use super::*;
use eredu_core::{
    cache::{StateResidencyClass, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};
use std::num::NonZeroU32;

fn policy(streams: usize) -> LayerCachePolicy {
    let mut tensors = Vec::new();
    for stream in 0..streams {
        let ratio = NonZeroU32::new(if stream == 0 { 4 } else { 6 }).unwrap();
        for component in POOLING_COMPONENTS {
            if streams == 1
                && matches!(
                    component,
                    PoolingStateComponent::OverlapValues | PoolingStateComponent::OverlapGates
                )
            {
                continue;
            }
            let extent = match component {
                PoolingStateComponent::PendingValues | PoolingStateComponent::PendingGates => {
                    StateTensorDimension::PrefixTokensRem(ratio)
                }
                PoolingStateComponent::Pooled => StateTensorDimension::PrefixTokensDiv(ratio),
                _ => StateTensorDimension::Fixed(ratio),
            };
            let tensor = StateTensorPolicy::new_with_residency(
                StateTensorRole::Pooling {
                    stream: stream as u32,
                    component,
                },
                vec![
                    StateTensorDimension::Batch,
                    extent,
                    StateTensorDimension::fixed(8).unwrap(),
                ],
                StateTensorDtype::Floating,
                if component == PoolingStateComponent::Pooled {
                    StateResidencyClass::SealablePaged
                } else {
                    StateResidencyClass::AlwaysDeviceMutable
                },
            )
            .unwrap();
            tensors.push(match component {
                PoolingStateComponent::PendingValues | PoolingStateComponent::PendingGates => {
                    tensor.when_prefix_remainder_nonzero(ratio)
                }
                _ => tensor.when_prefix_at_least(ratio),
            });
        }
    }
    if streams == 0 {
        LayerCachePolicy::key_only(AttentionPolicy::sliding(7).unwrap(), 1, 8).unwrap()
    } else {
        LayerCachePolicy::key_only_with_fixed_state(
            AttentionPolicy::sliding(7).unwrap(),
            1,
            8,
            tensors,
        )
        .unwrap()
    }
}

fn state(streams: usize, paged: bool) -> MlxPoolingAttentionState {
    let layout =
        StateLayout::new(LayerSchedule::new(2, vec![policy(streams); 2]).unwrap()).unwrap();
    if paged {
        let pool = eredu_runtime::CacheResidencyPool::new(
            eredu_runtime::CachePoolLimits::new(8 << 20, 64 << 20, 8 << 20, 0).unwrap(),
        );
        let manager = CacheResidencyManager::new(
            eredu_runtime::PagedCacheOptions::new(2, 1 << 20, 32 << 20, 1)
                .unwrap()
                .with_pool(pool)
                .unwrap(),
        )
        .unwrap();
        MlxPoolingAttentionStateFactory::paged(layout, manager, 3, 0, None).unwrap()
    } else {
        MlxPoolingAttentionStateFactory::device(layout).unwrap()
    }
}

fn values(array: &Array, stream: &Stream) -> Vec<f32> {
    array
        .contiguous(false, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

fn pools(
    state: &MlxPoolingAttentionState,
    stream: &Stream,
) -> Vec<(usize, StateTensorRole, Vec<f32>)> {
    state
        .as_ref()
        .iter()
        .enumerate()
        .flat_map(|(layer, cache)| {
            cache
                .prompt_cache_state_arrays(layer)
                .into_iter()
                .map(move |a| (layer, a.role, values(a.array, stream)))
        })
        .collect()
}

fn advance(
    state: &mut MlxPoolingAttentionState,
    tokens: i32,
    seed: f32,
    stream: &Stream,
) -> Vec<Vec<f32>> {
    let mut local_values = Vec::new();
    for (layer, cache) in state.as_mut().iter_mut().enumerate() {
        let offset = cache.offset();
        let input = |bias: f32| {
            // Transposed incoming tensors and sliced remainders exercise copies
            // of logical elements rather than the backing allocation's order.
            Array::from_slice(
                &(0..8 * tokens)
                    .map(|i| seed + bias + layer as f32 + i as f32 * 0.125)
                    .collect::<Vec<_>>(),
                &[1, 8, tokens],
            )
            .transpose_axes(&[0, 2, 1], stream)
            .unwrap()
        };
        let keys = cache
            .append_local(MlxTensor::from_array(input(0.)), stream)
            .unwrap();
        assert_eq!(keys.as_array().dim(1), offset.min(6) + tokens);
        local_values.push(values(keys.as_array(), stream));
        for stream_id in 0..2 {
            let Some(ratio) = cache.pooling_ratio(stream_id) else {
                continue;
            };
            let windows = cache
                .accumulate_pooling_windows(
                    stream_id,
                    MlxTensor::from_array(input(2. + stream_id as f32)),
                    MlxTensor::from_array(input(-4. - stream_id as f32)),
                    offset,
                    stream,
                )
                .unwrap();
            let count = windows.values.as_array().dim(1);
            let sources = values(windows.values.as_array(), stream);
            let gates = values(windows.gates.as_array(), stream);
            let mut pooled = Vec::new();
            for group in 0..count / ratio {
                for column in 0..8 {
                    pooled.push(
                        (0..ratio)
                            .map(|row| {
                                let i = ((group * ratio + row) * 8 + column) as usize;
                                sources[i] + gates[i] * 0.25
                            })
                            .sum::<f32>(),
                    );
                }
            }
            cache
                .append_pooled(
                    stream_id,
                    MlxTensor::from_array(Array::from_slice(&pooled, &[1, count / ratio, 8])),
                    stream,
                )
                .unwrap();
            if count > 0 && matches!(cache, MlxPoolingAttentionCache::Sparse { .. }) {
                cache
                    .replace_pooling_overlap(
                        stream_id,
                        MlxTensor::from_array(
                            windows
                                .values
                                .as_array()
                                .try_index_device((.., count - ratio.., ..), stream)
                                .unwrap(),
                        ),
                        MlxTensor::from_array(
                            windows
                                .gates
                                .as_array()
                                .try_index_device((.., count - ratio.., ..), stream)
                                .unwrap(),
                        ),
                    )
                    .unwrap();
            }
        }
    }
    local_values
}

#[test]
fn isolated_pooling_snapshots_copy_all_streams_and_preserve_interleaved_siblings() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for streams in 0..=2 {
        for paged in [false, true] {
            let mut parent = state(streams, paged);
            advance(&mut parent, 13, 1., &stream);
            if !paged {
                let checkpoint = parent.clone();
                let before = pools(&parent, &stream);
                advance(&mut parent, 2, 90., &stream);
                for (current, previous) in parent.as_mut().iter_mut().zip(checkpoint.as_ref()) {
                    PoolingAttentionCache::restore(current, previous, &stream).unwrap();
                }
                assert_eq!(pools(&parent, &stream), before);
            }
            assert!(MlxPoolingAttentionStateFactory::supports_isolated_snapshot(
                &parent
            ));
            assert!(
                MlxPoolingAttentionStateFactory::continuation_capacity_bound(&parent, u64::MAX)
                    .is_none()
            );
            assert!(
                MlxPoolingAttentionStateFactory::isolated_snapshot_auxiliary_growth(
                    &parent,
                    u64::MAX
                )
                .is_none()
            );
            assert!(
                MlxPoolingAttentionStateFactory::isolated_snapshot_auxiliary_growth(&parent, 32)
                    .unwrap()
                    > 0
            );
            let saved =
                MlxPoolingAttentionStateFactory::isolated_snapshot(&parent, &stream).unwrap();
            assert!(parent
                .shared_layout()
                .unwrap()
                .same_storage(saved.shared_layout().unwrap()));
            let expected_saved = pools(&parent, &stream);
            assert_eq!(pools(&saved, &stream), expected_saved);
            for (source, copy) in parent
                .as_ref()
                .iter()
                .flat_map(MlxPoolingAttentionCache::retained_arrays)
                .zip(
                    saved
                        .as_ref()
                        .iter()
                        .flat_map(MlxPoolingAttentionCache::retained_arrays),
                )
            {
                let source = source.evaluated().unwrap();
                let copy = copy.evaluated().unwrap();
                if source.as_array().size() > 0 {
                    assert_ne!(
                        source
                            .as_array()
                            .allocation_info()
                            .unwrap()
                            .unwrap()
                            .identity(),
                        copy.as_array()
                            .allocation_info()
                            .unwrap()
                            .unwrap()
                            .identity()
                    );
                }
            }
            let mut first =
                MlxPoolingAttentionStateFactory::isolated_snapshot(&saved, &stream).unwrap();
            let mut second =
                MlxPoolingAttentionStateFactory::isolated_snapshot(&saved, &stream).unwrap();
            for copy in [&first, &second] {
                assert!(parent
                    .shared_layout()
                    .unwrap()
                    .same_storage(copy.shared_layout().unwrap()));
            }
            if paged {
                let identities: std::collections::BTreeSet<_> = [&parent, &saved, &first, &second]
                    .map(|state| {
                        let id = state.as_ref()[0].residency_manager().unwrap().session_id();
                        assert_eq!(
                            state.as_ref()[1].residency_manager().unwrap().session_id(),
                            id
                        );
                        id
                    })
                    .into_iter()
                    .collect();
                assert_eq!(identities.len(), 4);
                let mut invalid = saved.clone();
                invalid.as_mut()[1]
                    .rebind_paging_manager(parent.as_ref()[0].residency_manager().unwrap().clone());
                assert!(!MlxPoolingAttentionStateFactory::supports_isolated_snapshot(&invalid));
                assert!(
                    MlxPoolingAttentionStateFactory::isolated_snapshot(&invalid, &stream).is_err()
                );
            }
            let mut oracles = (0..3)
                .map(|_| {
                    let mut oracle = state(streams, paged);
                    advance(&mut oracle, 13, 1., &stream);
                    oracle
                })
                .collect::<Vec<_>>();
            for tokens in [1, 3, 7, 11] {
                for (branch, current) in [&mut parent, &mut first, &mut second]
                    .into_iter()
                    .enumerate()
                {
                    let seed = 20. + branch as f32 * 100. + tokens as f32;
                    assert_eq!(
                        advance(current, tokens, seed, &stream),
                        advance(&mut oracles[branch], tokens, seed, &stream)
                    );
                    assert_eq!(pools(current, &stream), pools(&oracles[branch], &stream));
                    assert_eq!(pools(&saved, &stream), expected_saved);
                }
            }
            let mut forked =
                MlxPoolingAttentionStateFactory::fork_prediction_target_state(&saved, &stream)
                    .unwrap();
            assert!(parent
                .shared_layout()
                .unwrap()
                .same_storage(forked.shared_layout().unwrap()));
            assert_eq!(pools(&forked, &stream), expected_saved);
            let mut replay =
                MlxPoolingAttentionStateFactory::isolated_snapshot(&saved, &stream).unwrap();
            let mut oracle = state(streams, paged);
            advance(&mut oracle, 13, 1., &stream);
            assert_eq!(
                advance(&mut replay, 9, -30., &stream),
                advance(&mut oracle, 9, -30., &stream)
            );
            assert_eq!(pools(&replay, &stream), pools(&oracle, &stream));
            let mut fork_oracle = state(streams, paged);
            advance(&mut fork_oracle, 13, 1., &stream);
            assert_eq!(
                advance(&mut forked, 9, -30., &stream),
                advance(&mut fork_oracle, 9, -30., &stream)
            );
            assert_eq!(pools(&forked, &stream), pools(&fork_oracle, &stream));
        }
    }
}

fn sliding_prediction_rollback(device: safemlx::DeviceType) {
    let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
    for streams in 0..=2 {
        for paged in [false, true] {
            let mut live = state(streams, paged);
            advance(&mut live, 17, 1., &stream);
            let original_pools = pools(&live, &stream);
            let saved = live
                .as_ref()
                .iter()
                .map(|cache| PoolingAttentionCache::checkpoint(cache).unwrap())
                .collect::<Vec<_>>();
            // Verification advances beyond the final window and seals the saved
            // mutable tail. Reject it, then replay a shorter accepted prefix.
            for _ in 0..2 {
                advance(&mut live, 5, 90., &stream);
                for (current, previous) in live.as_mut().iter_mut().zip(&saved) {
                    PoolingAttentionCache::restore(current, previous, &stream).unwrap();
                }
                assert_eq!(pools(&live, &stream), original_pools);
                let mut oracle = state(streams, paged);
                advance(&mut oracle, 17, 1., &stream);
                assert_eq!(
                    advance(&mut live, 2, 20., &stream),
                    advance(&mut oracle, 2, 20., &stream)
                );
                assert_eq!(pools(&live, &stream), pools(&oracle, &stream));
                for (current, previous) in live.as_mut().iter_mut().zip(&saved) {
                    PoolingAttentionCache::restore(current, previous, &stream).unwrap();
                }
            }
            let discarded = live.as_ref()[0]
                .residency_manager()
                .map(|manager| manager.report().unwrap().discarded_sliding_blocks);
            drop(saved);
            advance(&mut live, 11, 30., &stream);
            if let Some(discarded) = discarded {
                assert!(
                    live.as_ref()[0]
                        .residency_manager()
                        .unwrap()
                        .report()
                        .unwrap()
                        .discarded_sliding_blocks
                        > discarded,
                    "released checkpoints must allow normal sliding-history discard"
                );
            }
        }
    }
}

#[test]
fn pooling_sliding_prediction_rollback_retains_sealed_history_cpu() {
    sliding_prediction_rollback(safemlx::DeviceType::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires a local MLX Metal device"]
fn pooling_sliding_prediction_rollback_retains_sealed_history_metal() {
    sliding_prediction_rollback(safemlx::DeviceType::Gpu);
}
