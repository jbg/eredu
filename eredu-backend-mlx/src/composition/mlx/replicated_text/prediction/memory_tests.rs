use super::*;
use crate::backend::managed_memory::NativeMemoryRetention;
use crate::backend::runtime::cache::kv::CompressedLatentCache;
use crate::composition::mlx::speculative::SpeculativeExecutionStreams;
use eredu_architectures::prediction_extension::PredictionExtensionMaterializer;
use eredu_core::{
    cache::{
        LayerCachePolicy, PoolingStateComponent, StateResidencyClass, StateTensorDimension,
        StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
    Admission, AttentionPolicy, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_nn::{CompressedAttentionCache, CompressedAttentionState, PoolingAttentionCache};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryError};
use eredu_runtime::StateLayout;
use safemlx::ops::indexing::IntoStrideBy;
use std::num::NonZeroU32;

type Sequential = <MlxEmbeddedPredictionMaterializer as PredictionExtensionMaterializer<
    MlxNeuralBackend,
>>::SequentialState;
type Pooling = <MlxEmbeddedPredictionMaterializer as PredictionExtensionMaterializer<
    MlxNeuralBackend,
>>::PoolingState;

fn zero_admission() -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless prediction ownership fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(0),
    })
}

fn settle(pool: &MemoryLedger, expected: usize) {
    submission_recovery::wait_for_retirement(|| {
        MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == expected
    });
}

fn blocked(pool: &MemoryLedger) {
    settle(pool, 1);
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &zero_admission()),
        Err(WorkingMemoryError::UnknownBound),
    ));
}

fn memory_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    let mut current = error;
    loop {
        if let Some(memory) = current.downcast_ref::<WorkingMemoryError>() {
            return Some(memory);
        }
        current = current.source()?;
    }
}

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn copy_streams() -> impl Iterator<Item = Stream> {
    [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu]
        .into_iter()
        .filter(|device| *device == safemlx::DeviceType::Cpu || cfg!(feature = "metal"))
        .map(|device| Stream::new_with_device(&safemlx::Device::new(device, 0)))
}

fn input(tokens: i32, width: i32, seed: f32, stream: &Stream) -> MlxTensor {
    MlxTensor::from_array(
        Array::from_slice(
            &(0..tokens * width)
                .map(|i| seed + i as f32 * 0.25)
                .collect::<Vec<_>>(),
            &[1, width, tokens],
        )
        .transpose_axes(&[0, 2, 1], stream)
        .unwrap(),
    )
}

pub(super) fn sequential(populated: bool, stream: &Stream) -> Sequential {
    let mut state = OwnedPredictionCache::new(
        CompressedLatentCache::new(),
        NativeMemoryRetention::default(),
    );
    if populated {
        state
            .append(
                CompressedAttentionState {
                    latent: input(3, 3, 0.5, stream),
                    rotary: input(3, 2, -4.5, stream),
                },
                stream,
            )
            .unwrap();
    }
    state
}

pub(super) fn pooling_policy() -> LayerCachePolicy {
    let mut tensors = Vec::new();
    for stream in 0..2 {
        let ratio = NonZeroU32::new(stream + 2).unwrap();
        for component in [
            PoolingStateComponent::PendingValues,
            PoolingStateComponent::PendingGates,
            PoolingStateComponent::Pooled,
            PoolingStateComponent::OverlapValues,
            PoolingStateComponent::OverlapGates,
        ] {
            let extent = match component {
                PoolingStateComponent::PendingValues | PoolingStateComponent::PendingGates => {
                    StateTensorDimension::PrefixTokensRem(ratio)
                }
                PoolingStateComponent::Pooled => StateTensorDimension::PrefixTokensDiv(ratio),
                _ => StateTensorDimension::Fixed(ratio),
            };
            let tensor = StateTensorPolicy::new_with_residency(
                StateTensorRole::Pooling { stream, component },
                vec![
                    StateTensorDimension::Batch,
                    extent,
                    StateTensorDimension::fixed(2).unwrap(),
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
    LayerCachePolicy::key_only_with_fixed_state(AttentionPolicy::sliding(4).unwrap(), 1, 2, tensors)
        .unwrap()
}

pub(super) fn pooling(populated: bool, stream: &Stream) -> Pooling {
    let mut state = OwnedPredictionCache::new(
        MlxPoolingAttentionCache::resident_from_policy(0, &pooling_policy()).unwrap(),
        NativeMemoryRetention::default(),
    );
    if populated {
        state
            .append_local(input(5, 2, 0.5, stream), stream)
            .unwrap();
        for id in 0..2 {
            let ratio = state.pooling_ratio(id).unwrap();
            let windows = state
                .accumulate_pooling_windows(
                    id,
                    input(5, 2, 3.5 + id as f32, stream),
                    input(5, 2, -9.5 - id as f32, stream),
                    0,
                    stream,
                )
                .unwrap();
            let count = windows.values.as_array().dim(1);
            state
                .append_pooled(
                    id,
                    input(count / ratio, 2, 12.5 + id as f32, stream),
                    stream,
                )
                .unwrap();
            state
                .replace_pooling_overlap(
                    id,
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
    state
}

fn numeric(arrays: Vec<&Array>, stream: &Stream) -> Vec<(Vec<i32>, Vec<f32>)> {
    arrays
        .into_iter()
        .map(|array| {
            array.evaluated().unwrap();
            (
                array.shape().to_vec(),
                array
                    .contiguous(false, stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec(),
            )
        })
        .collect()
}

fn independent_copies(originals: Vec<&Array>, copies: Vec<&Array>) {
    assert_eq!(originals.len(), copies.len());
    for (original, copy) in originals.into_iter().zip(copies) {
        assert_ne!(
            original.allocation_info().unwrap().unwrap().identity(),
            copy.allocation_info().unwrap().unwrap().identity()
        );
    }
}

#[test]
fn empty_sequential_snapshot_authority_survives_clone_checkpoint_clear_and_restore() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let source = sequential(false, &stream);
    let copied = MlxEmbeddedPredictionMaterializer::sequential_snapshot(
        &source,
        SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool),
    )
    .unwrap()
    .unwrap();
    assert!(copied.inner().retained_arrays().is_empty());
    let mut clone = copied.clone();
    drop((source, copied));
    blocked(&pool);
    let checkpoint = CompressedAttentionCache::checkpoint(&clone);
    CompressedAttentionCache::clear(&mut clone).unwrap();
    drop(clone);
    blocked(&pool);
    let mut restored = sequential(false, &stream);
    CompressedAttentionCache::restore(&mut restored, &checkpoint, &stream).unwrap();
    drop(checkpoint);
    CompressedAttentionCache::clear(&mut restored).unwrap();
    assert_eq!(restored.offset(), 0);
    blocked(&pool);
    drop(restored);
    settle(&pool, 0);
}

#[test]
fn empty_pooling_snapshot_authority_survives_clone_checkpoint_clear_and_restore() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let source = pooling(false, &stream);
    let copied = MlxEmbeddedPredictionMaterializer::pooling_snapshot(
        &source,
        SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool),
    )
    .unwrap()
    .unwrap();
    assert!(copied.inner().retained_arrays().is_empty());
    let mut clone = copied.clone();
    drop((source, copied));
    blocked(&pool);
    let checkpoint = PoolingAttentionCache::checkpoint(&clone).unwrap();
    PoolingAttentionCache::clear(&mut clone).unwrap();
    drop(clone);
    blocked(&pool);
    let mut restored = pooling(false, &stream);
    PoolingAttentionCache::restore(&mut restored, &checkpoint, &stream).unwrap();
    drop(checkpoint);
    PoolingAttentionCache::clear(&mut restored).unwrap();
    assert_eq!(restored.offset(), 0);
    blocked(&pool);
    drop(restored);
    settle(&pool, 0);
}

#[test]
fn sequential_snapshot_copies_nonzero_components_and_raw_alias_keeps_its_domain() {
    for stream in copy_streams() {
        let pool = crate::memory_fixture::ledger(0, 0).unwrap();
        let source = sequential(true, &stream);
        let (latent, rotary) = source.inner().arrays().unwrap();
        let expected = numeric(vec![latent, rotary], &stream);
        assert_eq!(expected.len(), 2);
        let copied = MlxEmbeddedPredictionMaterializer::sequential_snapshot(
            &source,
            SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool),
        )
        .unwrap()
        .unwrap();
        assert_eq!(copied.offset(), 3);
        let (latent, rotary) = copied.inner().arrays().unwrap();
        assert_eq!(numeric(vec![latent, rotary], &stream), expected);
        // Compare logical contents separately from all retained backing: the
        // source may retain padding while the isolated copy compacts its views.
        independent_copies(
            source.inner().retained_arrays(),
            copied.inner().retained_arrays(),
        );
        let escaped = copied.inner().arrays().unwrap().0.clone();
        let allocation = escaped.allocation_info().unwrap().unwrap();
        let alias = escaped
            .try_index_device((.., .., (..).stride_by(2)), &stream)
            .unwrap();
        alias.evaluated().unwrap();
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation));
        drop((copied, source, escaped));
        blocked(&pool);
        let expected_alias = expected[0]
            .1
            .chunks_exact(3)
            .flat_map(|row| row.iter().step_by(2).copied())
            .collect::<Vec<_>>();
        assert_eq!(numeric(vec![&alias], &stream)[0].1, expected_alias);
        drop(alias);
        settle(&pool, 0);
    }
}

#[test]
fn pooling_snapshot_copies_local_pending_pooled_and_overlap_frontiers_and_raw_alias_keeps_owner() {
    for stream in copy_streams() {
        let pool = crate::memory_fixture::ledger(0, 0).unwrap();
        let source = pooling(true, &stream);
        let expected = numeric(source.inner().retained_arrays(), &stream);
        assert_eq!(source.inner().prompt_cache_state_arrays(0).len(), 10);
        let copied = MlxEmbeddedPredictionMaterializer::pooling_snapshot(
            &source,
            SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool),
        )
        .unwrap()
        .unwrap();
        assert_eq!(copied.offset(), 5);
        assert_eq!(copied.pooling_ratio(0), Some(2));
        assert_eq!(copied.pooling_ratio(1), Some(3));
        assert_eq!(numeric(copied.inner().retained_arrays(), &stream), expected);
        independent_copies(
            source.inner().retained_arrays(),
            copied.inner().retained_arrays(),
        );
        let escaped = copied
            .inner()
            .prompt_cache_state_arrays(0)
            .into_iter()
            .find(|array| {
                array.role
                    == StateTensorRole::Pooling {
                        stream: 0,
                        component: PoolingStateComponent::OverlapValues,
                    }
            })
            .unwrap()
            .array
            .clone();
        let expected_alias = numeric(vec![&escaped], &stream)[0]
            .1
            .iter()
            .step_by(2)
            .copied()
            .collect::<Vec<_>>();
        let allocation = escaped.allocation_info().unwrap().unwrap();
        let alias = escaped
            .try_index_device((.., .., (..).stride_by(2)), &stream)
            .unwrap();
        alias.evaluated().unwrap();
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation));
        drop((copied, source, escaped));
        blocked(&pool);
        assert_eq!(numeric(vec![&alias], &stream)[0].1, expected_alias);
        drop(alias);
        settle(&pool, 0);
    }
}

#[test]
fn reserved_prediction_snapshot_domain_rejects_all_cache_kinds_preserving_sources() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let other = crate::memory_fixture::ledger(0, 0).unwrap();
    for populated in [false, true] {
        let sequential = sequential(populated, &stream);
        let pooling = pooling(populated, &stream);
        let layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![
                    LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 3, 2)
                        .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let mut model = MlxHybridState::device(layout).unwrap();
        if populated {
            model
                .layer(0)
                .unwrap()
                .append(
                    CompressedAttentionState {
                        latent: input(3, 3, 2.5, &stream),
                        rotary: input(3, 2, -5.5, &stream),
                    },
                    &stream,
                )
                .unwrap();
        }
        let before = [
            numeric(sequential.inner().retained_arrays(), &stream),
            numeric(pooling.inner().retained_arrays(), &stream),
            numeric(model.retained_arrays(), &stream),
        ];
        let allocations = [
            &sequential.inner().retained_arrays(),
            &pooling.inner().retained_arrays(),
            &model.retained_arrays(),
        ]
        .map(|arrays| {
            arrays
                .iter()
                .map(|array| array.allocation_info().unwrap().unwrap())
                .collect::<Vec<_>>()
        });
        let offsets = (sequential.offset(), pooling.offset(), model.offset());
        let reservation = pool
            .reserve(&InferenceExecutionIdentity::default(), &zero_admission())
            .unwrap();
        let context = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&pool);
        let errors = [
            MlxEmbeddedPredictionMaterializer::sequential_snapshot(&sequential, context)
                .unwrap_err(),
            MlxEmbeddedPredictionMaterializer::pooling_snapshot(&pooling, context).unwrap_err(),
            MlxEmbeddedPredictionMaterializer::model_snapshot(&model, context).unwrap_err(),
        ];
        for error in errors {
            assert_eq!(
                memory_error(&error),
                Some(&WorkingMemoryError::ReservedWorkActive)
            );
        }
        assert_eq!(
            [
                numeric(sequential.inner().retained_arrays(), &stream),
                numeric(pooling.inner().retained_arrays(), &stream),
                numeric(model.retained_arrays(), &stream)
            ],
            before
        );
        assert_eq!(
            [
                &sequential.inner().retained_arrays(),
                &pooling.inner().retained_arrays(),
                &model.retained_arrays()
            ]
            .map(|arrays| arrays
                .iter()
                .map(|array| array.allocation_info().unwrap().unwrap())
                .collect::<Vec<_>>()),
            allocations
        );
        assert_eq!(
            (sequential.offset(), pooling.offset(), model.offset()),
            offsets
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
        assert_eq!(pool.fixture_host_peak().unwrap(), 0);

        let other_context = SpeculativeExecutionStreams::single(&stream).with_memory_ledger(&other);
        let sequential_copy =
            MlxEmbeddedPredictionMaterializer::sequential_snapshot(&sequential, other_context)
                .unwrap()
                .unwrap();
        let pooling_copy =
            MlxEmbeddedPredictionMaterializer::pooling_snapshot(&pooling, other_context)
                .unwrap()
                .unwrap();
        let model_copy = MlxEmbeddedPredictionMaterializer::model_snapshot(&model, other_context)
            .unwrap()
            .unwrap();
        settle(&other, 3);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        drop((sequential_copy, pooling_copy, model_copy));
        settle(&other, 0);
        drop(reservation);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
