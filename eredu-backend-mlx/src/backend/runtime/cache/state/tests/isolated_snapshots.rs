use super::*;
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

fn values(arrays: Vec<&Array>, stream: &Stream) -> Vec<Vec<f32>> {
    arrays
        .into_iter()
        .map(|array| {
            array
                .contiguous(false, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        })
        .collect()
}

#[test]
fn isolated_native_kv_snapshots_preserve_windows_capacity_and_interleaved_siblings() {
    let stream = stream();
    for window in [None, Some(3)] {
        let policy = LayerCachePolicy::key_value(
            AttentionPolicy::from_sliding_window(window).unwrap(),
            2,
            1,
        )
        .unwrap();
        let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
        let mut parent = MlxKeyValueState::device(layout).unwrap();
        if window.is_none() {
            parent.layers[0] = MlxKeyValueLayerState::Device(ConcatKeyValueCache::new_with_step(4));
        }
        let input = Array::from_slice(
            &(0..10).map(|n| n as f32).collect::<Vec<_>>(),
            &[1, 2, 5, 1],
        );
        parent.layers[0]
            .update_and_fetch(input.clone(), input, &stream)
            .unwrap();
        let before = values(parent.retained_arrays(), &stream);
        let saved = parent.isolated_snapshot(&stream).unwrap();
        let capacity_bound = parent.continuation_capacity_bound(10).unwrap();
        assert_eq!(saved.offset(), 5);
        assert_eq!(values(saved.retained_arrays(), &stream), before);
        let mut first = saved.isolated_snapshot(&stream).unwrap();
        let mut second = saved.isolated_snapshot(&stream).unwrap();
        // Cross several capacity growth and sliding-window retention boundaries.
        for round in 0..10 {
            for (state, value) in [(&mut parent, 20.), (&mut first, 30.), (&mut second, 40.)] {
                let next = Array::from_slice(&[value + round as f32, value + 1.], &[1, 2, 1, 1]);
                state.layers[0]
                    .update_and_fetch(next.clone(), next, &stream)
                    .unwrap();
                assert_eq!(state.offset(), 6 + round);
                for array in state.retained_arrays() {
                    assert!(u64::try_from(array.dim(-2)).unwrap() <= capacity_bound);
                }
                assert_eq!(saved.offset(), 5);
                assert_eq!(values(saved.retained_arrays(), &stream), before);
            }
        }
        assert_ne!(
            values(parent.retained_arrays(), &stream),
            values(first.retained_arrays(), &stream)
        );
        assert_ne!(
            values(first.retained_arrays(), &stream),
            values(second.retained_arrays(), &stream)
        );
    }
}

#[test]
fn isolated_native_hybrid_snapshots_copy_strided_recurrent_and_convolution_storage() {
    let stream = stream();
    let roles = [
        StateTensorRole::Recurrent,
        StateTensorRole::Convolution { slot: 0 },
    ];
    let tensors = roles
        .iter()
        .map(|role| {
            StateTensorPolicy::new(
                *role,
                vec![
                    StateTensorDimension::fixed(2).unwrap(),
                    StateTensorDimension::fixed(2).unwrap(),
                ],
                StateTensorDtype::Float32,
                if matches!(role, StateTensorRole::Convolution { .. }) {
                    MutableStateResidency::AlwaysDeviceMutable
                } else {
                    MutableStateResidency::LayerScopedOffloadable
                },
            )
            .unwrap()
        })
        .collect();
    let policy = LayerCachePolicy::fixed_only(tensors).unwrap();
    let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
    let mut parent = MlxHybridState::device(layout).unwrap();
    for (index, role) in roles.iter().enumerate() {
        let array = Array::from_slice(&[1. + index as f32, 2., 3., 4.], &[2, 2])
            .transpose(&stream)
            .unwrap();
        parent.layers[0]
            .fixed
            .insert(*role, Some(MlxTensor::from_array(array)));
    }
    parent.layers[0].fixed_offset = 7;
    let saved = parent.isolated_snapshot(&stream).unwrap();
    assert_eq!(saved.layers[0].fixed_offset, 7);
    let before = values(parent.retained_arrays(), &stream);
    assert_eq!(values(saved.retained_arrays(), &stream), before);
    for (source, copied) in parent.retained_arrays().iter().zip(saved.retained_arrays()) {
        let source = source.evaluated().unwrap();
        let copied = copied.evaluated().unwrap();
        assert_ne!(
            source.as_slice::<f32>().as_ptr(),
            copied.as_slice::<f32>().as_ptr(),
            "native fixed storage must be independent, not just its Rust handle"
        );
    }
    let mut child = saved.isolated_snapshot(&stream).unwrap();
    for (state, value) in [(&mut parent, 10.), (&mut child, 20.)] {
        for role in roles {
            state.layers[0].fixed.insert(
                role,
                Some(MlxTensor::from_array(Array::from_slice(
                    &[value; 4],
                    &[2, 2],
                ))),
            );
        }
        state.layers[0].fixed_offset += 1;
        assert_eq!(values(saved.retained_arrays(), &stream), before);
        assert_eq!(saved.layers[0].fixed_offset, 7);
    }
    assert_ne!(
        values(parent.retained_arrays(), &stream),
        values(child.retained_arrays(), &stream)
    );
}

#[test]
fn isolated_paged_snapshots_copy_strided_tails_and_sealed_blocks_across_branches() {
    use eredu_runtime::PagedCacheOptions;
    let stream = stream();
    for window in [None, Some(3)] {
        let policy = LayerCachePolicy::key_value(
            AttentionPolicy::from_sliding_window(window).unwrap(),
            2,
            2,
        )
        .unwrap();
        let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 8192, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let mut parent = MlxKeyValueState::paged(layout, manager, None).unwrap();
        let input = Array::from_slice(
            &(0..12).map(|n| n as f32 * 0.25).collect::<Vec<_>>(),
            &[1, 3, 2, 2],
        )
        .transpose_axes(&[0, 2, 1, 3], &stream)
        .unwrap();
        parent.layers[0]
            .update_and_fetch(input.clone(), input.clone(), &stream)
            .unwrap();
        let saved = parent.isolated_snapshot(&stream).unwrap();
        let before = values(saved.retained_arrays(), &stream);
        let mut first = saved.isolated_snapshot(&stream).unwrap();
        let mut second = saved.isolated_snapshot(&stream).unwrap();
        let mut expected = [input.clone(), input.clone(), input];
        for round in 0..7 {
            for (branch, state) in [&mut parent, &mut first, &mut second]
                .into_iter()
                .enumerate()
            {
                let next = Array::from_slice(
                    &[
                        10. + branch as f32,
                        round as f32,
                        20. + branch as f32,
                        round as f32 + 0.5,
                    ],
                    &[1, 2, 1, 2],
                );
                expected[branch] =
                    safemlx::ops::concatenate_axis(&[&expected[branch], &next], -2, &stream)
                        .unwrap();
                state.layers[0]
                    .update_and_fetch(next.clone(), next, &stream)
                    .unwrap();
                let expected_array = if let Some(window) = window {
                    let end = expected[branch].dim(-2);
                    expected[branch]
                        .try_index_device((.., .., (end - window).max(0)..end, ..), &stream)
                        .unwrap()
                } else {
                    expected[branch].clone()
                };
                let query_values = [0.2_f32, -0.3, 0.7, -0.4];
                let query = Array::from_slice(&query_values, &[1, 2, 1, 2]);
                let actual = KeyValueCache::paged_attention(
                    &mut state.layers[0],
                    &query,
                    0.7,
                    None,
                    None,
                    None,
                    eredu_nn::AttentionArithmetic::Fused,
                    &stream,
                )
                .unwrap()
                .unwrap();
                let rows = values(vec![&expected_array], &stream).remove(0);
                let seq = expected_array.dim(-2) as usize;
                let mut oracle = Vec::new();
                for head in 0..2 {
                    let rows = &rows[head * seq * 2..(head + 1) * seq * 2];
                    let weights = rows
                        .chunks_exact(2)
                        .map(|v| {
                            (0.7 * (query_values[head * 2] * v[0]
                                + query_values[head * 2 + 1] * v[1]))
                                .exp()
                        })
                        .collect::<Vec<_>>();
                    let denominator: f32 = weights.iter().sum();
                    for column in 0..2 {
                        oracle.push(
                            rows.chunks_exact(2)
                                .zip(&weights)
                                .map(|(v, w)| v[column] * w / denominator)
                                .sum::<f32>(),
                        );
                    }
                }
                for (actual, expected) in
                    values(vec![&actual], &stream).remove(0).iter().zip(oracle)
                {
                    assert!((actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs());
                }
                assert_eq!(saved.offset(), 3);
                assert_eq!(values(saved.retained_arrays(), &stream), before);
            }
        }
    }
}

#[test]
fn paged_truncation_copies_logical_rows_from_a_strided_sealed_prefix() {
    let stream = stream();
    let options = eredu_runtime::PagedCacheOptions::new(4, 4096, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = PagedKeyValueCache::new(manager, 0, None).unwrap();
    let input = Array::from_slice(
        &(0..24).map(|i| i as f32 * 0.125).collect::<Vec<_>>(),
        &[1, 6, 2, 2],
    )
    .transpose_axes(&[0, 2, 1, 3], &stream)
    .unwrap();
    cache
        .update_and_fetch(input.clone(), input, &stream)
        .unwrap();
    cache.truncate(3, &stream).unwrap();
    let copied = cache.deep_clone_state(&stream).unwrap();
    assert_eq!(
        values(cache.retained_arrays(), &stream),
        values(copied.retained_arrays(), &stream)
    );
    let queries = Array::from_slice(&[0.0_f32; 4], &[1, 2, 1, 2]);
    let actual = cache
        .paged_attention(
            &queries,
            1.,
            None,
            None,
            None,
            eredu_nn::AttentionArithmetic::Fused,
            &stream,
        )
        .unwrap()
        .unwrap();
    // Uniform attention over tokens 0,1,2, with distinct head and channel values.
    assert_eq!(
        values(vec![&actual], &stream).remove(0),
        [0.5, 0.625, 0.75, 0.875]
    );
    let next = Array::from_slice(&[3., 4., 5., 6.], &[1, 2, 1, 2]);
    cache.update_and_fetch(next.clone(), next, &stream).unwrap();
    let actual = cache
        .paged_attention(
            &queries,
            1.,
            None,
            None,
            None,
            eredu_nn::AttentionArithmetic::Fused,
            &stream,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        values(vec![&actual], &stream).remove(0),
        [1.125, 1.46875, 1.8125, 2.15625]
    );
}
