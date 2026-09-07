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
