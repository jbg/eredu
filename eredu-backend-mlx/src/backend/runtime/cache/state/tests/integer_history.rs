use super::*;
use eredu_runtime::IntegerHistorySpec;
#[test]
fn integer_history_preserves_exact_bits_through_native_snapshot_and_restore() {
    let spec = IntegerHistorySpec::new(9, 3, 7).unwrap();
    let layout = StateLayout::new(
        eredu_core::LayerSchedule::new(
            1,
            vec![LayerCachePolicy::fixed_only(vec![spec.policy()]).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxHybridState::device(layout, &[]).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    assert_eq!(
        spec.read::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], 2, &stream)
            .unwrap(),
        [7; 6]
    );
    let values = [i32::MAX, i32::MIN, 16777217, -16777217, 1, 2];
    spec.write::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], &values, 2, &stream)
        .unwrap();
    state.layers_mut()[0].advance_fixed(3).unwrap();
    let saved = state.isolated_snapshot(&stream).unwrap();
    let next = spec.next(&values, &[101, -101], 2, 1).unwrap();
    spec.write::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], &next, 2, &stream)
        .unwrap();
    state.layers_mut()[0].advance_fixed(1).unwrap();
    state.restore_checkpoint(&saved, &stream).unwrap();
    assert_eq!(
        spec.read::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], 2, &stream)
            .unwrap(),
        values
    );
    assert_eq!(state.layers()[0].position(), 3);
    // Reject a floating tensor even when its current numeric values are integral.
    *state.layers_mut()[0].fixed_component(spec.role()).unwrap() = Some(MlxTensor::from_array(
        Array::from_slice(&[1f32; 6], &[2, 3]),
    ));
    assert!(spec
        .read::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], 2, &stream)
        .is_err());
    state.clear().unwrap();
    assert_eq!(
        spec.read::<MlxNeuralBackend, _>(&mut state.layers_mut()[0], 2, &stream)
            .unwrap(),
        [7; 6]
    );
}
