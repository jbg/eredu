use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp},
    Array, Device, DeviceType,
};

use crate::backend::ExecutionContext;

use super::gated_delta_scan;

#[test]
fn vector_decay_matches_scalar_when_channels_are_equal() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let query = Array::from_slice(&[0.5f32, -0.25, 0.1, 0.2], &[1, 2, 1, 2]);
    let key = Array::from_slice(&[0.3f32, 0.4, -0.2, 0.7], &[1, 2, 1, 2]);
    let value = Array::from_slice(&[1.0f32, -0.5, 0.25, 0.75], &[1, 2, 1, 2]);
    let beta = Array::from_slice(&[0.8f32, 0.6], &[1, 2, 1]);
    let scalar = Array::from_slice(&[-0.2f32, -0.4], &[1, 2, 1]);
    let vector = Array::from_slice(&[-0.2f32, -0.2, -0.4, -0.4], &[1, 2, 1, 2]);
    let (_, scalar_output) =
        gated_delta_scan(&query, &key, &value, &scalar, &beta, None, stream).unwrap();
    let (_, vector_output) =
        gated_delta_scan(&query, &key, &value, &vector, &beta, None, stream).unwrap();
    let scalar_output = scalar_output.evaluated().unwrap();
    let vector_output = vector_output.evaluated().unwrap();
    let scalar = scalar_output.as_slice::<f32>();
    let vector = vector_output.as_slice::<f32>();
    assert!(scalar
        .iter()
        .zip(vector)
        .all(|(left, right)| (left - right).abs() < 1e-6));
}

#[test]
fn cached_chunks_match_one_scan() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let query = Array::from_slice(&[0.5f32, -0.25, 0.1, 0.2, -0.4, 0.8], &[1, 3, 1, 2]);
    let key = Array::from_slice(&[0.3f32, 0.4, -0.2, 0.7, 0.6, -0.1], &[1, 3, 1, 2]);
    let value = Array::from_slice(&[1.0f32, -0.5, 0.25, 0.75, -0.3, 0.9], &[1, 3, 1, 2]);
    let beta = Array::from_slice(&[0.8f32, 0.6, 0.4], &[1, 3, 1]);
    let decay = Array::from_slice(&[-0.2f32, -0.1, -0.4, -0.3, -0.5, -0.25], &[1, 3, 1, 2]);
    let (expected_state, expected) =
        gated_delta_scan(&query, &key, &value, &decay, &beta, None, stream).unwrap();
    let (state, first) = gated_delta_scan(
        &query.try_index_device((.., ..2, .., ..), stream).unwrap(),
        &key.try_index_device((.., ..2, .., ..), stream).unwrap(),
        &value.try_index_device((.., ..2, .., ..), stream).unwrap(),
        &decay.try_index_device((.., ..2, .., ..), stream).unwrap(),
        &beta.try_index_device((.., ..2, ..), stream).unwrap(),
        None,
        stream,
    )
    .unwrap();
    let (actual_state, second) = gated_delta_scan(
        &query.try_index_device((.., 2.., .., ..), stream).unwrap(),
        &key.try_index_device((.., 2.., .., ..), stream).unwrap(),
        &value.try_index_device((.., 2.., .., ..), stream).unwrap(),
        &decay.try_index_device((.., 2.., .., ..), stream).unwrap(),
        &beta.try_index_device((.., 2.., ..), stream).unwrap(),
        Some(state),
        stream,
    )
    .unwrap();
    let actual = concatenate_axis(&[first, second], 1, stream).unwrap();
    let expected = expected.evaluated().unwrap();
    let actual = actual.evaluated().unwrap();
    assert!(expected
        .as_slice::<f32>()
        .iter()
        .zip(actual.as_slice::<f32>())
        .all(|(left, right)| (left - right).abs() < 1e-6));
    let expected_state = expected_state.evaluated().unwrap();
    let actual_state = actual_state.evaluated().unwrap();
    assert!(expected_state
        .as_slice::<f32>()
        .iter()
        .zip(actual_state.as_slice::<f32>())
        .all(|(left, right)| (left - right).abs() < 1e-6));
}

#[test]
#[ignore = "requires an MLX Metal device"]
fn metal_vector_decay_matches_cpu() {
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let values = |stream: &safemlx::Stream| {
        let query = Array::from_slice(&[0.5f32, -0.25, 0.1, 0.2], &[1, 2, 1, 2]);
        let key = Array::from_slice(&[0.3f32, 0.4, -0.2, 0.7], &[1, 2, 1, 2]);
        let value = Array::from_slice(&[1.0f32, -0.5, 0.25, 0.75], &[1, 2, 1, 2]);
        let beta = Array::from_slice(&[0.8f32, 0.6], &[1, 2, 1]);
        let decay = Array::from_slice(&[-0.2f32, -0.1, -0.4, -0.3], &[1, 2, 1, 2]);
        gated_delta_scan(&query, &key, &value, &decay, &beta, None, stream).unwrap()
    };
    let (cpu_state, cpu_output) = values(cpu.stream());
    let (gpu_state, gpu_output) = values(gpu.stream());
    let cpu_output = cpu_output.evaluated().unwrap();
    let gpu_output = gpu_output.evaluated().unwrap();
    assert!(cpu_output
        .as_slice::<f32>()
        .iter()
        .zip(gpu_output.as_slice::<f32>())
        .all(|(left, right)| (left - right).abs() < 1e-5));
    let cpu_state = cpu_state.evaluated().unwrap();
    let gpu_state = gpu_state.evaluated().unwrap();
    assert!(cpu_state
        .as_slice::<f32>()
        .iter()
        .zip(gpu_state.as_slice::<f32>())
        .all(|(left, right)| (left - right).abs() < 1e-5));
}
