//! Numerical and asynchronous admission contracts for fused stateless mechanisms.
use eredu_backend_mlx::{
    backend::nn::{shared::MlxNeuralBackend as Backend, tensor::TokenValidationScope},
    MlxTensor,
};
use eredu_nn::{Index, NeuralBackend, Tensor};
use safemlx::{transforms::async_eval_with_event, Device, DeviceType, Stream};

fn ordered_sum(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    // Cancellation distinguishes ordered addition from a reassociated reduction.
    let table = MlxTensor::from_f32_slice(
        &[1e20, 2., 7., 3., -1e20, 4., 8., 5., 3., 6., 9., 7.],
        &[3, 2, 2],
        &stream,
    )
    .unwrap();
    // Slice across time to exercise a strided, non-contiguous codes view.
    let codes =
        MlxTensor::from_i32_slice(&[0, 0, 0, 1, 2, 1, -1, 2, -1, 2, 2, 2], &[2, 2, 3], &stream)
            .unwrap()
            .index(&[Index::Full, Index::Range(0, 1), Index::Full], &stream)
            .unwrap();
    let actual = Backend::ordered_codebook_lookup_sum(&table, &codes, &stream).unwrap();
    assert_eq!(actual.to_f32_vec(&stream).unwrap(), [3., 12., 16., 10.]);
    let reference =
        eredu_nn::codebooks::ordered_codebook_lookup_sum(&table, &codes, &stream).unwrap();
    assert_eq!(
        actual.to_f32_vec(&stream).unwrap(),
        reference.to_f32_vec(&stream).unwrap()
    );
    for code in [-3, 3] {
        let codes = MlxTensor::from_i32_slice(&[code, 0, 0], &[1, 1, 3], &stream).unwrap();
        assert!(Backend::ordered_codebook_lookup_sum(&table, &codes, &stream).is_err());
        let scope = TokenValidationScope::begin().unwrap();
        let output = Backend::ordered_codebook_lookup_sum(&table, &codes, &stream).unwrap();
        let validations = scope.finish();
        let event =
            async_eval_with_event(std::iter::once(output.as_array()).chain(validations.arrays()))
                .unwrap();
        event.synchronize().unwrap();
        assert!(validations.validate_completed().is_err());
    }
    let padding = MlxTensor::from_i32_slice(&[2, 2, 2], &[1, 1, 3], &stream).unwrap();
    assert_eq!(
        Backend::ordered_codebook_lookup_sum(&table, &padding, &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap(),
        [0., 0.]
    );
}

#[test]
fn cpu_ordered_lookup_preserves_padding_signed_indices_and_validation() {
    ordered_sum(DeviceType::Cpu);
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn metal_ordered_lookup_preserves_rounding_views_and_lazy_validation() {
    ordered_sum(DeviceType::Gpu);
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn batched_codebook_norms_preserve_per_depth_rounding_at_released_geometry() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let values = (0..31 * 1024 * 512)
        .map(|i| ((i * 17 % 199) as f32 - 99.) / 31.)
        .collect::<Vec<_>>();
    let table = MlxTensor::from_f32_slice(&values, &[31, 1024, 512], &stream).unwrap();
    let batched = MlxTensor::sum_axis(&table.square(&stream).unwrap(), -1, false, &stream).unwrap();
    for depth in 0..31 {
        let means = table.index(&[Index::At(depth)], &stream).unwrap();
        let ordinary = MlxTensor::sum_axis(&means.square(&stream).unwrap(), -1, false, &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap();
        let cached = batched
            .index(&[Index::At(depth)], &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap();
        assert_eq!(ordinary, cached, "depth {depth} changed rounding");
    }
}
