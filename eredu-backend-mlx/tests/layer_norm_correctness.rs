//! Native affine normalization retains dtype promotion and scalar equations.
use eredu_backend_mlx::MlxTensor;
use eredu_nn::Tensor;
use safemlx::{Array, Device, DeviceType, Dtype, Stream};

fn tensor(values: &[f32], shape: &[i32], dtype: Dtype, stream: &Stream) -> MlxTensor {
    MlxTensor::from_array(
        Array::from_slice(values, shape)
            .as_dtype(dtype, stream)
            .unwrap(),
    )
}
fn values(value: &MlxTensor, stream: &Stream) -> Vec<f32> {
    value
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .unwrap()
        .into_evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}
fn promotion_cases(stream: &Stream) {
    let data: Vec<f32> = (0..21)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.125)
        .collect();
    for (input_dtype, weight_dtype, bias_dtype, expected_dtype) in [
        (
            Dtype::Float32,
            Some(Dtype::Float32),
            Some(Dtype::Float32),
            Dtype::Float32,
        ),
        (
            Dtype::Bfloat16,
            Some(Dtype::Bfloat16),
            Some(Dtype::Bfloat16),
            Dtype::Bfloat16,
        ),
        (
            Dtype::Bfloat16,
            Some(Dtype::Float32),
            Some(Dtype::Bfloat16),
            Dtype::Float32,
        ),
        // The released native fast operator casts a bias-only affine parameter
        // to the input representation; bias alone does not select output dtype.
        (Dtype::Bfloat16, None, Some(Dtype::Float32), Dtype::Bfloat16),
        (Dtype::Float32, None, None, Dtype::Float32),
    ] {
        let input = tensor(&data, &[3, 7], input_dtype, stream);
        let weight = weight_dtype
            .map(|dtype| tensor(&[0.5, 1.0, 0.75, 1.25, 0.25, 1.5, 2.0], &[7], dtype, stream));
        let bias = bias_dtype.map(|dtype| {
            tensor(
                &[0.125, -0.25, 0.375, 0.0, 0.25, -0.125, 0.5],
                &[7],
                dtype,
                stream,
            )
        });
        let actual =
            MlxTensor::layer_norm(&input, weight.as_ref(), bias.as_ref(), 1e-6, stream).unwrap();
        let reference = safemlx::fast::layer_norm(
            input.as_array(),
            weight.as_ref().map(MlxTensor::as_array),
            bias.as_ref().map(MlxTensor::as_array),
            1e-6,
            stream,
        )
        .unwrap();
        assert_eq!(actual.as_array().dtype(), expected_dtype);
        let actual = values(&actual, stream);
        assert_eq!(actual, values(&MlxTensor::from_array(reference), stream));
        assert!(actual.iter().all(|x| x.is_finite()));
        assert!(actual.iter().any(|x| x.abs() > 0.25));
        if input_dtype == Dtype::Float32 && weight.is_none() && bias.is_none() {
            for (input, actual) in data.chunks(7).zip(actual.chunks(7)) {
                let mean = input.iter().map(|x| f64::from(*x)).sum::<f64>() / 7.0;
                let variance = input
                    .iter()
                    .map(|x| (f64::from(*x) - mean).powi(2))
                    .sum::<f64>()
                    / 7.0;
                for (x, y) in input.iter().zip(actual) {
                    let expected = (f64::from(*x) - mean) / (variance + 1e-6).sqrt();
                    assert!((f64::from(*y) - expected).abs() < 2e-6);
                }
            }
        }
    }
}

#[test]
fn cpu_normalization_preserves_promotion_and_scalar_equations() {
    promotion_cases(&Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)));
}
