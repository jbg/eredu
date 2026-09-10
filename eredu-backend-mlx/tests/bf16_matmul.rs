//! Released-reference regression for CPU reduced-precision accumulation.
use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, Dtype, Stream};

#[test]
fn cpu_largest_partition_preserves_equal_cutoff_keys() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/topk_partition.safetensors"
        ),
        &stream,
    )
    .unwrap();
    for (width, count) in [(64, 4), (100, 8), (384, 8)] {
        let input = fixture[&format!("{width}.input")]
            .multiply(Array::from_f32(-1.0), &stream)
            .unwrap();
        let ids = safemlx::ops::argpartition_axis(input, count - 1, -1, &stream)
            .unwrap()
            .try_index_device((.., ..count), &stream)
            .unwrap()
            .as_dtype(Dtype::Int32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[&format!("{width}.indices")].evaluated().unwrap();
        assert_eq!(
            ids.as_slice::<i32>(),
            expected.as_slice::<i32>(),
            "width {width}"
        );
    }
}

#[test]
fn cpu_bf16_weighted_rms_preserves_reduction_rounding() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_normalization.safetensors"
        ),
        &stream,
    )
    .unwrap();
    for width in [32, 768, 1280, 2560] {
        let input = fixture[&format!("{width}.input")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap();
        let weight = fixture[&format!("{width}.weight")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap();
        let squared = input.multiply(&input, &stream).unwrap();
        let variance = safemlx::ops::mean_axis(squared, -1, true, &stream).unwrap();
        let inverse = variance
            .add(Array::from_f32(1e-6), &stream)
            .unwrap()
            .rsqrt(&stream)
            .unwrap();
        let output = input
            .multiply(&inverse, &stream)
            .unwrap()
            .multiply(&weight, &stream)
            .unwrap()
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap();
        for (name, actual) in [
            ("variance", variance),
            ("inverse", inverse),
            ("output", output),
        ] {
            let actual = actual
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let expected = fixture[&format!("{width}.{name}")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            assert_eq!(
                actual.as_slice::<f32>(),
                expected.as_slice::<f32>(),
                "width={width}, {name}"
            );
        }
    }
}

#[test]
fn cpu_bf16_matrix_products_preserve_full_precision_across_layouts() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_matmul.safetensors"
        ),
        &stream,
    )
    .unwrap();
    for width in [7, 24, 32, 128, 768, 2560] {
        let a = &fixture[&format!("{width}.input")];
        let w = &fixture[&format!("{width}.weight")];
        for rows in [1, 2, 3] {
            let a = a.try_index_device((..rows, ..), &stream).unwrap();
            let expected = &fixture[&format!("{width}.output.{rows}")];
            let mut baseline = None;
            for transpose_a in [false, true] {
                let a = if transpose_a {
                    a.transpose(&stream)
                        .unwrap()
                        .contiguous(false, &stream)
                        .unwrap()
                        .transpose(&stream)
                        .unwrap()
                } else {
                    a.clone()
                };
                for transpose_b in [false, true] {
                    let b = w.transpose(&stream).unwrap();
                    let b = if transpose_b {
                        b
                    } else {
                        b.contiguous(false, &stream).unwrap()
                    };
                    let output = a.matmul(&b, &stream).unwrap();
                    let actual = output
                        .as_dtype(Dtype::Float32, &stream)
                        .unwrap()
                        .into_evaluated()
                        .unwrap();
                    let reference = expected
                        .as_dtype(Dtype::Float32, &stream)
                        .unwrap()
                        .into_evaluated()
                        .unwrap();
                    let values = actual.as_slice::<f32>();
                    let reference = reference.as_slice::<f32>();
                    let error = values
                        .iter()
                        .zip(reference)
                        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                        .sum::<f64>();
                    let norm = reference.iter().map(|&y| f64::from(y).powi(2)).sum::<f64>();
                    assert!((error / norm).sqrt() < 1e-4, "width={width}, rows={rows}, A={transpose_a}, B={transpose_b}: relative L2 {}", (error/norm).sqrt());
                    if let Some(baseline) = &baseline {
                        assert_eq!(values, baseline, "physical layout changed accumulation");
                    } else {
                        baseline = Some(values.to_vec());
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_bf16_rotary_preserves_inverse_frequency_products() {
    use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use eredu_nn::{
        NeuralBackend, RotaryAlgorithm, RotaryArithmetic, RotaryOperator, RotaryPosition,
        RotarySpec,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_rotary.safetensors"
        ),
        &stream,
    )
    .unwrap();
    let mut rotary = MlxNeuralBackend::rotary(
        RotarySpec {
            dimensions: 64,
            base: 1_000_000.0,
            traditional: false,
            arithmetic: RotaryArithmetic::InputProducts,
            algorithm: RotaryAlgorithm::Yarn {
                factor: 16.0,
                original_max_positions: 8192,
                beta_fast: 128.0,
                beta_slow: 4.0,
                amplitude: 1.2772589,
                truncate: true,
            },
        },
        &stream,
    )
    .unwrap();
    for offset in [0, 128, 32768] {
        let input = MlxTensor::from_array(fixture[&format!("{offset}.input")].clone());
        let output = rotary
            .forward(&input, RotaryPosition::Offset(offset), &stream)
            .unwrap();
        let actual = output
            .as_array()
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[&format!("{offset}.output")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        assert_eq!(
            actual.as_slice::<f32>(),
            expected.as_slice::<f32>(),
            "offset {offset}"
        );
    }
}

#[test]
fn cpu_softmax_reduction_preserves_bf16_probability_rounding() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_softmax.safetensors"
        ),
        &stream,
    )
    .unwrap();
    for width in [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 64, 181, 193, 207, 513,
    ] {
        let output =
            safemlx::ops::softmax_axis(&fixture[&format!("{width}.input")], -1, true, &stream)
                .unwrap()
                .as_dtype(Dtype::Bfloat16, &stream)
                .unwrap();
        let actual = output
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[&format!("{width}.output")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let mismatches: Vec<_> = actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "width {width}: {} mismatches; first {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(12)]
        );
    }
}

#[test]
fn cpu_bf16_attention_query_tiles_preserve_complete_row_rounding() {
    use eredu_backend_mlx::backend::nn::attention::attention_with_softcap;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_attention.safetensors"
        ),
        &stream,
    )
    .unwrap();
    for width in [6, 7, 8, 9, 10, 93, 193, 513] {
        let allowed: Vec<bool> = (0..width)
            .flat_map(|q| (0..width).map(move |k| k <= q))
            .collect();
        let mask = Array::from_slice(&allowed, &[width, width]);
        let output = attention_with_softcap(
            &fixture[&format!("{width}.queries")],
            &fixture[&format!("{width}.keys")],
            &fixture[&format!("{width}.values")],
            0.125,
            Some(&mask),
            None,
            None,
            eredu_nn::AttentionArithmetic::InputScores,
            &stream,
        )
        .unwrap();
        let actual = output
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[&format!("{width}.output")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let mismatches: Vec<_> = actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "width {width}: {} mismatches; first {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(12)]
        );
    }
}

#[test]
fn cpu_bf16_default_rotary_preserves_inverse_frequency_products() {
    use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
    use eredu_nn::{
        NeuralBackend, RotaryAlgorithm, RotaryArithmetic, RotaryOperator, RotaryPosition,
        RotarySpec,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_rotary.safetensors"
        ),
        &stream,
    )
    .unwrap();
    let mut rotary = MlxNeuralBackend::rotary(
        RotarySpec {
            dimensions: 128,
            base: 10_000_000.0,
            traditional: false,
            arithmetic: RotaryArithmetic::InputProducts,
            algorithm: RotaryAlgorithm::Default,
        },
        &stream,
    )
    .unwrap();
    for offset in [0, 128, 32768] {
        let input = MlxTensor::from_array(fixture[&format!("default.{offset}.input")].clone());
        let output = rotary
            .forward(&input, RotaryPosition::Offset(offset), &stream)
            .unwrap();
        let actual = output
            .as_array()
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[&format!("default.{offset}.output")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        assert_eq!(
            actual.as_slice::<f32>(),
            expected.as_slice::<f32>(),
            "offset {offset}"
        );
    }
}

#[test]
fn cpu_sigmoid_scores_preserve_normalized_coefficient_rounding() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/f32_sigmoid.safetensors"
        ),
        &stream,
    )
    .unwrap();
    let scores = safemlx::ops::sigmoid(&fixture["input"], &stream).unwrap();
    let selected =
        safemlx::ops::indexing::take_along_axis(&scores, &fixture["ids"], -1, &stream).unwrap();
    let coefficients = selected
        .divide(selected.sum_axis(-1, true, &stream).unwrap(), &stream)
        .unwrap()
        .multiply(Array::from_f32(2.5), &stream)
        .unwrap()
        .as_dtype(Dtype::Bfloat16, &stream)
        .unwrap();
    for (name, values) in [("scores", scores), ("coefficients", coefficients)] {
        let actual = values
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[name]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let mismatches = actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(mismatches, 0, "{name}");
    }
}
