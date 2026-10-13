mod support {
    #[allow(dead_code)]
    pub mod numeric_backend;
}

use eredu_codec::prvq::{DistanceMetric, Geometry, ResidualQuantizer};
use eredu_nn::{Index, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use serde_json::Value;
use support::numeric_backend::{activate, Context, FailAt, NumericTensor, ReferenceBackend};

// The validation crate owns this independent artifact. Including its bytes
// avoids making a codec development dependency cycle through evaluation.
const REFERENCE: &str = eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON;
const MOG_REFERENCE: &str = eredu_evaluation::voicechat::MOG_REFERENCE_JSON;

fn tensor(value: &Value) -> NumericTensor {
    let shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect();
    NumericTensor::f32(values, &shape)
}

struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(output.shape(), replacement.shape());
        assert!(!metadata.trainable);
        *output = replacement;
    }
}

fn compare(actual: &NumericTensor, expected: &Value, tolerance: &Value, context: &Context) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let abs = tolerance["absolute"].as_f64().unwrap() as f32;
    let rel = tolerance["relative"].as_f64().unwrap() as f32;
    for (actual, expected) in actual
        .to_f32_vec(context)
        .unwrap()
        .into_iter()
        .zip(expected.to_f32_vec(context).unwrap())
    {
        assert!(
            (actual - expected).abs() <= abs + rel * expected.abs(),
            "{actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn fixed_codebooks_match_original_encode_and_ordered_decode() {
    let context = Context::new(FailAt::None);
    activate(&context);
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut module = ResidualQuantizer::<ReferenceBackend>::new(
        Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        "tts_model.audio_codec.prvq",
        &context,
    )
    .unwrap();
    module.visit_parameters_mut(&mut Bind(&reference["parameters"]));
    let latent = tensor(&reference["input"]);
    let result = module.quantize(&latent, &context).unwrap();
    assert_eq!(
        result.codes.to_i32_vec(&context).unwrap(),
        reference["codes"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    );
    for (actual, name) in [
        (&result.quantized, "quantized"),
        (&result.residual, "residual"),
    ] {
        compare(actual, &reference[name], &reference["tolerance"], &context);
    }
    compare(
        &module.decode(&result.codes, &context).unwrap(),
        &reference["decoded"],
        &reference["tolerance"],
        &context,
    );
    let prefix = result
        .codes
        .index(&[Index::Full, Index::Full, Index::Range(0, 2)], &context)
        .unwrap();
    compare(
        &module.decode(&prefix, &context).unwrap(),
        &reference["prefix_decoded"],
        &reference["tolerance"],
        &context,
    );
    assert_eq!(
        latent.to_f32_vec(&context).unwrap(),
        tensor(&reference["input"]).to_f32_vec(&context).unwrap()
    );
    assert!(module
        .quantize_range(&latent, 2..4, DistanceMetric::SquaredEuclidean, &context)
        .is_err());
    assert!(module
        .quantize_range(&latent, 2..2, DistanceMetric::SquaredEuclidean, &context)
        .is_err());
    assert!(module
        .decode(&NumericTensor::f32(vec![0.; 12], &[2, 2, 3]), &context)
        .is_err());
}

#[test]
fn suffix_quantization_starts_with_fresh_residual_and_uses_first_minimum() {
    let context = Context::new(FailAt::None);
    activate(&context);
    let reference: Value = serde_json::from_str(MOG_REFERENCE).unwrap();
    let cases = &reference["depthsum_cases"];
    let mut module = ResidualQuantizer::<ReferenceBackend>::new(
        Geometry {
            depth: 3,
            cardinality: 4,
            channels: 2,
        },
        "prvq",
        &context,
    )
    .unwrap();
    let tables = tensor(&cases["embeddings"]);
    for (depth, book) in module.codebooks_mut().enumerate() {
        book.means.replace(
            tables
                .index(
                    &[Index::At(depth as i32), Index::Full, Index::Full],
                    &context,
                )
                .unwrap(),
        );
        book.variance.replace(NumericTensor::f32(vec![0.1], &[]));
    }
    let residual = tensor(&cases["residual"]);
    let metric = DistanceMetric::CodebookNormMinusTwiceDot;
    let full = module
        .quantize_range(&residual, 0..3, metric, &context)
        .unwrap();
    assert_eq!(
        full.codes.to_i32_vec(&context).unwrap(),
        cases["full_codes"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    );
    let suffix = module
        .quantize_range(&residual, 1..3, metric, &context)
        .unwrap();
    let expected = cases["suffix_codes"]["values"]
        .as_array()
        .unwrap()
        .chunks(3)
        .flat_map(|row| row[1..].iter().map(|n| n.as_i64().unwrap() as i32))
        .collect::<Vec<_>>();
    assert_eq!(suffix.codes.to_i32_vec(&context).unwrap(), expected);
}

#[test]
fn channel_score_reduction_rejects_changed_vocabulary_shape_and_precision() {
    use eredu_codec::prvq::quantize_residual_with_score_reduction;
    let context = Context::new(FailAt::None);
    activate(&context);
    let residual = NumericTensor::f32(vec![0.5, -0.25], &[1, 1, 2]);
    let means = NumericTensor::f32(vec![0.0, 0.0, 1.0, 1.0], &[2, 2]);
    for replacement in [
        NumericTensor::full_f32(0.0, &[1, 3], &context).unwrap(),
        NumericTensor::full_i32(0, &[1, 2], &context).unwrap(),
    ] {
        assert!(matches!(
            quantize_residual_with_score_reduction(
                &residual,
                &means,
                DistanceMetric::CodebookNormMinusTwiceDot,
                &context,
                |_| Ok(replacement),
            ),
            Err(eredu_codec::Error::InvalidShape(_))
        ));
    }
}
