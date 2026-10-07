//! Native gather, argmin and residual arithmetic against the pinned codec class.

use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_codec::prvq::{Geometry, ResidualQuantizer};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor, TensorElementType};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;

fn tensor(value: &Value) -> MlxTensor {
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
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&values, &shape))
}

struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(output.shape(), replacement.shape());
        *output = replacement;
    }
}

#[test]
fn cpu_prvq_matches_original_codec_class_and_rejects_mixed_precision() {
    let reference: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut module = ResidualQuantizer::<Backend>::new(
        Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        "tts_model.audio_codec.prvq",
        &stream,
    )
    .unwrap();
    module.visit_parameters_mut(&mut Bind(&reference["parameters"]));
    let latent = tensor(&reference["input"]);
    let result = module.quantize(&latent, &stream).unwrap();
    assert_eq!(result.codes.element_type(), Some(TensorElementType::I32));
    assert_eq!(
        result.codes.to_i32_vec(&stream).unwrap(),
        reference["codes"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    );
    let decoded = module.decode(&result.codes, &stream).unwrap();
    let abs = reference["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let rel = reference["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (actual, name) in [
        (&result.quantized, "quantized"),
        (&result.residual, "residual"),
        (&decoded, "decoded"),
    ] {
        let expected = tensor(&reference[name]);
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual
            .to_f32_vec(&stream)
            .unwrap()
            .into_iter()
            .zip(expected.to_f32_vec(&stream).unwrap())
        {
            assert!(
                (actual - expected).abs() <= abs + rel * expected.abs(),
                "{actual} versus NVIDIA {expected}"
            );
        }
    }
    assert!(module
        .quantize(
            &latent.cast_float(TensorElementType::F16, &stream).unwrap(),
            &stream
        )
        .is_err());
    let invalid = MlxTensor::from_i32_slice(&[5], &[1, 1, 1], &stream).unwrap();
    assert!(module.decode(&invalid, &stream).is_err());
    let half = module
        .codebooks()
        .nth(1)
        .unwrap()
        .means
        .as_ref()
        .cast_float(TensorElementType::F16, &stream)
        .unwrap();
    module.codebooks_mut().nth(1).unwrap().means.replace(half);
    assert!(module.quantize(&latent, &stream).is_err());
}
