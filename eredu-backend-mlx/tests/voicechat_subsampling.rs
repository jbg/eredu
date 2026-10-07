//! Canonical OIHW subsampling parameters reproduce pinned NVIDIA window outputs.

use eredu_architectures::voicechat::subsampling::{
    CausalSubsampling, SubsamplingDimensions, SubsamplingError,
};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{
    Index, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor, TensorElementType,
};
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

fn compare(actual: &MlxTensor, expected: &Value, tolerance: &Value, stream: &Stream) {
    let expected_tensor = tensor(expected);
    assert_eq!(actual.shape(), expected_tensor.shape());
    let expected = expected_tensor.to_f32_vec(stream).unwrap();
    let actual = actual.to_f32_vec(stream).unwrap();
    let absolute = tolerance["absolute"].as_f64().unwrap() as f32;
    let relative = tolerance["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "element {index}: {actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn cpu_causal_subsampling_matches_nvidia_full_and_streaming_windows() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let subsampling = &fixture["subsampling"];
    let geometry = &subsampling["geometry"];
    let dimensions = SubsamplingDimensions {
        features: geometry["features"].as_i64().unwrap() as i32,
        channels: geometry["channels"].as_i64().unwrap() as i32,
        width: geometry["width"].as_i64().unwrap() as i32,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut module = CausalSubsampling::<Backend>::new(dimensions, &stream).unwrap();
    module.visit_parameters_mut(&mut Bind(&subsampling["parameters"]));
    let input = tensor(&subsampling["input"]);
    let whole = module.forward(&input, &stream).unwrap();
    compare(
        &whole,
        &subsampling["output"],
        &fixture["tolerance"],
        &stream,
    );
    let mut emitted = Vec::new();
    for step in subsampling["steps"].as_array().unwrap() {
        let input = tensor(&step["input"]);
        let output = module.forward(&input, &stream).unwrap();
        compare(&output, &step["raw_output"], &fixture["tolerance"], &stream);
        assert_eq!(
            output.dim(1),
            dimensions.output_frames(input.dim(1)).unwrap()
        );
        let drop = step["drop"].as_i64().unwrap() as i32;
        let retained = output
            .index(
                &[Index::Full, Index::Range(drop, output.dim(1)), Index::Full],
                &stream,
            )
            .unwrap();
        compare(&retained, &step["output"], &fixture["tolerance"], &stream);
        emitted.push(retained);
    }
    let streaming = MlxTensor::concatenate(&emitted, 1, &stream).unwrap();
    compare(
        &streaming,
        &subsampling["output"],
        &fixture["tolerance"],
        &stream,
    );

    let half = input.cast_float(TensorElementType::F16, &stream).unwrap();
    assert!(matches!(
        module.forward(&half, &stream),
        Err(SubsamplingError::Precision(_))
    ));
    // The first projection remains F32: checking only its precision would miss
    // a mixed-precision parameter in a later convolution.
    struct HalfBias<'s>(&'s Stream);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for HalfBias<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
            if metadata.id.as_str().ends_with("conv.2.bias") {
                *value = value.cast_float(TensorElementType::F16, self.0).unwrap();
            }
        }
    }
    module.visit_parameters_mut(&mut HalfBias(&stream));
    assert!(matches!(
        module.forward(&input, &stream),
        Err(SubsamplingError::Precision(_))
    ));
}
