//! CPU layer numerics and both streaming caches against pinned NVIDIA execution.

use eredu_architectures::voicechat::conformer::{
    ConformerBlock, ConformerDimensions, ConformerState,
};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
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
fn cpu_conformer_matches_nvidia_layer_and_streaming_caches() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let conformer = &fixture["conformer"];
    let geometry = &conformer["geometry"];
    let dimensions = ConformerDimensions {
        width: geometry["width"].as_i64().unwrap() as i32,
        heads: geometry["heads"].as_i64().unwrap() as i32,
        ff_hidden: geometry["feed_forward"].as_i64().unwrap() as i32,
        conv_kernel: geometry["kernel"].as_i64().unwrap() as i32,
        left_context: geometry["left_context"].as_i64().unwrap() as i32,
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut block = ConformerBlock::<Backend>::new(dimensions, 0, &stream).unwrap();
    block.visit_parameters_mut(&mut Bind(&conformer["parameters"]));
    let (whole, _) = block
        .forward(
            &tensor(&conformer["input"]),
            &ConformerState::default(),
            &stream,
        )
        .unwrap();
    compare(&whole, &conformer["output"], &fixture["tolerance"], &stream);
    let mut state = ConformerState::default();
    let mut first_snapshot = None;
    for (index, step) in conformer["steps"].as_array().unwrap().iter().enumerate() {
        let (output, next) = block
            .forward(&tensor(&step["input"]), &state, &stream)
            .unwrap();
        compare(&output, &step["output"], &fixture["tolerance"], &stream);
        compare(
            next.channel().unwrap(),
            &step["channel_cache"],
            &fixture["tolerance"],
            &stream,
        );
        compare(
            &next
                .convolution()
                .unwrap()
                .transpose_axes(&[0, 2, 1], &stream)
                .unwrap(),
            &step["time_cache"],
            &fixture["tolerance"],
            &stream,
        );
        assert_eq!(
            next.valid_channel(),
            step["valid_cache"].as_u64().unwrap() as usize
        );
        if index == 0 {
            first_snapshot = Some(next.clone());
        }
        state = next;
    }
    let first_snapshot = first_snapshot.unwrap();
    compare(
        first_snapshot.channel().unwrap(),
        &conformer["steps"][0]["channel_cache"],
        &fixture["tolerance"],
        &stream,
    );
    compare(
        &first_snapshot
            .convolution()
            .unwrap()
            .transpose_axes(&[0, 2, 1], &stream)
            .unwrap(),
        &conformer["steps"][0]["time_cache"],
        &fixture["tolerance"],
        &stream,
    );
}
