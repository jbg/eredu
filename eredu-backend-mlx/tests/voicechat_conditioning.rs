//! Native static character conditioning and gated fusion against pinned source methods.
use eredu_architectures::voicechat::tts_conditioning::{
    CharacterDimensions, CharacterEmbedding, CharacterFlags, CharacterMapping, GatedFusion,
};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor, TensorElementType};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;

fn tensor(value: &Value) -> MlxTensor {
    let shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&values, &shape))
}
fn integers(value: &Value) -> Vec<i32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}
struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(output.shape(), replacement.shape());
        *output = replacement;
    }
}
fn compare(actual: &MlxTensor, expected: &Value, fixture: &Value, context: &Stream) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let abs = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let rel = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
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
fn cpu_character_pool_scatter_and_gated_fusion_match_nvidia() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CONDITIONING_REFERENCE_JSON).unwrap();
    let context = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let vocabulary = fixture["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(id, token)| (id as u32, token.as_str().unwrap().to_owned()))
        .collect();
    let mapping = CharacterMapping::new(
        &vocabulary,
        CharacterFlags {
            continuation: integers(&fixture["continuation_flags"]),
            continuation_pad: fixture["continuation_pad"].as_i64().unwrap() as i32,
            special: integers(&fixture["special_flags"]),
            special_pad: fixture["special_pad"].as_i64().unwrap() as i32,
        },
    )
    .unwrap();
    let tokens = integers(&fixture["tokens"]);
    let mask = fixture["selected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect::<Vec<_>>();
    let batch = mapping.prepare(&tokens, Some(&mask), 2, 4, 1024).unwrap();
    let mut embedding = CharacterEmbedding::<Backend>::new(
        CharacterDimensions {
            characters: mapping.characters().len() as i32,
            width: 4,
            output_width: 4,
            use_continuation: true,
            use_special: true,
        },
        &context,
    )
    .unwrap();
    embedding.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let output = embedding
        .forward(
            &batch,
            |embedded, attention| {
                compare(embedded, &fixture["embedded"], &fixture, &context);
                let expected = fixture["attention_mask"]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| i32::from(v.as_bool().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(attention.to_i32_vec(&context).unwrap(), expected);
                Ok(tensor(&fixture["hidden"]))
            },
            &context,
        )
        .unwrap();
    compare(&output, &fixture["character_output"], &fixture, &context);
    let mut fusion = GatedFusion::<Backend>::new(4, &context).unwrap();
    fusion.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let audio = tensor(&fixture["audio"]);
    compare(
        &fusion.forward(&audio, &output, &context).unwrap(),
        &fixture["fusion_output"],
        &fixture,
        &context,
    );
    assert!(fusion
        .forward(
            &audio.cast_float(TensorElementType::F16, &context).unwrap(),
            &output,
            &context
        )
        .is_err());
    struct HalfBias<'s>(&'s Stream);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for HalfBias<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
            if metadata.id.as_str().ends_with("text_proj.bias") {
                *value = value.cast_float(TensorElementType::F16, self.0).unwrap();
            }
        }
    }
    fusion.visit_parameters_mut(&mut HalfBias(&context));
    assert!(fusion.forward(&audio, &output, &context).is_err());
}
