use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use safemlx::{Device, DeviceType, Stream};

use eredu_architectures::voicechat::{
    tts_conditioning::GatedFusion,
    tts_ingress::{PromptLayout, SpeakerPrompt, TtsIngress},
    tts_rvq::RvqRefinement,
};
use serde_json::Value;

fn tensor(value: &Value, context: &Stream) -> MlxTensor {
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
    MlxTensor::from_f32_slice(&values, &shape, context).unwrap()
}
fn codes(value: &Value, context: &Stream) -> MlxTensor {
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
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    MlxTensor::from_i32_slice(&values, &shape, context).unwrap()
}
struct Bind<'v> {
    parameters: &'v Value,
    speaker: &'v Value,
    context: &'v Stream,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        let value = if metadata.id.as_str() == "tts_model.audio_prompt_latents.Aria" {
            self.speaker
        } else {
            &self.parameters[metadata.id.as_str()]
        };
        let replacement = tensor(value, self.context);
        assert_eq!(
            replacement.shape(),
            output.shape(),
            "{}",
            metadata.id.as_str()
        );
        *output = replacement;
    }
}
fn compare(actual: &MlxTensor, expected: &Value, fixture: &Value, context: &Stream) {
    let expected = tensor(expected, context)
        .broadcast_to(actual.shape(), context)
        .unwrap();
    let tolerance = &fixture["comparison"];
    let absolute = tolerance["absolute_tolerance"].as_f64().unwrap() as f32;
    let relative = tolerance["relative_tolerance"].as_f64().unwrap() as f32;
    for (actual, expected) in actual
        .to_f32_vec(context)
        .unwrap()
        .iter()
        .zip(expected.to_f32_vec(context).unwrap())
    {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn voicechat_tts_ingress_matches_original_prompt_decode_and_guidance_lanes() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::INGRESS_REFERENCE_JSON).unwrap();
    let context = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut binder = Bind {
        parameters: &fixture["parameters"],
        speaker: &fixture["cases"][0]["supplied_speaker"],
        context: &context,
    };
    let mut feedback = RvqRefinement::<Backend>::new(5, 4, 4, &context).unwrap();
    feedback.visit_parameters_mut(&mut binder);
    let mut ingress = TtsIngress::<Backend>::new(4, 5, &context).unwrap();
    ingress.visit_parameters_mut(&mut binder);
    let mut fusion = GatedFusion::<Backend>::new(4, &context).unwrap();
    fusion.visit_parameters_mut(&mut binder);
    for case in fixture["cases"].as_array().unwrap() {
        let code = codes(&case["code"], &context);
        let condition = tensor(&case["supplied_subwords"], &context);
        let any = case["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool().unwrap());
        let condition = any.then_some(&condition);
        let lanes = if case["audio_mask"].is_null() {
            ingress
                .decode(&mut feedback, &code, condition, true, &context)
                .unwrap()
        } else {
            let mask = case["audio_mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect::<Vec<_>>();
            let layout = PromptLayout::new(code.dim(0), code.dim(1), &mask).unwrap();
            let speaker = if case["supplied_speaker"].is_null() {
                SpeakerPrompt::ProjectedCodes
            } else {
                SpeakerPrompt::Aria
            };
            ingress
                .prefill(
                    &mut feedback,
                    &code,
                    &layout,
                    speaker,
                    condition,
                    true,
                    &context,
                )
                .unwrap()
        };
        compare(&lanes.audio, &case["audio_embedding"], &fixture, &context);
        compare(&lanes.text, &case["conditioning"], &fixture, &context);
        let fused = fusion.forward(&lanes.audio, &lanes.text, &context).unwrap();
        compare(&fused, &case["backbone"]["inputs"], &fixture, &context);
    }
    // Exact Aria sequence geometry is checked before tensor work.
    let layout = PromptLayout::new(1, 1, &[true]).unwrap();
    let codes = MlxTensor::full_i32(0, &[1, 1, 31], &context).unwrap();
    assert!(ingress
        .prefill(
            &mut feedback,
            &codes,
            &layout,
            SpeakerPrompt::Aria,
            None,
            true,
            &context
        )
        .is_err());
}
