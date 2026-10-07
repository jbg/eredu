//! Original cached EAR-TTS composition on native CPU tensors and shared caches.
use super::super::*;
use super::voicechat_transformer::{compare, geometry, tensor};
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    tts_generator::TtsGenerator,
    tts_ingress::{PromptLayout, SpeakerPrompt},
};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use safemlx::{Device, DeviceType};
use serde_json::Value;

fn codes(value: &Value, stream: &Stream) -> MlxTensor {
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
    MlxTensor::from_i32_slice(&values, &shape, stream).unwrap()
}
struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        // Supplied-subword fixtures deliberately omit the unexecuted character module.
        if self.0.get(metadata.id.as_str()).is_none()
            && metadata
                .id
                .as_str()
                .starts_with("tts_model.tts_model.embed_subword.")
        {
            return;
        }
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(
            replacement.shape(),
            value.shape(),
            "{}",
            metadata.id.as_str()
        );
        *value = replacement;
    }
}
fn compare_codes(actual: &MlxTensor, expected: &Value, stream: &Stream) {
    assert_eq!(actual.shape(), codes(expected, stream).shape());
    assert_eq!(
        actual.to_i32_vec(stream).unwrap(),
        expected["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    );
}
fn compare_cache(caches: &[ConcatKeyValueCache], step: &Value, fixture: &Value, stream: &Stream) {
    for (cache, expected) in caches.iter().zip(step["cache"].as_array().unwrap()) {
        assert_eq!(
            KeyValueCache::offset(cache),
            expected["seen"].as_i64().unwrap() as i32
        );
        let arrays = KeyValueCache::retained_arrays(cache);
        assert_eq!(arrays.len(), 2);
        compare(
            &MlxTensor::from_array(arrays[0].clone()),
            &expected["keys"],
            fixture,
            stream,
        );
        compare(
            &MlxTensor::from_array(arrays[1].clone()),
            &expected["values"],
            fixture,
            stream,
        );
    }
}

#[test]
fn cpu_voicechat_tts_generator_matches_original_cached_refinement() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut config = VoiceChatConfig::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    )))
    .unwrap()
    .tts;
    config.backbone = geometry(&fixture["transformer_config"], true);
    // This fixture supplies already encoded subwords. Keep construction compact
    // without binding/executing a synthetic character encoder.
    let transformer_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    config.character_encoder =
        geometry(&transformer_fixture["t5gemma"]["config"]["encoder"], false);
    config.latent_size = 4;
    config.codebook_size = 5;
    config.mog.intermediate_size = 12;
    config.mog.num_layers = 3;
    config.mog.low_rank = 2;
    config.mog.num_predictions = 4;
    let mut model = TtsGenerator::<MlxNeuralBackend>::new(&config, 5, 5, &stream).unwrap();
    let mut bind = Bind(&fixture["parameters"]);
    model.conditioning.visit_parameters_mut(&mut bind);
    model.transformer.visit_parameters_mut(&mut bind);
    model.refinement.visit_parameters_mut(&mut bind);
    model.mog.visit_parameters_mut(&mut bind);
    let mut caches = [
        ConcatKeyValueCache::new_for_sliding_attention(3),
        ConcatKeyValueCache::new(),
    ];
    let calls = fixture["calls"].as_array().unwrap();
    let first = &calls[0];
    let mask = first["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect::<Vec<_>>();
    let layout = PromptLayout::new(1, 5, &mask).unwrap();
    let hidden = model
        .prefill(
            &codes(&first["input_codes"], &stream),
            &layout,
            SpeakerPrompt::Aria,
            Some(&tensor(&first["supplied_subwords"])),
            caches.as_mut_slice(),
            &stream,
        )
        .unwrap();
    compare(&hidden, &first["hidden"], &fixture, &stream);
    compare_cache(&caches, first, &fixture, &stream);
    for step in &calls[1..] {
        let condition = tensor(&step["supplied_subwords"]);
        let selected = step["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool().unwrap());
        let mut frame = model
            .begin_frame(
                &codes(&step["input_codes"], &stream),
                selected.then_some(&condition),
                caches.as_mut_slice(),
                &stream,
            )
            .unwrap();
        let lanes = MlxTensor::concatenate(
            &[
                frame.conditional().clone(),
                frame.unconditional().unwrap().clone(),
            ],
            0,
            &stream,
        )
        .unwrap();
        compare(&lanes, &step["hidden"], &fixture, &stream);
        for expected in step["refinement"].as_array().unwrap() {
            frame = model
                .advance_frame(
                    &frame,
                    |slot, logits: &MlxTensor, top_p| {
                        assert_eq!(slot, expected["slot"].as_u64().unwrap() as usize);
                        assert_eq!(top_p, fixture["policy"]["top_p_or_k"].as_f64().unwrap());
                        compare(logits, &expected["logits"], &fixture, &stream);
                        Ok((
                            codes(&expected["mixture_indices"], &stream)
                                .reshape(&[logits.dim(0), logits.dim(1)], &stream)
                                .unwrap(),
                            tensor(&expected["gaussian_draw"]),
                        ))
                    },
                    &stream,
                )
                .unwrap();
            compare_codes(frame.refinement().codes(), &expected["codes"], &stream);
        }
        compare_codes(frame.codes().unwrap(), &step["codes"], &stream);
        // Refinement advances no causal positions and changes neither guidance cache.
        compare_cache(&caches, step, &fixture, &stream);
    }
}
