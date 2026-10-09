//! Composed NVIDIA cached TTS transitions with explicit sampled decisions.
use super::voicechat_transformer::{geometry, CausalCache};
use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    tts_generator::TtsGenerator,
    tts_ingress::{PromptLayout, SpeakerPrompt},
    tts_mog::TtsError,
};
use serde_json::Value;

pub(super) fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
}
pub(super) fn codes(value: &Value, context: &NumericContext) -> NumericTensor {
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
        .map(|v| v.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    NumericTensor::from_i32_slice(&values, &shape, context).unwrap()
}
struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut NumericTensor) {
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
            target.shape(),
            replacement.shape(),
            "{}",
            metadata.id.as_str()
        );
        *target = replacement;
    }
}
pub(super) fn compare(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let absolute = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let relative = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (i, (actual, expected)) in actual.data.iter().zip(expected.data).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{i}: {actual} versus NVIDIA {expected}"
        );
    }
}
pub(super) fn compare_cache(caches: &[CausalCache], expected: &Value, fixture: &Value) {
    for (cache, expected) in caches.iter().zip(expected.as_array().unwrap()) {
        assert_eq!(cache.offset(), expected["seen"].as_i64().unwrap() as i32);
        compare(cache.0.keys.as_ref().unwrap(), &expected["keys"], fixture);
        compare(
            cache.0.values.as_ref().unwrap(),
            &expected["values"],
            fixture,
        );
    }
}

struct FailedBranch {
    cache: CausalCache,
    fail_after_append: bool,
}
impl RuntimeLayerState<NumericBackend> for FailedBranch {
    type RetainedValues<'a> =
        <NumericCache as RuntimeLayerState<NumericBackend>>::RetainedValues<'a>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.cache.retained_values()
    }
}

impl AttentionCache<NumericTensor> for FailedBranch {
    fn offset(&self) -> i32 {
        self.cache.offset()
    }
    fn max_size(&self) -> Option<i32> {
        self.cache.max_size()
    }
    fn update_for_attention(
        &mut self,
        keys: NumericTensor,
        values: NumericTensor,
        context: &NumericContext,
    ) -> Result<(NumericTensor, NumericTensor), Error> {
        let result = self.cache.update_for_attention(keys, values, context)?;
        if self.fail_after_append {
            return Err(Error::backend("injected post-append failure"));
        }
        Ok(result)
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.cache.attention(request, context)
    }
}

pub(super) fn model(fixture: &Value, context: &NumericContext) -> TtsGenerator<NumericBackend> {
    let mut config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap()
    .tts;
    config.backbone = geometry(&fixture["transformer_config"], true);
    config.character_encoder.hidden_size = 8;
    config.character_encoder.intermediate_size = 12;
    config.character_encoder.num_attention_heads = 2;
    config.character_encoder.num_key_value_heads = 1;
    config.character_encoder.head_dim = 4;
    if fixture.get("character").is_some() {
        config.character_encoder = geometry(&fixture["character"]["config"], false);
    }
    config.latent_size = 4;
    config.codebook_size = 5;
    config.mog.intermediate_size = 12;
    config.mog.low_rank = 2;
    config.mog.num_predictions = 4;
    let characters = fixture["character"]["characters"].as_i64().unwrap_or(16) as i32;
    let mut generator = TtsGenerator::new(&config, characters, 5, context).unwrap();
    let mut bind = Bind(&fixture["parameters"]);
    // The independent oracle supplies character embeddings; its weights do not
    // include a second, unexecuted character tower.
    generator.conditioning.visit_parameters_mut(&mut bind);
    generator.transformer.visit_parameters_mut(&mut bind);
    generator.refinement.visit_parameters_mut(&mut bind);
    generator.mog.visit_parameters_mut(&mut bind);
    if fixture.get("character").is_some() {
        generator
            .character_transformer
            .visit_parameters_mut(&mut bind);
    }
    generator
}

#[test]
fn voicechat_tts_generator_matches_original_cached_frames_and_retained_refinement() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut generator = model(&fixture, &context);
    let calls = fixture["calls"].as_array().unwrap();
    let prompt = &calls[0];
    let mask = prompt["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect::<Vec<_>>();
    let layout = PromptLayout::new(1, 5, &mask).unwrap();
    let mut caches = [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ];
    let hidden = generator
        .prefill(
            &codes(&prompt["input_codes"], &context),
            &layout,
            SpeakerPrompt::Aria,
            Some(&tensor(&prompt["supplied_subwords"])),
            &mut caches,
            &context,
        )
        .unwrap();
    compare(&hidden, &prompt["hidden"], &fixture);
    compare_cache(&caches, &prompt["cache"], &fixture);
    for (index, call) in calls[1..].iter().enumerate() {
        let previous = caches.clone();
        let any = call["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool().unwrap());
        let condition = tensor(&call["supplied_subwords"]);
        if index == 0 {
            let mut branch = previous
                .iter()
                .cloned()
                .enumerate()
                .map(|(layer, cache)| FailedBranch {
                    cache,
                    fail_after_append: layer == 1,
                })
                .collect::<Vec<_>>();
            assert!(generator
                .begin_frame(
                    &codes(&call["input_codes"], &context),
                    any.then_some(&condition),
                    &mut branch,
                    &context
                )
                .is_err());
            assert!(branch
                .iter()
                .all(|cache| cache.offset() == previous[0].offset() + 1));
            // Only the unpublished branch was mutated. The retained parent
            // remains a valid replay origin; the failed branch is discarded.
            compare_cache(&caches, &prompt["cache"], &fixture);
        }
        let initial = generator
            .begin_frame(
                &codes(&call["input_codes"], &context),
                any.then_some(&condition),
                &mut caches,
                &context,
            )
            .unwrap();
        let hidden = NumericTensor::concatenate(
            &[
                initial.conditional().clone(),
                initial.unconditional().unwrap().clone(),
            ],
            0,
            &context,
        )
        .unwrap();
        compare(&hidden, &call["hidden"], &fixture);
        assert_eq!(initial.model_position(), caches[0].offset());
        compare_cache(&caches, &call["cache"], &fixture);
        let retained = caches.clone();
        let expected_steps = call["refinement"].as_array().unwrap();
        let mut controlled = initial.clone();
        for expected in expected_steps {
            let snapshot = controlled.clone();
            let next = generator
                .advance_frame(
                    &controlled,
                    |slot, logits: &NumericTensor, top_p| {
                        assert_eq!(slot, expected["slot"].as_u64().unwrap() as usize);
                        assert_eq!(top_p, 0.95);
                        compare(logits, &expected["logits"], &fixture);
                        Ok((
                            codes(&expected["mixture_indices"], &context)
                                .reshape(&[1, 1], &context)
                                .unwrap(),
                            tensor(&expected["gaussian_draw"]),
                        ))
                    },
                    &context,
                )
                .unwrap();
            assert_eq!(
                next.refinement().codes().data,
                codes(&expected["codes"], &context).data
            );
            // An explicit draw failure leaves the retained frame reusable. The
            // causal cache was already submitted; refinement does not touch it.
            assert!(generator
                .advance_frame(
                    &snapshot,
                    |_, _: &NumericTensor, _| Err(TtsError::Shape("injected draw failure")),
                    &context
                )
                .is_err());
            let retry = generator
                .advance_frame(
                    &snapshot,
                    |_, _: &NumericTensor, _| {
                        Ok((
                            codes(&expected["mixture_indices"], &context)
                                .reshape(&[1, 1], &context)
                                .unwrap(),
                            tensor(&expected["gaussian_draw"]),
                        ))
                    },
                    &context,
                )
                .unwrap();
            assert_eq!(
                retry.refinement().codes().data,
                next.refinement().codes().data
            );
            controlled = next;
        }
        let mut draws = expected_steps.iter();
        let uninterrupted = generator
            .generate_frame(
                &initial,
                |slot, _: &NumericTensor, _| {
                    let expected = draws.next().unwrap();
                    assert_eq!(slot, expected["slot"].as_u64().unwrap() as usize);
                    Ok((
                        codes(&expected["mixture_indices"], &context)
                            .reshape(&[1, 1], &context)
                            .unwrap(),
                        tensor(&expected["gaussian_draw"]),
                    ))
                },
                &context,
            )
            .unwrap();
        assert!(draws.next().is_none());
        assert_eq!(
            uninterrupted.codes().unwrap().data,
            controlled.codes().unwrap().data
        );
        assert_eq!(
            controlled.codes().unwrap().data,
            codes(&call["codes"], &context).data
        );
        compare_cache(&caches, &call["cache"], &fixture);
        for (cache, snapshot) in caches.iter().zip(&retained) {
            assert_eq!(cache.offset(), snapshot.offset());
        }
        assert_eq!(previous[0].offset() + 1, caches[0].offset());
    }
}
