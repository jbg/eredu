//! Pinned TTS/character equations through the ordinary execution-unit providers.
use super::voicechat_transformer::{CausalCache, geometry};
use super::voicechat_tts_generator::{compare, compare_cache, tensor};
use super::*;
use eredu_architectures::voicechat::tts_transformer_layered::CausalTransformerExecution;
use eredu_architectures::voicechat::{
    tts_generator::{TtsGenerator, TtsGeneratorParts},
    tts_ingress::{PromptLayout, SpeakerPrompt},
    tts_transformer::{TransformerConfig, TtsTransformer},
    tts_transformer_layered::{
        LayerwiseTransformer, TransformerArchitecture, TransformerExecution, TransformerUnit,
    },
};
use serde_json::Value;

type Unit = TransformerUnit<NumericBackend>;
pub(super) struct BoundFixturePolicy {
    inner: RebuildingUnitPolicy,
    parameters: BTreeMap<String, NumericTensor>,
}
impl LayerwisePolicy<NumericBackend, Unit> for BoundFixturePolicy {
    type Lease = RebuiltUnitLease<Unit>;
    type Error = Error;
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::begin(
            &mut self.inner,
            input,
            context,
        )
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<Unit, E>,
    {
        let parameters = &self.parameters;
        self.inner.acquire(
            ordinal,
            address,
            |context| {
                let mut unit = build(context)?;
                replace_numeric_parameters(&mut unit, parameters);
                Ok(unit)
            },
            context,
        )
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: S,
        retained: C,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        NumericTensor: 'a,
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        self.inner
            .complete(ordinal, address, lease, output, state, retained, context)
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::finish(
            &mut self.inner,
            output,
            context,
        )
    }
}

fn weights(parameters: &Value) -> BTreeMap<String, NumericTensor> {
    parameters
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), tensor(value)))
        .collect()
}

pub(super) fn bounded(
    config: TransformerConfig,
    parameters: &Value,
) -> LayerwiseTransformer<NumericBackend, BoundFixturePolicy> {
    LayerwiseTransformer::new(
        TransformerArchitecture::new(config).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            parameters: weights(parameters),
        },
    )
}

fn resident(
    config: TransformerConfig,
    parameters: &Value,
    context: &NumericContext,
) -> LayerwiseTransformer<NumericBackend, NumericReplicatedPolicy<Unit>> {
    let architecture = TransformerArchitecture::new(config).unwrap();
    let values = weights(parameters);
    let mut units = Vec::new();
    for group in 0..2 {
        for index in 0..architecture.group_unit_count(group).unwrap() {
            let mut unit = architecture.build_unit(group, index, context).unwrap();
            replace_numeric_parameters(&mut unit, &values);
            units.push(unit);
        }
    }
    LayerwiseTransformer::new(architecture, NumericReplicatedPolicy::resident(units))
}

fn caches() -> [CausalCache; 2] {
    [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ]
}

fn events() -> Vec<StreamPolicyEvent> {
    vec![
        StreamPolicyEvent::Begin,
        StreamPolicyEvent::Acquire(0, 0, 0),
        StreamPolicyEvent::Complete(0, 0, 0),
        StreamPolicyEvent::Acquire(1, 0, 1),
        StreamPolicyEvent::Complete(1, 0, 1),
        StreamPolicyEvent::Acquire(2, 1, 0),
        StreamPolicyEvent::Complete(2, 1, 0),
        StreamPolicyEvent::Finish,
    ]
}

fn causal(
    model: &mut impl CausalTransformerExecution<NumericBackend, [CausalCache]>,
    data: &Value,
    fixture: &Value,
    context: &NumericContext,
) {
    let mut cache = caches();
    for step in data["chunks"].as_array().unwrap() {
        let result = model
            .forward_causal(&tensor(&step["input"]), &mut cache[..], context)
            .unwrap();
        compare(&result, &step["output"], fixture);
        compare_cache(&cache, &step["cache"], fixture);
    }
}

#[test]
fn voicechat_transformer_resident_and_bounded_causal_caches_match_pinned_hf() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let data = &fixture["gemma3"];
    let config = TransformerConfig::new(&geometry(&data["config"], true)).unwrap();
    let mut ordinary = TtsTransformer::new(config.clone(), &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights(&data["parameters"]));
    causal(&mut ordinary, data, &fixture, &context);
    let mut resident = resident(config.clone(), &data["parameters"], &context);
    causal(&mut resident, data, &fixture, &context);
    let mut bounded = bounded(config, &data["parameters"]);
    causal(&mut bounded, data, &fixture, &context);
    assert_eq!(bounded.runtime().policy().inner.events, events().repeat(4));
}

#[test]
fn voicechat_transformer_bounded_character_masks_match_pinned_bidirectional_outputs() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let data = &fixture["t5gemma"];
    let config = TransformerConfig::new(&geometry(&data["config"]["encoder"], false)).unwrap();
    let mask = tensor(&data["attention_mask"])
        .equal_i32(1, &context)
        .unwrap();
    let batch = &data["batches"][0];
    let input = tensor(&batch["input"]);
    let mut ordinary = TtsTransformer::<NumericBackend>::new(config.clone(), &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights(&data["parameters"]));
    let mut resident = resident(config.clone(), &data["parameters"], &context);
    let mut bounded = bounded(config, &data["parameters"]);
    for output in [
        ordinary.forward_encoder(&input, &mask, &context).unwrap(),
        resident.forward_encoder(&input, &mask, &context).unwrap(),
        bounded.forward_encoder(&input, &mask, &context).unwrap(),
    ] {
        compare(&output, &batch["output"], &fixture);
    }
    assert_eq!(bounded.runtime().policy().inner.events, events());
}

#[test]
fn voicechat_transformer_late_acquisition_failure_only_mutates_the_unpublished_cache_branch() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let data = &fixture["gemma3"];
    let mut model = bounded(
        TransformerConfig::new(&geometry(&data["config"], true)).unwrap(),
        &data["parameters"],
    );
    let mut parent = caches();
    let first = &data["chunks"][0];
    model
        .forward_causal(&tensor(&first["input"]), &mut parent, &context)
        .unwrap();
    let next = &data["chunks"][1];
    let input = tensor(&next["input"]);
    model.runtime_mut().policy_mut().inner.events.clear();
    model.runtime_mut().policy_mut().inner.fail_acquire = Some(2);
    let mut branch = parent.clone();
    let error = model
        .forward_causal(&input, &mut branch, &context)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected dense-stream acquisition failure")
    );
    assert!(
        branch
            .iter()
            .all(|cache| cache.offset() == parent[0].offset() + 1)
    );
    compare_cache(&parent, &first["cache"], &fixture);
    assert_eq!(model.runtime().policy().inner.events, events()[..6]);
    drop(branch);
    model.runtime_mut().policy_mut().inner.fail_acquire = None;
    let mut retry = parent.clone();
    let output = model.forward_causal(&input, &mut retry, &context).unwrap();
    compare(&output, &next["output"], &fixture);
    compare_cache(&retry, &next["cache"], &fixture);
    compare_cache(&parent, &first["cache"], &fixture);
}

#[test]
fn voicechat_bounded_tts_generator_preserves_both_guidance_lanes_through_prefill_and_decode() {
    use super::voicechat_tts_generator::{codes, model};
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let ordinary = model(&fixture, &context);
    let config = ordinary.config().clone();
    let parts = ordinary.into_parts();
    let mut generator = TtsGenerator::from_parts(
        &config,
        16,
        5,
        TtsGeneratorParts::new(
            parts.conditioning,
            bounded(
                TransformerConfig::new(&config.character_encoder).unwrap(),
                &fixture["parameters"],
            ),
            bounded(
                TransformerConfig::new(&config.backbone).unwrap(),
                &fixture["parameters"],
            ),
            parts.refinement,
            parts.mog,
        ),
    )
    .unwrap();
    let calls = fixture["calls"].as_array().unwrap();
    let prompt = &calls[0];
    let mask = prompt["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_bool().unwrap())
        .collect::<Vec<_>>();
    let layout = PromptLayout::new(1, 5, &mask).unwrap();
    let mut cache = caches();
    let hidden = generator
        .prefill(
            &codes(&prompt["input_codes"], &context),
            &layout,
            SpeakerPrompt::Aria,
            Some(&tensor(&prompt["supplied_subwords"])),
            &mut cache,
            &context,
        )
        .unwrap();
    compare(&hidden, &prompt["hidden"], &fixture);
    compare_cache(&cache, &prompt["cache"], &fixture);
    for call in &calls[1..3] {
        let any = call["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_bool().unwrap());
        let conditioning = tensor(&call["supplied_subwords"]);
        let frame = generator
            .begin_frame(
                &codes(&call["input_codes"], &context),
                any.then_some(&conditioning),
                &mut cache,
                &context,
            )
            .unwrap();
        let hidden = NumericTensor::concatenate(
            &[
                frame.conditional().clone(),
                frame.unconditional().unwrap().clone(),
            ],
            0,
            &context,
        )
        .unwrap();
        assert_eq!(hidden.dim(0), 2);
        compare(&hidden, &call["hidden"], &fixture);
        compare_cache(&cache, &call["cache"], &fixture);
    }
    assert_eq!(
        generator.transformer.runtime().policy().inner.events,
        events().repeat(3)
    );
    assert!(
        generator
            .character_transformer
            .runtime()
            .policy()
            .inner
            .events
            .is_empty(),
        "supplied character conditioning does not acquire an unneeded character stack"
    );
}
