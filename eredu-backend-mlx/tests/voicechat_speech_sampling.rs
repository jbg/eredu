//! Native EAR-TTS refinement uses the ordinary retained session RNG.
#[path = "voicechat_speech_sampling/coordination.rs"]
mod coordination;
use eredu_architectures::voicechat::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    channel_decision::{select_channels, ChannelDecisionPolicy},
    channels::DualChannelLogits,
    config::{GemmaConfig, VoiceChatConfig},
    tts_generator::TtsGenerator,
    tts_ingress::{PromptLayout, SpeakerPrompt},
};
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        random::RandomState,
        runtime::{cache::state::MlxKeyValueState, generation::MlxSamplingBackend},
    },
    MlxTensor,
};
use eredu_core::{scheduler::SemanticStateTransaction, Completion};
use eredu_nn::{AttentionCache, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use eredu_runtime::{
    DenseSamplingBackend, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeScheduleState, SamplingBackend, TokenDomain,
};
use safemlx::{Device, DeviceType, Stream};
use serde_json::Value;
use std::convert::Infallible;

fn tensor(value: &Value, stream: &Stream) -> MlxTensor {
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
    MlxTensor::from_f32_slice(&values, &shape, stream).unwrap()
}
fn codes(value: &Value, stream: &Stream) -> MlxTensor {
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
    MlxTensor::from_i32_slice(&values, &shape, stream).unwrap()
}
fn geometry(base: &GemmaConfig, source: &Value) -> GemmaConfig {
    let mut value = serde_json::to_value(base).unwrap();
    for (name, slot) in value.as_object_mut().unwrap() {
        if let Some(original) = source.get(name) {
            *slot = original.clone();
        }
    }
    serde_json::from_value(value).unwrap()
}
struct Bind<'a>(&'a Value, &'a Stream);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        // Supplied-subword fixtures deliberately omit the unexecuted character module.
        if self.0.get(metadata.id.as_str()).is_none()
            && metadata
                .id
                .as_str()
                .starts_with("tts_model.tts_model.embed_subword.")
        {
            return;
        }
        let replacement = tensor(&self.0[metadata.id.as_str()], self.1);
        assert_eq!(
            output.shape(),
            replacement.shape(),
            "{}",
            metadata.id.as_str()
        );
        *output = replacement;
    }
}
#[derive(Clone)]
struct Host;
impl SemanticStateTransaction for Host {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Infallible> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, _: Self) -> Result<(), Infallible> {
        unreachable!()
    }
}
impl RealtimeScheduleState for Host {
    type Identity = ();
    type Error = Infallible;
    fn identity(&self) -> &() {
        &()
    }
    fn validate_successor(&self, _: &Self) -> Result<(), Infallible> {
        unreachable!()
    }
}
// This mechanism test evaluates arrays to host but never submits/commits a session.
enum Unsubmitted {}
impl Completion for Unsubmitted {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        match *self {}
    }
    fn wait(&self) -> Result<(), Infallible> {
        match *self {}
    }
}
type Generation = RealtimeGenerationState<Host, Host, (), AgentSampler, RandomState, Unsubmitted>;

#[test]
fn voicechat_speech_sampling_native_controlled_and_uninterrupted_share_exact_rng_progress() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let agent_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let agent = agent_fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "nucleus_crossing")
        .unwrap();
    let floats = |value: &Value| {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let mut function = vec![-3.; 16];
    function[6] = 4.;
    let heads = DualChannelLogits {
        agent: MlxTensor::from_f32_slice(&floats(&agent["logits"]), &[1, 1, 16], &stream).unwrap(),
        function: MlxTensor::from_f32_slice(&function, &[1, 1, 16], &stream).unwrap(),
    };
    let original = VoiceChatConfig::from_json(include_str!(
        "../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    // Zero noise must still consume the released Gaussian draw. Compare both
    // policy settings using real native random keys, not forced mixture IDs.
    for noise_scale in [original.tts.noise_scale, 0.0] {
        let mut config = original.tts.clone();
        config.backbone = geometry(&config.backbone, &fixture["transformer_config"]);
        let transformers: Value =
            serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
        config.character_encoder = geometry(
            &config.character_encoder,
            &transformers["t5gemma"]["config"]["encoder"],
        );
        config.latent_size = 4;
        config.codebook_size = 5;
        config.mog.intermediate_size = 12;
        config.mog.num_layers = 3;
        config.mog.low_rank = 2;
        config.mog.num_predictions = 4;
        config.noise_scale = noise_scale;
        let mut model = TtsGenerator::<MlxNeuralBackend>::new(&config, 5, 5, &stream).unwrap();
        let mut bind = Bind(&fixture["parameters"], &stream);
        model.conditioning.visit_parameters_mut(&mut bind);
        model.transformer.visit_parameters_mut(&mut bind);
        model.refinement.visit_parameters_mut(&mut bind);
        model.mog.visit_parameters_mut(&mut bind);
        let first = &fixture["calls"][0];
        let call = &fixture["calls"][1];
        let mask = first["audio_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect::<Vec<_>>();
        let layout = eredu_runtime::StateLayout::new(
            eredu_core::LayerSchedule::new(
                2,
                vec![
                    eredu_core::cache::LayerCachePolicy::key_value(
                        eredu_core::AttentionPolicy::sliding(3).unwrap(),
                        1,
                        4,
                    )
                    .unwrap(),
                    eredu_core::cache::LayerCachePolicy::key_value(
                        eredu_core::AttentionPolicy::Full,
                        1,
                        4,
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let mut caches = MlxKeyValueState::device(layout).unwrap();
        model
            .prefill(
                &codes(&first["input_codes"], &stream),
                &PromptLayout::new(1, 5, &mask).unwrap(),
                SpeakerPrompt::Aria,
                Some(&tensor(&first["supplied_subwords"], &stream)),
                caches.as_mut(),
                &stream,
            )
            .unwrap();
        let frame = model
            .begin_frame(
                &codes(&call["input_codes"], &stream),
                Some(&tensor(&call["supplied_subwords"], &stream)),
                caches.as_mut(),
                &stream,
            )
            .unwrap();
        let random = RandomState::with_seed(317).unwrap();
        let mut expected_random = random.clone();
        let mut sampler = AgentSampler::new(AgentSamplingPolicy::new(0.9, 0.7, 1.4, 64).unwrap());
        sampler
            .set_history(
                &agent["history"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let state = Generation::new(
            Host,
            RealtimeGenerationContract::new((), 1),
            Host,
            (),
            vec![sampler],
            Some(random),
        )
        .unwrap();
        let mut controlled_rng = state.branch().unwrap();
        let mut ordinary_rng = state.branch().unwrap();
        let reference_logits = MlxTensor::from_f32_slice(
            &floats(&agent["expected"]["probabilities"])
                .iter()
                .map(|p| p.ln())
                .collect::<Vec<_>>(),
            &[1, 1, 16],
            &stream,
        )
        .unwrap();
        let expected_agent = MlxSamplingBackend::sample_processed(
            &reference_logits,
            1.,
            Some(&mut expected_random),
            &stream,
        )
        .unwrap();
        for branch in [&mut controlled_rng, &mut ordinary_rng] {
            let decision = select_channels::<MlxSamplingBackend, _, _, _, _>(
                branch,
                &heads,
                ChannelDecisionPolicy::Tool {
                    function_feedback: Some(9),
                    spoken_text: Some(3),
                }
                .into(),
                Default::default(),
                TokenDomain::new(16),
                None,
                &stream,
            )
            .unwrap();
            assert_eq!(
                decision.raw_agent_id,
                MlxSamplingBackend::token_id(&expected_agent, &stream).unwrap()
            );
            assert_eq!(
                decision.spoken_text,
                call["subword_ids"]["values"][0].as_u64().unwrap() as u32
            );
            assert_eq!(decision.feedback.agent(), 12);
        }
        let mut controlled = frame.clone();
        let mut slots = Vec::new();
        while let Some(slot) = controlled.refinement().next_slot() {
            slots.push(slot);
            controlled = model
                .advance_sampled_frame::<MlxSamplingBackend, _, _, _, _, _>(
                    &controlled,
                    &mut controlled_rng,
                    None,
                    &stream,
                )
                .unwrap();
        }
        let ordinary = model
            .generate_sampled_frame::<MlxSamplingBackend, _, _, _, _, _>(
                &frame,
                &mut ordinary_rng,
                None,
                &stream,
            )
            .unwrap();
        assert_eq!(slots, [3, 4, 5, 6, 7]);
        assert_eq!(
            controlled.codes().unwrap().to_i32_vec(&stream).unwrap(),
            ordinary.codes().unwrap().to_i32_vec(&stream).unwrap()
        );
        for _ in 0..5 {
            MlxSamplingBackend::uniform_f32(&[1, 1, 4], &mut expected_random, &stream).unwrap();
            MlxSamplingBackend::standard_normal_f32(&[1, 1, 4], &mut expected_random, &stream)
                .unwrap();
        }
        let expected = MlxSamplingBackend::standard_normal_f32(&[7], &mut expected_random, &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap();
        for branch in [&mut controlled_rng, &mut ordinary_rng] {
            assert_eq!(branch.samplers()[0].history().last(), Some(&12));
            let actual = branch
                .draw_standard_normal::<MlxSamplingBackend>(&[7], &stream)
                .unwrap();
            assert_eq!(actual.to_f32_vec(&stream).unwrap(), expected);
        }
        assert!(caches
            .as_ref()
            .iter()
            .all(|cache| AttentionCache::offset(cache) == 6));
    }
}
