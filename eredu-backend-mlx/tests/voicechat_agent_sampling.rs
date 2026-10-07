//! Agent policy uses native primitives and the same retained RNG as speech draws.

#[path = "voicechat_agent_sampling/coordination.rs"]
mod coordination;

use eredu_architectures::voicechat::agent_sampling::{AgentSampler, AgentSamplingPolicy};
use eredu_backend_mlx::{
    MlxTensor,
    backend::{random::RandomState, runtime::generation::MlxSamplingBackend as Backend},
};
use eredu_nn::Tensor;
use eredu_runtime::{DenseSamplingBackend, Sampler, SamplingBackend};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;

#[test]
fn voicechat_agent_sampling_matches_reference_distribution_and_native_rng_progress() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let policy = AgentSamplingPolicy::new(
            case["temperature"].as_f64().unwrap() as f32,
            case["top_p"].as_f64().unwrap() as f32,
            case["repetition_penalty"].as_f64().unwrap() as f32,
            64,
        )
        .unwrap();
        let history: Vec<_> = case["history"]
            .as_array()
            .unwrap()
            .iter()
            .take(case["current_step"].as_u64().unwrap() as usize)
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let values: Vec<_> = case["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let logits = MlxTensor::from_array(Array::from_slice(&values, &[1, values.len() as i32]));
        // Several native seeds exercise the independently fixed distribution;
        // their sequences are not compared with PyTorch's RNG implementation.
        for seed in [13, 317, 1701] {
            let mut sampler = AgentSampler::new(policy);
            sampler.set_history(&history).unwrap();
            let mut random = RandomState::with_seed(seed).unwrap();
            let mut reference = random.clone();
            let selected = <AgentSampler as Sampler<Backend>>::sample(
                &mut sampler,
                &logits,
                policy.temperature(),
                Some(&mut random),
                &stream,
            )
            .unwrap();
            assert_eq!(
                sampler.history(),
                history,
                "{name}: raw selection is not effective feedback"
            );
            let expected = if case["expected"]["draws"].as_u64().unwrap() == 0 {
                case["expected"]["token"].as_u64().unwrap() as u32
            } else {
                // The original NVIDIA method records these final probabilities.
                // Log probabilities differ from processed logits only by a scalar
                // normalization, so native categorical selection uses the same law.
                let log_probabilities: Vec<_> = case["expected"]["probabilities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p.as_f64().unwrap() as f32).ln())
                    .collect();
                let oracle = MlxTensor::from_array(Array::from_slice(
                    &log_probabilities,
                    &[1, log_probabilities.len() as i32],
                ));
                let expected =
                    Backend::sample_processed(&oracle, 1.0, Some(&mut reference), &stream).unwrap();
                Backend::token_id(&expected, &stream).unwrap()
            };
            assert_eq!(
                Backend::token_id(&selected, &stream).unwrap(),
                expected,
                "{name}, seed {seed}"
            );
            let after = Backend::standard_normal_f32(&[7], &mut random, &stream).unwrap();
            let replay = Backend::standard_normal_f32(&[7], &mut reference, &stream).unwrap();
            assert_eq!(
                after.to_f32_vec(&stream).unwrap(),
                replay.to_f32_vec(&stream).unwrap(),
                "{name}: categorical decision must preserve the following speech RNG position"
            );
            sampler.accept_effective(12).unwrap();
            assert_eq!(sampler.history().last(), Some(&12));
        }
    }
}

#[test]
fn voicechat_tool_handoff_keeps_native_rng_feedback_and_spoken_text_separate() {
    use eredu_architectures::voicechat::{
        channel_decision::{ChannelDecisionPolicy, select_channels},
        channels::DualChannelLogits,
    };
    use eredu_core::{Completion, scheduler::SemanticStateTransaction};
    use eredu_runtime::{
        RealtimeGenerationContract, RealtimeGenerationState, RealtimeScheduleState, TokenDomain,
    };
    use std::convert::Infallible;
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
    // This primitive test never creates a native submission or commits a branch.
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
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let case = fixture["cases"]
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
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut sampler = AgentSampler::new(AgentSamplingPolicy::new(0.9, 0.7, 1.4, 64).unwrap());
    sampler
        .set_history(
            &case["history"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let random = RandomState::with_seed(317).unwrap();
    let mut oracle_random = random.clone();
    let state = RealtimeGenerationState::<_, _, _, _, _, Unsubmitted>::new(
        Host,
        RealtimeGenerationContract::new((), 1),
        Host,
        (),
        vec![sampler],
        Some(random),
    )
    .unwrap();
    let mut branch = state.branch().unwrap();
    let mut function = vec![-3.; 16];
    function[6] = 4.;
    let logits = DualChannelLogits {
        agent: MlxTensor::from_f32_slice(&floats(&case["logits"]), &[1, 1, 16], &stream).unwrap(),
        function: MlxTensor::from_f32_slice(&function, &[1, 1, 16], &stream).unwrap(),
    };
    let decision = select_channels::<Backend, _, _, _, _>(
        &mut branch,
        &logits,
        ChannelDecisionPolicy::Tool {
            function_feedback: Some(9),
            spoken_text: Some(5),
        }
        .into(),
        Default::default(),
        TokenDomain::new(16),
        None,
        &stream,
    )
    .unwrap();
    let probabilities = floats(&case["expected"]["probabilities"]);
    let oracle = MlxTensor::from_f32_slice(
        &probabilities.iter().map(|p| p.ln()).collect::<Vec<_>>(),
        &[1, 1, 16],
        &stream,
    )
    .unwrap();
    let selected =
        Backend::sample_processed(&oracle, 1.0, Some(&mut oracle_random), &stream).unwrap();
    assert_eq!(
        decision.raw_agent_id,
        Backend::token_id(&selected, &stream).unwrap()
    );
    assert_eq!(decision.raw_function_id, 6);
    assert_eq!(decision.spoken_text, 5);
    let feedback = decision.feedback.materialize::<MlxTensor>(&stream).unwrap();
    assert_eq!(feedback.agent.to_i32_vec(&stream).unwrap(), [12]);
    assert_eq!(feedback.function.to_i32_vec(&stream).unwrap(), [9]);
    assert_eq!(branch.samplers()[0].history().last(), Some(&12));
    let speech_draw = branch
        .draw_standard_normal::<Backend>(&[7], &stream)
        .unwrap();
    let oracle_draw = Backend::standard_normal_f32(&[7], &mut oracle_random, &stream).unwrap();
    assert_eq!(
        speech_draw.to_f32_vec(&stream).unwrap(),
        oracle_draw.to_f32_vec(&stream).unwrap()
    );
    assert_eq!(
        state.samplers()[0].history().len() + 1,
        branch.samplers()[0].history().len()
    );
}
