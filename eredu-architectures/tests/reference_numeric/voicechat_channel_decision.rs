//! Agent/function handoff, effective feedback, and shared transaction ownership.
use super::voicechat_agent_sampling::{Draw, Sampling};
use super::*;
use eredu_architectures::voicechat::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    channel_decision::{select_channels, ChannelDecisionPolicy},
    channels::DualChannelLogits,
};
use eredu_core::scheduler::SemanticStateTransaction;
use eredu_runtime::{
    RealtimeGenerationBranch, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeScheduleState,
};
use serde_json::Value;
use std::convert::Infallible;

// Reuse the independently checked scalar sampling primitives with real tensor
// geometry, rather than NumericBackend's intentionally inert sampling stubs.
struct TensorSampling;
impl SamplingBackend for TensorSampling {
    type Logits = NumericTensor;
    type Token = u32;
    type RandomState = Draw;
    type Context = RefCell<Option<Vec<f32>>>;
    type Error = String;
    fn error(message: String) -> String {
        message
    }
    fn validate_token(
        token: &u32,
        domain: TokenDomain,
        context: &Self::Context,
    ) -> Result<u32, String> {
        Sampling::validate_token(token, domain, context)
    }
    fn scale_temperature(
        logits: &NumericTensor,
        temperature: f32,
        context: &Self::Context,
    ) -> Result<NumericTensor, String> {
        Ok(NumericTensor::new(
            logits.shape.clone(),
            Sampling::scale_temperature(&logits.data, temperature, context)?,
        ))
    }
    fn apply_penalties(
        logits: &NumericTensor,
        history: &[u32],
        config: PenaltyConfig,
        context: &Self::Context,
    ) -> Result<NumericTensor, String> {
        Ok(NumericTensor::new(
            logits.shape.clone(),
            Sampling::apply_penalties(&logits.data, history, config, context)?,
        ))
    }
    fn apply_top_p(
        logits: NumericTensor,
        top_p: f32,
        context: &Self::Context,
    ) -> Result<NumericTensor, String> {
        Ok(NumericTensor::new(
            logits.shape,
            Sampling::apply_top_p(logits.data, top_p, context)?,
        ))
    }
    fn sample_raw(
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut Draw>,
        context: &Self::Context,
    ) -> Result<u32, String> {
        if logits.data.iter().any(|x| x.is_nan()) {
            return Err("invalid greedy input".into());
        }
        Sampling::sample_raw(&logits.data, temperature, random, context)
    }
    fn sample_processed(
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut Draw>,
        context: &Self::Context,
    ) -> Result<u32, String> {
        Sampling::sample_processed(&logits.data, temperature, random, context)
    }
    fn token_id(token: &u32, context: &Self::Context) -> Result<u32, String> {
        Sampling::token_id(token, context)
    }
    fn token_probability(
        logits: &NumericTensor,
        token: u32,
        context: &Self::Context,
    ) -> Result<f32, String> {
        Sampling::token_probability(&logits.data, token, context)
    }
    fn apply_top_k(_: NumericTensor, _: i32, _: &Self::Context) -> Result<NumericTensor, String> {
        unreachable!()
    }
    fn apply_min_p(_: NumericTensor, _: f32, _: &Self::Context) -> Result<NumericTensor, String> {
        unreachable!()
    }
    fn apply_token_filter(
        _: &NumericTensor,
        _: &TokenFilter,
        _: &Self::Context,
    ) -> Result<NumericTensor, String> {
        unreachable!()
    }
    fn apply_mirostat(
        _: &NumericTensor,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &Self::Context,
    ) -> Result<NumericTensor, String> {
        unreachable!()
    }
}

#[derive(Clone)]
struct Model;
impl SemanticStateTransaction for Model {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Infallible> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, branch: Self) -> Result<(), Infallible> {
        *self = branch;
        Ok(())
    }
}
#[derive(Clone)]
struct Schedule(u64);
impl RealtimeScheduleState for Schedule {
    type Identity = ();
    type Error = std::io::Error;
    fn identity(&self) -> &() {
        &()
    }
    fn validate_successor(&self, next: &Self) -> Result<(), Self::Error> {
        if next.0 == self.0 + 1 {
            Ok(())
        } else {
            Err(std::io::Error::other("invalid successor"))
        }
    }
}
struct Ready;
impl Completion for Ready {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
type Generation = RealtimeGenerationState<Model, Schedule, (), AgentSampler, Draw, Ready>;
type Branch = RealtimeGenerationBranch<Model, Schedule, (), AgentSampler, Draw, Ready>;
fn generation(sampler: AgentSampler, random: Draw) -> Generation {
    Generation::new(
        Model,
        RealtimeGenerationContract::new((), 1),
        Schedule(0),
        (),
        vec![sampler],
        Some(random),
    )
    .unwrap()
}

#[test]
fn voicechat_channel_decision_keeps_raw_feedback_and_spoken_text_distinct_until_commit() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "nucleus_crossing")
        .unwrap();
    let history: Vec<_> = case["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut sampler = AgentSampler::new(AgentSamplingPolicy::new(0.9, 0.7, 1.4, 32).unwrap());
    sampler.set_history(&history).unwrap();
    let random = Draw {
        uniform: case["uniform"].as_f64().unwrap() as f32,
        count: 0,
    };
    let mut canonical = generation(sampler.clone(), random.clone());
    let agent = NumericTensor::new(
        vec![1, 1, 16],
        case["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    );
    let mut function = vec![-3.; 16];
    function[6] = 4.;
    let logits = DualChannelLogits {
        agent,
        function: NumericTensor::new(vec![1, 1, 16], function),
    };
    let context = RefCell::new(None);
    let decide = |branch: &mut Branch| {
        let output = select_channels::<TensorSampling, _, _, _, _>(
            branch,
            &logits,
            ChannelDecisionPolicy::Tool {
                function_feedback: Some(9),
                spoken_text: Some(5),
            }
            .into(),
            Default::default(),
            TokenDomain::new(16),
            None,
            &context,
        )
        .unwrap();
        assert_eq!(
            output.raw_agent_id as u64,
            case["expected"]["token"].as_u64().unwrap()
        );
        assert_eq!(
            output.raw_function_id, 6,
            "function reads its own head greedily"
        );
        assert_eq!(
            (
                output.feedback.agent(),
                output.feedback.function(),
                output.spoken_text
            ),
            (12, 9, 5)
        );
        assert_eq!(
            branch.random_state().unwrap().count as u64,
            case["expected"]["draws"].as_u64().unwrap(),
            "override does not bypass the raw agent draw"
        );
        let mut expected = history.clone();
        expected.push(12);
        assert_eq!(branch.samplers()[0].history(), expected);
        branch.schedule_state_mut().0 += 1;
        (
            output.raw_agent_id,
            output.raw_function_id,
            output.feedback,
            output.spoken_text,
        )
    };
    // Forced response tokens enter feedback and default speech only after the
    // same raw categorical decision; the independent function remains greedy.
    let mut response = canonical.branch().unwrap();
    let response_output = select_channels::<TensorSampling, _, _, _, _>(
        &mut response,
        &logits,
        ChannelDecisionPolicy::Main {
            agent_feedback: Some(10),
            function_feedback: None,
            spoken_text: None,
        }
        .into(),
        Default::default(),
        TokenDomain::new(16),
        None,
        &context,
    )
    .unwrap();
    assert_eq!(response_output.raw_agent_id, 8);
    assert_eq!(
        (
            response_output.feedback.agent(),
            response_output.feedback.function(),
            response_output.spoken_text
        ),
        (10, 6, 10)
    );
    assert_eq!(response.random_state().unwrap().count, 1);
    assert_eq!(response.samplers()[0].history().last(), Some(&10));
    let mut abandoned = canonical.branch().unwrap();
    let expected = decide(&mut abandoned);
    assert!(
        canonical.commit_branch(abandoned).is_err(),
        "uncompleted work cannot publish"
    );
    assert_eq!(canonical.samplers(), &[sampler]);
    assert_eq!(canonical.random_state(), Some(&random));
    let mut replay = canonical.branch().unwrap();
    let mut fork = canonical.branch().unwrap();
    assert_eq!(decide(&mut replay), expected);
    assert_eq!(decide(&mut fork), expected);
    replay.attach_submission_completion(Ready).unwrap();
    canonical.commit_branch(replay).unwrap();
    assert_eq!(canonical.samplers()[0].history().last(), Some(&12));
    assert_eq!(canonical.random_state().unwrap().count, 1);
    fork.attach_submission_completion(Ready).unwrap();
    assert!(
        canonical.commit_branch(fork).is_err(),
        "a stale handoff cannot publish twice"
    );
}

#[test]
fn voicechat_channel_decision_rejects_before_draws_and_rolls_back_partial_selection() {
    let sampler = AgentSampler::new(AgentSamplingPolicy::new(0.9, 1., 1., 2).unwrap());
    let random = Draw {
        uniform: 0.73,
        count: 0,
    };
    let state = generation(sampler.clone(), random.clone());
    let mut branch = state.branch().unwrap();
    let mut values = vec![-3.; 16];
    values[3] = 4.;
    let logits = DualChannelLogits {
        agent: NumericTensor::new(vec![1, 16], values.clone()),
        function: NumericTensor::new(vec![1, 16], values),
    };
    let context = RefCell::new(None);
    let invalid = ChannelDecisionPolicy::Main {
        agent_feedback: Some(16),
        function_feedback: None,
        spoken_text: None,
    };
    assert!(select_channels::<TensorSampling, _, _, _, _>(
        &mut branch,
        &logits,
        invalid.into(),
        Default::default(),
        TokenDomain::new(16),
        None,
        &context
    )
    .is_err());
    assert!(
        context.borrow().is_none(),
        "invalid effective feedback is rejected before sampling"
    );
    assert_eq!(branch.random_state(), Some(&random));
    assert_eq!(branch.samplers(), std::slice::from_ref(&sampler));
    let mut malformed = logits.clone();
    malformed.function = NumericTensor::new(vec![2, 8], vec![0.; 16]);
    assert!(select_channels::<TensorSampling, _, _, _, _>(
        &mut branch,
        &malformed,
        ChannelDecisionPolicy::Main {
            agent_feedback: None,
            function_feedback: None,
            spoken_text: None
        }
        .into(),
        Default::default(),
        TokenDomain::new(16),
        None,
        &context
    )
    .is_err());
    assert!(
        context.borrow().is_none(),
        "malformed function geometry cannot consume an agent draw"
    );
    let mut failure = logits.clone();
    failure.function.data[0] = f32::NAN;
    assert!(select_channels::<TensorSampling, _, _, _, _>(
        &mut branch,
        &failure,
        ChannelDecisionPolicy::Main {
            agent_feedback: None,
            function_feedback: None,
            spoken_text: None
        }
        .into(),
        Default::default(),
        TokenDomain::new(16),
        None,
        &context
    )
    .is_err());
    assert!(
        context.borrow().is_some(),
        "the injected function failure follows raw agent sampling"
    );
    assert_eq!(
        branch.random_state(),
        Some(&random),
        "partial channel selection does not adopt RNG"
    );
    assert_eq!(branch.samplers(), &[sampler]);
    branch.samplers_mut()[0].set_history(&[3, 4]).unwrap();
    *context.borrow_mut() = None;
    assert!(select_channels::<TensorSampling, _, _, _, _>(
        &mut branch,
        &logits,
        ChannelDecisionPolicy::Main {
            agent_feedback: None,
            function_feedback: None,
            spoken_text: None
        }
        .into(),
        Default::default(),
        TokenDomain::new(16),
        None,
        &context
    )
    .is_err());
    assert!(
        context.borrow().is_none(),
        "full effective history rejects before sampling"
    );
    assert_eq!(branch.random_state(), Some(&random));
}

#[test]
fn voicechat_channel_decision_effective_feedback_reaches_the_next_cached_backbone_step() {
    use super::voicechat_backbone_execution::{function, model, state};
    use eredu_architectures::voicechat::{
        backbone_execution::{execute, BackboneRequest, OrdinaryBackbone},
        channels::ChannelFusion,
        config::VoiceChatConfig,
    };
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let context = NumericContext::default();
    let fusion = ChannelFusion::from_config(&config.channels).unwrap();
    let mut runtime = ResidentRuntime::new(model(&fixture, &context), &context).unwrap();
    let mut reference = ResidentRuntime::new(model(&fixture, &context), &context).unwrap();
    let mut actual_state = state();
    let mut reference_state = state();
    let mut function_head = function(&fixture, &context);
    let embedding: Vec<f32> = fixture["parameters"]["stt_model.embed_tokens.weight"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    // Direct released channel equation over independently supplied token IDs.
    let fused = |audio: &[f32], agent: usize, function: usize| {
        NumericTensor::new(
            vec![1, 1, 4],
            (0..4)
                .map(|i| (embedding[agent * 4 + i] + audio[i]) + 2. * embedding[function * 4 + i])
                .collect(),
        )
    };
    let mut last_logits = None;
    for (agent, function, audio) in [
        (3, 9, [0.12, -0.31, 0.08, 0.42]),
        (6, 2, [-0.24, 0.17, 0.35, -0.09]),
    ] {
        let input = NumericTensor::new(vec![1, 1, 4], audio.to_vec());
        let agent_tensor = NumericTensor::from_i32_slice(&[agent], &[1, 1], &context).unwrap();
        let function_tensor =
            NumericTensor::from_i32_slice(&[function], &[1, 1], &context).unwrap();
        let output = execute(
            &mut OrdinaryBackbone::new(&mut runtime),
            &mut function_head,
            &fusion,
            BackboneRequest::Audio {
                projected_audio: &input,
                agent_tokens: &agent_tensor,
                function_tokens: &function_tensor,
            },
            &mut actual_state,
            &context,
        )
        .unwrap();
        let embedded = fused(&audio, agent as usize, function as usize);
        let expected = reference
            .forward(
                nemotron_h::EmbeddedInput::target_embedded(
                    &agent_tensor,
                    &embedded,
                    None,
                    nemotron_h::TargetReadout::Logits,
                ),
                &mut reference_state,
                &context,
            )
            .unwrap();
        assert_tensor_close(
            &output.logits.agent,
            &expected,
            "independently fused warmup",
        );
        last_logits = Some(output.logits);
    }
    let logits = last_logits.unwrap();
    let generation = generation(
        AgentSampler::new(AgentSamplingPolicy::streaming_defaults(32).unwrap()),
        Draw {
            uniform: 0.73,
            count: 0,
        },
    );
    let mut branch = generation.branch().unwrap();
    let sampled = select_channels::<TensorSampling, _, _, _, _>(
        &mut branch,
        &logits,
        ChannelDecisionPolicy::Tool {
            function_feedback: Some(9),
            spoken_text: Some(5),
        }
        .into(),
        Default::default(),
        TokenDomain::new(24),
        None,
        &RefCell::new(None),
    )
    .unwrap();
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    assert_eq!(sampled.raw_agent_id, argmax(&logits.agent.data));
    assert_eq!(sampled.raw_function_id, argmax(&logits.function.data));
    assert_eq!(sampled.spoken_text, 5);
    let feedback = sampled
        .feedback
        .materialize::<NumericTensor>(&context)
        .unwrap();
    let audio = [0.28, -0.13, 0.05, -0.19];
    let input = NumericTensor::new(vec![1, 1, 4], audio.to_vec());
    let actual = execute(
        &mut OrdinaryBackbone::new(&mut runtime),
        &mut function_head,
        &fusion,
        BackboneRequest::Audio {
            projected_audio: &input,
            agent_tokens: &feedback.agent,
            function_tokens: &feedback.function,
        },
        &mut actual_state,
        &context,
    )
    .unwrap();
    let effective_agent = NumericTensor::from_i32_slice(&[12], &[1, 1], &context).unwrap();
    let embedded = fused(&audio, 12, 9);
    assert_ne!(
        embedded.data,
        fused(&audio, 5, 9).data,
        "spoken acknowledgement has a distinct embedding from effective PAD"
    );
    let expected = reference
        .forward(
            nemotron_h::EmbeddedInput::target_embedded(
                &effective_agent,
                &embedded,
                None,
                nemotron_h::TargetReadout::Logits,
            ),
            &mut reference_state,
            &context,
        )
        .unwrap();
    assert_tensor_close(
        &actual.logits.agent,
        &expected,
        "effective feedback in the next cached step",
    );
    assert_state_exact(
        &actual_state,
        &reference_state,
        3,
        "effective feedback retained by cached Mamba and attention",
    );
    assert_eq!(branch.samplers()[0].history(), &[12]);
    assert_eq!(
        branch.random_state().unwrap().count,
        0,
        "streaming default is greedy"
    );
}

#[test]
fn voicechat_dynamic_channel_policy_sees_current_rnnt_and_raw_tokens_before_rng_adoption() {
    use eredu_architectures::voicechat::channel_decision::{
        DecisionEvidence, DecisionInput, DecisionResolver, DecisionSource,
    };
    use eredu_core::BackendFailure;

    struct Policy {
        token: u32,
        reject: bool,
    }
    impl DecisionResolver for Policy {
        fn preflight(&self, _domain: TokenDomain) -> Result<(), BackendFailure> {
            Ok(())
        }
        fn resolve(&self, input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
            assert_eq!((input.raw_agent, input.raw_function), (3, 6));
            assert_eq!(input.evidence.first_blank, Some(false));
            assert_eq!(input.evidence.audio_frame_index, Some(7));
            if self.reject {
                return Err(BackendFailure::from_error(std::io::Error::other(
                    "injected host policy failure",
                )));
            }
            Ok(ChannelDecisionPolicy::Main {
                agent_feedback: Some(self.token),
                function_feedback: None,
                spoken_text: Some(4),
            })
        }
    }
    let sampler = AgentSampler::new(AgentSamplingPolicy::new(0.9, 1., 1., 8).unwrap());
    let random = Draw {
        uniform: 0.73,
        count: 0,
    };
    let state = generation(sampler.clone(), random.clone());
    let mut branch = state.branch().unwrap();
    let mut agent = vec![-100.; 16];
    agent[3] = 100.;
    let mut function = vec![-100.; 16];
    function[6] = 100.;
    let logits = DualChannelLogits {
        agent: NumericTensor::new(vec![1, 16], agent),
        function: NumericTensor::new(vec![1, 16], function),
    };
    let context = RefCell::new(None);
    let evidence = DecisionEvidence {
        first_blank: Some(false),
        audio_frame_index: Some(7),
    };
    for policy in [
        Policy {
            token: 16,
            reject: false,
        },
        Policy {
            token: 1,
            reject: true,
        },
    ] {
        assert!(select_channels::<TensorSampling, _, _, _, _>(
            &mut branch,
            &logits,
            DecisionSource::Resolved(&policy),
            evidence,
            TokenDomain::new(16),
            None,
            &context,
        )
        .is_err());
        assert!(
            context.borrow().is_some(),
            "raw sampling precedes dynamic policy"
        );
        assert_eq!(branch.random_state(), Some(&random));
        assert_eq!(branch.samplers(), std::slice::from_ref(&sampler));
    }
    let decision = select_channels::<TensorSampling, _, _, _, _>(
        &mut branch,
        &logits,
        DecisionSource::Resolved(&Policy {
            token: 12,
            reject: false,
        }),
        evidence,
        TokenDomain::new(16),
        None,
        &context,
    )
    .unwrap();
    assert_eq!(
        (
            decision.feedback.agent(),
            decision.feedback.function(),
            decision.spoken_text
        ),
        (12, 6, 4)
    );
    assert_eq!(branch.samplers()[0].history(), &[12]);
    assert_eq!(branch.random_state().unwrap().count, 1);
}
