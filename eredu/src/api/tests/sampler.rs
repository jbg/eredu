//! Backend-neutral sampler tests for facade-owned chat policy.

use serde_json::json;

use crate::api::ConstraintError;
use crate::runtime::chat::constraints::{
    advance_trigger_prefix, completes_trigger, ConstraintController,
};
use crate::{
    runtime::chat::constraints::ConstraintCompiler,
    runtime::chat::dialect::{
        DeclarativeDialectSpec, DeclarativePayloadShape, DialectParameters, ExactEnvelope,
        GenerationPromptBehavior, JsonFunctionEnvelope, ParallelCallLayout, DECLARATIVE_DIALECT,
    },
    runtime::chat::{GenerationRuntimePlan, ParallelToolCallPolicy, ToolChoice},
};
use eredu_core::{
    generation::{FinishReason, SemanticEvent},
    TokenFilter,
};
use eredu_runtime::{
    ConstrainedSampler, DefaultSampler, GenerationSampler, MirostatV2Sampler, PenaltyConfig,
    Sampler, SamplingBackend, SpeculativeSampler, TokenDomain,
};

const SYNTHETIC_JSON_FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
    envelope: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    name_field: "name",
    arguments_field: "arguments",
    call_id: None,
};

const SYNTHETIC_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: r#"{"calls":"#,
        suffix: "}",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    json_function: Some(&SYNTHETIC_JSON_FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ",",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some(r#"{"calls":"#),
    required_structural_tokens: &[],
    stop_sequences: &[],
};
const SYNTHETIC_PARAMETERS: DialectParameters = DialectParameters::Declarative(&SYNTHETIC_SPEC);
const SYNTHETIC_VOCAB_SIZE: usize = 262;
const AUTO_TRIGGER: &[u8] = br#"{"calls":"#;
const COMPLETE_CALL: &[u8] = br#"{"calls":[{"name":"ping","arguments":{}}]}"#;
const BOUNDARY_TOKENS: &[&[u8]] = &[b"{\"", b"\n{\"", br#"{"calls":["#, br#"{"oops"#];
const QUOTED_OPEN_TOKEN: u32 = SYNTHETIC_VOCAB_SIZE as u32;
const PREFIXED_QUOTED_OPEN_TOKEN: u32 = QUOTED_OPEN_TOKEN + 1;
const TRIGGER_AND_ARGUMENT_TOKEN: u32 = QUOTED_OPEN_TOKEN + 2;
const INVALID_ACTIVATION_TOKEN: u32 = QUOTED_OPEN_TOKEN + 3;

const BOUNDARY_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    auto_activation_trigger: Some("{"),
    ..SYNTHETIC_SPEC
};
const BOUNDARY_PARAMETERS: DialectParameters = DialectParameters::Declarative(&BOUNDARY_SPEC);

#[derive(Clone, Default)]
struct CountingPolicy {
    commits: usize,
}

#[derive(Debug, Clone, Copy)]
struct TestSamplingBackend;

impl SamplingBackend for TestSamplingBackend {
    type Logits = Vec<f32>;
    type Token = u32;
    type RandomState = ();
    type Context = ();
    type Error = String;

    fn error(message: String) -> Self::Error {
        message
    }

    fn validate_token(
        token: &Self::Token,
        domain: TokenDomain,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        (usize::try_from(*token).unwrap() < domain.cardinality())
            .then_some(*token)
            .ok_or_else(|| "token is outside its decision domain".into())
    }

    fn scale_temperature(
        logits: &Self::Logits,
        temperature: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.iter().map(|logit| logit / temperature).collect())
    }

    fn apply_penalties(
        logits: &Self::Logits,
        _: &[u32],
        _: PenaltyConfig,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_top_k(
        mut logits: Self::Logits,
        top_k: i32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        if top_k > 0 && (top_k as usize) < logits.len() {
            let mut ranked = logits.clone();
            ranked.sort_by(|left, right| right.total_cmp(left));
            let threshold = ranked[top_k as usize - 1];
            for logit in &mut logits {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        Ok(logits)
    }

    fn apply_top_p(
        logits: Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_min_p(
        logits: Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_token_filter(
        logits: &Self::Logits,
        filter: &TokenFilter,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        let mut masked = logits.clone();
        if let Some(allowed) = filter.allowed_mask() {
            if allowed.len() != masked.len() {
                return Err("test filter does not match the vocabulary".into());
            }
            for (logit, allowed) in masked.iter_mut().zip(allowed) {
                if !allowed {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        Ok(masked)
    }

    fn apply_mirostat(
        logits: &Self::Logits,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn sample_raw(
        logits: &Self::Logits,
        _: f32,
        _: Option<&mut Self::RandomState>,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        argmax(logits)
    }

    fn sample_processed(
        logits: &Self::Logits,
        _: f32,
        _: Option<&mut Self::RandomState>,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        argmax(logits)
    }

    fn token_id(token: &Self::Token, _: &Self::Context) -> Result<u32, Self::Error> {
        Ok(*token)
    }

    fn token_probability(
        logits: &Self::Logits,
        token: u32,
        _: &Self::Context,
    ) -> Result<f32, Self::Error> {
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = logits
            .iter()
            .map(|logit| (*logit - maximum).exp())
            .sum::<f32>();
        logits
            .get(token as usize)
            .map(|logit| (*logit - maximum).exp() / denominator)
            .ok_or_else(|| "token is outside the test vocabulary".into())
    }
}

fn argmax(logits: &[f32]) -> Result<u32, String> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token, _)| token as u32)
        .ok_or_else(|| "cannot sample an empty vocabulary".into())
}

impl SpeculativeSampler<TestSamplingBackend> for CountingPolicy {
    fn process_logits(
        &mut self,
        logits: &Vec<f32>,
        _temperature: f32,
        _history: &[u32],
        _context: &(),
    ) -> Result<Vec<f32>, String> {
        Ok(logits.clone())
    }

    fn commit_token(
        &mut self,
        _processed_logits: &Vec<f32>,
        _token: u32,
        _context: &(),
    ) -> Result<(), String> {
        self.commits += 1;
        Ok(())
    }
}

fn constrained_sampler<S>(
    policy: S,
    plan: &GenerationRuntimePlan,
) -> Result<ConstrainedSampler<S, ConstraintController>, ConstraintError> {
    Ok(ConstrainedSampler::new(
        policy,
        ConstraintController::from_generation_plan(plan)?,
    ))
}

fn synthetic_plan(tool_choice: ToolChoice) -> GenerationRuntimePlan {
    ConstraintCompiler::synthetic_for_tests()
        .compile_tool_plan(
            &DECLARATIVE_DIALECT,
            SYNTHETIC_PARAMETERS,
            &[json!({
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            })],
            tool_choice,
            ParallelToolCallPolicy::Disabled,
            Vec::new(),
        )
        .unwrap()
}

fn boundary_plan() -> GenerationRuntimePlan {
    ConstraintCompiler::synthetic_with_tokens_for_tests(BOUNDARY_TOKENS)
        .compile_tool_plan(
            &DECLARATIVE_DIALECT,
            BOUNDARY_PARAMETERS,
            &[json!({
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            })],
            ToolChoice::Auto,
            ParallelToolCallPolicy::Disabled,
            Vec::new(),
        )
        .unwrap()
}

fn placeholder_logits() -> Vec<f32> {
    vec![0.0; SYNTHETIC_VOCAB_SIZE]
}

fn commit_bytes<S: SpeculativeSampler<TestSamplingBackend>>(
    sampler: &mut S,
    bytes: &[u8],
    logits: &Vec<f32>,
    context: &(),
) {
    for &byte in bytes {
        sampler
            .commit_token(logits, u32::from(byte), context)
            .unwrap();
    }
}

fn process_logits<S: SpeculativeSampler<TestSamplingBackend>>(
    sampler: &mut S,
    logits: &Vec<f32>,
    history: &[u32],
) -> Vec<f32> {
    sampler.process_logits(logits, 0.0, history, &()).unwrap()
}

fn sample_processed<S: SpeculativeSampler<TestSamplingBackend>>(
    sampler: &S,
    logits: &Vec<f32>,
) -> u32 {
    sampler.sample_processed(logits, 0.0, None, &()).unwrap()
}

fn commit_token<S: SpeculativeSampler<TestSamplingBackend>>(
    sampler: &mut S,
    logits: &Vec<f32>,
    token: u32,
) {
    sampler.commit_token(logits, token, &()).unwrap();
}

fn supports_exact_promotion<S: SpeculativeSampler<TestSamplingBackend>>(sampler: &S) -> bool {
    sampler.supports_exact_optimistic_promotion()
}

fn prefix_is_complete<S: SpeculativeSampler<TestSamplingBackend>>(
    sampler: &S,
    history: &[u32],
) -> bool {
    sampler.prefix_is_complete(history).unwrap()
}

#[test]
fn generation_sampler_accepts_external_token_history() {
    let mut sampler = GenerationSampler::new().with_generated_tokens([1, 2]);
    assert_eq!(sampler.generated_tokens(), &[1, 2]);

    sampler.accept_token(3);
    assert_eq!(sampler.generated_tokens(), &[1, 2, 3]);

    sampler.set_generated_tokens([5, 8]);
    assert_eq!(sampler.generated_tokens(), &[5, 8]);

    sampler.clear_generated_tokens();
    assert!(sampler.generated_tokens().is_empty());
}

#[test]
fn constraint_mask_precedes_existing_top_k_and_selects_lower_valid_token() {
    let context = &();
    let plan = synthetic_plan(ToolChoice::Required);
    let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
    let mut sampler = constrained_sampler(policy, &plan).unwrap();
    let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
    values[b'x' as usize] = 100.0;
    values[b'{' as usize] = 10.0;
    let raw = values;

    let processed = process_logits(&mut sampler, &raw, &[]);
    let invalid = processed[b'x' as usize];
    let valid = processed[b'{' as usize];
    let selected =
        Sampler::<TestSamplingBackend>::sample(&mut sampler, &raw, 0.0, None, context).unwrap();

    assert!(invalid.is_infinite() && invalid.is_sign_negative());
    assert_eq!(valid, 10.0);
    assert_eq!(selected, u32::from(b'{'));
    assert_eq!(sampler.policy().generated_tokens(), &[u32::from(b'{')]);
}

#[test]
fn auto_ignores_partial_and_near_triggers() {
    let context = &();
    let plan = synthetic_plan(ToolChoice::Auto);
    let logits = placeholder_logits();

    let mut partial = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
    commit_bytes(
        &mut partial,
        &AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1],
        &logits,
        context,
    );
    assert!(!partial.controller().constraint_is_active());
    assert_eq!(partial.controller_mut().valid_token_ids().unwrap(), None);

    let mut near = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
    commit_bytes(&mut near, br#"{"callx":"#, &logits, context);
    assert!(!near.controller().constraint_is_active());
    assert_eq!(near.controller_mut().valid_token_ids().unwrap(), None);
}

#[test]
fn exact_auto_trigger_spans_tokens_and_reports_completion_once() {
    let context = &();
    let plan = synthetic_plan(ToolChoice::Auto);
    let logits = placeholder_logits();
    let mut sampler = constrained_sampler(CountingPolicy::default(), &plan).unwrap();

    for (index, &byte) in AUTO_TRIGGER.iter().enumerate() {
        sampler
            .commit_token(&logits, u32::from(byte), context)
            .unwrap();
        assert_eq!(
            sampler.controller().constraint_is_active(),
            index + 1 == AUTO_TRIGGER.len()
        );
    }
    assert!(!sampler.grammar_is_complete().unwrap());
    commit_bytes(
        &mut sampler,
        &COMPLETE_CALL[AUTO_TRIGGER.len()..],
        &logits,
        context,
    );

    assert!(sampler.grammar_is_complete().unwrap());
    assert_eq!(sampler.policy().commits, COMPLETE_CALL.len());
}

#[test]
fn ordinary_auto_activation_masks_and_commits_a_token_past_the_trigger() {
    let context = &();
    let plan = boundary_plan();
    let vocab_size = SYNTHETIC_VOCAB_SIZE + BOUNDARY_TOKENS.len();
    let mut values = vec![-100.0f32; vocab_size];
    values[INVALID_ACTIVATION_TOKEN as usize] = 100.0;
    values[QUOTED_OPEN_TOKEN as usize] = 10.0;
    let logits = values;
    let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
    let mut sampler = constrained_sampler(policy, &plan).unwrap();

    let selected =
        Sampler::<TestSamplingBackend>::sample(&mut sampler, &logits, 0.0, None, context).unwrap();

    assert_eq!(selected, QUOTED_OPEN_TOKEN);
    assert!(sampler.controller().constraint_is_active());

    let mut continuation_values = vec![-100.0f32; vocab_size];
    continuation_values[b'x' as usize] = 100.0;
    continuation_values[b'c' as usize] = 10.0;
    let continuation = continuation_values;
    let next =
        Sampler::<TestSamplingBackend>::sample(&mut sampler, &continuation, 0.0, None, context)
            .unwrap();
    assert_eq!(next, u32::from(b'c'));
}

#[test]
fn canonical_speculative_history_activates_inside_a_prefixed_token() {
    let plan = boundary_plan();
    let vocab_size = SYNTHETIC_VOCAB_SIZE + BOUNDARY_TOKENS.len();
    let mut values = vec![-100.0f32; vocab_size];
    values[b'x' as usize] = 100.0;
    values[b'c' as usize] = 10.0;
    let logits = values;
    let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
    let mut sampler = constrained_sampler(policy, &plan).unwrap();

    let processed = process_logits(&mut sampler, &logits, &[PREFIXED_QUOTED_OPEN_TOKEN]);
    let selected = sample_processed(&sampler, &processed);

    assert_eq!(selected, u32::from(b'c'));
    assert!(!sampler.controller().constraint_is_active());
    commit_token(&mut sampler, &logits, PREFIXED_QUOTED_OPEN_TOKEN);
    commit_token(&mut sampler, &processed, u32::from(b'c'));
    assert!(sampler.controller().constraint_is_active());
}

#[test]
fn optimistic_speculative_fork_validates_trigger_and_argument_bytes_in_one_token() {
    let plan = boundary_plan();
    let vocab_size = SYNTHETIC_VOCAB_SIZE + BOUNDARY_TOKENS.len();
    let mut values = vec![-100.0f32; vocab_size];
    values[b'x' as usize] = 100.0;
    values[b'{' as usize] = 10.0;
    let logits = values;
    let mut sampler = constrained_sampler(DefaultSampler, &plan).unwrap();
    let mut optimistic = sampler.clone();

    let processed = process_logits(&mut optimistic, &logits, &[TRIGGER_AND_ARGUMENT_TOKEN]);
    let selected = sample_processed(&optimistic, &processed);

    assert_eq!(selected, u32::from(b'{'));
    assert!(!sampler.controller().constraint_is_active());
    commit_token(&mut sampler, &logits, TRIGGER_AND_ARGUMENT_TOKEN);
    commit_token(&mut sampler, &processed, u32::from(b'{'));
    assert!(sampler.controller().constraint_is_active());
}

#[test]
fn runtime_plan_creates_independent_sampler_instances() {
    let context = &();
    let plan = synthetic_plan(ToolChoice::Auto);
    let logits = placeholder_logits();
    let mut first = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
    let mut second = constrained_sampler(CountingPolicy::default(), &plan).unwrap();

    commit_bytes(&mut first, AUTO_TRIGGER, &logits, context);

    assert!(first.controller().constraint_is_active());
    assert!(!second.controller().constraint_is_active());
    assert_eq!(second.controller_mut().valid_token_ids().unwrap(), None);
    assert_eq!(first.policy().commits, AUTO_TRIGGER.len());
    assert_eq!(second.policy().commits, 0);
}

#[test]
fn constrained_sampler_advertises_only_wrapped_exact_promotion() {
    let plan = synthetic_plan(ToolChoice::Required);
    let exact = constrained_sampler(DefaultSampler, &plan).unwrap();
    let adaptive = constrained_sampler(MirostatV2Sampler::default(), &plan).unwrap();

    assert!(supports_exact_promotion(&exact));
    assert!(!supports_exact_promotion(&adaptive));
}

#[test]
fn none_masks_and_rejects_the_tool_call_trigger() {
    let context = &();
    let plan = synthetic_plan(ToolChoice::None);
    let logits = placeholder_logits();
    let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
    let mut sampler = constrained_sampler(policy, &plan).unwrap();

    assert!(!sampler.controller().constraint_is_active());
    assert_eq!(sampler.controller_mut().valid_token_ids().unwrap(), None);
    commit_bytes(
        &mut sampler,
        &AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1],
        &logits,
        context,
    );

    let final_trigger_byte = *AUTO_TRIGGER.last().unwrap();
    let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
    values[final_trigger_byte as usize] = 100.0;
    values[b'x' as usize] = 10.0;
    let raw = values;
    let selected =
        Sampler::<TestSamplingBackend>::sample(&mut sampler, &raw, 0.0, None, context).unwrap();
    assert_eq!(selected, u32::from(b'x'));

    let mut rejecting = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
    commit_bytes(
        &mut rejecting,
        &AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1],
        &logits,
        context,
    );
    let error = rejecting
        .commit_token(&logits, u32::from(final_trigger_byte), context)
        .unwrap_err();
    assert!(error.to_string().contains("tool_choice is None"), "{error}");
    assert_eq!(
        rejecting.policy().commits,
        AUTO_TRIGGER.len() - 1,
        "a rejected trigger token must roll back the wrapped policy"
    );
}

#[test]
fn none_masks_trigger_completion_in_speculative_history() {
    let plan = synthetic_plan(ToolChoice::None);
    let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
    let mut sampler = constrained_sampler(policy, &plan).unwrap();
    let final_trigger_byte = *AUTO_TRIGGER.last().unwrap();
    let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
    values[final_trigger_byte as usize] = 100.0;
    values[b'x' as usize] = 10.0;
    let raw = values;
    let history = AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1]
        .iter()
        .copied()
        .map(u32::from)
        .collect::<Vec<_>>();

    let processed = process_logits(&mut sampler, &raw, &history);
    let selected = sample_processed(&sampler, &processed);
    assert_eq!(selected, u32::from(b'x'));
}

#[test]
fn none_parser_never_emits_tool_events() {
    let plan = synthetic_plan(ToolChoice::None);
    let mut parser = plan.create_parser().unwrap();

    parser.push(str::from_utf8(COMPLETE_CALL).unwrap()).unwrap();
    parser.finish(FinishReason::MaxTokens).unwrap();

    assert!(parser.events().iter().all(|event| !matches!(
        event,
        SemanticEvent::ToolCallStart { .. }
            | SemanticEvent::ToolArgumentsDelta { .. }
            | SemanticEvent::ToolCallEnd
    )));
}

#[test]
fn forbidden_trigger_matching_handles_whole_tokens_and_overlaps() {
    assert!(completes_trigger(b"", b"xxababyy", b"abab"));
    assert!(completes_trigger(b"ab", b"ab", b"abab"));
    assert!(!completes_trigger(b"ab", b"ax", b"abab"));

    let mut pending = Vec::new();
    advance_trigger_prefix(&mut pending, b"aba", b"abab");
    assert_eq!(pending, b"aba");
    advance_trigger_prefix(&mut pending, b"a", b"abab");
    assert_eq!(pending, b"a");
}

#[test]
fn required_is_immediate_and_sampler_clones_are_independent() {
    let context = &();
    let plan = synthetic_plan(ToolChoice::Required);
    let logits = placeholder_logits();
    let mut sampler = constrained_sampler(CountingPolicy::default(), &plan).unwrap();

    assert!(sampler.controller().constraint_is_active());
    let initial = sampler.controller_mut().valid_token_ids().unwrap().unwrap();
    assert!(initial.contains(&u32::from(b'{')));
    sampler
        .commit_token(&logits, u32::from(b'{'), context)
        .unwrap();
    let after_open = sampler.controller_mut().valid_token_ids().unwrap().unwrap();
    let mut fork = sampler.clone();
    assert_eq!(
        fork.controller_mut().valid_token_ids().unwrap().unwrap(),
        after_open
    );

    sampler
        .commit_token(&logits, u32::from(b'"'), context)
        .unwrap();

    assert_eq!(
        fork.controller_mut().valid_token_ids().unwrap().unwrap(),
        after_open
    );
    assert_eq!(fork.policy().commits, 1);
    assert_eq!(sampler.policy().commits, 2);
}

#[test]
fn speculative_history_uses_a_state_fork_without_early_activation() {
    let plan = synthetic_plan(ToolChoice::Auto);
    let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
    let mut sampler = constrained_sampler(policy, &plan).unwrap();
    let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
    values[b'x' as usize] = 100.0;
    values[b'[' as usize] = 10.0;
    let raw = values;
    let history = AUTO_TRIGGER
        .iter()
        .copied()
        .map(u32::from)
        .collect::<Vec<_>>();

    let processed = process_logits(&mut sampler, &raw, &history);
    let selected = sample_processed(&sampler, &processed);

    assert_eq!(selected, u32::from(b'['));
    assert!(!sampler.controller().constraint_is_active());
    assert!(!prefix_is_complete(
        &sampler,
        &COMPLETE_CALL[..COMPLETE_CALL.len() - 1]
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>()
    ));
    assert!(prefix_is_complete(
        &sampler,
        &COMPLETE_CALL
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>()
    ));
    assert!(
        !sampler.controller().constraint_is_active(),
        "history-relative queries must not activate canonical grammar state"
    );
    assert!(sampler.policy().generated_tokens().is_empty());
}

#[test]
fn mirostat_v2_defaults_and_reset_restore_adaptive_state() {
    let mut sampler = MirostatV2Sampler::default();
    assert_eq!(sampler.tau(), 5.0);
    assert_eq!(sampler.eta(), 0.1);
    assert_eq!(sampler.mu(), 10.0);

    sampler.accept_token(42, 2.0f32.powi(-7)).unwrap();
    assert!((sampler.mu() - 9.8).abs() < 1e-6);
    assert_eq!(sampler.generated_tokens(), &[42]);

    sampler.reset();
    assert_eq!(sampler.mu(), 10.0);
    assert!(sampler.generated_tokens().is_empty());
}

#[test]
fn mirostat_v2_validates_configuration() {
    assert!(MirostatV2Sampler::new(3.0, 0.2).is_ok());
    assert!(MirostatV2Sampler::new(0.0, 0.2).is_err());
    assert!(MirostatV2Sampler::new(3.0, f32::NAN).is_err());

    let mut sampler = MirostatV2Sampler::default();
    assert!(sampler.accept_token(0, 0.0).is_err());
    assert!(sampler.accept_token(0, 1.1).is_err());
}
