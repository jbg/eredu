//! Backend-neutral sampling policies specialized to SafeMLX primitives.

use eredu_runtime::{
    Sampler as RuntimeSampler, SamplingBackend, SpeculativeSampler as RuntimeSpeculativeSampler,
};
use safemlx::{error::Exception, random::RandomState, Array, Stream};

pub(crate) use super::backend::MlxSamplingBackend;
use crate::core::TokenFilter;

pub use eredu_runtime::{ConstrainedSampler, DefaultSampler, GenerationSampler, MirostatV2Sampler};

/// SafeMLX-specialized token selection policy.
pub trait Sampler {
    /// Whether loaded checkpoint defaults should wrap this policy.
    fn uses_checkpoint_defaults(&self) -> bool {
        false
    }

    /// Selects one token from raw MLX logits.
    fn sample(
        &mut self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception>;
}

impl<T> Sampler for T
where
    T: RuntimeSampler<MlxSamplingBackend>,
{
    fn uses_checkpoint_defaults(&self) -> bool {
        RuntimeSampler::<MlxSamplingBackend>::uses_checkpoint_defaults(self)
    }

    fn sample(
        &mut self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        RuntimeSampler::<MlxSamplingBackend>::sample(self, logits, temperature, random, stream)
    }
}

/// SafeMLX-specialized lossless speculative sampling policy.
pub trait SpeculativeSampler {
    /// Whether loaded checkpoint defaults should wrap this policy.
    fn uses_checkpoint_defaults(&self) -> bool {
        false
    }

    /// Whether optimistic draft work is an exact discardable fork.
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    /// Whether the committed generation grammar is complete.
    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        Ok(false)
    }

    /// Whether an uncommitted logical prefix completes the grammar.
    fn prefix_is_complete(&self, _history: &[u32]) -> Result<bool, Exception> {
        Ok(false)
    }

    /// Applies penalties, filters, and temperature.
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception>;

    /// Selects from already processed logits.
    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        MlxSamplingBackend::sample_processed(logits, temperature, random, stream)
    }

    /// Commits an emitted token from a processed target distribution.
    fn commit_token(
        &mut self,
        _processed_logits: &Array,
        _token: u32,
        _stream: &Stream,
    ) -> Result<(), Exception> {
        Ok(())
    }
}

impl<T> SpeculativeSampler for T
where
    T: RuntimeSpeculativeSampler<MlxSamplingBackend>,
{
    fn uses_checkpoint_defaults(&self) -> bool {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::uses_checkpoint_defaults(self)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(self)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::grammar_is_complete(self)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::prefix_is_complete(self, history)
    }

    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::process_logits(
            self,
            logits,
            temperature,
            history,
            stream,
        )
    }

    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::sample_processed(
            self,
            logits,
            temperature,
            random,
            stream,
        )
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::commit_token(
            self,
            processed_logits,
            token,
            stream,
        )
    }
}

/// Applies a portable vocabulary filter to backend-owned MLX logits.
pub(crate) fn apply_token_filter(
    logits: &Array,
    filter: &TokenFilter,
    stream: &Stream,
) -> Result<Array, Exception> {
    MlxSamplingBackend::apply_token_filter(logits, filter, stream)
}

#[cfg(test)]
mod tests {
    use safemlx::{
        error::Exception, ops::indexing::TryIndexOp, transforms::eval, Array, Device, DeviceType,
        ExecutionContext, Stream,
    };
    use serde_json::json;

    use super::{
        ConstrainedSampler, DefaultSampler, GenerationSampler, MirostatV2Sampler,
        MlxSamplingBackend, Sampler, SpeculativeSampler,
    };
    use crate::runtime::chat::constraints::{
        advance_trigger_prefix, completes_trigger, ConstraintController, ConstraintError,
    };
    use crate::{
        core::generation::{FinishReason, SemanticEvent},
        runtime::chat::constraints::ConstraintCompiler,
        runtime::chat::dialect::{
            DeclarativeDialectSpec, DeclarativePayloadShape, DialectParameters, ExactEnvelope,
            GenerationPromptBehavior, JsonFunctionEnvelope, ParallelCallLayout,
            DECLARATIVE_DIALECT,
        },
        runtime::chat::{GenerationRuntimePlan, ParallelToolCallPolicy, ToolChoice},
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

    impl eredu_runtime::SpeculativeSampler<MlxSamplingBackend> for CountingPolicy {
        fn process_logits(
            &mut self,
            logits: &Array,
            _temperature: f32,
            _history: &[u32],
            _stream: &Stream,
        ) -> Result<Array, Exception> {
            Ok(logits.clone())
        }

        fn commit_token(
            &mut self,
            _processed_logits: &Array,
            _token: u32,
            _stream: &Stream,
        ) -> Result<(), Exception> {
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

    fn test_context() -> ExecutionContext {
        ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
    }

    fn placeholder_logits() -> Array {
        Array::from_slice(
            &vec![0.0f32; SYNTHETIC_VOCAB_SIZE],
            &[1, SYNTHETIC_VOCAB_SIZE as i32],
        )
    }

    fn commit_bytes<S: SpeculativeSampler>(
        sampler: &mut S,
        bytes: &[u8],
        logits: &Array,
        stream: &Stream,
    ) {
        for &byte in bytes {
            sampler
                .commit_token(logits, u32::from(byte), stream)
                .unwrap();
        }
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
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::Required);
        let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
        let mut sampler = constrained_sampler(policy, &plan).unwrap();
        let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
        values[b'x' as usize] = 100.0;
        values[b'{' as usize] = 10.0;
        let raw = Array::from_slice(&values, &[1, SYNTHETIC_VOCAB_SIZE as i32]);

        let processed = sampler.process_logits(&raw, 0.0, &[], stream).unwrap();
        let invalid = processed
            .try_index_device((0, i32::from(b'x')), stream)
            .unwrap();
        let valid = processed
            .try_index_device((0, i32::from(b'{')), stream)
            .unwrap();
        let selected = Sampler::sample(&mut sampler, &raw, 0.0, None, stream).unwrap();
        eval([&invalid, &valid, &selected]).unwrap();

        assert!(invalid.item::<f32>(stream) < -1.0e30);
        assert_eq!(valid.item::<f32>(stream), 10.0);
        assert_eq!(selected.item::<u32>(stream), u32::from(b'{'));
        assert_eq!(sampler.policy().generated_tokens(), &[u32::from(b'{')]);
    }

    #[test]
    fn auto_ignores_partial_and_near_triggers() {
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::Auto);
        let logits = placeholder_logits();

        let mut partial = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
        commit_bytes(
            &mut partial,
            &AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1],
            &logits,
            stream,
        );
        assert!(!partial.controller().constraint_is_active());
        assert_eq!(partial.controller_mut().valid_token_ids().unwrap(), None);

        let mut near = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
        commit_bytes(&mut near, br#"{"callx":"#, &logits, stream);
        assert!(!near.controller().constraint_is_active());
        assert_eq!(near.controller_mut().valid_token_ids().unwrap(), None);
    }

    #[test]
    fn exact_auto_trigger_spans_tokens_and_reports_completion_once() {
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::Auto);
        let logits = placeholder_logits();
        let mut sampler = constrained_sampler(CountingPolicy::default(), &plan).unwrap();

        for (index, &byte) in AUTO_TRIGGER.iter().enumerate() {
            sampler
                .commit_token(&logits, u32::from(byte), stream)
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
            stream,
        );

        assert!(sampler.grammar_is_complete().unwrap());
        assert_eq!(sampler.policy().commits, COMPLETE_CALL.len());
    }

    #[test]
    fn ordinary_auto_activation_masks_and_commits_a_token_past_the_trigger() {
        let context = test_context();
        let stream = context.stream();
        let plan = boundary_plan();
        let vocab_size = SYNTHETIC_VOCAB_SIZE + BOUNDARY_TOKENS.len();
        let mut values = vec![-100.0f32; vocab_size];
        values[INVALID_ACTIVATION_TOKEN as usize] = 100.0;
        values[QUOTED_OPEN_TOKEN as usize] = 10.0;
        let logits = Array::from_slice(&values, &[1, vocab_size as i32]);
        let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
        let mut sampler = constrained_sampler(policy, &plan).unwrap();

        let selected = Sampler::sample(&mut sampler, &logits, 0.0, None, stream).unwrap();
        eval([&selected]).unwrap();

        assert_eq!(selected.item::<u32>(stream), QUOTED_OPEN_TOKEN);
        assert!(sampler.controller().constraint_is_active());

        let mut continuation_values = vec![-100.0f32; vocab_size];
        continuation_values[b'x' as usize] = 100.0;
        continuation_values[b'c' as usize] = 10.0;
        let continuation = Array::from_slice(&continuation_values, &[1, vocab_size as i32]);
        let next = Sampler::sample(&mut sampler, &continuation, 0.0, None, stream).unwrap();
        eval([&next]).unwrap();
        assert_eq!(next.item::<u32>(stream), u32::from(b'c'));
    }

    #[test]
    fn canonical_mtp_history_activates_inside_a_prefixed_token() {
        let context = test_context();
        let stream = context.stream();
        let plan = boundary_plan();
        let vocab_size = SYNTHETIC_VOCAB_SIZE + BOUNDARY_TOKENS.len();
        let mut values = vec![-100.0f32; vocab_size];
        values[b'x' as usize] = 100.0;
        values[b'c' as usize] = 10.0;
        let logits = Array::from_slice(&values, &[1, vocab_size as i32]);
        let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
        let mut sampler = constrained_sampler(policy, &plan).unwrap();

        let processed = sampler
            .process_logits(&logits, 0.0, &[PREFIXED_QUOTED_OPEN_TOKEN], stream)
            .unwrap();
        let selected = sampler
            .sample_processed(&processed, 0.0, None, stream)
            .unwrap();
        eval([&selected]).unwrap();

        assert_eq!(selected.item::<u32>(stream), u32::from(b'c'));
        assert!(!sampler.controller().constraint_is_active());
        sampler
            .commit_token(&logits, PREFIXED_QUOTED_OPEN_TOKEN, stream)
            .unwrap();
        sampler
            .commit_token(&processed, u32::from(b'c'), stream)
            .unwrap();
        assert!(sampler.controller().constraint_is_active());
    }

    #[test]
    fn optimistic_mtp_fork_validates_trigger_and_argument_bytes_in_one_token() {
        let context = test_context();
        let stream = context.stream();
        let plan = boundary_plan();
        let vocab_size = SYNTHETIC_VOCAB_SIZE + BOUNDARY_TOKENS.len();
        let mut values = vec![-100.0f32; vocab_size];
        values[b'x' as usize] = 100.0;
        values[b'{' as usize] = 10.0;
        let logits = Array::from_slice(&values, &[1, vocab_size as i32]);
        let mut sampler = constrained_sampler(DefaultSampler, &plan).unwrap();
        let mut optimistic = sampler.clone();

        let processed = optimistic
            .process_logits(&logits, 0.0, &[TRIGGER_AND_ARGUMENT_TOKEN], stream)
            .unwrap();
        let selected = optimistic
            .sample_processed(&processed, 0.0, None, stream)
            .unwrap();
        eval([&selected]).unwrap();

        assert_eq!(selected.item::<u32>(stream), u32::from(b'{'));
        assert!(!sampler.controller().constraint_is_active());
        sampler
            .commit_token(&logits, TRIGGER_AND_ARGUMENT_TOKEN, stream)
            .unwrap();
        sampler
            .commit_token(&processed, u32::from(b'{'), stream)
            .unwrap();
        assert!(sampler.controller().constraint_is_active());
    }

    #[test]
    fn runtime_plan_creates_independent_sampler_instances() {
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::Auto);
        let logits = placeholder_logits();
        let mut first = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
        let mut second = constrained_sampler(CountingPolicy::default(), &plan).unwrap();

        commit_bytes(&mut first, AUTO_TRIGGER, &logits, stream);

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

        assert!(exact.supports_exact_optimistic_promotion());
        assert!(!adaptive.supports_exact_optimistic_promotion());
    }

    #[test]
    fn none_masks_and_rejects_the_tool_call_trigger() {
        let context = test_context();
        let stream = context.stream();
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
            stream,
        );

        let final_trigger_byte = *AUTO_TRIGGER.last().unwrap();
        let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
        values[final_trigger_byte as usize] = 100.0;
        values[b'x' as usize] = 10.0;
        let raw = Array::from_slice(&values, &[1, SYNTHETIC_VOCAB_SIZE as i32]);
        let selected = Sampler::sample(&mut sampler, &raw, 0.0, None, stream).unwrap();
        eval([&selected]).unwrap();
        assert_eq!(selected.item::<u32>(stream), u32::from(b'x'));

        let mut rejecting = constrained_sampler(CountingPolicy::default(), &plan).unwrap();
        commit_bytes(
            &mut rejecting,
            &AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1],
            &logits,
            stream,
        );
        let error = rejecting
            .commit_token(&logits, u32::from(final_trigger_byte), stream)
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
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::None);
        let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
        let mut sampler = constrained_sampler(policy, &plan).unwrap();
        let final_trigger_byte = *AUTO_TRIGGER.last().unwrap();
        let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
        values[final_trigger_byte as usize] = 100.0;
        values[b'x' as usize] = 10.0;
        let raw = Array::from_slice(&values, &[1, SYNTHETIC_VOCAB_SIZE as i32]);
        let history = AUTO_TRIGGER[..AUTO_TRIGGER.len() - 1]
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();

        let processed = sampler.process_logits(&raw, 0.0, &history, stream).unwrap();
        let selected = sampler
            .sample_processed(&processed, 0.0, None, stream)
            .unwrap();
        eval([&selected]).unwrap();
        assert_eq!(selected.item::<u32>(stream), u32::from(b'x'));
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
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::Required);
        let logits = placeholder_logits();
        let mut sampler = constrained_sampler(CountingPolicy::default(), &plan).unwrap();

        assert!(sampler.controller().constraint_is_active());
        let initial = sampler.controller_mut().valid_token_ids().unwrap().unwrap();
        assert!(initial.contains(&u32::from(b'{')));
        sampler
            .commit_token(&logits, u32::from(b'{'), stream)
            .unwrap();
        let after_open = sampler.controller_mut().valid_token_ids().unwrap().unwrap();
        let mut fork = sampler.clone();
        assert_eq!(
            fork.controller_mut().valid_token_ids().unwrap().unwrap(),
            after_open
        );

        sampler
            .commit_token(&logits, u32::from(b'"'), stream)
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
        let context = test_context();
        let stream = context.stream();
        let plan = synthetic_plan(ToolChoice::Auto);
        let policy = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);
        let mut sampler = constrained_sampler(policy, &plan).unwrap();
        let mut values = vec![-100.0f32; SYNTHETIC_VOCAB_SIZE];
        values[b'x' as usize] = 100.0;
        values[b'[' as usize] = 10.0;
        let raw = Array::from_slice(&values, &[1, SYNTHETIC_VOCAB_SIZE as i32]);
        let history = AUTO_TRIGGER
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();

        let processed = sampler.process_logits(&raw, 0.0, &history, stream).unwrap();
        let selected = sampler
            .sample_processed(&processed, 0.0, None, stream)
            .unwrap();
        eval([&selected]).unwrap();

        assert_eq!(selected.item::<u32>(stream), u32::from(b'['));
        assert!(!sampler.controller().constraint_is_active());
        assert!(!sampler
            .prefix_is_complete(
                &COMPLETE_CALL[..COMPLETE_CALL.len() - 1]
                    .iter()
                    .copied()
                    .map(u32::from)
                    .collect::<Vec<_>>()
            )
            .unwrap());
        assert!(sampler
            .prefix_is_complete(
                &COMPLETE_CALL
                    .iter()
                    .copied()
                    .map(u32::from)
                    .collect::<Vec<_>>()
            )
            .unwrap());
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

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn mirostat_v2_samples_and_updates_mu() {
        use safemlx::{
            random::{self, RandomState},
            Array, Device, DeviceType, ExecutionContext,
        };

        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let logits = Array::from_slice(&[0.0f32, -100.0, -100.0], &[1, 3]);
        let mut state = RandomState::from_key(random::key(0).unwrap());
        let mut sampler = MirostatV2Sampler::new(5.0, 0.1).unwrap();

        let token = sampler
            .sample(&logits, 1.0, Some(&mut state), stream)
            .unwrap();

        assert_eq!(token.item::<u32>(stream), 0);
        assert!(sampler.mu() > 10.0);
        assert_eq!(sampler.generated_tokens(), &[0]);
    }
}
