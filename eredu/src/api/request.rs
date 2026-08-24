//! Prepared-chat request types and semantic generation machinery.

use std::num::NonZeroUsize;

use eredu_core::{
    generation::{
        FinishReason, GenerationCancellationToken, GenerationConfigOverrides, SemanticEvent,
    },
    MtpSchedulerOptions, SpeculativeDraft, SpeculativeOutputError, SpeculativeSemanticState,
    TokenFilter, TokenFilterController,
};
use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};

use super::{TextDecoderError, TextModelError};
use crate::api::TextDecoder;
use crate::runtime::chat::constraints::{
    ConstraintCompiler, ConstraintController, ConstraintError,
};
use crate::runtime::chat::SemanticRuntimePlan;
use crate::runtime::chat::{
    prepare_format_profile, resolve_structural_tokens, CapabilitySupport, ChatCapabilities,
    ChatTemplateIdentity, ChatTemplateRequest, NativeToolSupport, PreparedChat, SemanticSupport,
    ToolChoice,
};
use crate::runtime::generation::streaming::{
    CommittedTokenPipeline, CommittedTokenSource, RawTokenDecoder, TokenDecoderBackend,
};
use std::collections::HashMap;

/// Model sampling and stopping settings for one prepared chat generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreparedChatGenerationSettings {
    /// Typed overrides layered over checkpoint-declared generation settings.
    pub overrides: GenerationConfigOverrides,
    /// Deterministic root seed used by the selected backend for stochastic sampling.
    pub seed: u64,
}

/// Explicit prompt source for structured generation from a [`PreparedChat`].
///
/// The prepared chat always owns the checkpoint-native generation semantics.
/// The selected variant determines only how the model prompt is prefetched.
pub enum PreparedChatInput<'a, B: eredu_core::TextGenerationBackend> {
    /// Tokenize and prefill the rendered prompt stored in the prepared chat.
    RenderedPrompt(&'a PreparedChat),
    /// Prefill an already-tokenized and backend-prepared model input.
    ///
    /// The caller must ensure that `model_input` represents the same rendered
    /// conversation as `prepared_chat`. This variant supports ordered image,
    /// audio, and video parts without discarding the chat's tool runtime plan.
    PreparedBackendInput {
        /// Prepared chat that supplies generation and semantic-streaming state.
        prepared_chat: &'a PreparedChat,
        /// Backend-owned prompt supplied directly to model prefill.
        prompt: B::Prompt,
    },
}

impl<'a, B: eredu_core::TextGenerationBackend> PreparedChatInput<'a, B> {
    /// Creates a text-only input from the prepared chat's rendered prompt.
    pub const fn rendered_prompt(prepared_chat: &'a PreparedChat) -> Self {
        Self::RenderedPrompt(prepared_chat)
    }

    /// Binds an opaque backend-prepared prompt to prepared-chat semantics.
    pub fn prepared_backend_input(prepared_chat: &'a PreparedChat, prompt: B::Prompt) -> Self {
        Self::PreparedBackendInput {
            prepared_chat,
            prompt,
        }
    }

    /// Returns the prepared chat that owns generation semantics.
    pub const fn prepared_chat(&self) -> &'a PreparedChat {
        match self {
            Self::RenderedPrompt(prepared_chat)
            | Self::PreparedBackendInput { prepared_chat, .. } => prepared_chat,
        }
    }

    /// Returns the explicitly prepared backend prompt, when present.
    pub const fn backend_prompt(&self) -> Option<&B::Prompt> {
        match self {
            Self::RenderedPrompt(_) => None,
            Self::PreparedBackendInput { prompt, .. } => Some(prompt),
        }
    }
}

/// Cohesive request for ordinary structured generation from a [`PreparedChat`].
///
/// Cache and execution-stream ownership belong to the selected backend
/// session. The prepared chat's portable constraint controller supplies a
/// vocabulary filter to the backend before each sampling submission.
pub struct PreparedChatGenerationRequest<'a, B: eredu_core::TextGenerationBackend, F> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a, B>,
    /// Portable sampling configuration, token limit, and random seed.
    pub settings: PreparedChatGenerationSettings,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// One-shot cooperative-cancellation token for this request.
    pub cancellation: GenerationCancellationToken,
    /// Called synchronously as each semantic event becomes available.
    pub on_event: F,
}

/// Terminal metadata returned by ordinary prepared-chat generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedChatGenerationOutput {
    /// Every committed generated tokenizer id; cancellation returns only its prefix.
    pub token_ids: Vec<u32>,
    /// Deterministically selected terminal condition.
    pub finish_reason: FinishReason,
}

/// Failure from ordinary prepared-chat generation on backend `E`.
///
/// Backend failures remain strongly typed all the way to the caller. Portable
/// tokenizer, constraint, semantic-streaming, and generation-lifecycle
/// failures have distinct variants and never masquerade as backend errors.
#[derive(Debug, thiserror::Error)]
pub enum PreparedChatError<E: std::error::Error + Send + Sync + 'static> {
    /// The selected backend failed submission, completion, or token extraction.
    #[error("selected backend failed prepared-chat generation: {0}")]
    Backend(#[source] E),
    /// Portable constraint construction or advancement failed.
    #[error(transparent)]
    Constraint(#[from] ConstraintError),
    /// Portable generation lifecycle state was invalid.
    #[error(transparent)]
    Generation(#[from] crate::core::generation::GenerationError),
    /// Prompt or incremental token decoding failed.
    #[error(transparent)]
    Tokenizer(#[from] TextDecoderError),
    /// The prepared semantic plan or event stream was invalid.
    #[error("prepared-chat semantic generation failed: {0}")]
    Semantic(String),
    /// The backend ended its stream without a terminal token.
    #[error("backend generation ended without a terminal token")]
    MissingTerminalToken,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum PreparedChatSetupError {
    #[error(transparent)]
    Constraint(#[from] ConstraintError),
    #[error("{0}")]
    Semantic(String),
}

/// MTP-specific controls for one prepared-chat request.
#[derive(Debug, Clone, Copy)]
pub struct PreparedChatMtpGenerationOptions {
    /// Maximum assistant proposals verified in one target block.
    pub max_draft_tokens: NonZeroUsize,
    /// Canonical scheduler controls.
    pub scheduler: MtpSchedulerOptions,
}

impl Default for PreparedChatMtpGenerationOptions {
    fn default() -> Self {
        Self {
            max_draft_tokens: NonZeroUsize::new(4).expect("4 is non-zero"),
            scheduler: MtpSchedulerOptions::default(),
        }
    }
}

/// Failure while the facade prepares or a backend executes speculative chat.
#[derive(Debug, thiserror::Error)]
pub enum PreparedChatMtpError<E: std::error::Error + Send + Sync + 'static> {
    /// The selected backend failed prompt preparation or speculative execution.
    #[error("selected backend failed prepared-chat speculative generation: {0}")]
    Backend(#[source] E),
    /// Portable generation configuration was invalid.
    #[error(transparent)]
    Generation(#[from] crate::core::generation::GenerationError),
    /// Portable tokenizer preparation failed.
    #[error(transparent)]
    Text(#[from] TextModelError),
    /// Portable constraint construction failed.
    #[error(transparent)]
    Constraint(#[from] ConstraintError),
    /// The prepared semantic plan or parser state was invalid.
    #[error("prepared-chat semantic generation failed: {0}")]
    Semantic(String),
    /// A backend returned the wrong number of results for the submitted lanes.
    #[error(
        "selected backend returned {actual} speculative results, but the facade expected {expected}"
    )]
    OutputCardinality {
        /// Number of results required by the facade operation.
        expected: usize,
        /// Number of results returned by the backend.
        actual: usize,
    },
}

/// One speculative MTP response from a [`PreparedChat`].
pub struct PreparedChatMtpGenerationRequest<'a, B: eredu_core::TextGenerationBackend, D, F> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a, B>,
    /// Embedded or separately loaded draft-model selection.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Portable sampling configuration, token limit, and random seed.
    pub settings: PreparedChatGenerationSettings,
    /// Proposal-block and scheduler controls.
    pub options: PreparedChatMtpGenerationOptions,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// One-shot cooperative-cancellation token for this request.
    pub cancellation: GenerationCancellationToken,
    /// Called synchronously for each event after its cache transaction commits.
    pub on_event: F,
}

/// One independently executable lane in a prepared-chat MTP batch.
///
/// Every lane owns portable sampling, a random root, stop configuration, and
/// callback. The facade constructs its constraint and decoder/parser pipeline
/// before dispatch; the backend allocates only its execution cache and sampling
/// state. Lanes are shared by the external-assistant and embedded-head drafting
/// strategies.
pub struct PreparedChatMtpBatchLane<'a, B: eredu_core::TextGenerationBackend> {
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a, B>,
    /// Portable sampling configuration, token limit, and independent random root.
    pub settings: PreparedChatGenerationSettings,
    /// Maximum assistant proposals verified in one target block.
    pub max_draft_tokens: NonZeroUsize,
    /// Additional decoded text sequences that terminate only this lane.
    pub caller_stop_sequences: &'a [String],
    /// One-shot cooperative-cancellation token owned only by this lane.
    pub cancellation: GenerationCancellationToken,
    /// Called synchronously for canonical events from only this lane.
    pub on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

/// Cohesive fair-scheduler request for independent prepared-chat MTP lanes.
pub struct PreparedChatMtpBatchRequest<'a, B: eredu_core::TextGenerationBackend, D> {
    /// Embedded or separately loaded draft-model selection.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Independently executable prepared-chat lanes.
    pub lanes: Vec<PreparedChatMtpBatchLane<'a, B>>,
    /// Bounded scheduler and optimistic-lookahead controls.
    pub scheduler: MtpSchedulerOptions,
}

/// Facade-prepared speculative grammar state consumed by a backend sampler.
///
/// The facade constructs this state from the prepared chat exactly once. A
/// backend applies its portable filters to backend-owned logits and commits
/// only tokens accepted by target verification.
#[derive(Clone)]
pub struct PreparedChatSpeculativeConstraint {
    controller: ConstraintController,
}

impl PreparedChatSpeculativeConstraint {
    pub(super) fn from_prepared_chat(
        prepared_chat: &PreparedChat,
    ) -> Result<Self, ConstraintError> {
        let generation_plan = prepared_chat
            .generation_runtime_plan()
            .expect("supported prepared chats carry a generation runtime plan");
        Ok(Self {
            controller: ConstraintController::from_generation_plan(generation_plan)?,
        })
    }
}

impl TokenFilterController for PreparedChatSpeculativeConstraint {
    type Error = ConstraintError;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.controller.current_filter()
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error> {
        self.controller.commit_token(token_id)
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.controller.is_complete()
    }
}

impl eredu_core::SpeculativeTokenFilterController for PreparedChatSpeculativeConstraint {
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error> {
        self.controller.filter_at(history)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        self.controller.prefix_is_complete(history)
    }
}

#[derive(Clone)]
pub(super) struct PreparedChatTokenDecoder {
    pub(super) decoder: TextDecoder,
}

impl TokenDecoderBackend for PreparedChatTokenDecoder {
    type Error = TextDecoderError;

    fn decode_token(
        &mut self,
        token_id: u32,
        _preserve_special: bool,
    ) -> Result<Vec<u8>, Self::Error> {
        let decoded = self.decoder.step(token_id)?.unwrap_or_default();
        Ok(decoded.into_bytes())
    }

    fn finish(&mut self) -> Result<Vec<u8>, Self::Error> {
        let decoded = self
            .decoder
            .tokenizer
            .decode(&self.decoder.ids, self.decoder.skip_special_tokens)
            .map_err(TextDecoderError::Tokenizer)?;
        if decoded.len() > self.decoder.prefix.len() {
            return Err(TextDecoderError::IncompleteByteSequence);
        }
        Ok(Vec::new())
    }
}

pub(super) struct PreparedChatControlRuntime {
    pub(super) controller: ConstraintController,
    pub(super) parser: crate::runtime::generation::streaming::ToolRuntimeParser,
    pub(super) structural_tokens: HashMap<u32, String>,
}

pub(super) struct PreparedChatSemanticState {
    initial_decoder: PreparedChatTokenDecoder,
    plan: SemanticRuntimePlan,
    caller_stop_sequences: Vec<String>,
    pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    token_ids: Vec<u32>,
    events: Vec<SemanticEvent>,
}

impl PreparedChatSemanticState {
    pub(super) fn new(
        initial_decoder: PreparedChatTokenDecoder,
        plan: SemanticRuntimePlan,
        caller_stop_sequences: &[String],
    ) -> Result<Self, SpeculativeOutputError> {
        let pipeline = Self::build_pipeline(initial_decoder.clone(), &plan, caller_stop_sequences)?;
        Ok(Self {
            initial_decoder,
            plan,
            caller_stop_sequences: caller_stop_sequences.to_vec(),
            pipeline,
            token_ids: Vec::new(),
            events: Vec::new(),
        })
    }

    fn build_pipeline(
        decoder: PreparedChatTokenDecoder,
        plan: &SemanticRuntimePlan,
        caller_stop_sequences: &[String],
    ) -> Result<CommittedTokenPipeline<PreparedChatTokenDecoder>, SpeculativeOutputError> {
        let parser = plan
            .create_parser_with_stops(caller_stop_sequences.iter().map(String::as_str))
            .map_err(|error| SpeculativeOutputError::semantic("create parser", error))?;
        Ok(CommittedTokenPipeline::new(
            RawTokenDecoder::with_structural_tokens(
                decoder,
                plan.structural_tokens()
                    .map(|(id, spelling)| (id, spelling.to_owned())),
            ),
            parser,
        ))
    }
}

impl SpeculativeSemanticState for PreparedChatSemanticState {
    fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError> {
        let mut pipeline = Self::build_pipeline(
            self.initial_decoder.clone(),
            &self.plan,
            &self.caller_stop_sequences,
        )?;
        for &token in &self.token_ids {
            pipeline.push(token, &mut |_| {}).map_err(|error| {
                SpeculativeOutputError::semantic("replay token", error.to_string())
            })?;
        }
        Ok(Box::new(Self {
            initial_decoder: self.initial_decoder.clone(),
            plan: self.plan.clone(),
            caller_stop_sequences: self.caller_stop_sequences.clone(),
            pipeline,
            token_ids: self.token_ids.clone(),
            events: Vec::new(),
        }))
    }

    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
        let matched = self
            .pipeline
            .push(token, &mut |event| self.events.push(event))
            .map_err(|error| SpeculativeOutputError::semantic("push token", error.to_string()))?;
        self.token_ids.push(token);
        Ok(matched)
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        self.pipeline
            .finish(reason, &mut |event| self.events.push(event))
            .map_err(|error| SpeculativeOutputError::semantic("finish", error.to_string()))
    }

    fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
        self.pipeline.cancel(&mut |event| self.events.push(event));
        Ok(())
    }

    fn take_events(&mut self) -> Vec<SemanticEvent> {
        std::mem::take(&mut self.events)
    }
}

pub(super) fn prepared_chat_control_runtime(
    prepared_chat: &PreparedChat,
    caller_stop_sequences: &[String],
) -> Result<PreparedChatControlRuntime, PreparedChatSetupError> {
    let semantic_plan = match prepared_chat.semantic_support() {
        SemanticSupport::Supported => prepared_chat
            .semantic_runtime_plan()
            .expect("supported prepared chats carry a semantic runtime plan"),
        SemanticSupport::Unsupported { reason } => {
            return Err(PreparedChatSetupError::Semantic(format!(
                "prepared chat does not have an executable semantic plan: {reason}"
            )));
        }
    };
    let generation_plan = prepared_chat
        .generation_runtime_plan()
        .expect("supported prepared chats carry a generation runtime plan");
    let controller = ConstraintController::from_generation_plan(generation_plan)?;
    let parser = semantic_plan
        .create_parser_with_stops(caller_stop_sequences.iter().map(String::as_str))
        .map_err(PreparedChatSetupError::Semantic)?;
    let structural_tokens = semantic_plan
        .structural_tokens()
        .map(|(id, spelling)| (id, spelling.to_owned()))
        .collect();
    Ok(PreparedChatControlRuntime {
        controller,
        parser,
        structural_tokens,
    })
}

pub(super) struct BackendGenerationTokenSource<'a, B>
where
    B: eredu_core::TextGenerationBackend,
{
    pub(super) generator: eredu_core::ControlledTextGeneration<'a, B, ConstraintController>,
}

impl<B> CommittedTokenSource for BackendGenerationTokenSource<'_, B>
where
    B: eredu_core::TextGenerationBackend,
{
    type Error = eredu_core::ControlledTextGenerationError<B::Error, ConstraintError>;

    fn next_token(&mut self) -> Result<Option<u32>, Self::Error> {
        self.generator
            .next()
            .transpose()
            .map(|token| token.map(|token| token.token_id()))
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.generator
            .controller_mut()
            .grammar_is_complete()
            .map_err(eredu_core::ControlledTextGenerationError::Controller)
    }
}

fn capability(condition: bool, reason: impl Into<String>) -> CapabilitySupport {
    if condition {
        CapabilitySupport::Supported
    } else {
        CapabilitySupport::Unsupported {
            reason: reason.into(),
        }
    }
}

fn validate_qwen_tagged_history(messages: &[serde_json::Value]) -> Result<(), TextModelError> {
    for (message_index, message) in messages.iter().enumerate() {
        let Some(tool_calls) = message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        for (call_index, call) in tool_calls.iter().enumerate() {
            let function = call.get("function").unwrap_or(call);
            let Some(arguments) = function.get("arguments") else {
                continue;
            };
            let arguments = arguments.as_object().ok_or_else(|| {
                TextModelError::ToolConstraint(format!(
                    "messages[{message_index}].tool_calls[{call_index}].function.arguments must be a mapping for Qwen tagged-parameter templates; serialized strings are unsupported"
                ))
            })?;
            for (name, value) in arguments {
                if name.is_empty()
                    || name
                        .chars()
                        .any(|character| matches!(character, '<' | '>' | '\r' | '\n'))
                {
                    return Err(TextModelError::ToolConstraint(format!(
                        "messages[{message_index}].tool_calls[{call_index}] contains unsafe tagged parameter name {name:?}"
                    )));
                }
                if value
                    .as_str()
                    .is_some_and(|value| value.contains("\n</parameter>"))
                {
                    return Err(TextModelError::ToolConstraint(format!(
                        "messages[{message_index}].tool_calls[{call_index}] parameter {name:?} contains the unescaped tagged-parameter closing delimiter"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn recognize_gemma_protocol(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
) -> Option<crate::runtime::chat::PreparedFormatProfile> {
    use crate::runtime::chat::{
        dialect::{DialectParameters, GenerationPromptBehavior},
        gemma::{
            self, CHANNEL_CLOSE, CHANNEL_OPEN, STRING_DELIMITER, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
            TOOL_RESPONSE_OPEN, TURN_CLOSE,
        },
        GEMMA4_STRUCTURAL_TOOL_SPEC,
    };

    let channel_tokens = [CHANNEL_OPEN.to_owned(), CHANNEL_CLOSE.to_owned()];
    let channel_ids = match resolve_structural_tokens(tokenizer, &channel_tokens) {
        Ok(ids) => ids,
        Err(_) => return None,
    };
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "eredu_probe",
            "description": "protocol probe",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }
        }
    })];
    let reasoning_sentinel = "__eredu_reasoning_probe_7c91__";
    let visible_sentinel = "__eredu_visible_probe_28ad__";
    let probe_messages = vec![
        serde_json::json!({"role": "user", "content": "__eredu_user_probe__"}),
        serde_json::json!({
            "role": "assistant",
            "reasoning_content": reasoning_sentinel,
            "content": visible_sentinel,
            "tool_calls": [{
                "id": "reasoning-probe-call",
                "type": "function",
                "function": {
                    "name": "eredu_probe",
                    "arguments": {"value": "reasoning-probe"}
                }
            }]
        }),
    ];
    let rendered = match tokenizer.apply_chat_template_json(
        selected_template.clone(),
        [probe_messages],
        Some(&tools),
        model_id,
        false,
        None,
    ) {
        Ok(rendered) => rendered.into_iter().next()?,
        Err(_) => return None,
    };
    let reasoning_frame = format!("{CHANNEL_OPEN}thought\n{reasoning_sentinel}\n{CHANNEL_CLOSE}");
    if !rendered.contains(&reasoning_frame) || !rendered.contains(visible_sentinel) {
        return None;
    }
    debug_assert_eq!(channel_ids.len(), 2);

    let mut thinking_on = serde_json::Map::new();
    thinking_on.insert("enable_thinking".into(), serde_json::Value::Bool(true));
    let mut thinking_off = serde_json::Map::new();
    thinking_off.insert("enable_thinking".into(), serde_json::Value::Bool(false));
    let generation_messages =
        vec![serde_json::json!({"role": "user", "content": "__eredu_prompt_probe__"})];
    for kwargs in [&thinking_on, &thinking_off] {
        let with_prompt = match tokenizer.apply_chat_template_json(
            selected_template.clone(),
            [generation_messages.clone()],
            Some(&[]),
            model_id,
            true,
            Some(kwargs),
        ) {
            Ok(rendered) => rendered.into_iter().next()?,
            Err(_) => return None,
        };
        if !with_prompt.contains("<|turn>model\n") {
            return None;
        }
    }

    let mut semantic_tokens = channel_tokens.to_vec();
    let mut semantic_stops = Vec::new();
    for spelling in [TOOL_RESPONSE_OPEN, TURN_CLOSE] {
        if resolve_structural_tokens(tokenizer, &[spelling.to_owned()]).is_ok() {
            semantic_tokens.push(spelling.to_owned());
            semantic_stops.push(spelling.to_owned());
        }
    }

    let full_tool_tokens = [
        CHANNEL_OPEN,
        CHANNEL_CLOSE,
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
        STRING_DELIMITER,
        TOOL_RESPONSE_OPEN,
        TURN_CLOSE,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    let tool_tokens_valid = resolve_structural_tokens(tokenizer, &full_tool_tokens).is_ok();
    let mapping_tool_messages = vec![
        serde_json::json!({"role": "user", "content": "probe"}),
        serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "probe-call",
                "type": "function",
                "function": {
                    "name": "eredu_probe",
                    "arguments": {"value": "probe-value"}
                }
            }]
        }),
        serde_json::json!({
            "role": "tool",
            "tool_call_id": "probe-call",
            "content": "probe-response"
        }),
    ];
    let mapping_tool_arguments = tool_tokens_valid
        && tokenizer
            .apply_chat_template_json(
                selected_template.clone(),
                [mapping_tool_messages],
                Some(&tools),
                model_id,
                true,
                Some(&thinking_on),
            )
            .ok()
            .and_then(|rendered| rendered.into_iter().next())
            .is_some_and(|rendered| {
                rendered.contains(&format!("{TOOL_CALL_OPEN}call:eredu_probe{{"))
                    && rendered.contains(TOOL_CALL_CLOSE)
                    && rendered.contains(TOOL_RESPONSE_OPEN)
                    && rendered.contains("probe-response")
            });
    let string_tool_messages = vec![
        serde_json::json!({"role": "user", "content": "probe"}),
        serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "probe-call",
                "type": "function",
                "function": {
                    "name": "eredu_probe",
                    "arguments": "{\"value\":\"probe-value\"}"
                }
            }]
        }),
        serde_json::json!({
            "role": "tool",
            "tool_call_id": "probe-call",
            "content": "probe-response"
        }),
    ];
    let string_tool_arguments = tool_tokens_valid
        && tokenizer
            .apply_chat_template_json(
                selected_template.clone(),
                [string_tool_messages],
                Some(&tools),
                model_id,
                true,
                Some(&thinking_on),
            )
            .ok()
            .and_then(|rendered| rendered.into_iter().next())
            .is_some_and(|rendered| {
                rendered.contains(&format!("{TOOL_CALL_OPEN}call:eredu_probe{{"))
                    && rendered.contains(TOOL_CALL_CLOSE)
                    && rendered.contains(TOOL_RESPONSE_OPEN)
                    && rendered.contains("probe-response")
            });
    let tool_output_protocol = mapping_tool_arguments || string_tool_arguments;
    let tool_input_rendering = tool_output_protocol;

    Some(crate::runtime::chat::PreparedFormatProfile {
        identity: Some("gemma.channels.v1".into()),
        dialect: Some(&gemma::GEMMA_CHANNEL_DIALECT),
        dialect_parameters: Some(gemma::parameters()),
        tool_dialect: tool_output_protocol.then_some(&gemma::GEMMA_TOOL_DIALECT),
        tool_dialect_parameters: tool_output_protocol
            .then_some(DialectParameters::Declarative(&GEMMA4_STRUCTURAL_TOOL_SPEC)),
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_control: crate::runtime::chat::ReasoningTemplateControl::Boolean(
            "enable_thinking",
        ),
        reasoning_effort_control: None,
        supports_reasoning_parsing: true,
        supports_tool_reasoning: true,
        supports_tool_input_rendering: tool_input_rendering,
        supports_mapping_tool_arguments: mapping_tool_arguments,
        supports_string_tool_arguments: string_tool_arguments,
        native_tool_unavailable_reason: (!tool_input_rendering).then(|| {
            "Gemma reasoning channels were recognized, but tool rendering probes failed".into()
        }),
        required_structural_tokens: semantic_tokens,
        tool_required_structural_tokens: if tool_output_protocol {
            full_tool_tokens
        } else {
            Vec::new()
        },
        stop_sequences: semantic_stops,
    })
}

fn recognize_inkling_protocol(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
) -> Option<crate::runtime::chat::PreparedFormatProfile> {
    use crate::runtime::chat::{
        dialect::GenerationPromptBehavior,
        inkling::{
            self, CONTENT_INVOKE_TOOL_JSON, CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE,
            END_SAMPLING, MESSAGE_MODEL,
        },
        ReasoningTemplateControl,
    };

    let structural_tokens = [
        MESSAGE_MODEL,
        CONTENT_TEXT,
        CONTENT_THINKING,
        END_MESSAGE,
        END_SAMPLING,
    ]
    .map(str::to_owned)
    .to_vec();
    resolve_structural_tokens(tokenizer, &structural_tokens).ok()?;

    let reasoning_sentinel = "__eredu_inkling_reasoning_probe_7c91__";
    let visible_sentinel = "__eredu_inkling_visible_probe_28ad__";
    let rendered = tokenizer
        .apply_chat_template_json(
            selected_template.clone(),
            [vec![
                serde_json::json!({"role": "user", "content": "__eredu_inkling_user_probe__"}),
                serde_json::json!({
                    "role": "assistant",
                    "reasoning_content": reasoning_sentinel,
                    "content": visible_sentinel,
                }),
            ]],
            Some(&[]),
            model_id,
            false,
            None,
        )
        .ok()?
        .into_iter()
        .next()?;
    let assistant_frames = format!(
        "{MESSAGE_MODEL}{CONTENT_THINKING}{reasoning_sentinel}{END_MESSAGE}{MESSAGE_MODEL}{CONTENT_TEXT}{visible_sentinel}{END_MESSAGE}{END_SAMPLING}"
    );
    if !rendered.contains(&assistant_frames) {
        return None;
    }

    let generation_messages =
        vec![serde_json::json!({"role": "user", "content": "__eredu_inkling_prompt_probe__"})];
    for (effort, expected) in [("none", "0"), ("high", "0.9")] {
        let kwargs = serde_json::Map::from_iter([(
            "reasoning_effort".into(),
            serde_json::Value::String(effort.into()),
        )]);
        let with_prompt = tokenizer
            .apply_chat_template_json(
                selected_template.clone(),
                [generation_messages.clone()],
                Some(&[]),
                model_id,
                true,
                Some(&kwargs),
            )
            .ok()?
            .into_iter()
            .next()?;
        let effort_frame = format!(
            "<|message_system|>{CONTENT_TEXT}Thinking effort level: {expected}{END_MESSAGE}"
        );
        if !with_prompt.contains(&effort_frame) || !with_prompt.ends_with(MESSAGE_MODEL) {
            return None;
        }
    }

    let tool_structural_tokens = [
        MESSAGE_MODEL,
        CONTENT_TEXT,
        CONTENT_THINKING,
        CONTENT_INVOKE_TOOL_JSON,
        END_MESSAGE,
        END_SAMPLING,
    ]
    .map(str::to_owned)
    .to_vec();
    let tool_tokens_valid = resolve_structural_tokens(tokenizer, &tool_structural_tokens).is_ok();
    let mapping_tool_arguments = tool_tokens_valid
        && render_protocol_probe(
            tokenizer,
            selected_template,
            model_id,
            serde_json::json!({"value": "probe-value"}),
        )
        .is_some_and(|rendered| {
            rendered.contains(concat!(
                "<|message_system|>tool_declare<|content_xml|>",
                "[{\"description\":\"protocol recognition probe\",\"name\":",
                "\"eredu_probe_7c91\",\"parameters\":"
            )) && rendered.contains(concat!(
                "<|message_model|>eredu_probe_7c91<|content_invoke_tool_json|>",
                "{\"name\":\"eredu_probe_7c91\",\"args\":{\"value\":\"probe-value\"}}",
                "<|end_message|>"
            )) && rendered.contains(concat!(
                "<|message_tool|>eredu_probe_7c91<|content_text|>",
                "\"__eredu_tool_result_probe__\"<|end_message|>"
            ))
        });
    let string_tool_arguments = tool_tokens_valid
        && render_protocol_probe(
            tokenizer,
            selected_template,
            model_id,
            serde_json::Value::String("{\"value\":\"probe-value\"}".into()),
        )
        .is_some_and(|rendered| {
            rendered.contains(CONTENT_INVOKE_TOOL_JSON)
                && rendered.contains("__eredu_tool_result_probe__")
        });
    let tool_output_protocol = mapping_tool_arguments || string_tool_arguments;

    Some(crate::runtime::chat::PreparedFormatProfile {
        identity: Some("inkling.messages.v1".into()),
        dialect: Some(&inkling::INKLING_MESSAGE_DIALECT),
        dialect_parameters: Some(inkling::parameters()),
        tool_dialect: tool_output_protocol.then_some(&inkling::INKLING_TOOL_DIALECT),
        tool_dialect_parameters: tool_output_protocol.then_some(inkling::parameters()),
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_control: ReasoningTemplateControl::NamedEffort {
            kwarg: "reasoning_effort",
            enabled: "high",
            disabled: "none",
        },
        reasoning_effort_control: None,
        supports_reasoning_parsing: true,
        supports_tool_reasoning: true,
        supports_tool_input_rendering: tool_output_protocol,
        supports_mapping_tool_arguments: mapping_tool_arguments,
        supports_string_tool_arguments: string_tool_arguments,
        native_tool_unavailable_reason: (!tool_output_protocol).then(|| {
            "Inkling reasoning and visible-text frames were recognized, but tool rendering probes failed"
                .into()
        }),
        required_structural_tokens: structural_tokens,
        tool_required_structural_tokens: if tool_output_protocol {
            tool_structural_tokens
        } else {
            Vec::new()
        },
        stop_sequences: vec![END_SAMPLING.into()],
    })
}

fn recognize_muse_atem_protocol(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
) -> Option<crate::runtime::chat::PreparedFormatProfile> {
    use crate::runtime::chat::{
        atem::{self, EOM, EOT, MESSAGE, START},
        dialect::GenerationPromptBehavior,
        ReasoningEffortControl, ReasoningTemplateControl,
    };

    let structural_tokens = [START, MESSAGE, EOM, EOT].map(str::to_owned).to_vec();
    resolve_structural_tokens(tokenizer, &structural_tokens).ok()?;

    let rendered = render_reasoning_protocol_probe(tokenizer, selected_template, model_id)?;
    let expected = concat!(
        "<|start|>assistant to=self<|message|>__eredu_reasoning_probe__<|eom|>",
        "<|start|>assistant to=user<|message|>__eredu_visible_probe__<|eot|>"
    );
    if !rendered.contains(expected) {
        return None;
    }

    let generation_messages =
        vec![serde_json::json!({"role": "user", "content": "__eredu_atem_prompt__"})];
    for strength in ["low", "medium", "high", "xhigh"] {
        let kwargs = serde_json::Map::from_iter([(
            "reasoning_strength".into(),
            serde_json::Value::String(strength.into()),
        )]);
        let prompt = tokenizer
            .apply_chat_template_json(
                selected_template.clone(),
                [generation_messages.clone()],
                Some(&[]),
                model_id,
                true,
                Some(&kwargs),
            )
            .ok()?
            .into_iter()
            .next()?;
        if !prompt.contains(&format!("Reasoning strength: {strength}."))
            || !prompt.ends_with("<|start|>assistant")
        {
            return None;
        }
    }

    let mapping = render_protocol_probe(
        tokenizer,
        selected_template,
        model_id,
        serde_json::json!({"value": "probe-value"}),
    );
    let mapping_tool_arguments = mapping.as_deref().is_some_and(|rendered| {
        [
            "<|start|>assistant to=self<|message|>__eredu_reasoning_probe__<|eom|>",
            "<|start|>assistant to=eredu_probe_7c91<|message|>",
            "<atem:function_calls>",
            "<atem:invoke name=\"eredu_probe_7c91\">",
            "<atem:parameter name=\"value\">probe-value</atem:parameter>",
            "<|start|>tool eredu_probe_7c91<|message|><tool_output name=\"eredu_probe_7c91\">",
            "__eredu_tool_result_probe__",
        ]
        .iter()
        .all(|marker| rendered.contains(marker))
    });

    Some(crate::runtime::chat::PreparedFormatProfile {
        identity: Some("muse-glimmer.atem.v1".into()),
        dialect: Some(&atem::ATEM_DIALECT),
        dialect_parameters: Some(atem::parameters()),
        tool_dialect: mapping_tool_arguments.then_some(&atem::ATEM_DIALECT),
        tool_dialect_parameters: mapping_tool_arguments.then_some(atem::parameters()),
        generation_prompt_behavior: GenerationPromptBehavior::Always,
        reasoning_template_control: ReasoningTemplateControl::NamedEffort {
            kwarg: "reasoning_strength",
            enabled: "high",
            disabled: "high",
        },
        reasoning_effort_control: Some(ReasoningEffortControl {
            kwarg: "reasoning_strength",
            supported: &["low", "medium", "high", "xhigh"],
        }),
        supports_reasoning_parsing: true,
        supports_tool_reasoning: true,
        supports_tool_input_rendering: mapping_tool_arguments,
        supports_mapping_tool_arguments: mapping_tool_arguments,
        supports_string_tool_arguments: false,
        native_tool_unavailable_reason: (!mapping_tool_arguments).then(|| {
            "Muse-Glimmer ATEM channels were recognized, but mapping-valued tool history did not pass the behavioral probe".into()
        }),
        required_structural_tokens: structural_tokens.clone(),
        tool_required_structural_tokens: if mapping_tool_arguments {
            structural_tokens
        } else {
            Vec::new()
        },
        stop_sequences: vec![EOT.into()],
    })
}

fn render_protocol_probe(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
    arguments: serde_json::Value,
) -> Option<String> {
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "eredu_probe_7c91",
            "description": "protocol recognition probe",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": false
            }
        }
    })];
    let messages = vec![
        serde_json::json!({"role": "user", "content": "__eredu_user_probe__"}),
        serde_json::json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "__eredu_reasoning_probe__",
            "thinking": "__eredu_reasoning_probe__",
            "tool_calls": [{
                "id": "abc123456",
                "type": "function",
                "function": {
                    "name": "eredu_probe_7c91",
                    "arguments": arguments
                }
            }]
        }),
        serde_json::json!({
            "role": "tool",
            "name": "eredu_probe_7c91",
            "tool_call_id": "abc123456",
            "content": "\"__eredu_tool_result_probe__\""
        }),
        serde_json::json!({
            "role": "assistant",
            "content": "__eredu_intermediate_assistant_probe__"
        }),
        serde_json::json!({"role": "user", "content": "__eredu_followup_probe__"}),
    ];
    tokenizer
        .apply_chat_template_json(
            selected_template.clone(),
            [messages],
            Some(&tools),
            model_id,
            true,
            None,
        )
        .ok()?
        .into_iter()
        .next()
}

fn render_reasoning_protocol_probe(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
) -> Option<String> {
    tokenizer
        .apply_chat_template_json(
            selected_template.clone(),
            [vec![
                serde_json::json!({"role": "user", "content": "__eredu_user_probe__"}),
                serde_json::json!({
                    "role": "assistant",
                    "reasoning_content": "__eredu_reasoning_probe__",
                    "content": "__eredu_visible_probe__"
                }),
            ]],
            Some(&[]),
            model_id,
            false,
            None,
        )
        .ok()?
        .into_iter()
        .next()
}

fn recognized_dialect_profile(
    tokenizer: &ChatTokenizer,
    identity: &'static str,
    dialect: &'static dyn crate::runtime::chat::dialect::FormatDialect,
    parameters: crate::runtime::chat::dialect::DialectParameters,
    mapping_tool_arguments: bool,
    string_tool_arguments: bool,
) -> Option<crate::runtime::chat::PreparedFormatProfile> {
    let generation_prompt_behavior = dialect.generation_prompt_behavior(parameters).ok()?;
    let reasoning_template_kwarg = dialect.reasoning_template_kwarg(parameters).ok()?;
    let supports_tool_reasoning = dialect.supports_tool_reasoning(parameters).ok()?;
    let required_structural_tokens = dialect
        .required_structural_tokens(parameters)
        .ok()?
        .iter()
        .map(|token| (*token).to_owned())
        .collect::<Vec<_>>();
    let stop_sequences = dialect
        .stop_sequences(parameters)
        .ok()?
        .iter()
        .map(|stop| (*stop).to_owned())
        .collect::<Vec<_>>();
    resolve_structural_tokens(tokenizer, &required_structural_tokens).ok()?;
    let supports_tool_input_rendering = mapping_tool_arguments || string_tool_arguments;

    Some(crate::runtime::chat::PreparedFormatProfile {
        identity: Some(identity.into()),
        dialect: Some(dialect),
        dialect_parameters: Some(parameters),
        tool_dialect: Some(dialect),
        tool_dialect_parameters: Some(parameters),
        generation_prompt_behavior,
        reasoning_template_control:
            crate::runtime::chat::ReasoningTemplateControl::Boolean(reasoning_template_kwarg),
        reasoning_effort_control: None,
        supports_reasoning_parsing: dialect.supports_reasoning_parsing(parameters),
        supports_tool_reasoning,
        supports_tool_input_rendering,
        supports_mapping_tool_arguments: mapping_tool_arguments,
        supports_string_tool_arguments: string_tool_arguments,
        native_tool_unavailable_reason: (!supports_tool_input_rendering).then(|| {
            format!(
                "{identity} output was recognized, but structured tool-history rendering probes failed"
            )
        }),
        required_structural_tokens: required_structural_tokens.clone(),
        tool_required_structural_tokens: required_structural_tokens,
        stop_sequences,
    })
}

fn recognize_remaining_protocols(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
) -> Option<crate::runtime::chat::PreparedFormatProfile> {
    use crate::runtime::chat::{
        dialect::{DialectParameters, DECLARATIVE_DIALECT},
        harmony::{GPT_OSS_HARMONY_PARAMETERS, HARMONY_DIALECT},
        lfm2::{LFM2_DIALECT, LFM2_PARAMETERS},
        DEEPSEEK31_STRUCTURAL_JSON_TOOL_SPEC, DEEPSEEK_STRUCTURAL_JSON_TOOL_SPEC,
        KIMI_K2_NATIVE_TOOL_SPEC, LLAMA3_JSON_TOOL_SPEC, LLAMA4_JSON_TOOL_SPEC,
        MINISTRAL_JSON_LIST_TOOL_SPEC, MISTRAL_JSON_LIST_TOOL_SPEC,
        NEMOTRON_NANO_JSON_LIST_TOOL_SPEC, NEMOTRON_NANO_V2_JSON_LIST_TOOL_SPEC,
        QWEN3_XML_TOOL_SPEC, QWEN_XML_TOOL_SPEC,
    };

    let mapping = render_protocol_probe(
        tokenizer,
        selected_template,
        model_id,
        serde_json::json!({"value": "__eredu_mapping_argument_probe__"}),
    );
    let string = render_protocol_probe(
        tokenizer,
        selected_template,
        model_id,
        serde_json::Value::String(r#"{"value":"__eredu_string_argument_probe__"}"#.into()),
    );
    let supports = |rendered: &Option<String>, required: &[&str]| {
        rendered
            .as_deref()
            .is_some_and(|rendered| required.iter().all(|part| rendered.contains(part)))
    };

    let kimi_markers = [
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        "## Return of abc123456",
        "__eredu_tool_result_probe__",
    ];
    let kimi_mapping = supports(&mapping, &kimi_markers);
    let kimi_string = supports(&string, &kimi_markers);
    if kimi_mapping || kimi_string {
        return recognized_dialect_profile(
            tokenizer,
            "kimi-k2.native-tools.v1",
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(&KIMI_K2_NATIVE_TOOL_SPEC),
            kimi_mapping,
            kimi_string,
        );
    }

    let harmony_markers = [
        "assistant to=functions.eredu_probe_7c91",
        "<|message|>",
        "<|call|>",
        "functions.eredu_probe_7c91 to=assistant",
        "__eredu_tool_result_probe__",
    ];
    let harmony_mapping = supports(&mapping, &harmony_markers);
    let harmony_string = supports(&string, &harmony_markers);
    if harmony_mapping || harmony_string {
        return recognized_dialect_profile(
            tokenizer,
            "harmony.channels.v1",
            &HARMONY_DIALECT,
            DialectParameters::Custom(&GPT_OSS_HARMONY_PARAMETERS),
            harmony_mapping,
            harmony_string,
        );
    }

    let deepseek_common = [
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁sep｜>",
        "eredu_probe_7c91",
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
        "__eredu_tool_result_probe__",
    ];
    let deepseek_mapping = supports(&mapping, &deepseek_common);
    let deepseek_string = supports(&string, &deepseek_common);
    if deepseek_mapping || deepseek_string {
        let rendered = mapping.as_deref().or(string.as_deref())?;
        let (identity, spec) = if rendered
            .contains("<｜tool▁call▁begin｜>function<｜tool▁sep｜>eredu_probe_7c91\n```json\n")
        {
            (
                "deepseek.structural-json-tools.v1",
                &DEEPSEEK_STRUCTURAL_JSON_TOOL_SPEC,
            )
        } else if rendered.contains("<｜tool▁call▁begin｜>eredu_probe_7c91<｜tool▁sep｜>")
        {
            (
                "deepseek.structural-json-tools.v2",
                &DEEPSEEK31_STRUCTURAL_JSON_TOOL_SPEC,
            )
        } else {
            return None;
        };
        return recognized_dialect_profile(
            tokenizer,
            identity,
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(spec),
            deepseek_mapping,
            deepseek_string,
        );
    }

    let lfm2_markers = [
        "<|tool_call_start|>",
        "eredu_probe_7c91(",
        "<|tool_call_end|>",
        "__eredu_tool_result_probe__",
    ];
    let lfm2_mapping = supports(&mapping, &lfm2_markers);
    let lfm2_string = supports(&string, &lfm2_markers);
    if lfm2_mapping || lfm2_string {
        return recognized_dialect_profile(
            tokenizer,
            "lfm2.python-tools.v1",
            &LFM2_DIALECT,
            DialectParameters::Custom(&LFM2_PARAMETERS),
            lfm2_mapping,
            lfm2_string,
        );
    }

    let qwen_markers = [
        "<tool_call>",
        "eredu_probe_7c91",
        "<tool_response>",
        "__eredu_tool_result_probe__",
    ];
    let qwen_mapping = supports(&mapping, &qwen_markers);
    let qwen_string = supports(&string, &qwen_markers);
    if qwen_mapping || qwen_string {
        let tagged_mapping = mapping.as_deref().is_some_and(|rendered| {
            rendered.contains(
                "<tool_call>\n<function=eredu_probe_7c91>\n<parameter=value>\n__eredu_mapping_argument_probe__\n</parameter>\n</function>\n</tool_call>",
            )
        });
        let tagged_string = string.as_deref().is_some_and(|rendered| {
            rendered.contains("<function=eredu_probe_7c91>\n<parameter=value>")
                && rendered.contains("__eredu_string_argument_probe__")
        });
        if tagged_mapping || tagged_string {
            let mut effort_kwargs = serde_json::Map::new();
            effort_kwargs.insert("reasoning_effort".into(), serde_json::json!("low"));
            let qwen38 = tokenizer
                .apply_chat_template_json(
                    selected_template.clone(),
                    [vec![serde_json::json!({
                        "role": "user",
                        "content": "__eredu_effort_probe__"
                    })]],
                    Some(&[]),
                    model_id,
                    false,
                    Some(&effort_kwargs),
                )
                .ok()
                .and_then(|rendered| rendered.into_iter().next())
                .is_some_and(|rendered| rendered.contains("Reasoning effort is set to low."));
            let identity = if qwen38 {
                "qwen3.8.tagged-parameter-tools.v1"
            } else {
                "qwen3.6.tagged-parameter-tools.v1"
            };
            let mut profile = recognized_dialect_profile(
                tokenizer,
                identity,
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(
                    &crate::runtime::chat::QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING,
                ),
                tagged_mapping,
                false,
            )?;
            if qwen38 {
                profile.reasoning_effort_control =
                    Some(crate::runtime::chat::ReasoningEffortControl {
                        kwarg: "reasoning_effort",
                        supported: &["low", "medium", "xhigh"],
                    });
            }
            return Some(profile);
        }
        let json_in_xml = |rendered: &Option<String>| {
            rendered.as_deref().is_some_and(|rendered| {
                rendered.contains("\"name\": \"eredu_probe_7c91\"")
                    && rendered.contains("\"arguments\":")
            })
        };
        let qwen_mapping = qwen_mapping && json_in_xml(&mapping);
        let qwen_string = qwen_string && json_in_xml(&string);
        if !qwen_mapping && !qwen_string {
            return None;
        }
        let reasoning = render_reasoning_protocol_probe(tokenizer, selected_template, model_id)
            .is_some_and(|rendered| {
                rendered.contains(
                    "<think>\n__eredu_reasoning_probe__\n</think>\n\n__eredu_visible_probe__",
                )
            });
        let (identity, spec) = if reasoning {
            ("qwen.xml-tools.reasoning.v1", &QWEN3_XML_TOOL_SPEC)
        } else {
            ("xml-tools.v1", &QWEN_XML_TOOL_SPEC)
        };
        return recognized_dialect_profile(
            tokenizer,
            identity,
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(spec),
            qwen_mapping,
            qwen_string,
        );
    }

    let mistral_markers = [
        "[TOOL_CALLS]",
        "eredu_probe_7c91",
        "abc123456",
        "[TOOL_RESULTS]",
        "__eredu_tool_result_probe__",
    ];
    let mistral_mapping = supports(&mapping, &mistral_markers);
    let mistral_string = supports(&string, &mistral_markers);
    if mistral_mapping || mistral_string {
        let rendered = mapping.as_deref().or(string.as_deref())?;
        let (identity, spec) = if rendered.contains("[TOOL_CALLS] [") {
            ("mistral.json-list-tools.v1", &MISTRAL_JSON_LIST_TOOL_SPEC)
        } else if rendered.contains("[TOOL_CALLS][") {
            (
                "mistral.json-list-tools.compact.v1",
                &MINISTRAL_JSON_LIST_TOOL_SPEC,
            )
        } else {
            return None;
        };
        return recognized_dialect_profile(
            tokenizer,
            identity,
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(spec),
            mistral_mapping,
            mistral_string,
        );
    }

    let nemotron_markers = [
        "<TOOLCALL>[",
        "eredu_probe_7c91",
        "<TOOL_RESPONSE>[",
        "__eredu_tool_result_probe__",
    ];
    let nemotron_mapping = supports(&mapping, &nemotron_markers);
    let nemotron_string = supports(&string, &nemotron_markers);
    if nemotron_mapping || nemotron_string {
        let v2 = resolve_structural_tokens(tokenizer, &["<SPECIAL_12>".into()]).is_ok();
        let (identity, spec) = if v2 {
            (
                "nemotron.json-list-tools.reasoning.v1",
                &NEMOTRON_NANO_V2_JSON_LIST_TOOL_SPEC,
            )
        } else {
            (
                "nemotron.json-list-tools.v1",
                &NEMOTRON_NANO_JSON_LIST_TOOL_SPEC,
            )
        };
        return recognized_dialect_profile(
            tokenizer,
            identity,
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(spec),
            nemotron_mapping,
            nemotron_string,
        );
    }

    let llama_markers = [
        "eredu_probe_7c91",
        "\"parameters\"",
        "__eredu_tool_result_probe__",
    ];
    let llama_mapping = supports(&mapping, &llama_markers);
    let llama_string = supports(&string, &llama_markers);
    if llama_mapping || llama_string {
        let llama4 = resolve_structural_tokens(
            tokenizer,
            &[
                "<|python_start|>".into(),
                "<|python_end|>".into(),
                "<|eot|>".into(),
            ],
        )
        .is_ok();
        let (identity, spec) = if llama4 {
            ("llama.python-channel-tools.v1", &LLAMA4_JSON_TOOL_SPEC)
        } else {
            ("llama.json-tools.v1", &LLAMA3_JSON_TOOL_SPEC)
        };
        return recognized_dialect_profile(
            tokenizer,
            identity,
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(spec),
            llama_mapping,
            llama_string,
        );
    }

    None
}

pub(crate) fn prepare_chat_from_parts(
    tokenizer: &mut ChatTokenizer,
    template: ModelChatTemplate,
    model_id: &str,
    eos_token_ids: &[u32],
    constraint_compiler: Option<&Result<ConstraintCompiler, String>>,
    request: ChatTemplateRequest,
) -> Result<PreparedChat, TextModelError> {
    let selected = template.select(Some(&request.tools))?;
    let template_identity = selected.identity().clone();
    let selected_template = match &template_identity {
        ChatTemplateIdentity::Single => ModelChatTemplate::Single(selected.template().to_owned()),
        ChatTemplateIdentity::Named(name) => ModelChatTemplate::Named(
            std::collections::BTreeMap::from([(name.clone(), selected.template().to_owned())]),
        ),
    };
    let mut profile = prepare_format_profile(selected.template());
    if profile.dialect.is_none() {
        if let Some(recognized) =
            recognize_muse_atem_protocol(tokenizer, &selected_template, model_id)
                .or_else(|| recognize_gemma_protocol(tokenizer, &selected_template, model_id))
                .or_else(|| recognize_inkling_protocol(tokenizer, &selected_template, model_id))
                .or_else(|| recognize_remaining_protocols(tokenizer, &selected_template, model_id))
        {
            profile = recognized;
        }
    }
    if profile
        .identity
        .as_deref()
        .is_some_and(|identity| identity.starts_with("qwen3.6."))
        && request.reasoning_effort.is_none()
        && request
            .extra_template_kwargs
            .contains_key("reasoning_effort")
    {
        return Err(TextModelError::ToolConstraint(
            "Qwen3.6 does not expose reasoning_effort control".into(),
        ));
    }
    let extra_reasoning_effort = if request.reasoning_effort.is_none() {
        profile
            .reasoning_effort_control
            .and_then(|control| {
                request
                    .extra_template_kwargs
                    .get(control.kwarg)
                    .map(|value| (control, value))
            })
            .map(|(control, value)| {
                value.as_str().map(|value| (control, value)).ok_or_else(|| {
                    TextModelError::ToolConstraint(format!(
                        "{} must be a string for format profile {:?}",
                        control.kwarg,
                        profile.identity.as_deref().unwrap_or("unregistered")
                    ))
                })
            })
            .transpose()?
    } else {
        None
    };
    if let Some(reasoning_effort) = request
        .reasoning_effort
        .as_deref()
        .or_else(|| extra_reasoning_effort.map(|(_, value)| value))
    {
        if request.enable_thinking == Some(false) {
            return Err(TextModelError::ToolConstraint(
                "reasoning_effort cannot be combined with enable_thinking=false".into(),
            ));
        }
        let control = profile.reasoning_effort_control.ok_or_else(|| {
            TextModelError::ToolConstraint(format!(
                "format profile {:?} does not expose reasoning_effort control",
                profile.identity.as_deref().unwrap_or("unregistered")
            ))
        })?;
        if !control.supported.contains(&reasoning_effort) {
            return Err(TextModelError::ToolConstraint(format!(
                "unsupported reasoning_effort {reasoning_effort:?} for format profile {:?}; expected {}",
                profile.identity.as_deref().unwrap_or("unregistered"),
                control.supported.join(", ")
            )));
        }
    }
    let add_generation_prompt = profile
        .generation_prompt_behavior
        .resolve(request.add_generation_prompt);
    if profile.identity.as_deref().is_some_and(|identity| {
        identity.starts_with("qwen3.6.") || identity.starts_with("qwen3.8.")
    }) {
        use crate::runtime::chat::{
            QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING, QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
            QWEN_TAGGED_TOOL_SPEC_PREFILLED_REASONING,
        };
        let spec = if request.enable_thinking == Some(false) {
            &QWEN_TAGGED_TOOL_SPEC_NO_REASONING
        } else if add_generation_prompt {
            &QWEN_TAGGED_TOOL_SPEC_PREFILLED_REASONING
        } else {
            &QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING
        };
        let parameters = crate::runtime::chat::dialect::DialectParameters::Declarative(spec);
        profile.dialect_parameters = Some(parameters);
        profile.tool_dialect_parameters = Some(parameters);
        profile.supports_reasoning_parsing = request.enable_thinking != Some(false);
        validate_qwen_tagged_history(&request.messages)?;
    }
    if profile.identity.as_deref() == Some("muse-glimmer.atem.v1") {
        if request.enable_thinking == Some(false) {
            return Err(TextModelError::ToolConstraint(
                "Muse-Glimmer does not expose a reasoning-disable control; enable_thinking=false is not supported".into(),
            ));
        }
        if let Some(value) = request.extra_template_kwargs.get("reasoning_strength") {
            let Some(strength) = value.as_str() else {
                return Err(TextModelError::ToolConstraint(
                    "Muse-Glimmer reasoning_strength must be one of low, medium, high, or xhigh"
                        .into(),
                ));
            };
            if !matches!(strength, "low" | "medium" | "high" | "xhigh") {
                return Err(TextModelError::ToolConstraint(format!(
                    "unsupported Muse-Glimmer reasoning_strength {strength:?}; expected low, medium, high, or xhigh"
                )));
            }
        }
    }
    if request.tool_choice != ToolChoice::None
        && request.enable_thinking == Some(true)
        && !request.tools.is_empty()
        && !profile.supports_tool_reasoning
    {
        return Err(TextModelError::ToolConstraint(format!(
            "format profile {:?} does not preserve reasoning semantics while native tools are active",
            profile.identity.as_deref().unwrap_or("unregistered")
        )));
    }
    let semantic_failure = profile
        .native_tool_unavailable_reason
        .clone()
        .unwrap_or_else(|| "no semantic protocol was recognized".into());
    let tool_surface_requested =
        !request.tools.is_empty() || request.tool_choice == ToolChoice::Required;
    let tool_protocol_available = profile.tool_dialect.is_some()
        && profile.tool_dialect_parameters.is_some()
        && constraint_compiler.is_some_and(Result::is_ok);
    let native_tool_support = if tool_protocol_available {
        NativeToolSupport::Supported
    } else {
        NativeToolSupport::Unsupported {
            reason: profile
                .native_tool_unavailable_reason
                .clone()
                .unwrap_or_else(|| semantic_failure.clone()),
        }
    };

    let selected_runtime = if tool_surface_requested && tool_protocol_available {
        profile
            .tool_dialect
            .zip(profile.tool_dialect_parameters)
            .map(|(dialect, parameters)| {
                (
                    dialect,
                    parameters,
                    &profile.tool_required_structural_tokens,
                    true,
                )
            })
    } else {
        profile
            .dialect
            .zip(profile.dialect_parameters)
            .map(|(dialect, parameters)| {
                (
                    dialect,
                    parameters,
                    &profile.required_structural_tokens,
                    false,
                )
            })
    };
    let generation_runtime_plan = match selected_runtime {
        Some((dialect, parameters, structural_tokens, has_tool_surface)) => {
            let resolved_ids = resolve_structural_tokens(tokenizer, structural_tokens)
                .map_err(TextModelError::ToolConstraint)?;
            let compiler = constraint_compiler
                .ok_or_else(|| {
                    TextModelError::ToolConstraint(
                        "the loaded model does not have tokenizer constraint data".into(),
                    )
                })?
                .as_ref()
                .map_err(|error| TextModelError::ToolConstraint(error.clone()))?;
            Some(
                compiler
                    .compile_generation_plan(
                        dialect,
                        parameters,
                        if has_tool_surface {
                            &request.tools
                        } else {
                            &[]
                        },
                        if has_tool_surface {
                            request.tool_choice
                        } else {
                            ToolChoice::None
                        },
                        request.parallel_tool_calls,
                        structural_tokens.clone(),
                        resolved_ids,
                        profile.stop_sequences.clone(),
                        has_tool_surface,
                    )
                    .map_err(TextModelError::ToolConstraint)?,
            )
        }
        None => None,
    };
    let preserved_structural_token_ids = generation_runtime_plan
        .as_ref()
        .map(|plan| {
            plan.semantic_plan()
                .structural_tokens()
                .map(|(id, _)| id)
                .collect()
        })
        .unwrap_or_default();
    let semantic_support = if generation_runtime_plan.is_some() {
        SemanticSupport::Supported
    } else {
        SemanticSupport::Unsupported {
            reason: semantic_failure.clone(),
        }
    };
    if request.enable_thinking == Some(true)
        && (generation_runtime_plan.is_none() || !profile.supports_reasoning_parsing)
        && !request.allow_unparsed_reasoning
    {
        return Err(TextModelError::ToolConstraint(format!(
            "thinking was explicitly enabled, but no semantic reasoning protocol was recognized: {semantic_failure}; set allow_unparsed_reasoning to opt into raw output"
        )));
    }

    let capabilities = ChatCapabilities {
        reasoning_parser: capability(
            generation_runtime_plan.is_some() && profile.supports_reasoning_parsing,
            "the selected protocol does not provide a recognized reasoning channel",
        ),
        visible_text_parser: capability(
            generation_runtime_plan.is_some(),
            "no semantic visible-text parser was recognized",
        ),
        tool_output_parser: capability(
            profile.tool_dialect.is_some(),
            "generated tool-call envelopes were not recognized",
        ),
        tool_input_rendering: capability(
            profile.supports_tool_input_rendering,
            "tool-call and tool-response rendering probes did not establish support",
        ),
        mapping_tool_arguments: capability(
            profile.supports_mapping_tool_arguments,
            "tool-call history with mapping arguments was not established",
        ),
        string_tool_arguments: capability(
            profile.supports_string_tool_arguments,
            "tool-call history with serialized string arguments was not established",
        ),
        constrained_tool_generation: capability(
            profile.tool_dialect.is_some() && constraint_compiler.is_some_and(Result::is_ok),
            "no compatible tokenizer constraint compiler is available",
        ),
    };

    let ChatTemplateRequest {
        messages,
        tools,
        tool_choice,
        parallel_tool_calls: _,
        enable_thinking,
        reasoning_effort,
        allow_unparsed_reasoning: _,
        add_generation_prompt,
        mut extra_template_kwargs,
    } = request;
    let template_tools = if tool_choice == ToolChoice::None {
        Vec::new()
    } else {
        tools
    };
    let add_generation_prompt = profile
        .generation_prompt_behavior
        .resolve(add_generation_prompt);

    if let Some(enable_thinking) = enable_thinking {
        let explicit_muse_strength = profile.identity.as_deref() == Some("muse-glimmer.atem.v1")
            && extra_template_kwargs.contains_key("reasoning_strength");
        if !explicit_muse_strength {
            let (kwarg, value) = profile
                .reasoning_template_control
                .template_entry(enable_thinking);
            extra_template_kwargs.insert(kwarg.into(), value);
        }
    }
    if let Some(reasoning_effort) = reasoning_effort {
        let control = profile
            .reasoning_effort_control
            .expect("reasoning effort was validated against the recognized profile");
        extra_template_kwargs.insert(
            control.kwarg.into(),
            serde_json::Value::String(reasoning_effort),
        );
    }

    let without_generation_prompt = tokenizer
        .apply_chat_template_json(
            selected_template.clone(),
            [messages.clone()],
            Some(&template_tools),
            model_id,
            false,
            Some(&extra_template_kwargs),
        )?
        .into_iter()
        .next()
        .expect("one input conversation must produce one rendered prompt");
    let with_generation_prompt = tokenizer
        .apply_chat_template_json(
            selected_template,
            [messages],
            Some(&template_tools),
            model_id,
            true,
            Some(&extra_template_kwargs),
        )?
        .into_iter()
        .next()
        .expect("one input conversation must produce one rendered prompt");
    let generation_prompt = with_generation_prompt
        .strip_prefix(&without_generation_prompt)
        .unwrap_or_default()
        .to_owned();
    let rendered_prompt = if add_generation_prompt {
        with_generation_prompt
    } else {
        without_generation_prompt
    };

    Ok(PreparedChat {
        rendered_prompt,
        generation_prompt,
        template_identity,
        format_profile_identity: profile.identity,
        native_tool_support,
        semantic_support,
        capabilities,
        generation_runtime_plan,
        eos_token_ids: eos_token_ids.to_vec(),
        preserved_structural_token_ids,
        profile_stop_sequences: profile.stop_sequences,
    })
}
