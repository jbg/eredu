//! Prepared-chat request types and semantic generation machinery.

use std::num::NonZeroUsize;

use eredu_core::{
    SharedTokenFilter, SpeculativeDraft, SpeculativeOutputError, SpeculativeSchedulerOptions,
    SpeculativeSemanticState, SpeculativeTokenFilterController, TokenFilter, TokenFilterController,
    generation::{
        FinishReason, GenerationCancellationToken, GenerationConfigOverrides, SemanticEvent,
    },
};
use eredu_text::tokenizer::{ChatTemplateIdentity, ModelChatTemplate, Tokenizer as ChatTokenizer};

use super::{ConstraintError, TextDecoderError, TextModelError};
use crate::api::TextDecoder;
use crate::runtime::chat::constraints::{
    ConstraintCompiler, ConstraintController, ConstraintPreparationError,
};
use crate::runtime::chat::{
    CapabilitySupport, ChatCapabilities, ChatTemplateRequest, NativeToolSupport, PreparedChat,
    SemanticSupport, ToolChoice, prepare_format_profile, resolve_structural_tokens,
};
use crate::runtime::generation::streaming::{
    CommittedTokenPipeline, CommittedTokenSource, RawTokenDecoder, TokenDecoderBackend,
};
use std::collections::HashMap;

mod ifm;
pub(crate) mod probes;
pub(crate) mod profile;

/// Model sampling and stopping settings for one prepared chat generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreparedChatGenerationSettings {
    /// Typed overrides layered over checkpoint-declared generation settings.
    pub overrides: GenerationConfigOverrides,
    /// Sampling strategy; defaults to [`eredu_core::TextSamplingStrategy::Standard`].
    ///
    /// Mirostat V2 requires a positive effective temperature after checkpoint
    /// defaults and overrides are resolved. It retains penalties and replaces
    /// top-k, top-p, and min-p. Vocabulary and tool constraints filter logits
    /// before sampling; adaptive state tracks the constrained distribution.
    /// Prepared speculative generation retains adaptive state per lane and
    /// disables exact optimistic lookahead promotion for Mirostat V2.
    pub strategy: eredu_core::TextSamplingStrategy,
    /// Deterministic root seed used by the selected backend for stochastic sampling.
    pub seed: u64,
    /// Shared prefill and enforced managed-memory policy for this request.
    pub inference: eredu_core::TextInferencePolicy,
}

/// Explicit prompt source for semantic or text generation from a [`PreparedChat`].
///
/// The generation method selects semantic parsing or literal text output.
/// The selected variant determines only how the model prompt is prefetched.
pub enum PreparedChatInput<'a, B: eredu_core::TextGenerationBackend> {
    /// Tokenize and prefill the rendered prompt stored in the prepared chat.
    RenderedPrompt(&'a PreparedChat),
    /// Prefill exact token IDs with the chat's semantic and termination policy.
    /// Native prompt construction occurs inside coordinated run preparation.
    TokenIds {
        /// Prepared chat supplying generation policy.
        prepared_chat: &'a PreparedChat,
        /// Exact prefix; generation never forces the tested prediction.
        token_ids: Vec<u32>,
    },
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

    /// Binds an exact token-ID prefix without creating native values early.
    pub fn token_ids(prepared_chat: &'a PreparedChat, token_ids: Vec<u32>) -> Self {
        Self::TokenIds {
            prepared_chat,
            token_ids,
        }
    }

    /// Returns the prepared chat that owns generation semantics.
    pub const fn prepared_chat(&self) -> &'a PreparedChat {
        match self {
            Self::RenderedPrompt(prepared_chat)
            | Self::TokenIds { prepared_chat, .. }
            | Self::PreparedBackendInput { prepared_chat, .. } => prepared_chat,
        }
    }

    /// Returns the explicitly prepared backend prompt, when present.
    pub const fn backend_prompt(&self) -> Option<&B::Prompt> {
        match self {
            Self::RenderedPrompt(_) | Self::TokenIds { .. } => None,
            Self::PreparedBackendInput { prompt, .. } => Some(prompt),
        }
    }
}

/// Request for ordinary semantic or text generation from a [`PreparedChat`].
///
/// Used by `LoadedModel::generate_prepared_chat` and `generate_prepared_text`.
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
/// Ordinary and observed generation share the neutral output with no extra statistics.
pub type PreparedChatGenerationOutput = eredu_core::GenerationOutput;

/// Backend-independent failure from ordinary prepared-chat generation.
///
/// Backend failures retain their original details through a common error source.
/// Portable tokenizer, constraint, semantic-streaming, and generation-lifecycle
/// failures have distinct variants.
#[derive(Debug, thiserror::Error)]
pub enum PreparedChatError {
    /// Capture admission, accounting, or transport bounds rejected the request.
    #[error(transparent)]
    Capture(#[from] eredu_core::capture::CaptureError),
    /// The selected backend failed submission, completion, or token extraction.
    #[error("selected backend failed prepared-chat generation: {0}")]
    Backend(#[source] eredu_core::BackendFailure),
    /// Portable constraint construction or advancement failed.
    #[error(transparent)]
    Constraint(#[from] ConstraintError),
    /// Portable generation lifecycle state was invalid.
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
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

impl From<eredu_core::run_preparation::TextCaptureSetupError> for PreparedChatError {
    fn from(error: eredu_core::run_preparation::TextCaptureSetupError) -> Self {
        match error {
            eredu_core::run_preparation::TextCaptureSetupError::Capture(error) => {
                Self::Capture(error)
            }
            eredu_core::run_preparation::TextCaptureSetupError::Preparation(error) => {
                Self::Backend(error)
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum PreparedChatSetupError {
    #[error("shared controller storage preparation failed: {0}")]
    Backend(#[source] eredu_core::BackendFailure),
    #[error(transparent)]
    Constraint(#[from] ConstraintError),
    #[error("{0}")]
    Semantic(String),
}

impl From<ConstraintPreparationError> for PreparedChatSetupError {
    fn from(error: ConstraintPreparationError) -> Self {
        match error {
            ConstraintPreparationError::Constraint(error) => Self::Constraint(error),
            ConstraintPreparationError::Backend(error) => Self::Backend(error),
        }
    }
}

/// Speculative-decoding controls for one prepared-chat request.
#[derive(Debug, Clone, Copy)]
pub struct PreparedChatSpeculativeGenerationOptions {
    /// Maximum assistant proposals verified in one target block.
    pub max_draft_tokens: NonZeroUsize,
    /// Canonical scheduler controls.
    pub scheduler: SpeculativeSchedulerOptions,
}

impl Default for PreparedChatSpeculativeGenerationOptions {
    fn default() -> Self {
        Self {
            max_draft_tokens: NonZeroUsize::new(4).expect("4 is non-zero"),
            scheduler: SpeculativeSchedulerOptions::default(),
        }
    }
}

/// Failure while the facade prepares or a backend executes speculative chat.
#[derive(Debug, thiserror::Error)]
pub enum PreparedChatSpeculativeError {
    /// The selected target/draft path does not accept the requested chunk policy,
    /// or speculative preparation lacks the requested memory/control admission.
    #[error(
        "prepared speculative inference does not yet implement the requested prefill or managed-memory policy: {0:?}"
    )]
    InferencePolicyUnavailable(eredu_core::TextInferencePolicy),
    /// A sampling strategy is unavailable for prepared speculative generation.
    /// Retained for source compatibility; all current strategies are supported.
    #[error(
        "prepared-chat speculative generation does not support sampling strategy {0:?}; use ordinary prepared-chat generation"
    )]
    UnsupportedSamplingStrategy(eredu_core::TextSamplingStrategy),
    /// The selected backend failed prompt preparation or speculative execution.
    #[error("selected backend failed prepared-chat speculative generation: {0}")]
    Backend(#[source] eredu_core::BackendFailure),
    /// Portable generation configuration was invalid.
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
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

/// One speculative semantic or text response from a [`PreparedChat`].
/// Used by `LoadedModel::generate_prepared_chat_speculative` and
/// `generate_prepared_text_speculative`.
pub struct PreparedChatSpeculativeGenerationRequest<'a, B: eredu_core::TextGenerationBackend, D, F>
{
    /// Explicit prompt source and embedded format/runtime plan.
    pub input: PreparedChatInput<'a, B>,
    /// Embedded or separately loaded draft-model selection.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Portable sampling configuration, token limit, and random seed.
    pub settings: PreparedChatGenerationSettings,
    /// Proposal-block and scheduler controls.
    pub options: PreparedChatSpeculativeGenerationOptions,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// One-shot cooperative-cancellation token for this request.
    pub cancellation: GenerationCancellationToken,
    /// Called synchronously for each event after its cache transaction commits.
    pub on_event: F,
}

/// One independently executable lane in a prepared-chat speculative batch.
///
/// Every lane owns portable sampling, a random root, stop configuration, and
/// callback. The facade constructs its constraint and decoder/parser pipeline
/// before dispatch; the backend allocates only its execution cache and sampling
/// state. Lanes are shared by the external-assistant and embedded-head drafting
/// strategies.
pub struct PreparedChatSpeculativeBatchLane<'a, B: eredu_core::TextGenerationBackend> {
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

/// Cohesive fair-scheduler request for independent prepared-chat speculative lanes.
/// Used by `LoadedModel::generate_prepared_chat_speculative_batch` and
/// `generate_prepared_text_speculative_batch`.
pub struct PreparedChatSpeculativeBatchRequest<'a, B: eredu_core::TextGenerationBackend, D> {
    /// Embedded or separately loaded draft-model selection.
    pub drafting: SpeculativeDraft<'a, D>,
    /// Independently executable prepared-chat lanes.
    pub lanes: Vec<PreparedChatSpeculativeBatchLane<'a, B>>,
    /// Bounded scheduler and optimistic-lookahead controls.
    pub scheduler: SpeculativeSchedulerOptions,
}

/// Facade-prepared speculative grammar state consumed by a backend sampler.
///
/// The facade constructs semantic constraints or text-only tokenizer validity
/// from the prepared request. A backend applies its portable filters to native
/// logits and commits only tokens accepted by target verification.
#[derive(Clone)]
pub struct PreparedChatSpeculativeConstraint {
    controller: ConstraintController,
}

impl PreparedChatSpeculativeConstraint {
    pub(super) fn new(controller: ConstraintController) -> Self {
        Self { controller }
    }
}

impl TokenFilterController for PreparedChatSpeculativeConstraint {
    type Error = ConstraintError;

    fn inference_storage(&self) -> eredu_core::TextControllerStorage<'_> {
        self.controller.inference_storage()
    }

    fn inference_workspace(
        &self,
        max_output_tokens: u64,
    ) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        self.controller.inference_workspace(max_output_tokens)
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.controller.current_filter()
    }

    fn current_decision(&mut self) -> Result<eredu_core::TokenSamplingDecision<'_>, Self::Error> {
        self.controller.current_decision()
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error> {
        self.controller.commit_token(token_id)
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.controller.is_complete()
    }
}

impl eredu_core::SpeculativeTokenFilterController for PreparedChatSpeculativeConstraint {
    type PreparedGrammar = crate::runtime::chat::constraints::OriginalPreparedGrammarController;
    fn prepared_grammar(&self) -> Option<&Self::PreparedGrammar> { self.controller.prepared_grammar() }
    fn prepared_grammar_replacement_bytes(&self) -> Option<usize> {
        self.controller.prepared_grammar_replacement_bytes()?.checked_add(
            eredu_core::speculative::PreparedGrammarInstallError::<Self::PreparedGrammar>::wrapper_control_bytes::<Self>()?)
    }
    fn replace_prepared_grammar(
        &self, grammar: Self::PreparedGrammar, funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Self, eredu_core::speculative::PreparedGrammarInstallError<Self::PreparedGrammar>> {
        let reserve = eredu_core::speculative::PreparedGrammarInstallError::<Self::PreparedGrammar>::wrapper_control_bytes::<Self>()
            .ok_or(eredu_core::HostMetadataFundingError::Overflow).and_then(|n| funding.reserve_metadata(n));
        if let Err(cause) = reserve {
            return Err(eredu_core::speculative::PreparedGrammarInstallError::new(cause.into(), grammar, funding));
        }
        self.controller.replace_prepared_grammar(grammar, funding).map(|controller| Self { controller })
    }

    fn prepared_plain_source(&self) -> Option<eredu_core::speculative::PlainControllerSource<'_>> {
        self.controller.prepared_plain_source()
    }
    fn prepared_plain_copy_bytes(&self, capacity: usize) -> Option<usize> {
        self.controller
            .prepared_plain_copy_bytes(capacity)?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(std::mem::size_of::<
                Result<Self, eredu_core::speculative::PlainControllerError>,
            >())
    }
    fn copy_prepared_plain(
        &self,
        capacity: usize,
        host: eredu_core::HostPreparationAuthority,
    ) -> Result<Self, eredu_core::speculative::PlainControllerError> {
        self.controller
            .copy_prepared_plain(capacity, host)
            .map(|controller| Self { controller })
    }
    fn prepared_plain_history_mut(
        &mut self,
    ) -> Option<&mut eredu_core::speculative::PlainControllerHistory> {
        self.controller.prepared_plain_history_mut()
    }
    fn prepared_forbidden_source(
        &self,
    ) -> Option<eredu_core::speculative::ForbiddenControllerSource<'_>> {
        self.controller.prepared_forbidden_source()
    }
    fn prepared_forbidden_copy_bytes(&self, capacity: usize) -> Option<usize> {
        self.controller
            .prepared_forbidden_copy_bytes(capacity)?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(std::mem::size_of::<
                Result<Self, eredu_core::speculative::ForbiddenControllerError>,
            >())
    }
    fn copy_prepared_forbidden(
        &self,
        capacity: usize,
        host: eredu_core::HostPreparationAuthority,
    ) -> Result<Self, eredu_core::speculative::ForbiddenControllerError> {
        self.controller
            .copy_prepared_forbidden(capacity, host)
            .map(|controller| Self { controller })
    }
    fn prepared_forbidden_mutation(
        &mut self,
    ) -> Result<
        eredu_core::speculative::ForbiddenControllerMutation<'_>,
        eredu_core::speculative::ForbiddenControllerError,
    > {
        self.controller.prepared_forbidden_mutation()
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        eredu_runtime::execution_control::SnapshotTokenController::snapshot_storage_bytes(
            &self.controller,
        )
    }

    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error> {
        self.controller.filter_at(history)
    }

    fn decision_at(
        &self,
        history: &[u32],
    ) -> Result<eredu_core::TokenSamplingDecision<'_>, Self::Error> {
        eredu_core::SpeculativeTokenFilterController::decision_at(&self.controller, history)
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

    fn snapshot_storage_bytes(&self) -> Option<u64> {
        self.decoder.snapshot_storage_bytes()
    }

    fn continuation_storage_bounds(&self, max_tokens: u64) -> Option<(u64, u64)> {
        self.decoder.continuation_storage_bounds(max_tokens)
    }

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
    // Map payload retires before the controller/parser lifetime owners, including
    // a prepared request discarded by a peer before this aggregate is split.
    pub(super) structural_tokens: HashMap<u32, String>,
    pub(super) controller: ConstraintController,
    pub(super) parser: crate::runtime::generation::streaming::ToolRuntimeParser,
}

pub(super) struct PreparedChatSemanticState {
    pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    token_ids: Vec<u32>,
    events: Vec<SemanticEvent>,
    authority: eredu_core::HostPreparationAuthority,
}

impl PreparedChatSemanticState {
    pub(super) fn new(
        decoder: PreparedChatTokenDecoder,
        parser: crate::runtime::generation::streaming::ToolRuntimeParser,
        structural_tokens: HashMap<u32, String>,
    ) -> Self {
        let authority = parser.host_preparation().clone();
        Self {
            pipeline: CommittedTokenPipeline::new(
                RawTokenDecoder::with_structural_tokens(decoder, structural_tokens),
                parser,
            ),
            token_ids: Vec::new(),
            events: Vec::new(),
            authority,
        }
    }
}

impl SpeculativeSemanticState for PreparedChatSemanticState {
    fn control_snapshot_bytes(&self) -> Option<u64> {
        use crate::runtime::generation::storage::SnapshotStorage;
        self.pipeline
            .snapshot_storage_bytes()?
            .checked_add(self.token_ids.heap_bytes()?)?
            .checked_add(self.events.heap_bytes()?)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn fork_box(&self) -> Result<Box<dyn SpeculativeSemanticState>, SpeculativeOutputError> {
        let pipeline = self
            .pipeline
            .fork()
            .map_err(|error| SpeculativeOutputError::semantic("fork semantic state", error))?;
        Ok(Box::new(Self {
            pipeline,
            token_ids: self.token_ids.clone(),
            events: Vec::new(),
            authority: self.authority.clone(),
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

    fn take_events(&mut self) -> eredu_core::SpeculativeBuffer<SemanticEvent> {
        std::mem::take(&mut self.events).into()
    }
}

#[derive(Clone, Copy)]
pub(super) enum PreparedGenerationMode {
    Semantic,
    Text,
}

impl PreparedGenerationMode {
    pub(super) fn prepare_control<B: eredu_core::TextGenerationBackend>(
        self,
        runtime: &eredu_core::ModelRuntime<B>,
        prepared_chat: &PreparedChat,
        caller_stop_sequences: &[String],
        validity: SharedTokenFilter,
    ) -> Result<PreparedChatControlRuntime, PreparedChatSetupError> {
        match self {
            Self::Semantic => prepared_chat_control_runtime(
                runtime,
                prepared_chat,
                caller_stop_sequences,
                validity,
            ),
            Self::Text => {
                prepared_text_control_runtime(prepared_chat, caller_stop_sequences, validity)
            }
        }
    }
}

fn prepared_chat_control_runtime<B: eredu_core::TextGenerationBackend>(
    runtime: &eredu_core::ModelRuntime<B>,
    prepared_chat: &PreparedChat,
    caller_stop_sequences: &[String],
    validity: SharedTokenFilter,
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
    let authority =
        B::acquire_host_preparation(runtime).map_err(PreparedChatSetupError::Backend)?;
    let controller =
        ConstraintController::from_generation_plan(generation_plan, runtime, &authority)?
            .with_validity(validity);
    let parser = semantic_plan
        .create_parser_with_stops_under_authority(
            caller_stop_sequences.iter().map(String::as_str),
            &authority,
        )
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

pub(super) fn prepared_text_control_runtime(
    prepared_chat: &PreparedChat,
    caller_stop_sequences: &[String],
    validity: SharedTokenFilter,
) -> Result<PreparedChatControlRuntime, PreparedChatSetupError> {
    if let CapabilitySupport::Unsupported { reason } = prepared_chat.text_generation_support() {
        return Err(PreparedChatSetupError::Semantic(reason.clone()));
    }
    Ok(PreparedChatControlRuntime {
        controller: ConstraintController::text(validity),
        parser: crate::runtime::generation::streaming::ToolRuntimeParser::text(
            caller_stop_sequences.iter().map(String::as_str),
        ),
        structural_tokens: HashMap::new(),
    })
}

pub(super) struct BackendGenerationTokenSource<'a, B, C = ConstraintController>
where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    pub(super) generator: eredu_core::ControlledTextGeneration<'a, B, C>,
    pub(super) on_token: Option<
        &'a mut dyn FnMut(Option<u32>, Option<eredu_core::capture::CapturedStepDelivery>, f64),
    >,
    pub(super) delivery_failure: Option<&'a dyn Fn() -> Option<eredu_core::capture::CaptureError>>,
    pub(super) generation_started: std::time::Instant,
    pub(super) time_to_first_token: Option<std::time::Duration>,
}

impl<B, C> CommittedTokenSource for BackendGenerationTokenSource<'_, B, C>
where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    type Error = eredu_core::ControlledTextGenerationError<B::Error, C::Error>;

    fn finish_step<T, E>(
        &mut self,
        local: Result<T, E>,
        cancelled: bool,
        map_source: impl FnOnce(Self::Error) -> E,
    ) -> Result<Option<T>, E> {
        self.generator.finish_text_preparation_cancellable(
            eredu_core::run_preparation::TextPreparationStage::Delivery,
            local.map(|value| (!cancelled).then_some(value)),
            |error| {
                map_source(eredu_core::ControlledTextGenerationError::Preparation(
                    error,
                ))
            },
        )
    }

    fn delivery_failure(&self) -> Option<eredu_core::capture::CaptureError> {
        self.delivery_failure.and_then(|failure| failure())
    }

    fn next_token(
        &mut self,
        cancellation: &eredu_core::GenerationCancellationToken,
    ) -> Result<Option<u32>, Self::Error> {
        let started = self.on_token.as_ref().map(|_| std::time::Instant::now());
        let result = self
            .generator
            .next_cancellable(cancellation)
            .transpose()
            .map(|token| token.map(|token| token.token_id()));
        let token = match result {
            Ok(token) => token,
            Err(error) => {
                // Empty capture plans still have terminal bookkeeping and
                // completion ownership. Always attempt the shared drain, while
                // preserving the original execution error if draining fails.
                if let (Some(callback), Ok(Some(capture))) =
                    (&mut self.on_token, self.generator.take_captured_delivery())
                {
                    callback(
                        None,
                        Some(capture),
                        started.unwrap().elapsed().as_secs_f64(),
                    );
                }
                return Err(error);
            }
        };
        if token.is_some() && self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(self.generation_started.elapsed());
        }
        // Readiness agreement requires a completed native boundary even when
        // this run has no capture plan. Empty capture draining allocates none.
        let capture = self
            .generator
            .take_captured_delivery()
            .map_err(eredu_core::ControlledTextGenerationError::Backend)?;
        if let Some(callback) = &mut self.on_token {
            // A completed cancellation may have a terminal frame without a
            // committed token. Deliver it through the same existing callback;
            // the callback cannot replace this result or clear cancellation.
            if token.is_some() || capture.is_some() {
                callback(token, capture, started.unwrap().elapsed().as_secs_f64());
            }
        }
        Ok(token)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.generator
            .controller_is_complete()
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

fn validate_tagged_history(
    messages: &[serde_json::Value],
    parameter_suffix: &str,
    profile: &str,
) -> Result<(), TextModelError> {
    profile::tagged::validate(messages, parameter_suffix)
        .map_err(|cause| cause.ordinary(messages, profile))
}

fn gemma_profile(
    recognition: probes::gemma::Recognition,
) -> crate::runtime::chat::PreparedFormatProfile {
    use crate::runtime::chat::{
        GEMMA4_STRUCTURAL_TOOL_SPEC,
        dialect::{DialectParameters, GenerationPromptBehavior},
        gemma::{self, TOOL_RESPONSE_OPEN, TURN_CLOSE},
    };

    let mut semantic_tokens = probes::gemma::CHANNELS.map(str::to_owned).to_vec();
    let mut semantic_stops = Vec::new();
    for (present, spelling) in [
        (recognition.response_token, TOOL_RESPONSE_OPEN),
        (recognition.turn_token, TURN_CLOSE),
    ] {
        if present {
            semantic_tokens.push(spelling.to_owned());
            semantic_stops.push(spelling.to_owned());
        }
    }
    let full_tool_tokens = probes::gemma::TOOLS.map(str::to_owned).to_vec();
    let mapping_tool_arguments = recognition.mapping_tool_arguments;
    let string_tool_arguments = recognition.string_tool_arguments;
    let tool_output_protocol = mapping_tool_arguments || string_tool_arguments;
    let tool_input_rendering = tool_output_protocol;

    crate::runtime::chat::PreparedFormatProfile {
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
    }
}

fn inkling_profile(
    recognition: probes::inkling::Recognition,
) -> crate::runtime::chat::PreparedFormatProfile {
    use crate::runtime::chat::{
        ReasoningTemplateControl,
        dialect::GenerationPromptBehavior,
        inkling::{self, END_SAMPLING},
    };

    let structural_tokens = probes::inkling::STRUCTURAL.map(str::to_owned).to_vec();
    let tool_structural_tokens = probes::inkling::TOOL_STRUCTURAL.map(str::to_owned).to_vec();
    let mapping_tool_arguments = recognition.mapping_tool_arguments;
    let string_tool_arguments = recognition.string_tool_arguments;
    let tool_output_protocol = mapping_tool_arguments || string_tool_arguments;

    crate::runtime::chat::PreparedFormatProfile {
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
    }
}

fn muse_profile(
    recognition: probes::muse::Recognition,
) -> crate::runtime::chat::PreparedFormatProfile {
    use crate::runtime::chat::{
        ReasoningEffortControl, ReasoningTemplateControl,
        atem::{self, EOT},
        dialect::GenerationPromptBehavior,
    };

    let structural_tokens = probes::muse::STRUCTURAL.map(str::to_owned).to_vec();
    let mapping_tool_arguments = recognition.mapping_tool_arguments;

    crate::runtime::chat::PreparedFormatProfile {
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
    }
}

fn render_protocol_probe(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
    arguments: probes::records::Arguments<'_>,
) -> Option<String> {
    probes::Ordinary {
        tokenizer,
        selected_template,
        model_id,
    }
    .render(probes::Probe::tool_input(arguments))
}

fn render_reasoning_protocol_probe(
    tokenizer: &mut ChatTokenizer,
    selected_template: &ModelChatTemplate,
    model_id: &str,
) -> Option<String> {
    probes::Ordinary {
        tokenizer,
        selected_template,
        model_id,
    }
    .render(probes::Probe::ReasoningHistory.construct())
}

fn recognized_dialect_profile(
    identity: &'static str,
    dialect: &'static dyn crate::runtime::chat::dialect::FormatDialect,
    parameters: crate::runtime::chat::dialect::DialectParameters,
    declaration: crate::runtime::chat::dialect::ProfileDeclaration,
    mapping_tool_arguments: bool,
    string_tool_arguments: bool,
) -> crate::runtime::chat::PreparedFormatProfile {
    let generation_prompt_behavior = declaration.generation;
    let reasoning_template_kwarg = declaration.reasoning_kwarg;
    let supports_tool_reasoning = declaration.tool_reasoning;
    let required_structural_tokens = declaration
        .structural
        .iter()
        .map(|token| (*token).to_owned())
        .collect::<Vec<_>>();
    let stop_sequences = declaration
        .stops
        .iter()
        .map(|stop| (*stop).to_owned())
        .collect::<Vec<_>>();
    let supports_tool_input_rendering = mapping_tool_arguments || string_tool_arguments;

    crate::runtime::chat::PreparedFormatProfile {
        identity: Some(identity.into()),
        dialect: Some(dialect),
        dialect_parameters: Some(parameters),
        tool_dialect: Some(dialect),
        tool_dialect_parameters: Some(parameters),
        generation_prompt_behavior,
        reasoning_template_control:
            crate::runtime::chat::ReasoningTemplateControl::Boolean(reasoning_template_kwarg),
        reasoning_effort_control: None,
        supports_reasoning_parsing: declaration.reasoning_parsing,
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
    }
}

fn selected_profile(
    selected: probes::selection::Selected,
    request: &ChatTemplateRequest,
) -> crate::runtime::chat::PreparedFormatProfile {
    use probes::selection::Selected;
    let policy = profile::selected::Policy::from_selected(selected);
    let mut profile = match selected {
        Selected::Ifm {
            behavior,
            declaration,
        } => {
            let spec = crate::runtime::chat::ifm::spec(
                behavior.format.spelling(),
                behavior.effort.spelling(),
                request.add_generation_prompt,
                behavior.disabled,
            )
            .expect("shared IFM controls were validated");
            let mut profile = recognized_dialect_profile(
                "ifm.tools.v1",
                &crate::runtime::chat::dialect::DECLARATIVE_DIALECT,
                crate::runtime::chat::dialect::DialectParameters::Declarative(spec),
                declaration,
                true,
                false,
            );
            profile.reasoning_effort_control = Some(crate::runtime::chat::ReasoningEffortControl {
                kwarg: "reasoning_effort",
                supported: &["high", "medium", "low"],
            });
            profile
        }
        Selected::Muse(behavior) => muse_profile(behavior),
        Selected::Gemma(behavior) => gemma_profile(behavior),
        Selected::Inkling(behavior) => inkling_profile(behavior),
        Selected::Remaining {
            behavior,
            declaration,
        } => {
            let (identity, dialect, parameters) = behavior.kind.declaration();
            let mut profile = recognized_dialect_profile(
                identity,
                dialect,
                parameters,
                declaration,
                behavior.mapping,
                behavior.string,
            );
            if behavior.kind == probes::remaining::Kind::Qwen38 {
                profile.reasoning_effort_control =
                    Some(crate::runtime::chat::ReasoningEffortControl {
                        kwarg: "reasoning_effort",
                        supported: &["low", "medium", "xhigh"],
                    });
            }
            profile
        }
        Selected::Unregistered => prepare_format_profile(""),
    };
    policy.apply_to(&mut profile);
    profile
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
        let selected = probes::selection::select(
            &mut ifm::Ordinary(probes::Ordinary {
                tokenizer,
                selected_template: &selected_template,
                model_id,
            }),
            &request,
        )?;
        profile = selected_profile(selected, &request);
    }
    let profile_controls = profile::ProfileRequestControls::from_profile(&profile);
    profile_controls
        .validate_effort(&request)
        .map_err(|cause| cause.ordinary(&profile, &request))?;
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
        validate_tagged_history(&request.messages, "</parameter>", "Qwen")?;
    }
    profile_controls
        .validate_strength(&request)
        .map_err(|cause| cause.ordinary(&profile, &request))?;
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
    let text_generation_support = match text_chat_eligibility(&request) {
        Ok(()) => CapabilitySupport::Supported,
        Err(reason) => CapabilitySupport::Unsupported {
            reason: reason.to_string(),
        },
    };
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

    let mapped_controls = profile_controls
        .bindings(&request)
        .map_err(|cause| cause.ordinary(&profile, &request))?
        .into_owned();
    let ChatTemplateRequest {
        messages,
        tools,
        tool_choice,
        parallel_tool_calls: _,
        enable_thinking: _,
        reasoning_effort: _,
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

    for (kwarg, value) in mapped_controls.into_iter().flatten() {
        extra_template_kwargs.insert(kwarg, value);
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
        text_generation_support,
        capabilities,
        generation_runtime_plan,
        eos_token_ids: eos_token_ids.to_vec(),
        preserved_structural_token_ids,
        profile_stop_sequences: profile.stop_sequences,
    })
}

/// The ordinary text-request eligibility predicate, factored without allocating
/// so original chat preparation can preserve the same semantic exclusions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum TextChatEligibilityError {
    #[error(
        "text generation does not support tool declarations or required tool calls; prepare a request without tools"
    )]
    Tools,
    #[error(
        "text generation does not parse explicit thinking; set allow_unparsed_reasoning to opt into raw output"
    )]
    Thinking,
}
pub(crate) fn text_chat_eligibility(
    request: &ChatTemplateRequest,
) -> Result<(), TextChatEligibilityError> {
    if !request.tools.is_empty() || request.tool_choice == ToolChoice::Required {
        Err(TextChatEligibilityError::Tools)
    } else if request.enable_thinking == Some(true) && !request.allow_unparsed_reasoning {
        Err(TextChatEligibilityError::Thinking)
    } else {
        Ok(())
    }
}
