//! Prepared-chat request types and semantic generation machinery.

use std::num::NonZeroUsize;

use eredu_core::{
    SpeculativeSchedulerOptions, TokenFilter, TokenFilterController,
    generation::GenerationConfigOverrides,
};
use eredu_text::tokenizer::{ChatTemplateIdentity, ModelChatTemplate, Tokenizer as ChatTokenizer};

use super::{ConstraintError, TextModelError};
use crate::runtime::chat::constraints::{ConstraintCompiler, ConstraintController};
use crate::runtime::chat::{
    CapabilitySupport, ChatTemplateRequest, PreparedChat, ProfileStrings, ToolChoice,
    prepare_format_profile, resolve_structural_tokens,
};
use crate::runtime::generation::streaming::CommittedTokenSource;

mod ifm;
pub(crate) mod policy;
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
    fn prepared_grammar(&self) -> Option<&Self::PreparedGrammar> {
        self.controller.prepared_grammar()
    }
    fn prepared_grammar_replacement_bytes(&self) -> Option<usize> {
        self.controller
            .prepared_grammar_replacement_bytes()?
            .checked_add(eredu_core::speculative::PreparedGrammarInstallError::<
            Self::PreparedGrammar,
        >::wrapper_control_bytes::<Self>()?)
    }
    fn replace_prepared_grammar(
        &self,
        grammar: Self::PreparedGrammar,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Self, eredu_core::speculative::PreparedGrammarInstallError<Self::PreparedGrammar>>
    {
        let reserve = eredu_core::speculative::PreparedGrammarInstallError::<Self::PreparedGrammar>::wrapper_control_bytes::<Self>()
            .ok_or(eredu_core::HostMetadataFundingError::Overflow).and_then(|n| funding.reserve_metadata(n));
        if let Err(cause) = reserve {
            return Err(eredu_core::speculative::PreparedGrammarInstallError::new(
                cause.into(),
                grammar,
                funding,
            ));
        }
        self.controller
            .replace_prepared_grammar(grammar, funding)
            .map(|controller| Self { controller })
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

pub(super) struct BackendGenerationTokenSource<'a, B, C = ConstraintController>
where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    pub(super) generator: eredu_core::ControlledTextGeneration<'a, B, C>,
    pub(super) on_token: Option<
        &'a mut dyn FnMut(Option<u32>, Option<eredu_core::capture::SharedCapturedStep>, f64),
    >,
    pub(super) delivery_failure: Option<&'a dyn Fn() -> Option<eredu_core::capture::CaptureError>>,
    pub(super) generation_started: std::time::Instant,
    pub(super) time_to_first_token: Option<std::time::Duration>,
}

/// Fixed facts from the actual committed-token worker, before semantic delivery.
#[derive(Clone, Copy)]
pub(crate) struct TokenDeliveryFacts {
    pub step_seconds: f64,
    pub timing: eredu_core::GenerationTiming,
    pub forced: bool,
}

/// A step borrows delivery state without extending that borrow over the run.
/// Both persistent and scoped observers see the same completed capture owner.
pub(crate) struct ScopedBackendGenerationTokenSource<
    'step,
    'run,
    'callback,
    B,
    C = ConstraintController,
> where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    source: &'step mut BackendGenerationTokenSource<'run, B, C>,
    observer: Option<
        &'step mut (
                       dyn FnMut(
            Option<u32>,
            Option<eredu_core::capture::SharedCapturedStep>,
            f64,
            eredu_core::GenerationTiming,
            &C,
        ) + 'callback
                   ),
    >,
    failure: Option<&'step dyn Fn() -> Option<eredu_core::capture::CaptureError>>,
}

impl<'run, B, C> BackendGenerationTokenSource<'run, B, C>
where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    pub(crate) fn scoped<'step, 'callback>(
        &'step mut self,
        observer: Option<
            &'step mut (
                           dyn FnMut(
                Option<u32>,
                Option<eredu_core::capture::SharedCapturedStep>,
                f64,
                eredu_core::GenerationTiming,
                &C,
            ) + 'callback
                       ),
        >,
        failure: Option<&'step dyn Fn() -> Option<eredu_core::capture::CaptureError>>,
    ) -> ScopedBackendGenerationTokenSource<'step, 'run, 'callback, B, C> {
        ScopedBackendGenerationTokenSource {
            source: self,
            observer,
            failure,
        }
    }

    fn next_token_with_observer(
        &mut self,
        cancellation: &eredu_core::GenerationCancellationToken,
        mut observer: Option<
            &mut (
                     dyn FnMut(
                Option<u32>,
                Option<eredu_core::capture::SharedCapturedStep>,
                f64,
                eredu_core::GenerationTiming,
                &C,
            ) + '_
                 ),
        >,
    ) -> Result<
        Option<u32>,
        eredu_core::ControlledTextGenerationError<eredu_core::BackendFailure, C::Error>,
    > {
        let started = (self.on_token.is_some() || observer.is_some()).then(std::time::Instant::now);
        let result = self
            .generator
            .next_cancellable(cancellation)
            .transpose()
            .map(|token| token.map(|token| token.token_id()));
        let token = match result {
            Ok(token) => token,
            Err(error) => {
                // Completion is drained once even for an empty capture plan.
                // A drain failure must preserve the original execution failure.
                if let Ok(Some(capture)) = self.generator.take_captured_delivery() {
                    self.deliver_token(None, Some(capture), started, &mut observer);
                }
                return Err(match error {
                    eredu_core::ControlledTextGenerationError::Backend(cause) => {
                        eredu_core::ControlledTextGenerationError::Backend(B::into_backend_failure(
                            cause,
                        ))
                    }
                    eredu_core::ControlledTextGenerationError::Preparation(cause) => {
                        eredu_core::ControlledTextGenerationError::Preparation(cause)
                    }
                    eredu_core::ControlledTextGenerationError::Controller(cause) => {
                        eredu_core::ControlledTextGenerationError::Controller(cause)
                    }
                });
            }
        };
        if token.is_some() && self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(self.generation_started.elapsed());
        }
        let capture = self.generator.take_captured_delivery().map_err(|cause| {
            eredu_core::ControlledTextGenerationError::Backend(B::into_backend_failure(cause))
        })?;
        self.deliver_token(token, capture, started, &mut observer);
        Ok(token)
    }

    fn deliver_token(
        &mut self,
        token: Option<u32>,
        capture: Option<eredu_core::capture::SharedCapturedStep>,
        started: Option<std::time::Instant>,
        observer: &mut Option<
            &mut (
                     dyn FnMut(
                Option<u32>,
                Option<eredu_core::capture::SharedCapturedStep>,
                f64,
                eredu_core::GenerationTiming,
                &C,
            ) + '_
                 ),
        >,
    ) {
        if token.is_none() && capture.is_none() {
            return;
        }
        let Some(started) = started else {
            return;
        };
        let seconds = started.elapsed().as_secs_f64();
        if let Some(callback) = &mut self.on_token {
            callback(token, capture.clone(), seconds);
        }
        if let Some(callback) = observer {
            callback(
                token,
                capture,
                seconds,
                eredu_core::GenerationTiming::new(self.time_to_first_token),
                self.generator.controller(),
            );
        }
    }
}

impl<B, C> CommittedTokenSource for ScopedBackendGenerationTokenSource<'_, '_, '_, B, C>
where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    type Error = eredu_core::ControlledTextGenerationError<eredu_core::BackendFailure, C::Error>;
    fn finish_step<T, E>(
        &mut self,
        local: Result<T, E>,
        cancelled: bool,
        map_source: impl FnOnce(Self::Error) -> E,
    ) -> Result<Option<T>, E> {
        self.source.finish_step(local, cancelled, map_source)
    }
    fn delivery_failure(&self) -> Option<eredu_core::capture::CaptureError> {
        self.failure
            .and_then(|failure| failure())
            .or_else(|| self.source.delivery_failure())
    }
    fn next_token(
        &mut self,
        cancellation: &eredu_core::GenerationCancellationToken,
    ) -> Result<Option<u32>, Self::Error> {
        match &mut self.observer {
            Some(observer) => self
                .source
                .next_token_with_observer(cancellation, Some(&mut **observer)),
            None => self.source.next_token_with_observer(cancellation, None),
        }
    }
    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.source.grammar_is_complete()
    }
}

impl<B, C> CommittedTokenSource for BackendGenerationTokenSource<'_, B, C>
where
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
{
    type Error = eredu_core::ControlledTextGenerationError<eredu_core::BackendFailure, C::Error>;

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
        self.next_token_with_observer(cancellation, None)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.generator
            .controller_is_complete()
            .map_err(eredu_core::ControlledTextGenerationError::Controller)
    }
}

fn gemma_profile(
    recognition: probes::gemma::Recognition,
) -> crate::runtime::chat::PreparedFormatProfile {
    use crate::runtime::chat::{
        GEMMA4_STRUCTURAL_TOOL_SPEC,
        dialect::{DialectParameters, GenerationPromptBehavior},
        gemma::{self, TOOL_RESPONSE_OPEN, TURN_CLOSE},
    };

    let extra = [
        recognition.response_token.then_some(TOOL_RESPONSE_OPEN),
        recognition.turn_token.then_some(TURN_CLOSE),
    ];
    let semantic_tokens = ProfileStrings::with_optional(&probes::gemma::CHANNELS, extra);
    let semantic_stops = ProfileStrings::with_optional(&[], extra);
    let full_tool_tokens = ProfileStrings::new(&probes::gemma::TOOLS);
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
            ProfileStrings::new(&[])
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

    let structural_tokens = ProfileStrings::new(&probes::inkling::STRUCTURAL);
    let tool_structural_tokens = ProfileStrings::new(&probes::inkling::TOOL_STRUCTURAL);
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
            ProfileStrings::new(&[])
        },
        stop_sequences: ProfileStrings::new(&[END_SAMPLING]),
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

    let structural_tokens = ProfileStrings::new(&probes::muse::STRUCTURAL);
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
            ProfileStrings::new(&[])
        },
        stop_sequences: ProfileStrings::new(&[EOT]),
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
    let required_structural_tokens = ProfileStrings::new(declaration.structural);
    let stop_sequences = ProfileStrings::new(declaration.stops);
    let supports_tool_input_rendering = mapping_tool_arguments || string_tool_arguments;

    crate::runtime::chat::PreparedFormatProfile {
        identity: Some(identity.into()),
        dialect: Some(dialect),
        dialect_parameters: Some(parameters),
        tool_dialect: Some(dialect),
        tool_dialect_parameters: Some(parameters),
        generation_prompt_behavior,
        reasoning_template_control: crate::runtime::chat::ReasoningTemplateControl::Boolean(
            reasoning_template_kwarg,
        ),
        reasoning_effort_control: None,
        supports_reasoning_parsing: declaration.reasoning_parsing,
        supports_tool_reasoning,
        supports_tool_input_rendering,
        supports_mapping_tool_arguments: mapping_tool_arguments,
        supports_string_tool_arguments: string_tool_arguments,
        native_tool_unavailable_reason: (!supports_tool_input_rendering)
            .then(|| "recognized output has no validated structured tool-history rendering"),
        required_structural_tokens: required_structural_tokens.clone(),
        tool_required_structural_tokens: required_structural_tokens,
        stop_sequences,
    }
}

pub(crate) fn selected_profile(
    selected: probes::selection::Selected,
    request: &ChatTemplateRequest,
) -> crate::runtime::chat::PreparedFormatProfile {
    use probes::selection::Selected;
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
    if profile.identity.is_some_and(|identity| {
        identity.starts_with("qwen3.6.") || identity.starts_with("qwen3.8.")
    }) {
        use crate::runtime::chat::{
            QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING, QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
            QWEN_TAGGED_TOOL_SPEC_PREFILLED_REASONING,
        };
        let spec = if request.enable_thinking == Some(false) {
            &QWEN_TAGGED_TOOL_SPEC_NO_REASONING
        } else if profile
            .generation_prompt_behavior
            .resolve(request.add_generation_prompt)
        {
            &QWEN_TAGGED_TOOL_SPEC_PREFILLED_REASONING
        } else {
            &QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING
        };
        let parameters = crate::runtime::chat::dialect::DialectParameters::Declarative(spec);
        profile.dialect_parameters = Some(parameters);
        profile.tool_dialect_parameters = Some(parameters);
        profile.supports_reasoning_parsing = request.enable_thinking != Some(false);
    }
    profile
}

pub(crate) fn inspect_chat_from_parts(
    tokenizer: &mut ChatTokenizer,
    template: ModelChatTemplate,
    model_id: &str,
    eos_token_ids: &[u32],
    constraint_compiler: Option<&Result<ConstraintCompiler, String>>,
    request: ChatTemplateRequest,
) -> Result<crate::runtime::chat::ChatInspection, TextModelError> {
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
    let mapped_controls = profile
        .request_bindings(&request)
        .map_err(|cause| cause.ordinary(&profile, &request))?
        .into_owned();
    let compiler = constraint_compiler.and_then(|compiler| compiler.as_ref().ok());
    let unbounded = crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged();
    let funding = compiler.map_or(&unbounded, ConstraintCompiler::allocation_funding);
    let policy = policy::CompiledChatPolicy::compile(
        &profile,
        &request,
        eos_token_ids,
        constraint_compiler.map_or(Ok(None), |result| {
            result.as_ref().map(Some).map_err(String::as_str)
        }),
        funding,
        |spellings, _funding| resolve_structural_tokens(tokenizer, spellings),
    )
    .map_err(policy::CompilationFailure::into_ordinary)?;
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

    Ok(crate::runtime::chat::ChatInspection {
        rendered_prompt,
        generation_prompt,
        template_identity,
        policy,
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
impl TextChatEligibilityError {
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::Tools => {
                "text generation does not support tool declarations or required tool calls; prepare a request without tools"
            }
            Self::Thinking => {
                "text generation does not parse explicit thinking; set allow_unparsed_reasoning to opt into raw output"
            }
        }
    }
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
