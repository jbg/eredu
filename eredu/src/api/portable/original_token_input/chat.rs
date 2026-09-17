//! Private original J/H preparation and the existing C/S/E/I/R plain cursor.
use super::plain::{
    OriginalPlainSession, OriginalPlainStartError, plain_consumer_layout,
    start_original_plain_string_for,
};
use crate::api::request::probes::original::{
    Original as ProfileOperations, PolicyOwner,
};
use crate::api::request::{TextChatEligibilityError, text_chat_eligibility};
use crate::runtime::chat::ChatTemplateRequest;
use eredu_core::{
    GenerationCancellationToken, ModelRuntime, TextGenerationConfig, TokenInputRejection,
};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalChatOperationError, OriginalChatTemplate, OriginalRenderedChat,
    OriginalTextSourceBudget, OriginalTextSourceError, OriginalTokenizer,
};
use eredu_text::chat_storage::{ChatMessageError, ChatRenderContext};

/// Policy integrations not yet supplied by the managed text-only chat route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ManagedChatPolicyRejection {
    /// No chat template is attached to the loaded model.
    #[error("managed chat requires a selected chat template")]
    MissingTemplate,
    /// Tool declarations/required calls need their original policy owners.
    #[error("managed chat tool policy integration is unfinished")]
    Tools,
    /// Raw explicit thinking requires the caller's unparsed-output opt-in.
    #[error("managed chat explicit thinking requires raw-output permission")]
    Reasoning,
    /// A backend does not supply a required template-variable operation.
    #[error("managed chat template-variable integration is unfinished")]
    TemplateVariables,
}

#[derive(Debug, thiserror::Error)]
enum RequestError {
    #[error(transparent)]
    Text(#[from] TextChatEligibilityError),
    #[error(transparent)]
    Messages(#[from] ChatMessageError),
}
#[derive(Debug, thiserror::Error)]
enum Cause<E: std::error::Error + 'static> {
    #[error(transparent)]
    Profile(#[from] crate::api::request::probes::original::Failure),
    #[error(transparent)]
    Request(#[from] RequestError),
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    #[error(transparent)]
    Operation(#[from] OriginalChatOperationError),
    #[error(transparent)]
    Source(#[from] OriginalTextSourceError),
    #[error(transparent)]
    Startup(OriginalPlainStartError<E>),
}
/// Closed by-value failure. Actual S/E/I/R errors retire before retained H, which
/// in turn keeps its exact original J/C sources until its final strong owner.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct OriginalChatError<E: std::error::Error + 'static> {
    #[source]
    cause: Cause<E>,
    render: Option<OriginalRenderedChat>,
    profile: Option<PolicyOwner>,
    // Last: failed render/encoding prefixes retire before their domain ceiling.
    source_budget: Option<OriginalTextSourceBudget>,
}
impl<E: std::error::Error + 'static> OriginalChatError<E> {
    fn before_render(cause: impl Into<Cause<E>>) -> Self {
        Self {
            cause: cause.into(),
            render: None,
            profile: None,
            source_budget: None,
        }
    }
    fn retaining(cause: impl Into<Cause<E>>, render: OriginalRenderedChat) -> Self {
        Self {
            cause: cause.into(),
            render: Some(render),
            profile: None,
            source_budget: None,
        }
    }
    /// Preserve a fixed input rejection through the by-value startup wrapper.
    pub(crate) fn input_rejection(&self) -> Option<&TokenInputRejection> {
        match &self.cause {
            Cause::Input(error) | Cause::Startup(OriginalPlainStartError::Input(error)) => {
                Some(error)
            }
            _ => None,
        }
    }
    pub(crate) fn policy_rejection(&self) -> Option<ManagedChatPolicyRejection> {
        use ManagedChatPolicyRejection as P;
        match &self.cause {
            Cause::Request(RequestError::Text(TextChatEligibilityError::Tools)) => Some(P::Tools),
            Cause::Request(RequestError::Text(TextChatEligibilityError::Thinking)) => {
                Some(P::Reasoning)
            }
            _ => None,
        }
    }
    pub(crate) fn into_neutral<B: OriginalChatBackend<Error = E>>(
        self,
    ) -> OriginalChatError<eredu_core::BackendFailure> {
        use eredu_core::ControlledTextGenerationError as C;
        let Self {
            cause,
            render,
            profile,
            source_budget,
        } = self;
        // Tuple fields retire in order even if backend error translation unwinds:
        // the render must always finish before the retained source ceiling.
        let retained = (render, profile, source_budget);
        let cause = match cause {
            Cause::Profile(e) => Cause::Profile(e),
            Cause::Request(e) => Cause::Request(e),
            Cause::Input(e) => Cause::Input(e),
            Cause::Operation(e) => Cause::Operation(e),
            Cause::Source(e) => Cause::Source(e),
            Cause::Startup(error) => Cause::Startup(match error {
                OriginalPlainStartError::Input(e) => OriginalPlainStartError::Input(e),
                OriginalPlainStartError::Generation(e) => OriginalPlainStartError::Generation(e),
                OriginalPlainStartError::Stop(e) => OriginalPlainStartError::Stop(e),
                OriginalPlainStartError::Backend(e) => OriginalPlainStartError::Backend(e),
                OriginalPlainStartError::Source(e) => OriginalPlainStartError::Source(e),
                OriginalPlainStartError::Startup(error) => {
                    OriginalPlainStartError::Startup(match error {
                        C::Preparation(e) => C::Preparation(e),
                        C::Backend(e) => C::Backend(B::into_backend_failure(e)),
                        C::Controller(e) => C::Controller(e),
                    })
                }
            }),
        };
        OriginalChatError {
            cause,
            render: retained.0,
            profile: retained.1,
            source_budget: retained.2,
        }
    }
    pub(crate) fn retained_render_bytes(&self) -> u64 {
        self.render
            .as_ref()
            .map_or(0, OriginalRenderedChat::original_bytes)
    }
}

/// Move-only rendering and the ceiling admitted before its construction.
/// No render alias escapes this preparation; startup consumes the same owner.
#[derive(Debug)]
pub(crate) struct OriginalTextChatPreparation {
    render: OriginalRenderedChat,
    capacity: u64,
    profile: Option<PolicyOwner>,
    source_budget: OriginalTextSourceBudget,
}
impl OriginalTextChatPreparation {
    pub(crate) fn prompt(&self, generation_prompt: bool) -> &str {
        self.render
            .prompt(self.profile.as_ref().map_or(generation_prompt, |profile| {
                profile.generation(generation_prompt)
            }))
    }
    pub(crate) fn generation_suffix(&self) -> &str {
        self.render.generation_suffix()
    }
    pub(crate) fn original_bytes(&self) -> u64 {
        self.render.original_bytes()
    }
    #[cfg(test)]
    pub(super) fn clone_render_for_test(&self) -> OriginalRenderedChat {
        self.render.clone()
    }
}

fn consumer_layout<B: OriginalChatBackend, E>()
-> Option<eredu_core::GenerationSequenceConsumerLayout> {
    plain_consumer_layout::<B, (OriginalChatError<B::Error>, OriginalTextChatPreparation, E)>()
}
fn context<'a>(
    request: &'a ChatTemplateRequest,
    defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
) -> Result<ChatRenderContext<'a>, RequestError> {
    text_chat_eligibility(request)?;
    // Scalar controls are projected only after the exact shared profile source
    // has been selected and paid. Tool/output permission remains this predicate.
    ChatRenderContext::from_json(
        &request.messages,
        defaults,
        Some(&request.extra_template_kwargs),
    )
    .map(|context| context.with_clock(eredu_text::chat_storage::chat_clock_snapshot()))
    .map_err(Into::into)
}
/// Consumed File producer. Path opening and File preparation precede original I;
/// its admitted bytes remain live through fresh J in the same backend pool.
pub(crate) fn compile_original_chat_file<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    file: std::fs::File,
    model_id: &str,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalChatTemplate>, OriginalChatOperationError> {
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let read = eredu_checkpoint::artifact::PreparedArtifactFileRead::new(file)?;
    let source = B::compile_original_chat_template_file(runtime, read, model_id)?;
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    Ok(Some(source))
}
/// Borrow the real request fields through measurement and rendering. No message
/// clone, ordinary Environment, third prompt String or semantic parser is used.
pub(crate) fn prepare_original_text_chat<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    request: &ChatTemplateRequest,
    config: TextGenerationConfig,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalTextChatPreparation>, OriginalChatError<B::Error>> {
    prepare_original_text_chat_for::<B, ()>(
        runtime,
        template,
        tokenizer,
        request,
        config,
        cancellation,
    )
}
pub(crate) fn prepare_original_text_chat_for<B: OriginalChatBackend, E>(
    runtime: &ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    request: &ChatTemplateRequest,
    config: TextGenerationConfig,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalTextChatPreparation>, OriginalChatError<B::Error>> {
    prepare_original_text_chat_with_defaults_for::<B, E>(
        runtime,
        template,
        tokenizer,
        request,
        None,
        config,
        cancellation,
    )
}
/// Loaded tokenizer defaults and actual caller replacements are borrowed until
/// both renders complete; semantic tool/reasoning policy remains facade-owned.
pub(crate) fn prepare_original_text_chat_with_defaults_for<B: OriginalChatBackend, E>(
    runtime: &ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    request: &ChatTemplateRequest,
    defaults: Option<&serde_json::Map<String, serde_json::Value>>,
    config: TextGenerationConfig,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalTextChatPreparation>, OriginalChatError<B::Error>> {
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let context = context(request, defaults).map_err(OriginalChatError::before_render)?;
    let consumer = consumer_layout::<B, E>()
        .ok_or_else(|| OriginalChatError::before_render(TokenInputRejection::Overflow))?;
    B::validate_original_chat_sources(runtime, template, tokenizer)
        .map_err(OriginalChatError::before_render)?;
    let capacity = config
        .inference_policy()
        .managed_memory_capacity_bytes
        .ok_or_else(|| OriginalChatError::before_render(TokenInputRejection::Unsupported))?;
    let source_budget = B::prepare_original_text_source_budget(runtime, tokenizer, capacity)
        .map_err(OriginalChatError::before_render)?;
    // Recognition also supplies default generation behavior and history checks.
    // Its source and controls are admitted before any probe or request render.
    let policy = match ProfileOperations::prepare(
        runtime, template, tokenizer, defaults, capacity, consumer,
    ).and_then(|mut operations| operations.prepare_policy(request)) {
        Ok(policy) => policy,
        Err(cause) => {
            let mut error = OriginalChatError::before_render(cause);
            error.source_budget = Some(source_budget);
            return Err(error);
        }
    };
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let context = context.with_scalar_overrides(policy.bindings());
    let rendered =
        B::render_original_chat_with_context(runtime, template, tokenizer, context, consumer);
    let profile = Some(policy.into_owner());
    let render = match rendered {
        Ok(render) => render,
        Err(cause) => {
            let mut error = OriginalChatError::before_render(cause);
            error.profile = profile;
            error.source_budget = Some(source_budget);
            return Err(error);
        }
    };
    if !render.has_sources(template, tokenizer)
        || !render.accepts_consumer(&consumer)
        || profile
            .as_ref()
            .is_some_and(|profile| !profile.has_sources(template, tokenizer))
    {
        let mut error = OriginalChatError::retaining(TokenInputRejection::IdentityMismatch, render);
        error.profile = profile;
        error.source_budget = Some(source_budget);
        return Err(error);
    }
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    // Controls and history have been consumed by the completed render. Only a
    // generation override remains a policy consumer at startup; pass-through
    // policies can release their completed probe account here.
    let profile = profile.filter(|owner| owner.generation(false));
    Ok(Some(OriginalTextChatPreparation {
        render,
        capacity,
        profile,
        source_budget,
    }))
}
/// Consume the closed H while its prompt is borrowed by the ordinary private
/// S/E→I startup. Success ends that borrow before retiring H. Failure retains H.
pub(crate) fn start_original_text_chat<'a, B: OriginalChatBackend>(
    runtime: &'a mut ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    prepared: OriginalTextChatPreparation,
    generation_prompt: bool,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalChatError<B::Error>> {
    start_original_text_chat_for::<B, ()>(
        runtime,
        template,
        tokenizer,
        prepared,
        generation_prompt,
        config,
        eos,
        stops,
        skip_special_tokens,
        cancellation,
    )
}
pub(crate) fn start_original_text_chat_for<'a, B: OriginalChatBackend, E>(
    runtime: &'a mut ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    prepared: OriginalTextChatPreparation,
    generation_prompt: bool,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalChatError<B::Error>> {
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let capacity = prepared.capacity;
    // The render moves into the inner call only after the outer ceiling exists;
    // its cleanup therefore precedes the ceiling even on unwind.
    let source_budget = prepared.source_budget;
    let profile = prepared.profile;
    let render = prepared.render;
    let generation_prompt = profile.as_ref().map_or(generation_prompt, |profile| {
        profile.generation(generation_prompt)
    });
    let result = if config.inference_policy().managed_memory_capacity_bytes != Some(capacity)
        || profile
            .as_ref()
            .is_some_and(|profile| !profile.has_sources(template, tokenizer))
    {
        Err(OriginalChatError::retaining(
            TokenInputRejection::IdentityMismatch,
            render,
        ))
    } else {
        start_rendered_text_chat::<B, E>(
            runtime,
            template,
            tokenizer,
            render,
            generation_prompt,
            config,
            eos,
            stops,
            skip_special_tokens,
            cancellation,
        )
    };
    match result {
        Ok(session) => Ok(session), // Request Q now enforces this same ceiling.
        Err(mut error) => {
            error.profile = profile;
            error.source_budget = Some(source_budget);
            Err(error)
        }
    }
}

fn start_rendered_text_chat<'a, B: OriginalChatBackend, E>(
    runtime: &'a mut ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    render: OriginalRenderedChat,
    generation_prompt: bool,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalChatError<B::Error>> {
    let consumer = match consumer_layout::<B, E>() {
        Some(consumer) => consumer,
        None => {
            return Err(OriginalChatError::retaining(
                TokenInputRejection::Overflow,
                render,
            ));
        }
    };
    if !render.has_sources(template, tokenizer) || !render.accepts_consumer(&consumer) {
        return Err(OriginalChatError::retaining(
            TokenInputRejection::IdentityMismatch,
            render,
        ));
    }
    if let Err(error) = B::validate_original_chat_render(runtime, &render) {
        return Err(OriginalChatError::retaining(error, render));
    }
    let result = start_original_plain_string_for::<
        B,
        (OriginalChatError<B::Error>, OriginalTextChatPreparation, E),
    >(
        runtime,
        tokenizer,
        render.prompt(generation_prompt),
        config,
        eos,
        stops,
        false,
        skip_special_tokens,
        cancellation,
    );
    match result {
        Ok(session) => {
            drop(render);
            Ok(session)
        }
        Err(error) => Err(OriginalChatError::retaining(Cause::Startup(error), render)),
    }
}
