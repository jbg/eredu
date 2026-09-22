//! Original source, policy and render preparation for every chat execution.
use crate::api::request::probes::original::{Original as ProfileOperations, PolicyOwner};
use crate::runtime::chat::ChatTemplateRequest;
use eredu_core::{GenerationCancellationToken, ModelRuntime, TokenInputRejection};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalChatRenderOperationError, OriginalChatSourceError,
    OriginalChatTemplate, OriginalControllerCompilation, OriginalControllerCompilationError,
    OriginalRenderedChat, OriginalTextSourceBudget, OriginalTextSourceError, OriginalTokenizer,
};
use eredu_text::chat_storage::{ChatMessageError, ChatRenderContext};

type PolicyCompilationError =
    OriginalControllerCompilationError<crate::api::request::policy::OriginalPolicyFailure>;

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Metadata(#[from] eredu_core::HostMetadataFundingError),
    #[error(transparent)]
    Profile(#[from] crate::api::request::probes::original::Failure),
    #[error(transparent)]
    Policy(#[from] PolicyCompilationError),
    #[error(transparent)]
    Request(#[from] ChatMessageError),
    #[error(transparent)]
    Input(#[from] TokenInputRejection),
    #[error(transparent)]
    Operation(#[from] OriginalChatRenderOperationError),
    #[error(transparent)]
    Source(#[from] OriginalTextSourceError),
}
/// By-value preparation failure. Render and policy storage retire before the
/// original source ceiling that admitted their construction.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct OriginalChatError {
    #[source]
    cause: Cause,
    render: Option<OriginalRenderedChat>,
    profile: Option<PolicyOwner>,
    // Last: failed render/encoding prefixes retire before their domain ceiling.
    source_budget: Option<OriginalTextSourceBudget>,
}
impl OriginalChatError {
    fn before_render(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            render: None,
            profile: None,
            source_budget: None,
        }
    }
    pub(crate) fn input_rejection(&self) -> Option<&TokenInputRejection> {
        match &self.cause {
            Cause::Input(error) => Some(error),
            _ => None,
        }
    }
}

/// Reusable rendering, selected policy and their original construction ceiling.
/// The execution consumer is bound separately when a session starts.
#[derive(Debug)]
pub(crate) struct PreparedChatRendering {
    render: OriginalRenderedChat,
    limits: eredu_core::MemoryLimitDeclarations,
    profile: Option<PolicyOwner>,
    source_budget: OriginalTextSourceBudget,
}
impl PreparedChatRendering {
    #[cfg(test)]
    pub(crate) fn prompt(&self, generation: bool) -> &str {
        self.render
            .prompt(self.profile.as_ref().unwrap().generation(generation))
    }
    #[cfg(test)]
    pub(crate) fn profile(&self) -> &crate::runtime::chat::PreparedFormatProfile {
        self.profile.as_ref().unwrap().profile()
    }
    #[cfg(test)]
    pub(crate) fn render(&self) -> &OriginalRenderedChat {
        &self.render
    }
    #[cfg(test)]
    pub(crate) fn metadata_funding(&self) -> &eredu_core::HostMetadataFunding {
        self.profile.as_ref().unwrap().metadata_funding()
    }
    pub(crate) fn publish(
        self,
        policy: crate::api::CompiledChatPolicy,
        compilation: OriginalControllerCompilation,
        named_entry: Option<&str>,
        requested_generation: bool,
    ) -> Result<
        crate::runtime::chat::PreparedChat,
        crate::runtime::chat::PreparedChatPublicationError,
    > {
        let generation = self
            .profile
            .as_ref()
            .expect("prepared profile")
            .generation(requested_generation);
        crate::runtime::chat::PreparedChat::publish(
            self.render,
            generation,
            self.limits,
            named_entry,
            policy,
            compilation,
            self.source_budget,
        )
    }
}

fn context<'a>(
    request: &'a ChatTemplateRequest,
    defaults: Option<&'a serde_json::Map<String, serde_json::Value>>,
) -> Result<ChatRenderContext<'a>, ChatMessageError> {
    // Scalar controls are projected only after the exact shared profile source
    // has been selected and paid. Tool/output permission remains this predicate.
    ChatRenderContext::from_json(
        &request.messages,
        defaults,
        Some(&request.extra_template_kwargs),
    )
    .map(|context| {
        context
            .with_tools(eredu_text::chat_storage::ChatInputArray::Json(
                if request.tool_choice == crate::runtime::chat::ToolChoice::None {
                    &[]
                } else {
                    &request.tools
                },
            ))
            .with_clock(eredu_text::chat_storage::chat_clock_snapshot())
    })
}
/// Consumed File producer. Path opening and File preparation precede original I;
/// its admitted bytes remain live through fresh J in the same backend pool.
pub(crate) fn compile_original_chat_file<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    file: std::fs::File,
    model_id: &str,
    has_tools: bool,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalChatTemplate>, OriginalChatSourceError> {
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let read = eredu_checkpoint::artifact::PreparedArtifactFileRead::new(file)?;
    let source = B::compile_original_chat_template_file(runtime, read, model_id, has_tools)?;
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    Ok(Some(source))
}
/// Compile semantic declarations before rendering the actual request. The
/// returned policy and receipt name the original owners used by all consumers.
pub(crate) fn prepare_original_chat<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    request: &ChatTemplateRequest,
    defaults: Option<&serde_json::Map<String, serde_json::Value>>,
    eos: &[u32],
    limits: &eredu_core::MemoryLimitDeclarations,
    grammar_memory: crate::runtime::chat::DependencyMemoryPolicy,
    cancellation: &GenerationCancellationToken,
) -> Result<
    Option<(
        PreparedChatRendering,
        (
            crate::api::CompiledChatPolicy,
            OriginalControllerCompilation,
        ),
    )>,
    OriginalChatError,
> {
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let domain = tokenizer
        .generation_domain()
        .ok_or_else(|| OriginalChatError::before_render(TokenInputRejection::Unavailable))?;
    if eos.iter().any(|&id| !domain.allows(id)) {
        return Err(OriginalChatError::before_render(
            TokenInputRejection::InvalidToken,
        ));
    }
    prepare_with_policy(
        runtime,
        template,
        tokenizer,
        request,
        defaults,
        limits,
        cancellation,
        |profile| {
            crate::api::request::policy::compile_original(
                runtime,
                profile.profile(),
                request,
                eos,
                profile.preparation(),
                grammar_memory,
            )
        },
    )
}

/// The canonical source, profile, declaration and request-render worker.
/// The callback permits direct cancellation/refusal coverage of this same producer.
pub(super) fn prepare_with_policy<B: OriginalChatBackend, P, F>(
    runtime: &ModelRuntime<B>,
    template: &OriginalChatTemplate,
    tokenizer: &OriginalTokenizer,
    request: &ChatTemplateRequest,
    defaults: Option<&serde_json::Map<String, serde_json::Value>>,
    limits: &eredu_core::MemoryLimitDeclarations,
    cancellation: &GenerationCancellationToken,
    compile: F,
) -> Result<Option<(PreparedChatRendering, P)>, OriginalChatError>
where
    F: FnOnce(&PolicyOwner) -> Result<P, PolicyCompilationError>,
{
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let context = context(request, defaults).map_err(OriginalChatError::before_render)?;
    B::validate_original_chat_sources(runtime, template, tokenizer)
        .map_err(OriginalChatError::before_render)?;
    let source_budget = B::prepare_original_text_source_budget(runtime, tokenizer, limits)
        .map_err(OriginalChatError::before_render)?;
    // Recognition also supplies default generation behavior and history checks.
    // Its source and controls are admitted before any probe or request render.
    let policy = match ProfileOperations::prepare(runtime, template, tokenizer, defaults, limits)
        .and_then(|mut operations| operations.prepare_policy(request))
    {
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
    // Rendering has no eventual execution consumer. Its own facade producer,
    // return and retained failure controls belong to the source preparation.
    let parts = [
        std::mem::size_of::<PreparedChatRendering>(),
        std::mem::size_of::<P>(),
        std::mem::size_of::<F>(),
        std::mem::size_of::<Result<P, PolicyCompilationError>>(),
        std::mem::size_of::<Option<(PreparedChatRendering, P)>>(),
        std::mem::size_of::<Result<Option<(PreparedChatRendering, P)>, OriginalChatError>>(),
        std::mem::size_of::<OriginalChatError>(),
        std::mem::size_of::<Result<Option<PreparedChatRendering>, OriginalChatError>>(),
        std::mem::size_of::<Result<OriginalRenderedChat, OriginalChatRenderOperationError>>(),
        std::mem::size_of::<Result<(), eredu_core::HostMetadataFundingError>>(),
        std::mem::size_of::<Option<OriginalRenderedChat>>(),
        std::mem::size_of::<Option<PolicyOwner>>(),
        std::mem::size_of::<(
            ChatRenderContext<'_>,
            &ModelRuntime<B>,
            &OriginalChatTemplate,
            &OriginalTokenizer,
            &ChatTemplateRequest,
            Option<&serde_json::Map<String, serde_json::Value>>,
            u64,
            &GenerationCancellationToken,
        )>(),
    ];
    let controls = parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
        .ok_or(eredu_core::HostMetadataFundingError::Overflow)
        .and_then(|bytes| policy.metadata_funding().reserve_metadata(bytes));
    if let Err(cause) = controls {
        let mut error = OriginalChatError::before_render(cause);
        error.profile = Some(policy.into_owner());
        error.source_budget = Some(source_budget);
        return Err(error);
    }
    let compiled = match compile(policy.owner()) {
        Ok(compiled) => compiled,
        Err(cause) => {
            let mut error = OriginalChatError::before_render(cause);
            error.profile = Some(policy.into_owner());
            error.source_budget = Some(source_budget);
            return Err(error);
        }
    };
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    let context = context.with_scalar_overrides(policy.bindings());
    let rendered = B::render_original_chat(runtime, template, tokenizer, context);
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
        || profile
            .as_ref()
            .is_some_and(|profile| !profile.has_sources(template, tokenizer))
    {
        let mut error = OriginalChatError::before_render(TokenInputRejection::IdentityMismatch);
        error.render = Some(render);
        error.profile = profile;
        error.source_budget = Some(source_budget);
        return Err(error);
    }
    if cancellation.is_cancelled() {
        return Ok(None);
    }
    if let Err(cause) = B::validate_original_chat_render(runtime, &render) {
        let mut error = OriginalChatError::before_render(cause);
        error.render = Some(render);
        error.profile = profile;
        error.source_budget = Some(source_budget);
        return Err(error);
    }
    Ok(Some((
        PreparedChatRendering {
            render,
            limits: limits.clone(),
            profile,
            source_budget,
        },
        compiled,
    )))
}
