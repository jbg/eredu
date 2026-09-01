//! Application-facing adapter for the execution backend selected by `eredu` features.
//!
//! Native device, stream, tensor, random-state, and allocator handles stay
//! behind this module. Applications configure and operate the selected local
//! backend through portable plans and facade-owned diagnostics.
//!
//! The selected backend and its session are not part of the application API.

use std::path::Path;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eredu_core::{RealtimeBackend as _, RealtimeModelLoadingBackend as _, TokenOutput as _};

/// Discovers hardware available to the selected local backend.
pub fn discover_local_hardware() -> eredu_core::HardwareProfile {
    eredu_backend_mlx::discover_hardware()
}

type SelectedBackend = eredu_backend_mlx::backend::MlxBackend<'static>;
type SelectedDrafter = eredu_backend_mlx::native::MlxDrafter;
type SelectedPrompt = <SelectedBackend as eredu_core::TextGenerationBackend>::Prompt;

/// Automatic planner and execution-plan factory for the selected local backend.
///
/// Its concrete backend factory is intentionally not exposed or implemented as
/// a public backend trait. Application code realizes plans through [`LocalModel`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalBackendFactory {
    inner: eredu_backend_mlx::MlxBackendFactory,
}

impl LocalBackendFactory {
    /// Enables allocator and process-memory sampling for bounded residency.
    pub const fn with_residency_diagnostics(
        mut self,
        sample_backend_memory: bool,
        sample_process_memory: bool,
    ) -> Self {
        self.inner = self
            .inner
            .with_residency_diagnostics(sample_backend_memory, sample_process_memory);
        self
    }

    /// Produces a portable execution plan without exposing the selected backend.
    pub fn plan(
        &self,
        planner: &crate::AutomaticPlanner,
        request: &crate::AutomaticPlanRequest,
    ) -> Result<crate::ExecutionPlanReport, crate::AutomaticPlanningError> {
        planner.plan(&self.inner, request)
    }
}

/// Scoped opt-in for selected-backend speculative component timing.
pub struct LocalSpeculativeComponentTimingGuard {
    _inner: eredu_backend_mlx::SpeculativeComponentTimingGuard,
}

impl LocalSpeculativeComponentTimingGuard {
    /// Enables component timing until the returned guard is dropped.
    pub fn enable() -> Self {
        Self {
            _inner: eredu_backend_mlx::SpeculativeComponentTimingGuard::enable(),
        }
    }
}

/// Opaque failure reported through the selected-local-backend facade.
///
/// The facade retains a stable error surface instead of exposing errors from
/// the selected native runtime. The diagnostic message contains the backend's
/// original context.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("selected local backend failed during {operation}: {message}")]
pub struct LocalBackendError {
    operation: &'static str,
    message: String,
}

impl LocalBackendError {
    fn new(operation: &'static str, error: impl std::fmt::Display) -> Self {
        Self {
            operation,
            message: error.to_string(),
        }
    }

    /// Facade operation that failed.
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Backend diagnostic without exposing its native error type.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Failure while loading a tokenizer-aware model through the local facade.
#[derive(Debug, thiserror::Error)]
pub enum LocalModelLoadError {
    /// Portable artifact inspection or preparation planning failed.
    #[error(transparent)]
    Artifact(#[from] eredu_core::artifact::ArtifactError),
    /// The selected backend failed materialization or session creation.
    #[error(transparent)]
    Backend(#[from] LocalBackendError),
    /// The inspected model/session route lacks a required capability.
    #[error(transparent)]
    SessionCapability(#[from] eredu_core::SessionCapabilityError),
    /// Portable tokenizer, chat-template, or generation metadata loading failed.
    #[error(transparent)]
    Metadata(#[from] super::TextMetadataError),
}

/// Failure while planning and loading a model through the local facade.
#[derive(Debug, thiserror::Error)]
pub enum LocalPlannedModelLoadError {
    /// Portable planning or device realization failed.
    #[error(transparent)]
    Planning(#[from] crate::AutomaticPlanningError),
    /// Artifact, metadata, materialization, or session creation failed.
    #[error(transparent)]
    Loading(#[from] LocalModelLoadError),
}

fn map_local_model_load_error(
    error: super::LoadedModelLoadError<eredu_backend_mlx::backend::error::Error>,
) -> LocalModelLoadError {
    match error {
        super::LoadedModelLoadError::Artifact(error) => LocalModelLoadError::Artifact(error),
        super::LoadedModelLoadError::Backend(error) => {
            LocalBackendError::new("model loading", error).into()
        }
        super::LoadedModelLoadError::SessionCapability(error) => {
            LocalModelLoadError::SessionCapability(error)
        }
        super::LoadedModelLoadError::Metadata(error) => LocalModelLoadError::Metadata(error),
    }
}

fn map_local_planned_model_load_error(
    error: super::PlannedModelLoadError<eredu_backend_mlx::backend::error::Error>,
) -> LocalPlannedModelLoadError {
    match error {
        super::PlannedModelLoadError::Planning(error) => {
            LocalPlannedModelLoadError::Planning(error)
        }
        super::PlannedModelLoadError::Loading(error) => {
            LocalPlannedModelLoadError::Loading(map_local_model_load_error(error))
        }
    }
}

/// Opaque prompt prepared for the selected local model.
pub struct LocalPrompt {
    inner: SelectedPrompt,
}

/// Explicit prompt source for local prepared-chat generation.
pub enum LocalPreparedChatInput<'a> {
    /// Tokenizes the rendered prompt stored in the prepared chat.
    RenderedPrompt(&'a super::PreparedChat),
    /// Uses a facade-owned prompt prepared from portable media.
    PreparedPrompt {
        /// Prepared chat that owns generation semantics.
        prepared_chat: &'a super::PreparedChat,
        /// Opaque selected-backend prompt.
        prompt: LocalPrompt,
    },
}

impl<'a> LocalPreparedChatInput<'a> {
    /// Creates a text-only input from the prepared chat's rendered prompt.
    pub const fn rendered_prompt(prepared_chat: &'a super::PreparedChat) -> Self {
        Self::RenderedPrompt(prepared_chat)
    }

    /// Binds a facade-prepared prompt to prepared-chat semantics.
    pub fn prepared_prompt(prepared_chat: &'a super::PreparedChat, prompt: LocalPrompt) -> Self {
        Self::PreparedPrompt {
            prepared_chat,
            prompt,
        }
    }

    fn into_backend(self) -> super::PreparedChatInput<'a, SelectedBackend> {
        match self {
            Self::RenderedPrompt(prepared_chat) => {
                super::PreparedChatInput::rendered_prompt(prepared_chat)
            }
            Self::PreparedPrompt {
                prepared_chat,
                prompt,
            } => super::PreparedChatInput::prepared_backend_input(prepared_chat, prompt.inner),
        }
    }
}

/// Request for ordinary semantic generation through the local facade.
pub struct LocalPreparedChatGenerationRequest<'a, F> {
    /// Text-only or facade-prepared multimodal prompt.
    pub input: LocalPreparedChatInput<'a>,
    /// Portable sampling configuration, token limit, and random seed.
    pub settings: super::PreparedChatGenerationSettings,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// Cooperative cancellation token.
    pub cancellation: eredu_core::generation::GenerationCancellationToken,
    /// Called synchronously as semantic events become available.
    pub on_event: F,
}

/// Request for speculative semantic generation through the local facade.
pub struct LocalPreparedChatSpeculativeGenerationRequest<'a, F> {
    /// Text-only or facade-prepared multimodal prompt.
    pub input: LocalPreparedChatInput<'a>,
    /// Opaque drafting realization loaded with the target model.
    pub drafting: &'a mut LocalDrafting,
    /// Portable sampling configuration, token limit, and random seed.
    pub settings: super::PreparedChatGenerationSettings,
    /// Proposal-block and scheduler controls.
    pub options: super::PreparedChatSpeculativeGenerationOptions,
    /// Additional decoded text sequences that terminate generation.
    pub caller_stop_sequences: &'a [String],
    /// Cooperative cancellation token.
    pub cancellation: eredu_core::generation::GenerationCancellationToken,
    /// Called synchronously as semantic events become available.
    pub on_event: F,
}

fn map_prepared_chat_error(
    error: super::PreparedChatError<eredu_backend_mlx::backend::error::Error>,
) -> super::PreparedChatError<LocalBackendError> {
    match error {
        super::PreparedChatError::Backend(error) => super::PreparedChatError::Backend(
            LocalBackendError::new("prepared-chat generation", error),
        ),
        super::PreparedChatError::Constraint(error) => super::PreparedChatError::Constraint(error),
        super::PreparedChatError::Generation(error) => super::PreparedChatError::Generation(error),
        super::PreparedChatError::Tokenizer(error) => super::PreparedChatError::Tokenizer(error),
        super::PreparedChatError::Semantic(error) => super::PreparedChatError::Semantic(error),
        super::PreparedChatError::MissingTerminalToken => {
            super::PreparedChatError::MissingTerminalToken
        }
    }
}

fn map_prepared_chat_speculative_error(
    error: super::PreparedChatSpeculativeError<eredu_backend_mlx::backend::error::Error>,
) -> super::PreparedChatSpeculativeError<LocalBackendError> {
    match error {
        super::PreparedChatSpeculativeError::Backend(error) => {
            super::PreparedChatSpeculativeError::Backend(LocalBackendError::new(
                "prepared-chat speculative generation",
                error,
            ))
        }
        super::PreparedChatSpeculativeError::Generation(error) => {
            super::PreparedChatSpeculativeError::Generation(error)
        }
        super::PreparedChatSpeculativeError::Text(error) => {
            super::PreparedChatSpeculativeError::Text(error)
        }
        super::PreparedChatSpeculativeError::Constraint(error) => {
            super::PreparedChatSpeculativeError::Constraint(error)
        }
        super::PreparedChatSpeculativeError::Semantic(error) => {
            super::PreparedChatSpeculativeError::Semantic(error)
        }
        super::PreparedChatSpeculativeError::OutputCardinality { expected, actual } => {
            super::PreparedChatSpeculativeError::OutputCardinality { expected, actual }
        }
    }
}

/// Opaque drafting resources realized with a local model.
pub struct LocalDrafting {
    inner: eredu_core::RealizedDrafting<SelectedDrafter>,
}

impl LocalDrafting {
    /// Returns whether speculative generation was selected.
    pub const fn is_enabled(&self) -> bool {
        !matches!(self.inner, eredu_core::RealizedDrafting::Disabled)
    }

    /// Returns whether a separately loaded assistant is owned by this plan.
    pub const fn is_external(&self) -> bool {
        self.inner.is_external()
    }
}

/// One local model plus the opaque drafting resources from its execution plan.
pub struct LocalPlannedModel {
    model: LocalModel,
    drafting: LocalDrafting,
}

impl LocalPlannedModel {
    /// Borrows the loaded target model.
    pub const fn model(&self) -> &LocalModel {
        &self.model
    }

    /// Mutably borrows the loaded target model.
    pub fn model_mut(&mut self) -> &mut LocalModel {
        &mut self.model
    }

    /// Borrows the realized drafting mode.
    pub const fn drafting(&self) -> &LocalDrafting {
        &self.drafting
    }

    /// Mutably borrows both facade-owned resources.
    pub fn parts_mut(&mut self) -> (&mut LocalModel, &mut LocalDrafting) {
        (&mut self.model, &mut self.drafting)
    }

    /// Consumes the plan into its facade-owned target and drafting resources.
    pub fn into_parts(self) -> (LocalModel, LocalDrafting) {
        (self.model, self.drafting)
    }
}

/// Asynchronous local token generation with portable token ids and opaque errors.
pub struct LocalTextGeneration<'a> {
    inner: eredu_core::TextGeneration<'a, SelectedBackend>,
}

impl Iterator for LocalTextGeneration<'_> {
    type Item = Result<u32, LocalBackendError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|result| {
            result
                .map_err(|error| LocalBackendError::new("text generation", error))
                .and_then(|token| {
                    token
                        .token_id()
                        .map_err(|error| LocalBackendError::new("token materialization", error))
                })
        })
    }
}

/// Loaded local model with no public backend, session, tensor, or completion type.
pub struct LocalModel {
    inner: super::LoadedModel<SelectedBackend>,
}

impl LocalModel {
    /// Realizes a complete execution plan through the selected local backend.
    pub fn load_execution_plan(
        factory: &LocalBackendFactory,
        artifact: impl AsRef<Path>,
        plan: &crate::ExecutionPlan,
    ) -> Result<LocalPlannedModel, LocalPlannedModelLoadError> {
        let planned = super::LoadedModel::load_execution_plan(&factory.inner, artifact, plan)
            .map_err(map_local_planned_model_load_error)?;
        let (model, drafting) = planned.into_parts();
        Ok(LocalPlannedModel {
            model: Self { inner: model },
            drafting: LocalDrafting { inner: drafting },
        })
    }

    /// Plans and loads a model without exposing backend realization types.
    pub fn plan_and_load(
        factory: &LocalBackendFactory,
        planner: &crate::AutomaticPlanner,
        request: &crate::AutomaticPlanRequest,
    ) -> Result<(LocalPlannedModel, crate::ExecutionPlanReport), LocalPlannedModelLoadError> {
        let report = factory.plan(planner, request)?;
        let model = Self::load_execution_plan(factory, &request.model_path, &report.plan)?;
        Ok((model, report))
    }

    /// Returns the canonical architecture family.
    pub const fn model_family(&self) -> super::ModelKind {
        self.inner.model_family()
    }

    /// Returns the parsed implementation or nested text-model type.
    pub fn effective_model_type(&self) -> &str {
        self.inner.effective_model_type()
    }

    /// Returns the model id used for chat-template rendering.
    pub fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    /// Borrows the portable tokenizer.
    pub const fn tokenizer(&self) -> &super::ChatTokenizer {
        self.inner.tokenizer()
    }

    /// Returns the stable token-id vocabulary fingerprint.
    pub const fn tokenizer_fingerprint(&self) -> &[u8; 32] {
        self.inner.tokenizer_fingerprint()
    }

    /// Encodes text to portable vocabulary ids.
    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, super::TextModelError> {
        self.inner.encode(text, add_special_tokens)
    }

    /// Decodes portable vocabulary ids.
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, super::TextModelError> {
        self.inner.decode(ids, skip_special_tokens)
    }

    /// Creates an incremental portable text decoder.
    pub fn text_decoder(&self, skip_special_tokens: bool) -> super::TextDecoder {
        self.inner.text_decoder(skip_special_tokens)
    }

    /// Returns checkpoint sampling recommendations, when present.
    pub fn checkpoint_generation_config(
        &self,
    ) -> Option<&eredu_core::generation::CheckpointGenerationConfig> {
        self.inner.checkpoint_generation_config()
    }

    /// Resolves request overrides over checkpoint recommendations and defaults.
    pub fn resolve_generation_config(
        &self,
        overrides: eredu_core::generation::GenerationConfigOverrides,
    ) -> Result<
        eredu_core::generation::ResolvedGenerationConfig,
        eredu_core::generation::GenerationError,
    > {
        self.inner.resolve_generation_config(overrides)
    }

    /// Starts local token generation without exposing backend token handles.
    pub fn generate_tokens(
        &mut self,
        prompt_token_ids: Vec<u32>,
        config: eredu_core::TextGenerationConfig,
    ) -> Result<LocalTextGeneration<'_>, LocalBackendError> {
        self.inner
            .generate_tokens(prompt_token_ids, config)
            .map(|inner| LocalTextGeneration { inner })
            .map_err(|error| LocalBackendError::new("text generation", error))
    }

    /// Returns whether a chat template is attached.
    pub fn has_chat_template(&self) -> bool {
        self.inner.has_chat_template()
    }

    /// Returns the configured EOS vocabulary ids.
    pub fn eos_token_ids(&self) -> &[u32] {
        self.inner.eos_token_ids()
    }

    /// Returns true when `id` is a configured EOS id.
    pub fn is_eos_token(&self, id: u32) -> bool {
        self.inner.is_eos_token(id)
    }

    /// Returns the selected chat-template identity for `tools`.
    pub fn selected_chat_template_identity(
        &self,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Option<eredu_text::tokenizer::ChatTemplateIdentity>, super::TextModelError> {
        self.inner.selected_chat_template_identity(tools)
    }

    /// Returns likely user-provided variables referenced by the chat template.
    pub fn chat_template_kwargs(&self) -> Result<Vec<String>, super::TextModelError> {
        self.inner.chat_template_kwargs()
    }

    /// Renders and validates a JSON-valued chat for generation.
    pub fn prepare_chat(
        &mut self,
        request: crate::runtime::chat::ChatTemplateRequest,
    ) -> Result<super::PreparedChat, super::TextModelError> {
        self.inner.prepare_chat(request)
    }

    /// Reports fail-closed speculative support for this local session.
    pub fn speculative_capability(&self) -> eredu_core::SpeculativeCapability {
        self.inner.speculative_capability()
    }

    /// Generates one constrained semantic response.
    pub fn generate_prepared_chat<F>(
        &mut self,
        request: LocalPreparedChatGenerationRequest<'_, F>,
    ) -> Result<super::PreparedChatGenerationOutput, super::PreparedChatError<LocalBackendError>>
    where
        F: FnMut(eredu_core::generation::SemanticEvent),
    {
        self.inner
            .generate_prepared_chat(super::PreparedChatGenerationRequest {
                input: request.input.into_backend(),
                settings: request.settings,
                caller_stop_sequences: request.caller_stop_sequences,
                cancellation: request.cancellation,
                on_event: request.on_event,
            })
            .map_err(map_prepared_chat_error)
    }

    /// Generates one constrained response using opaque drafting resources.
    pub fn generate_prepared_chat_speculative<F>(
        &mut self,
        request: LocalPreparedChatSpeculativeGenerationRequest<'_, F>,
    ) -> Result<
        eredu_core::SpeculativeGenerationOutput,
        super::PreparedChatSpeculativeError<LocalBackendError>,
    >
    where
        F: FnMut(eredu_core::generation::SemanticEvent),
    {
        let drafting = request
            .drafting
            .inner
            .as_speculative_draft()
            .ok_or_else(|| {
                super::PreparedChatSpeculativeError::Semantic(
                    "the execution plan did not enable speculative drafting".into(),
                )
            })?;
        self.inner
            .generate_prepared_chat_speculative(super::PreparedChatSpeculativeGenerationRequest {
                input: request.input.into_backend(),
                drafting,
                settings: request.settings,
                options: request.options,
                caller_stop_sequences: request.caller_stop_sequences,
                cancellation: request.cancellation,
                on_event: request.on_event,
            })
            .map_err(map_prepared_chat_speculative_error)
    }

    /// Tokenizes and prepares a portable multimodal request.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn prepare_multimodal_input(
        &self,
        request: &eredu_core::MultimodalRequest,
    ) -> Result<LocalPrompt, super::MultimodalPreparationError<LocalBackendError>> {
        self.inner
            .prepare_multimodal_input(request)
            .map(|inner| LocalPrompt { inner })
            .map_err(map_multimodal_error)
    }

    /// Binds decoded media to a prepared chat and prepares its prompt.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn prepare_chat_multimodal_input(
        &self,
        prepared_chat: &super::PreparedChat,
        bindings: &[eredu_core::MediaBinding],
    ) -> Result<LocalPrompt, super::MultimodalPreparationError<LocalBackendError>> {
        self.inner
            .prepare_chat_multimodal_input(prepared_chat, bindings)
            .map(|inner| LocalPrompt { inner })
            .map_err(map_multimodal_error)
    }

    /// Resets this model's session state.
    pub fn reset(&mut self) -> Result<(), LocalBackendError> {
        self.inner
            .runtime
            .session_mut()
            .reset()
            .map_err(|error| LocalBackendError::new("session reset", error))
    }

    /// Waits for all work submitted by this model.
    pub fn synchronize(&self) -> Result<(), LocalBackendError> {
        self.inner
            .runtime
            .backend()
            .synchronize()
            .map_err(|error| LocalBackendError::new("synchronization", error))
    }

    /// Synchronizes the model and samples allocator counters.
    pub fn allocator_telemetry(&self) -> Result<crate::AllocatorTelemetry, LocalBackendError> {
        self.synchronize()?;
        allocator_telemetry()
    }

    /// Returns portable sparse expert-cache telemetry when available.
    pub fn expert_cache_telemetry(
        &self,
    ) -> Result<Option<crate::ExpertCacheTelemetry>, LocalBackendError> {
        self.inner
            .runtime
            .session()
            .parameter_bank_report()
            .map(|report| {
                report
                    .as_ref()
                    .map(eredu_backend_mlx::parameter_bank_telemetry)
            })
            .map_err(|error| LocalBackendError::new("expert-cache telemetry", error))
    }

    /// Returns portable weight-residency telemetry when available.
    pub fn residency_telemetry(
        &self,
    ) -> Result<Option<crate::ResidencyTelemetry>, LocalBackendError> {
        self.inner
            .runtime
            .session()
            .residency_report()
            .map_err(|error| LocalBackendError::new("residency telemetry", error))
            .map(|report| report.as_ref().map(eredu_backend_mlx::residency_telemetry))
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
fn map_multimodal_error(
    error: super::MultimodalPreparationError<eredu_backend_mlx::backend::error::Error>,
) -> super::MultimodalPreparationError<LocalBackendError> {
    match error {
        super::MultimodalPreparationError::Request(error) => {
            super::MultimodalPreparationError::Request(error)
        }
        super::MultimodalPreparationError::Text(error) => {
            super::MultimodalPreparationError::Text(error)
        }
        super::MultimodalPreparationError::Backend(error) => {
            super::MultimodalPreparationError::Backend(LocalBackendError::new(
                "multimodal preparation",
                error,
            ))
        }
    }
}

/// Facade-owned policy for loading a model on the selected local backend.
///
/// Distributed native device bindings are intentionally absent. Application
/// clients select portable placement and topology through an
/// [`crate::ExecutionPlan`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LocalLoadOptions {
    quantization: Option<crate::QuantizationRequest>,
    weight_residency: eredu_runtime::WeightResidency,
    required_session_capabilities: crate::SessionCapabilities,
}

impl LocalLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: crate::QuantizationRequest) -> Self {
        Self {
            quantization: Some(quantization),
            ..Self::default()
        }
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub const fn with_weight_residency(
        mut self,
        residency: eredu_runtime::WeightResidency,
    ) -> Self {
        self.weight_residency = residency;
        self
    }

    /// Requires capabilities from the exact inspected and realized session.
    pub const fn with_required_session_capabilities(
        mut self,
        capabilities: crate::SessionCapabilities,
    ) -> Self {
        self.required_session_capabilities = capabilities;
        self
    }

    /// Requested dense-weight transformation, if any.
    pub const fn quantization(&self) -> Option<crate::QuantizationRequest> {
        self.quantization
    }

    /// Selected immutable-weight residency policy.
    pub const fn weight_residency(&self) -> eredu_runtime::WeightResidency {
        self.weight_residency
    }

    /// Capabilities required from the realized session.
    pub const fn required_session_capabilities(&self) -> crate::SessionCapabilities {
        self.required_session_capabilities
    }

    fn into_backend(self) -> eredu_backend_mlx::backend::config::ModelLoadOptions {
        let options = match self.quantization {
            Some(quantization) => {
                eredu_backend_mlx::backend::config::ModelLoadOptions::with_quantization(
                    quantization,
                )
            }
            None => eredu_backend_mlx::backend::config::ModelLoadOptions::default(),
        };
        options
            .with_weight_residency(self.weight_residency)
            .with_required_session_capabilities(self.required_session_capabilities)
    }

    fn from_backend(
        options: eredu_backend_mlx::backend::config::ModelLoadOptions,
    ) -> Result<Self, crate::AutomaticPlanningError> {
        if options.has_parallel_execution() {
            return Err(crate::AutomaticPlanningError::Invalid(
                "selected-local inspection options cannot contain a native parallel context; use a portable execution plan"
                    .into(),
            ));
        }
        Ok(Self {
            quantization: options.quantization(),
            weight_residency: options.weight_residency(),
            required_session_capabilities: options.required_session_capabilities(),
        })
    }
}

impl Default for LocalLoadOptions {
    fn default() -> Self {
        Self {
            quantization: None,
            weight_residency: eredu_runtime::WeightResidency::fully_resident(),
            required_session_capabilities: crate::SessionCapabilities::default(),
        }
    }
}

/// Facade-owned options for selected-local-backend model inspection.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct LocalInspectionOptions {
    /// The exact facade loading policy that admission should validate.
    pub load: LocalLoadOptions,
}

impl LocalInspectionOptions {
    /// Derives inspection options from a portable execution plan.
    pub fn for_execution_plan(
        factory: &LocalBackendFactory,
        plan: &crate::ExecutionPlan,
    ) -> Result<Self, crate::AutomaticPlanningError> {
        let realization = eredu_core::realize_execution_plan_target(&factory.inner, plan)?;
        let (_, options) = realization.into_parts();
        Ok(Self {
            load: LocalLoadOptions::from_backend(options)?,
        })
    }
}

/// Inspects a model using facade-owned options and errors.
pub fn inspect_local_model(
    path: impl AsRef<Path>,
    options: LocalInspectionOptions,
) -> Result<crate::ModelInspectionReport, LocalBackendError> {
    eredu_backend_mlx::native::inspect_model(
        path,
        eredu_backend_mlx::native::MlxInspectionOptions {
            load: options.load.into_backend(),
        },
    )
    .map_err(|error| LocalBackendError::new("model inspection", error))
}

/// A facade-level device class for the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LocalDevice {
    /// Host CPU execution.
    Cpu,
    /// The selected native accelerator at this zero-based index.
    Accelerator(u32),
}

/// Selects accelerator zero when this build includes a native accelerator
/// family, or the CPU for CPU-only builds.
pub const fn default_local_device() -> LocalDevice {
    if compiled_accelerator_family().is_some() {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    }
}

/// Failure to map a facade device choice to the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum LocalDevicePlanError {
    /// This build contains the MLX adapter but no native accelerator family.
    #[error("no local accelerator family is compiled for this target")]
    AcceleratorNotCompiled,
}

/// Factory for local realtime model loading and execution.
///
/// [`Default`] uses [`default_local_device`]. Pass an explicit [`LocalDevice`]
/// to [`Self::new`] to override that policy.
#[derive(Debug, Clone, Copy)]
pub struct LocalRealtimeBackendFactory {
    device: LocalDevice,
}

impl LocalRealtimeBackendFactory {
    /// Selects the device used for realtime model execution.
    pub const fn new(device: LocalDevice) -> Self {
        Self { device }
    }

    /// Loads an architecture-prepared realtime model with the default policy.
    pub fn load(
        &self,
        preparation: crate::RealtimePreparationPlan,
    ) -> Result<LocalRealtimeModel, LocalBackendError> {
        self.load_with_options(preparation, LocalLoadOptions::default())
    }

    /// Loads an architecture-prepared realtime model with explicit policy.
    pub fn load_with_options(
        &self,
        preparation: crate::RealtimePreparationPlan,
        options: LocalLoadOptions,
    ) -> Result<LocalRealtimeModel, LocalBackendError> {
        let device = local_device_plan(self.device)
            .map_err(|error| LocalBackendError::new("device planning", error))?;
        let backend = eredu_backend_mlx::create_realtime_backend(&device)
            .map_err(|error| LocalBackendError::new("realtime backend creation", error))?;
        let model = backend
            .materialize_realtime_model(preparation, options.into_backend())
            .map_err(|error| LocalBackendError::new("realtime model loading", error))?;
        Ok(LocalRealtimeModel {
            inner: eredu_core::RealtimeModel::new(backend, model),
        })
    }
}

impl Default for LocalRealtimeBackendFactory {
    fn default() -> Self {
        Self::new(default_local_device())
    }
}

/// A selected local realtime backend and its loaded model.
///
/// Native models, tensors, streams, and backend traits remain private. Use a
/// [`LocalRealtimeScheduler`] to submit portable [`crate::RealtimeInputFrame`]
/// values and observe portable [`crate::RealtimeOutputFrame`] values.
pub struct LocalRealtimeModel {
    inner: eredu_core::RealtimeModel<eredu_backend_mlx::MlxRealtimeAdapter>,
}

impl LocalRealtimeModel {
    /// Name of the selected execution backend.
    pub fn backend_name(&self) -> &str {
        self.inner.backend().name()
    }

    /// Portable codec-token geometry for this model.
    pub fn speech_config(&self) -> crate::RealtimeSpeechConfig {
        self.inner.speech_config()
    }

    /// Fail-closed capabilities of the exact loaded realtime session.
    pub fn session_capabilities(&self) -> crate::SessionCapabilities {
        self.inner.session_capabilities()
    }
}

/// Request-local selected-backend state released from a realtime scheduler.
pub struct LocalRealtimeSession {
    inner: eredu_core::RealtimeSession<eredu_backend_mlx::MlxRealtimeAdapter>,
}

impl LocalRealtimeSession {
    /// Committed batch dimension, when at least one frame was accepted.
    pub fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size()
    }
}

/// One completed, portable realtime transition.
pub struct LocalRealtimeCompletedStep {
    work: crate::WorkId,
    output: crate::RealtimeOutputFrame,
}

impl LocalRealtimeCompletedStep {
    /// Scheduler-assigned work identity.
    pub const fn work(&self) -> crate::WorkId {
        self.work
    }

    /// Borrows the completed host token frame.
    pub const fn output(&self) -> &crate::RealtimeOutputFrame {
        &self.output
    }

    /// Consumes this completion into its work identity and host token frame.
    pub fn into_parts(self) -> (crate::WorkId, crate::RealtimeOutputFrame) {
        (self.work, self.output)
    }
}

/// Fair bounded realtime scheduler for a [`LocalRealtimeModel`].
///
/// This facade materializes portable host token frames before execution and
/// observes completed backend outputs before returning them to the caller.
pub struct LocalRealtimeScheduler {
    inner: eredu_core::RealtimeScheduler<eredu_backend_mlx::MlxRealtimeAdapter>,
}

impl LocalRealtimeScheduler {
    /// Creates an empty scheduler bound to one loaded local model.
    pub fn new(
        model: &LocalRealtimeModel,
        limits: crate::SchedulerLimits,
    ) -> Result<Self, eredu_core::RealtimeError<LocalBackendError>> {
        Ok(Self {
            inner: eredu_core::RealtimeScheduler::new(&model.inner, limits)
                .map_err(map_local_realtime_error)?,
        })
    }

    /// Registers a request with fresh selected-backend state.
    pub fn register_request(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        sampling: crate::RealtimeSampling,
    ) -> Result<(), eredu_core::RealtimeError<LocalBackendError>> {
        self.inner
            .register_request(&model.inner, request, sampling)
            .map_err(map_local_realtime_error)
    }

    /// Registers a previously released request session.
    pub fn register_request_with_session(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        session: LocalRealtimeSession,
    ) -> Result<(), eredu_core::RealtimeError<LocalBackendError>> {
        self.inner
            .register_request_with_session(&model.inner, request, session.inner)
            .map_err(map_local_realtime_error)
    }

    /// Materializes and enqueues one portable host token frame.
    pub fn enqueue(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        frame: crate::RealtimeInputFrame,
    ) -> Result<crate::WorkId, eredu_core::RealtimeError<LocalBackendError>> {
        self.enqueue_with_deadline(model, request, frame, None)
    }

    /// Materializes and enqueues one frame with an absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        frame: crate::RealtimeInputFrame,
        deadline: Option<Instant>,
    ) -> Result<crate::WorkId, eredu_core::RealtimeError<LocalBackendError>> {
        let input = model
            .inner
            .backend()
            .materialize_input(model.inner.model(), &frame)
            .map_err(|error| {
                eredu_core::RealtimeError::Backend(LocalBackendError::new(
                    "realtime input materialization",
                    error,
                ))
            })?;
        self.inner
            .enqueue_with_deadline(&model.inner, request, input, deadline)
            .map_err(map_local_realtime_error)
    }

    /// Atomically materializes and enqueues ordered portable host token frames.
    pub fn enqueue_batch(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        frames: Vec<crate::RealtimeInputFrame>,
    ) -> Result<Vec<crate::WorkId>, eredu_core::RealtimeError<LocalBackendError>> {
        let inputs = frames
            .iter()
            .map(|frame| {
                model
                    .inner
                    .backend()
                    .materialize_input(model.inner.model(), frame)
                    .map_err(|error| {
                        eredu_core::RealtimeError::Backend(LocalBackendError::new(
                            "realtime input materialization",
                            error,
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.inner
            .enqueue_batch(&model.inner, request, inputs)
            .map_err(map_local_realtime_error)
    }

    /// Advances one unbounded fair scheduling turn and observes completed frames.
    pub fn run_queued(
        &mut self,
        model: &mut LocalRealtimeModel,
    ) -> Result<Vec<LocalRealtimeCompletedStep>, eredu_core::RealtimeError<LocalBackendError>> {
        let completed = self
            .inner
            .run_queued(&mut model.inner)
            .map_err(map_local_realtime_error)?;
        observe_realtime_steps(model, completed)
    }

    /// Advances at most `max_frames` transitions and observes completed frames.
    pub fn run_bounded(
        &mut self,
        model: &mut LocalRealtimeModel,
        max_frames: usize,
    ) -> Result<Vec<LocalRealtimeCompletedStep>, eredu_core::RealtimeError<LocalBackendError>> {
        let completed = self
            .inner
            .run_bounded(&mut model.inner, max_frames)
            .map_err(map_local_realtime_error)?;
        observe_realtime_steps(model, completed)
    }

    /// Completes one request and drops its backend session.
    pub fn finish_request(
        &mut self,
        request: crate::RequestId,
    ) -> Result<(), eredu_core::RealtimeError<LocalBackendError>> {
        self.inner
            .finish_request(request)
            .map_err(map_local_realtime_error)
    }

    /// Cancels one request and discards queued frames.
    pub fn cancel_request(
        &mut self,
        request: crate::RequestId,
    ) -> Result<(), eredu_core::RealtimeError<LocalBackendError>> {
        self.inner
            .cancel_request(request)
            .map_err(map_local_realtime_error)
    }

    /// Releases an idle request for persistence or resumption.
    pub fn release_request(
        &mut self,
        request: crate::RequestId,
    ) -> Result<LocalRealtimeSession, eredu_core::RealtimeError<LocalBackendError>> {
        Ok(LocalRealtimeSession {
            inner: self
                .inner
                .release_request(request)
                .map_err(map_local_realtime_error)?,
        })
    }

    /// Removes a terminal identity for explicit reuse.
    pub fn forget_terminal_request(
        &mut self,
        request: crate::RequestId,
    ) -> Result<crate::RequestStatus, eredu_core::RealtimeError<LocalBackendError>> {
        self.inner
            .forget_terminal_request(request)
            .map_err(map_local_realtime_error)
    }

    /// Lifecycle state for a known request.
    pub fn request_status(&self, request: crate::RequestId) -> Option<crate::RequestStatus> {
        self.inner.request_status(request)
    }

    /// Queued frame count for one request.
    pub fn queued_for_request(&self, request: crate::RequestId) -> usize {
        self.inner.queued_for_request(request)
    }

    /// Replaces sampling controls for an idle active request.
    pub fn set_request_sampling(
        &mut self,
        model: &LocalRealtimeModel,
        request: crate::RequestId,
        sampling: crate::RealtimeSampling,
    ) -> Result<(), eredu_core::RealtimeError<LocalBackendError>> {
        self.inner
            .set_request_sampling(&model.inner, request, sampling)
            .map_err(map_local_realtime_error)
    }

    /// Generic occupancy and lifecycle telemetry.
    pub fn report(&self) -> crate::SchedulerReport {
        self.inner.report()
    }

    /// Configured bounds and observed backend capabilities.
    pub fn capabilities(&self) -> crate::SchedulerCapabilities {
        self.inner.capabilities()
    }
}

fn observe_realtime_steps(
    model: &LocalRealtimeModel,
    completed: Vec<eredu_core::RealtimeCompletedStep<eredu_backend_mlx::native::MlxRealtimeOutput>>,
) -> Result<Vec<LocalRealtimeCompletedStep>, eredu_core::RealtimeError<LocalBackendError>> {
    completed
        .into_iter()
        .map(|step| {
            let (work, output) = step.into_parts();
            let output = model
                .inner
                .backend()
                .observe_output(&output)
                .map_err(|error| {
                    eredu_core::RealtimeError::Backend(LocalBackendError::new(
                        "realtime output observation",
                        error,
                    ))
                })?;
            Ok(LocalRealtimeCompletedStep { work, output })
        })
        .collect()
}

fn map_local_realtime_error(
    error: eredu_core::RealtimeError<eredu_backend_mlx::backend::error::Error>,
) -> eredu_core::RealtimeError<LocalBackendError> {
    match error {
        eredu_core::RealtimeError::Backend(error) => {
            eredu_core::RealtimeError::Backend(LocalBackendError::new("realtime execution", error))
        }
        eredu_core::RealtimeError::Scheduler(error) => eredu_core::RealtimeError::Scheduler(error),
        eredu_core::RealtimeError::ModelMismatch { component } => {
            eredu_core::RealtimeError::ModelMismatch { component }
        }
        eredu_core::RealtimeError::BatchSize {
            request,
            expected,
            actual,
        } => eredu_core::RealtimeError::BatchSize {
            request,
            expected,
            actual,
        },
        eredu_core::RealtimeError::EmptyRunBound => eredu_core::RealtimeError::EmptyRunBound,
        eredu_core::RealtimeError::SamplingWhileQueued { request, queued } => {
            eredu_core::RealtimeError::SamplingWhileQueued { request, queued }
        }
        eredu_core::RealtimeError::Asynchronous { work, message } => {
            eredu_core::RealtimeError::Asynchronous { work, message }
        }
    }
}

/// Process-global configuration for the selected local runtime.
#[derive(Debug, Clone, Default)]
pub struct LocalRuntimeConfiguration {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    accelerator_library_path: Option<PathBuf>,
    allocator_cache_limit: Option<usize>,
}

impl LocalRuntimeConfiguration {
    /// Overrides the native accelerator kernel-library path.
    ///
    /// Embedded Apple applications use this when their bundled library cannot
    /// be found through the runtime's default search path.
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    pub fn with_accelerator_library(mut self, path: impl Into<PathBuf>) -> Self {
        self.accelerator_library_path = Some(path.into());
        self
    }

    /// Sets the selected runtime's process-global allocator-cache limit.
    pub const fn with_allocator_cache_limit(mut self, bytes: usize) -> Self {
        self.allocator_cache_limit = Some(bytes);
        self
    }
}

/// Applies process-global configuration before creating a local model session.
pub fn configure_local_runtime(
    configuration: &LocalRuntimeConfiguration,
) -> Result<(), LocalBackendError> {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    if let Some(path) = &configuration.accelerator_library_path {
        eredu_backend_mlx::set_accelerator_library_path(path)
            .map_err(|error| LocalBackendError::new("runtime configuration", error))?;
    }
    if let Some(bytes) = configuration.allocator_cache_limit {
        eredu_backend_mlx::set_allocator_cache_limit(bytes)
            .map_err(|error| LocalBackendError::new("allocator configuration", error))?;
    }
    Ok(())
}

/// Creates a portable plan device for the selected local backend.
pub fn local_device_plan(device: LocalDevice) -> Result<crate::DevicePlan, LocalDevicePlanError> {
    let device = match device {
        LocalDevice::Cpu => "cpu:0".to_owned(),
        LocalDevice::Accelerator(index) => {
            let family = compiled_accelerator_family()
                .ok_or(LocalDevicePlanError::AcceleratorNotCompiled)?;
            format!("{family}:{index}")
        }
    };
    Ok(crate::DevicePlan::new("mlx", device)
        .expect("the selected local backend and generated device identifier are non-empty"))
}

const fn compiled_accelerator_family() -> Option<&'static str> {
    if cfg!(feature = "cuda") {
        Some("cuda")
    } else if cfg!(all(feature = "metal", target_vendor = "apple")) {
        Some("metal")
    } else {
        None
    }
}

/// Resets the selected runtime's allocator high-water mark.
pub fn reset_local_allocator_peak() -> Result<(), LocalBackendError> {
    eredu_backend_mlx::reset_allocator_peak()
        .map_err(|error| LocalBackendError::new("allocator peak reset", error))?;
    Ok(())
}

/// Converts neutral speculative statistics into portable execution telemetry.
pub fn local_speculative_decoding_telemetry(
    stats: &eredu_core::speculative::SpeculativeStats,
) -> crate::SpeculativeDecodingTelemetry {
    eredu_backend_mlx::speculative_decoding_telemetry(stats)
}

fn allocator_telemetry() -> Result<crate::AllocatorTelemetry, LocalBackendError> {
    let memory = eredu_backend_mlx::allocator_memory()
        .map_err(|error| LocalBackendError::new("allocator telemetry", error))?;
    Ok(crate::AllocatorTelemetry {
        peak_bytes: memory.peak_bytes(),
        active_bytes: memory.active_bytes(),
        cache_bytes: memory.cached_bytes(),
    })
}

/// One measured phase of a selected-backend expert-cache benchmark.
#[derive(Debug, Clone, Copy)]
pub struct LocalExpertCacheBenchmarkSample {
    /// End-to-end phase latency after exact completion.
    pub elapsed: Duration,
    /// Route rows requested by the router.
    pub requested_routes: u64,
    /// Distinct logical experts requested after coalescing.
    pub distinct_experts: u64,
    /// Duplicate requests eliminated before materialization.
    pub coalesced_duplicates: u64,
    /// Temporary compact banks built.
    pub compact_banks: u64,
    /// Temporary compact-bank bytes built.
    pub compact_bank_bytes: u64,
    /// Host-cache hits.
    pub host_hits: u64,
    /// Host-cache misses.
    pub host_misses: u64,
    /// Host-cache evictions.
    pub host_evictions: u64,
    /// Device-cache hits.
    pub device_hits: u64,
    /// Device-cache misses.
    pub device_misses: u64,
    /// Device-cache evictions.
    pub device_evictions: u64,
    /// Host-resident expert count after the phase.
    pub host_resident_experts: usize,
    /// Host-resident expert bytes after the phase.
    pub host_resident_bytes: u64,
    /// Device-resident expert count after the phase.
    pub device_resident_experts: usize,
    /// Device-resident expert bytes after the phase.
    pub device_resident_bytes: u64,
}

/// Cold prefill, repeated prefill, and cached decode measurements.
#[derive(Debug, Clone, Copy)]
pub struct LocalExpertCacheBenchmark {
    /// Prefill after resetting the model session.
    pub cold_prefill: LocalExpertCacheBenchmarkSample,
    /// A second prefill after resetting only model state.
    pub repeated_prefill: LocalExpertCacheBenchmarkSample,
    /// One decode using the state produced by repeated prefill.
    pub cached_decode: LocalExpertCacheBenchmarkSample,
}

/// Failure while running the facade-owned expert-cache benchmark workflow.
#[derive(Debug, thiserror::Error)]
pub enum LocalExpertCacheBenchmarkError {
    /// The benchmark needs a non-empty prompt for prefill and cached decode.
    #[error("expert-cache benchmark requires at least one prompt token")]
    EmptyPrompt,
    /// The selected model does not expose sparse expert-cache telemetry.
    #[error("sparse expert-cache benchmark requires an expert-cache model")]
    ExpertCacheUnavailable,
    /// The local rank did not produce logits needed to complete a benchmark phase.
    #[error("expert-cache benchmark requires logits on the local rank")]
    LogitsUnavailable,
    /// The selected backend failed while preparing or executing the benchmark.
    #[error(transparent)]
    Backend(#[from] LocalBackendError),
}

#[derive(Clone, Copy)]
struct ExpertSnapshot {
    prefill: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    decode: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    host_resident_experts: usize,
    host_resident_bytes: u64,
    device_resident_experts: usize,
    device_resident_bytes: u64,
}

fn expert_snapshot(model: &LocalModel) -> Result<ExpertSnapshot, LocalExpertCacheBenchmarkError> {
    let report = model
        .inner
        .runtime
        .session()
        .parameter_bank_report()
        .map_err(|error| LocalBackendError::new("expert-cache telemetry", error))?
        .ok_or(LocalExpertCacheBenchmarkError::ExpertCacheUnavailable)?;
    Ok(ExpertSnapshot {
        prefill: *report.bulk(),
        decode: *report.incremental(),
        host_resident_experts: report.host_resident_entries(),
        host_resident_bytes: report.host_resident_bytes(),
        device_resident_experts: report.device_resident_entries(),
        device_resident_bytes: report.device_resident_bytes(),
    })
}

fn benchmark_sample(
    elapsed: Duration,
    before: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    after: eredu_backend_mlx::backend::runtime::residency::parameter_bank::BankPassStatistics,
    occupancy: ExpertSnapshot,
) -> LocalExpertCacheBenchmarkSample {
    LocalExpertCacheBenchmarkSample {
        elapsed,
        requested_routes: after
            .requested_selections()
            .saturating_sub(before.requested_selections()),
        distinct_experts: after
            .distinct_entries()
            .saturating_sub(before.distinct_entries()),
        coalesced_duplicates: after
            .coalesced_duplicates()
            .saturating_sub(before.coalesced_duplicates()),
        compact_banks: after.compact_banks().saturating_sub(before.compact_banks()),
        compact_bank_bytes: after
            .compact_bank_bytes()
            .saturating_sub(before.compact_bank_bytes()),
        host_hits: after.host().hits().saturating_sub(before.host().hits()),
        host_misses: after.host().misses().saturating_sub(before.host().misses()),
        host_evictions: after
            .host()
            .evictions()
            .saturating_sub(before.host().evictions()),
        device_hits: after.device().hits().saturating_sub(before.device().hits()),
        device_misses: after
            .device()
            .misses()
            .saturating_sub(before.device().misses()),
        device_evictions: after
            .device()
            .evictions()
            .saturating_sub(before.device().evictions()),
        host_resident_experts: occupancy.host_resident_experts,
        host_resident_bytes: occupancy.host_resident_bytes,
        device_resident_experts: occupancy.device_resident_experts,
        device_resident_bytes: occupancy.device_resident_bytes,
    }
}

fn validate_expert_cache_benchmark_prompt(
    token_ids: &[u32],
) -> Result<(), LocalExpertCacheBenchmarkError> {
    if token_ids.is_empty() {
        return Err(LocalExpertCacheBenchmarkError::EmptyPrompt);
    }
    Ok(())
}

/// Benchmarks selected-backend expert-cache reuse without exposing tensors or streams.
pub fn benchmark_local_expert_cache(
    model: &mut LocalModel,
    token_ids: &[u32],
) -> Result<LocalExpertCacheBenchmark, LocalExpertCacheBenchmarkError> {
    validate_expert_cache_benchmark_prompt(token_ids)?;
    let prompt = <SelectedBackend as eredu_core::TextGenerationBackend>::prepare_text_prompt(
        model.inner.runtime.backend(),
        token_ids.to_vec(),
    )
    .map_err(|error| LocalBackendError::new("expert-cache prompt preparation", error))?;

    let before_cold = expert_snapshot(model)?;
    model
        .inner
        .runtime
        .session_mut()
        .reset()
        .map_err(|error| LocalBackendError::new("expert-cache session reset", error))?;
    let started = std::time::Instant::now();
    let logits = model
        .inner
        .runtime
        .prefill(prompt.clone())
        .map_err(|error| LocalBackendError::new("expert-cache prefill submission", error))?
        .wait()
        .map_err(|error| LocalBackendError::new("expert-cache prefill completion", error))?
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_cold = expert_snapshot(model)?;
    let cold_prefill = benchmark_sample(
        started.elapsed(),
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    model
        .inner
        .runtime
        .session_mut()
        .reset()
        .map_err(|error| LocalBackendError::new("expert-cache session reset", error))?;
    let started = std::time::Instant::now();
    let logits = model
        .inner
        .runtime
        .prefill(prompt)
        .map_err(|error| LocalBackendError::new("expert-cache prefill submission", error))?
        .wait()
        .map_err(|error| LocalBackendError::new("expert-cache prefill completion", error))?
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_repeated = expert_snapshot(model)?;
    let repeated_prefill = benchmark_sample(
        started.elapsed(),
        after_cold.prefill,
        after_repeated.prefill,
        after_repeated,
    );

    let started = std::time::Instant::now();
    let output = {
        let (backend, session) = model.inner.runtime.parts_mut();
        session
            .submit_token_decode(backend, token_ids[token_ids.len() - 1])
            .map_err(|error| LocalBackendError::new("expert-cache decode submission", error))?
    }
    .wait()
    .map_err(|error| LocalBackendError::new("expert-cache decode completion", error))?;
    let logits = output
        .into_logits()
        .ok_or(LocalExpertCacheBenchmarkError::LogitsUnavailable)?;
    drop(logits);
    let after_decode = expert_snapshot(model)?;
    let cached_decode = benchmark_sample(
        started.elapsed(),
        after_repeated.decode,
        after_decode.decode,
        after_decode,
    );

    Ok(LocalExpertCacheBenchmark {
        cold_prefill,
        repeated_prefill,
        cached_decode,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        default_local_device, local_device_plan, validate_expert_cache_benchmark_prompt,
        LocalDevice, LocalDevicePlanError, LocalExpertCacheBenchmarkError,
        LocalRealtimeBackendFactory,
    };

    #[test]
    fn empty_benchmark_prompt_is_a_facade_input_error() {
        assert!(matches!(
            validate_expert_cache_benchmark_prompt(&[]),
            Err(LocalExpertCacheBenchmarkError::EmptyPrompt)
        ));
        validate_expert_cache_benchmark_prompt(&[1]).unwrap();
    }

    #[test]
    fn local_accelerator_plan_names_the_compiled_family() {
        let plan = local_device_plan(LocalDevice::Accelerator(3));
        if cfg!(feature = "cuda") {
            assert_eq!(plan.unwrap().device, "cuda:3");
        } else if cfg!(all(feature = "metal", target_vendor = "apple")) {
            assert_eq!(plan.unwrap().device, "metal:3");
        } else {
            assert_eq!(plan, Err(LocalDevicePlanError::AcceleratorNotCompiled));
        }
    }

    #[test]
    fn default_device_uses_an_available_accelerator_or_cpu() {
        let expected = if cfg!(any(
            feature = "cuda",
            all(feature = "metal", target_vendor = "apple")
        )) {
            LocalDevice::Accelerator(0)
        } else {
            LocalDevice::Cpu
        };
        let device = default_local_device();
        assert_eq!(device, expected);
        assert!(local_device_plan(device).is_ok());
        assert_eq!(LocalRealtimeBackendFactory::default().device, device);
    }
}
