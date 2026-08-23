//! High-level contract implemented once per execution backend.

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

use crate::{
    artifact::{
        inspect_artifact, plan_model_preparation, ArtifactError, ArtifactInspection,
        ModelConfigurationResolver, ModelPreparationPlan, PreparationPolicy,
    },
    capability::{
        CapabilityError, InputTokenCount, ModelCapabilities, RuntimeStateEstimate,
        StaticMemoryReport,
    },
    checkpoint::TensorDtype,
    generation::ResolvedGenerationConfig,
    media::TokenizedMultimodalRequest,
    topology::{ParallelAxis, ParallelTopology},
};

/// Stable, extensible description of an execution backend.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendDescriptor {
    /// Backend implementation name, such as `mlx` or `iree`.
    pub name: String,
    /// Backend implementation version.
    pub version: String,
}

/// Portable description of one backend-visible device.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    /// Backend-stable device identifier.
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Backend-specific device family without a closed core enum.
    pub family: String,
    /// Total memory when discoverable.
    pub memory_bytes: Option<u64>,
}

/// Fail-closed capabilities discovered from a backend and device.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Supports exact completion observation for submissions.
    pub exact_completion: bool,
    /// Supports device-to-device transfer for backend-owned values.
    pub transfers: bool,
    /// Supports collective execution for a complete session.
    pub collectives: bool,
    /// Supports backend-managed persistent decode caches.
    pub persistent_cache: bool,
}

/// Fail-closed distributed operations exposed by one selected session.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DistributedCapabilities {
    /// World-scoped collectives are available.
    pub world_collectives: bool,
    /// Active topology axes with subgroup collective support.
    pub collective_axes: Vec<ParallelAxis>,
    /// Point-to-point transfers are available.
    pub point_to_point: bool,
    /// Variable-count all-to-all exchange is available.
    pub variable_all_to_all: bool,
    /// Collective and transfer submissions have exact completion objects.
    pub exact_completion: bool,
}

/// Scope of a collective or point-to-point operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "axis", rename_all = "snake_case")]
pub enum CollectiveScope {
    /// All ranks in the selected distributed session.
    World,
    /// The topology subgroup containing this rank on one axis.
    Axis(ParallelAxis),
}

/// Portable shape and element type for a backend-owned received value.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueDescriptor {
    /// Row-major logical shape. An empty shape describes a scalar.
    pub shape: Vec<usize>,
    /// Logical element type.
    pub dtype: TensorDtype,
}

/// Portable identity of one selected distributed session.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct DistributedSessionDescriptor {
    /// Backend-neutral Cartesian topology.
    pub topology: ParallelTopology,
    /// World rank represented by this process-local session.
    pub rank: usize,
}

impl DistributedSessionDescriptor {
    /// Validates that `rank` belongs to `topology`.
    pub fn new(topology: ParallelTopology, rank: usize) -> Result<Self, BackendError> {
        let topology = ParallelTopology::new(
            topology.tensor,
            topology.pipeline,
            topology.expert,
            topology.data,
        )
        .map_err(|error| BackendError::Preparation {
            operation: "distributed session topology".into(),
            message: error.to_string(),
        })?;
        if rank >= topology.world_size() {
            return Err(BackendError::Preparation {
                operation: "distributed session topology".into(),
                message: format!(
                    "rank {rank} is outside topology world size {}",
                    topology.world_size()
                ),
            });
        }
        Ok(Self { topology, rank })
    }
}

impl<'de> Deserialize<'de> for DistributedSessionDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDescriptor {
            topology: ParallelTopology,
            rank: usize,
        }

        let raw = RawDescriptor::deserialize(deserializer)?;
        Self::new(raw.topology, raw.rank).map_err(serde::de::Error::custom)
    }
}

/// Structured backend failure that does not expose a runtime exception type.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum BackendError {
    /// A required capability is absent.
    #[error("backend {backend} does not support required capability {capability}")]
    Unsupported {
        /// Backend implementation name.
        backend: String,
        /// Required capability.
        capability: String,
    },
    /// Model preparation failed.
    #[error("backend model preparation failed during {operation}: {message}")]
    Preparation {
        /// Preparation operation.
        operation: String,
        /// Backend-provided context.
        message: String,
    },
    /// Session execution failed.
    #[error("backend session {session} failed during {operation}: {message}")]
    Execution {
        /// Stable session identifier.
        session: String,
        /// Operation being executed.
        operation: String,
        /// Backend-provided context.
        message: String,
    },
    /// Exact completion observation failed.
    #[error("backend completion observation failed: {message}")]
    Completion {
        /// Backend-provided context.
        message: String,
    },
}

/// Exact completion owned by one backend submission.
pub trait Completion {
    /// Error produced while observing the completion.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Nonblocking exact-completion observation.
    fn is_complete(&self) -> Result<bool, Self::Error>;

    /// Blocks on this exact completion only.
    fn wait(&self) -> Result<(), Self::Error>;
}

/// Output and exact completion returned by a backend submission.
#[derive(Debug)]
pub struct Submission<T, C> {
    /// Backend-owned output value.
    pub output: T,
    /// Completion retaining everything needed by the submitted work.
    pub completion: C,
}

impl<T, C> Submission<T, C>
where
    C: Completion,
{
    /// Waits for this exact submission and returns its output.
    pub fn wait(self) -> Result<T, C::Error> {
        self.completion.wait()?;
        Ok(self.output)
    }
}

/// Marker wrapper proving that a model was prepared by a backend.
#[derive(Debug)]
pub struct PreparedModel<M> {
    model: M,
}

impl<M> PreparedModel<M> {
    /// Wraps a backend-prepared model.
    pub const fn new(model: M) -> Self {
        Self { model }
    }
    /// Borrows the backend model.
    pub const fn get(&self) -> &M {
        &self.model
    }
    /// Mutably borrows the backend model.
    pub fn get_mut(&mut self) -> &mut M {
        &mut self.model
    }
    /// Consumes the marker.
    pub fn into_inner(self) -> M {
        self.model
    }
}

impl<M> std::ops::Deref for PreparedModel<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<M> std::ops::DerefMut for PreparedModel<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// One backend selected for an entire prepared model and all its sessions.
pub trait BackendProvider: Sized {
    /// Portable model preparation request.
    type ModelConfig;
    /// Opaque backend model/executable.
    type Model;
    /// Opaque backend session/cache state and execution implementation.
    type Session: BackendSession<Self>;
    /// Backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Backend identity.
    fn descriptor(&self) -> BackendDescriptor;
    /// Discovers devices and their fail-closed capabilities.
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error>;
    /// Loads, compiles, or materializes a model for this backend.
    fn prepare_model(
        &self,
        config: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error>;
    /// Consumes a prepared model into one backend-owned execution session.
    ///
    /// The executable and its mutable cache state have one owner after this
    /// call. This prevents callers from pairing session state with a different
    /// model or submitting the same mutable executable through two sessions.
    fn create_session(
        &self,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error>;
}

/// Artifact-loading extension for a selected whole-model backend.
///
/// Core owns checkpoint inspection and preparation planning. Implementations
/// translate the resulting neutral plan and their associated load options into
/// the backend's concrete [`BackendProvider::ModelConfig`]. Tensor materialization
/// remains entirely inside [`BackendProvider::prepare_model`].
pub trait ModelLoadingBackend: BackendProvider {
    /// Backend load policy exposed to a generic caller.
    type LoadOptions;

    /// Architecture registry selected by this backend adapter.
    type ConfigurationResolver: ModelConfigurationResolver;

    /// Returns the architecture-owned model configuration registry.
    fn configuration_resolver(&self) -> &Self::ConfigurationResolver;

    /// Resolves backend options into the policy used during neutral planning.
    fn preparation_policy(
        &self,
        options: &Self::LoadOptions,
    ) -> Result<PreparationPolicy, Self::Error>;

    /// Intersects normalized architecture requirements with backend support.
    ///
    /// Core deliberately does not infer architecture capabilities from a
    /// coarse model-family identity. Implementations must fail closed for
    /// requested routes that the exact normalized architecture or backend
    /// cannot realize.
    fn validate_preparation(
        &self,
        inspection: &ArtifactInspection,
        policy: PreparationPolicy,
    ) -> Result<(), Self::Error>;

    /// Binds a neutral preparation plan to backend-owned materialization input.
    fn model_config(
        &self,
        plan: ModelPreparationPlan,
        options: Self::LoadOptions,
    ) -> Result<Self::ModelConfig, Self::Error>;
}

/// Failure while inspecting, planning, or materializing a model artifact.
#[derive(Debug, thiserror::Error)]
pub enum ModelLoadError<E: std::error::Error + Send + Sync + 'static> {
    /// Portable artifact inspection or preparation planning failed.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// The selected backend failed policy resolution or materialization.
    #[error("selected backend failed to prepare the model: {0}")]
    Backend(#[source] E),
}

/// Inspects, plans, and prepares one artifact on the selected backend.
///
/// This is the sole generic artifact-loading entry point. The backend instance
/// already owns its device, execution queues, transfer queues, and optional
/// distributed communication state; none are passed separately to loading.
pub fn load_model<B: ModelLoadingBackend>(
    backend: &B,
    artifact: impl AsRef<Path>,
    options: B::LoadOptions,
) -> Result<PreparedModel<B::Model>, ModelLoadError<B::Error>> {
    let inspection = inspect_artifact(artifact, backend.configuration_resolver())?;
    prepare_inspected_model(backend, inspection, options)
}

/// Plans and prepares an artifact that the caller has already inspected.
///
/// This is the canonical lower-level entry point for facade loaders which
/// must derive tokenizer, chat, or other portable sidecar state from the same
/// inspection before transferring ownership to the selected backend.
pub fn prepare_inspected_model<B: ModelLoadingBackend>(
    backend: &B,
    inspection: ArtifactInspection,
    options: B::LoadOptions,
) -> Result<PreparedModel<B::Model>, ModelLoadError<B::Error>> {
    let policy = backend
        .preparation_policy(&options)
        .map_err(ModelLoadError::Backend)?;
    backend
        .validate_preparation(&inspection, policy)
        .map_err(ModelLoadError::Backend)?;
    let plan = plan_model_preparation(inspection, policy)?;
    let config = backend
        .model_config(plan, options)
        .map_err(ModelLoadError::Backend)?;
    backend
        .prepare_model(config)
        .map_err(ModelLoadError::Backend)
}

/// Prefill/decode interface for an already selected backend session.
///
/// The session owns its prepared executable and cache. The contract
/// intentionally models language-model submissions rather than primitive
/// tensor operations. Input, output, cache and completion stay opaque.
pub trait BackendSession<B: BackendProvider> {
    /// Backend-owned prefill input.
    type PrefillInput;
    /// Backend-owned decode input.
    type DecodeInput;
    /// Backend-owned logits/output.
    type Output;
    /// Exact completion type.
    type Completion: Completion<Error = B::Error>;

    /// Submits prompt prefill against this session.
    fn prefill(
        &mut self,
        backend: &B,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error>;

    /// Submits one or more cached decode positions against this session.
    fn decode(
        &mut self,
        backend: &B,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, B::Error>;
}

/// One submission produced by the selected backend session.
pub type SessionSubmission<B> = Submission<
    <<B as BackendProvider>::Session as BackendSession<B>>::Output,
    <<B as BackendProvider>::Session as BackendSession<B>>::Completion,
>;

/// A prepared model, its selected backend, and its backend-owned session.
///
/// This is the canonical client-side execution owner. Keeping the backend and
/// session together makes backend selection a whole-model decision and makes
/// it impossible to submit a session through a different backend instance.
/// Backend-owned executable, cache, tensor, and completion types remain
/// associated types and never enter the portable API.
pub struct ModelRuntime<B: BackendProvider> {
    backend: B,
    session: B::Session,
}

impl<B: BackendProvider> ModelRuntime<B> {
    /// Prepares `config` and creates its sole execution session.
    pub fn prepare(backend: B, config: B::ModelConfig) -> Result<Self, B::Error> {
        let model = backend.prepare_model(config)?;
        Self::from_prepared(backend, model)
    }

    /// Creates the sole execution session for an already prepared model.
    pub fn from_prepared(backend: B, model: PreparedModel<B::Model>) -> Result<Self, B::Error> {
        let session = backend.create_session(model)?;
        Ok(Self { backend, session })
    }

    /// Returns the selected backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the backend-owned session for optional backend capabilities.
    pub const fn session(&self) -> &B::Session {
        &self.session
    }

    /// Returns the backend-owned session for optional backend capabilities.
    pub fn session_mut(&mut self) -> &mut B::Session {
        &mut self.session
    }

    /// Borrows the selected backend and its mutable session together.
    pub fn parts_mut(&mut self) -> (&B, &mut B::Session) {
        (&self.backend, &mut self.session)
    }

    /// Submits prompt prefill through the selected backend and session.
    pub fn prefill(
        &mut self,
        input: <B::Session as BackendSession<B>>::PrefillInput,
    ) -> Result<SessionSubmission<B>, B::Error> {
        self.session.prefill(&self.backend, input)
    }

    /// Submits cached decode through the selected backend and session.
    pub fn decode(
        &mut self,
        input: <B::Session as BackendSession<B>>::DecodeInput,
    ) -> Result<SessionSubmission<B>, B::Error> {
        self.session.decode(&self.backend, input)
    }
}

impl<B: ModelLoadingBackend> ModelRuntime<B> {
    /// Loads an artifact and creates its sole session on `backend`.
    pub fn load(
        backend: B,
        artifact: impl AsRef<Path>,
        options: B::LoadOptions,
    ) -> Result<Self, ModelLoadError<B::Error>> {
        let model = load_model(&backend, artifact, options)?;
        Self::from_prepared(backend, model).map_err(ModelLoadError::Backend)
    }
}

/// Portable sampling inputs for one text-generation session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextGenerationConfig {
    sampling: ResolvedGenerationConfig,
    seed: u64,
}

impl TextGenerationConfig {
    /// Uses resolved checkpoint/request sampling with deterministic seed zero.
    pub const fn new(sampling: ResolvedGenerationConfig) -> Self {
        Self { sampling, seed: 0 }
    }

    /// Selects the deterministic root seed used by a stochastic backend.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the validated sampling configuration.
    pub const fn sampling(&self) -> ResolvedGenerationConfig {
        self.sampling
    }

    /// Returns the deterministic root seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
}

/// Backend-owned generated token that exposes only its portable token id.
pub trait TokenOutput: Clone {
    /// Error produced while observing the token value.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Waits only as required to read this token's canonical vocabulary id.
    fn token_id(&self) -> Result<u32, Self::Error>;
}

impl TokenOutput for u32 {
    type Error = std::convert::Infallible;

    fn token_id(&self) -> Result<u32, Self::Error> {
        Ok(*self)
    }
}

/// Portable vocabulary filter applied before backend-owned sampling.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenFilter {
    /// Every vocabulary token may be selected.
    All,
    /// One boolean per canonical vocabulary id; `true` permits selection.
    Allowed(Vec<bool>),
}

impl TokenFilter {
    /// Validates an explicit canonical-vocabulary allow mask.
    pub fn allowed(mask: Vec<bool>) -> Result<Self, TokenFilterError> {
        if mask.is_empty() {
            return Err(TokenFilterError::EmptyVocabulary);
        }
        if !mask.iter().any(|allowed| *allowed) {
            return Err(TokenFilterError::NoAllowedToken);
        }
        Ok(Self::Allowed(mask))
    }

    /// Returns the explicit allow mask, or `None` when all tokens are allowed.
    pub fn allowed_mask(&self) -> Option<&[bool]> {
        match self {
            Self::All => None,
            Self::Allowed(mask) => Some(mask),
        }
    }
}

/// Invalid portable token-filter construction.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum TokenFilterError {
    /// An explicit mask must describe a nonempty vocabulary.
    #[error("token filter vocabulary must not be empty")]
    EmptyVocabulary,
    /// Fail closed instead of asking a backend to sample an impossible row.
    #[error("token filter does not allow any vocabulary token")]
    NoAllowedToken,
}

/// Backend-independent logical controller for constrained token selection.
pub trait TokenFilterController {
    /// Constraint or grammar error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the filter for the current durable logical prefix.
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error>;

    /// Commits one backend-selected canonical vocabulary id.
    fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error>;

    /// Returns whether the committed logical prefix satisfies the constraint.
    fn is_complete(&mut self) -> Result<bool, Self::Error>;
}

/// Portable constraint controller that can evaluate speculative token histories.
///
/// Speculative backends use these queries for discardable draft branches. The
/// durable controller state must not change until [`TokenFilterController::commit_token`]
/// is called for a target-accepted token.
pub trait SpeculativeTokenFilterController: TokenFilterController + Clone {
    /// Returns the filter at `history` without committing its uncommitted suffix.
    ///
    /// `history` contains the controller's durable prefix followed by zero or
    /// more speculative tokens. Implementations must reject histories that do
    /// not begin with the durable prefix.
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error>;

    /// Returns whether `history` completes the constraint without committing it.
    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error>;
}

#[derive(Debug, Clone, Copy)]
struct UnconstrainedTokens;

impl TokenFilterController for UnconstrainedTokens {
    type Error = std::convert::Infallible;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

/// High-level text-generation extension implemented once per backend.
///
/// The contract deliberately combines model execution and sampling. Core does
/// not see logits or ask a backend to implement tensor primitives. The token,
/// sampling state, cache state, and exact completion remain backend-owned.
pub trait TextGenerationBackend: BackendProvider {
    /// Opaque prepared prompt, including any backend-owned multimodal values.
    type Prompt;
    /// Backend-owned generated token handle.
    type Token: TokenOutput<Error = Self::Error>;
    /// Backend-owned sampler and randomness state for one sequence.
    type TextGenerationState;
    /// Exact completion retaining model execution and token sampling.
    type TextCompletion: Completion<Error = Self::Error>;

    /// Creates backend sampling state for one sequence.
    fn start_text_generation(
        backend: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error>;

    /// Converts portable tokenizer ids into a backend-owned text prompt.
    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Self::Error>;

    /// Submits prompt prefill followed by sampling one token.
    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error>;

    /// Submits cached decode from the preceding token and samples its successor.
    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error>;
}

/// Failure from backend preprocessing or backend-requested text encoding.
#[derive(Debug, thiserror::Error)]
pub enum MultimodalPreparationFailure<B, T>
where
    B: std::error::Error + 'static,
    T: std::error::Error + 'static,
{
    /// The selected backend rejected or failed media preprocessing.
    #[error("backend multimodal preparation failed: {0}")]
    Backend(#[source] B),
    /// The facade tokenizer failed on backend-required framing text.
    #[error("multimodal framing text encoding failed: {0}")]
    Text(#[source] T),
}

/// Backend preparation of portable decoded media for one selected session.
///
/// Caller text is tokenized before this boundary. Some processors introduce
/// checkpoint-defined framing text or video timestamps, so the callback keeps
/// that work on the facade's tokenizer. The selected backend returns its
/// existing opaque prompt type; core never observes tensors or streams.
pub trait MultimodalPreparationBackend: TextGenerationBackend {
    /// Converts ordered token and decoded-media segments into a backend prompt.
    fn prepare_multimodal_input<E>(
        runtime: &ModelRuntime<Self>,
        request: &TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self::Prompt, MultimodalPreparationFailure<Self::Error, E>>
    where
        E: std::error::Error + Send + Sync + 'static;
}

/// Capability and resource observations for one selected text-model session.
///
/// Implementations inspect the opaque model and cache already owned by
/// [`ModelRuntime`]. The contract exposes only portable documents and the
/// backend's existing opaque prompt type; tensor, stream, allocator, and
/// executable types remain inside the adapter.
pub trait ModelCapabilityBackend: TextGenerationBackend {
    /// Reports validated model capabilities for the selected session.
    fn model_capabilities(
        runtime: &ModelRuntime<Self>,
    ) -> Result<ModelCapabilities, CapabilityError>;

    /// Counts text and backend-specific model positions in a prepared prompt.
    fn count_prepared_input(
        runtime: &ModelRuntime<Self>,
        input: &Self::Prompt,
    ) -> Result<InputTokenCount, CapabilityError>;

    /// Estimates persistent and transient request state for this session.
    fn estimate_runtime_state(
        runtime: &ModelRuntime<Self>,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError>;

    /// Reports static model storage and current backend memory observations.
    fn static_memory(runtime: &ModelRuntime<Self>) -> Result<StaticMemoryReport, CapabilityError>;
}

enum TextGenerationStep<P, T> {
    Prefill(P),
    Decode(T),
}

/// Failure from either backend execution or portable constraint control.
#[derive(Debug, thiserror::Error)]
pub enum ControlledTextGenerationError<B, C>
where
    B: std::error::Error + 'static,
    C: std::error::Error + 'static,
{
    /// Backend preparation, execution, sampling, or completion failed.
    #[error("backend text generation failed: {0}")]
    Backend(#[source] B),
    /// Portable constraint filtering or commitment failed.
    #[error("text generation constraint failed: {0}")]
    Controller(#[source] C),
}

/// One constraint-committed token and its backend-owned output handle.
#[derive(Debug, Clone)]
pub struct ControlledToken<T> {
    output: T,
    token_id: u32,
}

impl<T> ControlledToken<T> {
    /// Returns the committed canonical vocabulary id.
    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    /// Borrows the backend-owned token handle.
    pub const fn output(&self) -> &T {
        &self.output
    }

    /// Consumes the committed token into its backend-owned handle.
    pub fn into_output(self) -> T {
        self.output
    }
}

/// Backend-generic generation driven by a portable token-filter controller.
pub struct ControlledTextGeneration<'a, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    inner: TextGenerationMachine<'a, B, C>,
}

struct TextGenerationMachine<'a, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    runtime: &'a mut ModelRuntime<B>,
    backend_state: B::TextGenerationState,
    controller: C,
    step: Option<TextGenerationStep<B::Prompt, B::Token>>,
    completions: Vec<B::TextCompletion>,
    remaining_tokens: Option<usize>,
}

type ControlledGenerationResult<B, C> = Result<
    <B as TextGenerationBackend>::Token,
    ControlledTextGenerationError<
        <B as BackendProvider>::Error,
        <C as TokenFilterController>::Error,
    >,
>;

impl<'a, B, C> ControlledTextGeneration<'a, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    /// Starts controlled generation from portable prompt token ids.
    pub fn new(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let prompt = B::prepare_text_prompt(runtime.backend(), prompt_token_ids)
            .map_err(ControlledTextGenerationError::Backend)?;
        Self::from_prompt(runtime, prompt, config, controller)
    }

    /// Starts controlled generation from an opaque backend-prepared prompt.
    pub fn from_prompt(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        TextGenerationMachine::new(runtime, prompt, config, controller).map(|inner| Self { inner })
    }

    /// Mutably borrows the canonical constraint state.
    pub fn controller_mut(&mut self) -> &mut C {
        &mut self.inner.controller
    }
}

impl<'a, B, C> TextGenerationMachine<'a, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    fn new(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let backend_state = B::start_text_generation(runtime.backend(), config)
            .map_err(ControlledTextGenerationError::Backend)?;
        Ok(Self {
            runtime,
            backend_state,
            controller,
            step: Some(TextGenerationStep::Prefill(prompt)),
            completions: Vec::new(),
            remaining_tokens: config.sampling().max_new_tokens,
        })
    }

    fn retain_completion(&mut self, completion: B::TextCompletion) -> Result<(), B::Error> {
        let existing = std::mem::take(&mut self.completions);
        let mut retained = Vec::with_capacity(existing.len() + 1);
        for pending in existing {
            match pending.is_complete() {
                Ok(true) => {}
                Ok(false) => retained.push(pending),
                Err(error) => {
                    let _ = pending.wait();
                    for retained_completion in retained.drain(..) {
                        let _ = retained_completion.wait();
                    }
                    let _ = completion.wait();
                    return Err(error);
                }
            }
        }
        retained.push(completion);
        self.completions = retained;
        Ok(())
    }

    fn next_output(&mut self) -> Option<ControlledGenerationResult<B, C>> {
        if self.remaining_tokens == Some(0) {
            self.step = None;
            return None;
        }
        let step = self.step.take()?;
        let filter = match self.controller.current_filter() {
            Ok(filter) => filter,
            Err(error) => return Some(Err(ControlledTextGenerationError::Controller(error))),
        };
        let submission = match step {
            TextGenerationStep::Prefill(prompt) => {
                B::submit_text_prefill(self.runtime, prompt, &filter, &mut self.backend_state)
            }
            TextGenerationStep::Decode(token) => {
                B::submit_text_decode(self.runtime, token, &filter, &mut self.backend_state)
            }
        };
        let submission = match submission {
            Ok(submission) => submission,
            Err(error) => return Some(Err(ControlledTextGenerationError::Backend(error))),
        };
        let token = submission.output;
        if let Err(error) = self.retain_completion(submission.completion) {
            return Some(Err(ControlledTextGenerationError::Backend(error)));
        }
        self.step = Some(TextGenerationStep::Decode(token.clone()));
        if let Some(remaining_tokens) = &mut self.remaining_tokens {
            *remaining_tokens -= 1;
        }
        Some(Ok(token))
    }
}

impl<B, C> Iterator for ControlledTextGeneration<'_, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    type Item =
        Result<ControlledToken<B::Token>, ControlledTextGenerationError<B::Error, C::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        let token = match self.inner.next_output()? {
            Ok(token) => token,
            Err(error) => return Some(Err(error)),
        };
        let token_id = match token.token_id() {
            Ok(token_id) => token_id,
            Err(error) => {
                self.inner.step = None;
                return Some(Err(ControlledTextGenerationError::Backend(error)));
            }
        };
        if let Err(error) = self.inner.controller.commit_token(token_id) {
            self.inner.step = None;
            return Some(Err(ControlledTextGenerationError::Controller(error)));
        }
        Some(Ok(ControlledToken {
            output: token,
            token_id,
        }))
    }
}

impl<B, C> Drop for TextGenerationMachine<'_, B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    fn drop(&mut self) {
        for completion in self.completions.drain(..) {
            let _ = completion.wait();
        }
    }
}

/// Backend-generic asynchronous token-generation iterator.
///
/// Every yielded token handle may be fed into the following decode before its
/// id is read on the host. Exact completions are retained until finished, and
/// dropping the iterator waits for all still-retained submissions.
pub struct TextGeneration<'a, B: TextGenerationBackend> {
    inner: TextGenerationMachine<'a, B, UnconstrainedTokens>,
}

impl<'a, B: TextGenerationBackend> TextGeneration<'a, B> {
    /// Starts generation from portable prompt token ids.
    pub fn new(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
    ) -> Result<Self, B::Error> {
        let prompt = B::prepare_text_prompt(runtime.backend(), prompt_token_ids)?;
        Self::from_prompt(runtime, prompt, config)
    }

    /// Starts generation from an opaque backend-prepared prompt.
    pub fn from_prompt(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
    ) -> Result<Self, B::Error> {
        TextGenerationMachine::new(runtime, prompt, config, UnconstrainedTokens)
            .map(|inner| Self { inner })
            .map_err(unreachable_unconstrained_error)
    }
}

fn unreachable_unconstrained_error<B>(
    error: ControlledTextGenerationError<B, std::convert::Infallible>,
) -> B
where
    B: std::error::Error + 'static,
{
    match error {
        ControlledTextGenerationError::Backend(error) => error,
        ControlledTextGenerationError::Controller(error) => match error {},
    }
}

impl<B: TextGenerationBackend> Iterator for TextGeneration<'_, B> {
    type Item = Result<B::Token, B::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next_output()
            .map(|result| result.map_err(unreachable_unconstrained_error))
    }
}

/// Optional high-level transfer and collective capability of a selected session.
///
/// This contract deliberately operates on an opaque backend value. It models
/// the few communication submissions needed by model execution without making
/// core define a tensor algebra or exposing native groups, streams, or events.
/// Every operation is scoped to the session selected for the complete model.
pub trait DistributedSession {
    /// Backend-owned tensor or buffer value.
    type Value;
    /// Exact completion retaining the submitted communication resources.
    type Completion: Completion<Error = Self::Error>;
    /// Structured backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable topology and rank identity.
    fn descriptor(&self) -> DistributedSessionDescriptor;
    /// Fail-closed communication support.
    fn capabilities(&self) -> DistributedCapabilities;

    /// Submits a sum reduction over `scope`.
    fn all_reduce_sum(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a leading-rank-axis gather over `scope`.
    fn all_gather(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a variable-count all-to-all exchange over `scope`.
    fn all_to_all_v(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a point-to-point send to a rank within `scope`.
    fn send(
        &self,
        scope: CollectiveScope,
        peer: usize,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Submits a point-to-point receive from a rank within `scope`.
    fn receive(
        &self,
        scope: CollectiveScope,
        peer: usize,
        value: &ValueDescriptor,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error>;

    /// Synchronously gathers portable scheduler metadata across the world.
    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error>;
}

/// Backend extension exposing communication attached to a model session.
pub trait DistributedBackend: BackendProvider {
    /// Selected distributed session implementation.
    type DistributedSession: DistributedSession<Error = Self::Error>;

    /// Returns communication for a distributed model session.
    fn distributed_session(session: &Self::Session) -> Option<&Self::DistributedSession>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{convert::Infallible, io::Write};

    #[derive(Debug, Clone, Copy)]
    struct Done;
    impl Completion for Done {
        type Error = Infallible;
        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    struct Mock;
    impl BackendProvider for Mock {
        type ModelConfig = u32;
        type Model = u32;
        type Session = MockSession;
        type Error = Infallible;
        fn descriptor(&self) -> BackendDescriptor {
            BackendDescriptor {
                name: "mock".into(),
                version: "1".into(),
            }
        }
        fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error> {
            Ok(vec![])
        }
        fn prepare_model(&self, config: u32) -> Result<PreparedModel<u32>, Self::Error> {
            Ok(PreparedModel::new(config))
        }
        fn create_session(&self, model: PreparedModel<u32>) -> Result<MockSession, Self::Error> {
            Ok(MockSession {
                model: model.into_inner(),
                tokens: vec![],
                distributed: None,
            })
        }
    }

    struct LoadingMock;
    struct LoadingMockSession;

    struct LoadingConfigurationResolver;

    impl ModelConfigurationResolver for LoadingConfigurationResolver {
        fn resolve_safetensors(
            &self,
            json: &serde_json::Value,
        ) -> Result<crate::ModelConfiguration, ArtifactError> {
            Ok(crate::ModelConfiguration {
                declared_model_type: "llama".into(),
                effective_model_type: "llama".into(),
                kind: crate::ModelKind::Llama,
                json: Some(json.clone()),
            })
        }

        fn resolve_gguf(
            &self,
            architecture: &str,
            _checkpoint: &eredu_gguf::Checkpoint,
        ) -> Result<crate::ModelConfiguration, ArtifactError> {
            if architecture != "llama" {
                return Err(ArtifactError::UnsupportedGgufArchitecture(
                    architecture.into(),
                ));
            }
            Ok(crate::ModelConfiguration {
                declared_model_type: architecture.into(),
                effective_model_type: architecture.into(),
                kind: crate::ModelKind::Llama,
                json: None,
            })
        }
    }

    static LOADING_CONFIGURATION_RESOLVER: LoadingConfigurationResolver =
        LoadingConfigurationResolver;

    impl BackendProvider for LoadingMock {
        type ModelConfig = (ModelPreparationPlan, u32);
        type Model = u32;
        type Session = LoadingMockSession;
        type Error = std::convert::Infallible;

        fn descriptor(&self) -> BackendDescriptor {
            BackendDescriptor {
                name: "loading-mock".into(),
                version: "1".into(),
            }
        }

        fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error> {
            Ok(Vec::new())
        }

        fn prepare_model(
            &self,
            (plan, model): Self::ModelConfig,
        ) -> Result<PreparedModel<Self::Model>, Self::Error> {
            assert_eq!(
                plan.inspection().configuration().kind,
                crate::ModelKind::Llama
            );
            Ok(PreparedModel::new(model))
        }

        fn create_session(
            &self,
            _: PreparedModel<Self::Model>,
        ) -> Result<Self::Session, Self::Error> {
            Ok(LoadingMockSession)
        }
    }

    impl BackendSession<LoadingMock> for LoadingMockSession {
        type PrefillInput = ();
        type DecodeInput = ();
        type Output = ();
        type Completion = LoadingDone;

        fn prefill(
            &mut self,
            _: &LoadingMock,
            _: (),
        ) -> Result<Submission<(), LoadingDone>, std::convert::Infallible> {
            Ok(Submission {
                output: (),
                completion: LoadingDone,
            })
        }

        fn decode(
            &mut self,
            _: &LoadingMock,
            _: (),
        ) -> Result<Submission<(), LoadingDone>, std::convert::Infallible> {
            Ok(Submission {
                output: (),
                completion: LoadingDone,
            })
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct LoadingDone;

    impl Completion for LoadingDone {
        type Error = std::convert::Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl ModelLoadingBackend for LoadingMock {
        type LoadOptions = u32;
        type ConfigurationResolver = LoadingConfigurationResolver;

        fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
            &LOADING_CONFIGURATION_RESOLVER
        }

        fn preparation_policy(
            &self,
            _: &Self::LoadOptions,
        ) -> Result<PreparationPolicy, Self::Error> {
            Ok(PreparationPolicy::default())
        }

        fn validate_preparation(
            &self,
            _: &ArtifactInspection,
            _: PreparationPolicy,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn model_config(
            &self,
            plan: ModelPreparationPlan,
            options: Self::LoadOptions,
        ) -> Result<Self::ModelConfig, Self::Error> {
            Ok((plan, options))
        }
    }

    fn write_loading_fixture(root: &Path) {
        std::fs::write(root.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        let header = br#"{"token_embd.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0; 4]).unwrap();
    }
    struct MockSession {
        model: u32,
        tokens: Vec<u32>,
        distributed: Option<MockDistributed>,
    }
    impl BackendSession<Mock> for MockSession {
        type PrefillInput = Vec<u32>;
        type DecodeInput = u32;
        type Output = u32;
        type Completion = Done;
        fn prefill(
            &mut self,
            _: &Mock,
            input: Vec<u32>,
        ) -> Result<Submission<u32, Done>, Infallible> {
            self.tokens.extend(input);
            Ok(Submission {
                output: self.tokens.len() as u32 + self.model,
                completion: Done,
            })
        }
        fn decode(&mut self, _: &Mock, input: u32) -> Result<Submission<u32, Done>, Infallible> {
            self.tokens.push(input);
            Ok(Submission {
                output: self.tokens.len() as u32 + self.model,
                completion: Done,
            })
        }
    }

    impl TextGenerationBackend for Mock {
        type Prompt = Vec<u32>;
        type Token = u32;
        type TextGenerationState = (u32, u64);
        type TextCompletion = Done;

        fn start_text_generation(
            _: &Self,
            config: TextGenerationConfig,
        ) -> Result<Self::TextGenerationState, Self::Error> {
            Ok((config.sampling().top_k as u32, config.seed()))
        }

        fn prepare_text_prompt(
            _: &Self,
            prompt_token_ids: Vec<u32>,
        ) -> Result<Self::Prompt, Self::Error> {
            Ok(prompt_token_ids)
        }

        fn submit_text_prefill(
            runtime: &mut ModelRuntime<Self>,
            prompt: Self::Prompt,
            filter: &TokenFilter,
            state: &mut Self::TextGenerationState,
        ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
            let submission = runtime.prefill(prompt)?;
            Ok(Submission {
                output: apply_mock_filter(submission.output + state.0 + state.1 as u32, filter),
                completion: submission.completion,
            })
        }

        fn submit_text_decode(
            runtime: &mut ModelRuntime<Self>,
            token: Self::Token,
            filter: &TokenFilter,
            _: &mut Self::TextGenerationState,
        ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
            let submission = runtime.decode(token)?;
            Ok(Submission {
                output: apply_mock_filter(submission.output, filter),
                completion: submission.completion,
            })
        }
    }

    impl MultimodalPreparationBackend for Mock {
        fn prepare_multimodal_input<E>(
            _: &ModelRuntime<Self>,
            request: &TokenizedMultimodalRequest,
            _: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        ) -> Result<Self::Prompt, MultimodalPreparationFailure<Self::Error, E>>
        where
            E: std::error::Error + Send + Sync + 'static,
        {
            let mut prompt = Vec::new();
            for segment in request.segments() {
                match segment {
                    crate::TokenizedMultimodalSegment::TokenIds(ids) => {
                        prompt.extend_from_slice(ids);
                    }
                    crate::TokenizedMultimodalSegment::Media(crate::Media::Image(_)) => {
                        prompt.push(1_001);
                    }
                    crate::TokenizedMultimodalSegment::Media(crate::Media::Video(_)) => {
                        prompt.push(1_002);
                    }
                    crate::TokenizedMultimodalSegment::Media(crate::Media::Audio(_)) => {
                        prompt.push(1_003);
                    }
                }
            }
            Ok(prompt)
        }
    }

    impl ModelCapabilityBackend for Mock {
        fn model_capabilities(
            _: &ModelRuntime<Self>,
        ) -> Result<ModelCapabilities, CapabilityError> {
            Ok(ModelCapabilities {
                model_type: "mock".into(),
                native_max_context: crate::Observed::exact(64, "mock configuration"),
                effective_max_context: crate::Observed::exact(64, "mock configuration"),
                state_strategy: crate::CacheStateStrategy::FullKv,
                modalities: crate::InputModalities::TEXT,
                estimation: crate::EstimationCompleteness::Complete,
            })
        }

        fn count_prepared_input(
            _: &ModelRuntime<Self>,
            input: &Self::Prompt,
        ) -> Result<InputTokenCount, CapabilityError> {
            Ok(InputTokenCount::text(input.len() as u64))
        }

        fn estimate_runtime_state(
            _: &ModelRuntime<Self>,
            input: InputTokenCount,
            max_output_tokens: u64,
            batch_size: u64,
        ) -> Result<RuntimeStateEstimate, CapabilityError> {
            crate::estimate_runtime_state(
                &crate::StateLayout {
                    fixed_scalars_per_batch: 0,
                    growing: vec![crate::GrowingState {
                        layers: 1,
                        scalars_per_position: 2,
                        window: None,
                    }],
                    hidden_size: 1,
                    allocation_granularity: 1,
                    completeness: crate::EstimationCompleteness::Complete,
                },
                input,
                max_output_tokens,
                batch_size,
                std::num::NonZeroU8::new(4).unwrap(),
            )
        }

        fn static_memory(
            runtime: &ModelRuntime<Self>,
        ) -> Result<StaticMemoryReport, CapabilityError> {
            let unavailable = || crate::Observed::unavailable("mock does not expose this counter");
            Ok(StaticMemoryReport {
                logical_parameter_bytes: crate::Observed::exact(
                    u64::from(runtime.session().model),
                    "mock model",
                ),
                current_host_resident_bytes: unavailable(),
                current_device_resident_bytes: unavailable(),
                planned_disk_backed_bytes: unavailable(),
                backend_active_allocation_bytes: unavailable(),
                backend_allocator_cache_bytes: unavailable(),
                physical_semantics: crate::PhysicalMemorySemantics::Unknown,
                currently_mapped_shards: unavailable(),
            })
        }
    }

    fn apply_mock_filter(candidate: u32, filter: &TokenFilter) -> u32 {
        let Some(allowed) = filter.allowed_mask() else {
            return candidate;
        };
        allowed
            .get(candidate as usize)
            .copied()
            .unwrap_or(false)
            .then_some(candidate)
            .or_else(|| {
                allowed
                    .iter()
                    .position(|allowed| *allowed)
                    .map(|token| token as u32)
            })
            .expect("validated token filters allow at least one token")
    }

    #[test]
    fn generic_loader_inspects_plans_and_prepares_on_the_selected_backend() {
        let root = tempfile::tempdir().unwrap();
        write_loading_fixture(root.path());
        let prepared = load_model(&LoadingMock, root.path(), 41).unwrap();
        assert_eq!(*prepared, 41);

        let runtime = ModelRuntime::load(LoadingMock, root.path(), 7).unwrap();
        assert_eq!(runtime.backend().descriptor().name, "loading-mock");

        let missing = root.path().join("missing");
        assert!(matches!(
            load_model(&LoadingMock, &missing, 1),
            Err(ModelLoadError::Artifact(ArtifactError::MissingArtifact(path)))
                if path == missing
        ));
    }

    struct FixedController {
        tokens: Vec<u32>,
        committed: usize,
    }

    impl TokenFilterController for FixedController {
        type Error = Infallible;

        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            let mut allowed = vec![false; 64];
            allowed[self.tokens[self.committed] as usize] = true;
            Ok(TokenFilter::allowed(allowed).unwrap())
        }

        fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error> {
            assert_eq!(token_id, self.tokens[self.committed]);
            self.committed += 1;
            Ok(())
        }

        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(self.committed == self.tokens.len())
        }
    }

    #[test]
    fn mock_prefill_and_multiple_decode_steps() {
        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let prefill = runtime.prefill(vec![1, 2]).unwrap();
        assert_eq!(prefill.output, 12);
        assert!(prefill.completion.is_complete().unwrap());
        assert_eq!(runtime.decode(3).unwrap().output, 13);
        assert_eq!(runtime.decode(4).unwrap().output, 14);
    }

    #[test]
    fn portable_text_generation_prefills_and_decodes_without_tensor_types() {
        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let sampling = crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        let mut generation = TextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(sampling).with_seed(3),
        )
        .unwrap();
        assert_eq!(generation.next().unwrap().unwrap().token_id().unwrap(), 55);
        assert_eq!(generation.next().unwrap().unwrap().token_id().unwrap(), 13);
        assert_eq!(generation.next().unwrap().unwrap().token_id().unwrap(), 14);
        assert!(generation.next().is_none());
    }

    #[test]
    fn portable_media_preparation_feeds_the_existing_generation_contract() {
        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let request = crate::MultimodalRequest::new(vec![
            crate::MultimodalSegment::TokenIds(vec![7, 8]),
            crate::MultimodalSegment::Media(crate::Media::Image(
                crate::RgbImage::new(vec![5, 6, 7], 1, 1).unwrap(),
            )),
            crate::MultimodalSegment::TokenIds(vec![9]),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!("request is already tokenized"))
        .unwrap();
        let prompt = Mock::prepare_multimodal_input(&runtime, &request, &mut |_| {
            Ok::<_, Infallible>(Vec::new())
        })
        .unwrap();
        assert_eq!(prompt, vec![7, 8, 1_001, 9]);

        let sampling = crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let mut generation =
            TextGeneration::from_prompt(&mut runtime, prompt, TextGenerationConfig::new(sampling))
                .unwrap();
        assert!(generation.next().unwrap().is_ok());
        assert!(generation.next().unwrap().is_ok());
        assert!(generation.next().is_none());
    }

    #[test]
    fn model_capability_extension_observes_the_selected_mock_session() {
        let runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let capabilities = Mock::model_capabilities(&runtime).unwrap();
        assert_eq!(capabilities.model_type, "mock");
        let input = Mock::count_prepared_input(&runtime, &vec![1, 2, 3]).unwrap();
        assert_eq!(input.model_positions, 3);
        let state = Mock::estimate_runtime_state(&runtime, input, 2, 1).unwrap();
        assert_eq!(state.requested_state_bytes, 5 * 2 * 4);
        assert_eq!(
            Mock::static_memory(&runtime)
                .unwrap()
                .logical_parameter_bytes
                .value(),
            Some(&10)
        );
    }

    #[test]
    fn controlled_generation_applies_portable_filters_and_commits_tokens() {
        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let sampling = crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let controller = FixedController {
            tokens: vec![7, 8],
            committed: 0,
        };
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(sampling),
            controller,
        )
        .unwrap();
        assert_eq!(generation.next().unwrap().unwrap().token_id(), 7);
        assert_eq!(generation.next().unwrap().unwrap().token_id(), 8);
        assert!(generation.controller_mut().is_complete().unwrap());
        assert!(generation.next().is_none());
    }

    #[derive(Debug, Clone, Copy)]
    struct MockDistributed {
        descriptor: DistributedSessionDescriptor,
    }

    impl DistributedSession for MockDistributed {
        type Value = Vec<u32>;
        type Completion = Done;
        type Error = Infallible;

        fn descriptor(&self) -> DistributedSessionDescriptor {
            self.descriptor.clone()
        }

        fn capabilities(&self) -> DistributedCapabilities {
            DistributedCapabilities {
                world_collectives: true,
                collective_axes: vec![ParallelAxis::Tensor],
                point_to_point: true,
                variable_all_to_all: true,
                exact_completion: true,
            }
        }

        fn all_reduce_sum(
            &self,
            _: CollectiveScope,
            input: &Vec<u32>,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: input.iter().map(|value| value * 2).collect(),
                completion: Done,
            })
        }

        fn all_gather(
            &self,
            _: CollectiveScope,
            input: &Vec<u32>,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            let mut output = input.clone();
            output.extend(input);
            Ok(Submission {
                output,
                completion: Done,
            })
        }

        fn all_to_all_v(
            &self,
            _: CollectiveScope,
            input: &Vec<u32>,
            _: &[usize],
            _: &[usize],
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: input.clone(),
                completion: Done,
            })
        }

        fn send(
            &self,
            _: CollectiveScope,
            _: usize,
            input: &Vec<u32>,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: input.clone(),
                completion: Done,
            })
        }

        fn receive(
            &self,
            _: CollectiveScope,
            peer: usize,
            value: &ValueDescriptor,
        ) -> Result<Submission<Vec<u32>, Done>, Infallible> {
            Ok(Submission {
                output: vec![peer as u32; value.shape.iter().product()],
                completion: Done,
            })
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Infallible> {
            let mut output = local.to_vec();
            output.extend_from_slice(local);
            Ok(output)
        }
    }

    impl DistributedBackend for Mock {
        type DistributedSession = MockDistributed;

        fn distributed_session(session: &MockSession) -> Option<&Self::DistributedSession> {
            session.distributed.as_ref()
        }
    }

    #[test]
    fn mock_distributed_session_owns_collective_and_transfer_lifecycle() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let session = MockDistributed {
            descriptor: DistributedSessionDescriptor::new(topology, 0).unwrap(),
        };
        let capabilities = session.capabilities();
        assert!(capabilities.exact_completion);
        assert_eq!(capabilities.collective_axes, vec![ParallelAxis::Tensor]);
        assert_eq!(
            session
                .all_reduce_sum(CollectiveScope::Axis(ParallelAxis::Tensor), &vec![2, 3])
                .unwrap()
                .wait()
                .unwrap(),
            vec![4, 6]
        );
        assert_eq!(
            session
                .receive(
                    CollectiveScope::World,
                    1,
                    &ValueDescriptor {
                        shape: vec![2],
                        dtype: TensorDtype::U32,
                    },
                )
                .unwrap()
                .wait()
                .unwrap(),
            vec![1, 1]
        );
        assert_eq!(session.all_gather_words(&[7]).unwrap(), vec![7, 7]);

        let model_session = MockSession {
            model: 0,
            tokens: Vec::new(),
            distributed: Some(session),
        };
        assert_eq!(
            Mock::distributed_session(&model_session)
                .unwrap()
                .descriptor(),
            session.descriptor()
        );
    }

    #[test]
    fn distributed_descriptors_round_trip_and_reject_invalid_ranks() {
        let descriptor =
            DistributedSessionDescriptor::new(ParallelTopology::new(2, 3, 1, 1).unwrap(), 4)
                .unwrap();
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert_eq!(
            serde_json::from_str::<DistributedSessionDescriptor>(&encoded).unwrap(),
            descriptor
        );
        let scope = CollectiveScope::Axis(ParallelAxis::Pipeline);
        assert_eq!(
            serde_json::from_str::<CollectiveScope>(&serde_json::to_string(&scope).unwrap())
                .unwrap(),
            scope
        );
        assert!(DistributedSessionDescriptor::new(descriptor.topology, 6).is_err());
        assert!(serde_json::from_str::<DistributedSessionDescriptor>(
            r#"{"topology":{"tensor":2,"pipeline":3,"expert":1,"data":1},"rank":6}"#
        )
        .is_err());
    }
}
