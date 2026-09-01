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
    generation::{GenerationError, ResolvedGenerationConfig},
    media::TokenizedMultimodalRequest,
    observation::{InspectedOutput, ObservationRequest, ObservationSet},
};

/// Stable, extensible description of an execution backend.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendDescriptor {
    /// Backend implementation name, such as `example-backend`.
    name: String,
    /// Backend implementation version.
    version: String,
}

impl BackendDescriptor {
    /// Creates a backend identity without freezing future descriptor fields.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }

    /// Returns the backend implementation name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the backend implementation version.
    pub fn version(&self) -> &str {
        &self.version
    }
}

/// Portable description of one backend-visible device.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    /// Backend-stable device identifier.
    id: String,
    /// Human-readable device name.
    name: String,
    /// Backend-specific device family without a closed core enum.
    family: String,
    /// Total memory when discoverable.
    memory_bytes: Option<u64>,
}

impl DeviceDescriptor {
    /// Creates a backend-stable device description.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        family: impl Into<String>,
        memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            family: family.into(),
            memory_bytes,
        }
    }

    /// Returns the backend-stable device identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns the human-readable device name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the backend-defined device family.
    pub fn family(&self) -> &str {
        &self.family
    }
    /// Returns total device memory when known.
    pub const fn memory_bytes(&self) -> Option<u64> {
        self.memory_bytes
    }
}

/// Fail-closed capabilities discovered from a backend and device.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Supports exact completion observation for submissions.
    exact_completion: bool,
    /// Supports device-to-device transfer for backend-owned values.
    transfers: bool,
    /// Supports collective execution for a complete session.
    collectives: bool,
}

impl DeviceCapabilities {
    /// Creates an exact fail-closed device mechanism report.
    pub const fn new(exact_completion: bool, transfers: bool, collectives: bool) -> Self {
        Self {
            exact_completion,
            transfers,
            collectives,
        }
    }

    /// Returns whether exact completion observation is available.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
    /// Returns whether device transfers are available.
    pub const fn transfers(&self) -> bool {
        self.transfers
    }
    /// Returns whether collective execution is available.
    pub const fn collectives(&self) -> bool {
        self.collectives
    }
}

/// Fail-closed capabilities of one exact prepared model session.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionCapabilities {
    /// Supports backend-managed persistent decode caches.
    persistent_cache: bool,
    /// Supports explicit host observation of completed session outputs.
    output_observation: bool,
    /// Supports named activation inspection for instrumented session passes.
    activation_inspection: bool,
}

impl SessionCapabilities {
    /// Creates an exact fail-closed session mechanism report.
    pub const fn new(
        persistent_cache: bool,
        output_observation: bool,
        activation_inspection: bool,
    ) -> Self {
        Self {
            persistent_cache,
            output_observation,
            activation_inspection,
        }
    }

    /// Returns whether persistent cache storage is available.
    pub const fn persistent_cache(self) -> bool {
        self.persistent_cache
    }
    /// Returns whether completed outputs may be observed on the host.
    pub const fn output_observation(self) -> bool {
        self.output_observation
    }
    /// Returns whether named activation inspection is available.
    pub const fn activation_inspection(self) -> bool {
        self.activation_inspection
    }

    /// Returns a report with persistent cache storage configured.
    pub const fn with_persistent_cache(mut self, supported: bool) -> Self {
        self.persistent_cache = supported;
        self
    }
    /// Returns a report with output observation configured.
    pub const fn with_output_observation(mut self, supported: bool) -> Self {
        self.output_observation = supported;
        self
    }
    /// Returns a report with activation inspection configured.
    pub const fn with_activation_inspection(mut self, supported: bool) -> Self {
        self.activation_inspection = supported;
        self
    }
    /// Validates fail-closed requirements against an exact available report.
    pub fn validate(&self, available: &Self) -> Result<(), SessionCapabilityError> {
        for (required, supported, capability) in [
            (
                self.persistent_cache,
                available.persistent_cache,
                "persistent_cache",
            ),
            (
                self.output_observation,
                available.output_observation,
                "output_observation",
            ),
            (
                self.activation_inspection,
                available.activation_inspection,
                "activation_inspection",
            ),
        ] {
            if required && !supported {
                return Err(SessionCapabilityError { capability });
            }
        }
        Ok(())
    }
}

/// One unavailable exact-session requirement.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("prepared session does not support required capability {capability}")]
pub struct SessionCapabilityError {
    capability: &'static str,
}

impl SessionCapabilityError {
    /// Returns the stable capability name.
    pub const fn capability(self) -> &'static str {
        self.capability
    }
}

/// Fail-closed distributed operations exposed by one selected session.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DistributedCapabilities {
    world_collectives: bool,
    collective_groups: Vec<CollectiveGroupId>,
    point_to_point: bool,
    variable_all_to_all: bool,
    exact_completion: bool,
}

impl DistributedCapabilities {
    /// Creates an exact mechanism capability report.
    pub fn new(
        world_collectives: bool,
        collective_groups: impl IntoIterator<Item = CollectiveGroupId>,
        point_to_point: bool,
        variable_all_to_all: bool,
        exact_completion: bool,
    ) -> Self {
        Self {
            world_collectives,
            collective_groups: collective_groups.into_iter().collect(),
            point_to_point,
            variable_all_to_all,
            exact_completion,
        }
    }

    /// Returns whether world-scoped collectives are available.
    pub const fn world_collectives(&self) -> bool {
        self.world_collectives
    }
    /// Returns opaque groups supporting collectives.
    pub fn collective_groups(&self) -> &[CollectiveGroupId] {
        &self.collective_groups
    }
    /// Returns whether point-to-point transfers are available.
    pub const fn point_to_point(&self) -> bool {
        self.point_to_point
    }
    /// Returns whether variable-count all-to-all is available.
    pub const fn variable_all_to_all(&self) -> bool {
        self.variable_all_to_all
    }
    /// Returns whether submissions have exact completion objects.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
}

/// Opaque stable identity of a selected collective group.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CollectiveGroupId(u32);

impl CollectiveGroupId {
    /// Creates an opaque group identity selected by architecture/runtime composition.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
    /// Returns the stable numeric representation for serialization and backend maps.
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Ordered membership for one opaque collective group containing this rank.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CollectiveGroupDescriptor {
    id: CollectiveGroupId,
    members: Vec<usize>,
    local_rank: usize,
}

impl CollectiveGroupDescriptor {
    /// Validates ordered group membership and the process-local rank.
    pub fn new(
        id: CollectiveGroupId,
        members: Vec<usize>,
        local_rank: usize,
    ) -> Result<Self, BackendError> {
        if members.is_empty() || local_rank >= members.len() {
            return Err(BackendError::Preparation {
                operation: "collective group realization".into(),
                message: "collective membership must be non-empty and contain local rank".into(),
            });
        }
        let mut unique = std::collections::BTreeSet::new();
        if !members.iter().all(|rank| unique.insert(*rank)) {
            return Err(BackendError::Preparation {
                operation: "collective group realization".into(),
                message: "collective membership contains duplicate world ranks".into(),
            });
        }
        Ok(Self {
            id,
            members,
            local_rank,
        })
    }

    /// Returns the opaque group identity.
    pub const fn id(&self) -> CollectiveGroupId {
        self.id
    }
    /// Returns ordered world-rank membership.
    pub fn members(&self) -> &[usize] {
        &self.members
    }
    /// Returns this process's rank within the ordered group.
    pub const fn local_rank(&self) -> usize {
        self.local_rank
    }
}

impl<'de> Deserialize<'de> for CollectiveGroupDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            id: CollectiveGroupId,
            members: Vec<usize>,
            local_rank: usize,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.id, raw.members, raw.local_rank).map_err(serde::de::Error::custom)
    }
}

/// Scope of a collective or point-to-point operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "group", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CollectiveScope {
    /// All ranks in the selected distributed session.
    World,
    /// One opaque selected collective group containing this rank.
    Group(CollectiveGroupId),
}

/// Portable shape and element type for a backend-owned received value.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct ValueDescriptor {
    /// Row-major logical shape. An empty shape describes a scalar.
    shape: Vec<usize>,
    /// Logical element type.
    dtype: TensorDtype,
}

impl ValueDescriptor {
    /// Validates a portable value shape and element type.
    pub fn new(shape: Vec<usize>, dtype: TensorDtype) -> Result<Self, BackendError> {
        if shape.contains(&0) {
            return Err(BackendError::Preparation {
                operation: "distributed value descriptor".into(),
                message: "non-scalar distributed values require positive dimensions".into(),
            });
        }
        Ok(Self { shape, dtype })
    }

    /// Returns the row-major logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the logical element type.
    pub const fn dtype(&self) -> &TensorDtype {
        &self.dtype
    }
}

impl<'de> Deserialize<'de> for ValueDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDescriptor {
            shape: Vec<usize>,
            dtype: TensorDtype,
        }

        let raw = RawDescriptor::deserialize(deserializer)?;
        Self::new(raw.shape, raw.dtype).map_err(serde::de::Error::custom)
    }
}

/// Portable identity of one selected distributed session.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct DistributedSessionDescriptor {
    world_size: usize,
    rank: usize,
    groups: Vec<CollectiveGroupDescriptor>,
}

impl DistributedSessionDescriptor {
    /// Validates a mechanism-only distributed session realization.
    pub fn new(
        world_size: usize,
        rank: usize,
        groups: Vec<CollectiveGroupDescriptor>,
    ) -> Result<Self, BackendError> {
        if world_size == 0 || rank >= world_size {
            return Err(BackendError::Preparation {
                operation: "distributed session realization".into(),
                message: format!("rank {rank} is outside world size {world_size}"),
            });
        }
        let mut ids = std::collections::BTreeSet::new();
        for group in &groups {
            if !ids.insert(group.id())
                || group.members().iter().any(|member| *member >= world_size)
                || group.members()[group.local_rank()] != rank
            {
                return Err(BackendError::Preparation {
                    operation: "distributed session realization".into(),
                    message: "collective groups must have unique IDs, in-range members, and the declared local world rank".into(),
                });
            }
        }
        Ok(Self {
            world_size,
            rank,
            groups,
        })
    }

    /// Returns the total process count.
    pub const fn world_size(&self) -> usize {
        self.world_size
    }
    /// Returns this process's world rank.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Returns ordered opaque group realizations.
    pub fn groups(&self) -> &[CollectiveGroupDescriptor] {
        &self.groups
    }
}

impl<'de> Deserialize<'de> for DistributedSessionDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDescriptor {
            world_size: usize,
            rank: usize,
            groups: Vec<CollectiveGroupDescriptor>,
        }

        let raw = RawDescriptor::deserialize(deserializer)?;
        Self::new(raw.world_size, raw.rank, raw.groups).map_err(serde::de::Error::custom)
    }
}

/// Structured backend failure that does not expose a runtime exception type.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
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
    capabilities: SessionCapabilities,
}

impl<M> PreparedModel<M> {
    /// Wraps a backend-prepared model.
    pub const fn new(model: M, capabilities: SessionCapabilities) -> Self {
        Self {
            model,
            capabilities,
        }
    }
    /// Borrows the backend model.
    pub const fn get(&self) -> &M {
        &self.model
    }
    /// Mutably borrows the backend model.
    pub fn get_mut(&mut self) -> &mut M {
        &mut self.model
    }
    /// Returns the session capabilities admitted before materialization.
    pub const fn capabilities(&self) -> SessionCapabilities {
        self.capabilities
    }
    /// Consumes the marker.
    pub fn into_inner(self) -> M {
        self.model
    }
    /// Consumes the marker into the backend model and admitted capabilities.
    pub fn into_parts(self) -> (M, SessionCapabilities) {
        (self.model, self.capabilities)
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
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error>;
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

    /// Constructs an internal backend error for an admission/realization mismatch.
    ///
    /// Backends whose prepared and realized reports cannot differ may leave
    /// this unreachable default in place.
    fn session_capability_mismatch(
        &self,
        admitted: SessionCapabilities,
        realized: SessionCapabilities,
    ) -> Self::Error {
        panic!("backend realized session capabilities {realized:?} after admitting {admitted:?}")
    }
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

    /// Exact pre-materialization realization selected from the inspected
    /// artifact, caller request, and backend mechanisms.
    type SelectedPreparation;

    /// Architecture registry selected by this backend adapter.
    type ConfigurationResolver: ModelConfigurationResolver;

    /// Returns the architecture-owned model configuration registry.
    fn configuration_resolver(&self) -> &Self::ConfigurationResolver;

    /// Resolves backend options into the policy used during neutral planning.
    fn preparation_policy(
        &self,
        options: &Self::LoadOptions,
    ) -> Result<PreparationPolicy, Self::Error>;

    /// Intersects normalized architecture requirements and the caller request
    /// with backend support, returning the sole construction-policy handoff.
    ///
    /// Core deliberately does not infer architecture capabilities from a
    /// coarse model-family identity. Implementations must fail closed for
    /// requested routes that the exact normalized architecture or backend
    /// cannot realize.
    fn select_preparation(
        &self,
        inspection: &ArtifactInspection<
            <Self::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
        >,
        options: &Self::LoadOptions,
        policy: PreparationPolicy,
    ) -> Result<Self::SelectedPreparation, Self::Error>;

    /// Derives exact session capabilities from header/configuration state only.
    fn session_capabilities(
        &self,
        inspection: &ArtifactInspection<
            <Self::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
        >,
        policy: PreparationPolicy,
    ) -> Result<SessionCapabilities, Self::Error>;

    /// Binds a neutral preparation plan and its authoritative selected
    /// realization to backend-owned materialization input.
    fn model_config(
        &self,
        plan: ModelPreparationPlan<
            <Self::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
        >,
        selected: Self::SelectedPreparation,
    ) -> Result<Self::ModelConfig, Self::Error>;
}

/// Failure while inspecting, planning, or materializing a model artifact.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelLoadError<E: std::error::Error + Send + Sync + 'static> {
    /// Portable artifact inspection or preparation planning failed.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// The selected backend failed policy resolution or materialization.
    #[error("selected backend failed to prepare the model: {0}")]
    Backend(#[source] E),
    /// The inspected architecture/load-policy/topology route lacks a requirement.
    #[error(transparent)]
    SessionCapability(#[from] SessionCapabilityError),
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
    inspection: ArtifactInspection<
        <B::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
    >,
    options: B::LoadOptions,
) -> Result<PreparedModel<B::Model>, ModelLoadError<B::Error>> {
    let policy = backend
        .preparation_policy(&options)
        .map_err(ModelLoadError::Backend)?;
    let selected = backend
        .select_preparation(&inspection, &options, policy)
        .map_err(ModelLoadError::Backend)?;
    let capabilities = backend
        .session_capabilities(&inspection, policy)
        .map_err(ModelLoadError::Backend)?;
    policy
        .validate_session_capabilities(&capabilities)
        .map_err(ModelLoadError::SessionCapability)?;
    let plan = plan_model_preparation(inspection, policy, capabilities)?;
    let config = backend
        .model_config(plan, selected)
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

    /// Reports capabilities of this exact realized session.
    fn capabilities(&self) -> SessionCapabilities;

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

    /// Materializes an already completed opaque output into portable records.
    ///
    /// Calling this method is an explicit synchronization and host-transfer
    /// boundary. Ordinary inference does not invoke it.
    fn observe_output(
        &self,
        backend: &B,
        output: &Self::Output,
    ) -> Result<ObservationSet, B::Error>;
}

/// Optional named-activation inspection for a selected backend session.
///
/// This is a general diagnostics and observability capability. Implementations
/// execute the requested operation to completion and materialize only selected
/// observation points. It is intentionally separate from ordinary asynchronous
/// submission so production inference pays no instrumentation cost.
pub trait InspectableBackendSession<B: BackendProvider>: BackendSession<B> {
    /// Executes and inspects one prompt prefill operation.
    fn inspect_prefill(
        &mut self,
        backend: &B,
        input: Self::PrefillInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, B::Error>;

    /// Executes and inspects one cached decode operation.
    fn inspect_decode(
        &mut self,
        backend: &B,
        input: Self::DecodeInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, B::Error>;
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
        let admitted = model.capabilities();
        let session = backend.create_session(model)?;
        let realized = session.capabilities();
        if admitted != realized {
            return Err(backend.session_capability_mismatch(admitted, realized));
        }
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

    /// Reports capabilities of the exact prepared model session.
    pub fn capabilities(&self) -> SessionCapabilities {
        self.session.capabilities()
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

    /// Materializes portable observations from an already completed output.
    pub fn observe_output(
        &self,
        output: &<B::Session as BackendSession<B>>::Output,
    ) -> Result<ObservationSet, B::Error> {
        self.session.observe_output(&self.backend, output)
    }
}

impl<B> ModelRuntime<B>
where
    B: BackendProvider,
    B::Session: InspectableBackendSession<B>,
{
    /// Executes a completed, explicitly instrumented prefill operation.
    pub fn inspect_prefill(
        &mut self,
        input: <B::Session as BackendSession<B>>::PrefillInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<<B::Session as BackendSession<B>>::Output>, B::Error> {
        self.session.inspect_prefill(&self.backend, input, request)
    }

    /// Executes a completed, explicitly instrumented decode operation.
    pub fn inspect_decode(
        &mut self,
        input: <B::Session as BackendSession<B>>::DecodeInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<<B::Session as BackendSession<B>>::Output>, B::Error> {
        self.session.inspect_decode(&self.backend, input, request)
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
    strategy: TextSamplingStrategy,
}

/// Backend-neutral token-sampling strategy for one text-generation session.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum TextSamplingStrategy {
    /// Apply the resolved top-k, top-p, min-p, and penalty controls.
    #[default]
    Standard,
    /// Adapt the surprise cutoff toward `tau` bits at rate `eta`.
    MirostatV2 {
        /// Target surprise in bits.
        tau: f32,
        /// Adaptation rate.
        eta: f32,
    },
}

impl TextGenerationConfig {
    /// Uses resolved checkpoint/request sampling with deterministic seed zero.
    pub const fn new(sampling: ResolvedGenerationConfig) -> Self {
        Self {
            sampling,
            seed: 0,
            strategy: TextSamplingStrategy::Standard,
        }
    }

    /// Selects the deterministic root seed used by a stochastic backend.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Selects adaptive Mirostat V2 sampling.
    pub fn with_mirostat_v2(mut self, tau: f32, eta: f32) -> Result<Self, GenerationError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(GenerationError::InvalidMirostatTau(tau));
        }
        if !eta.is_finite() || eta <= 0.0 {
            return Err(GenerationError::InvalidMirostatEta(eta));
        }
        self.strategy = TextSamplingStrategy::MirostatV2 { tau, eta };
        Ok(self)
    }

    /// Returns the validated sampling configuration.
    pub const fn sampling(&self) -> ResolvedGenerationConfig {
        self.sampling
    }

    /// Returns the deterministic root seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the selected backend-neutral sampling strategy.
    pub const fn strategy(&self) -> TextSamplingStrategy {
        self.strategy
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

    #[test]
    fn text_generation_config_validates_portable_mirostat_strategy() {
        let sampling = crate::generation::resolve_generation_config(
            None,
            crate::generation::GenerationConfigOverrides {
                temperature: Some(0.8),
                ..crate::generation::GenerationConfigOverrides::default()
            },
        )
        .unwrap();
        let config = TextGenerationConfig::new(sampling)
            .with_seed(7)
            .with_mirostat_v2(5.0, 0.1)
            .unwrap();
        assert_eq!(config.seed(), 7);
        assert_eq!(
            config.strategy(),
            TextSamplingStrategy::MirostatV2 { tau: 5.0, eta: 0.1 }
        );
        assert!(matches!(
            TextGenerationConfig::new(sampling).with_mirostat_v2(0.0, 0.1),
            Err(GenerationError::InvalidMirostatTau(0.0))
        ));
        assert!(matches!(
            TextGenerationConfig::new(sampling).with_mirostat_v2(5.0, f32::NAN),
            Err(GenerationError::InvalidMirostatEta(value)) if value.is_nan()
        ));
    }

    #[derive(Debug, Clone)]
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
            BackendDescriptor::new("mock", "1")
        }
        fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
            Ok(vec![])
        }
        fn prepare_model(&self, config: u32) -> Result<PreparedModel<u32>, Self::Error> {
            Ok(PreparedModel::new(config, SessionCapabilities::default()))
        }
        fn create_session(&self, model: PreparedModel<u32>) -> Result<MockSession, Self::Error> {
            Ok(MockSession {
                model: model.into_inner(),
                tokens: vec![],
                distributed: None,
            })
        }
    }

    #[derive(Default)]
    struct LoadingMock {
        materializations: std::sync::atomic::AtomicUsize,
    }
    struct LoadingMockSession;

    struct LoadingConfigurationResolver;

    impl ModelConfigurationResolver for LoadingConfigurationResolver {
        type ArtifactPlan = ();

        fn resolve_safetensors(
            &self,
            json: &serde_json::Value,
        ) -> Result<crate::ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
            Ok(crate::ResolvedModelConfiguration::new(
                crate::ModelConfiguration::new(
                    "llama",
                    "llama",
                    "llama",
                    crate::LoadingProtocol::Model,
                    Some(json.clone()),
                )?,
                (),
            ))
        }

        fn resolve_gguf(
            &self,
            architecture: &str,
            _checkpoint: &eredu_gguf::Checkpoint,
        ) -> Result<crate::ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
            if architecture != "llama" {
                return Err(ArtifactError::UnsupportedGgufArchitecture(
                    architecture.into(),
                ));
            }
            Ok(crate::ResolvedModelConfiguration::new(
                crate::ModelConfiguration::new(
                    architecture,
                    architecture,
                    "llama",
                    crate::LoadingProtocol::Model,
                    None,
                )?,
                (),
            ))
        }

        fn gguf_companion_requirements(
            &self,
            _architecture: &str,
            _checkpoint: &eredu_gguf::Checkpoint,
        ) -> Result<Vec<crate::GgufCompanionRequirement>, ArtifactError> {
            Ok(Vec::new())
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
            BackendDescriptor::new("loading-mock", "1")
        }

        fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
            Ok(Vec::new())
        }

        fn prepare_model(
            &self,
            (plan, model): Self::ModelConfig,
        ) -> Result<PreparedModel<Self::Model>, Self::Error> {
            self.materializations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            assert_eq!(plan.inspection().configuration().family(), "llama");
            Ok(PreparedModel::new(
                model,
                plan.admitted_session_capabilities(),
            ))
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

        fn capabilities(&self) -> SessionCapabilities {
            SessionCapabilities::default()
        }

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

        fn observe_output(
            &self,
            _: &LoadingMock,
            _: &(),
        ) -> Result<ObservationSet, std::convert::Infallible> {
            Ok(ObservationSet::new())
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
        type SelectedPreparation = u32;
        type ConfigurationResolver = LoadingConfigurationResolver;

        fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
            &LOADING_CONFIGURATION_RESOLVER
        }

        fn preparation_policy(
            &self,
            options: &Self::LoadOptions,
        ) -> Result<PreparationPolicy, Self::Error> {
            Ok(
                PreparationPolicy::default().with_required_session_capabilities(
                    SessionCapabilities::default().with_activation_inspection(*options == 99),
                ),
            )
        }

        fn select_preparation(
            &self,
            _: &ArtifactInspection,
            options: &Self::LoadOptions,
            _: PreparationPolicy,
        ) -> Result<Self::SelectedPreparation, Self::Error> {
            Ok(*options)
        }

        fn session_capabilities(
            &self,
            _: &ArtifactInspection,
            _: PreparationPolicy,
        ) -> Result<SessionCapabilities, Self::Error> {
            Ok(SessionCapabilities::default())
        }

        fn model_config(
            &self,
            plan: ModelPreparationPlan,
            selected: Self::SelectedPreparation,
        ) -> Result<Self::ModelConfig, Self::Error> {
            Ok((plan, selected))
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
        fn capabilities(&self) -> SessionCapabilities {
            SessionCapabilities::default()
        }
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

        fn observe_output(&self, _: &Mock, output: &u32) -> Result<ObservationSet, Infallible> {
            let mut observations = ObservationSet::new();
            observations
                .insert(
                    "mock.output",
                    crate::ObservationValue::Unsigned(u64::from(*output)),
                )
                .unwrap();
            Ok(observations)
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
                effective_model_type: "mock".into(),
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
                &crate::StateMemoryLayout::new(
                    crate::LayerSchedule::new(
                        1,
                        vec![crate::cache::LayerCachePolicy::key_only(
                            crate::AttentionPolicy::Full,
                            1,
                            2,
                        )
                        .unwrap()],
                    )
                    .unwrap(),
                    vec![0],
                    1,
                    1,
                    crate::EstimationCompleteness::Complete,
                )
                .unwrap(),
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
                currently_cached_shards: unavailable(),
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
        let prepared = load_model(&LoadingMock::default(), root.path(), 41).unwrap();
        assert_eq!(*prepared, 41);

        let runtime = ModelRuntime::load(LoadingMock::default(), root.path(), 7).unwrap();
        assert_eq!(runtime.backend().descriptor().name, "loading-mock");

        let missing = root.path().join("missing");
        assert!(matches!(
            load_model(&LoadingMock::default(), &missing, 1),
            Err(ModelLoadError::Artifact(ArtifactError::MissingArtifact(path)))
                if path == missing
        ));
    }

    #[test]
    fn session_requirement_is_rejected_before_materialization() {
        let root = tempfile::tempdir().unwrap();
        write_loading_fixture(root.path());
        let backend = LoadingMock::default();

        let error = load_model(&backend, root.path(), 99).unwrap_err();

        assert!(matches!(
            error,
            ModelLoadError::SessionCapability(error)
                if error.capability() == "activation_inspection"
        ));
        assert_eq!(
            backend
                .materializations
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
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
        assert_eq!(capabilities.effective_model_type, "mock");
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

    #[derive(Debug, Clone)]
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
            DistributedCapabilities::new(true, [CollectiveGroupId::new(7)], true, true, true)
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
                output: vec![peer as u32; value.shape().iter().product()],
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
        let tensor_group =
            CollectiveGroupDescriptor::new(CollectiveGroupId::new(7), vec![0, 1], 0).unwrap();
        let session = MockDistributed {
            descriptor: DistributedSessionDescriptor::new(2, 0, vec![tensor_group]).unwrap(),
        };
        let capabilities = session.capabilities();
        assert!(capabilities.exact_completion());
        assert_eq!(
            capabilities.collective_groups(),
            &[CollectiveGroupId::new(7)]
        );
        assert_eq!(
            session
                .all_reduce_sum(
                    CollectiveScope::Group(CollectiveGroupId::new(7)),
                    &vec![2, 3]
                )
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
                    &ValueDescriptor::new(vec![2], TensorDtype::U32).unwrap(),
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
            distributed: Some(session.clone()),
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
        let descriptor = DistributedSessionDescriptor::new(
            6,
            4,
            vec![CollectiveGroupDescriptor::new(CollectiveGroupId::new(9), vec![1, 4], 1).unwrap()],
        )
        .unwrap();
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert_eq!(
            serde_json::from_str::<DistributedSessionDescriptor>(&encoded).unwrap(),
            descriptor
        );
        let scope = CollectiveScope::Group(CollectiveGroupId::new(9));
        assert_eq!(
            serde_json::from_str::<CollectiveScope>(&serde_json::to_string(&scope).unwrap())
                .unwrap(),
            scope
        );
        assert!(DistributedSessionDescriptor::new(descriptor.world_size(), 6, Vec::new()).is_err());
        assert!(serde_json::from_str::<DistributedSessionDescriptor>(
            r#"{"world_size":6,"rank":6,"groups":[]}"#
        )
        .is_err());
    }
}
