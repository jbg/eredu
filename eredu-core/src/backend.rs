//! High-level contract implemented once per execution backend.

mod capture_delivery;
mod branch;
pub use branch::{TextBranchSource, TextGenerationBranch, TextBranchFenced};
mod continuation;
mod controller_workspace;
mod failure;
mod host_preparation;
mod metadata_funding;
mod packed_filter;
mod preparation;
mod prepared_control;
pub use prepared_control::*;
mod reset_preparation;
mod resume;
mod shared_filter;
mod shared_storage;
mod text_step;
mod token_choice;
mod sampling_boundary;
pub use sampling_boundary::TextSamplingBoundary;
pub use token_choice::{ProspectiveTokenController, TextTokenChoiceBoundary};
pub use continuation::{
    TextContinuationBoundary, TextContinuationError, TextContinuationIdentity, TextDriverIdentity,
    TextGenerationContinuation, TextGenerationDriver, TextSnapshotSource,
};
pub use controller_workspace::{TextControllerContract, TextControllerContractError};
pub use failure::{
    BackendFailure, BackendFailureKind, GenerationSequenceBankRejection, HostMetadataFundingError,
    PreparedRequestRejection, SharedBackendFailure, TokenInputRejection,
};
pub use host_preparation::HostPreparationAuthority;
pub use metadata_funding::{HostMetadataAccount, HostMetadataFunding};
pub use packed_filter::{PackedTokenFilter, PackedTokenFilterError};
pub use preparation::{
    GenerationDecoderError, GenerationDecoderInput, GenerationDecoderOutput, GenerationPlainText,
    GenerationPlainTextEvent, GenerationPlainTextEvents, GenerationPlainTextProjection,
    GenerationSequenceAdmissionError, GenerationSequenceConsumerLayout,
    GenerationSequencePreparation, GenerationSequenceRequest, TextGenerationInput,
    TextPreparationInput, TextPreparationOptions, TokenIdsInputPlan,
};
pub use reset_preparation::{
    PreparedSessionReset, SessionResetPreparationBackend, SessionResetReadiness,
};
pub use resume::{OriginalTextResumeKind, OriginalTextResumeOptions, TextResumeBackend, TextResumeFacts, TextResumeSourceFacts, text_resume_control_bytes};
pub use shared_filter::SharedTokenFilter;
pub(crate) use shared_storage::SharedStorageCustody;
pub use shared_storage::{
    ControllerDeclarationData, ErasedSharedStorageOwner, SharedControllerBytes,
    SharedControllerDeclaration, SharedControllerSource, SharedStorageAttachmentError,
    SharedStorageDomain, SharedStorageIdentity, SharedStorageOwner, SharedStorageRetirement,
};
pub use text_step::{TextContextError, TextPolicyIdentity, TextRunIdentity, TextStepContext};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

use crate::{
    PreparationAdmission,
    artifact::{
        ArtifactError, ArtifactInspection, ModelConfigurationResolver, ModelPreparationPlan,
        inspect_artifact_with_prepared_gguf_headers,
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
    /// An error reports the outcome, not whether native resource use has stopped.
    fn wait(&self) -> Result<(), Self::Error>;

    /// Nonblocking proof that submitted work and its consumers no longer borrow
    /// retained resources. This is independent of successful output publication:
    /// a failed submission may become releasable while its error remains sticky.
    ///
    /// `false` includes pending work, unavailable evidence, and runtime contention.
    /// An observation error, timeout, or cancellation request is never proof of
    /// release. Implementations must include outstanding child/observation work
    /// and must not wait, retry failed execution, or run application destructors.
    /// Readiness does not make a failed session reusable or validate its outputs.
    /// Backends still own retention when the public completion is dropped.
    ///
    /// The default recognizes successful exact completion only. Backends that
    /// can establish safe release after failure should override this method.
    fn resources_releasable(&self) -> bool {
        matches!(self.is_complete(), Ok(true))
    }
}

/// Mechanism selected for work which has not completed by a bounded deadline.
#[derive(
    Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CompletionCancellationMode {
    /// The backend can request native cancellation and safely complete its teardown.
    NativeCancel,
    /// Native cancellation is unavailable, so the backend retains the orphaned work
    /// and every borrowed resource until exact completion can be observed.
    QuarantineUntilComplete,
}

/// Monotonic identity of one distributed session transaction.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct DistributedCommitEpoch(u64);

impl DistributedCommitEpoch {
    /// First epoch allocated by a newly constructed session.
    pub const FIRST: Self = Self(1);

    /// Creates a positive durable epoch identity.
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    /// Stable serialized epoch value.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Returns the next representable session epoch.
    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl<'de> Deserialize<'de> for DistributedCommitEpoch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| serde::de::Error::custom("commit epoch must be positive"))
    }
}

/// Final decision cut reached by a distributed commit attempt.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributedCommitPhase {
    /// The rank could not submit its final decision contribution.
    DecisionSubmission,
    /// The submitted final decision did not complete within its exact contract.
    DecisionCompletion,
    /// The rank observed the globally fixed decision.
    DecisionObservation,
}

/// Honest rank-local observation of one globally identified commit attempt.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributedCommitOutcome {
    /// This rank observed the globally fixed all-success decision.
    Committed(DistributedCommitEpoch),
    /// This rank observed the globally fixed abort decision before publication became final.
    Aborted(DistributedCommitEpoch),
    /// The rank may have contributed, but could not observe the fixed final decision.
    Indeterminate {
        /// Transaction identity retained for recovery and diagnosis.
        epoch: DistributedCommitEpoch,
        /// Exact final decision cut whose outcome could not be observed.
        phase: DistributedCommitPhase,
    },
}

impl DistributedCommitOutcome {
    /// Transaction identity shared by every outcome variant.
    pub const fn epoch(self) -> DistributedCommitEpoch {
        match self {
            Self::Committed(epoch) | Self::Aborted(epoch) | Self::Indeterminate { epoch, .. } => {
                epoch
            }
        }
    }

    /// Whether further use requires external recovery of the named epoch.
    pub const fn is_indeterminate(self) -> bool {
        matches!(self, Self::Indeterminate { .. })
    }
}

/// One explicit bounded-wait policy selected before a backend submission is created.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BoundedCompletionWait {
    timeout: std::time::Duration,
    cancellation: CompletionCancellationMode,
}

impl BoundedCompletionWait {
    /// Creates a positive relative deadline and its required safe cancellation mode.
    pub fn new(
        timeout: std::time::Duration,
        cancellation: CompletionCancellationMode,
    ) -> Result<Self, BoundedCompletionWaitError> {
        if timeout.is_zero() {
            return Err(BoundedCompletionWaitError::ZeroTimeout);
        }
        Ok(Self {
            timeout,
            cancellation,
        })
    }

    /// Maximum time spent waiting for exact completion.
    pub const fn timeout(self) -> std::time::Duration {
        self.timeout
    }

    /// Required safe disposition for work still live at the deadline.
    pub const fn cancellation(self) -> CompletionCancellationMode {
        self.cancellation
    }
}

/// Invalid bounded-completion policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum BoundedCompletionWaitError {
    /// A zero duration cannot establish a meaningful completion deadline.
    #[error("bounded completion timeout must be positive")]
    ZeroTimeout,
}

/// Stable result of observing one completion through an explicit bounded policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BoundedCompletionOutcome {
    /// Exact backend completion was observed before the deadline.
    Completed,
    /// The deadline expired and the backend safely applied the selected disposition.
    DeadlineExceeded {
        /// Cancellation or retained-orphan mechanism actually applied.
        cancellation: CompletionCancellationMode,
    },
}

/// Completion which can enforce a selected deadline without dropping live resources.
///
/// Implementations must consume themselves on timeout. `NativeCancel` means native
/// cancellation and teardown have completed before returning. `QuarantineUntilComplete`
/// means the implementation has transferred ownership of the completion and all work
/// resources to a safe owner which will release them only after exact completion. An
/// `Err` return has the same ownership obligation: it may represent a terminally
/// completed backend failure, or it must first transfer still-live work to the selected
/// safe disposition. Implementations must never return an error by merely dropping live
/// resources.
pub trait BoundedCompletion: Completion + Sized {
    /// Whether this completion implementation can honor the selected disposition.
    /// Implementations with a restricted native mechanism set must override this
    /// so schedulers can reject unsupported policy before submission.
    fn supports_cancellation(_cancellation: CompletionCancellationMode) -> bool {
        true
    }

    /// Observes exact completion until the policy deadline, then performs its selected
    /// safe cancellation disposition.
    fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error>;
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

impl<T, C> Submission<T, C>
where
    C: BoundedCompletion,
{
    /// Waits through the selected bounded policy and returns output only after exact
    /// completion. Timed-out output is dropped only after its completion has transferred
    /// ownership of every live backend resource to its safe cancellation owner.
    pub fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedSubmissionOutcome<T>, C::Error> {
        match self.completion.wait_bounded(policy)? {
            BoundedCompletionOutcome::Completed => {
                Ok(BoundedSubmissionOutcome::Completed(self.output))
            }
            BoundedCompletionOutcome::DeadlineExceeded { cancellation } => {
                Ok(BoundedSubmissionOutcome::DeadlineExceeded { cancellation })
            }
        }
    }
}

/// Result of a bounded submission wait. Output is authoritative only on completion.
#[derive(Debug, Eq, PartialEq)]
pub enum BoundedSubmissionOutcome<T> {
    /// Exact completion was observed and the output may be consumed.
    Completed(T),
    /// The deadline expired and live work entered its selected safe disposition.
    DeadlineExceeded {
        /// Cancellation or retained-orphan mechanism actually applied.
        cancellation: CompletionCancellationMode,
    },
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

    /// Consumes a backend failure at a public neutral boundary. The default
    /// preserves ordinary owned-source conversion. An adapter may transfer an
    /// already-retained source without allocating another wrapper; this grants
    /// neither completion nor memory authority and must preserve its cause.
    fn into_backend_failure(error: Self::Error) -> BackendFailure {
        BackendFailure::from_error(error)
    }

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
    ) -> Result<Self::SelectedPreparation, Self::Error>;

    /// Returns the exact portable admission retained by the authoritative selection.
    fn selected_preparation_admission(
        &self,
        selected: &Self::SelectedPreparation,
    ) -> PreparationAdmission;

    /// Binds a neutral preparation plan and its authoritative selected
    /// realization to backend-owned materialization input.
    fn model_config(
        &self,
        selected: SelectedModelPreparation<Self>,
    ) -> Result<Self::ModelConfig, Self::Error>;
}

/// One neutral preparation plan inseparably paired with its backend selection.
///
/// Core creates this value only after policy, architecture, and session-capability
/// admission have succeeded against the same artifact inspection. Backend adapters
/// may consume the pair, but callers cannot substitute a different plan after
/// selection.
pub struct SelectedModelPreparation<B: ModelLoadingBackend> {
    plan: ModelPreparationPlan<
        <B::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
    >,
    selected: B::SelectedPreparation,
}

impl<B: ModelLoadingBackend> SelectedModelPreparation<B> {
    pub(crate) fn new(
        plan: ModelPreparationPlan<
            <B::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
        >,
        selected: B::SelectedPreparation,
    ) -> Self {
        Self { plan, selected }
    }

    pub(crate) const fn plan(
        &self,
    ) -> &ModelPreparationPlan<<B::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan>
    {
        &self.plan
    }

    /// Consumes the binding into the exact admitted plan and backend selection.
    pub fn into_parts(
        self,
    ) -> (
        ModelPreparationPlan<
            <B::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
        >,
        B::SelectedPreparation,
    ) {
        (self.plan, self.selected)
    }
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
/// GGUF headers are retained from this initial inspection for later prepared
/// reads; this does not make tensor payloads immutable or certify a memory bound.
pub fn load_model<B: ModelLoadingBackend>(
    backend: &B,
    artifact: impl AsRef<Path>,
    options: B::LoadOptions,
) -> Result<PreparedModel<B::Model>, ModelLoadError<B::Error>> {
    let inspection =
        inspect_artifact_with_prepared_gguf_headers(artifact, backend.configuration_resolver())?;
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
    let config = prepare_inspected_model_config(backend, inspection, options)?;
    backend
        .prepare_model(config)
        .map_err(ModelLoadError::Backend)
}

/// Selects and prepares the exact source configuration without native model
/// materialization. This uses the same admission as [`prepare_inspected_model`].
/// A backend may borrow this configuration for cold source compilation before
/// entering its ordinary load path. No native execution or input admission is
/// conferred by the returned configuration.
pub fn prepare_inspected_model_config<B: ModelLoadingBackend>(
    backend: &B,
    inspection: ArtifactInspection<
        <B::ConfigurationResolver as ModelConfigurationResolver>::ArtifactPlan,
    >,
    options: B::LoadOptions,
) -> Result<B::ModelConfig, ModelLoadError<B::Error>> {
    let selected = backend
        .select_preparation(&inspection, &options)
        .map_err(ModelLoadError::Backend)?;
    let admission = backend.selected_preparation_admission(&selected);
    let plan = ModelPreparationPlan::from_retained_admission(inspection, admission)?;
    backend
        .model_config(SelectedModelPreparation::new(plan, selected))
        .map_err(ModelLoadError::Backend)
}

/// Materializes an already planned artifact with its retained backend selection.
///
/// Selection and capability admission must have completed before the backend
/// instance was realized. This entry point therefore performs no policy or
/// route selection after native resources exist.
pub(crate) fn prepare_selected_model<B: ModelLoadingBackend>(
    backend: &B,
    selected: SelectedModelPreparation<B>,
) -> Result<PreparedModel<B::Model>, ModelLoadError<B::Error>> {
    let config = backend
        .model_config(selected)
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

    /// Prefill with cooperative cancellation. `None` means cancellation was
    /// agreed and all submitted work safely settled, without an output. Native
    /// chunked implementations must check between completed spans. The default
    /// retains the existing indivisible submission; it cannot cancel mid-call.
    fn prefill_cancellable(
        &mut self,
        backend: &B,
        input: Self::PrefillInput,
        _cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Submission<Self::Output, Self::Completion>>, B::Error> {
        self.prefill(backend, input).map(Some)
    }

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
/// session together makes backend selection a whole-model decision: generic
/// operations always supply this retained backend. Backend session hooks still
/// validate native identity when unrestricted session mutation is exposed.
/// Backend-owned executable, cache, tensor, and completion types remain
/// associated types and never enter the portable API.
pub struct ModelRuntime<B: BackendProvider> {
    backend: B,
    session: B::Session,
    admission: crate::SessionAdmission,
    execution_plan_target_id: Option<u64>,
}

impl<B: BackendProvider> ModelRuntime<B> {
    /// Prepares `config` and creates its sole execution session.
    pub fn prepare(backend: B, config: B::ModelConfig) -> Result<Self, B::Error> {
        let model = backend.prepare_model(config)?;
        Self::from_prepared(backend, model)
    }

    /// Creates the sole execution session for an already prepared model.
    pub fn from_prepared(backend: B, model: PreparedModel<B::Model>) -> Result<Self, B::Error> {
        Self::from_prepared_with_execution_plan_target(backend, model, None)
    }

    pub(crate) fn from_prepared_execution_plan_target(
        backend: B,
        model: PreparedModel<B::Model>,
        execution_plan_target_id: u64,
    ) -> Result<Self, B::Error> {
        Self::from_prepared_with_execution_plan_target(
            backend,
            model,
            Some(execution_plan_target_id),
        )
    }

    fn from_prepared_with_execution_plan_target(
        backend: B,
        model: PreparedModel<B::Model>,
        execution_plan_target_id: Option<u64>,
    ) -> Result<Self, B::Error> {
        let admitted = model.capabilities();
        let session = backend.create_session(model)?;
        let realized = session.capabilities();
        if crate::SessionAdmission::new(admitted)
            .validate(realized)
            .is_err()
        {
            return Err(backend.session_capability_mismatch(admitted, realized));
        }
        Ok(Self {
            backend,
            session,
            admission: crate::SessionAdmission::new(admitted),
            execution_plan_target_id,
        })
    }

    pub(crate) const fn execution_plan_target_id(&self) -> Option<u64> {
        self.execution_plan_target_id
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
    ///
    /// Unrestricted mutation can replace the session, so this invalidates any
    /// retained execution-plan target proof. Previously selected drafting can
    /// no longer attach to this runtime. Generic operations still enforce the
    /// original exact capability admission after mutation.
    pub fn session_mut(&mut self) -> &mut B::Session {
        self.execution_plan_target_id = None;
        &mut self.session
    }

    /// Borrows the selected backend and its mutable session together.
    ///
    /// Like [`Self::session_mut`], this invalidates retained drafting target proof.
    pub fn parts_mut(&mut self) -> (&B, &mut B::Session) {
        self.execution_plan_target_id = None;
        (&self.backend, &mut self.session)
    }

    /// Read-only comparison of this session with its exact retained admission.
    /// Backend-specific composed operations must call this when using a path
    /// outside the ordinary runtime submission methods. Success grants no
    /// submission or native allocation authority and does not rebind admission.
    pub fn validate_session_admission(&self) -> Result<(), B::Error> {
        self.admission
            .validate(self.session.capabilities())
            .map_err(|error| {
                self.backend
                    .session_capability_mismatch(error.admitted(), error.realized())
            })
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
        self.validate_session_admission()?;
        self.session.prefill(&self.backend, input)
    }

    /// Executes cancellable prefill with the same exact session admission.
    /// Cancellation produces no output and implies safe native settlement.
    pub fn prefill_cancellable(
        &mut self,
        input: <B::Session as BackendSession<B>>::PrefillInput,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<SessionSubmission<B>>, B::Error> {
        self.validate_session_admission()?;
        self.session
            .prefill_cancellable(&self.backend, input, cancellation)
    }

    /// Submits cached decode through the selected backend and session.
    pub fn decode(
        &mut self,
        input: <B::Session as BackendSession<B>>::DecodeInput,
    ) -> Result<SessionSubmission<B>, B::Error> {
        self.validate_session_admission()?;
        self.session.decode(&self.backend, input)
    }

    /// Materializes portable observations from an already completed output.
    pub fn observe_output(
        &self,
        output: &<B::Session as BackendSession<B>>::Output,
    ) -> Result<ObservationSet, B::Error> {
        self.validate_session_admission()?;
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
        self.validate_session_admission()?;
        self.session.inspect_prefill(&self.backend, input, request)
    }

    /// Executes a completed, explicitly instrumented decode operation.
    pub fn inspect_decode(
        &mut self,
        input: <B::Session as BackendSession<B>>::DecodeInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<<B::Session as BackendSession<B>>::Output>, B::Error> {
        self.validate_session_admission()?;
        self.session.inspect_decode(&self.backend, input, request)
    }
}

impl<B: TextGenerationBackend> ModelRuntime<B> {
    /// Completes fallible local preparation without stranding successful peers.
    /// Local failures retain their original type. Agreement errors independently
    /// fence native authority; an original local error never proves safe reuse.
    pub fn finish_text_preparation<T, E>(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<T, E> {
        crate::run_preparation::finish_preparation(
            stage,
            local,
            |status| self.agree_text_preparation(stage, status),
            map_backend,
        )
    }

    /// As above, with `Ok(None)` representing cancellation before model work.
    /// Every successful participant receives cancellation when any peer cancels;
    /// a reported failure takes precedence over cancellation.
    pub fn finish_text_preparation_cancellable<T, E>(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<Option<T>, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        crate::run_preparation::finish_preparation_cancellable(
            stage,
            local,
            |status| self.agree_text_preparation(stage, status),
            map_backend,
        )
    }

    /// Agrees readiness at a cold run boundary without entering model execution.
    /// Every participant uses the same stage order, including failed/cancelled
    /// preparation. A completed rejection is distinct from transport failure.
    pub fn agree_text_preparation(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, BackendFailure> {
        let admission = self
            .admission
            .validate(self.session.capabilities())
            .map_err(BackendFailure::from);
        // Invalid local admission still participates in bounded failure
        // agreement, so ready peers cannot proceed into a stranded prediction.
        let outcome = B::agree_text_preparation(
            self,
            stage,
            if admission.is_err() {
                crate::run_preparation::TextPreparationStatus::Failed
            } else {
                status
            },
        );
        admission?;
        outcome
    }

    pub(crate) fn finish_text_preparation_control<T, E>(
        &self,
        control: Option<&B::TextPreparationControl>,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<T, E> {
        crate::run_preparation::finish_preparation(
            stage,
            local,
            |status| self.agree_text_preparation_control(control, stage, status),
            map_backend,
        )
    }
    pub(crate) fn finish_text_preparation_control_cancellable<T, E>(
        &self,
        control: Option<&B::TextPreparationControl>,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<Option<T>, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        crate::run_preparation::finish_preparation_cancellable(
            stage,
            local,
            |status| self.agree_text_preparation_control(control, stage, status),
            map_backend,
        )
    }
    fn agree_text_preparation_control(
        &self,
        control: Option<&B::TextPreparationControl>,
        stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, BackendFailure> {
        let admission = self
            .admission
            .validate(self.session.capabilities())
            .map_err(BackendFailure::from);
        let outcome = B::agree_text_preparation_with_control(
            self,
            control,
            stage,
            if admission.is_err() {
                crate::run_preparation::TextPreparationStatus::Failed
            } else {
                status
            },
        );
        admission?;
        outcome
    }

    /// Reports session lifetime preparation reservations, unaffected by restore.
    pub fn text_preparation_usage(
        &self,
    ) -> Result<crate::run_preparation::TextPreparationUsage, BackendFailure> {
        B::text_preparation_usage(self)
    }

    /// Settles prior work and clears request state while retaining the loaded model.
    ///
    /// Success establishes a fresh session for the next request. Model identity,
    /// admitted capabilities, placement and retained drafting target proof are
    /// preserved. Failure does not establish that the session can be reused.
    pub fn reset(&mut self) -> Result<(), BackendFailure> {
        self.synchronize()?;
        B::reset_session(&self.backend, &mut self.session)
    }

    /// Requests an originally funded reset without implicitly waiting for prior work.
    ///
    /// The provider must establish exact readiness without unreserved native work,
    /// or reject before construction. Busy leaves the old session installed and
    /// does not establish completion. The default provider rejects unsupported
    /// execution. Ordinary [`Self::reset`] still synchronizes; earlier generation
    /// funding and numeric reports do not authorize this new operation.
    pub fn reset_admitted(
        &mut self,
        limits: crate::SessionResetLimits,
    ) -> Result<(), BackendFailure> {
        self.admission.validate(self.session.capabilities())?;
        let claim = crate::SessionResetClaim::new(&self.session, &self.admission, limits);
        B::reset_session_admitted(&self.backend, &mut self.session, claim)
    }

    /// Waits for this session's submitted work, including abandoned generation.
    ///
    /// Success establishes that no work or submission authority remains pending
    /// and the session is healthy. It preserves request state. An error is not
    /// completion evidence; the backend must retain any unresolved resources.
    /// Allocator caches and process-wide memory reclamation are outside this
    /// contract.
    pub fn synchronize(&self) -> Result<(), BackendFailure> {
        self.admission.validate(self.session.capabilities())?;
        B::synchronize_session(&self.backend, &self.session)
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
    inference: crate::TextInferencePolicy,
}

/// Backend-neutral token-sampling strategy for one text-generation session.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum TextSamplingStrategy {
    /// Apply the resolved top-k, top-p, min-p, and penalty controls.
    #[default]
    Standard,
    /// Adapt the surprise cutoff toward `tau` bits at rate `eta`.
    ///
    /// Requires positive temperature. Retains penalties and replaces top-k,
    /// top-p, and min-p; vocabulary filters apply before this strategy.
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
            inference: crate::TextInferencePolicy {
                prefill_chunk_positions: None,
                managed_memory_capacity_bytes: None,
                submission_tracking_capacity_bytes: None,
                graph_metadata_capacity_bytes: None,
            },
        }
    }

    /// Selects the deterministic root seed used by a stochastic backend.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Applies the same execution limits to ordinary and controlled startup.
    /// Admission validates finite output allowance and complete memory bounds
    /// before native prompt or sampling-state construction.
    pub const fn with_inference_policy(mut self, policy: crate::TextInferencePolicy) -> Self {
        self.inference = policy;
        self
    }

    /// Requested execution limits, before admission selects a smaller chunk.
    pub const fn inference_policy(&self) -> crate::TextInferencePolicy {
        self.inference
    }

    /// Selects adaptive Mirostat V2 sampling, requiring positive temperature.
    pub fn with_mirostat_v2(mut self, tau: f32, eta: f32) -> Result<Self, GenerationError> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(GenerationError::InvalidMirostatTau(tau));
        }
        if !eta.is_finite() || eta <= 0.0 {
            return Err(GenerationError::InvalidMirostatEta(eta));
        }
        if !self.sampling.temperature.is_finite() || self.sampling.temperature <= 0.0 {
            return Err(GenerationError::InvalidMirostatTemperature(
                self.sampling.temperature,
            ));
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
    /// IDs beyond this mask are forbidden, even when model logits are wider.
    /// A narrower model output uses only the executable prefix of the mask.
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

    /// Whether this closed set permits a canonical token ID.
    pub fn allows(&self, token: u32) -> bool {
        self.allowed_mask()
            .is_none_or(|mask| mask.get(token as usize).copied().unwrap_or(false))
    }

    /// Intersects independent restrictions, rejecting an empty allowed set.
    pub fn intersection(&self, other: &Self) -> Result<Self, TokenFilterError> {
        match (self.allowed_mask(), other.allowed_mask()) {
            (None, None) => Ok(Self::All),
            (Some(mask), None) | (None, Some(mask)) => Self::allowed(mask.to_vec()),
            (Some(left), Some(right)) => {
                Self::allowed(left.iter().zip(right).map(|(a, b)| *a && *b).collect())
            }
        }
    }

    /// Validates an executable vocabulary without expanding or copying a mask.
    pub fn validate_output_width(&self, output_width: usize) -> Result<(), TokenFilterError> {
        if output_width == 0 {
            return Err(TokenFilterError::EmptyVocabulary);
        }
        if let Some(mask) = self.allowed_mask() {
            if !mask[..mask.len().min(output_width)]
                .iter()
                .any(|allowed| *allowed)
            {
                return Err(TokenFilterError::NoExecutableToken { output_width });
            }
        }
        Ok(())
    }

    /// Realizes a closed allow set at the model's actual output width.
    /// Missing IDs are false; a truncated prefix must still allow a token.
    pub fn allowed_mask_for(
        &self,
        output_width: usize,
    ) -> Result<Option<std::borrow::Cow<'_, [bool]>>, TokenFilterError> {
        self.validate_output_width(output_width)?;
        let Some(mask) = self.allowed_mask() else {
            return Ok(None);
        };
        let prefix = &mask[..mask.len().min(output_width)];
        Ok(Some(if mask.len() >= output_width {
            std::borrow::Cow::Borrowed(prefix)
        } else {
            let mut mask = vec![false; output_width];
            mask[..prefix.len()].copy_from_slice(prefix);
            std::borrow::Cow::Owned(mask)
        }))
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
    /// The intersection with the model's output domain is empty.
    #[error("token filter permits no token in the model output vocabulary of size {output_width}")]
    NoExecutableToken {
        /// Actual logits width.
        output_width: usize,
    },
}

/// One exact sampling filter with optional pre-override observation provenance.
/// Existing controllers can return an unknown domain without inferring whether
/// their exclusions came from tokenizer validity, semantics or forcing.
#[derive(Debug)]
pub struct TokenSamplingDecision<'a> {
    filter: TokenFilter,
    pre_override_filter: Option<TokenFilter>,
    tokenizer_validity: Option<&'a TokenFilter>,
    shared_tokenizer_validity: Option<&'a SharedTokenFilter>,
    original_tokenizer_validity: Option<OriginalSourceWitness<'a>>,
    controller_storage: Option<TextControllerStorage<'a>>,
}

impl<'a> TokenSamplingDecision<'a> {
    /// Creates a decision with unknown observation provenance.
    pub fn new(filter: TokenFilter) -> Self {
        Self {
            filter,
            pre_override_filter: None,
            tokenizer_validity: None,
            shared_tokenizer_validity: None,
            original_tokenizer_validity: None,
            controller_storage: None,
        }
    }

    /// Declares the tokenizer baseline for this exact pre-override filter.
    pub fn with_tokenizer_validity(mut self, validity: &'a TokenFilter) -> Self {
        self.tokenizer_validity = Some(validity);
        self.shared_tokenizer_validity = None;
        self.original_tokenizer_validity = None;
        self
    }

    /// Declares the exact shared owner of the tokenizer domain. Its immutable
    /// identity lets admitted execution validate lifetime coverage after the
    /// controller callback as well as before it.
    pub fn with_shared_tokenizer_validity(mut self, validity: &'a SharedTokenFilter) -> Self {
        self.tokenizer_validity = Some(validity.as_ref());
        self.shared_tokenizer_validity = Some(validity);
        self.original_tokenizer_validity = None;
        self
    }

    /// Borrows a separately original token domain. This declaration alone grants
    /// no reduced bound: the runtime must authenticate its concrete source, the
    /// exact filter object and the original request before validating ownership.
    pub fn with_original_tokenizer_validity(
        mut self,
        validity: &'a TokenFilter,
        source: OriginalSourceWitness<'a>,
    ) -> Self {
        self.tokenizer_validity = Some(validity);
        self.shared_tokenizer_validity = None;
        self.original_tokenizer_validity = Some(source);
        self
    }

    /// Actual borrowed source provenance, still requiring runtime authentication.
    pub fn original_tokenizer_validity(&self) -> Option<OriginalSourceWitness<'a>> {
        self.original_tokenizer_validity
    }

    /// Borrowed tokenizer domain, when explicitly supplied by the controller.
    pub fn tokenizer_validity(&self) -> Option<&'a TokenFilter> {
        self.tokenizer_validity
    }

    /// Exact immutable owner of a separately retained tokenizer domain.
    pub fn shared_tokenizer_validity(&self) -> Option<&'a SharedTokenFilter> {
        self.shared_tokenizer_validity
    }

    /// Supplies the complete shared-source inventory after this controller
    /// callback. The descriptor borrows its owners without copying or owning
    /// their payloads. Managed submission with declared byte sources requires
    /// this evidence and checks it against the admitted immutable inventory.
    /// Legacy filter-only decisions may omit it.
    pub fn with_controller_storage(mut self, storage: TextControllerStorage<'a>) -> Self {
        self.controller_storage = Some(storage);
        self
    }

    /// Complete post-callback source evidence, when supplied. This declaration
    /// grants no admission, attachment or permission to replace shared owners.
    pub fn controller_storage(&self) -> Option<TextControllerStorage<'a>> {
        self.controller_storage
    }

    /// The final filter to apply to native sampling.
    pub fn filter(&self) -> &TokenFilter {
        &self.filter
    }

    /// Applies an already validated forced choice while retaining the original
    /// domain for observation and the post-callback source inventory. This
    /// moves filters rather than copying masks.
    pub fn override_filter(&mut self, filter: TokenFilter) {
        let original = std::mem::replace(&mut self.filter, filter);
        if self.pre_override_filter.is_none() {
            self.pre_override_filter = Some(original);
        }
    }

    /// Exact pre-override domain, or explicitly unknown.
    pub fn capture_domain(&self) -> Option<crate::capture::CaptureTokenDomain<'_>> {
        Some(crate::capture::CaptureTokenDomain {
            filter: self
                .pre_override_filter
                .as_ref()
                .unwrap_or(&self.filter)
                .into(),
            tokenizer_validity: self.tokenizer_validity?,
        })
    }
}

/// Borrowed concrete source evidence, with no ownership or admission authority.
/// The runtime must downcast and authenticate the actual closed source against
/// the original request. A matching type, value or size alone is insufficient.
///
/// ```compile_fail
/// fn escape() -> eredu_core::OriginalSourceWitness<'static> {
///     let source = 7u8;
///     eredu_core::OriginalSourceWitness::new(&source)
/// }
/// ```
#[derive(Clone, Copy)]
pub struct OriginalSourceWitness<'a>(&'a (dyn std::any::Any + Send + Sync));
impl<'a> OriginalSourceWitness<'a> {
    /// Borrows an actual immutable source without allocating or invoking it.
    pub fn new(source: &'a (dyn std::any::Any + Send + Sync)) -> Self {
        Self(source)
    }
    /// Whether both witnesses borrow the same concrete object. This is lexical
    /// source identity only; it neither authenticates the object nor grants work.
    pub fn same_borrowed_source(self, other: Self) -> bool {
        self.0.type_id() == other.0.type_id() && std::ptr::addr_eq(self.0, other.0)
    }
    /// Checks the concrete borrowed type; the result still carries no authority.
    pub fn downcast_ref<T: std::any::Any>(&self) -> Option<&'a T> {
        self.0.downcast_ref()
    }
}
impl Debug for OriginalSourceWitness<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalSourceWitness(..)")
    }
}

/// Actual borrowed state source for an independently paid semantic controller.
/// The runtime must authenticate these owners and their retained account.
#[derive(Debug, Clone, Copy)]
pub enum PreparedControllerSource<'a> {
    /// Plain canonical token history and its immutable validity source.
    Plain {
        /// Actual paid mutable history.
        history: &'a crate::speculative::PlainControllerHistory,
        /// Exact loaded tokenizer-validity owner.
        validity: &'a SharedTokenFilter,
    },
    /// Mutable grammar state and its exact immutable recipe/trie inputs.
    Grammar(crate::speculative::PreparedGrammarSource<'a>),
    /// Fixed forbidden inputs and their paid canonical history.
    Forbidden(crate::speculative::ForbiddenControllerSource<'a>),
}

/// Complete lifetime declaration for controller numerical payloads.
///
/// Shared sources are included in the ordinary size witness; enumerating them
/// does not discount that witness or register their storage. All remaining
/// numerical payload must retire with the controller or its owned decisions,
/// before run funding retires, without escaping through errors or callbacks.
#[derive(Debug, Clone, Copy)]
pub enum TextControllerStorage<'a> {
    /// No complete storage-lifetime declaration is available.
    Unknown,
    /// Every numerical payload covered by the size witness is run-owned.
    RunOwned,
    /// One separately original immutable token domain and a run-owned remainder.
    /// Legacy inventory/pinning cannot accept this declaration. A source-aware
    /// runtime must authenticate the actual owner and original request; no source
    /// bytes are credited, adopted or attached to the request account.
    RunOwnedWithOriginalTokenDomain(OriginalSourceWitness<'a>),
    /// Authenticated independently paid semantic state, plus a run-owned emitted
    /// mask. The witness is descriptive until a source-aware runtime validates
    /// its exact closed binding and the actual state source. Registry adoption
    /// cannot accept this declaration or discount its independent funding.
    RunOwnedWithPreparedSemantic {
        /// Borrow of the closed runtime source/execution binding.
        binding: OriginalSourceWitness<'a>,
        /// Actual state inputs and retained history payer at this callback.
        source: PreparedControllerSource<'a>,
    },
    /// The enumerated immutable filters have independent shared lifetimes;
    /// every other numerical payload is run-owned. The inventory must remain
    /// unchanged throughout the admitted run, including callbacks: every
    /// enumerated source stays live alongside the run-owned remainder. Their
    /// capacities are a fixed part of the additional host allowance, beyond
    /// the final emitted sampling mask. Runtime may independently pin already
    /// registered identities and credit that fixed contribution. Borrowed
    /// shared tokenizer domains must preserve their owner in the decision.
    RunOwnedWithSharedFilters(&'a [SharedTokenFilter]),
    /// The enumerated immutable filters and byte sources have independent
    /// shared lifetimes; every other numerical payload is run-owned. All
    /// sources remain live and unchanged throughout the admitted run and are
    /// included in the fixed additional host allowance. Byte sources do not
    /// establish tokenizer-filter provenance.
    RunOwnedWithSharedStorage {
        /// Immutable mask sources, including any shared tokenizer domain.
        filters: &'a [SharedTokenFilter],
        /// Immutable non-mask source allocations.
        bytes: &'a [SharedControllerBytes],
    },
    /// Complete immutable filter/byte/declaration inventory and a run-owned
    /// remainder. Every source contributes its exact capacity to the additional
    /// host allowance; declarations supply no tokenizer-filter provenance.
    RunOwnedWithSharedDeclarations {
        /// Immutable filter sources.
        filters: &'a [SharedTokenFilter],
        /// Immutable byte sources.
        bytes: &'a [SharedControllerBytes],
        /// Immutable independently constructed declaration sources.
        declarations: &'a [SharedControllerDeclaration],
    },
}

impl<'a> TextControllerStorage<'a> {
    /// Actual source-bearing declaration, never a grant or numeric size witness.
    pub fn original_token_domain(self) -> Option<OriginalSourceWitness<'a>> {
        match self {
            Self::RunOwnedWithOriginalTokenDomain(source) => Some(source),
            _ => None,
        }
    }
    /// Complete shared inventory expressed only as filters. Returns None when
    /// ownership is unknown or any non-mask source would be omitted.
    pub fn shared_filters(self) -> Option<&'a [SharedTokenFilter]> {
        match self {
            Self::Unknown | Self::RunOwnedWithOriginalTokenDomain(_) | Self::RunOwnedWithPreparedSemantic { .. } => None,
            Self::RunOwned => Some(&[]),
            Self::RunOwnedWithSharedFilters(filters) => Some(filters),
            Self::RunOwnedWithSharedStorage { filters, bytes } if bytes.is_empty() => Some(filters),
            Self::RunOwnedWithSharedStorage { .. } => None,
            Self::RunOwnedWithSharedDeclarations {
                filters,
                bytes,
                declarations,
            } if bytes.is_empty() && declarations.is_empty() => Some(filters),
            Self::RunOwnedWithSharedDeclarations { .. } => None,
        }
    }

    /// Complete shared source inventory, or None when ownership is unknown.
    /// Iteration only borrows the declarations; it allocates no source list and
    /// does not clone, register or attach the numerical owners.
    pub fn shared_sources(self) -> Option<impl Iterator<Item = SharedControllerSource<'a>>> {
        let (filters, bytes, declarations): (
            &'a [SharedTokenFilter],
            &'a [SharedControllerBytes],
            &'a [SharedControllerDeclaration],
        ) = match self {
            Self::Unknown | Self::RunOwnedWithOriginalTokenDomain(_) | Self::RunOwnedWithPreparedSemantic { .. } => return None,
            Self::RunOwned => (&[], &[], &[]),
            Self::RunOwnedWithSharedFilters(filters) => (filters, &[], &[]),
            Self::RunOwnedWithSharedStorage { filters, bytes } => (filters, bytes, &[]),
            Self::RunOwnedWithSharedDeclarations {
                filters,
                bytes,
                declarations,
            } => (filters, bytes, declarations),
        };
        Some(
            filters
                .iter()
                .map(SharedControllerSource::Filter)
                .chain(bytes.iter().map(SharedControllerSource::Bytes))
                .chain(declarations.iter().map(SharedControllerSource::Declaration)),
        )
    }
}

/// Backend-independent logical controller for constrained token selection.
pub trait TokenFilterController {
    /// Constraint or grammar error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Describes a complete cold bound through the remaining output allowance.
    /// Unknown controller/filter growth prevents enforced admission; this must
    /// not query or advance the current decision. Application controllers may
    /// supply a bound when their retained state and every emitted filter fit it.
    fn inference_workspace(
        &self,
        _max_output_tokens: u64,
    ) -> Option<crate::TextControllerWorkspace<'_>> {
        None
    }

    /// Certifies the lifetime of opaque numerical payload covered by run funding.
    ///
    /// Returning true promises that every such payload described by
    /// [`Self::inference_workspace`] retires with this controller or its emitted
    /// decisions, before the backend's run state retires. It must not survive
    /// through shared clones, callback side effects, returned errors or other
    /// escapes. Descriptive bookkeeping is not numerical payload.
    ///
    /// This is a cold ownership declaration, independent of the size bound.
    /// Strict admission cannot release run funding using a size witness alone.
    fn inference_workspace_is_run_owned(&self) -> bool {
        false
    }

    /// Declares all shared numerical owners and the run-owned remainder without
    /// advancing a decision. The default preserves existing wholly run-owned
    /// controllers; shared or opaque controllers must supply complete evidence.
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        if self.inference_workspace_is_run_owned() {
            TextControllerStorage::RunOwned
        } else {
            TextControllerStorage::Unknown
        }
    }

    /// Returns the filter for the current durable logical prefix.
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error>;

    /// Returns the sampling filter and, when available, its exact observation
    /// domain. Implementations must reuse the same filter query and preserve the
    /// domain before forcing. The default explicitly leaves capture unknown.
    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        self.current_filter().map(TokenSamplingDecision::new)
    }

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
    /// Actual move-only prepared grammar state, or the uninhabited fixed-source
    /// type. This association does not authenticate a source or grant execution.
    type PreparedGrammar: crate::speculative::PreparedGrammarController;

    /// Borrows the complete paid grammar. Unknown and fixed controllers return
    /// None; they must not supply plain/forbidden and grammar sources together.
    fn prepared_grammar(&self) -> Option<&Self::PreparedGrammar> { None }

    /// Prospective payment for publishing a copied prepared grammar through
    /// this complete controller wrapper. The grammar copy is priced separately;
    /// absent or unknown publication producers return None.
    fn prepared_grammar_replacement_bytes(&self) -> Option<usize> { None }

    /// Publishes an independently prepared successor, retaining its exact inputs
    /// and supplied account. All allocation must be paid before construction;
    /// refusal retains the uninstalled successor. Ordinary Clone is not a copy
    /// producer for mutable grammar state.
    fn replace_prepared_grammar(
        &self, grammar: Self::PreparedGrammar, funding: &crate::HostMetadataFunding,
    ) -> Result<Self, crate::speculative::PreparedGrammarInstallError<Self::PreparedGrammar>> {
        Err(crate::speculative::PreparedGrammarInstallError::new(
            crate::speculative::PreparedGrammarInstallCause::Unknown, grammar, funding,
        ))
    }

    /// Actual fixed plain controller state, never an arbitrary callback summary.
    /// The complete decision semantics must be canonical-prefix validation plus
    /// this exact tokenizer-validity filter. Unknown/grammar controllers return None.
    fn prepared_plain_source(&self) -> Option<crate::speculative::PlainControllerSource<'_>> {
        None
    }

    /// Exact host copy destinations and controls for the requested capacity.
    /// Shared source accounting is separate; unknown opaque fields return None.
    fn prepared_plain_copy_bytes(&self, _capacity: usize) -> Option<usize> {
        None
    }

    /// Copies the complete plain controller under prepaid destination custody.
    /// All copied payload, clones and allocating failures must retain that custody.
    /// The returned source must preserve the same immutable filter and prefix.
    fn copy_prepared_plain(
        &self,
        _capacity: usize,
        _host: crate::HostPreparationAuthority,
    ) -> Result<Self, crate::speculative::PlainControllerError> {
        Err(crate::speculative::PlainControllerError::Unknown)
    }

    /// Borrows the actual history named by prepared_plain_source. Prepared
    /// commitment updates only this fixed destination; it never invokes opaque
    /// current_filter/commit_token callbacks or grows ordinary storage.
    fn prepared_plain_history_mut(
        &mut self,
    ) -> Option<&mut crate::speculative::PlainControllerHistory> {
        None
    }

    /// Actual prepared forbidden-trigger state. It is distinct from plain state:
    /// candidate validity also depends on the exact current byte-trigger prefix.
    fn prepared_forbidden_source(
        &self,
    ) -> Option<crate::speculative::ForbiddenControllerSource<'_>> {
        None
    }
    /// Exact independent history and fixed controls; immutable paid inputs alias
    /// their existing owner rather than being adopted into a new account.
    fn prepared_forbidden_copy_bytes(&self, _capacity: usize) -> Option<usize> {
        None
    }
    /// Copies a complete forbidden controller under paid destination custody.
    /// The copied source preserves the actual immutable inputs and validity.
    fn copy_prepared_forbidden(
        &self,
        _capacity: usize,
        _host: crate::HostPreparationAuthority,
    ) -> Result<Self, crate::speculative::ForbiddenControllerError> {
        Err(crate::speculative::ForbiddenControllerError::Unknown)
    }
    /// Borrows the exact mutable history and prefix named by the forbidden
    /// source. The returned worker performs atomic, nonallocating commitment.
    fn prepared_forbidden_mutation(
        &mut self,
    ) -> Result<
        crate::speculative::ForbiddenControllerMutation<'_>,
        crate::speculative::ForbiddenControllerError,
    > {
        Err(crate::speculative::ForbiddenControllerError::Unknown)
    }

    /// Complete bytes retained by an isolated clone, or unknown.
    fn control_snapshot_bytes(&self) -> Option<u64> {
        None
    }

    /// Returns the filter at `history` without committing its uncommitted suffix.
    ///
    /// `history` contains the controller's durable prefix followed by zero or
    /// more speculative tokens. Implementations must reject histories that do
    /// not begin with the durable prefix.
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error>;

    /// Speculative counterpart of [`TokenFilterController::current_decision`].
    fn decision_at(&self, history: &[u32]) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        self.filter_at(history).map(TokenSamplingDecision::new)
    }

    /// Returns whether `history` completes the constraint without committing it.
    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error>;
}

#[derive(Debug, Clone)]
struct FixedTokenFilter(TokenFilter);

impl TokenFilterController for FixedTokenFilter {
    type Error = std::convert::Infallible;

    fn inference_workspace_is_run_owned(&self) -> bool {
        // The fixed mask and emitted clones own separate vectors; no callback
        // exports them and the controller cannot return a payload-bearing error.
        true
    }

    fn inference_workspace(&self, _: u64) -> Option<crate::TextControllerWorkspace<'_>> {
        Some(crate::TextControllerWorkspace {
            filter: (&self.0).into(),
            // current_filter clones the fixed source. Sampling already prices
            // one supplied filter; the source allocation remains alive too.
            additional_host_bytes: match &self.0 {
                TokenFilter::All => 0,
                TokenFilter::Allowed(mask) => mask.capacity() as u64,
            },
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(self.0.clone())
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
/// Lifecycle hooks return the common [`BackendFailure`]; preserve the original
/// error as its source and classify only from known facts, not diagnostic text.
pub trait TextGenerationBackend: BackendProvider {
    /// Retained request admission. Clones share its charge and cannot authorize
    /// another run. Native completions must independently retain unresolved work.
    type TextPreparation: Clone;

    /// Retained readiness-consensus source for this request. Clones share exact
    /// source custody and monotone spending, never a fresh model admission.
    /// Ordinary backends may use ().
    type TextPreparationControl: Clone;

    /// Borrows the actual accepted request's diagnostics without reserving,
    /// cloning reports, or inspecting native execution. Report selected geometry
    /// and the retained successful admission, never the requested policy cap.
    /// None means this preparation has no available accepted-admission report.
    fn text_preparation_report(
        _preparation: &Self::TextPreparation,
    ) -> Option<crate::TextPreparationReport<'_>> {
        None
    }

    /// Move-only authority for one current prediction, separate from retained
    /// charges. Dropping an unfinished permit must preserve or fence its work;
    /// native completions independently retain resources through settlement.
    type TextStepPermit;

    /// Validates the exact current request, policy and input before controller
    /// callbacks or native prediction work. Historical retained reservations
    /// alone grant no authority. Rejection must not consume executable authority.
    fn begin_text_step<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Self::TextPreparation,
        state: &Self::TextGenerationState,
        controller: &C,
        input: PendingTextInput<&Self::Prompt, &Self::Token>,
        context: &TextStepContext,
    ) -> Result<Self::TextStepPermit, Self::Error>;

    /// Finalizes one successful prediction after any controlled token callback.
    /// This need not wait for native completion: unresolved native work retains
    /// its own resources. Failure must leave the run fenced or safely retained.
    fn finish_text_step(permit: Self::TextStepPermit) -> Result<(), Self::Error>;

    /// Quotes and admits one exact request before prompt/sampler allocation.
    /// This may inspect metadata and reserve shared capacity, but must not create
    /// native tensors or advance the controller. Unknown required bounds must
    /// reject an enforced policy. Explicit unbudgeted policy may return a unit
    /// owner without claiming memory coverage.
    fn admit_text_preparation<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
    ) -> Result<Self::TextPreparation, BackendFailure>;

    /// Admits owned preparation sources before prompt or sampler construction.
    ///
    /// An implementation must account for the exact shared source and the whole
    /// supported capture schedule in this original admission. It must reject
    /// unknown bounds before creating payloads or invoking controller decisions.
    /// The default delegates only when both sources are absent; an empty shared plan still
    /// requires source ownership and is therefore unsupported by this default.
    fn admit_text_preparation_with_options<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Self::Prompt>,
        config: TextGenerationConfig,
        controller: &C,
        options: &TextPreparationOptions,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        if options.capture.is_some() || options.interventions.is_some() {
            return Err(BackendFailure::new(
                BackendFailureKind::Unsupported,
                crate::capture::CaptureError::Unsupported(
                    "backend has no original-admission shared instrumentation support".into(),
                ),
            ));
        }
        Self::admit_text_preparation(runtime, input, config, controller)
    }

    /// Quotes the exact borrowed sequence request with all original preparation.
    ///
    /// This opt-in replaces the local admission call, not its quote or readiness
    /// boundary. Include EOS/max and optional capture in the same original plan.
    /// No prompt, sampler or token-slot materialization may precede acceptance.
    /// The default rejects rather than falling back to an unpriced sequence.
    fn admit_text_preparation_with_sequence<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _controller: &C,
        _options: Option<&TextPreparationOptions>,
        _sequence: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            GenerationSequenceAdmissionError::Unsupported,
        ))
    }

    /// Explicit original admission for a sequence with a fixed consumer contribution.
    ///
    /// This replaces the local admission hook in the same Admission phase. An
    /// implementation must seal the actual descriptor in the original quote/bank
    /// and return its consumed association with the provider. Existing sequence
    /// support does not opt in implicitly. No late contribution or second grant.
    fn admit_text_preparation_with_sequence_consumer<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _controller: &C,
        _options: Option<&TextPreparationOptions>,
        _sequence: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            GenerationSequenceAdmissionError::ConsumerUnsupported,
        ))
    }

    /// Explicit original admission for a concrete decoder input and optional
    /// consumer/capture. Take the unique source once before candidate retries;
    /// seal its actual source, N and policy with the same original sequence bank.
    /// Existing sequence/consumer hooks do not implicitly opt in. This creates
    /// no second quote, readiness phase or model operation.
    fn admit_text_preparation_with_decoder<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _controller: &C,
        _options: Option<&TextPreparationOptions>,
        _sequence: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            GenerationSequenceAdmissionError::DecoderUnsupported,
        ))
    }

    /// Explicit admission of one actual decoder plus literal-stop source. The
    /// same original quote must seal the combined program and returned provider.
    /// Suffix-only decoder support defaults to rejection before any preparation.
    fn admit_text_preparation_with_plain_text<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _controller: &C,
        _options: Option<&TextPreparationOptions>,
        _sequence: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            GenerationSequenceAdmissionError::PlainTextUnsupported,
        ))
    }

    /// Consumes the same core claim after original run binding, before Admission
    /// readiness. Extract only the already accepted one-use storage bank; no new
    /// reservation, quote, source replacement, or bytes-plus-guard attachment.
    /// Return an initial retained sequence with its exact sorted EOS policy.
    /// Partial failures must retain their own original custody in the error.
    /// The later cursor calls prepare_storage before its existing first delivery
    /// readiness, and submits no model work until that agreement succeeds.
    fn prepare_generation_sequence_admitted(
        _runtime: &ModelRuntime<Self>,
        _preparation: &Self::TextPreparation,
        _sequence: GenerationSequencePreparation<'_, '_>,
    ) -> Result<crate::generation::RetainedGenerationSequence, BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            GenerationSequenceAdmissionError::Unsupported,
        ))
    }

    /// Installs the exact shared source already priced by original admission.
    ///
    /// Validate its identity against preparation and the original core-issued
    /// context. Use only that preparation's account: no second reservation,
    /// source readmission/copy, policy revision, or late configure hook. Retain
    /// source custody on any aliases escaping the machine. Finish local work
    /// before returning; the driver then agrees Instrumentation exactly once.
    /// Failure/unwind must leave state safely owned through readiness cleanup.
    fn install_text_capture_admitted(
        _runtime: &ModelRuntime<Self>,
        _state: &mut Self::TextGenerationState,
        _preparation: &Self::TextPreparation,
        _source: &crate::capture::SharedCapturePlan,
        _context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            crate::capture::CaptureError::Unsupported(
                "backend has no admitted shared capture installation".into(),
            ),
        ))
    }

    /// Binds admitted preparation to the machine's original run and policy.
    ///
    /// The shared driver calls this once, at attempt zero, before prompt/sampler
    /// creation or mutable run exposure. It is part of the Admission readiness
    /// result. This cold hook must not create native values or advance a
    /// controller. Quoted backends must retain and check this original context;
    /// the default supports unbudgeted preparation and grants no budget proof.
    /// Calling admission directly does not construct authentic run evidence.
    fn bind_text_preparation_run<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _preparation: &Self::TextPreparation,
        _controller: &C,
        _context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }

    /// Acquires lifetime custody before host preparation or destination copying.
    ///
    /// Managed backends validate the exact runtime domain and acquire exclusion
    /// before any host constructor, copy, or callback runs. Acquisition must
    /// release internal locks before returning. Callers retain clones on every
    /// escaping payload owner and destroy payloads before their authority,
    /// including failure and unwind paths. Rejection must precede those writes.
    ///
    /// This grants no finite-byte proof or inference permission. The portable
    /// default supplies no managed-domain evidence. Historical reservations and
    /// an unmanaged value cannot substitute for a managed backend's acquisition.
    fn acquire_host_preparation(
        _runtime: &ModelRuntime<Self>,
    ) -> Result<HostPreparationAuthority, BackendFailure> {
        Ok(HostPreparationAuthority::unmanaged())
    }

    /// Creates an immutable shared filter for a facade-owned tokenizer domain.
    ///
    /// Managed backends must obtain resource authority before invoking the
    /// deferred factory, publish its exact storage, and attach lifetime custody
    /// before returning. This includes models that never start inference. The
    /// default supports backends without a managed storage domain and supplies
    /// no memory-budget evidence. Failure must precede the factory when another
    /// admitted request excludes new unquoted preparation.
    fn prepare_shared_token_filter(
        _runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> TokenFilter,
    ) -> Result<SharedTokenFilter, BackendFailure> {
        Ok(SharedTokenFilter::new(factory()))
    }

    /// Creates one immutable controller source allocation under backend custody.
    ///
    /// Size and vocabulary metadata preflight occurs before this hook. Managed
    /// backends acquire resource authority before invoking the deferred factory,
    /// then register exact capacity and attach custody before returning. Another
    /// admitted request must reject construction before the factory runs.
    ///
    /// The infallible factory transfers all numerical payload it creates into
    /// its returned vector; it must not escape additional unaccounted owners.
    /// Temporary allocations must retire before it returns or unwinds. The
    /// portable default grants no managed-domain or memory-budget evidence.
    fn prepare_shared_controller_bytes(
        _runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> Vec<u8>,
    ) -> Result<SharedControllerBytes, BackendFailure> {
        Ok(SharedControllerBytes::new(factory(), crate::HostPreparationAuthority::unmanaged()))
    }

    /// Creates an immutable declaration under the same source-accounting
    /// protocol as controller bytes. Managed backends reject before `factory`
    /// while admitted work excludes construction, then register the exact
    /// completed capacity before publication. A failed factory may retain a
    /// partial destination: its error must retain construction custody until
    /// that destination retires. This cold hook supplies no finite original
    /// request authority; original copies use their own admitted constructors.
    fn prepare_shared_controller_declaration<T: ControllerDeclarationData>(
        _runtime: &ModelRuntime<Self>,
        factory: impl FnOnce() -> Result<T, BackendFailure>,
    ) -> Result<SharedControllerDeclaration, BackendFailure> {
        Ok(SharedControllerDeclaration::new(factory()?, HostPreparationAuthority::unmanaged()))
    }

    /// Opaque prepared prompt, including any backend-owned multimodal values.
    type Prompt;
    /// Backend-owned generated token handle.
    type Token: TokenOutput<Error = Self::Error>;
    /// Backend-owned sampler and randomness state for one sequence.
    type TextGenerationState;
    /// Exact completion retaining model execution and token sampling.
    type TextCompletion: Completion<Error = Self::Error>;

    /// Clears all request-dependent session state after successful synchronization.
    ///
    /// On success, subsequent generation must behave as on a fresh session with
    /// the same model, prompt and generation configuration. Preserve the loaded
    /// parameters, model identity, selected placement and admitted capabilities.
    /// Any work submitted by reset itself must be settled before success. Never
    /// clear an unresolved submission lease or silently recover a poisoned session.
    /// Stateless backends must explicitly implement this as a no-op.
    fn reset_session(backend: &Self, session: &mut Self::Session) -> Result<(), BackendFailure>;

    /// Explicit original-reset opt-in. Validate the genuine claim against this
    /// exact session, bind selected source/geometry and admit all construction
    /// before work. The caller has not synchronized. Readiness must not allocate,
    /// submit, reap or progress native work before its original acceptance; return
    /// Busy when exact terminal evidence is unavailable. Unknown producer coverage
    /// is unsupported, not completion. An incomplete provider retains this default.
    fn reset_session_admitted(
        _backend: &Self,
        _session: &mut Self::Session,
        _claim: crate::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        Err(BackendFailure::new(
            BackendFailureKind::Unsupported,
            crate::SessionResetRejection::Unsupported,
        ))
    }

    /// Establishes settled, healthy session work without changing request state.
    ///
    /// Wait for all work associated with this session, including loading,
    /// transfers and work retained after cancellation or iterator drop. Return
    /// success only when no pending submission authority remains and the session
    /// is reusable. A live detached continuation/submission may be rejected as
    /// busy. Errors must preserve ownership of unresolved native resources and
    /// must not be treated as completion. Synchronous backends must explicitly
    /// implement the corresponding idle/health check, or a no-op when always idle.
    fn synchronize_session(backend: &Self, session: &Self::Session) -> Result<(), BackendFailure>;

    /// Exact support for serial, completed-token control on this loaded session.
    /// Backends opt in only for verified ordinary single-sequence execution.
    /// Snapshot mechanisms and complete facade-state support are separate facts.
    fn text_execution_control_support(
        _runtime: &ModelRuntime<Self>,
    ) -> crate::execution_control::ControlSupport<&'static str> {
        crate::execution_control::ControlSupport::Unsupported {
            reason: "backend has not declared completed-token control support",
        }
    }

    /// Exact support for prospective temperature changes and explicit reseeding.
    /// The runtime validates policy; the adapter supplies atomic native changes.
    fn text_sampling_control_support(
        _runtime: &ModelRuntime<Self>,
    ) -> crate::execution_control::ControlSupport<&'static str> {
        crate::execution_control::ControlSupport::Unsupported {
            reason: "backend has no prospective sampling controls",
        }
    }

    /// Active parameter-edit provenance for ordinary and controlled generation.
    fn active_parameter_overlay(_runtime: &ModelRuntime<Self>) -> Option<&str> {
        None
    }

    /// Content identity already established during source discovery. This cold
    /// borrowed query must not hash, reopen artifacts or allocate a discovery.
    fn prepared_artifact_identity(_runtime: &ModelRuntime<Self>) -> Option<crate::artifact::ArtifactIdentity> {
        None
    }

    /// Returns genuine mutable points for this exact loaded session.
    fn intervention_discovery(
        _runtime: &ModelRuntime<Self>,
    ) -> Result<crate::intervention::InterventionDiscovery, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no intervention discovery".into(),
        ))
    }

    /// Checks both plans without submitting work or changing model/sampler state.
    fn validate_text_interventions(
        runtime: &ModelRuntime<Self>,
        capture: &crate::capture::AdmittedCapturePlan,
        plan: &crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::capture::CaptureError> {
        if !plan.is_empty() {
            return Err(crate::capture::CaptureError::Unsupported(
                "backend has no text interventions".into(),
            ));
        }
        Self::validate_text_capture(runtime, capture)
    }

    /// Installs immutable plans before the first submission. Diagnostics and
    /// evidence must share capture accounting and ordinary completion ownership.
    fn configure_text_interventions(
        runtime: &ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        capture: crate::capture::AdmittedCapturePlan,
        plan: crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::capture::CaptureError> {
        Self::validate_text_interventions(runtime, &capture, &plan)?;
        Self::configure_text_capture(runtime, state, capture)
    }

    /// Returns observation facts retained from this session's admitted preparation.
    fn capture_discovery(
        _runtime: &ModelRuntime<Self>,
    ) -> Result<crate::capture::CaptureDiscovery, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no bounded capture discovery".into(),
        ))
    }

    /// Ordinary shared-source installation against the actual pending prompt.
    /// This is distinct from original admitted capture and defaults to rejection.
    fn configure_text_prepared_capture(
        _runtime: &ModelRuntime<Self>,
        _state: &mut Self::TextGenerationState,
        _prompt: &Self::Prompt,
        _source: crate::capture::SharedCapturePlan,
    ) -> Result<(), BackendFailure> {
        Err(PreparedControlInputError::InstrumentationUnavailable.into_backend_failure())
    }

    /// Enables an immutable capture plan before the first submission. Implementations
    /// must validate it against this session and reject unsupported combinations.
    fn configure_text_capture(
        _runtime: &ModelRuntime<Self>,
        _state: &mut Self::TextGenerationState,
        plan: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::capture::CaptureError> {
        if plan.is_empty() {
            Ok(())
        } else {
            Err(crate::capture::CaptureError::Unsupported(
                "backend has no bounded text capture".into(),
            ))
        }
    }

    /// Validates known capture costs and execution combinations without submitting
    /// native work or advancing sampler state.
    fn validate_text_capture(
        _runtime: &ModelRuntime<Self>,
        plan: &crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::capture::CaptureError> {
        if plan.is_empty() {
            Ok(())
        } else {
            Err(crate::capture::CaptureError::Unsupported(
                "backend has no bounded text capture".into(),
            ))
        }
    }

    /// Fallibly moves one completed frame without detaching retained custody.
    /// The shared driver establishes exact completion before calling this hook.
    /// Failure leaves undelivered payloads owned by the state. Implementations
    /// with capture must also report pending transactions through
    /// `text_capture_pending`; additional frames require admitted buffering.
    fn try_take_text_capture(
        _state: &mut Self::TextGenerationState,
    ) -> Result<Option<crate::capture::SharedCapturedStep>, Self::Error> {
        Ok(None)
    }

    /// Whether any capture transaction or undelivered frame remains, including
    /// failed/aborted work with no ready frame. This read-only query performs no
    /// completion, allocation, publication or retry. Retained/fallible collectors
    /// must override it; uninstrumented backends have no capture transactions.
    fn text_capture_pending(_state: &Self::TextGenerationState) -> bool {
        false
    }

    /// Explicit opt-in to originally constructed TokenIds, including all other
    /// source/consumer requirements carried by this same genuine claim.
    fn admit_text_preparation_with_token_input<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _controller: &C,
        _options: Option<&TextPreparationOptions>,
        _claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Err(TokenInputRejection::Unsupported.into_backend_failure())
    }

    /// Admits the actual closed prepared source with the same genuine sequence
    /// claim, controller, decoder/consumer and optional capture contributions.
    /// The input variant is not authority: implementations must authenticate
    /// source/selection/pool/current revision before any allocating inspection.
    /// A missing contribution rejects before native work; no ordinary fallback.
    fn admit_text_preparation_with_original_prepared<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _controller: &C,
        _options: Option<&TextPreparationOptions>,
        _claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self::TextPreparation, BackendFailure> {
        Err(PreparedRequestRejection::Unsupported.into_backend_failure())
    }

    /// Takes the input already constructed inside genuine sequence extraction.
    /// No source replacement, allocation permission or extra readiness is supplied.
    fn prepare_original_text_prompt_admitted(
        _backend: &Self,
        _preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, BackendFailure> {
        Err(TokenInputRejection::Unsupported.into_backend_failure())
    }

    /// Creates backend sampling state for one sequence.
    fn start_text_generation(
        backend: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error>;

    /// Initializes sampling within admitted native recovery ownership. A
    /// nontrivial preparation owner must be retained by unresolved native work.
    fn start_text_generation_admitted(
        backend: &Self,
        config: TextGenerationConfig,
        _preparation: &Self::TextPreparation,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        Self::start_text_generation(backend, config)
    }

    /// Whether this actual runtime requires an explicit transport source even
    /// to report failure of original preparation. This is an applicability
    /// refusal condition, never proof of a bound or permission to communicate.
    fn requires_original_preparation_control(_runtime: &ModelRuntime<Self>) -> bool {
        false
    }

    /// Prepare explicit readiness transport from the genuine original request
    /// before model admission. This source has its own complete stage accounting;
    /// later model admission cannot pay for it retroactively. Failure must fence
    /// unresolved work: the driver never falls back to ordinary consensus then.
    fn prepare_text_preparation_control(
        _runtime: &ModelRuntime<Self>,
        _input: &TextPreparationInput<'_, Self::Prompt>,
        _config: TextGenerationConfig,
        _sequence: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Option<Self::TextPreparationControl>, BackendFailure> {
        Ok(None)
    }

    /// Uses the explicit retained source, authenticated against this runtime.
    /// None preserves ordinary behavior; a mismatched Some must fail.
    fn agree_text_preparation_with_control(
        runtime: &ModelRuntime<Self>,
        control: Option<&Self::TextPreparationControl>,
        stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, BackendFailure> {
        if control.is_some() {
            return Err(TokenInputRejection::Unsupported.into_backend_failure());
        }
        Self::agree_text_preparation(runtime, stage, status)
    }

    /// Completes a cold preparation stage using this session's actual authority.
    /// Distributed implementations must agree exact setup, monotone attempt,
    /// stage and disposition under bounded completion before returning readiness.
    /// Failed/indeterminate exchanges fence the shared execution owner. The
    /// default is for single-participant backends and submits no native work.
    fn agree_text_preparation(
        _runtime: &ModelRuntime<Self>,
        _stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, BackendFailure> {
        use crate::run_preparation::{
            TextPreparationOutcome as Outcome, TextPreparationStatus as Status,
        };
        Ok(match status {
            Status::Ready => Outcome::Ready,
            Status::Cancelled => Outcome::Cancelled,
            Status::Failed => Outcome::Rejected { rank: 0 },
        })
    }

    /// Cumulative logical agreement costs of the actual session, not model state.
    fn text_preparation_usage(
        _runtime: &ModelRuntime<Self>,
    ) -> Result<crate::run_preparation::TextPreparationUsage, BackendFailure> {
        Ok(Default::default())
    }

    /// Converts portable tokenizer ids into a backend-owned text prompt.
    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Self::Error>;

    /// Constructs native input under this already admitted request. Backends
    /// with native allocation retain its owner through failed/unknown completion.
    fn prepare_text_prompt_admitted(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
        _preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Self::Error> {
        Self::prepare_text_prompt(backend, prompt_token_ids)
    }

    /// Binds both new and externally prepared input to the admitted request.
    fn bind_text_prompt_preparation(
        _backend: &Self,
        prompt: Self::Prompt,
        _preparation: &Self::TextPreparation,
    ) -> Result<Self::Prompt, Self::Error> {
        Ok(prompt)
    }

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

    /// Prefill with optional exact capture provenance. Backends without domain
    /// observation retain their existing submission behavior.
    fn submit_text_prefill_decision(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        Self::submit_text_prefill(runtime, prompt, decision.filter(), state)
    }

    /// Cooperatively cancels prefill before sampling or committing a token.
    /// `None` certifies agreed cancellation and safe settlement of native work.
    /// Errors remain errors even when cancellation was concurrently requested.
    /// The default preserves indivisible prefill for existing implementations.
    fn submit_text_prefill_cancellable_decision(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        _cancellation: &crate::GenerationCancellationToken,
    ) -> Result<Option<Submission<Self::Token, Self::TextCompletion>>, Self::Error> {
        Self::submit_text_prefill_decision(runtime, prompt, decision, state).map(Some)
    }

    /// Submits prefill under the exact permit acquired before the decision.
    /// Existing backends can keep their decision/cancellation implementation;
    /// accounting backends override this hook to use the supplied authority.
    fn submit_text_prefill_permitted(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        decision: &TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        cancellation: &crate::GenerationCancellationToken,
        _permit: &mut Self::TextStepPermit,
    ) -> Result<Option<Submission<Self::Token, Self::TextCompletion>>, Self::Error> {
        Self::submit_text_prefill_cancellable_decision(
            runtime,
            prompt,
            decision,
            state,
            cancellation,
        )
    }

    /// Submits decode under the exact permit acquired before the decision.
    fn submit_text_decode_permitted(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        decision: &TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
        _permit: &mut Self::TextStepPermit,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        Self::submit_text_decode_decision(runtime, token, decision, state)
    }

    /// Decode with the same pre-override domain contract as prefill.
    fn submit_text_decode_decision(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        decision: &TokenSamplingDecision<'_>,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        Self::submit_text_decode(runtime, token, decision.filter(), state)
    }
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

/// Input for the next ordinary prediction. A committed token may still be the
/// pending decode input and must not be inserted into model state until that step.
pub enum PendingTextInput<P, T> {
    /// The original prepared prompt, before the first model prediction.
    Prefill(P),
    /// The preceding committed canonical token, before its successor is predicted.
    Decode(T),
}

impl<P, T> PendingTextInput<P, T> {
    /// Borrows the pending native input without cloning or evaluating it.
    pub fn as_ref(&self) -> PendingTextInput<&P, &T> {
        match self {
            Self::Prefill(prompt) => PendingTextInput::Prefill(prompt),
            Self::Decode(token) => PendingTextInput::Decode(token),
        }
    }
}

/// Failure from either backend execution or portable constraint control.
#[derive(Debug, thiserror::Error)]
pub enum ControlledTextGenerationError<B, C>
where
    B: std::error::Error + 'static,
    C: std::error::Error + 'static,
{
    /// Cold preparation failed across the selected session's participants.
    #[error("text preparation failed: {0}")]
    Preparation(#[source] BackendFailure),
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
    runtime: &'a mut ModelRuntime<B>,
    inner: TextGenerationMachine<B, C>,
}

struct TextGenerationMachine<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    controller: C,
    step: Option<PendingTextInput<B::Prompt, B::Token>>,
    completions: Vec<B::TextCompletion>,
    // A legacy raw drain cannot detach or copy a retained frame.
    remaining_tokens: Option<usize>,
    // Core-issued evidence is never copied into a snapshot or a child machine.
    step_context: TextStepContext,
    // Stable exclusive borrowed branch tree, independently of fresh resume runs.
    branch_owner: TextRunIdentity,
    branch_fenced: bool,
    // A backend may keep the run's remaining allocation authority here. Retire
    // machine-owned controller/input/completion payloads before closing it.
    // Escaped outputs and unresolved work require independent backend retention.
    prepared_sequence: preparation::PreparedSequence,
    backend_state: B::TextGenerationState,
    // Drop after all ordinary payloads. Unresolved native work independently
    // retains its own clone in backend completion/recovery resources.
    preparation: B::TextPreparation,
    preparation_control: Option<B::TextPreparationControl>,
    // Exact shared source retires after all machine payload and run custody.
    // This ownership is not a grant to copy, reserve, or reconfigure capture.
    capture_source: Option<crate::capture::SharedCapturePlan>,
    intervention_source: Option<crate::intervention::SharedInterventionPlan>,
    // Fresh host-copy custody outlives every resumed machine field. No run grant.
    resume_host: Option<HostPreparationAuthority>,
}

/// Checked fixed controls for the actual shared startup, generation and manual
/// continuation types instantiated by a backend and controller.
///
/// This sums their named constructor/return/error overlaps without constructing
/// them. It excludes nested backend/controller/source payloads, completion-vector
/// backing, cancellation allocations, and formatting. Driver identity uses the
/// non-repeating core evidence issuer without a heap owner. A concrete
/// original producer must add those actual populations before the same admission;
/// this layout alone grants no storage, completeness or execution authority.
pub fn text_generation_control_bytes<B: TextGenerationBackend, C: TokenFilterController>()
-> Option<usize> {
    use std::mem::size_of;
    [
        preparation::control_bytes::<B, C>()?,
        size_of::<TextGenerationMachine<B, C>>(),
        size_of::<TextGeneration<'_, B>>(),
        size_of::<TokenSamplingDecision<'_>>(),
        size_of::<Result<TokenSamplingDecision<'_>, C::Error>>(),
        size_of::<ControlledTextGeneration<'_, B, C>>(),
        size_of::<TextGenerationContinuation<B, C>>(),
        size_of::<TextGenerationDriver<'_, B>>(),
        size_of::<TextContinuationBoundary<'_, '_, B, C>>(),
        size_of::<TextSnapshotSource<'_, B, C>>(),
        size_of::<
            Result<
                ControlledTextGeneration<'_, B, C>,
                ControlledTextGenerationError<B::Error, C::Error>,
            >,
        >(),
        size_of::<
            Result<
                TextGenerationContinuation<B, C>,
                ControlledTextGenerationError<B::Error, C::Error>,
            >,
        >(),
        size_of::<
            Result<
                Option<TextGenerationContinuation<B, C>>,
                ControlledTextGenerationError<B::Error, C::Error>,
            >,
        >(),
        size_of::<ControlledToken<B::Token>>(),
        size_of::<ControlledGenerationResult<B, C>>(),
        size_of::<
            Result<Option<ControlledToken<B::Token>>, TextContinuationError<B::Error, C::Error>>,
        >(),
        size_of::<PendingTextInput<B::Prompt, B::Token>>(),
        size_of::<B::TextStepPermit>(),
        size_of::<TextStepContext>(),
        size_of::<crate::GenerationCancellationToken>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

type ControlledGenerationResult<B, C> = Result<
    (
        <B as TextGenerationBackend>::Token,
        <B as TextGenerationBackend>::TextStepPermit,
    ),
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
    /// Borrows the selected request geometry and historical admission requirement.
    /// This reports the same retained preparation as uninterrupted generation.
    pub fn preparation_report(&self) -> Option<crate::TextPreparationReport<'_>> {
        self.inner.preparation_report()
    }

    /// Starts controlled generation from portable prompt token ids.
    pub fn new(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::from_input(
            runtime,
            TextGenerationInput::TokenIds(prompt_token_ids),
            config,
            controller,
        )
    }

    /// Starts controlled generation from an opaque backend-prepared prompt.
    pub fn from_prompt(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::from_input(
            runtime,
            TextGenerationInput::Prepared(prompt),
            config,
            controller,
        )
    }

    /// Prepares either host tokens or existing input through the same admission.
    pub fn from_input(
        runtime: &'a mut ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let inner = TextGenerationMachine::new(runtime, input, config, controller)?;
        Ok(Self { runtime, inner })
    }

    /// Starts controlled generation with sources included in original admission.
    pub fn new_with_options(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        controller: C,
        options: TextPreparationOptions,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::from_input_with_options(
            runtime,
            TextGenerationInput::TokenIds(prompt_token_ids),
            config,
            controller,
            options,
        )
    }

    /// Starts from prepared input with options included in original admission.
    pub fn from_prompt_with_options(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        controller: C,
        options: TextPreparationOptions,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::from_input_with_options(
            runtime,
            TextGenerationInput::Prepared(prompt),
            config,
            controller,
            options,
        )
    }

    /// Uses the ordinary preparation runner and completes Instrumentation before
    /// exposing a machine, including when the options contain no capture plan.
    pub fn from_input_with_options(
        runtime: &'a mut ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
        options: TextPreparationOptions,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let inner = TextGenerationMachine::new_preparation(
            runtime,
            input,
            config,
            controller,
            Some(options),
        )?;
        Ok(Self { runtime, inner })
    }

    /// Queries the existing controller's termination condition without exposing
    /// mutable policy or revising this run's bound policy identity.
    ///
    /// This preserves `TokenFilterController::is_complete` semantics, including
    /// internal query/cache mutation and its exact error. It grants no step or
    /// completion authority and does not make the continuation quiescent.
    pub fn controller_is_complete(&mut self) -> Result<bool, C::Error> {
        self.inner.controller.is_complete()
    }

    /// Exact immutable source installed by this machine's original preparation.
    pub fn capture_source(&self) -> Option<&crate::capture::SharedCapturePlan> {
        self.inner.capture_source.as_ref()
    }
    /// Exact immutable intervention source, including a fresh resumed revision.
    pub fn intervention_source(&self) -> Option<&crate::intervention::SharedInterventionPlan> {
        self.inner.intervention_source.as_ref()
    }

    /// Borrows canonical constraint state without revising policy or advancing.
    pub fn controller(&self) -> &C {
        &self.inner.controller
    }

    /// Mutably borrows the canonical constraint state.
    pub fn controller_mut(&mut self) -> &mut C {
        self.inner.step_context.revise_policy();
        &mut self.inner.controller
    }

    /// Advances the ordinary machine with cancellation between predictions and
    /// at safe completed prefill spans. Cancellation commits no token and
    /// terminates this source.
    pub fn next_cancellable(
        &mut self,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Option<Result<ControlledToken<B::Token>, ControlledTextGenerationError<B::Error, C::Error>>>
    {
        self.inner.next_committed(self.runtime, cancellation)
    }

    /// Completes additional fallible host preparation before advancing this run.
    pub fn finish_text_preparation<T, E>(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<T, E> {
        if let Err(error) = self.inner.step_context.validate() {
            return match local {
                // Preserve the original local error and its complete custody;
                // no mapping callback or backend agreement runs in this branch.
                Err(local_error) => Err(local_error),
                Ok(value) => {
                    drop(value);
                    Err(map_backend(error))
                }
            };
        }
        self.runtime.finish_text_preparation_control(
            self.inner.preparation_control.as_ref(),
            stage,
            local,
            map_backend,
        )
    }

    /// Agrees initial cancellation/delivery without advancing this run.
    pub fn finish_text_preparation_cancellable<T, E>(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<Option<T>, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        if let Err(error) = self.inner.step_context.validate() {
            return match local {
                // Preserve the original local error and its complete custody;
                // no mapping callback or backend agreement runs in this branch.
                Err(local_error) => Err(local_error),
                Ok(value) => {
                    drop(value);
                    Err(map_backend(error))
                }
            };
        }
        self.runtime.finish_text_preparation_control_cancellable(
            self.inner.preparation_control.as_ref(),
            stage,
            local,
            map_backend,
        )
    }

    /// Installs capture while preserving the existing sampling/controller state.
    pub fn enable_capture(
        &mut self,
        plan: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        self.inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        let local = if !matches!(self.inner.step, Some(PendingTextInput::Prefill(_))) {
            Err(crate::capture::CaptureError::Invalid(
                "capture must be configured before generation".into(),
            ))
        } else {
            // A backend may mutate instrumentation before a later setup error.
            self.inner.step_context.revise_policy();
            self.inner
                .step_context
                .validate()
                .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
            B::configure_text_capture(self.runtime, &mut self.inner.backend_state, plan)
        };
        self.inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        self.runtime.finish_text_preparation_control(
            self.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Capture),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }

    /// Installs one source-bound ordinary prepared-input capture.
    pub fn enable_prepared_capture(
        &mut self,
        source: crate::capture::SharedCapturePlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        self.inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        let local = self.inner.configure_prepared_capture(self.runtime, source);
        self.inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        self.runtime.finish_text_preparation_control(
            self.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Preparation),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }

    /// Installs admitted interventions and captures before ordinary generation.
    pub fn enable_interventions(
        &mut self,
        capture: crate::capture::AdmittedCapturePlan,
        plan: crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        self.inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        let local = if !matches!(self.inner.step, Some(PendingTextInput::Prefill(_))) {
            Err(crate::capture::CaptureError::Invalid(
                "interventions must be configured before generation".into(),
            ))
        } else {
            self.inner.step_context.revise_policy();
            self.inner
                .step_context
                .validate()
                .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
            B::configure_text_interventions(
                self.runtime,
                &mut self.inner.backend_state,
                capture,
                plan,
            )
        };
        self.inner
            .step_context
            .validate()
            .map_err(crate::run_preparation::TextCaptureSetupError::Preparation)?;
        self.runtime.finish_text_preparation_control(
            self.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Capture),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }


}

impl<B, C> TextGenerationMachine<B, C>
where
    B: TextGenerationBackend,
    C: TokenFilterController,
{
    fn preparation_report(&self) -> Option<crate::TextPreparationReport<'_>> {
        B::text_preparation_report(&self.preparation)
    }

    fn configure_prepared_capture(
        &mut self,
        runtime: &ModelRuntime<B>,
        source: crate::capture::SharedCapturePlan,
    ) -> Result<(), BackendFailure> {
        let Some(PendingTextInput::Prefill(prompt)) = self.step.as_ref() else {
            return Err(PreparedControlInputError::SourceMismatch.into_backend_failure());
        };
        self.step_context.revise_policy();
        self.step_context.validate()?;
        B::configure_text_prepared_capture(runtime, &mut self.backend_state, prompt, source)
    }

    fn new(
        runtime: &ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::new_preparation(runtime, input, config, controller, None)
    }

    fn new_preparation(
        runtime: &ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
        options: Option<TextPreparationOptions>,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        Self::new_preparation_with_sequence(runtime, input, config, controller, options, None)
    }

    fn new_preparation_with_sequence(
        runtime: &ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        controller: C,
        options: Option<TextPreparationOptions>,
        sequence: Option<GenerationSequenceRequest<'_>>,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        // Establish payload-before-source order before even issuing the context.
        // The runner subsequently owns the payloads while borrowing this source.
        let mut owned = preparation::InputOwner::<B, C> {
            controller: Some(controller),
            input: Some(input),
            sequence,
            options,
        };
        let step_context = TextStepContext::new();
        step_context
            .validate()
            .map_err(ControlledTextGenerationError::Preparation)?;
        let (
            controller,
            prompt,
            backend_state,
            prepared_sequence,
            preparation,
            preparation_control,
        ) = preparation::prepare(
            runtime,
            owned.input.take().expect("owned preparation input"),
            config,
            owned.controller.take().expect("owned controller"),
            &step_context,
            owned.options.as_ref(),
            owned.sequence.as_ref(),
        )?;
        Ok(Self {
            capture_source: owned.options.as_ref().and_then(|options| options.capture.clone()),
            intervention_source: owned.options.take().and_then(|options| options.interventions),
            resume_host: None,
            prepared_sequence,
            preparation,
            preparation_control,
            backend_state,
            controller,
            step: Some(PendingTextInput::Prefill(prompt)),
            completions: Vec::new(),
            remaining_tokens: config.sampling().max_new_tokens,
            branch_owner: step_context.run_identity().clone(),
            branch_fenced: false,
            step_context,
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

    fn resolve_completions_before_decode(&mut self) -> Result<(), B::Error> {
        // Keep exact handles on error: an empty list must never manufacture
        // completion evidence for a subsequent capture drain or retry.
        for (index, completion) in self.completions.iter().enumerate() {
            let result = match completion.is_complete() {
                Ok(true) => Ok(()),
                Ok(false) => completion.wait(),
                Err(error) => {
                    let _ = completion.wait();
                    Err(error)
                }
            };
            if let Err(error) = result {
                for pending in &self.completions[index + 1..] {
                    let _ = pending.wait();
                }
                return Err(error);
            }
        }
        self.completions.clear();
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn next_committed(
        &mut self,
        runtime: &mut ModelRuntime<B>,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Option<Result<ControlledToken<B::Token>, ControlledTextGenerationError<B::Error, C::Error>>>
    {
        let (token, permit) = match self.next_output(runtime, cancellation)? {
            Ok(output) => output,
            Err(error) => return Some(Err(error)),
        };

        let local = (|| {
            let token_id = token
                .token_id()
                .map_err(ControlledTextGenerationError::Backend)?;
            // Token observation can settle only its readback while the
            // retained completion still owns session submission authority.
            // Controlled commitment requires that exact completion to settle;
            // the move-only permit stays live through this boundary.

            self.resolve_completions_before_decode()
                .map_err(ControlledTextGenerationError::Backend)?;

            self.controller
                .commit_token(token_id)
                .map_err(ControlledTextGenerationError::Controller)?;

            B::finish_text_step(permit).map_err(ControlledTextGenerationError::Backend)?;
            Ok(ControlledToken {
                output: token,
                token_id,
            })
        })();

        let committed = runtime.finish_text_preparation_control(
            self.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Commitment,
            local,
            ControlledTextGenerationError::Preparation,
        );
        if committed.is_err() {
            self.step = None;
        }
        Some(committed)
    }

    fn next_output(
        &mut self,
        runtime: &mut ModelRuntime<B>,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Option<ControlledGenerationResult<B, C>> {
        if self.branch_fenced {
            return Some(Err(ControlledTextGenerationError::Preparation(
                BackendFailure::new(BackendFailureKind::Other, TextBranchFenced))));
        }
        if let Err(error) = self.step_context.validate() {
            self.step = None;
            return Some(Err(ControlledTextGenerationError::Preparation(error)));
        }
        if self.remaining_tokens == Some(0) {
            self.step = None;
            return None;
        }

        let step = self.step.take()?;
        let local = (|| {
            if matches!(step, PendingTextInput::Decode(_)) {
                self.resolve_completions_before_decode()
                    .map_err(ControlledTextGenerationError::Backend)?;
            }

            if self.capture_pending() {
                return Err(ControlledTextGenerationError::Preparation(
                    BackendFailure::from_error(crate::capture::CaptureDeliveryPending),
                ));
            }

            runtime
                .validate_session_admission()
                .map_err(ControlledTextGenerationError::Backend)?;
            if cancellation.is_cancelled() {
                return Ok(None);
            }

            let following_context = self.step_context.following_attempt().ok_or_else(|| {
                ControlledTextGenerationError::Preparation(BackendFailure::from_error(
                    CapabilityError::ArithmeticOverflow {
                        operation: "text prediction attempt ordinal",
                    },
                ))
            })?;

            let permit = B::begin_text_step(
                runtime,
                &self.preparation,
                &self.backend_state,
                &self.controller,
                step.as_ref(),
                &self.step_context,
            )
            .map_err(ControlledTextGenerationError::Backend)?;
            // Issuing local authority consumes the attempt even when a peer
            // subsequently vetoes readiness. Restoration cannot refund it.

            self.step_context = following_context;
            Ok(Some(permit))
        })();
        // Every lane agrees even on local admission/permit failure. Successful
        // local permits stay alive during agreement and drop on peer rejection.

        let mut permit = match runtime.finish_text_preparation_control_cancellable(
            self.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Prediction,
            local,
            ControlledTextGenerationError::Preparation,
        ) {
            Ok(Some(ready)) => ready,
            Ok(None) => {
                // All lanes use the existing prediction boundary, even when
                // locally cancelled. A ready peer drops its unused permit;
                // completed work and local failures retain their disposition.
                cancellation.cancel();
                return None;
            }
            Err(error) => return Some(Err(error)),
        };

        let decision = self
            .controller
            .current_decision()
            .map_err(ControlledTextGenerationError::Controller);

        let decision = match runtime.finish_text_preparation_control(
            self.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Decision,
            decision,
            ControlledTextGenerationError::Preparation,
        ) {
            Ok(decision) => decision,
            Err(error) => return Some(Err(error)),
        };

        let submission = match step {
            PendingTextInput::Prefill(prompt) => B::submit_text_prefill_permitted(
                runtime,
                prompt,
                &decision,
                &mut self.backend_state,
                cancellation,
                &mut permit,
            ),
            PendingTextInput::Decode(token) => B::submit_text_decode_permitted(
                runtime,
                token,
                &decision,
                &mut self.backend_state,
                &mut permit,
            )
            .map(Some),
        };

        drop(decision);
        let submission = match submission {
            Ok(Some(submission)) => submission,
            Ok(None) => {
                // A peer may have requested cancellation. Publish it locally so
                // shared commitment reports cancellation rather than exhaustion.
                cancellation.cancel();
                return None;
            }
            Err(error) => return Some(Err(ControlledTextGenerationError::Backend(error))),
        };

        let token = submission.output;
        if let Err(error) = self.retain_completion(submission.completion) {
            return Some(Err(ControlledTextGenerationError::Backend(error)));
        }

        self.step = Some(PendingTextInput::Decode(token.clone()));
        if let Some(remaining_tokens) = &mut self.remaining_tokens {
            *remaining_tokens -= 1;
        }
        Some(Ok((token, permit)))
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
        self.next_cancellable(&crate::GenerationCancellationToken::new())
    }
}

impl<B, C> Drop for TextGenerationMachine<B, C>
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
/// Every yielded token handle may be fed into the following decode without
/// first reading its id on the host. The preceding completion resolves before
/// that state-mutating decode is submitted, and dropping the iterator waits
/// for every still-retained submission. Drop cannot report settlement errors;
/// use [`ModelRuntime::synchronize`] to establish successful settlement before
/// reuse or eviction, and [`ModelRuntime::reset`] for fresh request state.
pub struct TextGeneration<'a, B: TextGenerationBackend> {
    runtime: &'a mut ModelRuntime<B>,
    inner: TextGenerationMachine<B, FixedTokenFilter>,
}

impl<'a, B: TextGenerationBackend> TextGeneration<'a, B> {
    /// Borrows the selected request geometry and historical admission requirement.
    /// This reports the same retained preparation as controlled generation.
    pub fn preparation_report(&self) -> Option<crate::TextPreparationReport<'_>> {
        self.inner.preparation_report()
    }

    /// Installs ordinary prepared-input capture using the same pending prompt,
    /// policy revision and Instrumentation agreement as controlled generation.
    pub fn enable_prepared_capture(
        &mut self,
        source: crate::capture::SharedCapturePlan,
    ) -> Result<(), BackendFailure> {
        self.inner.step_context.validate()?;
        let local = self.inner.configure_prepared_capture(self.runtime, source);
        self.inner.step_context.validate()?;
        self.runtime.finish_text_preparation_control(
            self.inner.preparation_control.as_ref(),
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local,
            |error| error,
        )
    }
    /// Advances with cancellation between predictions and at safe completed
    /// prefill spans, without producing a synthetic token.
    pub fn next_cancellable(
        &mut self,
        cancellation: &crate::GenerationCancellationToken,
    ) -> Option<Result<B::Token, BackendFailure>> {
        self.inner
            .next_output(self.runtime, cancellation)
            .map(|result| {
                result
                    .and_then(|(token, permit)| {
                        B::finish_text_step(permit).map_err(|error| {
                            self.inner.step = None;
                            ControlledTextGenerationError::Backend(error)
                        })?;
                        Ok(token)
                    })
                    .map_err(unreachable_unconstrained_error::<B>)
            })
    }
    /// Starts generation from portable prompt token ids.
    pub fn new(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
    ) -> Result<Self, BackendFailure> {
        Self::with_token_filter(runtime, prompt_token_ids, config, TokenFilter::All)
    }

    /// Starts ordinary generation with a fixed portable validity restriction.
    /// Tokenizer-owning callers supply their mapped IDs here; sampling, history,
    /// completions and limits use the same machine as controlled generation.
    pub fn with_token_filter(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        filter: TokenFilter,
    ) -> Result<Self, BackendFailure> {
        let inner = TextGenerationMachine::new(
            runtime,
            TextGenerationInput::TokenIds(prompt_token_ids),
            config,
            FixedTokenFilter(filter),
        )
        .map_err(unreachable_unconstrained_error::<B>)?;
        Ok(Self { runtime, inner })
    }

    /// Starts generation from an opaque backend-prepared prompt.
    pub fn from_prompt(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
    ) -> Result<Self, BackendFailure> {
        let inner = TextGenerationMachine::new(
            runtime,
            TextGenerationInput::Prepared(prompt),
            config,
            FixedTokenFilter(TokenFilter::All),
        )
        .map_err(unreachable_unconstrained_error::<B>)?;
        Ok(Self { runtime, inner })
    }
    /// Starts ordinary generation with sources included in original admission.
    pub fn new_with_options(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        options: TextPreparationOptions,
    ) -> Result<Self, BackendFailure> {
        Self::from_input_with_options(
            runtime,
            TextGenerationInput::TokenIds(prompt_token_ids),
            config,
            options,
        )
    }

    /// Starts ordinary generation from prepared input and owned options.
    pub fn from_prompt_with_options(
        runtime: &'a mut ModelRuntime<B>,
        prompt: B::Prompt,
        config: TextGenerationConfig,
        options: TextPreparationOptions,
    ) -> Result<Self, BackendFailure> {
        Self::from_input_with_options(
            runtime,
            TextGenerationInput::Prepared(prompt),
            config,
            options,
        )
    }

    /// Prepares either input form through original admission and the single
    /// Instrumentation readiness stage, even when capture is absent.
    pub fn from_input_with_options(
        runtime: &'a mut ModelRuntime<B>,
        input: TextGenerationInput<B::Prompt>,
        config: TextGenerationConfig,
        options: TextPreparationOptions,
    ) -> Result<Self, BackendFailure> {
        let inner = TextGenerationMachine::new_preparation(
            runtime,
            input,
            config,
            FixedTokenFilter(TokenFilter::All),
            Some(options),
        )
        .map_err(unreachable_unconstrained_error::<B>)?;
        Ok(Self { runtime, inner })
    }

    /// Starts fixed-filter generation with options included in original admission.
    pub fn with_token_filter_and_options(
        runtime: &'a mut ModelRuntime<B>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        filter: TokenFilter,
        options: TextPreparationOptions,
    ) -> Result<Self, BackendFailure> {
        let inner = TextGenerationMachine::new_preparation(
            runtime,
            TextGenerationInput::TokenIds(prompt_token_ids),
            config,
            FixedTokenFilter(filter),
            Some(options),
        )
        .map_err(unreachable_unconstrained_error::<B>)?;
        Ok(Self { runtime, inner })
    }
}

fn unreachable_unconstrained_error<B: BackendProvider>(
    error: ControlledTextGenerationError<B::Error, std::convert::Infallible>,
) -> BackendFailure {
    match error {
        ControlledTextGenerationError::Backend(error) => B::into_backend_failure(error),
        ControlledTextGenerationError::Preparation(error) => error,
        ControlledTextGenerationError::Controller(error) => match error {},
    }
}

impl<B: TextGenerationBackend> Iterator for TextGeneration<'_, B> {
    type Item = Result<B::Token, BackendFailure>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_cancellable(&crate::GenerationCancellationToken::new())
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
    mod capture_delivery_defaults;
    mod preparation_options_defaults;
    mod reset_preparation;
    #[test]
    fn closed_token_sets_intersect_and_project_to_executable_output_width() {
        use super::{TokenFilter, TokenFilterError};
        let valid = TokenFilter::allowed(vec![false, true, false, true]).unwrap();
        assert_eq!(
            valid.allowed_mask_for(6).unwrap().unwrap().as_ref(),
            &[false, true, false, true, false, false]
        );
        assert_eq!(
            valid.allowed_mask_for(3).unwrap().unwrap().as_ref(),
            &[false, true, false]
        );
        assert_eq!(
            valid.allowed_mask_for(1),
            Err(TokenFilterError::NoExecutableToken { output_width: 1 })
        );
        assert_eq!(
            valid.allowed_mask_for(0),
            Err(TokenFilterError::EmptyVocabulary)
        );
        assert!(!valid.allows(4));
        let grammar = TokenFilter::allowed(vec![true, true]).unwrap();
        assert_eq!(
            valid.intersection(&grammar).unwrap(),
            TokenFilter::Allowed(vec![false, true])
        );
        assert_eq!(valid.intersection(&TokenFilter::All).unwrap(), valid);
        assert_eq!(
            valid.intersection(&TokenFilter::Allowed(vec![true])),
            Err(TokenFilterError::NoAllowedToken)
        );
        assert!(TokenFilter::Allowed(vec![]).allowed_mask_for(2).is_err());
    }

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
        let greedy = crate::generation::resolve_generation_config(
            None,
            crate::generation::GenerationConfigOverrides {
                do_sample: Some(false),
                temperature: Some(0.8),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(
            TextGenerationConfig::new(greedy).with_mirostat_v2(5.0, 0.1),
            Err(GenerationError::InvalidMirostatTemperature(0.0))
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
        selections: std::sync::atomic::AtomicUsize,
        materializations: std::sync::atomic::AtomicUsize,
        selected_gguf: std::cell::RefCell<Option<eredu_gguf::Checkpoint>>,
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
            if let Some(checkpoint) = plan.inspection().gguf_checkpoint() {
                let selected = self.selected_gguf.borrow();
                for (before, after) in selected
                    .as_ref()
                    .unwrap()
                    .shards()
                    .iter()
                    .zip(checkpoint.shards())
                {
                    assert!(std::ptr::eq(
                        before.prepared_header().unwrap(),
                        after.prepared_header().unwrap(),
                    ));
                }
            }
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
        type SelectedPreparation = (u32, crate::PreparationAdmission);
        type ConfigurationResolver = LoadingConfigurationResolver;

        fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
            &LOADING_CONFIGURATION_RESOLVER
        }

        fn select_preparation(
            &self,
            inspection: &ArtifactInspection,
            options: &Self::LoadOptions,
        ) -> Result<Self::SelectedPreparation, Self::Error> {
            self.selections
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *self.selected_gguf.borrow_mut() = inspection.gguf_checkpoint().cloned();
            let policy = crate::PreparationPolicy::default().with_required_session_capabilities(
                SessionCapabilities::default().with_activation_inspection(*options == 99),
            );
            let request = crate::PreparationAdmissionRequest::new(
                crate::LoadingProtocol::Model,
                inspection.format(),
                policy,
                crate::ArchitecturePreparationCapabilities::new(
                    false,
                    true,
                    false,
                    false,
                    false,
                    crate::InputModalities::TEXT,
                ),
            );
            let admission = crate::admit_preparation(
                request,
                crate::PreparationMechanismCapabilities::new(true, true)
                    .with_residency(crate::ResidencyRequest::FullyResident, true)
                    .with_input_modalities(crate::InputModalities::TEXT)
                    .with_session(
                        SessionCapabilities::default().with_activation_inspection(*options == 99),
                    ),
            )
            .expect("mock admission facts are coherent");
            Ok((*options, admission))
        }

        fn selected_preparation_admission(
            &self,
            selected: &Self::SelectedPreparation,
        ) -> crate::PreparationAdmission {
            selected.1
        }

        fn model_config(
            &self,
            selected: SelectedModelPreparation<Self>,
        ) -> Result<Self::ModelConfig, Self::Error> {
            let (plan, (selected, _admission)) = selected.into_parts();
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
        type TextPreparation = ();
        type TextPreparationControl = ();
        type TextStepPermit = ();
        fn begin_text_step<C: TokenFilterController>(
            _: &ModelRuntime<Self>,
            _: &Self::TextPreparation,
            _: &Self::TextGenerationState,
            _: &C,
            _: PendingTextInput<&Self::Prompt, &Self::Token>,
            _: &TextStepContext,
        ) -> Result<Self::TextStepPermit, Self::Error> {
            Ok(())
        }
        fn finish_text_step(_: Self::TextStepPermit) -> Result<(), Self::Error> {
            Ok(())
        }

        fn admit_text_preparation<C: TokenFilterController>(
            _: &ModelRuntime<Self>,
            _: &TextPreparationInput<'_, Self::Prompt>,
            _: TextGenerationConfig,
            _: &C,
        ) -> Result<(), BackendFailure> {
            Ok(())
        }
        fn reset_session(_: &Self, session: &mut Self::Session) -> Result<(), BackendFailure> {
            session.tokens.clear();
            Ok(())
        }

        fn synchronize_session(_: &Self, _: &Self::Session) -> Result<(), BackendFailure> {
            Ok(())
        }

        type Prompt = Vec<u32>;
        type Token = u32;
        type TextGenerationState = (u32, u64);
        type TextCompletion = Done;

        fn try_take_text_capture(
            _: &mut Self::TextGenerationState,
        ) -> Result<Option<crate::capture::SharedCapturedStep>, Self::Error> {
            Ok(capture_delivery_defaults::take())
        }
        fn text_capture_pending(_: &Self::TextGenerationState) -> bool {
            capture_delivery_defaults::pending()
        }

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
                        vec![
                            crate::cache::LayerCachePolicy::key_only(
                                crate::AttentionPolicy::Full,
                                1,
                                2,
                            )
                            .unwrap(),
                        ],
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
    fn generic_path_loader_retains_first_gguf_headers_and_lazy_nonzero_payload() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let bytes: Vec<u8> = [1.25f32, -3.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let metadata = std::collections::BTreeMap::from([(
            "general.architecture".into(),
            eredu_gguf::MetadataValue::String("llama".into()),
        )]);
        eredu_gguf::Writer::default()
            .write(
                std::fs::File::create(&path).unwrap(),
                &metadata,
                &[eredu_gguf::TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[2],
                    ggml_type: eredu_gguf::GgmlType::F32,
                    data: &bytes,
                }],
            )
            .unwrap();
        let backend = LoadingMock::default();
        assert_eq!(*load_model(&backend, &path, 41).unwrap(), 41);
        assert_eq!(
            backend
                .selections
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            backend
                .materializations
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let retained = backend.selected_gguf.borrow_mut().take().unwrap();
        assert!(
            retained
                .shards()
                .iter()
                .all(|shard| shard.prepared_header().is_some())
        );
        let mut materializer = retained.into_materializer();
        let converted = materializer.converted_tensor("token_embd.weight").unwrap();
        let eredu_gguf::ConvertedTensor::Dense(dense) = converted.converted() else {
            panic!("expected dense fixture");
        };
        assert_eq!(dense.shape, [2]);
        assert_eq!(dense.data, bytes);
    }

    #[test]
    fn session_requirement_is_retained_by_the_single_admission() {
        let root = tempfile::tempdir().unwrap();
        write_loading_fixture(root.path());
        let backend = LoadingMock::default();

        let prepared = load_model(&backend, root.path(), 99).unwrap();

        assert_eq!(*prepared, 99);
        assert_eq!(
            backend
                .selections
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            backend
                .materializations
                .load(std::sync::atomic::Ordering::Relaxed),
            1
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
    fn lifecycle_preserves_selected_target_and_loaded_parameters() {
        let prepared = Mock.prepare_model(10).unwrap();
        let mut runtime =
            ModelRuntime::from_prepared_execution_plan_target(Mock, prepared, 42).unwrap();
        let capabilities = runtime.capabilities();
        assert_eq!(runtime.prefill(vec![1, 2]).unwrap().output, 12);
        runtime.synchronize().unwrap();
        assert_eq!(runtime.decode(3).unwrap().output, 13);
        runtime.reset().unwrap();
        assert_eq!(runtime.execution_plan_target_id(), Some(42));
        assert_eq!(runtime.capabilities(), capabilities);
        assert_eq!(runtime.prefill(vec![1, 2]).unwrap().output, 12);
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

    fn continuation_config(limit: usize) -> TextGenerationConfig {
        TextGenerationConfig::new(
            crate::resolve_generation_config(
                None,
                crate::GenerationConfigOverrides {
                    max_new_tokens: Some(limit),
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    }

    #[test]
    fn detached_ordinary_continuation_preserves_pending_input_and_commit_order() {
        let controller = || FixedController {
            tokens: vec![7, 8, 9],
            committed: 0,
        };
        let mut ordinary_runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let ordinary: Vec<_> = ControlledTextGeneration::new(
            &mut ordinary_runtime,
            vec![1, 2],
            continuation_config(3),
            controller(),
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect();

        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start(vec![1, 2], continuation_config(3), controller())
            .unwrap();
        state.require_quiescent().unwrap();
        assert!(state.is_prefill_pending());
        assert!(driver.runtime().session().tokens.is_empty());
        let mut actual = Vec::new();
        for (index, expected) in ordinary.iter().enumerate() {
            let token = driver.advance(&mut state).unwrap().unwrap().token_id();
            actual.push(token);
            assert_eq!(token, *expected);
            assert_eq!(state.controller().committed, index + 1);
            assert_eq!(state.remaining_tokens(), Some(2 - index));
            // The newest token has been committed but is not in model state.
            let mut model_inputs = vec![1, 2];
            model_inputs.extend_from_slice(&ordinary[..index]);
            assert_eq!(driver.runtime().session().tokens, model_inputs);
            assert!(!state.is_prefill_pending());
            assert!(matches!(
                driver.advance(&mut state),
                Err(TextContinuationError::NotQuiescent)
            ));
            assert_eq!(state.controller().committed, index + 1);
            assert!(driver.take_completed_delivery(&mut state).unwrap().is_none());
            state.require_quiescent().unwrap();
        }
        assert!(driver.advance(&mut state).unwrap().is_none());
        assert_eq!(actual, ordinary);
        assert_eq!(
            driver.runtime().session().tokens,
            ordinary_runtime.session().tokens
        );
    }

    #[test]
    fn detached_continuation_cannot_attach_to_another_driver() {
        let mut first = ModelRuntime::prepare(Mock, 10).unwrap();
        let mut other = ModelRuntime::prepare(Mock, 10).unwrap();
        let mut owner = TextGenerationDriver::new(&mut first);
        let mut state = owner
            .start(
                vec![1, 2],
                continuation_config(2),
                FixedTokenFilter(TokenFilter::All),
            )
            .unwrap();
        let mut foreign = TextGenerationDriver::new(&mut other);
        assert!(matches!(
            foreign.advance(&mut state),
            Err(TextContinuationError::IncompatibleDriver)
        ));
        assert!(foreign.runtime().session().tokens.is_empty());
        assert!(owner.advance(&mut state).unwrap().is_some());
        assert!(matches!(
            foreign.take_completed_delivery(&mut state),
            Err(TextContinuationError::IncompatibleDriver)
        ));
        owner.take_completed_delivery(&mut state).unwrap();
        drop(owner);
        let mut replacement = TextGenerationDriver::new(&mut first);
        assert!(matches!(
            replacement.advance(&mut state),
            Err(TextContinuationError::IncompatibleDriver)
        ));
        assert_eq!(replacement.runtime().session().tokens, vec![1, 2]);
    }

    #[test]
    fn detached_continuation_failure_remains_fenced_after_draining() {
        struct RejectCommit;
        impl TokenFilterController for RejectCommit {
            type Error = std::io::Error;
            fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
                Ok(TokenFilter::All)
            }
            fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
                Err(std::io::Error::other("commit rejected"))
            }
            fn is_complete(&mut self) -> Result<bool, Self::Error> {
                Ok(false)
            }
        }
        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start(vec![1, 2], continuation_config(3), RejectCommit)
            .unwrap();
        assert!(matches!(
            driver.advance(&mut state),
            Err(TextContinuationError::Generation(
                ControlledTextGenerationError::Controller(_)
            ))
        ));
        driver.take_completed_delivery(&mut state).unwrap();
        assert!(matches!(
            state.require_quiescent(),
            Err(TextContinuationError::Failed)
        ));
        assert!(matches!(
            driver.advance(&mut state),
            Err(TextContinuationError::Failed)
        ));
        assert_eq!(driver.runtime().session().tokens, vec![1, 2]);
    }

    #[test]
    fn detached_continuation_caught_unwind_cannot_be_resumed() {
        struct PanickingFilter;
        impl TokenFilterController for PanickingFilter {
            type Error = Infallible;
            fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
                panic!("filter failed while preparing the decision")
            }
            fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
                Ok(())
            }
            fn is_complete(&mut self) -> Result<bool, Self::Error> {
                Ok(false)
            }
        }
        let mut runtime = ModelRuntime::prepare(Mock, 10).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start(vec![1, 2], continuation_config(3), PanickingFilter)
            .unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driver.advance(&mut state)
            }))
            .is_err()
        );
        driver.take_completed_delivery(&mut state).unwrap();
        assert!(matches!(
            state.require_quiescent(),
            Err(TextContinuationError::Failed)
        ));
        assert!(matches!(
            driver.advance(&mut state),
            Err(TextContinuationError::Failed)
        ));
        assert!(driver.runtime().session().tokens.is_empty());
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
        assert!(
            serde_json::from_str::<DistributedSessionDescriptor>(
                r#"{"world_size":6,"rank":6,"groups":[]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn distributed_commit_epoch_round_trips_and_rejects_zero() {
        let outcome = DistributedCommitOutcome::Indeterminate {
            epoch: DistributedCommitEpoch::new(17).unwrap(),
            phase: DistributedCommitPhase::DecisionCompletion,
        };
        let encoded = serde_json::to_string(&outcome).unwrap();
        assert_eq!(
            serde_json::from_str::<DistributedCommitOutcome>(&encoded).unwrap(),
            outcome
        );
        assert!(serde_json::from_str::<DistributedCommitEpoch>("0").is_err());
    }
}
