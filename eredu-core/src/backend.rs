//! High-level contract implemented once per execution backend.

mod continuation;
mod failure;
pub mod local_text;
mod steps;
pub use continuation::{
    TextContinuationBoundary, TextContinuationError, TextContinuationIdentity, TextDriverIdentity,
    TextGenerationContinuation, TextGenerationDriver,
};
pub use failure::{BackendFailure, BackendFailureKind};
mod decision_source;
use decision_source::TextRequestDriver;
pub use steps::{
    TextAdvancement, TextOperationError, TextOperationKind, TextOperationOutput,
    TextOperationPolicyError, TextProgress,
};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

use crate::{
    artifact::{
        inspect_artifact, ArtifactError, ArtifactInspection, ModelConfigurationResolver,
        ModelPreparationPlan,
    },
    capability::{
        CapabilityError, InputTokenCount, ModelCapabilities, RuntimeStateEstimate,
        StaticMemoryReport,
    },
    checkpoint::TensorDtype,
    generation::{GenerationError, ResolvedGenerationConfig},
    media::TokenizedMultimodalRequest,
    observation::ObservationSet,
    PreparationAdmission,
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

/// Submits backend-native work and retains values through exact completion.
pub trait SubmissionBackend: eredu_nn::NeuralBackend {
    /// Backend executor, queue, stream, or equivalent submission context.
    type Executor: ?Sized;
    /// Owned executor used for an independently schedulable graph lane.
    type OwnedExecutor: std::borrow::Borrow<Self::Executor>;
    /// Exact completion object for one submission.
    type Completion: Completion;

    /// Creates independently schedulable executors on the same backend device.
    fn fork_executors(
        executor: &Self::Executor,
        count: usize,
    ) -> Result<Vec<Self::OwnedExecutor>, <Self::Completion as Completion>::Error>;

    /// Submits evaluation of backend-native values on one executor.
    fn submit<'a, I>(
        executor: &Self::Executor,
        values: I,
    ) -> Result<Self::Completion, <Self::Completion as Completion>::Error>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>;

    /// Orders future work on `executor` after an exact producer completion.
    fn order_after(
        completion: &Self::Completion,
        executor: &Self::Executor,
    ) -> Result<(), <Self::Completion as Completion>::Error>;

    /// Retains an owned value until `completion` has completed exactly.
    fn retain_until_complete<T: Send + 'static>(
        executor: &Self::Executor,
        completion: &Self::Completion,
        value: T,
    ) -> Result<(), <Self::Completion as Completion>::Error>;
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
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
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

/// Native completion of a text invocation. The scope must retain the resource
/// root and submission lease through failure, unwinding, and cancellation.
pub trait TextSubmissionScope<O, E>: Sized {
    /// Completion of this exact invocation.
    type Completion: Completion<Error = E>;

    /// Keeps the resource root alive through a dropped or unwound invocation.
    fn abandon(self, root: std::rc::Rc<dyn std::any::Any>);

    /// Seals and publishes the invocation into its native completion.
    fn finish(
        self,
        root: std::rc::Rc<dyn std::any::Any>,
        output: Result<O, E>,
    ) -> Result<Submission<O, Self::Completion>, E>;

    /// Settles and checks token-validation work before publishing synchronous state.
    fn validate_tokens(&mut self) -> Result<(), E>;

    /// Establishes completion of a synchronous state transaction.
    fn settle<T>(self, root: std::rc::Rc<dyn std::any::Any>, result: Result<T, E>) -> Result<T, E>;
}

/// Native submission authority for synchronous transactions on an installed model.
pub trait TextTransactionMechanisms: crate::session_authority::SessionRetirement {
    /// Scores retained by the native submission scope.
    type Scores;
    /// Exact native scope for settle or abandonment.
    type Scope<'a>: TextSubmissionScope<Self::Scores, Self::Error>
    where
        Self: 'a;

    /// Opens a scope without entering any model workflow.
    fn open_transaction(
        &self,
        lease: crate::SubmissionLease,
    ) -> Result<Self::Scope<'_>, Self::Error>;

    /// Translates a portable rejection into the backend's operation error.
    fn transaction_error(error: TextTransactionError) -> Self::Error;
}

/// Preflight rejection before any transaction may mutate the executable.
#[derive(Debug, thiserror::Error)]
pub enum TextTransactionError {
    /// The resource root is shared by outstanding work.
    #[error("installed text resources are still retained by a submission")]
    Busy,
    /// A preparation result failed to propagate the local rejection.
    #[error("model operation preparation accepted absent submission authority")]
    AbsentAuthority,
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

/// Native backend identity, device capabilities, and prepared model resources.
pub trait BackendProvider: Sized {
    /// Backend error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Backend identity.
    fn descriptor(&self) -> BackendDescriptor;
    /// Discovers devices and their fail-closed capabilities.
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error>;
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

/// Portable projection of separately supplied policy inputs and native facts.
pub trait AdmissionFacts: Sized {
    /// Inputs available without inspecting or materializing an artifact.
    type Inputs;

    /// Forms the facts consumed by a portable admission implementation.
    fn from_inputs(inputs: Self::Inputs) -> Self;
}

/// Portable cold selection for a directly loaded artifact.
pub trait ModelLoadAdmission<P> {
    /// Architecture-owned source preparation associated with this admission.
    type SourcePreparation: ModelSourcePreparation<P>;
    /// Reason portable selection failed.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Projects the admitted source layout using cold mechanism facts before selection.
    fn project_inspection(
        &self,
        inspection: ArtifactInspection<P>,
    ) -> Result<ArtifactInspection<P>, ArtifactError> {
        Ok(inspection)
    }

    /// Selects a preparation for this inspection without materializing sources.
    fn select(
        self,
        inspection: &ArtifactInspection<P>,
    ) -> Result<Self::SourcePreparation, Self::Error>;
}

/// Portable transformation of an admitted artifact into prepared sources.
pub trait ModelSourcePreparation<P> {
    /// Exact sources supplied to backend binding.
    type Prepared;

    /// Failure to prepare the admitted sources.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the preparation admission retained before materialization.
    fn admission(&self) -> PreparationAdmission;

    /// Prepares sources from the exact retained plan and source selection.
    fn prepare(self, plan: ModelPreparationPlan<P>) -> Result<Self::Prepared, Self::Error>;
}

/// A portable prepared source that can bind to a backend's model configuration.
pub trait BindModelSources<B: ModelLoadingBackend> {
    /// Executable and retained resources produced by this source binding.
    type Model;

    /// Constructs the selected executable with the backend's bound mechanisms.
    fn construct_model(
        self,
        backend: &B,
        placement: B::Binding,
    ) -> Result<PreparedModel<Self::Model>, BackendFailure>;
}

/// Loaded model assembled by the source preparation selected for an artifact.
pub type LoadedModelFor<B, P, S> =
    <<S as ModelSourcePreparation<P>>::Prepared as BindModelSources<B>>::Model;

/// Portable source preparation paired with opaque native placement.
#[derive(Debug, Clone)]
pub struct SelectedModelBinding<S, N> {
    source: S,
    native: N,
}

impl<S, N> SelectedModelBinding<S, N> {
    /// Pairs a portable selection with its native placement.
    pub(crate) const fn new(source: S, native: N) -> Self {
        Self { source, native }
    }

    /// Borrows the selected native placement.
    pub const fn native(&self) -> &N {
        &self.native
    }

    /// Consumes the pair for source preparation and native binding.
    pub fn into_parts(self) -> (S, N) {
        (self.source, self.native)
    }
}

impl<S, N> AsRef<S> for SelectedModelBinding<S, N> {
    fn as_ref(&self) -> &S {
        &self.source
    }
}

/// Cold direct-load request and independent backend device assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLoadOptions<R, P> {
    request: R,
    placement: Option<P>,
}

impl<R: Default, P> Default for ModelLoadOptions<R, P> {
    fn default() -> Self {
        Self::new(R::default())
    }
}

impl<R, P> ModelLoadOptions<R, P> {
    /// Selects the portable request for an already realized backend.
    pub const fn new(request: R) -> Self {
        Self {
            request,
            placement: None,
        }
    }

    /// Supplies the native device assignment independently of portable rank policy.
    pub fn with_placement(mut self, placement: P) -> Self {
        self.placement = Some(placement);
        self
    }

    /// Borrows the portable request.
    pub const fn request(&self) -> &R {
        &self.request
    }

    /// Borrows the optional native device assignment.
    pub const fn placement(&self) -> Option<&P> {
        self.placement.as_ref()
    }
}

/// Portable facts projected from a normalized request and exact mechanism inputs.
pub trait DirectLoadFacts {
    /// Normalized policy supplied by the caller.
    type Request;
    /// Exact side-effect-free mechanism descriptions.
    type Mechanisms;

    /// Reads the admitted portable rank without opening native resources.
    fn rank(request: &Self::Request) -> Option<crate::ParallelRankTopology>;

    /// Combines the portable policy and backend's independent mechanism facts.
    fn from_request(request: Self::Request, mechanisms: Self::Mechanisms) -> Self;
}

/// Native mechanisms used by a portable model-loading policy.
pub trait ModelLoadingBackend: BackendProvider {
    /// Exact device assignment selected by a distributed caller.
    type LoadPlacement;

    /// Native inputs retained separately from the portable source selection.
    type Binding;

    /// Side-effect-free mechanisms parameterizing the portable catalog's facts.
    type LoadMechanisms;

    /// Supplies mechanism facts without consuming model policy.
    fn load_mechanisms(&self) -> Self::LoadMechanisms;

    /// Validates and retains an exact rank and matching native device assignment.
    fn bind_load_placement(
        &self,
        rank: Option<crate::ParallelRankTopology>,
        placement: Option<Self::LoadPlacement>,
    ) -> Result<Self::Binding, Self::Error>;

    /// Validates native placement before portable source preparation begins.
    fn validate_model_binding(&self, _binding: &Self::Binding) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// One neutral preparation plan inseparably paired with its backend selection.
///
/// Core creates this value only after policy, architecture, and session-capability
/// admission have succeeded against the same artifact inspection. Backend adapters
/// may consume the pair, but callers cannot substitute a different plan after
/// selection.
pub struct SelectedModelPreparation<B: ModelLoadingBackend, P, S> {
    plan: ModelPreparationPlan<P>,
    selected: SelectedModelBinding<S, B::Binding>,
}

impl<B: ModelLoadingBackend, P, S> SelectedModelPreparation<B, P, S> {
    pub(crate) fn new(
        plan: ModelPreparationPlan<P>,
        selected: SelectedModelBinding<S, B::Binding>,
    ) -> Self {
        Self { plan, selected }
    }

    /// Borrows the exact admitted artifact and preparation policy.
    pub const fn plan(&self) -> &ModelPreparationPlan<P> {
        &self.plan
    }

    /// Borrows the selected source and placement without changing admission.
    pub const fn selected(&self) -> &SelectedModelBinding<S, B::Binding> {
        &self.selected
    }

    /// Consumes the binding into the exact admitted plan and backend selection.
    pub fn into_parts(self) -> (ModelPreparationPlan<P>, SelectedModelBinding<S, B::Binding>) {
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
    /// Portable cold selection failed.
    #[error("failed to select model preparation: {0}")]
    Selection(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The portable source preparation failed.
    #[error("failed to prepare checkpoint sources: {0}")]
    SourcePreparation(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The selected backend failed policy resolution or materialization.
    #[error("selected backend failed to prepare the model: {0}")]
    Backend(#[source] E),
    /// Constructing the executable from its prepared sources failed.
    #[error(transparent)]
    Construction(BackendFailure),
    /// The inspected architecture/load-policy/topology route lacks a requirement.
    #[error(transparent)]
    SessionCapability(#[from] SessionCapabilityError),
}

/// Inspects, plans, and prepares one artifact on the selected backend.
///
/// The caller supplies the resolver. The backend instance already owns its
/// device, queues, and optional distributed communication state.
pub fn load_model<B: ModelLoadingBackend, P, A>(
    backend: &B,
    artifact: impl AsRef<Path>,
    resolver: &impl ModelConfigurationResolver<ArtifactPlan = P>,
    options: ModelLoadOptions<A::Request, B::LoadPlacement>,
) -> Result<PreparedModel<LoadedModelFor<B, P, A::SourcePreparation>>, ModelLoadError<B::Error>>
where
    P: Clone + std::fmt::Debug,
    A: DirectLoadFacts<Mechanisms = B::LoadMechanisms>
        + ModelLoadAdmission<
            P,
            SourcePreparation: ModelSourcePreparation<P, Prepared: BindModelSources<B>>,
        >,
{
    let inspection = inspect_artifact(artifact, resolver)?;
    prepare_inspected_model::<B, P, A>(backend, inspection, options)
}

/// Plans and prepares an artifact that the caller has already inspected.
///
/// This is the canonical lower-level entry point for facade loaders which
/// must derive tokenizer, chat, or other portable sidecar state from the same
/// inspection before transferring ownership to the selected backend.
pub fn prepare_inspected_model<B: ModelLoadingBackend, P, A>(
    backend: &B,
    inspection: ArtifactInspection<P>,
    options: ModelLoadOptions<A::Request, B::LoadPlacement>,
) -> Result<PreparedModel<LoadedModelFor<B, P, A::SourcePreparation>>, ModelLoadError<B::Error>>
where
    P: Clone + std::fmt::Debug,
    A: DirectLoadFacts<Mechanisms = B::LoadMechanisms>
        + ModelLoadAdmission<
            P,
            SourcePreparation: ModelSourcePreparation<P, Prepared: BindModelSources<B>>,
        >,
{
    let selected = select_inspected_model::<B, P, A>(backend, inspection, options)?;
    prepare_selected_model(backend, selected)
}

/// Selects an inspected model before opening checkpoint payloads.
pub fn select_inspected_model<B: ModelLoadingBackend, P, A>(
    backend: &B,
    inspection: ArtifactInspection<P>,
    options: ModelLoadOptions<A::Request, B::LoadPlacement>,
) -> Result<SelectedModelPreparation<B, P, A::SourcePreparation>, ModelLoadError<B::Error>>
where
    P: Clone + std::fmt::Debug,
    A: DirectLoadFacts<Mechanisms = B::LoadMechanisms> + ModelLoadAdmission<P>,
{
    let ModelLoadOptions { request, placement } = options;
    let rank = A::rank(&request);
    let facts = A::from_request(request, backend.load_mechanisms());
    let inspection = facts.project_inspection(inspection)?;
    let binding = backend
        .bind_load_placement(rank, placement)
        .map_err(ModelLoadError::Backend)?;
    let selected = facts
        .select(&inspection)
        .map_err(|error| ModelLoadError::Selection(Box::new(error)))?;
    let admission = selected.admission();
    let selected = SelectedModelBinding::new(selected, binding);
    let plan = ModelPreparationPlan::from_retained_admission(inspection, admission)?;
    Ok(SelectedModelPreparation::new(plan, selected))
}

/// Materializes an already planned artifact with its retained backend selection.
///
/// Selection and capability admission must have completed before the backend
/// instance was realized. This entry point therefore performs no policy or
/// route selection after native resources exist.
pub fn prepare_selected_model<B: ModelLoadingBackend, P, S>(
    backend: &B,
    selected: SelectedModelPreparation<B, P, S>,
) -> Result<PreparedModel<LoadedModelFor<B, P, S>>, ModelLoadError<B::Error>>
where
    P: Clone + std::fmt::Debug,
    S: ModelSourcePreparation<P, Prepared: BindModelSources<B>>,
{
    let (plan, selected) = selected.into_parts();
    let (source_preparation, binding) = selected.into_parts();
    backend
        .validate_model_binding(&binding)
        .map_err(ModelLoadError::Backend)?;
    let sources = source_preparation
        .prepare(plan)
        .map_err(|error| ModelLoadError::SourcePreparation(Box::new(error)))?;
    sources
        .construct_model(backend, binding)
        .map_err(ModelLoadError::Construction)
}

/// Installs an admitted model on an already selected backend.
pub trait ModelSessionInstaller<B: BackendProvider> {
    /// Prepared model consumed by this installer.
    type Model;

    /// Session chosen by the caller's execution composition.
    type Session: BackendSession<B>;

    /// Installs the prepared model and settles initial session work.
    fn install_session(
        backend: &B,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, B::Error>;
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

    /// Reports the exact speculative support of this installed target.
    fn speculative_capability(&self) -> crate::SpeculativeCapability {
        crate::SpeculativeCapability::Unavailable
    }

    /// Returns the installed target identity for binding a separate draft.
    fn draft_target_identity(&self) -> Option<&str> {
        None
    }

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

/// Lifecycle and cold readiness of an installed text session.
pub trait TextSessionLifecycle<B: BackendProvider>: BackendSession<B> {
    /// Clears all request-dependent session state after successful synchronization.
    ///
    /// On success, subsequent generation must behave as on a fresh session with
    /// the same model, prompt and generation configuration. Preserve the loaded
    /// parameters, model identity, selected placement and admitted capabilities.
    /// Any work submitted by reset itself must be settled before success. Never
    /// clear an unresolved submission lease or silently recover a poisoned session.
    /// Stateless backends must explicitly implement this as a no-op.
    fn reset_session(&mut self, backend: &B) -> Result<(), BackendFailure>;

    /// Establishes settled, healthy session work without changing request state.
    ///
    /// Wait for all work associated with this session, including loading,
    /// transfers and work retained after cancellation or iterator drop. Return
    /// success only when no pending submission authority remains and the session
    /// is reusable. A live detached continuation/submission may be rejected as
    /// busy. Errors must preserve ownership of unresolved native resources and
    /// must not be treated as completion. Synchronous backends must explicitly
    /// implement the corresponding idle/health check, or a no-op when always idle.
    fn synchronize_session(&self, backend: &B) -> Result<(), BackendFailure>;

    /// Completes a cold preparation stage using this session's actual authority.
    /// Distributed implementations must agree exact setup, monotone attempt,
    /// stage and disposition under bounded completion before returning readiness.
    /// Failed/indeterminate exchanges fence the shared execution owner. The
    /// default is for single-participant backends and submits no native work.
    fn agree_text_preparation(
        &self,
        _backend: &B,
        _stage: crate::run_preparation::TextPreparationStage,
        status: crate::run_preparation::TextPreparationStatus,
    ) -> Result<crate::run_preparation::TextPreparationOutcome, BackendFailure> {
        Ok(status.local_outcome())
    }

    /// Cumulative logical agreement costs of the actual session, not model state.
    fn text_preparation_usage(
        &self,
        _backend: &B,
    ) -> Result<crate::run_preparation::TextPreparationUsage, BackendFailure> {
        Ok(Default::default())
    }
}

/// One submission produced by the selected backend session.
pub type SessionSubmission<B, S> =
    Submission<<S as BackendSession<B>>::Output, <S as BackendSession<B>>::Completion>;

/// A prepared model, its selected backend, and its installed execution session.
///
/// This is the canonical client-side execution owner. Keeping the backend and
/// session together makes backend selection a whole-model decision: generic
/// operations always supply this retained backend. Backend session hooks still
/// validate native identity when unrestricted session mutation is exposed.
/// Backend-owned executable, cache, tensor, and completion types remain
/// associated types and never enter the portable API.
/// The execution composition supplies the installed session type.
pub struct ModelRuntime<B: BackendProvider, S: BackendSession<B>> {
    backend: B,
    session: S,
    admission: crate::SessionAdmission,
    execution_plan_target_id: Option<u64>,
}

impl<B: BackendProvider, S: BackendSession<B>> ModelRuntime<B, S> {
    /// Installs the prepared model using the caller's selected installer.
    pub fn install<I: ModelSessionInstaller<B, Session = S>>(
        backend: B,
        model: PreparedModel<I::Model>,
    ) -> Result<Self, B::Error> {
        Self::install_with_target::<I>(backend, model, None)
    }

    pub(crate) fn install_with_target<I: ModelSessionInstaller<B, Session = S>>(
        backend: B,
        model: PreparedModel<I::Model>,
        execution_plan_target_id: Option<u64>,
    ) -> Result<Self, B::Error> {
        let admitted = model.capabilities();
        let session = I::install_session(&backend, model)?;
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

    /// Returns the installed session for optional execution capabilities.
    pub const fn session(&self) -> &S {
        &self.session
    }

    /// Returns the installed session for optional execution capabilities.
    ///
    /// Unrestricted mutation can replace the session, so this invalidates any
    /// retained execution-plan target proof. Previously selected drafting can
    /// no longer attach to this runtime. Generic operations still enforce the
    /// original exact capability admission after mutation.
    pub fn session_mut(&mut self) -> &mut S {
        self.execution_plan_target_id = None;
        &mut self.session
    }

    /// Borrows the selected backend and its mutable session together.
    ///
    /// Like [`Self::session_mut`], this invalidates retained drafting target proof.
    pub fn parts_mut(&mut self) -> (&B, &mut S) {
        self.execution_plan_target_id = None;
        (&self.backend, &mut self.session)
    }

    fn validate_session_admission(&self) -> Result<(), B::Error> {
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
    pub fn prefill(&mut self, input: S::PrefillInput) -> Result<SessionSubmission<B, S>, B::Error> {
        self.validate_session_admission()?;
        self.session.prefill(&self.backend, input)
    }

    /// Submits cached decode through the selected backend and session.
    pub fn decode(&mut self, input: S::DecodeInput) -> Result<SessionSubmission<B, S>, B::Error> {
        self.validate_session_admission()?;
        self.session.decode(&self.backend, input)
    }

    /// Materializes portable observations from an already completed output.
    pub fn observe_output(&self, output: &S::Output) -> Result<ObservationSet, B::Error> {
        self.validate_session_admission()?;
        self.session.observe_output(&self.backend, output)
    }
}

impl<B: BackendProvider, S: TextSessionLifecycle<B>> ModelRuntime<B, S> {
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
        self.admission.validate(self.session.capabilities())?;
        self.session
            .agree_text_preparation(&self.backend, stage, status)
    }

    /// Reports session lifetime preparation reservations, unaffected by restore.
    pub fn text_preparation_usage(
        &self,
    ) -> Result<crate::run_preparation::TextPreparationUsage, BackendFailure> {
        self.session.text_preparation_usage(&self.backend)
    }

    /// Settles prior work and clears request state while retaining the loaded model.
    ///
    /// Success establishes a fresh session for the next request. Model identity,
    /// admitted capabilities, placement and retained drafting target proof are
    /// preserved. Failure does not establish that the session can be reused.
    pub fn reset(&mut self) -> Result<(), BackendFailure> {
        self.synchronize()?;
        self.session.reset_session(&self.backend)
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
        self.session.synchronize_session(&self.backend)
    }
}

impl<B, S> ModelRuntime<B, S>
where
    B: TextGenerationBackend,
    S: ModelCapabilitySession<B> + TextSessionLifecycle<B>,
{
    /// Settles ordinary work and releases optional parameter-conversion claims.
    /// Preserves source weights and request state; allocator caches are separate.
    /// Controlled drivers use their canonical boundary instead of settling here.
    pub fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        Vec<crate::residency::ParameterConversionRetentionTrimReport>,
        crate::residency::ParameterConversionTrimError,
    > {
        self.synchronize()?;
        S::trim_parameter_conversions(self)
    }
}

/// Portable sampling inputs for one text-generation session.
#[derive(Debug, Clone, PartialEq)]
pub struct TextGenerationConfig {
    sampling: ResolvedGenerationConfig,
    seed: u64,
    strategy: TextSamplingStrategy,
    prefill: PrefillChunkPolicy,
    decoding_mode: Option<crate::decoding::DecodingMode>,
    draft_events: bool,
    canvas: Option<std::sync::Arc<crate::refinement::RefinementCanvas>>,
}

/// Bounds the number of prompt tokens processed together when the selected
/// backend and request support incremental prefill. Architecture-required
/// invocation limits apply even without a caller-imposed bound.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefillChunkPolicy {
    /// Selects a chunk for the request using installed execution and memory facts.
    Automatic,
    /// Selects a chunk while reserving these bytes in each physical memory pool.
    AutomaticWithReserve {
        /// Bytes set aside for other work and uncertainty.
        reserve_bytes: u64,
    },
    /// Add no caller-imposed chunk limit; architecture invocation limits remain.
    Unchunked,
    /// Process at most this many tokens in each eligible prefill pass.
    Bounded(std::num::NonZeroUsize),
}

impl Default for PrefillChunkPolicy {
    fn default() -> Self {
        Self::Automatic
    }
}

impl PrefillChunkPolicy {
    /// Bound used when there is no usable forecast for automatic selection.
    pub const AUTOMATIC_FALLBACK: std::num::NonZeroUsize =
        std::num::NonZeroUsize::new(512).unwrap();

    /// The caller's reserve for automatic sizing, or None for explicit chunking.
    pub const fn automatic_reserve_bytes(self) -> Option<u64> {
        match self {
            Self::Automatic => Some(0),
            Self::AutomaticWithReserve { reserve_bytes } => Some(reserve_bytes),
            Self::Unchunked | Self::Bounded(_) => None,
        }
    }

    /// Effective explicit bound, or the automatic fallback before selection.
    pub const fn unresolved_limit(self) -> std::num::NonZeroUsize {
        match self {
            Self::Automatic | Self::AutomaticWithReserve { .. } => Self::AUTOMATIC_FALLBACK,
            Self::Unchunked => std::num::NonZeroUsize::MAX,
            Self::Bounded(limit) => limit,
        }
    }
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
            prefill: PrefillChunkPolicy::Automatic,
            decoding_mode: None,
            draft_events: false,
            canvas: None,
        }
    }

    /// Selects a seeded and optionally pinned canvas for every iterative block.
    pub fn with_refinement_canvas(
        mut self,
        canvas: Option<std::sync::Arc<crate::refinement::RefinementCanvas>>,
    ) -> Self {
        self.canvas = canvas;
        self
    }

    /// Immutable canvas controls retained independently from numerical state.
    pub fn refinement_canvas(
        &self,
    ) -> Option<&std::sync::Arc<crate::refinement::RefinementCanvas>> {
        self.canvas.as_ref()
    }

    /// Selects the deterministic root seed used by a stochastic backend.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Selects the initial execution policy. Absent uses causal generation when
    /// available, or the trained policy of an intrinsically iterative architecture.
    pub const fn with_decoding_mode(mut self, mode: Option<crate::decoding::DecodingMode>) -> Self {
        self.decoding_mode = mode;
        self
    }

    /// Explicit initial mode; the selected architecture resolves an absent value.
    pub const fn decoding_mode(&self) -> Option<crate::decoding::DecodingMode> {
        self.decoding_mode
    }

    /// Explicit iterative policy, independently of authoritative token sampling.
    pub const fn refinement_policy(&self) -> Option<crate::refinement::RefinementPolicy> {
        match self.decoding_mode {
            Some(mode) => mode.refinement_policy(),
            None => None,
        }
    }

    /// Requests tentative IDs for bounded facade observation. Commitments and
    /// semantic output are independent of this request.
    pub const fn with_draft_events(mut self, enabled: bool) -> Self {
        self.draft_events = enabled;
        self
    }

    /// Whether tentative block observations are requested.
    pub const fn draft_events(&self) -> bool {
        self.draft_events
    }

    /// Selects prompt chunking for both ordinary and controlled generation.
    pub const fn with_prefill_chunk_policy(mut self, policy: PrefillChunkPolicy) -> Self {
        self.prefill = policy;
        self
    }

    /// Returns the requested prefill policy; backends preserve unsupported input
    /// and observation contracts by retaining an unchunked pass.
    pub const fn prefill_chunk_policy(&self) -> PrefillChunkPolicy {
        self.prefill
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

    /// Realizes a closed allow set at the model's actual output width.
    /// Missing IDs are false; a truncated prefix must still allow a token.
    pub fn allowed_mask_for(
        &self,
        output_width: usize,
    ) -> Result<Option<std::borrow::Cow<'_, [bool]>>, TokenFilterError> {
        if output_width == 0 {
            return Err(TokenFilterError::EmptyVocabulary);
        }
        let Some(mask) = self.allowed_mask() else {
            return Ok(None);
        };
        let prefix = &mask[..mask.len().min(output_width)];
        if !prefix.iter().any(|allowed| *allowed) {
            return Err(TokenFilterError::NoExecutableToken { output_width });
        }
        Ok(Some(if mask.len() >= output_width {
            std::borrow::Cow::Borrowed(prefix)
        } else {
            let mut mask = mask.to_vec();
            mask.resize(output_width, false);
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
}

impl<'a> TokenSamplingDecision<'a> {
    /// Creates a decision with unknown observation provenance.
    pub fn new(filter: TokenFilter) -> Self {
        Self {
            filter,
            pre_override_filter: None,
            tokenizer_validity: None,
        }
    }

    /// Declares the tokenizer baseline for this exact pre-override filter.
    pub fn with_tokenizer_validity(mut self, validity: &'a TokenFilter) -> Self {
        self.tokenizer_validity = Some(validity);
        self
    }

    /// The final filter to apply to native sampling.
    pub fn filter(&self) -> &TokenFilter {
        &self.filter
    }

    /// Applies an already validated forced choice while retaining the original
    /// domain for observation. This moves filters rather than copying masks.
    pub fn override_filter(&mut self, filter: TokenFilter) {
        let original = std::mem::replace(&mut self.filter, filter);
        if self.pre_override_filter.is_none() {
            self.pre_override_filter = Some(original);
        }
    }

    /// Exact pre-override domain, or explicitly unknown.
    pub fn capture_domain(&self) -> Option<crate::capture::CaptureTokenDomain<'_>> {
        Some(crate::capture::CaptureTokenDomain {
            filter: self.pre_override_filter.as_ref().unwrap_or(&self.filter),
            tokenizer_validity: self.tokenizer_validity?,
        })
    }
}

/// Backend-independent logical controller for constrained token selection.
pub trait TokenFilterController {
    /// Constraint or grammar error.
    type Error: std::error::Error + Send + Sync + 'static;

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
    /// Complete bytes retained by an isolated clone, or unknown.
    fn control_snapshot_bytes(&self) -> Option<u64> {
        None
    }

    /// Total current and future storage for an additional token horizon.
    /// This read-only query must not commit or fork grammar state.
    fn continuation_storage_bytes(&self, _additional_tokens: u64) -> Option<u64> {
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

/// Causal submissions owned by an installed text session.
pub trait CausalTextSession<B: BackendProvider, Prompt, Token, State>: BackendSession<B> {
    /// Submits native causal prompt work using the selected session.
    fn submit_causal_prefill(
        &mut self,
        _backend: &B,
        _prompt: Prompt,
        _domain: Option<crate::capture::CaptureTokenDomain<'_>>,
        _state: &mut State,
    ) -> Result<Submission<Self::Output, Self::Completion>, TextOperationError<B::Error>> {
        Err(TextOperationPolicyError::UnsupportedCausalSubmission.into())
    }

    /// Submits native causal decode work using the last committed token.
    fn submit_causal_decode(
        &mut self,
        _backend: &B,
        _token: Token,
        _domain: Option<crate::capture::CaptureTokenDomain<'_>>,
        _state: &mut State,
    ) -> Result<Submission<Self::Output, Self::Completion>, TextOperationError<B::Error>> {
        Err(TextOperationPolicyError::UnsupportedCausalSubmission.into())
    }

    /// Supplies previously retained scores after a settled mode transition.
    fn submit_causal_continuation(
        &mut self,
        _backend: &B,
        _state: &mut State,
    ) -> Result<Submission<Self::Output, Self::Completion>, TextOperationError<B::Error>> {
        Err(TextOperationPolicyError::UnsupportedContinuation.into())
    }
}

/// Portable dispatch of a text session's physical operations.
pub trait TextOperationDriver<B: TextGenerationBackend>: BackendSession<B> + Sized {
    /// Prepared input owned by this portable operation driver.
    type Prompt: 'static;

    /// Counts an unadorned prompt for automatic prefill sizing.
    fn plain_text_prefill_tokens(_prompt: &Self::Prompt) -> Option<usize> {
        None
    }

    /// Resolves automatic prefill sizing without submitting model work.
    /// The input count excludes the verified prefix already restored.
    fn automatic_prefill_chunk(
        _runtime: &ModelRuntime<B, Self>,
        _input_tokens: usize,
        _restored_tokens: usize,
        _reserve_bytes: u64,
    ) -> std::num::NonZeroUsize {
        PrefillChunkPolicy::AUTOMATIC_FALLBACK
    }

    /// Admits tokenized text and prepares the exact prompt for this installed driver.
    fn prepare_text_prompt(
        backend: &B,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, B::Error>;

    /// Mutable request state owned by the installed operation driver.
    type TextGenerationState;

    /// Starts one portable generation request after admission has agreed.
    fn start_text_generation(
        &self,
        backend: &B,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, B::Error>;

    /// Samples once from the submitted causal scores under the current filter.
    fn sample_causal_submission(
        &self,
        _backend: &B,
        _submission: SessionSubmission<B, Self>,
        _filter: &TokenFilter,
        _state: &mut Self::TextGenerationState,
    ) -> Result<Submission<B::Token, B::TextCompletion>, TextOperationError<B::Error>> {
        Err(TextOperationPolicyError::UnsupportedCausalSubmission.into())
    }

    /// Exact support for serial, completed-token control on this loaded session.
    /// Backends opt in only for verified ordinary single-sequence execution.
    /// Snapshot mechanisms and complete facade-state support are separate facts.
    fn text_execution_control_support(&self, _: &B) -> crate::execution_control::ControlSupport {
        crate::execution_control::ControlSupport::Unsupported {
            reason: "backend has not declared completed-token control support".into(),
        }
    }

    /// Exact support for prospective temperature changes and explicit reseeding.
    /// The runtime validates policy; the adapter supplies atomic native changes.
    fn text_sampling_control_support(&self, _: &B) -> crate::execution_control::ControlSupport {
        crate::execution_control::ControlSupport::Unsupported {
            reason: "backend has no prospective sampling controls".into(),
        }
    }

    /// Active parameter-edit provenance for ordinary and controlled generation.
    fn active_parameter_overlay(&self) -> Option<&str> {
        None
    }

    /// Trained canvas extent of the selected iterative realization, without execution.
    fn refinement_canvas_length(&self) -> Option<std::num::NonZeroU32> {
        None
    }

    /// Admits ordered selected scores for the next refinement operation.
    fn configure_refinement_readout(
        &self,
        _state: &mut Self::TextGenerationState,
        _selections: Vec<crate::refinement::RefinementScoreSelection>,
    ) -> Result<(), crate::refinement::RefinementReadoutError> {
        Err(crate::refinement::RefinementReadoutError::Unsupported)
    }

    /// Drains selected scores following exact completion.
    fn take_refinement_readout(
        _state: &mut Self::TextGenerationState,
    ) -> Option<crate::refinement::RefinementScoreRows> {
        None
    }

    /// Complete score vocabulary of the selected iterative realization.
    fn refinement_vocabulary_size(&self) -> Option<std::num::NonZeroU32> {
        None
    }

    /// Whether the selected executable supports iterative token blocks.
    fn supports_block_refinement(&self) -> bool {
        false
    }

    /// Source-free facts for all modes already realized by this loaded execution.
    fn decoding_capabilities(&self) -> crate::decoding::DecodingCapabilities {
        let iterative = self.supports_block_refinement();
        crate::decoding::DecodingCapabilities {
            autoregressive: !iterative,
            diffusion: iterative,
            entropy_feedback: iterative,
            ..Default::default()
        }
    }

    /// Trained initial policy when the request has no explicit mode.
    fn default_decoding_mode(&self) -> crate::decoding::DecodingMode {
        if self.decoding_capabilities().autoregressive {
            crate::decoding::DecodingMode::Autoregressive
        } else {
            crate::decoding::DecodingMode::Diffusion { policy: None }
        }
    }

    /// Exact numerical policy supported by the retained iterative architecture.
    /// A mechanism implementing unmasking does not imply family support for it.
    fn supports_refinement_policy(&self, policy: crate::refinement::RefinementPolicy) -> bool {
        self.supports_block_refinement()
            && matches!(
                policy,
                crate::refinement::RefinementPolicy::EntropyFeedback(_)
            )
    }

    /// Meaning of scores produced by the retained generation realization.
    fn generation_score_kind(&self) -> crate::refinement::PredictionScoreKind {
        if self.supports_block_refinement() {
            crate::refinement::PredictionScoreKind::RefinementPosition
        } else {
            crate::refinement::PredictionScoreKind::CausalConditional
        }
    }

    /// Reports whether this executable can chunk ordinary plain-text prefill.
    /// Media/structured inputs and requested captures can still require the
    /// complete pass; this capability does not describe those request contracts.
    fn text_prefill_chunking_support(&self) -> Result<(), &'static str> {
        Err("backend retains a complete prefill pass")
    }

    /// Observes weight residency for this loaded session when supported.
    /// An unavailable report is distinct from a failed observation.
    fn residency_telemetry(&self) -> Result<Option<crate::ResidencyTelemetry>, BackendFailure> {
        Ok(None)
    }

    /// Observes expert-cache occupancy for this loaded session when supported.
    /// An unavailable report is distinct from a failed observation.
    fn expert_cache_telemetry(
        &self,
    ) -> Result<Option<crate::ExpertCacheTelemetry>, BackendFailure> {
        Ok(None)
    }

    /// Returns genuine mutable points for this exact loaded session.
    #[cfg(feature = "tensor-capture")]
    fn intervention_discovery(
        &self,
    ) -> Result<crate::intervention::InterventionDiscovery, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no intervention discovery".into(),
        ))
    }

    /// Checks both plans without submitting work or changing model/sampler state.
    #[cfg(feature = "tensor-capture")]
    fn validate_text_interventions(
        &self,
        capture: &crate::capture::AdmittedCapturePlan,
        plan: &crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::capture::CaptureError> {
        if !plan.is_empty() {
            return Err(crate::capture::CaptureError::Unsupported(
                "backend has no text interventions".into(),
            ));
        }
        self.validate_text_capture(capture)
    }

    /// Installs immutable plans before the first submission. Diagnostics and
    /// evidence must share capture accounting and ordinary completion ownership.
    #[cfg(feature = "tensor-capture")]
    fn configure_text_interventions(
        &self,
        state: &mut Self::TextGenerationState,
        capture: crate::capture::AdmittedCapturePlan,
        plan: crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::capture::CaptureError> {
        self.validate_text_interventions(&capture, &plan)?;
        self.configure_text_capture(state, capture)
    }

    /// Returns retained admission facts without resolving checkpoint content identity.
    #[cfg(feature = "tensor-capture")]
    fn capture_declarations(
        &self,
    ) -> Result<crate::capture::CaptureDeclarations, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no bounded capture declarations".into(),
        ))
    }

    /// Resolves content-exact provenance together with retained observation facts.
    #[cfg(feature = "tensor-capture")]
    fn capture_discovery(
        &self,
    ) -> Result<crate::capture::CaptureDiscovery, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "backend has no bounded capture discovery".into(),
        ))
    }

    /// Exact bounds for independently shaped physical forwards. An absent value
    /// selects ordinary causal prefill/decode geometry. This reads retained
    /// executable facts without model work or native allocations.
    #[cfg(feature = "tensor-capture")]
    fn text_capture_invocation_bounds(
        &self,
        _request: crate::capture::CaptureRequestShape,
    ) -> Result<Option<crate::capture::CaptureInvocationBounds>, crate::capture::CaptureError> {
        Ok(None)
    }

    /// Enables an immutable capture plan before the first submission. Implementations
    /// must validate it against this session and reject unsupported combinations.
    #[cfg(feature = "tensor-capture")]
    fn configure_text_capture(
        &self,
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
    #[cfg(feature = "tensor-capture")]
    fn validate_text_capture(
        &self,
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

    /// Moves out at most one completed capture step. No implementation may queue
    /// unconsumed steps without a separately admitted finite buffering contract.
    #[cfg(feature = "tensor-capture")]
    fn take_text_capture(
        _state: &mut Self::TextGenerationState,
    ) -> Option<crate::capture::CapturedStep> {
        None
    }

    /// Declares the next physical operation without execution or controller queries.
    fn text_operation_kind(
        _runtime: &ModelRuntime<B, Self>,
        _input: &PendingTextInput<Self::Prompt, B::Token>,
        _state: &Self::TextGenerationState,
    ) -> TextOperationKind {
        TextOperationKind::Decision(crate::refinement::PredictionScoreKind::CausalConditional)
    }

    /// Submits a specialized declared operation such as iterative numerical refinement.
    /// The portable driver handles ordinary causal submissions separately.
    #[allow(clippy::type_complexity)]
    fn submit_specialized_text_operation(
        _runtime: &mut ModelRuntime<B, Self>,
        _input: PendingTextInput<Self::Prompt, B::Token>,
        _decision: Option<&TokenSamplingDecision<'_>>,
        _state: &mut Self::TextGenerationState,
        _remaining_tokens: Option<usize>,
    ) -> Result<
        Submission<TextOperationOutput<Self::Prompt, B::Token>, B::TextCompletion>,
        TextOperationError<B::Error>,
    > {
        Err(TextOperationPolicyError::OutcomeMismatch.into())
    }

    /// Completes one nonfinal prefill step without sampling.
    /// Returns true after replacing `prompt` with its remaining suffix or
    /// advancing its retained cursor by one ingress step or decoder chunk.
    /// Returning false performs no model work and retains final prefill input.
    ///
    /// Implementations must preserve cache positions and settle native work
    /// before returning true, so transient graphs do not accumulate across
    /// chunks. Captures describe this physical chunk and are drained before the
    /// next prefix by shared facade drivers. Inputs without a supported partial
    /// invocation contract retain the full pass.
    fn prefill_text_prefix(
        _runtime: &mut ModelRuntime<B, Self>,
        _prompt: &mut Self::Prompt,
        _max_tokens: std::num::NonZeroUsize,
        _state: &mut Self::TextGenerationState,
    ) -> Result<bool, B::Error> {
        Ok(false)
    }

    /// Agrees pending commands and cancellation before entering distributed work.
    /// The default is a single-owner execution with no transport.
    fn coordinate_decoding_control(
        _runtime: &mut ModelRuntime<B, Self>,
        local: crate::decoding::DecodingCoordination,
    ) -> Result<crate::decoding::DecodingCoordination, TextOperationError<B::Error>> {
        Ok(local)
    }

    /// Settles a queued handoff without sampling or advancing commitment controls.
    /// `false` leaves a diffusion block or verification transaction active. `true`
    /// replaces pending input after appending its committed suffix exactly once,
    /// preserving next-target logits and removing all tentative output. Returned
    /// work must have exact completion before this method returns. Admission uses
    /// `decoding_capabilities` before this method can be called.
    fn settle_decoding_mode(
        _runtime: &mut ModelRuntime<B, Self>,
        _input: &mut Option<PendingTextInput<Self::Prompt, B::Token>>,
        _state: &mut Self::TextGenerationState,
        _previous: crate::decoding::DecodingMode,
        _requested: crate::decoding::DecodingMode,
    ) -> Result<bool, TextOperationError<B::Error>> {
        Err(TextOperationPolicyError::DecodingMode(
            crate::decoding::DecodingModeError::Unsupported("settled execution handoff"),
        )
        .into())
    }

    /// Publishes accepted sampling or verification state after a canonical token commits.
    fn commit_text_token(
        _runtime: &mut ModelRuntime<B, Self>,
        _token_id: u32,
        _position: crate::refinement::GenerationPosition,
        _state: &mut Self::TextGenerationState,
    ) -> Result<(), B::Error> {
        Ok(())
    }
}

/// Automatic prefix reuse through the installed session's cache operations.
pub trait AutomaticPromptCacheSession<B: TextGenerationBackend>: BackendSession<B> + Sized {
    /// Model-derived prompt-cache identity for the selected execution state.
    fn prompt_cache_model_identity(
        _runtime: &ModelRuntime<B, Self>,
    ) -> Result<crate::cache::PromptCacheModelIdentity, BackendFailure> {
        Err(BackendFailure::from_error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "prompt-cache storage is unavailable",
        )))
    }
    /// Content identity for the exact backend-prepared plain-text prefix.
    fn prompt_cache_prefix_fingerprint(
        _runtime: &ModelRuntime<B, Self>,
        prefix: &[u32],
    ) -> Result<String, BackendFailure> {
        Ok(crate::cache::prompt_cache_token_fingerprint(prefix))
    }
    /// Native isolated-state byte bound, before portable encoding/publication costs.
    fn estimate_prompt_cache_state(
        _runtime: &ModelRuntime<B, Self>,
    ) -> Result<Option<u64>, BackendFailure> {
        Ok(None)
    }
    /// Saves through the installed session's ordinary prompt-cache export path.
    fn save_prompt_cache(
        _runtime: &mut ModelRuntime<B, Self>,
        _store: &dyn crate::cache::PromptCacheStore,
        _descriptor: crate::cache::PromptCacheDescriptor,
        _prefix: &[u32],
        _options: &crate::cache::PromptCacheOptions,
    ) -> Result<crate::cache::PromptCacheManifest, BackendFailure> {
        Err(BackendFailure::from_error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "prompt-cache export is unavailable",
        )))
    }
    /// Restores the exact immutable generation admitted by the caller.
    fn restore_prompt_cache(
        _runtime: &mut ModelRuntime<B, Self>,
        _read: &dyn crate::cache::PromptCacheRead,
        _descriptor: &crate::cache::PromptCacheDescriptor,
        _prefix: &[u32],
    ) -> Result<crate::cache::PromptCacheManifest, BackendFailure> {
        Err(BackendFailure::from_error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "prompt-cache import is unavailable",
        )))
    }
}

/// Native tensor representations and operations for an installed text executable.
pub trait NativeTextValues: 'static {
    /// Tensor stored in prepared prompt parts.
    type InputTensor: Clone + 'static;
    /// Native prepared decode positions.
    type Decode;
    /// Native score handle.
    type Scores;
    /// Tensor used by the installed neural operators and parameter owner.
    type Tensor: eredu_nn::Tensor<Context = Self::Context>;
    /// Tensor passed to the model's invocation observer.
    type InvocationTensor;
    /// Native mapping between the two tensor representations.
    type ObserverMapping: crate::observation::ObserverTensorMapping<
        InvocationTensor = Self::InvocationTensor,
        ObserverTensor = Self::Tensor,
        InvocationError = Self::Error,
        ObserverError = Self::Error,
    >;
    /// Exact native execution context.
    type Context: Clone;
    /// Failure of model invocation or native resource access.
    type Error: std::error::Error + Send + Sync + From<BackendFailure> + 'static;
    /// Failure reading an installed native state observation.
    type StateError: std::error::Error + Send + Sync + 'static;
    /// Exact native group used for coordinated token sampling.
    type Group;
    /// Side-effect-free metadata inspection for prepared input.
    type InputInspector: crate::input::PreparedInputInspector<Self::InputTensor>;

    /// Optional guard for native retirement after isolated state copies are dropped.
    fn snapshot_retirement() -> Option<Box<dyn std::any::Any>> {
        None
    }

    /// Borrows tensor metadata without submitting model work.
    fn input_inspector() -> Self::InputInspector;

    /// Reads the actual token values carried by an incoming tensor.
    fn read_token_ids(
        tokens: &Self::InputTensor,
        context: &Self::Context,
    ) -> Result<Vec<u32>, Self::Error>;

    /// Native metadata inspection on decoder tensors.
    type TensorInputInspector: crate::input::PreparedInputInspector<Self::Tensor>;
    /// Maps decoder tensors to native observers.
    type TensorObserver: crate::observation::ObserverTensorMapping<
        InvocationTensor = Self::Tensor,
        ObserverTensor = Self::InvocationTensor,
        InvocationError = eredu_nn::Error,
        ObserverError = Self::Error,
    >;
    /// Native neural operators.
    type Neural: eredu_nn::NeuralBackend<Tensor = Self::Tensor>;
    /// Native tensor and page storage for portable text state.
    type Cache: crate::cache::mechanisms::CacheTensorStorage<
        Value = Self::Tensor,
        Queue = Self::Context,
    >;
    /// Realized rank communication.
    type Communication: 'static;

    /// Borrows the active native group for sampling, if it has collective work.
    fn sampling_group(
        communication: &Self::Communication,
        group: crate::CollectiveGroupId,
    ) -> Option<&Self::Group>;

    /// Borrows the exact shared authority retained with native communication.
    fn authority(
        communication: &Self::Communication,
    ) -> &crate::communication::CommunicationAuthority;
    /// Converts one prepared input handle to the decoder tensor representation.
    fn input_tensor(value: &Self::InputTensor) -> Self::Tensor;
    /// Supplies the metadata inspector for decoder tensors.
    fn tensor_input_inspector() -> Self::TensorInputInspector;
    /// Converts one prepared decode handle.
    fn decode_tensor(tokens: &Self::Decode) -> Self::Tensor;
    /// Converts scores to the representation exposed by native submission.
    fn scores(value: Self::Tensor) -> Self::Scores;
    /// Concatenates exact token parts on the specified queue.
    fn concatenate_token_parts(
        parts: &[&Self::InputTensor],
        queue: &Self::Context,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Converts a token handle from the invocation observer.
    fn prompt_tokens(value: &Self::InvocationTensor) -> Self::InputTensor;
    /// Converts a prepared decode handle into an input handle.
    fn decode_prompt_tokens(value: &Self::Decode) -> Self::InputTensor;
    /// Converts a portable state observation error to the native error domain.
    fn state_error(error: BackendFailure) -> Self::StateError;
    /// Measures the exact logical payload bytes of a replacement tensor.
    fn tensor_bytes(value: &Self::Tensor) -> u64;
    /// Reads the materialized decoder token IDs.
    fn decoder_token_ids(value: &Self::Tensor) -> Result<Vec<u32>, Self::Error>;

    /// Native predictor materialization operations.
    type Materializer;
    /// Native tensor operators and collectors for embedded verification.
    type Embedded: crate::speculative::embedded::EmbeddedPredictionOperators<Tensor = Self::Tensor>;

    /// Settles the retained input handle.
    fn settle_input(
        value: &Self::Tensor,
    ) -> Result<
        (),
        <Self::Embedded as crate::speculative::embedded::EmbeddedPredictionOperators>::Error,
    >;
    /// Preserves a typed input failure in the predictor error domain.
    fn input_failure(
        error: BackendFailure,
    ) -> <Self::Embedded as crate::speculative::embedded::EmbeddedPredictionOperators>::Error;
}

/// Backend representations needed to erase an installed executable.
pub trait TextExecutableTypes: 'static {
    /// Tensor stored in prepared prompt parts.
    type InputTensor: Clone + 'static;
    /// Native prepared decode positions.
    type Decode;
    /// Native score handle.
    type Scores;
    /// Tensor used by the installed neural operators and parameter owner.
    type Tensor: eredu_nn::Tensor;
    /// Tensor passed to the model's invocation observer.
    type InvocationTensor;
    /// Native mapping between the two tensor representations.
    type ObserverMapping: crate::observation::ObserverTensorMapping<
        InvocationTensor = Self::InvocationTensor,
        ObserverTensor = Self::Tensor,
        InvocationError = Self::Error,
        ObserverError = Self::Error,
    >;
    /// Exact native execution context.
    type Context;
    /// Failure of model invocation or native resource access.
    type Error: std::error::Error + Send + Sync + From<BackendFailure> + 'static;
    /// Failure reading an installed native state observation.
    type StateError: std::error::Error + Send + Sync + 'static;
    /// Native state retained by an independently advancing prediction lane.
    type PredictionState;
    /// Exact native group used for coordinated token sampling.
    type Group;
    /// Native completion and input types for the bound embedded executor.
    type EmbeddedExecutor: crate::speculative::erasure::GenerationExecutorTypes<Binding = ()>;
    /// Native representations used by the installed prediction collectors.
    type EmbeddedObservers;
    /// Side-effect-free metadata inspection for prepared input.
    type InputInspector: crate::input::PreparedInputInspector<Self::InputTensor>;

    /// Optional guard for native retirement after isolated state copies are dropped.
    fn snapshot_retirement() -> Option<Box<dyn std::any::Any>> {
        None
    }

    /// Borrows tensor metadata without submitting model work.
    fn input_inspector() -> Self::InputInspector;

    /// Reads the actual token values carried by an incoming tensor.
    fn read_token_ids(
        tokens: &Self::InputTensor,
        context: &Self::Context,
    ) -> Result<Vec<u32>, Self::Error>;
}

/// Local host generation with optional masked block refinement.
pub struct CausalTextGeneration;

/// A session that can realize the advanced text dispatch routes.
pub struct AdvancedTextGeneration;

/// The native resource, queue and submission operations used by the driver.
pub trait TextSessionNative<B: BackendProvider>:
    Sized + crate::session_authority::SessionRetirement<Error = B::Error>
{
    /// Generation dispatch profile supported by this native implementation.
    type Generation;
    /// Exact native placement of prepared resources.
    type Target;
    /// Native communication retained for a prepared executable.
    type Communication;

    /// Reports an existing native failure without progressing outstanding work.
    fn check_native_health(&self) -> Result<(), B::Error>;

    /// Validates native placement before creating installed resources.
    fn validate_target(backend: &B, target: &Self::Target) -> Result<(), B::Error>;

    /// Attaches native placement and completion lifetimes for installation.
    fn attach_native(
        target: Self::Target,
        communication: Option<Self::Communication>,
    ) -> Result<(Self::Retained, Self), B::Error>;

    /// Exact native values supplied to the portable executable.
    type TextValues: NativeTextValues<Error = B::Error>;
    /// Retained native target data.
    type Retained;
    /// Native parameter handles retained by the portable overlay state.
    type ParameterValue: Clone + 'static;
    /// Native retirement owner specialized to the supplied opaque resource root.
    type Retired<R: 'static>: std::ops::DerefMut<Target = R> + 'static;
    /// Applies native retirement to an opaque resource root.
    fn retain_resources<R: 'static>(resources: R) -> Self::Retired<R>;

    /// Published score handle.
    type Output;
    /// Native completion of the invocation.
    type Completion: Completion<Error = B::Error>;
    /// Native submission scope for this session.
    type Scope<'a>: TextSubmissionScope<
        <Self::TextValues as NativeTextValues>::Scores,
        B::Error,
        Completion = Self::Completion,
    >
    where
        Self: 'a;

    /// Adopts exact submission authority before the executable is accessed.
    /// Token submissions collect deferred native token-index validation.
    fn begin(
        &self,
        lease: crate::SubmissionLease,
        token_validation: bool,
    ) -> Result<Self::Scope<'_>, B::Error>;

    /// Makes a one-position input from an exact committed token ID.
    fn decode_token_id(
        token: u32,
        backend: &B,
    ) -> Result<<Self::TextValues as NativeTextValues>::Decode, B::Error>;

    /// Native sampling result before cached decode preparation.
    type SampledDecode;

    /// Prepares cached decode positions from an already admitted native sample.
    fn decode_sample(
        token: Self::SampledDecode,
        backend: &B,
    ) -> Result<<Self::TextValues as NativeTextValues>::Decode, B::Error>;

    /// Error for a resource root still shared at an idle boundary.
    fn busy() -> B::Error;

    /// The selected device context.
    fn context(backend: &B) -> &<Self::TextValues as NativeTextValues>::Context;

    /// Collector for local capture and intervention, borrowing the exact context.
    type LocalCollector<'a>: crate::intervention::InterventionBackend<
        Tensor = <Self::TextValues as NativeTextValues>::Tensor,
        Error = B::Error,
    >;

    /// Opens the optional local collector on its exact native context.
    fn local_capture<'a>(
        _context: &'a <Self::TextValues as NativeTextValues>::Context,
        _domain: Option<crate::capture::CaptureTokenDomain<'a>>,
    ) -> Option<Self::LocalCollector<'a>> {
        None
    }

    /// Side-effect-free mechanisms and the estimator used by local intervention.
    fn local_intervention(
        &self,
    ) -> Option<(
        crate::intervention::InterventionMechanisms,
        std::sync::Arc<dyn crate::intervention::InterventionEstimator>,
    )> {
        None
    }

    /// Whether the selected local device can realize host-backed refinement.
    fn supports_host_refinement(&self) -> bool {
        false
    }

    /// Uploads the exact tentative canvas for one local refinement invocation.
    fn host_refinement_canvas(
        _ids: &[u32],
        _context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<<Self::TextValues as NativeTextValues>::Tensor, B::Error>
    where
        B::Error: From<crate::BackendFailure>,
    {
        Err(crate::BackendFailure::new(
            crate::BackendFailureKind::Unsupported,
            std::io::Error::other("no host refinement canvas adapter"),
        )
        .into())
    }

    /// Reports the floating representation retained by soft-feedback ingress.
    fn host_refinement_format(
        _scores: &<Self::TextValues as NativeTextValues>::Scores,
    ) -> Option<eredu_nn::TensorElementType> {
        None
    }

    /// Uploads immutable feedback on the exact execution context and settles
    /// the upload before returning the tensor for a refinement invocation.
    fn host_refinement_feedback(
        _scores: &[f32],
        _shape: &[i32],
        _element: eredu_nn::TensorElementType,
        _context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<<Self::TextValues as NativeTextValues>::Tensor, B::Error>
    where
        B::Error: From<crate::BackendFailure>,
    {
        Err(crate::BackendFailure::new(
            crate::BackendFailureKind::Unsupported,
            std::io::Error::other("no host refinement feedback adapter"),
        )
        .into())
    }

    /// Copies the exact admitted canvas score matrix after native completion.
    fn host_refinement_scores(
        _scores: &<Self::TextValues as NativeTextValues>::Scores,
        _shape: [usize; 2],
        _context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Vec<f32>, B::Error>
    where
        B::Error: From<crate::BackendFailure>,
    {
        Err(crate::BackendFailure::new(
            crate::BackendFailureKind::Unsupported,
            std::io::Error::other("no host refinement score adapter"),
        )
        .into())
    }

    /// Prices all work and host staging before creating a local native capture view.
    fn estimate_local_capture(
        _source: &[u64],
        _selection: &crate::capture::CaptureSelection,
        _slice: &crate::capture::ResolvedCaptureSlice,
    ) -> Result<crate::capture::CaptureUsage, crate::capture::CaptureError> {
        Err(crate::capture::CaptureError::Unsupported(
            "no local native capture collector".into(),
        ))
    }

    /// Publishes scores on the selected output rank.
    fn publish_scores(
        scores: <Self::TextValues as NativeTextValues>::Scores,
        publish: bool,
    ) -> Self::Output;

    /// Validates that lifecycle operations target the exact installed native session.
    fn validate_native_target(
        &self,
        retained: &Self::Retained,
        backend: &B,
    ) -> Result<(), B::Error>;

    /// Rejects synchronization when communication has fenced the native session.
    fn check_transport_active(retained: &Self::Retained) -> Result<(), BackendFailure>;

    /// Waits for the native parameter and execution queues.
    fn synchronize_queues(backend: &B) -> Result<(), B::Error>;

    /// Transport bound to the selected distributed session.
    type ControlTransport: crate::run_preparation::TextPreparationTransport<Error = B::Error>
        + crate::parameters::ParameterOperationTransport
        + Clone;

    /// Borrows the retained cohort transport, if the session is distributed.
    fn control_transport(retained: &Self::Retained) -> Option<&Self::ControlTransport>;

    /// Exact participant identity for the retained transport.
    fn control_transport_identity(
        transport: &Self::ControlTransport,
    ) -> crate::communication::CommunicationSessionIdentity;

    /// Reports whether the retained cohort has bounded parameter control.
    fn parameter_control_supported(transport: &Self::ControlTransport) -> bool;

    /// Describes the native bounds for a fixed preparation exchange.
    fn preparation_admission(
        transport: &Self::ControlTransport,
        words: usize,
    ) -> Result<Option<crate::run_preparation::TextPreparationAdmission>, B::Error>;

    /// Borrows the exact published score tensor, if this rank owns it.
    fn output_scores(
        output: &Self::Output,
    ) -> Option<&<Self::TextValues as NativeTextValues>::Tensor>;

    /// Reads completed native tensors in the supplied order. Retains the tensors,
    /// invocation root and submission authority until native transfer settles.
    fn read_observation_tensors(
        &self,
        lease: crate::SubmissionLease,
        root: std::rc::Rc<dyn std::any::Any>,
        backend: &B,
        tensors: Vec<<Self::TextValues as NativeTextValues>::Tensor>,
    ) -> Result<Vec<crate::observation::TensorObservation>, B::Error>;
}

/// Native reads for committed token storage at a settled mode boundary.
pub trait CommittedTokenReadback<T, C, Queue: ?Sized, E> {
    /// Reads the pending committed token.
    fn pending_token(token: &T, queue: &Queue) -> Result<u32, E>;

    /// Reads an authoritative refinement block in position order.
    fn committed_block(tokens: &[C], queue: &Queue) -> Result<Vec<u32>, E>;
}

/// Numerical and completion mechanisms for sampling an admitted text submission.
pub trait TextSamplingNative<B: TextGenerationBackend>: TextSessionNative<B> {
    /// Native cancellation behavior for an outstanding text submission.
    fn completion_cancellation_mode() -> Option<crate::CompletionCancellationMode>;

    /// Exact completion for a sample from already submitted causal scores.
    type SamplingCompletion: crate::Completion<Error = B::Error>;

    /// Final vocabulary row owned by the sampling route. Host routes return
    /// one owned FP32 vector, while device routes retain a native tensor.
    type ScoreRow;

    /// Selected value handed to the native completion scope.
    type Sample;

    /// Exact native resources retained while a token is derived from submitted scores.
    type SamplingScope;

    /// Attaches the token operation to the live score submission. A host
    /// adapter establishes safe readback here before its score-row method runs.
    fn begin_sampling(completion: &Self::Completion) -> Result<Self::SamplingScope, B::Error>;

    /// Returns the final physical score row when the backend owns the submitted output.
    fn last_score_row(
        output: &Self::Output,
        context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Option<Self::ScoreRow>, B::Error>;

    /// Observes the realized collective coordinates without submitting work.
    fn sampling_group_coordinates(
        group: &<Self::TextValues as NativeTextValues>::Group,
    ) -> (usize, usize);

    /// Broadcasts exactly one selected token on the admitted group.
    fn exchange_sampled_token(
        token: Option<Self::Sample>,
        rank: usize,
        group: &<Self::TextValues as NativeTextValues>::Group,
        authority: &crate::communication::CommunicationAuthority,
        context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Self::Sample, B::Error>;

    /// Combines score and token completion or safely retires a failed token operation.
    fn finish_sampling(
        scope: Self::SamplingScope,
        completion: Self::Completion,
        token: Result<Self::Sample, B::Error>,
        context: &<Self::TextValues as NativeTextValues>::Context,
    ) -> Result<Submission<Self::Token, Self::SamplingCompletion>, B::Error>;

    /// Publishes settled host-only refinement progress under installed authority.
    fn complete_host_progress(
        _lease: crate::SubmissionLease,
    ) -> Result<Self::SamplingCompletion, B::Error>
    where
        B::Error: From<crate::BackendFailure>,
    {
        Err(crate::BackendFailure::new(
            crate::BackendFailureKind::Unsupported,
            std::io::Error::other("no host progress completion"),
        )
        .into())
    }

    /// Publishes a host-selected token under an acquired installed-session lease.
    /// There is no pending device computation for this host-only decision.
    fn complete_host_token(
        _token: u32,
        _lease: crate::SubmissionLease,
    ) -> Result<Submission<Self::Token, Self::SamplingCompletion>, B::Error>
    where
        B::Error: From<crate::BackendFailure>,
    {
        Err(crate::BackendFailure::new(
            crate::BackendFailureKind::Unsupported,
            std::io::Error::other("no host token completion"),
        )
        .into())
    }

    /// Native token value consumed by one cached decode.
    type Token;
    /// Extracts the sampled value without preparing or submitting a decode.
    fn sampled_token_value(token: Self::Token) -> Self::SampledDecode;
}

/// Device-native numerical operations for a causal text submission.
pub trait DeviceTextSampling<B: TextGenerationBackend>: TextSamplingNative<B> {
    /// Exact native numerical failure.
    type SamplingError: std::error::Error + Send + Sync + 'static;

    /// Numerical operations over native scores and the native random stream.
    type Sampling: crate::sampling::SamplingBackend<
        Logits = Self::ScoreRow,
        Token = Self::Sample,
        Context = <Self::TextValues as NativeTextValues>::Context,
        Error = Self::SamplingError,
    >;
}

/// Native preparation and sampling state for text generation.
pub trait TextGenerationBackend: BackendProvider {
    /// Native tensor in prepared inputs. The driver owns their composition.
    type InputTensor: Clone + 'static;
    /// Backend-owned generated token handle.
    type Token: TokenOutput<Error = Self::Error>;
    /// Exact completion retaining model execution and token sampling.
    type TextCompletion: Completion<Error = Self::Error>;
}

/// Native recovery and tensor conversion for portable text media preparation.
pub trait TextMediaNative<B: TextGenerationBackend>: TextSessionNative<B> {
    /// Scope retaining tensors created for the portable processor.
    type ProcessorScope: crate::input::ProcessorPreparationScope<
        Tensor = <Self::TextValues as NativeTextValues>::InputTensor,
        Error = B::Error,
    >;

    /// Opens native allocation and submission recovery for the installed placement.
    fn begin_processor(&self) -> Result<Self::ProcessorScope, B::Error>;
}

/// Native values borrowed by the portable shared-target text driver.
pub trait SharedTextNative: TextGenerationBackend {
    /// Placement, submission, and retirement mechanisms for text.
    type Native: TextSessionNative<Self>;
    /// Borrows the already selected native target queue.
    fn shared_queue(
        &self,
    ) -> &<<Self::Native as TextSessionNative<Self>>::TextValues as NativeTextValues>::Context;

    /// Reads one completed native token.
    fn shared_token_id(&self, token: &Self::Token) -> Result<u32, Self::Error>;

    /// Native placeholder retained on a progress-only completion.
    fn shared_progress_value(
    ) -> <<Self::Native as TextSessionNative<Self>>::TextValues as NativeTextValues>::Scores;

    /// Combines the exact model fence with completion of the retained progress value.
    fn finish_shared_progress(
        submission: Submission<
            <<Self::Native as TextSessionNative<Self>>::TextValues as NativeTextValues>::Scores,
            <Self::Native as TextSessionNative<Self>>::Completion,
        >,
    ) -> Result<Self::TextCompletion, Self::Error>;
}

/// Installed session operations used by the ordinary generation driver.
pub trait TextExecutionSession<B: TextGenerationBackend>:
    TextSessionLifecycle<B>
    + CausalTextSession<B, Self::Prompt, B::Token, Self::TextGenerationState>
    + TextOperationDriver<B>
{
}

impl<B, S> TextExecutionSession<B> for S
where
    B: TextGenerationBackend,
    S: TextSessionLifecycle<B>
        + CausalTextSession<B, S::Prompt, B::Token, S::TextGenerationState>
        + TextOperationDriver<B>,
{
}

impl<B, S> ModelRuntime<B, S>
where
    B: TextGenerationBackend,
    S: TextOperationDriver<B> + CausalTextSession<B, S::Prompt, B::Token, S::TextGenerationState>,
{
    /// Executes one declared text operation through the portable advancement protocol.
    #[allow(clippy::type_complexity)]
    pub fn submit_text_operation(
        &mut self,
        input: PendingTextInput<S::Prompt, B::Token>,
        decision: Option<&TokenSamplingDecision<'_>>,
        state: &mut S::TextGenerationState,
        remaining_tokens: Option<usize>,
    ) -> Result<
        Submission<TextOperationOutput<S::Prompt, B::Token>, B::TextCompletion>,
        TextOperationError<B::Error>,
    > {
        if S::text_operation_kind(self, &input, state) != TextOperationKind::CausalDecision {
            return S::submit_specialized_text_operation(
                self,
                input,
                decision,
                state,
                remaining_tokens,
            );
        }
        let decision = decision.ok_or(TextOperationPolicyError::MissingDecision)?;
        self.validate_session_admission()
            .map_err(TextOperationError::Backend)?;
        let (backend, session) = self.parts_mut();
        let submission = match input {
            PendingTextInput::Prefill(prompt) => {
                session.submit_causal_prefill(backend, prompt, decision.capture_domain(), state)
            }
            PendingTextInput::Decode(token) => {
                session.submit_causal_decode(backend, token, decision.capture_domain(), state)
            }
            PendingTextInput::Continue => session.submit_causal_continuation(backend, state),
        }?;
        let sampled =
            session.sample_causal_submission(backend, submission, decision.filter(), state)?;
        Ok(Submission {
            output: TextOperationOutput::Token(sampled.output),
            completion: sampled.completion,
        })
    }
}

/// Failure from backend preprocessing or backend-requested text encoding.
#[derive(Debug, thiserror::Error)]
pub enum MultimodalPreparationFailure<B, T>
where
    B: std::error::Error + 'static,
    T: std::error::Error + 'static,
{
    /// The selected architecture rejected the request before native preprocessing.
    #[error(transparent)]
    Capability(#[from] crate::CapabilityError),
    /// The selected backend rejected or failed media preprocessing.
    #[error("backend multimodal preparation failed: {0}")]
    Backend(#[source] B),
    /// The facade tokenizer failed on backend-required framing text.
    #[error("multimodal framing text encoding failed: {0}")]
    Text(#[source] T),
}

/// Preparation of decoded media by the installed session driver.
///
/// Caller text is tokenized before this boundary. Some processors introduce
/// checkpoint-defined framing text or video timestamps, so the callback keeps
/// that work on the facade's tokenizer. The driver returns its prepared prompt.
pub trait MultimodalPreparationSession<B: TextGenerationBackend>: TextOperationDriver<B> {
    /// Converts ordered token and decoded-media segments into the driver's prompt.
    fn prepare_multimodal_input<E>(
        &self,
        request: &TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self::Prompt, MultimodalPreparationFailure<B::Error, E>>
    where
        E: std::error::Error + Send + Sync + 'static;
}

/// Capability and resource observations for one selected text-model session.
///
/// Implementations inspect the opaque model and cache already owned by
/// [`ModelRuntime`]. The contract exposes only portable documents and the
/// driver's prepared prompt type. Tensor, stream, allocator, and
/// executable types remain inside the adapter.
pub trait ModelCapabilitySession<B: TextGenerationBackend>: TextOperationDriver<B> + Sized {
    /// Releases this execution's optional conversion-retention claims at an
    /// already settled boundary. Require healthy, idle submission authority;
    /// never settle pending controlled transactions or invalidate parameters.
    /// Preserve request/sampling state, snapshots, epochs and all run budgets.
    /// Keep eligible bindings available for future bounded admission. Other
    /// owners and native graphs may keep physical storage alive.
    fn trim_parameter_conversions(
        _runtime: &mut ModelRuntime<B, Self>,
    ) -> Result<
        Vec<crate::residency::ParameterConversionRetentionTrimReport>,
        crate::residency::ParameterConversionTrimError,
    > {
        Err(crate::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Reads load-selected conversion-retention policy and live usage. This must
    /// not allocate execution resources, evaluate, synchronize or mutate state.
    fn parameter_conversion_retention(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<
        crate::Observed<Vec<crate::residency::ParameterConversionRetentionReport>>,
        crate::BackendFailure,
    > {
        Self::static_memory(runtime)
            .map(|report| report.parameter_conversion_retention)
            .map_err(crate::BackendFailure::from_error)
    }

    /// Reports validated model capabilities for the selected session.
    fn model_capabilities(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<ModelCapabilities, CapabilityError>;

    /// Counts text and backend-specific model positions in a prepared prompt.
    fn count_prepared_input(
        runtime: &ModelRuntime<B, Self>,
        input: &Self::Prompt,
    ) -> Result<InputTokenCount, CapabilityError>;

    /// Observes complete ordered decoder IDs for capture and controlled admission.
    /// Includes architecture-declared media placeholders and supplied original
    /// IDs for projected text. Never reconstructs IDs from embeddings. An adapter
    /// without alignment observation returns a typed unavailable-capability failure.
    fn prepared_input_token_ids(
        _runtime: &ModelRuntime<B, Self>,
        _input: &Self::Prompt,
    ) -> Result<Vec<u32>, crate::BackendFailure> {
        Err(crate::BackendFailure::new(
            crate::BackendFailureKind::Unsupported,
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "prepared decoder token alignment is unavailable",
            ),
        ))
    }

    /// Estimates persistent and transient request state for this session.
    fn estimate_runtime_state(
        runtime: &ModelRuntime<B, Self>,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError>;

    /// Reports static model storage and current backend memory observations.
    fn static_memory(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<StaticMemoryReport, CapabilityError>;
}

/// Input for the next ordinary prediction. A committed token may still be the
/// pending decode input and must not be inserted into model state until that step.
pub enum PendingTextInput<P, T> {
    /// The original prepared prompt, before the first model prediction.
    Prefill(P),
    /// The preceding committed canonical token, before its successor is predicted.
    Decode(T),
    /// Resume physical work retained by the generation state.
    Continue,
}

impl<P, T> PendingTextInput<P, T> {
    /// Borrows the pending native input without cloning or evaluating it.
    pub fn as_ref(&self) -> PendingTextInput<&P, &T> {
        match self {
            Self::Prefill(prompt) => PendingTextInput::Prefill(prompt),
            Self::Decode(token) => PendingTextInput::Decode(token),
            Self::Continue => PendingTextInput::Continue,
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
    /// A declared physical operation violated the neutral advancement contract.
    #[error(transparent)]
    Operation(#[from] TextOperationPolicyError),
}

/// One constraint-committed token and its backend-owned output handle.
#[derive(Debug, Clone)]
pub struct ControlledToken<T> {
    output: T,
    token_id: u32,
    score_kind: crate::refinement::PredictionScoreKind,
    position: crate::refinement::GenerationPosition,
}

impl<T> ControlledToken<T> {
    /// Completed physical work and authoritative output coordinates.
    pub const fn position(&self) -> crate::refinement::GenerationPosition {
        self.position
    }
    /// Semantic meaning of scores associated with this committed token.
    pub const fn score_kind(&self) -> crate::refinement::PredictionScoreKind {
        self.score_kind
    }
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
pub struct ControlledTextGeneration<'a, B, C, S>
where
    B: TextGenerationBackend,
    S: TextExecutionSession<B>,
    C: TokenFilterController,
{
    runtime: &'a mut ModelRuntime<B, S>,
    inner: TextRequestDriver<B, C, S>,
}

type ControlledGenerationResult<B, C> = Result<
    <B as TextGenerationBackend>::Token,
    ControlledTextGenerationError<
        <B as BackendProvider>::Error,
        <C as TokenFilterController>::Error,
    >,
>;

impl<'a, B, C, S> ControlledTextGeneration<'a, B, C, S>
where
    B: TextGenerationBackend,
    S: TextExecutionSession<B>,
    C: TokenFilterController,
{
    /// Independent completed-work and authoritative-output coordinates.
    pub fn position(&self) -> crate::refinement::GenerationPosition {
        self.inner.position
    }
    /// Largest completed plain-text prefill chunk for this run, in tokens.
    pub fn prefill_chunk_tokens(&self) -> Option<usize> {
        self.inner.prefill_chunk_tokens
    }
    /// Active/pending mode and effective handoff position of this shared driver.
    pub fn decoding_state(&self) -> crate::decoding::DecodingState {
        self.inner.decoding.state()
    }
    /// Binds cross-thread mode commands to uninterrupted or explicit advancement.
    pub fn bind_control_handle(
        &self,
        handle: &crate::execution_control::GenerationControlHandle,
    ) -> Result<(), crate::decoding::DecodingModeError> {
        handle.bind_decoding(
            self.inner.decoding.clone(),
            self.inner.cancellation.as_ref(),
        )
    }

    /// Advances one physical operation using the same driver as the token iterator.
    #[allow(clippy::type_complexity)]
    pub fn advance_operation(
        &mut self,
    ) -> Result<Option<TextAdvancement<B::Token>>, ControlledTextGenerationError<B::Error, C::Error>>
    {
        self.inner.next_physical_committed(self.runtime).transpose()
    }
    /// Starts controlled generation from portable prompt token ids.
    pub fn new(
        runtime: &'a mut ModelRuntime<B, S>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let prompt = S::prepare_text_prompt(runtime.backend(), prompt_token_ids)
            .map_err(ControlledTextGenerationError::Backend);
        let prompt = runtime.finish_text_preparation(
            crate::run_preparation::TextPreparationStage::Prompt,
            prompt,
            ControlledTextGenerationError::Preparation,
        )?;
        Self::from_prompt(runtime, prompt, config, controller)
    }

    /// Starts controlled generation from the installed driver's prepared prompt.
    pub fn from_prompt(
        runtime: &'a mut ModelRuntime<B, S>,
        prompt: S::Prompt,
        config: TextGenerationConfig,
        controller: C,
    ) -> Result<Self, ControlledTextGenerationError<B::Error, C::Error>> {
        let inner = TextRequestDriver::new(runtime, prompt, config, controller)?;
        Ok(Self { runtime, inner })
    }

    /// Mutably borrows the canonical constraint state.
    pub fn controller_mut(&mut self) -> &mut C {
        &mut self.inner.controller
    }

    /// Observes cancellation between completed prefill chunks and predictions.
    pub fn set_cancellation_token(&mut self, cancellation: crate::GenerationCancellationToken) {
        self.inner.set_cancellation_token(cancellation);
    }

    /// Completes additional fallible host preparation before advancing this run.
    pub fn finish_text_preparation<T, E>(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<T, E> {
        self.runtime
            .finish_text_preparation(stage, local, map_backend)
    }

    /// Agrees initial cancellation/delivery without advancing this run.
    pub fn finish_text_preparation_cancellable<T, E>(
        &self,
        stage: crate::run_preparation::TextPreparationStage,
        local: Result<Option<T>, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        self.runtime
            .finish_text_preparation_cancellable(stage, local, map_backend)
    }

    /// Installs capture while preserving the existing sampling/controller state.
    #[cfg(feature = "tensor-capture")]
    pub fn enable_capture(
        &mut self,
        plan: crate::capture::AdmittedCapturePlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        let local = if !matches!(self.inner.step, Some(PendingTextInput::Prefill(_))) {
            Err(crate::capture::CaptureError::Invalid(
                "capture must be configured before generation".into(),
            ))
        } else {
            self.runtime
                .session()
                .configure_text_capture(&mut self.inner.backend_state, plan)
        };
        self.runtime.finish_text_preparation(
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Capture),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }

    /// Installs admitted interventions and captures before ordinary generation.
    #[cfg(feature = "tensor-capture")]
    pub fn enable_interventions(
        &mut self,
        capture: crate::capture::AdmittedCapturePlan,
        plan: crate::intervention::AdmittedInterventionPlan,
    ) -> Result<(), crate::run_preparation::TextCaptureSetupError> {
        let local = if !matches!(self.inner.step, Some(PendingTextInput::Prefill(_))) {
            Err(crate::capture::CaptureError::Invalid(
                "interventions must be configured before generation".into(),
            ))
        } else {
            self.runtime.session().configure_text_interventions(
                &mut self.inner.backend_state,
                capture,
                plan,
            )
        };
        self.runtime.finish_text_preparation(
            crate::run_preparation::TextPreparationStage::Instrumentation,
            local.map_err(crate::run_preparation::TextCaptureSetupError::Capture),
            crate::run_preparation::TextCaptureSetupError::Preparation,
        )
    }

    /// Borrows the model at a completed boundary before the final prefill.
    /// The caller drains the step's captures before accessing the runtime.
    pub fn prefill_boundary(&mut self) -> Result<Option<&mut ModelRuntime<B, S>>, B::Error> {
        self.settle_step()?;
        Ok(
            matches!(self.inner.step, Some(PendingTextInput::Prefill(_)))
                .then_some(&mut *self.runtime),
        )
    }

    /// Establishes exact native completion at the current operation boundary.
    pub fn settle_step(&mut self) -> Result<(), B::Error> {
        self.inner.resolve_completions_before_decode()
    }

    /// Establishes exact completion before delivering this step's host captures.
    /// The ordinary machine retains authority and handles errors/drop as before.
    #[cfg(feature = "tensor-capture")]
    pub fn take_captured_step(&mut self) -> Result<Option<crate::capture::CapturedStep>, B::Error> {
        self.inner.resolve_completions_before_decode()?;
        Ok(S::take_text_capture(&mut self.inner.backend_state))
    }

    /// Completes one nonfinal prefill chunk without sampling or constraint
    /// advancement. Drain captures before advancing another chunk or token.
    pub fn advance_prefill(
        &mut self,
    ) -> Result<bool, ControlledTextGenerationError<B::Error, C::Error>> {
        if self.inner.position.execution_step == u64::MAX {
            return Err(TextOperationPolicyError::PositionOverflow.into());
        }
        self.inner.advance_prefill(self.runtime)
    }
}

impl<B, C, S> Iterator for ControlledTextGeneration<'_, B, C, S>
where
    B: TextGenerationBackend,
    S: TextExecutionSession<B>,
    C: TokenFilterController,
{
    type Item =
        Result<ControlledToken<B::Token>, ControlledTextGenerationError<B::Error, C::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_committed(self.runtime)
    }
}

/// Backend-generic iterator over committed native token handles.
///
/// Exact prediction completion and canonical identity validation precede each
/// yield through the common request driver. The native handle remains available
/// for the following decode; callers need not reconstruct or copy token tensors.
/// Dropping the iterator settles retained submissions. Drop cannot report
/// settlement errors; use [`ModelRuntime::synchronize`] to establish successful
/// settlement before reuse or eviction, and [`ModelRuntime::reset`] for a fresh run.
pub struct TextGeneration<'a, B: TextGenerationBackend, S: TextExecutionSession<B>> {
    runtime: &'a mut ModelRuntime<B, S>,
    inner: TextRequestDriver<B, FixedTokenFilter, S>,
}

impl<'a, B: TextGenerationBackend, S: TextExecutionSession<B>> TextGeneration<'a, B, S> {
    /// Largest completed plain-text prefill chunk for this run, in tokens.
    pub fn prefill_chunk_tokens(&self) -> Option<usize> {
        self.inner.prefill_chunk_tokens
    }

    /// Establishes a healthy settled boundary without consuming pending input.
    /// Completion failure closes the iterator; it must not resume unknown state.
    pub fn synchronize(&mut self) -> Result<(), BackendFailure> {
        let result = self
            .inner
            .resolve_completions_before_decode()
            .map_err(BackendFailure::from_error)
            .and_then(|()| self.runtime.synchronize());
        if result.is_err() {
            self.inner.step = None;
        }
        result
    }

    /// Borrows the installed sampler and pending input without executing or settling
    /// work. Native observers must independently reject an in-flight session.
    #[allow(clippy::type_complexity)]
    pub fn observation_parts(
        &self,
    ) -> (
        &ModelRuntime<B, S>,
        &S::TextGenerationState,
        Option<PendingTextInput<&S::Prompt, &B::Token>>,
    ) {
        (
            self.runtime,
            &self.inner.backend_state,
            if self.inner.remaining_tokens == Some(0) {
                None
            } else {
                self.inner.step.as_ref().map(PendingTextInput::as_ref)
            },
        )
    }

    /// Canonical sequence owned by the common request driver.
    pub fn committed_sequence(&self) -> &crate::GenerationSequence {
        self.inner.sequence()
    }

    /// Read-only active and queued policies of this logical generation.
    pub fn decoding_state(&self) -> crate::decoding::DecodingState {
        self.inner.decoding.state()
    }

    /// Starts generation from portable prompt token ids.
    pub fn new(
        runtime: &'a mut ModelRuntime<B, S>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
    ) -> Result<Self, BackendFailure> {
        Self::with_token_filter(runtime, prompt_token_ids, config, TokenFilter::All)
    }

    /// Starts ordinary generation with a fixed portable validity restriction.
    /// Tokenizer-owning callers supply their mapped IDs here; sampling, history,
    /// completions and limits use the same request driver as controlled generation.
    pub fn with_token_filter(
        runtime: &'a mut ModelRuntime<B, S>,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
        filter: TokenFilter,
    ) -> Result<Self, BackendFailure> {
        let prompt = S::prepare_text_prompt(runtime.backend(), prompt_token_ids)
            .map_err(BackendFailure::from_error);
        let prompt = runtime.finish_text_preparation(
            crate::run_preparation::TextPreparationStage::Prompt,
            prompt,
            std::convert::identity,
        )?;
        let inner = TextRequestDriver::new(runtime, prompt, config, FixedTokenFilter(filter))
            .map_err(unreachable_unconstrained_error)?;
        Ok(Self { runtime, inner })
    }

    /// Starts generation from the installed driver's prepared prompt.
    pub fn from_prompt(
        runtime: &'a mut ModelRuntime<B, S>,
        prompt: S::Prompt,
        config: TextGenerationConfig,
    ) -> Result<Self, BackendFailure> {
        let inner =
            TextRequestDriver::new(runtime, prompt, config, FixedTokenFilter(TokenFilter::All))
                .map_err(unreachable_unconstrained_error)?;
        Ok(Self { runtime, inner })
    }
}

fn unreachable_unconstrained_error<B>(
    error: ControlledTextGenerationError<B, std::convert::Infallible>,
) -> BackendFailure
where
    B: std::error::Error + Send + Sync + 'static,
{
    match error {
        ControlledTextGenerationError::Backend(error) => BackendFailure::from_error(error),
        ControlledTextGenerationError::Preparation(error) => error,
        ControlledTextGenerationError::Operation(error) => BackendFailure::from_error(error),
        ControlledTextGenerationError::Controller(error) => match error {},
    }
}

impl<B: TextGenerationBackend, S: TextExecutionSession<B>> Iterator for TextGeneration<'_, B, S> {
    type Item = Result<B::Token, BackendFailure>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next_output(self.runtime)
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
pub trait DistributedModelSession<B: BackendProvider>: BackendSession<B> {
    /// Selected distributed session implementation.
    type DistributedSession: DistributedSession<Error = B::Error>;

    /// Returns communication for a distributed model session.
    fn distributed_session(&self) -> Option<&Self::DistributedSession>;
}

/// Native cache storage and completion operations for local components.
pub trait LocalComponentStorage: eredu_nn::NeuralBackend {
    /// Native tensors and transfers for the selected component cache.
    type CacheStorage: crate::cache::mechanisms::CacheTensorStorage<Value = Self::Tensor>;

    /// Establishes completion of the supplied values.
    fn settle_local_values<'a>(
        values: impl Iterator<Item = &'a Self::Tensor>,
    ) -> Result<(), crate::BackendFailure>
    where
        Self::Tensor: 'a;
}

#[cfg(test)]
mod tests {
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
        type Error = Infallible;
        fn descriptor(&self) -> BackendDescriptor {
            BackendDescriptor::new("mock", "1")
        }
        fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
            Ok(vec![])
        }
    }

    #[derive(Default)]
    struct LoadingMock {
        selections: std::sync::atomic::AtomicUsize,
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
        type Error = std::convert::Infallible;

        fn descriptor(&self) -> BackendDescriptor {
            BackendDescriptor::new("loading-mock", "1")
        }

        fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
            Ok(Vec::new())
        }
    }

    impl ModelSessionInstaller<LoadingMock> for LoadingMockSession {
        type Model = u32;
        type Session = Self;

        fn install_session(
            _: &LoadingMock,
            _: PreparedModel<u32>,
        ) -> Result<Self, std::convert::Infallible> {
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

    struct RetainLoadingPlan(crate::PreparationAdmission);

    impl ModelSourcePreparation<()> for RetainLoadingPlan {
        type Prepared = ModelPreparationPlan<()>;
        type Error = Infallible;

        fn admission(&self) -> crate::PreparationAdmission {
            self.0
        }

        fn prepare(self, plan: ModelPreparationPlan<()>) -> Result<Self::Prepared, Self::Error> {
            Ok(plan)
        }
    }

    struct FixtureLoadFacts(u32);

    impl DirectLoadFacts for FixtureLoadFacts {
        type Request = u32;
        type Mechanisms = ();
        fn rank(_: &u32) -> Option<crate::ParallelRankTopology> {
            None
        }
        fn from_request(options: u32, (): ()) -> Self {
            Self(options)
        }
    }

    impl ModelLoadAdmission<()> for FixtureLoadFacts {
        type SourcePreparation = RetainLoadingPlan;
        type Error = Infallible;

        fn select(self, _: &ArtifactInspection) -> Result<Self::SourcePreparation, Self::Error> {
            let Self(options) = self;
            let policy = crate::PreparationPolicy::default().with_required_session_capabilities(
                SessionCapabilities::default().with_activation_inspection(options == 99),
            );
            let request = crate::PreparationAdmissionRequest::new(
                crate::LoadingProtocol::Model,
                crate::ArtifactFormat::SafeTensors,
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
                        SessionCapabilities::default().with_activation_inspection(options == 99),
                    ),
            )
            .expect("mock admission facts are coherent");
            Ok(RetainLoadingPlan(admission))
        }
    }

    impl ModelLoadingBackend for LoadingMock {
        type LoadPlacement = u32;
        type Binding = u32;
        type LoadMechanisms = ();

        fn load_mechanisms(&self) {
            self.selections
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        fn bind_load_placement(
            &self,
            _: Option<crate::ParallelRankTopology>,
            selected: Option<u32>,
        ) -> Result<u32, Self::Error> {
            Ok(selected.unwrap())
        }
    }

    impl BindModelSources<LoadingMock> for ModelPreparationPlan<()> {
        type Model = u32;
        fn construct_model(
            self,
            backend: &LoadingMock,
            selected: u32,
        ) -> Result<PreparedModel<u32>, BackendFailure> {
            backend
                .materializations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            assert_eq!(self.inspection().configuration().family(), "llama");
            Ok(PreparedModel::new(
                selected,
                self.admitted_session_capabilities(),
            ))
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
        prefill_lengths: Vec<usize>,
        cancel_on_prefill: Option<crate::GenerationCancellationToken>,
        prepare_ingress: bool,
    }
    impl ModelSessionInstaller<Mock> for MockSession {
        type Model = u32;
        type Session = Self;

        fn install_session(_: &Mock, model: PreparedModel<u32>) -> Result<Self, Infallible> {
            Ok(MockSession {
                model: model.into_inner(),
                tokens: vec![],
                distributed: None,
                prefill_lengths: Vec::new(),
                cancel_on_prefill: None,
                prepare_ingress: false,
            })
        }
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
            self.prefill_lengths.push(input.len());
            self.tokens.extend(input);
            if let Some(cancellation) = &self.cancel_on_prefill {
                cancellation.cancel();
            }
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

    impl TextSessionLifecycle<Mock> for MockSession {
        fn reset_session(&mut self, _: &Mock) -> Result<(), BackendFailure> {
            self.tokens.clear();
            Ok(())
        }

        fn synchronize_session(&self, _: &Mock) -> Result<(), BackendFailure> {
            Ok(())
        }
    }

    impl CausalTextSession<Mock, Vec<u32>, u32, (u32, u64)> for MockSession {}

    impl TextOperationDriver<Mock> for MockSession {
        type Prompt = Vec<u32>;
        fn prepare_text_prompt(
            _: &Mock,
            prompt_token_ids: Vec<u32>,
        ) -> Result<
            <MockSession as TextOperationDriver<Mock>>::Prompt,
            <Mock as BackendProvider>::Error,
        > {
            Ok(prompt_token_ids)
        }

        type TextGenerationState = (u32, u64);

        fn start_text_generation(
            &self,
            _: &Mock,
            config: TextGenerationConfig,
        ) -> Result<
            <Self as TextOperationDriver<Mock>>::TextGenerationState,
            <Mock as BackendProvider>::Error,
        > {
            Ok((config.sampling().top_k as u32, config.seed()))
        }

        fn submit_specialized_text_operation(
            runtime: &mut ModelRuntime<Mock, MockSession>,
            input: crate::PendingTextInput<
                <MockSession as TextOperationDriver<Mock>>::Prompt,
                <Mock as TextGenerationBackend>::Token,
            >,
            decision: Option<&crate::TokenSamplingDecision<'_>>,
            state: &mut <Self as TextOperationDriver<Mock>>::TextGenerationState,
            _remaining_tokens: Option<usize>,
        ) -> Result<
            Submission<
                crate::TextOperationOutput<
                    <MockSession as TextOperationDriver<Mock>>::Prompt,
                    <Mock as TextGenerationBackend>::Token,
                >,
                <Mock as TextGenerationBackend>::TextCompletion,
            >,
            crate::TextOperationError<<Mock as BackendProvider>::Error>,
        > {
            let decision = decision.ok_or(crate::TextOperationPolicyError::MissingDecision)?;
            let filter = decision.filter();
            if matches!(&input, crate::PendingTextInput::Continue) {
                return Err(crate::TextOperationPolicyError::UnsupportedContinuation.into());
            }
            let submission =
                || -> Result<Submission<<Mock as TextGenerationBackend>::Token, <Mock as TextGenerationBackend>::TextCompletion>, <Mock as BackendProvider>::Error> {
                    match input {
                        crate::PendingTextInput::Prefill(prompt) => {
                            let submission = runtime.prefill(prompt)?;
                            Ok(Submission {
                                output: apply_mock_filter(
                                    submission.output + state.0 + state.1 as u32,
                                    filter,
                                ),
                                completion: submission.completion,
                            })
                        }
                        crate::PendingTextInput::Decode(token) => {
                            let submission = runtime.decode(token)?;
                            Ok(Submission {
                                output: apply_mock_filter(submission.output, filter),
                                completion: submission.completion,
                            })
                        }
                        crate::PendingTextInput::Continue => {
                            unreachable!("continuation is rejected before causal submission")
                        }
                    }
                };
            let submission = submission().map_err(crate::TextOperationError::Backend)?;
            Ok(Submission {
                output: crate::TextOperationOutput::Token(submission.output),
                completion: submission.completion,
            })
        }

        fn prefill_text_prefix(
            runtime: &mut ModelRuntime<Mock, MockSession>,
            prompt: &mut Vec<u32>,
            maximum: std::num::NonZeroUsize,
            _: &mut (u32, u64),
        ) -> Result<bool, Infallible> {
            if std::mem::take(&mut runtime.session_mut().prepare_ingress) {
                return Ok(true);
            }
            if prompt.len() <= maximum.get() {
                return Ok(false);
            }
            let suffix = prompt.split_off(maximum.get());
            let prefix = std::mem::replace(prompt, suffix);
            runtime.prefill(prefix)?.completion.wait()?;
            Ok(true)
        }
    }

    impl AutomaticPromptCacheSession<Mock> for MockSession {}

    impl TextGenerationBackend for Mock {
        type InputTensor = u32;

        type Token = u32;

        type TextCompletion = Done;
    }

    impl MultimodalPreparationSession<Mock> for MockSession {
        fn prepare_multimodal_input<E>(
            &self,
            request: &TokenizedMultimodalRequest,
            _: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        ) -> Result<
            <MockSession as TextOperationDriver<Mock>>::Prompt,
            MultimodalPreparationFailure<<Mock as BackendProvider>::Error, E>,
        >
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

    impl ModelCapabilitySession<Mock> for MockSession {
        fn model_capabilities(
            _: &ModelRuntime<Mock, Self>,
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
            _: &ModelRuntime<Mock, Self>,
            input: &<MockSession as TextOperationDriver<Mock>>::Prompt,
        ) -> Result<InputTokenCount, CapabilityError> {
            Ok(InputTokenCount::text(input.len() as u64))
        }

        fn estimate_runtime_state(
            _: &ModelRuntime<Mock, Self>,
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
            runtime: &ModelRuntime<Mock, Self>,
        ) -> Result<StaticMemoryReport, CapabilityError> {
            let unavailable = || crate::Observed::unavailable("mock does not expose this counter");
            Ok(StaticMemoryReport {
                parameter_conversion_retention:
                    crate::residency::unreported_parameter_conversion_retention(),
                logical_parameter_bytes: crate::Observed::exact(
                    u64::from(runtime.session().model),
                    "mock model",
                ),
                current_host_resident_bytes: unavailable(),
                current_device_resident_bytes: unavailable(),
                current_device_parameter_conversion_bytes: unavailable(),
                planned_disk_backed_bytes: unavailable(),
                backend_active_allocation_bytes: unavailable(),
                backend_allocator_cache_bytes: unavailable(),
                backend_device_allocated_bytes: unavailable(),
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
        let prepared = load_model::<_, _, FixtureLoadFacts>(
            &LoadingMock::default(),
            root.path(),
            &LOADING_CONFIGURATION_RESOLVER,
            ModelLoadOptions::new(41).with_placement(41),
        )
        .unwrap();
        assert_eq!(*prepared, 41);

        let backend = LoadingMock::default();
        let prepared = load_model::<_, _, FixtureLoadFacts>(
            &backend,
            root.path(),
            &LOADING_CONFIGURATION_RESOLVER,
            ModelLoadOptions::new(7).with_placement(7),
        )
        .unwrap();
        let runtime = ModelRuntime::install::<LoadingMockSession>(backend, prepared).unwrap();
        assert_eq!(runtime.backend().descriptor().name, "loading-mock");

        let missing = root.path().join("missing");
        assert!(matches!(
            load_model::<_, _, FixtureLoadFacts>(&LoadingMock::default(), &missing, &LOADING_CONFIGURATION_RESOLVER, ModelLoadOptions::new(1).with_placement(1)),
            Err(ModelLoadError::Artifact(ArtifactError::MissingArtifact(path)))
                if path == missing
        ));
    }

    #[test]
    fn session_requirement_is_retained_by_the_single_admission() {
        let root = tempfile::tempdir().unwrap();
        write_loading_fixture(root.path());
        let backend = LoadingMock::default();

        let prepared = load_model::<_, _, FixtureLoadFacts>(
            &backend,
            root.path(),
            &LOADING_CONFIGURATION_RESOLVER,
            ModelLoadOptions::new(99).with_placement(99),
        )
        .unwrap();

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

    #[derive(Clone)]
    struct FixedController {
        queries: usize,
        tokens: Vec<u32>,
        committed: usize,
    }

    impl TokenFilterController for FixedController {
        type Error = Infallible;

        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            self.queries += 1;
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

    impl TokenFilterController for &mut FixedController {
        type Error = Infallible;
        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            (**self).current_filter()
        }
        fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
            (**self).commit_token(token)
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            (**self).is_complete()
        }
    }

    struct PhasedMock;
    impl BackendProvider for PhasedMock {
        type Error = Infallible;
        fn descriptor(&self) -> BackendDescriptor {
            BackendDescriptor::new("phased_mock", "1")
        }
        fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Infallible> {
            Mock.devices()
        }
    }

    impl ModelSessionInstaller<PhasedMock> for MockSession {
        type Model = u32;
        type Session = Self;

        fn install_session(_: &PhasedMock, model: PreparedModel<u32>) -> Result<Self, Infallible> {
            <Self as ModelSessionInstaller<Mock>>::install_session(&Mock, model)
        }
    }

    impl BackendSession<PhasedMock> for MockSession {
        type PrefillInput = Vec<u32>;
        type DecodeInput = u32;
        type Output = u32;
        type Completion = Done;
        fn capabilities(&self) -> SessionCapabilities {
            SessionCapabilities::default()
        }
        fn prefill(
            &mut self,
            _: &PhasedMock,
            input: Vec<u32>,
        ) -> Result<Submission<u32, Done>, Infallible> {
            <Self as BackendSession<Mock>>::prefill(self, &Mock, input)
        }
        fn decode(
            &mut self,
            _: &PhasedMock,
            input: u32,
        ) -> Result<Submission<u32, Done>, Infallible> {
            <Self as BackendSession<Mock>>::decode(self, &Mock, input)
        }
        fn observe_output(
            &self,
            _: &PhasedMock,
            output: &u32,
        ) -> Result<ObservationSet, Infallible> {
            <Self as BackendSession<Mock>>::observe_output(self, &Mock, output)
        }
    }
    #[derive(Clone, Default)]
    struct PhasedState {
        context: u32,
        iterations: u32,
        block: u64,
    }
    impl TextSessionLifecycle<PhasedMock> for MockSession {
        fn reset_session(&mut self, _: &PhasedMock) -> Result<(), BackendFailure> {
            <Self as TextSessionLifecycle<Mock>>::reset_session(self, &Mock)
        }
        fn synchronize_session(&self, _: &PhasedMock) -> Result<(), BackendFailure> {
            <Self as TextSessionLifecycle<Mock>>::synchronize_session(self, &Mock)
        }
    }

    impl CausalTextSession<PhasedMock, Vec<u32>, u32, PhasedState> for MockSession {}

    impl TextOperationDriver<PhasedMock> for MockSession {
        type Prompt = Vec<u32>;
        fn prepare_text_prompt(_: &PhasedMock, prompt: Vec<u32>) -> Result<Vec<u32>, Infallible> {
            Ok(prompt)
        }

        type TextGenerationState = PhasedState;

        fn start_text_generation(
            &self,
            _: &PhasedMock,
            _: TextGenerationConfig,
        ) -> Result<PhasedState, Infallible> {
            Ok(PhasedState::default())
        }

        fn supports_block_refinement(&self) -> bool {
            true
        }

        fn text_operation_kind(
            _: &ModelRuntime<PhasedMock, MockSession>,
            input: &PendingTextInput<Vec<u32>, u32>,
            state: &PhasedState,
        ) -> TextOperationKind {
            use crate::refinement::{GenerationPhase, PredictionScoreKind};
            match input {
                PendingTextInput::Prefill(_) | PendingTextInput::Decode(_) => {
                    TextOperationKind::Progress(GenerationPhase::Context)
                }
                PendingTextInput::Continue if state.iterations < 2 => {
                    TextOperationKind::Progress(GenerationPhase::Refinement)
                }
                PendingTextInput::Continue => {
                    TextOperationKind::Decision(PredictionScoreKind::ConstrainedFinalization)
                }
            }
        }

        fn submit_specialized_text_operation(
            runtime: &mut ModelRuntime<PhasedMock, MockSession>,
            input: PendingTextInput<Vec<u32>, u32>,
            decision: Option<&TokenSamplingDecision<'_>>,
            state: &mut PhasedState,
            _remaining_tokens: Option<usize>,
        ) -> Result<
            Submission<TextOperationOutput<Vec<u32>, u32>, Done>,
            TextOperationError<Infallible>,
        > {
            use crate::refinement::GenerationPhase;
            let phase = match input {
                PendingTextInput::Prefill(prompt) => {
                    assert!(decision.is_none());
                    state.context = runtime.prefill(prompt).unwrap().wait().unwrap();
                    state.iterations = 0;
                    GenerationPhase::Context
                }
                PendingTextInput::Decode(token) => {
                    assert!(decision.is_none());
                    state.context = runtime.decode(token).unwrap().wait().unwrap();
                    state.iterations = 0;
                    GenerationPhase::Context
                }
                PendingTextInput::Continue if state.iterations < 2 => {
                    assert!(decision.is_none());
                    state.iterations += 1;
                    GenerationPhase::Refinement
                }
                PendingTextInput::Continue => {
                    let decision = decision.ok_or(TextOperationPolicyError::MissingDecision)?;
                    let token =
                        apply_mock_filter(state.context + state.iterations, decision.filter());
                    state.block += 1;
                    return Ok(Submission {
                        output: TextOperationOutput::Token(token),
                        completion: Done,
                    });
                }
            };
            Ok(Submission {
                output: TextOperationOutput::Progress {
                    pending: PendingTextInput::Continue,
                    progress: TextProgress {
                        phase,
                        block: state.block,
                        iteration: state.iterations,
                        tentative_token_ids: None,
                    },
                },
                completion: Done,
            })
        }
    }

    impl AutomaticPromptCacheSession<PhasedMock> for MockSession {}

    impl TextGenerationBackend for PhasedMock {
        type InputTensor = u32;

        type Token = u32;

        type TextCompletion = Done;
    }

    #[test]
    fn mismatched_refinement_policy_is_rejected_before_context_or_sampling() {
        let sampling =
            crate::resolve_generation_config(None, crate::GenerationConfigOverrides::default())
                .unwrap();
        let config = TextGenerationConfig::new(sampling).with_decoding_mode(Some(
            crate::decoding::DecodingMode::Diffusion {
                policy: Some(crate::refinement::RefinementPolicy::MaskedUnmasking {
                    config: Default::default(),
                    mask_token_id: 2,
                }),
            },
        ));
        let mut runtime = ModelRuntime::<PhasedMock, MockSession>::install::<MockSession>(
            PhasedMock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let controller = FixedController {
            queries: 0,
            tokens: vec![5],
            committed: 0,
        };
        let result = ControlledTextGeneration::new(&mut runtime, vec![1, 2], config, controller);
        assert!(matches!(
            result,
            Err(ControlledTextGenerationError::Operation(
                TextOperationPolicyError::DecodingMode(
                    crate::decoding::DecodingModeError::Unsupported("refinement numerical policy")
                )
            ))
        ));
        drop(result);
        assert!(runtime.session().tokens.is_empty());
    }

    #[test]
    fn physical_advancement_preserves_commitment_controller_and_ordinary_parity() {
        use crate::refinement::{GenerationPhase, PredictionScoreKind};
        let sampling = crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let config = TextGenerationConfig::new(sampling);
        let controller = FixedController {
            queries: 0,
            tokens: vec![5, 8],
            committed: 0,
        };
        let mut ordinary_runtime = ModelRuntime::<PhasedMock, MockSession>::install::<MockSession>(
            PhasedMock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let mut ordinary = ControlledTextGeneration::new(
            &mut ordinary_runtime,
            vec![1, 2],
            config.clone(),
            controller.clone(),
        )
        .unwrap();
        let expected = ordinary
            .by_ref()
            .map(|token| token.unwrap().token_id())
            .collect::<Vec<_>>();
        assert_eq!(ordinary.inner.controller.queries, 2);
        drop(ordinary);
        let mut runtime = ModelRuntime::<PhasedMock, MockSession>::install::<MockSession>(
            PhasedMock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver.start(vec![1, 2], config, controller).unwrap();
        let mut actual = Vec::new();
        let mut phases = Vec::new();
        while let Some(advance) = driver.advance_operation(&mut state).unwrap() {
            match advance {
                TextAdvancement::Progress(progress) => phases.push(progress.phase),
                TextAdvancement::ModeChanged(_) => panic!("no transition was requested"),
                TextAdvancement::Token(token) => {
                    assert_eq!(
                        token.score_kind(),
                        PredictionScoreKind::ConstrainedFinalization
                    );
                    actual.push(token.token_id());
                    phases.push(GenerationPhase::Finalization);
                }
            }
            assert_eq!(state.position().execution_step, phases.len() as u64);
            assert_eq!(state.position().committed_tokens, actual.len() as u64);
            assert_eq!(state.controller().queries, actual.len());
            assert_eq!(state.controller().committed, actual.len());
            assert!(matches!(
                driver.advance_operation(&mut state),
                Err(TextContinuationError::NotQuiescent)
            ));
            driver.take_completed_step(&mut state).unwrap();
        }
        assert_eq!(actual, [5, 8]);
        assert_eq!(actual, expected);
        assert_eq!(
            phases,
            [
                GenerationPhase::Context,
                GenerationPhase::Refinement,
                GenerationPhase::Refinement,
                GenerationPhase::Finalization
            ]
            .repeat(2)
        );
        assert_eq!(driver.runtime().session().tokens, [1, 2, 5]);
        assert_eq!(
            driver.runtime().session().tokens,
            ordinary_runtime.session().tokens
        );
    }

    #[test]
    fn lifecycle_preserves_selected_target_and_loaded_parameters() {
        let prepared = PreparedModel::new(10, SessionCapabilities::default());
        let mut runtime = ModelRuntime::<Mock, MockSession>::install_with_target::<MockSession>(
            Mock,
            prepared,
            Some(42),
        )
        .unwrap();
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
    fn shared_prefill_chunks_preserve_tokens_and_only_sample_the_final_chunk() {
        for length in [1, 2, 5, 8] {
            let config = continuation_config(4);
            let prompt = (0..length)
                .map(|index| index as u32 + 1)
                .collect::<Vec<_>>();
            let mut ordinary_runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
                Mock,
                PreparedModel::new(10, SessionCapabilities::default()),
            )
            .unwrap();
            let ordinary = TextGeneration::new(
                &mut ordinary_runtime,
                prompt.clone(),
                config
                    .clone()
                    .with_prefill_chunk_policy(PrefillChunkPolicy::Unchunked),
            )
            .unwrap()
            .map(|token| token.unwrap())
            .collect::<Vec<_>>();
            let mut chunk_runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
                Mock,
                PreparedModel::new(10, SessionCapabilities::default()),
            )
            .unwrap();
            let chunked = ControlledTextGeneration::new(
                &mut chunk_runtime,
                prompt.clone(),
                config.with_prefill_chunk_policy(PrefillChunkPolicy::Bounded(
                    std::num::NonZeroUsize::new(2).unwrap(),
                )),
                FixedTokenFilter(TokenFilter::All),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id())
            .collect::<Vec<_>>();
            assert_eq!(ordinary, chunked);
            assert_eq!(
                ordinary_runtime.session().tokens,
                chunk_runtime.session().tokens
            );
            assert_eq!(
                chunk_runtime.session().prefill_lengths,
                prompt.chunks(2).map(<[u32]>::len).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn cancellation_after_a_prefill_chunk_does_not_sample_or_decode() {
        let cancellation = crate::GenerationCancellationToken::new();
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        runtime.session_mut().cancel_on_prefill = Some(cancellation.clone());
        let config = continuation_config(4).with_prefill_chunk_policy(PrefillChunkPolicy::Bounded(
            std::num::NonZeroUsize::new(2).unwrap(),
        ));
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2, 3, 4, 5],
            config,
            FixedTokenFilter(TokenFilter::All),
        )
        .unwrap();
        generation.set_cancellation_token(cancellation);
        assert!(generation.next().is_none());
        assert!(generation.next().is_none());
        drop(generation);
        assert_eq!(runtime.session().tokens, [1, 2]);
        assert_eq!(runtime.session().prefill_lengths, [2]);
    }

    #[test]
    fn portable_text_generation_prefills_and_decodes_without_tensor_types() {
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
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
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
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
        let prompt = runtime
            .session()
            .prepare_multimodal_input(&request, &mut |_| Ok::<_, Infallible>(Vec::new()))
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
    fn controlled_generation_applies_portable_filters_and_commits_tokens() {
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let sampling = crate::resolve_generation_config(
            None,
            crate::GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let mut controller = FixedController {
            queries: 0,
            tokens: vec![7, 8],
            committed: 0,
        };
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(sampling),
            &mut controller,
        )
        .unwrap();
        let handle = crate::execution_control::GenerationControlHandle::default();
        generation.bind_control_handle(&handle).unwrap();
        assert_eq!(generation.next().unwrap().unwrap().token_id(), 7);
        assert!(!handle.decoding_state().unwrap().terminal);
        assert_eq!(generation.next().unwrap().unwrap().token_id(), 8);
        assert!(handle.decoding_state().unwrap().terminal);
        assert_eq!(
            handle.request_decoding_mode(crate::decoding::DecodingMode::Autoregressive),
            Err(crate::decoding::DecodingModeError::Terminal)
        );
        assert!(generation.controller_mut().is_complete().unwrap());
        assert!(generation.next().is_none());
        drop(generation);
        assert_eq!(controller.queries, 2);
        assert_eq!(controller.committed, 2);
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
    fn detached_prefill_advances_one_completed_chunk_without_sampling() {
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        runtime.session_mut().prepare_ingress = true;
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start(
                vec![1, 2, 3, 4, 5],
                continuation_config(2).with_prefill_chunk_policy(PrefillChunkPolicy::Bounded(
                    std::num::NonZeroUsize::new(2).unwrap(),
                )),
                FixedController {
                    queries: 0,
                    tokens: vec![7, 8],
                    committed: 0,
                },
            )
            .unwrap();
        assert!(driver.advance_prefill(&mut state).unwrap());
        assert!(driver.runtime().session().tokens.is_empty());
        assert_eq!(state.controller().queries, 0);
        assert_eq!(state.remaining_tokens(), Some(2));
        driver.take_completed_step(&mut state).unwrap();
        assert!(driver.advance_prefill(&mut state).unwrap());
        assert_eq!(driver.runtime().session().tokens, [1, 2]);
        assert_eq!(state.controller().queries, 0);
        assert_eq!(state.remaining_tokens(), Some(2));
        assert!(matches!(
            state.require_quiescent(),
            Err(TextContinuationError::NotQuiescent)
        ));
        driver.take_completed_step(&mut state).unwrap();
        let mut child = {
            let boundary = driver.quiescent(&mut state).unwrap();
            let (_, sampler, pending) = boundary.parts();
            let Some(PendingTextInput::Prefill(pending)) = pending else {
                panic!("pending prefill")
            };
            assert_eq!(pending, &[3, 4, 5]);
            boundary.fork_host_state(
                *sampler,
                FixedController {
                    queries: 0,
                    tokens: vec![7, 8],
                    committed: 0,
                },
                Some(PendingTextInput::Prefill(pending.clone())),
                Some(2),
                boundary.position(),
                boundary.decoding_state(),
                boundary.sequence().clone(),
            )
        };
        // The child shares only execution authority, with its own remaining cursor.
        assert!(driver.advance_prefill(&mut child).unwrap());
        driver.take_completed_step(&mut child).unwrap();
        assert_eq!(driver.runtime().session().tokens, [1, 2, 3, 4]);
        assert!(state.is_prefill_pending());
        assert_eq!(child.controller().queries, 0);
        assert!(!driver.advance_prefill(&mut child).unwrap());
        assert_eq!(driver.advance(&mut child).unwrap().unwrap().token_id(), 7);
        driver.take_completed_step(&mut child).unwrap();
        assert_eq!(child.controller().queries, 1);
        assert_eq!(child.controller().committed, 1);
        assert_eq!(child.remaining_tokens(), Some(1));
        assert_eq!(driver.runtime().session().prefill_lengths, [2, 2, 1]);
    }

    #[test]
    fn detached_prefill_cancellation_keeps_sampler_and_pending_input_untouched() {
        let cancellation = crate::GenerationCancellationToken::new();
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start(
                vec![1, 2, 3, 4, 5],
                continuation_config(2).with_prefill_chunk_policy(PrefillChunkPolicy::Bounded(
                    std::num::NonZeroUsize::new(2).unwrap(),
                )),
                FixedController {
                    queries: 0,
                    tokens: vec![7, 8],
                    committed: 0,
                },
            )
            .unwrap();
        state.set_cancellation_token(cancellation.clone());
        assert!(driver.advance_prefill(&mut state).unwrap());
        driver.take_completed_step(&mut state).unwrap();
        cancellation.cancel();
        assert!(!driver.advance_prefill(&mut state).unwrap());
        assert!(driver.advance(&mut state).unwrap().is_none());
        assert_eq!(state.controller().queries, 0);
        assert_eq!(state.controller().committed, 0);
        assert_eq!(driver.runtime().session().tokens, [1, 2]);
    }

    #[test]
    fn detached_ordinary_continuation_preserves_pending_input_and_commit_order() {
        let controller = || FixedController {
            queries: 0,
            tokens: vec![7, 8, 9],
            committed: 0,
        };
        let mut ordinary_runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let ordinary: Vec<_> = ControlledTextGeneration::new(
            &mut ordinary_runtime,
            vec![1, 2],
            continuation_config(3),
            controller(),
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect();

        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
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
            #[cfg(feature = "tensor-capture")]
            assert!(driver.take_completed_step(&mut state).unwrap().is_none());
            #[cfg(not(feature = "tensor-capture"))]
            driver.take_completed_step(&mut state).unwrap();
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
        let mut first = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let mut other = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
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
            foreign.take_completed_step(&mut state),
            Err(TextContinuationError::IncompatibleDriver)
        ));
        owner.take_completed_step(&mut state).unwrap();
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
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
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
        driver.take_completed_step(&mut state).unwrap();
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
        let mut runtime = ModelRuntime::<Mock, MockSession>::install::<MockSession>(
            Mock,
            PreparedModel::new(10, SessionCapabilities::default()),
        )
        .unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start(vec![1, 2], continuation_config(3), PanickingFilter)
            .unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            driver.advance(&mut state)
        }))
        .is_err());
        driver.take_completed_step(&mut state).unwrap();
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

    impl DistributedModelSession<Mock> for MockSession {
        type DistributedSession = MockDistributed;

        fn distributed_session(&self) -> Option<&Self::DistributedSession> {
            self.distributed.as_ref()
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
            prefill_lengths: Vec::new(),
            cancel_on_prefill: None,
            prepare_ingress: false,
        };
        assert_eq!(
            DistributedModelSession::<Mock>::distributed_session(&model_session)
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
