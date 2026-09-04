//! Narrow capability contracts implemented by execution backends.

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointLease, CheckpointSource},
};
use eredu_core::{BoundedCompletion, Completion, Submission};
use eredu_nn::NeuralBackend;

use crate::CommunicationPeerCounts;

/// Submits backend-native work and retains values through exact completion.
pub trait SubmissionBackend: NeuralBackend {
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

/// Materializes and binds checkpoint data to backend-native parameter slots.
pub trait ParameterBackend: NeuralBackend {
    /// One backend-native parameter slot.
    type Parameter: 'static;
    /// Materialized backend-native checkpoint weight.
    type MaterializedWeight;
    /// Backend context used only while realizing checkpoint parameters.
    type MaterializationContext: ?Sized;
    /// In-flight guard retaining encoded sources through exact realization completion.
    type Materialization;
    /// Backend-specific loading failure.
    type ParameterError: std::error::Error + Send + Sync + 'static;

    /// Lowers one format-preserving encoded lease into a native weight.
    fn materialize(
        lease: CheckpointLease,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError>;

    /// Lowers a validated neutral recipe directly into a native weight.
    fn materialize_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError>;

    /// Borrows the native weight retained by an in-flight materialization.
    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight;

    /// Waits for this exact realization and releases its encoded source lease.
    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError>;

    /// Creates another native handle to identical materialized storage without
    /// rereading or rematerializing checkpoint data.
    fn share_materialized_weight(
        weight: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError>;

    /// Validates destination shape/storage compatibility without publication.
    fn validate_bind(
        parameter: &Self::Parameter,
        weight: &Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError>;

    /// Binds one materialized weight to its destination parameter.
    ///
    /// After successful [`Self::validate_bind`] on unchanged arguments this
    /// operation must not fail, allowing orchestration to validate an entire
    /// atomic unit before publishing any destination.
    fn bind(
        parameter: &mut Self::Parameter,
        weight: Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError>;
}

/// Promotes and demotes backend-native storage without changing its semantics.
pub trait TransferBackend: SubmissionBackend + ParameterBackend {
    /// Backend-owned host representation.
    type HostBuffer;
    /// In-flight transfer guard retaining all source and destination storage.
    type Transfer: Completion<Error = Self::TransferError>;
    /// Backend-specific transfer failure.
    type TransferError: std::error::Error + Send + Sync + 'static;

    /// Promotes host storage into a materialized execution weight.
    fn promote(
        executor: &Self::Executor,
        host: &Self::HostBuffer,
    ) -> Result<(Self::MaterializedWeight, Self::Transfer), Self::TransferError>;

    /// Demotes a materialized execution weight into backend-owned host storage.
    fn demote(
        executor: &Self::Executor,
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Self::Transfer), Self::TransferError>;
}

/// Collective operations available to distributed runtime policies.
pub trait CollectiveBackend: SubmissionBackend {
    /// Backend-native collective group.
    type Group: ?Sized;
    /// Backend-specific collective failure.
    type CollectiveError: std::error::Error + Send + Sync + 'static;

    /// Reduces a tensor across the selected group.
    fn all_reduce(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;

    /// Gathers a tensor across the selected group.
    fn all_gather(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;

    /// Exchanges tensor partitions across the selected group.
    fn all_to_all(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;
}

/// Common opaque handles and exact completion used by communication extensions.
///
/// This trait deliberately declares no operation. Backends implement only the
/// fine-grained operation traits selected for a concrete architecture.
pub trait CommunicationBackend: SubmissionBackend {
    /// Backend-native realization of one opaque communication group.
    type CommunicationGroup: ?Sized;
    /// Backend-native realization of one opaque directed route.
    type CommunicationRoute: ?Sized;
    /// Exact completion retaining tensors, buffers, groups, routes, and streams.
    type CommunicationCompletion: Completion<Error = Self::CommunicationError>
        + BoundedCompletion<Error = Self::CommunicationError>;
    /// Stable mechanism failure with no architecture-family policy.
    type CommunicationError: std::error::Error + Send + Sync + 'static;

    /// Submits evaluation of rank-local tensor dependencies before a
    /// communication-readiness agreement.
    ///
    /// The returned communication completion must retain every submitted
    /// tensor and native execution resource through exact completion or safe
    /// cancellation teardown. This operation does not select or infer a
    /// collective group.
    fn submit_local_dependencies<'a, I>(
        values: I,
        executor: &Self::Executor,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>;
}

/// Sum reduction on an opaque communication group.
pub trait SumReductionBackend: CommunicationBackend {
    /// Submits one elementwise sum and returns its exact completion.
    fn all_reduce_sum(
        value: Self::Tensor,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Equal-size gathering on an opaque communication group.
pub trait EvenGatherBackend: CommunicationBackend {
    /// Gathers equal-sized values and concatenates in member order on `axis`.
    fn all_gather_even(
        value: Self::Tensor,
        axis: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Unequal-size gathering on an opaque communication group.
pub trait UnevenGatherBackend: CommunicationBackend {
    /// Gathers values and concatenates in member order using exact element counts.
    fn all_gather_uneven(
        value: Self::Tensor,
        counts: &[usize],
        axis: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Variable-count exchange on an opaque communication group.
pub trait VariableAllToAllBackend: CommunicationBackend {
    /// Exchanges exact per-peer partitions on `axis` and returns exact completion.
    fn variable_all_to_all(
        value: Self::Tensor,
        counts: &CommunicationPeerCounts,
        axis: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Ordered point-to-point boundary transfer on one opaque route.
pub trait PointToPointBackend: CommunicationBackend {
    /// Sends or receives the route's exact ordered tensor bundle.
    #[allow(
        clippy::type_complexity,
        reason = "the signature exposes the tensor bundle and exact completion without erasure"
    )]
    fn send_receive(
        values: Vec<RoleExactBoundaryValue<Self::Tensor>>,
        route: &Self::CommunicationRoute,
        executor: &Self::Executor,
    ) -> Result<
        Submission<Vec<Self::Tensor>, Self::CommunicationCompletion>,
        Self::CommunicationError,
    >;
}

/// One logical boundary tensor coupled to the exact in-band header that must
/// be transmitted with its payload.
///
/// A point-to-point implementation must place `header` and the byte
/// representation of `tensor` in the same native message. On receive it must
/// compare the bytes actually received with `header` before its completion can
/// report success. Returning a backend-synthesized tag does not satisfy this
/// contract.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoleExactBoundaryValue<T> {
    header: Vec<u8>,
    tensor: T,
}

impl<T> RoleExactBoundaryValue<T> {
    pub(crate) fn new(header: Vec<u8>, tensor: T) -> Self {
        Self { header, tensor }
    }

    /// Exact expected in-band header bytes.
    pub fn header(&self) -> &[u8] {
        &self.header
    }

    /// Logical tensor payload.
    pub const fn tensor(&self) -> &T {
        &self.tensor
    }

    /// Consumes the framed value into its expected header and payload.
    pub fn into_parts(self) -> (Vec<u8>, T) {
        (self.header, self.tensor)
    }
}

/// Root-to-group publication on an opaque communication group.
pub trait BroadcastBackend: CommunicationBackend {
    /// Broadcasts one tensor from an ordered member index.
    fn broadcast(
        value: Self::Tensor,
        root: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Payload-free agreement on an opaque communication group.
pub trait BarrierBackend: CommunicationBackend {
    /// Submits a barrier and returns its exact completion.
    fn barrier(
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Self::CommunicationCompletion, Self::CommunicationError>;
}

/// All-rank success-status agreement on an opaque communication group.
///
/// Unlike a barrier, this operation carries one boolean status from every
/// member and returns `true` only when every submitted status was `true`.
pub trait FailureAgreementBackend: CommunicationBackend {
    /// Backend-owned result whose host boolean becomes authoritative only after
    /// exact communication completion.
    type FailureAgreementOutput;

    /// Submits one local phase status without reading the lazy result eagerly.
    fn agree_success(
        local_success: bool,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<
        Submission<Self::FailureAgreementOutput, Self::CommunicationCompletion>,
        Self::CommunicationError,
    >;

    /// Resolves the completed backend result without starting new native work.
    fn resolve_failure_agreement(
        output: Self::FailureAgreementOutput,
    ) -> Result<bool, Self::CommunicationError>;
}
