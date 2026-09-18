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
    /// Owned exact completion for one submission. It may outlive the borrowed
    /// executor, inputs and policy call; those lifetimes cannot hide resources.
    type Completion: Completion + 'static;

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

    /// Validates that one neutral recipe can be represented by this backend.
    ///
    /// This is a metadata-only selection gate. Implementations must not acquire
    /// payload leases, allocate native tensors, or submit backend work.
    fn preflight_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<(), Self::ParameterError>;

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
    /// This operation is infallible so orchestration can validate an entire
    /// atomic unit before publishing any destination.
    fn bind(parameter: &mut Self::Parameter, weight: Self::MaterializedWeight);
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

    /// Borrows control funding attached to the exact current execution context.
    /// None selects ordinary execution. A backend with an original context must
    /// report invalid/missing request ownership as an error instead of None.
    /// The callback permits a weak source to be held for this complete loan.
    fn with_parallel_control_context<T, E, F>(
        context: &Self::ParallelContext, run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where F: FnOnce(Option<(&Self::ParallelContext,
        &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        let _ = context;
        Ok(run(None))
    }

    /// Lends a source-bound group for one actual shared lifecycle event.
    /// The prepared context and funding belong to the enclosing request, which
    /// may precede a numerical forward. None explicitly means unavailable;
    /// callers with a prepared context must refuse rather than submit ordinary
    /// communication. A provided group and its resources outlive native completion.
    fn with_prepared_control_group<T, E, F>(
        event: crate::replicated_session::ParallelControlEvent,
        group: &Self::CommunicationGroup, prepared: &Self::ParallelContext,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        executor: &Self::Executor, run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where F: FnOnce(Option<&Self::CommunicationGroup>) -> Result<T, E>,
    {
        let _ = (event, group, prepared, funding, executor);
        Ok(run(None))
    }

    /// Retains the selected route declaration and paid destination account from
    /// an explicit execution/control context. This descriptive loan grants no
    /// native transport authority. None preserves the ordinary default; original
    /// contexts must return a retained failure when their source is unavailable.
    fn prepare_boundary_source(context:&Self::ParallelContext,route:&Self::CommunicationRoute)
        ->Result<Option<crate::PreparedBoundarySource>,eredu_core::BackendFailure>{
        let _=(context,route);Ok(None)
    }

    /// Authenticates an exact quoted publication group against the selected
    /// stored resource. None keeps missing prepared support explicit.
    fn with_prepared_publication_group<T,E,F>(group:&Self::CommunicationGroup,
        prepared:&Self::ParallelContext,funding:&eredu_nn::workspace::HostMetadataFunding,
        executor:&Self::Executor,run:F)->Result<Result<T,E>,Self::CommunicationError>
    where F:FnOnce(Option<&Self::CommunicationGroup>)->Result<T,E> {
        let _=(group,prepared,funding,executor);Ok(run(None))
    }

    /// Completes exactly the selected boundary's model dependencies through
    /// their existing original scope. None is a missing producer, never a
    /// request to evaluate the same values through the ordinary constructor.
    fn submit_prepared_boundary_dependencies(
        values:&[crate::ArchitectureBoundaryValue<Self::Tensor>],
        source:&crate::PreparedBoundarySource,route:&Self::CommunicationRoute,
        context:&Self::ParallelContext,executor:&Self::Executor,
    )->Result<Option<Submission<(),Self::CommunicationCompletion>>,Self::CommunicationError>{
        let _=(values,source,route,context,executor);Ok(None)
    }

    /// Submits evaluation of rank-local tensor dependencies before a
    /// communication-readiness agreement.
    ///
    /// The returned communication completion must retain every submitted
    /// tensor and native execution resource through exact completion or safe
    /// cancellation teardown. This operation does not select or infer a
    /// collective group.
    /// Complete exact local roots through a retained model context when that
    /// context owns their submission. Some means successful native completion;
    /// None leaves the unchanged ordinary submission path to the caller.
    /// A present but invalid retained context must return an error.
    fn complete_model_dependencies(
        values: &[&Self::Tensor], context: &Self::ParallelContext,
        executor: &Self::Executor,
    ) -> Result<Option<()>, Self::CommunicationError> {
        let _ = (values, context, executor);
        Ok(None)
    }

    /// Executes one architecture-declared dynamic expert region. Cold backends
    /// retain the finite source and symbolic output; native backends bind the
    /// actual completed IDs/counts and lend a separately admitted child graph.
    /// The default preserves the ordinary route body and selects no new work.
    fn with_expert_route_region<P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P, input: &Self::Tensor,
        routes: &eredu_nn::GroupSelection<Self::Tensor>,
        context: Option<&Self::ParallelContext>, executor: &Self::Executor, run: F,
    ) -> Result<Result<crate::RoutedExpertTensorParallelOutput<Self::Tensor>, E>, Self::CommunicationError>
    where P: eredu_nn::Parameterized<Self::Tensor>,
        F: FnOnce(&mut P, Option<crate::PreparedExpertMovementLoan<'_>>) -> Result<crate::RoutedExpertTensorParallelOutput<Self::Tensor>, E>,
    {
        let _ = (source, input, routes, context, executor);
        Ok(run(bank, None))
    }

    /// Carries the actual routed observer through the same region boundary.
    /// A cold realization records its prospective source; a native realization
    /// lends this observer to the ordinary route body and its real callbacks.
    /// Describing a source never supplies received rows or completed evidence.
    fn with_observed_expert_route_region<'observer, P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P, input: &Self::Tensor,
        routes: &eredu_nn::GroupSelection<Self::Tensor>,
        context: Option<&Self::ParallelContext>, executor: &Self::Executor,
        observer: Option<&'observer mut dyn crate::RoutedUnitObserver<Self::Tensor>>,
        run: F,
    ) -> Result<Result<crate::RoutedExpertTensorParallelOutput<Self::Tensor>, E>, Self::CommunicationError>
    where P: eredu_nn::Parameterized<Self::Tensor>,
        F: FnOnce(&mut P, Option<crate::PreparedExpertMovementLoan<'_>>,
            Option<&'observer mut dyn crate::RoutedUnitObserver<Self::Tensor>>)
            -> Result<crate::RoutedExpertTensorParallelOutput<Self::Tensor>, E>,
    {
        Self::with_expert_route_region(source, bank, input, routes, context, executor,
            observed_expert_region_body(run, observer))
    }

    /// Lends one retained control-only inactive provider occurrence. Cold
    /// backends record the declared votes; ordinary backends run this same body.
    fn with_expert_provider_wave<E,F>(source:eredu_nn::workspace::WorkspaceExpertProviderWave,
        context:Option<&Self::ParallelContext>,executor:&Self::Executor,run:F)
        ->Result<Result<(),E>,Self::CommunicationError>
    where F:FnOnce()->Result<(),E>{
        let _=(source,context,executor);Ok(run())
    }

    /// Runs the existing zero-work count/dispatch/return itinerary through one
    /// exact retained wave occurrence. Cold recording grants no actual counts.
    fn with_expert_inactive_wave<E,F>(source:Option<eredu_nn::workspace::WorkspaceExpertInactiveWave>,
        context:Option<&Self::ParallelContext>,executor:&Self::Executor,run:F)
        ->Result<Result<(),E>,Self::CommunicationError>
    where F:FnOnce()->Result<(),E>{let _=(source,context,executor);Ok(run())}

    /// Copies an actual validated integer row source through the selected
    /// original input producer. None preserves ordinary construction. A backend
    /// with an original context must retain the source or return an error.
    fn prepare_expert_route_input(values: &[i32], source: &eredu_core::ErasedSharedStorageOwner,
        context: &Self::ParallelContext, executor: &Self::Executor) -> Result<Option<Self::Tensor>, Self::CommunicationError> {
        let _ = (values, source, context, executor); Ok(None)
    }

    /// The selected local grouped worker runs after completed count/ID binding.
    /// The same typed output returns to the ordinary forward/reverse driver.
    fn with_expert_route_local<P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>, bank: &mut P,
        input: &Self::Tensor, scores: &Self::Tensor, coefficients: &Self::Tensor,
        completed: &eredu_core::ErasedSharedStorageOwner,
        local_expert_rows: &[usize], context: &Self::ParallelContext,
        executor: &Self::Executor, run: F,
    ) -> Result<Result<crate::RoutedExpertTensorParallelOutput<Self::Tensor>, E>, Self::CommunicationError>
    where P: eredu_nn::Parameterized<Self::Tensor>,
        F: FnOnce(&mut P) -> Result<crate::RoutedExpertTensorParallelOutput<Self::Tensor>, E>,
    {
        let _ = (source, input, scores, coefficients, completed, local_expert_rows, context, executor);
        Ok(run(bank))
    }

    /// Completes and lends the ordinary I32 readout of a selected route source under the exact
    /// current model invocation. The callback cannot retain the borrowed slice;
    /// any derived host destination retains the supplied cumulative account.
    /// None selects ordinary readout. An invalid original context must fail.
    fn with_prepared_expert_route_indices<T, E, F>(
        value: &Self::Tensor, context: &Self::ParallelContext,
        executor: &Self::Executor, run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where F: for<'loan> FnOnce(Option<(&'loan [i32],
        &'loan eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        let _ = (value, context, executor);
        Ok(run(None))
    }

    /// Source-bound integer peer-row consensus. The completed rank-major
    /// matrix is borrowed only during the callback; derived destinations retain
    /// the supplied account. None preserves the ordinary collective driver.
    fn with_prepared_peer_count_consensus<T, E, F>(
        local: &[i32], group: &Self::CommunicationGroup,
        context: &Self::ParallelContext, executor: &Self::Executor, run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where F: for<'loan> FnOnce(Option<(&'loan [i32],
        &'loan eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        let _ = (local, group, context, executor);
        Ok(run(None))
    }

    /// Companion carrying a closed completed source through the same consensus.
    /// Existing backends keep their established consensus and provide no owner.
    fn with_prepared_peer_count_source<T, E, F>(
        local: &[i32], group: &Self::CommunicationGroup,
        context: &Self::ParallelContext, executor: &Self::Executor, run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where F: for<'loan> FnOnce(Option<crate::PreparedPeerCountLoan<'loan>>) -> Result<T, E> {
        Self::with_prepared_peer_count_consensus(local, group, context, executor, |loan| {
            run(loan.map(|(matrix, funding)| crate::PreparedPeerCountLoan::new(matrix, funding, None)))
        })
    }

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
    /// Constructs every exact retained occurrence before completing the wave.
    /// This preserves lazy cross-rank ordering: completing one member before
    /// constructing its peers can deadlock an active rank's combined graph.
    /// The validator borrows the actual source payer before any constructor,
    /// then validates the complete output list after completion. Native failures
    /// already own their typed source and custody. None means no retained source
    /// was selected; an invalid source must fail without calling ordinary work.
    fn complete_model_sum_wave<E, V>(
        values: &[Self::Tensor], group: &Self::CommunicationGroup,
        context: &Self::ParallelContext, executor: &Self::Executor,
        validate: V,
    ) -> Result<Option<Vec<Self::Tensor>>, eredu_core::BackendFailure>
    where E: std::error::Error + Send + Sync + 'static, V: FnMut(&[Self::Tensor], &eredu_nn::workspace::HostMetadataFunding, bool) -> Result<(), E>,
    {
        let _ = (values, group, context, executor, validate);
        Ok(None)
    }
    /// Execute and complete a sum using the exact retained model occurrence.
    /// The borrowed input permits ordinary fallback without copying when None.
    /// Some must retain authority until successful completion; an invalid
    /// retained context is an error and never ordinary fallback.
    fn complete_model_sum(
        value: &Self::Tensor, group: &Self::CommunicationGroup,
        context: &Self::ParallelContext, executor: &Self::Executor,
    ) -> Result<Option<Self::Tensor>, Self::CommunicationError> {
        let _ = (value, group, context, executor);
        Ok(None)
    }
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
    /// Complete a gather through its retained model occurrence, with the same
    /// completion and ordinary-fallback contract as complete_model_sum.
    fn complete_model_gather(
        value: &Self::Tensor, counts: &[usize], axis: usize,
        group: &Self::CommunicationGroup, context: &Self::ParallelContext,
        executor: &Self::Executor,
    ) -> Result<Option<Self::Tensor>, Self::CommunicationError> {
        let _ = (value, counts, axis, group, context, executor);
        Ok(None)
    }
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
    /// Completes one variable exchange from its exact completed count matrix.
    /// The retained context authenticates native source/stream/owner identity;
    /// matrix geometry alone grants no submission authority. The implementation
    /// retains accepted inputs, matrix storage and resources until completion or
    /// safe failure. None means missing source and must never select ordinary
    /// execution when called with an original context.
    fn complete_prepared_variable_all_to_all(
        value: &Self::Tensor, counts: &CommunicationPeerCounts, axis: usize,
        matrix: &crate::CommunicationPeerMatrix<'_>, group: &Self::CommunicationGroup,
        context: &Self::ParallelContext, executor: &Self::Executor,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Option<Self::Tensor>, Self::CommunicationError> {
        let _=(value,counts,axis,matrix,group,context,executor,funding); Ok(None)
    }
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

    /// Same selected transfer with canonical headers and their exact paid
    /// source. None is a typed missing producer, never an ordinary fallback.
    fn send_receive_prepared(values:crate::PreparedBoundaryFrames<Self::Tensor>,
        route:&Self::CommunicationRoute,context:&Self::ParallelContext,executor:&Self::Executor)
        ->Result<Option<Submission<Vec<Self::Tensor>,Self::CommunicationCompletion>>,Self::CommunicationError>{
        let _=(values,route,context,executor);Ok(None)
    }

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

    /// Optional exact source-owned vote. A returned boolean is already terminal
    /// under the selected communication authority; this creates no detached
    /// completion or substitute group. Defaults preserve ordinary submission.
    fn agree_success_from_source(
        _local_success: bool, _group: &Self::CommunicationGroup,
        _phase: crate::DistributedExecutionPhase, _executor: &Self::Executor,
        _source: &Self::ParallelContext,
    ) -> Result<Option<bool>, Self::CommunicationError> { Ok(None) }

    /// Resolves the completed backend result without starting new native work.
    fn resolve_failure_agreement(
        output: Self::FailureAgreementOutput,
    ) -> Result<bool, Self::CommunicationError>;
}

/// Canonical terminal rejection for already-realized communication resources.
/// Marking must be infallible, allocation-free and nonblocking, and submit,
/// evaluate, poll or retire no work. All aliases of that native incarnation,
/// including newly wrapped cached handles, must reject later submissions.
/// Existing accepted work retains its owners: this proves no completion.
pub trait TerminalCommunicationBackend: CommunicationBackend {
    /// Irreversibly prevents new submissions on the retained incarnation.
    fn mark_terminal_submission(group: &Self::CommunicationGroup);
}

/// The same forwarding closure is constructed by native dispatch and inspected
/// by its cold source. Its captured observer option changes no storage layout.
pub(crate) fn observed_expert_region_body<'observer, T: eredu_nn::Tensor, P, E, F>(
    run: F, observer: Option<&'observer mut dyn crate::RoutedUnitObserver<T>>,
) -> impl FnOnce(&mut P, Option<crate::PreparedExpertMovementLoan<'_>>)
    -> Result<crate::RoutedExpertTensorParallelOutput<T>, E>
    + use<'observer, T, P, E, F>
where F: FnOnce(&mut P, Option<crate::PreparedExpertMovementLoan<'_>>,
    Option<&'observer mut dyn crate::RoutedUnitObserver<T>>)
    -> Result<crate::RoutedExpertTensorParallelOutput<T>, E>,
{
    move |bank, movement| run(bank, movement, observer)
}
