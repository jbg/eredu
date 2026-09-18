//! Backend-neutral rank-local execution over opaque communication resources.
//!
//! Architecture adapters remain statically dispatched and own tensor equations,
//! routed providers, and typed boundary encoding. This module owns the common
//! partition schedule, mechanical wire validation, exact completion, publication,
//! and the execution strategy installed in [`crate::ReplicatedTextSession`].

mod media_prefill;
mod publication;
mod peer_exchange;
pub use media_prefill::PartitionedMediaGroupExecutor;

use std::{borrow::Borrow, marker::PhantomData};

use eredu_core::{
    checkpoint::TensorDtype, BoundedCompletion, BoundedCompletionOutcome, BoundedSubmissionOutcome,
    CollectiveGroupId, CompletionCancellationMode, DistributedCommitEpoch,
    DistributedCommitOutcome, DistributedCommitPhase,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    ActivationObserver, ArchitectureGroupKind, BarrierBackend, BroadcastBackend,
    CommunicationBackend, CommunicationGroupDescriptor, CommunicationManifest,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationPeerCounts,
    CommunicationRouteDescriptor, CommunicationRouteId, EvenGatherBackend, ExecutionGraph,
    ExecutionResidency, ExpertPass, FailureAgreementBackend, LayeredArchitecture,
    LayeredPartitionDriver, LayeredPipelineSchedule, LayeredPipelineScheduleError,
    LayeredTraversalHook, LayerwisePolicy, LayerwiseRuntime, LayerwiseRuntimeError,
    ParallelLayeredArchitecture, PipelineActivationDtype, PipelineWireContract,
    PointToPointBackend, ReplicatedTextExecutionStrategy, ReplicatedTextSessionError,
    ResolvedBoundaryTensorSpec, ResolvedBoundaryWireSchema, RuntimeState, SubmissionBackend,
    SumReductionBackend, UnevenGatherBackend, VariableAllToAllBackend,
};

/// Backend-native tensor metadata used only for mechanical communication validation.
pub trait CommunicationTensorMetadata<B: NeuralBackend> {
    /// Returns the portable logical dtype of one native tensor.
    fn dtype(&self, tensor: &B::Tensor) -> TensorDtype;

    /// Returns the exact logical shape of one native tensor.
    fn shape(&self, tensor: &B::Tensor) -> Vec<usize>;

    /// Compare the actual dimensions with a resolved boundary without creating
    /// an owning shape vector. None leaves this original producer unqualified.
    fn matches_shape_with_funding(&self,tensor:&B::Tensor,shape:&[i32],
        funding:&eredu_nn::workspace::HostMetadataFunding)
        ->Result<Option<bool>,eredu_nn::workspace::HostMetadataFundingError>{
        let _=(tensor,shape,funding);Ok(None)
    }

    /// Paid, allocation-free metadata for a prepared communication. A backend
    /// returns None when it has no such producer; the caller must then refuse.
    /// The dtype must be a built-in scalar, with no allocated encoded label.
    fn fixed_metadata_with_funding(&self,tensor:&B::Tensor,
        funding:&eredu_nn::workspace::HostMetadataFunding)
        ->Result<Option<(TensorDtype,usize,Option<usize>)>,eredu_nn::workspace::HostMetadataFundingError> {
        let _=(tensor,funding);Ok(None)
    }
}

/// One backend-native group paired with the opaque identity selected by the manifest.
pub struct RealizedCommunicationGroup<G> {
    id: CollectiveGroupId,
    resource: G,
}

impl<G> RealizedCommunicationGroup<G> {
    /// Binds one native resource to its selected opaque group identity.
    pub const fn new(id: CollectiveGroupId, resource: G) -> Self {
        Self { id, resource }
    }
}

/// One backend-native route paired with the opaque identity selected by the manifest.
pub struct RealizedCommunicationRoute<R> {
    id: CommunicationRouteId,
    resource: R,
}

impl<R> RealizedCommunicationRoute<R> {
    /// Binds one native resource to its selected opaque route identity.
    pub const fn new(id: CommunicationRouteId, resource: R) -> Self {
        Self { id, resource }
    }
}

/// Opaque native communication resources paired in manifest order.
pub struct PartitionCommunication<B, G, R, I>
where
    B: CommunicationBackend,
{
    manifest: CommunicationManifest,
    groups: Vec<RealizedCommunicationGroup<G>>,
    routes: Vec<RealizedCommunicationRoute<R>>,
    inspector: I,
    authority: PartitionCommunicationAuthority,
    backend: PhantomData<fn() -> B>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CommunicationPoison {
    operation: CommunicationOperation,
    phase: DistributedExecutionPhase,
    route: Option<CommunicationRouteId>,
    cancellation: CompletionCancellationMode,
}

/// Cloneable bounded-completion and poison authority for one selected communication session.
///
/// Backend-owned operations adjacent to the neutral partition driver must
/// retain this authority instead of a bare native group. Every clone observes
/// the first terminal communication failure and fails before later submission.
#[derive(Debug, Clone)]
pub struct PartitionCommunicationAuthority {
    policy: Option<crate::CommunicationCompletionPolicy>,
    shared: std::sync::Arc<CommunicationAuthorityState>,
}

#[derive(Debug)]
struct CommunicationAuthorityState {
    terminal: std::sync::atomic::AtomicBool,
    poison: std::sync::Mutex<Option<CommunicationPoison>>,
}

impl PartitionCommunicationAuthority {
    fn new(policy: Option<crate::CommunicationCompletionPolicy>) -> Self {
        Self {
            policy,
            shared: std::sync::Arc::new(CommunicationAuthorityState {
                terminal: std::sync::atomic::AtomicBool::new(false),
                poison: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Whether two loans refer to the same retained communication incarnation.
    pub fn same_authority(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// Creates the poison and bounded-completion authority selected by one manifest.
    pub fn from_manifest(
        manifest: &CommunicationManifest,
    ) -> Result<Self, PartitionExecutionError> {
        if manifest.completion_policy().is_none()
            && (!manifest.groups().is_empty() || !manifest.routes().is_empty())
        {
            return Err(PartitionExecutionError::MissingBoundedCompletionPolicy);
        }
        Ok(Self::new(manifest.completion_policy()))
    }

    /// Exact retained shared representation, including its Arc control header.
    /// Handle values and construction moves are separate. This is a diagnostic,
    /// not a reservation or a later allocation grant; the original communicator
    /// constructor must include it in its own physical host budget.
    pub const fn shared_control_bytes() -> usize {
        std::mem::size_of::<CommunicationAuthorityState>() + 2 * std::mem::size_of::<usize>()
    }

    fn mark_terminal(&self) {
        self.shared
            .terminal
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn poison_guard(&self) -> std::sync::MutexGuard<'_, Option<CommunicationPoison>> {
        self.shared
            .poison
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Rejects work after any submission, completion, cancellation, or deadline failure.
    pub fn ensure_active(&self) -> Result<(), PartitionExecutionError> {
        if self
            .shared
            .terminal
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(PartitionExecutionError::CommunicationTerminal);
        }
        match *self.poison_guard() {
            Some(poison) => Err(PartitionExecutionError::CommunicationPoisoned {
                operation: poison.operation,
                phase: poison.phase,
                route: poison.route,
                cancellation: poison.cancellation,
            }),
            None => Ok(()),
        }
    }

    /// Returns the bounded completion policy selected by the communication manifest.
    pub const fn completion_policy(&self) -> Option<crate::CommunicationCompletionPolicy> {
        self.policy
    }

    fn mark_poisoned(&self, poison: CommunicationPoison) {
        let mut current = self.poison_guard();
        if current.is_none() {
            *current = Some(poison);
        }
    }

    fn is_poisoned(&self) -> bool {
        self.shared
            .terminal
            .load(std::sync::atomic::Ordering::Acquire)
            || self.poison_guard().is_some()
    }

    /// Records a diagnostic-only submission rejection and poisons the authority.
    /// Use [`Self::submission_failure`] when an owned error is available.
    pub fn submission_error(
        &self,
        error: impl std::fmt::Display,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> PartitionExecutionError {
        let cancellation = self
            .policy
            .expect("communication submission requires a selected completion policy")
            .cancellation();
        self.mark_poisoned(CommunicationPoison {
            operation,
            phase,
            route,
            cancellation,
        });
        PartitionExecutionError::CommunicationSubmissionFailed {
            operation,
            phase,
            route,
            error: error.to_string(),
            source: None,
        }
    }

    /// Records a diagnostic-only completion rejection in the shared poison domain.
    /// Use [`Self::completion_failure`] when an owned error is available.
    pub fn completion_error(
        &self,
        error: impl std::fmt::Display,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> PartitionExecutionError {
        let cancellation = self
            .policy
            .expect("communication completion requires a selected completion policy")
            .cancellation();
        self.mark_poisoned(CommunicationPoison {
            operation,
            phase,
            route,
            cancellation,
        });
        PartitionExecutionError::CommunicationCompletionFailed {
            operation,
            phase,
            route,
            error: error.to_string(),
            source: None,
        }
    }

    /// Records a submission failure while retaining its original typed cause.
    pub fn submission_failure(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> PartitionExecutionError {
        let mut failure = self.submission_error(&error, operation, phase, route);
        if let PartitionExecutionError::CommunicationSubmissionFailed { source, .. } = &mut failure
        {
            *source = Some(eredu_core::BackendFailure::from_error(error));
        }
        failure
    }

    /// Records a completion failure without treating the error as release evidence.
    pub fn completion_failure(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> PartitionExecutionError {
        let mut failure = self.completion_error(&error, operation, phase, route);
        if let PartitionExecutionError::CommunicationCompletionFailed { source, .. } = &mut failure
        {
            *source = Some(eredu_core::BackendFailure::from_error(error));
        }
        failure
    }

    /// Waits under the selected bound and permanently poisons this session on failure.
    pub fn wait<T, C>(
        &self,
        submission: eredu_core::Submission<T, C>,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> Result<T, PartitionExecutionError>
    where
        C: BoundedCompletion,
        C::Error: std::fmt::Display,
    {
        self.wait_with_error(submission,operation,phase,route,|error| {
            PartitionExecutionError::CommunicationCompletionFailed {
                operation,phase,route,error:error.to_string(),
                source:Some(eredu_core::BackendFailure::from_error(error)),
            }
        })
    }

    /// Waits through the same selected completion/poison worker with a funded
    /// failure producer. The callback replaces diagnostics, never completion.
    pub fn wait_with_error<T,C,F>(&self,submission:eredu_core::Submission<T,C>,
        operation:CommunicationOperation,phase:DistributedExecutionPhase,
        route:Option<CommunicationRouteId>,failure:F)->Result<T,PartitionExecutionError>
    where C:BoundedCompletion,F:FnOnce(C::Error)->PartitionExecutionError {
        self.ensure_active()?;
        let policy = self
            .policy
            .ok_or(PartitionExecutionError::MissingBoundedCompletionPolicy)?
            .bounded_wait();
        let outcome = match submission.wait_bounded(policy) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.mark_poisoned(CommunicationPoison {
                    operation,
                    phase,
                    route,
                    cancellation: policy.cancellation(),
                });
                return Err(failure(error));
            }
        };
        match outcome {
            BoundedSubmissionOutcome::Completed(output) => Ok(output),
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                self.mark_poisoned(CommunicationPoison {
                    operation,
                    phase,
                    route,
                    cancellation,
                });
                Err(PartitionExecutionError::CommunicationDeadlineExceeded {
                    operation,
                    phase,
                    route,
                    cancellation,
                })
            }
        }
    }

    fn wait_after_prior_failure<T, C>(
        &self,
        submission: eredu_core::Submission<T, C>,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> Result<T, PartitionExecutionError>
    where
        C: BoundedCompletion,
        C::Error: std::fmt::Display,
    {
        let policy = self
            .policy
            .ok_or(PartitionExecutionError::MissingBoundedCompletionPolicy)?
            .bounded_wait();
        let outcome = submission.wait_bounded(policy).map_err(|error| {
            self.mark_poisoned(CommunicationPoison {
                operation,
                phase,
                route,
                cancellation: policy.cancellation(),
            });
            PartitionExecutionError::CommunicationCompletionFailed {
                operation,
                phase,
                route,
                error: error.to_string(),
                source: Some(eredu_core::BackendFailure::from_error(error)),
            }
        })?;
        match outcome {
            BoundedSubmissionOutcome::Completed(output) => Ok(output),
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                self.mark_poisoned(CommunicationPoison {
                    operation,
                    phase,
                    route,
                    cancellation,
                });
                Err(PartitionExecutionError::CommunicationDeadlineExceeded {
                    operation,
                    phase,
                    route,
                    cancellation,
                })
            }
        }
    }

    fn wait_before_failure_agreement<T, C>(
        &self,
        submission: eredu_core::Submission<T, C>,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> Result<T, PartitionExecutionError>
    where
        C: BoundedCompletion,
        C::Error: std::fmt::Display,
    {
        self.wait_before_failure_agreement_with_error(submission,operation,phase,route,|error|
            PartitionExecutionError::CommunicationCompletionFailed {
                operation,phase,route,error:error.to_string(),
                source:Some(eredu_core::BackendFailure::from_error(error)),
            })
    }

    fn wait_before_failure_agreement_with_error<T,C,F>(&self,
        submission:eredu_core::Submission<T,C>,operation:CommunicationOperation,
        phase:DistributedExecutionPhase,route:Option<CommunicationRouteId>,failure:F,
    )->Result<T,PartitionExecutionError>
    where C:BoundedCompletion,F:FnOnce(C::Error)->PartitionExecutionError {
        self.ensure_active()?;
        let policy=self.policy.ok_or(PartitionExecutionError::MissingBoundedCompletionPolicy)?.bounded_wait();
        match submission.wait_bounded(policy).map_err(failure)? {
            BoundedSubmissionOutcome::Completed(output)=>Ok(output),
            BoundedSubmissionOutcome::DeadlineExceeded{cancellation}=>Err(
                PartitionExecutionError::CommunicationDeadlineExceeded{operation,phase,route,cancellation}),
        }
    }

    /// Poison the same selected completion domain without formatting a second
    /// diagnostic. The caller retains its original typed protocol/native cause.
    pub fn fence_protocol_failure(
        &self,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) {
        let cancellation = self
            .policy
            .expect("communication protocol failure requires a selected completion policy")
            .cancellation();
        self.mark_poisoned(CommunicationPoison {
            operation,
            phase,
            route,
            cancellation,
        });
    }
}

impl<B, G, R, I> PartitionCommunication<B, G, R, I>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Pairs already-created opaque resources with their authoritative manifest order.
    pub fn new(
        manifest: CommunicationManifest,
        groups: Vec<RealizedCommunicationGroup<G>>,
        routes: Vec<RealizedCommunicationRoute<R>>,
        inspector: I,
    ) -> Result<Self, PartitionExecutionError> {
        let authority = PartitionCommunicationAuthority::from_manifest(&manifest)?;
        Self::new_with_authority(manifest, groups, routes, inspector, authority)
    }

    /// Pairs native resources with an authority already retained by adjacent session APIs.
    pub fn new_with_authority(
        manifest: CommunicationManifest,
        groups: Vec<RealizedCommunicationGroup<G>>,
        routes: Vec<RealizedCommunicationRoute<R>>,
        inspector: I,
        authority: PartitionCommunicationAuthority,
    ) -> Result<Self, PartitionExecutionError> {
        if authority.policy != manifest.completion_policy() {
            return Err(PartitionExecutionError::MissingBoundedCompletionPolicy);
        }
        if groups.len() != manifest.groups().len() || routes.len() != manifest.routes().len() {
            return Err(PartitionExecutionError::ResourceCount {
                expected_groups: manifest.groups().len(),
                actual_groups: groups.len(),
                expected_routes: manifest.routes().len(),
                actual_routes: routes.len(),
            });
        }
        for (descriptor, resource) in manifest.groups().iter().zip(&groups) {
            if descriptor.id() != resource.id {
                return Err(PartitionExecutionError::ResourceIdentity {
                    expected: u64::from(descriptor.id().value()),
                    actual: u64::from(resource.id.value()),
                });
            }
        }
        for (descriptor, resource) in manifest.routes().iter().zip(&routes) {
            if descriptor.id() != resource.id {
                return Err(PartitionExecutionError::ResourceIdentity {
                    expected: descriptor.id().value(),
                    actual: resource.id.value(),
                });
            }
        }
        Ok(Self {
            manifest,
            groups,
            routes,
            inspector,
            authority,
            backend: PhantomData,
        })
    }

    /// Authoritative local-rank manifest.
    pub const fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }

    // Marking precedes every possible poison-mutex access and native retry.
    fn mark_terminal_failure(&self)
    where
        B: crate::TerminalCommunicationBackend,
    {
        self.authority.mark_terminal();
        for group in &self.groups {
            B::mark_terminal_submission(group.resource.borrow());
        }
    }

    /// Shares the selected deadline and terminal poison domain with adjacent operations.
    pub fn authority(&self) -> PartitionCommunicationAuthority {
        self.authority.clone()
    }

    fn ensure_active(&self) -> Result<(), PartitionExecutionError> {
        self.authority.ensure_active()
    }

    fn submission_error(
        &self,
        error: impl std::error::Error + Send + Sync + 'static,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> PartitionExecutionError {
        self.authority
            .submission_failure(error, operation, phase, route)
    }

    fn group(
        &self,
        id: CollectiveGroupId,
        operation: CommunicationOperation,
    ) -> Result<(&CommunicationGroupDescriptor, &B::CommunicationGroup), PartitionExecutionError>
    {
        self.ensure_active()?;
        let selected=self.manifest.select_group_operation(id,operation).map_err(|error| match error {
            crate::CommunicationGroupOperationError::Unknown(id)=>PartitionExecutionError::UnknownGroup(id),
            crate::CommunicationGroupOperationError::NotMember(id)=>PartitionExecutionError::NotGroupMember(id),
            crate::CommunicationGroupOperationError::NotSelected{group,operation}=>
                PartitionExecutionError::OperationNotSelected{resource:format!("group {}",group.value()),operation},
        })?;
        Ok((selected.descriptor(), self.groups[selected.order()].resource.borrow()))
    }


    fn with_control_group<T, F>(
        &self,
        group: CollectiveGroupId,
        operation: CommunicationOperation,
        event: crate::replicated_session::ParallelControlEvent,
        prepared: Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>,
        executor: &B::Executor,
        run: F,
    ) -> Result<T, PartitionExecutionError>
    where
        F: FnOnce(Option<&B::CommunicationGroup>) -> Result<T, PartitionExecutionError>,
    {
        let Some((context, funding)) = prepared else {
            return run(None);
        };
        let selected = self.manifest.select_group_operation(group, operation)
            .map_err(|error| match error {
                crate::CommunicationGroupOperationError::Unknown(id) =>
                    PartitionExecutionError::UnknownGroup(id),
                crate::CommunicationGroupOperationError::NotMember(id) =>
                    PartitionExecutionError::NotGroupMember(id),
                crate::CommunicationGroupOperationError::NotSelected { group, operation } =>
                    PartitionExecutionError::OperationNotSelected {
                        resource: format!("group {}", group.value()), operation,
                    },
            })?;
        let native = self.groups[selected.order()].resource.borrow();
        B::with_prepared_control_group(event, native, context, funding, executor, |bound| {
            let bound = bound.ok_or(PartitionExecutionError::CommunicationPolicyMismatch)?;
            run(Some(bound))
        }).map_err(|error| {
            let phase = match event {
                crate::replicated_session::ParallelControlEvent::Phase(phase) => phase,
                crate::replicated_session::ParallelControlEvent::Commit => DistributedExecutionPhase::Commit,
            };
            self.submission_error(error, operation, phase, None)
        })?
    }

    /// Runs one reached provider or pipeline vote with the explicitly lent
    /// request control context. The selected group and existing policy still
    /// own the operation; this method creates no collective schedule.
    pub fn agree_phase_with_parallel_context<V>(
        &self, policy: &mut V, group: CollectiveGroupId,
        phase: DistributedExecutionPhase, local_success: bool,
        context: &B::Executor, current: Option<&B::ParallelContext>,
    ) -> Result<bool, PartitionExecutionError>
    where V: PartitionCommitAgreement<B, G, R, I>,
    {
        if let Some(source)=current {
            if let Some(completed)=policy.agree_phase_from_source(self,group,phase,local_success,context,source)? {
                return Ok(completed);
            }
        }
        let mut run = |prepared: Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>| self.with_control_group(
            group, CommunicationOperation::FailureAgreement,
            crate::replicated_session::ParallelControlEvent::Phase(phase),
            prepared, context,
            |bound| policy.agree_phase_with_group(self, group, phase, local_success, context, bound),
        );
        match current {
            Some(current) => B::with_parallel_control_context(current, run)
                .map_err(|error| self.submission_error(
                    error, CommunicationOperation::FailureAgreement, phase, None))?,
            None => run(None),
        }
    }

    fn route(
        &self,
        id: CommunicationRouteId,
    ) -> Result<(&CommunicationRouteDescriptor, &B::CommunicationRoute), PartitionExecutionError>
    {
        self.ensure_active()?;
        let index = self
            .manifest
            .routes()
            .iter()
            .position(|candidate| candidate.id() == id)
            .ok_or(PartitionExecutionError::UnknownRoute(id))?;
        Ok((
            &self.manifest.routes()[index],
            self.routes[index].resource.borrow(),
        ))
    }

    fn group_requirement(
        descriptor: &CommunicationGroupDescriptor,
        operation: CommunicationOperation,
    ) -> &CommunicationOperationRequirement {
        descriptor
            .requirements()
            .operations()
            .iter()
            .find(|requirement| requirement.operation() == operation)
            .expect("group() established the selected operation")
    }

    fn validate_tensor(
        &self,
        value: &B::Tensor,
        requirement: &CommunicationOperationRequirement,
        completed: bool,
    ) -> Result<(), PartitionExecutionError> {
        let dtype = self.inspector.dtype(value);
        let shape = self.inspector.shape(value);
        let elements=shape.iter().try_fold(1usize,|product,dimension|product.checked_mul(*dimension));
        requirement.validate_tensor_metadata(&dtype,shape.len(),elements,completed).map_err(|error| match error {
            crate::CommunicationTensorContractError::Dtype=>PartitionExecutionError::TensorDtype{dtype},
            crate::CommunicationTensorContractError::MissingLimits=>PartitionExecutionError::MissingTensorLimits,
            crate::CommunicationTensorContractError::Limits=>PartitionExecutionError::TensorLimits{shape},
        })
    }

    fn wait<T>(
        &self,
        submission: eredu_core::Submission<T, B::CommunicationCompletion>,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> Result<T, PartitionExecutionError> {
        self.authority.wait(submission, operation, phase, route)
    }

    fn expected_axis_output_shape(
        &self,
        value: &B::Tensor,
        axis: usize,
        output_width: usize,
    ) -> Result<Vec<usize>, PartitionExecutionError> {
        let mut shape = self.inspector.shape(value);
        let rank = shape.len();
        let dimension = shape
            .get_mut(axis)
            .ok_or(PartitionExecutionError::CommunicationAxis { axis, rank })?;
        *dimension = output_width;
        Ok(shape)
    }

    fn validate_output_shape(
        &self,
        output: &B::Tensor,
        expected: Vec<usize>,
    ) -> Result<(), PartitionExecutionError> {
        let actual = self.inspector.shape(output);
        if actual != expected {
            return Err(PartitionExecutionError::CommunicationOutputShape { expected, actual });
        }
        Ok(())
    }

    fn output_contract_error(
        &self,
        error: PartitionExecutionError,
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
    ) -> PartitionExecutionError {
        self.authority
            .fence_protocol_failure(operation, phase, route);
        error
    }

    /// Executes one exact sum reduction using only its narrow backend capability.
    pub fn all_reduce_sum(
        &self,
        value: B::Tensor,
        group: CollectiveGroupId,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where
        B: SumReductionBackend,
    {
        self.all_reduce_sum_with_parallel(value, group, executor, None)
    }

    /// The same validated sum, using an explicitly retained model occurrence
    /// when supplied by the architecture driver.
    pub fn all_reduce_sum_with_parallel(
        &self, value: B::Tensor, group: CollectiveGroupId,
        executor: &B::Executor, context: Option<&B::ParallelContext>,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where B: SumReductionBackend,
    {
        let (descriptor, native) = self.group(group, CommunicationOperation::AllReduceSum)?;
        let requirement = Self::group_requirement(descriptor, CommunicationOperation::AllReduceSum);
        self.validate_tensor(&value, requirement, false)?;
        let completed = match context {
            Some(context) => B::complete_model_sum(&value, native, context, executor)
                .map_err(|error|self.submission_error(error, CommunicationOperation::AllReduceSum,
                    DistributedExecutionPhase::Execution, None))?,
            None => None,
        };
        let output = if let Some(output) = completed { output } else {
        let output = B::all_reduce_sum(value, native, executor).map_err(|error| {
            self.submission_error(
                error,
                CommunicationOperation::AllReduceSum,
                DistributedExecutionPhase::Execution,
                None,
            )
        })?;
        let output = self.wait(
            output,
            CommunicationOperation::AllReduceSum,
            DistributedExecutionPhase::Execution,
            None,
        )?;
        output
        };
        self.validate_tensor(&output, requirement, true)
            .map_err(|error| {
                self.output_contract_error(
                    error,
                    CommunicationOperation::AllReduceSum,
                    DistributedExecutionPhase::Execution,
                    None,
                )
            })?;
        Ok(output)
    }

    /// Submits one exact sum wave before waiting for any member. A selected
    /// source lends its actual payer to shared validation before constructing
    /// every occurrence; ordinary execution uses the same group and limits.
    pub fn all_reduce_sum_wave(
        &self, values: Vec<B::Tensor>, group: CollectiveGroupId,
        executor: &B::Executor, context: Option<&B::ParallelContext>,
    ) -> Result<Vec<B::Tensor>, PartitionExecutionError>
    where B: SumReductionBackend,
    {
        self.ensure_active()?;
        // This selector has fixed typed failure and allocates no diagnostic.
        let selected = self.manifest.select_group_operation(group, CommunicationOperation::AllReduceSum)
            .map_err(PartitionExecutionError::PreparedGroup)?;
        let descriptor = selected.descriptor();
        let native = self.groups[selected.order()].resource.borrow();
        let requirement = Self::group_requirement(descriptor, CommunicationOperation::AllReduceSum);
        if let Some(context) = context {
            let mut controls = None;
            let mut validated_input = false;
            let mut validated_output = false;
            let validate = |actual: &[B::Tensor], funding: &eredu_nn::workspace::HostMetadataFunding, completed: bool| {
                if controls.is_none() {
                    let frames = std::mem::size_of::<(
                        &Self, Vec<B::Tensor>, CollectiveGroupId, &B::Executor,
                        Option<&B::ParallelContext>, &CommunicationGroupDescriptor,
                        &B::CommunicationGroup, &CommunicationOperationRequirement,
                        std::slice::Iter<'_, B::Tensor>, Result<(), PartitionExecutionError>,
                        Vec<B::Tensor>, Result<Option<Vec<B::Tensor>>, eredu_core::BackendFailure>,
                        Result<Vec<B::Tensor>, PartitionExecutionError>, usize, bool, bool,
                    )>();
                    controls = Some(publication::Controls::<publication::WaveCause>::prepare::<Vec<B::Tensor>>(
                        funding, CommunicationOperation::AllReduceSum, frames)?);
                }
                let controls = controls.as_ref().expect("prepared source controls");
                if !controls.same_account(funding) || actual.len() != values.len()
                    || if completed { !validated_input || validated_output } else { validated_input } {
                    return Err(PartitionExecutionError::CommunicationPolicyMismatch);
                }
                for value in actual { controls.validate::<B, I>(&self.inspector, value, requirement, completed)?; }
                if completed { validated_output = true; } else { validated_input = true; }
                Ok(())
            };
            let result = B::complete_model_sum_wave(&values, native, context, executor, validate);
            match result {
                Ok(Some(outputs)) => {
                    if !validated_input || !validated_output || outputs.len() != values.len() {
                        return Err(match controls {
                            Some(controls) => controls.failure(&self.authority,
                                publication::WaveCause::Contract(PartitionExecutionError::CommunicationPolicyMismatch),
                                DistributedExecutionPhase::Execution, false),
                            None => self.output_contract_error(PartitionExecutionError::CommunicationPolicyMismatch,
                                CommunicationOperation::AllReduceSum, DistributedExecutionPhase::Execution, None),
                        });
                    }
                    return Ok(outputs);
                }
                Ok(None) if controls.is_none() => {},
                Ok(None) => return Err(controls.expect("selected source controls").failure(&self.authority,
                    publication::WaveCause::Contract(PartitionExecutionError::CommunicationPolicyMismatch),
                    DistributedExecutionPhase::Execution, false)),
                Err(cause) => {
                    return Err(match controls {
                        Some(controls) => controls.failure(&self.authority, publication::WaveCause::Native(cause), DistributedExecutionPhase::Execution, false),
                        None => {
                            self.authority.fence_protocol_failure(CommunicationOperation::AllReduceSum,
                                DistributedExecutionPhase::Execution, None);
                            PartitionExecutionError::PreparedCommunication {
                                operation: CommunicationOperation::AllReduceSum, phase: DistributedExecutionPhase::Execution,
                                completion: false, source: cause,
                            }
                        }
                    });
                }
            }
        }
        // Ordinary worker: all submissions precede every wait. No retained
        // source has been selected or consumed on this branch.
        let mut submissions = Vec::with_capacity(values.len());
        for value in values {
            self.validate_tensor(&value, requirement, false)?;
            submissions.push(B::all_reduce_sum(value, native, executor).map_err(|cause|
                self.submission_error(cause, CommunicationOperation::AllReduceSum,
                    DistributedExecutionPhase::Execution, None))?);
        }
        submissions.into_iter().map(|submission| {
            let output = self.wait(submission, CommunicationOperation::AllReduceSum,
                DistributedExecutionPhase::Execution, None)?;
            self.validate_tensor(&output, requirement, true).map_err(|cause|
                self.output_contract_error(cause, CommunicationOperation::AllReduceSum,
                    DistributedExecutionPhase::Execution, None))?;
            Ok(output)
        }).collect()
    }

    /// Lends the actual completed I32 matrix of one selected peer-row gather.
    /// The backend supplies native source/custody; this existing communication
    /// owner retains group selection, tensor limits and failure fencing.
    pub fn with_prepared_peer_count_consensus<T, E, F>(
        &self, local: &[i32], group: CollectiveGroupId,
        context: &B::ParallelContext, executor: &B::Executor, run: F,
    ) -> Result<Result<T, E>, PartitionExecutionError>
    where F: for<'loan> FnOnce(Option<(&'loan [i32],
        &'loan eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        self.with_prepared_peer_count_source(local, group, context, executor, |loan| {
            run(loan.map(|loan| { let (matrix, funding, _source) = loan.into_parts(); (matrix, funding) }))
        })
    }

    pub fn with_prepared_peer_count_source<T, E, F>(
        &self, local: &[i32], group: CollectiveGroupId,
        context: &B::ParallelContext, executor: &B::Executor, run: F,
    ) -> Result<Result<T, E>, PartitionExecutionError>
    where
        F: for<'loan> FnOnce(Option<crate::PreparedPeerCountLoan<'loan>>) -> Result<T, E>,
    {
        let operation = CommunicationOperation::AllGatherEven;
        let phase = DistributedExecutionPhase::Execution;
        let (descriptor, native) = self.group(group, operation)?;
        let peers = descriptor.members().len();
        if local.len()!=peers {
            return Err(PartitionExecutionError::PeerCount { expected: peers, actual: local.len() });
        }
        let expected = peers.checked_mul(peers)
            .ok_or(PartitionExecutionError::CommunicationShapeOverflow)?;
        let requirement = Self::group_requirement(descriptor, operation);
        requirement.validate_tensor_metadata(&TensorDtype::I32, 1, Some(local.len()), false)
            .map_err(|cause|PartitionExecutionError::PeerConsensusTensor { completed: false, cause })?;
        B::with_prepared_peer_count_source(local, native, context, executor, |matrix| {
            if let Some(loan) = &matrix {
                let matrix = loan.matrix();
                if matrix.len()!=expected {
                    return Err(self.output_contract_error(
                        PartitionExecutionError::PeerCount { expected, actual: matrix.len() }, operation, phase, None));
                }
                requirement.validate_tensor_metadata(&TensorDtype::I32, 1, Some(matrix.len()), true)
                    .map_err(|cause|self.output_contract_error(
                        PartitionExecutionError::PeerConsensusTensor { completed: true, cause }, operation, phase, None))?;
            }
            Ok::<_, PartitionExecutionError>(run(matrix))
        }).map_err(|cause|self.output_contract_error(
            PartitionExecutionError::PreparedCommunication {
                operation, phase, completion: false,
                source: eredu_core::BackendFailure::from_error(cause),
            }, operation, phase, None))?
    }

    /// Executes one exact equal-count gather.
    pub fn all_gather_even(
        &self,
        value: B::Tensor,
        axis: usize,
        group: CollectiveGroupId,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where
        B: EvenGatherBackend,
    {
        let (descriptor, native) = self.group(group, CommunicationOperation::AllGatherEven)?;
        let requirement =
            Self::group_requirement(descriptor, CommunicationOperation::AllGatherEven);
        self.validate_tensor(&value, requirement, false)?;
        let input_shape = self.inspector.shape(&value);
        let input_width =
            *input_shape
                .get(axis)
                .ok_or(PartitionExecutionError::CommunicationAxis {
                    axis,
                    rank: input_shape.len(),
                })?;
        let output_width = input_width
            .checked_mul(descriptor.members().len())
            .ok_or(PartitionExecutionError::CommunicationShapeOverflow)?;
        let expected = self.expected_axis_output_shape(&value, axis, output_width)?;
        let output = B::all_gather_even(value, axis, native, executor).map_err(|error| {
            self.submission_error(
                error,
                CommunicationOperation::AllGatherEven,
                DistributedExecutionPhase::Execution,
                None,
            )
        })?;
        let output = self.wait(
            output,
            CommunicationOperation::AllGatherEven,
            DistributedExecutionPhase::Execution,
            None,
        )?;
        self.validate_tensor(&output, requirement, true)
            .and_then(|()| self.validate_output_shape(&output, expected))
            .map_err(|error| {
                self.output_contract_error(
                    error,
                    CommunicationOperation::AllGatherEven,
                    DistributedExecutionPhase::Execution,
                    None,
                )
            })?;
        Ok(output)
    }

    /// Executes one exact unequal-count gather.
    pub fn all_gather_uneven(
        &self,
        value: B::Tensor,
        counts: &[usize],
        axis: usize,
        group: CollectiveGroupId,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where
        B: UnevenGatherBackend,
    {
        self.all_gather_uneven_with_parallel(value, counts, axis, group, executor, None)
    }

    /// The same exact gather using its explicitly retained model occurrence.
    pub fn all_gather_uneven_with_parallel(
        &self, value: B::Tensor, counts: &[usize], axis: usize,
        group: CollectiveGroupId, executor: &B::Executor,
        context: Option<&B::ParallelContext>,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where B: UnevenGatherBackend,
    {
        let (descriptor, native) = self.group(group, CommunicationOperation::AllGatherUneven)?;
        if counts.len() != descriptor.members().len() {
            return Err(PartitionExecutionError::PeerCount {
                expected: descriptor.members().len(),
                actual: counts.len(),
            });
        }
        let requirement =
            Self::group_requirement(descriptor, CommunicationOperation::AllGatherUneven);
        self.validate_tensor(&value, requirement, false)?;
        let output_width = counts.iter().try_fold(0usize, |total, count| {
            total
                .checked_add(*count)
                .ok_or(PartitionExecutionError::CommunicationShapeOverflow)
        })?;
        let expected = self.expected_axis_output_shape(&value, axis, output_width)?;
        let completed = match context {
            Some(context) => B::complete_model_gather(&value, counts, axis, native, context, executor)
                .map_err(|error|self.submission_error(error, CommunicationOperation::AllGatherUneven,
                    DistributedExecutionPhase::Execution, None))?,
            None => None,
        };
        let output = if let Some(output) = completed { output } else {
        let output =
            B::all_gather_uneven(value, counts, axis, native, executor).map_err(|error| {
                self.submission_error(
                    error,
                    CommunicationOperation::AllGatherUneven,
                    DistributedExecutionPhase::Execution,
                    None,
                )
            })?;
        let output = self.wait(
            output,
            CommunicationOperation::AllGatherUneven,
            DistributedExecutionPhase::Execution,
            None,
        )?;
        output
        };
        self.validate_tensor(&output, requirement, true)
            .and_then(|()| self.validate_output_shape(&output, expected))
            .map_err(|error| {
                self.output_contract_error(
                    error,
                    CommunicationOperation::AllGatherUneven,
                    DistributedExecutionPhase::Execution,
                    None,
                )
            })?;
        Ok(output)
    }

    /// Executes the selected expert plan's exact variable-count exchange without exposing EP semantics.
    pub fn variable_all_to_all(
        &self,
        value: B::Tensor,
        counts: &CommunicationPeerCounts,
        axis: usize,
        group: CollectiveGroupId,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where
        B: VariableAllToAllBackend,
    {
        let (descriptor, native) = self.group(group, CommunicationOperation::VariableAllToAll)?;
        if counts.group_size() != descriptor.members().len() {
            return Err(PartitionExecutionError::PeerCount {
                expected: descriptor.members().len(),
                actual: counts.group_size(),
            });
        }
        let requirement =
            Self::group_requirement(descriptor, CommunicationOperation::VariableAllToAll);
        self.validate_tensor(&value, requirement, false)?;
        let max = requirement
            .limits()
            .and_then(|limits| limits.max_count_per_peer())
            .ok_or(PartitionExecutionError::MissingTensorLimits)?;
        if counts
            .send()
            .iter()
            .chain(counts.receive())
            .any(|count| *count > max)
        {
            return Err(PartitionExecutionError::PeerCountLimit { maximum: max });
        }
        let output_width = counts.receive().iter().try_fold(0usize, |total, count| {
            total
                .checked_add(*count)
                .ok_or(PartitionExecutionError::CommunicationShapeOverflow)
        })?;
        let expected = self.expected_axis_output_shape(&value, axis, output_width)?;
        let output =
            B::variable_all_to_all(value, counts, axis, native, executor).map_err(|error| {
                self.submission_error(
                    error,
                    CommunicationOperation::VariableAllToAll,
                    DistributedExecutionPhase::Execution,
                    None,
                )
            })?;
        let output = self.wait(
            output,
            CommunicationOperation::VariableAllToAll,
            DistributedExecutionPhase::Execution,
            None,
        )?;
        self.validate_tensor(&output, requirement, true)
            .and_then(|()| self.validate_output_shape(&output, expected))
            .map_err(|error| {
                self.output_contract_error(
                    error,
                    CommunicationOperation::VariableAllToAll,
                    DistributedExecutionPhase::Execution,
                    None,
                )
            })?;
        Ok(output)
    }

    fn transfer_boundary(
        &self,
        route: CommunicationRouteId,
        values: Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,
        schema: &ResolvedBoundaryWireSchema,
        wire: PipelineWireContract,
        executor: &B::Executor,
    ) -> Result<Vec<B::Tensor>, PartitionExecutionError>
    where
        B: PointToPointBackend,
    {
        let (descriptor, native) = self.route(route)?;
        let rank = self.manifest.rank();
        if rank != descriptor.source() && rank != descriptor.destination() {
            return Err(PartitionExecutionError::NotRouteEndpoint(route));
        }
        validate_tagged_boundary_bundle::<B, I>(&self.inspector, &values, schema, wire)?;
        let tensors = values
            .iter()
            .map(crate::ArchitectureBoundaryValue::tensor)
            .cloned()
            .collect::<Vec<_>>();
        validate_bundle_requirement::<B, I>(
            &self.inspector,
            &tensors,
            descriptor.requirement(),
            false,
        )?;
        let contract = descriptor.boundary_contract().ok_or_else(|| {
            PartitionExecutionError::BoundaryFraming(
                "point-to-point boundary route has no role-exact framing contract".into(),
            )
        })?;
        if contract.schema() != schema.identity() {
            return Err(PartitionExecutionError::BoundaryFraming(format!(
                "route schema {:?} differs from execution schema {:?}",
                contract.schema(),
                schema.identity()
            )));
        }
        let actual_roles = resolved_tagged_boundary_roles(&values, schema, wire)?;
        let values = contract
            .frame_values(
                route,
                &actual_roles,
                values
                    .into_iter()
                    .map(crate::ArchitectureBoundaryValue::into_parts)
                    .map(|(_, tensor)| tensor)
                    .collect(),
            )
            .map_err(|error| PartitionExecutionError::BoundaryFraming(error.to_string()))?;
        let submission = B::send_receive(values, native, executor).map_err(|error| {
            self.submission_error(
                error,
                CommunicationOperation::SendReceive,
                DistributedExecutionPhase::Execution,
                Some(route),
            )
        })?;
        let output = self.wait(
            submission,
            CommunicationOperation::SendReceive,
            DistributedExecutionPhase::Execution,
            Some(route),
        )?;
        validate_boundary_bundle::<B, I>(&self.inspector, &output, schema, wire)
            .and_then(|()| {
                validate_bundle_requirement::<B, I>(
                    &self.inspector,
                    &output,
                    descriptor.requirement(),
                    true,
                )
            })
            .map_err(|error| {
                self.output_contract_error(
                    error,
                    CommunicationOperation::SendReceive,
                    DistributedExecutionPhase::Execution,
                    Some(route),
                )
            })?;
        Ok(output)
    }

    fn boundary_endpoint_is_source(
        &self,
        route: CommunicationRouteId,
    ) -> Result<bool, PartitionExecutionError> {
        let descriptor = self
            .manifest
            .routes()
            .iter()
            .find(|candidate| candidate.id() == route)
            .ok_or(PartitionExecutionError::UnknownRoute(route))?;
        let rank = self.manifest.rank();
        if rank != descriptor.source() && rank != descriptor.destination() {
            return Err(PartitionExecutionError::NotRouteEndpoint(route));
        }
        Ok(rank == descriptor.source())
    }

    fn validate_prepared_boundary(
        &self,
        route: CommunicationRouteId,
        values: &[crate::ArchitectureBoundaryValue<B::Tensor>],
        schema: &ResolvedBoundaryWireSchema,
        wire: PipelineWireContract,
    ) -> Result<(), PartitionExecutionError> {
        let descriptor = self
            .manifest
            .routes()
            .iter()
            .find(|candidate| candidate.id() == route)
            .ok_or(PartitionExecutionError::UnknownRoute(route))?;
        let rank = self.manifest.rank();
        if rank != descriptor.source() && rank != descriptor.destination() {
            return Err(PartitionExecutionError::NotRouteEndpoint(route));
        }
        validate_tagged_boundary_bundle::<B, I>(&self.inspector, values, schema, wire)?;
        let contract = descriptor.boundary_contract().ok_or_else(|| {
            PartitionExecutionError::BoundaryFraming(
                "point-to-point boundary route has no role-exact framing contract".into(),
            )
        })?;
        if contract.schema() != schema.identity() {
            return Err(PartitionExecutionError::BoundaryFraming(format!(
                "route schema {:?} differs from prepared schema {:?}",
                contract.schema(),
                schema.identity(),
            )));
        }
        contract
            .validate_invocation(&resolved_tagged_boundary_roles(values, schema, wire)?)
            .map_err(|error| PartitionExecutionError::BoundaryFraming(error.to_string()))?;
        let tensors = values
            .iter()
            .map(crate::ArchitectureBoundaryValue::tensor)
            .cloned()
            .collect::<Vec<_>>();
        validate_bundle_requirement::<B, I>(
            &self.inspector,
            &tensors,
            descriptor.requirement(),
            false,
        )
    }

    fn broadcast_output(
        &self,
        value: B::Tensor,
        publication: PartitionOutputPublication,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError>
    where
        B: BroadcastBackend,
    {
        self.broadcast_output_with_parallel(value,publication,phase,executor,None)
    }

    fn broadcast_output_with_parallel(&self,value:B::Tensor,publication:PartitionOutputPublication,
        phase:DistributedExecutionPhase,executor:&B::Executor,
        prepared:Option<(&B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)
        ->Result<B::Tensor,PartitionExecutionError> where B:BroadcastBackend {
        let controls=prepared.map(|(_,funding)|publication::Controls::<B::CommunicationError>::prepare::<B::Tensor>(funding,
            CommunicationOperation::Broadcast, std::mem::size_of::<(
                PartitionOutputPublication, DistributedExecutionPhase,
                eredu_core::Submission<B::Tensor, B::CommunicationCompletion>,
                Result<eredu_core::Submission<B::Tensor, B::CommunicationCompletion>, B::CommunicationError>,
                BoundedSubmissionOutcome<B::Tensor>, Result<BoundedSubmissionOutcome<B::Tensor>, B::CommunicationError>,
            )>())).transpose()?;
        self.ensure_active()?;
        // Same selected-group validator; prepared errors retain its finite
        // typed cause instead of allocating an ordinary diagnostic label.
        let (descriptor,native)=if controls.is_some() {
            let selected=self.manifest.select_group_operation(publication.group,CommunicationOperation::Broadcast)
                .map_err(PartitionExecutionError::PreparedGroup)?;
            (selected.descriptor(),self.groups[selected.order()].resource.borrow())
        } else {self.group(publication.group,CommunicationOperation::Broadcast)?};
        let root=descriptor.members().iter().position(|rank|*rank==publication.owner_rank)
            .ok_or(PartitionExecutionError::OutputOwnerNotMember {rank:publication.owner_rank,group:publication.group})?;
        let requirement=Self::group_requirement(descriptor,CommunicationOperation::Broadcast);
        let validate=|value:&B::Tensor,completed|->Result<(),PartitionExecutionError> {
            match &controls {
                Some(controls)=>controls.validate::<B,I>(&self.inspector,value,requirement,completed),
                None=>self.validate_tensor(value,requirement,completed),
            }
        };
        validate(&value,false)?;
        let failed=|error,completion|match &controls {
            Some(controls)=>controls.failure(&self.authority,error,phase,completion),
            None=>self.submission_error(error,CommunicationOperation::Broadcast,phase,None),
        };
        let run=|native:&B::CommunicationGroup|->Result<B::Tensor,PartitionExecutionError> {
            let submission=B::broadcast(value,root,native,executor).map_err(|error|failed(error,false))?;
            let output=match &controls {
                Some(_)=>self.authority.wait_with_error(submission,CommunicationOperation::Broadcast,phase,None,
                    |error|failed(error,true))?,
                None=>self.wait(submission,CommunicationOperation::Broadcast,phase,None)?,
            };
            validate(&output,true).map_err(|error|self.output_contract_error(error,CommunicationOperation::Broadcast,phase,None))?;
            Ok(output)
        };
        match prepared {
            None=>run(native),
            Some((prepared,funding))=>B::with_prepared_publication_group(native,prepared,funding,executor,|bound| {
                run(bound.ok_or(PartitionExecutionError::CommunicationPolicyMismatch)?)
            }).map_err(|error|failed(error,false))?,
        }
    }

    fn barrier(
        &self,
        group: CollectiveGroupId,
        executor: &B::Executor,
    ) -> Result<(), PartitionExecutionError>
    where
        B: BarrierBackend,
    {
        self.barrier_with_group(group, executor, None)
    }

    fn barrier_with_group(
        &self,
        group: CollectiveGroupId,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<(), PartitionExecutionError>
    where
        B: BarrierBackend,
    {
        let (_, selected) = self.group(group, CommunicationOperation::Barrier)?;
        let native = prepared.unwrap_or(selected);
        let completion = B::barrier(native, executor).map_err(|error| {
            self.submission_error(
                error,
                CommunicationOperation::Barrier,
                DistributedExecutionPhase::Commit,
                None,
            )
        })?;
        let policy = self
            .manifest
            .completion_policy()
            .expect("partition communication requires bounded completion")
            .bounded_wait();
        let outcome = match completion.wait_bounded(policy) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.authority.mark_poisoned(CommunicationPoison {
                    operation: CommunicationOperation::Barrier,
                    phase: DistributedExecutionPhase::Commit,
                    route: None,
                    cancellation: policy.cancellation(),
                });
                return Err(PartitionExecutionError::CommunicationCompletionFailed {
                    operation: CommunicationOperation::Barrier,
                    phase: DistributedExecutionPhase::Commit,
                    route: None,
                    error: error.to_string(),
                    source: Some(eredu_core::BackendFailure::from_error(error)),
                });
            }
        };
        match outcome {
            BoundedCompletionOutcome::Completed => Ok(()),
            BoundedCompletionOutcome::DeadlineExceeded { cancellation } => {
                self.authority.mark_poisoned(CommunicationPoison {
                    operation: CommunicationOperation::Barrier,
                    phase: DistributedExecutionPhase::Commit,
                    route: None,
                    cancellation,
                });
                Err(PartitionExecutionError::CommunicationDeadlineExceeded {
                    operation: CommunicationOperation::Barrier,
                    phase: DistributedExecutionPhase::Commit,
                    route: None,
                    cancellation,
                })
            }
        }
    }

    fn agree_success(
        &self,
        local_success: bool,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
    ) -> Result<bool, PartitionExecutionError>
    where
        B: FailureAgreementBackend,
    {
        self.agree_success_with_group(local_success, group, phase, executor, None)
    }

    fn agree_success_with_group(
        &self,
        local_success: bool,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<bool, PartitionExecutionError>
    where
        B: FailureAgreementBackend,
    {
        let (_, selected) = self.group(group, CommunicationOperation::FailureAgreement)?;
        let native = prepared.unwrap_or(selected);
        let output = self.wait(
            B::agree_success(local_success, native, executor).map_err(|error| {
                self.submission_error(error, CommunicationOperation::FailureAgreement, phase, None)
            })?,
            CommunicationOperation::FailureAgreement,
            phase,
            None,
        )?;
        B::resolve_failure_agreement(output).map_err(|error| {
            self.authority.mark_poisoned(CommunicationPoison {
                operation: CommunicationOperation::FailureAgreement,
                phase,
                route: None,
                cancellation: self
                    .manifest
                    .completion_policy()
                    .expect("partition communication requires bounded completion")
                    .cancellation(),
            });
            PartitionExecutionError::CommunicationCompletionFailed {
                operation: CommunicationOperation::FailureAgreement,
                phase,
                route: None,
                error: error.to_string(),
                source: Some(eredu_core::BackendFailure::from_error(error)),
            }
        })
    }

    fn agree_success_after_prior_failure(
        &self,
        local_success: bool,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
    ) -> Result<bool, PartitionExecutionError>
    where
        B: FailureAgreementBackend,
    {
        self.agree_success_after_prior_failure_with_group(local_success, group, phase, executor, None)
    }

    fn agree_success_after_prior_failure_with_group(
        &self,
        local_success: bool,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<bool, PartitionExecutionError>
    where
        B: FailureAgreementBackend,
    {
        if self
            .authority
            .shared
            .terminal
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(PartitionExecutionError::CommunicationTerminal);
        }
        if local_success && !self.authority.is_poisoned() {
            return Err(PartitionExecutionError::RecoveryAgreementWithoutFailure { phase });
        }
        let index = self
            .manifest
            .groups()
            .iter()
            .position(|candidate| candidate.id() == group)
            .ok_or(PartitionExecutionError::UnknownGroup(group))?;
        let descriptor = &self.manifest.groups()[index];
        if descriptor.local_index().is_none() {
            return Err(PartitionExecutionError::NotGroupMember(group));
        }
        if !descriptor
            .requirements()
            .operations()
            .iter()
            .any(|requirement| requirement.operation() == CommunicationOperation::FailureAgreement)
        {
            return Err(PartitionExecutionError::OperationNotSelected {
                resource: format!("group {}", group.value()),
                operation: CommunicationOperation::FailureAgreement,
            });
        }
        let native = prepared.unwrap_or_else(|| self.groups[index].resource.borrow());
        let submission = B::agree_success(local_success, native, executor).map_err(|error| {
            self.authority.mark_poisoned(CommunicationPoison {
                operation: CommunicationOperation::FailureAgreement,
                phase,
                route: None,
                cancellation: self
                    .manifest
                    .completion_policy()
                    .expect("partition communication requires bounded completion")
                    .cancellation(),
            });
            PartitionExecutionError::CommunicationSubmissionFailed {
                operation: CommunicationOperation::FailureAgreement,
                phase,
                route: None,
                error: error.to_string(),
                source: Some(eredu_core::BackendFailure::from_error(error)),
            }
        })?;
        let output = self.authority.wait_after_prior_failure(
            submission,
            CommunicationOperation::FailureAgreement,
            phase,
            None,
        )?;
        B::resolve_failure_agreement(output).map_err(|error| {
            self.authority.mark_poisoned(CommunicationPoison {
                operation: CommunicationOperation::FailureAgreement,
                phase,
                route: None,
                cancellation: self
                    .manifest
                    .completion_policy()
                    .expect("partition communication requires bounded completion")
                    .cancellation(),
            });
            PartitionExecutionError::CommunicationCompletionFailed {
                operation: CommunicationOperation::FailureAgreement,
                phase,
                route: None,
                error: error.to_string(),
                source: Some(eredu_core::BackendFailure::from_error(error)),
            }
        })
    }

    fn complete_local_dependencies(
        &self,
        values: &[crate::ArchitectureBoundaryValue<B::Tensor>],
        route: CommunicationRouteId,
        executor: &B::Executor,
        before_failure_agreement: bool,
        source:Option<&crate::PreparedBoundarySource>,
        context:Option<&B::ParallelContext>,
    ) -> Result<(), PartitionExecutionError> {
        if let Some(source)=source {
            return self.complete_boundary_prepared(values,route,executor,before_failure_agreement,
                source,context.ok_or_else(||source.error(crate::PreparedBoundaryFrameCause::Contract))?);
        }
        self.ensure_active()?;
        let descriptor = self
            .manifest
            .routes()
            .iter()
            .find(|candidate| candidate.id() == route)
            .ok_or(PartitionExecutionError::UnknownRoute(route))?;
        if descriptor.source() != self.manifest.rank() {
            return Err(PartitionExecutionError::NotRouteEndpoint(route));
        }
        let phase = DistributedExecutionPhase::BoundarySourceCompletion(route);
        let submission = B::submit_local_dependencies(
            values.iter().map(crate::ArchitectureBoundaryValue::tensor),
            executor,
        )
        .map_err(|error| {
            if before_failure_agreement {
                PartitionExecutionError::CommunicationSubmissionFailed {
                    operation: CommunicationOperation::SendReceive,
                    phase,
                    route: Some(route),
                    error: error.to_string(),
                    source: Some(eredu_core::BackendFailure::from_error(error)),
                }
            } else {
                self.submission_error(
                    error,
                    CommunicationOperation::SendReceive,
                    phase,
                    Some(route),
                )
            }
        })?;
        if before_failure_agreement {
            self.authority.wait_before_failure_agreement(
                submission,
                CommunicationOperation::SendReceive,
                phase,
                Some(route),
            )
        } else {
            self.wait(
                submission,
                CommunicationOperation::SendReceive,
                phase,
                Some(route),
            )
        }
    }

    /// Completes exact local tensor dependencies before the caller advances an
    /// architecture-declared distributed execution wave.
    ///
    /// This is distinct from a collective: it contributes no tensor value and
    /// exists only to make lazy predecessors reach the selected bounded
    /// completion policy while every rank is still at the matching wave
    /// position.
    pub fn complete_execution_dependencies_with_parallel(
        &self, values: &[&B::Tensor], executor: &B::Executor,
        context: Option<&B::ParallelContext>,
    ) -> Result<(), PartitionExecutionError> {
        self.ensure_active()?;
        if let Some(context) = context {
            if B::complete_model_dependencies(values, context, executor).map_err(|error|
                self.submission_error(error, CommunicationOperation::SendReceive,
                    DistributedExecutionPhase::Execution, None))?.is_some() {
                return Ok(());
            }
        }
        self.complete_execution_dependencies(values.iter().copied(), executor)
    }

    pub fn complete_execution_dependencies<'a, V>(
        &self,
        values: V,
        executor: &B::Executor,
    ) -> Result<(), PartitionExecutionError>
    where
        V: IntoIterator<Item = &'a B::Tensor>,
        B::Tensor: 'a,
    {
        self.ensure_active()?;
        let submission = B::submit_local_dependencies(values, executor).map_err(|error| {
            self.submission_error(
                error,
                CommunicationOperation::SendReceive,
                DistributedExecutionPhase::Execution,
                None,
            )
        })?;
        self.wait(
            submission,
            CommunicationOperation::SendReceive,
            DistributedExecutionPhase::Execution,
            None,
        )
    }
}

/// Canonical shared-session phases whose local status must be propagated.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum DistributedExecutionPhase {
    /// The exact request span fits its retained inference workspace admission.
    InferenceWorkspace,
    /// Every rank prepared its input before observation admission or mutable execution.
    InputPreparation,
    /// Every rank can project only the final position; a sequence observer on
    /// any participant keeps full readout geometry on every participant.
    ReadoutSelection,
    /// Every participant requests no scores after agreeing that no sequence
    /// observer requires readout. Otherwise participants keep final-row scores.
    StateOnlyReadout,
    /// All participants retained request capacity before preparing a chunk.
    PrefillReservation,
    /// All ranks prepared a source without mutating decoder state.
    PrefillSourcePreparation,
    /// Every rank can consume the same scheduled source protocol.
    PrefillSourceSelection,
    /// Every active prefill rank stops if any participant requests cancellation.
    PrefillCancellation,
    /// Native reservation scopes settled before further chunk execution.
    PrefillReservationCompletion,
    /// Checks whether every rank omits a transactional observer.
    ObservationParticipation,
    /// Checks whether every rank supplies a transactional observer.
    ObservationParticipationRequired,
    /// Every transactional observer completed local admission before any model work.
    ObservationPreparation,
    /// Prepared observer identities and quotas agree before the forward.
    ObservationCoordination,
    /// Completed observer records are staged before state commitment.
    ObservationDelivery,
    /// Every rank captured the state checkpoint required for transactional rollback.
    StateCheckpoint,
    /// Every rank accepted the exact prompt-cache load identity and input.
    PromptCacheLoadPreflight,
    /// Every stateful rank loaded and validated a provisional cache shard.
    PromptCacheLoadPreparation,
    /// Every rank accepted the exact prompt-cache save identity and input.
    PromptCacheSavePreflight,
    /// Every stateful rank serialized and validated an unpublished cache shard.
    PromptCacheSavePreparation,
    /// Every stateful rank reversibly published its prepared cache shard.
    PromptCacheSavePublication,
    /// Every stateful rank captured an opaque manual-control checkpoint.
    SessionCheckpoint,
    /// Every stateful rank prepared a replacement selected state.
    SessionResetPreparation,
    /// Every stateful rank restored a checkpoint into provisional state.
    SessionRollbackPreparation,
    /// Every rank completed an independent copy of its installed state.
    ControlCapturePreparation,
    /// Every rank completed an independent copy of the same saved branch.
    ControlCopyPreparation,
    /// Every rank validated the same branch exchange before any state changes.
    ControlExchangePreparation,
    /// Source tensor dependencies reached exact bounded completion for one route.
    BoundarySourceCompletion(CommunicationRouteId),
    /// Source execution and both endpoint preparations completed for one route.
    BoundarySourceReady(CommunicationRouteId),
    /// Rank-local graph execution, including boundary transfer and publication.
    Execution,
    /// Final output observation and intervention.
    OutputObservation,
    /// Output-owner hidden capture is present and published for prediction.
    PredictionTargetCapture,
    /// Every rank completed publication of the exact prediction target capture.
    PredictionTargetCapturePublication,
    /// Every rank validated a provisional prediction-lane target state swap.
    PredictionTargetStatePreparation,
    /// Every rank completed one typed prediction-only unit operation.
    PredictionExtensionExecution,
    /// Publication of the authoritative intervened output to every rank.
    OutputPublication,
    /// Publication of the authoritative sampled token and stop status.
    SamplingSynchronization,
    /// Exact output and mutable-state mechanism completion.
    MechanismCompletion,
    /// Final distributed state-commit decision.
    Commit,
}

/// One exact directed architecture boundary after a graph group completes.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionBoundaryRoute {
    /// Producing architecture group.
    pub source_group: usize,
    /// Consuming architecture group.
    pub destination_group: usize,
    /// Exact exclusive source-unit endpoint for a continuation within one group.
    /// None preserves manually constructed legacy routes without cut metadata.
    pub source_unit_end: Option<usize>,
    /// World rank selected by architecture ownership as the producer.
    pub source_rank: usize,
    /// World rank selected by architecture ownership as the consumer.
    pub destination_rank: usize,
    /// Opaque route selected in the communication manifest.
    pub route: CommunicationRouteId,
}

/// Authoritative output owner and opaque publication group.
#[derive(Debug, Clone, Copy)]
pub struct PartitionOutputPublication {
    /// Opaque group containing the output owner and every session participant.
    pub group: CollectiveGroupId,
    /// World rank that owns architecture projection and supplies source data.
    pub owner_rank: usize,
}

/// Exact public-output authority projected into one selected communication group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PartitionOutputAuthority {
    owner_rank: usize,
    owner_group_rank: usize,
    local_public_output: bool,
}

impl PartitionOutputAuthority {
    /// Architecture-selected world rank supplying authoritative logits.
    pub const fn owner_rank(self) -> usize {
        self.owner_rank
    }

    /// Owner's exact local rank within the selected publication group.
    pub const fn owner_group_rank(self) -> usize {
        self.owner_group_rank
    }

    /// Whether this manifest rank may expose logits through the public adapter.
    pub const fn local_public_output(self) -> bool {
        self.local_public_output
    }
}

// A replay retains the exact initial validated plan; ordinary construction
// continues to own its moved plan without adding a shared allocation. Both
// forms enter the same manifest/policy validator and execution driver.
enum PartitionPlanStorage {
    Owned(PartitionedExecutionPlan),
    Retained(std::sync::Arc<PartitionedExecutionPlan>),
}
impl std::ops::Deref for PartitionPlanStorage {
    type Target = PartitionedExecutionPlan;
    fn deref(&self) -> &Self::Target {
        match self { Self::Owned(plan) => plan, Self::Retained(plan) => plan }
    }
}

/// Immutable rank-local execution and communication plan.
#[derive(Debug, Clone)]
pub struct PartitionedExecutionPlan {
    graph: ExecutionGraph,
    group_contracts: Vec<(ArchitectureGroupKind, bool)>,
    drivers: Vec<Option<LayeredPartitionDriver>>,
    routes: Vec<PartitionBoundaryRoute>,
    publication: Option<PartitionOutputPublication>,
    commit_barrier: Option<CollectiveGroupId>,
    wire: PipelineWireContract,
}

impl PartitionedExecutionPlan {
    /// Validates group slots, local ownership, route dependencies, and output ownership.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph: ExecutionGraph,
        group_contracts: Vec<(ArchitectureGroupKind, bool)>,
        drivers: Vec<Option<LayeredPartitionDriver>>,
        routes: Vec<PartitionBoundaryRoute>,
        publication: Option<PartitionOutputPublication>,
        commit_barrier: Option<CollectiveGroupId>,
        wire: PipelineWireContract,
    ) -> Result<Self, PartitionExecutionError> {
        if group_contracts.len() != graph.groups().len() || drivers.len() != graph.groups().len() {
            return Err(PartitionExecutionError::GroupCount {
                graph: graph.groups().len(),
                contracts: group_contracts.len(),
                drivers: drivers.len(),
            });
        }
        for (group, driver) in drivers.iter().enumerate() {
            if driver
                .as_ref()
                .is_some_and(|driver| driver.group_index() != group)
            {
                return Err(PartitionExecutionError::DriverGroup { group });
            }
        }
        for route in &routes {
            if route.source_group >= graph.groups().len()
                || route.destination_group >= graph.groups().len()
            {
                return Err(PartitionExecutionError::RouteDependency(route.route));
            }
            if route.source_group != route.destination_group {
                let dependencies = graph
                    .dependencies(route.destination_group)
                    .ok_or(PartitionExecutionError::RouteGroup(route.route))?;
                if !dependencies.contains(&route.source_group) {
                    return Err(PartitionExecutionError::RouteDependency(route.route));
                }
            }
        }
        Ok(Self {
            graph,
            group_contracts,
            drivers,
            routes,
            publication,
            commit_barrier,
            wire,
        })
    }

    fn validate_manifest(
        &self,
        manifest: &CommunicationManifest,
    ) -> Result<(), PartitionExecutionError> {
        if self.routes.len() != manifest.routes().len() {
            return Err(PartitionExecutionError::RouteCount {
                plan: self.routes.len(),
                manifest: manifest.routes().len(),
            });
        }
        for (planned, descriptor) in self.routes.iter().zip(manifest.routes()) {
            if planned.route != descriptor.id()
                || planned.source_rank != descriptor.source()
                || planned.destination_rank != descriptor.destination()
            {
                return Err(PartitionExecutionError::RouteDescriptorMismatch {
                    route: planned.route,
                    planned_source: planned.source_rank,
                    planned_destination: planned.destination_rank,
                    manifest_source: descriptor.source(),
                    manifest_destination: descriptor.destination(),
                });
            }
            let rank = manifest.rank();
            if rank == planned.source_rank && self.drivers[planned.source_group].is_none() {
                return Err(PartitionExecutionError::RouteOwnerMissing {
                    route: planned.route,
                    rank,
                    group: planned.source_group,
                });
            }
            if rank == planned.destination_rank && self.drivers[planned.destination_group].is_none()
            {
                return Err(PartitionExecutionError::RouteOwnerMissing {
                    route: planned.route,
                    rank,
                    group: planned.destination_group,
                });
            }
        }
        if let Some(publication) = self.publication {
            let descriptor = manifest
                .groups()
                .iter()
                .find(|descriptor| descriptor.id() == publication.group)
                .ok_or(PartitionExecutionError::UnknownGroup(publication.group))?;
            if !descriptor.members().contains(&publication.owner_rank) {
                return Err(PartitionExecutionError::OutputOwnerNotMember {
                    rank: publication.owner_rank,
                    group: publication.group,
                });
            }
            let local_owns_output = self
                .drivers
                .iter()
                .flatten()
                .any(LayeredPartitionDriver::owns_output);
            if manifest.rank() == publication.owner_rank && !local_owns_output {
                return Err(PartitionExecutionError::OutputOwnership {
                    rank: manifest.rank(),
                    owner: publication.owner_rank,
                });
            }
            Self::validate_exact_group_operation(descriptor, CommunicationOperation::Broadcast)?;
        }
        Ok(())
    }

    fn validate_exact_group_operation(
        descriptor: &CommunicationGroupDescriptor,
        operation: CommunicationOperation,
    ) -> Result<(), PartitionExecutionError> {
        let selected = descriptor
            .requirements()
            .operations()
            .iter()
            .find(|requirement| requirement.operation() == operation)
            .ok_or_else(|| PartitionExecutionError::OperationNotSelected {
                resource: format!("group {}", descriptor.id().value()),
                operation,
            })?;
        if !selected.exact_completion() {
            return Err(PartitionExecutionError::InexactOperationRequirement {
                group: descriptor.id(),
                operation,
            });
        }
        Ok(())
    }

    fn validate_exact_commit_agreement(
        &self,
        manifest: &CommunicationManifest,
    ) -> Result<(), PartitionExecutionError> {
        let group = self
            .commit_barrier
            .ok_or(PartitionExecutionError::CommunicationPolicyMismatch)?;
        let descriptor = manifest
            .groups()
            .iter()
            .find(|descriptor| descriptor.id() == group)
            .ok_or(PartitionExecutionError::UnknownGroup(group))?;
        Self::validate_exact_group_operation(descriptor, CommunicationOperation::FailureAgreement)
    }

    /// Rank-local execution drivers in architecture-group order.
    pub fn drivers(&self) -> &[Option<LayeredPartitionDriver>] {
        &self.drivers
    }

    /// Architecture-selected point-to-point boundary routes.
    pub fn routes(&self) -> &[PartitionBoundaryRoute] {
        &self.routes
    }

    /// Selected output publication, including its exact opaque session group.
    pub const fn publication(&self) -> Option<PartitionOutputPublication> {
        self.publication
    }

    /// Resolves architecture-selected publication authority without backend
    /// topology inference.
    pub fn publication_authority(
        &self,
        manifest: &CommunicationManifest,
    ) -> Result<Option<PartitionOutputAuthority>, PartitionExecutionError> {
        let Some(publication) = self.publication else {
            return Ok(None);
        };
        let descriptor = manifest
            .groups()
            .iter()
            .find(|descriptor| descriptor.id() == publication.group)
            .ok_or(PartitionExecutionError::UnknownGroup(publication.group))?;
        let owner_group_rank = descriptor
            .members()
            .iter()
            .position(|rank| *rank == publication.owner_rank)
            .ok_or(PartitionExecutionError::OutputOwnerNotMember {
                rank: publication.owner_rank,
                group: publication.group,
            })?;
        Ok(Some(PartitionOutputAuthority {
            owner_rank: publication.owner_rank,
            owner_group_rank,
            local_public_output: manifest.rank() == publication.owner_rank,
        }))
    }

    /// Exact opaque session group used for post-publication commit synchronization.
    pub const fn commit_barrier(&self) -> Option<CollectiveGroupId> {
        self.commit_barrier
    }

    fn selects_phase_failure_agreement(&self, manifest: &CommunicationManifest) -> bool {
        self.commit_barrier.is_some_and(|group| {
            manifest.groups().iter().any(|descriptor| {
                descriptor.id() == group
                    && descriptor
                        .requirements()
                        .operations()
                        .iter()
                        .any(|requirement| {
                            requirement.operation() == CommunicationOperation::FailureAgreement
                        })
            })
        })
    }
}

/// Statically dispatched architecture/provider adapter used by the production driver.
///
/// Direct, routed-provider, and composite prepared-input implementations share this
/// interface without erasing tensors or per-unit execution. Typed architecture boundary
/// values are encoded and decoded inside the adapter; only their validated native tensor
/// bundle crosses the communication seam.
pub trait PartitionedGroupExecutor<A, B, S, G, R, I>
where
    B: CommunicationBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Declares this executor's actual group-completion worker independently
    /// of its parameter owner. Custom executors must classify it explicitly.
    fn group_submission_mechanism(&self) -> crate::GroupSubmissionMechanism;

    /// Internal hook coverage of this exact architecture/executor pairing.
    /// The shared partition session adds its final-publication seam separately.
    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Produce stable observation declarations under initial loading authority.
    /// This optional producer does not grant transport or finite execution.
    fn prepare_observation_paths(&self) -> Result<Option<crate::PreparedLayeredObservationPaths>,
        crate::PreparedLayeredObservationError<A::Error>> { Ok(None) }

    /// Rebind the same immutable source after authorized semantic validation.
    fn bind_observation_paths(&self, _source: &crate::SharedLayeredObservationPaths, metadata: Option<crate::layered::LayeredMetadata<A::Error>>)
        -> Result<crate::PreparedLayeredObservationPaths, crate::PreparedLayeredObservationError<A::Error>> {
        Err(crate::PreparedLayeredObservationError::BindingMismatch)
    }

    /// Validate the existing runtime token without creating source payloads.
    fn validate_observation_paths(&self, _paths: &crate::PreparedLayeredObservationPaths)
        -> Result<(), crate::PreparedLayeredObservationError<std::convert::Infallible>> {
        Err(crate::PreparedLayeredObservationError::BindingMismatch)
    }

    /// Begin the same invocation with an explicitly borrowed observation token.
    /// The selected executor must consume this loan in its ordinary unit worker.
    #[allow(clippy::too_many_arguments)]
    fn begin_with_prepared_observer<'a, 'control>(
        &mut self, _input: A::Input<'a>, _state: &mut S, _pass: ExpertPass,
        _demand: eredu_core::OutputDemand, _context: &<B::Tensor as Tensor>::Context,
        _paths: &'control crate::PreparedLayeredObservationPaths,
    ) -> Result<Self::Pass<'a, 'control>, crate::PreparedLayeredObservationError<A::Error>>
    where S: 'control {
        Err(crate::PreparedLayeredObservationError::BindingMismatch)
    }

    /// Per-invocation state with independent prepared-input and control loans.
    /// The state type need only outlive the short control loan; ordinary input
    /// may have a longer lifetime. The pass does not borrow the mutable state.
    type Pass<'input, 'control>
    where
        S: 'control;

    /// Borrows the actual current context, including an explicitly exchanged
    /// request loan. No context is reconstructed from rank or world metadata.
    fn current_parallel_context(&self) -> Option<&B::ParallelContext> { None }

    /// Exchanges an explicit context loan; success promises reversible,
    /// allocation-free restoration without a new source lookup.
    fn exchange_parallel_context(&mut self,_context:&mut B::ParallelContext)->bool { false }

    /// One scoped invocation may lend an already prepared backend context.
    /// Implementations accepting it must return the exact prior value and
    /// support restoration without allocation or a new source lookup.
    fn replace_parallel_context(&mut self,replacement:B::ParallelContext)
        ->Result<B::ParallelContext,B::ParallelContext>
    where B::ParallelContext:Sized { Err(replacement) }

    /// Starts one invocation without traversing unowned groups.
    fn begin<'a, 'control>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Pass<'a, 'control>, A::Error>
    where
        S: 'control;

    /// Reports request activity only for architecture-declared optional roots.
    fn request_group_active(
        &self,
        pass: &Self::Pass<'_, '_>,
        group: usize,
    ) -> Result<bool, A::Error>;

    /// Observes the existing schedule's disabled group without creating output.
    /// Ordinary executors need no retained dependency bookkeeping.
    fn retain_inactive_group(&mut self, _pass: &mut Self::Pass<'_, '_>, _group: usize) {}

    /// Whether inactive ranks submit collectives during pipeline-stage waves.
    ///
    /// A failed wave is communication-indeterminate and therefore permanently
    /// fences the selected communication authority on every rank after phase
    /// agreement. Ordinary pipeline execution leaves this disabled so a
    /// deterministic architecture failure remains retryable after rollback.
    fn has_cross_stage_collective_waves(&self) -> bool {
        false
    }

    /// Executes exactly one locally owned group through its validated partition driver.
    #[allow(clippy::too_many_arguments)]
    fn execute_group<'control, O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_, 'control>,
        driver: &LayeredPartitionDriver,
        state: &mut S,
        communication: &PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), A::Error>
    where S: 'control;

    /// Lends the enclosing request's exact control source to the selected
    /// group worker without changing its neural parallel context.
    #[allow(clippy::too_many_arguments)]
    fn execute_group_with_parallel_context<'control, O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self, pass: &mut Self::Pass<'_, 'control>, driver: &LayeredPartitionDriver,
        state: &mut S, communication: &PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor, context: &<B::Tensor as Tensor>::Context,
        _source_context: Option<&B::ParallelContext>, observer: &mut O,
    ) -> Result<(), A::Error>
    where S: 'control {
        self.execute_group(pass, driver, state, communication, communication_executor, context, observer)
    }

    /// Participates in one globally ordered pipeline-stage execution wave.
    ///
    /// Ordinary executors perform work only on the active stage. Executors
    /// with architecture-selected cross-stage collectives may use the wave
    /// ordinal on inactive stages to submit their exact zero-work protocol.
    /// The execution plan has already proved that an active rank owns a local
    /// driver, so absence of a driver is meaningful only for inactive ranks.
    #[allow(clippy::too_many_arguments)]
    fn execute_pipeline_wave<'control, O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_, 'control>,
        group: usize,
        driver: Option<&LayeredPartitionDriver>,
        active: bool,
        _wave: usize,
        state: &mut S,
        communication: &PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), A::Error>
    where S: 'control {
        if active {
            let driver = driver.expect("validated active pipeline rank owns a partition driver");
            assert_eq!(
                driver.group_index(),
                group,
                "active pipeline driver differs from the scheduled architecture group"
            );
            self.execute_group(
                pass,
                driver,
                state,
                communication,
                communication_executor,
                context,
                observer,
            )
        } else {
            Ok(())
        }
    }

    /// The same selected wave with an explicitly lent request control source.
    /// Default delegates to the existing wave, preserving custom wave semantics.
    #[allow(clippy::too_many_arguments)]
    fn execute_pipeline_wave_with_parallel_context<'control, O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self, pass: &mut Self::Pass<'_, 'control>, group: usize,
        driver: Option<&LayeredPartitionDriver>, active: bool, wave: usize,
        state: &mut S, communication: &PartitionCommunication<B, G, R, I>,
        communication_executor: &B::Executor, context: &<B::Tensor as Tensor>::Context,
        _source_context: Option<&B::ParallelContext>, observer: &mut O,
    ) -> Result<(), A::Error>
    where S: 'control {
        self.execute_pipeline_wave(pass, group, driver, active, wave, state,
            communication, communication_executor, context, observer)
    }

    /// Produces source tensors or destination placeholders for one endpoint route.
    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_, '_>,
        route: &PartitionBoundaryRoute,
        schema: &ResolvedBoundaryWireSchema,
        source: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<crate::ArchitectureBoundaryValue<B::Tensor>>, A::Error>;

    /// Resolves the exact invocation-dependent architecture boundary schema.
    fn boundary_schema(
        &self,
        pass: &Self::Pass<'_, '_>,
        route: &PartitionBoundaryRoute,
    ) -> Result<ResolvedBoundaryWireSchema, A::Error>;

    /// Installs a validated received typed-boundary bundle before its consumer runs.
    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_, '_>,
        route: &PartitionBoundaryRoute,
        values: Vec<B::Tensor>,
    ) -> Result<(), A::Error>;

    /// Returns projected output on the owner and matching destination storage elsewhere.
    ///
    /// The value is source data only on the publication root. On other ranks it is the
    /// architecture-selected destination/placeholder tensor validated before submission.
    fn finish(
        &mut self,
        pass: Self::Pass<'_, '_>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), A::Error>;

    /// Resolves an output-owner target capture or a matching rank-local placeholder.
    ///
    /// The default serves non-pipeline executors, where every participant owns
    /// the complete target result. Pipeline executors override this through
    /// their retained architecture and allocator.
    fn prediction_target_capture(
        &mut self,
        forward: &A::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, A::Error> {
        Ok(<A as crate::LayeredArchitecture<B, S>>::prediction_target_capture(forward).cloned())
    }

    /// Runs one typed prediction-only operation against this rank's target partition.
    fn apply_prediction_target_operation<O>(
        &mut self,
        _state: &mut S,
        _operation: O,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<O::Output>, A::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
    {
        Ok(None)
    }
}

/// Additive policy for an optional point-to-point boundary path.
pub trait PartitionBoundaryTransport<B, G, R, I>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Whether this policy admits a selected boundary route.
    const ENABLED: bool;

    /// Moves one exact architecture boundary.
    #[allow(clippy::too_many_arguments)]
    fn transfer(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        route: CommunicationRouteId,
        values: Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,
        schema: &ResolvedBoundaryWireSchema,
        wire: PipelineWireContract,
        executor: &B::Executor,
    ) -> Result<Vec<B::Tensor>, PartitionExecutionError>;
    /// Transfer through the same policy with the explicit selected source loan.
    /// Missing prepared policy support refuses instead of using ordinary work.
    #[allow(clippy::too_many_arguments)]
    fn transfer_with_source(&mut self,communication:&PartitionCommunication<B,G,R,I>,route:CommunicationRouteId,
        values:Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,schema:&ResolvedBoundaryWireSchema,
        wire:PipelineWireContract,executor:&B::Executor,source:Option<&crate::PreparedBoundarySource>,
        context:Option<&B::ParallelContext>)->Result<Vec<B::Tensor>,PartitionExecutionError>{
        let _=context;
        if let Some(source)=source{return Err(source.error(crate::PreparedBoundaryFrameCause::Contract).into());}
        self.transfer(communication,route,values,schema,wire,executor)
    }

}

/// Point-to-point boundary transport requiring no collective capabilities.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpaqueBoundaryTransport;

impl<B, G, R, I> PartitionBoundaryTransport<B, G, R, I> for OpaqueBoundaryTransport
where
    B: CommunicationBackend + PointToPointBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = true;

    fn transfer(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        route: CommunicationRouteId,
        values: Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,
        schema: &ResolvedBoundaryWireSchema,
        wire: PipelineWireContract,
        executor: &B::Executor,
    ) -> Result<Vec<B::Tensor>, PartitionExecutionError> {
        communication.transfer_boundary(route, values, schema, wire, executor)
    }
    fn transfer_with_source(&mut self,communication:&PartitionCommunication<B,G,R,I>,route:CommunicationRouteId,
        values:Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,schema:&ResolvedBoundaryWireSchema,
        wire:PipelineWireContract,executor:&B::Executor,source:Option<&crate::PreparedBoundarySource>,
        context:Option<&B::ParallelContext>)->Result<Vec<B::Tensor>,PartitionExecutionError>{
        match source {
            Some(source)=>communication.transfer_boundary_prepared(route,values,schema,wire,executor,source,
                context.ok_or_else(||source.error(crate::PreparedBoundaryFrameCause::Contract))?),
            None=>communication.transfer_boundary(route,values,schema,wire,executor),
        }
    }

}

/// Proof that this execution path selects no point-to-point boundary routes.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoBoundaryTransport;

impl<B, G, R, I> PartitionBoundaryTransport<B, G, R, I> for NoBoundaryTransport
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = false;

    fn transfer(
        &mut self,
        _communication: &PartitionCommunication<B, G, R, I>,
        _route: CommunicationRouteId,
        _values: Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,
        _schema: &ResolvedBoundaryWireSchema,
        _wire: PipelineWireContract,
        _executor: &B::Executor,
    ) -> Result<Vec<B::Tensor>, PartitionExecutionError> {
        Err(PartitionExecutionError::OperationPolicyUnavailable(
            CommunicationOperation::SendReceive,
        ))
    }
}

/// Additive policy for optional root-owned output publication.
pub trait PartitionOutputPublisher<B, G, R, I>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Whether this policy admits selected output publication.
    const ENABLED: bool;

    /// Publishes source data on root and destination storage on other members.
    fn publish(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        value: B::Tensor,
        publication: PartitionOutputPublication,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError>;
    /// Same policy with an explicit source-bound publication occurrence.
    fn publish_with_parallel(&mut self,communication:&PartitionCommunication<B,G,R,I>,
        value:B::Tensor,publication:PartitionOutputPublication,phase:DistributedExecutionPhase,
        executor:&B::Executor,prepared:Option<(&B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)
        ->Result<B::Tensor,PartitionExecutionError> {
        if prepared.is_some(){return Err(PartitionExecutionError::CommunicationPolicyMismatch);}
        self.publish(communication,value,publication,phase,executor)
    }
}

/// Broadcast output publication requiring no point-to-point or barrier capability.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpaqueOutputPublisher;

impl<B, G, R, I> PartitionOutputPublisher<B, G, R, I> for OpaqueOutputPublisher
where
    B: CommunicationBackend + BroadcastBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = true;

    fn publish(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        value: B::Tensor,
        publication: PartitionOutputPublication,
        phase: DistributedExecutionPhase,
        executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError> {
        communication.broadcast_output(value, publication, phase, executor)
    }
    fn publish_with_parallel(&mut self,communication:&PartitionCommunication<B,G,R,I>,
        value:B::Tensor,publication:PartitionOutputPublication,phase:DistributedExecutionPhase,
        executor:&B::Executor,prepared:Option<(&B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)
        ->Result<B::Tensor,PartitionExecutionError> {
        communication.broadcast_output_with_parallel(value,publication,phase,executor,prepared)
    }

}

/// Proof that this execution path selects no root-to-group publication.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOutputPublisher;

impl<B, G, R, I> PartitionOutputPublisher<B, G, R, I> for NoOutputPublisher
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = false;

    fn publish(
        &mut self,
        _communication: &PartitionCommunication<B, G, R, I>,
        _value: B::Tensor,
        _publication: PartitionOutputPublication,
        _phase: DistributedExecutionPhase,
        _executor: &B::Executor,
    ) -> Result<B::Tensor, PartitionExecutionError> {
        Err(PartitionExecutionError::OperationPolicyUnavailable(
            CommunicationOperation::Broadcast,
        ))
    }
}

/// Additive policy for optional distributed state-commit agreement.
pub trait PartitionCommitAgreement<B, G, R, I>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    /// Whether this policy admits distributed commit agreement.
    const ENABLED: bool;

    /// Whether this policy propagates an explicit success status at every
    /// canonical shared-session phase.
    const PHASE_FAILURE_AGREEMENT: bool = false;

    /// Operation performed by the final decision policy.
    const COMMIT_OPERATION: CommunicationOperation = CommunicationOperation::Barrier;

    /// Optional completed vote from an explicit retained request source.
    /// Custom policies opt in; absence keeps the ordinary prepared-group path.
    fn agree_phase_from_source(
        &mut self, _communication:&PartitionCommunication<B,G,R,I>,
        _group:CollectiveGroupId, _phase:DistributedExecutionPhase, _success:bool,
        _executor:&B::Executor, _source:&B::ParallelContext,
    )->Result<Option<bool>,PartitionExecutionError>{Ok(None)}

    /// Existing phase policy consuming an explicitly prepared native resource.
    /// An implementation must opt in before a prepared group can be submitted.
    fn agree_phase_with_group(
        &mut self, communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId, phase: DistributedExecutionPhase, local_success: bool,
        executor: &B::Executor, prepared: Option<&B::CommunicationGroup>,
    ) -> Result<bool, PartitionExecutionError> {
        if prepared.is_some() { return Err(PartitionExecutionError::CommunicationPolicyMismatch); }
        self.agree_phase(communication, group, phase, local_success, executor)
    }

    /// Bounded recovery agreement with the same explicit resource loan.
    fn agree_phase_after_prior_failure_with_group(
        &mut self, communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId, phase: DistributedExecutionPhase, local_success: bool,
        executor: &B::Executor, prepared: Option<&B::CommunicationGroup>,
    ) -> Result<bool, PartitionExecutionError> {
        if prepared.is_some() { return Err(PartitionExecutionError::CommunicationPolicyMismatch); }
        self.agree_phase_after_prior_failure(communication, group, phase, local_success, executor)
    }

    /// Final decision under the prepared group's original completion authority.
    fn commit_with_group(
        &mut self, communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId, epoch: DistributedCommitEpoch, executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<DistributedCommitOutcome, PartitionExecutionError> {
        if prepared.is_some() { return Err(PartitionExecutionError::CommunicationPolicyMismatch); }
        Ok(self.commit(communication, group, epoch, executor))
    }

    /// Returns the conjunction of every member's local phase status.
    ///
    /// Barrier-only policies intentionally retain the local status here; they
    /// do not claim failure propagation.
    fn agree_phase(
        &mut self,
        _communication: &PartitionCommunication<B, G, R, I>,
        _group: CollectiveGroupId,
        _phase: DistributedExecutionPhase,
        local_success: bool,
        _executor: &B::Executor,
    ) -> Result<bool, PartitionExecutionError> {
        Ok(local_success)
    }

    /// Performs the canonical all-rank agreement after an earlier subgroup
    /// communication failure poisoned the local session authority.
    ///
    /// Implementations may bypass the prior poison only for this bounded
    /// recovery agreement. The selected session remains poisoned afterwards.
    fn agree_phase_after_prior_failure(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        executor: &B::Executor,
    ) -> Result<bool, PartitionExecutionError> {
        self.agree_phase(communication, group, phase, local_success, executor)
    }

    /// Returns this rank's honest observation of the globally identified final decision.
    fn commit(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        executor: &B::Executor,
    ) -> DistributedCommitOutcome;
}

/// Barrier commit agreement requiring no tensor communication capabilities.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpaqueCommitAgreement;

impl<B, G, R, I> PartitionCommitAgreement<B, G, R, I> for OpaqueCommitAgreement
where
    B: CommunicationBackend + BarrierBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = true;

    fn commit(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        executor: &B::Executor,
    ) -> DistributedCommitOutcome {
        match communication.barrier(group, executor) {
            Ok(()) => DistributedCommitOutcome::Committed(epoch),
            Err(error) => indeterminate_commit(epoch, &error),
        }
    }

    fn commit_with_group(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<DistributedCommitOutcome, PartitionExecutionError> {
        Ok(match communication.barrier_with_group(group, executor, prepared) {
            Ok(()) => DistributedCommitOutcome::Committed(epoch),
            Err(error) => indeterminate_commit(epoch, &error),
        })
    }
}

/// Explicit all-rank failure propagation and final commit agreement.
///
/// This policy is intentionally separate from [`OpaqueCommitAgreement`]: a
/// barrier cannot reveal that another rank reported a local phase failure.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpaqueFailureAgreement;

impl<B, G, R, I> PartitionCommitAgreement<B, G, R, I> for OpaqueFailureAgreement
where
    B: CommunicationBackend + FailureAgreementBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = true;
    const PHASE_FAILURE_AGREEMENT: bool = true;
    const COMMIT_OPERATION: CommunicationOperation = CommunicationOperation::FailureAgreement;

    fn agree_phase(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        executor: &B::Executor,
    ) -> Result<bool, PartitionExecutionError> {
        communication.agree_success(local_success, group, phase, executor)
    }

    fn agree_phase_with_group(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<bool, PartitionExecutionError> {
        communication.agree_success_with_group(local_success, group, phase, executor, prepared)
    }

    fn agree_phase_from_source(&mut self,communication:&PartitionCommunication<B,G,R,I>,
        group:CollectiveGroupId,phase:DistributedExecutionPhase,success:bool,
        executor:&B::Executor,source:&B::ParallelContext)->Result<Option<bool>,PartitionExecutionError>{
        let (_,selected)=communication.group(group,CommunicationOperation::FailureAgreement)?;
        B::agree_success_from_source(success,selected,phase,executor,source).map_err(|error|
            communication.submission_error(error,CommunicationOperation::FailureAgreement,phase,None))
    }

    fn agree_phase_after_prior_failure(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        executor: &B::Executor,
    ) -> Result<bool, PartitionExecutionError> {
        communication.agree_success_after_prior_failure(local_success, group, phase, executor)
    }

    fn agree_phase_after_prior_failure_with_group(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<bool, PartitionExecutionError> {
        communication.agree_success_after_prior_failure_with_group(local_success, group, phase, executor, prepared)
    }

    fn commit(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        executor: &B::Executor,
    ) -> DistributedCommitOutcome {
        match communication.agree_success(true, group, DistributedExecutionPhase::Commit, executor)
        {
            Ok(true) => DistributedCommitOutcome::Committed(epoch),
            Ok(false) => DistributedCommitOutcome::Aborted(epoch),
            Err(error) => indeterminate_commit(epoch, &error),
        }
    }

    fn commit_with_group(
        &mut self,
        communication: &PartitionCommunication<B, G, R, I>,
        group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        executor: &B::Executor,
        prepared: Option<&B::CommunicationGroup>,
    ) -> Result<DistributedCommitOutcome, PartitionExecutionError> {
        Ok(match communication.agree_success_with_group(true, group, DistributedExecutionPhase::Commit, executor, prepared)
        {
            Ok(true) => DistributedCommitOutcome::Committed(epoch),
            Ok(false) => DistributedCommitOutcome::Aborted(epoch),
            Err(error) => indeterminate_commit(epoch, &error),
        })
    }
}

/// Proof that this execution path selects no distributed commit barrier.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCommitAgreement;

impl<B, G, R, I> PartitionCommitAgreement<B, G, R, I> for NoCommitAgreement
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    const ENABLED: bool = false;

    fn commit(
        &mut self,
        _communication: &PartitionCommunication<B, G, R, I>,
        _group: CollectiveGroupId,
        epoch: DistributedCommitEpoch,
        _executor: &B::Executor,
    ) -> DistributedCommitOutcome {
        DistributedCommitOutcome::Aborted(epoch)
    }
}

fn indeterminate_commit(
    epoch: DistributedCommitEpoch,
    error: &PartitionExecutionError,
) -> DistributedCommitOutcome {
    let phase = match error {
        PartitionExecutionError::CommunicationSubmissionFailed { .. } => {
            DistributedCommitPhase::DecisionSubmission
        }
        PartitionExecutionError::CommunicationCompletionFailed { .. }
        | PartitionExecutionError::CommunicationDeadlineExceeded { .. }
        | PartitionExecutionError::CommunicationPoisoned { .. } => {
            DistributedCommitPhase::DecisionCompletion
        }
        _ => DistributedCommitPhase::DecisionObservation,
    };
    DistributedCommitOutcome::Indeterminate { epoch, phase }
}

/// Complete rank-local runtime installed behind the shared replicated session.
pub struct PartitionedTextRuntime<A, B, S, P, E, G, R, I, T, U, V>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
{
    plan: PartitionPlanStorage,
    executor: E,
    communication: PartitionCommunication<B, G, R, I>,
    communication_executor: B::OwnedExecutor,
    parallel_control: Option<Box<B::ParallelContext>>,
    boundary_transport: T,
    output_publisher: U,
    commit_agreement: V,
    execution_rejection_agreed: bool,
    residency: ExecutionResidency,
    bounded_policy: Option<P>,
    marker: PhantomData<fn(A, S)>,
}

/// Failure while a fully local layered traversal runs through a selected
/// partitioned runtime.
///
/// This narrow path serves architectures whose public invocation contains
/// more than one tensor and whose architecture-owned traversal hook must run
/// between execution groups. Communication ownership and admission remain in
/// [`PartitionedTextRuntime`]; only the already selected local layered
/// traversal is delegated to its reusable executor.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedTraversalError<ArchitectureError, PolicyError>
where
    ArchitectureError: std::fmt::Display,
    PolicyError: std::fmt::Display,
{
    /// The selected partition plan is not a fully local traversal.
    #[error("partitioned traversal contract failed: {0}")]
    Contract(String),
    /// The architecture or residency policy rejected execution.
    #[error(transparent)]
    Execution(#[from] LayerwiseRuntimeError<ArchitectureError, PolicyError>),
}

/// Output of one architecture-owned traversal through a neutral partition runtime.
pub type PartitionedTraversalResult<Tensor, ForwardContext, ArchitectureError, PolicyError> =
    Result<(Tensor, ForwardContext), PartitionedTraversalError<ArchitectureError, PolicyError>>;

/// Reusable executor for one architecture-owned, fully local traversal inside
/// a neutral partitioned runtime.
///
/// The parallel context is an opaque backend mechanism selected from the
/// manifest. Family input, forward context, and traversal decisions remain
/// statically typed by the architecture.
pub struct LayerwiseTraversalPartitionExecutor<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    runtime: LayerwiseRuntime<A, B, S, P>,
    parallel: B::ParallelContext,
}

/// Neutral choice between an ordinary local traversal and the same traversal
/// installed in a selected partition runtime.
///
/// Concrete backends construct one of these alternatives but do not retain a
/// parallel execution enum or select a separate forward implementation.
pub enum LayerwiseTraversalRuntime<Direct, Partitioned> {
    /// Replicated local traversal.
    Direct(Direct),
    /// Architecture-selected partition traversal.
    Partitioned(Partitioned),
}

impl<Direct, Partitioned> LayerwiseTraversalRuntime<Direct, Partitioned> {
    /// Wraps an ordinary local layered runtime.
    pub const fn direct(runtime: Direct) -> Self {
        Self::Direct(runtime)
    }

    /// Wraps an architecture-selected partition runtime.
    pub const fn partitioned(runtime: Partitioned) -> Self {
        Self::Partitioned(runtime)
    }
}

impl<A, B, S, P> LayerwiseTraversalPartitionExecutor<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    B::ParallelContext: Sized,
{
    /// Pairs one local layered runtime with its selected opaque parallel context.
    pub const fn new(runtime: LayerwiseRuntime<A, B, S, P>, parallel: B::ParallelContext) -> Self {
        Self { runtime, parallel }
    }

    /// Borrows the original parallel context selected with this local executor.
    pub const fn parallel_context(&self)->&B::ParallelContext {&self.parallel}

    /// Borrows the installed layered runtime for residency reporting.
    pub const fn runtime(&self) -> &LayerwiseRuntime<A, B, S, P> {
        &self.runtime
    }
}

impl<A, B, S, P, E, G, R, I, T, U, V> PartitionedTextRuntime<A, B, S, P, E, G, R, I, T, U, V>
where
    B: CommunicationBackend,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
{
    /// Borrows the actual retained communication executor. This performs no
    /// initialization, cloning, submission, waiting, or authority selection.
    pub fn communication_executor(&self) -> &B::Executor { self.communication_executor.borrow() }

    /// Pairs one immutable plan with its rank-local executable and opaque resources.
    #[allow(
        clippy::too_many_arguments,
        reason = "constructor pairs independently selected runtime and communication policies"
    )]
    pub fn new(
        plan: PartitionedExecutionPlan,
        executor: E,
        communication: PartitionCommunication<B, G, R, I>,
        communication_executor: B::OwnedExecutor,
        boundary_transport: T,
        output_publisher: U,
        commit_agreement: V,
        residency: ExecutionResidency,
        bounded_policy: Option<P>,
    ) -> Result<Self, PartitionExecutionError> {
        Self::with_plan_storage(PartitionPlanStorage::Owned(plan), executor, communication,
            communication_executor, boundary_transport, output_publisher, commit_agreement,
            residency, bounded_policy)
    }

    /// Reuses the exact immutable plan retained by a completed initial
    /// construction. This copies no declarations and grants no communication
    /// authority: the same manifest and selected-policy checks still run.
    /// The caller must fund the shared alias and concrete runtime destination.
    #[allow(clippy::too_many_arguments)]
    pub fn with_retained_plan(
        plan: std::sync::Arc<PartitionedExecutionPlan>,
        executor: E,
        communication: PartitionCommunication<B, G, R, I>,
        communication_executor: B::OwnedExecutor,
        boundary_transport: T,
        output_publisher: U,
        commit_agreement: V,
        residency: ExecutionResidency,
        bounded_policy: Option<P>,
    ) -> Result<Self, PartitionExecutionError> {
        Self::with_plan_storage(PartitionPlanStorage::Retained(plan), executor, communication,
            communication_executor, boundary_transport, output_publisher, commit_agreement,
            residency, bounded_policy)
    }

    #[allow(clippy::too_many_arguments)]
    fn with_plan_storage(
        plan: PartitionPlanStorage,
        executor: E,
        communication: PartitionCommunication<B, G, R, I>,
        communication_executor: B::OwnedExecutor,
        boundary_transport: T,
        output_publisher: U,
        commit_agreement: V,
        residency: ExecutionResidency,
        bounded_policy: Option<P>,
    ) -> Result<Self, PartitionExecutionError> {
        plan.validate_manifest(communication.manifest())?;
        if V::PHASE_FAILURE_AGREEMENT {
            plan.validate_exact_commit_agreement(communication.manifest())?;
        }
        let bounded = !matches!(residency, ExecutionResidency::FullyResident);
        if bounded != bounded_policy.is_some() {
            return Err(PartitionExecutionError::ResidencyPolicyMismatch);
        }
        if (!T::ENABLED && !plan.routes.is_empty())
            || (!U::ENABLED && plan.publication.is_some())
            || (!V::ENABLED && plan.commit_barrier.is_some())
            || (V::PHASE_FAILURE_AGREEMENT
                != plan.selects_phase_failure_agreement(communication.manifest()))
        {
            return Err(PartitionExecutionError::CommunicationPolicyMismatch);
        }
        Ok(Self {
            plan,
            executor,
            communication,
            communication_executor,
            parallel_control: None,
            boundary_transport,
            output_publisher,
            commit_agreement,
            execution_rejection_agreed: false,
            residency,
            bounded_policy,
            marker: PhantomData,
        })
    }

    fn agree_phase(
        &mut self,
        phase: DistributedExecutionPhase,
        local_success: bool,
    ) -> Result<bool, PartitionExecutionError> {
        let Some(group) = self.plan.commit_barrier else {
            return Ok(local_success);
        };
        self.commit_agreement.agree_phase(
            &self.communication,
            group,
            phase,
            local_success,
            self.communication_executor.borrow(),
        )
    }
}

impl<A, B, S, P, ExecutionPolicy, G, R, I, T, U, V>
    PartitionedTextRuntime<
        A,
        B,
        S,
        P,
        LayerwiseTraversalPartitionExecutor<A, B, S, ExecutionPolicy>,
        G,
        R,
        I,
        T,
        U,
        V,
    >
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: ParallelLayeredArchitecture<B, S>,
    ExecutionPolicy: LayerwisePolicy<B, A::Unit>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
    B::ParallelContext: Sized,
    A::Error: std::fmt::Display,
    ExecutionPolicy::Error: std::fmt::Display,
{
    /// Runs one architecture-owned traversal through the selected neutral
    /// partitioned base.
    ///
    /// Every architecture group must be local. Point-to-point routes, output
    /// publication, and phase agreement use the ordinary partitioned session
    /// entry instead and are rejected here rather than silently bypassed.
    pub fn forward_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> PartitionedTraversalResult<B::Tensor, A::ForwardContext, A::Error, ExecutionPolicy::Error>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_traversal_hook_with_readout(
            input,
            state,
            context,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same selected local traversal with explicit public demand.
    pub fn forward_with_traversal_hook_with_readout<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> PartitionedTraversalResult<
        Option<B::Tensor>,
        A::ForwardContext,
        A::Error,
        ExecutionPolicy::Error,
    >
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_traversal_hook_with_readout_in(input,state,context,hook,demand,None)
    }

    /// Lends an explicitly prepared context to the unchanged local traversal.
    /// The backend authenticates its native identity before this neutral call.
    pub fn forward_with_traversal_hook_with_readout_in<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
        parallel: Option<&B::ParallelContext>,
    ) -> PartitionedTraversalResult<
        Option<B::Tensor>,
        A::ForwardContext,
        A::Error,
        ExecutionPolicy::Error,
    >
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        if !self.is_fully_local_traversal() {
            return Err(PartitionedTraversalError::Contract(
                "fully local traversal requires every group locally owned and no route, publication, or agreement policy"
                    .into(),
            ));
        }
        self.executor
            .runtime
            .forward_parallel_with_traversal_hook_with_readout(
                input,
                state,
                parallel.unwrap_or(&self.executor.parallel),
                context,
                hook,
                demand,
            )
            .map_err(PartitionedTraversalError::Execution)
    }

    /// Allocation-free source predicate shared by ordinary fully local execution
    /// and its cold projection. This does not grant communication authority.
    pub fn is_fully_local_traversal(&self)->bool {
        self.plan.routes.is_empty() && self.plan.publication.is_none()
            && self.plan.commit_barrier.is_none() && self.plan.drivers.iter().all(Option::is_some)
    }

    /// Borrows the installed traversal executor for neutral residency reports.
    pub const fn traversal_executor(
        &self,
    ) -> &LayerwiseTraversalPartitionExecutor<A, B, S, ExecutionPolicy> {
        &self.executor
    }
}

impl<A, B, S, ExecutionPolicy, G, R, I, T, U, V>
    LayerwiseTraversalRuntime<
        LayerwiseRuntime<A, B, S, ExecutionPolicy>,
        Box<
            PartitionedTextRuntime<
                A,
                B,
                S,
                (),
                LayerwiseTraversalPartitionExecutor<A, B, S, ExecutionPolicy>,
                G,
                R,
                I,
                T,
                U,
                V,
            >,
        >,
    >
where
    B: CommunicationBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: ParallelLayeredArchitecture<B, S>,
    ExecutionPolicy: LayerwisePolicy<B, A::Unit>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
    B::ParallelContext: Sized,
    A::Error: std::fmt::Display,
    ExecutionPolicy::Error: std::fmt::Display,
{
    /// Runs the selected local traversal without exposing its realization kind
    /// to concrete composition.
    pub fn forward_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> PartitionedTraversalResult<B::Tensor, A::ForwardContext, A::Error, ExecutionPolicy::Error>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_traversal_hook_with_readout(
            input,
            state,
            context,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same selected local traversal with explicit public demand.
    pub fn forward_with_traversal_hook_with_readout<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> PartitionedTraversalResult<
        Option<B::Tensor>,
        A::ForwardContext,
        A::Error,
        ExecutionPolicy::Error,
    >
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_traversal_hook_with_readout_in(input,state,context,hook,demand,None)
    }

    /// Lends an explicitly prepared context to the unchanged local traversal.
    /// The backend authenticates its native identity before this neutral call.
    pub fn forward_with_traversal_hook_with_readout_in<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
        parallel: Option<&B::ParallelContext>,
    ) -> PartitionedTraversalResult<
        Option<B::Tensor>,
        A::ForwardContext,
        A::Error,
        ExecutionPolicy::Error,
    >
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        match self {
            Self::Direct(_) if parallel.is_some()=>Err(PartitionedTraversalError::Contract(
                "a direct local traversal has no selected parallel context".into())),
            Self::Direct(runtime) => runtime
                .forward_with_traversal_hook_with_readout(input, state, context, hook, demand)
                .map_err(PartitionedTraversalError::Execution),
            Self::Partitioned(runtime) => runtime
                .forward_with_traversal_hook_with_readout_in(input, state, context, hook, demand,parallel),
        }
    }

    /// Borrows the selected residency policy independently of realization kind.
    pub fn policy(&self) -> &ExecutionPolicy {
        match self {
            Self::Direct(runtime) => runtime.policy(),
            Self::Partitioned(runtime) => runtime.traversal_executor().runtime().policy(),
        }
    }

    /// Borrows the selected local architecture independently of realization kind.
    pub fn architecture(&self) -> &A {
        match self {
            Self::Direct(runtime) => runtime.architecture(),
            Self::Partitioned(runtime) => runtime.traversal_executor().runtime().architecture(),
        }
    }
}

/// Stateless strategy marker selecting [`PartitionedTextRuntime`] in the shared session.
#[allow(
    clippy::type_complexity,
    reason = "marker preserves static dispatch across independent additive policies"
)]
pub struct PartitionedTextExecution<E, G, R, I, T, U, V>(
    PhantomData<fn() -> (E, G, R, I, T, U, V)>,
);

impl<E, G, R, I, T, U, V> PartitionedTextExecution<E, G, R, I, T, U, V> {
    /// Creates the statically dispatched partition strategy.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<E, G, R, I, T, U, V> Default for PartitionedTextExecution<E, G, R, I, T, U, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A, B, S, Bounded, E, G, R, I, T, U, V>
    ReplicatedTextExecutionStrategy<A, B, S, E::Policy, Bounded>
    for PartitionedTextExecution<E, G, R, I, T, U, V>
where
    B: CommunicationBackend + crate::TerminalCommunicationBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    Bounded: LayerwisePolicy<B, A::Unit, Error = <E::Policy as LayerwisePolicy<B, A::Unit>>::Error>,
    E: PartitionedGroupExecutor<A, B, S, G, R, I>
        + crate::parameter_operations::LayeredParameterOwner<B, S, Architecture = A>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
    A::Error: std::fmt::Display,
    <E::Policy as LayerwisePolicy<B, A::Unit>>::Error: std::fmt::Display,
{
    fn mark_terminal_failure(runtime: &Self::Runtime, _phase: DistributedExecutionPhase) {
        runtime.communication.mark_terminal_failure();
    }

    const PARTITIONED_SESSION: bool = true;
    const DISTRIBUTED_PHASE_AGREEMENT: bool = V::PHASE_FAILURE_AGREEMENT;

    type Runtime = PartitionedTextRuntime<A, B, S, Bounded, E, G, R, I, T, U, V>;

    fn observation_hooks(runtime: &Self::Runtime) -> crate::inspection::ObservationHookSupport {
        runtime.executor.observation_hooks().with_publication(true)
    }
    fn prepare_observation_paths(runtime: &Self::Runtime)
        -> Result<Option<crate::PreparedLayeredObservationPaths>,
            ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>> {
        runtime.executor.prepare_observation_paths().map_err(|cause|
            crate::replicated_session::observation_paths::map_prepared(cause,
                ReplicatedTextSessionError::Architecture))
    }

    fn bind_observation_paths(runtime: &Self::Runtime, source: &crate::SharedLayeredObservationPaths, metadata: Option<crate::layered::LayeredMetadata<A::Error>>)
        -> Result<crate::PreparedLayeredObservationPaths,
            ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>> {
        runtime.executor.bind_observation_paths(source, metadata).map_err(|cause|
            crate::replicated_session::observation_paths::map_prepared(cause,
                ReplicatedTextSessionError::Architecture))
    }

    fn validate_observation_paths(runtime: &Self::Runtime, paths: &crate::PreparedLayeredObservationPaths)
        -> Result<(), ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>> {
        runtime.executor.validate_observation_paths(paths).map_err(|cause|
            crate::replicated_session::observation_paths::map_prepared(cause, |never| match never {}))
    }

    fn forward_with_prepared_observer<'a, O>(
        &mut self, runtime: &mut Self::Runtime, input: A::Input<'a>, state: &mut S,
        pass_kind: ExpertPass, context: &<B::Tensor as Tensor>::Context,
        observer: &mut O, paths: &crate::PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>>
    where O: ActivationObserver<B::Tensor, A::Error> + ?Sized {
        runtime.execution_rejection_agreed = false;
        let pass = runtime.executor.begin_with_prepared_observer(
            input, state, pass_kind, demand, context, paths,
        ).map_err(|cause| crate::replicated_session::observation_paths::map_prepared(cause,
            ReplicatedTextSessionError::Architecture))?;
        run_partition_pass::<A, B, S, E::Policy, Bounded, E, G, R, I, T, U, V, O>(
            runtime, pass, state, context, observer,
        )
    }

    fn exchange_parallel_control_context(
        runtime: &mut Self::Runtime, context: &mut Option<Box<B::ParallelContext>>,
    ) -> bool {
        std::mem::swap(&mut runtime.parallel_control, context);
        true
    }

    fn exchange_parallel_context(runtime:&mut Self::Runtime,context:&mut B::ParallelContext)->bool {
        runtime.executor.exchange_parallel_context(context)
    }
    fn replace_parallel_context(runtime:&mut Self::Runtime,replacement:B::ParallelContext)
        ->Result<B::ParallelContext,B::ParallelContext>
    where B::ParallelContext:Sized {
        runtime.executor.replace_parallel_context(replacement)
    }


    fn visit_loaded_parameters(
        runtime: &mut Self::Runtime,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        runtime.executor.visit_loaded_parameters(visitor)
    }

    fn static_modules_ref(runtime: &Self::Runtime) -> Option<&A::StaticModules> {
        let (architecture, _) = runtime.executor.parameter_parts_ref()?;
        Some(architecture.static_modules())
    }

    fn visit_parameter_sources<Visitor>(
        runtime: &Self::Runtime,
        visitor: &mut Visitor,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error>
    where
        Visitor: for<'source> eredu_nn::ParameterSourceVisitor<'source, B::Tensor>,
    {
        context.charge_metadata(std::mem::size_of::<(
            &Self::Runtime,
            &mut Visitor,
            &eredu_nn::workspace::WorkspaceContext,
            Option<(&A, &E::Policy)>,
            Result<bool, eredu_nn::Error>,
        )>())?;
        let Some((architecture, policy)) = runtime.executor.parameter_parts_ref() else {
            return Ok(false);
        };
        crate::parameter_operations::visit_parameter_sources_in_parts::<
            B,
            A::Unit,
            E::Policy,
            Visitor,
        >(architecture.static_modules(), policy, visitor, context)
    }

    fn visit_retained_values(runtime: &Self::Runtime, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        runtime.executor.visit_retained_values(visitor)
    }

    fn with_parameter_slots(
        runtime: &mut Self::Runtime,
        location: &crate::parameter_operations::PreparedParameterLocation,
        operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            <E::Policy as LayerwisePolicy<B, A::Unit>>::Error,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, crate::LayerwiseAcquireError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error>> {
        runtime
            .executor
            .with_parameter_slots(location, operation, context)
    }

    fn publish_parameter_replacements(
        runtime: &mut Self::Runtime,
        values: &std::collections::BTreeMap<String, B::Tensor>,
        active: bool,
    ) -> Result<bool, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error> {
        runtime
            .executor
            .publish_parameter_replacements(values, active)
    }

    fn resident_policy(runtime: &Self::Runtime) -> Option<&E::Policy> {
        if !matches!(runtime.residency, ExecutionResidency::FullyResident) {
            return None;
        }
        // The executor owns the selected policy. Lending it here lets cold
        // inspection and original operation installation use that same owner;
        // a bounded policy is never promoted to resident by its concrete type.
        runtime.executor.parameter_parts_ref().map(|(_, policy)| policy)
    }

    fn group_submission_mechanism(runtime: &Self::Runtime) -> crate::GroupSubmissionMechanism {
        runtime.executor.group_submission_mechanism()
    }

    fn bounded_policy(runtime: &Self::Runtime) -> Option<&Bounded> {
        runtime.bounded_policy.as_ref()
    }

    fn execution_residency(
        runtime: &Self::Runtime,
        _selected: &crate::SelectedReplicatedTextRealization,
    ) -> ExecutionResidency {
        runtime.residency
    }

    fn forward_with_observer<'a, O>(
        &mut self,
        runtime: &mut Self::Runtime,
        input: A::Input<'a>,
        state: &mut S,
        pass_kind: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.execution_rejection_agreed = false;
        let pass = runtime
            .executor
            .begin(input, state, pass_kind, demand, context)
            .map_err(ReplicatedTextSessionError::Architecture)?;
        run_partition_pass::<A, B, S, E::Policy, Bounded, E, G, R, I, T, U, V, O>(
            runtime, pass, state, context, observer,
        )
    }

    fn observe_output<O>(
        runtime: &mut Self::Runtime,
        output: &B::Tensor,
        observer: &mut O,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        B::Tensor,
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        if let Some(publication) = runtime.plan.publication {
            let rank = runtime.communication.manifest().rank();
            let local_output = runtime
                .plan
                .drivers
                .iter()
                .flatten()
                .any(LayeredPartitionDriver::owns_output);
            if rank == publication.owner_rank && !local_output {
                return Err(ReplicatedTextSessionError::Contract(
                    PartitionExecutionError::OutputOwnership {
                        rank,
                        owner: publication.owner_rank,
                    }
                    .to_string(),
                ));
            }
            if rank != publication.owner_rank {
                if local_output {
                    observer
                        .observe_replica(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, output)
                        .map_err(ReplicatedTextSessionError::Architecture)?;
                }
                return Ok(output.clone());
            }
        }
        crate::observe_model_logits(observer, output)
            .map_err(ReplicatedTextSessionError::Architecture)
    }

    fn publish_observed_output(
        runtime: &mut Self::Runtime,
        output: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        B::Tensor,
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    > {
        match runtime.plan.publication {
            Some(publication) => runtime
                .output_publisher
                .publish(
                    &runtime.communication,
                    output,
                    publication,
                    DistributedExecutionPhase::OutputPublication,
                    runtime.communication_executor.borrow(),
                )
                .map_err(ReplicatedTextSessionError::Partition),
            None => Ok(output),
        }
    }

    fn publish_observed_output_with_parallel(runtime:&mut Self::Runtime,output:B::Tensor,
        _context:&<B::Tensor as Tensor>::Context,
        prepared:Option<(&B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)
        ->Result<B::Tensor,ReplicatedTextSessionError<A::Error,<E::Policy as LayerwisePolicy<B, A::Unit>>::Error,std::convert::Infallible>> {
        match runtime.plan.publication {
            Some(publication)=>runtime.output_publisher.publish_with_parallel(&runtime.communication,output,
                publication,DistributedExecutionPhase::OutputPublication,runtime.communication_executor.borrow(),prepared)
                .map_err(ReplicatedTextSessionError::Partition),
            None if prepared.is_some()=>Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::CommunicationPolicyMismatch)),
            None=>Ok(output),
        }
    }

    fn prediction_target_capture(
        runtime: &mut Self::Runtime,
        forward: &A::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<B::Tensor>,
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    > {
        runtime
            .executor
            .prediction_target_capture(forward, context)
            .map_err(ReplicatedTextSessionError::Architecture)
    }

    fn publish_prediction_target_capture(
        runtime: &mut Self::Runtime,
        capture: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        B::Tensor,
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    > {
        match runtime.plan.publication {
            Some(publication) => runtime
                .output_publisher
                .publish(
                    &runtime.communication,
                    capture,
                    publication,
                    DistributedExecutionPhase::PredictionTargetCapture,
                    runtime.communication_executor.borrow(),
                )
                .map_err(ReplicatedTextSessionError::Partition),
            None => Ok(capture),
        }
    }

    fn apply_prediction_target_operation<O>(
        runtime: &mut Self::Runtime,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<O::Output>,
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>,
    >
    where
        O: crate::PredictionTargetOperation<A, B, S>,
    {
        runtime
            .executor
            .apply_prediction_target_operation(state, operation, context)
            .map_err(ReplicatedTextSessionError::Architecture)
    }

    fn parallel_control_operation(
        runtime: &Self::Runtime, event: crate::replicated_session::ParallelControlEvent,
    ) -> Option<CommunicationOperation> {
        runtime.plan.commit_barrier?;
        match event {
            crate::replicated_session::ParallelControlEvent::Phase(_) if V::PHASE_FAILURE_AGREEMENT =>
                Some(CommunicationOperation::FailureAgreement),
            crate::replicated_session::ParallelControlEvent::Commit => Some(V::COMMIT_OPERATION),
            _ => None,
        }
    }

    fn commit_after_completion(
        runtime: &mut Self::Runtime, epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> DistributedCommitOutcome {
        match <Self as ReplicatedTextExecutionStrategy<A, B, S, E::Policy, Bounded>>::commit_after_completion_with_parallel(runtime, epoch, context, None) {
            Ok(outcome) => outcome,
            Err(ReplicatedTextSessionError::Partition(error)) => indeterminate_commit(epoch, &error),
            Err(_) => DistributedCommitOutcome::Indeterminate {
                epoch, phase: DistributedCommitPhase::DecisionSubmission,
            },
        }
    }

    fn commit_after_completion_with_parallel(
        runtime: &mut Self::Runtime, epoch: DistributedCommitEpoch,
        _context: &<B::Tensor as Tensor>::Context,
        prepared: Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>,
    ) -> Result<DistributedCommitOutcome,
        ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>>
    {
        let Some(group) = runtime.plan.commit_barrier else {
            return Ok(DistributedCommitOutcome::Committed(epoch));
        };
        let communication = &runtime.communication;
        let executor = runtime.communication_executor.borrow();
        let policy = &mut runtime.commit_agreement;
        communication.with_control_group(group, V::COMMIT_OPERATION,
            crate::replicated_session::ParallelControlEvent::Commit, prepared, executor,
            |bound| policy.commit_with_group(communication, group, epoch, executor, bound),
        ).map_err(ReplicatedTextSessionError::Partition)
    }

    fn agree_distributed_phase(
        runtime: &mut Self::Runtime, phase: DistributedExecutionPhase, local_success: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>>
    {
        <Self as ReplicatedTextExecutionStrategy<A, B, S, E::Policy, Bounded>>::agree_distributed_phase_with_parallel(runtime, phase, local_success, context, None)
    }

    fn agree_distributed_phase_with_parallel(
        runtime: &mut Self::Runtime,
        phase: DistributedExecutionPhase,
        local_success: bool,
        _context: &<B::Tensor as Tensor>::Context,
        prepared: Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, <E::Policy as LayerwisePolicy<B, A::Unit>>::Error, std::convert::Infallible>>
    {
        let cross_stage_failure = phase == DistributedExecutionPhase::Execution
            && runtime.executor.has_cross_stage_collective_waves();
        let needs_recovery_agreement = cross_stage_failure
            && !runtime.execution_rejection_agreed
            && (!local_success || runtime.communication.authority.is_poisoned());
        let agreed = if let Some(group) = runtime.plan.commit_barrier {
            let communication = &runtime.communication;
            let executor = runtime.communication_executor.borrow();
            let policy = &mut runtime.commit_agreement;
            if !V::PHASE_FAILURE_AGREEMENT {
                // Barrier-only policy declares no phase collective to bind.
                policy.agree_phase(communication, group, phase, local_success, executor)
            } else {
                communication.with_control_group(group, CommunicationOperation::FailureAgreement,
                    crate::replicated_session::ParallelControlEvent::Phase(phase),
                    prepared, executor, |bound| {
                        if needs_recovery_agreement {
                            policy.agree_phase_after_prior_failure_with_group(
                                communication, group, phase, local_success, executor, bound)
                        } else {
                            policy.agree_phase_with_group(
                                communication, group, phase, local_success, executor, bound)
                        }
                    })
            }
        } else { Ok(local_success) }
        .map_err(ReplicatedTextSessionError::Partition)?;
        if cross_stage_failure && !agreed && !runtime.execution_rejection_agreed {
            let _ = runtime.communication.authority.submission_error(
                "routed pipeline collective wave failed on at least one rank",
                CommunicationOperation::AllGatherEven,
                DistributedExecutionPhase::Execution,
                None,
            );
        }
        Ok(agreed)
    }
}

enum PartitionRouteTransferError<E> {
    Architecture(E),
    Contract(PartitionExecutionError),
}

struct PreparedPartitionBoundary<T> {
    source: bool,
    schema: ResolvedBoundaryWireSchema,
    values: Vec<crate::ArchitectureBoundaryValue<T>>,
    source_context: Option<crate::PreparedBoundarySource>,
}

fn map_partition_route_error<E: std::fmt::Display, P: std::fmt::Display>(
    error: PartitionRouteTransferError<E>,
) -> ReplicatedTextSessionError<E, P, std::convert::Infallible> {
    match error {
        PartitionRouteTransferError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        PartitionRouteTransferError::Contract(error) => {
            ReplicatedTextSessionError::Partition(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_partition_boundary<'a, 'control, A, B, S, E, G, R, I>(
    executor: &mut E,
    control: Option<&B::ParallelContext>,
    pass: &mut E::Pass<'a, 'control>,
    communication: &PartitionCommunication<B, G, R, I>,
    route: &PartitionBoundaryRoute,
    wire: PipelineWireContract,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedPartitionBoundary<B::Tensor>, PartitionRouteTransferError<A::Error>>
where
    B: CommunicationBackend,
    S: RuntimeState<B> + 'control,
    A: LayeredArchitecture<B, S>,
    E: PartitionedGroupExecutor<A, B, S, G, R, I>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    let source_context=match control.or_else(||executor.current_parallel_context()) {
        Some(context)=>{
            let (_,native)=communication.route(route.route).map_err(PartitionRouteTransferError::Contract)?;
            B::prepare_boundary_source(context,native)
                .map_err(|cause|PartitionRouteTransferError::Contract(PartitionExecutionError::BoundarySource(cause)))?
        }
        None=>None,
    };
    let source = communication
        .boundary_endpoint_is_source(route.route)
        .map_err(PartitionRouteTransferError::Contract)?;
    let schema = executor
        .boundary_schema(pass, route)
        .map_err(PartitionRouteTransferError::Architecture)?;
    let values = executor
        .boundary_values(pass, route, &schema, source, context)
        .map_err(PartitionRouteTransferError::Architecture)?;
    match &source_context {
        Some(source)=>boundary::Controls::<B::CommunicationError>::new(source).and_then(|controls|
            controls.validate_tagged::<B,I>(&communication.inspector,&values,&schema,wire)),
        None=>communication.validate_prepared_boundary(route.route,&values,&schema,wire),
    }.map_err(PartitionRouteTransferError::Contract)?;
    Ok(PreparedPartitionBoundary {
        source,
        schema,
        values,
        source_context,
    })
}

#[allow(clippy::too_many_arguments)]
fn transfer_prepared_partition_boundary<'a, 'control, A, B, S, E, G, R, I, T>(
    executor: &mut E,
    control: Option<&B::ParallelContext>,
    pass: &mut E::Pass<'a, 'control>,
    communication: &PartitionCommunication<B, G, R, I>,
    transport: &mut T,
    route: &PartitionBoundaryRoute,
    wire: PipelineWireContract,
    communication_executor: &B::Executor,
    prepared: PreparedPartitionBoundary<B::Tensor>,
) -> Result<(), PartitionRouteTransferError<A::Error>>
where
    B: CommunicationBackend,
    S: RuntimeState<B> + 'control,
    A: LayeredArchitecture<B, S>,
    E: PartitionedGroupExecutor<A, B, S, G, R, I>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
{
    let values = transport.transfer_with_source(
        communication,route.route,prepared.values,&prepared.schema,wire,
        communication_executor,prepared.source_context.as_ref(),
        control.or_else(||executor.current_parallel_context()),
    ).map_err(PartitionRouteTransferError::Contract)?;
    if !prepared.source {
        executor
            .accept_boundary(pass, route, values)
            .map_err(PartitionRouteTransferError::Architecture)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_and_transfer_partition_boundary<'a, 'control, A, B, S, E, G, R, I, T>(
    executor: &mut E,
    control: Option<&B::ParallelContext>,
    pass: &mut E::Pass<'a, 'control>,
    communication: &PartitionCommunication<B, G, R, I>,
    transport: &mut T,
    route: &PartitionBoundaryRoute,
    wire: PipelineWireContract,
    communication_executor: &B::Executor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), PartitionRouteTransferError<A::Error>>
where
    B: CommunicationBackend,
    S: RuntimeState<B> + 'control,
    A: LayeredArchitecture<B, S>,
    E: PartitionedGroupExecutor<A, B, S, G, R, I>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
{
    let prepared = prepare_partition_boundary::<A, B, S, E, G, R, I>(
        executor,
        control,
        pass,
        communication,
        route,
        wire,
        context,
    )?;
    if prepared.source {
        communication
            .complete_local_dependencies(
                &prepared.values,
                route.route,
                communication_executor,
                false,
                prepared.source_context.as_ref(),
                control.or_else(||executor.current_parallel_context()),
            )
            .map_err(PartitionRouteTransferError::Contract)?;
    }
    transfer_prepared_partition_boundary::<A, B, S, E, G, R, I, T>(
        executor,
        control,
        pass,
        communication,
        transport,
        route,
        wire,
        communication_executor,
        prepared,
    )
}

fn validate_bundle_requirement<B, I>(
    inspector: &I,
    values: &[B::Tensor],
    requirement: &CommunicationOperationRequirement,
    completed: bool,
) -> Result<(), PartitionExecutionError>
where
    B: NeuralBackend,
    I: CommunicationTensorMetadata<B>,
{
    let limits = requirement
        .limits()
        .ok_or(PartitionExecutionError::MissingTensorLimits)?;
    if values.len() > limits.max_tensors() {
        return Err(PartitionExecutionError::TensorCount {
            expected_at_most: limits.max_tensors(),
            actual: values.len(),
        });
    }
    for value in values {
        let dtype = inspector.dtype(value);
        let shape = inspector.shape(value);
        let elements = shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension));
        if !requirement.dtypes().contains(&dtype) {
            return Err(PartitionExecutionError::TensorDtype { dtype });
        }
        let maximum = if completed {
            limits.max_output_tensor_elements()
        } else {
            limits.max_tensor_elements()
        };
        if shape.len() > limits.max_tensor_rank()
            || elements.is_none_or(|elements| elements > maximum)
        {
            return Err(PartitionExecutionError::TensorLimits { shape });
        }
    }
    Ok(())
}

fn validate_boundary_bundle<B, I>(
    inspector: &I,
    values: &[B::Tensor],
    schema: &ResolvedBoundaryWireSchema,
    wire: PipelineWireContract,
) -> Result<(), PartitionExecutionError>
where
    B: NeuralBackend,
    I: CommunicationTensorMetadata<B>,
{
    let specs = std::iter::once(schema.primary()).chain(schema.auxiliary());
    let expected = 1 + schema.auxiliary().len();
    if values.len() != expected {
        return Err(PartitionExecutionError::BoundaryTensorCount {
            boundary: schema.identity(),
            expected,
            actual: values.len(),
        });
    }
    for (value, spec) in values.iter().zip(specs) {
        validate_boundary_tensor::<B, I>(inspector, value, spec, wire)?;
    }
    Ok(())
}

fn validate_tagged_boundary_bundle<B, I>(
    inspector: &I,
    values: &[crate::ArchitectureBoundaryValue<B::Tensor>],
    schema: &ResolvedBoundaryWireSchema,
    wire: PipelineWireContract,
) -> Result<(), PartitionExecutionError>
where
    B: NeuralBackend,
    I: CommunicationTensorMetadata<B>,
{
    let specs = std::iter::once(schema.primary()).chain(schema.auxiliary());
    let expected = 1 + schema.auxiliary().len();
    if values.len() != expected {
        return Err(PartitionExecutionError::BoundaryTensorCount {
            boundary: schema.identity(),
            expected,
            actual: values.len(),
        });
    }
    for (value, spec) in values.iter().zip(specs) {
        if value.role() != spec.role() {
            return Err(PartitionExecutionError::BoundaryFraming(format!(
                "architecture boundary role {:?} differs from selected role {:?}",
                value.role(),
                spec.role(),
            )));
        }
        validate_boundary_tensor::<B, I>(inspector, value.tensor(), spec, wire)?;
    }
    Ok(())
}

fn boundary_dtype(spec:&ResolvedBoundaryTensorSpec,wire:PipelineWireContract)->TensorDtype{
    match spec.dtype(){
        crate::BoundaryTensorDtype::Activation=>match wire.activation_dtype(){
            PipelineActivationDtype::Float16=>TensorDtype::F16,
            PipelineActivationDtype::Bfloat16=>TensorDtype::Bf16,
            PipelineActivationDtype::Float32=>TensorDtype::F32,
        },
        crate::BoundaryTensorDtype::Uint32=>TensorDtype::U32,
        crate::BoundaryTensorDtype::Int32=>TensorDtype::I32,
    }
}

fn resolved_tagged_boundary_roles<T>(
    values: &[crate::ArchitectureBoundaryValue<T>],
    schema: &ResolvedBoundaryWireSchema,
    wire: PipelineWireContract,
) -> Result<Vec<crate::BoundaryRoleContract>, PartitionExecutionError> {
    values
        .iter()
        .zip(std::iter::once(schema.primary()).chain(schema.auxiliary()))
        .map(|(value, spec)| {
            let dtype = boundary_dtype(spec,wire);
            let shape = spec
                .shape()
                .iter()
                .map(|dimension| usize::try_from(*dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    PartitionExecutionError::BoundaryFraming(
                        "resolved boundary shape is not representable".into(),
                    )
                })?;
            crate::BoundaryRoleContract::new(value.role(), dtype, shape)
                .map_err(|error| PartitionExecutionError::BoundaryFraming(error.to_string()))
        })
        .collect()
}

fn validate_boundary_tensor<B, I>(
    inspector: &I,
    value: &B::Tensor,
    spec: &ResolvedBoundaryTensorSpec,
    wire: PipelineWireContract,
) -> Result<(), PartitionExecutionError>
where
    B: NeuralBackend,
    I: CommunicationTensorMetadata<B>,
{
    let shape = inspector.shape(value);
    let expected_shape = spec
        .shape()
        .iter()
        .map(|dimension| usize::try_from(*dimension).unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    if shape != expected_shape {
        return Err(PartitionExecutionError::BoundaryShape {
            role: spec.role().to_owned(),
            expected: expected_shape,
            actual: shape,
        });
    }
    let expected_dtype = boundary_dtype(spec,wire);
    let actual = inspector.dtype(value);
    if actual != expected_dtype {
        return Err(PartitionExecutionError::BoundaryDtype {
            role: spec.role().to_owned(),
            expected: expected_dtype,
            actual,
        });
    }
    Ok(())
}

enum PartitionScheduleSetupError<E> {
    Architecture(E),
    Schedule(LayeredPipelineScheduleError),
}

impl<E> From<LayeredPipelineScheduleError> for PartitionScheduleSetupError<E> {
    fn from(error: LayeredPipelineScheduleError) -> Self {
        Self::Schedule(error)
    }
}

/// Cold-path validation or execution failure in the neutral partition driver.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
#[allow(
    missing_docs,
    reason = "field names and error messages document mechanical validation diagnostics"
)]
pub enum PartitionExecutionError {
    #[error(transparent)]
    PipelineSchedule(#[from] LayeredPipelineScheduleError),
    #[error("{0}")]
    PipelineProgress(&'static str),
    #[error("partition workspace preparation failed: {0}")]
    Workspace(#[source] eredu_nn::Error),
    #[error("prepared boundary source failed: {0}")]
    BoundarySource(#[source] eredu_core::BackendFailure),
    #[error(transparent)]
    PreparedBoundary(#[from] crate::PreparedBoundaryFrameError),
    /// Prepared publication metadata could not be retained under its source.
    #[error("prepared publication metadata failed: {0}")]
    PublicationMetadata(#[source] eredu_nn::workspace::HostMetadataFundingError),
    /// The selected backend has no bounded actual tensor metadata producer.
    #[error("prepared publication tensor metadata is unavailable")]
    PreparedTensorMetadataUnavailable,
    /// Actual prepared metadata failed the same shared tensor contract check.
    #[error("prepared communication tensor contract failed: {0}")]
    PreparedTensor(#[source] crate::CommunicationTensorContractError),
    /// The exact publication group is absent from the retained declaration.
    #[error("prepared communication group selection failed: {0}")]
    PreparedGroup(#[source] crate::CommunicationGroupOperationError),
    /// A paid failure shell keeps its concrete source without a formatted copy.
    #[error("prepared communication {operation:?} failed during {phase:?} (completion={completion}): {source}")]
    PreparedCommunication {
        operation:CommunicationOperation,
        phase:DistributedExecutionPhase,
        completion:bool,
        #[source]
        source:eredu_core::BackendFailure,
    },

    /// New work is forbidden after a guard or communication terminal failure.
    #[error("communication incarnation is terminal for new submissions")]
    CommunicationTerminal,

    /// A partition communication object omitted the selected bounded-wait contract.
    #[error("partition communication manifest has no bounded completion policy")]
    MissingBoundedCompletionPolicy,
    /// Exact communication did not complete by its selected deadline.
    #[error(
        "communication {operation:?} exceeded its selected deadline during {phase:?} (route {route:?}, disposition {cancellation:?})"
    )]
    CommunicationDeadlineExceeded {
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
        cancellation: CompletionCancellationMode,
    },
    /// A submitted communication reported a terminal error while completing.
    #[error(
        "communication {operation:?} failed during exact completion in {phase:?} (route {route:?}): {error}"
    )]
    CommunicationCompletionFailed {
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
        error: String,
        #[source]
        source: Option<eredu_core::BackendFailure>,
    },
    /// Native submission failed after entering one communication operation.
    #[error(
        "communication {operation:?} submission failed in {phase:?} (route {route:?}): {error}"
    )]
    CommunicationSubmissionFailed {
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
        error: String,
        #[source]
        source: Option<eredu_core::BackendFailure>,
    },
    /// A prior timed-out or failed operation left this communicator unsafe for reuse.
    #[error(
        "communication is poisoned by prior {operation:?} during {phase:?} (route {route:?}, disposition {cancellation:?})"
    )]
    CommunicationPoisoned {
        operation: CommunicationOperation,
        phase: DistributedExecutionPhase,
        route: Option<CommunicationRouteId>,
        cancellation: CompletionCancellationMode,
    },
    /// A recovery-only agreement was invoked without a local failure or prior poison.
    #[error("recovery agreement was requested without a prior failure during {phase:?}")]
    RecoveryAgreementWithoutFailure { phase: DistributedExecutionPhase },
    /// Native resources were not paired one-for-one with manifest descriptors.
    #[error(
        "communication resource count mismatch (groups {actual_groups}/{expected_groups}, routes {actual_routes}/{expected_routes})"
    )]
    ResourceCount {
        expected_groups: usize,
        actual_groups: usize,
        expected_routes: usize,
        actual_routes: usize,
    },
    /// A native resource was paired with a different opaque manifest identity.
    #[error("communication resource identity mismatch (actual {actual}, expected {expected})")]
    ResourceIdentity { expected: u64, actual: u64 },
    #[error("unknown opaque communication group {0:?}")]
    UnknownGroup(CollectiveGroupId),
    #[error("local rank is not a member of opaque communication group {0:?}")]
    NotGroupMember(CollectiveGroupId),
    #[error("unknown opaque communication route {0:?}")]
    UnknownRoute(CommunicationRouteId),
    #[error(
        "route submission wave beginning at {first:?} has {actual} local routes/endpoints, expected {expected}"
    )]
    RouteSubmissionWave {
        first: CommunicationRouteId,
        expected: usize,
        actual: usize,
    },
    #[error("local rank is not an endpoint of opaque communication route {0:?}")]
    NotRouteEndpoint(CommunicationRouteId),
    #[error("{resource} did not select operation {operation:?}")]
    OperationNotSelected {
        resource: String,
        operation: CommunicationOperation,
    },
    #[error("peer count consensus tensor contract failed (completed={completed}): {cause}")]
    PeerConsensusTensor {
        completed: bool,
        #[source]
        cause: crate::CommunicationTensorContractError,
    },
    #[error("communication tensor has unselected dtype {dtype:?}")]
    TensorDtype { dtype: TensorDtype },
    #[error("communication tensor shape exceeds selected limits: {shape:?}")]
    TensorLimits { shape: Vec<usize> },
    #[error("communication axis {axis} is outside tensor rank {rank}")]
    CommunicationAxis { axis: usize, rank: usize },
    #[error("communication result shape arithmetic overflowed")]
    CommunicationShapeOverflow,
    #[error("communication result shape {actual:?} differs from expected {expected:?}")]
    CommunicationOutputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("communication tensor requirement has no payload limits")]
    MissingTensorLimits,
    #[error("communication bundle contains {actual} tensors, maximum is {expected_at_most}")]
    TensorCount {
        expected_at_most: usize,
        actual: usize,
    },
    #[error("peer-count cardinality is {actual}, expected {expected}")]
    PeerCount { expected: usize, actual: usize },
    #[error("a peer count exceeds the selected maximum {maximum}")]
    PeerCountLimit { maximum: usize },
    #[error("architecture boundary {boundary:?} contains {actual} tensors, expected {expected}")]
    BoundaryTensorCount {
        boundary: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("architecture boundary role {role:?} has shape {actual:?}, expected {expected:?}")]
    BoundaryShape {
        role: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("architecture boundary role {role:?} has dtype {actual:?}, expected {expected:?}")]
    BoundaryDtype {
        role: String,
        expected: TensorDtype,
        actual: TensorDtype,
    },
    #[error("role-exact boundary framing failed: {0}")]
    BoundaryFraming(String),
    #[error("opaque output group {group:?} does not contain output owner rank {rank}")]
    OutputOwnerNotMember {
        rank: usize,
        group: CollectiveGroupId,
    },
    #[error("rank {rank} output ownership disagrees with selected owner {owner}")]
    OutputOwnership { rank: usize, owner: usize },
    #[error("a local output owner has no publication operation")]
    MissingOutputPublication,
    #[error(
        "partition plan declares {contracts} contracts and {drivers} drivers for {graph} groups"
    )]
    GroupCount {
        graph: usize,
        contracts: usize,
        drivers: usize,
    },
    #[error("rank-local driver is stored under the wrong group slot {group}")]
    DriverGroup { group: usize },
    #[error("route {0:?} names a missing architecture group")]
    RouteGroup(CommunicationRouteId),
    #[error("route {0:?} does not connect an architecture dependency edge")]
    RouteDependency(CommunicationRouteId),
    #[error("partition plan contains {plan} routes but its manifest contains {manifest}")]
    RouteCount { plan: usize, manifest: usize },
    #[error(
        "route {route:?} ownership {planned_source} -> {planned_destination} differs from manifest endpoints {manifest_source} -> {manifest_destination}"
    )]
    RouteDescriptorMismatch {
        route: CommunicationRouteId,
        planned_source: usize,
        planned_destination: usize,
        manifest_source: usize,
        manifest_destination: usize,
    },
    #[error("route {route:?} endpoint rank {rank} does not own architecture group {group}")]
    RouteOwnerMissing {
        route: CommunicationRouteId,
        rank: usize,
        group: usize,
    },
    #[error("communication mechanism failed: {0}")]
    Communication(String),
    #[error("another rank reported failure during distributed phase {0:?}")]
    RemotePhaseFailure(DistributedExecutionPhase),
    #[error("rank-local execution residency and bounded policy disagree")]
    ResidencyPolicyMismatch,
    #[error("communication operation {0:?} has no selected execution policy")]
    OperationPolicyUnavailable(CommunicationOperation),
    #[error("opaque communication group {group:?} selected an inexact {operation:?} requirement")]
    InexactOperationRequirement {
        group: CollectiveGroupId,
        operation: CommunicationOperation,
    },
    #[error("partition communication plan and additive operation policies disagree")]
    CommunicationPolicyMismatch,
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::{CommunicationGroupRequirements, CommunicationTensorLimits};

    #[test]
    fn plan_exposes_the_exact_selected_publication_and_commit_group() {
        let graph =
            ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder")
                .unwrap();
        let group = CollectiveGroupId::new(17);
        let plan = PartitionedExecutionPlan::new(
            graph,
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![None],
            Vec::new(),
            Some(PartitionOutputPublication {
                group,
                owner_rank: 3,
            }),
            Some(group),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();

        assert_eq!(plan.publication().unwrap().group, group);
        assert_eq!(plan.publication().unwrap().owner_rank, 3);
        assert_eq!(plan.commit_barrier(), Some(group));
    }

    #[test]
    fn publication_authority_preserves_selected_owner_and_group_local_rank() {
        let graph =
            ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder")
                .unwrap();
        let group = CollectiveGroupId::new(17);
        let plan = PartitionedExecutionPlan::new(
            graph,
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![None],
            Vec::new(),
            Some(PartitionOutputPublication {
                group,
                owner_rank: 3,
            }),
            Some(group),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        let requirements = CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 32, None).unwrap(),
                true,
            )
            .unwrap(),
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap();
        let manifest = CommunicationManifest::new(
            8,
            7,
            vec![CommunicationGroupDescriptor::new(
                group,
                0,
                vec![3, 7],
                Some(1),
                requirements.clone(),
            )
            .unwrap()],
            Vec::new(),
        )
        .unwrap();
        let authority = plan.publication_authority(&manifest).unwrap().unwrap();
        assert_eq!(authority.owner_rank(), 3);
        assert_eq!(authority.owner_group_rank(), 0);
        assert!(!authority.local_public_output());

        let substituted = CommunicationManifest::new(
            8,
            7,
            vec![
                CommunicationGroupDescriptor::new(group, 0, vec![4, 7], Some(1), requirements)
                    .unwrap(),
            ],
            Vec::new(),
        )
        .unwrap();
        assert!(matches!(
            plan.publication_authority(&substituted),
            Err(PartitionExecutionError::OutputOwnerNotMember { rank: 3, .. })
        ));
    }

    #[test]
    fn plan_rejects_manifest_route_endpoints_that_differ_from_architecture_ownership() {
        let graph =
            ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder")
                .unwrap();
        let route = CommunicationRouteId::new(9);
        let plan = PartitionedExecutionPlan::new(
            graph,
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![None],
            vec![PartitionBoundaryRoute {
                source_group: 0,
                destination_group: 0,
                source_rank: 0,
                destination_rank: 1,
                source_unit_end: None,
                route,
            }],
            None,
            None,
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            crate::CommunicationTensorLimits::new(1, 3, 8, None).unwrap(),
            true,
        )
        .unwrap();
        let manifest = CommunicationManifest::new(
            3,
            2,
            Vec::new(),
            vec![CommunicationRouteDescriptor::new(route, 0, 1, 0, requirement).unwrap()],
        )
        .unwrap();

        assert!(matches!(
            plan.validate_manifest(&manifest),
            Err(PartitionExecutionError::RouteDescriptorMismatch {
                planned_source: 0,
                planned_destination: 1,
                manifest_source: 1,
                manifest_destination: 0,
                ..
            })
        ));
    }

    fn publication_plan(group: CollectiveGroupId) -> PartitionedExecutionPlan {
        PartitionedExecutionPlan::new(
            ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder")
                .unwrap(),
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![None],
            Vec::new(),
            Some(PartitionOutputPublication {
                group,
                owner_rank: 0,
            }),
            Some(group),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap()
    }

    fn single_group_manifest(
        group: CollectiveGroupId,
        requirements: CommunicationGroupRequirements,
    ) -> CommunicationManifest {
        CommunicationManifest::new(
            2,
            1,
            vec![
                CommunicationGroupDescriptor::new(group, 0, vec![0, 1], Some(1), requirements)
                    .unwrap(),
            ],
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn plan_rejects_publication_group_without_exact_broadcast_before_execution() {
        let group = CollectiveGroupId::new(29);
        let plan = publication_plan(group);
        let missing = single_group_manifest(
            group,
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        );
        assert!(matches!(
            plan.validate_manifest(&missing),
            Err(PartitionExecutionError::OperationNotSelected {
                operation: CommunicationOperation::Broadcast,
                ..
            })
        ));

        let inexact = single_group_manifest(
            group,
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::F32],
                    CommunicationTensorLimits::new(1, 3, 32, None).unwrap(),
                    false,
                )
                .unwrap(),
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        );
        assert!(matches!(
            plan.validate_manifest(&inexact),
            Err(PartitionExecutionError::InexactOperationRequirement {
                operation: CommunicationOperation::Broadcast,
                ..
            })
        ));
    }

    #[test]
    fn plan_rejects_inexact_failure_agreement_before_execution() {
        let group = CollectiveGroupId::new(31);
        let plan = publication_plan(group);
        let manifest = single_group_manifest(
            group,
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::F32],
                    CommunicationTensorLimits::new(1, 3, 32, None).unwrap(),
                    true,
                )
                .unwrap(),
                CommunicationOperationRequirement::failure_agreement(false),
            ])
            .unwrap(),
        );
        plan.validate_manifest(&manifest).unwrap();
        assert!(matches!(
            plan.validate_exact_commit_agreement(&manifest),
            Err(PartitionExecutionError::InexactOperationRequirement {
                operation: CommunicationOperation::FailureAgreement,
                ..
            })
        ));
    }
}

#[cfg(test)]
#[path = "partitioned_execution/terminal_tests.rs"]
mod terminal_tests;

// Inner pipeline votes consume the same request source as outer lifecycle votes.
// Borrow disjoint runtime fields so the active pass and semantic route stay live.
fn agree_partition_phase<A, B, S, E, G, R, I, V>(
    executor: &E,
    policy: &mut V,
    control: Option<&B::ParallelContext>,
    communication: &PartitionCommunication<B, G, R, I>,
    group: CollectiveGroupId,
    phase: DistributedExecutionPhase,
    local_success: bool,
    context: &B::Executor,
) -> Result<bool, PartitionExecutionError>
where
    B: CommunicationBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    E: PartitionedGroupExecutor<A, B, S, G, R, I>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    V: PartitionCommitAgreement<B, G, R, I>,
{
    communication.agree_phase_with_parallel_context(policy, group, phase, local_success,
        context, control.or_else(|| executor.current_parallel_context()))
}

// One scheduler for ordinary and already-prepared retained-ingress passes.
fn run_partition_pass<'a, 'control, A, B, S, Resident, Bounded, E, G, R, I, T, U, V, O>(
    runtime: &mut PartitionedTextRuntime<A, B, S, Bounded, E, G, R, I, T, U, V>,
    mut pass: E::Pass<'a, 'control>,
    state: &mut S,
    context: &<B::Tensor as Tensor>::Context,
    observer: &mut O,
) -> Result<
    (Option<B::Tensor>, A::ForwardContext),
    ReplicatedTextSessionError<A::Error, Resident::Error, std::convert::Infallible>,
>
where
    B: CommunicationBackend + crate::TerminalCommunicationBackend,
    S: RuntimeState<B> + 'control,
    A: LayeredArchitecture<B, S>,
    Resident: LayerwisePolicy<B, A::Unit>,
    Bounded: LayerwisePolicy<B, A::Unit, Error = Resident::Error>,
    E: PartitionedGroupExecutor<A, B, S, G, R, I>
        + crate::parameter_operations::LayeredParameterOwner<B, S, Architecture = A>,
    E::Policy: LayerwisePolicy<B, A::Unit, Error = Resident::Error>,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
    T: PartitionBoundaryTransport<B, G, R, I>,
    U: PartitionOutputPublisher<B, G, R, I>,
    V: PartitionCommitAgreement<B, G, R, I>,
    A::Error: std::fmt::Display,
    Resident::Error: std::fmt::Display,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
{
    let metadata=B::construction_metadata(context).filter(|value|value.uses_checked_metadata());
    if let Some(metadata)=metadata {
        metadata.charge_metadata(std::mem::size_of::<(PartitionExecutionError,LayeredPipelineScheduleError,
            PartitionScheduleSetupError<A::Error>,Result<(Option<B::Tensor>,A::ForwardContext),
                ReplicatedTextSessionError<A::Error,Resident::Error,std::convert::Infallible>>)>() )
            .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause.into())))?;
    }
    let phase_agreement_group = runtime.plan.commit_barrier;
    let request=|group| runtime.executor.request_group_active(&pass,group)
        .map_err(PartitionScheduleSetupError::Architecture);
    let mut schedule = match metadata {
        Some(metadata)=>LayeredPipelineSchedule::try_new_with_metadata(&runtime.plan.graph,
            &runtime.plan.group_contracts,request,metadata)
            .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?,
        None=>LayeredPipelineSchedule::try_new(
        &runtime.plan.graph,
        runtime.plan.group_contracts.iter().copied(),
        request,
    ),
    }.map_err(|error| match error {
        PartitionScheduleSetupError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        PartitionScheduleSetupError::Schedule(error) => {
            ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineSchedule(error))
        }
    })?;

    while !schedule.is_complete() {
        let Some(group) = schedule.ready_groups().next() else {
            return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineProgress("partitioned graph schedule made no progress")));
        };
        schedule
            .started_without_release(group)
            .map_err(|error| ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineSchedule(error)))?;
        if schedule.is_active(group) == Some(true) {
            let same_group_routes=partition_metadata_collect(metadata,runtime.plan.routes.iter()
                .filter(|route|route.source_group==group&&route.destination_group==group).cloned())
                .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
            let rank = runtime.communication.manifest().rank();
            let mut executed = false;
            let mut pipeline_wave = 0usize;

            // With explicit failure agreement, advance the rank graph one pipeline wave at
            // a time. Every source in a wave executes concurrently so TP peers reach their
            // collectives together; only after agreement may matching destinations receive.
            // A middle stage becomes a source only after its incoming wave was removed.
            if V::PHASE_FAILURE_AGREEMENT && !same_group_routes.is_empty() {
                let mut remaining=partition_metadata_vec(metadata,same_group_routes.len())
                    .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
                remaining.resize(same_group_routes.len(),true);
                while remaining.iter().any(|remaining| *remaining) {
                    let wave = partition_metadata_collect(metadata,same_group_routes
                        .iter()
                        .enumerate()
                        .filter(|(index, route)| {
                            remaining[*index]
                                && !same_group_routes.iter().enumerate().any(
                                    |(predecessor, candidate)| {
                                        remaining[predecessor]
                                            && candidate.destination_rank == route.source_rank
                                    },
                                )
                        })
                        .map(|(index, _)| index)
                        )
                        .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
                    if wave.is_empty() {
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineProgress("same-group pipeline routes contain a rank cycle")));
                    }
                    let active = wave
                        .iter()
                        .any(|index| same_group_routes[*index].source_rank == rank)
                        && !executed;
                    if active {
                        executed = true;
                    }
                    let local_execution = Some(runtime.executor.execute_pipeline_wave_with_parallel_context(
                        &mut pass,
                        group,
                        runtime.plan.drivers[group].as_ref(),
                        active,
                        pipeline_wave,
                        state,
                        &runtime.communication,
                        runtime.communication_executor.borrow(),
                        context,
                        runtime.parallel_control.as_deref(),
                        observer,
                    ));
                    let mut local_error = match local_execution {
                        Some(Err(error)) => Some(PartitionRouteTransferError::Architecture(error)),
                        _ => None,
                    };
                    let execution_agreement = agree_partition_phase::<A, B, S, E, G, R, I, V>(
                            &runtime.executor, &mut runtime.commit_agreement, runtime.parallel_control.as_deref(),
                        &runtime.communication,
                        phase_agreement_group.expect("selected execution agreement group"),
                        DistributedExecutionPhase::Execution,
                        local_error.is_none(),
                        runtime.communication_executor.borrow(),
                    );
                    // A completed vote proves all ranks reached this boundary
                    // before any boundary payload was prepared or submitted.
                    runtime.execution_rejection_agreed |= matches!(execution_agreement, Ok(false));
                    if let Some(error) = local_error.take() {
                        return Err(map_partition_route_error(error));
                    }
                    if !execution_agreement.map_err(ReplicatedTextSessionError::Partition)? {
                        return Err(ReplicatedTextSessionError::Partition(
                            PartitionExecutionError::RemotePhaseFailure(
                                DistributedExecutionPhase::Execution,
                            ),
                        ));
                    }
                    let mut prepared_wave=partition_metadata_vec(metadata,wave.len())
                        .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
                    for index in wave.iter().copied() {
                        let route = &same_group_routes[index];
                        let prepared = if local_error.is_none()
                            && (rank == route.source_rank || rank == route.destination_rank)
                        {
                            match prepare_partition_boundary::<A, B, S, E, G, R, I>(
                                &mut runtime.executor,
                                runtime.parallel_control.as_deref(),
                                &mut pass,
                                &runtime.communication,
                                route,
                                runtime.plan.wire,
                                context,
                            ) {
                                Ok(prepared) => Some(prepared),
                                Err(error) => {
                                    local_error = Some(error);
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        prepared_wave.push((index, prepared));
                    }
                    if local_error.is_none() {
                        for (index, prepared) in &prepared_wave {
                            let Some(prepared) = prepared.as_ref().filter(|value| value.source)
                            else {
                                continue;
                            };
                            let route = &same_group_routes[*index];
                            if let Err(error) = runtime.communication.complete_local_dependencies(
                                &prepared.values,
                                route.route,
                                runtime.communication_executor.borrow(),
                                true,
                                prepared.source_context.as_ref(),
                                runtime.parallel_control.as_deref().or_else(||runtime.executor.current_parallel_context()),
                            ) {
                                local_error = Some(PartitionRouteTransferError::Contract(error));
                                break;
                            }
                        }
                    }
                    let local_success = local_error.is_none();
                    let mut remote_completion_failure = None;
                    for index in wave.iter().copied() {
                        let route = &same_group_routes[index];
                        let completed = match agree_partition_phase::<A, B, S, E, G, R, I, V>(
                            &runtime.executor, &mut runtime.commit_agreement, runtime.parallel_control.as_deref(),
                            &runtime.communication,
                            phase_agreement_group.expect(
                                "failure-agreement policy requires a selected session group",
                            ),
                            DistributedExecutionPhase::BoundarySourceCompletion(route.route),
                            local_success,
                            runtime.communication_executor.borrow(),
                        ) {
                            Ok(completed) => completed,
                            Err(error) => {
                                if let Some(local) = local_error.take() {
                                    return Err(map_partition_route_error(local));
                                }
                                return Err(ReplicatedTextSessionError::Partition(error));
                            }
                        };
                        if !completed && remote_completion_failure.is_none() {
                            remote_completion_failure = Some(route.route);
                        }
                    }
                    if let Some(error) = local_error {
                        let route = wave.first().map(|index| same_group_routes[*index].route);
                        runtime.communication.authority.fence_protocol_failure(
                            CommunicationOperation::SendReceive,
                            route.map_or(
                                DistributedExecutionPhase::Execution,
                                DistributedExecutionPhase::BoundarySourceCompletion,
                            ),
                            route,
                        );
                        return Err(map_partition_route_error(error));
                    }
                    if let Some(route) = remote_completion_failure {
                        runtime.communication.authority.fence_protocol_failure(
                            CommunicationOperation::SendReceive,
                            DistributedExecutionPhase::BoundarySourceCompletion(route),
                            Some(route),
                        );
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::RemotePhaseFailure(
                                DistributedExecutionPhase::BoundarySourceCompletion(route),
                            )
                            ));
                    }
                    let mut remote_failure = None;
                    for index in wave.iter().copied() {
                        let route = &same_group_routes[index];
                        let ready = agree_partition_phase::<A, B, S, E, G, R, I, V>(
                            &runtime.executor, &mut runtime.commit_agreement, runtime.parallel_control.as_deref(),
                                &runtime.communication,
                                phase_agreement_group.expect(
                                    "failure-agreement policy requires a selected session group",
                                ),
                                DistributedExecutionPhase::BoundarySourceReady(route.route),
                                true,
                                runtime.communication_executor.borrow(),
                            )
                            .map_err(|error| ReplicatedTextSessionError::Partition(error))?;
                        if !ready && remote_failure.is_none() {
                            remote_failure = Some(route.route);
                        }
                    }
                    if let Some(route) = remote_failure {
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::RemotePhaseFailure(
                                DistributedExecutionPhase::BoundarySourceReady(route),
                            )
                            ));
                    }
                    for (index, prepared) in prepared_wave {
                        let route = &same_group_routes[index];
                        if let Some(prepared) = prepared {
                            transfer_prepared_partition_boundary::<A, B, S, E, G, R, I, T>(
                                &mut runtime.executor,
                                runtime.parallel_control.as_deref(),
                                &mut pass,
                                &runtime.communication,
                                &mut runtime.boundary_transport,
                                route,
                                runtime.plan.wire,
                                runtime.communication_executor.borrow(),
                                prepared,
                            )
                            .map_err(map_partition_route_error)?;
                        }
                        remaining[index] = false;
                    }
                    pipeline_wave = pipeline_wave.checked_add(1).ok_or_else(|| {
                        ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineProgress("pipeline execution wave ordinal overflowed"))
                    })?;
                }
            } else {
                for route in &same_group_routes {
                    if rank == route.destination_rank {
                        prepare_and_transfer_partition_boundary::<A, B, S, E, G, R, I, T>(
                            &mut runtime.executor,
                                runtime.parallel_control.as_deref(),
                            &mut pass,
                            &runtime.communication,
                            &mut runtime.boundary_transport,
                            route,
                            runtime.plan.wire,
                            runtime.communication_executor.borrow(),
                            context,
                        )
                        .map_err(map_partition_route_error)?;
                    }
                }
            }

            let mut local_execution = if V::PHASE_FAILURE_AGREEMENT && !same_group_routes.is_empty()
            {
                let active = !executed && runtime.plan.drivers[group].is_some();
                Some(runtime.executor.execute_pipeline_wave_with_parallel_context(
                    &mut pass,
                    group,
                    runtime.plan.drivers[group].as_ref(),
                    active,
                    pipeline_wave,
                    state,
                    &runtime.communication,
                    runtime.communication_executor.borrow(),
                    context,
                    runtime.parallel_control.as_deref(),
                        observer,
                ))
            } else if executed {
                None
            } else {
                runtime.plan.drivers[group].as_ref().map(|driver| {
                    runtime.executor.execute_group_with_parallel_context(
                        &mut pass,
                        driver,
                        state,
                        &runtime.communication,
                        runtime.communication_executor.borrow(),
                        context,
                        runtime.parallel_control.as_deref(),
                        observer,
                    )
                })
            };

            if V::PHASE_FAILURE_AGREEMENT {
                let execution_agreement = agree_partition_phase::<A, B, S, E, G, R, I, V>(
                            &runtime.executor, &mut runtime.commit_agreement, runtime.parallel_control.as_deref(),
                    &runtime.communication,
                    phase_agreement_group.expect("selected execution agreement group"),
                    DistributedExecutionPhase::Execution,
                    local_execution.as_ref().is_none_or(Result::is_ok),
                    runtime.communication_executor.borrow(),
                );
                runtime.execution_rejection_agreed |= matches!(execution_agreement, Ok(false));
                if local_execution.as_ref().is_some_and(Result::is_err) {
                    let Some(Err(error)) = local_execution.take() else {
                        unreachable!("local execution failure was checked")
                    };
                    return Err(ReplicatedTextSessionError::Architecture(error));
                }
                if !execution_agreement.map_err(ReplicatedTextSessionError::Partition)? {
                    return Err(ReplicatedTextSessionError::Partition(
                        PartitionExecutionError::RemotePhaseFailure(
                            DistributedExecutionPhase::Execution,
                        ),
                    ));
                }
            }

            let outgoing_routes = partition_metadata_collect(metadata,runtime
                .plan
                .routes
                .iter()
                .filter(|route| {
                    route.source_group == group
                        && (!V::PHASE_FAILURE_AGREEMENT
                            || route.source_group != route.destination_group)
                })
                .cloned()
                )
                .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
            if V::PHASE_FAILURE_AGREEMENT {
                let manifest = runtime.communication.manifest();
                let mut waves=partition_metadata_vec(metadata,manifest.routes().len())
                    .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
                for descriptor_range in manifest.route_submission_waves() {
                    let wave = partition_metadata_collect(metadata,descriptor_range
                        .clone()
                        .filter_map(|index| {
                            let id = manifest.routes()[index].id();
                            outgoing_routes
                                .iter()
                                .find(|route| route.route == id)
                                .cloned()
                        })
                        )
                        .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
                    if wave.is_empty() {
                        continue;
                    }
                    if wave.len() != descriptor_range.len() {
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::RouteSubmissionWave {
                                first: manifest.routes()[descriptor_range.start].id(),
                                expected: descriptor_range.len(),
                                actual: wave.len(),
                            }
                            ));
                    }
                    waves.push(wave);
                }
                if waves.iter().map(Vec::len).sum::<usize>() != outgoing_routes.len() {
                    return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineProgress("outgoing routes were omitted from manifest submission waves")));
                }

                for wave in waves {
                    let phase_route = wave[0].route;
                    let endpoints = partition_metadata_collect(metadata,wave
                        .iter()
                        .filter(|route| rank == route.source_rank || rank == route.destination_rank)
                        )
                        .map_err(|cause|ReplicatedTextSessionError::Partition(PartitionExecutionError::Workspace(cause)))?;
                    if wave.len() > 1 && endpoints.len() != 1 {
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::RouteSubmissionWave {
                                first: phase_route,
                                expected: 1,
                                actual: endpoints.len(),
                            }
                            ));
                    }
                    let local_route = endpoints.first().copied();
                    let mut local_error = if local_route
                        .is_some_and(|route| rank == route.source_rank)
                        && local_execution.as_ref().is_some_and(Result::is_err)
                    {
                        let Some(Err(error)) = local_execution.take() else {
                            unreachable!("the local execution result was checked as an error")
                        };
                        Some(PartitionRouteTransferError::Architecture(error))
                    } else {
                        None
                    };
                    let prepared = if local_error.is_none() {
                        local_route.and_then(|route| {
                            match prepare_partition_boundary::<A, B, S, E, G, R, I>(
                                &mut runtime.executor,
                                runtime.parallel_control.as_deref(),
                                &mut pass,
                                &runtime.communication,
                                route,
                                runtime.plan.wire,
                                context,
                            ) {
                                Ok(prepared) => Some(prepared),
                                Err(error) => {
                                    local_error = Some(error);
                                    None
                                }
                            }
                        })
                    } else {
                        None
                    };
                    if local_error.is_none() {
                        if let (Some(route), Some(prepared)) =
                            (local_route, prepared.as_ref().filter(|value| value.source))
                        {
                            if let Err(error) = runtime.communication.complete_local_dependencies(
                                &prepared.values,
                                route.route,
                                runtime.communication_executor.borrow(),
                                true,
                                prepared.source_context.as_ref(),
                                runtime.parallel_control.as_deref().or_else(||runtime.executor.current_parallel_context()),
                            ) {
                                local_error = Some(PartitionRouteTransferError::Contract(error));
                            }
                        }
                    }
                    let completed = agree_partition_phase::<A, B, S, E, G, R, I, V>(
                            &runtime.executor, &mut runtime.commit_agreement, runtime.parallel_control.as_deref(),
                            &runtime.communication,
                            phase_agreement_group.expect(
                                "failure-agreement policy requires a selected session group",
                            ),
                            DistributedExecutionPhase::BoundarySourceCompletion(phase_route),
                            local_error.is_none(),
                            runtime.communication_executor.borrow(),
                        )
                        .map_err(|error| ReplicatedTextSessionError::Partition(error))?;
                    if let Some(error) = local_error {
                        runtime.communication.authority.fence_protocol_failure(
                            CommunicationOperation::SendReceive,
                            DistributedExecutionPhase::BoundarySourceCompletion(phase_route),
                            Some(phase_route),
                        );
                        return Err(map_partition_route_error(error));
                    }
                    if !completed {
                        runtime.communication.authority.fence_protocol_failure(
                            CommunicationOperation::SendReceive,
                            DistributedExecutionPhase::BoundarySourceCompletion(phase_route),
                            Some(phase_route),
                        );
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::RemotePhaseFailure(
                                DistributedExecutionPhase::BoundarySourceCompletion(phase_route),
                            )
                            ));
                    }
                    let ready = agree_partition_phase::<A, B, S, E, G, R, I, V>(
                            &runtime.executor, &mut runtime.commit_agreement, runtime.parallel_control.as_deref(),
                            &runtime.communication,
                            phase_agreement_group.expect(
                                "failure-agreement policy requires a selected session group",
                            ),
                            DistributedExecutionPhase::BoundarySourceReady(phase_route),
                            true,
                            runtime.communication_executor.borrow(),
                        )
                        .map_err(|error| ReplicatedTextSessionError::Partition(error))?;
                    if !ready {
                        return Err(ReplicatedTextSessionError::Partition(PartitionExecutionError::RemotePhaseFailure(
                                DistributedExecutionPhase::BoundarySourceReady(phase_route),
                            )
                            ));
                    }
                    if let (Some(route), Some(prepared)) = (local_route, prepared) {
                        transfer_prepared_partition_boundary::<A, B, S, E, G, R, I, T>(
                            &mut runtime.executor,
                                runtime.parallel_control.as_deref(),
                            &mut pass,
                            &runtime.communication,
                            &mut runtime.boundary_transport,
                            route,
                            runtime.plan.wire,
                            runtime.communication_executor.borrow(),
                            prepared,
                        )
                        .map_err(map_partition_route_error)?;
                    }
                }
            } else {
                for route in &outgoing_routes {
                    let descriptor = runtime
                        .communication
                        .manifest()
                        .routes()
                        .iter()
                        .find(|candidate| candidate.id() == route.route)
                        .ok_or_else(|| {
                            ReplicatedTextSessionError::Partition(PartitionExecutionError::UnknownRoute(route.route))
                        })?;
                    let same_group = route.source_group == route.destination_group;
                    if rank != descriptor.source()
                        && (same_group || rank != descriptor.destination())
                    {
                        continue;
                    }
                    prepare_and_transfer_partition_boundary::<A, B, S, E, G, R, I, T>(
                        &mut runtime.executor,
                                runtime.parallel_control.as_deref(),
                        &mut pass,
                        &runtime.communication,
                        &mut runtime.boundary_transport,
                        route,
                        runtime.plan.wire,
                        runtime.communication_executor.borrow(),
                        context,
                    )
                    .map_err(map_partition_route_error)?;
                }
            }
            if let Some(Err(error)) = local_execution {
                return Err(ReplicatedTextSessionError::Architecture(error));
            }
        } else {
            runtime.executor.retain_inactive_group(&mut pass, group);
        }
        schedule
            .ordered(group)
            .map_err(|error| ReplicatedTextSessionError::Partition(PartitionExecutionError::PipelineSchedule(error)))?;
    }

    let (output, forward) = runtime
        .executor
        .finish(pass, state, context)
        .map_err(ReplicatedTextSessionError::Architecture)?;
    Ok((output, forward))
}

mod boundary;

fn partition_metadata_vec<T>(context:Option<&eredu_nn::workspace::WorkspaceContext>,capacity:usize)
    ->Result<Vec<T>,eredu_nn::Error>{
    match context{Some(context)=>context.metadata_vec(capacity),None=>Ok(Vec::with_capacity(capacity))}
}
fn partition_metadata_collect<T>(context:Option<&eredu_nn::workspace::WorkspaceContext>,values:impl Iterator<Item=T>)
    ->Result<Vec<T>,eredu_nn::Error>{
    match context{
        None=>Ok(values.collect()),
        Some(context)=>{
            context.charge_metadata(std::mem::size_of::<(Vec<T>,Result<Vec<T>,eredu_nn::Error>,T)>()
                .checked_add(std::mem::size_of_val(&values)).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
            let capacity=values.size_hint().1.ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
            let mut output=context.metadata_vec(capacity)?;
            for value in values{
                if output.len()==capacity{return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());}
                output.push(value);
            }
            Ok(output)
        }
    }
}
