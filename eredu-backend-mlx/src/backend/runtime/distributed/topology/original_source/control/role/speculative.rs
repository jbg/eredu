//! Closed native custody for the existing parallel-control worker.
use super::super::speculative::PreparedSpeculativeControl;
use super::*;
use eredu_nn::workspace::{
    WorkspaceMetadataAllocation, WorkspaceModelControl, WorkspaceModelControlPhase,
};
use eredu_runtime::replicated_session::{
    SessionModelControlCursor, SessionModelControlPlan, SessionTransactionControlError,
};
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, OriginalExternalSpeculativeRole, OriginalHostSourceCustody,
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeRequest, OriginalSpeculativeRole,
    WorkingMemoryError,
};

/// Closed adapters retain the actual admitted role; equal geometry is not authority.
#[derive(Clone)]
pub(crate) enum SpeculativeModelRole {
    Autoregressive(OriginalSpeculativeRole),
    Embedded(OriginalEmbeddedSpeculativeRole),
    External(OriginalExternalSpeculativeRole),
}
impl SpeculativeModelRole {
    pub(crate) fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        match self {
            Self::Autoregressive(role) => role.budget_custody(),
            Self::Embedded(role) => role.budget_custody(),
            Self::External(role) => role.budget_custody(),
        }
    }
    pub(crate) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Autoregressive(role) => role.validate_execution(execution),
            Self::Embedded(role) => role.validate_execution(execution),
            Self::External(role) => role.validate_execution(execution),
        }
    }
}
pub(super) enum RequestOwner {
    Text(OriginalParallelControlRequest),
    Speculative(PreparedSpeculativeControl),
}
impl std::ops::Deref for RequestOwner {
    type Target = OriginalParallelControlRequest;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Text(request) => request,
            Self::Speculative(source) => source.request(),
        }
    }
}
pub(super) enum NativeOwner {
    Text {
        bank: BankOwner,
        execution: InferenceRequest,
        controls: OriginalTextControlGuard,
    },
    Speculative {
        role: SpeculativeModelRole,
        budget: OriginalBufferBudget,
        observer: OriginalScopeObserver,
        cursor: RefCell<SessionModelControlCursor>,
    },
}
impl NativeOwner {
    pub(super) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Text {
                execution: request, ..
            } => request.validate(execution, request.geometry()),
            Self::Speculative { role, .. } => role.validate_execution(execution),
        }
    }
    pub(super) fn text_execution(&self) -> Result<&InferenceRequest, Error> {
        match self {
            Self::Text { execution, .. } => Ok(execution),
            Self::Speculative { .. } => Err(Error::PredictionScopeUnavailable),
        }
    }
    pub(super) fn text_controls(&self) -> Result<&OriginalTextControlGuard, Error> {
        match self {
            Self::Text { controls, .. } => Ok(controls),
            Self::Speculative { .. } => Err(Error::PredictionScopeUnavailable),
        }
    }
    pub(super) fn host_source_custody(&self) -> OriginalHostSourceCustody {
        match self {
            Self::Text { controls, .. } => controls.clone().into(),
            Self::Speculative { role, .. } => role.budget_custody().into(),
        }
    }
    pub(super) fn claim_model(
        &self,
        invocation: &OriginalParallelControlInvocation,
        funding: &HostMetadataFunding,
    ) -> Result<(), Error> {
        let Self::Speculative { cursor, .. } = self else {
            return Ok(());
        };
        use eredu_runtime::DistributedExecutionPhase;
        let phase = match invocation.claim().event() {
            ParallelControlEvent::Phase(DistributedExecutionPhase::Execution) => {
                WorkspaceModelControlPhase::Execution
            }
            ParallelControlEvent::Phase(DistributedExecutionPhase::BoundarySourceCompletion(
                route,
            )) => WorkspaceModelControlPhase::BoundarySourceCompletion {
                route: route.value(),
            },
            ParallelControlEvent::Phase(DistributedExecutionPhase::BoundarySourceReady(route)) => {
                WorkspaceModelControlPhase::BoundarySourceReady {
                    route: route.value(),
                }
            }
            _ => return Err(Error::PrefillScopeUnavailable),
        };
        let state = invocation.state();
        let group = state
            .retained
            .manifest()
            .groups()
            .get(state.group_order)
            .ok_or(Error::PrefillScopeUnavailable)?
            .id();
        cursor
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .claim(WorkspaceModelControl { group, phase })
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        Ok(())
    }
}
impl OriginalParallelControlOwner {
    pub(crate) fn speculative_activation_control_bytes() -> Option<usize> {
        Self::speculative_owner_control_bytes()?.checked_add(Self::installation_control_bytes()?)
    }
    fn speculative_owner_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Owner>(),
            size_of::<NativeOwner>(),
            size_of::<SpeculativeModelRole>(),
            size_of::<OriginalSpeculativeBudgetCustody>(),
            size_of::<&super::super::speculative::ModelRuntimeFunding>(),
            size_of::<Result<&HostMetadataFunding, Error>>(),
            size_of::<RequestOwner>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Custody>(),
            Layout::new::<[usize; 2]>()
                .extend(Layout::new::<Owner>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
            size_of::<(
                &PreparedSpeculativeControl,
                &OriginalSpeculativeRequest,
                &SpeculativeModelRole,
                &OriginalBufferBudget,
                &OriginalScopeObserver,
                SessionModelControlPlan,
                &HostMetadataFunding,
            )>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            SessionModelControlPlan::control_bytes()?,
            eredu_nn::workspace::WorkspaceContext::metadata_source_bytes::<
                SessionTransactionControlError,
            >()?,
            failure_control_bytes()?,
            OriginalScopeObserver::control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn new_speculative_role(
        source: &PreparedSpeculativeControl,
        request: &OriginalSpeculativeRequest,
        role: &SpeculativeModelRole,
        budget: &OriginalBufferBudget,
        observer: &OriginalScopeObserver,
        plan: SessionModelControlPlan,
        funding: &HostMetadataFunding,
        runtime: &super::super::speculative::ModelRuntimeFunding,
    ) -> Result<Self, Error> {
        let model_funding = runtime.validate(source)?;
        model_funding
            .reserve_metadata(Self::speculative_owner_control_bytes().ok_or_else(overflow)?)?;
        source.validate_model_custody(request, &role.budget_custody())?;
        if !source.request().funding.same_account(funding)
            || !OriginalScopeObserver::require_current()?.same_scope(observer)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(Self(Some(Rc::new(Owner {
            request: RequestOwner::Speculative(source.clone()),
            native: NativeOwner::Speculative {
                role: role.clone(),
                budget: budget.clone(),
                observer: observer.clone(),
                cursor: RefCell::new(plan.into_cursor()),
            },
            token_sources: RefCell::new(None),
            running: Cell::new(false),
            failed: Cell::new(false),
            custody: Custody {
                source: source.request().retained.clone(),
                raw: role.budget_custody().into(),
                funding: funding.clone(),
                model_funding: Some(model_funding.clone()),
            },
        }))))
    }
    /// Closes the exact model sequence before enclosing transaction completion.
    /// A failed prefix is consumed; escaped weak projections gain no new claim.
    pub(crate) fn finish_model(&self, success: bool) -> Result<(), Error> {
        let owner = self.owner();
        let NativeOwner::Speculative { cursor, .. } = &owner.native else {
            return Err(Error::PrefillScopeUnavailable);
        };
        if success && owner.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let result = cursor
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .finish(success)
            .map_err(|cause| Error::Neural(owner.custody.model_funding().metadata_source(cause)));
        if !success || result.is_err() {
            owner.failed.set(true);
        }
        result
    }
}

impl OriginalParallelControlOwner {
    pub(crate) fn model_context_control_bytes<T, E, F>(
        source: &PreparedSpeculativeControl,
    ) -> Result<usize, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        source
            .request()
            .funding
            .reserve_metadata(size_of::<(&PreparedSpeculativeControl, Result<usize, Error>)>())?;
        OriginalParallelControlProjection::context_control_bytes::<T, E, F>()
            .and_then(|n| n.checked_add(super::super::super::parallel::OriginalParallelBinding::control_source_validation_control_bytes()?))
            .ok_or_else(overflow)
    }
    /// The one inner agreement owns both immutable native vote alternatives.
    /// Exact generic wrappers below are the same factories used by with_group.
    pub(crate) fn model_call_requirements<T, E, F>(
        prepared: &PreparedSpeculativeControl,
        group_id: CollectiveGroupId,
    ) -> Result<GatherRequirements, Error>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        use crate::backend::submission_recovery::native_role::{self, NativeRoleCapacity};
        let request = prepared.request();
        request.funding.reserve_metadata(size_of::<(
            &PreparedSpeculativeControl,
            CollectiveGroupId,
            Result<GatherRequirements, Error>,
        )>())?;
        let actual = request
            .source
            .model()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let source = actual.communication_source()?;
        let inputs = actual
            .agreement_inputs()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let selected = source
            .source()
            .manifest()
            .select_group_operation(group_id, CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        let group = source
            .group(selected.order())
            .ok_or(Error::PrefillScopeUnavailable)?
            .0;
        let stream = group
            .retained_transport_stream()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let vote = inputs.requirements_for_group(&source, group_id)?;
        let source_loan = OriginalParallelSource::communication_source_funded_control_bytes()
            .ok_or_else(overflow)?;
        let comparison = StreamCopyPlan::<()>::capture(stream)
            .map_err(|_| Error::PrefillScopeUnavailable)?
            .source_comparison_control_bytes()
            .ok_or_else(overflow)?;
        let capacity = NativeRoleCapacity {
            graph: vote.capacity.graph,
            records: vote.capacity.records,
            backing: vote.capacity.backing,
        };
        let parts = [
            OriginalParallelControlRequest::prepare_control_bytes().ok_or_else(overflow)?,
            source_loan,
            super::super::source::agreement_capacity_for_group_control_bytes()
                .ok_or_else(overflow)?,
            vote.capacity_metadata.checked_mul(2).ok_or_else(overflow)?,
            group
                .retention_copy_bytes()
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(overflow)?,
            native_role::control_bytes::<OriginalParallelControlInvocation, Custody>(
                capacity, None,
            )
            .map_err(|_| Error::PrefillScopeUnavailable)?,
            model_callbacks::group_controls::<T, E, F>().ok_or_else(overflow)?,
            OriginalParallelControlProjection::group_control_bytes::<T, E, F>()
                .ok_or_else(overflow)?,
            OriginalControlBinding::group_control_bytes::<T, E>(size_of::<F>())
                .ok_or_else(overflow)?,
            OriginalControlBinding::agreement_control_bytes().ok_or_else(overflow)?,
            OriginalControlBinding::loan_control_bytes()
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(overflow)?,
            OriginalControlBinding::validation_control_bytes()
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(overflow)?,
            comparison
                .checked_add(size_of::<[usize; 1]>())
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(overflow)?,
            source_loan.checked_mul(4).ok_or_else(overflow)?,
            vote.execution_metadata,
            size_of::<(
                &NativeOwner,
                &OriginalParallelControlInvocation,
                &HostMetadataFunding,
            )>(),
            size_of::<std::cell::RefMut<'_, SessionModelControlCursor>>(),
            size_of::<Running<'_>>(),
            size_of::<Result<(), Error>>(),
            failure_control_bytes().ok_or_else(overflow)?,
            eredu_nn::workspace::WorkspaceContext::metadata_source_bytes::<
                SessionTransactionControlError,
            >()
            .ok_or_else(overflow)?,
        ];
        let metadata = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(GatherRequirements {
            capacity: vote.capacity,
            metadata,
        })
    }
}
