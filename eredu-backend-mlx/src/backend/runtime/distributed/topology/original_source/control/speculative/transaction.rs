//! Exact shared-session control callbacks under one admitted model role.
use super::super::role::{CaptureSourceCustody, CaptureSourceOwner};
use super::capture::CaptureCustody;
use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::submission_recovery::native_role::{self, NativeRoleCapacity};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::replicated_session::{
    ParallelControlCallbackVisitor, SessionTransactionControlCursor,
    SessionTransactionControlError, SessionTransactionControlOccurrence,
    SessionTransactionControlPlan,
};
use eredu_runtime::working_memory::{OriginalSpeculativeRequest, OriginalSpeculativeRole};
use safemlx::OriginalBufferBudget;

fn callback<T, E, F>(run: F) -> impl FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>
where
    F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
{
    move |group, funding| run(Some((group, funding)))
}

fn callback_controls<T, E, F>() -> Option<usize>
where
    F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
{
    fn describe<T, E, F, R>(_: impl FnOnce(F) -> R) -> Option<usize>
    where
        R: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
    {
        let context =
            <SpeculativeCaptureOwner as CaptureSourceOwner>::agreement_context_size::<T, E, R>();
        native_role::callback_control_bytes::<T, E>(context)?.checked_add(
            OriginalParallelControlInvocation::context_control_bytes::<T, E>(size_of::<R>())?,
        )
    }
    describe(callback::<T, E, F>)
}

/// Native capacities come from both actual immutable status inputs. The
/// generic callback is the same type visited by the neutral session producer.
impl PreparedSpeculativeControl {
    pub(crate) fn transaction_call_requirements<T, E, F>(
        &self,
    ) -> Result<super::super::role::GatherRequirements, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let body = self.body();
        body.funding.reserve_metadata(size_of::<(
            &Self,
            AgreementCapacity,
            NativeRoleCapacity,
            super::super::role::GatherRequirements,
            Result<super::super::role::GatherRequirements, Error>,
        )>())?;
        if body.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let actual = body
            .request
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
            .select_group_operation(inputs.group(), CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        let group = source
            .group(selected.order())
            .ok_or(Error::PrefillScopeUnavailable)?
            .0;
        let stream = group
            .retained_transport_stream()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let vote = inputs.requirements_for_group(&source, inputs.group())?;
        let capacity = NativeRoleCapacity {
            graph: vote.capacity.graph,
            records: vote.capacity.records,
            backing: vote.capacity.backing,
        };
        let source_loan = OriginalParallelSource::communication_source_funded_control_bytes()
            .ok_or_else(overflow)?;
        let comparison = StreamCopyPlan::<()>::capture(stream)
            .map_err(|_| Error::PrefillScopeUnavailable)?
            .source_comparison_control_bytes()
            .ok_or_else(overflow)?;
        let parts = [
            OriginalParallelControlRequest::prepare_control_bytes().ok_or_else(overflow)?,
            source_loan,
            super::super::source::agreement_capacity_for_group_control_bytes()
                .ok_or_else(overflow)?,
            vote.capacity_metadata.checked_mul(2).ok_or_else(overflow)?,
            group
                .retention_copy_bytes()
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(size_of::<[usize; 1]>()))
                .ok_or_else(overflow)?,
            native_role::control_bytes::<OriginalParallelControlInvocation, CaptureCustody>(
                capacity, None,
            )
            .map_err(|_| Error::PrefillScopeUnavailable)?,
            callback_controls::<T, E, F>().ok_or_else(overflow)?,
            OriginalControlBinding::agreement_control_bytes().ok_or_else(overflow)?,
            OriginalControlBinding::loan_control_bytes().ok_or_else(overflow)?,
            OriginalControlBinding::validation_control_bytes().ok_or_else(overflow)?,
            comparison
                .checked_add(size_of::<[usize; 1]>())
                .ok_or_else(overflow)?,
            source_loan,
            source_loan,
            vote.execution_metadata,
            SpeculativeTransactionControl::call_control_bytes::<T, E, F>().ok_or_else(overflow)?,
            eredu_nn::workspace::WorkspaceContext::metadata_source_bytes::<
                SessionTransactionControlError,
            >()
            .ok_or_else(overflow)?,
        ];
        let metadata = parts
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(super::super::role::GatherRequirements {
            capacity: vote.capacity,
            metadata,
        })
    }
}

/// One immutable source quote for one actual transaction. A clone of the
/// descriptive plan cannot repeat this owner's producing activation.
pub(crate) struct SpeculativeTransactionQuote {
    source: PreparedSpeculativeControl,
    plan: SessionTransactionControlPlan,
    capacity: AgreementCapacity,
    metadata: usize,
    used: Cell<bool>,
}

pub(crate) struct SpeculativeTransactionControl {
    cursor: RefCell<SessionTransactionControlCursor>,
    _activation: CaptureActivation,
    owner: SpeculativeCaptureOwner,
}

impl SpeculativeTransactionQuote {
    pub(crate) fn new(
        source: &PreparedSpeculativeControl,
        plan: SessionTransactionControlPlan,
        capacity: AgreementCapacity,
        metadata: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        funding.reserve_metadata(size_of::<(
            Self,
            Result<Self, Error>,
            &PreparedSpeculativeControl,
            SessionTransactionControlPlan,
            AgreementCapacity,
            usize,
            &HostMetadataFunding,
        )>())?;
        Ok(Self {
            source: source.clone(),
            plan,
            capacity,
            metadata: metadata
                .checked_add(Self::activation_control_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?,
            used: Cell::new(false),
        })
    }

    fn activation_control_bytes() -> Option<usize> {
        let parts = [
            SpeculativeCaptureOwner::control_bytes()?,
            Self::activation_frames()?,
            SpeculativeTransactionControl::finish_control_bytes()?.checked_mul(2)?,
            eredu_nn::workspace::WorkspaceContext::metadata_source_bytes::<
                SessionTransactionControlError,
            >()?
            .checked_mul(2)?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    fn activation_frames() -> Option<usize> {
        let parts = [
            SessionTransactionControlPlan::control_bytes()?,
            size_of::<SpeculativeTransactionControl>(),
            size_of::<Option<SpeculativeTransactionControl>>(),
            size_of::<Result<SpeculativeTransactionControl, Error>>(),
            size_of::<(
                &Self,
                &OriginalSpeculativeRequest,
                &OriginalSpeculativeRole,
                &OriginalBufferBudget,
                &OriginalScopeObserver,
                &HostMetadataFunding,
            )>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    pub(crate) fn metadata_bytes(&self) -> usize {
        self.metadata
    }
    pub(crate) fn backing_bytes(&self) -> usize {
        self.capacity.backing
    }

    pub(crate) fn activate(
        &self,
        request: &OriginalSpeculativeRequest,
        role: &OriginalSpeculativeRole,
        budget: &OriginalBufferBudget,
        observer: &OriginalScopeObserver,
        funding: &HostMetadataFunding,
    ) -> Result<SpeculativeTransactionControl, Error> {
        funding.reserve_metadata(Self::activation_frames().ok_or_else(overflow)?)?;
        if self.used.replace(true) {
            return Err(Error::PrefillScopeUnavailable);
        }
        let owner = SpeculativeCaptureOwner::new(
            &self.source,
            request,
            role,
            funding.clone(),
            self.capacity,
        )?;
        let activation = owner.activate(role, budget, observer)?;
        Ok(SpeculativeTransactionControl {
            cursor: RefCell::new(self.plan.cursor()),
            _activation: activation,
            owner,
        })
    }
}

impl SpeculativeTransactionControl {
    pub(crate) fn run<T, E, F>(
        &self,
        event: ParallelControlEvent,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.run_inner(event, stream, run)
            .map_err(Error::into_backend_failure)
    }

    fn run_inner<T, E, F>(
        &self,
        event: ParallelControlEvent,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        let funding = self.owner.custody().funding();
        funding.reserve_metadata(Self::call_control_bytes::<T, E, F>().ok_or_else(overflow)?)?;
        self.owner.validate_active()?;
        self.cursor
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .claim(event)
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let invocation = self.owner.prepare_agreement(event, None).inspect_err(|_| {
            self.owner.failed().set(true);
        })?;
        if self.owner.failed().get() || self.owner.running().replace(true) {
            self.owner.failed().set(true);
            return Err(Error::PrefillScopeUnavailable);
        }
        let _running = Running {
            running: self.owner.running(),
            failed: self.owner.failed(),
        };
        self.owner
            .run_agreement(invocation, stream, callback(run))
            .map_err(|cause| Error::with_original_control_source(cause, false))
    }

    pub(crate) fn call_control_bytes<T, E, F>() -> Option<usize> {
        let parts = [
            size_of::<(&Self, ParallelControlEvent, &Stream, F)>(),
            size_of::<Result<T, E>>(),
            size_of::<Result<Result<T, E>, Error>>(),
            size_of::<Result<Result<T, E>, eredu_core::BackendFailure>>(),
            size_of::<std::cell::RefMut<'_, SessionTransactionControlCursor>>(),
            size_of::<OriginalParallelControlInvocation>(),
            size_of::<Result<OriginalParallelControlInvocation, Error>>(),
            failure_control_bytes()?,
            size_of::<Running<'_>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    fn finish_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(&Self, bool)>(),
            size_of::<Result<(), Error>>(),
            size_of::<std::cell::RefMut<'_, SessionTransactionControlCursor>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    pub(crate) fn finish_transaction(&self, success: bool) -> Result<(), Error> {
        self.owner
            .custody()
            .funding()
            .reserve_metadata(Self::finish_control_bytes().ok_or_else(overflow)?)?;
        self.cursor
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .finish_transaction(success)
            .map_err(|cause| Error::Neural(self.owner.custody().funding().metadata_source(cause)))
    }

    pub(crate) fn finish_loan(&self, success: bool) -> Result<(), Error> {
        self.owner
            .custody()
            .funding()
            .reserve_metadata(Self::finish_control_bytes().ok_or_else(overflow)?)?;
        self.cursor
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .finish_loan(success)
            .map_err(|cause| Error::Neural(self.owner.custody().funding().metadata_source(cause)))
    }
}

struct Running<'a> {
    running: &'a Cell<bool>,
    failed: &'a Cell<bool>,
}
impl Drop for Running<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.failed.set(true);
        }
        self.running.set(false);
    }
}

impl PreparedSpeculativeControl {
    fn transaction_group_controls<T, E, F>(&self) -> Result<usize, Error> {
        let body = self.body();
        body.funding
            .reserve_metadata(size_of::<(&Self, usize, Result<usize, Error>)>())?;
        let actual = body
            .request
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
            .select_group_operation(inputs.group(), CommunicationOperation::FailureAgreement)
            .map_err(|cause| failure(Cause::Rank(cause), source.source(), source.funding()))?;
        let group = source
            .group(selected.order())
            .ok_or(Error::PrefillScopeUnavailable)?
            .0;
        let stream = group
            .retained_transport_stream()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let parts = [
            OriginalControlBinding::group_control_bytes::<T, E>(size_of::<F>())
                .ok_or_else(overflow)?,
            OriginalControlBinding::loan_control_bytes().ok_or_else(overflow)?,
            OriginalControlBinding::validation_control_bytes().ok_or_else(overflow)?,
            StreamCopyPlan::<()>::capture(stream)
                .map_err(|_| Error::PrefillScopeUnavailable)?
                .source_comparison_control_bytes()
                .ok_or_else(overflow)?,
            size_of::<[usize; 1]>(),
            OriginalParallelSource::communication_source_funded_control_bytes()
                .ok_or_else(overflow)?,
            OriginalParallelSource::communication_source_funded_control_bytes()
                .ok_or_else(overflow)?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(overflow)
    }
}

/// Collects both actual callback layers from the selected shared strategy.
pub(crate) struct SpeculativeTransactionVisitor<'a> {
    source: &'a PreparedSpeculativeControl,
    plan: SessionTransactionControlPlan,
    capacity: AgreementCapacity,
    metadata: usize,
    outer: usize,
    inner: usize,
}
impl<'a> SpeculativeTransactionVisitor<'a> {
    pub(crate) fn new(
        source: &'a PreparedSpeculativeControl,
        plan: SessionTransactionControlPlan,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        funding.reserve_metadata(size_of::<(Self, Result<Self, Error>)>())?;
        Ok(Self {
            source,
            plan,
            capacity: AgreementCapacity {
                graph: 0,
                records: 0,
                backing: 0,
            },
            metadata: 0,
            outer: 0,
            inner: 0,
        })
    }
    pub(crate) fn finish(
        self,
        additional: usize,
        funding: &HostMetadataFunding,
    ) -> Result<SpeculativeTransactionQuote, Error> {
        let count = self.plan.occurrences().count();
        if self.outer != count || self.inner != count {
            return Err(Error::PrefillScopeUnavailable);
        }
        SpeculativeTransactionQuote::new(
            self.source,
            self.plan,
            self.capacity,
            self.metadata.checked_add(additional).ok_or_else(overflow)?,
            funding,
        )
    }
    fn validate(
        &self,
        occurrence: SessionTransactionControlOccurrence,
        expected: usize,
    ) -> Result<(), Error> {
        if occurrence.ordinal() != expected
            || occurrence.operation() != CommunicationOperation::FailureAgreement
            || self.plan.occurrences().nth(expected) != Some(occurrence)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
}
impl ParallelControlCallbackVisitor<MlxNeuralBackend> for SpeculativeTransactionVisitor<'_> {
    type Error = Error;
    fn visit<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Error>
    where
        F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
    {
        self.validate(occurrence, self.outer)?;
        let quote = self.source.transaction_call_requirements::<T, E, F>()?;
        self.capacity = AgreementCapacity {
            graph: self
                .capacity
                .graph
                .checked_add(quote.capacity.graph)
                .ok_or_else(overflow)?,
            records: self
                .capacity
                .records
                .checked_add(quote.capacity.records)
                .ok_or_else(overflow)?,
            backing: self
                .capacity
                .backing
                .checked_add(quote.capacity.backing)
                .ok_or_else(overflow)?,
        };
        self.metadata = self
            .metadata
            .checked_add(quote.metadata)
            .ok_or_else(overflow)?;
        self.outer = self.outer.checked_add(1).ok_or_else(overflow)?;
        Ok(())
    }
    fn visit_group<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Error>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        self.validate(occurrence, self.inner)?;
        self.metadata = self
            .metadata
            .checked_add(self.source.transaction_group_controls::<T, E, F>()?)
            .ok_or_else(overflow)?;
        self.inner = self.inner.checked_add(1).ok_or_else(overflow)?;
        Ok(())
    }
}
