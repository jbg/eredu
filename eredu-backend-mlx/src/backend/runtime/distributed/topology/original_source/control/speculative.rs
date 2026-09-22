//! Source-qualified speculative agreements use the shared native role worker.
use super::*;
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, MemoryLedger, SpeculativePrefillScheduleAuthority,
    WorkingMemoryError,
};
use std::rc::{Rc, Weak};

struct Body {
    request: OriginalParallelControlRequest,
    authority: RefCell<Option<SpeculativePrefillScheduleAuthority>>,
    execution: InferenceExecutionIdentity,
    pool: MemoryLedger,
    active: Cell<usize>,
    failed: Cell<bool>,
    funding: HostMetadataFunding,
}
/// Retains the actual initialized communication source captured before model loans.
pub(crate) struct PreparedSpeculativeControl(Option<Rc<Body>>);
impl Clone for PreparedSpeculativeControl {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for PreparedSpeculativeControl {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
impl PreparedSpeculativeControl {
    fn body(&self) -> &Body {
        self.0.as_deref().expect("live speculative control source")
    }
    pub(super) fn same_control_source(&self, other: &Self) -> bool {
        Rc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    pub(super) fn request(&self) -> &OriginalParallelControlRequest {
        &self.body().request
    }
    pub(super) fn validate_model_role(
        &self,
        request: &eredu_runtime::working_memory::OriginalSpeculativeRequest,
        role: &eredu_runtime::working_memory::OriginalSpeculativeRole,
    ) -> Result<(), Error> {
        request
            .validate_pool(&self.body().pool)
            .map_err(Error::PrefillControl)?;
        request
            .validate_model_role(role)
            .map_err(Error::PrefillControl)?;
        role.validate_execution(&self.body().execution)
            .map_err(Error::PrefillControl)?;
        if self.body().failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
    pub(super) fn validate_model_custody(
        &self,
        request: &eredu_runtime::working_memory::OriginalSpeculativeRequest,
        custody: &eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Result<(), Error> {
        request
            .validate_pool(&self.body().pool)
            .map_err(Error::PrefillControl)?;
        request
            .validate_model_custody(custody)
            .map_err(Error::PrefillControl)?;
        if !request
            .execution_identity()
            .same_execution(&self.body().execution)
            || self.body().failed.get()
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
    /// The immutable model-equation source belonging to this exact control owner.
    /// Borrowing it creates no invocation or permission to submit native work.
    pub(crate) fn workspace_source(&self) -> &OriginalParallelSource {
        self.body()
            .request
            .source
            .model()
            .expect("speculative control retains model source")
    }
    pub(crate) fn new(
        source: OriginalParallelSource,
        execution: &InferenceExecutionIdentity,
        pool: &MemoryLedger,
    ) -> Result<Self, Error> {
        let funding = source.funding().clone();
        reserve(
            &funding,
            &[
                size_of::<Self>(),
                size_of::<Body>(),
                size_of::<Result<Self, Error>>(),
                Layout::new::<[usize; 2]>()
                    .extend(Layout::new::<Body>())
                    .map_err(|_| overflow())?
                    .0
                    .pad_to_align()
                    .size(),
            ],
        )?;
        let request = OriginalParallelControlRequest::new(source)?;
        Ok(Self(Some(Rc::new(Body {
            request,
            authority: RefCell::new(None),
            execution: execution.clone(),
            pool: pool.clone(),
            active: Cell::new(0),
            failed: Cell::new(false),
            funding,
        }))))
    }
    pub(crate) fn bind(
        &self,
        authority: &SpeculativePrefillScheduleAuthority,
    ) -> Result<(), Error> {
        let body = self.body();
        authority
            .validate(&body.execution, authority.geometry())
            .map_err(Error::PrefillControl)?;
        if !body.pool.same_ledger(authority.pool()) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut slot = body
            .authority
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        if let Some(prior) = slot.as_ref() {
            if !prior.same_schedule(authority) {
                return Err(Error::PrefillScopeUnavailable);
            }
        } else {
            *slot = Some(authority.clone());
        }
        Ok(())
    }
    pub(crate) fn projection(&self) -> Result<SpeculativeControlProjection, Error> {
        reserve(
            &self.body().funding,
            &[
                size_of::<SpeculativeControlProjection>(),
                size_of::<Result<SpeculativeControlProjection, Error>>(),
            ],
        )?;
        Ok(SpeculativeControlProjection {
            body: Rc::downgrade(self.0.as_ref().expect("live source")),
            funding: self.body().funding.clone(),
        })
    }
}
#[derive(Clone)]
pub(crate) struct SpeculativeControlProjection {
    body: Weak<Body>,
    funding: HostMetadataFunding,
}
impl SpeculativeControlProjection {
    fn owner(&self) -> Result<PreparedSpeculativeControl, Error> {
        self.body
            .upgrade()
            .map(|body| PreparedSpeculativeControl(Some(body)))
            .ok_or(Error::PrefillScopeUnavailable)
    }
    pub(crate) fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), Error> {
        if !self.owner()?.body().execution.same_execution(execution) {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
    pub(crate) fn matches(
        &self,
        authority: &SpeculativePrefillScheduleAuthority,
    ) -> Result<bool, Error> {
        let owner = self.owner()?;
        let slot = owner
            .body()
            .authority
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        Ok(slot
            .as_ref()
            .is_some_and(|scope| scope.same_schedule(authority)))
    }
    pub(crate) fn activate(&self) -> Result<SpeculativeControlActivation, Error> {
        let owner = self.owner()?;
        let body = owner.body();
        reserve(
            &body.funding,
            &[
                size_of::<SpeculativeControlActivation>(),
                size_of::<Result<SpeculativeControlActivation, Error>>(),
            ],
        )?;
        if body.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        body.active
            .set(body.active.get().checked_add(1).ok_or_else(overflow)?);
        Ok(SpeculativeControlActivation { owner })
    }
    pub(crate) fn is_active(&self) -> bool {
        self.body
            .upgrade()
            .is_some_and(|body| body.active.get() != 0)
    }
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
        let owner = self.owner()?;
        let body = owner.body();
        if body.active.get() == 0 || body.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let scope = body
            .authority
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?
            .clone();
        scope
            .validate(&body.execution, scope.geometry())
            .map_err(Error::PrefillControl)?;
        let invocation = body.request.prepare(event, None)?;
        let capacity = invocation.capacity();
        let source = &body.request.source;
        let inputs = source
            .agreement_inputs()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let runtime = inputs.runtime();
        let timeout = body
            .request
            .retained
            .manifest()
            .completion_policy()
            .ok_or(Error::PrefillScopeUnavailable)?
            .timeout();
        let parent = safemlx::OriginalScopeObserver::try_current()?;
        let operation=|value:&(OriginalParallelControlInvocation,SpeculativePrefillScheduleAuthority),
            native:&crate::backend::submission_recovery::native_role::NativeRoleContext<'_>| {
            value.0.with_context(native.observer(),stream,|group,funding|run(Some((group,funding))))
        };
        reserve(
            &body.funding,
            &[
                size_of_val(&operation),
                size_of::<T>(),
                size_of::<E>(),
                size_of::<Result<T, E>>(),
                size_of::<Result<Result<T, E>, Error>>(),
            ],
        )?;
        let plan = crate::backend::submission_recovery::native_role::cold::Plan::new(
            runtime,
            crate::backend::submission_recovery::native_role::NativeRoleCapacity {
                graph: capacity.graph,
                records: capacity.records,
                backing: capacity.backing,
            },
            None,
            (invocation, scope),
            operation,
        )
        .with_parent(parent, timeout);
        let result = match plan.submit(&body.pool) {
            Ok(submitted) => submitted
                .finish()
                .map_err(|cause| Error::with_original_control_source(cause, false)),
            Err(cause) => {
                let (plan, failure) = cause.into_parts();
                drop(plan);
                let failure = failure.retire_output_and_map_error(|cause| cause);
                Err(Error::with_original_control_source(
                    eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::Other, failure),
                    false,
                ))
            }
        };
        if !matches!(&result, Ok(Ok(_))) {
            body.failed.set(true);
        }
        result
    }
}
pub(crate) struct SpeculativeControlActivation {
    owner: PreparedSpeculativeControl,
}
impl Drop for SpeculativeControlActivation {
    fn drop(&mut self) {
        let body = self.owner.body();
        if std::thread::panicking() {
            body.failed.set(true);
        }
        body.active.set(
            body.active
                .get()
                .checked_sub(1)
                .expect("live control activation"),
        );
    }
}

mod capture;
pub(crate) use capture::{CaptureActivation, SpeculativeCaptureOwner};

mod transaction;
pub(crate) use transaction::{
    SpeculativeTransactionControl, SpeculativeTransactionQuote, SpeculativeTransactionVisitor,
};

mod model;
pub(crate) use model::{
    ModelRuntimeFunding, SpeculativeModelControlQuote, SpeculativeModelControlVisitor,
};
