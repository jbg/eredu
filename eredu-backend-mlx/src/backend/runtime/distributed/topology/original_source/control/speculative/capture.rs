//! A lexical loan from the accepted independent model into the shared protocol.
use super::super::role::{CaptureSourceCustody, CaptureSourceOwner};
use super::*;
use crate::backend::submission_recovery::native_role::{self, NativeRoleCapacity};
use eredu_runtime::working_memory::{
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeRequest, OriginalSpeculativeRole,
};
use safemlx::OriginalBufferBudget;

#[derive(Clone, Debug)]
pub(crate) struct CaptureCustody {
    source: RetainedCommunicationSource,
    role: OriginalSpeculativeBudgetCustody,
    funding: HostMetadataFunding,
}
impl CaptureSourceCustody for CaptureCustody {
    fn source(&self) -> &RetainedCommunicationSource {
        &self.source
    }
    fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
}
struct Loan {
    budget: OriginalBufferBudget,
    observer: OriginalScopeObserver,
}
struct CaptureOwner {
    source: PreparedSpeculativeControl,
    role: OriginalSpeculativeRole,
    loan: RefCell<Option<Loan>>,
    remaining: Cell<AgreementCapacity>,
    activated: Cell<bool>,
    running: Cell<bool>,
    custody: CaptureCustody,
}
/// Source and paid Host custody can outlive execution; the producing native
/// budget is present only during one authenticated enclosing model callback.
pub(crate) struct SpeculativeCaptureOwner(Rc<CaptureOwner>);
impl SpeculativeCaptureOwner {
    pub(crate) fn control_bytes() -> Option<usize> {
        let parts = [
            Layout::new::<[usize; 2]>()
                .extend(Layout::new::<CaptureOwner>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
            size_of::<Self>(),
            size_of::<CaptureCustody>(),
            size_of::<Loan>(),
            size_of::<Option<Loan>>(),
            size_of::<std::cell::RefMut<'_, Option<Loan>>>(),
            size_of::<CaptureActivation>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<CaptureActivation, Error>>(),
            size_of::<(
                &PreparedSpeculativeControl,
                &OriginalSpeculativeRequest,
                &OriginalSpeculativeRole,
                HostMetadataFunding,
                AgreementCapacity,
            )>(),
            size_of::<(
                &Self,
                &OriginalSpeculativeRole,
                &OriginalBufferBudget,
                &OriginalScopeObserver,
            )>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn new(
        source: &PreparedSpeculativeControl,
        request: &OriginalSpeculativeRequest,
        role: &OriginalSpeculativeRole,
        funding: HostMetadataFunding,
        capacity: AgreementCapacity,
    ) -> Result<Self, Error> {
        funding.reserve_metadata(Self::control_bytes().ok_or_else(overflow)?)?;
        request
            .validate_pool(&source.body().pool)
            .map_err(Error::PrefillControl)?;
        request
            .validate_model_role(role)
            .map_err(Error::PrefillControl)?;
        role.validate_execution(&source.body().execution)
            .map_err(Error::PrefillControl)?;
        if source.body().failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(Self(Rc::new(CaptureOwner {
            source: source.clone(),
            role: role.clone(),
            loan: RefCell::new(None),
            remaining: Cell::new(capacity),
            activated: Cell::new(false),
            running: Cell::new(false),
            custody: CaptureCustody {
                source: source.body().request.retained.clone(),
                role: role.budget_custody(),
                funding,
            },
        })))
    }
    pub(crate) fn activate(
        &self,
        role: &OriginalSpeculativeRole,
        budget: &OriginalBufferBudget,
        observer: &OriginalScopeObserver,
    ) -> Result<CaptureActivation, Error> {
        if !self.0.role.same_role(role)
            || self.0.source.body().failed.get()
            || !OriginalScopeObserver::require_current()?.same_scope(observer)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        let mut slot = self
            .0
            .loan
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        if slot.is_some() || self.0.activated.replace(true) {
            return Err(Error::PrefillScopeUnavailable);
        }
        // The child worker still authenticates budget origin against this exact
        // parent before native construction. No ambient producing budget is read.
        *slot = Some(Loan {
            budget: budget.clone(),
            observer: observer.clone(),
        });
        Ok(CaptureActivation {
            owner: self.retained(),
        })
    }
}
pub(crate) struct CaptureActivation {
    owner: SpeculativeCaptureOwner,
}
impl Drop for CaptureActivation {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.owner.0.source.body().failed.set(true);
        }
        // The guard closes the only producing loan before enclosing completion;
        // pending child Recovery retains its own exact native/account custody.
        self.owner.0.loan.borrow_mut().take();
    }
}
impl CaptureSourceOwner for SpeculativeCaptureOwner {
    type Custody = CaptureCustody;
    fn retained(&self) -> Self {
        Self(self.0.clone())
    }
    fn request(&self) -> &OriginalParallelControlRequest {
        &self.0.source.body().request
    }
    fn custody(&self) -> &CaptureCustody {
        &self.0.custody
    }
    fn running(&self) -> &Cell<bool> {
        &self.0.running
    }
    fn failed(&self) -> &Cell<bool> {
        &self.0.source.body().failed
    }
    fn validate_active(&self) -> Result<(), Error> {
        let slot = self
            .0
            .loan
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        let loan = slot.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
        if self.failed().get()
            || !OriginalScopeObserver::require_current()?.same_scope(&loan.observer)
        {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(())
    }
    fn run_native<I: 'static, T, E, F>(
        &self,
        invocation: I,
        capacity: AgreementCapacity,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>,
    {
        self.validate_active()
            .map_err(Error::into_backend_failure)?;
        let remaining = self.0.remaining.get();
        let Some(next) = (|| {
            Some(AgreementCapacity {
                graph: remaining.graph.checked_sub(capacity.graph)?,
                records: remaining.records.checked_sub(capacity.records)?,
                backing: remaining.backing.checked_sub(capacity.backing)?,
            })
        })() else {
            return Err(Error::PrefillScopeUnavailable.into_backend_failure());
        };
        self.0.remaining.set(next);
        let loan = self.0.loan.borrow();
        let loan = loan.as_ref().expect("authenticated lexical loan");
        let timeout = self
            .custody()
            .source
            .manifest()
            .completion_policy()
            .ok_or_else(|| Error::PrefillScopeUnavailable.into_backend_failure())?
            .timeout();
        let result = native_role::run_with_prepared_budget(
            invocation,
            NativeRoleCapacity {
                graph: capacity.graph,
                records: capacity.records,
                backing: capacity.backing,
            },
            None,
            &loan.budget,
            &loan.observer,
            self.custody(),
            self.custody().funding(),
            Some(timeout),
            |invocation, context| run(invocation, context.observer()),
        );
        if !matches!(&result, Ok(Ok(_))) {
            self.failed().set(true);
        }
        result
    }
}

mod quote;
