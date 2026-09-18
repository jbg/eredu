//! Exact accepted-scope aliases, never a new source or submission authority.
use super::*;

/// Created with the quote after Q. Roles may retain this slot before the first
/// prefill claim installs its bank. It contains no quote, session or model edge.
/// Every escaping clone keeps the weak header alive before independent Q custody.
#[derive(Clone)]
pub(crate) struct OriginalOperationRegistration {
    slot: Rc<Cell<(bool, Option<Weak<Registry>>)>>,
    controls: OriginalTextControlGuard,
}
impl std::fmt::Debug for OriginalOperationRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalOperationRegistration")
            .finish_non_exhaustive()
    }
}
impl OriginalOperationRegistration {
    #[cfg(test)]
    pub(crate) fn test_scope_counts(&self) -> Result<(usize, usize), Error> {
        let registry = self.registry()?.ok_or(Error::PrefillScopeUnavailable)?;
        let scopes = registry.scopes.try_borrow().map_err(|_| Error::PrefillScopeReentrant)?;
        Ok((registry.registered_scopes.get(), scopes.len()))
    }
    #[cfg(test)]
    pub(crate) fn test_with_scope_loan<T>(&self, operation: impl FnOnce() -> T) -> T {
        let registry = self.registry().unwrap().unwrap();
        let _loan = registry.scopes.borrow();
        operation()
    }
    pub(crate) fn new(controls: OriginalTextControlGuard) -> Self {
        Self {
            slot: Rc::new(Cell::new((false, None))),
            controls,
        }
    }
    pub(super) fn install(&self, registry: &Rc<Registry>) -> Result<(), Error> {
        self.controls
            .validate_reservation(registry.request.memory_reservation().ok_or_else(identity)?)
            .map_err(memory)?;
        let (retired, previous) = self.slot.take();
        if retired {
            self.slot.set((retired, previous));
            return Err(Error::PrefillScopeUnavailable);
        }
        if previous.as_ref().is_some_and(|old| old.strong_count() != 0) {
            self.slot.set((retired, previous));
            return Err(Error::PrefillScopeReentrant);
        }
        self.slot.set((false, Some(Rc::downgrade(registry))));
        // The Cell contains no loan while the last old weak header retires.
        drop(previous);
        Ok(())
    }
    pub(super) fn clear(&self, registry: &Rc<Registry>) {
        let (retired, previous) = self.slot.take();
        if previous
            .as_ref()
            .is_some_and(|old| Weak::ptr_eq(old, &Rc::downgrade(registry)))
        {
            self.slot.set((true, None));
            drop(previous);
        } else {
            self.slot.set((retired, previous));
        }
    }
    fn registry(&self) -> Result<Option<Rc<Registry>>, Error> {
        let (retired, weak) = self.slot.take();
        let live = weak.as_ref().map(Weak::upgrade);
        self.slot.set((retired, weak));
        if retired {
            return Err(Error::PrefillScopeUnavailable);
        }
        match live {
            None => Ok(None), // Resident execution needs no unit-operation bank.
            Some(Some(live)) => Ok(Some(live)),
            Some(None) => Err(Error::PrefillScopeUnavailable),
        }
    }
    /// Same registration with an exact alias carried to the owning role's finish.
    /// Dropping this handle alone cannot remove a failed or unfinished role.
    pub(crate) fn register_for_retirement(
        &self,
        request: &InferenceRequest,
        scope: &safemlx::SubmissionScope,
    ) -> Result<Option<RegisteredOriginalScope>, Error> {
        let Some(registry) = self.registry()? else {
            return Ok(None);
        };
        registry.check_retirement()?;
        if !registry.active.get() || !registry.entered.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        registry
            .request
            .validate_same_request(request)
            .map_err(memory)?;
        self.controls
            .validate_reservation(request.memory_reservation().ok_or_else(identity)?)
            .map_err(memory)?;
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        if !observer.belongs_to(scope) {
            return Err(identity());
        }
        // Native calls and request/account checks precede the finite storage loan.
        let mut scopes = registry
            .scopes
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        if !scopes.iter().any(|old| old.same_scope(&observer)) {
            let issued = registry.registered_scopes.get();
            if issued == registry.scope_limit {
                return Err(Error::PrefillScopeUnavailable);
            }
            // Removal of terminal aliases never restores this request's count.
            registry.registered_scopes.set(issued + 1);
            scopes.push(observer.clone());
        }
        drop(scopes);
        Ok(Some(RegisteredOriginalScope {
            observer,
            registry,
            _controls: self.controls.clone(),
        }))
    }
    /// Authenticate a sibling consumer through the existing model completion
    /// owner. A resident sequential model has no unit-operation registry; when
    /// a registry exists, its exact request and active scope must agree too.
    pub(crate) fn authenticate_scope(
        &self,
        request: &InferenceRequest,
        scope: &safemlx::SubmissionScope,
        roots: &crate::backend::submission_recovery::prefill::TransientRootsProjection,
    ) -> Result<safemlx::OriginalScopeObserver, Error> {
        self.controls
            .validate_reservation(request.memory_reservation().ok_or_else(identity)?)
            .map_err(memory)?;
        // An expired or terminally cleared installation remains an error. Only
        // the deliberately absent unit bank uses the model owner on its own.
        let registry = self.registry()?;
        let observer = roots.authenticate_scope(scope)?;
        if let Some(registry) = registry {
            registry
                .request
                .validate_same_request(request)
                .map_err(memory)?;
            let registered = registry.authenticate()?;
            if !registered.same_scope(&observer) {
                return Err(identity());
            }
        }
        Ok(observer)
    }
    /// Loans the exact installed model registry, including resident execution.
    /// Deliberately absent or retired registrations cannot create an owner.
    pub(crate) fn selected_residency_access(&self)->Result<OriginalSelectedResidencyAccess,Error> {
        let registry=self.registry()?.ok_or(Error::OriginalSourceContract {
            stage: "selected residency operation registry is absent",
            cause: WorkingMemoryError::IdentityMismatch,
        })?;
        OriginalSelectedResidencyAccess::registered(registry,OperationControls::Text(self.controls.clone()))
    }
    pub(crate) fn control_bytes() -> Option<u64> {
        let controls = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Cell<(bool, Option<Weak<Registry>>)>>(),
            size_of::<Rc<Registry>>(),
            size_of::<Option<Rc<Registry>>>(),
            size_of::<Result<Option<Rc<Registry>>, Error>>(),
            size_of::<Option<Weak<Registry>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<safemlx::OriginalScopeObserver, Error>>(),
            size_of::<safemlx::OriginalScopeObserver>(),
            size_of::<(
                &Self,
                &InferenceRequest,
                &safemlx::SubmissionScope,
                &crate::backend::submission_recovery::prefill::TransientRootsProjection,
            )>(),
            size_of::<Result<Option<RegisteredOriginalScope>, Error>>(),
        ]
        .into_iter()
        .try_fold(
            rc_layout::<Cell<(bool, Option<Weak<Registry>>)>>()?,
            usize::checked_add,
        )?;
        u64::try_from(controls)
            .ok()?
            .checked_add(RegisteredOriginalScope::control_bytes()?)
    }
}

/// Exact registered role, with independent Q custody through retirement.
/// This is neither a success witness nor permission to begin another role.
pub(crate) struct RegisteredOriginalScope {
    observer: safemlx::OriginalScopeObserver,
    registry: Rc<Registry>,
    _controls: OriginalTextControlGuard,
}
#[derive(Debug, thiserror::Error)]
pub enum RegisteredScopeRetirementCause {
    #[error(transparent)]
    Native(#[from] safemlx::error::Exception),
    #[error("original scope retirement remains {0:?}")]
    Incomplete(safemlx::SubmissionRetirement),
    #[error("original scope registration is borrowed during retirement")]
    Reentrant,
}
impl RegisteredScopeRetirementCause {
    pub(crate) fn into_error(self) -> Error {
        match self {
            Self::Native(cause) => Error::Exception(cause),
            Self::Incomplete(_) => Error::PrefillScopeUnavailable,
            Self::Reentrant => Error::PrefillScopeReentrant,
        }
    }
}
impl From<RegisteredScopeRetirementCause> for Error {
    fn from(cause: RegisteredScopeRetirementCause) -> Self { cause.into_error() }
}
/// Private move-only delivery. The Rc never enters a public error: synchronous
/// callers take the original typed cause; abandoned delivery retains it in the
/// fenced bank. Only native/fixed retirement causes enter this owner.
pub(crate) struct RegisteredScopeRetirementFailure {
    registration: RegisteredOriginalScope,
    cause: Option<RegisteredScopeRetirementCause>,
}
impl std::fmt::Debug for RegisteredScopeRetirementFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredScopeRetirementFailure").field("cause", &self.cause).finish()
    }
}
impl RegisteredScopeRetirementFailure {
    pub(crate) fn into_cause(mut self) -> RegisteredScopeRetirementCause {
        self.cause.take().expect("retained retirement cause")
    }
    pub(crate) fn into_error(self) -> Error { self.into_cause().into_error() }
}
impl Drop for RegisteredScopeRetirementFailure {
    fn drop(&mut self) {
        if let Some(cause) = self.cause.take() {
            let previous = self.registration.registry.retirement_failure.take();
            self.registration.registry.retirement_failure.set(Some(previous.unwrap_or(cause)));
        }
    }
}
impl RegisteredOriginalScope {
    /// The same Recovery node has already destroyed its probe and actual
    /// payload. Deferred observed payloads keep this owner in their same node.
    /// A cleanup refusal fences the bank and preserves its first typed cause.
    pub(crate) fn finish_after_payload(self, status: Option<crate::backend::submission_recovery::Status>) -> Result<(), RegisteredScopeRetirementFailure> {
        let Some(status) = status else { return Ok(()); };
        if !status.settled || status.failed || status.blocked { return Ok(()); }
        if let Err(cause) = self.finish(status) {
            self.registry.active.set(false);
            return Err(RegisteredScopeRetirementFailure { registration: self, cause: Some(cause) });
        }
        Ok(())
    }

    /// Only the existing owning Recovery finish supplies this terminal result.
    /// Failed, blocked, pending and reentrant paths retain the registry alias.
    pub(crate) fn finish(&self, status: crate::backend::submission_recovery::Status) -> Result<(), RegisteredScopeRetirementCause> {
        if !status.settled || status.failed || status.blocked {
            return Ok(());
        }
        crate::backend::submission_recovery::retirement::complete(&self.observer)?;
        let removed = {
            let mut scopes = self.registry.scopes.try_borrow_mut()
                .map_err(|_| RegisteredScopeRetirementCause::Reentrant)?;
            let index = scopes
                .iter()
                .position(|scope| scope.same_scope(&self.observer));
            index.map(|index| scopes.swap_remove(index))
        };
        // The last native Scope can release controls here, outside the loan.
        // The cumulative registration count deliberately remains unchanged.
        drop(removed);
        Ok(())
    }

    pub(crate) fn payload_retirement_control_bytes() -> Option<u64> {
        use crate::backend::submission_recovery::Status;
        let parts = [
            size_of::<Self>(), size_of::<Option<Self>>(), size_of::<&Self>(),
            size_of::<&Registry>(), size_of::<Option<Status>>(), size_of::<Status>(),
            size_of::<Result<(), Error>>(), size_of::<Error>(), size_of::<Option<Error>>(),
            size_of::<std::cell::RefMut<'_, Vec<safemlx::OriginalScopeObserver>>>(),
            size_of::<Option<safemlx::OriginalScopeObserver>>(), size_of::<Option<usize>>(),
            size_of::<bool>(),
            size_of::<RegisteredScopeRetirementFailure>(),
            size_of::<Result<(), RegisteredScopeRetirementFailure>>(),
            size_of::<RegisteredScopeRetirementCause>(), size_of::<Option<RegisteredScopeRetirementCause>>(),
            size_of::<Result<(), RegisteredScopeRetirementCause>>(),
        ];
        let bytes = parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
        u64::try_from(bytes).ok()?.checked_add(crate::backend::submission_recovery::retirement::control_bytes()?)
    }

    pub(crate) fn control_bytes() -> Option<u64> {
        let values = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<crate::backend::submission_recovery::Status>(),
            size_of::<Option<safemlx::OriginalScopeObserver>>(),
            size_of::<Option<usize>>(),
            size_of::<std::cell::RefMut<'static, Vec<safemlx::OriginalScopeObserver>>>(),
        ];
        u64::try_from(
            values
                .into_iter()
                .try_fold(size_of::<[usize; 6]>(), usize::checked_add)?,
        )
        .ok()
    }
}

pub(super) struct Registry {
    pub(super) request: OperationRequest,
    pub(super) scopes: RefCell<Vec<safemlx::OriginalScopeObserver>>,
    pub(super) scope_limit: usize,
    pub(super) registered_scopes: Cell<usize>,
    pub(super) active: Cell<bool>,
    pub(super) entered: Cell<bool>,
    pub(super) retirement_failure: Cell<Option<RegisteredScopeRetirementCause>>,
}
impl Registry {
    pub(super) fn check_retirement(&self) -> Result<(), Error> {
        match self.retirement_failure.take() { Some(cause) => Err(cause.into_error()), None => Ok(()) }
    }
    pub(super) fn authenticate(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        self.authenticate_observer(&observer)?;
        Ok(observer)
    }
    pub(super) fn authenticate_observer(&self,observer:&safemlx::OriginalScopeObserver)->Result<(),Error> {
        self.check_retirement()?;
        if !self.active.get() || !self.entered.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let belongs = self
            .scopes
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .iter()
            .any(|scope| scope.same_scope(&observer));
        if !belongs {
            return Err(Error::OriginalSourceContract {
                stage: "selected residency current scope is not registered",
                cause: WorkingMemoryError::IdentityMismatch,
            });
        }
        Ok(())
    }
}
