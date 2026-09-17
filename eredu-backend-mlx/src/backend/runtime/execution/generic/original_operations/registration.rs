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
    /// Called only after the same neutral role has accepted and configured its
    /// actual native Scope. No current-observer lookup chooses an ordinary path.
    pub(crate) fn register(
        &self,
        request: &InferenceRequest,
        scope: &safemlx::SubmissionScope,
    ) -> Result<(), Error> {
        self.register_for_retirement(request, scope).map(drop)
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
        if !registry.active.get() {
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
        let registry=self.registry()?.ok_or_else(identity)?;
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
impl RegisteredOriginalScope {
    /// Only the existing owning Recovery finish supplies this terminal result.
    /// Failed, blocked, pending and reentrant paths retain the registry alias.
    pub(crate) fn finish(self, status: crate::backend::submission_recovery::Status) {
        if !status.settled || status.failed || status.blocked {
            return;
        }
        let removed = {
            let Ok(mut scopes) = self.registry.scopes.try_borrow_mut() else {
                return;
            };
            let index = scopes
                .iter()
                .position(|scope| scope.same_scope(&self.observer));
            index.map(|index| scopes.swap_remove(index))
        };
        // The last native Scope can release controls here, outside the loan.
        // The cumulative registration count deliberately remains unchanged.
        drop(removed);
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
}
impl Registry {
    pub(super) fn authenticate(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
        let observer = safemlx::OriginalScopeObserver::require_current()?;
        self.authenticate_observer(&observer)?;
        Ok(observer)
    }
    pub(super) fn authenticate_observer(&self,observer:&safemlx::OriginalScopeObserver)->Result<(),Error> {
        if !self.active.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let belongs = self
            .scopes
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .iter()
            .any(|scope| scope.same_scope(&observer));
        if !belongs {
            return Err(identity());
        }
        Ok(())
    }
}
