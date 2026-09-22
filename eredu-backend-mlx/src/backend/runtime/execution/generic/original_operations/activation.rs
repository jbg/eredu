//! Lexical use of a retained request bank, independent of its source lifetime.
use super::*;

/// Held by the actual submission resources until their completion cut. An
/// inactive branch keeps its bank, spent slots, registration and original Q.
pub(crate) struct OriginalOperationActivation {
    registry: Rc<Registry>,
}
impl Drop for OriginalOperationActivation {
    fn drop(&mut self) {
        self.registry.entered.set(false);
    }
}
impl OriginalOperationActivation {
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Option<Self>, Error>>(),
            size_of::<&OriginalOperationBankOwner>(),
            size_of::<&dyn ErasedOwner>(),
            size_of::<&OperationControls>(),
            size_of::<&Registry>(),
            size_of::<Rc<Registry>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Error>(),
            size_of::<Option<Error>>(),
            size_of::<RegisteredScopeRetirementCause>(),
            size_of::<Option<RegisteredScopeRetirementCause>>(),
            size_of::<&Cell<Option<RegisteredScopeRetirementCause>>>(),
            size_of::<std::cell::Ref<'static, Vec<safemlx::OriginalScopeObserver>>>(),
            size_of::<std::cell::Ref<'static, Option<OriginalOperationBankOwner>>>(),
            size_of::<std::cell::RefMut<'static, Option<Self>>>(),
            size_of::<Option<&OriginalOperationBankOwner>>(),
            size_of::<(&InferenceRequest, &InferenceTextStep)>(),
            size_of::<bool>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl Registry {
    pub(super) fn require_idle(&self) -> Result<(), Error> {
        self.check_retirement()?;
        if !self.active.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        if self.entered.get()
            || !self
                .scopes
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .is_empty()
        {
            return Err(Error::PrefillScopeReentrant);
        }
        Ok(())
    }
    pub(super) fn enter(self: &Rc<Self>) -> OriginalOperationActivation {
        let previous = self.entered.replace(true);
        debug_assert!(!previous);
        OriginalOperationActivation {
            registry: Rc::clone(self),
        }
    }
}
impl<U: 'static> Bank<U> {
    fn require_idle(&self) -> Result<(), Error> {
        self.registry.require_idle()?;
        if !self
            .pending
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?
            .is_empty()
        {
            return Err(Error::PrefillScopeReentrant);
        }
        let background = self
            .background
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        if background.as_ref().is_some_and(|value| !value.is_idle()) {
            return Err(Error::PrefillScopeReentrant);
        }
        // A live preparation/access loan is not an idle source, even before it
        // has published a lease into the pending queue.
        drop(
            self.prepared
                .try_borrow_mut()
                .map_err(|_| Error::PrefillScopeReentrant)?,
        );
        Ok(())
    }
}
impl<U: 'static> OriginalOperationSlot<U> {
    pub(super) fn require_bounded_idle(&self) -> Result<(), Error> {
        let view = self.take();
        let bank = view.as_ref().and_then(|view| view.value.upgrade());
        self.set(view);
        match bank {
            Some(bank) => bank.require_idle(),
            None => Ok(()),
        }
    }
}
impl OriginalOperationBankOwner {
    pub(crate) fn activate(&self) -> Result<OriginalOperationActivation, Error> {
        self.value.activate(&self.controls)
    }
}
impl<U: 'static> OwnedBank<U> {
    pub(super) fn activate_text(
        &self,
        controls: &OperationControls,
    ) -> Result<OriginalOperationActivation, Error> {
        if self.registration.is_none() || !matches!(controls, OperationControls::Text(_)) {
            return Err(Error::OriginalSourceContract {
                stage: "text operation activation source",
                cause: WorkingMemoryError::IdentityMismatch,
            });
        }
        controls.validate(&self.bank.registry)?;
        self.bank.require_idle()?;
        self.slot.require_bounded_idle()?;
        let view = OriginalOperationProjection {
            value: Rc::downgrade(&self.bank),
            identity: Arc::clone(&self.bank.identity),
            controls: controls.clone(),
        };
        let previous = self.slot.replace(Some(view));
        let guard = self.bank.registry.enter();
        drop(previous);
        Ok(guard)
    }
}

pub(super) fn typed_control_bytes<U: 'static>() -> Option<usize> {
    let frames = [
        size_of::<&Bank<U>>(),
        size_of::<&OwnedBank<U>>(),
        size_of::<&OriginalOperationSlot<U>>(),
        size_of::<Option<OriginalOperationProjection<U>>>(),
        size_of::<OriginalOperationProjection<U>>(),
        size_of::<Option<Rc<Bank<U>>>>(),
        size_of::<Rc<Bank<U>>>(),
        size_of::<std::cell::Ref<'static, VecDeque<MlxUnitLease<U>>>>(),
        size_of::<std::cell::RefMut<'static, PreparedStorage<U>>>(),
        size_of::<
            std::cell::Ref<
                'static,
                Option<crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>,
            >,
        >(),
        size_of::<
            Option<&crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>,
        >(),
        size_of::<&crate::backend::runtime::residency::dense_stream::BackgroundHostCoordinator>(),
        size_of::<Result<(), Error>>(),
        size_of::<bool>(),
    ];
    frames.into_iter().try_fold(
        OriginalOperationActivation::control_bytes()?
            .checked_add(std::mem::size_of_val(&frames))?,
        usize::checked_add,
    )
}
