//! Temporary ownership during native attachment and its reentrant cleanup.
use super::{Error, WorkingMemoryFundingScope};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::{
    cell::{Cell, RefCell, RefMut},
    mem::size_of,
};

fn fenced() -> Error {
    Error::Other(Box::new(WorkingMemoryError::ExecutionFenced))
}

pub(super) fn ensure_idle(active: &Cell<bool>) -> Result<(), Error> {
    if active.get() {
        Err(fenced())
    } else {
        Ok(())
    }
}

pub(super) struct Activity<'a>(&'a Cell<bool>);
impl<'a> Activity<'a> {
    pub(super) fn begin(active: &'a Cell<bool>) -> Result<Self, Error> {
        ensure_idle(active)?;
        active.set(true);
        Ok(Self(active))
    }
}
impl Drop for Activity<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

pub(super) struct OwnedScope<'a> {
    slot: &'a RefCell<Option<WorkingMemoryFundingScope>>,
    scope: Option<WorkingMemoryFundingScope>,
}
impl<'a> OwnedScope<'a> {
    pub(super) fn take(
        slot: &'a RefCell<Option<WorkingMemoryFundingScope>>,
    ) -> Result<Self, Error> {
        let scope = slot
            .try_borrow_mut()
            .map_err(|_| fenced())?
            .take()
            .ok_or_else(|| Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)))?;
        Ok(Self {
            slot,
            scope: Some(scope),
        })
    }
    pub(super) fn get_mut(&mut self) -> &mut WorkingMemoryFundingScope {
        self.scope.as_mut().expect("owned publication scope")
    }
    pub(super) fn get(&self) -> &WorkingMemoryFundingScope {
        self.scope.as_ref().expect("owned publication scope")
    }
}
impl Drop for OwnedScope<'_> {
    fn drop(&mut self) {
        // This slot has only two producers: one-shot prepared activation and
        // this guard. No callback can retain a slot loan or activate this Work
        // again; recursive publication/certification is fenced by Activity.
        let mut slot = self.slot.borrow_mut();
        assert!(slot.is_none(), "exclusive publication scope slot");
        *slot = self.scope.take();
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    [
        size_of::<Activity<'static>>(),
        size_of::<Result<Activity<'static>, Error>>(),
        size_of::<OwnedScope<'static>>(),
        size_of::<Result<OwnedScope<'static>, Error>>(),
        size_of::<WorkingMemoryFundingScope>(),
        size_of::<Option<WorkingMemoryFundingScope>>(),
        size_of::<RefMut<'static, Option<WorkingMemoryFundingScope>>>(),
        size_of::<Result<(), Error>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
