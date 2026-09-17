//! Closed active aliases; semantic payload destruction stays in ordinary retirement.
use super::SessionPayload;
use crate::backend::ordinary_retirement::OrdinaryRetirement;
#[cfg(test)]
use std::cell::Cell;
use std::{ops::Deref, rc::Rc};

// No raw Rc/Weak or owning OrdinaryRetirement escapes this module.
pub(in crate::composition::mlx::session) struct SessionPayloadOwner {
    inner: Option<Rc<OrdinaryRetirement<SessionPayload>>>,
    // Separate from semantic Drop. All clones observe the same final active alias.
    #[cfg(test)]
    active_retired: Rc<Cell<bool>>,
}
impl SessionPayloadOwner {
    pub(in crate::composition::mlx::session) fn new(value: SessionPayload) -> Self {
        Self {
            inner: Some(Rc::new(OrdinaryRetirement::new(value))),
            #[cfg(test)]
            active_retired: Rc::new(Cell::new(false)),
        }
    }

    pub(in crate::composition::mlx::session) fn same_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(
            self.inner.as_ref().expect("live payload owner"),
            other.inner.as_ref().expect("live payload owner"),
        )
    }
    pub(in crate::composition::mlx::session) fn get_mut(&mut self) -> Option<&mut SessionPayload> {
        Rc::get_mut(self.inner.as_mut()?).map(|payload| &mut **payload)
    }
    pub(super) fn is_exclusive(&self) -> bool {
        self.inner.as_ref().is_some_and(|owner|
            Rc::strong_count(owner) == 1 && Rc::weak_count(owner) == 0)
    }

    #[cfg(test)]
    pub(in crate::composition::mlx::session) fn active_owner_count(&self) -> usize {
        Rc::strong_count(self.inner.as_ref().expect("live payload owner"))
    }

    #[cfg(test)]
    pub(in crate::composition::mlx::session) fn retirement_probe(
        &self,
    ) -> impl Fn() -> bool + 'static {
        let retired = self.active_retired.clone();
        move || retired.get()
    }
}
impl Clone for SessionPayloadOwner {
    fn clone(&self) -> Self {
        Self {
            inner: Some(self.inner.as_ref().expect("live payload owner").clone()),
            #[cfg(test)]
            active_retired: self.active_retired.clone(),
        }
    }
}
impl Deref for SessionPayloadOwner {
    type Target = SessionPayload;
    fn deref(&self) -> &SessionPayload {
        self.inner.as_deref().expect("live payload owner")
    }
}

// With no exported Weak, Rc::into_inner retires the last Rc block before this
// helper returns the unchanged ordinary owner. Do not extract its semantic T.
fn retire_active(
    owner: Rc<OrdinaryRetirement<SessionPayload>>,
) -> Option<OrdinaryRetirement<SessionPayload>> {
    Rc::into_inner(owner)
}
impl Drop for SessionPayloadOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.inner.take() {
            let pending = retire_active(owner);
            #[cfg(test)]
            if pending.is_some() {
                self.active_retired.set(true);
            }
            // Only queues the existing Box. No callback, wait or eager reclaim.
            drop(pending);
        }
    }
}
