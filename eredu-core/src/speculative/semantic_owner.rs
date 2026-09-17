//! Closed ownership for the existing transactional semantic driver.
use super::{SpeculativeOutputError, SpeculativeSemanticState};
use crate::HostPreparationAuthority;

/// An owning semantic state with no raw Box exit. A prepared implementation's
/// retirement frees its concrete box before its host funding can be returned.
/// Moving an ordinary Box here preserves ordinary ownership; it certifies no
/// prior allocation, source identity or future callback behavior.
pub struct SpeculativeSemanticOwner(Option<Box<dyn SpeculativeSemanticState>>);
impl std::fmt::Debug for SpeculativeSemanticOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeSemanticOwner")
            .finish_non_exhaustive()
    }
}
impl SpeculativeSemanticOwner {
    /// Allocates the exact concrete state box after its constructor is paid.
    /// The implementation must retire payload after the box shell and retain
    /// each independently escaping allocation's actual funding.
    pub fn from_prepared<T: SpeculativeSemanticState + 'static>(state: T) -> Self {
        Self(Some(Box::new(state)))
    }
    /// Source identity for the selected caller's read-only authentication. This
    /// grants no construction or native permission and exposes no owning exit.
    pub fn prepared_source(&self) -> Option<&dyn std::any::Any> {
        self.state().prepared_source()
    }
    pub(super) fn state(&self) -> &dyn SpeculativeSemanticState {
        &**self.0.as_ref().expect("live semantic owner")
    }
    pub(super) fn state_mut(&mut self) -> &mut dyn SpeculativeSemanticState {
        &mut **self.0.as_mut().expect("live semantic owner")
    }
    pub(super) fn fork(&self) -> Result<Self, SpeculativeOutputError> {
        self.state().fork_owned()
    }
    pub(super) fn fork_prepared(
        &self,
        host: HostPreparationAuthority,
    ) -> Result<Self, SpeculativeOutputError> {
        self.state().fork_prepared(host)
    }
}
impl<T: SpeculativeSemanticState + 'static> From<Box<T>> for SpeculativeSemanticOwner {
    fn from(state: Box<T>) -> Self {
        Self(Some(state))
    }
}
impl From<Box<dyn SpeculativeSemanticState>> for SpeculativeSemanticOwner {
    fn from(state: Box<dyn SpeculativeSemanticState>) -> Self {
        Self(Some(state))
    }
}
impl Drop for SpeculativeSemanticOwner {
    fn drop(&mut self) {
        if let Some(state) = self.0.take() {
            state.retire();
        }
    }
}
