//! Shared semantic state and custody, independent of execution strategy.
use super::{FinishReason, SemanticEvent};
use crate::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeOutputError};

/// Object-safe semantic state shared by committed generation and speculative transactions.
///
/// This interface owns decoded semantic events and never exposes a backend
/// tensor, stream, completion, or error type.
pub trait SemanticState {
    /// Complete conservative bytes retained by a semantic fork, or unknown.
    fn control_snapshot_bytes(&self) -> Option<u64> {
        None
    }

    /// Actual isolated host copy allocations; unknown callbacks remain unknown.
    fn control_snapshot_metadata_bytes(&self) -> Option<usize> {
        None
    }
    /// Actual prepared source for read-only concrete authentication only.
    fn prepared_source(&self) -> Option<&dyn std::any::Any> {
        None
    }
    /// Forks the exact committed semantic prefix into a closed owner. Each
    /// implementation must explicitly construct and fund its own destination.
    fn fork_owned(&self) -> Result<SemanticStateOwner, SpeculativeOutputError>;
    /// Copy after the shared snapshot driver pays this source's exact query.
    fn fork_prepared(
        &self,
        host: crate::HostPreparationAuthority,
    ) -> Result<SemanticStateOwner, SpeculativeOutputError> {
        if !host.is_unmanaged() {
            return Err(SpeculativeOutputError::Storage(
                "semantic state has no paid isolated copy",
            ));
        }
        self.fork_owned()
    }
    /// Ordinary boxes retain ordinary drop; prepared implementations move the
    /// concrete payload out so the shell retires before its funding.
    fn retire(self: Box<Self>) {
        drop(self);
    }
    /// Stages one token and reports whether a stop sequence matched.
    fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError>;
    /// Stages normal terminal output.
    fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError>;
    /// Stages cancellation output.
    fn cancel(&mut self) -> Result<(), SpeculativeOutputError>;
    /// Drains events authorized by the next exact commit boundary.
    fn take_events(&mut self) -> SpeculativeBuffer<crate::generation::SemanticEvent>;
    /// Synchronous publication through the same callback. Prepared implementations
    /// can drain in place, retaining their exact fixed queue for the next round.
    fn publish_events(&mut self, emit: &mut dyn FnMut(crate::generation::SemanticEvent)) {
        for event in self.take_events() {
            emit(event);
        }
    }
}

/// An owning semantic state with no raw Box exit. A prepared implementation's
/// retirement frees its concrete box before its host funding can be returned.
/// Moving an ordinary Box here preserves ordinary ownership; it certifies no
/// prior allocation, source identity or future callback behavior.
pub struct SemanticStateOwner(Option<Box<dyn SemanticState>>);
impl std::fmt::Debug for SemanticStateOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticStateOwner").finish_non_exhaustive()
    }
}
impl SemanticStateOwner {
    /// Allocates the exact concrete state box after its constructor is paid.
    /// The implementation must retire payload after the box shell and retain
    /// each independently escaping allocation's actual funding.
    pub fn from_prepared<T: SemanticState + 'static>(state: T) -> Self {
        Self(Some(Box::new(state)))
    }
    /// Stages one token. The execution driver authorizes publication only after
    /// committing it; speculative consumers stage on an isolated semantic fork.
    pub fn push_token(&mut self, token: u32) -> Result<bool, SpeculativeOutputError> {
        self.state_mut().push_token(token)
    }
    /// Stages normal termination without duplicating prior terminal events.
    pub fn finish(&mut self, reason: FinishReason) -> Result<(), SpeculativeOutputError> {
        self.state_mut().finish(reason)
    }
    /// Stages cancellation; allocation or parser failure remains fallible.
    pub fn cancel(&mut self) -> Result<(), SpeculativeOutputError> {
        self.state_mut().cancel()
    }
    /// Publishes committed events in order while retaining the state and queue.
    pub fn publish_events(&mut self, emit: &mut dyn FnMut(SemanticEvent)) {
        self.state_mut().publish_events(emit);
    }
    /// Source identity for the selected caller's read-only authentication. This
    /// grants no construction or native permission and exposes no owning exit.
    pub fn prepared_source(&self) -> Option<&dyn std::any::Any> {
        self.state().prepared_source()
    }
    pub(crate) fn state(&self) -> &dyn SemanticState {
        &**self.0.as_ref().expect("live semantic owner")
    }
    pub(crate) fn state_mut(&mut self) -> &mut dyn SemanticState {
        &mut **self.0.as_mut().expect("live semantic owner")
    }
    pub(crate) fn fork(&self) -> Result<Self, SpeculativeOutputError> {
        self.state().fork_owned()
    }
    pub(crate) fn fork_prepared(
        &self,
        host: HostPreparationAuthority,
    ) -> Result<Self, SpeculativeOutputError> {
        self.state().fork_prepared(host)
    }
}
impl<T: SemanticState + 'static> From<Box<T>> for SemanticStateOwner {
    fn from(state: Box<T>) -> Self {
        Self(Some(state))
    }
}
impl From<Box<dyn SemanticState>> for SemanticStateOwner {
    fn from(state: Box<dyn SemanticState>) -> Self {
        Self(Some(state))
    }
}
impl Drop for SemanticStateOwner {
    fn drop(&mut self) {
        if let Some(state) = self.0.take() {
            state.retire();
        }
    }
}
