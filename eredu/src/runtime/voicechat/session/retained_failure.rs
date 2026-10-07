//! Retained error causes across the shared scheduler's diagnostic boundary.

use std::{cell::RefCell, error::Error, fmt, rc::Rc, sync::Arc};

use eredu_core::{
    scheduler::{SemanticStateTransaction, TransitionOutput},
    BackendFailure,
};

use super::super::{VoiceChatPublicationBranch, VoiceChatPublicationState, VoiceChatSubmittedWork};

pub(super) type FailureSlot = Rc<RefCell<Option<Arc<BackendFailure>>>>;

/// Keeps the first failure while allowing later retirement failures to be reported.
pub(super) fn capture(
    slot: &FailureSlot,
    error: impl Error + Send + Sync + 'static,
) -> SharedFailure {
    let failure = Arc::new(BackendFailure::from_error(error));
    let mut retained = slot.borrow_mut();
    if retained.is_none() {
        *retained = Some(Arc::clone(&failure));
    }
    SharedFailure(failure)
}

#[derive(Debug, Clone)]
pub(super) struct SharedFailure(Arc<BackendFailure>);

impl fmt::Display for SharedFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0.as_ref(), formatter)
    }
}

impl Error for SharedFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.0.as_ref())
    }
}

pub(super) struct TrackedState<S: SemanticStateTransaction, D> {
    pub(super) inner: VoiceChatPublicationState<S, D>,
    pub(super) failure: FailureSlot,
}

pub(super) struct TrackedBranch<B, D> {
    pub(super) inner: VoiceChatPublicationBranch<B, D>,
    failure: FailureSlot,
}

impl<S, D> SemanticStateTransaction for TrackedState<S, D>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
{
    type Branch = TrackedBranch<S::Branch, D>;
    type Error = SharedFailure;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        self.inner
            .branch()
            .map(|inner| TrackedBranch {
                inner,
                failure: Rc::clone(&self.failure),
            })
            .map_err(|error| capture(&self.failure, error))
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.inner
            .commit_branch(branch.inner)
            .map_err(|error| capture(&self.failure, error))
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        VoiceChatPublicationState::<S, D>::discard_branch(branch.inner)
            .map_err(|error| capture(&branch.failure, error))
    }

    fn permits_parallel_branches(&self) -> bool {
        self.inner.permits_parallel_branches()
    }
}

pub(super) struct TrackedOutput {
    pub(super) inner: VoiceChatSubmittedWork,
    pub(super) failure: FailureSlot,
}

impl TransitionOutput for TrackedOutput {
    type Error = SharedFailure;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.inner
            .is_complete()
            .map_err(|error| capture(&self.failure, error))
    }

    fn retained_resources(&self) -> usize {
        self.inner.retained_resources()
    }

    fn backend_name(&self) -> Option<String> {
        self.inner.backend_name()
    }

    fn physically_preemptible(&self) -> bool {
        self.inner.physically_preemptible()
    }
}
