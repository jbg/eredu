use crate::SubmissionBackend;
use eredu_core::Completion;

/// Owned graph-boundary submission with the narrow operation needed by the
/// shared layered scheduler. A policy may retain its own prepared completion
/// without exposing an unrestricted application-owned resource retention API.
pub trait OrderedLayerwiseCompletion<C: ?Sized>: eredu_core::Completion {
    /// Order future work on this context after the retained submission.
    fn order_after(&self, context: &C) -> Result<(), Self::Error>;

    /// Establish terminal completion and retire this actual owner before the
    /// forward returns. Errors keep unresolved resources on its safe Drop path.
    /// The ordinary default waits on the unchanged concrete completion.
    fn finish(self) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        self.wait()
    }
}

// Stack wrapper only. The ordinary concrete backend completion remains the
// sole resource owner; there is no Box, Rc, clone or alternate submission path.
pub(super) struct BackendLayerwiseCompletion<B: SubmissionBackend> {
    completion: B::Completion,
    backend: std::marker::PhantomData<fn() -> B>,
}
impl<B: SubmissionBackend> BackendLayerwiseCompletion<B> {
    pub(super) fn new(completion: B::Completion) -> Self {
        Self {
            completion,
            backend: std::marker::PhantomData,
        }
    }
}
impl<B: SubmissionBackend> OrderedLayerwiseCompletion<B::Executor>
    for BackendLayerwiseCompletion<B>
{
    fn order_after(&self, context: &B::Executor) -> Result<(), Self::Error> {
        B::order_after(&self.completion, context)
    }
}

impl<B: SubmissionBackend> eredu_core::Completion for BackendLayerwiseCompletion<B> {
    type Error = <B::Completion as eredu_core::Completion>::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.completion.is_complete()
    }
    fn wait(&self) -> Result<(), Self::Error> {
        self.completion.wait()
    }
    fn resources_releasable(&self) -> bool {
        self.completion.resources_releasable()
    }
}

// Infer the exact same P::submit_group opaque owner from the initial slot,
// including G=1 where it is None. This retains the existing one Vec allocation.
pub(super) fn empty_completions<T>(_: &Option<T>, count: usize) -> Vec<Option<T>> {
    (0..count).map(|_| None).collect()
}
