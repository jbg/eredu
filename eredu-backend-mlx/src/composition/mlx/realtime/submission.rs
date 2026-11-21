//! MLX queue submission, recovery and retirement for realtime invocations.

use eredu_core::realtime_mechanisms::{
    RealtimeCompletionCreationError, RealtimeQueue, RealtimeSubmissionOwner,
};
use std::{ops::Deref, sync::Arc};

use eredu_core::BackendFailure;

use safemlx::Stream;

use crate::{
    backend::{
        ordinary_retirement::{self, OrdinaryRetirement},
        runtime::{distributed::Group, realtime::MlxRealtimeReadback},
        submission_recovery,
    },
    MlxTensor,
};

use super::{completion::MlxRealtimeSubmissionScope, MlxRealtimeCompletion};

/// Native stream and collective resources selected for a realtime session.
pub struct MlxRealtimeQueue {
    pub(super) stream: Stream,
    pub(super) world: Option<Arc<Group>>,
    pub(super) _weights_stream: Stream,
}

#[doc(hidden)]
pub struct MlxRealtimeRetirement<T: 'static>(
    OrdinaryRetirement<(T, ordinary_retirement::TrimCacheOnRetirement)>,
);

impl<T: 'static> Deref for MlxRealtimeRetirement<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0 .0
    }
}

impl<T: RealtimeSubmissionOwner + 'static> RealtimeSubmissionOwner for OrdinaryRetirement<T> {
    fn poison(&self) {
        self.deref().poison();
    }
}

impl RealtimeQueue for MlxRealtimeQueue {
    type Tensor = MlxTensor;
    type Completion = MlxRealtimeCompletion;
    type Readback = MlxRealtimeReadback;
    type Scope = MlxRealtimeSubmissionScope;
    type Retired<T: 'static> = MlxRealtimeRetirement<T>;

    fn context(&self) -> &Stream {
        &self.stream
    }

    fn readback(&self) -> Self::Readback {
        MlxRealtimeReadback::new(&self.stream)
    }

    fn retire<T: 'static>(value: T) -> Self::Retired<T> {
        MlxRealtimeRetirement(OrdinaryRetirement::new((
            value,
            ordinary_retirement::TrimCacheOnRetirement,
        )))
    }

    fn reclaim() {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
    }

    fn release_retired() {
        drop(ordinary_retirement::ReclaimOnDrop);
    }

    fn begin(
        &self,
        owner: impl RealtimeSubmissionOwner + 'static,
        initial: &[MlxTensor],
    ) -> Result<Self::Scope, BackendFailure> {
        let owner = Arc::new(OrdinaryRetirement::new(owner));
        let mut scope =
            MlxRealtimeSubmissionScope::begin(Some(owner)).map_err(BackendFailure::from_error)?;
        scope.retain_initial(initial.iter().map(|tensor| tensor.as_array().clone()));
        Ok(scope)
    }

    fn finish(
        scope: Self::Scope,
        roots: Vec<MlxTensor>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, BackendFailure>>
    {
        use RealtimeCompletionCreationError::*;
        scope
            .finish(roots.into_iter().map(MlxTensor::into_array).collect())
            .map_err(|failure| match failure {
                BeforeSubmission(error) => BeforeSubmission(BackendFailure::from_error(error)),
                AfterSubmission { error, completion } => AfterSubmission {
                    error: BackendFailure::from_error(error),
                    completion,
                },
            })
    }

    fn retained_resources(completion: &Self::Completion) -> usize {
        completion.retained_resources()
    }

    fn abandon(scope: Self::Scope, roots: Vec<MlxTensor>) -> Self::Completion {
        scope.fail(roots.into_iter().map(MlxTensor::into_array).collect())
    }

    fn finish_startup() -> Result<(), BackendFailure> {
        crate::trim_allocator_cache().map_err(BackendFailure::from_error)
    }
}
