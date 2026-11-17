//! Native ownership and recovery for realtime candidates.

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

use crate::MlxTensor;
use eredu_core::{backend::Completion, BackendFailure};
use eredu_nn::RetainedTensorRoots;
use eredu_runtime::{
    submit_realtime_output, PrepublicationRealtimeFrame, RealtimeCompletionAttachmentError,
    RealtimeCompletionCreationError, RealtimeFrameHostObserver, RealtimeGenerationBranch,
    RealtimeOutputSubmissionError, RealtimeScheduleState,
};

use super::{
    completion::MlxRealtimeSubmissionScope, Error, MlxRealtimeCompletion,
    MlxRealtimeExecutionResources,
};
use crate::backend::{
    ordinary_retirement::{self, OrdinaryRetirement},
    submission_recovery,
};
use safemlx::{Array, Stream};

/// Execution shared by sibling sessions, with an exclusive invocation gate.
pub struct SharedRealtimeExecution<T: 'static> {
    inner: Arc<OrdinaryRetirement<Execution<T>>>,
    _reclaim: ordinary_retirement::ReclaimOnDrop,
}

struct Execution<T> {
    pub components: RefCell<T>,
    pub stream: Stream,
    poisoned: Cell<bool>,
    active: Cell<bool>,
    _cache_cleanup: ordinary_retirement::TrimCacheOnRetirement,
}

impl<T: 'static> SharedRealtimeExecution<T> {
    /// Adopts prepared components and their execution stream.
    pub fn new(components: T, stream: &Stream) -> Self {
        Self {
            inner: Arc::new(OrdinaryRetirement::new(Execution {
                components: RefCell::new(components),
                stream: stream.clone(),
                poisoned: Cell::new(false),
                active: Cell::new(false),
                _cache_cleanup: ordinary_retirement::TrimCacheOnRetirement,
            })),
            _reclaim: ordinary_retirement::ReclaimOnDrop,
        }
    }

    /// Shares components and their invocation gate.
    pub fn share(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _reclaim: ordinary_retirement::ReclaimOnDrop,
        }
    }

    /// Returns the selected native execution stream.
    pub fn stream(&self) -> &Stream {
        &self.inner.stream
    }

    pub(crate) fn components(&self) -> &RefCell<T> {
        &self.inner.components
    }

    /// Reads retained component metadata under the shared owner.
    pub fn inspect<O>(&self, read: impl FnOnce(&T) -> O) -> O {
        read(&self.components().borrow())
    }

    fn reclaim() {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
    }

    /// Reports unresolved work that temporarily holds the execution.
    pub fn is_busy(&self) -> bool {
        Self::reclaim();
        !self.inner.poisoned.get() && self.inner.active.get()
    }

    pub(crate) fn require_healthy(&self) -> Result<(), &'static str> {
        if self.inner.poisoned.get() {
            Err("realtime execution is poisoned")
        } else {
            Ok(())
        }
    }

    pub(crate) fn require_available(&self) -> Result<(), &'static str> {
        Self::reclaim();
        self.require_healthy()?;
        if self.inner.active.get() {
            Err("realtime execution retains an outstanding invocation")
        } else {
            Ok(())
        }
    }

    /// Acquires completion ownership after preflight and an availability check.
    pub(crate) fn retain<N: 'static>(
        &self,
        resources: N,
    ) -> Arc<dyn MlxRealtimeExecutionResources> {
        self.inner.active.set(true);
        Arc::new(SubmissionOwner(OrdinaryRetirement::new(
            SubmissionPayload {
                execution: self.inner.clone(),
                _resources: resources,
            },
        )))
    }
}

type NativeBranch<M, Q, P, S, R> = RealtimeGenerationBranch<M, Q, P, S, R, MlxRealtimeCompletion>;
type BeginInvocation<'a, B> = dyn FnMut(&B) -> Result<(), BackendFailure> + 'a;

impl<T: 'static> SharedRealtimeExecution<T> {
    /// Runs a portable invocation with native recovery and prepublication observation.
    ///
    /// The invocation performs preflight before calling the supplied begin function.
    /// It must begin before mutating the branch or submitting work. Branch roots and
    /// opaque native retention are captured at that boundary, and roots are captured
    /// again after execution or on failure to cover updates made by the invocation.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn submit<M, Q, P, S, R, I, O, H, N>(
        &self,
        branch: &mut NativeBranch<M, Q, P, S, R>,
        input: &I,
        branch_roots: impl Fn(&NativeBranch<M, Q, P, S, R>) -> Vec<Array>,
        retain_native: impl FnOnce(&NativeBranch<M, Q, P, S, R>) -> N,
        observer: H,
        invoke: impl FnOnce(
            &mut T,
            &mut NativeBranch<M, Q, P, S, R>,
            &Stream,
            &mut BeginInvocation<'_, NativeBranch<M, Q, P, S, R>>,
        ) -> Result<O, BackendFailure>,
    ) -> Result<PrepublicationRealtimeFrame<O, MlxRealtimeCompletion, H>, BackendFailure>
    where
        Q: RealtimeScheduleState,
        I: RetainedTensorRoots<MlxTensor> + ?Sized,
        O: RetainedTensorRoots<MlxTensor>,
        H: RealtimeFrameHostObserver<O>,
        N: 'static,
    {
        self.require_available().map_err(invocation_failure)?;
        if branch.has_submission_completion() {
            return Err(BackendFailure::from_error(
                RealtimeCompletionAttachmentError::AlreadyAttached,
            ));
        }

        let stream = self.stream();
        let mut components = self.components().borrow_mut();
        let submitted = with_realtime_submission(
            branch,
            &branch_roots,
            retain_native,
            |branch, submission| {
                let mut initial_roots = Vec::new();
                let output = invoke(&mut components, branch, stream, &mut |branch| {
                    submission
                        .begin(|retain_native| {
                            initial_roots = branch_roots(branch);
                            input.visit_retained_tensors(&mut |tensor| {
                                initial_roots.push(tensor.as_array().clone());
                            });
                            Ok(self.retain((retain_native(branch), initial_roots.clone())))
                        })
                        .map_err(BackendFailure::from_error)
                })?;
                initial_roots.extend(branch_roots(branch));
                output.visit_retained_tensors(&mut |tensor| {
                    initial_roots.push(tensor.as_array().clone());
                });
                submit_realtime_output(branch, output, |_, _| {
                    submission.finish(initial_roots).map(|completion| {
                        let count = completion.retained_resources();
                        (completion, count)
                    })
                })
                .map_err(|error| match error {
                    RealtimeOutputSubmissionError::BeforeSubmission(error)
                    | RealtimeOutputSubmissionError::AfterSubmission(error) => {
                        BackendFailure::from_error(error)
                    }
                    RealtimeOutputSubmissionError::Attachment(error) => {
                        BackendFailure::from_error(error)
                    }
                })
            },
        )?;
        Ok(PrepublicationRealtimeFrame::new(submitted, observer))
    }
}

fn invocation_failure(message: &'static str) -> BackendFailure {
    BackendFailure::from_error(Error::ArchitectureModel(message.into()))
}

impl<T: 'static> eredu_runtime::RealtimeStartupPreparation<T, MlxTensor>
    for SharedRealtimeExecution<T>
{
    fn prepare_startup<O: RetainedTensorRoots<MlxTensor>>(
        &self,
        operation: impl FnOnce(&mut T, &Stream) -> Result<O, BackendFailure>,
    ) -> Result<O, BackendFailure> {
        self.require_available().map_err(|message| {
            BackendFailure::from_error(Error::ArchitectureModel(message.into()))
        })?;
        let mut components = self.components().borrow_mut();
        let scope = MlxRealtimeSubmissionScope::begin(Some(self.retain(())))
            .map_err(BackendFailure::from_error)?;
        let output = operation(&mut components, self.stream())?;
        let mut roots = Vec::new();
        output.visit_retained_tensors(&mut |tensor| roots.push(tensor.as_array().clone()));
        let completion = scope.finish(roots).map_err(|error| {
            use eredu_runtime::RealtimeCompletionCreationError::*;
            match error {
                BeforeSubmission(error) | AfterSubmission { error, .. } => {
                    BackendFailure::from_error(error)
                }
            }
        })?;
        completion.wait().map_err(BackendFailure::from_error)?;
        drop(completion);
        drop(components);
        crate::trim_allocator_cache().map_err(BackendFailure::from_error)?;
        Ok(output)
    }
}

struct SubmissionPayload<T: 'static, N> {
    execution: Arc<OrdinaryRetirement<Execution<T>>>,
    _resources: N,
}

impl<T: 'static, N> Drop for SubmissionPayload<T, N> {
    fn drop(&mut self) {
        self.execution.active.set(false);
    }
}

struct SubmissionOwner<T: 'static, N: 'static>(OrdinaryRetirement<SubmissionPayload<T, N>>);

impl<T: 'static, N: 'static> MlxRealtimeExecutionResources for SubmissionOwner<T, N> {
    fn poison(&self) {
        self.0.execution.poisoned.set(true);
    }
}

/// A one-shot native lifecycle for an unpublished scheduler candidate.
pub(super) struct CandidateSubmission<A> {
    authority: Option<A>,
    scope: Option<MlxRealtimeSubmissionScope>,
}

impl<A> CandidateSubmission<A> {
    pub(super) fn begin(
        &mut self,
        retain: impl FnOnce(A) -> Result<Arc<dyn MlxRealtimeExecutionResources>, Error>,
    ) -> Result<(), Error> {
        let authority = self.authority.take().ok_or_else(|| {
            Error::Parallel("realtime candidate submission already attempted".into())
        })?;
        let owner = retain(authority)?;
        self.scope = Some(MlxRealtimeSubmissionScope::begin(Some(owner))?);
        Ok(())
    }

    pub(super) fn finish(
        &mut self,
        retained: Vec<Array>,
    ) -> Result<MlxRealtimeCompletion, RealtimeCompletionCreationError<MlxRealtimeCompletion, Error>>
    {
        let scope = self.scope.take().ok_or_else(|| {
            RealtimeCompletionCreationError::before_submission(Error::Parallel(
                "realtime candidate has no active submission scope".into(),
            ))
        })?;
        scope.finish(retained)
    }
}

/// Owns a candidate's recovery scope. The driver enters it only after preflight.
/// Early exit after that boundary attaches failed completion to the branch.
pub(super) fn with_realtime_submission<M, Q, P, S, R, A, O, E>(
    branch: &mut NativeBranch<M, Q, P, S, R>,
    failure_roots: impl Fn(&NativeBranch<M, Q, P, S, R>) -> Vec<Array>,
    authority: A,
    operation: impl FnOnce(
        &mut NativeBranch<M, Q, P, S, R>,
        &mut CandidateSubmission<A>,
    ) -> Result<O, E>,
) -> Result<O, E>
where
    Q: RealtimeScheduleState,
{
    let mut guard = SubmissionGuard {
        branch,
        submission: CandidateSubmission {
            authority: Some(authority),
            scope: None,
        },
        fail: |branch: &mut NativeBranch<M, Q, P, S, R>, scope: MlxRealtimeSubmissionScope| {
            let completion = scope.fail(failure_roots(branch));
            // Recovery retains work independently even if attachment is rejected.
            let _ = branch.attach_submission_completion(completion);
        },
    };
    operation(guard.branch, &mut guard.submission)
}

struct SubmissionGuard<'a, B, A, F: FnMut(&mut B, MlxRealtimeSubmissionScope)> {
    branch: &'a mut B,
    submission: CandidateSubmission<A>,
    fail: F,
}

impl<B, A, F: FnMut(&mut B, MlxRealtimeSubmissionScope)> Drop for SubmissionGuard<'_, B, A, F> {
    fn drop(&mut self) {
        if let Some(scope) = self.submission.scope.take() {
            (self.fail)(self.branch, scope);
        }
    }
}
