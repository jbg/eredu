use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use eredu_core::realtime_mechanisms::RealtimeSubmissionOwner;

/// Exact MLX completion retaining the frame's arrays and native ownership.
/// Dropping a pending or failed completion never waits or retries evaluation.
#[derive(Clone)]
pub struct MlxRealtimeCompletion {
    inner: Rc<MlxRealtimeCompletionInner>,
}

struct MlxRealtimeCompletionInner {
    recovery: Recovery<RealtimeCompletionResources>,
    observations: Cell<usize>,
    validated_complete: Cell<bool>,
}

struct RealtimeCompletionResources {
    event: Option<Event>,
    submission_failed: Cell<bool>,
    observation_error: RefCell<Option<String>>,
    retained: Vec<Array>,
    token_validations: TokenValidationBatch,
    _execution_resources: Option<Arc<dyn RealtimeSubmissionOwner>>,
}

impl Retention for RealtimeCompletionResources {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.submission_failed.set(true);
            if let Some(resources) = &self._execution_resources {
                resources.poison();
            }
        }
    }
}

impl Retention for MlxRealtimeCompletionInner {
    fn observe(&self, status: Status) {
        self.recovery.retention().observe(status);
    }
}

struct ObservationTicket(Rc<MlxRealtimeCompletionInner>);

impl ObservationTicket {
    fn new(inner: &Rc<MlxRealtimeCompletionInner>) -> Result<Self, Error> {
        let count = inner.observations.get().checked_add(1).ok_or_else(|| {
            Error::Parallel("MLX realtime observation ticket count exhausted".into())
        })?;
        // New observation work must establish its own terminal proof, even when
        // the original invocation was already observed successfully.
        inner.validated_complete.set(false);
        inner.observations.set(count);
        Ok(Self(Rc::clone(inner)))
    }
}

impl Retention for ObservationTicket {
    fn observe(&self, status: Status) {
        self.0.observe(status);
    }
}

impl Drop for ObservationTicket {
    fn drop(&mut self) {
        self.0.observations.set(self.0.observations.get() - 1);
    }
}

struct ObservationUnwind<'a>(&'a RealtimeCompletionResources);
impl Drop for ObservationUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.submission_failed.set(true);
            if let Some(execution) = &self.0._execution_resources {
                execution.poison();
            }
        }
    }
}

/// Captures every native submission in a coupled invocation, including eager
/// work performed before its final output arrays exist.
#[doc(hidden)]
pub struct MlxRealtimeSubmissionScope {
    recovery: Option<Recovery<RealtimeCompletionResources>>,
    token_validations: Option<TokenValidationScope>,
}

impl MlxRealtimeSubmissionScope {
    /// Enters recovery and index validation before any execution begins.
    pub(crate) fn begin(
        execution_resources: Option<Arc<dyn RealtimeSubmissionOwner>>,
    ) -> Result<Self, Error> {
        let validations = TokenValidationScope::begin()?;
        Ok(Self {
            recovery: Some(Recovery::begin(RealtimeCompletionResources {
                event: None,
                submission_failed: Cell::new(false),
                observation_error: RefCell::new(None),
                retained: Vec::new(),
                token_validations: TokenValidationBatch::default(),
                _execution_resources: execution_resources,
            })?),
            token_validations: Some(validations),
        })
    }

    pub(super) fn retain_initial(&mut self, retained: impl Iterator<Item = Array>) {
        self.recovery
            .as_mut()
            .expect("live submission scope")
            .retention_mut()
            .retained
            .extend(retained);
    }

    fn retain(&mut self, mut retained: Vec<Array>) {
        let resources = self
            .recovery
            .as_mut()
            .expect("live submission scope")
            .retention_mut();
        if let Some(validations) = self.token_validations.take() {
            resources.token_validations = validations.finish();
        }
        retained.extend(resources.token_validations.arrays().cloned());
        resources.retained.extend(retained);
    }

    /// Submits final arrays inside the scope that captured earlier eager work.
    #[allow(clippy::result_large_err)]
    pub(crate) fn finish(
        mut self,
        retained: Vec<Array>,
    ) -> Result<MlxRealtimeCompletion, RealtimeCompletionCreationError<MlxRealtimeCompletion, Error>>
    {
        self.retain(retained);
        let recovery = self.recovery.as_mut().expect("live submission scope");
        let result = async_eval_with_event(recovery.retention().retained.iter());
        recovery.seal();
        let status = recovery.progress();
        let error = match result {
            Ok(event) => {
                recovery.retention_mut().event = Some(event);
                (status.failed || status.blocked).then(|| Error::Parallel(
                    "MLX realtime native submission failed; unresolved resources remain retained".into(),
                ))
            }
            Err(error) => Some(error.into()),
        };
        if error.is_some() {
            mark_failed(recovery.retention());
        }
        let completion = self.into_completion();
        match error {
            None => Ok(completion),
            Some(error) => Err(submission_failure(completion, error, status.settled)),
        }
    }

    /// Seals failed execution without submitting any additional graph. The
    /// returned completion retains eager work and permanently rejects publication.
    pub(crate) fn fail(mut self, retained: Vec<Array>) -> MlxRealtimeCompletion {
        self.retain(retained);
        let recovery = self.recovery.as_mut().expect("live submission scope");
        mark_failed(recovery.retention());
        recovery.seal();
        self.into_completion()
    }

    fn into_completion(mut self) -> MlxRealtimeCompletion {
        MlxRealtimeCompletion {
            inner: Rc::new(MlxRealtimeCompletionInner {
                recovery: self.recovery.take().expect("live submission scope"),
                observations: Cell::new(0),
                validated_complete: Cell::new(false),
            }),
        }
    }
}

fn mark_failed(resources: &RealtimeCompletionResources) {
    resources.submission_failed.set(true);
    if let Some(execution) = &resources._execution_resources {
        execution.poison();
    }
}

impl Drop for MlxRealtimeSubmissionScope {
    fn drop(&mut self) {
        if self.recovery.is_some() {
            self.retain(Vec::new());
            let recovery = self.recovery.as_mut().expect("live submission scope");
            mark_failed(recovery.retention());
            recovery.seal();
        }
    }
}

pub(super) fn submission_failure<C, E>(
    completion: C,
    error: E,
    settled: bool,
) -> RealtimeCompletionCreationError<C, E> {
    if settled {
        RealtimeCompletionCreationError::before_submission(error)
    } else {
        RealtimeCompletionCreationError::after_submission(error, completion)
    }
}

impl std::fmt::Debug for MlxRealtimeCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxRealtimeCompletion")
            .field("retained_resources", &self.retained_resources())
            .finish_non_exhaustive()
    }
}

impl MlxRealtimeCompletion {
    /// Number of array handles retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.inner.recovery.retention().retained.len()
    }

    fn ready(&self) -> Result<bool, Error> {
        crate::backend::submission_recovery::reap();
        let status = self.inner.recovery.progress();
        let resources = self.inner.recovery.retention();
        if let Some(error) = resources.observation_error.borrow().as_ref() {
            return Err(Error::Parallel(error.clone()));
        }
        if resources.submission_failed.get() || status.failed || status.blocked {
            return Err(Error::Parallel(
                "MLX realtime completion failed; unresolved resources remain retained".into(),
            ));
        }
        Ok(status.settled && self.inner.observations.get() == 0)
    }
}

impl Completion for MlxRealtimeCompletion {
    type Error = Error;

    fn resources_releasable(&self) -> bool {
        safemlx::try_with_submission_retirement(|| {
            crate::backend::submission_recovery::reap();
            self.inner.recovery.progress().settled && self.inner.observations.get() == 0
        })
        .unwrap_or(false)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        // Successful publication evidence is monotonic across runtime-lock
        // contention. Re-polling an already validated event must not discard a
        // ready semantic branch merely because an unrelated thread is submitting.
        // Retirement still takes the native lock in resources_releasable().
        let resources = self.inner.recovery.retention();
        if self.inner.validated_complete.get()
            && self.inner.observations.get() == 0
            && !resources.submission_failed.get()
            && resources.observation_error.borrow().is_none()
        {
            return Ok(true);
        }
        safemlx::try_with_submission_retirement(|| {
            if !self.ready()? {
                return Ok(false);
            }
            let _unwind = ObservationUnwind(self.inner.recovery.retention());
            let mut observation = Recovery::begin(ObservationTicket::new(&self.inner)?)?;
            let result = (|| {
                let resources = self.inner.recovery.retention();
                let complete = resources
                    .event
                    .as_ref()
                    .expect("successful event submission")
                    .is_complete()?;
                if complete {
                    // Token validity is a publication gate, independent of resource lifetime.
                    resources.token_validations.validate_completed()?;
                }
                Ok::<_, Error>(complete)
            })();
            observation.seal();
            let status = observation.progress();
            if let Err(error) = &result {
                let resources = self.inner.recovery.retention();
                *resources.observation_error.borrow_mut() = Some(error.to_string());
                if let Some(execution) = &resources._execution_resources {
                    execution.poison();
                }
            }
            if status.failed || status.blocked {
                return Err(Error::Parallel(
                    "MLX realtime observation failed; unresolved resources remain retained".into(),
                ));
            }
            drop(observation);
            result.map(|complete| {
                let complete = complete && status.settled && self.inner.observations.get() == 0;
                self.inner.validated_complete.set(complete);
                complete
            })
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}
