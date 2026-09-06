use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// Exact MLX completion retaining the frame's arrays and native ownership.
/// Dropping a pending or failed completion never waits or retries evaluation.
#[derive(Clone)]
pub struct MlxRealtimeCompletion {
    inner: Rc<MlxRealtimeCompletionInner>,
}

struct MlxRealtimeCompletionInner {
    recovery: Recovery<RealtimeCompletionResources>,
    observations: Cell<usize>,
}

struct RealtimeCompletionResources {
    event: Option<Event>,
    submission_failed: Cell<bool>,
    observation_error: RefCell<Option<String>>,
    retained: Vec<Array>,
    token_validations: TokenValidationBatch,
    _execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
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
    #[cfg(test)]
    pub(super) fn submit_retained(
        retained: Vec<Array>,
        token_validations: TokenValidationBatch,
    ) -> Result<Self, Error> {
        Self::submit_retained_with_resources(retained, token_validations, None).map_err(|error| {
            match error {
                RealtimeCompletionCreationError::BeforeSubmission(error) => error,
                RealtimeCompletionCreationError::AfterSubmission {
                    error,
                    completion: _,
                } => error,
            }
        })
    }

    pub(super) fn submit_retained_with_resources(
        mut retained: Vec<Array>,
        token_validations: TokenValidationBatch,
        execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
    ) -> Result<Self, RealtimeCompletionCreationError<Self, Error>> {
        retained.extend(token_validations.arrays().cloned());
        let mut recovery = Recovery::begin(RealtimeCompletionResources {
            event: None,
            submission_failed: Cell::new(false),
            observation_error: RefCell::new(None),
            retained,
            token_validations,
            _execution_resources: execution_resources,
        })
        .map_err(|error| RealtimeCompletionCreationError::before_submission(error.into()))?;
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
        recovery.retention().submission_failed.set(error.is_some());
        let completion = Self {
            inner: Rc::new(MlxRealtimeCompletionInner {
                recovery,
                observations: Cell::new(0),
            }),
        };
        match error {
            None => Ok(completion),
            Some(error) => Err(submission_failure(completion, error, status.settled)),
        }
    }

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

    fn is_complete(&self) -> Result<bool, Self::Error> {
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
            result.map(|complete| complete && status.settled && self.inner.observations.get() == 0)
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

#[cfg(test)]
mod observation_tests {
    use super::*;
    use crate::backend::submission_recovery::Probe;
    struct FakeProbe(Rc<Cell<Status>>);
    impl Probe for FakeProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.0.get()
        }
    }

    fn completion() -> MlxRealtimeCompletion {
        let validations = TokenValidationScope::begin().unwrap().finish();
        MlxRealtimeCompletion::submit_retained(Vec::new(), validations).unwrap()
    }

    #[test]
    fn original_completion_cannot_hide_an_unresolved_observation() {
        let completion = completion();
        let state = Rc::new(Cell::new(Status {
            settled: false,
            failed: false,
            blocked: false,
        }));
        let observation = Recovery::with_probe(
            ObservationTicket::new(&completion.inner).unwrap(),
            FakeProbe(Rc::clone(&state)),
        );
        drop(observation);
        assert!(!completion.is_complete().unwrap());
        state.set(Status {
            settled: true,
            failed: false,
            blocked: false,
        });
        assert!(completion.is_complete().unwrap());
    }

    #[test]
    fn observation_unwind_poison_does_not_wait_for_native_terminal_status() {
        let completion = completion();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _unwind = ObservationUnwind(completion.inner.recovery.retention());
            panic!("injected observation unwind");
        }));
        assert!(result.is_err());
        assert!(completion.is_complete().is_err());
    }
}
