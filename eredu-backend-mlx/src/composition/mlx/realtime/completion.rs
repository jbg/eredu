use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// Execution ownership retained by shared native completion and observation.
///
/// Implementations retain selected stores, streams, collectives and other native
/// authorities needed by outstanding work. Resource destruction must obey native
/// retirement restrictions, for example through `OrdinaryRetirement`. Poisoning
/// permanently rejects further submissions through the associated execution;
/// it must not wait for completion or release outstanding resources.
pub(crate) trait MlxRealtimeExecutionResources {
    fn poison(&self);
}

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
    _execution_resources: Option<Arc<dyn MlxRealtimeExecutionResources>>,
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
pub(crate) struct MlxRealtimeSubmissionScope {
    recovery: Option<Recovery<RealtimeCompletionResources>>,
    token_validations: Option<TokenValidationScope>,
}

impl MlxRealtimeSubmissionScope {
    /// Enters recovery and index validation before any execution begins.
    pub(crate) fn begin(
        execution_resources: Option<Arc<dyn MlxRealtimeExecutionResources>>,
    ) -> Result<Self, Error> {
        let validations = TokenValidationScope::begin()?;
        let mut scope =
            Self::with_validations(execution_resources, TokenValidationBatch::default())?;
        scope.token_validations = Some(validations);
        Ok(scope)
    }

    fn with_validations(
        execution_resources: Option<Arc<dyn MlxRealtimeExecutionResources>>,
        token_validations: TokenValidationBatch,
    ) -> Result<Self, Error> {
        Ok(Self {
            recovery: Some(Recovery::begin(RealtimeCompletionResources {
                event: None,
                submission_failed: Cell::new(false),
                observation_error: RefCell::new(None),
                retained: Vec::new(),
                token_validations,
                _execution_resources: execution_resources,
            })?),
            token_validations: None,
        })
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
        resources.retained = retained;
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
        if let Some(recovery) = &mut self.recovery {
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

    // Failure must carry the retained completion authority back to its owner.
    #[allow(clippy::result_large_err)]
    pub(crate) fn submit_retained_with_resources(
        retained: Vec<Array>,
        token_validations: TokenValidationBatch,
        execution_resources: Option<Arc<dyn MlxRealtimeExecutionResources>>,
    ) -> Result<Self, RealtimeCompletionCreationError<Self, Error>> {
        MlxRealtimeSubmissionScope::with_validations(execution_resources, token_validations)
            .map_err(RealtimeCompletionCreationError::before_submission)?
            .finish(retained)
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
    fn validated_realtime_completion_stays_ready_while_another_thread_owns_runtime() {
        use std::{
            sync::mpsc,
            time::{Duration, Instant},
        };

        let completion = completion();
        completion.wait().unwrap();
        let retained_copy = completion.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || loop {
            if safemlx::try_with_submission_retirement(|| {
                ready_tx.send(()).unwrap();
                // Test-only contention with a watchdog, matching the native
                // retirement tests: an accidental blocking poll cannot hang.
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let started = Instant::now();
        let ready = retained_copy.is_complete();
        // Cached publication proof does not bypass native retirement exclusion.
        let releasable = retained_copy.resources_releasable();
        let elapsed = started.elapsed();
        let _ = release_tx.send(());
        holder.join().unwrap();
        assert!(ready.unwrap());
        assert!(!releasable);
        assert!(elapsed < Duration::from_secs(1));
    }

    #[test]
    fn failed_invocation_scope_retains_eager_work_without_a_final_event() {
        struct Owner(Rc<Cell<bool>>);
        impl MlxRealtimeExecutionResources for Owner {
            fn poison(&self) {
                self.0.set(true);
            }
        }
        let poisoned = Rc::new(Cell::new(false));
        let owner = Arc::new(Owner(Rc::clone(&poisoned)));
        let weak = Arc::downgrade(&owner);
        let scope = MlxRealtimeSubmissionScope::begin(Some(owner)).unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let eager = Array::from_slice(&[2.0_f32, 3.0], &[2])
            .multiply(Array::from_f32(7.0), &stream)
            .unwrap();
        let _event = async_eval_with_event([&eager]).unwrap();
        let completion = scope.fail(vec![eager]);
        assert!(poisoned.get());
        assert!(completion.inner.recovery.retention().event.is_none());
        assert!(completion.wait().is_err());
        assert!(weak.upgrade().is_some());
        crate::backend::submission_recovery::wait_for_retirement(|| {
            completion.resources_releasable()
        });
        // Terminal failure permits release but can never become publishable.
        assert!(completion.is_complete().is_err());
        drop(completion);
        crate::backend::submission_recovery::wait_for_retirement(|| weak.upgrade().is_none());
    }

    #[test]
    fn unwinding_invocation_scope_poison_closes_token_validation() {
        struct Owner(Rc<Cell<bool>>);
        impl MlxRealtimeExecutionResources for Owner {
            fn poison(&self) {
                self.0.set(true);
            }
        }
        let poisoned = Rc::new(Cell::new(false));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope =
                MlxRealtimeSubmissionScope::begin(Some(Arc::new(Owner(Rc::clone(&poisoned)))))
                    .unwrap();
            panic!("injected eager execution unwind");
        }));
        assert!(result.is_err());
        assert!(poisoned.get());
        // A failed invocation must not leave the thread-local index collector open.
        let _next_scope = TokenValidationScope::begin().unwrap();
    }

    #[test]
    fn original_completion_cannot_hide_an_unresolved_observation() {
        let completion = completion();
        completion.wait().unwrap();
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
        crate::backend::submission_recovery::wait_for_retirement(|| {
            completion.is_complete().unwrap()
        });
    }

    #[test]
    fn failed_observation_becomes_releasable_only_after_its_native_scope_settles() {
        struct ExecutionOwner(Rc<Cell<bool>>);
        impl MlxRealtimeExecutionResources for ExecutionOwner {
            fn poison(&self) {
                self.0.set(true);
            }
        }
        let poisoned = Rc::new(Cell::new(false));
        let owner = Arc::new(ExecutionOwner(Rc::clone(&poisoned)));
        let retained_owner = Arc::downgrade(&owner);
        let completion = MlxRealtimeCompletion::submit_retained_with_resources(
            Vec::new(),
            TokenValidationScope::begin().unwrap().finish(),
            Some(owner),
        )
        .unwrap_or_else(|failure| match failure {
            RealtimeCompletionCreationError::BeforeSubmission(error)
            | RealtimeCompletionCreationError::AfterSubmission { error, .. } => {
                panic!("completion submission failed: {error}")
            }
        });
        completion.wait().unwrap();
        let state = Rc::new(Cell::new(Status {
            settled: false,
            failed: true,
            blocked: true,
        }));
        drop(Recovery::with_probe(
            ObservationTicket::new(&completion.inner).unwrap(),
            FakeProbe(Rc::clone(&state)),
        ));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            completion.is_complete().is_err()
        });
        assert!(poisoned.get());
        assert!(retained_owner.upgrade().is_some());
        assert!(!completion.resources_releasable());
        state.set(Status {
            settled: true,
            failed: true,
            blocked: false,
        });
        crate::backend::submission_recovery::wait_for_retirement(|| {
            completion.resources_releasable()
        });
        crate::backend::submission_recovery::wait_for_retirement(|| {
            completion.is_complete().is_err()
        });
        drop(completion);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            retained_owner.upgrade().is_none()
        });
    }

    #[test]
    fn observation_unwind_poison_does_not_wait_for_native_terminal_status() {
        let completion = completion();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _unwind = ObservationUnwind(completion.inner.recovery.retention());
            panic!("injected observation unwind");
        }));
        assert!(result.is_err());
        crate::backend::submission_recovery::wait_for_retirement(|| {
            completion.is_complete().is_err()
        });
    }
}
