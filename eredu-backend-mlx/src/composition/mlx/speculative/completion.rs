use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use std::{cell::Cell, rc::Rc};

struct SpeculativeRetention {
    event: Option<Event>,
    arrays: Rc<Vec<Array>>,
}

impl Retention for SpeculativeRetention {
    fn observe(&self, _: Status) {}
}

/// Exact completion for one retained MLX speculative verification.
pub struct MlxSpeculativeCompletion {
    recovery: Recovery<SpeculativeRetention>,
    observations: Rc<SpeculativeObservations>,
}

#[derive(Default)]
struct SpeculativeObservations {
    active: Cell<usize>,
    failed: Cell<bool>,
}

struct SpeculativeObservation {
    _arrays: Rc<Vec<Array>>,
    state: Rc<SpeculativeObservations>,
}
impl SpeculativeObservation {
    fn new(arrays: &Rc<Vec<Array>>, state: &Rc<SpeculativeObservations>) -> Self {
        state.active.set(
            state
                .active
                .get()
                .checked_add(1)
                .expect("speculative observation ticket overflow"),
        );
        Self {
            _arrays: Rc::clone(arrays),
            state: Rc::clone(state),
        }
    }
}
impl Retention for SpeculativeObservation {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.state.failed.set(true);
        }
    }
}
impl Drop for SpeculativeObservation {
    fn drop(&mut self) {
        self.state.active.set(self.state.active.get() - 1);
    }
}
struct ObservationUnwind<'a>(&'a SpeculativeObservations);
impl Drop for ObservationUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.failed.set(true);
        }
    }
}

impl MlxSpeculativeCompletion {
    /// Submits all retained verification outputs as one exact completion.
    pub fn submit<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<Self, Exception> {
        let arrays = outputs.into_iter().cloned().collect::<Vec<_>>();
        let mut recovery = Recovery::begin(SpeculativeRetention {
            event: None,
            arrays: Rc::new(arrays),
        })?;
        let event = async_eval_with_event(recovery.retention().arrays.iter())?;
        recovery.retention_mut().event = Some(event);
        recovery.seal();
        let status = recovery.progress();
        if status.failed || status.blocked {
            return Err(Exception::custom(
                "speculative submission failed or is unobservable",
            ));
        }
        Ok(Self {
            recovery,
            observations: Rc::new(SpeculativeObservations::default()),
        })
    }

    #[cfg(test)]
    pub(super) fn retained(&self) -> &[Array] {
        &self.recovery.retention().arrays
    }
}

impl Completion for MlxSpeculativeCompletion {
    type Error = Exception;

    fn resources_releasable(&self) -> bool {
        safemlx::try_with_submission_retirement(|| {
            crate::backend::submission_recovery::reap();
            self.recovery.progress().settled && self.observations.active.get() == 0
        })
        .unwrap_or(false)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            crate::backend::submission_recovery::reap();
            if self.observations.failed.get() {
                return Err(Exception::custom(
                    "speculative completion observation previously failed",
                ));
            }
            if self.observations.active.get() != 0 {
                return Ok(false);
            }
            let submitted = self.recovery.progress();
            if submitted.failed || submitted.blocked {
                self.observations.failed.set(true);
                return Err(Exception::custom(
                    "speculative native work failed or is unobservable",
                ));
            }
            if !submitted.settled {
                return Ok(false);
            }
            let mut observation = Recovery::begin(SpeculativeObservation::new(
                &self.recovery.retention().arrays,
                &self.observations,
            ))?;
            let _unwind = ObservationUnwind(&self.observations);
            let ready = self
                .recovery
                .retention()
                .event
                .as_ref()
                .expect("submitted event")
                .is_complete();
            observation.seal();
            let observed = observation.progress();
            let status = self.recovery.progress();
            if ready.is_err()
                || status.failed
                || status.blocked
                || observed.failed
                || observed.blocked
            {
                self.observations.failed.set(true);
                return Err(Exception::custom(
                    "speculative native work failed or is unobservable",
                ));
            }
            ready.map(|ready| ready && status.settled && observed.settled)
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        loop {
            if self.is_complete()? {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}

fn quarantine_speculative_completion(completion: MlxSpeculativeCompletion) {
    // Recovery's preallocated node owns both the event and every root. Drop
    // performs bounded progress, retaining unresolved work through thread exit.
    drop(completion);
}

impl BoundedCompletion for MlxSpeculativeCompletion {
    fn supports_cancellation(cancellation: CompletionCancellationMode) -> bool {
        cancellation == CompletionCancellationMode::QuarantineUntilComplete
    }

    fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        let Some(deadline) = std::time::Instant::now().checked_add(policy.timeout()) else {
            quarantine_speculative_completion(self);
            return Err(Exception::custom(
                "speculative completion deadline exceeds the host monotonic clock range; live work was quarantined safely",
            ));
        };
        loop {
            if self.is_complete()? {
                return Ok(BoundedCompletionOutcome::Completed);
            }
            if std::time::Instant::now() >= deadline {
                let selected = policy.cancellation();
                quarantine_speculative_completion(self);
                if selected != CompletionCancellationMode::QuarantineUntilComplete {
                    return Err(Exception::custom(
                        "MLX speculative execution has no native cancellation; timed-out work was quarantined safely",
                    ));
                }
                return Ok(BoundedCompletionOutcome::DeadlineExceeded {
                    cancellation: CompletionCancellationMode::QuarantineUntilComplete,
                });
            }
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod observation_scope_tests {
    use super::*;
    use crate::backend::submission_recovery::Probe;
    struct FakeProbe(Rc<Cell<Status>>);
    impl Probe for FakeProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.0.get()
        }
    }

    #[test]
    fn prior_pending_observation_stays_active_until_terminal_and_failure_is_sticky() {
        let state = Rc::new(SpeculativeObservations::default());
        let native = Rc::new(Cell::new(Status {
            settled: false,
            failed: true,
            blocked: true,
        }));
        let observation = Recovery::with_probe(
            SpeculativeObservation::new(&Rc::new(Vec::new()), &state),
            FakeProbe(Rc::clone(&native)),
        );
        drop(observation);
        assert_eq!(state.active.get(), 1);
        assert!(state.failed.get());
        native.set(Status {
            settled: true,
            failed: false,
            blocked: false,
        });
        crate::backend::submission_recovery::reap();
        crate::backend::submission_recovery::wait_for_retirement(|| state.active.get() == 0);
        assert_eq!(state.active.get(), 0);
        assert!(state.failed.get());
    }
}
