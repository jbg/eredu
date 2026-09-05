use super::*;

/// Exact completion for one retained MLX speculative verification.
pub struct MlxSpeculativeCompletion {
    event: Event,
    _retained: Vec<Array>,
}

impl MlxSpeculativeCompletion {
    /// Submits all retained verification outputs as one exact completion.
    pub fn submit<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<Self, Exception> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self {
            event,
            _retained: retained,
        })
    }

    #[cfg(test)]
    pub(super) fn retained(&self) -> &[Array] {
        &self._retained
    }
}

impl Completion for MlxSpeculativeCompletion {
    type Error = Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize()
    }
}

#[derive(Default)]
struct SpeculativeCompletionQuarantine {
    work: Vec<MlxSpeculativeCompletion>,
}

impl SpeculativeCompletionQuarantine {
    fn reap(&mut self) {
        self.work
            .retain(|completion| !matches!(completion.is_complete(), Ok(true) | Err(_)));
    }
}

impl Drop for SpeculativeCompletionQuarantine {
    fn drop(&mut self) {
        self.reap();
        for completion in self.work.drain(..) {
            let _ = completion.wait();
        }
    }
}

thread_local! {
    static SPECULATIVE_COMPLETION_ORPHANS: RefCell<SpeculativeCompletionQuarantine> =
        RefCell::new(SpeculativeCompletionQuarantine::default());
}

fn reap_speculative_completion_orphans() {
    let empty = SPECULATIVE_COMPLETION_ORPHANS.try_with(|orphans| {
        if let Ok(mut orphans) = orphans.try_borrow_mut() {
            orphans.reap();
            return orphans.work.is_empty();
        }
        false
    });
    if matches!(empty, Ok(true)) {
        safemlx::unregister_thread_runtime_housekeeping(reap_speculative_completion_orphans);
    }
}

fn quarantine_speculative_completion(completion: MlxSpeculativeCompletion) {
    safemlx::register_thread_runtime_housekeeping(reap_speculative_completion_orphans);
    SPECULATIVE_COMPLETION_ORPHANS.with(|orphans| {
        let mut orphans = orphans.borrow_mut();
        orphans.reap();
        orphans.work.push(completion);
    });
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

impl Drop for MlxSpeculativeCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}
