//! Core-issued identity evidence for one ordinary prediction attempt.

use std::sync::atomic::{AtomicU64, Ordering};

/// Exhaustion of non-reusable core identity evidence.
///
/// This is terminal for the affected context. It grants no execution authority,
/// and its neutral conversion retains a fixed static source without allocating.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TextContextError {
    /// Every representable core run identity has already been issued.
    #[error("text run identity space exhausted")]
    RunExhausted,
    /// This run has consumed every representable policy revision.
    #[error("text policy revision space exhausted")]
    PolicyExhausted,
}
impl TextContextError {
    /// Returns the fixed typed failure without constructing an error allocation.
    pub fn into_backend_failure(self) -> super::BackendFailure {
        super::BackendFailure::text_context(self)
    }
}

/// Identity of one ordinary machine, independent of cloned preparation charges.
/// IDs never repeat, even after all evidence for a previous machine is dropped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextRunIdentity(Option<std::num::NonZeroU64>);

/// Identity of the current policy revision, replaced before mutable exposure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPolicyIdentity {
    run: TextRunIdentity,
    revision: u64,
    valid: bool,
}

// Equality-only evidence: this counter owns no request, payload or allocation.
// MAX is issued once; checked update then leaves the issuer permanently exhausted.
static LAST_RUN: AtomicU64 = AtomicU64::new(0);
fn issue_run(issuer: &AtomicU64) -> Option<std::num::NonZeroU64> {
    issuer
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
            last.checked_add(1)
        })
        .ok()
        .and_then(|last| std::num::NonZeroU64::new(last + 1))
}

/// Evidence issued by the shared machine before a prediction begins.
///
/// A clone retains evidence only; it grants no execution authority. Backends
/// separately issue a move-only permit for the exact current request. Snapshot
/// restoration never rewinds this context, and a fork receives a fresh run.
/// Exhausted evidence is terminal and cannot become executable after restoration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextStepContext {
    run: TextRunIdentity,
    policy: TextPolicyIdentity,
    attempt: u64,
    exhausted: Option<TextContextError>,
}

impl TextStepContext {
    pub(super) fn new() -> Self {
        #[cfg(test)]
        if FORCE_RUN_EXHAUSTION.with(std::cell::Cell::get) {
            return Self::from_run(None);
        }
        Self::from_run(issue_run(&LAST_RUN))
    }

    fn from_run(value: Option<std::num::NonZeroU64>) -> Self {
        let run = TextRunIdentity(value);
        Self {
            policy: TextPolicyIdentity {
                run: run.clone(),
                revision: 0,
                valid: value.is_some(),
            },
            run,
            attempt: 0,
            exhausted: value.is_none().then_some(TextContextError::RunExhausted),
        }
    }

    /// The machine that issued this evidence.
    pub fn run_identity(&self) -> &TextRunIdentity {
        &self.run
    }

    /// The policy revision observed before this attempt.
    pub fn policy_identity(&self) -> &TextPolicyIdentity {
        &self.policy
    }

    /// Zero-based ordinal; successful permit acquisition consumes this attempt.
    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub(super) fn validate(&self) -> Result<(), super::BackendFailure> {
        self.exhausted
            .map_or(Ok(()), |cause| Err(cause.into_backend_failure()))
    }

    pub(super) fn following_attempt(&self) -> Option<Self> {
        if self.exhausted.is_some() {
            return None;
        }
        Some(Self {
            run: self.run.clone(),
            policy: self.policy.clone(),
            attempt: self.attempt.checked_add(1)?,
            exhausted: None,
        })
    }

    pub(super) fn revise_policy(&mut self) {
        if self.exhausted.is_some() {
            return;
        }
        match self.policy.revision.checked_add(1) {
            Some(revision) => self.policy.revision = revision,
            None => {
                self.policy.valid = false;
                self.exhausted = Some(TextContextError::PolicyExhausted);
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    // A thread-local test-only exhausted issuer; never substitutes a live ID or
    // mutates the production counter used by concurrently executing fixtures.
    static FORCE_RUN_EXHAUSTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn with_exhausted_run<T>(work: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_RUN_EXHAUSTION.with(|value| value.set(self.0));
        }
    }
    let _reset = Reset(FORCE_RUN_EXHAUSTION.with(|value| value.replace(true)));
    work()
}

#[cfg(test)]
mod tests;
