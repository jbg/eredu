use super::*;

/// Caller coordination for one ordinary submission-scope constructor.
///
/// This waits for the existing runtime owner without reclamation, housekeeping,
/// native calls or scope creation. Keep the loan through the existing one-shot
/// constructor, then drop it before releasing old resources, calling providers,
/// acquiring resource locks, evaluating, voting or waiting for completion.
/// An original-required scope is refused before waiting. This supplies no
/// original authority, admission, completion proof or bounded waiting latency.
#[must_use = "retain through one scope-entry attempt, then release before other work"]
pub struct OrdinarySubmissionEntry {
    _guard: runtime_lock::RuntimeLockGuard,
}

impl OrdinarySubmissionEntry {
    /// Coordinate ordinary entry without attempting or retrying a constructor.
    pub fn enter() -> Result<Self, crate::OriginalNativeControlError> {
        // SAFETY: the existing TLS-only predicate reads this thread's actual
        // Scope, including an inherited or incompletely configured requirement.
        if !unsafe { safemlx_sys::mlx_submission_runtime_preparation_allowed() } {
            return Err(crate::OriginalNativeControlError::ForeignDomain);
        }
        let guard = runtime_lock::coordinate_entry();
        // No callback ran while acquiring the loan, so the thread's domain
        // cannot have changed. The constructor still performs its own checks.
        Ok(Self { _guard: guard })
    }
}

impl std::fmt::Debug for OrdinarySubmissionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinarySubmissionEntry").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, sync::mpsc, time::Duration};

    thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
    fn hook() {
        HOOKS.with(|n| n.set(n.get() + 1));
    }
    struct Hook;
    impl Drop for Hook {
        fn drop(&mut self) {
            runtime_lock::unregister_housekeeping_hook(hook);
        }
    }

    #[test]
    fn ordinary_entry_retains_the_runtime_through_one_scope_attempt_without_hooks() {
        let owner = runtime_lock::enter();
        let (started_tx, started_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (released_tx, released_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            runtime_lock::register_housekeeping_hook(hook);
            let _hook = Hook;
            HOOKS.with(|n| n.set(0));
            started_tx.send(()).unwrap();
            let loan = OrdinarySubmissionEntry::enter().unwrap();
            let mut scope = SubmissionScope::try_begin().unwrap();
            assert_eq!(scope.status().activity(), SubmissionActivity::None);
            assert_eq!(HOOKS.with(Cell::get), 0);
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            drop(loan);
            released_tx.send(()).unwrap();
            scope.seal();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let waited = matches!(
            entered_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        drop(owner);
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            runtime_lock::try_enter_for_recovery().is_none(),
            "loan must still own the runtime"
        );
        release_tx.send(()).unwrap();
        released_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert!(
            waited,
            "foreign runtime ownership must precede the one-shot constructor"
        );
        assert!(runtime_lock::try_enter_for_recovery().is_some());
    }

    #[test]
    fn ordinary_entry_refuses_original_before_waiting_or_running_hooks() {
        let mut scope = SubmissionScope::begin().unwrap();
        scope.require_original_native_controls().unwrap();
        let before = scope.status();
        runtime_lock::register_housekeeping_hook(hook);
        let _hook = Hook;
        HOOKS.with(|n| n.set(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _guard = runtime_lock::enter();
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
        });
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let result = OrdinarySubmissionEntry::enter();
        let released = release_tx.send(()).is_ok();
        let held = worker.join().unwrap();
        assert!(
            released && held,
            "original refusal must not wait for a foreign owner"
        );
        assert!(matches!(
            result,
            Err(crate::OriginalNativeControlError::ForeignDomain)
        ));
        assert_eq!(scope.status(), before);
        assert_eq!(HOOKS.with(Cell::get), 0);
        scope.seal();
    }
}
