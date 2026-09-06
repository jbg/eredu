use super::*;
use crate::{BackendProvider, BackendSession, Completion, ModelRuntime, PreparedModel, Submission};
use std::sync::atomic::{AtomicBool, AtomicU8};

struct ForeignNativeCompletion {
    // 0: pending, 1: ready, 2: terminal failure.
    state: Arc<AtomicU8>,
    active: Arc<AtomicU64>,
    drained: Arc<AtomicBool>,
}

impl Drop for ForeignNativeCompletion {
    fn drop(&mut self) {
        // A still-pending backend resource must be torn down with authority held.
        if self.state.load(Ordering::Acquire) == 0 {
            assert_ne!(self.active.load(Ordering::Acquire), 0);
        }
        self.drained.store(true, Ordering::Release);
    }
}

struct ForeignCompletion {
    native: ForeignNativeCompletion,
    lease: SubmissionLease,
}

impl Completion for ForeignCompletion {
    type Error = std::io::Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        match self.native.state.load(Ordering::Acquire) {
            0 => Ok(false),
            1 => {
                self.lease.resolve();
                Ok(true)
            }
            _ => {
                self.lease.resolve();
                Err(std::io::Error::other("terminal native failure"))
            }
        }
    }

    fn wait(&self) -> Result<(), Self::Error> {
        if self.native.state.load(Ordering::Acquire) == 0 {
            self.native.state.store(1, Ordering::Release);
        }
        self.is_complete().map(|_| ())
    }
}

fn pending(authority: &mut SessionAuthority) -> ForeignCompletion {
    let lease = authority.begin_submission().unwrap();
    ForeignCompletion {
        native: ForeignNativeCompletion {
            state: Arc::new(AtomicU8::new(0)),
            active: Arc::clone(&authority.active),
            drained: Arc::new(AtomicBool::new(false)),
        },
        lease,
    }
}

#[test]
fn admission_requires_exact_equality_including_unrequested_extra_capabilities() {
    for admitted in 0..8 {
        for realized in 0..8 {
            let report =
                |bits| SessionCapabilities::new(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
            let result = SessionAdmission::new(report(admitted)).validate(report(realized));
            assert_eq!(result.is_ok(), admitted == realized);
            if let Err(error) = result {
                assert_eq!(error.admitted(), report(admitted));
                assert_eq!(error.realized(), report(realized));
            }
        }
    }
}

#[test]
fn nested_submission_and_mutation_are_excluded_until_own_lease_resolves() {
    let mut authority = SessionAuthority::new();
    let first = authority.begin_submission().unwrap();
    assert_eq!(authority.require_idle(), Err(SessionAuthorityError::Busy));
    assert!(matches!(
        authority.begin_submission(),
        Err(SessionAuthorityError::Busy)
    ));
    assert!(first.resolve());
    assert!(!first.resolve());
    let second = authority.begin_submission().unwrap();
    assert!(!first.resolve());
    drop(first);
    assert_eq!(authority.require_idle(), Err(SessionAuthorityError::Busy));
    drop(second);
    assert_eq!(authority.require_idle(), Ok(()));
}

#[test]
fn aborted_and_unwinding_submissions_release_without_manual_cleanup() {
    let mut authority = SessionAuthority::new();
    let fail = |authority: &mut SessionAuthority| -> Result<(), &'static str> {
        let _lease = authority.begin_submission().unwrap();
        Err("native submission failed")
    };
    assert!(fail(&mut authority).is_err());
    assert_eq!(authority.require_idle(), Ok(()));
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lease = authority.begin_submission().unwrap();
        panic!("native constructor unwound");
    }));
    assert!(unwind.is_err());
    assert_eq!(authority.require_idle(), Ok(()));
}

#[test]
fn completion_can_resolve_on_another_thread_without_weakening_exclusivity() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SessionAuthority>();
    assert_send_sync::<SubmissionLease>();
    let mut authority = SessionAuthority::new();
    let lease = authority.begin_submission().unwrap();
    std::thread::spawn(move || {
        assert!(lease.resolve());
    })
    .join()
    .unwrap();
    assert_eq!(authority.require_idle(), Ok(()));
    let lease = authority.begin_submission().unwrap();
    let active = Arc::clone(&authority.active);
    drop(authority);
    assert_ne!(active.load(Ordering::Acquire), 0);
    drop(lease);
    assert_eq!(active.load(Ordering::Acquire), 0);
}

#[test]
fn ticket_exhaustion_is_deterministic_and_never_reuses_a_ticket() {
    let mut authority = SessionAuthority::new();
    authority.next_ticket = u64::MAX - 1;
    let last = authority.begin_submission().unwrap();
    assert_eq!(last.ticket, u64::MAX - 1);
    assert!(matches!(
        authority.begin_submission(),
        Err(SessionAuthorityError::Busy)
    ));
    drop(last);
    for _ in 0..2 {
        assert!(matches!(
            authority.begin_submission(),
            Err(SessionAuthorityError::TicketExhausted)
        ));
        assert_eq!(authority.require_idle(), Ok(()));
    }
}

#[test]
fn foreign_pending_ready_failed_and_cancelled_completions_preserve_mutation_authority() {
    let mut authority = SessionAuthority::new();
    let completion = pending(&mut authority);
    assert!(!completion.is_complete().unwrap());
    let mut mutations = Vec::new();
    for operation in [
        "reset",
        "rollback",
        "replacement",
        "prompt-cache",
        "inspection",
    ] {
        let result = authority.require_idle().map(|()| mutations.push(operation));
        assert_eq!(result, Err(SessionAuthorityError::Busy));
    }
    assert!(mutations.is_empty());
    completion.wait().unwrap();
    authority.require_idle().unwrap();
    let next = pending(&mut authority);
    assert!(completion.is_complete().unwrap());
    drop(completion);
    assert_eq!(authority.require_idle(), Err(SessionAuthorityError::Busy));
    next.native.state.store(2, Ordering::Release);
    assert!(next.wait().is_err());
    authority.require_idle().unwrap();
    assert!(next.is_complete().is_err());
    let cancelled = pending(&mut authority);
    let drained = Arc::clone(&cancelled.native.drained);
    drop(next);
    assert_eq!(authority.require_idle(), Err(SessionAuthorityError::Busy));
    drop(cancelled); // Native Drop asserts that the pending lease remains held.
    assert!(drained.load(Ordering::Acquire));
    authority.require_idle().unwrap();
}

struct ForeignBackend {
    constructed_active: Arc<std::sync::Mutex<Option<Arc<AtomicU64>>>>,
    native_drained: Arc<AtomicBool>,
}

struct ForeignSession {
    authority: SessionAuthority,
    _construction: ForeignCompletion,
}

impl BackendProvider for ForeignBackend {
    type ModelConfig = ();
    type Model = ();
    type Session = ForeignSession;
    type Error = std::io::Error;

    fn descriptor(&self) -> crate::BackendDescriptor {
        crate::BackendDescriptor::new("foreign", "test")
    }

    fn devices(
        &self,
    ) -> Result<Vec<(crate::DeviceDescriptor, crate::DeviceCapabilities)>, Self::Error> {
        Ok(Vec::new())
    }

    fn prepare_model(&self, (): ()) -> Result<PreparedModel<()>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }

    fn create_session(&self, _: PreparedModel<()>) -> Result<ForeignSession, Self::Error> {
        let mut authority = SessionAuthority::new();
        let mut completion = pending(&mut authority);
        completion.native.drained = Arc::clone(&self.native_drained);
        *self.constructed_active.lock().unwrap() = Some(Arc::clone(&authority.active));
        Ok(ForeignSession {
            authority,
            _construction: completion,
        })
    }

    fn session_capability_mismatch(
        &self,
        admitted: SessionCapabilities,
        realized: SessionCapabilities,
    ) -> Self::Error {
        std::io::Error::other(
            SessionAdmission::new(admitted)
                .validate(realized)
                .unwrap_err(),
        )
    }
}

impl BackendSession<ForeignBackend> for ForeignSession {
    type PrefillInput = ();
    type DecodeInput = ();
    type Output = ();
    type Completion = ForeignCompletion;

    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::new(true, false, false)
    }

    fn prefill(
        &mut self,
        _: &ForeignBackend,
        (): (),
    ) -> Result<Submission<(), ForeignCompletion>, std::io::Error> {
        self.authority
            .require_idle()
            .map_err(std::io::Error::other)?;
        Ok(Submission {
            output: (),
            completion: pending(&mut self.authority),
        })
    }

    fn decode(
        &mut self,
        backend: &ForeignBackend,
        input: (),
    ) -> Result<Submission<(), ForeignCompletion>, std::io::Error> {
        self.prefill(backend, input)
    }

    fn observe_output(
        &self,
        _: &ForeignBackend,
        _: &(),
    ) -> Result<crate::ObservationSet, std::io::Error> {
        self.authority
            .require_idle()
            .map_err(std::io::Error::other)?;
        Ok(crate::ObservationSet::new())
    }
}

#[test]
fn failed_generic_session_publication_drains_native_resources_and_releases_active_authority() {
    let active = Arc::new(std::sync::Mutex::new(None));
    let drained = Arc::new(AtomicBool::new(false));
    let backend = ForeignBackend {
        constructed_active: Arc::clone(&active),
        native_drained: Arc::clone(&drained),
    };
    let result = ModelRuntime::prepare(backend, ());
    assert!(result.is_err());
    assert!(drained.load(Ordering::Acquire));
    assert_eq!(
        active
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load(Ordering::Acquire),
        0
    );
}
