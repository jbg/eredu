use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use super::{
    model_session::{ResourceOperation, ScopeRetention, SubmissionResources},
    recovery::{self, Probe, Recovery, Retention, Status},
    SessionAuthority,
};
use eredu_core::TextGenerationBackend as _;

struct FakeProbe(Rc<Cell<Status>>);
impl Probe for FakeProbe {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        self.0.get()
    }
}

struct Payload(Arc<AtomicUsize>);
impl Drop for Payload {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct Retained {
    // Native resources drop before the scope ticket can release authority.
    _payload: Rc<Payload>,
    ticket: ScopeRetention,
}
impl Retention for Retained {
    fn observe(&self, status: Status) {
        self.ticket.observe(status);
    }
}

fn status(settled: bool, failed: bool, blocked: bool) -> Status {
    Status {
        settled,
        failed,
        blocked,
    }
}

#[test]
fn late_terminal_failure_cannot_publish_a_successful_model_operation() {
    let mut authority = SessionAuthority::new();
    let poison = Rc::new(Cell::new(false));
    let owner = SubmissionResources::new(authority.begin_submission().unwrap(), Rc::clone(&poison));
    let native = Rc::new(Cell::new(status(false, false, false)));
    let recovery = Recovery::with_probe(owner.ticket(), FakeProbe(Rc::clone(&native)));
    let first = recovery.progress();
    assert!(!first.failed && !first.blocked);
    native.set(status(true, true, false));
    let result = super::model_session::complete_model_operation(42, owner, recovery);
    assert!(result.is_err());
    assert!(poison.get());
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
}

#[test]
fn recovery_requires_fresh_restoration_and_no_native_failure() {
    use super::model_session::restored_error_permits_recovery;
    use super::Error;
    let restored = Error::after_model_call("restored", Some(1), Some(2));
    for settled in [false, true] {
        assert!(restored_error_permits_recovery(
            status(settled, false, false),
            &restored
        ));
    }
    for native in [
        status(false, true, false),
        status(false, false, true),
        status(true, true, false),
        status(true, false, true),
    ] {
        assert!(!restored_error_permits_recovery(native, &restored));
    }
    for (before, after) in [
        (Some(1), Some(1)),
        (Some(2), Some(1)),
        (Some(u64::MAX), None),
        (None, None),
        (None, Some(1)),
    ] {
        let stale = Error::after_model_call("not newly restored", before, after);
        assert!(!restored_error_permits_recovery(
            status(true, false, false),
            &stale
        ));
    }
    assert!(!restored_error_permits_recovery(
        status(true, false, false),
        &Error::ArchitectureModel("unknown host failure".into())
    ));
}

#[test]
fn unresolved_scope_retains_payload_and_lease_after_public_owners_drop() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut authority = SessionAuthority::new();
    let poisoned = Rc::new(Cell::new(false));
    let owner =
        SubmissionResources::new(authority.begin_submission().unwrap(), Rc::clone(&poisoned));
    let native = Rc::new(Cell::new(status(false, false, true)));
    let payload = Rc::new(Payload(Arc::clone(&drops)));
    let guard = Recovery::with_probe(
        Retained {
            _payload: Rc::clone(&payload),
            ticket: owner.ticket(),
        },
        FakeProbe(Rc::clone(&native)),
    );
    owner.request_release();
    drop(payload);
    drop(owner);
    drop(guard);
    recovery::wait_for_retirement(|| poisoned.get());
    assert!(poisoned.get());
    assert!(authority.require_idle().is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    recovery::reap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    native.set(status(true, true, false));
    recovery::reap();
    recovery::wait_for_retirement(|| drops.load(Ordering::SeqCst) == 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
    assert!(
        poisoned.get(),
        "terminal recovery does not unpoison the session"
    );
}

#[test]
fn one_settled_scope_cannot_release_another_unobservable_scope_ticket() {
    let mut authority = SessionAuthority::new();
    let owner = SubmissionResources::new(
        authority.begin_submission().unwrap(),
        Rc::new(Cell::new(false)),
    );
    let first = Recovery::with_probe(
        owner.ticket(),
        FakeProbe(Rc::new(Cell::new(status(true, false, false)))),
    );
    let native = Rc::new(Cell::new(status(false, false, true)));
    let second = Recovery::with_probe(owner.ticket(), FakeProbe(Rc::clone(&native)));
    owner.request_release();
    drop(first);
    assert!(authority.require_idle().is_err());
    drop(second);
    assert!(authority.require_idle().is_err());
    native.set(status(true, false, false));
    recovery::reap();
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
    let next = authority.begin_submission().unwrap();
    owner.request_release();
    drop(owner);
    assert!(
        authority.require_idle().is_err(),
        "old owner cannot resolve a newer ticket"
    );
    drop(next);
}

#[test]
fn retained_payload_is_released_before_a_resolved_owner_allows_new_mutation() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut payload = Rc::new(Payload(Arc::clone(&drops)));
    let mut authority = SessionAuthority::new();
    let owner = SubmissionResources::new(
        authority.begin_submission().unwrap(),
        Rc::new(Cell::new(false)),
    );
    let guard = Recovery::with_probe(
        Retained {
            _payload: Rc::clone(&payload),
            ticket: owner.ticket(),
        },
        FakeProbe(Rc::new(Cell::new(status(true, false, false)))),
    );
    owner.request_release();
    assert!(Rc::get_mut(&mut payload).is_none());
    drop(guard);
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
    assert!(Rc::get_mut(&mut payload).is_some());
    drop(payload);
    recovery::wait_for_retirement(|| drops.load(Ordering::SeqCst) == 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(owner);
}

#[test]
fn unwinding_does_not_release_unobservable_native_resources() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut authority = SessionAuthority::new();
    let poisoned = Rc::new(Cell::new(false));
    let owner =
        SubmissionResources::new(authority.begin_submission().unwrap(), Rc::clone(&poisoned));
    let native = Rc::new(Cell::new(status(false, false, true)));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = Recovery::with_probe(
            Retained {
                _payload: Rc::new(Payload(Arc::clone(&drops))),
                ticket: owner.ticket(),
            },
            FakeProbe(Rc::clone(&native)),
        );
        owner.request_release();
        panic!("injected unwind after native submission");
    }));
    assert!(result.is_err());
    assert!(authority.require_idle().is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    native.set(status(true, false, false));
    recovery::reap();
    recovery::wait_for_retirement(|| drops.load(Ordering::SeqCst) == 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
}

#[test]
fn submitting_thread_exit_permanently_retains_unobservable_payload_and_authority() {
    let drops = Arc::new(AtomicUsize::new(0));
    let thread_drops = Arc::clone(&drops);
    let authority = std::thread::spawn(move || {
        let mut authority = SessionAuthority::new();
        let owner = SubmissionResources::new(
            authority.begin_submission().unwrap(),
            Rc::new(Cell::new(false)),
        );
        let guard = Recovery::with_probe(
            Retained {
                _payload: Rc::new(Payload(thread_drops)),
                ticket: owner.ticket(),
            },
            FakeProbe(Rc::new(Cell::new(status(false, false, true)))),
        );
        owner.request_release();
        drop(guard);
        authority
    })
    .join()
    .unwrap();
    assert!(authority.require_idle().is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn swallowed_native_failure_cannot_turn_resource_operation_into_success() {
    let mut authority = SessionAuthority::new();
    let poisoned = Rc::new(Cell::new(false));
    let owner =
        SubmissionResources::new(authority.begin_submission().unwrap(), Rc::clone(&poisoned));
    let operation = ResourceOperation::with_probe(
        &owner,
        FakeProbe(Rc::new(Cell::new(status(true, true, false)))),
    );
    assert!(operation.finish(Ok(7)).is_err());
    assert!(poisoned.get());
    owner.request_release();
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
}

#[test]
fn successful_pending_resource_operation_retains_authority_without_poisoning() {
    let mut authority = SessionAuthority::new();
    let poisoned = Rc::new(Cell::new(false));
    let owner =
        SubmissionResources::new(authority.begin_submission().unwrap(), Rc::clone(&poisoned));
    let native = Rc::new(Cell::new(status(false, false, false)));
    let operation = ResourceOperation::with_probe(&owner, FakeProbe(Rc::clone(&native)));
    let (value, guard) = operation.finish(Ok(7)).unwrap();
    assert_eq!(value, 7);
    assert!(!poisoned.get());
    owner.request_release();
    drop(guard);
    assert!(authority.require_idle().is_err());
    assert!(!poisoned.get());
    native.set(status(true, false, false));
    recovery::reap();
    recovery::wait_for_retirement(|| authority.require_idle().is_ok());
    assert!(authority.require_idle().is_ok());
    assert!(!poisoned.get());
}

#[test]
fn host_error_and_unwind_poison_and_retain_pending_resource_operations() {
    for unwind in [false, true] {
        let mut authority = SessionAuthority::new();
        let poisoned = Rc::new(Cell::new(false));
        let owner =
            SubmissionResources::new(authority.begin_submission().unwrap(), Rc::clone(&poisoned));
        let native = Rc::new(Cell::new(status(false, false, false)));
        let operation = ResourceOperation::with_probe(&owner, FakeProbe(Rc::clone(&native)));
        if unwind {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _operation = operation;
                panic!("injected resource-operation unwind");
            }))
            .is_err());
        } else {
            assert!(operation
                .finish::<()>(Err(super::Error::ArchitectureModel(
                    "host error after accepted work".into()
                )))
                .is_err());
        }
        owner.request_release();
        assert!(poisoned.get());
        assert!(authority.require_idle().is_err());
        native.set(status(true, false, false));
        recovery::reap();
        recovery::wait_for_retirement(|| authority.require_idle().is_ok());
        assert!(authority.require_idle().is_ok());
        assert!(
            poisoned.get(),
            "settling work must not unpoison a mutated session"
        );
    }
}

fn native_session(backend: &super::MlxBackend<'_>) -> (tempfile::TempDir, super::MlxModelSession) {
    use eredu_core::BackendProvider as _;
    let artifact = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let model =
        eredu_core::load_model(backend, artifact.path(), crate::MlxLoadRequest::default()).unwrap();
    let session = backend.create_session(model).unwrap();
    (artifact, session)
}

#[test]
fn restored_execution_error_excludes_reuse_until_native_retirement() {
    use eredu_core::Completion as _;
    let stream = crate::test_stream();
    let backend = super::MlxBackend::new(stream, stream);
    let (_artifact, mut session) = native_session(&backend);
    let native = Rc::new(Cell::new(status(false, false, false)));
    let error = session
        .test_failed_operation(
            FakeProbe(Rc::clone(&native)),
            super::Error::after_model_call("restored model failure", Some(1), Some(2)),
            true,
        )
        .unwrap_err();
    assert!(error.model_state_preserved());

    let entered = Cell::new(false);
    let pending = session
        .with_model_operation(|_| {
            entered.set(true);
            Ok(())
        })
        .unwrap_err();
    assert!(
        !entered.get(),
        "pending recovery must exclude model mutation"
    );
    assert!(!pending.to_string().contains("fenced"));
    assert!(session.reset().is_err());
    for error in [
        super::MlxBackend::synchronize_session(&backend, &session).unwrap_err(),
        super::MlxBackend::reset_session(&backend, &mut session).unwrap_err(),
    ] {
        assert_eq!(error.kind(), eredu_core::BackendFailureKind::Busy);
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<super::Error>());
    }
    assert!(session.submit_token_decode(&backend, 2).is_err());

    native.set(status(true, false, false));
    recovery::wait_for_retirement(|| session.ensure_no_submission_in_flight().is_ok());
    super::MlxBackend::synchronize_session(&backend, &session).unwrap();
    super::MlxBackend::reset_session(&backend, &mut session).unwrap();
    session
        .submit_token_decode(&backend, 2)
        .unwrap()
        .completion
        .wait()
        .unwrap();
}

#[test]
fn restored_execution_error_keeps_late_native_failure_terminal() {
    let stream = crate::test_stream();
    let backend = super::MlxBackend::new(stream, stream);
    for failed in [
        status(false, true, false),
        status(false, false, true),
        status(true, true, false),
    ] {
        let (_artifact, mut session) = native_session(&backend);
        let native = Rc::new(Cell::new(status(false, false, false)));
        assert!(session
            .test_failed_operation(
                FakeProbe(Rc::clone(&native)),
                super::Error::after_model_call("restored model failure", Some(1), Some(2)),
                true,
            )
            .unwrap_err()
            .model_state_preserved());
        native.set(failed);
        recovery::wait_for_retirement(|| {
            session
                .ensure_no_submission_in_flight()
                .unwrap_err()
                .to_string()
                .contains("fenced")
        });
        native.set(status(true, false, false));
        recovery::wait_for_retirement(|| session.test_payload_weak().strong_count() == 1);
        assert!(session.reset().unwrap_err().to_string().contains("fenced"));
        assert_eq!(
            super::MlxBackend::synchronize_session(&backend, &session)
                .unwrap_err()
                .kind(),
            eredu_core::BackendFailureKind::InvalidSession,
        );
        assert_eq!(
            super::MlxBackend::reset_session(&backend, &mut session)
                .unwrap_err()
                .kind(),
            eredu_core::BackendFailureKind::InvalidSession,
        );
        assert!(session.submit_token_decode(&backend, 2).is_err());
    }
}

#[test]
fn restored_execution_error_retains_payload_after_session_drop() {
    let stream = crate::test_stream();
    let backend = super::MlxBackend::new(stream, stream);
    let (_artifact, mut session) = native_session(&backend);
    let native = Rc::new(Cell::new(status(false, false, false)));
    let payload = session.test_payload_weak();
    assert!(session
        .test_failed_operation(
            FakeProbe(Rc::clone(&native)),
            super::Error::after_model_call("restored model failure", Some(1), Some(2)),
            true,
        )
        .unwrap_err()
        .model_state_preserved());
    drop(session);
    recovery::reap();
    assert!(payload.upgrade().is_some());
    native.set(status(true, false, false));
    recovery::wait_for_retirement(|| payload.upgrade().is_none());
}

#[test]
fn pending_recovery_does_not_excuse_unknown_or_enclosing_operation_errors() {
    let stream = crate::test_stream();
    let backend = super::MlxBackend::new(stream, stream);
    for (error, allow_preservation) in [
        (
            super::Error::ArchitectureModel("unknown host failure".into()),
            true,
        ),
        (
            super::Error::after_model_call("stale restoration", Some(1), Some(1)),
            true,
        ),
        (
            super::Error::after_model_call("enclosed model restored", Some(1), Some(2)),
            false,
        ),
    ] {
        let (_artifact, mut session) = native_session(&backend);
        let native = Rc::new(Cell::new(status(false, false, false)));
        assert!(session
            .test_failed_operation(FakeProbe(Rc::clone(&native)), error, allow_preservation)
            .is_err());
        assert!(session.reset().unwrap_err().to_string().contains("fenced"));
        native.set(status(true, false, false));
        recovery::wait_for_retirement(|| session.test_payload_weak().strong_count() == 1);
        assert!(session.reset().unwrap_err().to_string().contains("fenced"));
    }
}

#[test]
fn unsupported_observer_preflight_does_not_poison_an_unmodified_session() {
    use eredu_core::Completion as _;
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = super::MlxBackend::new(context.stream(), context.stream());
    let (_artifact, mut session) = native_session(&backend);
    assert!(session
        .install_embedded_prediction_observers(
            eredu_runtime::NoopObserver,
            eredu_runtime::NoopObserver
        )
        .is_err());
    session
        .submit_token_decode(&backend, 1)
        .unwrap()
        .completion
        .wait()
        .unwrap();
}

#[test]
fn completed_native_output_retains_entire_session_payload_after_session_drop() {
    use eredu_core::Completion as _;
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = super::MlxBackend::new(context.stream(), context.stream());
    let (_artifact, mut session) = native_session(&backend);
    let output = session.submit_token_decode(&backend, 1).unwrap();
    let payload = session.test_payload_weak();
    drop(session);
    assert!(payload.upgrade().is_some());
    output.completion.wait().unwrap();
    assert!(
        payload.upgrade().is_none(),
        "terminal completion releases active session ownership; ordinary cleanup may remain staged"
    );
}

#[test]
fn terminal_session_retirement_defers_manager_drop_until_runtime_is_unlocked() {
    use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};
    use std::sync::{mpsc, Mutex};
    use std::time::Duration;

    struct ManagerOwner {
        manager: Arc<Mutex<()>>,
        drops: Arc<AtomicUsize>,
        unsafe_drop: Arc<AtomicUsize>,
    }
    impl Drop for ManagerOwner {
        fn drop(&mut self) {
            // A real resident-manager lease locks this mutex. Use try_lock in
            // the probe so the regression fails causally instead of hanging.
            if !safemlx::can_reclaim_submission_resources() || self.manager.try_lock().is_err() {
                self.unsafe_drop.fetch_add(1, Ordering::SeqCst);
            }
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct WholeSession {
        _payload: Rc<OrdinaryRetirement<super::model_session::SessionPayload>>,
    }
    impl Retention for WholeSession {
        fn observe(&self, _: Status) {}
    }

    ordinary_retirement::reclaim();
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = super::MlxBackend::new(context.stream(), context.stream());
    let (_artifact, mut session) = native_session(&backend);
    let manager = Arc::new(Mutex::new(()));
    let drops = Arc::new(AtomicUsize::new(0));
    let unsafe_drop = Arc::new(AtomicUsize::new(0));
    session.set_retirement_probe(Box::new(ManagerOwner {
        manager: Arc::clone(&manager),
        drops: Arc::clone(&drops),
        unsafe_drop: Arc::clone(&unsafe_drop),
    }));
    let native = Rc::new(Cell::new(status(false, false, false)));
    let retained = Recovery::with_probe(
        WholeSession {
            _payload: session.test_payload_weak().upgrade().unwrap(),
        },
        FakeProbe(Rc::clone(&native)),
    );
    drop(session);
    drop(retained);
    let (ready_tx, ready_rx) = mpsc::channel();
    let (start_tx, start_rx) = mpsc::channel();
    let (attempt_tx, attempt_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _manager = manager.lock().unwrap();
        ready_tx.send(()).unwrap();
        start_rx.recv().unwrap();
        attempt_tx.send(()).unwrap();
        // The worker needs runtime while retaining its manager state, exactly
        // the lock order that a runtime-held payload destructor must not invert.
        let value = safemlx::Array::from_slice(&[1.0_f32], &[1]);
        safemlx::transforms::eval([&value]).unwrap();
        drop(value);
        done_tx.send(()).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    recovery::wait_for_retirement(|| {
        safemlx::try_with_submission_retirement(|| {
            start_tx.send(()).unwrap();
            attempt_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            native.set(status(true, false, false));
            recovery::reap();
            ordinary_retirement::reclaim();
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        })
        .is_some()
    });
    done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    worker.join().unwrap();
    ordinary_retirement::reclaim();
    recovery::wait_for_retirement(|| drops.load(Ordering::SeqCst) == 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(unsafe_drop.load(Ordering::SeqCst), 0);
}

#[test]
fn unknown_host_failure_after_real_cpu_work_poison_excludes_later_model_mutation() {
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = super::MlxBackend::new(context.stream(), context.stream());
    let (_artifact, mut session) = native_session(&backend);
    let mut evidence = safemlx::SubmissionScope::begin().unwrap();
    let result: Result<(), super::Error> = session.with_model_operation(|_| {
        let value = safemlx::Array::from_slice(&[1.0_f32, 2.0], &[2]).copy(context.stream())?;
        safemlx::transforms::eval([&value])?;
        Err(super::Error::ArchitectureModel(
            "injected host failure after native execution".into(),
        ))
    });
    evidence.seal();
    assert!(evidence.status().has_work());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("injected host failure"));
    let entered = Cell::new(false);
    let retry_error = session
        .with_model_operation(|_| {
            entered.set(true);
            Ok(())
        })
        .unwrap_err()
        .to_string();
    assert!(retry_error.contains("session is fenced"));
    assert!(retry_error.contains("injected host failure"));
    assert!(!entered.get());
    assert!(session.reset().is_err());
}

#[test]
fn unwinding_after_real_cpu_work_poison_excludes_native_session_reuse() {
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = super::MlxBackend::new(context.stream(), context.stream());
    let (_artifact, mut session) = native_session(&backend);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<(), super::Error> = session.with_model_operation(|_| {
            let value = safemlx::Array::from_slice(&[3.0_f32], &[1]).copy(context.stream())?;
            safemlx::transforms::eval([&value])?;
            panic!("injected unwind after real CPU work");
        });
    }));
    assert!(result.is_err());
    assert!(session.submit_token_decode(&backend, 2).is_err());
}
