use super::*;
use crate::backend::{nn::tensor::validate_token_domain, ExecutionContext};
use std::{
    error::Error as _,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

struct ForeignRuntime {
    release: mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    released: Arc<AtomicBool>,
}
impl ForeignRuntime {
    fn hold() -> Self {
        let (held_tx, held_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let released = Arc::new(AtomicBool::new(false));
        let done = released.clone();
        let thread = std::thread::spawn(move || {
            crate::backend::submission_recovery::wait_for_retirement(|| {
                safemlx::try_with_submission_retirement(|| {
                    held_tx.send(()).unwrap();
                    // A regression cannot strand the test: expiry unlocks, and
                    // the caller checks it returned before that happened.
                    let _ = release_rx.recv_timeout(Duration::from_secs(5));
                    done.store(true, Ordering::SeqCst);
                })
                .is_some()
            });
        });
        held_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        Self {
            release,
            thread: Some(thread),
            released,
        }
    }
    fn assert_still_held(&self) {
        assert!(!self.released.load(Ordering::SeqCst));
    }
}
impl Drop for ForeignRuntime {
    fn drop(&mut self) {
        let _ = self.release.send(());
        if let Some(thread) = self.thread.take() {
            let result = thread.join();
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }
}

fn original(error: &Error) -> &Error {
    let Error::OutputObservation(failure) = error else {
        panic!("closed observation error: {error}")
    };
    failure.source().unwrap().downcast_ref::<Error>().unwrap()
}
fn model(token: i32, stream: &Stream) -> (MlxSessionCompletion, SessionAuthority) {
    let validations = TokenValidationScope::begin().unwrap();
    validate_token_domain(&Array::from_int(token), 4, None, stream).unwrap();
    let validations = validations.finish();
    safemlx::transforms::async_eval_with_event(validations.arrays())
        .unwrap()
        .synchronize()
        .unwrap();
    let mut authority = SessionAuthority::new();
    let output = model_session::model_submission(
        Array::from_slice(&[3.0_f32, 7.0], &[1, 2]),
        validations,
        true,
        authority.begin_submission().unwrap(),
    );
    (output.completion, authority)
}

#[test]
fn token_clones_reuse_the_actual_scalar_while_distinct_tokens_and_health_remain_exact() {
    let context = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let stream = context.stream();
    let (model, authority) = model(3, stream);
    let first = MlxTextToken::new(
        Array::from_slice(&[17_u32], &[1]),
        stream.clone(),
        model.owner().clone(),
    );
    let clone = first.clone();
    // This is the same constructor used for copied/restored token payloads:
    // an identical submission owner does not imply an identical token value.
    let distinct = MlxTextToken::new(
        Array::from_slice(&[31_i32], &[1]),
        stream.clone(),
        model.owner().clone(),
    );
    assert_eq!(first.token_id().unwrap(), 17);
    drop(first);
    {
        let held = ForeignRuntime::hold();
        for _ in 0..16 {
            assert_eq!(clone.token_id().unwrap(), 17);
        }
        held.assert_still_held();
    }
    assert_eq!(distinct.token_id().unwrap(), 31);
    model.wait().unwrap();
    assert_eq!(authority.require_idle(), Ok(()));
    model.owner().reject_unresolved();
    let first_error = clone.token_id().unwrap_err();
    let second_error = clone.token_id().unwrap_err();
    assert!(std::ptr::eq(
        original(&first_error),
        original(&second_error)
    ));
    assert!(
        distinct.token_id().is_err(),
        "cached success must recheck owner health"
    );
}

#[test]
fn actual_model_validation_retains_its_first_typed_source_after_completion_drop() {
    let context = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (model, authority) = model(4, context.stream());
    let first = model.wait().unwrap_err();
    assert!(matches!(original(&first), Error::Exception(_)));
    assert!(first.to_string().contains("outside 0..4"));
    let second = {
        let held = ForeignRuntime::hold();
        let second = model.wait().unwrap_err();
        held.assert_still_held();
        second
    };
    assert!(std::ptr::eq(original(&first), original(&second)));
    assert_eq!(authority.require_idle(), Ok(()));
    drop(model);
    assert!(std::ptr::eq(original(&first), original(&second)));
    drop(first);
    assert!(second.to_string().contains("outside 0..4"));
}

#[test]
fn actual_scalar_failure_is_not_retried_or_replaced_by_the_poison_diagnostic() {
    let context = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (model, _) = model(3, context.stream());
    let token = MlxTextToken::new(
        Array::from_slice(&[13_u32, 29], &[2]),
        context.stream().clone(),
        model.owner().clone(),
    );
    let alias = token.clone();
    let first = token.token_id().unwrap_err();
    assert!(matches!(original(&first), Error::Exception(_)));
    assert!(model.owner().ensure_healthy().is_err());
    drop(token);
    let held = ForeignRuntime::hold();
    let second = alias.token_id().unwrap_err();
    assert!(std::ptr::eq(original(&first), original(&second)));
    held.assert_still_held();
}

#[test]
fn sampled_completion_repeats_success_but_rejects_reentry_and_later_poison() {
    let context = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let (model, authority) = model(3, context.stream());
    let sampled = MlxCompletion::submission(Array::from_slice(&[23_u32], &[1])).unwrap();
    let mut sampling = model.owner().recovery().unwrap();
    sampling.seal();
    let completion = MlxTextCompletion {
        model,
        token: sampled.completion,
        recovery: RefCell::new(Some(sampling)),
        observation: Observation::new(),
    };
    {
        let _active = completion.observation.enter().unwrap();
        assert!(matches!(
            completion.is_complete(),
            Err(Error::OutputObservationReentrant)
        ));
        assert!(matches!(
            completion.wait(),
            Err(Error::OutputObservationReentrant)
        ));
        assert!(!completion.resources_releasable());
        assert!(authority.require_idle().is_err());
    }
    completion.wait().unwrap();
    {
        let held = ForeignRuntime::hold();
        for _ in 0..16 {
            assert!(completion.is_complete().unwrap());
            completion.wait().unwrap();
            assert!(completion.model.is_complete().unwrap());
            completion.model.wait().unwrap();
        }
        held.assert_still_held();
    }
    assert_eq!(sampled.output.evaluated().unwrap().as_slice::<u32>(), &[23]);
    assert_eq!(authority.require_idle(), Ok(()));
    completion.model.owner().reject_unresolved();
    let first = completion.wait().unwrap_err();
    let second = {
        let held = ForeignRuntime::hold();
        let second = completion.wait().unwrap_err();
        held.assert_still_held();
        second
    };
    assert!(std::ptr::eq(original(&first), original(&second)));
}

#[test]
fn recovery_progress_callbacks_run_after_the_actual_completion_slot_loan_ends() {
    use crate::backend::submission_recovery::Status;
    use std::{
        cell::Cell,
        rc::{Rc, Weak},
    };
    struct Terminal;
    impl Probe for Terminal {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            Status {
                settled: true,
                failed: false,
                blocked: false,
            }
        }
    }
    type Slot = RefCell<Option<Recovery<Check, Terminal>>>;
    struct Check {
        slot: Weak<Slot>,
        observed: Rc<Cell<usize>>,
    }
    impl Retention for Check {
        fn observe(&self, _: Status) {
            let slot = self.slot.upgrade().unwrap();
            let loan = slot
                .try_borrow_mut()
                .expect("progress callback under a completion loan");
            assert!(loan.is_none(), "the actual original owner must be detached");
            self.observed.set(self.observed.get() + 1);
        }
    }
    let slot = Rc::new(RefCell::new(None));
    let observed = Rc::new(Cell::new(0));
    *slot.borrow_mut() = Some(Recovery::with_probe(
        Check {
            slot: Rc::downgrade(&slot),
            observed: observed.clone(),
        },
        Terminal,
    ));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        progress_recovery(&slot).is_some_and(|status| status.settled)
    });
    assert!(observed.get() > 0);
    assert!(
        slot.borrow().is_some(),
        "progress restores the same original owner"
    );
    let owner = take_recovery(&slot);
    drop(owner);
}

#[test]
fn actual_model_and_scalar_failures_retain_ordinary_custody_through_cached_error_aliases() {
    fn exception<'a>(
        mut error: &'a (dyn std::error::Error + 'static),
    ) -> &'a safemlx::error::Exception {
        loop {
            if let Some(value) = error.downcast_ref() {
                return value;
            }
            error = error
                .source()
                .expect("actual native failure remains in source chain");
        }
    }
    let context = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for scalar in [false, true] {
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let host = eredu_core::HostPreparationAuthority::retain(pool.acquire_unquoted().unwrap());
        let (mut model, authority) = model(if scalar { 3 } else { 4 }, context.stream());
        model.retain_ordinary_capture(Some(host.clone()));
        let (first, second) = if scalar {
            model.wait().unwrap();
            let token = MlxTextToken::new_with_scalar_scope_and_capture(
                Array::from_slice(&[13_u32, 29], &[2]),
                context.stream().clone(),
                model.owner().clone(),
                None,
                Some(host.clone()),
            );
            let alias = token.clone();
            let first = token.token_id().unwrap_err();
            assert!(model.owner().ensure_healthy().is_err());
            drop(token);
            let held = ForeignRuntime::hold();
            let second = alias.token_id().unwrap_err();
            held.assert_still_held();
            drop(held);
            drop(alias);
            (first, second)
        } else {
            let first = model.wait().unwrap_err();
            let held = ForeignRuntime::hold();
            let second = model.wait().unwrap_err();
            held.assert_still_held();
            drop(held);
            assert_eq!(authority.require_idle(), Ok(()));
            (first, second)
        };
        assert!(std::ptr::eq(original(&first), original(&second)));
        assert!(matches!(original(&first), Error::OrdinaryCapture(_)));
        let diagnostic = exception(&first).what().to_owned();
        assert!(!diagnostic.is_empty());
        drop(model);
        drop(host);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(first);
        assert_eq!(exception(&second).what(), diagnostic);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(second);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}
