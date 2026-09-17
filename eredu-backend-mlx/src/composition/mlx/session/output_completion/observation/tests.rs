use super::*;
use std::{
    error::Error as _,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

thread_local! {
    static CHECK_DROP: RefCell<Option<Rc<dyn Fn()>>> = RefCell::new(None);
}
struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        CHECK_DROP.with(|slot| {
            slot.replace(None);
        });
    }
}
#[derive(Debug)]
struct Cause(Arc<AtomicUsize>);
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("exact observation cause")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        let check = CHECK_DROP.with(|slot| slot.borrow().clone());
        if let Some(check) = check {
            check();
        }
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn original(error: &Error) -> &Error {
    let Error::OutputObservation(failure) = error else {
        panic!("expected retained source")
    };
    failure.source().unwrap().downcast_ref().unwrap()
}

#[test]
fn first_error_source_is_immutable_and_later_input_drops_after_the_state_loan() {
    let state = Rc::new(Observation::<()>::new());
    let weak = Rc::downgrade(&state); // Synthetic state checker; no native owner Weak.
    CHECK_DROP.with(|slot| {
        slot.replace(Some(Rc::new(move || {
            if let Some(state) = weak.upgrade() {
                assert!(state.result.try_borrow_mut().is_ok());
            }
        })))
    });
    let _reset = Reset;
    let drops = Arc::new(AtomicUsize::new(0));
    let loan = state.enter().unwrap();
    let first = loan
        .complete(Err(Error::Other(Box::new(Cause(drops.clone())))))
        .unwrap_err();
    let replacement = loan
        .complete(Err(Error::Other(Box::new(Cause(drops.clone())))))
        .unwrap_err();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(std::ptr::eq(original(&first), original(&replacement)));
    drop(loan);
    drop(state);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(replacement);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn interrupted_observation_releases_its_loan_without_allowing_another_attempt() {
    let state = Observation::<u32>::new();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _loan = state.enter().unwrap();
        assert!(matches!(
            state.enter(),
            Err(Error::OutputObservationReentrant)
        ));
        assert!(state.result.try_borrow_mut().is_ok());
        panic!("original observation panic");
    }))
    .unwrap_err();
    assert_eq!(
        *panic.downcast_ref::<&str>().unwrap(),
        "original observation panic"
    );
    assert!(state.result.try_borrow_mut().is_ok());
    let loan = state.enter().unwrap();
    assert!(matches!(
        loan.cached(),
        Some(Err(Error::OutputObservationInterrupted))
    ));
}

#[test]
fn ordinary_cached_observation_errors_keep_actual_host_after_completion_state_drops() {
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let host = eredu_core::HostPreparationAuthority::retain(pool.acquire_unquoted().unwrap());
    let state = Observation::<u32>::with_ordinary_capture(Some(host));
    let loan = state.enter().unwrap();
    let first = loan
        .complete(Err(Error::Exception(safemlx::error::Exception::custom(
            "exact asynchronous native cause",
        ))))
        .unwrap_err();
    let second = loan.cached().unwrap().unwrap_err();
    assert!(std::ptr::eq(original(&first), original(&second)));
    assert!(matches!(original(&first), Error::OrdinaryCapture(_)));
    drop(loan);
    drop(state);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(first);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(second);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}
#[test]
fn attaching_ordinary_observation_custody_preserves_cached_success_and_first_failure() {
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let mut state = Observation::<u32>::new();
    {
        let loan = state.enter().unwrap();
        assert_eq!(loan.complete(Ok(17)).unwrap(), 17);
    }
    state.retain_ordinary_capture(Some(eredu_core::HostPreparationAuthority::retain(
        pool.acquire_unquoted().unwrap(),
    )));
    let loan = state.enter().unwrap();
    assert_eq!(loan.cached().unwrap().unwrap(), 17);
    let failure = loan
        .complete(Err(Error::ArchitectureModel("late owner failure".into())))
        .unwrap_err();
    assert!(matches!(original(&failure), Error::OrdinaryCapture(_)));
    drop(loan);
    drop(state);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(failure);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}
