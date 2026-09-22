use super::*;
use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

pub(in crate::working_memory) fn wait_for_contention<T>(lock: &ControlMutex<T>) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !lock.observed_contention() {
        assert!(
            Instant::now() < deadline,
            "worker never reached actual WouldBlock"
        );
        std::thread::yield_now(); // Test harness only, outside production locks.
    }
}

#[test]
fn request_control_mutex_contenders_progress_and_retire_payload_once() {
    struct Payload {
        value: usize,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    let lock = Arc::new(ControlMutex::new(Payload {
        value: 0,
        retired: retired.clone(),
    }));
    let hold = lock.lock().unwrap();
    let start = Arc::new(Barrier::new(9));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let alias = lock.clone();
        let start = start.clone();
        workers.push(std::thread::spawn(move || {
            start.wait();
            for _ in 0..64 {
                alias.lock().unwrap().value += 1;
            }
        }));
    }
    start.wait();
    wait_for_contention(&lock);
    assert_eq!(hold.value, 0);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(hold);
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(lock.lock().unwrap().value, 512);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(lock);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn request_control_mutex_poison_returns_actual_guard_and_remains_poisoned() {
    let lock = ControlMutex::new(7);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = lock.lock().unwrap();
        *guard = 11;
        panic!("fixture poisons an actual held exclusive guard");
    }));
    assert!(panic.is_err());
    let mut guard = lock.lock().unwrap_err().into_inner();
    assert_eq!(*guard, 11);
    *guard = 19;
    drop(guard);
    assert_eq!(**lock.lock().unwrap_err().get_ref(), 19);
}

#[test]
fn request_control_mutex_debug_never_reads_or_formats_the_locked_payload() {
    struct NoDebug;
    let lock = ControlMutex::new(NoDebug);
    let guard = lock.lock().unwrap();
    assert_eq!(format!("{lock:?}"), "ControlMutex { .. }");
    assert!(!lock.observed_contention());
    drop(guard);
}

#[test]
fn request_control_mutex_unknown_mechanism_has_no_original_layout() {
    assert!(matches!(
        require_layout(false),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert!(require_layout(true).is_ok());
    assert_eq!(
        require_known_layout().is_ok(),
        !cfg!(target_os = "solid_asp3")
    );
}
