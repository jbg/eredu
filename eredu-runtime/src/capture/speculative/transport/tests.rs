use super::super::signal;
use super::*;
use eredu_core::{BackendFailureKind, SharedBackendFailure};
use std::{
    cell::{Cell, RefCell},
    error::Error as _,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Debug)]
struct Cause {
    id: u32,
    drops: Arc<AtomicUsize>,
}
impl std::fmt::Display for Cause {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("retained source was formatted")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn source(id: u32) -> (SharedBackendFailure, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    (
        SharedBackendFailure::new(
            BackendFailureKind::Busy,
            Cause {
                id,
                drops: drops.clone(),
            },
        ),
        drops,
    )
}
struct Retained {
    calls: Cell<usize>,
}
impl SpeculativeCaptureErrorTransport<SharedBackendFailure, SharedBackendFailure> for Retained {
    fn take_retained(
        &self,
        error: CaptureExecutionError<SharedBackendFailure>,
    ) -> Result<
        (SharedBackendFailure, SpeculativeControlError),
        CaptureExecutionError<SharedBackendFailure>,
    > {
        self.calls.set(self.calls.get() + 1);
        match error {
            CaptureExecutionError::Backend(source) => Ok((
                source.retained(),
                SpeculativeControlError::Backend(source.into_failure()),
            )),
            error => Err(error),
        }
    }
    fn ordinary(&self, _: &CaptureExecutionError<SharedBackendFailure>) -> SharedBackendFailure {
        panic!("retained path called ordinary mapper")
    }
}
fn assert_control(error: &SpeculativeControlError, expected: &SharedBackendFailure) {
    let SpeculativeControlError::Backend(error) = error else {
        panic!("wrong branch")
    };
    assert_eq!(error.kind(), BackendFailureKind::Busy);
    let actual = error.source().unwrap().downcast_ref::<Cause>().unwrap();
    let expected = expected.source_error().downcast_ref::<Cause>().unwrap();
    assert!(std::ptr::eq(actual, expected));
}

#[test]
fn paired_capture_sources_survive_observer_drop_in_both_final_orders() {
    for control_first in [false, true] {
        let (source, drops) = source(7);
        let failure = RefCell::new(None);
        let transport = Retained {
            calls: Cell::new(0),
        };
        let execution = signal(
            &failure,
            &transport,
            CaptureExecutionError::Backend(source.retained()),
        );
        let control = failure.borrow_mut().take().unwrap();
        assert_control(&control, &execution);
        assert_eq!(
            execution.source_error().downcast_ref::<Cause>().unwrap().id,
            7
        );
        assert_eq!(transport.calls.get(), 1);
        drop(source);
        drop(transport);
        drop(failure);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        if control_first {
            drop(control);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(execution);
        } else {
            drop(execution);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(control);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn first_control_failure_stays_while_later_execution_retains_its_own_source() {
    let (first, first_drops) = source(11);
    let (later, later_drops) = source(19);
    let failure = RefCell::new(None);
    let transport = Retained {
        calls: Cell::new(0),
    };
    let first_execution = signal(
        &failure,
        &transport,
        CaptureExecutionError::Backend(first.retained()),
    );
    let later_execution = signal(
        &failure,
        &transport,
        CaptureExecutionError::Backend(later.retained()),
    );
    assert_control(failure.borrow().as_ref().unwrap(), &first);
    assert_eq!(
        later_execution
            .source_error()
            .downcast_ref::<Cause>()
            .unwrap()
            .id,
        19
    );
    assert_eq!(transport.calls.get(), 2);
    drop(first);
    drop(later);
    drop(first_execution);
    drop(failure);
    assert_eq!(first_drops.load(Ordering::SeqCst), 1);
    assert_eq!(later_drops.load(Ordering::SeqCst), 0);
    drop(later_execution);
    assert_eq!(later_drops.load(Ordering::SeqCst), 1);
}

#[derive(Debug)]
struct Ordinary {
    id: u32,
    drops: Arc<AtomicUsize>,
}
impl std::fmt::Display for Ordinary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ordinary {}", self.id)
    }
}
impl std::error::Error for Ordinary {}
impl Drop for Ordinary {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn ordinary_mapper_runs_before_cell_loan_and_keeps_first_source_and_latest_signal() {
    let failure = RefCell::new(None);
    let calls = Cell::new(0);
    let first_drops = Arc::new(AtomicUsize::new(0));
    let later_drops = Arc::new(AtomicUsize::new(0));
    let map = |error: &CaptureExecutionError<Ordinary>| {
        assert!(
            failure.try_borrow_mut().is_ok(),
            "ordinary callback ran under cell loan"
        );
        calls.set(calls.get() + 1);
        error.to_string()
    };
    let first = signal(
        &failure,
        &map,
        CaptureExecutionError::Backend(Ordinary {
            id: 3,
            drops: first_drops.clone(),
        }),
    );
    let later = signal(
        &failure,
        &map,
        CaptureExecutionError::Backend(Ordinary {
            id: 5,
            drops: later_drops.clone(),
        }),
    );
    assert!(first.contains("ordinary 3"));
    assert!(later.contains("ordinary 5"));
    assert_eq!(calls.get(), 2);
    let stored = failure.borrow();
    let Some(SpeculativeControlError::Backend(error)) = stored.as_ref() else {
        panic!("wrong ordinary branch")
    };
    assert_eq!(error.kind(), BackendFailureKind::Other);
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<Ordinary>()
            .unwrap()
            .id,
        3
    );
    assert_eq!(first_drops.load(Ordering::SeqCst), 0);
    assert_eq!(later_drops.load(Ordering::SeqCst), 1);
    drop(stored);
    drop(failure);
    assert_eq!(first_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_callback_unwind_keeps_failure_cell_empty_and_retires_input_once() {
    let failure = RefCell::new(None);
    let drops = Arc::new(AtomicUsize::new(0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        signal(
            &failure,
            &|_: &CaptureExecutionError<Ordinary>| -> () {
                assert!(failure.try_borrow_mut().is_ok());
                std::panic::panic_any(23_u32);
            },
            CaptureExecutionError::Backend(Ordinary {
                id: 3,
                drops: drops.clone(),
            }),
        );
    }));
    assert_eq!(result.unwrap_err().downcast_ref::<u32>(), Some(&23));
    assert!(failure.borrow().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn paired_cell_loan_refusal_retires_both_local_aliases_without_early_source_drop() {
    let (source, drops) = source(31);
    let failure = RefCell::new(None);
    let held = failure.borrow_mut();
    let transport = Retained {
        calls: Cell::new(0),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        signal(
            &failure,
            &transport,
            CaptureExecutionError::Backend(source.retained()),
        )
    }));
    assert!(result.is_err());
    assert_eq!(transport.calls.get(), 1);
    assert!(held.is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(held);
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_admission_stays_typed_after_owned_transfer_is_refused() {
    let failure = RefCell::new(None);
    let signal = signal(
        &failure,
        &|error: &CaptureExecutionError<std::io::Error>| error.to_string(),
        CaptureExecutionError::Admission(eredu_core::capture::CaptureError::Invalid(
            "exact admission".into(),
        )),
    );
    assert!(signal.contains("exact admission"));
    assert!(
        matches!(failure.into_inner(), Some(SpeculativeControlError::Capture(eredu_core::capture::CaptureError::Invalid(message))) if message == "exact admission")
    );
}
