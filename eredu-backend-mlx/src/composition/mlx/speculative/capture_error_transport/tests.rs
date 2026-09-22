use super::*;
use eredu_core::{BackendFailureKind, SharedBackendFailure};
use eredu_runtime::working_memory::{MemoryLedger, OriginalTextControlGuard};
use std::{
    error::Error as _,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Debug)]
struct Cause {
    drops: Arc<AtomicUsize>,
    held_at_drop: Arc<AtomicBool>,
    pool: Option<MemoryLedger>,
    // Last: concrete payload and enclosing source allocation retire before Q.
    _controls: Option<OriginalTextControlGuard>,
}
impl std::fmt::Display for Cause {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("paired retained cause formatted")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            self.held_at_drop
                .store(pool.fixture_host_charge().unwrap() > 0, Ordering::SeqCst);
        }
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn pair(
    state_preserved: bool,
    controls: Option<OriginalTextControlGuard>,
    pool: Option<MemoryLedger>,
) -> (
    Error,
    SpeculativeControlError,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
) {
    // This construction is outside the transfer itself. The genuine-controls
    // test proves custody, not that an additional source has been admitted.
    let drops = Arc::new(AtomicUsize::new(0));
    let held = Arc::new(AtomicBool::new(false));
    let source = SharedBackendFailure::new(
        BackendFailureKind::Busy,
        Cause {
            drops: drops.clone(),
            held_at_drop: held.clone(),
            pool,
            _controls: controls,
        },
    );
    let incoming = Error::retained_original(source.retained(), state_preserved);
    let transport = RetainedCaptureTransport::new(|_: &CaptureExecutionError<Error>| -> Error {
        panic!("retained pair used ordinary callback")
    });
    let (execution, control) =
        match transport.take_retained(CaptureExecutionError::Backend(incoming)) {
            Ok(pair) => pair,
            Err(_) => panic!("closed original source was refused"),
        };
    assert_eq!(execution.model_state_preserved(), state_preserved);
    let SpeculativeControlError::Backend(neutral) = &control else {
        panic!("wrong control branch")
    };
    assert_eq!(neutral.kind(), BackendFailureKind::Busy);
    assert_eq!(neutral.operation(), "backend operation");
    assert!(std::ptr::eq(
        neutral.source().unwrap().downcast_ref::<Cause>().unwrap(),
        source.source_error().downcast_ref::<Cause>().unwrap()
    ));
    assert!(std::ptr::eq(
        execution.source().unwrap().downcast_ref::<Cause>().unwrap(),
        source.source_error().downcast_ref::<Cause>().unwrap()
    ));
    drop(source);
    (execution, control, drops, held)
}

pub(in crate::composition::mlx) fn fixture_pair(
    controls: &OriginalTextControlGuard,
    pool: &MemoryLedger,
    state_preserved: bool,
) -> (
    Error,
    SpeculativeControlError,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
) {
    pair(state_preserved, Some(controls.clone()), Some(pool.clone()))
}

#[test]
fn retained_capture_pair_preserves_actual_backend_source_and_state_preservation() {
    for state_preserved in [false, true] {
        let (execution, control, drops, _) = pair(state_preserved, None, None);
        let neutral = execution.into_backend_failure();
        assert_eq!(neutral.kind(), BackendFailureKind::Busy);
        let SpeculativeControlError::Backend(control) = control else {
            panic!("wrong control branch")
        };
        assert!(std::ptr::eq(
            neutral.source().unwrap().downcast_ref::<Cause>().unwrap(),
            control.source().unwrap().downcast_ref::<Cause>().unwrap()
        ));
        drop(neutral);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(control);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn retained_capture_transport_returns_admission_and_other_backend_owners_unchanged() {
    let transport = RetainedCaptureTransport::new(|_: &CaptureExecutionError<Error>| -> Error {
        panic!("extraction invoked ordinary callback")
    });
    let message = String::from("exact admission source");
    let address = message.as_ptr();
    let error = transport.take_retained(CaptureExecutionError::Admission(
        eredu_core::capture::CaptureError::Invalid(message),
    ));
    let Err(CaptureExecutionError::Admission(eredu_core::capture::CaptureError::Invalid(message))) =
        error
    else {
        panic!("admission was rewritten")
    };
    assert_eq!(message.as_ptr(), address);
    let drops = Arc::new(AtomicUsize::new(0));
    let boxed = Box::new(Cause {
        drops: drops.clone(),
        held_at_drop: Arc::new(AtomicBool::new(false)),
        pool: None,
        _controls: None,
    });
    let address = std::ptr::from_ref(&*boxed);
    let error = transport.take_retained(CaptureExecutionError::Backend(Error::Other(boxed)));
    let Err(CaptureExecutionError::Backend(Error::Other(error))) = error else {
        panic!("ordinary backend variant was rewritten")
    };
    assert!(std::ptr::eq(
        error.downcast_ref::<Cause>().unwrap(),
        address
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
