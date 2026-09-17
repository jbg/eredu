use super::*;
use crate::{BackendFailureKind, SharedBackendFailure};
use std::{
    cell::Cell,
    error::Error as _,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[derive(Debug)]
struct Cause(Arc<AtomicUsize>);
impl std::fmt::Display for Cause {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("retained transfer formatted its source")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn retained_control_transfer_calls_hook_once_without_formatting_or_replacing_source() {
    let drops = Arc::new(AtomicUsize::new(0));
    let source = SharedBackendFailure::new(BackendFailureKind::Busy, Cause(drops.clone()));
    let calls = Cell::new(0);
    let mut errors = Vec::new();
    for driver in [false, true] {
        let error = source
            .retained()
            .into_failure()
            .with_operation("retained speculative owner");
        let take = |error| {
            calls.set(calls.get() + 1);
            Ok(error)
        };
        let error = if driver {
            SpeculativeControlError::driver_with_retained(
                SpeculativeDriverError::Backend(error),
                take,
            )
        } else {
            SpeculativeControlError::backend_with_retained(error, take)
        };
        let SpeculativeControlError::Backend(neutral) = &error else {
            panic!("wrong branch")
        };
        assert_eq!(neutral.kind(), BackendFailureKind::Busy);
        assert_eq!(neutral.operation(), "retained speculative owner");
        assert!(std::ptr::eq(
            neutral.source().unwrap().downcast_ref::<Cause>().unwrap(),
            source.source_error().downcast_ref::<Cause>().unwrap(),
        ));
        errors.push(error);
    }
    assert_eq!(calls.get(), 2);
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(errors.pop());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(errors);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn refused_control_transfer_preserves_ordinary_raw_and_driver_classification() {
    let raw = SpeculativeControlError::backend_with_retained(
        std::io::Error::from(std::io::ErrorKind::Interrupted),
        Err,
    );
    let driver = SpeculativeControlError::driver_with_retained(
        SpeculativeDriverError::Backend(std::io::Error::from(std::io::ErrorKind::Interrupted)),
        Err,
    );
    let SpeculativeControlError::Backend(raw) = raw else {
        panic!("wrong raw branch")
    };
    let SpeculativeControlError::Backend(driver) = driver else {
        panic!("wrong driver branch")
    };
    assert_eq!(raw.kind(), BackendFailureKind::Io);
    assert_eq!(driver.kind(), BackendFailureKind::Other);
    assert_eq!(
        raw.source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::Interrupted
    );
    let wrapped = driver
        .source()
        .unwrap()
        .downcast_ref::<SpeculativeDriverError<std::io::Error>>()
        .unwrap();
    let SpeculativeDriverError::Backend(error) = wrapped else {
        panic!("lost ordinary wrapper")
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
}

#[test]
fn nonbackend_driver_branches_never_invoke_retained_native_hook() {
    let errors: [SpeculativeDriverError<std::io::Error>; 6] = [
        SpeculativeDriverError::Preparation(BackendFailure::from_error(std::io::Error::from(
            std::io::ErrorKind::PermissionDenied,
        ))),
        SpeculativeDriverError::Coordination("exact coordination cause"),
        SpeculativeDriverError::Output(SpeculativeOutputError::publication("exact output cause")),
        SpeculativeDriverError::Generation(GenerationError::UnknownSpeculativeRequest { index: 7 }),
        SpeculativeDriverError::CompletionDeadline {
            cancellation: CompletionCancellationMode::QuarantineUntilComplete,
        },
        SpeculativeDriverError::UnsupportedCompletionCancellation {
            cancellation: CompletionCancellationMode::QuarantineUntilComplete,
        },
    ];
    for original in errors {
        let message = original.to_string();
        let discriminant = std::mem::discriminant(&original);
        let error = SpeculativeControlError::driver_with_retained(
            original,
            |_| -> Result<BackendFailure, std::io::Error> {
                panic!("nonbackend branch called native hook")
            },
        );
        let SpeculativeControlError::Backend(neutral) = error else {
            panic!("wrong branch")
        };
        assert_eq!(neutral.kind(), BackendFailureKind::Other);
        let retained = neutral
            .source()
            .unwrap()
            .downcast_ref::<SpeculativeDriverError<std::io::Error>>()
            .unwrap();
        assert_eq!(std::mem::discriminant(retained), discriminant);
        assert_eq!(retained.to_string(), message);
    }
}
