use super::{submit_or_synchronously_drain, CompletionSubmissionFailure, MlxRealtimeCompletion};
use crate::backend::nn::tensor::{validate_token_domain, TokenValidationScope};
use eredu_core::backend::Completion;
use safemlx::{Array, Device, DeviceType, Stream};
use std::{cell::RefCell, rc::Rc};

#[derive(Debug)]
struct RetainedRoot {
    calls: Rc<RefCell<Vec<&'static str>>>,
}

impl Drop for RetainedRoot {
    fn drop(&mut self) {
        self.calls.borrow_mut().push("drop");
    }
}

#[test]
fn successful_exact_submission_does_not_run_the_synchronous_drain() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let retained = vec![RetainedRoot {
        calls: Rc::clone(&calls),
    }];

    let (completion, retained) = submit_or_synchronously_drain(
        retained,
        |roots| {
            assert_eq!(roots.len(), 1);
            calls.borrow_mut().push("submit");
            Ok::<_, &'static str>("event")
        },
        |_| {
            calls.borrow_mut().push("drain");
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(completion, "event");
    assert_eq!(&*calls.borrow(), &["submit"]);
    drop(retained);
    assert_eq!(&*calls.borrow(), &["submit", "drop"]);
}

#[test]
fn failed_event_creation_retains_roots_through_the_synchronous_drain() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let retained = vec![RetainedRoot {
        calls: Rc::clone(&calls),
    }];

    let error = submit_or_synchronously_drain(
        retained,
        |_| {
            calls.borrow_mut().push("submit");
            Err::<(), _>("submission")
        },
        |roots| {
            assert_eq!(roots.len(), 1);
            assert_eq!(&*calls.borrow(), &["submit"]);
            calls.borrow_mut().push("drain");
            Ok(())
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        CompletionSubmissionFailure::Drained {
            submission: "submission"
        }
    );
    assert_eq!(&*calls.borrow(), &["submit", "drain", "drop"]);
}

#[test]
fn synchronous_drain_failure_is_reported_only_after_retained_roots_are_held() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let retained = vec![RetainedRoot {
        calls: Rc::clone(&calls),
    }];

    let error = submit_or_synchronously_drain(
        retained,
        |_| {
            calls.borrow_mut().push("submit");
            Err::<(), _>("submission")
        },
        |roots| {
            assert_eq!(roots.len(), 1);
            calls.borrow_mut().push("drain");
            Err("asynchronous execution failure")
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        CompletionSubmissionFailure::DrainReported {
            submission: "submission",
            drain: "asynchronous execution failure"
        }
    );
    assert_eq!(&*calls.borrow(), &["submit", "drain", "drop"]);
}

#[test]
fn completed_invalid_token_scope_fails_readiness_before_publication() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let scope = TokenValidationScope::begin().unwrap();
    let tokens =
        validate_token_domain(&Array::from_slice(&[5_u32], &[1]), 5, None, &stream).unwrap();
    let validations = scope.finish();
    let expected_retained = 1 + validations.arrays().count();
    let completion = MlxRealtimeCompletion::submit_retained(vec![tokens], validations).unwrap();

    assert_eq!(completion.retained_resources(), expected_retained);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let readiness_error = loop {
        match completion.is_complete() {
            Ok(false) => {
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
            Ok(true) => panic!("invalid token scope must not become publishable"),
            Err(error) => break error.to_string(),
        }
    };
    assert!(
        readiness_error.contains("outside 0..5"),
        "{readiness_error}"
    );
    let wait_error = completion.wait().unwrap_err().to_string();
    assert!(wait_error.contains("outside 0..5"), "{wait_error}");
}
