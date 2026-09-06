use super::{completion::submission_failure, MlxRealtimeCompletion};
use crate::backend::{
    nn::tensor::{validate_token_domain, TokenValidationScope},
    submission_recovery::{self, Probe, Recovery, Retention, Status},
};
use eredu_core::backend::Completion;
use eredu_runtime::RealtimeCompletionCreationError;
use safemlx::{Array, Device, DeviceType, Stream};
use std::{cell::Cell, rc::Rc};

struct RetainedRoot(Rc<Cell<usize>>);
impl Drop for RetainedRoot {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
impl Retention for RetainedRoot {
    fn observe(&self, _: Status) {}
}
struct FakeProbe(Rc<Cell<Status>>);
impl Probe for FakeProbe {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        self.0.get()
    }
}

#[test]
fn failed_submission_keeps_completion_and_roots_without_retrying_evaluation() {
    let drops = Rc::new(Cell::new(0));
    let state = Rc::new(Cell::new(Status {
        settled: false,
        failed: true,
        blocked: true,
    }));
    let completion = Recovery::with_probe(
        RetainedRoot(Rc::clone(&drops)),
        FakeProbe(Rc::clone(&state)),
    );
    let failure = submission_failure(completion, "event creation failed", false);
    assert!(matches!(
        failure,
        RealtimeCompletionCreationError::AfterSubmission { .. }
    ));
    assert_eq!(drops.get(), 0);
    drop(failure);
    submission_recovery::reap();
    assert_eq!(
        drops.get(),
        0,
        "dropping an error cannot free unresolved work"
    );
    state.set(Status {
        settled: true,
        failed: true,
        blocked: false,
    });
    submission_recovery::reap();
    assert_eq!(drops.get(), 1, "terminal failure is safe to release");
}

#[test]
fn only_terminal_proof_allows_an_error_without_an_owned_completion() {
    let drops = Rc::new(Cell::new(0));
    let state = Rc::new(Cell::new(Status {
        settled: true,
        failed: true,
        blocked: false,
    }));
    let completion = Recovery::with_probe(RetainedRoot(Rc::clone(&drops)), FakeProbe(state));
    let failure = submission_failure(completion, "event creation failed", true);
    assert!(matches!(
        failure,
        RealtimeCompletionCreationError::BeforeSubmission(_)
    ));
    assert_eq!(drops.get(), 1);
}

#[test]
fn pending_without_a_failure_is_not_a_release_proof() {
    let drops = Rc::new(Cell::new(0));
    let state = Rc::new(Cell::new(Status {
        settled: false,
        failed: false,
        blocked: false,
    }));
    let completion = Recovery::with_probe(
        RetainedRoot(Rc::clone(&drops)),
        FakeProbe(Rc::clone(&state)),
    );
    let failure = submission_failure(completion, "host operation failed", false);
    drop(failure);
    submission_recovery::reap();
    assert_eq!(drops.get(), 0);
    state.set(Status {
        settled: true,
        failed: false,
        blocked: false,
    });
    submission_recovery::reap();
    assert_eq!(drops.get(), 1);
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
