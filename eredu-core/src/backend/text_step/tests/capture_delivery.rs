use super::*;
use crate::capture::*;
use std::error::Error as _;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, Default)]
pub(super) struct CaptureFacts {
    ready: Option<CapturedStepDelivery>,
    pub(super) pending: bool,
    armed: bool,
    pub(super) no_output: bool,
    pub(super) fail_query: bool,
    pub(super) fail_wait: bool,
    fail_drain: bool,
    fail_provider_drain: bool,
    empty_pending: bool,
    drains: usize,
}
#[derive(Debug, thiserror::Error)]
#[error("capture fixture {0}")]
struct Fault(&'static str);
pub(super) fn failure(at: &'static str) -> io::Error {
    io::Error::other(Fault(at))
}
pub(super) fn submitted(facts: &mut Facts) {
    if facts.capture.armed {
        facts.capture.pending = true;
    }
}
pub(super) fn drain(facts: &Rc<RefCell<Facts>>) -> Result<Option<CapturedStepDelivery>, io::Error> {
    let mut facts = facts.borrow_mut();
    if !facts.capture.armed {
        return Ok(None);
    }
    assert_eq!(
        facts.outstanding_completions, 0,
        "delivery must follow exact completion"
    );
    facts.events.push("drain");
    facts.capture.drains += 1;
    if facts.capture.fail_provider_drain {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            Fault("provider drain"),
        ));
    }
    if facts.capture.fail_drain {
        return Err(failure("drain"));
    }
    if facts.capture.empty_pending {
        return Ok(None);
    }
    facts.capture.pending = false;
    Ok(facts.capture.ready.take())
}
#[derive(Debug)]
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn frame(outcome: CaptureStepOutcome) -> CapturedStep {
    CapturedStep {
        outcome,
        phase: CapturePhase::Decode,
        invocation: Some(CaptureInvocationShape {
            batch: 1,
            sequence: 1,
            context: Some(7),
        }),
        prediction_index: 3,
        records: vec![CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: "kept-source".into(),
            path: "layer.2.output".into(),
            node_id: "layer.2".into(),
            position: crate::ObservationPosition::BeforeIntervention,
            source_shape: Some(vec![1, 3]),
            source_dtype: Some(crate::checkpoint::TensorDtype::F32),
            selected_shape: Some(vec![1, 3]),
            outcome: CaptureOutcome::Captured,
            payload: Some(CapturePayload::Tensor(
                crate::TensorObservation::new(
                    vec![1, 3],
                    crate::TensorObservationData::F32(vec![1.25, -2.5, 7.0]),
                )
                .unwrap(),
            )),
            charged: CaptureUsage {
                captures: 1,
                retained_bytes: 128,
                host_bytes: 128,
                encoded_bytes: 0,
            },
        }],
        partitions: Vec::new(),
        interventions: Vec::new(),
        step_usage: CaptureUsage {
            captures: 1,
            retained_bytes: 128,
            host_bytes: 128,
            encoded_bytes: 0,
        },
        cumulative_usage: CaptureUsage {
            captures: 2,
            retained_bytes: 256,
            host_bytes: 256,
            encoded_bytes: 0,
        },
        capture_seconds: 0.0,
    }
}
fn arm(
    facts: &Rc<RefCell<Facts>>,
    outcome: CaptureStepOutcome,
) -> (SharedCapturedStep, Arc<AtomicUsize>) {
    let retired = Arc::new(AtomicUsize::new(0));
    let source = SharedCapturedStep::retain(frame(outcome), Retired(retired.clone()));
    let mut facts = facts.borrow_mut();
    facts.capture.armed = true;
    facts.capture.ready = Some(CapturedStepDelivery::Shared(source.clone()));
    (source, retired)
}
fn same(delivery: &CapturedStepDelivery, source: &SharedCapturedStep) {
    let actual = delivery.shared().unwrap();
    assert!(actual.same_storage(source));
    assert_eq!(actual.records().as_ptr(), source.records().as_ptr());
    assert_eq!(
        actual.records()[0].path.as_ptr(),
        source.records()[0].path.as_ptr()
    );
    assert_eq!(delivery.as_step(), source.as_step());
}
fn io_fault(error: &io::Error, at: &'static str) {
    assert_eq!(
        error.get_ref().unwrap().downcast_ref::<Fault>().unwrap().0,
        at
    );
}

#[test]
fn ordinary_and_controlled_legacy_parking_preserves_exact_shared_payload_and_blocks_advance() {
    for controlled in [false, true] {
        let (mut runtime, facts) = fixture();
        let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
        let ptr = source.records().as_ptr();
        let delivered = if controlled {
            let mut run = ControlledTextGeneration::new(
                &mut runtime,
                vec![1, 2],
                config(),
                controller(&facts, ControllerFailure::None),
            )
            .unwrap();
            drop(run.next().unwrap().unwrap());
            assert!(run.capture_pending());
            assert!(run.take_captured_step().unwrap().is_none());
            assert!(run.take_captured_step().unwrap().is_none());
            assert_eq!(facts.borrow().capture.drains, 1);
            assert!(matches!(
                run.next().unwrap(),
                Err(ControlledTextGenerationError::Preparation(_))
            ));
            assert!(run.next().is_none());
            assert!(run.capture_pending());
            assert_eq!(
                facts
                    .borrow()
                    .events
                    .iter()
                    .filter(|e| **e == "decode")
                    .count(),
                0
            );
            run.take_captured_delivery().unwrap().unwrap()
        } else {
            let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config()).unwrap();
            drop(run.next().unwrap().unwrap());
            assert!(run.take_captured_step().unwrap().is_none());
            assert!(run.take_captured_step().unwrap().is_none());
            assert_eq!(facts.borrow().capture.drains, 1);
            assert!(run
                .next()
                .unwrap()
                .unwrap_err()
                .source()
                .unwrap()
                .is::<CaptureDeliveryPending>());
            assert!(run.next().is_none());
            assert!(run.capture_pending());
            run.take_captured_delivery().unwrap().unwrap()
        };
        same(&delivered, &source);
        let delivered = delivered.into_legacy().unwrap_err();
        assert_eq!(delivered.as_step().records.as_ptr(), ptr);
        drop((runtime, facts, source));
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(delivered);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn detached_no_output_and_aborted_frame_do_not_create_a_drained_boundary() {
    for empty in [false, true] {
        let (mut runtime, facts) = fixture();
        let (source, retired) = arm(&facts, CaptureStepOutcome::Aborted);
        facts.borrow_mut().capture.no_output = true;
        facts.borrow_mut().capture.empty_pending = empty;
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .start_input(
                TextGenerationInput::TokenIds(vec![2, 3]),
                config(),
                controller(&facts, ControllerFailure::None),
            )
            .unwrap();
        assert!(driver.advance(&mut state).unwrap().is_none());
        assert!(matches!(
            state.require_quiescent(),
            Err(TextContinuationError::NotQuiescent)
        ));
        if empty {
            assert!(driver
                .take_completed_delivery(&mut state)
                .unwrap()
                .is_none());
            assert!(driver.capture_pending(&state).unwrap());
            assert!(matches!(
                state.require_quiescent(),
                Err(TextContinuationError::NotQuiescent)
            ));
            facts.borrow_mut().capture.empty_pending = false;
        }
        assert!(driver.take_completed_step(&mut state).unwrap().is_none());
        assert!(matches!(
            state.require_quiescent(),
            Err(TextContinuationError::NotQuiescent)
        ));
        let delivered = driver.take_completed_delivery(&mut state).unwrap().unwrap();
        same(&delivered, &source);
        assert_eq!(delivered.as_step().outcome, CaptureStepOutcome::Aborted);
        state.require_quiescent().unwrap();
        assert!(!driver.capture_pending(&state).unwrap());
        drop(state);
        drop(driver);
        drop((runtime, facts, source));
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(delivered);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn detached_fallible_drain_retries_same_frame_without_clearing_failed_lifecycle() {
    let (mut runtime, facts) = fixture();
    let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    drop(driver.advance(&mut state).unwrap().unwrap());
    facts.borrow_mut().capture.fail_drain = true;
    match driver.take_completed_delivery(&mut state) {
        Err(TextContinuationError::Generation(ControlledTextGenerationError::Backend(error))) => {
            io_fault(&error, "drain")
        }
        _ => panic!("expected original drain cause"),
    }
    assert!(driver.capture_pending(&state).unwrap());
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    facts.borrow_mut().capture.fail_drain = false;
    let delivered = driver.take_completed_delivery(&mut state).unwrap().unwrap();
    same(&delivered, &source);
    assert!(!driver.capture_pending(&state).unwrap());
    assert!(matches!(
        state.require_quiescent(),
        Err(TextContinuationError::Failed)
    ));
    drop(state);
    drop(driver);
    drop((runtime, facts, source));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(delivered);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn polling_and_wait_failures_retain_exact_completion_until_positive_retry_before_delivery() {
    for query in [false, true] {
        let (mut runtime, facts) = fixture();
        let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
        let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config()).unwrap();
        drop(run.next().unwrap().unwrap());
        facts.borrow_mut().capture.fail_query = query;
        facts.borrow_mut().capture.fail_wait = !query;
        let error = run.take_captured_delivery().unwrap_err();
        io_fault(
            error.source().unwrap().downcast_ref::<io::Error>().unwrap(),
            if query { "poll" } else { "wait" },
        );
        assert_eq!(facts.borrow().capture.drains, 0);
        assert_eq!(run.inner.completions.len(), 1);
        assert!(run.capture_pending());
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        facts.borrow_mut().capture.fail_query = false;
        facts.borrow_mut().capture.fail_wait = false;
        let delivered = run.take_captured_delivery().unwrap().unwrap();
        same(&delivered, &source);
        assert!(run.inner.completions.is_empty());
        assert_eq!(facts.borrow().capture.drains, 1);
        {
            let borrowed = facts.borrow();
            let events = &borrowed.events;
            assert!(
                events.iter().position(|e| *e == "wait").unwrap()
                    < events.iter().position(|e| *e == "drain").unwrap()
            );
        }
        drop(run);
        drop((runtime, facts, source));
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(delivered);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn completion_error_keeps_parked_frame_until_machine_and_external_aliases_retire() {
    let (mut runtime, facts) = fixture();
    let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
    let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config()).unwrap();
    drop(run.next().unwrap().unwrap());
    assert!(run.take_captured_step().unwrap().is_none());
    // Another retained completion is a distinct unresolved backend obligation.
    facts.borrow_mut().outstanding_completions += 1;
    run.inner.completions.push(Pending {
        facts: facts.clone(),
        settled: Cell::new(false),
    });
    facts.borrow_mut().capture.fail_wait = true;
    assert!(run.take_captured_delivery().is_err());
    assert!(run.capture_pending());
    drop(source);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    facts.borrow_mut().capture.fail_wait = false;
    drop(run);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(facts.borrow().outstanding_completions, 0);
}

#[test]
fn ordinary_and_controlled_fallible_drains_retry_without_losing_shared_owner() {
    for controlled in [false, true] {
        let (mut runtime, facts) = fixture();
        let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
        let delivered = if controlled {
            let mut run = ControlledTextGeneration::new(
                &mut runtime,
                vec![1, 2],
                config(),
                controller(&facts, ControllerFailure::None),
            )
            .unwrap();
            drop(run.next().unwrap().unwrap());
            facts.borrow_mut().capture.fail_drain = true;
            io_fault(&run.take_captured_delivery().unwrap_err(), "drain");
            assert!(run.capture_pending());
            facts.borrow_mut().capture.fail_drain = false;
            run.take_captured_delivery().unwrap().unwrap()
        } else {
            let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config()).unwrap();
            drop(run.next().unwrap().unwrap());
            facts.borrow_mut().capture.fail_drain = true;
            let error = run.take_captured_delivery().unwrap_err();
            io_fault(
                error.source().unwrap().downcast_ref::<io::Error>().unwrap(),
                "drain",
            );
            assert!(run.capture_pending());
            facts.borrow_mut().capture.fail_drain = false;
            run.take_captured_delivery().unwrap().unwrap()
        };
        same(&delivered, &source);
        drop((runtime, facts, source));
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(delivered);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn no_output_empty_terminal_transaction_requires_explicit_successful_drain() {
    let (mut runtime, facts) = fixture();
    facts.borrow_mut().capture.armed = true;
    facts.borrow_mut().capture.no_output = true;
    facts.borrow_mut().capture.empty_pending = true;
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    assert!(driver.advance(&mut state).unwrap().is_none());
    assert!(driver
        .take_completed_delivery(&mut state)
        .unwrap()
        .is_none());
    assert!(driver.capture_pending(&state).unwrap());
    assert!(matches!(
        state.require_quiescent(),
        Err(TextContinuationError::NotQuiescent)
    ));
    facts.borrow_mut().capture.empty_pending = false;
    assert!(driver
        .take_completed_delivery(&mut state)
        .unwrap()
        .is_none());
    state.require_quiescent().unwrap();
    assert!(!driver.capture_pending(&state).unwrap());
}

#[test]
fn detached_pending_block_preserves_input_and_attempt_until_retained_frame_drains() {
    let (mut runtime, facts) = fixture();
    let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(vec![1, 2]),
            config(),
            controller(&facts, ControllerFailure::None),
        )
        .unwrap();
    drop(driver.advance(&mut state).unwrap().unwrap());
    let remaining = state.remaining_tokens();
    assert!(matches!(
        driver.advance(&mut state),
        Err(TextContinuationError::NotQuiescent)
    ));
    assert_eq!(state.remaining_tokens(), remaining);
    assert_eq!(facts.borrow().contexts.len(), 1);
    let delivered = driver.take_completed_delivery(&mut state).unwrap().unwrap();
    same(&delivered, &source);
    state.require_quiescent().unwrap();
    facts.borrow_mut().capture.armed = false;
    drop(driver.advance(&mut state).unwrap().unwrap());
    assert_eq!(facts.borrow().contexts.len(), 2);
    assert_eq!(facts.borrow().contexts[1].attempt(), 1);
    driver.take_completed_delivery(&mut state).unwrap();
    drop(state);
    drop(driver);
    drop((runtime, facts, source));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(delivered);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_raw_and_retained_capture_errors_use_provider_hook_without_consuming_frame() {
    for legacy in [false, true] {
        let (mut runtime, facts) = fixture();
        let (source, retired) = arm(&facts, CaptureStepOutcome::Committed);
        let mut run = TextGeneration::new(&mut runtime, vec![1, 2], config()).unwrap();
        drop(run.next().unwrap().unwrap());
        facts.borrow_mut().capture.fail_provider_drain = true;
        let error = if legacy {
            run.take_captured_step().unwrap_err()
        } else {
            run.take_captured_delivery().unwrap_err()
        };
        assert_eq!(error.kind(), BackendFailureKind::Busy);
        assert_eq!(error.operation(), "step-provider-hook");
        io_fault(
            error.source().unwrap().downcast_ref::<io::Error>().unwrap(),
            "provider drain",
        );
        assert!(run.capture_pending());
        assert_eq!(facts.borrow().outstanding_completions, 0);
        assert_eq!(facts.borrow().capture.drains, 1);
        same(facts.borrow().capture.ready.as_ref().unwrap(), &source);
        drop(run);
        drop(runtime);
        drop(facts);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(source);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
