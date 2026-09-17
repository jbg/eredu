use super::*;
use std::error::Error as _;

fn count() -> usize {
    crate::backend::failure::source_retirement_count_for_test()
}

#[test]
fn actual_rejected_sequence_source_retires_before_its_retained_provider() {
    let baseline = count();
    let (mut runtime, facts, probe) = setup();
    facts.borrow_mut().sequence.bad_result = true;
    let error = match TextGeneration::from_input_with_sequence(
        &mut runtime,
        input(&facts, false),
        config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(8, &[7]),
    ) {
        Err(error) => error,
        Ok(_) => panic!("actual core rejection required"),
    };
    assert_eq!(
        facts.borrow().sequence.votes,
        [(Stage::Admission, Status::Failed)]
    );
    assert_eq!(facts.borrow().preparation_events, ["admit", "bind"]);
    assert!(facts.borrow().events.is_empty());
    assert_eq!(probe.prepares.load(Ordering::SeqCst), 0);
    assert_eq!(error.kind(), BackendFailureKind::InvalidInput);
    assert_eq!(
        error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceAdmissionError>(),
        Some(&GenerationSequenceAdmissionError::InvalidSequence)
    );
    drop(runtime);
    drop(facts);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
    assert_eq!(count(), baseline);
    drop(error);
    assert_eq!(
        probe.source_retirements_at_drop.load(Ordering::SeqCst),
        baseline + 1
    );
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    assert_eq!(count(), baseline + 1);
}

#[test]
fn actual_extraction_failure_flattening_preserves_provider_until_closed_retirement() {
    let baseline = count();
    let (mut runtime, facts, probe) = setup();
    facts.borrow_mut().sequence.fail_extract = true;
    let error = match ControlledTextGeneration::from_input_with_sequence(
        &mut runtime,
        input(&facts, false),
        config(),
        controller(&facts, ControllerFailure::None),
        None,
        GenerationSequenceRequest::new(8, &[7]),
    ) {
        Err(ControlledTextGenerationError::Preparation(error)) => error,
        _ => panic!("actual extraction failure required"),
    };
    let address = std::ptr::from_ref(
        error
            .source()
            .unwrap()
            .downcast_ref::<ExtractFailure>()
            .unwrap(),
    );
    let error = BackendFailure::from_error(error.with_operation("sequence admission"));
    assert!(std::ptr::eq(
        error
            .source()
            .unwrap()
            .downcast_ref::<ExtractFailure>()
            .unwrap(),
        address
    ));
    assert_eq!(error.operation(), "sequence admission");
    assert_eq!(
        facts.borrow().sequence.votes,
        [(Stage::Admission, Status::Failed)]
    );
    assert!(facts.borrow().events.is_empty());
    drop(runtime);
    drop(facts);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
    assert_eq!(count(), baseline);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _error = error;
        std::panic::panic_any(91_u64);
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<u64>(), Some(&91));
    assert_eq!(
        probe.source_retirements_at_drop.load(Ordering::SeqCst),
        baseline + 1
    );
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    assert_eq!(count(), baseline + 1);
}

#[test]
fn actual_rejection_diagnostic_is_the_private_core_source_envelope() {
    let baseline = count();
    let bytes = GenerationSequenceAdmissionError::rejected_sequence_retention_peak_bytes().unwrap();
    assert!(
        bytes
            >= BackendFailure::source_retention_peak_bytes::<GenerationSequenceAdmissionError>()
                .unwrap()
    );
    assert!(bytes >= std::mem::size_of::<RetainedGenerationSequence>());
    assert_eq!(count(), baseline);
}
