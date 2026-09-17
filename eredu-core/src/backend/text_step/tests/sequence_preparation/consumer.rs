use super::*;
use crate::GenerationSequenceConsumerLayout;
fn layout() -> GenerationSequenceConsumerLayout {
    GenerationSequenceConsumerLayout::for_driver_types::<[u8; 1], (), io::Error>().unwrap()
}

#[test]
fn consumer_request_default_rejects_before_existing_sequence_hook_or_work() {
    let (mut runtime, facts, probe) = setup();
    let consumer = layout();
    let result = TextGeneration::from_input_with_sequence(
        &mut runtime,
        input(&facts, false),
        config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(8, &[7]).with_consumer(&consumer),
    );
    let error = match result {
        Err(e) => e,
        _ => panic!("old sequence admission must not consume new controls"),
    };
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<GenerationSequenceAdmissionError>(),
        Some(&GenerationSequenceAdmissionError::ConsumerUnsupported)
    );
    assert_eq!(
        facts.borrow().sequence.votes,
        [(Stage::Admission, Status::Failed)]
    );
    assert_eq!(
        facts.borrow().sequence.order,
        ["Admission"],
        "only failed Admission readiness runs; old quote/extract hooks were not called"
    );
    assert!(facts.borrow().preparation_events.is_empty());
    assert!(facts.borrow().events.is_empty());
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
}

#[test]
fn consumer_provider_verification_rejects_missing_and_equal_size_foreign_types_with_owner() {
    let expected = layout();
    let foreign =
        GenerationSequenceConsumerLayout::for_driver_types::<[i8; 1], (), io::Error>().unwrap();
    assert_eq!(
        expected.retention_peak_bytes(),
        foreign.retention_peak_bytes()
    );
    assert_ne!(expected, foreign);
    for returned in [None, Some(foreign), Some(expected)] {
        let probe = Arc::new(Probe::default());
        probe.alive.store(1, Ordering::SeqCst);
        let sequence = RetainedGenerationSequence::from_retained_storage(Box::new(Storage {
            consumer: returned,
            payload: Arc::new(Payload {
                max: 8,
                eos: vec![7],
                tokens: vec![],
                committed: 0,
                probe: probe.clone(),
            }),
            fail: false,
        }))
        .unwrap();
        let request = GenerationSequenceRequest::new(8, &[7]).with_consumer(&expected);
        // This is the exact core check used after extraction and before Admission
        // readiness. This neutral provider test creates no funding certificate.
        let result = crate::backend::preparation::PreparedSequence::install(sequence, &request);
        if returned == Some(expected) {
            assert!(result.is_ok());
        } else {
            let error = result.as_ref().unwrap_err();
            assert_eq!(
                std::error::Error::source(error)
                    .unwrap()
                    .source()
                    .unwrap()
                    .downcast_ref::<GenerationSequenceAdmissionError>(),
                Some(&GenerationSequenceAdmissionError::InvalidSequence)
            );
        }
        assert_eq!(probe.prepares.load(Ordering::SeqCst), 0);
        assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
        drop(result);
        assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    }
}
