use super::*;
#[derive(Debug)]
struct Input;
impl crate::GenerationDecoderInput for Input {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
#[test]
fn original_decoder_default_rejects_before_any_existing_sequence_hook() {
    for consumer in [false, true] {
        let (mut runtime, facts, probe) = setup();
        let layout =
            crate::GenerationSequenceConsumerLayout::for_driver_types::<[u8; 1], (), io::Error>()
                .unwrap();
        let source = Input;
        let request = GenerationSequenceRequest::new(8, &[7]).with_decoder(&source);
        let request = if consumer {
            request.with_consumer(&layout)
        } else {
            request
        };
        let error = match TextGeneration::from_input_with_sequence(
            &mut runtime,
            input(&facts, false),
            config(),
            TokenFilter::All,
            None,
            request,
        ) {
            Err(e) => e,
            _ => panic!("decoder admission must be explicit"),
        };
        assert_eq!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<GenerationSequenceAdmissionError>(),
            Some(&GenerationSequenceAdmissionError::DecoderUnsupported)
        );
        assert_eq!(
            facts.borrow().sequence.votes,
            [(Stage::Admission, Status::Failed)]
        );
        assert_eq!(facts.borrow().sequence.order, ["Admission"]);
        assert!(facts.borrow().preparation_events.is_empty());
        assert!(facts.borrow().events.is_empty());
        assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn returned_sequence_without_original_decoder_keeps_provider_in_core_rejection() {
    let probe = Arc::new(Probe::default());
    probe.alive.store(1, Ordering::SeqCst);
    let sequence = RetainedGenerationSequence::from_retained_storage(Box::new(Storage {
        consumer: None,
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
    let source = Input;
    let request = GenerationSequenceRequest::new(8, &[7]).with_decoder(&source);
    let error =
        crate::backend::preparation::PreparedSequence::install(sequence, &request).unwrap_err();
    assert_eq!(probe.prepares.load(Ordering::SeqCst), 0);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 1);
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceAdmissionError>(),
        Some(&GenerationSequenceAdmissionError::InvalidSequence)
    );
    drop(error);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
}
