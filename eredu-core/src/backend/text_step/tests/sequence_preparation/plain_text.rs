use super::*;
#[derive(Debug)]
struct PlainInput;
impl crate::GenerationDecoderInput for PlainInput {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn output_kind(&self) -> crate::GenerationDecoderOutput {
        crate::GenerationDecoderOutput::PlainText
    }
}
#[test]
fn original_plain_projection_defaults_to_rejection_in_the_original_admission_phase() {
    for route in 0..3 {
        let (mut runtime, facts, probe) = setup();
        let source = PlainInput;
        let request = GenerationSequenceRequest::new(8, &[7]).with_decoder(&source);
        let ids = input(&facts, false);
        let error = match route {
            0 => match TextGeneration::from_input_with_sequence(
                &mut runtime,
                ids,
                config(),
                TokenFilter::All,
                None,
                request,
            ) {
                Err(e) => e,
                Ok(_) => panic!("explicit original projection hook required"),
            },
            1 => match ControlledTextGeneration::from_input_with_sequence(
                &mut runtime,
                ids,
                config(),
                controller(&facts, ControllerFailure::None),
                None,
                request,
            ) {
                Err(ControlledTextGenerationError::Preparation(e)) => e,
                _ => panic!("original Admission rejection"),
            },
            _ => {
                let mut driver = TextGenerationDriver::new(&mut runtime);
                match driver.start_input_with_sequence(
                    ids,
                    config(),
                    controller(&facts, ControllerFailure::None),
                    None,
                    request,
                ) {
                    Err(ControlledTextGenerationError::Preparation(e)) => e,
                    _ => panic!("original Admission rejection"),
                }
            }
        };
        assert_eq!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<GenerationSequenceAdmissionError>(),
            Some(&GenerationSequenceAdmissionError::PlainTextUnsupported)
        );
        assert_eq!(facts.borrow().sequence.order, ["Admission"]);
        assert_eq!(
            facts.borrow().sequence.votes,
            [(Stage::Admission, Status::Failed)]
        );
        assert!(facts.borrow().preparation_events.is_empty());
        assert!(facts.borrow().events.is_empty());
        assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    }
}
#[derive(Debug)]
struct SuffixOnly(Storage);
impl RetainedGenerationStorage for SuffixOnly {
    fn matches_decoder_input(&self, _: Option<&dyn crate::GenerationDecoderInput>) -> bool {
        true
    }
    fn max_tokens(&self) -> usize {
        self.0.max_tokens()
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.0.eos_token_ids()
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        self.0.prepare_tokens()
    }
    fn token_slots(&self) -> &[u32] {
        self.0.token_slots()
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        self.0.token_slots_mut()
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> crate::GenerationTokenIds {
        Box::new(self.0).into_token_ids(committed)
    }
}
#[test]
fn accepting_a_decoder_identity_does_not_authorize_a_missing_plain_projection() {
    let probe = Arc::new(Probe::default());
    probe.alive.store(1, Ordering::SeqCst);
    let sequence =
        RetainedGenerationSequence::from_retained_storage(Box::new(SuffixOnly(Storage {
            consumer: None,
            fail: false,
            payload: Arc::new(Payload {
                max: 8,
                eos: vec![7],
                tokens: vec![],
                committed: 0,
                probe: probe.clone(),
            }),
        })))
        .unwrap();
    let request = GenerationSequenceRequest::new(8, &[7]).with_decoder(&PlainInput);
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
