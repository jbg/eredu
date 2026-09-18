use super::*;
use crate::PreparedRequestRejection;

fn original(facts: &Rc<RefCell<Facts>>) -> TextGenerationInput<Prompt> {
    TextGenerationInput::OriginalPrepared(Prompt {
        ids: vec![2, 3],
        facts: facts.clone(),
    })
}

#[test]
fn original_prepared_default_rejects_real_sequence_without_legacy_fallback_in_all_routes() {
    for route in 0..3 {
        for installed in [false, true] {
            let (mut runtime, facts, probe) = setup();
            let options = installed.then(TextPreparationOptions::default);
            let request = GenerationSequenceRequest::new(8, &[7, 9]);
            let value = original(&facts);
            let map = |e| match e {
                ControlledTextGenerationError::Preparation(e) => e,
                _ => panic!("typed original-prepared admission failure required"),
            };
            let error = match route {
                0 => TextGeneration::from_input_with_sequence(
                    &mut runtime,
                    value,
                    config(),
                    TokenFilter::All,
                    options,
                    request,
                )
                .err()
                .unwrap(),
                1 => ControlledTextGeneration::from_input_with_sequence(
                    &mut runtime,
                    value,
                    config(),
                    controller(&facts, ControllerFailure::None),
                    options,
                    request,
                )
                .err()
                .map(map)
                .unwrap(),
                _ => {
                    let mut driver = TextGenerationDriver::new(&mut runtime);
                    driver
                        .start_input_with_sequence(
                            value,
                            config(),
                            controller(&facts, ControllerFailure::None),
                            options,
                            request,
                        )
                        .err()
                        .map(map)
                        .unwrap()
                }
            };
            assert_eq!(
                std::error::Error::source(&error)
                    .unwrap()
                    .downcast_ref::<PreparedRequestRejection>(),
                Some(&PreparedRequestRejection::Unsupported)
            );
            let f = facts.borrow();
            assert_eq!(f.sequence.votes, [(Stage::Admission, Status::Failed)]);
            assert_eq!(f.sequence.order, ["Admission"]);
            assert!(f.preparation_events.is_empty());
            assert!(f.events.is_empty());
            assert!(f.sequence.context.is_none());
            assert!(!f.sequence.extracted);
            assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn original_prepared_requires_sequence_and_preserves_token_source_pairing_precedence() {
    let (mut runtime, facts, _) = setup();
    let error = TextGeneration::from_input_with_options(
        &mut runtime,
        original(&facts),
        config(),
        TextPreparationOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<PreparedRequestRejection>(),
        Some(&PreparedRequestRejection::MissingSequence)
    );
    assert!(facts.borrow().preparation_events.is_empty());
    let plan = crate::TokenIdsInputPlan::new(&[2, 3]).unwrap();
    let request = GenerationSequenceRequest::new(8, &[7]).with_token_input(&plan);
    let error = TextGeneration::from_input_with_sequence(
        &mut runtime,
        original(&facts),
        config(),
        TokenFilter::All,
        None,
        request,
    )
    .err()
    .unwrap();
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<crate::TokenInputRejection>(),
        Some(&crate::TokenInputRejection::IdentityMismatch)
    );
    assert!(!facts.borrow().sequence.extracted);
    assert!(facts.borrow().preparation_events.is_empty());
}

#[test]
fn original_prepared_authentication_refusal_uses_explicit_readiness_before_any_work() {
    let (mut runtime, facts, probe) = setup();
    facts.borrow_mut().sequence.explicit_control = true;
    let error = ControlledTextGeneration::from_input_with_sequence(
        &mut runtime,
        original(&facts),
        config(),
        controller(&facts, ControllerFailure::None),
        None,
        GenerationSequenceRequest::new(8, &[7, 9]),
    ).err().expect("the mock has no completed-source admission mechanism");
    let ControlledTextGenerationError::Preparation(error) = error else {
        panic!("original prepared admission must retain its typed cause");
    };
    assert_eq!(
        std::error::Error::source(&error).unwrap().downcast_ref::<PreparedRequestRejection>(),
        Some(&PreparedRequestRejection::Unsupported),
    );
    let f = facts.borrow();
    assert_eq!(f.sequence.controls, 1);
    assert_eq!(f.sequence.votes, [(Stage::Admission, Status::Failed)]);
    assert!(f.preparation_events.is_empty());
    assert!(f.events.is_empty());
    assert!(!f.sequence.extracted);
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
}
