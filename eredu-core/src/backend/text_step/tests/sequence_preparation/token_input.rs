use super::*;
use crate::{TokenIdsInputPlan, TokenInputRejection};

#[test]
fn original_input_pairing_rejects_both_directions_in_every_startup_route_including_zero() {
    for route in 0..3 {
        for maximum in [0, 8] {
            for mismatch in 0..3 {
                let (mut runtime, facts, probe) = setup();
                let plan = TokenIdsInputPlan::new(&[2, 3]).unwrap();
                let mut sampling = config().sampling();
                sampling.max_new_tokens = Some(maximum);
                let config = TextGenerationConfig::new(sampling);
                let request = GenerationSequenceRequest::new(maximum, &[7]);
                let request = if mismatch == 0 {
                    request
                } else {
                    request.with_token_input(&plan)
                };
                let value = match mismatch {
                    0 => TextGenerationInput::OriginalTokenIds,
                    1 => input(&facts, false),
                    _ => input(&facts, true),
                };
                let map = |e| match e {
                    ControlledTextGenerationError::Preparation(e) => e,
                    _ => panic!("expected typed original pairing rejection"),
                };
                let error = match route {
                    0 => TextGeneration::from_input_with_sequence(
                        &mut runtime,
                        value,
                        config,
                        TokenFilter::All,
                        None,
                        request,
                    )
                    .err()
                    .unwrap(),
                    1 => ControlledTextGeneration::from_input_with_sequence(
                        &mut runtime,
                        value,
                        config,
                        controller(&facts, ControllerFailure::None),
                        None,
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
                                config,
                                controller(&facts, ControllerFailure::None),
                                None,
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
                        .downcast_ref::<TokenInputRejection>(),
                    Some(&TokenInputRejection::IdentityMismatch)
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
    }
}
#[test]
fn original_input_default_rejects_without_existing_quote_or_extract_and_plan_is_borrowed() {
    let mut source = Vec::with_capacity(1024);
    source.extend([2, 3]);
    let plan = TokenIdsInputPlan::new(&source).unwrap();
    assert_eq!(plan.tokens().as_ptr(), source.as_ptr());
    assert_eq!(plan.destination_bytes(), 8);
    assert_eq!(
        TokenIdsInputPlan::new(&[]).unwrap_err(),
        TokenInputRejection::Empty
    );
    let (mut runtime, facts, probe) = setup();
    let error = TextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        plan,
        config(),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(8, &[7]),
    )
    .err()
    .unwrap();
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<TokenInputRejection>(),
        Some(&TokenInputRejection::Unsupported)
    );
    assert_eq!(facts.borrow().sequence.order, ["Admission"]);
    assert!(facts.borrow().preparation_events.is_empty());
    assert_eq!(probe.alive.load(Ordering::SeqCst), 0);
    source.clear(); // no source borrow is retained after synchronous startup
}
