use super::*;
use crate::working_memory::{InferenceRequest, OriginalPredictionScopeRole};
use eredu_core::{PendingTextInput, TokenOutput};

fn take_roles(
    mut set: crate::working_memory::OriginalTextPredictionScopeSet,
) -> Vec<OriginalPredictionScopeRole> {
    let roles = vec![
        set.take_model_execution().unwrap(),
        set.take_sampling().unwrap(),
        set.take_sampling_event().unwrap(),
        set.take_model_validation().unwrap(),
        set.take_token_scalar().unwrap(),
    ];
    for result in [
        set.take_model_execution(),
        set.take_sampling(),
        set.take_sampling_event(),
        set.take_model_validation(),
        set.take_token_scalar(),
    ] {
        assert!(matches!(result, Err(WorkingMemoryError::AlreadyStarted)));
    }
    roles
}

#[test]
fn original_prediction_bank_accepts_real_started_decode_and_independent_retirement_tails() {
    for controlled in [false, true] {
        for capture in [false, true] {
            let (mut rt, state) = runtime(Mode {
                prediction_scopes: true,
                capture,
                ..Mode::default()
            });
            let source = capture.then(capture_source);
            let options = source.as_ref().map(|s| TextPreparationOptions {
                interventions: None,
                capture: Some(s.clone()),
            });
            let tokens = if controlled {
                ControlledTextGeneration::from_input_with_sequence(
                    &mut rt,
                    TextGenerationInput::TokenIds(vec![2]),
                    config(2),
                    Controller,
                    options,
                    GenerationSequenceRequest::new(2, &[]),
                )
                .unwrap()
                .map(|token| token.unwrap().into_output())
                .collect::<Vec<_>>()
            } else {
                TextGeneration::from_input_with_sequence(
                    &mut rt,
                    TextGenerationInput::TokenIds(vec![2]),
                    config(2),
                    TokenFilter::All,
                    options,
                    GenerationSequenceRequest::new(2, &[]),
                )
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>()
            };
            assert_eq!(
                tokens
                    .iter()
                    .map(|t| t.token_id().unwrap())
                    .collect::<Vec<_>>(),
                [7, 8]
            );
            let issued = std::mem::take(&mut state.borrow_mut().issued);
            assert_eq!(issued.len(), 2);
            let mut native = Vec::new();
            let mut recovery = Vec::new();
            for role in issued.into_iter().flat_map(take_roles) {
                let (n, r) = role.into_custody();
                native.push(n);
                recovery.push(r);
            }
            let (pool, held) = retire_request(&state);
            drop((tokens, rt, state, source));
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(native);
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                held,
                "independent Rust Recovery custody still owns original Q"
            );
            drop(recovery);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn original_prediction_bank_rejects_foreign_preparation_and_abandoned_active_step() {
    let (mut rt, state) = runtime(Mode {
        prediction_scopes: true,
        ..Mode::default()
    });
    let mut run = TextGeneration::from_input_with_sequence(
        &mut rt,
        TextGenerationInput::TokenIds(vec![2]),
        config(2),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(2, &[]),
    )
    .unwrap();
    let prepared = Rc::clone(state.borrow().active.as_ref().unwrap());
    let context = state.borrow().bound.clone().unwrap();
    let original = prepared
        .request
        .claim_step(&context, PendingTextInput::Prefill(()))
        .unwrap();
    let mut bank = prepared.prediction_bank.borrow_mut().take().unwrap();
    let roles = bank.claim(&original).unwrap();
    let reservation = prepared.owner.borrow().reservation().clone();
    assert!(matches!(
        InferenceRequest::from(&reservation).prepare_text(
            &reservation.0.execution,
            reservation.geometry(),
            config(2)
        ),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    let (mut other_rt, other_state) = runtime(Mode {
        prediction_scopes: true,
        ..Mode::default()
    });
    let other_run = TextGeneration::from_input_with_sequence(
        &mut other_rt,
        TextGenerationInput::TokenIds(vec![2]),
        config(2),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(2, &[]),
    )
    .unwrap();
    let foreign = Rc::clone(other_state.borrow().active.as_ref().unwrap());
    let other_context = other_state.borrow().bound.clone().unwrap();
    let other = foreign
        .request
        .claim_step(&other_context, PendingTextInput::Prefill(()))
        .unwrap();
    assert!(matches!(
        bank.claim(&other),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        bank.claim(&original),
        Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: 1,
            actual: 0
        })
    ));
    drop(original);
    assert!(matches!(
        prepared
            .request
            .claim_step(&context, PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(run.next().unwrap().is_err());
    assert!(run.next().is_none());
    drop((
        run,
        other_run,
        bank,
        roles,
        other,
        foreign,
        reservation,
        prepared,
    ));
    let (other_pool, _) = retire_request(&other_state);
    drop((other_rt, other_state));
    assert_eq!(other_pool.payload_used_bytes().unwrap(), 0);
    let (pool, _) = retire_request(&state);
    drop((rt, state));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn zero_output_and_initial_cancellation_never_issue_prediction_roles() {
    for maximum in [0, 2] {
        let (mut rt, state) = runtime(Mode {
            prediction_scopes: true,
            ..Mode::default()
        });
        let mut run = TextGeneration::from_input_with_sequence(
            &mut rt,
            TextGenerationInput::TokenIds(vec![2]),
            config(maximum),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(maximum, &[]),
        )
        .unwrap();
        let cancel = eredu_core::GenerationCancellationToken::new();
        if maximum != 0 {
            cancel.cancel();
        }
        assert!(run.next_cancellable(&cancel).is_none());
        assert!(state.borrow().issued.is_empty());
        drop(run);
        let (pool, _) = retire_request(&state);
        drop((rt, state));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_prediction_bank_checks_registered_source_health_before_issuing_any_role() {
    let (mut rt, state) = runtime(Mode {
        prediction_scopes: true,
        source: true,
        ..Mode::default()
    });
    let mut run = TextGeneration::from_input_with_sequence(
        &mut rt,
        TextGenerationInput::TokenIds(vec![2]),
        config(2),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(2, &[]),
    )
    .unwrap();
    state.borrow_mut().source_scope.take();
    let error = run.next().unwrap().unwrap_err();
    assert!(matches!(
        error.source().unwrap().downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(state.borrow().issued.is_empty());
    assert!(run.next().is_none());
    drop((run, error));
    state.borrow_mut().source_run.take();
    state.borrow_mut().source_reservation.take();
    let source_envelope = state.borrow().source_envelope;
    let (pool, _) = retire_request(&state);
    drop((rt, state));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        source_envelope + publication_controls() + 64
    );
}
