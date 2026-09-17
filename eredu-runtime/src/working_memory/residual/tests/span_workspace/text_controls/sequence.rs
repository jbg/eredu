use super::*;
use crate::working_memory::residual::span_workspace::SequenceExtractionError as RuntimeSequenceError;
use crate::working_memory::{
    InferenceTextPreparation, OwnedTextSpanWorkspace, WorkingMemoryFundingScope,
};
use eredu_core::run_preparation::{TextPreparationStage as Stage, TextPreparationStatus as Status};
use eredu_core::{
    ControlledTextGeneration, GenerationSequenceRequest, GenerationTokenIds, ModelRuntime,
    RetainedGenerationSequence, TextGeneration, TextGenerationInput, TextPreparationOptions,
    TokenFilter, TokenFilterController, TokenTerminalSignals,
};
use std::{cell::RefCell, error::Error as _, rc::Rc};
mod bank;
mod consumer;
mod harness;
use harness::{config, runtime, Backend, Mode, State};

struct Controller;
impl TokenFilterController for Controller {
    type Error = WorkingMemoryError;
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
fn extract(
    runtime: &mut ModelRuntime<Backend>,
    maximum: usize,
    eos: &[u32],
    controlled: bool,
    owned_input: bool,
    options: Option<TextPreparationOptions>,
) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
    let input = if owned_input {
        TextGenerationInput::Prepared(vec![2])
    } else {
        TextGenerationInput::TokenIds(vec![2])
    };
    let claim = GenerationSequenceRequest::new(maximum, eos);
    if controlled {
        let mut generation = ControlledTextGeneration::from_input_with_sequence(
            runtime,
            input,
            config(maximum),
            Controller,
            options,
            claim,
        )
        .map_err(|e| match e {
            eredu_core::ControlledTextGenerationError::Preparation(e) => e,
            _ => panic!("unexpected preparation error"),
        })?;
        let sequence = generation.take_prepared_sequence().unwrap();
        assert!(generation.take_prepared_sequence().is_none());
        Ok(sequence)
    } else {
        let mut generation = TextGeneration::from_input_with_sequence(
            runtime,
            input,
            config(maximum),
            TokenFilter::All,
            options,
            claim,
        )?;
        let sequence = generation.take_prepared_sequence().unwrap();
        assert!(generation.take_prepared_sequence().is_none());
        Ok(sequence)
    }
}
fn retire_request(state: &Rc<RefCell<State>>) -> (WorkingMemoryPool, u64) {
    let mut state = state.borrow_mut();
    let pool = state.pool.clone();
    let held = state.held;
    state.active.take();
    state.foreign.take();
    state.root.take();
    (pool, held)
}
#[test]
fn original_sequence_exact_admission_and_shared_phase_order_for_optional_capture() {
    for controlled in [false, true] {
        for owned in [false, true] {
            for capture in [false, true] {
                let (mut runtime, state) = runtime(Mode {
                    capture,
                    ..Mode::default()
                });
                let options = capture.then(|| TextPreparationOptions {
                    interventions: None, capture: Some(capture_source()),
                });
                let sequence =
                    extract(&mut runtime, 2, &[9, 7, 9], controlled, owned, options).unwrap();
                let state_ref = state.borrow();
                assert_eq!(
                    state_ref.order,
                    if owned {
                        vec!["admit", "bind", "extract", "sampling"]
                    } else {
                        vec!["admit", "bind", "extract", "prompt", "sampling"]
                    }
                );
                let stages = state_ref
                    .votes
                    .iter()
                    .map(|(stage, status)| {
                        assert_eq!(*status, Status::Ready);
                        *stage
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    stages,
                    if capture {
                        vec![
                            Stage::Admission,
                            Stage::Prompt,
                            Stage::Sampling,
                            Stage::Instrumentation,
                        ]
                    } else {
                        vec![Stage::Admission, Stage::Prompt, Stage::Sampling]
                    }
                );
                assert!(sequence.tokens().is_empty());
                assert!(state_ref.r > 0);
                assert_eq!(state_ref.held, state_ref.p + 51 + state_ref.r);
                drop(state_ref);
                let (pool, held) = retire_request(&state);
                drop(runtime);
                assert_eq!(
                    pool.used_bytes().unwrap(),
                    held,
                    "mutable provider retains the host hold; the closed run releases its decoder pin"
                );
                assert!(matches!(
                    pool.pin_registered_storage([(1u32, 64)]),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
                drop(sequence);
                assert_eq!(pool.used_bytes().unwrap(), 0);
            }
        }
    }
    let (mut runtime, state) = runtime(Mode {
        short: true,
        ..Mode::default()
    });
    assert!(extract(&mut runtime, 2, &[], false, false, None).is_err());
    assert_eq!(state.borrow().order, vec!["admit"]);
    assert_eq!(
        state.borrow().votes,
        vec![(Stage::Admission, Status::Failed)]
    );
    assert_eq!(state.borrow().pool.used_bytes().unwrap(), 64);
}
#[test]
fn original_sequence_payload_freezes_without_copy_and_partial_iterator_keeps_r() {
    let (mut runtime, state) = runtime(Mode::default());
    let mut eos = [9, 7, 9];
    let sequence = extract(&mut runtime, 7, &eos, false, false, None).unwrap();
    eos.fill(42);
    let before_votes = state.borrow().votes.clone();
    let mut sequence = sequence.prepare_storage().unwrap();
    assert_eq!(
        state.borrow().votes,
        before_votes,
        "materialization supplies no agreement"
    );
    let pointer = sequence.tokens().as_ptr();
    sequence.commit(5, TokenTerminalSignals::default()).unwrap();
    let final_token = sequence.commit(7, TokenTerminalSignals::default()).unwrap();
    assert_eq!(
        final_token.finish_reason,
        Some(eredu_core::FinishReason::Eos)
    );
    let ids = sequence.into_token_ids();
    assert_eq!(&*ids, &[5, 7]);
    assert_eq!(ids.as_ptr(), pointer);
    let alias = ids.clone();
    let mut iterator = ids.into_iter();
    assert_eq!(iterator.next(), Some(5));
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert_eq!(iterator.next_back(), Some(7));
    drop(iterator);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_sequence_zero_output_and_dormant_cancel_keep_one_original_owner() {
    for maximum in [0, 5] {
        let (mut runtime, state) = runtime(Mode::default());
        let sequence = extract(&mut runtime, maximum, &[], false, false, None).unwrap();
        let mut sequence = if maximum == 0 {
            let error = sequence.prepare_storage().unwrap_err();
            assert!(matches!(
                error
                    .cause()
                    .source()
                    .unwrap()
                    .downcast_ref::<eredu_core::GenerationError>(),
                Some(eredu_core::GenerationError::AlreadyFinished)
            ));
            error.into_sequence()
        } else {
            sequence
        };
        assert_eq!(
            sequence.finish_reason(),
            (maximum == 0).then_some(eredu_core::FinishReason::MaxTokens)
        );
        assert!(sequence.cancel());
        assert_eq!(
            sequence.finish_reason(),
            Some(eredu_core::FinishReason::Cancelled)
        );
        assert!(!sequence.cancel());
        let ids = sequence.into_token_ids();
        assert!(ids.is_empty());
        let (pool, held) = retire_request(&state);
        drop(runtime);
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop(ids);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn original_sequence_abandoned_extraction_blocks_new_and_owned_prompt_paths() {
    for owned in [false, true] {
        let (mut runtime, state) = runtime(Mode {
            abandon: true,
            ..Mode::default()
        });
        let error = extract(&mut runtime, 2, &[], false, owned, None).unwrap_err();
        assert_eq!(state.borrow().order, vec!["admit", "bind", "extract"]);
        assert_eq!(
            state.borrow().votes,
            vec![(Stage::Admission, Status::Failed)]
        );
        let state_ref = state.borrow();
        let preparation = &state_ref.active.as_ref().unwrap().request;
        assert!(preparation.claim_prompt().is_err());
        assert!(preparation.bind_prompt().is_err());
        assert!(preparation.claim_sampling(config(2)).is_err());
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<RuntimeSequenceError>()
            .unwrap()
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>()
            .unwrap();
        assert!(matches!(
            cause,
            WorkingMemoryError::PreparationAlreadyStarted
        ));
        assert_eq!(error.kind(), eredu_core::BackendFailureKind::InvalidSession);
        drop(state_ref);
        let (pool, held) = retire_request(&state);
        drop(runtime);
        assert_eq!(
            pool.used_bytes().unwrap(),
            held,
            "consumed error retains original metadata and host custody, not the retired decoder pin"
        );
        assert!(matches!(
            pool.pin_registered_storage([(1u32, 64)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn original_sequence_foreign_core_claim_cannot_take_an_earlier_bank() {
    let (mut first, old) = runtime(Mode::default());
    let sequence = extract(&mut first, 2, &[7], false, false, None).unwrap();
    drop(sequence);
    let original = Rc::clone(old.borrow().active.as_ref().unwrap());
    let (mut second, new) = runtime(Mode::default());
    new.borrow_mut().foreign = Some(Rc::clone(&original));
    let error = extract(&mut second, 2, &[7], false, false, None).unwrap_err();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(),
        Some(&eredu_core::GenerationSequenceBankRejection::Unavailable)
    );
    assert_eq!(new.borrow().votes, vec![(Stage::Admission, Status::Failed)]);
    drop(original);
    new.borrow_mut().foreign.take();
    let (pool, _) = retire_request(&old);
    drop(first);
    assert_eq!(
        pool.used_bytes().unwrap(),
        0,
        "replayed rejection holds no original custody"
    );
    drop(error);
    let (pool, _) = retire_request(&new);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_sequence_preparation_failure_owns_error_until_after_run_close() {
    let (mut runtime, state) = runtime(Mode::default());
    let sequence = extract(&mut runtime, 2, &[], false, false, None).unwrap();
    state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .run
        .borrow_mut()
        .take();
    let error = sequence.prepare_storage().unwrap_err();
    assert!(matches!(
        error
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let sequence = error.into_sequence();
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(sequence);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn original_sequence_explicit_source_survives_run_close_and_retires_at_freeze() {
    let (mut runtime, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let mut sequence = extract(&mut runtime, 2, &[], false, false, None).unwrap();
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), held + 64);
    // This source is explicitly declared on the accepted quote. Its full guard
    // outlives the run's separate credited decoder pin and the fixture root.
    drop(pool.pin_registered_storage([(1u32, 64)]).unwrap());
    assert_eq!(pool.used_bytes().unwrap(), held + 64);
    assert!(sequence.cancel());
    let ids = sequence.into_token_ids();
    assert!(ids.is_empty());
    assert_eq!(pool.used_bytes().unwrap(), held);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(ids);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_sequence_source_quarantine_is_checked_before_dormant_token_allocation() {
    let (mut runtime, state) = runtime(Mode {
        source: true,
        ..Mode::default()
    });
    let sequence = extract(&mut runtime, 2, &[], false, false, None).unwrap();
    state.borrow_mut().source_scope.take();
    let error = sequence.prepare_storage().unwrap_err();
    assert!(matches!(
        error
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    state.borrow_mut().source_run.take();
    state.borrow_mut().source_reservation.take();
    let source_envelope = state.borrow().source_envelope;
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), source_envelope + held + 64);
    drop(error);
    assert_eq!(
        pool.used_bytes().unwrap(),
        source_envelope + 64,
        "original source quarantine preserves its borrowed root after R retires"
    );
}
#[test]
fn original_sequence_actual_result_aliases_retire_concurrently_after_request() {
    let (mut runtime, state) = runtime(Mode::default());
    let mut sequence = extract(&mut runtime, 2, &[], false, false, None)
        .unwrap()
        .prepare_storage()
        .unwrap();
    sequence.commit(3, TokenTerminalSignals::default()).unwrap();
    let ids = sequence.into_token_ids();
    let aliases = (0..4)
        .map(|_| ids.clone())
        .collect::<Vec<GenerationTokenIds>>();
    drop(ids);
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), held);
    std::thread::scope(|scope| {
        let workers = aliases
            .into_iter()
            .map(|ids| {
                scope.spawn(move || {
                    assert_eq!(&*ids, &[3]);
                    drop(ids);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_sequence_controls_reject_unknown_foreign_plan_and_overflow_before_acceptance() {
    let (mut runtime, state) = runtime(Mode {
        audit: true,
        ..Mode::default()
    });
    let sequence = extract(&mut runtime, 2, &[9], false, false, None).unwrap();
    drop(sequence);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_sequence_owner_unwind_retires_actual_provider_without_refunding_live_bank() {
    let (mut runtime, state) = runtime(Mode::default());
    let sequence = extract(&mut runtime, 2, &[9], false, false, None)
        .unwrap()
        .prepare_storage()
        .unwrap();
    let panic = catch_unwind(AssertUnwindSafe(move || {
        let _sequence = sequence;
        panic!("actual original R owner unwind");
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"actual original R owner unwind")
    );
    let state_ref = state.borrow();
    let active = state_ref.active.as_ref().unwrap();
    assert_eq!(
        account(&state_ref.pool, active.owner.borrow().reservation()).1,
        state_ref.held
    );
    drop(state_ref);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod decoder;

mod prediction;

mod plain_text;

mod token_input;

mod prefill;
