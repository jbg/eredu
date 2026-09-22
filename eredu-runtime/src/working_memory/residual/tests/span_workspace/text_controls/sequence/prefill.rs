use super::*;
use crate::prefill::PrefillControlRole;
use eredu_core::{PendingTextInput, TokenOutput};
#[test]
fn actual_prefill_step_issues_once_and_four_independent_custodies_retire_last() {
    for controlled in [false, true] {
        for capture in [false, true] {
            let (mut rt, state) = runtime(Mode {
                prediction_scopes: true,
                prefill_scopes: true,
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
                .map(|t| t.unwrap().into_output())
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
            let mut sets = std::mem::take(&mut state.borrow_mut().prefill_issued);
            assert_eq!(sets.len(), 1);
            let mut set = sets.pop().unwrap();
            let facts = set.facts();
            assert!(matches!(
                set.take_next(PrefillControlRole::FinalIndex),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            let mut native = Vec::new();
            let mut recovery = Vec::new();
            let mut roots = Vec::new();
            let mut projections = Vec::new();
            for ordinal in 0..facts.plan().scope_count() {
                let descriptor = facts.plan().role(ordinal).unwrap();
                let role = set.take_next(descriptor).unwrap();
                assert_eq!(role.role(), descriptor);
                let (n, r, v, p) = role.into_custody();
                native.push(n);
                recovery.push(r);
                roots.push(v);
                projections.push(p);
            }
            assert!(
                set.take_next(PrefillControlRole::SourcePreparation)
                    .is_err()
            );
            state.borrow_mut().issued.clear();
            let (pool, held) = retire_request(&state);
            drop((set, sets, tokens, rt, state, source));
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(native);
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(recovery);
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(roots);
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(projections);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn prefill_claim_rejects_equal_geometry_foreign_step_and_spent_original_without_refund() {
    let mode = Mode {
        prediction_scopes: true,
        prefill_scopes: true,
        ..Mode::default()
    };
    let (mut rt, state) = runtime(mode);
    let (mut other_rt, other_state) = runtime(mode);
    let mut run = TextGeneration::from_input_with_sequence(
        &mut rt,
        TextGenerationInput::TokenIds(vec![2]),
        config(2),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(2, &[]),
    )
    .unwrap();
    let other_run = TextGeneration::from_input_with_sequence(
        &mut other_rt,
        TextGenerationInput::TokenIds(vec![2]),
        config(2),
        TokenFilter::All,
        None,
        GenerationSequenceRequest::new(2, &[]),
    )
    .unwrap();
    let prepared = Rc::clone(state.borrow().active.as_ref().unwrap());
    let other = Rc::clone(other_state.borrow().active.as_ref().unwrap());
    let context = state.borrow().bound.clone().unwrap();
    let other_context = other_state.borrow().bound.clone().unwrap();
    let step = prepared
        .request
        .claim_step(&context, PendingTextInput::Prefill(()))
        .unwrap();
    let foreign = other
        .request
        .claim_step(&other_context, PendingTextInput::Prefill(()))
        .unwrap();
    let mut bank = prepared.prefill_bank.borrow_mut().take().unwrap();
    let before = state.borrow().pool.payload_used_bytes().unwrap();
    assert!(matches!(
        bank.claim(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(state.borrow().pool.payload_used_bytes().unwrap(), before);
    let set = bank.claim(&step).unwrap();
    assert!(matches!(
        bank.claim(&step),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    drop(step);
    assert!(run.next().unwrap().is_err());
    assert!(run.next().is_none());
    drop((run, other_run, prepared, other, foreign, bank, set));
    let (pool, _) = retire_request(&state);
    let (other_pool, _) = retire_request(&other_state);
    drop((rt, state, other_rt, other_state));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(other_pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn cancelled_initial_step_never_consumes_prefill_scope_bank() {
    let (mut rt, state) = runtime(Mode {
        prediction_scopes: true,
        prefill_scopes: true,
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
    let cancel = eredu_core::GenerationCancellationToken::new();
    cancel.cancel();
    assert!(run.next_cancellable(&cancel).is_none());
    assert!(state.borrow().prefill_issued.is_empty());
    drop(run);
    let (pool, _) = retire_request(&state);
    drop((rt, state));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn source_component_handoff_keeps_exact_account_remaining_destinations_and_escaped_spending() {
    for bytes in [0, 4] {
        let mode = Mode {
            prefill_scopes: true,
            host_source_component: Some(bytes),
            ..Mode::default()
        };
        let (mut rt, state) = runtime(mode);
        let (mut other_rt, other_state) = runtime(mode);
        let run = TextGeneration::from_input_with_sequence(
            &mut rt,
            TextGenerationInput::TokenIds(vec![2]),
            config(2),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(2, &[]),
        )
        .unwrap();
        let other_run = TextGeneration::from_input_with_sequence(
            &mut other_rt,
            TextGenerationInput::TokenIds(vec![2]),
            config(2),
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(2, &[]),
        )
        .unwrap();
        let prepared = Rc::clone(state.borrow().active.as_ref().unwrap());
        let other = Rc::clone(other_state.borrow().active.as_ref().unwrap());
        let context = state.borrow().bound.clone().unwrap();
        let step = prepared
            .request
            .claim_step(&context, PendingTextInput::Prefill(()))
            .unwrap();
        let mut bank = prepared.prefill_bank.borrow_mut().take().unwrap();
        let mut set = bank.claim(&step).unwrap();
        let facts = set.facts();
        let selected = facts.source_construction_facts().unwrap();
        let mut host = prepared
            .owner
            .borrow_mut()
            .take_host_destinations()
            .unwrap();
        let mut foreign = other.owner.borrow_mut().take_host_destinations().unwrap();
        let pool = state.borrow().pool.clone();
        let before = pool.payload_used_bytes().unwrap();
        assert!(matches!(
            set.take_source_component(&mut foreign, selected),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(set.facts(), facts);
        assert!(
            foreign
                .as_ref()
                .unwrap()
                .matches_facts(facts.host_destination_facts().unwrap())
        );
        let mut source = set.take_source_component(&mut host, selected).unwrap();
        assert_eq!(set.facts().source_construction_facts(), None);
        assert_eq!(set.facts().plan(), facts.plan());
        assert_eq!(
            set.facts().operation_control_bytes(),
            facts.operation_control_bytes()
        );
        match (host.as_ref(), set.facts().host_destination_facts()) {
            (Some(host), Some(remaining)) => assert!(host.matches_facts(remaining)),
            (None, None) => assert_eq!(bytes, 0),
            _ => panic!("source extraction and remaining destination disagree"),
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), before);
        assert!(set.take_source_component(&mut host, selected).is_err());
        let receipt = source.try_debit(32).unwrap();
        let error = source.try_debit(33).unwrap_err();
        assert_eq!(
            (source.remaining_bytes(), source.remaining_attempts()),
            (32, 0)
        );
        let values = host.as_mut().map(|host| {
            let mut values = host.try_vec::<u32>(1).unwrap();
            values.try_fill(1, [0x1234abcd]).unwrap();
            assert_eq!(values.as_slice(), &[0x1234abcd]);
            values
        });
        drop((
            step, run, other_run, prepared, other, bank, set, source, host, foreign, values,
        ));
        let (pool, held) = retire_request(&state);
        let (other_pool, _) = retire_request(&other_state);
        drop((rt, state, other_rt, other_state));
        assert_eq!(other_pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(pool.payload_used_bytes().unwrap(), held);
        drop(receipt);
        assert_eq!(pool.payload_used_bytes().unwrap(), held);
        drop(error);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
