//! Read-only validation against contexts issued by the ordinary core machine.
use super::*;

fn unchanged(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    )
}

#[test]
fn ordinary_and_controlled_ready_checks_preserve_first_permit_and_outputs() {
    for controlled in [false, true] {
        let (prepared, pool, _) = new_preparation(3, true);
        let (mut runtime, facts) = runtime(Some(prepared.clone()));
        let expected = [7, 8, 9];
        if controlled {
            let generation = ControlledTextGeneration::new(
                &mut runtime,
                vec![1; 7],
                config(3),
                Controller(facts.clone()),
            )
            .unwrap();
            let context = facts.borrow().bound[0].clone();
            let before = unchanged(&pool);
            for _ in 0..4 {
                prepared.validate_initial_ready_context(&context).unwrap();
                prepared
                    .clone()
                    .validate_initial_ready_context(&context.clone())
                    .unwrap();
            }
            assert_eq!(unchanged(&pool), before);
            assert_eq!(
                (facts.borrow().decisions, facts.borrow().submissions),
                (0, 0)
            );
            let actual = generation
                .map(|v| v.unwrap().token_id())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert_eq!(facts.borrow().commits, 3);
        } else {
            let generation = TextGeneration::new(&mut runtime, vec![1; 7], config(3)).unwrap();
            let context = facts.borrow().bound[0].clone();
            let before = unchanged(&pool);
            for _ in 0..4 {
                prepared.validate_initial_ready_context(&context).unwrap();
            }
            assert_eq!(unchanged(&pool), before);
            assert_eq!(
                (facts.borrow().decisions, facts.borrow().submissions),
                (0, 0)
            );
            let actual = generation.map(|v| v.unwrap().value).collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        assert_eq!(facts.borrow().submissions, 3);
        assert_eq!(facts.borrow().attempts[0].attempt(), 0);
        drop((runtime, prepared));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn unbound_and_each_partial_preparation_stage_reject_without_consumption() {
    let context = issued_contexts().remove(0);
    for prompt_first in [false, true] {
        let (prepared, pool, _) = new_preparation(3, true);
        let before = unchanged(&pool);
        assert_eq!(
            prepared.validate_initial_ready_context(&context),
            Err(WorkingMemoryError::TextRunUnbound)
        );
        prepared.bind_run(&context).unwrap();
        let not_ready = || {
            assert_eq!(
                prepared.validate_initial_ready_context(&context),
                Err(WorkingMemoryError::PreparationNotReady)
            );
            assert_eq!(unchanged(&pool), before);
        };
        not_ready();
        if prompt_first {
            let prompt = prepared.claim_prompt().unwrap();
            not_ready();
            prompt.finish().unwrap();
            not_ready(); // Constructed is not published/bound.
            prepared.bind_prompt().unwrap();
            not_ready();
            let sampling = prepared.claim_sampling(config(3)).unwrap();
            not_ready();
            sampling.finish().unwrap();
        } else {
            let sampling = prepared.claim_sampling(config(3)).unwrap();
            not_ready();
            sampling.finish().unwrap();
            not_ready();
            let prompt = prepared.claim_prompt().unwrap();
            not_ready();
            prompt.finish().unwrap();
            not_ready();
            prepared.bind_prompt().unwrap();
        }
        prepared.validate_initial_ready_context(&context).unwrap();
        prepared
            .claim_step(&context, PendingTextInput::Prefill(()))
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(unchanged(&pool), before);
    }
}

#[test]
fn foreign_run_changed_initial_policy_and_nonzero_context_are_distinct_rejections() {
    // controller_mut revises policy through the real core machine at attempt0.
    let (mut runtime, facts) = runtime(None);
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1; 7],
        config(3),
        Controller(facts.clone()),
    )
    .unwrap();
    let original = facts.borrow().bound[0].clone();
    let _ = generation.controller_mut();
    generation.next().unwrap().unwrap();
    generation.next().unwrap().unwrap();
    drop(generation);
    let revised = facts.borrow().attempts[0].clone();
    let next = facts.borrow().attempts[1].clone();
    assert_eq!(original.run_identity(), revised.run_identity());
    assert_ne!(original.policy_identity(), revised.policy_identity());
    assert_eq!(
        (original.attempt(), revised.attempt(), next.attempt()),
        (0, 0, 1)
    );
    let foreign = issued_contexts().remove(0);
    let (prepared, pool, _) = new_preparation(3, true);
    prepared.bind_run(&original).unwrap();
    ready(&prepared);
    let before = unchanged(&pool);
    for wrong in [&foreign, &revised] {
        assert_eq!(
            prepared.validate_initial_ready_context(wrong),
            Err(WorkingMemoryError::IdentityMismatch)
        );
    }
    prepared.validate_initial_ready_context(&original).unwrap();
    assert_eq!(unchanged(&pool), before);
    let (changed, _, _) = new_preparation(3, false);
    changed.bind_run(&revised).unwrap();
    ready(&changed);
    assert_eq!(
        changed.validate_initial_ready_context(&next),
        Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: 0,
            actual: 1
        })
    );
    changed.validate_initial_ready_context(&revised).unwrap();
}

#[test]
fn active_completed_and_abandoned_steps_never_become_initial_ready_again() {
    let context = issued_contexts().remove(0);
    for abandon in [false, true] {
        let (prepared, pool, _) = new_preparation(3, true);
        prepared.bind_run(&context).unwrap();
        ready(&prepared);
        prepared.validate_initial_ready_context(&context).unwrap();
        let first = prepared
            .claim_step(&context, PendingTextInput::Prefill(()))
            .unwrap();
        let before = unchanged(&pool);
        assert_eq!(
            prepared.validate_initial_ready_context(&context),
            Err(WorkingMemoryError::TextStepActive)
        );
        if abandon {
            drop(first);
        } else {
            first.finish().unwrap();
        }
        for _ in 0..3 {
            assert_eq!(
                prepared.validate_initial_ready_context(&context),
                Err(if abandon {
                    WorkingMemoryError::ExecutionFenced
                } else {
                    WorkingMemoryError::PreparationAlreadyStarted
                })
            );
        }
        assert_eq!(unchanged(&pool), before);
        drop(prepared);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn zero_allowance_and_separate_request_domain_checks_do_not_gain_authority() {
    // The application resolver rejects zero tokens. Exercise the existing
    // low-level preparation contract directly, without constructing a run.
    fn zero_config() -> TextGenerationConfig {
        let mut sampling = config(1).sampling();
        sampling.max_new_tokens = Some(0);
        TextGenerationConfig::new(sampling).with_seed(19)
    }
    fn zero_preparation() -> (
        InferenceTextPreparation,
        WorkingMemoryPool,
        InferenceExecutionIdentity,
    ) {
        let mut g = geometry(3, OutputDemand::LastPosition);
        g.max_output_tokens = 0;
        let execution = InferenceExecutionIdentity::default();
        let pool = WorkingMemoryPool::new(4096, 0).unwrap();
        let request = InferenceRequest::from(pool.reserve(&execution, &admission(g)).unwrap());
        let preparation = request.prepare_text(&execution, g, zero_config()).unwrap();
        (preparation, pool, execution)
    }
    let context = issued_contexts().remove(0);
    let (prepared, pool, execution) = zero_preparation();
    prepared.bind_run(&context).unwrap();
    prepared.claim_prompt().unwrap().finish().unwrap();
    prepared.bind_prompt().unwrap();
    prepared
        .claim_sampling(zero_config())
        .unwrap()
        .finish()
        .unwrap();
    let before = unchanged(&pool);
    prepared.validate_initial_ready_context(&context).unwrap();
    let request = prepared.request();
    request.validate(&execution, request.geometry()).unwrap();
    request
        .memory_reservation()
        .unwrap()
        .validate_domain(&pool)
        .unwrap();
    let (other, foreign_pool, foreign_execution) = zero_preparation();
    assert_eq!(
        request.validate_same_request(other.request()),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        request.validate(&foreign_execution, request.geometry()),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        request
            .memory_reservation()
            .unwrap()
            .validate_domain(&foreign_pool),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    let mut wrong_geometry = request.geometry();
    wrong_geometry.cached_positions += 1;
    assert_eq!(
        request.validate(&execution, wrong_geometry),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert!(matches!(
        prepared.claim_step(&context, PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 0,
            limit: 0
        })
    ));
    // Validation is not a permit; a rejected output claim did not change readiness.
    prepared.validate_initial_ready_context(&context).unwrap();
    assert_eq!(unchanged(&pool), before);
}

#[test]
fn legacy_prefill_start_is_not_mistaken_for_an_unissued_initial_core_step() {
    let context = issued_contexts().remove(0);
    let (prepared, pool, execution) = new_preparation(3, true);
    prepared.bind_run(&context).unwrap();
    ready(&prepared);
    prepared.validate_initial_ready_context(&context).unwrap();
    let before = unchanged(&pool);
    let driver = PrefillDriver::<Vec<f64>, NativeCompletion>::new(
        &execution,
        prepared.request(),
        prepared.request().geometry(),
        GenerationCancellationToken::new(),
    )
    .unwrap();
    for _ in 0..3 {
        assert_eq!(
            prepared.validate_initial_ready_context(&context),
            Err(WorkingMemoryError::AlreadyStarted)
        );
    }
    assert_eq!(unchanged(&pool), before);
    drop(driver);
    assert_eq!(
        prepared.validate_initial_ready_context(&context),
        Err(WorkingMemoryError::AlreadyStarted)
    );
}
