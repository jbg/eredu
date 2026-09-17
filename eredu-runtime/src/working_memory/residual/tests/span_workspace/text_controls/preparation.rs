use super::*;
use crate::working_memory::{InferenceRequest, TextPreparationScopeFacts};

#[test]
fn original_scope_pair_is_sealed_once_and_excluded_from_native_headroom() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let plain = prepared(&source, &original);
    let controls = plain
        .clone()
        .with_preparation_scopes(TextPreparationScopeFacts::new(Some(113), Some(197)))
        .unwrap();
    let delta = controls.facts().total_bytes().unwrap().unwrap()
        - plain.facts().total_bytes().unwrap().unwrap();
    assert!(
        delta > 113 + 197,
        "the actual neutral bank and claim controls are also original Q"
    );
    assert!(!controls.same_binding(&plain));
    assert!(controls.same_binding(&controls.clone()));
    assert!(matches!(
        controls
            .clone()
            .with_preparation_scopes(TextPreparationScopeFacts::new(Some(0), Some(0))),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    let before = original.incremental_bytes();
    let quote = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let p = quote.span_workspace().retention_peak_bytes().unwrap();
    assert_eq!(quote.incremental_bytes(), before + p + 51 + delta);
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (_, r, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    drop(quote);
    let (r, run) = r.into_funding().unwrap();
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    assert_eq!(protected, p + 51 + delta);
    let bank = span.take_preparation_scopes().unwrap().unwrap();
    assert!(matches!(
        span.take_preparation_scopes(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    let native = run.scope().unwrap();
    let free = r.bytes() - protected;
    assert!(
        matches!(native.adopt_storage_individually([(39u32, free+1)]), Err(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if required_bytes == free+1 && available_bytes == free)
    );
    let payload = native.adopt_storage_individually([(39u32, free)]).unwrap();
    drop(payload);
    native.certify().unwrap();
    drop((span, r, run, root, plain, source));
    assert_eq!(
        pool.used_bytes().unwrap(),
        protected,
        "the extracted bank retains the original aggregate"
    );
    drop(bank);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn scope_bank_rejects_unbound_or_foreign_preparation_without_spending_roles() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let controls = prepared(&source, &original)
        .with_preparation_scopes(TextPreparationScopeFacts::new(Some(113), Some(197)))
        .unwrap();
    let quote = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let (r, run, quote) = accept(&pool, quote);
    let (mut span, _) = quote.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span.take_preparation_scopes().unwrap().unwrap();
    let execution = r.0.execution.clone();
    let config = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(geometry().max_output_tokens as usize),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let p = InferenceRequest::from(&r)
        .prepare_text(&execution, geometry(), config)
        .unwrap();
    assert!(matches!(
        bank.claim_prompt(&p),
        Err(WorkingMemoryError::TextRunUnbound)
    ));
    assert!(matches!(
        bank.claim_sampling(&p, config),
        Err(WorkingMemoryError::TextRunUnbound)
    ));
    let other = super::quote(&pool, &source);
    let (r2, run2, other) = accept(&pool, other);
    let foreign = InferenceRequest::from(&r2)
        .prepare_text(&r2.0.execution, geometry(), config)
        .unwrap();
    assert!(matches!(
        bank.claim_prompt(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        bank.claim_prompt(&p),
        Err(WorkingMemoryError::TextRunUnbound)
    ));
    drop((bank, span, p, foreign, r, run, r2, run2, other, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn absent_scope_pair_stays_absent_and_unknown_facts_cannot_hide_overflow() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let controls = prepared(&source, &original);
    let unknown = controls
        .clone()
        .with_preparation_scopes(TextPreparationScopeFacts::new(None, Some(1)))
        .unwrap();
    assert!(matches!(
        original
            .clone()
            .with_span_workspace_and_text_controls(unknown),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert!(matches!(
        controls
            .clone()
            .with_preparation_scopes(TextPreparationScopeFacts::new(None, Some(u64::MAX))),
        Err(WorkingMemoryError::Overflow)
    ));
    let q = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let (r, run, q) = accept(&pool, q);
    let (mut owner, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    assert!(owner.take_preparation_scopes().unwrap().is_none());
    assert!(owner.take_preparation_scopes().unwrap().is_none());
    drop((owner, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_control_guard_checks_unexposed_scope_and_same_scope_after_activation() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let controls = prepared(&source, &original)
        .with_preparation_scopes(TextPreparationScopeFacts::new(Some(113), Some(197)))
        .unwrap();
    let quote = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let (r, run, quote) = accept(&pool, quote);
    let (span, _) = quote.into_funded_text_span_workspace(&run, &r).unwrap();
    let guard = span.control_guard();
    let before = pool.used_bytes().unwrap();
    let prepared = run.prepare_scope().unwrap();
    guard.validate_prepared_scope(&prepared).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), before);
    let foreign = replacement_quote(&pool, geometry(), 0).into_incremental();
    let (other_r, other_run, other_quote) = accept(&pool, foreign);
    let other = other_run.prepare_scope().unwrap();
    let before = pool.used_bytes().unwrap();
    assert_eq!(
        guard.validate_prepared_scope(&other),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop((other, other_quote, other_run, other_r));
    let native = prepared.activate();
    guard.validate_native_scope(&native).unwrap();
    let free = r.bytes() - span.protected_host_bytes();
    assert!(
        matches!(native.adopt_storage_individually([(71u32, free + 1)]),
        Err(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
            if required_bytes == free + 1 && available_bytes == free)
    );
    let allocation = native.adopt_storage_individually([(71u32, free)]).unwrap();
    drop(allocation);
    native.certify().unwrap();
    drop((guard, span, r, run, root, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
