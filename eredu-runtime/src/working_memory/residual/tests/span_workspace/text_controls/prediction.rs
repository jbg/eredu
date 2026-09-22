use super::*;
use crate::working_memory::TextPredictionScopeFacts;

fn role_facts() -> TextPredictionScopeFacts {
    TextPredictionScopeFacts::new(Some(101), Some(103), Some(107), Some(109), Some(113))
}

#[test]
fn prediction_bank_is_original_exact_admission_and_cannot_spend_protected_headroom() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), publication_controls()).into_incremental();
    let plain = prepared(&source, &original);
    let controls = plain.clone().with_prediction_scopes(role_facts()).unwrap();
    assert!(!controls.same_binding(&plain));
    assert!(controls.same_binding(&controls.clone()));
    assert!(matches!(
        controls.clone().with_prediction_scopes(role_facts()),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let q = controls.facts().total_bytes().unwrap().unwrap();
    assert!(q > 51 + geometry().max_output_tokens * (101 + 103 + 107 + 109 + 113));
    let before = original.incremental_bytes().unwrap();
    let quote = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let p = quote.span_workspace().retention_peak_bytes().unwrap();
    assert_eq!(quote.incremental_bytes().unwrap(), before + p + q);
    let exact = exact_capacity(&pool, &quote);
    assert!(matches!(sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(capacity_error)) if matches!(capacity_numbers(&capacity_error), Some((required_bytes, available_bytes)) if required_bytes == quote_reservation_bytes(&quote) && available_bytes + 1 == required_bytes)));
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let (r, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    drop(quote);
    let (r, run) = r.into_funding().unwrap();
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let held = span.protected_host_bytes();
    assert_eq!(held, p + q);
    let bank = span.take_prediction_scopes().unwrap().unwrap();
    assert!(matches!(
        span.take_prediction_scopes(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let scope = run.scope().unwrap();
    let native = reservation_payload_bytes(&r) - held;
    assert!(
        matches!(scope.adopt_host_storage_individually([(33u32, native + 1)]),
        Err(capacity_error) if matches!(capacity_numbers(&capacity_error), Some((required_bytes, available_bytes)) if required_bytes > available_bytes && available_bytes <= native))
    );
    let payload = scope
        .adopt_host_storage_individually([(33u32, native - publication_controls())])
        .unwrap();
    drop(payload);
    scope.certify().unwrap();
    drop((span, r, run, root, plain, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), held);
    drop(bank);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn unknown_prediction_role_and_overflow_reject_without_a_partial_original_seal() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = replacement_quote(&pool, geometry(), publication_controls()).into_incremental();
    let plain = prepared(&source, &q);
    for absent in 0..5 {
        let mut values = [Some(1); 5];
        values[absent] = None;
        let facts =
            TextPredictionScopeFacts::new(values[0], values[1], values[2], values[3], values[4]);
        let unknown = plain.clone().with_prediction_scopes(facts).unwrap();
        assert!(matches!(
            q.clone().with_span_workspace_and_text_controls(unknown),
            Err(ResidualQuoteError::Storage(
                WorkingMemoryError::UnknownBound
            ))
        ));
    }
    assert!(matches!(
        plain
            .clone()
            .with_prediction_scopes(TextPredictionScopeFacts::new(
                None,
                Some(u64::MAX),
                Some(1),
                Some(0),
                Some(0)
            )),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(
        TextPredictionScopeFacts::new(None, Some(0), Some(0), Some(0), Some(0))
            .total_bytes()
            .unwrap(),
        None
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    let q = q.with_span_workspace_and_text_controls(plain).unwrap();
    let (r, run, q) = accept(&pool, q);
    let (mut span, _) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    assert!(span.take_prediction_scopes().unwrap().is_none());
    assert!(span.take_prediction_scopes().unwrap().is_none());
    drop((span, r, run, root));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
