use super::*;
use crate::working_memory::{GraphMetadataFacts, TextPrefillScopeFacts};
fn graph() -> GraphMetadataFacts {
    GraphMetadataFacts::new(std::num::NonZeroU64::new(4096).unwrap(), 4096 + 59).unwrap()
}
fn roles(g: InferenceGeometry) -> TextPrefillScopeFacts {
    TextPrefillScopeFacts::new(g, [Some(37); 7], Some(41), 8, 64).unwrap()
}
#[test]
fn prefill_original_exact_and_short_admission_preserve_single_hold_and_residual_ceiling() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let plain = prepared(&source, &original)
        .with_graph_metadata(graph())
        .unwrap();
    // The actual accepted component includes request-wide operation storage;
    // this fixture checks custody/comparison, not a native slot implementation.
    let facts = roles(geometry()).with_operation_controls(Some(73)).unwrap();
    assert_eq!(facts.operation_control_bytes(), Some(73));
    assert_eq!(
        facts.total_bytes().unwrap(),
        roles(geometry()).total_bytes().unwrap().map(|n| n + 73)
    );
    assert_eq!(
        facts.graph_residual_ceiling(4096).unwrap(),
        4096 - facts.graph_bytes().unwrap()
    );
    assert!(matches!(
        facts.graph_residual_ceiling(facts.graph_bytes().unwrap() - 1),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let controls = plain.clone().with_prefill_scopes(facts).unwrap();
    assert!(!controls.same_binding(&plain));
    assert!(controls.same_binding(&controls.clone()));
    assert!(matches!(
        controls.clone().with_prefill_scopes(facts),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let q = controls.facts().total_bytes().unwrap().unwrap();
    let quote = original
        .with_span_workspace_and_text_controls(controls)
        .unwrap();
    let p = quote.span_workspace().retention_peak_bytes().unwrap();
    let exact = 64 + quote.incremental_bytes();
    assert!(
        matches!(sealed_plan(&pool,&quote,exact-1),Err(PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes})) if required_bytes==available_bytes+1)
    );
    assert_eq!(pool.used_bytes().unwrap(), 64);
    let (reservation, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted
        .into_funded_text_span_workspace(&run, &reservation)
        .unwrap();
    let bank = span.take_prefill_scopes().unwrap().unwrap();
    assert!(matches!(
        span.take_prefill_scopes(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let hold = span.protected_host_bytes();
    assert_eq!(hold, p + q);
    drop((span, reservation, run, root, source, plain, quote));
    assert_eq!(pool.used_bytes().unwrap(), hold);
    drop(bank);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn prefill_unknown_geometry_and_graph_population_never_produce_partial_authority() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let plain = prepared(&source, &original);
    assert!(matches!(
        plain.clone().with_prefill_scopes(roles(geometry())),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let plain = plain.with_graph_metadata(graph()).unwrap();
    let unknown_operations = plain
        .clone()
        .with_prefill_scopes(roles(geometry()).with_operation_controls(None).unwrap())
        .unwrap();
    assert!(matches!(
        original
            .clone()
            .with_span_workspace_and_text_controls(unknown_operations),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert!(matches!(
        roles(geometry()).with_operation_controls(Some(u64::MAX)),
        Err(WorkingMemoryError::Overflow)
    ));
    for missing in 0..7 {
        let mut controls = [Some(1); 7];
        controls[missing] = None;
        let facts = TextPrefillScopeFacts::new(geometry(), controls, Some(1), 8, 64).unwrap();
        let unknown = plain.clone().with_prefill_scopes(facts).unwrap();
        assert!(matches!(
            original
                .clone()
                .with_span_workspace_and_text_controls(unknown),
            Err(ResidualQuoteError::Storage(
                WorkingMemoryError::UnknownBound
            ))
        ));
    }
    let changed = InferenceGeometry {
        input_positions: geometry().input_positions + 1,
        ..geometry()
    };
    assert!(matches!(
        plain.clone().with_prefill_scopes(roles(changed)),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        TextPrefillScopeFacts::new(geometry(), [Some(u64::MAX); 7], Some(1), 1, 1),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop((root, source, plain));
    assert_eq!(
        pool.used_bytes().unwrap(),
        64,
        "the original quote pins its source"
    );
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
