mod owned;
mod text_controls;

use super::*;
use crate::working_memory::InferenceRequest;

fn sealed_plan(
    pool: &WorkingMemoryPool,
    quote: &IncrementalInferenceQuote,
    capacity: u64,
) -> Result<
    (
        WorkingMemoryReservation,
        IncrementalInferenceQuote,
    ),
    PrefillPlanningError,
> {
    plan_prefill_incremental_with_capacity(
        &InferenceExecutionIdentity::default(),
        pool,
        &capabilities(),
        request(quote.geometry()),
        quote.geometry(),
        capacity,
        |_| Ok(quote.clone()),
    )
}
#[test]
fn opt_in_prices_host_once_preserves_full_source_charges_and_exact_admission_boundary() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let escaped = source.clone();
    let original = replacement_quote(&pool, geometry(), 11).into_incremental();
    let before = original.incremental_bytes();
    let host = original.span_workspace().retention_peak_bytes().unwrap();
    assert_eq!(
        original.span_workspace().source_preparation_bytes(),
        Some(11)
    );
    assert_eq!(
        (0..3)
            .map(|i| original.span_workspace().span_bytes(i))
            .collect::<Vec<_>>(),
        vec![Some(11), Some(27), Some(91)]
    );
    let quote = original.clone().with_span_workspace().unwrap();
    assert_eq!(quote.incremental_bytes(), before + host);
    assert_eq!(
        quote
            .state()
            .execution_workspace
            .as_ref()
            .unwrap()
            .retained
            .bytes(),
        Some(
            original
                .state()
                .execution_workspace
                .as_ref()
                .unwrap()
                .retained
                .bytes()
                .unwrap()
                + host
        )
    );
    assert_eq!(
        pool.used_bytes().unwrap(),
        64,
        "cold opt-in neither registers nor reserves"
    );
    assert!(matches!(
        quote.clone().with_span_workspace(),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let exact = 64 + quote.incremental_bytes();
    assert!(
        matches!(sealed_plan(&pool,&quote,exact-1),Err(PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes})) if required_bytes==quote.incremental_bytes() && available_bytes+1==required_bytes)
    );
    assert_eq!(pool.used_bytes().unwrap(), 64);
    let (reservation, accepted) = sealed_plan(&pool, &quote, exact).unwrap();
    assert_eq!(reservation.admission().state, *quote.state());
    assert_eq!(pool.used_bytes().unwrap(), exact);
    let association = accepted.reserved_span_workspace(&reservation).unwrap();
    assert!(association
        .workspace()
        .plan()
        .same_plan(original.span_workspace().plan()));
    assert_eq!(
        association.span_bytes(&InferenceWorkspaceSpan::Decode {
            index: 1,
            position: 6,
            output: OutputDemand::LastPosition
        }),
        Some(91)
    );
    assert!(original.reserved_span_workspace(&reservation).is_err());
    drop(association);
    drop((reservation, accepted, quote, original, source));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn equal_independent_seals_and_unsealed_reservations_cannot_rebind_candidate_identity() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let a = original.clone().with_span_workspace().unwrap();
    let b = original.clone().with_span_workspace().unwrap();
    assert_eq!(a.state(), b.state());
    let (reservation, accepted) = sealed_plan(&pool, &a, 1_000_000).unwrap();
    assert!(b.reserved_span_workspace(&reservation).is_err());
    assert!(a
        .clone()
        .reserved_span_workspace(&reservation.clone())
        .is_ok());
    let request = InferenceRequest::from(&reservation);
    accepted
        .reserved_span_workspace(&reservation)
        .unwrap()
        .validate_request(&request)
        .unwrap();
    let (other, _) = sealed_plan(&pool, &b, 1_000_000).unwrap();
    assert!(accepted
        .reserved_span_workspace(&reservation)
        .unwrap()
        .validate_request(&InferenceRequest::from(&other))
        .is_err());
    let (raw, _) = sealed_plan(&pool, &original, 1_000_000).unwrap();
    assert!(a.reserved_span_workspace(&raw).is_err());
    drop((
        request,
        reservation,
        other,
        raw,
        accepted,
        a,
        b,
        original,
        source,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn mechanism_terms_are_original_full_contributions_and_unknown_or_overflow_never_becomes_a_plan() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let context = WorkspaceContext::new(Facts::default());
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let registered =
        RegisteredWorkspaceStorage::bind(&pool, &context, [(1u32, root.clone())]).unwrap();
    let report = replacement_report(&context, &root, geometry());
    let mut full = outside(geometry(), 13);
    full.state_update = WorkspaceBound::bounded(17, "full original preparation");
    full.materialization = WorkspaceBound::bounded(19, "selected original materialization");
    full.attention = WorkspaceBound::bounded(23, "separate original attention");
    let quote =
        ResidualInferenceQuote::compose(&report, state(geometry()), full.clone(), &registered)
            .unwrap()
            .into_incremental();
    assert!(quote
        .span_workspace()
        .plan()
        .same_plan(report.span_workspace_plan()));
    assert_eq!(quote.span_workspace().source_preparation_bytes(), Some(30));
    assert_eq!(quote.span_workspace().materialization_bytes(), Some(19));
    assert_eq!(
        quote.span_workspace().span_bytes(2),
        Some(80 + 30 + 19 + 23)
    );
    full.materialization = WorkspaceBound::Unknown {
        reason: "selected materialization missing".into(),
    };
    assert!(matches!(
        ResidualInferenceQuote::compose(&report, state(geometry()), full.clone(), &registered),
        Err(ResidualQuoteError::IncompleteWorkspace(_))
    ));
    let diagnostic = InferenceSpanWorkspace::new(report.span_workspace_plan(), &full).unwrap();
    assert_eq!(diagnostic.span_bytes(0), None);
    full.activations = WorkspaceBound::bounded(u64::MAX, "overflow must remain visible");
    assert!(matches!(
        InferenceSpanWorkspace::new(report.span_workspace_plan(), &full),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    drop((quote, registered, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn reservation_keeps_only_identity_after_diagnostics_retire_and_funding_conversion_preserves_binding(
) {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0)
        .into_incremental()
        .with_span_workspace()
        .unwrap();
    let (reservation, accepted) = sealed_plan(&pool, &quote, 1_000_000).unwrap();
    let diagnostic = accepted.span_workspace().plan().clone();
    let owners = diagnostic.strong_owner_count();
    let (reservation, run) = reservation.into_funding().unwrap();
    assert!(accepted.reserved_span_workspace(&reservation).is_ok());
    drop((accepted, quote));
    assert_eq!(
        diagnostic.strong_owner_count(),
        owners - 2,
        "historical identity must not retain the record array"
    );
    assert_eq!(diagnostic.strong_owner_count(), 1);
    drop(diagnostic);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), 64);
    drop((reservation, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
