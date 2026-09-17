use super::*;
use crate::working_memory::HostSourceConstructionFacts;

fn source_request(
    pool: &WorkingMemoryPool,
    bytes: u64,
    attempts: usize,
    partitions: usize,
) -> Option<IncrementalInferenceQuote> {
    source_request_with_facts(pool, HostSourceConstructionFacts::new(bytes, attempts, partitions).unwrap())
}

fn source_request_with_facts(
    pool: &WorkingMemoryPool,
    source: HostSourceConstructionFacts,
) -> Option<IncrementalInferenceQuote> {
    let facts = match HostDestinationFacts::new(4, 1) {
        Ok(facts) => facts.with_source_constructions(source).unwrap(),
        Err(WorkingMemoryError::UnknownBound) => return None,
        Err(cause) => panic!("host facts: {cause}"),
    };
    assert!(matches!(
        facts.with_source_constructions(source),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let quote = replacement_quote(pool, geometry(), 0).into_incremental();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        quote.span_workspace().plan(),
        super::facts(),
    )
    .unwrap()
    .with_host_destinations(facts)
    .unwrap();
    Some(
        quote
            .with_span_workspace_and_text_controls(controls)
            .unwrap(),
    )
}

#[test]
fn source_component_is_reserved_before_debit_and_retains_the_actual_accepted_hold() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = source_request(&pool, 24, 2, 0) else {
        return;
    };
    let total = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, total - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (_, r, accepted) = sealed_plan(&pool, &quote, total).unwrap();
    let (r, run) = r.into_funding().unwrap();
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut source = host.take_source_constructions().unwrap();
    assert!(host.take_source_constructions().is_none());
    assert!(matches!(
        span.take_host_destinations(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    let receipt = source.try_debit(24).unwrap();
    assert!(receipt.belongs_to(&span.control_guard()));
    assert_eq!(
        (source.remaining_bytes(), source.remaining_attempts()),
        (0, 1)
    );
    // Source debit cannot consume the separately accepted vector component.
    let mut values = host.try_vec::<u32>(1).unwrap();
    values.try_fill(1, [0x12345678]).unwrap();
    assert_eq!(values.as_slice(), &[0x12345678]);
    drop((quote, values, span, source, host, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(receipt);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_split_conserves_bytes_attempts_and_cannot_refill_after_retirement() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = source_request(&pool, 24, 3, 2) else {
        return;
    };
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut source = host.take_source_constructions().unwrap();
    assert!(matches!(
        source.split(25, 1),
        Err(HostDestinationCause::Capacity { .. })
    ));
    assert_eq!(
        (
            source.remaining_bytes(),
            source.remaining_attempts(),
            source.remaining_partitions()
        ),
        (24, 3, 2)
    );
    let mut a = source.split(8, 1).unwrap();
    let mut b = source.split(16, 2).unwrap();
    assert_eq!(
        (
            source.remaining_bytes(),
            source.remaining_attempts(),
            source.remaining_partitions()
        ),
        (0, 0, 0)
    );
    assert!(matches!(a.split(0, 0), Err(HostDestinationCause::Attempts)));
    assert!(matches!(
        source.split(0, 0),
        Err(HostDestinationCause::Attempts)
    ));
    let a_receipt = a.try_debit(8).unwrap();
    drop((a, a_receipt));
    let b_receipt = b.try_debit(16).unwrap();
    let error = b.try_debit(1).unwrap_err();
    assert!(matches!(
        error.cause(),
        HostDestinationCause::Capacity {
            required: 1,
            remaining: 0
        }
    ));
    assert!(error.retains_receipt());
    let exhausted = b.try_debit(0).unwrap_err();
    assert!(!exhausted.retains_receipt());
    drop((
        b_receipt, error, exhausted, b, source, host, span, r, run, root,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_one_short_refuses_before_constructor_and_owning_failures_are_finite() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = source_request(&pool, 7, 1, 0) else {
        return;
    };
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut source = host.take_source_constructions().unwrap();
    let mut constructor_calls = 0;
    let error = source
        .try_debit(8)
        .map(|receipt| {
            constructor_calls += 1;
            receipt
        })
        .unwrap_err();
    assert_eq!(constructor_calls, 0);
    assert!(matches!(
        error.cause(),
        HostDestinationCause::Capacity {
            required: 8,
            remaining: 7
        }
    ));
    assert_eq!(
        (source.remaining_bytes(), source.remaining_attempts()),
        (7, 0)
    );
    let exhausted: Vec<_> = (0..16).map(|_| source.try_debit(0).unwrap_err()).collect();
    assert!(exhausted.iter().all(|e| !e.retains_receipt()));
    drop((source, host, span, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(exhausted);
}

#[test]
fn source_checks_real_account_health_before_constructor_and_preserves_attempt_custody() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some(quote) = source_request(&pool, 12, 1, 1) else {
        return;
    };
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut source = host.take_source_constructions().unwrap();
    drop(run);
    assert!(matches!(
        source.split(12, 1),
        Err(HostDestinationCause::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(
        (
            source.remaining_bytes(),
            source.remaining_attempts(),
            source.remaining_partitions()
        ),
        (12, 1, 1)
    );
    let error = source.try_debit(12).unwrap_err();
    assert!(matches!(
        error.cause(),
        HostDestinationCause::Memory(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(error.retains_receipt());
    assert_eq!(
        (source.remaining_bytes(), source.remaining_attempts()),
        (12, 0)
    );
    drop((source, host, span, r, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_zero_attempts_issue_no_owner_and_equal_foreign_banks_do_not_match() {
    let a = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let b = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let ar = a.register_storage([(1u32, 64)]).unwrap();
    let br = b.register_storage([(1u32, 64)]).unwrap();
    let Some(aq) = source_request(&a, 0, 0, 0) else {
        return;
    };
    let bq = source_request(&b, 0, 0, 0).unwrap();
    let (ra, run_a, aq) = accept(&a, aq);
    let (rb, run_b, bq) = accept(&b, bq);
    let (mut a_span, _) = aq.into_funded_text_span_workspace(&run_a, &ra).unwrap();
    let (b_span, _) = bq.into_funded_text_span_workspace(&run_b, &rb).unwrap();
    let mut host = a_span.take_host_destinations().unwrap().unwrap();
    let mut source = host.take_source_constructions().unwrap();
    assert!(source.belongs_to(&a_span.control_guard()));
    assert!(!source.belongs_to(&b_span.control_guard()));
    let error = source.try_debit(0).unwrap_err();
    assert!(!error.retains_receipt());
    drop((source, host, a_span, b_span, ra, rb, run_a, run_b, ar, br));
    assert_eq!((a.used_bytes().unwrap(), b.used_bytes().unwrap()), (0, 0));
    drop(error);
}

#[test]
fn prefill_source_join_extends_same_host_binding_once_without_accepting_unknown_operations() {
    use crate::working_memory::{GraphMetadataFacts, TextPrefillScopeFacts};
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let original = replacement_quote(&pool, geometry(), 0).into_incremental();
    let host = match HostDestinationFacts::new(4, 1) {
        Ok(facts) => facts,
        Err(WorkingMemoryError::UnknownBound) => return,
        Err(cause) => panic!("host facts: {cause}"),
    };
    let source = HostSourceConstructionFacts::new(32, 1, 0).unwrap();
    let base = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        original.span_workspace().plan(),
        super::facts(),
    )
    .unwrap()
    .with_host_destinations(host)
    .unwrap()
    .with_graph_metadata(
        GraphMetadataFacts::new(std::num::NonZeroU64::new(4096).unwrap(), 4096 + 59).unwrap(),
    )
    .unwrap();
    let roles = TextPrefillScopeFacts::new(geometry(), [Some(37); 7], Some(41), 8, 64).unwrap();
    let plain = base.clone().with_prefill_scopes(roles).unwrap();
    let combined = base
        .clone()
        .with_prefill_scopes(roles.with_source_constructions(Some(source)))
        .unwrap();
    assert_eq!(
        combined.facts().total_bytes().unwrap(),
        plain
            .facts()
            .total_bytes()
            .unwrap()
            .map(|n| n + source.protected_bytes())
    );
    let unknown = base
        .clone()
        .with_prefill_scopes(
            roles
                .with_operation_controls(None)
                .unwrap()
                .with_source_constructions(Some(source)),
        )
        .unwrap();
    assert!(matches!(
        original
            .clone()
            .with_span_workspace_and_text_controls(unknown),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    let quote = original
        .with_span_workspace_and_text_controls(combined)
        .unwrap();
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    let roles = span.take_prefill_scopes().unwrap().unwrap();
    assert_eq!(
        roles.facts().unwrap().source_construction_facts(),
        Some(source)
    );
    let mut host = span.take_host_destinations().unwrap().unwrap();
    assert_eq!((host.remaining_bytes(), host.remaining_attempts()), (4, 1));
    let mut source = host.take_source_constructions().unwrap();
    assert!(host.take_source_constructions().is_none());
    let receipt = source.try_debit(32).unwrap();
    drop((source, host, roles, span, r, run, root, base, plain));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(receipt);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod publication;

#[test]
fn metadata_retention_preserves_admitted_backing_and_only_the_original_host_hold() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1_u32, 64)]).unwrap();
    let Some(quote) = source_request(&pool, 24, 1, 0) else {
        assert!(matches!(
            crate::working_memory::OriginalHostMetadataCustody::boxed_storage_bytes(
                std::alloc::Layout::new::<u32>()
            ),
            Err(WorkingMemoryError::UnknownBound)
        ));
        return;
    };
    let total = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, total - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (_, reservation, accepted) = sealed_plan(&pool, &quote, total).unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let (mut span, _) = accepted
        .into_funded_text_span_workspace(&run, &reservation)
        .unwrap();
    let protected = span.protected_host_bytes();
    let mut host = span.take_host_destinations().unwrap().unwrap();
    let mut source = host.take_source_constructions().unwrap();
    let raw = source.try_debit(24).unwrap().into_metadata();
    let alias = raw.clone();
    let mut values = host.try_vec::<u32>(1).unwrap();
    values.try_fill(1, [0x1234abcd]).unwrap();
    let pointer = values.as_slice().as_ptr();
    let capacity = values.capacity();
    let mut metadata = values.into_metadata();
    assert_eq!(metadata.as_slice().as_ptr(), pointer);
    assert_eq!(metadata.capacity(), capacity);
    assert_eq!(metadata.as_slice(), &[0x1234abcd]);
    metadata.as_mut_slice()[0] = 0x9876fedc;
    assert_eq!(metadata.as_slice(), &[0x9876fedc]);
    assert_eq!(
        (source.remaining_bytes(), source.remaining_attempts()),
        (0, 0)
    );
    drop((quote, span, source, host, reservation, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop((raw, alias));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(metadata);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod program;
