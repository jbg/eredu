use super::*;
use crate::working_memory::HostSourceConstructionFacts;

fn paired(pool: &WorkingMemoryPool) -> Option<(IncrementalInferenceQuote, HostDestinationFacts)> {
    let host = match HostDestinationFacts::new(12, 3) {
        Ok(facts) => facts
            .with_partitions(2)
            .unwrap()
            .with_source_constructions(HostSourceConstructionFacts::new(24, 3, 2).unwrap())
            .unwrap(),
        Err(WorkingMemoryError::UnknownBound) => return None,
        Err(cause) => panic!("host facts: {cause}"),
    };
    let quote = replacement_quote(pool, geometry(), 0).into_incremental();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        quote.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_host_destinations(host)
    .unwrap();
    Some((
        quote
            .with_span_workspace_and_text_controls(controls)
            .unwrap(),
        host,
    ))
}

#[test]
fn paired_split_checks_both_populations_before_mutation_and_preserves_real_storage() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((quote, facts)) = paired(&pool) else {
        return;
    };
    let exact = 64 + quote.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &quote, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let protected = span.protected_host_bytes();
    let mut bank = span.take_host_destinations().unwrap().unwrap();
    for (bytes, attempts, source) in [
        (13, 1, Some((8, 1))),
        (8, 4, Some((8, 1))),
        (8, 1, Some((25, 1))),
        (8, 1, Some((8, 4))),
        (8, 1, None),
    ] {
        assert!(bank.split(bytes, attempts, source).is_err());
        assert!(
            bank.matches_facts(facts),
            "neither component mutated on refusal"
        );
    }
    let mut a = bank.split(8, 1, Some((8, 1))).unwrap();
    let mut b = bank.split(4, 2, Some((16, 2))).unwrap();
    assert_eq!(
        (
            bank.remaining_bytes(),
            bank.remaining_attempts(),
            bank.remaining_partitions()
        ),
        (0, 0, 0)
    );
    assert!(matches!(
        a.split(0, 0, Some((0, 0))),
        Err(HostDestinationCause::Attempts)
    ));
    let mut values = a.try_vec::<u32>(2).unwrap();
    values.try_fill(2, [0x12345678, 0x87654321]).unwrap();
    assert_eq!(values.capacity(), 2);
    let mut a_source = a.take_source_constructions().unwrap();
    let receipt = a_source.try_debit(8).unwrap();
    drop((a, a_source, receipt));
    let mut b_source = b.take_source_constructions().unwrap();
    let source = b_source.try_debit(16).unwrap();
    let mut tail = b.try_vec::<u32>(1).unwrap();
    tail.try_fill(1, [41]).unwrap();
    let error = b.try_vec::<u8>(1).unwrap_err();
    assert!(matches!(
        error.cause(),
        HostDestinationCause::Capacity {
            required: 1,
            remaining: 0
        }
    ));
    drop((tail, error, source, b_source, b, bank, span, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    assert_eq!(values.as_slice(), &[0x12345678, 0x87654321]);
    drop(values);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn paired_split_health_and_foreign_owner_checks_do_not_reissue_or_refund() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((quote, facts)) = paired(&pool) else {
        return;
    };
    let (r, run, accepted) = accept(&pool, quote);
    let (mut span, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let mut bank = span.take_host_destinations().unwrap().unwrap();
    let other = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let other_root = other.register_storage([(1u32, 64)]).unwrap();
    let (quote, _) = paired(&other).unwrap();
    let (or, of, oa) = accept(&other, quote);
    let (other_span, _) = oa.into_funded_text_span_workspace(&of, &or).unwrap();
    assert!(!bank.belongs_to(&other_span.control_guard()));
    assert!(bank.matches_facts(facts));
    drop((other_span, or, of, other_root));
    assert_eq!(other.used_bytes().unwrap(), 0);
    drop(run);
    assert!(matches!(
        bank.split(8, 1, Some((8, 1))),
        Err(HostDestinationCause::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert!(bank.matches_facts(facts));
    let sources = bank.take_source_constructions().unwrap();
    assert!(!bank.matches_facts(facts));
    assert!(matches!(
        span.take_host_destinations(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    drop((sources, bank, span, r, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn paired_prefill_binding_charges_whole_host_once_and_keeps_unknown_operations() {
    use crate::working_memory::{GraphMetadataFacts, TextPrefillScopeFacts};
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let Some((_, host)) = paired(&pool) else {
        return;
    };
    let quote = replacement_quote(&pool, geometry(), 0).into_incremental();
    let base = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        quote.span_workspace().plan(),
        facts(),
    )
    .unwrap()
    .with_graph_metadata(
        GraphMetadataFacts::new(std::num::NonZeroU64::new(4096).unwrap(), 4096 + 59).unwrap(),
    )
    .unwrap();
    let roles = TextPrefillScopeFacts::new(geometry(), [Some(37); 7], Some(41), 8, 64).unwrap();
    let plain = base.clone().with_prefill_scopes(roles).unwrap();
    let combined = base
        .clone()
        .with_prefill_scopes(
            roles
                .with_source_constructions(host.source_constructions())
                .with_host_destinations(Some(host)),
        )
        .unwrap();
    assert_eq!(
        combined.facts().total_bytes().unwrap(),
        plain
            .facts()
            .total_bytes()
            .unwrap()
            .map(|n| n + host.protected_bytes())
    );
    assert!(matches!(
        base.clone()
            .with_prefill_scopes(roles.with_host_destinations(Some(host))),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let unknown = base
        .with_prefill_scopes(
            roles
                .with_operation_controls(None)
                .unwrap()
                .with_source_constructions(host.source_constructions())
                .with_host_destinations(Some(host)),
        )
        .unwrap();
    assert!(matches!(
        quote.with_span_workspace_and_text_controls(unknown),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    drop((combined, plain, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
