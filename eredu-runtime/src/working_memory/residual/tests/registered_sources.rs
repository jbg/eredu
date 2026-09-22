mod witness;

use super::*;
use crate::working_memory::{WorkingMemoryCapacityHandoff, WorkingMemoryFundingRun};

fn reserve(
    pool: &MemoryLedger,
    quote: &IncrementalInferenceQuote,
    capacity: u64,
    execution: &InferenceExecutionIdentity,
    handoffs: &[WorkingMemoryCapacityHandoff],
) -> Result<(WorkingMemoryReservation, IncrementalInferenceQuote), PrefillPlanningError> {
    let capacity = payload_capacity_with_quote(pool, quote, &request(quote.geometry()), capacity);
    plan_prefill_incremental_with_capacity_handoff(
        execution,
        pool,
        &capabilities(),
        request(quote.geometry()),
        quote.geometry(),
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        handoffs,
        |_| Ok(quote.clone()),
    )
}

#[test]
fn additional_sources_keep_full_quote_and_exact_capacity_without_duplicate_credit_or_charge() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let additional = pool.register_host_storage([(2u32, 24), (3, 0)]).unwrap();
    let different_key_type = pool.register_host_storage([(2u64, 8)]).unwrap();
    let ordinary = replacement_quote(&pool, geometry(), 0).into_incremental();
    let joined = ordinary
        .clone()
        .with_registered_sources(root.clone())
        .unwrap()
        .with_registered_sources(additional.clone())
        .unwrap()
        .with_registered_sources(additional.clone())
        .unwrap()
        .with_registered_sources(different_key_type.clone())
        .unwrap();
    assert_eq!(joined.state(), ordinary.state());
    assert_eq!(joined.geometry(), ordinary.geometry());
    assert_eq!(joined.controller_contract(), ordinary.controller_contract());
    assert_eq!(joined.incremental_bytes().unwrap(), 96);
    assert_eq!(pool.payload_used_bytes().unwrap(), 96);
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        reserve(&pool, &joined, 191, &execution, &[]),
        Err(PrefillPlanningError::Reservation(
            capacity_error
        )) if matches!(capacity_numbers(&capacity_error), Some((required, available)) if required == available + 1)));
    assert_eq!(pool.payload_used_bytes().unwrap(), 96);
    let pin_metadata = crate::working_memory::StoragePublicationLayout::<u32>::new(1)
        .unwrap()
        .fund(&pool)
        .unwrap();
    let (reservation, accepted) = reserve(&pool, &joined, 192, &execution, &[]).unwrap();
    assert_eq!(reservation.admission().state, *ordinary.state());
    assert_eq!(reservation_payload_bytes(&reservation), 96);
    drop((
        ordinary,
        joined,
        accepted,
        root,
        additional,
        different_key_type,
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 192);
    assert_eq!(
        pin_metadata
            .pin_registered_storage([(3u32, 0)])
            .unwrap()
            .bytes(),
        Some(0)
    );
    drop(reservation);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(3u32, 0)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}

#[test]
fn foreign_zero_and_empty_registrations_reject_without_mutation() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let other = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let _root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let ordinary = replacement_quote(&pool, geometry(), 0).into_incremental();
    for source in [
        other.register_host_storage([(2u32, 0)]).unwrap(),
        other.register_storage::<u32>([]).unwrap(),
    ] {
        assert!(matches!(
            ordinary.clone().with_registered_sources(source),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), 64);
        assert_eq!(other.payload_used_bytes().unwrap(), 0);
    }
    let empty = pool.register_storage::<u32>([]).unwrap();
    let joined = ordinary.with_registered_sources(empty).unwrap();
    let (reservation, _) = reserve(
        &pool,
        &joined,
        160,
        &InferenceExecutionIdentity::default(),
        &[],
    )
    .unwrap();
    assert_eq!(reservation_payload_bytes(&reservation), 96);
}

fn funded(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    bytes: u64,
    capacity: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    let g = geometry();
    let bytes = bytes.checked_add(publication_controls()).unwrap();
    pool.reserve_with_capacity(
        execution,
        &Admission {
            requested_positions: g.cached_positions + g.input_positions + g.max_output_tokens,
            state: state(g)
                .with_execution_workspace(outside(g, bytes))
                .unwrap(),
            incremental_required_bytes: Some(bytes),
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
        },
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
    )
    .unwrap()
    .into_funding()
    .unwrap()
}

#[test]
fn every_source_origin_is_rechecked_before_reservation_and_eligible_handoff_commit() {
    for source_bytes in [0, 32] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let execution = InferenceExecutionIdentity::default();
        let (source_metadata, source_run) = funded(&pool, &execution, source_bytes, 1_000_000);
        let source_scope = source_run.scope().unwrap();
        let source = source_scope
            .adopt_host_storage_individually([(2u32, source_bytes)])
            .unwrap()
            .remove(&2)
            .unwrap();
        let (old_metadata, mut old_run) = funded(&pool, &execution, 64, 100_000);
        let handoff = old_run.take_capacity_handoff().unwrap();
        let old_scope = old_run.scope().unwrap();
        let old_root = old_scope
            .adopt_host_storage_individually([(1u32, 64)])
            .unwrap();
        old_scope.certify().unwrap();
        old_run.close().unwrap();
        let quote = replacement_quote(&pool, geometry(), 0)
            .into_incremental()
            .with_registered_sources(source.clone())
            .unwrap();
        drop((source_scope, source_run));
        let before = (used(&pool), pool.payload_effective_capacity().unwrap());
        assert_eq!(before.0.0, 64 + source_bytes);
        assert_eq!(before.1, 100_000);
        assert!(matches!(
            reserve(
                &pool,
                &quote,
                200_000,
                &execution,
                std::slice::from_ref(&handoff)
            ),
            Err(PrefillPlanningError::Reservation(
                WorkingMemoryError::ExecutionFenced
            ))
        ));
        assert_eq!(
            (used(&pool), pool.payload_effective_capacity().unwrap()),
            before
        );
        assert!(!handoff.is_retired().unwrap());
        // Adding a healthy input also rechecks earlier attached origins.
        let healthy = pool.pin_registered_storage([(1u32, 64)]).unwrap();
        let before = (used(&pool), pool.payload_effective_capacity().unwrap());
        assert!(matches!(
            quote.clone().with_registered_sources(healthy),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(
            (used(&pool), pool.payload_effective_capacity().unwrap()),
            before
        );
        drop((
            quote,
            source,
            source_metadata,
            old_root,
            old_metadata,
            handoff,
        ));
    }
}

#[test]
fn additional_source_pins_flow_through_cloned_reservation_and_all_original_scopes() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source = pool.register_host_storage([(2u32, 24), (3, 0)]).unwrap();
    let quote = replacement_quote(&pool, geometry(), 0)
        .into_incremental()
        .with_registered_sources(source.clone())
        .unwrap();
    let pin_metadata = crate::working_memory::StoragePublicationLayout::<u32>::new(2)
        .unwrap()
        .fund(&pool)
        .unwrap();
    let (reservation, accepted) = reserve(
        &pool,
        &quote,
        184,
        &InferenceExecutionIdentity::default(),
        &[],
    )
    .unwrap();
    let alias = reservation.clone();
    drop((reservation, quote, accepted, root, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 184);
    let (metadata, run) = alias.into_funding().unwrap();
    let first = run.scope().unwrap();
    let last = run.scope().unwrap();
    run.close().unwrap();
    first.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 184);
    assert_eq!(
        pin_metadata
            .pin_registered_storage([(2u32, 24), (3, 0)])
            .unwrap()
            .bytes(),
        Some(24)
    );
    last.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(3u32, 0)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(metadata.requires_funding_scope());
    drop(metadata);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn abandoned_scope_preserves_all_additional_sources_and_original_envelope() {
    for abandon_first in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let root = pool.register_host_storage([(1u32, 64)]).unwrap();
        let source = pool.register_host_storage([(2u32, 24), (3, 0)]).unwrap();
        let quote = replacement_quote(&pool, geometry(), 0)
            .into_incremental()
            .with_registered_sources(source.clone())
            .unwrap();
        let pin_metadata = crate::working_memory::StoragePublicationLayout::<u32>::new(3)
            .unwrap()
            .fund(&pool)
            .unwrap();
        let (reservation, accepted) = reserve(
            &pool,
            &quote,
            184,
            &InferenceExecutionIdentity::default(),
            &[],
        )
        .unwrap();
        let (metadata, run) = reservation.into_funding().unwrap();
        let good = run.scope().unwrap();
        let abandoned = run.scope().unwrap();
        drop((metadata, run, quote, accepted, root, source));
        if abandon_first {
            drop(abandoned);
            good.certify().unwrap();
        } else {
            good.certify().unwrap();
            drop(abandoned);
        }
        assert_eq!(pool.payload_used_bytes().unwrap(), 184);
        assert_eq!(
            pin_metadata
                .pin_registered_storage([(1u32, 64), (2, 24), (3, 0)])
                .unwrap()
                .bytes(),
            Some(88)
        );
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}
