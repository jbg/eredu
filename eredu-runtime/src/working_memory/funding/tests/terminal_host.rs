//! Actual original accounts/closed holds with the parent's stateless workspace.
use super::*;
use crate::working_memory::HostSlotStorageKey;
use crate::{HostMetadataKey, HostSlotTable};
fn counts(pool: &WorkingMemoryPool, r: &WorkingMemoryReservation) -> (usize, usize, u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    let a = &usage.funding[&r.0.funding.unwrap()];
    (a.scopes, a.native_scopes, a.remaining, a.host_held)
}
fn empty(pool: &WorkingMemoryPool) -> WorkingMemoryStorage<u32> {
    pool.pin_registered_storage([]).unwrap()
}
#[test]
fn terminal_host_tail_waits_for_run_and_every_native_scope_in_both_orders() {
    for close_first in [false, true] {
        let pool = WorkingMemoryPool::new(2_000, 0).unwrap();
        let (r, run) = reservation(&pool, 1_000, 1_000).into_funding().unwrap();
        let host = run.open_pending_input_scope(&r, 200).unwrap();
        let a = run.scope().unwrap();
        let b = run.scope().unwrap();
        let mut storage = a
            .adopt_storage_individually([(1u32, 100), (2, 50)])
            .unwrap();
        let c = storage.remove(&1).unwrap();
        let n = storage.remove(&2).unwrap();
        drop(storage);
        assert_eq!(counts(&pool, &r), (3, 2, 850, 200));
        if close_first {
            run.close().unwrap();
            a.certify().unwrap();
            assert_eq!(balances(&pool), (850, 150, 1_000));
            b.certify().unwrap();
        } else {
            a.certify().unwrap();
            b.certify().unwrap();
            assert_eq!(balances(&pool), (850, 150, 1_000));
            run.close().unwrap();
        }
        assert_eq!(counts(&pool, &r), (1, 0, 200, 200));
        assert_eq!(balances(&pool), (200, 150, 1_000));
        assert_eq!(r.bytes(), 1_000);
        assert_eq!(pool.effective_capacity().unwrap(), 1_000);
        blocked(&pool);
        // Independent new admission uses exactly released headroom, keeping the old ceiling.
        assert!(matches!(
            try_reservation(&pool, 651, 2_000),
            Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: 651,
                available_bytes: 650
            })
        ));
        assert_eq!(balances(&pool), (200, 150, 1_000));
        let next = reservation(&pool, 650, 2_000);
        assert_eq!(pool.used_bytes().unwrap(), 1_000);
        assert!(matches!(
            pool.register_storage([(99u32, 1)]),
            Err(WorkingMemoryError::BudgetExceeded { .. })
        ));
        drop(next);
        let c_alias = c.clone();
        drop(c);
        drop(n);
        assert_eq!(balances(&pool), (200, 100, 1_000));
        drop(r);
        drop(host);
        assert_eq!(balances(&pool), (0, 100, 1_000));
        assert_eq!(pool.effective_capacity().unwrap(), 1_000);
        blocked(&pool);
        drop(c_alias);
        assert_eq!(balances(&pool), (0, 0, 1_000));
        assert_eq!(pool.effective_capacity().unwrap(), 2_000);
        drop(pool.acquire_unquoted().unwrap());
    }
}
#[test]
fn allocation_retirement_refills_only_live_work_then_releases_without_refill() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
    let host = run.open_pending_input_scope(&r, 80).unwrap();
    let native = run.scope().unwrap();
    let first = native.adopt_storage_individually([(1u32, 120)]).unwrap();
    assert_eq!(balances(&pool), (380, 120, 500));
    drop(first);
    assert_eq!(balances(&pool), (500, 0, 500));
    let later = native.adopt_storage_individually([(2u32, 90)]).unwrap();
    native.certify().unwrap();
    drop(run);
    assert_eq!(balances(&pool), (80, 90, 500));
    drop(later);
    assert_eq!(balances(&pool), (80, 0, 500));
    drop((host, r));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn host_only_source_alias_keeps_borrowed_pins_and_zero_byte_origin_ceiling() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let original = pool.register_storage([(1u32, 40)]).unwrap();
    let pin = pool.pin_registered_storage([(1u32, 40)]).unwrap();
    let (r, run) = with_borrowed_storage(reservation(&pool, 400, 500), pin)
        .into_funding()
        .unwrap();
    let host = run.sampler_scope().unwrap();
    assert_eq!(counts(&pool, &r), (1, 0, 400, 0));
    let native = run.scope().unwrap();
    let zero = native.adopt_storage_individually([(2u32, 0)]).unwrap();
    native.certify().unwrap();
    drop((original, r, run));
    assert_eq!(balances(&pool), (0, 40, 440));
    blocked(&pool);
    drop(host);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 500);
    blocked(&pool);
    drop(zero);
    assert_eq!(pool.effective_capacity().unwrap(), 1_000);
    drop(pool.acquire_unquoted().unwrap());
}
#[test]
fn quarantine_never_trims_even_after_every_host_payload_retires() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
    let host = run.open_pending_input_scope(&r, 80).unwrap();
    let native = run.scope().unwrap();
    let storage = native.adopt_storage_individually([(1u32, 120)]).unwrap();
    drop((native, run));
    assert_eq!(counts(&pool, &r), (1, 0, 380, 80));
    assert_eq!(balances(&pool), (380, 120, 500));
    drop(storage);
    assert_eq!(balances(&pool), (500, 0, 500));
    drop((host, r));
    assert_eq!(balances(&pool), (500, 0, 500));
    blocked(&pool);
}
#[test]
fn original_sampler_pending_and_grouped_holds_retire_independently() {
    let pool = WorkingMemoryPool::new(2_000, 0).unwrap();
    let (r, run) = reservation(&pool, 1_000, 1_500).into_funding().unwrap();
    let mut sampler = run.sampler_scope().unwrap();
    sampler.hold_sampler_payload(100).unwrap();
    let pending = run.open_pending_input_scope(&r, 50).unwrap();
    let source = empty(&pool);
    let sources = [
        DecoderCopySource::Registered(&source),
        DecoderCopySource::Registered(&source),
    ];
    let (_, tables, native) = run
        .open_grouped_dense_prompt_scopes(
            &r,
            &sources,
            &[30, 70],
            &source,
            RegisteredStoragePin::new(source.clone()),
        )
        .unwrap();
    assert_eq!(counts(&pool, &r), (5, 1, 1_000, 250));
    native.certify().unwrap();
    drop(run);
    assert_eq!(counts(&pool, &r), (4, 0, 250, 250));
    drop(pending);
    assert_eq!(pool.used_bytes().unwrap(), 200);
    let mut tables = tables.into_iter();
    drop(tables.next().unwrap());
    assert_eq!(pool.used_bytes().unwrap(), 170);
    drop(sampler);
    assert_eq!(pool.used_bytes().unwrap(), 70);
    drop((tables, source, r));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn dense_and_no_decoder_prompt_keep_one_actual_native_scope() {
    for dense in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
        let source = empty(&pool);
        let (host, native) = if dense {
            let (_, host, native) = run
                .open_dense_prompt_scopes(
                    &r,
                    DecoderCopySource::Registered(&source),
                    &source,
                    RegisteredStoragePin::new(source.clone()),
                    80,
                )
                .unwrap();
            (Some(host), native)
        } else {
            (
                None,
                run.open_no_decoder_prompt_scope(&r, source.clone())
                    .unwrap(),
            )
        };
        assert_eq!(counts(&pool, &r).1, 1);
        drop(run);
        assert_eq!(pool.used_bytes().unwrap(), 500);
        native.certify().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), if dense { 80 } else { 0 });
        drop((host, source, r));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn hidden_host_scope_cannot_publish_or_validate_as_native_with_live_sibling() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
    let host = run.sampler_scope().unwrap();
    let sibling = run.scope().unwrap();
    let hidden = host.scope.as_ref().unwrap();
    let before = counts(&pool, &r);
    assert!(matches!(
        hidden.adopt_storage_individually([(1u32, 10)]),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(matches!(
        hidden.adopt_storage_individually::<u32>([]),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    {
        let usage = pool.0.usage.lock().unwrap();
        assert!(matches!(
            FundingSource::NativeScope(hidden).validate(&usage, &r.0.execution),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        FundingSource::NativeScope(&sibling)
            .validate(&usage, &r.0.execution)
            .unwrap();
    }
    assert_eq!(counts(&pool, &r), before);
    sibling.certify().unwrap();
    drop((host, run, r));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn native_counter_overflow_rejects_both_counters_before_any_mint() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
    {
        let mut u = pool.0.usage.lock().unwrap();
        let a = u.funding.get_mut(&run.id).unwrap();
        a.scopes = usize::MAX - 1;
        a.native_scopes = usize::MAX;
    }
    let before = counts(&pool, &r);
    let rejected = run.scope();
    let after = counts(&pool, &r);
    // Restore artificial boundary before assertions/Drop; no synthetic scope.
    {
        let mut u = pool.0.usage.lock().unwrap();
        let a = u.funding.get_mut(&run.id).unwrap();
        a.scopes = 0;
        a.native_scopes = 0;
    }
    assert!(matches!(rejected, Err(WorkingMemoryError::Overflow)));
    assert_eq!(before, after);
    drop((r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HostKey(HostMetadataKey);
impl HostSlotStorageKey for HostKey {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        Some(&self.0)
    }
}
#[test]
fn completed_dense_table_transfers_held_bytes_before_or_after_terminal_trim() {
    for close_first in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let source = HostSlotTable::new(vec![7u64, 11, 13].into_boxed_slice());
        let plan = source
            .prepare_copy_slots()
            .unwrap()
            .for_dense_destination::<u64>()
            .unwrap();
        let retained = plan.retained_bytes();
        let protected = plan.initialization_peak_bytes();
        let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
        let mut host = run.open_pending_input_scope(&r, protected).unwrap();
        let mut builder = plan.initialize();
        for value in [17, 19, 23] {
            builder.push(value).unwrap();
        }
        let slots = builder.finish().unwrap();
        let mut run = Some(run);
        if close_first {
            drop(run.take());
        }
        crate::working_memory::storage::publish_dense_host_slots(
            &slots,
            &mut host,
            &r.0.execution,
            retained,
            protected,
            HostKey(slots.metadata().identity().registry_key().clone()),
            None,
        )
        .unwrap();
        assert_eq!(slots.get(1), Some(&19));
        drop(run);
        assert_eq!(balances(&pool), (protected - retained, retained, 500));
        let alias = slots.metadata().clone();
        drop((slots, host, r));
        assert_eq!(balances(&pool), (0, retained, 500));
        drop(alias);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn concurrent_native_close_and_storage_retirement_commit_one_terminal_tail() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
    let host = run.open_pending_input_scope(&r, 80).unwrap();
    let native = run.scope().unwrap();
    let storage = native.adopt_storage_individually([(1u32, 120)]).unwrap();
    let (go, receive) = std::sync::mpsc::sync_channel(1);
    let worker = std::thread::spawn(move || {
        receive
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        native.certify()
    });
    go.send(()).unwrap();
    drop((run, storage));
    let outcome = worker.join().unwrap();
    outcome.unwrap();
    assert_eq!(balances(&pool), (80, 0, 500));
    drop((host, r));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn copy_accounts_classify_sampler_paired_and_grouped_host_scopes_at_commit() {
    let pool = WorkingMemoryPool::new(2_000, 0).unwrap();
    let (r, original) = reservation(&pool, 100, 2_000).into_funding().unwrap();
    let source = original.sampler_scope().unwrap();
    let storage = empty(&pool);
    for kind in 0..4 {
        let pin = RegisteredStoragePin::new(storage.clone());
        let (run, native, sampler, tables, held) = match kind {
            0 => {
                let (run, native) = pool
                    .open_sampler_copy_account(source.source(), &r.0.execution, 500, 2_000)
                    .unwrap();
                (run, native, None, vec![], 0)
            }
            1 => {
                let (run, host, native) = pool
                    .open_sampling_copy_account(
                        source.source(),
                        &r.0.execution,
                        &storage,
                        Some(&storage),
                        pin,
                        &r.0.execution,
                        500,
                        30,
                        2_000,
                    )
                    .unwrap();
                (run, native, Some(host), vec![], 30)
            }
            2 => {
                let (run, sampler, table, native) = pool
                    .open_text_components_copy_account(
                        source.source(),
                        &r.0.execution,
                        DecoderCopySource::Registered(&storage),
                        &storage,
                        &storage,
                        pin,
                        &r.0.execution,
                        500,
                        30,
                        50,
                        2_000,
                    )
                    .unwrap();
                (run, native, Some(sampler), vec![table], 80)
            }
            _ => {
                let sources = [
                    DecoderCopySource::Registered(&storage),
                    DecoderCopySource::Registered(&storage),
                ];
                let (run, sampler, tables, native) = pool
                    .open_grouped_text_components_account(
                        source.source(),
                        &r.0.execution,
                        &sources,
                        &[20, 40],
                        &storage,
                        &storage,
                        pin,
                        &r.0.execution,
                        500,
                        30,
                        2_000,
                    )
                    .unwrap();
                (run, native, Some(sampler), tables, 90)
            }
        };
        {
            let u = pool.0.usage.lock().unwrap();
            let a = &u.funding[&run.id];
            assert_eq!(a.native_scopes, 1);
            assert_eq!(a.scopes, 1 + usize::from(sampler.is_some()) + tables.len());
            assert_eq!(a.host_held, held);
        }
        drop(run);
        assert_eq!(pool.used_bytes().unwrap(), 600);
        native.certify().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 100 + held);
        drop((sampler, tables));
        assert_eq!(pool.used_bytes().unwrap(), 100);
    }
    drop((source, original, r, storage));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

fn poison_usage(pool: &WorkingMemoryPool) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _usage = pool.0.usage.lock().unwrap();
        panic!("terminal host accounting poison");
    }))
    .is_err());
}

#[test]
fn poison_before_run_host_or_metadata_drop_keeps_the_original_envelope() {
    for first in 0..3 {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
        let id = r.0.funding.unwrap();
        let host = run.open_pending_input_scope(&r, 80).unwrap();
        poison_usage(&pool);
        let mut run = Some(run);
        let mut host = Some(host);
        let mut metadata = Some(r);
        match first {
            0 => drop(run.take()),
            1 => drop(host.take()),
            _ => drop(metadata.take()),
        }
        {
            let usage = pool.0.usage.lock().unwrap_err().into_inner();
            let state = &usage.funding[&id];
            assert!(state.quarantined);
            assert_eq!(state.remaining, 500);
            assert_eq!(state.host_held, if first == 1 { 0 } else { 80 });
            assert_eq!(usage.reserved, 500);
            assert_eq!(usage.registered, 0);
        }
        // Explicit close still reports the original poison; its Drop cannot
        // trim before the surviving host wrapper reaches its own failure path.
        if let Some(run) = run.take() {
            assert!(matches!(run.close(), Err(WorkingMemoryError::Poisoned)));
        }
        drop((host, metadata));
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        let state = &usage.funding[&id];
        assert!(state.quarantined);
        assert!(!state.run_open);
        assert_eq!((state.scopes, state.native_scopes), (0, 0));
        assert_eq!((state.remaining, state.host_held), (500, 0));
        assert_eq!(usage.reserved, 500);
        assert_eq!(usage.funding.capacity_counts().get(&700), Some(&1));
    }
}

#[test]
fn poisoned_storage_retirement_refills_conservative_custody_before_settle() {
    for already_trimmed in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let (r, run) = reservation(&pool, 500, 700).into_funding().unwrap();
        let id = r.0.funding.unwrap();
        let host = run.open_pending_input_scope(&r, 80).unwrap();
        let native = run.scope().unwrap();
        let storage = native.adopt_storage_individually([(1u32, 100)]).unwrap();
        native.certify().unwrap();
        let mut run = Some(run);
        if already_trimmed {
            drop(run.take());
            assert_eq!(balances(&pool), (80, 100, 500));
        }
        poison_usage(&pool);
        drop(storage);
        let retained = if already_trimmed { 180 } else { 500 };
        {
            let usage = pool.0.usage.lock().unwrap_err().into_inner();
            assert!(usage.funding[&id].quarantined);
            assert_eq!(usage.funding[&id].remaining, retained);
            assert_eq!(usage.reserved, retained);
            assert_eq!(usage.registered, 0);
            assert!(usage.storage.is_empty());
        }
        drop((run, r, host));
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert!(usage.funding[&id].quarantined);
        assert_eq!(usage.funding[&id].host_held, 0);
        assert_eq!(usage.reserved, retained);
        assert_eq!(usage.funding.capacity_counts().get(&700), Some(&1));
    }
}
