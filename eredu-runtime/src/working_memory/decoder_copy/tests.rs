use super::*;
use crate::working_memory::{
    BorrowedFundedSampler, RegisteredWorkspaceCopy, RegisteredWorkspaceStorage, RunOwnedTextSampler,
};
use crate::{ConfiguredTextSampler, HostSlotTable};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceHostBound,
    WorkspaceIsolatedCopyPlan, WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation,
    WorkspaceOperationBound, WorkspaceOperationKind, WorkspaceOutputStorage, WorkspaceTensor,
};
use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[path = "fixtures.rs"]
mod fixtures;
use fixtures::{Facts, grow, history, source};

thread_local! {
    static COPY_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
    static FAIL_COPY: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn before_copy() {
    COPY_ATTEMPTS.with(|v| v.set(v.get() + 1));
    assert!(
        !FAIL_COPY.with(|v| v.replace(false)),
        "injected before host copying"
    );
}
fn attempts() -> usize {
    COPY_ATTEMPTS.with(Cell::get)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Host(HostMetadataKey),
    Array(u32),
}
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        match self {
            Self::Host(id) => Some(id),
            Self::Array(_) => None,
        }
    }
}
const CAP: u64 = 65_536;
const ARRAY: u64 = 64;
const EXTRA: u64 = 80;
const NATIVE: u64 = 48;
fn host_key<T>(table: &HostSlotTable<T>) -> Key {
    Key::Host(table.metadata().identity().registry_key().clone())
}
fn register_table<T>(pool: &MemoryLedger, table: &HostSlotTable<T>) -> WorkingMemoryStorage<Key> {
    pool.register_host_storage([(host_key(table), table.metadata().capacity_bytes().unwrap())])
        .unwrap()
}
fn physical(pool: &MemoryLedger) -> WorkingMemoryStorage<Key> {
    pool.register_host_storage([(Key::Array(1), ARRAY), (Key::Array(2), EXTRA)])
        .unwrap()
}
fn array_plan(pool: &MemoryLedger, key: u32, capacity: u64) -> RegisteredWorkspaceCopy<Key> {
    let context = WorkspaceContext::new(Facts::default().with_pool(&pool));
    let root = WorkspaceExistingStorage::try_new_placed(
        Some(capacity),
        &pool.host_placement_handle(),
        &context,
    )
    .unwrap();
    let registered =
        RegisteredWorkspaceStorage::bind(pool, &context, [(Key::Array(key), root.clone())])
            .unwrap();
    let array = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
        &root,
        &context,
    )
    .unwrap();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, registered.borrowed_storage(), &[array])
            .unwrap();
    assert_eq!(plan.incremental_bytes(), Some(NATIVE));
    RegisteredWorkspaceCopy::bind(plan, registered).unwrap()
}
fn joint<'a, T>(
    pool: &MemoryLedger,
    sampler: BorrowedFundedSampler<'a>,
    table: &'a HostSlotTable<T>,
    complete: WorkingMemoryStorage<Key>,
) -> RegisteredTextComponentsCopy<'a, T, Key> {
    let decoder =
        RegisteredDecoderHostCopy::bind(pool, table.prepare_copy_slots().unwrap(), host_key(table))
            .unwrap();
    RegisteredSamplingCopy::prepare(sampler, array_plan(pool, 1, ARRAY))
        .unwrap()
        .with_decoder_slots(decoder, complete)
        .unwrap()
}
fn usage(pool: &MemoryLedger) -> (u64, u64, u64) {
    (
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
        pool.payload_effective_capacity().unwrap(),
    )
}
fn held(pool: &MemoryLedger) -> u64 {
    pool.0
        .usage
        .lock()
        .unwrap()
        .funding
        .values()
        .map(|s| s.host_held.checked_sub(s.control_floor).unwrap())
        .sum()
}

#[test]
fn exact_and_one_short_admit_one_account_before_both_real_host_copies() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![11u32, 23, 47].into_boxed_slice());
    let table_charge = register_table(&pool, &table);
    let (mut sampler, preparation, run) = source(&pool, CAP, 8, true);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let h = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    let plan = table.prepare_copy_slots().unwrap();
    let (d, p) = (plan.retained_bytes(), plan.initialization_peak_bytes());
    assert!(p >= d && d > table.metadata().capacity_bytes().unwrap());
    let required = joint(&pool, sampler.borrow_funded(), &table, arrays.clone())
        .required_bytes()
        .unwrap();
    assert_eq!(required, h + p + NATIVE);
    let rejected = joint(&pool, sampler.borrow_funded(), &table, arrays.clone());
    let full = pool
        .text_components_copy_requirements(&rejected, &WorkspaceCopyLimits::default())
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap();
    let snapshot = pool.snapshot().unwrap();
    let domain = &snapshot.domains[0];
    let initial = domain.current_charge_bytes - domain.fixed_baseline.total().unwrap();
    let before = (usage(&pool), attempts());
    let mut limits =
        WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP));
    limits.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 7)]);
    limits.memory_limits = crate::working_memory::memory_fixture::host_limits(initial + full + 6);
    assert!(matches!(pool.copy_text_components(rejected,limits.clone()),
        Err(DecoderCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes==full+7 && limit_bytes-existing_bytes==full+6));
    assert_eq!((usage(&pool), attempts()), before);
    let accounts = pool.0.usage.lock().unwrap().funding.len();
    limits.memory_limits = crate::working_memory::memory_fixture::host_limits(initial + full + 7);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            limits.clone(),
        )
        .unwrap();
    assert_eq!(pool.0.usage.lock().unwrap().funding.len(), accounts + 1);
    assert_eq!(attempts(), before.1 + 1);
    assert_eq!(
        native
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        full + 7
    );
    assert_eq!(
        (
            copied.bytes(),
            slots.retained_bytes(),
            slots.protected_bytes()
        ),
        (h, d, p)
    );
    assert_eq!(history(copied.as_sampler()), history(sampler.as_sampler()));
    assert_ne!(
        history(copied.as_sampler()).as_ptr(),
        history(sampler.as_sampler()).as_ptr()
    );
    for value in table.slots() {
        slots.push(*value).unwrap();
    }
    let saved = slots.finish().unwrap();
    assert_eq!(saved.iter().copied().collect::<Vec<_>>(), [11, 23, 47]);
    assert_ne!(
        saved.get(0).unwrap() as *const u32,
        &table.slots()[0] as *const u32
    );
    let source_bytes = ARRAY + EXTRA + table.metadata().capacity_bytes().unwrap();
    drop((table, table_charge, arrays, sampler, preparation, run));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        required + 7 + source_bytes
    );
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), required + 7);
    drop((copied, saved, custody));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn table_identity_capacity_and_every_source_domain_are_checked() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let other = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let foreign = physical(&other);
    let a = HostSlotTable::new(vec![2u32, 9].into_boxed_slice());
    let b = HostSlotTable::new(vec![2u32, 9].into_boxed_slice());
    let _a = register_table(&pool, &a);
    let _b = register_table(&pool, &b);
    let (sampler, _preparation, _run) = source(&pool, CAP, 8, false);
    let before = (usage(&pool), usage(&other), attempts());
    for key in [host_key(&b), Key::Array(1)] {
        assert!(matches!(
            RegisteredDecoderHostCopy::bind(&pool, a.prepare_copy_slots().unwrap(), key),
            Err(DecoderCopyAdmissionError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
    }
    assert!(matches!(
        RegisteredDecoderHostCopy::bind(&other, a.prepare_copy_slots().unwrap(), host_key(&a)),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let missing = HostSlotTable::new(vec![7u32].into_boxed_slice());
    assert!(matches!(
        RegisteredDecoderHostCopy::bind(
            &pool,
            missing.prepare_copy_slots().unwrap(),
            host_key(&missing)
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let wrong_capacity = pool
        .register_host_storage([(host_key(&missing), 3)])
        .unwrap();
    assert!(matches!(
        RegisteredDecoderHostCopy::bind(
            &pool,
            missing.prepare_copy_slots().unwrap(),
            host_key(&missing)
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    drop(wrong_capacity);
    // Cold source descriptors receive their own metadata funding before the
    // rejected destination transaction begins.
    let foreign_source = joint(&pool, sampler.borrow_funded(), &a, foreign.clone());
    let foreign_destination = joint(&pool, sampler.borrow_funded(), &a, arrays.clone());
    let after_temporary = (usage(&pool), usage(&other), attempts());
    assert_eq!(after_temporary.0.0, before.0.0);
    assert!(matches!(
        pool.copy_text_components(
            foreign_source,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP))
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert!(matches!(
        other.copy_text_components(
            foreign_destination,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP))
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!((usage(&pool), usage(&other), attempts()), after_temporary);
}

#[test]
fn native_adoption_respects_both_holds_and_each_owner_releases_only_its_own() {
    for decoder_first in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
        let arrays = physical(&pool);
        let table = HostSlotTable::new(vec![1u32, 2, 3].into_boxed_slice());
        let table_charge = register_table(&pool, &table);
        let (sampler, preparation, run) = source(&pool, CAP, 8, false);
        let (copied, mut slots, native) = pool
            .copy_text_components(
                joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
            )
            .unwrap();
        let (h, p) = (copied.bytes(), slots.protected_bytes());
        for value in table.slots() {
            slots.push(*value).unwrap();
        }
        let saved = slots.finish().unwrap();
        let total = copied.bytes() + saved.protected_bytes() + NATIVE;
        let (custody, scope) = native.into_parts();
        drop((sampler, preparation, run, table, table_charge, arrays));
        assert_eq!(held(&pool), h + p);
        let native_charge = scope
            .publish_host_storage_fixture([(Key::Array(10), NATIVE)])
            .unwrap();
        assert!(matches!(
            scope.publish_host_storage_fixture([(Key::Array(11), 1)]),
            Err(WorkingMemoryError::DomainAllowanceExceeded {
                available_bytes: 0,
                ..
            })
        ));
        let mut copied = Some(copied);
        let mut saved = Some(saved);
        let first = if decoder_first {
            drop(saved.take());
            p
        } else {
            drop(copied.take());
            h
        };
        assert_eq!(held(&pool), h + p - first);
        let first_charge = scope
            .publish_host_storage_fixture([(Key::Array(11), first)])
            .unwrap();
        assert!(matches!(
            scope.publish_host_storage_fixture([(Key::Array(12), 1)]),
            Err(WorkingMemoryError::DomainAllowanceExceeded {
                available_bytes: 0,
                ..
            })
        ));
        drop((saved, copied));
        assert_eq!(held(&pool), 0);
        let second_charge = scope
            .publish_host_storage_fixture([(Key::Array(12), h + p - first)])
            .unwrap();
        drop(custody);
        scope.certify().unwrap();
        assert_eq!(pool.payload_used_bytes().unwrap(), total);
        drop((native_charge, first_charge, second_charge));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn incomplete_finish_keeps_actual_payload_and_hold_through_error_and_recovery() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![1u32, 2].into_boxed_slice());
    let _table_charge = register_table(&pool, &table);
    let (sampler, _preparation, _run) = source(&pool, CAP, 8, false);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    slots.push(17).unwrap();
    let before = held(&pool);
    let error = slots.finish().unwrap_err();
    assert_eq!((error.expected(), error.initialized()), (2, 1));
    assert!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<HostSlotFinishError<u32>>()
            .is_some()
    );
    assert_eq!(held(&pool), before);
    let mut slots = error.into_slots();
    slots.push(29).unwrap();
    let rejected = slots.push(41).unwrap_err();
    assert_eq!(rejected.into_value(), 41);
    let saved = slots.finish().unwrap();
    assert_eq!(saved.iter().copied().collect::<Vec<_>>(), [17, 29]);
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((copied, saved, custody));
}

#[test]
fn independent_full_source_origin_quarantine_is_rechecked_at_commit() {
    for table_origin in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
        let base = pool
            .register_host_storage([(Key::Array(1), ARRAY)])
            .unwrap();
        let table = HostSlotTable::new(vec![3u32, 5].into_boxed_slice());
        let (sampler, _preparation, _run) = source(&pool, CAP, 8, false);
        let origin = pool
            .admit_workspace_copy(
                array_plan(&pool, 1, ARRAY),
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
            )
            .unwrap();
        let (origin, scope) = origin.into_parts();
        let source_key = if table_origin {
            host_key(&table)
        } else {
            Key::Array(2)
        };
        let capacity = if table_origin {
            table.metadata().capacity_bytes().unwrap()
        } else {
            16
        };
        let adopted = scope
            .publish_host_storage_fixture([(source_key.clone(), capacity)])
            .unwrap();
        let table_charge = if table_origin {
            None
        } else {
            Some(register_table(&pool, &table))
        };
        let complete = pool
            .pin_registered_storage([(Key::Array(1), ARRAY), (source_key, capacity)])
            .unwrap();
        let plan = joint(&pool, sampler.borrow_funded(), &table, complete);
        drop(scope); // Healthy when bound, quarantined before one destination commit.
        let before = (usage(&pool), attempts());
        assert!(matches!(
            pool.copy_text_components(
                plan,
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP))
            ),
            Err(DecoderCopyAdmissionError::Memory(
                WorkingMemoryError::ExecutionFenced
            ))
        ));
        assert_eq!((usage(&pool), attempts()), before);
        drop((base, adopted, table_charge, origin));
    }
}

#[test]
fn host_payload_drop_precedes_hold_release_and_never_certifies_native_work() {
    struct Value {
        pool: MemoryLedger,
        drops: Arc<AtomicUsize>,
        minimum: u64,
    }
    impl Drop for Value {
        fn drop(&mut self) {
            // Reenters the usage lock while the host payload is being destroyed.
            assert!(held(&self.pool) >= self.minimum);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let drops = Arc::new(AtomicUsize::new(0));
    let table = HostSlotTable::new(
        vec![Value {
            pool: pool.clone(),
            drops: drops.clone(),
            minimum: 0,
        }]
        .into_boxed_slice(),
    );
    let table_charge = register_table(&pool, &table);
    let (sampler, preparation, run) = source(&pool, CAP, 8, false);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    let p = slots.protected_bytes();
    slots
        .push(Value {
            pool: pool.clone(),
            drops: drops.clone(),
            minimum: p,
        })
        .unwrap();
    let saved = slots.finish().unwrap();
    let bytes = copied.bytes() + saved.protected_bytes() + NATIVE;
    let source_bytes = ARRAY + EXTRA + table.metadata().capacity_bytes().unwrap();
    let (custody, scope) = native.into_parts();
    drop((
        sampler,
        preparation,
        run,
        table,
        table_charge,
        arrays,
        copied,
        custody,
    ));
    drop(saved);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert_eq!(held(&pool), 0);
    assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes + bytes);
    drop(scope);
    assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes + bytes);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn abandonment_retains_one_complete_pin_bundle_and_rejects_saved_source_reuse() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![7u32, 13].into_boxed_slice());
    let key = host_key(&table);
    let table_bytes = table.metadata().capacity_bytes().unwrap();
    let table_charge = register_table(&pool, &table);
    let (sampler, preparation, run) = source(&pool, CAP, 8, false);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    slots.push(7).unwrap();
    slots.push(13).unwrap();
    let saved = slots.finish().unwrap();
    let total = copied.bytes() + saved.protected_bytes() + NATIVE;
    let (custody, scope) = native.into_parts();
    let outputs = scope
        .publish_host_storage_fixture([(Key::Array(3), 16)])
        .unwrap();
    let next =
        RegisteredSamplingCopy::prepare(sampler.borrow_funded(), array_plan(&pool, 1, ARRAY))
            .unwrap()
            .with_decoder_slots(saved.prepare_copy().unwrap(), arrays.clone())
            .unwrap();
    drop(scope);
    let before = (usage(&pool), attempts());
    assert!(matches!(
        pool.copy_text_components(
            next,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP))
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), attempts()), before);
    drop((
        sampler,
        preparation,
        run,
        table,
        table_charge,
        arrays,
        copied,
        saved,
        custody,
        outputs,
    ));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        ARRAY + EXTRA + table_bytes + total
    );
    // Every original source, including the uncopied root and table, survived in
    // the one quarantined bundle. No destination-only custody owns these pins.
    drop(
        pool.pin_registered_storage([
            (Key::Array(1), ARRAY),
            (Key::Array(2), EXTRA),
            (key, table_bytes),
        ])
        .unwrap(),
    );
}

#[test]
fn saved_host_sources_copy_after_original_run_and_native_custody_retire() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![2u32, 17].into_boxed_slice());
    let table_charge = register_table(&pool, &table);
    let (mut sampler, preparation, run) = source(&pool, CAP, 8, true);
    grow(&mut sampler, &[3, 9, 5]);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    for v in table.slots() {
        slots.push(*v).unwrap();
    }
    let saved = slots.finish().unwrap();
    let (custody, scope) = native.into_parts();
    let mut outputs = scope
        .publish_host_storage_fixture([(Key::Array(3), 16)])
        .unwrap();
    scope.certify().unwrap();
    drop((
        custody,
        sampler,
        preparation,
        run,
        table,
        table_charge,
        arrays,
    ));
    let full = outputs.remove(&Key::Array(3)).unwrap();
    let next = RegisteredSamplingCopy::prepare(copied.borrow_funded(), array_plan(&pool, 3, 16))
        .unwrap()
        .with_decoder_slots(saved.prepare_copy().unwrap(), full.clone())
        .unwrap();
    let (copied2, mut slots2, native2) = pool
        .copy_text_components(
            next,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    assert_eq!(slots2.retained_bytes(), saved.retained_bytes());
    assert_eq!(slots2.protected_bytes(), saved.protected_bytes());
    for v in saved.iter() {
        slots2.push(*v).unwrap();
    }
    let saved2 = slots2.finish().unwrap();
    assert_eq!(saved2.iter().copied().collect::<Vec<_>>(), [2, 17]);
    assert_ne!(
        saved.get(0).unwrap() as *const u32,
        saved2.get(0).unwrap() as *const u32
    );
    assert_eq!(history(copied2.as_sampler()), history(copied.as_sampler()));
    let total = copied2.bytes() + saved2.protected_bytes() + NATIVE;
    drop((copied, saved, full));
    let (custody2, scope2) = native2.into_parts();
    scope2.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), total);
    drop((copied2, saved2, custody2));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn failure_before_host_initialization_quarantines_all_source_pins() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![7u32].into_boxed_slice());
    let table_charge = register_table(&pool, &table);
    let key = host_key(&table);
    let extent = table.metadata().capacity_bytes().unwrap();
    let (sampler, preparation, run) = source(&pool, CAP, 8, false);
    let total = joint(&pool, sampler.borrow_funded(), &table, arrays.clone())
        .required_bytes()
        .unwrap();
    FAIL_COPY.with(|v| v.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    }));
    assert!(result.is_err());
    drop((sampler, preparation, run, table, table_charge, arrays));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        ARRAY + EXTRA + extent + total
    );
    drop(
        pool.pin_registered_storage([
            (Key::Array(1), ARRAY),
            (Key::Array(2), EXTRA),
            (key, extent),
        ])
        .unwrap(),
    );
}

#[test]
fn empty_slots_keep_a_distinct_host_scope_without_a_fabricated_byte_charge() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::<u32>::new(Vec::new().into_boxed_slice());
    let table_charge = register_table(&pool, &table);
    let (sampler, preparation, run) = source(&pool, CAP, 8, false);
    let (copied, slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    assert_eq!(
        (slots.len(), slots.retained_bytes(), slots.protected_bytes()),
        (0, 0, 0)
    );
    let saved = slots.finish().unwrap();
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((
        copied,
        custody,
        sampler,
        preparation,
        run,
        table,
        table_charge,
        arrays,
    ));
    // A zero-byte host owner keeps its original account/exclusion without
    // retaining unrelated transient bytes after healthy native/run closure.
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(pool.acquire_unquoted().is_err());
    let decoder = saved.prepare_copy::<Key>().unwrap();
    assert!(decoder.is_empty());
    drop(decoder);
    drop(saved);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn partial_finish_error_drops_payload_under_its_hold_and_full_push_returns_original_value() {
    struct Value {
        id: u32,
        pool: MemoryLedger,
        drops: Arc<AtomicUsize>,
        minimum: u64,
    }
    impl Drop for Value {
        fn drop(&mut self) {
            assert!(held(&self.pool) >= self.minimum);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let source_drops = Arc::new(AtomicUsize::new(0));
    let table = HostSlotTable::new(
        (0..2)
            .map(|id| Value {
                id,
                pool: pool.clone(),
                drops: source_drops.clone(),
                minimum: 0,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let _table_charge = register_table(&pool, &table);
    let (sampler, _preparation, _run) = source(&pool, CAP, 8, false);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    let p = slots.protected_bytes();
    let drops = Arc::new(AtomicUsize::new(0));
    slots
        .push(Value {
            id: 17,
            pool: pool.clone(),
            drops: drops.clone(),
            minimum: p,
        })
        .unwrap();
    let before = held(&pool);
    let error = slots.finish().unwrap_err();
    assert_eq!(held(&pool), before);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(held(&pool), before - p);
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((copied, custody));

    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    let p = slots.protected_bytes();
    let installed = Arc::new(AtomicUsize::new(0));
    for id in [23, 29] {
        slots
            .push(Value {
                id,
                pool: pool.clone(),
                drops: installed.clone(),
                minimum: p,
            })
            .unwrap();
    }
    let incoming_drops = Arc::new(AtomicUsize::new(0));
    let incoming = Value {
        id: 41,
        pool: pool.clone(),
        drops: incoming_drops.clone(),
        minimum: 0,
    };
    let rejected = slots.push(incoming).unwrap_err();
    assert_eq!((slots.initialized_count(), rejected.capacity()), (2, 2));
    assert_eq!(incoming_drops.load(Ordering::SeqCst), 0);
    // The never-installed value remains caller-owned; rejection grants it no
    // host or nested native funding and does not overwrite either funded slot.
    let returned = rejected.into_value();
    assert_eq!(returned.id, 41);
    drop(returned);
    assert_eq!(incoming_drops.load(Ordering::SeqCst), 1);
    let saved = slots.finish().unwrap();
    assert_eq!(saved.iter().map(|v| v.id).collect::<Vec<_>>(), [23, 29]);
    drop(saved);
    assert_eq!(installed.load(Ordering::SeqCst), 2);
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((copied, custody));
}

#[test]
fn sampler_quarantine_and_foreign_saved_host_proof_reject_without_copying() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![1u32].into_boxed_slice());
    let _table_charge = register_table(&pool, &table);
    let (sampler, _preparation, run) = source(&pool, CAP, 8, false);
    let plan = joint(&pool, sampler.borrow_funded(), &table, arrays.clone());
    drop(run.scope().unwrap());
    let before = (usage(&pool), attempts());
    assert!(matches!(
        pool.copy_text_components(
            plan,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP))
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), attempts()), before);

    let a = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let b = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays_a = physical(&a);
    let arrays_b = physical(&b);
    let table = HostSlotTable::new(vec![17u32].into_boxed_slice());
    let _table_charge = register_table(&a, &table);
    let (sampler_a, _prep_a, _run_a) = source(&a, CAP, 8, false);
    let (sampler_b, _prep_b, _run_b) = source(&b, CAP, 8, false);
    let (copied, mut slots, native) = a
        .copy_text_components(
            joint(&a, sampler_a.borrow_funded(), &table, arrays_a),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    slots.push(17).unwrap();
    let saved = slots.finish().unwrap();
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((copied, custody));
    let plan = RegisteredSamplingCopy::prepare(sampler_b.borrow_funded(), array_plan(&b, 1, ARRAY))
        .unwrap()
        .with_decoder_slots(saved.prepare_copy().unwrap(), arrays_b.clone())
        .unwrap();
    let before = (usage(&a), usage(&b), attempts());
    assert!(matches!(
        b.copy_text_components(
            plan,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP))
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!((usage(&a), usage(&b), attempts()), before);
}

#[test]
fn admitted_builder_matches_exact_actual_source_before_any_fill() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAP, 0).unwrap();
    let arrays = physical(&pool);
    let table = HostSlotTable::new(vec![3u32, 5].into_boxed_slice());
    let _table_charge = register_table(&pool, &table);
    let replacement = HostSlotTable::new(vec![3u32, 5].into_boxed_slice());
    let (sampler, _preparation, _run) = source(&pool, CAP, 8, false);
    let (copied, mut slots, native) = pool
        .copy_text_components(
            joint(&pool, sampler.borrow_funded(), &table, arrays.clone()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    let before = usage(&pool);
    assert!(matches!(
        slots.validate_source(&replacement.prepare_copy_slots().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!((slots.initialized_count(), usage(&pool)), (0, before));
    slots
        .validate_source(&table.prepare_copy_slots().unwrap())
        .unwrap();
    slots.push(3).unwrap();
    assert!(matches!(
        slots.validate_source(&table.prepare_copy_slots().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    slots.push(5).unwrap();
    let saved = slots.finish().unwrap();
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((copied, custody));
    let next =
        RegisteredSamplingCopy::prepare(sampler.borrow_funded(), array_plan(&pool, 1, ARRAY))
            .unwrap()
            .with_decoder_slots(saved.prepare_copy().unwrap(), arrays)
            .unwrap();
    let (copied, mut slots, native) = pool
        .copy_text_components(
            next,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(CAP)),
        )
        .unwrap();
    // Saved-to-saved uses the actual saved table, not its original live owner.
    assert!(matches!(
        slots.validate_source(&table.prepare_copy_slots().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    slots
        .validate_source(&saved.prepare_copy_slots().unwrap())
        .unwrap();
    for v in saved.iter() {
        slots.push(*v).unwrap();
    }
    let saved2 = slots.finish().unwrap();
    let (custody, scope) = native.into_parts();
    scope.certify().unwrap();
    drop((copied, custody, saved, saved2));
}
