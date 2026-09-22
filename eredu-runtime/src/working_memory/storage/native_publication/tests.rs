use super::*;
use crate::working_memory::InferenceExecutionIdentity;
use crate::working_memory::WorkingMemoryReservation;
use crate::working_memory::funding::native_partition::test_receipt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

// These ownership fixtures permit two live request accounts. Their real host
// metadata is priced separately from the backing capacities under assertion.
fn test_ledger(capacity: u64, existing: u64) -> Result<MemoryLedger, WorkingMemoryError> {
    let seed = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0)?;
    let mut admission = crate::working_memory::memory_fixture::host_admission(&seed, 200);
    admission.memory_limits = seed.configured_limits().named(seed.topology()).unwrap();
    let requirements = crate::working_memory::memory_fixture::host_requirements(&seed, 200);
    let metadata = crate::working_memory::memory_fixture::reservation_metadata_bytes(
        &seed,
        &admission,
        &requirements,
    );
    // Three ordinary namespaces can coexist with the two native accounts.
    let publication = crate::working_memory::StoragePublicationLayout::<Key>::new(1)?
        .requested_bytes()
        .checked_add(MemoryLedger::storage_metadata_control_bytes()?)
        .unwrap();
    crate::working_memory::memory_fixture::host_ledger(
        capacity
            .checked_add(metadata.checked_mul(2).unwrap())
            .unwrap()
            .checked_add(publication.checked_mul(3).unwrap())
            .unwrap(),
        existing,
    )
}
fn reservation(pool: &MemoryLedger, bytes: u64, _capacity: u64) -> WorkingMemoryReservation {
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &crate::working_memory::memory_fixture::host_admission(pool, bytes),
        pool.configured_limits().clone(),
    )
    .unwrap()
}
impl MemoryLedger {
    fn native_payload_bytes(&self) -> Result<u64, WorkingMemoryError> {
        self.payload_used_bytes()
    }
}

fn account(
    pool: &MemoryLedger,
) -> (
    super::super::super::WorkingMemoryReservation,
    funding::WorkingMemoryFundingRun,
    NativePartition,
) {
    let (metadata, run) = reservation(pool, 200, 1_000).into_funding().unwrap();
    let partition = run.take_native_partition(test_receipt(&run, 100)).unwrap();
    (metadata, run, partition)
}

fn publish(
    partition: &NativePartition,
    scope: &WorkingMemoryFundingScope,
    inputs: Vec<NativePublicationInput<u32>>,
) -> Vec<WorkingMemoryStorage<u32>> {
    let mut attempt = PreparedNativePublication::prepare(partition.clone(), inputs);
    attempt.publish(scope).unwrap();
    (0..attempt.rows.len())
        .map(|i| attempt.take(i).unwrap())
        .collect()
}

fn balances(pool: &MemoryLedger) -> (u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.reserved
            - usage
                .funding
                .values()
                .map(|state| state.control_floor)
                .sum::<u64>(),
        usage.registered - usage.registry_metadata,
    )
}

#[test]
fn mixed_rows_charge_only_ordinary_bytes_and_cross_request_aliases_keep_donor_origin() {
    for drop_birth_first in [false, true] {
        let pool = test_ledger(1_000, 0).unwrap();
        let namespace = pool.register_host_storage([(0u32, 0)]).unwrap();
        let (a, ar, ap) = account(&pool);
        let (b, br, bp) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let b_scope = br.scope().unwrap();
        let mut a_rows = publish(
            &ap,
            &a_scope,
            vec![
                test_native(1, 64, &ap),
                NativePublicationInput::Ordinary(2, 30),
            ],
        );
        assert_eq!(balances(&pool), (306, 94));
        assert_eq!(a_rows[0].bytes(), Some(64));
        assert_eq!(pool.native_payload_bytes().unwrap(), 400);
        let b_rows = publish(
            &bp,
            &b_scope,
            vec![test_native(1, 64, &ap), test_native(1, 64, &ap)],
        );
        assert_eq!(b_rows.len(), 1);
        assert_eq!(balances(&pool), (306, 94));
        drop(a_rows.pop()); // the independently funded ordinary row
        a_scope.certify().unwrap();
        b_scope.certify().unwrap();
        drop((ar, br, a, b, bp));
        assert_eq!(pool.native_payload_bytes().unwrap(), 100);
        // The native birth's accounting projection and a stale registry alias
        // may retire in either order. Only the complete partition is released.
        if drop_birth_first {
            drop(ap);
            drop(a_rows);
            assert_eq!(pool.native_payload_bytes().unwrap(), 100);
            drop(b_rows);
        } else {
            drop(a_rows);
            drop(b_rows);
            assert_eq!(pool.native_payload_bytes().unwrap(), 100);
            drop(ap);
        }
        assert_eq!(pool.native_payload_bytes().unwrap(), 0);
        assert!(pool.0.usage.lock().unwrap().funding.is_empty());
        drop(namespace);
    }
}

#[test]
fn cumulative_native_rows_do_not_replay_credit_or_bound_rows_by_partition_capacity() {
    let pool = test_ledger(1_000, 0).unwrap();
    let namespace = pool.register_host_storage([(0u32, 0)]).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let first = publish(&partition, &scope, vec![test_native(1, 80, &partition)]);
    let before = balances(&pool);
    let mut premature =
        PreparedNativePublication::prepare(partition.clone(), vec![test_native(2, 80, &partition)]);
    assert!(premature.publish(&scope).is_err());
    assert_eq!(balances(&pool), before);
    drop(premature);
    // The attached sidecar certifies physical retirement independently of this
    // stale canonical alias. The native producer can then reuse its capacity.
    first[0].retire_native_backing();
    let second = publish(&partition, &scope, vec![test_native(2, 80, &partition)]);
    assert_eq!(first[0].bytes().unwrap() + second[0].bytes().unwrap(), 160);
    assert_eq!(balances(&pool), (120, 80));
    scope.certify().unwrap();
    drop((run, metadata, partition));
    assert_eq!(pool.native_payload_bytes().unwrap(), 100);
    drop(first);
    assert_eq!(pool.native_payload_bytes().unwrap(), 100);
    drop(second);
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
    drop(namespace);
}

#[test]
fn native_capacity_origin_and_new_foreign_birth_failures_are_atomic_and_terminal() {
    let pool = test_ledger(1_000, 0).unwrap();
    let namespace = pool.register_host_storage([(0u32, 0)]).unwrap();
    let (a, ar, ap) = account(&pool);
    let (b, br, bp) = account(&pool);
    let a_scope = ar.scope().unwrap();
    let b_scope = br.scope().unwrap();
    let rows = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
    for (witness, capacity_error) in [
        (test_native(1u32, 65, &ap), true),
        (test_native(1, 64, &bp), false),
        (test_native(3, 64, &ap), false),
        (test_native(4, 101, &bp), false),
    ] {
        let mut attempt = PreparedNativePublication::prepare(
            bp.clone(),
            vec![NativePublicationInput::Ordinary(2, 30), witness],
        );
        let error = attempt.publish(&b_scope).unwrap_err();
        if capacity_error {
            assert!(matches!(
                error,
                WorkingMemoryError::StorageCapacityMismatch {
                    expected_bytes: 64,
                    actual_bytes: 65
                }
            ));
        } else {
            assert_eq!(error, WorkingMemoryError::IdentityMismatch);
        }
        assert_eq!(balances(&pool), (336, 64));
        assert!(attempt.take(0).is_none());
        assert_eq!(
            attempt.publish(&b_scope).unwrap_err(),
            WorkingMemoryError::PreparationAlreadyStarted
        );
        assert!(pool.pin_registered_storage([(2u32, 30)]).is_err());
    }
    a_scope.certify().unwrap();
    b_scope.certify().unwrap();
    drop((rows, ar, br, a, b, ap, bp, namespace));
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
}

#[test]
fn donor_and_publisher_health_are_independent_even_for_zero_increment_aliases() {
    for quarantine_donor in [false, true] {
        let pool = test_ledger(1_000, 0).unwrap();
        let namespace = pool.register_host_storage([(0u32, 0)]).unwrap();
        let (a, ar, ap) = account(&pool);
        let (b, br, bp) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let b_scope = br.scope().unwrap();
        let rows = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
        let ordinary_alias = crate::working_memory::StoragePublicationLayout::<u32>::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap();
        // Actual uncertified scope retirement fences exactly its own account.
        drop(if quarantine_donor {
            ar.scope().unwrap()
        } else {
            br.scope().unwrap()
        });
        let mut attempt =
            PreparedNativePublication::prepare(bp.clone(), vec![test_native(1u32, 64, &ap)]);
        assert_eq!(
            attempt.publish(&b_scope).unwrap_err(),
            WorkingMemoryError::ExecutionFenced
        );
        assert_eq!(balances(&pool), (336, 64));
        assert!(attempt.take(0).is_none());
        // The ordinary funded alias route must preserve the same native origin.
        assert!(matches!(
            ordinary_alias.adopt_storage_individually(
                &b_scope,
                [(
                    1u32,
                    crate::working_memory::StorageAllocation::new(64, pool.host_placement_handle())
                )]
            ),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        a_scope.certify().unwrap();
        b_scope.certify().unwrap();
        drop((attempt, rows, ar, br, a, b, ap, bp, namespace));
        assert_eq!(pool.native_payload_bytes().unwrap(), 200);
    }
}

struct Events {
    pool: MemoryLedger,
    armed: AtomicBool,
    drops: AtomicUsize,
    clones: AtomicUsize,
    panic_at_clone: usize,
    panic_on_drop: AtomicBool,
    minimum_charge: u64,
}
struct Key {
    id: u32,
    events: Arc<Events>,
}
impl Clone for Key {
    fn clone(&self) -> Self {
        assert!(self.events.pool.0.usage.try_lock().is_ok());
        let clone = self.events.clones.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        assert_ne!(
            clone, self.events.panic_at_clone,
            "prepared key clone failure"
        );
        Self {
            id: self.id,
            events: self.events.clone(),
        }
    }
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        if self.id != 0 && self.events.armed.load(AtomicOrdering::SeqCst) {
            assert!(
                self.events.pool.0.usage.try_lock().is_ok(),
                "provider key retired under Usage"
            );
            assert!(
                self.events.pool.native_payload_bytes().unwrap() >= self.events.minimum_charge,
                "partition retired before key"
            );
            self.events.drops.fetch_add(1, AtomicOrdering::SeqCst);
            assert!(
                !self
                    .events
                    .panic_on_drop
                    .swap(false, AtomicOrdering::SeqCst),
                "provider key destructor failure"
            );
        }
    }
}

#[test]
fn native_keys_retire_unlocked_before_final_partition_including_preparation_unwind() {
    for panic_on_clone in [false, true] {
        let pool = test_ledger(1_000, 0).unwrap();
        let events = Arc::new(Events {
            pool: pool.clone(),
            armed: AtomicBool::new(false),
            drops: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            panic_at_clone: if panic_on_clone { 3 } else { usize::MAX },
            panic_on_drop: AtomicBool::new(false),
            minimum_charge: 100,
        });
        let namespace = pool
            .register_host_storage([(
                Key {
                    id: 0,
                    events: events.clone(),
                },
                0,
            )])
            .unwrap();
        events.clones.store(0, AtomicOrdering::SeqCst);
        let (metadata, run, partition) = account(&pool);
        let scope = run.scope().unwrap();
        let mut attempt = PreparedNativePublication::prepare(
            partition.clone(),
            vec![
                test_native(
                    Key {
                        id: 1,
                        events: events.clone(),
                    },
                    40,
                    &partition,
                ),
                test_native(
                    Key {
                        id: 2,
                        events: events.clone(),
                    },
                    50,
                    &partition,
                ),
            ],
        );
        events.armed.store(true, AtomicOrdering::SeqCst);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| attempt.publish(&scope)));
        if panic_on_clone {
            assert!(result.is_err());
            assert_eq!(attempt.inputs.len(), 2);
            assert!(attempt.take(0).is_none());
            assert_eq!(
                attempt.publish(&scope).unwrap_err(),
                WorkingMemoryError::PreparationAlreadyStarted
            );
        } else {
            result.unwrap().unwrap();
        }
        let first = attempt.take(0);
        let last = attempt.take(1);
        scope.certify().unwrap();
        drop((run, metadata, partition));
        assert_eq!(pool.native_payload_bytes().unwrap(), 100);
        drop(attempt);
        if !panic_on_clone {
            drop(first);
            assert_eq!(pool.native_payload_bytes().unwrap(), 100);
            drop(last);
        }
        assert!(events.drops.load(AtomicOrdering::SeqCst) > 0);
        assert_eq!(pool.native_payload_bytes().unwrap(), 0);
        drop(namespace);
    }
}

#[test]
fn reset_layout_final_pin_uses_native_coverage_without_an_ordinary_refund() {
    use crate::working_memory::{HostSlotStorageKey, storage::reset_layout::ResetLayoutPin};
    use crate::{HostMetadataKey, HostSlotTable};
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct MetadataKey(HostMetadataKey);
    impl HostSlotStorageKey for MetadataKey {
        fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
            Some(&self.0)
        }
    }
    let pool = test_ledger(10_000, 0).unwrap();
    let sentinel = HostSlotTable::new(vec![1u32].into_boxed_slice());
    let table = HostSlotTable::new(vec![3u32, 4].into_boxed_slice());
    let sentinel_bytes = sentinel.metadata().capacity_bytes().unwrap();
    let bytes = table.metadata().capacity_bytes().unwrap();
    let sentinel_key = MetadataKey(sentinel.metadata().identity().registry_key().clone());
    let key = MetadataKey(table.metadata().identity().registry_key().clone());
    let namespace = pool
        .register_host_storage([(sentinel_key, sentinel_bytes)])
        .unwrap();
    let (metadata, run) = reservation(&pool, bytes * 4, 10_000)
        .into_funding()
        .unwrap();
    let partition = run
        .take_native_partition(test_receipt(&run, bytes * 2))
        .unwrap();
    let scope = run.scope().unwrap();
    // This is the private neutral realization, not a claim that a host table
    // was allocated by a native budget. It exercises the actual reset pin route.
    let mut attempt = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_native(key.clone(), bytes, &partition)],
    );
    attempt.publish(&scope).unwrap();
    let owner = attempt.take(0).unwrap();
    let mut first = ResetLayoutPin::prepare::<MetadataKey>(&pool, &key.0, bytes).unwrap();
    let mut last = ResetLayoutPin::prepare::<MetadataKey>(&pool, &key.0, bytes).unwrap();
    {
        let mut usage = pool.0.usage.lock().unwrap();
        let locator = ResetLayoutPin::locate_existing::<MetadataKey>(&usage, &key, bytes).unwrap();
        ResetLayoutPin::acquire_pair::<MetadataKey>(
            &mut first, &mut last, locator, locator, &mut usage,
        )
        .unwrap();
    }
    scope.certify().unwrap();
    drop((attempt, owner, run, metadata, partition));
    assert_eq!(
        pool.native_payload_bytes().unwrap(),
        sentinel_bytes + bytes * 2
    );
    drop(first);
    assert_eq!(
        pool.native_payload_bytes().unwrap(),
        sentinel_bytes + bytes * 2
    );
    drop(last);
    assert_eq!(pool.native_payload_bytes().unwrap(), sentinel_bytes);
    drop((key, table, sentinel, namespace));
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
}

#[test]
fn provider_destructor_unwind_keeps_native_partition_charge_quarantined() {
    for keep_alias in [false, true] {
        let pool = test_ledger(1_000, 0).unwrap();
        let events = Arc::new(Events {
            pool: pool.clone(),
            armed: AtomicBool::new(false),
            drops: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            panic_at_clone: usize::MAX,
            panic_on_drop: AtomicBool::new(false),
            minimum_charge: 100,
        });
        let namespace = pool
            .register_host_storage([(
                Key {
                    id: 0,
                    events: events.clone(),
                },
                0,
            )])
            .unwrap();
        let (metadata, run, partition) = account(&pool);
        let scope = run.scope().unwrap();
        let mut attempt = PreparedNativePublication::prepare(
            partition.clone(),
            vec![test_native(
                Key {
                    id: 1,
                    events: events.clone(),
                },
                64,
                &partition,
            )],
        );
        attempt.publish(&scope).unwrap();
        let owner = attempt.take(0).unwrap();
        scope.certify().unwrap();
        let retained_alias = keep_alias.then(|| partition.clone());
        drop((attempt, run, metadata, partition));
        assert_eq!(pool.native_payload_bytes().unwrap(), 100);
        events.armed.store(true, AtomicOrdering::SeqCst);
        events.panic_on_drop.store(true, AtomicOrdering::SeqCst);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner))).is_err());
        // The mutex itself is healthy: the provider ran outside Usage. Its unwind
        // still preserves the original pool credit rather than certifying cleanup.
        assert_eq!(pool.native_payload_bytes().unwrap(), 100);
        let usage = pool.0.usage.lock().unwrap();
        if let Some(alias) = &retained_alias {
            assert_eq!(
                alias.validate_pool(&pool, &usage).unwrap_err(),
                WorkingMemoryError::ExecutionFenced
            );
        }
        assert!(matches!(
            usage
                .funding
                .values()
                .next()
                .unwrap()
                .validate_registered_copy_origin(),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        drop(usage);
        // A later normal alias drop must not turn the caught failure into credit.
        drop(retained_alias);
        assert_eq!(pool.native_payload_bytes().unwrap(), 100);
        drop(namespace);
    }
}

#[test]
fn failed_foreign_prefix_keeps_donor_until_pending_output_keys_retire() {
    let pool = test_ledger(1_000, 0).unwrap();
    let events = Arc::new(Events {
        pool: pool.clone(),
        armed: AtomicBool::new(false),
        drops: AtomicUsize::new(0),
        clones: AtomicUsize::new(0),
        panic_at_clone: usize::MAX,
        panic_on_drop: AtomicBool::new(false),
        minimum_charge: 200,
    });
    let namespace = pool
        .register_host_storage([(
            Key {
                id: 0,
                events: events.clone(),
            },
            0,
        )])
        .unwrap();
    let (a, ar, ap) = account(&pool);
    let (b, br, bp) = account(&pool);
    let b_scope = br.scope().unwrap();
    let mut attempt = PreparedNativePublication::prepare(
        bp.clone(),
        vec![test_native(
            Key {
                id: 1,
                events: events.clone(),
            },
            64,
            &ap,
        )],
    );
    assert_eq!(
        attempt.publish(&b_scope).unwrap_err(),
        WorkingMemoryError::IdentityMismatch
    );
    b_scope.certify().unwrap();
    drop((ar, br, a, b, ap, bp));
    assert_eq!(pool.native_payload_bytes().unwrap(), 200);
    events.armed.store(true, AtomicOrdering::SeqCst);
    drop(attempt);
    assert!(events.drops.load(AtomicOrdering::SeqCst) >= 3);
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
    drop(namespace);
}

mod existing_alias;

#[test]
fn qualified_rows_keep_prefix_and_surviving_partition_alias_on_provider_unwind() {
    if !crate::working_memory::qualified_storage::qualified() {
        return;
    }
    for clone_panic in [true, false] {
        let pool = test_ledger(1_000, 0).unwrap();
        let events = Arc::new(Events {
            pool: pool.clone(),
            armed: AtomicBool::new(false),
            drops: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            panic_at_clone: if clone_panic { 3 } else { usize::MAX },
            panic_on_drop: AtomicBool::new(false),
            minimum_charge: 100,
        });
        let namespace = pool
            .register_host_storage([(
                Key {
                    id: 0,
                    events: events.clone(),
                },
                0,
            )])
            .unwrap();
        let (metadata, run, partition) = account(&pool);
        let scope = run.scope().unwrap();
        // Private ledger fixture tests the actual qualified storage worker;
        // it is not a generic-Key complete control quote or native birth proof.
        let mut attempt =
            PreparedNativePublication::prepare_slots_exact(partition.clone(), 2).unwrap();
        for id in [1, 2] {
            attempt
                .push_observation(
                    crate::working_memory::NativeStorageObservation::Originating(
                        Key {
                            id,
                            events: events.clone(),
                        },
                        40,
                    ),
                )
                .unwrap();
        }
        events.clones.store(0, AtomicOrdering::SeqCst);
        events.armed.store(true, AtomicOrdering::SeqCst);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| attempt.publish(&scope)));
        if clone_panic {
            assert!(result.is_err());
            assert_eq!(attempt.inputs.len(), 2);
            assert_eq!(attempt.rows.len(), 1);
            assert!(attempt.take(0).is_none());
            assert_eq!(attempt.control_capacities(), [2; 3]);
            scope.certify().unwrap();
            drop((attempt, run, metadata, partition));
            assert_eq!(pool.native_payload_bytes().unwrap(), 0);
        } else {
            result.unwrap().unwrap();
            let first = attempt.take(0).unwrap();
            let last = attempt.take(1).unwrap();
            scope.certify().unwrap();
            let retained = partition.clone();
            drop((attempt, run, metadata, partition));
            events.panic_on_drop.store(true, AtomicOrdering::SeqCst);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(first))).is_err()
            );
            {
                let usage = pool.0.usage.lock().unwrap();
                assert_eq!(
                    retained.validate_pool(&pool, &usage).unwrap_err(),
                    WorkingMemoryError::ExecutionFenced
                );
            }
            drop((last, retained));
            assert_eq!(
                pool.native_payload_bytes().unwrap(),
                100,
                "later normal retirement cannot refund quarantined origin"
            );
        }
        drop(namespace);
    }
}

#[test]
fn first_native_publication_installs_real_namespace_and_foreign_alias_keeps_origin() {
    let pool = test_ledger(1_000, 0).unwrap();
    let (a, ar, ap) = account(&pool);
    let (b, br, bp) = account(&pool);
    let a_scope = ar.scope().unwrap();
    let b_scope = br.scope().unwrap();
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
    let a_rows = publish(&ap, &a_scope, vec![test_native(7, 64, &ap)]);
    let b_rows = publish(&bp, &b_scope, vec![test_existing_native(7, 64)]);
    assert_eq!(balances(&pool), (336, 64));
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 1);
    assert_eq!(b_rows[0].bytes(), Some(64));
    a_scope.certify().unwrap();
    b_scope.certify().unwrap();
    drop((ar, br, a, b, ap, bp, a_rows));
    assert_eq!(pool.native_payload_bytes().unwrap(), 100);
    drop(b_rows);
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
    let usage = pool.0.usage.lock().unwrap();
    assert!(usage.storage.is_empty());
    assert!(usage.funding.is_empty());
}

#[test]
fn refused_first_rows_and_empty_attempt_leave_no_namespace_or_partial_registration() {
    let pool = test_ledger(1_000, 0).unwrap();
    let (r, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let before = balances(&pool);
    let mut missing =
        PreparedNativePublication::prepare(partition.clone(), vec![test_existing_native(7u32, 64)]);
    assert!(matches!(
        missing.publish(&scope),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut refused = PreparedNativePublication::prepare(
        partition.clone(),
        vec![
            test_native(7u32, 64, &partition),
            test_native(8, 101, &partition),
        ],
    );
    assert!(matches!(
        refused.publish(&scope),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(refused.take(0).is_none());
    assert_eq!(
        refused.inputs.len(),
        2,
        "actual refused input owners survive"
    );
    let mut empty = PreparedNativePublication::<u32>::prepare(partition.clone(), vec![]);
    empty.publish(&scope).unwrap();
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
    assert_eq!(balances(&pool), before);
    let rows = publish(&partition, &scope, vec![test_native(7, 64, &partition)]);
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 1);
    scope.certify().unwrap();
    drop((rows, empty, r, run, partition));
    assert_eq!(
        pool.native_payload_bytes().unwrap(),
        100,
        "refused attempts retain their origin"
    );
    drop((missing, refused));
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
}

#[test]
fn prepared_and_ordinary_namespaces_share_lookup_and_detach_middle_head_and_tail() {
    struct Retirement {
        pool: MemoryLedger,
        armed: AtomicBool,
        exits: AtomicUsize,
    }
    #[derive(Clone)]
    struct Key(u32, Arc<Retirement>);
    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.cmp(&other.0)
        }
    }
    impl Drop for Key {
        fn drop(&mut self) {
            if self.1.armed.load(AtomicOrdering::SeqCst) {
                let usage = self
                    .1
                    .pool
                    .0
                    .usage
                    .try_lock()
                    .expect("final provider retirement must run outside Usage");
                assert!(
                    usage.storage.get(&TypeId::of::<Key>()).is_none(),
                    "the final namespace is already detached"
                );
                drop(usage);
                assert_eq!(
                    self.1.pool.native_payload_bytes().unwrap(),
                    41,
                    "provider keys retire before ordinary payload credit"
                );
                self.1.exits.fetch_add(1, AtomicOrdering::SeqCst);
            }
        }
    }
    let pool = test_ledger(1_000, 0).unwrap();
    let retirement = Arc::new(Retirement {
        pool: pool.clone(),
        armed: AtomicBool::new(false),
        exits: AtomicUsize::new(0),
    });
    let tail = pool.register_host_storage([(1u16, 11)]).unwrap();
    let (r, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let mut publication = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_native(Key(7, retirement.clone()), 64, &partition)],
    );
    publication.publish(&scope).unwrap();
    let row = publication.take(0).unwrap();
    let head = pool.register_host_storage([(1u64, 13)]).unwrap();
    let same_namespace = pool
        .register_host_storage([(Key(8, retirement.clone()), 17)])
        .unwrap();
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 3);
    scope.certify().unwrap();
    drop((r, run, partition, row, publication));
    assert_eq!(pool.native_payload_bytes().unwrap(), 41);
    retirement.armed.store(true, AtomicOrdering::SeqCst);
    drop(same_namespace); // middle: prepared namespace, final ordinary row
    assert!(retirement.exits.load(AtomicOrdering::SeqCst) > 0);
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 2);
    drop(pool.pin_registered_storage([(1u16, 11)]).unwrap());
    drop(pool.pin_registered_storage([(1u64, 13)]).unwrap());
    drop(head);
    assert_eq!(pool.native_payload_bytes().unwrap(), 11);
    drop(tail);
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
}

mod registered_source;

#[test]
fn physical_owner_alias_preserves_closed_ordinary_and_copy_funding() {
    use crate::working_memory::{
        AdmittedWorkspaceCopy, InferenceExecutionIdentity, residual::RegisteredStoragePin,
    };
    for copied in [false, true] {
        let pool = test_ledger(10_000, 0).unwrap();
        let source = pool
            .register_host_storage(std::iter::empty::<(u32, u64)>())
            .unwrap();
        let original = if copied {
            let execution = InferenceExecutionIdentity::default();
            let pin = RegisteredStoragePin::new(
                pool.pin_registered_storage(std::iter::empty::<(u32, u64)>())
                    .unwrap(),
            );
            let control = crate::working_memory::StoragePublicationLayout::<u32>::new(1)
                .unwrap()
                .requested_bytes()
                + MemoryLedger::storage_metadata_control_bytes().unwrap();
            let requirements =
                crate::working_memory::memory_fixture::host_requirements(&pool, 64 + control);
            let limits = pool.configured_limits().clone();
            let (run, scope) = pool
                .open_workspace_copy_account(&source, pin, &execution, &requirements, limits)
                .unwrap();
            let (copy, scope) =
                AdmittedWorkspaceCopy::from_account(execution, requirements, run, scope)
                    .into_parts();
            let mut rows = scope
                .adopt_host_storage_individually([(71u32, 64)])
                .unwrap();
            let row = rows.remove(&71).unwrap();
            scope.certify().unwrap();
            drop((rows, copy));
            row
        } else {
            pool.register_host_storage([(71u32, 64)]).unwrap()
        };
        assert_eq!(pool.native_payload_bytes().unwrap(), 64);
        let (metadata, run, partition) = account(&pool);
        let scope = run.scope().unwrap();
        let before = balances(&pool);
        // A prepaid-source claim remains false for either ordinary origin.
        let mut strict = PreparedNativePublication::prepare_slots(partition.clone(), 1);
        strict
            .push_observation(
                crate::working_memory::NativeStorageObservation::ExistingImmutable(71u32, 64),
            )
            .unwrap();
        assert_eq!(
            strict.publish(&scope),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(balances(&pool), before);
        drop(strict);
        let mut alias = PreparedNativePublication::prepare_slots(partition.clone(), 2);
        for _ in 0..2 {
            alias
                .push_observation(
                    crate::working_memory::NativeStorageObservation::ExistingPhysical(71u32, 64),
                )
                .unwrap();
        }
        alias.publish(&scope).unwrap();
        assert_eq!(balances(&pool), before);
        let retained = alias.take(0).unwrap();
        assert!(alias.take(1).is_none());
        scope.certify().unwrap();
        drop((alias, original, metadata, run, partition, source));
        assert_eq!(
            pool.native_payload_bytes().unwrap(),
            64,
            "the last exact physical alias preserves its donor's full charge"
        );
        drop(retained);
        assert_eq!(pool.native_payload_bytes().unwrap(), 0);
    }
}

#[test]
fn physical_owner_alias_refuses_missing_capacity_and_conflicting_classifications() {
    let pool = test_ledger(10_000, 0).unwrap();
    let source = pool.register_host_storage([(71u32, 64)]).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    for (key, bytes, ordinary_duplicate) in [(72u32, 64, false), (71, 65, false), (71, 64, true)] {
        let before = balances(&pool);
        let mut attempt = PreparedNativePublication::prepare_slots(partition.clone(), 3);
        attempt
            .push_observation(crate::working_memory::NativeStorageObservation::Ordinary(
                90, 8,
            ))
            .unwrap();
        attempt
            .push_observation(
                crate::working_memory::NativeStorageObservation::ExistingPhysical(key, bytes),
            )
            .unwrap();
        if ordinary_duplicate {
            attempt
                .push_observation(crate::working_memory::NativeStorageObservation::Ordinary(
                    key, bytes,
                ))
                .unwrap();
        }
        let error = attempt.publish(&scope).unwrap_err();
        if bytes == 65 {
            assert!(matches!(
                error,
                WorkingMemoryError::StorageCapacityMismatch {
                    expected_bytes: 64,
                    actual_bytes: 65
                }
            ));
        } else {
            assert_eq!(error, WorkingMemoryError::IdentityMismatch);
        }
        assert_eq!(attempt.missing_existing_input(), (key == 72).then_some(1));
        assert_eq!(balances(&pool), before);
        assert!(pool.pin_registered_storage([(90u32, 8)]).is_err());
        assert!(attempt.take(0).is_none());
        assert_eq!(
            attempt.publish(&scope),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        );
    }
    scope.certify().unwrap();
    drop((metadata, run, partition, source));
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
}

#[test]
fn physical_owner_alias_retains_closed_prepaid_origin_and_refuses_unhealthy_accounts() {
    for unhealthy in [None, Some(true), Some(false)] {
        let pool = test_ledger(1_000, 0).unwrap();
        let (a, ar, ap) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let donor = publish(&ap, &a_scope, vec![test_native(1u32, 64, &ap)]);
        let (b, br, bp) = account(&pool);
        let b_scope = br.scope().unwrap();
        if let Some(donor_unhealthy) = unhealthy {
            drop(if donor_unhealthy {
                ar.scope().unwrap()
            } else {
                br.scope().unwrap()
            });
        }
        a_scope.certify().unwrap();
        drop((a, ar, ap));
        let before = balances(&pool);
        let mut alias = PreparedNativePublication::prepare_slots(bp.clone(), 1);
        alias
            .push_observation(
                crate::working_memory::NativeStorageObservation::ExistingPhysical(1u32, 64),
            )
            .unwrap();
        let result = alias.publish(&b_scope);
        assert_eq!(balances(&pool), before);
        let retained = if unhealthy.is_some() {
            assert_eq!(result, Err(WorkingMemoryError::ExecutionFenced));
            assert!(alias.take(0).is_none());
            None
        } else {
            result.unwrap();
            Some(alias.take(0).unwrap())
        };
        b_scope.certify().unwrap();
        drop((alias, donor, b, br, bp));
        if unhealthy.is_none() {
            assert_eq!(pool.native_payload_bytes().unwrap(), 100);
            drop(retained);
            assert_eq!(pool.native_payload_bytes().unwrap(), 0);
        } else {
            assert_eq!(
                pool.native_payload_bytes().unwrap(),
                200,
                "quarantine retains the entire account"
            );
        }
    }
}

#[test]
fn native_publication_converts_allowance_and_independent_backings_retire_with_stale_aliases() {
    let pool = test_ledger(10_000, 0).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let mut rows = publish(
        &partition,
        &scope,
        vec![
            test_native(1, 16, &partition),
            test_native(2, 64, &partition),
        ],
    );
    let destination = rows.pop().unwrap();
    let staging = rows.pop().unwrap();
    assert_eq!(balances(&pool), (120, 80));
    let alias = destination.clone();
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(balances(&pool), (20, 80));
    partition.retire_completed_occupancy(80);
    assert_eq!(balances(&pool), (0, 80));
    // Native physical free precedes the deferred attachment payload. Neither
    // callback can release the still-live destination or charge it twice.
    partition.retire_completed_occupancy(64);
    staging.retire_native_backing();
    assert_eq!(balances(&pool), (0, 64));
    staging.retire_native_backing();
    assert_eq!(balances(&pool), (0, 64));
    drop((staging, destination));
    assert_eq!(balances(&pool), (0, 64));
    partition.retire_completed_occupancy(0);
    alias.retire_native_backing();
    assert_eq!(balances(&pool), (0, 0));
    drop((alias, partition));
    assert!(pool.0.usage.lock().unwrap().funding.is_empty());
}

fn numerical_account(
    pool: &MemoryLedger,
    placement: &eredu_core::MemoryPlacement,
    bytes: u64,
) -> crate::working_memory::OriginalNumericalSource {
    let mut native = eredu_core::DomainMemoryRequirements::zero(pool.topology());
    native.add_allocation(bytes, placement).unwrap();
    pool.reserve_numerical_source(
        &InferenceExecutionIdentity::default(),
        crate::working_memory::NumericalSourceRequirements::new(
            native,
            Some(128),
            Some(0),
            crate::working_memory::HostSourceConstructionFacts::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        pool.configured_limits().clone(),
    )
    .unwrap()
}

#[test]
fn completed_numerical_publication_keeps_exact_accounts_without_registry_or_payload_charge() {
    use crate::working_memory::{
        NativeStorageObservation,
        memory_fixture::{self, separate},
    };
    for unified in [false, true] {
        for managed in [false, true] {
            let pool = if unified {
                memory_fixture::host_ledger(1 << 24, 0).unwrap()
            } else {
                separate::device_ledger(64, 0).unwrap()
            };
            let baseline = pool.snapshot().unwrap();
            let target = if unified {
                pool.topology().host_domain()
            } else {
                separate::device_domain(&pool)
            };
            let placement = Arc::new(if managed {
                eredu_core::MemoryPlacement::possible(
                    pool.topology(),
                    vec![pool.topology().host_domain(), target],
                    "native fixture candidate set".into(),
                )
                .unwrap()
            } else {
                eredu_core::MemoryPlacement::fixed(pool.topology(), target).unwrap()
            });
            let source = numerical_account(&pool, &placement, 64);
            let custody = source.budget_custody();
            let (metadata, run, partition) = account(&pool);
            let scope = run.scope().unwrap();
            let mut attempt = PreparedNativePublication::prepare_slots(partition.clone(), 3);
            for key in [11u32, 12, 11] {
                attempt
                    .push_placed_observation(
                        NativeStorageObservation::CompletedNumerical(key, 32, custody.clone()),
                        placement.clone(),
                    )
                    .unwrap();
            }
            let before = pool.snapshot().unwrap();
            attempt.publish(&scope).unwrap();
            assert!(pool.registered_allocation(&11u32).unwrap().is_none());
            assert!(pool.registered_allocation(&12u32).unwrap().is_none());
            for (a, b) in pool
                .snapshot()
                .unwrap()
                .domains
                .iter()
                .zip(before.domains.iter())
            {
                assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
                assert_eq!(a.registered_storage_bytes, b.registered_storage_bytes);
                assert_eq!(a.historical_peak_bytes, b.historical_peak_bytes);
            }
            let first = attempt.take_input(0).unwrap();
            let second = attempt.take_input(1).unwrap();
            assert!(attempt.take_input(2).is_none());
            assert_eq!(first.bytes(), Some(32));
            assert!(!first.same_registered_storage(&second));
            let transient = crate::working_memory::NativeStorageRegistration {
                registration: std::sync::OnceLock::from(first.clone()),
                native_attached: AtomicBool::new(true),
                _raw: crate::working_memory::OriginalHostMetadataCustody::from_numerical(
                    custody.clone(),
                ),
            };
            transient
                .confirm_completed_numerical_custody(&custody)
                .unwrap();
            assert!(!transient.native_attached.load(AtomicOrdering::Acquire));
            drop(transient);
            {
                let usage = pool.0.usage.lock().unwrap();
                first.validate_copy_source(&pool, &usage).unwrap();
            }
            scope.certify().unwrap();
            drop((attempt, source, custody, metadata, run, partition, first));
            assert!(
                pool.snapshot()
                    .unwrap()
                    .domains
                    .iter()
                    .zip(baseline.domains.iter())
                    .any(|(a, b)| a.current_charge_bytes > b.current_charge_bytes)
            );
            drop(second);
            for (a, b) in pool
                .snapshot()
                .unwrap()
                .domains
                .iter()
                .zip(baseline.domains.iter())
            {
                assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
            }
        }
    }
}

#[test]
fn completed_publication_confirmation_rejects_different_and_quarantined_accounts() {
    use crate::working_memory::{NativeStorageObservation, NativeStorageRegistration};
    let pool = test_ledger(1 << 20, 0).unwrap();
    let placement = pool.host_placement_handle();
    let source = numerical_account(&pool, &placement, 64);
    let other = numerical_account(&pool, &placement, 64);
    let custody = source.budget_custody();
    let (_metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let mut attempt = PreparedNativePublication::prepare_slots(partition, 1);
    attempt
        .push_placed_observation(
            NativeStorageObservation::CompletedNumerical(17u32, 64, custody.clone()),
            placement,
        )
        .unwrap();
    attempt.publish(&scope).unwrap();
    let registration = NativeStorageRegistration {
        registration: std::sync::OnceLock::from(attempt.take_input(0).unwrap()),
        native_attached: AtomicBool::new(true),
        _raw: crate::working_memory::OriginalHostMetadataCustody::from_numerical(custody.clone()),
    };
    let before = pool.snapshot().unwrap();
    assert_eq!(
        registration.confirm_completed_numerical_custody(&other.budget_custody()),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert!(registration.native_attached.load(AtomicOrdering::Acquire));
    assert_eq!(pool.snapshot().unwrap(), before);
    custody.quarantine();
    assert_eq!(
        registration.confirm_completed_numerical_custody(&custody),
        Err(WorkingMemoryError::ExecutionFenced)
    );
    assert!(registration.native_attached.load(AtomicOrdering::Acquire));
    assert_eq!(pool.snapshot().unwrap(), before);
    scope.certify().unwrap();
}

#[test]
fn completed_numerical_final_row_rejections_preserve_every_publication_domain() {
    use crate::working_memory::{NativeStorageObservation, memory_fixture::separate};
    for failure in 0..4 {
        let pool = separate::device_ledger(64, 0).unwrap();
        let placement = separate::device_placement(&pool);
        let source = numerical_account(&pool, &placement, 64);
        let custody = source.budget_custody();
        let foreign = separate::device_ledger(64, 0).unwrap();
        let foreign_source = numerical_account(&foreign, &placement, 64);
        let (metadata, run, partition) = account(&pool);
        let scope = run.scope().unwrap();
        let mut attempt = PreparedNativePublication::prepare_slots(partition.clone(), 3);
        attempt
            .push_observation(NativeStorageObservation::Originating(1u32, 16))
            .unwrap();
        let (bytes, selected, source_account) = match failure {
            0 => (65, placement.clone(), custody.clone()),
            1 => (64, pool.host_placement_handle(), custody.clone()),
            2 => (64, placement.clone(), foreign_source.budget_custody()),
            _ => {
                custody.quarantine();
                (64, placement.clone(), custody.clone())
            }
        };
        attempt
            .push_placed_observation(
                NativeStorageObservation::CompletedNumerical(2u32, bytes, source_account),
                selected,
            )
            .unwrap();
        let before = pool.snapshot().unwrap();
        assert!(attempt.publish(&scope).is_err());
        assert!(pool.registered_allocation(&1u32).unwrap().is_none());
        for (a, b) in pool
            .snapshot()
            .unwrap()
            .domains
            .iter()
            .zip(before.domains.iter())
        {
            assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
            assert_eq!(a.historical_peak_bytes, b.historical_peak_bytes);
            assert_eq!(a.registered_storage_bytes, b.registered_storage_bytes);
        }
        scope.certify().unwrap();
        drop((
            attempt,
            source,
            custody,
            foreign_source,
            metadata,
            run,
            partition,
        ));
    }
}
