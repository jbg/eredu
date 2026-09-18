use super::*;
use crate::working_memory::funding::{native_partition::test_receipt, tests::reservation};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

fn account(
    pool: &WorkingMemoryPool,
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

fn balances(pool: &WorkingMemoryPool) -> (u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (usage.reserved, usage.registered)
}

#[test]
fn mixed_rows_charge_only_ordinary_bytes_and_cross_request_aliases_keep_donor_origin() {
    for drop_birth_first in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let namespace = pool.register_storage([(0u32, 0)]).unwrap();
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
        assert_eq!(balances(&pool), (370, 30));
        assert_eq!(a_rows[0].bytes(), 64);
        assert_eq!(pool.used_bytes().unwrap(), 400);
        let b_rows = publish(
            &bp,
            &b_scope,
            vec![test_native(1, 64, &ap), test_native(1, 64, &ap)],
        );
        assert_eq!(b_rows.len(), 1);
        assert_eq!(balances(&pool), (370, 30));
        drop(a_rows.pop()); // the independently funded ordinary row
        a_scope.certify().unwrap();
        b_scope.certify().unwrap();
        drop((ar, br, a, b, bp));
        assert_eq!(pool.used_bytes().unwrap(), 100);
        // The native birth's accounting projection and a stale registry alias
        // may retire in either order. Only the complete partition is released.
        if drop_birth_first {
            drop(ap);
            drop(a_rows);
            assert_eq!(pool.used_bytes().unwrap(), 100);
            drop(b_rows);
        } else {
            drop(a_rows);
            drop(b_rows);
            assert_eq!(pool.used_bytes().unwrap(), 100);
            drop(ap);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(pool.0.usage.lock().unwrap().funding.is_empty());
        drop(namespace);
    }
}

#[test]
fn cumulative_native_rows_do_not_replay_credit_or_bound_rows_by_partition_capacity() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let namespace = pool.register_storage([(0u32, 0)]).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let first = publish(&partition, &scope, vec![test_native(1, 80, &partition)]);
    // The private neutral realization describes a later birth after physical
    // reuse. Both accounting rows survive; no native occupancy is inferred.
    let second = publish(&partition, &scope, vec![test_native(2, 80, &partition)]);
    assert_eq!(first[0].bytes() + second[0].bytes(), 160);
    assert_eq!(balances(&pool), (200, 0));
    scope.certify().unwrap();
    drop((run, metadata, partition));
    assert_eq!(pool.used_bytes().unwrap(), 100);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), 100);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(namespace);
}

#[test]
fn native_capacity_origin_and_new_foreign_birth_failures_are_atomic_and_terminal() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let namespace = pool.register_storage([(0u32, 0)]).unwrap();
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
        assert_eq!(balances(&pool), (400, 0));
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
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn donor_and_publisher_health_are_independent_even_for_zero_increment_aliases() {
    for quarantine_donor in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let namespace = pool.register_storage([(0u32, 0)]).unwrap();
        let (a, ar, ap) = account(&pool);
        let (b, br, bp) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let b_scope = br.scope().unwrap();
        let rows = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
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
        assert_eq!(balances(&pool), (400, 0));
        assert!(attempt.take(0).is_none());
        // The ordinary funded alias route must preserve the same native origin.
        assert!(matches!(
            b_scope.adopt_storage_individually([(1u32, 64)]),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        a_scope.certify().unwrap();
        b_scope.certify().unwrap();
        drop((attempt, rows, ar, br, a, b, ap, bp, namespace));
        assert_eq!(pool.used_bytes().unwrap(), 200);
    }
}

struct Events {
    pool: WorkingMemoryPool,
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
                self.events.pool.used_bytes().unwrap() >= self.events.minimum_charge,
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
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
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
            .register_storage([(
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
        assert_eq!(pool.used_bytes().unwrap(), 100);
        drop(attempt);
        if !panic_on_clone {
            drop(first);
            assert_eq!(pool.used_bytes().unwrap(), 100);
            drop(last);
        }
        assert!(events.drops.load(AtomicOrdering::SeqCst) > 0);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        drop(namespace);
    }
}

#[test]
fn reset_layout_final_pin_uses_native_coverage_without_an_ordinary_refund() {
    use crate::working_memory::{storage::reset_layout::ResetLayoutPin, HostSlotStorageKey};
    use crate::{HostMetadataKey, HostSlotTable};
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct MetadataKey(HostMetadataKey);
    impl HostSlotStorageKey for MetadataKey {
        fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
            Some(&self.0)
        }
    }
    let pool = WorkingMemoryPool::new(10_000, 0).unwrap();
    let sentinel = HostSlotTable::new(vec![1u32].into_boxed_slice());
    let table = HostSlotTable::new(vec![3u32, 4].into_boxed_slice());
    let sentinel_bytes = sentinel.metadata().capacity_bytes().unwrap();
    let bytes = table.metadata().capacity_bytes().unwrap();
    let sentinel_key = MetadataKey(sentinel.metadata().identity().registry_key().clone());
    let key = MetadataKey(table.metadata().identity().registry_key().clone());
    let namespace = pool
        .register_storage([(sentinel_key, sentinel_bytes)])
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
    assert_eq!(pool.used_bytes().unwrap(), sentinel_bytes + bytes * 2);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), sentinel_bytes + bytes * 2);
    drop(last);
    assert_eq!(pool.used_bytes().unwrap(), sentinel_bytes);
    drop((key, table, sentinel, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn provider_destructor_unwind_keeps_native_partition_charge_quarantined() {
    for keep_alias in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
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
            .register_storage([(
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
        assert_eq!(pool.used_bytes().unwrap(), 100);
        events.armed.store(true, AtomicOrdering::SeqCst);
        events.panic_on_drop.store(true, AtomicOrdering::SeqCst);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner))).is_err());
        // The mutex itself is healthy: the provider ran outside Usage. Its unwind
        // still preserves the original pool credit rather than certifying cleanup.
        assert_eq!(pool.used_bytes().unwrap(), 100);
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
        assert_eq!(pool.used_bytes().unwrap(), 100);
        drop(namespace);
    }
}

#[test]
fn failed_foreign_prefix_keeps_donor_until_pending_output_keys_retire() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
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
        .register_storage([(
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
    assert_eq!(pool.used_bytes().unwrap(), 200);
    events.armed.store(true, AtomicOrdering::SeqCst);
    drop(attempt);
    assert!(events.drops.load(AtomicOrdering::SeqCst) >= 3);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(namespace);
}

mod existing_alias;

#[test]
fn qualified_rows_keep_prefix_and_surviving_partition_alias_on_provider_unwind() {
    if !crate::working_memory::qualified_storage::qualified() {
        return;
    }
    for clone_panic in [true, false] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
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
            .register_storage([(
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
            assert_eq!(pool.used_bytes().unwrap(), 0);
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
                pool.used_bytes().unwrap(),
                100,
                "later normal retirement cannot refund quarantined origin"
            );
        }
        drop(namespace);
    }
}

#[test]
fn first_native_publication_installs_real_namespace_and_foreign_alias_keeps_origin() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let (a, ar, ap) = account(&pool);
    let (b, br, bp) = account(&pool);
    let a_scope = ar.scope().unwrap();
    let b_scope = br.scope().unwrap();
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
    let a_rows = publish(&ap, &a_scope, vec![test_native(7, 64, &ap)]);
    let b_rows = publish(&bp, &b_scope, vec![test_existing_native(7, 64)]);
    assert_eq!(balances(&pool), (400, 0));
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 1);
    assert_eq!(b_rows[0].bytes(), 64);
    a_scope.certify().unwrap();
    b_scope.certify().unwrap();
    drop((ar, br, a, b, ap, bp, a_rows));
    assert_eq!(pool.used_bytes().unwrap(), 100);
    drop(b_rows);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let usage = pool.0.usage.lock().unwrap();
    assert!(usage.storage.is_empty());
    assert!(usage.funding.is_empty());
}

#[test]
fn refused_first_rows_and_empty_attempt_leave_no_namespace_or_partial_registration() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
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
        pool.used_bytes().unwrap(),
        100,
        "refused attempts retain their origin"
    );
    drop((missing, refused));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
}

#[test]
fn prepared_and_ordinary_namespaces_share_lookup_and_detach_middle_head_and_tail() {
    struct Retirement {
        pool: WorkingMemoryPool,
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
                    self.1.pool.used_bytes().unwrap(),
                    41,
                    "provider keys retire before ordinary payload credit"
                );
                self.1.exits.fetch_add(1, AtomicOrdering::SeqCst);
            }
        }
    }
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let retirement = Arc::new(Retirement {
        pool: pool.clone(),
        armed: AtomicBool::new(false),
        exits: AtomicUsize::new(0),
    });
    let tail = pool.register_storage([(1u16, 11)]).unwrap();
    let (r, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let mut publication = PreparedNativePublication::prepare(
        partition.clone(),
        vec![test_native(Key(7, retirement.clone()), 64, &partition)],
    );
    publication.publish(&scope).unwrap();
    let row = publication.take(0).unwrap();
    let head = pool.register_storage([(1u64, 13)]).unwrap();
    let same_namespace = pool
        .register_storage([(Key(8, retirement.clone()), 17)])
        .unwrap();
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 3);
    scope.certify().unwrap();
    drop((r, run, partition, row, publication));
    assert_eq!(pool.used_bytes().unwrap(), 41);
    retirement.armed.store(true, AtomicOrdering::SeqCst);
    drop(same_namespace); // middle: prepared namespace, final ordinary row
    assert!(retirement.exits.load(AtomicOrdering::SeqCst) > 0);
    assert_eq!(pool.0.usage.lock().unwrap().storage.len(), 2);
    drop(pool.pin_registered_storage([(1u16, 11)]).unwrap());
    drop(pool.pin_registered_storage([(1u64, 13)]).unwrap());
    drop(head);
    assert_eq!(pool.used_bytes().unwrap(), 11);
    drop(tail);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
}

mod registered_source;

#[test]
fn physical_owner_alias_preserves_closed_ordinary_and_copy_funding() {
    use crate::working_memory::{
        residual::RegisteredStoragePin, AdmittedWorkspaceCopy, InferenceExecutionIdentity,
    };
    for copied in [false, true] {
        let pool = WorkingMemoryPool::new(10_000, 0).unwrap();
        let source = pool
            .register_storage(std::iter::empty::<(u32, u64)>())
            .unwrap();
        let original = if copied {
            let execution = InferenceExecutionIdentity::default();
            let pin = RegisteredStoragePin::new(
                pool.pin_registered_storage(std::iter::empty::<(u32, u64)>())
                    .unwrap(),
            );
            let (run, scope) = pool
                .open_workspace_copy_account(&source, pin, &execution, 64, 10_000)
                .unwrap();
            let (copy, scope) =
                AdmittedWorkspaceCopy::from_account(execution, 64, run, scope).into_parts();
            let mut rows = scope.adopt_storage_individually([(71u32, 64)]).unwrap();
            let row = rows.remove(&71).unwrap();
            scope.certify().unwrap();
            drop((rows, copy));
            row
        } else {
            pool.register_storage([(71u32, 64)]).unwrap()
        };
        assert_eq!(pool.used_bytes().unwrap(), 64);
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
            pool.used_bytes().unwrap(),
            64,
            "the last exact physical alias preserves its donor's full charge"
        );
        drop(retained);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn physical_owner_alias_refuses_missing_capacity_and_conflicting_classifications() {
    let pool = WorkingMemoryPool::new(10_000, 0).unwrap();
    let source = pool.register_storage([(71u32, 64)]).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    for (key, bytes, ordinary_duplicate) in [(72u32, 64, false), (71, 65, false), (71, 64, true)] {
        let before = balances(&pool);
        let mut attempt = PreparedNativePublication::prepare_slots(partition.clone(), 2);
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
        assert_eq!(balances(&pool), before);
        assert!(attempt.take(0).is_none());
        assert_eq!(
            attempt.publish(&scope),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        );
    }
    scope.certify().unwrap();
    drop((metadata, run, partition, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
