use super::*;

#[test]
fn canonical_alias_derives_closed_donor_origin_and_survives_both_final_drop_orders() {
    for drop_donor_first in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let namespace = pool.register_storage([(0u32, 0)]).unwrap();
        let (a, ar, ap) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let donor = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
        a_scope.certify().unwrap();
        // No original installation pair, explicit partition alias or live A
        // execution is available when B recognizes this existing allocation.
        drop((ar, a, ap));
        assert_eq!(balances(&pool), (100, 0));
        let (b, br, bp) = account(&pool);
        let b_scope = br.scope().unwrap();
        let aliases = publish(
            &bp,
            &b_scope,
            vec![test_existing_native(1, 64), test_existing_native(1, 64)],
        );
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].bytes(), 64);
        assert_eq!(balances(&pool), (300, 0));
        b_scope.certify().unwrap();
        drop((br, b, bp));
        assert_eq!(pool.used_bytes().unwrap(), 100);
        if drop_donor_first {
            drop(donor);
            assert_eq!(pool.used_bytes().unwrap(), 100);
            drop(aliases);
        } else {
            drop(aliases);
            assert_eq!(pool.used_bytes().unwrap(), 100);
            drop(donor);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(pool.0.usage.lock().unwrap().funding.is_empty());
        drop(namespace);
    }
}

#[test]
fn existing_route_refuses_missing_ordinary_and_same_batch_new_birth_without_committing_prefix() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let ordinary = pool.register_storage([(7u32, 64)]).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let cases = [
        vec![test_existing_native(1u32, 64)],
        vec![test_existing_native(7, 64)],
        vec![test_existing_native(2, 64), test_native(2, 64, &partition)],
        vec![test_native(2, 64, &partition), test_existing_native(2, 64)],
    ];
    let before = balances(&pool);
    for case in cases {
        let mut inputs = vec![NativePublicationInput::Ordinary(9, 30)];
        inputs.extend(case);
        let mut attempt = PreparedNativePublication::prepare(partition.clone(), inputs);
        assert_eq!(
            attempt.publish(&scope),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert!(attempt.take(0).is_none());
        assert_eq!(balances(&pool), before);
        assert!(pool.pin_registered_storage([(9u32, 30)]).is_err());
        assert!(pool.pin_registered_storage([(2u32, 64)]).is_err());
        assert_eq!(
            attempt.publish(&scope),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        );
    }
    scope.certify().unwrap();
    drop((run, metadata, partition, ordinary));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn mixed_duplicate_native_claims_use_canonical_origin_and_charge_only_new_ordinary_storage() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let namespace = pool.register_storage([(0u32, 0)]).unwrap();
    let (a, ar, ap) = account(&pool);
    let (b, br, bp) = account(&pool);
    let a_scope = ar.scope().unwrap();
    let b_scope = br.scope().unwrap();
    let donor = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
    let mut aliases = Vec::new();
    for existing_first in [false, true] {
        let mut duplicate = vec![test_native(1, 64, &ap), test_existing_native(1, 64)];
        if existing_first {
            duplicate.reverse();
        }
        duplicate.push(NativePublicationInput::Ordinary(2, 30));
        let result = publish(&bp, &b_scope, duplicate);
        assert_eq!(result.len(), 2);
        assert_eq!((result[0].bytes(), result[1].bytes()), (64, 30));
        assert_eq!(balances(&pool), (370, 30));
        aliases.push(result);
    }
    for existing_first in [false, true] {
        for wrong_origin in [false, true] {
            let conflict = if wrong_origin {
                test_native(1u32, 64, &bp)
            } else {
                NativePublicationInput::Ordinary(1, 64)
            };
            let mut inputs = vec![test_existing_native(1, 64), conflict];
            if !existing_first {
                inputs.reverse();
            }
            let mut attempt = PreparedNativePublication::prepare(bp.clone(), inputs);
            assert_eq!(
                attempt.publish(&b_scope),
                Err(WorkingMemoryError::IdentityMismatch)
            );
            assert!(attempt.take(0).is_none());
            assert_eq!(balances(&pool), (370, 30));
        }
    }
    a_scope.certify().unwrap();
    b_scope.certify().unwrap();
    drop((aliases, donor, ar, br, a, b, ap, bp, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn existing_alias_capacity_mismatch_retains_typed_cause_and_does_not_publish_ordinary_prefix() {
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
    let namespace = pool.register_storage([(0u32, 0)]).unwrap();
    let (a, ar, ap) = account(&pool);
    let (b, br, bp) = account(&pool);
    let a_scope = ar.scope().unwrap();
    let b_scope = br.scope().unwrap();
    let donor = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
    let mut attempt = PreparedNativePublication::prepare(
        bp.clone(),
        vec![
            NativePublicationInput::Ordinary(2u32, 30),
            test_existing_native(1, 65),
        ],
    );
    assert_eq!(
        attempt.publish(&b_scope),
        Err(WorkingMemoryError::StorageCapacityMismatch {
            expected_bytes: 64,
            actual_bytes: 65,
        })
    );
    assert_eq!(balances(&pool), (400, 0));
    assert!(attempt.take(0).is_none());
    assert!(pool.pin_registered_storage([(2u32, 30)]).is_err());
    a_scope.certify().unwrap();
    b_scope.certify().unwrap();
    drop((attempt, donor, ar, br, a, b, ap, bp, namespace));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn canonical_alias_checks_donor_and_publisher_health_independently() {
    for quarantine_donor in [false, true] {
        let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
        let namespace = pool.register_storage([(0u32, 0)]).unwrap();
        let (a, ar, ap) = account(&pool);
        let (b, br, bp) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let b_scope = br.scope().unwrap();
        let donor = publish(&ap, &a_scope, vec![test_native(1, 64, &ap)]);
        drop(if quarantine_donor {
            ar.scope().unwrap()
        } else {
            br.scope().unwrap()
        });
        let mut attempt =
            PreparedNativePublication::prepare(bp.clone(), vec![test_existing_native(1u32, 64)]);
        assert_eq!(
            attempt.publish(&b_scope),
            Err(WorkingMemoryError::ExecutionFenced)
        );
        assert_eq!(balances(&pool), (400, 0));
        assert!(attempt.take(0).is_none());
        a_scope.certify().unwrap();
        b_scope.certify().unwrap();
        drop((attempt, donor, ar, br, a, b, ap, bp, namespace));
        assert_eq!(pool.used_bytes().unwrap(), 200);
    }
}

#[test]
fn derived_donor_custody_survives_later_refusal_and_unlocked_provider_retirement() {
    for panic_on_drop in [false, true] {
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
        let missing_events = Arc::new(Events {
            pool: pool.clone(),
            armed: AtomicBool::new(false),
            drops: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            panic_at_clone: usize::MAX,
            panic_on_drop: AtomicBool::new(false),
            minimum_charge: 100,
        });
        let key = |id| Key {
            id,
            events: if id == 2 {
                missing_events.clone()
            } else {
                events.clone()
            },
        };
        let namespace = pool.register_storage([(key(0), 0)]).unwrap();
        let (a, ar, ap) = account(&pool);
        let (b, br, bp) = account(&pool);
        let a_scope = ar.scope().unwrap();
        let b_scope = br.scope().unwrap();
        let mut birth =
            PreparedNativePublication::prepare(ap.clone(), vec![test_native(key(1), 64, &ap)]);
        birth.publish(&a_scope).unwrap();
        let donor = birth.take(0).unwrap();
        drop(birth);
        let mut attempt = PreparedNativePublication::prepare(
            bp.clone(),
            vec![
                test_existing_native(key(1), 64),
                test_existing_native(key(2), 48),
            ],
        );
        assert_eq!(
            attempt.publish(&b_scope),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert!(attempt.take(0).is_none());
        assert_eq!(balances(&pool), (400, 0));
        a_scope.certify().unwrap();
        b_scope.certify().unwrap();
        let retained_donor = panic_on_drop.then(|| ap.clone());
        drop((donor, ar, br, a, b, ap, bp));
        // The canonical row and both original request owners are gone. The
        // failed transaction itself still pins A while its provider keys live.
        assert_eq!(pool.used_bytes().unwrap(), 200);
        events.armed.store(true, AtomicOrdering::SeqCst);
        missing_events.armed.store(true, AtomicOrdering::SeqCst);
        events
            .panic_on_drop
            .store(panic_on_drop, AtomicOrdering::SeqCst);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(attempt)));
        assert_eq!(result.is_err(), panic_on_drop);
        assert!(events.drops.load(AtomicOrdering::SeqCst) > 0);
        if let Some(alias) = retained_donor {
            let usage = pool.0.usage.lock().unwrap();
            assert_eq!(
                alias.validate_pool(&pool, &usage),
                Err(WorkingMemoryError::ExecutionFenced)
            );
            drop(usage);
            drop(alias);
            assert_eq!(pool.used_bytes().unwrap(), 200);
        } else {
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
        drop(namespace);
    }
}
