use super::*;
use eredu_checkpoint::store::{CheckpointSource, MemoryWeightStore, SourceStorageIdentity};

#[test]
fn original_source_alias_preserves_ordinary_load_charge_and_constructor_custody() {
    let pool = test_ledger(10_000_000, 0).unwrap();
    let mut buffer = pool
        .allocate_memory_tensor_buffer(
            "weight",
            safetensors::Dtype::F32,
            &[2],
            8,
            crate::working_memory::DependencyMemoryPolicy::default(),
        )
        .unwrap();
    buffer
        .bytes_mut()
        .copy_from_slice(&[1.25f32, -3.5].map(f32::to_le_bytes).concat());
    let store = MemoryWeightStore::from_buffers([buffer]).unwrap();
    let mut inventory = Vec::new();
    assert!(
        store
            .visit_source_storage(&mut |source| {
                inventory.push((source.identity(), source.bytes()));
            })
            .unwrap()
    );
    assert_eq!(inventory.len(), 1);
    let (key, physical) = inventory.pop().unwrap();
    assert_eq!(physical, 8);
    let source_charge = pool.native_payload_bytes().unwrap();
    assert!(source_charge > physical);
    let (metadata, run) = reservation(&pool, 200, 10_000_000).into_funding().unwrap();
    let partition = run.take_native_partition(test_receipt(&run, 100)).unwrap();
    let scope = run.scope().unwrap();
    let before = balances(&pool);
    let mut missing = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    missing
        .push_source(&key, physical, Some(&key), &pool)
        .unwrap();
    assert_eq!(
        missing.publish(&scope),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(balances(&pool), before);
    drop(missing);

    let mut changed = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    assert!(matches!(
        changed.push_source(&key, physical + 1, Some(&key), &pool),
        Err(WorkingMemoryError::StorageCapacityMismatch { .. })
    ));
    let foreign = test_ledger(10_000_000, 0).unwrap();
    assert_eq!(
        changed.push_source(&key, physical, Some(&key), &foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    drop(changed);
    let mut funded = PreparedNativePublication::prepare(
        partition.clone(),
        vec![NativePublicationInput::Ordinary(key.clone(), physical)],
    );
    funded.publish(&scope).unwrap();
    let before = balances(&pool);
    let mut wrong_account = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    wrong_account
        .push_source(&key, physical, Some(&key), &pool)
        .unwrap();
    assert_eq!(
        wrong_account.publish(&scope),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(balances(&pool), before);
    drop((wrong_account, funded));

    let ordinary = pool
        .register_host_storage([(key.clone(), physical)])
        .unwrap();
    let before = balances(&pool);
    let mut alias = PreparedNativePublication::prepare_slots(partition.clone(), 2);
    for _ in 0..2 {
        alias
            .push_source(&key, physical, Some(&key), &pool)
            .unwrap();
    }
    alias.publish(&scope).unwrap();
    assert_eq!(balances(&pool), before);
    assert_eq!(alias.rows.len(), 1);
    let retained = alias.take_input(0).unwrap();
    {
        let usage = pool.0.usage.lock().unwrap();
        let registry = usage
            .storage
            .get(&std::any::TypeId::of::<SourceStorageIdentity>())
            .unwrap()
            .downcast_ref::<Registry<SourceStorageIdentity>>()
            .unwrap();
        let (_, entry) = registry.locate(&key).unwrap();
        assert!(entry.prepaid.is_none() && entry.funding.is_none());
        assert_eq!(entry.bytes, physical);
    }
    scope.certify().unwrap();
    drop((alias, metadata, run, partition, ordinary, store, key));
    assert_eq!(
        pool.native_payload_bytes().unwrap(),
        source_charge + physical
    );
    drop(retained);
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
}

#[test]
fn registered_source_alias_preserves_full_charge_and_refuses_missing_foreign_changed_or_retired_rows()
 {
    // Actual ordinary load source: encoded logical length differs from the
    // retained allocation capacity. The source owner, not a test byte guess,
    // supplies the identity and full physical registration amount.
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(&1.25_f32.to_le_bytes());
    bytes.extend_from_slice(&(-3.5_f32).to_le_bytes());
    let source = MemoryWeightStore::from_safetensors([(
        "weight".into(),
        safetensors::Dtype::F32,
        vec![2],
        bytes,
    )])
    .unwrap();
    let mut row = None;
    assert!(
        source
            .visit_source_storage(&mut |owner| {
                assert!(row.replace((owner.identity(), owner.bytes())).is_none());
            })
            .unwrap()
    );
    let (key, physical) = row.unwrap();
    assert!(physical > 8);
    let pool = test_ledger(1_000, 0).unwrap();
    let (metadata, run, partition) = account(&pool);
    let scope = run.scope().unwrap();
    let native_balance = || {
        let usage = pool.0.usage.lock().unwrap();
        let state = usage.funding.get(&scope.id).unwrap();
        (state.remaining, state.allocations)
    };
    let canonical = || {
        let usage = pool.0.usage.lock().unwrap();
        let registry = usage
            .storage
            .get(&std::any::TypeId::of::<SourceStorageIdentity>())
            .and_then(|value| value.downcast_ref::<Registry<SourceStorageIdentity>>())?;
        let (location, entry) = registry.locate(&key)?;
        assert_eq!(entry.bytes, physical);
        assert!(entry.prepaid.is_none() && entry.funding.is_none());
        Some((location, entry.owners))
    };
    let before = (balances(&pool), native_balance());
    assert_eq!(
        pool.validate_retained_source_inventory(&key, physical),
        Err(WorkingMemoryError::UnknownBound)
    );
    let mut missing = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    assert_eq!(
        missing.push_source(&key, physical, Some(&key), &pool),
        Err(WorkingMemoryError::UnknownBound)
    );
    assert_eq!((balances(&pool), native_balance()), before);
    assert!(canonical().is_none());

    let existing = pool
        .register_storage_with_gguf_sources([(
            key.clone(),
            crate::working_memory::StorageAllocation::new(physical, pool.host_placement_handle()),
        )])
        .unwrap();
    pool.validate_retained_source_inventory(&key, physical)
        .unwrap();
    let before = (balances(&pool), native_balance());
    let foreign = test_ledger(1_000, 0).unwrap();
    let foreign_before = balances(&foreign);
    let mut wrong_pool = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    assert_eq!(
        foreign.validate_retained_source_inventory(&key, physical),
        Err(WorkingMemoryError::UnknownBound)
    );
    assert_eq!(
        wrong_pool.push_source(&key, physical, Some(&key), &foreign),
        Err(WorkingMemoryError::UnknownBound)
    );
    assert_eq!(balances(&foreign), foreign_before);
    let mut changed = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    for actual in [8, physical + 1] {
        let expected = WorkingMemoryError::StorageCapacityMismatch {
            expected_bytes: physical,
            actual_bytes: actual,
        };
        assert_eq!(
            pool.validate_retained_source_inventory(&key, actual),
            Err(expected.clone())
        );
        assert_eq!(
            changed.push_source(&key, actual, Some(&key), &pool),
            Err(expected)
        );
    }
    assert_eq!((balances(&pool), native_balance()), before);

    // A successful preflight is not a source pin. Retiring the sole canonical
    // registration before publication must not let the transaction recreate it.
    let mut retired = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    retired
        .push_source(&key, physical, Some(&key), &pool)
        .unwrap();
    drop(existing);
    let after_retirement = (balances(&pool), native_balance());
    assert!(canonical().is_none());
    assert_eq!(
        retired.publish(&scope),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert!(retired.take_input(0).is_none());
    assert_eq!((balances(&pool), native_balance()), after_retirement);
    assert!(canonical().is_none());

    let existing = pool
        .register_storage_with_gguf_sources([(
            key.clone(),
            crate::working_memory::StorageAllocation::new(physical, pool.host_placement_handle()),
        )])
        .unwrap();
    let before = (balances(&pool), native_balance());
    let (location, owners) = canonical().unwrap();
    let mut alias = PreparedNativePublication::prepare_slots(partition.clone(), 2);
    for _ in 0..2 {
        alias
            .push_source(&key, physical, Some(&key), &pool)
            .unwrap();
    }
    alias.publish(&scope).unwrap();
    assert_eq!(
        alias.rows.len(),
        1,
        "duplicate source aliases share one canonical output"
    );
    assert_eq!(
        canonical(),
        Some((location, owners + 1)),
        "no new physical row"
    );
    assert_eq!(
        (balances(&pool), native_balance()),
        before,
        "no Q or physical charge consumed"
    );
    let retained = alias.take_input(0).unwrap();
    assert_eq!(retained.bytes(), Some(physical));
    drop(existing);
    assert_eq!(
        canonical(),
        Some((location, owners)),
        "published alias pins the original paid row"
    );
    assert_eq!((balances(&pool), native_balance()), before);
    scope.certify().unwrap();
    drop((
        missing, wrong_pool, changed, retired, alias, partition, metadata, run,
    ));
    assert!(pool.native_payload_bytes().unwrap() >= physical);
    drop(retained);
    assert_eq!(pool.native_payload_bytes().unwrap(), 0);
}
