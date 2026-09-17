use super::*;
use eredu_checkpoint::store::{CheckpointSource, MemoryWeightStore, SourceStorageIdentity};

#[test]
fn registered_source_alias_preserves_full_charge_and_refuses_missing_foreign_changed_or_retired_rows(
) {
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
    assert!(source
        .visit_source_storage(&mut |owner| {
            assert!(row.replace((owner.identity(), owner.bytes())).is_none());
        })
        .unwrap());
    let (key, physical) = row.unwrap();
    assert!(physical > 8);
    let pool = WorkingMemoryPool::new(1_000, 0).unwrap();
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
        .register_storage_with_gguf_sources([(key.clone(), physical)])
        .unwrap();
    pool.validate_retained_source_inventory(&key, physical)
        .unwrap();
    let before = (balances(&pool), native_balance());
    let foreign = WorkingMemoryPool::new(1_000, 0).unwrap();
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
        .register_storage_with_gguf_sources([(key.clone(), physical)])
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
    assert_eq!(retained.bytes(), physical);
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
    assert!(pool.used_bytes().unwrap() >= physical);
    drop(retained);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
