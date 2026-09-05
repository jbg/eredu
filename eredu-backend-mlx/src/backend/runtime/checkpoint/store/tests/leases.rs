#[test]
fn maps_only_acquired_shards_and_reuses_one_mapping() {
    let dir = tempfile::tempdir().unwrap();
    write_two_i32(&dir.path().join("local.safetensors"));
    write_i32(
        &dir.path().join("other.safetensors"),
        "other",
        &[5, 6],
        vec![2],
    );
    write_index(
        dir.path(),
        &[
            ("a_tensor", "local.safetensors"),
            ("z_tensor", "local.safetensors"),
            ("other", "other.safetensors"),
        ],
    );
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    let first = store.acquire("a_tensor", TensorSelection::Full).unwrap();
    let second = store.acquire("z_tensor", TensorSelection::Full).unwrap();
    assert_eq!(first.backing_shard(), second.backing_shard());
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert_eq!(diagnostics.cache_misses, 1);
    assert_eq!(diagnostics.cache_hits, 1);
    assert_eq!(diagnostics.touched_shard_paths.len(), 1);
}

#[test]
fn enforces_capacity_until_leases_drop_then_evicts_lru() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(&dir.path().join("one.safetensors"), "one", &[1], vec![1]);
    write_i32(&dir.path().join("two.safetensors"), "two", &[2], vec![1]);
    write_index(
        dir.path(),
        &[("one", "one.safetensors"), ("two", "two.safetensors")],
    );
    let store = SafetensorsWeightStore::open_with_max_cached_shards(dir.path(), 1).unwrap();
    let one = store.acquire("one", TensorSelection::Full).unwrap();
    let error = store.acquire("two", TensorSelection::Full).unwrap_err();
    assert!(matches!(
        error,
        CheckpointMaterializationError::Store(StoreError::CapacityExhausted { maximum: 1, .. })
    ));
    assert_eq!(one.metadata().logical_shape, [1]);
    let stream = cpu_stream();
    let pinned_value = one
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(pinned_value.evaluated().unwrap().as_slice::<i32>(), &[1]);
    drop(one);

    let two = store.acquire("two", TensorSelection::Full).unwrap();
    assert_eq!(two.metadata().logical_shape, [1]);
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert_eq!(diagnostics.evictions, 1);
    assert_eq!(diagnostics.touched_shard_paths.len(), 2);
}

#[test]
fn returned_array_survives_lease_and_store_drop() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(
        &dir.path().join("model.safetensors"),
        "weight",
        &[7, 8, 9],
        vec![3],
    );
    let stream = cpu_stream();
    let value = {
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let lease = store.acquire("weight", TensorSelection::Full).unwrap();
        lease
            .materialize(&stream, &stream)
            .unwrap()
            .synchronize()
            .unwrap()
    };
    assert_eq!(value.evaluated().unwrap().as_slice::<i32>(), &[7, 8, 9]);
}
