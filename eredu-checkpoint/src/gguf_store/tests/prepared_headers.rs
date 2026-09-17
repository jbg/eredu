use super::*;
use eredu_gguf::MetadataValue;

fn write(path: &Path, label: &str, value: f32) {
    let metadata = BTreeMap::from([("label".into(), MetadataValue::String(label.into()))]);
    Writer::default()
        .write(
            File::create(path).unwrap(),
            &metadata,
            &[TensorInput {
                name: "weight",
                dimensions: &[2],
                ggml_type: GgmlType::F32,
                data: &[value.to_le_bytes(), (-3.5f32).to_le_bytes()].concat(),
            }],
        )
        .unwrap();
}
fn source(paths: &[std::path::PathBuf], prepared: bool) -> GgufWeightStore {
    let mut builder = GgufWeightStore::builder().max_cached_readers(1).unwrap();
    for (i, path) in paths.iter().enumerate() {
        let c = if prepared {
            Checkpoint::open_with_prepared_headers(path)
        } else {
            Checkpoint::open(path)
        }
        .unwrap();
        let plan = test_plan(&c);
        let mapping = c.translated_outputs(|n| format!("{i}.{n}")).unwrap();
        builder = builder.add_checkpoint(c, &plan, &mapping).unwrap();
    }
    builder.build().unwrap()
}
fn request(i: usize) -> TensorReadRequest {
    TensorReadRequest {
        key: format!("{i}.weight"),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
#[test]
fn prepared_header_source_shares_cache_and_preserves_typed_change_until_failure_drop() {
    let dir = tempfile::tempdir().unwrap();
    let paths = [dir.path().join("a.gguf"), dir.path().join("b.gguf")];
    for path in &paths {
        write(path, "old", 1.25);
    }
    let prepared = source(&paths, true);
    let ordinary = source(&paths, false);
    assert!(prepared.inner.prepared_header_storage_bytes > 0);
    assert_eq!(ordinary.inner.prepared_header_storage_bytes, 0);
    assert_eq!(
        prepared.source_storage_bytes().unwrap() - ordinary.source_storage_bytes().unwrap(),
        prepared.inner.prepared_header_storage_bytes
    );
    for i in [0, 0, 1] {
        assert_eq!(
            prepared
                .acquire(request(i))
                .unwrap()
                .materialize_portable()
                .unwrap(),
            ordinary
                .acquire(request(i))
                .unwrap()
                .materialize_portable()
                .unwrap()
        );
    }
    // Both caches evicted a; its same-geometry metadata change succeeds ordinarily.
    write(&paths[0], "new", 1.25);
    ordinary
        .acquire(request(0))
        .unwrap()
        .materialize_portable()
        .unwrap();
    let weak = prepared.inner.ordinary_weak();
    let failure = prepared
        .acquire(request(0))
        .unwrap()
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap()
        .prepare_result_metadata()
        .unwrap()
        .materialize()
        .unwrap_err();
    match failure.store_error().unwrap() {
        StoreError::GgufPreparedHeaderChanged { key, source } => {
            assert_eq!(key, "0.weight");
            assert!(source.prepared_header_change().is_some());
            assert!(matches!(source.as_ref(), eredu_gguf::Error::Shard { .. }));
        }
        other => panic!("{other:?}"),
    }
    let cache = prepared.inner.readers.lock().unwrap();
    assert_eq!(cache.misses, 3);
    assert_eq!(cache.hits, 1);
    assert!(cache
        .materializers
        .iter()
        .all(|m| m.open_shard_path().is_none()));
    drop(cache);
    drop(prepared);
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert!(weak.upgrade().is_none());
}

#[test]
fn prepared_header_restore_retries_and_payload_changes_keep_real_nonzero_reads() {
    let dir = tempfile::tempdir().unwrap();
    let paths = [dir.path().join("a.gguf"), dir.path().join("b.gguf")];
    for path in &paths {
        write(path, "old", 1.25);
    }
    let prepared = source(&paths, true);
    write(&paths[0], "new", 1.25);
    assert!(matches!(
        prepared.acquire(request(0)).unwrap().materialize_portable(),
        Err(StoreError::GgufPreparedHeaderChanged { .. })
    ));
    write(&paths[0], "old", 19.0);
    let actual = prepared
        .acquire(request(0))
        .unwrap()
        .materialize_portable()
        .unwrap();
    let reference = source(&paths, false)
        .acquire(request(0))
        .unwrap()
        .materialize_portable()
        .unwrap();
    assert_eq!(actual, reference);
    prepared
        .acquire(request(1))
        .unwrap()
        .materialize_portable()
        .unwrap();
    let file = File::options().write(true).open(&paths[0]).unwrap();
    let len = file.metadata().unwrap().len();
    file.set_len(len - 1).unwrap();
    drop(file);
    assert!(matches!(
        prepared.acquire(request(0)).unwrap().materialize_portable(),
        Err(StoreError::Gguf { .. })
    ));
}

#[test]
fn prepared_header_cloned_checkpoint_aliases_charge_one_physical_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.gguf");
    write(&path, "old", 1.25);
    let checkpoint = Checkpoint::open_with_prepared_headers(&path).unwrap();
    let header = checkpoint.shards()[0].prepared_header().unwrap();
    let expected = header.retained_payload_bytes().unwrap() as u64;
    let clone = checkpoint.clone();
    assert!(std::ptr::eq(
        header,
        clone.shards()[0].prepared_header().unwrap()
    ));
    let mut builder = GgufWeightStore::builder().max_cached_readers(1).unwrap();
    for (i, c) in [checkpoint, clone].into_iter().enumerate() {
        let plan = test_plan(&c);
        let mapping = c.translated_outputs(|n| format!("{i}.{n}")).unwrap();
        builder = builder.add_checkpoint(c, &plan, &mapping).unwrap();
    }
    let source = builder.build().unwrap();
    let scratch: u64 = source
        .inner
        .readers
        .lock()
        .unwrap()
        .materializers
        .iter()
        .map(|m| m.prepared_header_scratch_layout().unwrap().size() as u64)
        .sum();
    assert!(scratch > 0);
    assert_eq!(
        source.inner.prepared_header_storage_bytes,
        expected + scratch
    );
    let first = source
        .acquire(request(0))
        .unwrap()
        .materialize_portable()
        .unwrap();
    let second = source
        .acquire(request(1))
        .unwrap()
        .materialize_portable()
        .unwrap();
    assert_eq!(first, second);
    let cache = source.inner.readers.lock().unwrap();
    let a = cache.materializers[0].shards()[0]
        .prepared_header()
        .unwrap();
    let b = cache.materializers[1].shards()[0]
        .prepared_header()
        .unwrap();
    assert!(std::ptr::eq(a, b));
}
