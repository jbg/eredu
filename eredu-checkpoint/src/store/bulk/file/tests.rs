use super::*;
use safetensors::tensor::{TensorView, serialize_to_file};

pub(super) fn fixture() -> (tempfile::TempDir, SafetensorsWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    for (name, rows) in [
        (
            "a.safetensors",
            vec![("a", vec![13u8, 14]), ("b", vec![21, 22, 23])],
        ),
        ("b.safetensors", vec![("c", vec![3u8, 7, 11])]),
    ] {
        serialize_to_file(
            rows.iter().map(|(key, bytes)| {
                (
                    *key,
                    TensorView::new(Dtype::U8, vec![bytes.len()], bytes).unwrap(),
                )
            }),
            None,
            &directory.path().join(name),
        )
        .unwrap();
    }
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"a.safetensors","b":"a.safetensors","c":"b.safetensors"}}"#,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(directory.path()).unwrap();
    (directory, store)
}
pub(super) fn warm(store: &SafetensorsWeightStore) {
    for key in ["a", "c"] {
        WeightStore::metadata(store, key).unwrap();
    }
}

#[test]
fn file_plan_refuses_unprepared_headers_without_opening_or_touching_them() {
    let (_directory, store) = fixture();
    let keys = ["c".into(), "a".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::new(&store, &keys),
        Err(SafetensorsEncodedReadPlanError::HeaderUnavailable { index: 0 })
    ));
    for path in store.shards.payload_paths() {
        assert_eq!(
            store
                .shards
                .admission(path)
                .header_reads
                .load(Ordering::Relaxed),
            0
        );
    }
    let d = store.diagnostics().unwrap();
    assert!(d.touched_shard_paths.is_empty());
    assert!(d.payload_shard_paths.is_empty());
    assert_eq!(d.physical_reads, 0);
    warm(&store);
    let keys = ["a".into(), "missing".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::new(&store, &keys),
        Err(SafetensorsEncodedReadPlanError::UnknownTensor { index: 1 })
    ));
}

#[test]
fn file_constructor_preserves_repeated_order_sources_and_custody_after_store_retirement() {
    let (_directory, store) = fixture();
    warm(&store);
    let keys = ["c".into(), "a".into(), "b".into(), "a".into()];
    let file = Arc::downgrade(&store.shards.admission(&store.catalog["a"].shard).file);
    let custody = Arc::new(());
    let held = Arc::downgrade(&custody);
    let read = SafetensorsEncodedReadPlan::new(&store, &keys)
        .unwrap()
        .construct(custody)
        .unwrap();
    assert_eq!(read.byte_len(), 10);
    assert!(read.read_layout().unwrap().required_bytes() > 0);
    assert_eq!(
        read.tensors()
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        ["c", "a", "b", "a"]
    );
    assert_eq!(store.diagnostics().unwrap().physical_reads, 0);
    drop((store, keys));
    assert!(file.upgrade().is_some());
    let mut wrong = [99; 9];
    assert!(matches!(
        read.read_into(&mut wrong).unwrap_err().cause,
        EncodedReadFailureCause::DestinationLengths
    ));
    assert_eq!(wrong, [99; 9]);
    let mut output = [0; 10];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, [3, 7, 11, 13, 14, 21, 22, 23, 13, 14]);
    assert!(held.upgrade().is_some());
    drop(read);
    assert!(file.upgrade().is_none());
    assert!(held.upgrade().is_none());
}

#[test]
fn metadata_and_payload_diagnostics_use_independent_flags_in_fixed_source_storage() {
    let (_directory, store) = fixture();
    let path = store.catalog["a"].shard.clone();
    let (rows, capacity) = {
        let cache = store.cache.lock().unwrap();
        (cache.paths.0.as_ptr(), cache.paths.0.capacity())
    };
    let keys = ["a".into(), "b".into()];
    let ordinary = prepare(&store, &keys).unwrap();
    let before = store.diagnostics().unwrap();
    assert_eq!(before.touched_shard_paths, [path.clone()]);
    assert!(before.payload_shard_paths.is_empty());
    assert_eq!(before.physical_reads, 0);
    let mut output = [0; 5];
    ordinary.read_into(&mut output).unwrap();
    assert_eq!(output, [13, 14, 21, 22, 23]);
    let after = store.diagnostics().unwrap();
    assert_eq!(after.touched_shard_paths, [path.clone()]);
    assert_eq!(after.payload_shard_paths, [path]);
    assert_eq!(after.physical_reads, 1);
    assert_eq!(after.physical_read_bytes, 5);
    assert_eq!(after.currently_cached_shards, 0);
    let mut cache = store.cache.lock().unwrap();
    assert!(!cache.paths.mark_touched(Path::new("foreign")));
    assert!(!cache.paths.mark_payload(Path::new("foreign")));
    assert_eq!(cache.paths.0.as_ptr(), rows);
    assert_eq!(cache.paths.0.capacity(), capacity);
}

#[test]
fn poisoned_diagnostics_preserve_constructed_metadata_prefix_and_custody() {
    let (_directory, store) = fixture();
    warm(&store);
    let keys = ["a".into(), "b".into()];
    let plan = SafetensorsEncodedReadPlan::new(&store, &keys).unwrap();
    let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _cache = store.cache.lock().unwrap();
        panic!("poison this fixture's diagnostic lock");
    }));
    assert!(poison.is_err());
    let custody = Arc::new(());
    let held = Arc::downgrade(&custody);
    let error = plan.construct(custody).unwrap_err();
    assert!(matches!(
        error.cause(),
        SafetensorsEncodedReadBuildCause::CachePoisoned
    ));
    assert_eq!(error.completed_tensors(), 2);
    drop((store, keys));
    assert!(held.upgrade().is_some());
    drop(error);
    assert!(held.upgrade().is_none());
}

#[test]
fn completed_file_read_keeps_original_identity_after_path_replacement() {
    let (directory, store) = fixture();
    warm(&store);
    let keys = ["a".into()];
    let read = SafetensorsEncodedReadPlan::new(&store, &keys)
        .unwrap()
        .construct(())
        .unwrap();
    let path = store.catalog["a"].shard.clone();
    drop(store);
    std::fs::rename(&path, directory.path().join("old.safetensors")).unwrap();
    std::fs::copy(directory.path().join("old.safetensors"), &path).unwrap();
    let mut output = [99; 2];
    assert!(matches!(
        read.read_into(&mut output).unwrap_err().cause,
        EncodedReadFailureCause::Changed
    ));
    assert_eq!(output, [99; 2]);
}

#[test]
fn failed_header_inspection_lends_the_original_error_without_retrying() {
    let (_directory, store) = fixture();
    let path = store.catalog["a"].shard.clone();
    std::fs::remove_file(&path).unwrap();
    assert!(WeightStore::metadata(&store, "a").is_err());
    let admission = store.shards.admission(&path);
    let original = admission.header.get().unwrap().as_ref().unwrap_err();
    let keys = ["a".into()];
    let Err(error) = SafetensorsEncodedReadPlan::new(&store, &keys) else {
        panic!("expected the retained header failure")
    };
    assert!(matches!(
        error,
        SafetensorsEncodedReadPlanError::Header { index: 0, .. }
    ));
    let cause = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<StoreError>()
        .unwrap();
    assert!(std::ptr::eq(cause, original));
    assert_eq!(admission.header_reads.load(Ordering::Relaxed), 1);
    assert!(store.diagnostics().unwrap().touched_shard_paths.is_empty());
}
