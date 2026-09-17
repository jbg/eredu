use super::super::read_destination;
use super::*;

mod original;

fn add_named(builder: GgufWeightStoreBuilder, path: &Path, prefix: &str) -> GgufWeightStoreBuilder {
    let checkpoint = Checkpoint::open(path).unwrap();
    let plan = test_plan(&checkpoint);
    let mapping = checkpoint
        .translated_outputs(|name| format!("{prefix}{name}"))
        .unwrap();
    builder.add_checkpoint(checkpoint, &plan, &mapping).unwrap()
}
fn fixture(path: &Path) {
    write_tensor(
        path,
        "matrix.weight",
        &[2],
        GgmlType::F32,
        &[1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat(),
    );
}
fn request(key: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
fn snapshot(source: &GgufWeightStore) -> (String, u64, Vec<u64>, Vec<Option<PathBuf>>) {
    let diagnostics = format!("{:?}", source.diagnostics().unwrap());
    let cache = source.inner.readers.lock().unwrap();
    (
        diagnostics,
        cache.tick,
        cache.last_used.clone(),
        cache
            .materializers
            .iter()
            .map(|m| m.open_shard_path().map(Path::to_path_buf))
            .collect(),
    )
}
fn old_read(lease: &ReferenceLease) -> Result<ConvertedCheckpointTensor, read_destination::Cause> {
    let converted = lease
        .cache
        .lock()
        .unwrap()
        .old_g4_materialize_with_destination(
            lease.entry.checkpoint,
            &lease.entry.physical_name,
            lease.identity.selection.as_ref(),
            lease.store.max_cached_readers,
            &lease.entry.metadata.name,
            None,
            None,
            None,
        )?;
    lease
        .store
        .statistics
        .physical_reads
        .fetch_add(1, Ordering::Relaxed);
    lease
        .store
        .statistics
        .physical_read_bytes
        .fetch_add(lease.proof.length_bytes, Ordering::Relaxed);
    Ok(converted)
}
fn error_text(cause: read_destination::Cause) -> String {
    match cause {
        read_destination::Cause::Store(error) => error.to_string(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn shared_cache_preserves_saturated_lru_and_unknown_lookup_order() {
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..3)
        .map(|i| dir.path().join(format!("source{i}.gguf")))
        .collect();
    for path in &paths {
        fixture(path);
    }
    for maximum in [1, 2] {
        let make = || {
            let mut builder = GgufWeightStore::builder()
                .max_cached_readers(maximum)
                .unwrap();
            for (i, path) in paths.iter().enumerate() {
                builder = add_named(builder, path, &format!("{i}."));
            }
            builder.build().unwrap()
        };
        let old = ReferenceStore::new(make());
        let source = make();
        {
            let mut cache = source.inner.readers.lock().unwrap();
            cache.tick = u64::MAX;
            cache.last_used.fill(u64::MAX);
        }
        {
            let mut cache = old.cache.lock().unwrap();
            cache.tick = u64::MAX;
            cache.last_used.fill(u64::MAX);
        }
        for (step, index) in [0, 1, 2, 2, 0, 1].into_iter().enumerate() {
            let req = request(&format!("{index}.matrix.weight"));
            let expected = old_read(&old.acquire(req.clone()).unwrap()).unwrap();
            let actual = if step % 2 == 0 {
                source
                    .clone()
                    .acquire(req)
                    .unwrap()
                    .prepare_portable_read()
                    .unwrap()
                    .prepare_conversion()
                    .unwrap()
                    .prepare_result_metadata()
                    .unwrap()
                    .materialize()
                    .unwrap()
            } else {
                source.acquire(req).unwrap().materialize_portable().unwrap()
            };
            assert_eq!(actual, expected);
            assert_eq!(snapshot(&source), old.snapshot());
            if maximum == 2 && step == 2 {
                let cache = source.inner.readers.lock().unwrap();
                assert!(cache.materializers[0].open_shard_path().is_none());
                assert!(cache.materializers[1].open_shard_path().is_some());
            }
        }
        for (checkpoint, name) in [(99, "matrix.weight"), (0, "missing")] {
            let before = snapshot(&source);
            let expected = old
                .cache
                .lock()
                .unwrap()
                .old_g4_materialize_with_destination(
                    checkpoint, name, None, maximum, "logical", None, None, None,
                )
                .unwrap_err();
            let actual = source
                .inner
                .readers
                .lock()
                .unwrap()
                .materialize_with_destination(
                    checkpoint, name, None, maximum, "logical", None, None, None,
                )
                .unwrap_err();
            assert_eq!(error_text(actual), error_text(expected));
            assert_eq!(snapshot(&source), before);
            assert_eq!(snapshot(&source), old.snapshot());
        }
    }
}

#[test]
fn first_successful_equivalent_path_spelling_survives_failure_and_repeat() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("same.gguf");
    let second = dir.path().join(".").join("same.gguf");
    assert_eq!(first.as_path(), second.as_path());
    assert_ne!(first.as_os_str(), second.as_os_str());
    fixture(&first);
    let make = || {
        add_named(
            add_named(
                GgufWeightStore::builder().max_cached_readers(1).unwrap(),
                &first,
                "a.",
            ),
            &second,
            "b.",
        )
        .build()
        .unwrap()
    };
    let old = ReferenceStore::new(make());
    let source = make();
    let old_lease = old.acquire(request("a.matrix.weight")).unwrap();
    let prepared = source
        .acquire(request("a.matrix.weight"))
        .unwrap()
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap()
        .prepare_result_metadata()
        .unwrap();
    std::fs::remove_file(&first).unwrap();
    let expected = error_text(old_read(&old_lease).unwrap_err());
    let failure = prepared.materialize().unwrap_err();
    assert_eq!(failure.store_error().unwrap().to_string(), expected);
    assert_eq!(failure.raw_bytes(), &[0; 8]);
    assert!(failure.lease().store.same(&source.inner));
    assert_eq!(snapshot(&source), old.snapshot());
    assert!(source.inner.readers.lock().unwrap().touched.is_empty());
    fixture(&first);
    for key in ["b.matrix.weight", "a.matrix.weight", "b.matrix.weight"] {
        let expected = old_read(&old.acquire(request(key)).unwrap()).unwrap();
        let actual = source
            .acquire(request(key))
            .unwrap()
            .materialize_portable()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(snapshot(&source), old.snapshot());
        let diagnostics = source.diagnostics().unwrap();
        assert_eq!(diagnostics.touched_shard_paths.len(), 1);
        assert_eq!(
            diagnostics.touched_shard_paths[0].as_os_str(),
            second.as_os_str()
        );
        assert_eq!(
            diagnostics.payload_shard_paths[0].as_os_str(),
            second.as_os_str()
        );
    }
}

#[test]
fn builder_moves_catalog_paths_and_lease_keeps_lazy_shared_source_alive() {
    assert!(GgufWeightStore::builder().build().is_err());
    let dir = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..2)
        .map(|i| dir.path().join(format!("moved{i}.gguf")))
        .collect();
    let mut builder = GgufWeightStore::builder();
    for (i, path) in paths.iter().enumerate() {
        fixture(path);
        builder = add_named(builder, path, &format!("{i}."));
    }
    let addresses: Vec<_> = builder
        .checkpoints
        .iter()
        .map(|c| c.shards()[0].path().as_os_str().as_encoded_bytes().as_ptr() as usize)
        .collect();
    let source = builder.build().unwrap();
    {
        let cache = source.inner.readers.lock().unwrap();
        assert_eq!(cache.materializers.len(), 2);
        for (materializer, address) in cache.materializers.iter().zip(addresses) {
            assert_eq!(
                materializer
                    .shard_path_for_tensor("matrix.weight")
                    .unwrap()
                    .as_os_str()
                    .as_encoded_bytes()
                    .as_ptr() as usize,
                address
            );
            assert!(materializer.open_shard_path().is_none());
        }
    }
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    let weak = source.inner.ordinary_weak();
    let lease = source.acquire(request("1.matrix.weight")).unwrap();
    drop(source);
    assert!(weak.upgrade().is_some());
    let tensor = lease.materialize_portable().unwrap();
    drop(lease);
    assert!(weak.upgrade().is_none());
    let ConvertedTensor::Dense(dense) = tensor.converted() else {
        panic!("dense")
    };
    assert_eq!(
        dense.data,
        [1.25_f32.to_ne_bytes(), (-3.5_f32).to_ne_bytes()].concat()
    );
}
