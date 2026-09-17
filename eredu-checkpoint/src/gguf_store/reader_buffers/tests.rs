use super::super::tests::test_plan;
use super::*;
use eredu_gguf::MetadataValue;
use eredu_gguf::{GgmlType, TensorInput, Writer};
use std::fs::File;

fn write(path: &Path, label: &str) {
    Writer::default()
        .write(
            File::create(path).unwrap(),
            &BTreeMap::from([("label".into(), MetadataValue::String(label.into()))]),
            &[TensorInput {
                name: "weight",
                dimensions: &[2],
                ggml_type: GgmlType::F32,
                data: &[1.25f32.to_le_bytes(), (-3.5f32).to_le_bytes()].concat(),
            }],
        )
        .unwrap();
}
fn builder(paths: &[PathBuf], maximum: usize, captured: bool) -> GgufWeightStoreBuilder {
    let mut builder = GgufWeightStore::builder()
        .max_cached_readers(maximum)
        .unwrap();
    for (i, path) in paths.iter().enumerate() {
        let checkpoint = if captured {
            Checkpoint::open_with_prepared_headers(path)
        } else {
            Checkpoint::open(path)
        }
        .unwrap();
        let plan = test_plan(&checkpoint);
        let mapping = checkpoint
            .translated_outputs(|name| format!("{i}.{name}"))
            .unwrap();
        builder = builder.add_checkpoint(checkpoint, &plan, &mapping).unwrap();
    }
    builder
}
fn request(i: usize) -> TensorReadRequest {
    TensorReadRequest {
        key: format!("{i}.weight"),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
fn cache_inventory(source: &GgufWeightStore) -> (usize, usize, u64) {
    let cache = source.inner.readers.lock().unwrap();
    let bank = cache.buffers.as_ref().unwrap();
    let active = cache
        .materializers
        .iter()
        .filter(|m| m.open_shard_path().is_some())
        .count();
    assert_eq!(bank.free.len() + active, bank.count());
    (
        bank.free.as_ptr() as usize,
        bank.free.capacity(),
        bank.storage_bytes().unwrap(),
    )
}
#[test]
fn reader_bank_inventory_is_constant_across_nonzero_cache_hits_evictions_and_retries() {
    let dir = tempfile::tempdir().unwrap();
    let paths = (0..3)
        .map(|i| dir.path().join(format!("{i}.gguf")))
        .collect::<Vec<_>>();
    for path in &paths {
        write(path, "old");
    }
    for captured in [false, true] {
        for maximum in [1, 2] {
            let ordinary = builder(&paths, maximum, captured).build().unwrap();
            let build = builder(&paths, maximum, captured);
            let buffers = build.prepare_reader_buffers().unwrap();
            assert_eq!(buffers.count(), maximum);
            let bytes = buffers.storage_bytes().unwrap();
            assert!(
                bytes
                    >= maximum as u64
                        * (8192 + std::mem::size_of::<eredu_gguf::ReaderBuffer>() as u64)
            );
            let source = build.build_with_reader_buffers(buffers).unwrap();
            let inventory = cache_inventory(&source);
            let controls = source.prepared_materializer_storage_bytes().unwrap();
            assert!(ordinary.prepared_materializer_storage_bytes().is_none());
            {
                let cache = source.inner.readers.lock().unwrap();
                let actual =
                    std::alloc::Layout::array::<TensorMaterializer>(cache.materializers.capacity())
                        .unwrap()
                        .size()
                        + std::alloc::Layout::array::<u64>(cache.last_used.capacity())
                            .unwrap()
                            .size()
                        + cache
                            .materializers
                            .iter()
                            .map(|m| m.prepared_index_layout().unwrap().size())
                            .sum::<usize>();
                assert_eq!(controls, actual as u64);
                assert!(controls > 0);
            }
            assert_eq!(source.prepared_reader_storage_bytes(), Some(bytes));
            assert_eq!(
                source.source_storage_bytes().unwrap() - ordinary.source_storage_bytes().unwrap(),
                bytes - maximum as u64 * 8192 + controls
            );
            for i in [0, 0, 1, 2, 0, 1, 2, 2, 0] {
                assert_eq!(
                    source
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
                assert_eq!(
                    source.diagnostics().unwrap(),
                    ordinary.diagnostics().unwrap()
                );
                assert_eq!(cache_inventory(&source), inventory);
                assert_eq!(source.reader_storage_bytes().unwrap(), bytes);
                assert_eq!(source.prepared_materializer_storage_bytes(), Some(controls));
            }
            // Closing/opening failures retain the same cache effects and return the
            // newly attempted buffer. Successful retry reuses the fixed bank.
            std::fs::remove_file(&paths[1]).unwrap();
            assert_eq!(
                source
                    .acquire(request(1))
                    .unwrap()
                    .materialize_portable()
                    .unwrap_err()
                    .to_string(),
                ordinary
                    .acquire(request(1))
                    .unwrap()
                    .materialize_portable()
                    .unwrap_err()
                    .to_string()
            );
            assert_eq!(
                source.diagnostics().unwrap(),
                ordinary.diagnostics().unwrap()
            );
            assert_eq!(cache_inventory(&source), inventory);
            write(&paths[1], "old");
            assert_eq!(
                source
                    .acquire(request(1))
                    .unwrap()
                    .materialize_portable()
                    .unwrap(),
                ordinary
                    .acquire(request(1))
                    .unwrap()
                    .materialize_portable()
                    .unwrap()
            );
            assert_eq!(cache_inventory(&source), inventory);
        }
    }
}
#[test]
fn prepared_header_failure_retains_real_source_bank_until_last_lease_owner() {
    let dir = tempfile::tempdir().unwrap();
    let paths = [dir.path().join("a.gguf"), dir.path().join("b.gguf")];
    for path in &paths {
        write(path, "old");
    }
    let build = builder(&paths, 1, true);
    let buffers = build.prepare_reader_buffers().unwrap();
    let source = build.build_with_reader_buffers(buffers).unwrap();
    let inventory = cache_inventory(&source);
    let retained = source.acquire(request(0)).unwrap();
    retained.materialize_portable().unwrap();
    source
        .acquire(request(1))
        .unwrap()
        .materialize_portable()
        .unwrap();
    write(&paths[0], "new");
    let failure = source
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
    assert!(matches!(
        failure.store_error(),
        Some(StoreError::GgufPreparedHeaderChanged { .. })
    ));
    assert_eq!(cache_inventory(&source), inventory);
    let weak = source.inner.ordinary_weak();
    drop(source);
    drop(retained);
    let inner = weak.upgrade().expect("actual failed lease retains source");
    assert_eq!(inner.prepared_reader_storage_bytes, Some(inventory.2));
    drop(inner);
    drop(failure);
    assert!(weak.upgrade().is_none());
}
#[test]
fn source_reader_bank_wrong_count_and_empty_catalog_return_the_exact_bank() {
    let dir = tempfile::tempdir().unwrap();
    let paths = [dir.path().join("a.gguf"), dir.path().join("b.gguf")];
    for path in &paths {
        write(path, "old");
    }
    let build = builder(&paths, 2, false);
    let buffers = build.prepare_reader_buffers().unwrap();
    let pointer = buffers.free.as_ptr();
    let capacity = buffers.free.capacity();
    let bytes = buffers.storage_bytes();
    let (error, buffers) = builder(&paths, 1, false)
        .build_with_reader_buffers(buffers)
        .unwrap_err();
    assert!(matches!(
        error,
        StoreError::GgufPreparedReaderStorage { .. }
    ));
    assert_eq!(buffers.free.as_ptr(), pointer);
    assert_eq!(buffers.free.capacity(), capacity);
    assert_eq!(buffers.storage_bytes(), bytes);
    let empty = GgufWeightStore::builder();
    let buffers = empty.prepare_reader_buffers().unwrap();
    let (error, buffers) = empty.build_with_reader_buffers(buffers).unwrap_err();
    assert!(matches!(error, StoreError::Gguf { .. }));
    assert_eq!(buffers.count(), 0);
    assert_eq!(buffers.storage_bytes(), Some(0));
}

#[test]
fn cold_source_convenience_selects_actual_inventory_and_preserves_nonzero_reads() {
    let dir = tempfile::tempdir().unwrap();
    let paths = [dir.path().join("a.gguf"), dir.path().join("b.gguf")];
    for path in &paths {
        write(path, "old");
    }
    for maximum in [1, 8] {
        let source = builder(&paths, maximum, true)
            .build_with_prepared_reader_buffers()
            .unwrap();
        let ordinary = builder(&paths, maximum, true).build().unwrap();
        assert_eq!(
            source
                .inner
                .readers
                .lock()
                .unwrap()
                .buffers
                .as_ref()
                .unwrap()
                .count(),
            maximum.min(2)
        );
        assert!(source.prepared_reader_storage_bytes().unwrap() > 0);
        assert!(source.prepared_materializer_storage_bytes().unwrap() > 0);
        assert_eq!(ordinary.prepared_reader_storage_bytes(), None);
        assert_eq!(ordinary.prepared_materializer_storage_bytes(), None);
        let inventory = cache_inventory(&source);
        for i in [0, 1, 0, 1] {
            assert_eq!(
                source
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
            assert_eq!(
                source.diagnostics().unwrap(),
                ordinary.diagnostics().unwrap()
            );
            assert_eq!(cache_inventory(&source), inventory);
        }
    }
    let checkpoint = Checkpoint::open_with_prepared_headers(&paths[0]).unwrap();
    let plan = test_plan(&checkpoint);
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    let ordinary = open_prepared_gguf_source(checkpoint.clone(), &plan, &mapping, 1).unwrap();
    let source =
        open_prepared_gguf_source_with_reader_buffers(checkpoint, &plan, &mapping, 1).unwrap();
    assert!(source.prepared_reader_storage_bytes().is_some());
    assert!(source.prepared_materializer_storage_bytes().is_some());
    assert_eq!(ordinary.prepared_reader_storage_bytes(), None);
    let request = TensorReadRequest {
        key: "weight".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    assert_eq!(
        source
            .acquire(request.clone())
            .unwrap()
            .materialize_portable()
            .unwrap(),
        ordinary
            .acquire(request)
            .unwrap()
            .materialize_portable()
            .unwrap()
    );
}

#[test]
fn cold_source_validation_retains_untouched_builder_before_reader_preparation() {
    use std::error::Error;
    let mut empty = GgufWeightStore::builder();
    empty.checkpoints.reserve_exact(3);
    let backing = empty.checkpoints.as_ptr();
    let capacity = empty.checkpoints.capacity();
    let expected = GgufWeightStore::builder().build().unwrap_err();
    let error = empty.build_with_prepared_reader_buffers().unwrap_err();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<StoreError>()
            .unwrap()
            .to_string(),
        expected.to_string()
    );
    let (cause, builder, bank) = error.into_parts();
    assert!(matches!(
        cause,
        GgufSourcePreparationCause::Store(StoreError::Gguf { .. })
    ));
    assert!(bank.is_none());
    let builder = builder.unwrap();
    assert_eq!(builder.checkpoints.as_ptr(), backing);
    assert_eq!(builder.checkpoints.capacity(), capacity);
    assert!(builder.catalog.is_empty());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.gguf");
    write(&path, "old");
    let checkpoint = Checkpoint::open(&path).unwrap();
    let plan = test_plan(&checkpoint);
    // Limit rejection must precede the deliberately invalid mapping.
    let error = open_prepared_gguf_source_with_reader_buffers(checkpoint.clone(), &plan, &[], 0)
        .unwrap_err();
    assert!(matches!(
        error.source().unwrap().downcast_ref::<StoreError>(),
        Some(StoreError::InvalidShardCacheLimit)
    ));
    let (cause, builder, bank) = error.into_parts();
    assert!(matches!(
        cause,
        GgufSourcePreparationCause::Store(StoreError::InvalidShardCacheLimit)
    ));
    assert!(builder.is_none() && bank.is_none());
    let expected = open_prepared_gguf_source(checkpoint.clone(), &plan, &[], 1).unwrap_err();
    let error =
        open_prepared_gguf_source_with_reader_buffers(checkpoint, &plan, &[], 1).unwrap_err();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<StoreError>()
            .unwrap()
            .to_string(),
        expected.to_string()
    );
    let (_, builder, bank) = error.into_parts();
    assert!(builder.is_none() && bank.is_none());
}
