use super::*;

#[test]
fn boxed_conversion_and_metadata_refusals_keep_prepared_raw_and_same_box() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preparation.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let mut lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    let pointer = std::ptr::from_ref(lease.as_ref());
    // Private descriptor corruption reaches the actual pre-reserve layout check;
    // the raw destination remains the genuine 24-byte physical selection.
    lease.entry.physical_descriptor.byte_len = u64::MAX;
    let failure = GgufLease::materialize_prepared_boxed(lease).unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    let PreparedGgufBoxedFailure::Conversion(partial) = failure else {
        panic!("conversion refusal")
    };
    assert!(matches!(
        partial.conversion_error(),
        Some(eredu_gguf::ConversionDestinationError::Layout)
    ));
    assert_eq!(partial.raw_bytes(), &[0; 24]);
    drop(partial);
    let lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    let pointer = std::ptr::from_ref(lease.as_ref());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = source.inner.readers.lock().unwrap();
        panic!("real metadata cache poison");
    }));
    assert!(poisoned.is_err());
    let failure = GgufLease::materialize_prepared_boxed(lease).unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    let PreparedGgufBoxedFailure::Tensor(partial) = failure else {
        panic!("metadata refusal")
    };
    assert!(matches!(
        partial.store_error(),
        Some(StoreError::Internal(_))
    ));
    assert_eq!(partial.raw_bytes(), &[0; 24]);
    assert!(partial.metadata().is_none());
}

#[test]
fn boxed_captured_header_eviction_change_and_restored_retry_use_original_source_policy() {
    fn write(path: &Path, label: &str, value: f32) {
        Writer::default()
            .write(
                File::create(path).unwrap(),
                &BTreeMap::from([(
                    "label".to_owned(),
                    eredu_gguf::MetadataValue::String(label.into()),
                )]),
                &[TensorInput {
                    name: "weight",
                    dimensions: &[2],
                    ggml_type: GgmlType::F32,
                    data: &[value.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat(),
                }],
            )
            .unwrap();
    }
    let directory = tempfile::tempdir().unwrap();
    let paths = [
        directory.path().join("a.gguf"),
        directory.path().join("b.gguf"),
    ];
    let mut builder = GgufWeightStore::builder().max_cached_readers(1).unwrap();
    for (index, path) in paths.iter().enumerate() {
        write(path, "old", 1.25);
        let checkpoint = Checkpoint::open_with_prepared_headers(path).unwrap();
        let plan = test_plan(&checkpoint);
        let names = checkpoint
            .translated_outputs(|name| format!("{index}.{name}"))
            .unwrap();
        builder = builder.add_checkpoint(checkpoint, &plan, &names).unwrap();
    }
    let source = builder.build().unwrap();
    for key in ["0.weight", "0.weight", "1.weight"] {
        let lease = Box::new(source.acquire(request(key, TensorSelection::Full)).unwrap());
        let (output, lease) = GgufLease::materialize_prepared_boxed(lease).unwrap();
        assert_eq!(decoded(&output), [1.25, -3.5]);
        drop(lease);
    }
    write(&paths[0], "new", 1.25);
    let lease = Box::new(
        source
            .acquire(request("0.weight", TensorSelection::Full))
            .unwrap(),
    );
    let pointer = std::ptr::from_ref(lease.as_ref());
    let failure = GgufLease::materialize_prepared_boxed(lease).unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    assert!(
        matches!(failure.store_error(), Some(StoreError::GgufPreparedHeaderChanged { key, source }) if key == "0.weight" && source.prepared_header_change().is_some())
    );
    let cache = source.inner.readers.lock().unwrap();
    assert_eq!((cache.misses, cache.hits), (3, 1));
    assert!(cache
        .materializers
        .iter()
        .all(|m| m.open_shard_path().is_none()));
    drop(cache);
    // Restoring only the captured header permits new numerical payload bytes.
    write(&paths[0], "old", 19.0);
    let lease = Box::new(
        source
            .acquire(request("0.weight", TensorSelection::Full))
            .unwrap(),
    );
    let (output, lease) = GgufLease::materialize_prepared_boxed(lease).unwrap();
    assert_eq!(decoded(&output), [19.0, -3.5]);
    assert!(
        failure.store_error().is_some(),
        "old failure remains an owning snapshot"
    );
    drop((output, lease, source, failure));
}

#[test]
fn boxed_layout_refusal_retains_same_allocation_without_reading() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("layout.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let weak = source.inner.ordinary_weak();
    let mut lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    // Private malformed-layout probe, never an admitted payload size.
    lease.proof.length_bytes = u64::MAX;
    let pointer = std::ptr::from_ref(lease.as_ref());
    let failure = GgufLease::materialize_prepared_boxed(lease).unwrap_err();
    assert!(matches!(failure, PreparedGgufBoxedFailure::Read(_)));
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    drop(source);
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert!(weak.upgrade().is_none());
}

#[test]
fn boxed_partial_read_retains_same_source_and_actual_initialized_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("partial.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let weak = source.inner.ordinary_weak();
    let lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    lease.materialize_portable().unwrap();
    let pointer = std::ptr::from_ref(lease.as_ref());
    let offset = lease.entry.physical_descriptor.data_offset;
    let reads = source.diagnostics().unwrap().physical_reads;
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(offset + 3)
        .unwrap();
    let failure = GgufLease::materialize_prepared_boxed(lease).unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    let PreparedGgufBoxedFailure::Tensor(ref partial) = failure else {
        panic!("read failure must retain G3")
    };
    assert!(partial.store_error().is_some());
    assert_eq!(&partial.raw_bytes()[..3], &1.5_f32.to_le_bytes()[..3]);
    assert_eq!(&partial.raw_bytes()[3..], &[0; 21]);
    assert!(partial.metadata().unwrap().capacity_bytes().unwrap() > 0);
    assert_eq!(source.diagnostics().unwrap().physical_reads, reads);
    drop(source);
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert!(weak.upgrade().is_none());
}

#[test]
fn boxed_reopen_failure_preserves_original_cause_cache_effects_and_owner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("changed.gguf");
    dense_file(&path, "matrix.weight");
    let old = ReferenceStore::new(test_store(&path));
    let source = test_store(&path);
    let request = request("matrix.weight", TensorSelection::Full);
    let ordinary = old.acquire(request.clone()).unwrap();
    let lease = Box::new(source.acquire(request).unwrap());
    let pointer = std::ptr::from_ref(lease.as_ref());
    write_tensor(
        &path,
        "different.weight",
        &[1],
        GgmlType::F32,
        &7.5_f32.to_le_bytes(),
    );
    let expected = ordinary.old_materialize_portable().unwrap_err();
    let failure = GgufLease::materialize_prepared_boxed(lease).unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    assert_eq!(
        failure.store_error().unwrap().to_string(),
        expected.to_string()
    );
    assert_eq!(stats(&source), old.stats());
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
}

#[test]
fn returned_box_keeps_source_after_store_and_output_retire() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("returned.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let weak = source.inner.ordinary_weak();
    let lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    let pointer = std::ptr::from_ref(lease.as_ref());
    drop(source);
    let (output, lease) = GgufLease::materialize_prepared_boxed(lease).unwrap();
    assert_eq!(decoded(&output), [1.5, -2.5, 3.5, 4.5, -5.5, 6.5]);
    assert_eq!(std::ptr::from_ref(lease.as_ref()), pointer);
    drop(output);
    assert!(weak.upgrade().is_some());
    drop(lease);
    assert!(weak.upgrade().is_none());
}

#[test]
fn boxed_source_returns_identical_lease_for_nonzero_formats_selections_and_fallback() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("formats.gguf");
    for endian in [Endian::Little, Endian::Big] {
        for ty in [
            GgmlType::F32,
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::MxFp4,
        ] {
            let raw = if ty == GgmlType::F32 {
                (0..128)
                    .flat_map(|i| {
                        let x = (i as f32 - 31.0) * 0.25;
                        match endian {
                            Endian::Little => x.to_le_bytes(),
                            Endian::Big => x.to_be_bytes(),
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                let size = ty.block_and_bytes().unwrap().1 as usize;
                (0..4)
                    .flat_map(|block| {
                        let mut bytes = vec![0u8; size];
                        if ty == GgmlType::MxFp4 {
                            bytes[0] = 130;
                            for i in 0..16 {
                                bytes[i + 1] = (i as u8) | ((15 - i as u8) << 4);
                            }
                        } else {
                            let scale = match endian {
                                Endian::Little => 0x3c00_u16.to_le_bytes(),
                                Endian::Big => 0x3c00_u16.to_be_bytes(),
                            };
                            bytes[..2].copy_from_slice(&scale);
                            for (i, x) in bytes.iter_mut().enumerate().skip(2) {
                                *x = ((i + block) % 63 + 1) as u8;
                            }
                        }
                        bytes
                    })
                    .collect::<Vec<_>>()
            };
            Writer::new(WriterOptions {
                version: 3,
                endian,
                alignment: 32,
            })
            .unwrap()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "matrix.weight",
                    dimensions: &[64, 2],
                    ggml_type: ty,
                    data: &raw,
                }],
            )
            .unwrap();
            let old = ReferenceStore::new(test_store(&path));
            let source = test_store(&path);
            for key in source.keys() {
                let entry = &source.inner.catalog[&key];
                let units = entry.logical_last_units_per_block.unwrap();
                for selection in [
                    TensorSelection::Full,
                    TensorSelection::Range {
                        axis: 0,
                        start: 1,
                        end: 2,
                    },
                    TensorSelection::Indices {
                        axis: 0,
                        indices: vec![1, 0, 1],
                    },
                    TensorSelection::Contiguous {
                        offset_elements: units,
                        shape: vec![1, units],
                    },
                ] {
                    let req = request(&key, selection);
                    let expected = old
                        .acquire(req.clone())
                        .unwrap()
                        .old_materialize_portable()
                        .unwrap();
                    let lease = Box::new(source.acquire(req).unwrap());
                    let pointer = std::ptr::from_ref(lease.as_ref());
                    let (converted, lease) = GgufLease::materialize_prepared_boxed(lease).unwrap();
                    assert_eq!(std::ptr::from_ref(lease.as_ref()), pointer);
                    assert!(lease.store.same(&source.inner));
                    assert_eq!(converted, expected, "{ty:?} {endian:?} {key}");
                    assert_eq!(stats(&source), old.stats());
                    if ty == GgmlType::MxFp4 {
                        let ConvertedTensor::MxFp4(t) = converted.converted() else {
                            panic!("MXFP4")
                        };
                        assert_eq!(
                            &t.weights[..4],
                            &[0x76543210, 0xfedcba98, 0x89abcdef, 0x01234567]
                        );
                        assert!(t.scales.iter().all(|x| *x == 130));
                    }
                    if ty == GgmlType::Q8_0 {
                        assert!(
                            matches!(converted.converted(), ConvertedTensor::IQuant(_))
                                == (endian == Endian::Little)
                        );
                    }
                }
                if ty == GgmlType::Q4_0 && key == "matrix.weight" {
                    let mut req = request(
                        &key,
                        TensorSelection::Range {
                            axis: 1,
                            start: 1,
                            end: 2,
                        },
                    );
                    assert!(matches!(
                        source.acquire(req.clone()),
                        Err(StoreError::BoundedSelectionUnavailable { .. })
                    ));
                    req.policy = ReadPolicy::AllowFullTensorRead;
                    let expected = old
                        .acquire(req.clone())
                        .unwrap()
                        .old_materialize_portable()
                        .unwrap();
                    let lease = Box::new(source.acquire(req).unwrap());
                    assert!(!lease.selection_is_materialized());
                    let pointer = std::ptr::from_ref(lease.as_ref());
                    let (actual, lease) = GgufLease::materialize_prepared_boxed(lease).unwrap();
                    assert_eq!(std::ptr::from_ref(lease.as_ref()), pointer);
                    assert_eq!(actual, expected);
                    assert_eq!(stats(&source), old.stats());
                }
            }
        }
    }
}
