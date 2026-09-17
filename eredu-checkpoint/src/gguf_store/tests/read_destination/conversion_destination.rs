use super::*;

#[test]
fn prepared_conversion_source_preserves_nonzero_outputs_across_all_physical_conversion_kinds() {
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
                    let before = stats(&source);
                    let lease = source.acquire(req).unwrap();
                    let extent = lease.proof.length_bytes as usize;
                    let prepared = lease
                        .prepare_portable_read()
                        .unwrap()
                        .prepare_conversion()
                        .unwrap();
                    assert_eq!(prepared.raw_layout().size(), extent);
                    assert!(prepared.raw_capacity() >= extent);
                    assert_eq!(stats(&source), before);
                    let converted = prepared.materialize().unwrap();
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
                    let lease = source.acquire(req).unwrap();
                    assert!(!lease.selection_is_materialized());
                    let full_bytes = lease.entry.physical_descriptor.byte_len as usize;
                    let prepared = lease
                        .prepare_portable_read()
                        .unwrap()
                        .prepare_conversion()
                        .unwrap();
                    assert_eq!(prepared.raw_layout().size(), full_bytes);
                    assert_eq!(prepared.materialize().unwrap(), expected);
                    assert_eq!(stats(&source), old.stats());
                }
            }
        }
    }
}

#[test]
fn prepared_conversion_shares_actual_cached_reader_and_source_retirement() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let weak = source.inner.ordinary_weak();
    source
        .acquire(request("matrix.weight", TensorSelection::Full))
        .unwrap()
        .materialize_portable()
        .unwrap();
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: vec![1, 0, 1],
    };
    let prepared = source
        .acquire(request("matrix.weight", selection))
        .unwrap()
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap();
    let ordinary = source
        .acquire(request("matrix.weight", TensorSelection::Full))
        .unwrap();
    assert_eq!(source.diagnostics().unwrap().cache_misses, 1);
    #[cfg(unix)]
    std::fs::remove_file(&path).unwrap();
    let selected = prepared.materialize().unwrap();
    let full = ordinary.materialize_portable().unwrap();
    assert_eq!(
        decoded(&selected),
        [4.5, -5.5, 6.5, 1.5, -2.5, 3.5, 4.5, -5.5, 6.5]
    );
    assert_eq!(decoded(&full), [1.5, -2.5, 3.5, 4.5, -5.5, 6.5]);
    assert_eq!(source.diagnostics().unwrap().cache_hits, 2);
    drop(source);
    assert!(weak.upgrade().is_some());
    drop(ordinary);
    assert!(weak.upgrade().is_none());
    assert_eq!(decoded(&selected)[0], 4.5);
}

#[test]
fn prepared_conversion_preserves_lru_and_intervening_ordinary_access() {
    let directory = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..3)
        .map(|i| directory.path().join(format!("shard{i}.gguf")))
        .collect();
    for (i, path) in paths.iter().enumerate() {
        dense_file(path, &format!("matrix{i}.weight"));
    }
    for maximum in [1, 2] {
        let make = || {
            let mut builder = GgufWeightStore::builder()
                .max_cached_readers(maximum)
                .unwrap();
            for path in &paths {
                builder = add(builder, path);
            }
            builder.build().unwrap()
        };
        let old = ReferenceStore::new(make());
        let source = make();
        let prepared = source
            .acquire(request("matrix0.weight", TensorSelection::Full))
            .unwrap()
            .prepare_portable_read()
            .unwrap()
            .prepare_conversion()
            .unwrap();
        let mut prepared = Some(prepared);
        for (step, index) in [1, 2, 0, 0, 1, 2, 0].into_iter().enumerate() {
            let key = format!("matrix{index}.weight");
            let req = request(&key, TensorSelection::Full);
            let expected = old
                .acquire(req.clone())
                .unwrap()
                .old_materialize_portable()
                .unwrap();
            let actual = if step == 2 {
                prepared.take().unwrap().materialize().unwrap()
            } else if step % 2 == 0 {
                source
                    .acquire(req)
                    .unwrap()
                    .prepare_portable_read()
                    .unwrap()
                    .prepare_conversion()
                    .unwrap()
                    .materialize()
                    .unwrap()
            } else {
                source.acquire(req).unwrap().materialize_portable().unwrap()
            };
            assert_eq!(actual, expected);
            assert_eq!(stats(&source), old.stats());
            assert!(source.diagnostics().unwrap().currently_cached_shards <= maximum);
        }
    }
}

#[test]
fn failed_conversion_read_keeps_outputs_raw_and_source_until_failure_drop() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("partial.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let weak = source.inner.ordinary_weak();
    let lease = source
        .acquire(request("matrix.weight", TensorSelection::Full))
        .unwrap();
    lease.materialize_portable().unwrap();
    let offset = lease.entry.physical_descriptor.data_offset;
    let prepared = lease
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap();
    let before = source.diagnostics().unwrap().physical_reads;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(offset + 3)
        .unwrap();
    let failure = prepared.materialize().unwrap_err();
    assert!(failure.store_error().is_some());
    assert_eq!(failure.raw_bytes().len(), 24);
    assert!(failure.conversion().unwrap().capacities()[2] >= 24);
    assert_eq!(&failure.raw_bytes()[..3], &1.5_f32.to_le_bytes()[..3]);
    assert_eq!(&failure.raw_bytes()[3..], &[0; 21]);
    assert_eq!(source.diagnostics().unwrap().physical_reads, before);
    assert!(failure.lease().store.same(&source.inner));
    drop(source);
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert!(weak.upgrade().is_none());
}

#[test]
fn conversion_reopen_failure_matches_old_cache_effects_and_retains_all_buffers() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("changed.gguf");
    dense_file(&path, "matrix.weight");
    let old = ReferenceStore::new(test_store(&path));
    let source = test_store(&path);
    let req = request("matrix.weight", TensorSelection::Full);
    let ordinary = old.acquire(req.clone()).unwrap();
    let prepared = source
        .acquire(req)
        .unwrap()
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap();
    write_tensor(
        &path,
        "different.weight",
        &[1],
        GgmlType::F32,
        &7.5_f32.to_le_bytes(),
    );
    let expected = ordinary.old_materialize_portable().unwrap_err();
    let failure = prepared.materialize().unwrap_err();
    assert_eq!(
        failure.store_error().unwrap().to_string(),
        expected.to_string()
    );
    assert_eq!(stats(&source), old.stats());
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    assert_eq!(failure.raw_bytes(), &[0; 24]);
    assert!(source.inner.readers.lock().unwrap().touched.is_empty());
}
