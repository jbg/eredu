#[test]
fn neutral_store_errors_remain_typed_and_lossless() {
    let io = CheckpointMaterializationError::from(StoreError::Io {
        path: PathBuf::from("checkpoint.bin"),
        message: "permission denied".into(),
    });
    assert!(matches!(
        io,
        CheckpointMaterializationError::Store(StoreError::Io {
            ref path,
            ref message,
        }) if path == Path::new("checkpoint.bin") && message == "permission denied"
    ));

    let internal = CheckpointMaterializationError::from(StoreError::Internal(
        "reader registry invariant".into(),
    ));
    assert!(matches!(
        internal,
        CheckpointMaterializationError::Store(StoreError::Internal(ref message))
            if message == "reader registry invariant"
    ));
}

trait AcquireBoundedForTest {
    fn acquire(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<WeightLease, CheckpointMaterializationError>;

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, CheckpointMaterializationError>;
}

impl<T: CheckpointSource> AcquireBoundedForTest for T {
    fn acquire(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<WeightLease, CheckpointMaterializationError> {
        self.acquire_with_policy(key, selection, WeightReadPolicy::RequireBounded)
    }

    fn acquire_with_policy(
        &self,
        key: &str,
        selection: TensorSelection,
        policy: WeightReadPolicy,
    ) -> Result<WeightLease, CheckpointMaterializationError> {
        let lease = self
            .acquire_lease(TensorReadRequest {
                key: key.into(),
                selection,
                policy,
            })
            .map_err(CheckpointMaterializationError::from)?;
        let stream = cpu_stream();
        MlxParameterMaterializationContext::new(&stream, &stream).weight_lease(lease)
    }
}

fn acquire_with_context<T: CheckpointSource>(
    store: &T,
    key: &str,
    selection: TensorSelection,
    policy: WeightReadPolicy,
    context: &MlxParameterMaterializationContext,
) -> Result<WeightLease, CheckpointMaterializationError> {
    let lease = store
        .acquire_lease(TensorReadRequest {
            key: key.into(),
            selection,
            policy,
        })
        .map_err(CheckpointMaterializationError::from)?;
    context.weight_lease(lease)
}

#[test]
fn gguf_store_rejects_translated_collisions_across_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.gguf");
    let second = dir.path().join("second.gguf");
    write_dense_gguf(&first, "text.weight", 1.0);
    write_dense_gguf(&second, "vision.weight", 2.0);
    let first = GgufCheckpoint::open(first).unwrap();
    let first_plan = gguf_test_plan(first.catalog());
    let first_plan =
        eredu_checkpoint::validation::resolve_gguf_plan(first.catalog(), &first_plan).unwrap();
    let first_mapping = first
        .catalog()
        .translated_outputs(|_| "shared.weight".into())
        .unwrap();
    let builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .add_resolved_checkpoint(first.catalog().clone(), &first_plan, &first_mapping)
        .unwrap();
    let second = GgufCheckpoint::open(second).unwrap();
    let second_plan = gguf_test_plan(second.catalog());
    let second_plan =
        eredu_checkpoint::validation::resolve_gguf_plan(second.catalog(), &second_plan).unwrap();
    let second_mapping = second
        .catalog()
        .translated_outputs(|_| "shared.weight".into())
        .unwrap();
    let error = builder
        .add_resolved_checkpoint(second.catalog().clone(), &second_plan, &second_mapping)
        .unwrap_err();
    assert!(matches!(
        error,
        eredu_checkpoint::store::StoreError::Gguf { .. }
    ));
}

#[test]
fn gguf_store_cataloging_does_not_touch_payload_readers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_dense_gguf(&path, "value.weight", 3.0);
    let store = open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
        name.to_string()
    })
    .unwrap();
    assert_eq!(store.source_keys(), ["value.weight"]);
    assert_eq!(
        store.source_metadata("value.weight").unwrap().logical_shape,
        [1]
    );
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 0);
    assert!(diagnostics.touched_shard_paths.is_empty());
    assert_eq!(diagnostics.physical_reads, 0);
}

#[test]
fn gguf_store_catalog_contains_only_contract_selected_sources() {
    use eredu_checkpoint::schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_two_dense_gguf(&path);
    let plan = GgufCheckpointPlan::new(
        "selected layout",
        vec![GgufTensorConstraint::required(
            "selected.weight",
            vec![1],
            GgufTypeConstraint::OperationClass(TensorOperation::Dense),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let checkpoint = GgufCheckpoint::open(path).unwrap();
    let tensor_mapping = checkpoint
        .catalog()
        .translated_outputs(str::to_string)
        .unwrap();
    let store = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .add_checkpoint(checkpoint.catalog().clone(), &plan, &tensor_mapping)
        .unwrap()
        .build()
        .unwrap();

    assert!(store.is_checkpoint_contract_resolved());
    assert_eq!(store.source_keys(), ["selected.weight"]);
    assert_eq!(store.unclaimed_checkpoint_keys(), ["unselected.weight"]);
    assert!(matches!(
        store.source_metadata("unselected.weight"),
        Err(eredu_checkpoint::store::StoreError::UnknownTensor { key })
            if key == "unselected.weight"
    ));
}

#[test]
fn native_affine_store_bytes_equal_checkpoint_payload_bytes() {
    for ty in [GgmlType::Q4K, GgmlType::Q5_1, GgmlType::Q8_0] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
        let payload = vec![0; (2 * block_bytes) as usize];
        Writer::default()
            .write(
                std::fs::File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "bank.weight",
                    dimensions: &[block_values, 2],
                    ggml_type: ty,
                    data: &payload,
                }],
            )
            .unwrap();
        let store = open_gguf_checkpoint_source_for_test(
            GgufCheckpoint::open(path).unwrap(),
            str::to_string,
        )
        .unwrap();
        let metadata = store.source_metadata("bank.weight").unwrap();
        assert_eq!(metadata.logical_shape, [2, block_bytes as usize], "{ty:?}");
        assert_eq!(metadata.stored_dtype, StoredDtype::U8, "{ty:?}");
        assert_eq!(metadata.encoded_byte_len, payload.len() as u64, "{ty:?}");
        assert_eq!(store.source_keys(), ["bank.weight"], "{ty:?}");
    }
}

#[test]
fn gguf_dense_contiguous_span_reads_only_the_reshaped_interval() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    let values = write_dense_bank_gguf(&path);
    let store = open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
        name.to_string()
    })
    .unwrap();
    let lease = store
        .acquire(
            "bank.weight",
            TensorSelection::Contiguous {
                offset_elements: 8,
                shape: vec![1, 2, 4],
            },
        )
        .unwrap();
    assert_eq!(lease.output_shape(), [1, 2, 4]);
    assert_eq!(lease.selected_byte_len(), 32);
    let WeightLeaseSource::Gguf(source) = &lease.source else {
        panic!("expected GGUF lease");
    };
    assert!(matches!(
        source.lease.identity().physical_selection(),
        Some(GgufPhysicalSelection::DenseSpan(_))
    ));

    let stream = cpu_stream();
    let selected = lease
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(selected.shape(), [1, 2, 4]);
    assert_eq!(
        selected.evaluated().unwrap().as_slice::<f32>(),
        &values[8..16]
    );
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 1);
    assert_eq!(diagnostics.physical_read_bytes, 32);
}

#[test]
fn gguf_native_contiguous_span_requires_block_alignment_before_payload_io() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_affine_gguf(&path);
    let store = open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
        name.to_string()
    })
    .unwrap();

    let lease = store
        .acquire(
            "bank.weight",
            TensorSelection::Contiguous {
                offset_elements: 0,
                shape: vec![1, 4],
            },
        )
        .unwrap();
    assert_eq!(lease.output_shape(), [1, 4]);
    assert_eq!(lease.selected_byte_len(), 16);
    let WeightLeaseSource::Gguf(source) = &lease.source else {
        panic!("expected GGUF lease");
    };
    assert!(matches!(
        source.lease.identity().physical_selection(),
        Some(GgufPhysicalSelection::DenseSpan(_))
    ));
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
    assert!(diagnostics.touched_shard_paths.is_empty());

    let error = store
        .acquire(
            "bank.weight",
            TensorSelection::Contiguous {
                offset_elements: 1,
                shape: vec![1, 4],
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CheckpointMaterializationError::Store(StoreError::BoundedSelectionUnavailable { .. })
    ));
    assert!(error.to_string().contains("must align"));
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
    assert!(diagnostics.touched_shard_paths.is_empty());
}

#[test]
fn gguf_affine_companions_coalesce_selected_physical_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_affine_gguf(&path);
    let store = open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
        name.to_string()
    })
    .unwrap();
    let stream = cpu_stream();
    let context = MlxParameterMaterializationContext::new(&stream, &stream);
    let selection = TensorSelection::Range {
        axis: 0,
        start: 1,
        end: 2,
    };
    let weight = acquire_with_context(
        &store,
        "bank.weight",
        selection.clone(),
        WeightReadPolicy::RequireBounded,
        &context,
    )
    .unwrap()
    .prepare_materialization(&stream, &stream)
    .unwrap();
    let scales = acquire_with_context(
        &store,
        "bank.scales",
        selection.clone(),
        WeightReadPolicy::RequireBounded,
        &context,
    )
    .unwrap()
    .prepare_materialization(&stream, &stream)
    .unwrap();
    let biases = acquire_with_context(
        &store,
        "bank.biases",
        selection,
        WeightReadPolicy::RequireBounded,
        &context,
    )
    .unwrap()
    .prepare_materialization(&stream, &stream)
    .unwrap();
    weight.finish().unwrap();
    scales.finish().unwrap();
    biases.finish().unwrap();

    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.backend, WeightStoreBackend::Gguf);
    assert_eq!(diagnostics.physical_reads, 1);
    assert_eq!(diagnostics.physical_read_bytes, 18);
    assert_eq!(diagnostics.coalesced_group_hits, 2);
}

#[test]
fn gguf_inner_affine_companions_normalize_to_one_bounded_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_wide_affine_gguf(&path);
    let store = open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
        name.to_string()
    })
    .unwrap();
    let stream = cpu_stream();
    let context = MlxParameterMaterializationContext::new(&stream, &stream);
    let weight = acquire_with_context(
        &store,
        "bank.weight",
        TensorSelection::Range {
            axis: 1,
            start: 4,
            end: 8,
        },
        WeightReadPolicy::RequireBounded,
        &context,
    )
    .unwrap()
    .prepare_materialization(&stream, &stream)
    .unwrap();
    let scales = acquire_with_context(
        &store,
        "bank.scales",
        TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 2,
        },
        WeightReadPolicy::RequireBounded,
        &context,
    )
    .unwrap()
    .prepare_materialization(&stream, &stream)
    .unwrap();
    let biases = acquire_with_context(
        &store,
        "bank.biases",
        TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 2,
        },
        WeightReadPolicy::RequireBounded,
        &context,
    )
    .unwrap()
    .prepare_materialization(&stream, &stream)
    .unwrap();
    assert_eq!(weight.output().shape(), [2, 4]);
    assert_eq!(scales.output().shape(), [2, 1]);
    assert_eq!(biases.output().shape(), [2, 1]);
    weight.finish().unwrap();
    scales.finish().unwrap();
    biases.finish().unwrap();

    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 1);
    assert_eq!(diagnostics.physical_read_bytes, 36);
    assert_eq!(diagnostics.coalesced_group_hits, 2);
}

#[test]
fn gguf_bounded_policy_rejects_misalignment_before_payload_io() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    write_wide_affine_gguf(&path);
    let store = open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
        name.to_string()
    })
    .unwrap();
    let selection = TensorSelection::Range {
        axis: 1,
        start: 1,
        end: 5,
    };
    assert!(matches!(
        store.acquire_with_policy(
            "bank.weight",
            selection.clone(),
            WeightReadPolicy::RequireBounded,
        ),
        Err(CheckpointMaterializationError::Store(
            StoreError::BoundedSelectionUnavailable { .. }
        ))
    ));
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
    assert!(diagnostics.touched_shard_paths.is_empty());

    let stream = cpu_stream();
    let value = store
        .acquire_with_policy(
            "bank.weight",
            selection,
            WeightReadPolicy::AllowFullTensorRead,
        )
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(value.shape(), [2, 4]);
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 1);
    assert_eq!(diagnostics.physical_read_bytes, 72);
}

#[test]
fn gguf_mxfp4_and_iq_outputs_map_to_native_block_coordinates() {
    let expected = Some(GgufTensorSelection::Range {
        axis: 1,
        start: 32,
        end: 64,
    });
    for (ty, byte_len, outputs) in [
        (
            GgmlType::MxFp4,
            68,
            vec![("bank.weight", 4, 8), ("bank.scales", 1, 2)],
        ),
        (GgmlType::IQ4NL, 72, vec![("bank.weight", 18, 36)]),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_block_gguf(&path, ty, byte_len);
        let store =
            open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(path).unwrap(), |name| {
                name.to_string()
            })
            .unwrap();
        for (key, start, end) in outputs {
            let lease = store
                .acquire_with_policy(
                    key,
                    TensorSelection::Range {
                        axis: 1,
                        start,
                        end,
                    },
                    WeightReadPolicy::RequireBounded,
                )
                .unwrap();
            assert_eq!(gguf_physical_selection(&lease), expected);
        }
        assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
    }
}

#[test]
fn indexed_catalog_is_sorted_without_mapping_payloads() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("broken.safetensors"), b"not a checkpoint").unwrap();
    std::fs::write(
        dir.path().join("also-broken.safetensors"),
        b"not a checkpoint either",
    )
    .unwrap();
    write_index(
        dir.path(),
        &[
            ("z.weight", "broken.safetensors"),
            ("a.weight", "also-broken.safetensors"),
        ],
    );

    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    assert_eq!(store.source_keys(), ["a.weight", "z.weight"]);
    assert_eq!(
        store.source_diagnostics().unwrap(),
        WeightStoreDiagnostics {
            backend: WeightStoreBackend::Safetensors,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            currently_cached_shards: 0,
            touched_shard_paths: vec![],
            payload_shard_paths: vec![],
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        }
    );
    assert!(matches!(
        store.acquire("a.weight", TensorSelection::Full),
        Err(CheckpointMaterializationError::Store(
            StoreError::MalformedSafetensors { .. }
        ))
    ));
    assert!(matches!(
        store.acquire("z.weight", TensorSelection::Full),
        Err(CheckpointMaterializationError::Store(
            StoreError::MalformedSafetensors { .. }
        ))
    ));
}

#[test]
fn rejects_contradictory_index_mapping_when_accessed() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(
        &dir.path().join("payload.safetensors"),
        "actual",
        &[1],
        vec![1],
    );
    write_index(dir.path(), &[("claimed", "payload.safetensors")]);
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    assert!(matches!(
        store.source_metadata("claimed"),
        Err(StoreError::ContradictoryIndexMapping { key, .. }) if key == "claimed"
    ));
}

#[test]
fn discovers_direct_and_single_file_directory_catalogs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_two_i32(&path);

    let directory = SafetensorsWeightStore::open(dir.path()).unwrap();
    let direct = SafetensorsWeightStore::open(&path).unwrap();
    assert_eq!(directory.source_keys(), ["a_tensor", "z_tensor"]);
    assert_eq!(direct.source_keys(), directory.source_keys());
    assert_eq!(
        directory
            .source_diagnostics()
            .unwrap()
            .currently_cached_shards,
        0
    );
}

#[test]
fn rejects_malformed_indexes_and_unsafe_shard_paths() {
    let malformed = tempfile::tempdir().unwrap();
    std::fs::write(
        malformed.path().join("model.safetensors.index.json"),
        b"{invalid",
    )
    .unwrap();
    assert!(matches!(
        SafetensorsWeightStore::open(malformed.path()),
        Err(eredu_checkpoint::store::StoreError::SafetensorsShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::MalformedIndex { .. }
        ))
    ));

    let duplicate = tempfile::tempdir().unwrap();
    std::fs::write(
        duplicate.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"weight":"one.safetensors","weight":"two.safetensors"}}"#,
    )
    .unwrap();
    assert!(matches!(
        SafetensorsWeightStore::open(duplicate.path()),
        Err(eredu_checkpoint::store::StoreError::SafetensorsShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::MalformedIndex { .. }
        ))
    ));

    for shard in ["../escape.safetensors", "/absolute.safetensors"] {
        let dir = tempfile::tempdir().unwrap();
        write_index(dir.path(), &[("weight", shard)]);
        assert!(matches!(
            SafetensorsWeightStore::open(dir.path()),
            Err(eredu_checkpoint::store::StoreError::SafetensorsShards(
                eredu_checkpoint::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
            ))
        ));
    }
}

#[cfg(unix)]
#[test]
fn rejects_indexed_symlinks_that_escape_the_model_directory() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let outside_file = outside.path().join("outside.safetensors");
    write_i32(&outside_file, "weight", &[1], vec![1]);
    std::os::unix::fs::symlink(&outside_file, dir.path().join("linked.safetensors")).unwrap();
    write_index(dir.path(), &[("weight", "linked.safetensors")]);
    assert!(matches!(
        SafetensorsWeightStore::open(dir.path()),
        Err(StoreError::SafetensorsShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
        ))
    ));
}

#[cfg(unix)]
#[test]
fn accepts_hugging_face_snapshot_symlinks_into_repository_blobs() {
    let cache = tempfile::tempdir().unwrap();
    let repository = cache.path().join("models--owner--model");
    let snapshot = repository.join("snapshots/revision");
    let blobs = repository.join("blobs");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(&blobs).unwrap();
    write_i32(&blobs.join("payload"), "weight", &[7], vec![1]);
    std::os::unix::fs::symlink(
        "../../blobs/payload",
        snapshot.join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    write_index(&snapshot, &[("weight", "model-00001-of-00001.safetensors")]);

    let store = SafetensorsWeightStore::open(&snapshot).unwrap();
    let stream = cpu_stream();
    let materialized = store
        .acquire("weight", TensorSelection::Full)
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let value = materialized.evaluated().unwrap();
    assert_eq!(value.as_slice::<i32>(), &[7]);
}
