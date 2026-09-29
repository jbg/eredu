#[test]
fn prompt_cache_save_selects_only_descriptor_owned_layers() {
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(1, 64, 64, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let block_id = |global_layer| CacheBlockId {
        session_id: manager.session_id,
        global_layer,
        representation: CacheRepresentation::KeyValue,
        start: 0,
        end: 1,
        rank: None,
    };
    let prompt_host_block = || {
        let arrays = CacheBlockArrays::KeyValue {
            keys: Array::from_slice(&[0.0f32], &[1, 1, 1, 1]),
            values: Array::from_slice(&[0.0f32], &[1, 1, 1, 1]),
        };
        HostCacheBlock::from_device_arrays(&arrays, &cpu_stream()).unwrap()
    };
    {
        let mut state = manager.lock().unwrap();
        for (global_layer, leases) in [(0, 0), (2, 1)] {
            let id = block_id(global_layer);
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical: MlxCacheBlockStorage::host(id, prompt_host_block(), None),
                    bytes: 8,
                    shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                },
                false,
                leases,
            );
        }
    }
    let descriptor = prompt_descriptor().with_layer_count(3).unwrap();
    let destination = tempfile::tempdir().unwrap();

    let manifest = manager
        .save_prompt_cache(
            destination.path().join("target-only"),
            descriptor,
            &[7],
            &[],
            &PromptCacheOptions::default(),
        )
        .unwrap();

    assert_eq!(
        manifest
            .blocks
            .iter()
            .map(|block| block.global_layer)
            .collect::<Vec<_>>(),
        [0]
    );
}

const TEST_PROMPT_CACHE_GENERATION: &str = "generation-test";

#[test]
fn schema_v3_is_rejected_before_v4_fields_are_decoded() {
    let directory = tempfile::tempdir().unwrap();
    let generation = create_prompt_fixture_generation(directory.path());
    fs::write(
        generation.join("manifest.json"),
        br#"{"schema_version":3,"layer_layout":[]}"#,
    )
    .unwrap();
    assert!(matches!(
        inspect_prompt_cache(directory.path()),
        Err(PromptCachePersistenceError::PromptCache(
            PromptCacheError::UnsupportedSchema(3)
        ))
    ));
}

#[test]
fn v7_layer_frontiers_validate_speculative_cache_coverage() {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = write_prompt_fixture(directory.path(), "speculative-frontier");
    manifest.layer_count = 2;
    manifest.global_layer_end = 2;
    manifest.state_segments =
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..2).unwrap()];
    manifest.total_prefix_tokens = 2;
    manifest.prefix_sha256 = prompt_cache_token_fingerprint(&[7, 8]);
    manifest.layer_layout = key_value_layout([None, None]);
    manifest.layer_prefix_offsets = vec![0, -1];

    let first = manifest.blocks[0].clone();
    let mut target_tail = first.clone();
    target_tail.start = 1;
    target_tail.end = 2;
    let mut draft = first;
    draft.global_layer = 1;
    manifest.blocks = vec![manifest.blocks[0].clone(), target_tail, draft];
    fs::write(
        prompt_fixture_manifest_path(directory.path()),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    assert_eq!(
        inspect_prompt_cache(directory.path())
            .unwrap()
            .layer_prefix_offsets,
        [0, -1]
    );

    manifest.layer_prefix_offsets = vec![0, 0];
    fs::write(
        prompt_fixture_manifest_path(directory.path()),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(inspect_prompt_cache(directory.path())
        .unwrap_err()
        .to_string()
        .contains("ends at 1, expected 2"));
}

#[test]
fn fixed_state_validation_rejects_missing_kind_and_geometry() {
    let directory = tempfile::tempdir().unwrap();
    let base = write_fixed_state_fixture(directory.path());
    inspect_prompt_cache(directory.path()).unwrap();
    let write = |manifest: &PromptCacheManifest| {
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap();
        inspect_prompt_cache(directory.path())
            .unwrap_err()
            .to_string()
    };

    let mut missing = base.clone();
    missing.state_tensors.clear();
    assert!(write(&missing).contains("count"));

    let mut unexpected = base.clone();
    unexpected.state_tensors[0].role = StateTensorRole::Recurrent;
    assert!(write(&unexpected).contains("undeclared owner or role"));

    let mut geometry = base.clone();
    geometry.state_tensors[0].shape = vec![1, 2, 4];
    assert!(write(&geometry).contains("does not match its policy"));

    let mut dtype = base;
    dtype.state_tensors[0].dtype = "Int32".into();
    assert!(write(&dtype).contains("does not match its policy"));
}

#[test]
fn manifest_rejects_reordered_duplicate_missing_and_unexpected_layers() {
    let directory = tempfile::tempdir().unwrap();
    let base = write_prompt_fixture(directory.path(), "ordered-layout");
    let mut two_layers = base.clone();
    two_layers.layer_count = 2;
    two_layers.global_layer_end = 2;
    two_layers.state_segments =
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..2).unwrap()];
    two_layers.layer_layout = key_value_layout([None, Some(7)]);
    two_layers.layer_prefix_offsets = vec![0, 0];
    let mut second = two_layers.blocks[0].clone();
    second.global_layer = 1;
    two_layers.blocks.push(second);

    let write = |manifest: &PromptCacheManifest| {
        fs::write(
            prompt_fixture_manifest_path(directory.path()),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap();
        inspect_prompt_cache(directory.path()).unwrap_err()
    };

    let mut reordered = two_layers.clone();
    reordered.blocks.reverse();
    assert!(write(&reordered).to_string().contains("reordered"));

    let mut duplicate = two_layers.clone();
    duplicate.blocks.insert(1, duplicate.blocks[0].clone());
    assert!(write(&duplicate).to_string().contains("duplicated"));

    let mut missing = two_layers.clone();
    missing.blocks.pop();
    assert!(write(&missing).to_string().contains("missing blocks"));

    let mut unexpected = two_layers.clone();
    unexpected.blocks[1].global_layer = 2;
    assert!(write(&unexpected)
        .to_string()
        .contains("outside the owned range"));
}

#[test]
fn manifest_rejects_policy_payload_kind_and_geometry_mismatches() {
    let directory = tempfile::tempdir().unwrap();
    let base = write_prompt_fixture(directory.path(), "policy-mismatch");
    let mut kind = base.clone();
    kind.layer_layout = PromptCacheModelIdentity::compressed_layouts(1, 1, 1).unwrap();
    fs::write(
        prompt_fixture_manifest_path(directory.path()),
        serde_json::to_vec(&kind).unwrap(),
    )
    .unwrap();
    assert!(inspect_prompt_cache(directory.path())
        .unwrap_err()
        .to_string()
        .contains("does not match its policy"));

    let mut geometry = base;
    geometry.layer_layout = PromptCacheModelIdentity::key_value_layouts([None], 2, 1).unwrap();
    fs::write(
        prompt_fixture_manifest_path(directory.path()),
        serde_json::to_vec(&geometry).unwrap(),
    )
    .unwrap();
    assert!(inspect_prompt_cache(directory.path())
        .unwrap_err()
        .to_string()
        .contains("does not match its policy"));
}

#[test]
fn same_length_prompt_payload_corruption_is_rejected_before_array_conversion() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = write_prompt_fixture(directory.path(), "payload-checksum");
    let shard = prompt_fixture_root(directory.path()).join(&manifest.blocks[0].shard);
    let mut bytes = fs::read(&shard).unwrap();
    let final_byte = bytes.last_mut().expect("fixture shard has a payload");
    *final_byte ^= 0x01;
    fs::write(&shard, &bytes).unwrap();

    // Header-only inspection remains valid because metadata and length did
    // not change. The retained payload gate must still reject the shard.
    inspect_prompt_cache(directory.path()).unwrap();
    let location = DiskLocation {
        path: shard.clone(),
        first_name: "keys".into(),
        second_name: "values".into(),
        persistent: true,
        source: Some(Arc::new(
            eredu_runtime::RetainedCacheShard::open_block(&shard, &manifest.blocks[0]).unwrap(),
        )),
        logical_bytes: manifest.blocks[0].logical_bytes,
        payload_sha256: Some(manifest.blocks[0].payload_sha256.clone()),
    };
    let error = load_host_cache_block_direct(&location, CacheRepresentation::KeyValue).unwrap_err();
    assert!(error.to_string().contains("payload SHA-256 mismatch"));
}

#[test]
fn imported_prompt_shards_retain_handles_without_payload_buffers() {
    let directory = tempfile::tempdir().unwrap();
    write_prompt_fixture(directory.path(), "retained");
    let options = PagedCacheOptions::new(1, 64, 64, 1).unwrap();
    let (manager, _) = open_prompt_cache(
        directory.path(),
        &prompt_descriptor(),
        &prompt_model_identity(),
        &[7],
        options,
    )
    .unwrap();
    let state = manager.lock().unwrap();
    assert_eq!(state.telemetry.report.imported_retained_shards, 1);
    assert!(state.blocks.values().all(|record| record
        .disk()
        .and_then(|location| location.source.as_ref())
        .is_some()));
    for record in state.blocks.values() {
        load_host_cache_block_direct(record.disk().unwrap(), record.physical.id().representation)
            .unwrap();
    }
}

#[test]
fn loaded_model_identity_rejects_a_forged_caller_descriptor() {
    let descriptor = PromptCacheDescriptor::new(
        "decoder",
        "decoder",
        "checkpoint",
        "text:prefix",
        "architecture",
        2,
        0,
        2,
        1,
        PromptCacheModelIdentity::key_value_layouts([None, None], 1, 1).unwrap(),
        vec![0, 0],
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..2).unwrap()],
        0,
        PromptCacheTopology::default(),
    )
    .unwrap();
    let loaded_model = PromptCacheModelIdentity::new(
        "decoder",
        "decoder",
        descriptor.architecture_fingerprint(),
        1,
        0,
        1,
        0,
        PromptCacheTopology::default(),
        PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
        vec![0],
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap()],
    )
    .unwrap();
    assert!(matches!(
        validate_prompt_cache_model_identity(&descriptor, &loaded_model),
        Err(PromptCacheError::Incompatible(_))
    ));
}

#[test]
fn loaded_model_identity_rejects_a_forged_architecture_fingerprint() {
    let descriptor = prompt_descriptor()
        .with_architecture_fingerprint("sha256:caller-repeated-stale-value")
        .unwrap();
    let loaded_model = PromptCacheModelIdentity::new(
        descriptor.model_family(),
        descriptor.effective_model_type(),
        "sha256:derived-from-loaded-model",
        descriptor.layer_count(),
        descriptor.global_layer_start(),
        descriptor.global_layer_end(),
        descriptor.sink_tokens(),
        descriptor.topology().clone(),
        descriptor.layer_layout().clone(),
        descriptor.layer_prefix_offsets().to_vec(),
        descriptor.state_segments().to_vec(),
    )
    .unwrap();
    let error = validate_prompt_cache_model_identity(&descriptor, &loaded_model).unwrap_err();
    assert!(error.to_string().contains("architecture_fingerprint"));
}

#[test]
fn loaded_model_identity_rejects_a_forged_layer_frontier() {
    let descriptor = PromptCacheDescriptor::new(
        "decoder",
        "decoder",
        "checkpoint",
        "text:prefix",
        "architecture",
        1,
        0,
        1,
        1,
        PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
        vec![-1],
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap()],
        0,
        PromptCacheTopology::default(),
    )
    .unwrap();
    let loaded_model = prompt_model_identity();
    let error = validate_prompt_cache_model_identity(&descriptor, &loaded_model).unwrap_err();
    assert!(error.to_string().contains("layer_prefix_offsets"));
}

#[test]
fn prompt_load_rejects_model_incompatible_key_value_dimensions() {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = write_prompt_fixture(directory.path(), "wrong-kv-dimensions");
    let keys = vec![0u8; 32];
    let values = vec![0u8; 32];
    let key_view = TensorView::new(StoredDtype::F32, vec![1, 2, 1, 4], &keys).unwrap();
    let value_view = TensorView::new(StoredDtype::F32, vec![1, 2, 1, 4], &values).unwrap();
    serialize_to_file(
        [("keys", key_view), ("values", value_view)],
        None,
        &prompt_fixture_root(directory.path()).join("block.safetensors"),
    )
    .unwrap();
    manifest.blocks[0].first_shape = vec![1, 2, 1, 4];
    manifest.blocks[0].second_shape = vec![1, 2, 1, 4];
    manifest.blocks[0].logical_bytes = 64;
    fs::write(
        prompt_fixture_manifest_path(directory.path()),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let error = open_prompt_cache(
        directory.path(),
        &prompt_descriptor(),
        &prompt_model_identity(),
        &[7],
        PagedCacheOptions::new(1, 64, 64, 1).unwrap(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not match its policy"));
}

#[test]
fn prompt_load_rejects_model_incompatible_layer_representation() {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = write_prompt_fixture(directory.path(), "wrong-representation");
    let latent = vec![0u8; 16];
    let rotary = vec![0u8; 8];
    let latent_view = TensorView::new(StoredDtype::F32, vec![1, 1, 4], &latent).unwrap();
    let rotary_view = TensorView::new(StoredDtype::F32, vec![1, 1, 2], &rotary).unwrap();
    serialize_to_file(
        [("latent", latent_view), ("rotary_key", rotary_view)],
        None,
        &prompt_fixture_root(directory.path()).join("block.safetensors"),
    )
    .unwrap();
    manifest.blocks[0].representation = CacheRepresentation::CompressedLatentRotary;
    manifest.blocks[0].first_array = "latent".into();
    manifest.blocks[0].second_array = "rotary_key".into();
    manifest.blocks[0].first_shape = vec![1, 1, 4];
    manifest.blocks[0].second_shape = vec![1, 1, 2];
    manifest.blocks[0].logical_bytes = 24;
    fs::write(
        prompt_fixture_manifest_path(directory.path()),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let error = open_prompt_cache(
        directory.path(),
        &prompt_descriptor(),
        &prompt_model_identity(),
        &[7],
        PagedCacheOptions::new(1, 64, 64, 1).unwrap(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not match its policy"));
}
