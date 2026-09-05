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
fn paged_options_require_finite_nonzero_limits() {
    assert!(PagedCacheOptions::new(0, 1, 1, 1).is_err());
    assert!(PagedCacheOptions::new(16, 0, 1, 1).is_err());
    assert!(PagedCacheOptions::new(16, 1, 1, 0).is_err());
    assert!(PagedCacheOptions::new(16, 1, 0, 1).is_ok());
}

#[test]
fn prefix_hash_is_order_sensitive() {
    assert_ne!(
        prompt_cache_token_fingerprint(&[1, 2, 3]),
        prompt_cache_token_fingerprint(&[3, 2, 1])
    );
}

#[test]
fn prompt_cache_topology_preserves_parallel_coordinates_and_rank_identity() {
    let topology = crate::test_parallel_rank(5, 2, 2, 2);
    let coordinates = topology.coordinates();
    let cache_topology = PromptCacheTopology::new(
        Some((topology.pipeline_parallel_size(), coordinates.pipeline())),
        Some((topology.tensor_parallel_size(), coordinates.tensor())),
        Some((topology.expert_parallel_size(), coordinates.expert())),
        true,
    )
    .unwrap();

    assert_eq!(cache_topology.stage(), Some((2, 1)));
    assert_eq!(cache_topology.shard(), Some((2, 0)));
    assert_eq!(cache_topology.addressable(), Some((2, 1)));
    assert_eq!(
        cache_topology.cache_rank_identity(),
        Some(CacheRankIdentity::new(Some(1), Some(0), Some(1)))
    );

    let replicated = PromptCacheTopology::default();
    assert_eq!(replicated, PromptCacheTopology::default());
    assert_eq!(replicated.cache_rank_identity(), None);
}

#[test]
fn ordered_layer_layout_round_trips_all_attention_patterns() {
    let schedules = [
        vec![None, None, None, None],
        vec![Some(4), Some(4), Some(4), Some(4)],
        vec![None, Some(4), None, Some(4)],
        vec![Some(3), None, None, Some(9)],
        vec![Some(2), Some(5), Some(11), None],
    ];
    for windows in schedules {
        let layout = key_value_layout(windows);
        let json = serde_json::to_string(&layout).unwrap();
        let restored: LayerSchedule<LayerCachePolicy> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, layout);
    }
}

#[test]
fn attention_windows_reject_zero_negative_and_overflowing_sources() {
    assert!(AttentionPolicy::from_sliding_window(Some(0)).is_err());
    assert!(AttentionPolicy::from_sliding_window(Some(-1)).is_err());
    for json in [
        r#"{"sliding":{"window":0}}"#,
        r#"{"sliding":{"window":-1}}"#,
        r#"{"sliding":{"window":4294967296}}"#,
    ] {
        assert!(serde_json::from_str::<AttentionPolicy>(json).is_err());
    }
}

#[test]
fn cache_identity_hashes_the_complete_ordered_layout() {
    let base = prompt_descriptor();
    let variants = [
        key_value_layout([Some(4)]),
        key_value_layout([Some(5)]),
        key_value_layout([None]),
        key_value_layout([None, Some(4)]),
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        PromptCacheModelIdentity::compressed_layouts(1, 1, 1).unwrap(),
    ];
    let hashes = variants
        .into_iter()
        .map(|layer_layout| {
            let layer_count = layer_layout.len();
            stable_hash(
                &PromptCacheDescriptor::new(
                    base.model_family(),
                    base.effective_model_type(),
                    base.checkpoint_fingerprint(),
                    base.prefix_content_fingerprint(),
                    base.architecture_fingerprint(),
                    layer_count,
                    0,
                    layer_count,
                    base.batch_size(),
                    layer_layout,
                    vec![0; layer_count],
                    vec![
                        eredu_core::cache::PromptCacheStateSegment::new("state", 0..layer_count)
                            .unwrap(),
                    ],
                    base.sink_tokens(),
                    base.topology().clone(),
                )
                .unwrap(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(hashes.len(), 6);

    let first = key_value_layout([None, Some(4)]);
    let reordered = key_value_layout([Some(4), None]);
    assert_ne!(stable_hash(&first), stable_hash(&reordered));
}

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
fn v5_behavioral_state_layout_round_trips_and_changes_identity() {
    let incoherent = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![StateTensorDimension::Scalar],
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .expect_err("large recurrent state cannot use the rolling-state lifecycle");
    assert!(incoherent
        .to_string()
        .contains("requires LayerScopedOffloadable"));
    let convolution = StateTensorPolicy::new(
        StateTensorRole::Convolution { slot: 0 },
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(3).unwrap(),
            StateTensorDimension::fixed(4).unwrap(),
        ],
        StateTensorDtype::Floating,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let recurrent = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(4).unwrap(),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    let layouts = [
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::fixed_only(vec![convolution.clone()]).unwrap()],
        )
        .unwrap(),
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::fixed_only(vec![recurrent.clone()]).unwrap()],
        )
        .unwrap(),
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value_with_fixed_state(
                AttentionPolicy::Full,
                1,
                1,
                vec![convolution.clone()],
            )
            .unwrap()],
        )
        .unwrap(),
    ];
    let hashes = layouts
        .iter()
        .map(|layout| {
            let json = serde_json::to_string(layout).unwrap();
            let restored: LayerSchedule<LayerCachePolicy> = serde_json::from_str(&json).unwrap();
            assert_eq!(&restored, layout);
            stable_hash(layout)
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(hashes.len(), layouts.len());
    assert_eq!(
        convolution.residency_class(),
        StateResidencyClass::AlwaysDeviceMutable
    );
    assert_eq!(
        recurrent.residency_class(),
        StateResidencyClass::LayerScopedOffloadable
    );
    assert_eq!(
        layouts[2].get(0).unwrap().attention_residency_class(),
        Some(StateResidencyClass::SealablePaged)
    );
}

#[test]
fn v4_fixed_state_validation_rejects_missing_reordered_kind_and_geometry() {
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
    assert!(write(&unexpected).contains("does not match its policy"));

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
    // not change. The buffered payload gate must still reject the shard.
    inspect_prompt_cache(directory.path()).unwrap();
    let location = DiskLocation {
        path: shard.clone(),
        first_name: "keys".into(),
        second_name: "values".into(),
        persistent: true,
        buffered: Some(buffer_prompt_cache_shard(&shard).unwrap()),
        payload_sha256: Some(manifest.blocks[0].payload_sha256.clone()),
        payload_verification: Arc::new(OnceLock::new()),
    };
    let error = verify_disk_payload(&location).unwrap_err();
    assert!(error.to_string().contains("payload SHA-256 mismatch"));
}

#[test]
fn imported_prompt_shards_are_buffered_and_retained() {
    let directory = tempfile::tempdir().unwrap();
    write_prompt_fixture(directory.path(), "buffered");
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
    assert_eq!(state.telemetry.report.imported_buffered_shards, 1);
    assert!(state.blocks.values().all(|record| record
        .disk()
        .and_then(|location| location.buffered.as_ref())
        .is_some()));
    for record in state.blocks.values() {
        verify_disk_payload(record.disk().unwrap()).unwrap();
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
