// Residency, persistence, and resource-lifetime tests.

use super::{
    buffer_prompt_cache_shard, cpu_stream, hash_prompt_cache_shard_payload,
    host_cache_capacity_upper_bound, inspect_prompt_cache, open_prompt_cache, verify_disk_payload,
    CacheBlockArrays, CacheBlockId, CacheBlockRecord, CacheIoOperationKey, CacheIoOperationKind,
    CacheLayerResidencyStats, CacheManagerState, CachePoolError, CachePoolResource,
    CacheRankIdentity, CacheRepresentation, CacheResidencyError, CacheResidencyManager,
    CacheResidencyPool, CacheStoragePhase, CacheTier, DiskLocation, DiskResult, DiskTask,
    DiskWorker, DiskWriteCommit, HostCacheBlock, HostDemotionCompletion, HostDemotionTicket,
    HostWriteReservation, MlxCacheBlockStorage, MlxCacheIoOperation, PagedCacheOptions,
    StateTensorOwner, StateTensorRole,
};
use eredu_core::cache::{
    prompt_cache_token_fingerprint, validate_prompt_cache_model_identity, LayerCachePolicy,
    MutableStateResidency, PromptCacheBlock, PromptCacheDescriptor, PromptCacheError,
    PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheStateTensor,
    PromptCacheTopology, StateResidencyClass, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, PROMPT_CACHE_SCHEMA_VERSION,
};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_runtime::{
    resolve_prompt_cache_root, CachePoolLimits, CacheResidencyConfigurationError, MutableCacheTail,
    PromptCachePersistenceError, CACHE_RESIDENCY_LAYER_REPORT_LIMIT, PROMPT_CACHE_CURRENT_FILE,
    PROMPT_CACHE_GENERATIONS_DIRECTORY,
};
use safemlx::{
    host_transfer_capacity_upper_bound, transforms::async_eval_with_event, Array, Device,
    DeviceType, HostTransferPolicy, HostTransferStorageKind, Stream,
};
use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};
use std::{
    fs,
    hash::{DefaultHasher, Hash, Hasher},
    path::{Path, PathBuf},
    sync::{mpsc, Arc, OnceLock},
    thread,
    time::Duration,
};

fn disk_test_id(start: i64) -> CacheBlockId {
    CacheBlockId {
        session_id: 7,
        global_layer: 0,
        representation: CacheRepresentation::KeyValue,
        start,
        end: start + 1,
        rank: None,
    }
}

fn missing_location(root: &Path, name: &str) -> DiskLocation {
    DiskLocation {
        path: root.join(name),
        first_name: "keys".into(),
        second_name: "values".into(),
        persistent: false,
        buffered: None,
        payload_sha256: None,
        payload_verification: Arc::new(OnceLock::new()),
    }
}

fn test_device_block() -> CacheBlockArrays {
    CacheBlockArrays::KeyValue {
        keys: Array::from_slice(&[0.0f32], &[1]),
        values: Array::from_slice(&[0.0f32], &[1]),
    }
}

fn test_host_block() -> HostCacheBlock {
    HostCacheBlock::from_device_arrays(&test_device_block(), &super::cpu_stream()).unwrap()
}

fn test_host_writing(block: HostCacheBlock, ticket: super::DiskTicket) -> MlxCacheBlockStorage {
    let mut physical = MlxCacheBlockStorage::host(ticket.key.id.clone(), block, None);
    physical
        .begin_write(MlxCacheIoOperation {
            ticket,
            reserved_host_bytes: None,
        })
        .unwrap();
    physical
}

fn test_demoting(arrays: CacheBlockArrays, ticket: HostDemotionTicket) -> MlxCacheBlockStorage {
    let mut physical = MlxCacheBlockStorage::device(ticket.id.clone(), arrays, None);
    physical.begin_host_demotion(ticket).unwrap();
    physical
}

fn insert_test_record(
    state: &mut CacheManagerState,
    record: CacheBlockRecord,
    protected_prefix: bool,
    leases: usize,
) {
    let id = record.physical.id().clone();
    state
        .lifecycle
        .insert(id.clone(), protected_prefix)
        .unwrap();
    for _ in 0..leases {
        state.lifecycle.acquire(&id).unwrap();
    }
    assert!(state.blocks.insert(id, record).is_none());
}

fn manager_with_leased_block() -> CacheResidencyManager {
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(1, 64, 64, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let id = CacheBlockId {
        session_id: manager.session_id,
        global_layer: 0,
        representation: CacheRepresentation::KeyValue,
        start: 0,
        end: 1,
        rank: None,
    };
    insert_test_record(
        &mut manager.lock().unwrap(),
        CacheBlockRecord {
            physical: MlxCacheBlockStorage::host(id.clone(), test_host_block(), None),
            bytes: 0,
            shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
            dtypes: ["Float32".into(), "Float32".into()],
            imported: false,
        },
        false,
        1,
    );
    manager
}

fn prompt_descriptor() -> PromptCacheDescriptor {
    PromptCacheDescriptor::new(
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
        vec![0],
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap()],
        0,
        PromptCacheTopology::default(),
    )
    .unwrap()
}

fn key_value_layout(
    windows: impl IntoIterator<Item = Option<i32>>,
) -> LayerSchedule<LayerCachePolicy> {
    PromptCacheModelIdentity::key_value_layouts(windows, 1, 1).unwrap()
}

fn stable_hash(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn prompt_model_identity() -> PromptCacheModelIdentity {
    let descriptor = prompt_descriptor();
    PromptCacheModelIdentity::new(
        descriptor.model_family(),
        descriptor.effective_model_type(),
        descriptor.architecture_fingerprint(),
        descriptor.layer_count(),
        descriptor.global_layer_start(),
        descriptor.global_layer_end(),
        descriptor.sink_tokens(),
        descriptor.topology().clone(),
        descriptor.layer_layout().clone(),
        descriptor.layer_prefix_offsets().to_vec(),
        descriptor.state_segments().to_vec(),
    )
    .unwrap()
}

fn create_prompt_fixture_generation(root: &Path) -> PathBuf {
    let generation = root
        .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
        .join(TEST_PROMPT_CACHE_GENERATION);
    fs::create_dir_all(&generation).unwrap();
    fs::write(
        root.join(PROMPT_CACHE_CURRENT_FILE),
        format!("{TEST_PROMPT_CACHE_GENERATION}\n"),
    )
    .unwrap();
    generation
}

fn prompt_fixture_root(root: &Path) -> PathBuf {
    resolve_prompt_cache_root(root).unwrap()
}

fn prompt_fixture_manifest_path(root: &Path) -> PathBuf {
    prompt_fixture_root(root).join("manifest.json")
}

fn write_prompt_fixture(root: &Path, namespace: &str) -> PromptCacheManifest {
    let generation = create_prompt_fixture_generation(root);
    let keys = 1.0f32.to_le_bytes();
    let values = 2.0f32.to_le_bytes();
    let key_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &keys).unwrap();
    let value_view = TensorView::new(StoredDtype::F32, vec![1, 1, 1, 1], &values).unwrap();
    serialize_to_file(
        [("keys", key_view), ("values", value_view)],
        None,
        &generation.join("block.safetensors"),
    )
    .unwrap();
    let descriptor = prompt_descriptor();
    let manifest = PromptCacheManifest {
        schema_version: PROMPT_CACHE_SCHEMA_VERSION,
        model_family: descriptor.model_family().to_owned(),
        effective_model_type: descriptor.effective_model_type().to_owned(),
        checkpoint_fingerprint: descriptor.checkpoint_fingerprint().to_owned(),
        prefix_content_fingerprint: descriptor.prefix_content_fingerprint().to_owned(),
        architecture_fingerprint: descriptor.architecture_fingerprint().to_owned(),
        layer_count: 1,
        global_layer_start: 0,
        global_layer_end: 1,
        block_size_tokens: 1,
        batch_size: 1,
        total_prefix_tokens: 1,
        prefix_sha256: prompt_cache_token_fingerprint(&[7]),
        layer_layout: descriptor.layer_layout().clone(),
        sink_tokens: 0,
        layer_prefix_offsets: vec![0],
        state_segments: descriptor.state_segments().to_vec(),
        topology: PromptCacheTopology::default(),
        distributed_commit: None,
        application_namespace: Some(namespace.into()),
        blocks: vec![PromptCacheBlock {
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 1,
            rank: None,
            shard: "block.safetensors".into(),
            first_array: "keys".into(),
            second_array: "values".into(),
            first_shape: vec![1, 1, 1, 1],
            second_shape: vec![1, 1, 1, 1],
            first_dtype: "Float32".into(),
            second_dtype: "Float32".into(),
            logical_bytes: 8,
            payload_sha256: hash_prompt_cache_shard_payload(&generation.join("block.safetensors"))
                .unwrap(),
        }],
        state_tensors: Vec::new(),
    };
    fs::write(
        generation.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    manifest
}

fn write_fixed_state_fixture(root: &Path) -> PromptCacheManifest {
    let generation = create_prompt_fixture_generation(root);
    let values = (0..12)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(StoredDtype::F32, vec![1, 3, 4], &values).unwrap();
    let shard = generation.join("state.safetensors");
    serialize_to_file([("state", view)], None, &shard).unwrap();
    let policy = StateTensorPolicy::new(
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
    let manifest = PromptCacheManifest {
        schema_version: PROMPT_CACHE_SCHEMA_VERSION,
        model_family: "fixed".into(),
        effective_model_type: "fixed".into(),
        checkpoint_fingerprint: "checkpoint".into(),
        prefix_content_fingerprint: "state:prefix".into(),
        architecture_fingerprint: "architecture".into(),
        layer_count: 1,
        global_layer_start: 0,
        global_layer_end: 1,
        block_size_tokens: 1,
        batch_size: 1,
        total_prefix_tokens: 1,
        prefix_sha256: prompt_cache_token_fingerprint(&[7]),
        layer_layout: LayerSchedule::new(
            1,
            vec![LayerCachePolicy::fixed_only(vec![policy]).unwrap()],
        )
        .unwrap(),
        layer_prefix_offsets: vec![0],
        state_segments: vec![
            eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap(),
        ],
        sink_tokens: 0,
        topology: PromptCacheTopology::default(),
        distributed_commit: None,
        application_namespace: None,
        blocks: Vec::new(),
        state_tensors: vec![PromptCacheStateTensor {
            owner: StateTensorOwner::Layer(0),
            role: StateTensorRole::Convolution { slot: 0 },
            shard: "state.safetensors".into(),
            array: "state".into(),
            shape: vec![1, 3, 4],
            dtype: "Float32".into(),
            logical_bytes: 48,
            payload_sha256: hash_prompt_cache_shard_payload(&shard).unwrap(),
        }],
    };
    fs::write(
        generation.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    manifest
}

fn execution_key_value_block(stream: &Stream) -> CacheBlockArrays {
    let keys = Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap();
    let values = Array::ones::<f32>(&[1, 1, 2, 1], stream).unwrap();
    async_eval_with_event([&keys, &values])
        .unwrap()
        .synchronize()
        .unwrap();
    CacheBlockArrays::KeyValue { keys, values }
}

fn two_buffer_host_capacity(logical_bytes_each: usize) -> u64 {
    let capacity =
        host_transfer_capacity_upper_bound(logical_bytes_each, HostTransferPolicy::Transfer)
            .unwrap() as u64;
    capacity.checked_mul(2).unwrap()
}

fn f32_storage_pointers(arrays: &CacheBlockArrays) -> [usize; 2] {
    arrays
        .arrays()
        .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr() as usize)
}

fn backend_key_value_block(stream: &Stream) -> CacheBlockArrays {
    let keys = Array::ones::<f32>(&[1, 1, 1, 1], stream).unwrap();
    let values = Array::ones::<f32>(&[1, 1, 1, 1], stream)
        .unwrap()
        .multiply(Array::from(2.0f32), stream)
        .unwrap();
    async_eval_with_event([&keys, &values])
        .unwrap()
        .synchronize()
        .unwrap();
    CacheBlockArrays::KeyValue { keys, values }
}

fn assert_backend_key_value_block(arrays: &CacheBlockArrays) {
    let CacheBlockArrays::KeyValue { keys, values } = arrays else {
        panic!("expected key/value cache arrays");
    };
    assert_eq!(
        keys.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[1.0]
    );
    assert_eq!(
        values.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[2.0]
    );
}

fn exercise_backend_cache_lifecycle(device: Device, expected_storage: HostTransferStorageKind) {
    let stream = Stream::new_with_device(&device);
    let consumer = Stream::new_with_device(&device);
    let host_capacity = two_buffer_host_capacity(size_of::<f32>());
    let pool = CacheResidencyPool::new(
        CachePoolLimits::new(16, host_capacity * 2, host_capacity * 2, 16).unwrap(),
    );
    let options = || {
        PagedCacheOptions::new(1, 16, host_capacity, 1)
            .unwrap()
            .with_full_attention(true)
            .with_pool(pool.clone())
            .unwrap()
    };
    let manager = CacheResidencyManager::new(options()).unwrap();
    manager.bind_transfer_device(&stream).unwrap();
    let id = manager
        .seal_block(0, 0, 1, None, backend_key_value_block(&stream), false)
        .unwrap();
    let competing = CacheResidencyManager::new(options()).unwrap();
    competing.set_tail_state(0, 8, 1).unwrap();
    let aggregate_error = competing.set_tail_state(0, 12, 1).unwrap_err();
    assert!(matches!(
        aggregate_error,
        CacheResidencyError::Pool(CachePoolError::BudgetExceeded {
            resource: CachePoolResource::Device,
            required: 20,
            budget: 16,
        })
    ));

    let demotion = manager.begin_device_demotion(&id).unwrap();
    let in_flight = pool.report().unwrap();
    assert_eq!(in_flight.current_device_bytes, 16);
    assert_eq!(in_flight.current_host_bytes, host_capacity);
    assert_eq!(in_flight.current_transfer_in_flight_bytes, host_capacity);
    manager.finish_device_demotion(&demotion).unwrap();
    {
        let state = manager.lock().unwrap();
        let block = state.blocks.get(&id).unwrap().host_block().unwrap();
        for buffer in block.buffers() {
            assert_eq!(buffer.storage_kind().unwrap(), expected_storage);
        }
    }
    let demoted = pool.report().unwrap();
    assert_eq!(demoted.current_device_bytes, 8);
    assert_eq!(demoted.current_host_bytes, host_capacity);
    assert_eq!(demoted.current_transfer_in_flight_bytes, 0);

    let destination = tempfile::tempdir().unwrap();
    let cache_path = destination.path().join("backend-cache");
    manager
        .save_prompt_cache(
            &cache_path,
            prompt_descriptor(),
            &[7],
            &[],
            &PromptCacheOptions::new(Some("backend-verification".into()), false).unwrap(),
        )
        .unwrap();
    let (restored, manifest) = open_prompt_cache(
        &cache_path,
        &prompt_descriptor(),
        &prompt_model_identity(),
        &[7],
        options(),
    )
    .unwrap();
    assert_eq!(
        manifest.application_namespace.as_deref(),
        Some("backend-verification")
    );
    restored.bind_transfer_device(&stream).unwrap();
    let restored_id = restored
        .lock()
        .unwrap()
        .blocks
        .keys()
        .next()
        .unwrap()
        .clone();
    let transfer = restored
        .prepare_block_transfer(&restored_id, &stream)
        .unwrap();
    transfer.wait_on(&consumer).unwrap();
    let consumed = transfer.arrays().arrays()[0].square(&consumer).unwrap();
    async_eval_with_event([&consumed])
        .unwrap()
        .synchronize()
        .unwrap();
    assert_backend_key_value_block(transfer.arrays());
    drop(transfer);

    let restored_report = restored.report().unwrap();
    assert_eq!(restored_report.prompt_cache_loads, 1);
    assert_eq!(restored_report.disk_promotions, 1);
    assert!(restored_report.transfer_bytes >= 8);
    let aggregate = pool.report().unwrap();
    assert_eq!(aggregate.managers, 3);
    assert!(aggregate.current_device_bytes <= aggregate.limits.device_bytes());
    assert!(aggregate.current_host_bytes <= aggregate.limits.host_bytes());
    assert!(
        aggregate.current_transfer_in_flight_bytes <= aggregate.limits.transfer_in_flight_bytes()
    );

    manager.clear().unwrap();
    competing.clear().unwrap();
    restored.clear().unwrap();
    let cleared = pool.report().unwrap();
    assert_eq!(cleared.current_device_bytes, 0);
    assert_eq!(cleared.current_host_bytes, 0);
    assert_eq!(cleared.current_transfer_in_flight_bytes, 0);
    drop((manager, competing, restored));
    assert_eq!(pool.report().unwrap().managers, 0);
}
