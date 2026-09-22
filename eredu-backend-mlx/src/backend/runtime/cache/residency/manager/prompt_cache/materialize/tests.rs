use super::*;
use crate::backend::nn::workspace::{ExistingArrayProjection, registered_storage_row};
use crate::backend::runtime::residency::storage::StorageIdentity;
use crate::backend::{
    MlxAcceleratorFamily, MlxBackend, MlxDeviceIdentity,
    managed_memory::gpu_stream::PreparedExecutionStreams,
};
use eredu_core::cache::{PromptCacheStateTensor, StateTensorOwner, StateTensorRole};
use eredu_runtime::working_memory::{DependencyMemoryPolicy, RegisteredWorkspaceStorageLayout};
use safemlx::{Device, DeviceType};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

#[test]
#[ignore = "requires the native constructor, canonical source directory and completion worker"]
fn imported_fixed_state_keeps_canonical_source_across_aliases_and_retires_host_staging() {
    run(DeviceType::Cpu, "imported-fixed-state-source-cpu");
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
#[ignore = "requires actual Metal constructor and allocation-free descriptor completion"]
fn imported_fixed_state_metal_keeps_canonical_and_empty_source_custody() {
    run(DeviceType::Gpu, "imported-fixed-state-source-metal");
}

fn run(device: DeviceType, child: &'static str) {
    if !crate::tests::support::native_process::enter(child) {
        return;
    }
    let ledger = crate::tests::support::test_utils::initialize_original_sources();
    let streams = PreparedExecutionStreams::for_device_factory(&ledger, device)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(
            &Device::new(device, 0),
            (device == DeviceType::Gpu).then_some(MlxAcceleratorFamily::Metal),
        )
        .unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    let execution = InferenceExecutionIdentity::default();
    let account = ledger
        .prepare_workspace_metadata(&execution, ledger.configured_limits().clone())
        .unwrap();
    let dependency = ledger
        .prepare_workspace_metadata(&execution, ledger.configured_limits().clone())
        .unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(
        MlxMetalWorkspaceMechanisms::current_host().unwrap(),
        account.clone(),
    )
    .unwrap();
    let funding = PromptCachePersistenceFunding::new(
        &context,
        dependency,
        DependencyMemoryPolicy::default(),
        eredu_runtime::cache::DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT,
    )
    .unwrap();
    let materialization = PromptCacheMaterialization::new(
        &ledger,
        &execution,
        environment.input_runtime().unwrap(),
        &context,
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("recurrent.safetensors");
    let expected = [2.5f32, -7.0, 11.25, 13.0];
    let payload: Vec<_> = expected.into_iter().flat_map(f32::to_le_bytes).collect();
    serialize_to_file(
        [(
            "state",
            TensorView::new(Dtype::F32, vec![2, 2], &payload).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    let declaration = PromptCacheStateTensor {
        owner: StateTensorOwner::Layer(0),
        role: StateTensorRole::Recurrent,
        shard: "recurrent.safetensors".into(),
        array: "state".into(),
        shape: vec![2, 2],
        dtype: "Float32".into(),
        logical_bytes: payload.len() as u64,
        payload_sha256: eredu_runtime::cache::hash_prompt_cache_shard_payload(&path).unwrap(),
    };
    let source = PersistentCacheStateTensor::read_with(
        File::open(&path).unwrap(),
        &path,
        &declaration,
        &funding,
        |bytes| Staging::new(materialization.body(), bytes, |_| 0),
    )
    .unwrap();
    let completed = construct::<f32, 4>(
        &source,
        &materialization,
        environment.stream(),
        f32::from_le_bytes,
    )
    .unwrap();
    let array = publication::publish(completed, materialization.body()).unwrap();
    assert_eq!(array.evaluated().unwrap().as_slice::<f32>(), expected);
    let allocation = array.try_allocation_info().unwrap().unwrap();
    let key = StorageIdentity::Native(allocation.identity());
    let host_charge = || {
        ledger
            .snapshot()
            .unwrap()
            .domains
            .into_iter()
            .find(|row| row.domain == ledger.topology().host_domain())
            .unwrap()
            .current_charge_bytes
    };
    let with_file_staging = host_charge();
    drop(source);
    // The independent payload and its own account metadata retire together.
    assert!(with_file_staging - host_charge() >= file_bytes);
    let alias = array.try_clone_handle().unwrap();
    drop(array);
    drop(materialization);
    assert_eq!(alias.evaluated().unwrap().as_slice::<f32>(), expected);
    // The ordinary copy binder sees the actual canonical rows without a state-
    // specific source bank or the constructor's completed-source wrapper.
    let mut projection = ExistingArrayProjection::with_source_count(&context, 1).unwrap();
    projection.project(&alias).unwrap();
    let layout = RegisteredWorkspaceStorageLayout::<StorageIdentity>::new(1).unwrap();
    context.charge_metadata(layout.requested_bytes()).unwrap();
    let pin = layout
        .construct(
            &ledger,
            &context,
            projection
                .storage_roots()
                .map(|(identity, _, root)| registered_storage_row(identity, root)),
        )
        .unwrap();
    assert!(ledger.registered_allocation(&key).unwrap().is_some());
    drop((pin, projection));
    let retained = host_charge();
    drop(alias);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::try_retire_completed_submissions().unwrap();
        safemlx::reclaim_allocation_owners();
        assert!(std::time::Instant::now() < deadline);
        ledger.registered_allocation(&key).unwrap().is_none()
    });
    assert!(host_charge() < retained);
    assert!(path.exists());

    let empty_path = root.path().join("empty.safetensors");
    serialize_to_file(
        [(
            "state",
            TensorView::new(Dtype::F32, vec![2, 0], &[]).unwrap(),
        )],
        None,
        &empty_path,
    )
    .unwrap();
    let mut empty_declaration = declaration;
    empty_declaration.shape = vec![2, 0];
    empty_declaration.logical_bytes = 0;
    empty_declaration.payload_sha256 =
        eredu_runtime::cache::hash_prompt_cache_shard_payload(&empty_path).unwrap();
    let materialization = PromptCacheMaterialization::new(
        &ledger,
        &execution,
        environment.input_runtime().unwrap(),
        &context,
    )
    .unwrap();
    let empty_source = PersistentCacheStateTensor::read_with(
        File::open(&empty_path).unwrap(),
        &empty_path,
        &empty_declaration,
        &funding,
        |bytes| Staging::new(materialization.body(), bytes, |_| 0),
    )
    .unwrap();
    let completed = construct::<f32, 4>(
        &empty_source,
        &materialization,
        environment.stream(),
        f32::from_le_bytes,
    )
    .unwrap();
    let empty = publication::publish(completed, materialization.body()).unwrap();
    if device == DeviceType::Gpu {
        assert!(matches!(
            empty.inspect_ordinary_buffer().unwrap(),
            safemlx::OrdinaryBufferInspection::Empty
        ));
    }
    let empty_alias = empty.try_clone_handle().unwrap();
    drop((empty, empty_source, materialization));
    assert_eq!(empty_alias.shape(), [2, 0]);
    assert!(
        empty_alias
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .is_empty()
    );
    drop(empty_alias);
    safemlx::reclaim_allocation_owners();
}
