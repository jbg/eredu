use super::*;
use crate::backend::{
    nn::workspace::MlxMetalWorkspaceMechanisms, runtime::cache::kv::KeyValueCache,
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::workspace::WorkspaceTensor;
use eredu_runtime::RuntimeLayerState;
use safemlx::{Device, DeviceType};
use std::cell::Cell;

fn layout() -> StateLayout {
    StateLayout::new(
        LayerSchedule::new(
            2,
            vec![
                LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap(),
                LayerCachePolicy::key_value(AttentionPolicy::sliding(8).unwrap(), 1, 2).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap()
}
fn source() -> MlxHybridState {
    MlxHybridState::device_with_global_layer_start(layout(), 17).unwrap()
}
fn unknown<T>(result: Result<T, Error>) {
    match result {
        Err(Error::Other(e)) => assert_eq!(
            e.downcast_ref::<WorkingMemoryError>(),
            Some(&WorkingMemoryError::UnknownBound)
        ),
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("unsupported payload was accepted"),
    }
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct Cold;
impl Drop for Cold {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn actual_outer_geometry_and_empty_children_prepare_without_native_work() {
    let source = source();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let _cold = Cold;
    HOUSEKEEPING.set(0);
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    assert_eq!(plan.len(), 2);
    let saved = source
        .layers
        .prepare_copy_slots()
        .unwrap()
        .for_destination::<SavedHybridGroupedLayer>()
        .unwrap();
    let dense = source
        .layers
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<MlxHybridLayerState>()
        .unwrap();
    assert!(saved
        .source_metadata()
        .same_storage(source.layers.metadata()));
    assert!(dense
        .source_metadata()
        .same_storage(source.layers.metadata()));
    assert!(std::ptr::eq(
        saved.source_at(0).unwrap(),
        &source.layers.slots()[0]
    ));
    assert_eq!(
        dense.retained_bytes(),
        (2 * std::mem::size_of::<MlxHybridLayerState>()) as u64
    );
    assert_eq!(
        saved.retained_bytes(),
        (2 * std::mem::size_of::<Option<SavedHybridGroupedLayer>>()) as u64
    );
    let children: u64 = source
        .layers
        .slots()
        .iter()
        .map(|layer| {
            layer
                .fixed
                .prepare_slots()
                .unwrap()
                .initialization_peak_bytes()
        })
        .sum();
    assert_eq!(children, 0, "empty children have no inline slot payload");
    let child_controls = eredu_runtime::working_memory::RegisteredDecoderHostCopy::<
        Slot,
        StorageIdentity,
    >::preparation_control_bytes(true)
    .unwrap()
    .checked_mul(plan.len())
    .unwrap();
    assert!(
        child_controls > 0,
        "each empty child still constructs metadata controls"
    );
    assert!(
        plan.host_copy_preparation_bytes().unwrap() > child_controls,
        "the parent preparation also includes its outer table and group controls"
    );
    assert_eq!(
        plan.host_copy_initialization_peak_bytes().unwrap(),
        saved.initialization_peak_bytes() + children
    );
    let dense_children: u64 = source
        .layers
        .slots()
        .iter()
        .map(|layer| {
            layer
                .fixed
                .prepare_slots()
                .unwrap()
                .for_dense_destination::<Slot>()
                .unwrap()
                .initialization_peak_bytes()
        })
        .sum();
    assert_eq!(
        plan.dense_initialization_peak_bytes_fixed().unwrap(),
        dense.initialization_peak_bytes() + dense_children
    );
    assert_eq!(plan.global_layer_start(), 17);
    assert!(plan.shared_layout().same_storage(&source.layout));
    assert!(source
        .layers
        .slots()
        .iter()
        .all(|s| s.fixed.is_empty() && s.fixed.payload_bytes() == Some(0)));
    assert_eq!(HOUSEKEEPING.get(), 0);
}

#[test]
fn absent_fixed_values_still_reject_real_role_payload_and_missing_attention() {
    let mut source = source();
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Convolution { slot: 0 },
        vec![StateTensorDimension::fixed(1).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    source.layers.slots_mut()[0].fixed =
        FixedStateSlots::from_policy(&LayerCachePolicy::FixedState {
            tensors: vec![fixed],
        })
        .unwrap();
    assert!(source.layers.slots()[0].fixed.values().all(Option::is_none));
    assert!(source.layers.slots()[0].fixed.payload_bytes().unwrap() > 0);
    unknown(PreparedHybridGroupedCopy::prepare(&source));
    source.layers.slots_mut()[0].fixed =
        FixedStateSlots::from_policy(&LayerCachePolicy::NoState).unwrap();
    source.layers.slots_mut()[0].attention = None;
    unknown(PreparedHybridGroupedCopy::prepare(&source));
    source.layers.slots_mut()[0].attention = Some(MlxHybridAttentionState::Compressed(
        CompressedLatentCache::new(),
    ));
    unknown(PreparedHybridGroupedCopy::prepare(&source));
}

fn populated(stream: &Stream) -> MlxHybridState {
    let mut source = source();
    let array = Array::from_slice(&[1_f32, 3., 5., 7., 9., 11.], &[1, 1, 3, 2]);
    for layer in source.layers.slots_mut() {
        let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) =
            &mut layer.attention
        else {
            unreachable!()
        };
        cache
            .update_and_fetch(array.clone(), array.clone(), stream)
            .unwrap();
        layer.fixed_offset = 4;
    }
    for array in source.retained_arrays() {
        array.evaluated().unwrap();
    }
    source
}
fn arrays(attention: Option<&MlxHybridAttentionState>) -> Vec<&Array> {
    let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) = attention
    else {
        unreachable!()
    };
    let mut values = Vec::new();
    cache
        .prepare_isolated_copy()
        .visit_operands(&mut |a| values.push(a));
    values
}
#[test]
fn numerical_group_copies_aliases_and_funds_each_empty_child_table() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = populated(&stream);
    super::publish(&source, &loading);
    drop(loading);
    super::settle(&pool, pool.fixture_host_charge().unwrap());
    let (sampler, preparation, run) = super::sampler(&pool);
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    let (sampling, ordinary_host, ordinary_complete, _required) =
        super::prepared(&plan, sampler.borrow_funded(), &pool);
    drop((ordinary_host, ordinary_complete));
    // Payload accounting is zero for an empty table. Its actual header and
    // identity instead retain the prospectively funded preparation account.
    use eredu_core::HostPreparationAuthority;
    use eredu_nn::workspace::HostMetadataFunding;
    use eredu_runtime::working_memory::WorkingMemoryStorage;
    let mut inventory = RetainedStorage::default();
    plan.visit_retained_arrays(&mut |array| inventory.include_array(array).unwrap());
    inventory
        .include_metadata(SharedHostMetadata::Layout(plan.shared_layout().clone()))
        .unwrap();
    let source_pin = inventory.source_pin_plan(&pool).unwrap();
    let metadata_pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let controls = [
        plan.host_copy_preparation_bytes().unwrap(),
        source_pin.requested_bytes(),
        plan.copy_account_layout().unwrap().requested_bytes(),
        WorkingMemoryStorage::<StorageIdentity>::copy_source_wrapper_bytes(
            1,
            1 + plan.registered_source_tables().unwrap(),
        )
        .unwrap(),
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .unwrap();
    let funding = metadata_pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(u64::MAX),
        )
        .unwrap();
    let account_controls = metadata_pool.fixture_host_charge().unwrap();
    funding.reserve_metadata(controls).unwrap();
    assert_eq!(
        metadata_pool.fixture_host_charge().unwrap(),
        account_controls + controls as u64
    );
    let preparation_owner = HostPreparationAuthority::retain(funding.clone());
    // Destination metadata uses the complete source's preparation authority.
    // Funding only the table-binding plans leaves the ordinary destination
    // constructor selected, so retain the same accepted owner on the real
    // existing-only source inventory as the production snapshot worker does.
    let complete = inventory
        .pin_registered_with_host(&pool, &preparation_owner)
        .unwrap();
    let host = plan
        .host_copy_with_preparation(&pool, Some(&preparation_owner))
        .unwrap();
    let (copied_sampler, slots, account) = host
        .admit_exact_fixture(
            &pool,
            sampling,
            complete,
            super::publication_roots(&plan),
            false,
        )
        .unwrap();
    let (custody, scope) = account.into_parts();
    let roots = RefCell::new(Vec::new());
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    assert_eq!(roots.borrow().len(), 8);
    super::finish(&saved, scope, &roots);
    let mut escaped_children = Vec::new();
    for index in 0..2 {
        let copied = saved.layers.get(index).unwrap();
        assert_eq!(copied.fixed_offset, 4);
        assert_eq!(copied.fixed.len(), 0);
        assert_eq!(copied.fixed.retained_bytes(), 0);
        assert_eq!(copied.fixed.protected_bytes(), 0);
        let metadata = copied
            .fixed
            .prepare_copy_slots()
            .unwrap()
            .source_metadata()
            .clone();
        assert_eq!(metadata.capacity_bytes(), Some(0));
        assert!(!metadata.same_storage(source.layers.slots()[index].fixed.metadata()));
        escaped_children.push(metadata);
        let old = arrays(source.layers.slots()[index].attention.as_ref());
        let new = arrays(copied.attention.as_ref());
        for (a, b) in old.iter().zip(&new) {
            assert_eq!(
                a.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                b.evaluated().unwrap().try_to_vec::<f32>().unwrap()
            );
            assert_ne!(
                a.allocation_info().unwrap().unwrap().identity(),
                b.allocation_info().unwrap().unwrap().identity()
            );
        }
        assert_ne!(
            new[0].allocation_info().unwrap().unwrap().identity(),
            new[1].allocation_info().unwrap().unwrap().identity()
        );
        assert_eq!(
            old[0].evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            vec![1., 3., 5., 7., 9., 11.]
        );
    }
    assert_eq!(saved.global_layer_start(), 17);
    assert!(saved.shared_layout().same_storage(&source.layout));
    drop((
        saved,
        copied_sampler,
        custody,
        roots,
        source,
        sampler,
        preparation,
        run,
        preparation_owner,
        funding,
        inventory,
    ));
    super::settle(&pool, 0);
    assert!(
        metadata_pool.fixture_host_charge().unwrap() > 0,
        "escaped empty-child headers retain their actual preparation account"
    );
    drop(escaped_children.pop());
    assert!(
        metadata_pool.fixture_host_charge().unwrap() > 0,
        "either independently constructed child keeps the common preparation alive"
    );
    drop(escaped_children);
    assert_eq!(metadata_pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn copied_workspace_uses_independent_roots_and_preserves_full_sliding_controls() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = populated(&stream);
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let copied = plan
        .project_dense_workspace(NonZeroU32::new(1).unwrap(), &context)
        .unwrap();
    assert!(copied.source_storage.is_complete());
    assert_eq!(copied.source_storage.iter().len(), 1);
    let roots: Vec<WorkspaceTensor> = copied
        .state
        .as_ref()
        .iter()
        .flat_map(RuntimeLayerState::<WorkspaceBackend>::retained_values)
        .cloned()
        .collect();
    assert_eq!(roots.len(), 4);
    context.begin_state_span(&roots).unwrap();
    let retained = context
        .report(&roots)
        .unwrap()
        .state
        .unwrap()
        .retained_bytes
        .unwrap();
    let individual: u64 = roots
        .iter()
        .map(|root| {
            context
                .report(std::slice::from_ref(root))
                .unwrap()
                .state
                .unwrap()
                .retained_bytes
                .unwrap()
        })
        .sum();
    assert!(retained > 0);
    assert_eq!(
        retained, individual,
        "every copied role has an independent bounded root"
    );
    assert!(copied.copy.state.as_ref().unwrap().retained_bytes.is_some());
    assert!(copied
        .copy
        .state
        .as_ref()
        .unwrap()
        .transient_bytes
        .is_some());
    assert_eq!(
        eredu_nn::AttentionCache::offset(&source.layers.slots()[1]),
        3
    );
    assert_eq!(copied.state.layout(), source.layout.layout());
}

#[test]
fn all_kv_paged_ordinary_and_fixed_copy_inspect_the_same_source_without_native_work() {
    use crate::backend::runtime::cache::state::PreparedResidentDecoderCopy;
    let manager = CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 4096, 4096, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let source =
        MlxHybridState::paged_with_global_layer_start(layout(), manager, None, 17).unwrap();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let _cold = Cold;
    HOUSEKEEPING.set(0);
    let ordinary = PreparedResidentDecoderCopy::hybrid(&source).unwrap();
    let fixed = PreparedResidentDecoderCopy::hybrid_fixed(&source).unwrap();
    assert!(ordinary.is_paged());
    assert!(fixed.is_paged());
    assert!(ordinary
        .shared_layout()
        .unwrap()
        .same_storage(&source.layout));
    assert!(fixed.shared_layout().unwrap().same_storage(&source.layout));
    assert_eq!(ordinary.global_layer_start(), Some(17));
    assert_eq!(fixed.global_layer_start(), Some(17));
    let mut children = 0;
    ordinary
        .visit_registered_child_metadata(&mut |metadata| {
            assert!(metadata.same_storage(source.layers.slots()[children].fixed.metadata()));
            children += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(children, 2);
    let ordinary = ordinary.into_dense_fixed().unwrap();
    let fixed = fixed.into_dense_fixed().unwrap();
    assert_eq!(
        ordinary.initialization_peak_bytes_fixed().unwrap(),
        fixed.initialization_peak_bytes_fixed().unwrap()
    );
    assert_eq!(
        ordinary.logical_snapshot_estimate(),
        fixed.logical_snapshot_estimate()
    );
    assert_eq!(HOUSEKEEPING.get(), 0);
}
