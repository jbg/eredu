use super::*;
use crate::backend::runtime::cache::state::resident_copy::host_copy::copy_slots;
use eredu_core::cache::{
    MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
};
use eredu_core::{AttentionPolicy, HostPreparationAuthority, LayerSchedule};
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::{
    PagedCacheOptions,
    working_memory::{InferenceExecutionIdentity, WorkingMemoryPool},
};
use safemlx::{Device, DeviceType};

fn fixed() -> StateTensorPolicy {
    StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![StateTensorDimension::fixed(2).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap()
}
fn layout(attention: bool) -> StateLayout {
    let policy = if attention {
        LayerCachePolicy::key_value_with_fixed_state(AttentionPolicy::Full, 1, 1, vec![fixed()])
            .unwrap()
    } else {
        LayerCachePolicy::fixed_only(vec![fixed()]).unwrap()
    };
    StateLayout::new(LayerSchedule::new(2, vec![policy, LayerCachePolicy::NoState]).unwrap())
        .unwrap()
}
fn source(attention: bool) -> MlxHybridState {
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    MlxHybridState::paged_with_global_layer_start(
        layout(attention),
        CacheResidencyManager::new(options).unwrap(),
        None,
        17,
    )
    .unwrap()
}
fn funding(pool: &WorkingMemoryPool) -> HostMetadataFunding {
    pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 24)
        .unwrap()
}
// Exercise the actual paid manager, table and per-layer constructors directly.
// Core reset claim/comparison/publication remains covered by the neutral reset suite.
fn construct(source: &MlxHybridState, funding: &HostMetadataFunding) -> MlxHybridState {
    let (plan, bytes) = source.resident_reset_plan().unwrap();
    assert!(bytes > 0);
    let mut context = source
        .prepare_resident_reset_context(&plan, Some(funding))
        .unwrap();
    funding
        .reserve_metadata(
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
        )
        .unwrap();
    let host = HostPreparationAuthority::retain(funding.clone());
    let layers = copy_slots(
        source
            .layers
            .prepare_copy_slots()
            .unwrap()
            .for_dense_destination()
            .unwrap(),
        &host,
        funding,
        |index, layer| {
            let child = copy_slots(
                layer
                    .fixed
                    .table()
                    .prepare_copy_slots()
                    .unwrap()
                    .for_dense_destination()
                    .unwrap(),
                &host,
                funding,
                |_, child| Ok(MlxHybridState::empty_resident_reset_child(child)),
            )?;
            MlxHybridState::empty_resident_reset_layer_prepared(
                &mut context,
                layer,
                source.layout.layout().layer(index).unwrap(),
                Some(child),
            )
            .map_err(crate::backend::Error::StorageSource)
        },
    )
    .unwrap();
    MlxHybridState::from_resident_reset(
        &mut context,
        source.layout.clone(),
        source.global_layer_start,
        layers,
    )
}
#[test]
#[ignore = "requires native paged cache sources"]
fn hybrid_paged_reset_preserves_nonzero_sources_fixed_children_and_independent_manager_custody() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut source = source(true);
    let input = Array::from_slice(&[1.25_f32, -2.5, 3.75], &[1, 1, 3, 1]);
    let cache = match source.layers.slots_mut()[0].attention.as_mut().unwrap() {
        MlxHybridAttentionState::KeyValue(cache) => cache,
        _ => unreachable!(),
    };
    drop(KeyValueCache::update_for_attention(cache, input.clone(), input, &stream).unwrap());
    let fixed = MlxTensor::from_array(Array::from_slice(&[7.5_f32, -9.25], &[2]));
    *source.layers.slots_mut()[0]
        .fixed
        .get_mut(&StateTensorRole::Recurrent)
        .unwrap() = Some(fixed.clone());
    source.layers.slots_mut()[0].fixed_offset = 3;
    for value in source.retained_arrays() {
        value.evaluated().unwrap();
    }
    let old_values: Vec<_> = source
        .retained_arrays()
        .into_iter()
        .map(|v| v.evaluated().unwrap().try_to_vec::<f32>().unwrap())
        .collect();
    assert!(old_values.iter().flatten().any(|v| *v != 0.));
    let old_manager = source.manager.as_ref().unwrap().clone();
    let old_report = old_manager.report().unwrap();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let funding = funding(&pool);
    let reset = construct(&source, &funding);
    assert!(reset.layout.same_storage(&source.layout));
    assert_eq!(reset.global_layer_start, 17);
    let manager = reset.manager.as_ref().unwrap();
    assert_ne!(manager.session_id(), old_manager.session_id());
    assert_eq!(manager.pool(), old_manager.pool());
    assert_eq!(manager.report().unwrap().current_device_bytes, 0);
    for layer in reset.layers.slots() {
        assert_eq!(layer.fixed_offset, 0);
        assert!(
            layer
                .fixed
                .table()
                .slots()
                .iter()
                .all(|(_, value)| value.is_none())
        );
        if let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) =
            &layer.attention
        {
            assert_eq!(KeyValueCache::offset(cache), 0);
            assert_eq!(cache.global_layer(), 17);
            assert!(cache.manager().same_catalog(manager));
        }
    }
    assert_eq!(
        reset.layers.slots()[0].fixed.table().slots()[0].0,
        StateTensorRole::Recurrent
    );
    assert!(reset.layers.slots()[1].fixed.is_empty());
    assert_eq!(source.layers.slots()[0].fixed_offset, 3);
    assert_eq!(
        source
            .retained_arrays()
            .into_iter()
            .map(|v| v.evaluated().unwrap().try_to_vec::<f32>().unwrap())
            .collect::<Vec<_>>(),
        old_values
    );
    assert_eq!(old_manager.report().unwrap(), old_report);
    let escaped_manager = manager.clone();
    let escaped_empty_child = reset.layers.slots()[1].fixed.metadata().clone();
    drop((reset, funding, source, old_manager));
    assert_eq!(
        fixed
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        vec![7.5, -9.25]
    );
    assert!(pool.used_bytes().unwrap() > 0);
    drop(escaped_manager);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "even an empty child retains its actual table funding"
    );
    drop(escaped_empty_child);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn hybrid_reset_component_placement_source_identity_and_empty_local_manager_are_exact() {
    use eredu_runtime::StateComponentPlacement::{Device, Paged};
    let mut source = source(true);
    let layer = &source.layers.slots()[0];
    for role in [
        StateComponentRole::AttentionKeys,
        StateComponentRole::AttentionValues,
    ] {
        assert!(MlxHybridState::validate_resident_reset_placement(
            layer, role, Paged
        ));
        assert!(!MlxHybridState::validate_resident_reset_placement(
            layer, role, Device
        ));
    }
    assert!(MlxHybridState::validate_resident_reset_placement(
        layer,
        StateComponentRole::Fixed(StateTensorRole::Recurrent),
        Device
    ));
    assert!(!MlxHybridState::validate_resident_reset_placement(
        layer,
        StateComponentRole::Fixed(StateTensorRole::Recurrent),
        Paged
    ));
    let (plan, _) = source.resident_reset_plan().unwrap();
    let short_pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let short = funding(&short_pool);
    // Construct the real account first, then occupy exactly its remaining
    // configured capacity so the reset's first constructor is the refusal.
    short
        .reserve_metadata(usize::try_from((1 << 24) - short_pool.used_bytes().unwrap()).unwrap())
        .unwrap();
    assert_eq!(short_pool.used_bytes().unwrap(), 1 << 24);
    let before_report = source.manager.as_ref().unwrap().report().unwrap();
    assert!(
        source
            .prepare_resident_reset_context(&plan, Some(&short))
            .is_err()
    );
    assert_eq!(
        source.manager.as_ref().unwrap().report().unwrap(),
        before_report
    );
    drop(short);
    assert_eq!(short_pool.used_bytes().unwrap(), 0);
    let foreign = super::tests::source(true);
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let funding = funding(&pool);
    assert!(
        plan.prepare(source.manager.as_ref(), 0, Some(&funding))
            .is_err(),
        "the same manager cannot authenticate a different constructor control layout"
    );
    assert!(
        foreign
            .prepare_resident_reset_context(&plan, Some(&funding))
            .is_err()
    );
    source.global_layer_start += 1;
    assert!(matches!(
        source.resident_reset_plan(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    source.global_layer_start -= 1;
    let source_manager = source.manager.take().unwrap();
    assert!(!source.validate_resident_reset_state());
    source.manager = Some(source_manager);
    let local = super::tests::source(false);
    let before = local.manager.as_ref().unwrap().session_id();
    let reset = construct(&local, &funding);
    assert!(
        reset
            .layers
            .slots()
            .iter()
            .all(|layer| layer.attention.is_none())
    );
    assert_ne!(
        reset.manager.as_ref().unwrap().session_id(),
        before,
        "a selected manager survives even when this local partition has no paged attention rows"
    );
    let repeated = construct(&reset, &funding);
    assert_ne!(
        reset.manager.as_ref().unwrap().session_id(),
        repeated.manager.as_ref().unwrap().session_id()
    );
    drop((source, local, reset, repeated, funding));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
