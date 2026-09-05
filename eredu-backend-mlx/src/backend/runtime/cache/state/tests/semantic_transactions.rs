use super::*;
use eredu_core::{
    cache::{
        CacheRepresentation, MutableStateResidency, PromptCacheModelIdentity,
        PromptCacheStateSegment, PromptCacheTopology, StateResidencyClass, StateTensorDimension,
        StateTensorDtype, StateTensorPolicy,
    },
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::NeuralOperatorCapabilities;
use eredu_runtime::{
    select_replicated_text_realization, ArchitectureGroupKind, ArchitectureGroupPlacement,
    ArchitectureGroupTransport, ArchitectureMergeDestination, BackendMechanismCapabilities,
    CacheResidencyPolicy, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    LayerWeightResidency, PagedCacheOptions, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, ReplicatedTextStateAccess, StateComponentMechanism,
    StateMechanismCapabilities, StateSegmentLifetime, WeightResidencyMechanism,
};

fn layout(window: Option<i32>) -> StateLayout {
    StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value(
                AttentionPolicy::from_sliding_window(window).unwrap(),
                1,
                8,
            )
            .unwrap()],
        )
        .unwrap(),
    )
    .unwrap()
}

fn manager() -> CacheResidencyManager {
    CacheResidencyManager::new(
        PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap()
}

fn selected_state(
    layout: StateLayout,
    access: ReplicatedTextStateAccess,
    policy: CacheResidencyPolicy,
) -> SelectedStateRealization {
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [layout.len()]).unwrap();
    let requirements = ReplicatedTextRequirements::new(
        "test.mlx-selected-state",
        NeuralOperatorCapabilities::NONE,
        graph,
        units,
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Decoder,
            first_owner_static_roles: Vec::new(),
            last_owner_static_roles: Vec::new(),
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        layout,
        access,
        Vec::new(),
    )
    .unwrap();
    let components = (0..requirements.state_layout().len()).flat_map(|layer| {
        requirements
            .state_layout()
            .components(layer)
            .unwrap()
            .iter()
            .cloned()
            .map(move |component| {
                let paged = match component.residency() {
                    StateResidencyClass::SealablePaged => StateComponentPlacement::Paged,
                    StateResidencyClass::AlwaysDeviceMutable
                    | StateResidencyClass::LayerScopedOffloadable => {
                        StateComponentPlacement::Device
                    }
                };
                StateComponentMechanism::new(
                    layer,
                    component,
                    Some(StateComponentPlacement::Device),
                    Some(paged),
                )
            })
    });
    let state = StateMechanismCapabilities::new(components)
        .with_transactions(true, true)
        .with_reset(true);
    let capabilities = BackendMechanismCapabilities::new(
        NeuralOperatorCapabilities::NONE,
        Vec::new(),
        vec![WeightResidencyMechanism::Resident],
        state,
    );
    select_replicated_text_realization(
        &requirements,
        &ReplicatedTextSelectionRequest::new(LayerWeightResidency::FullyResident, policy),
        &capabilities,
    )
    .unwrap()
    .state()
    .clone()
}

fn segmented_layout() -> StateLayout {
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap();
    StateLayout::segmented(
        LayerSchedule::new(2, vec![policy.clone(), policy]).unwrap(),
        [
            StateSegmentSpec::new("temporal", 0..1, StateSegmentLifetime::Persistent, 0).unwrap(),
            StateSegmentSpec::new("depth", 1..2, StateSegmentLifetime::FrameLocal, 0).unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn key_value_constructor_uses_selected_component_placement() {
    let layout = layout(Some(4));
    let device = selected_state(
        layout.clone(),
        ReplicatedTextStateAccess::KeyValue,
        CacheResidencyPolicy::Device,
    );
    let device = MlxKeyValueState::from_selected(&device, None, None).unwrap();
    assert!(matches!(
        device.as_ref(),
        [MlxKeyValueLayerState::Device(_)]
    ));

    let paged_policy = CacheResidencyPolicy::Paged(
        PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    );
    let paged = selected_state(layout, ReplicatedTextStateAccess::KeyValue, paged_policy);
    assert!(MlxKeyValueState::from_selected(&paged, None, None)
        .unwrap_err()
        .to_string()
        .contains("no residency manager"));
    let paged = MlxKeyValueState::from_selected(&paged, Some(manager()), None).unwrap();
    assert!(matches!(paged.as_ref(), [MlxKeyValueLayerState::Paged(_)]));
}

#[test]
fn hybrid_constructor_preserves_paged_attention_and_device_fixed_state() {
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![StateTensorDimension::fixed(4).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    let policy =
        LayerCachePolicy::key_value_with_fixed_state(AttentionPolicy::Full, 1, 8, vec![fixed])
            .unwrap();
    let layout = StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap();
    let paged_policy = CacheResidencyPolicy::Paged(
        PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    );
    let selected = selected_state(
        layout,
        ReplicatedTextStateAccess::AttentionWithFixed,
        paged_policy,
    );
    assert_eq!(
        selected
            .components()
            .iter()
            .map(SelectedStateComponentRealization::placement)
            .collect::<Vec<_>>(),
        vec![
            StateComponentPlacement::Paged,
            StateComponentPlacement::Paged,
            StateComponentPlacement::Device,
        ]
    );

    let state = MlxHybridState::from_selected(&selected, Some(manager()), None).unwrap();
    assert!(matches!(
        state.layers[0].attention,
        Some(MlxHybridAttentionState::KeyValue(
            MlxKeyValueLayerState::Paged(_)
        ))
    ));
    assert!(state.layers[0]
        .fixed
        .contains_key(&StateTensorRole::Recurrent));
}

#[test]
#[ignore = "requires MLX native array persistence"]
fn fixed_only_hybrid_prompt_cache_round_trips_without_attention() {
    use crate::backend::runtime::cache::residency::{
        load_prompt_cache_state_tensors, open_prompt_cache,
    };
    use crate::native::ExecutionContext;
    use safemlx::{Device, DeviceType};

    let role = StateTensorRole::Convolution { slot: 0 };
    let fixed = StateTensorPolicy::new(
        role,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(2).unwrap(),
            StateTensorDimension::fixed(3).unwrap(),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::fixed_only(vec![fixed]).unwrap()]).unwrap(),
    )
    .unwrap();
    let paging = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let selected = selected_state(
        layout.clone(),
        ReplicatedTextStateAccess::Fixed,
        CacheResidencyPolicy::Paged(paging.clone()),
    );
    let mut state = MlxHybridState::from_selected(&selected, Some(manager()), None).unwrap();
    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    state.layers[0].fixed.insert(
        role,
        Some(MlxTensor::from_array(Array::from_slice(
            &values,
            &[1, 2, 3],
        ))),
    );
    state.layers[0].fixed_offset = 3;

    let segment = PromptCacheStateSegment::new("state", 0..1).unwrap();
    let identity = PromptCacheModelIdentity::new(
        "test-fixed-only",
        "test-fixed-only",
        "test-fixed-only-architecture",
        1,
        0,
        1,
        0,
        PromptCacheTopology::default(),
        layout.layers().clone(),
        vec![0],
        vec![segment],
    )
    .unwrap();
    let descriptor = PromptCacheDescriptor::from_model_identity(
        identity.clone(),
        "test-fixed-only-checkpoint",
        "tokens:1,2,3",
        1,
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("cache");
    let tokens = [1_u32, 2, 3];
    let manifest = state
        .save_prompt_cache(
            &destination,
            descriptor.clone(),
            &tokens,
            &PromptCacheOptions::default(),
        )
        .unwrap();
    assert!(manifest.blocks.is_empty());
    assert_eq!(manifest.state_tensors.len(), 1);

    let (restored_manager, manifest) =
        open_prompt_cache(&destination, &descriptor, &identity, &tokens, paging).unwrap();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let tensors = load_prompt_cache_state_tensors(&destination, &manifest, stream).unwrap();
    let mut restored =
        MlxHybridState::from_selected(&selected, Some(restored_manager), None).unwrap();
    restored
        .restore_prompt_cache_state(tensors, 3, &[0])
        .unwrap();
    assert_eq!(restored.layers[0].position(), 3);
    let restored = restored.layers[0].fixed[&role]
        .as_ref()
        .unwrap()
        .as_array()
        .evaluated()
        .unwrap();
    assert_eq!(restored.as_slice::<f32>(), values);
}

#[test]
fn device_transaction_uses_an_independent_semantic_branch() {
    let mut canonical = MlxKeyValueState::device(layout(Some(4))).unwrap();
    let branch = canonical.branch().unwrap();

    assert!(canonical.permits_parallel_branches());
    assert!(canonical.has_same_transaction_identity(&branch));
    canonical.commit_branch(branch).unwrap();
}

#[test]
fn hybrid_commit_clones_only_the_architecture_named_segment() {
    let layout = StateLayout::segmented(
        LayerSchedule::new(3, vec![LayerCachePolicy::NoState; 3]).unwrap(),
        [
            StateSegmentSpec::new("target", 0..2, StateSegmentLifetime::Persistent, 0).unwrap(),
            StateSegmentSpec::new("prediction", 2..3, StateSegmentLifetime::Persistent, -1)
                .unwrap(),
        ],
    )
    .unwrap();
    let mut canonical = MlxHybridState::device(layout.clone()).unwrap();
    let mut draft = MlxHybridState::device(layout).unwrap();
    canonical.layers[0].fixed_offset = 1;
    canonical.layers[1].fixed_offset = 2;
    canonical.layers[2].fixed_offset = 3;
    draft.layers[0].fixed_offset = 10;
    draft.layers[1].fixed_offset = 20;
    draft.layers[2].fixed_offset = 30;

    canonical.commit_segment_from(&draft, "prediction").unwrap();

    assert_eq!(canonical.layers[0].fixed_offset, 1);
    assert_eq!(canonical.layers[1].fixed_offset, 2);
    assert_eq!(canonical.layers[2].fixed_offset, 30);
}

#[test]
fn transaction_rejects_layout_and_residency_changes() {
    let canonical = MlxKeyValueState::device(layout(Some(4))).unwrap();
    let different_layout = MlxKeyValueState::device(layout(Some(8))).unwrap();
    assert!(!canonical.has_same_transaction_identity(&different_layout));

    let paged = MlxKeyValueState::paged(layout(Some(4)), manager(), None).unwrap();
    assert!(!canonical.has_same_transaction_identity(&paged));
}

#[test]
fn paged_transaction_discard_restores_shared_manager_frontier() {
    let canonical = MlxKeyValueState::paged(layout(None), manager(), None).unwrap();
    assert!(!canonical.permits_parallel_branches());
    let branch = canonical.branch().unwrap();
    let manager = match &branch.state.layers[0] {
        MlxKeyValueLayerState::Paged(cache) => cache.manager().clone(),
        MlxKeyValueLayerState::Device(_) => unreachable!(),
    };
    manager.set_tail_state(0, 0, 3).unwrap();
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 3);

    MlxKeyValueState::discard_branch(branch).unwrap();

    assert_eq!(canonical.offset(), 0);
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 0);
}

#[test]
fn paged_transaction_rejects_segment_reset_before_shared_page_mutation() {
    let canonical = MlxKeyValueState::paged(segmented_layout(), manager(), None).unwrap();
    let mut branch = canonical.branch().unwrap();

    let error = branch
        .reset_segment(&StateSegmentId::new("depth").unwrap())
        .unwrap_err();

    assert!(error.to_string().contains("copy-on-write page ownership"));
    MlxKeyValueState::discard_branch(branch).unwrap();
    assert_eq!(canonical.offset(), 0);
}

#[test]
#[ignore = "requires local MLX execution"]
fn paged_depth_segment_reset_preserves_temporal_pages_and_later_rollback() {
    let manager = manager();
    let mut canonical = MlxKeyValueState::paged(segmented_layout(), manager.clone(), None).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for (layer, value) in [(0usize, 1.0f32), (1, 2.0)] {
        canonical.layers[layer]
            .update_and_fetch(
                Array::from_slice(&[value; 5], &[1, 1, 5, 1]),
                Array::from_slice(&[value + 10.0; 5], &[1, 1, 5, 1]),
                &stream,
            )
            .unwrap();
    }
    assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
    assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 5);
    assert!(!manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap()
        .is_empty());
    assert!(!manager
        .layer_block_ids(1, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap()
        .is_empty());

    canonical
        .reset_segment(&StateSegmentId::new("depth").unwrap())
        .unwrap();

    assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
    assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 0);
    assert!(!manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap()
        .is_empty());
    assert!(manager
        .layer_block_ids(1, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap()
        .is_empty());

    let mut discarded = canonical.branch().unwrap();
    discarded.layers[1]
        .update_and_fetch(
            Array::from_slice(&[3.0f32; 2], &[1, 1, 2, 1]),
            Array::from_slice(&[4.0f32; 2], &[1, 1, 2, 1]),
            &stream,
        )
        .unwrap();
    MlxKeyValueState::discard_branch(discarded).unwrap();
    assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
    assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 0);

    let mut resumed = canonical.branch().unwrap();
    resumed.layers[1]
        .update_and_fetch(
            Array::from_slice(&[5.0f32; 2], &[1, 1, 2, 1]),
            Array::from_slice(&[6.0f32; 2], &[1, 1, 2, 1]),
            &stream,
        )
        .unwrap();
    canonical.commit_branch(resumed).unwrap();
    assert_eq!(KeyValueCache::offset(&canonical.layers[0]), 5);
    assert_eq!(KeyValueCache::offset(&canonical.layers[1]), 2);
}

#[test]
fn paged_sliding_transaction_fails_before_branch_publication() {
    let canonical = MlxKeyValueState::paged(layout(Some(4)), manager(), None).unwrap();
    let error = canonical.branch().unwrap_err();
    assert!(error.to_string().contains("paged sliding attention"));
}

#[test]
#[ignore = "requires local MLX execution"]
fn mlx_realtime_transaction_paged_rollback_release_resume() {
    let mut canonical = MlxKeyValueState::paged(layout(None), manager(), None).unwrap();
    let mut branch = canonical.branch().unwrap();
    let manager = match &branch.state.layers[0] {
        MlxKeyValueLayerState::Paged(cache) => cache.manager().clone(),
        MlxKeyValueLayerState::Device(_) => unreachable!(),
    };
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let keys = Array::from_slice(&[1.0_f32; 5], &[1, 1, 5, 1]);
    let values = Array::from_slice(&[2.0_f32; 5], &[1, 1, 5, 1]);
    branch.state.layers[0]
        .update_and_fetch(keys, values, &stream)
        .unwrap();
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 5);

    MlxKeyValueState::discard_branch(branch).unwrap();

    let report = manager.report().unwrap();
    assert_eq!(report.logical_cached_tokens, 0);
    assert_eq!(canonical.offset(), 0);

    let mut resumed = canonical.branch().unwrap();
    let keys = Array::from_slice(&[3.0_f32; 2], &[1, 1, 2, 1]);
    let values = Array::from_slice(&[4.0_f32; 2], &[1, 1, 2, 1]);
    resumed.state.layers[0]
        .update_and_fetch(keys, values, &stream)
        .unwrap();
    canonical.commit_branch(resumed).unwrap();
    assert_eq!(canonical.offset(), 2);
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 2);
}
