use super::*;
use crate::{AppendStreamBinding, AppendStreamLimits, AppendStreamSpec, PartitionState};
use eredu_core::{cache::AppendStreamPolicy, AttentionPolicy, LayerSchedule};

fn layout(width: i32, layers: usize, cadence: i32) -> StateLayout {
    StateLayout::new(
        LayerSchedule::new(
            layers,
            vec![
                eredu_core::cache::LayerCachePolicy::key_value_with_state(
                    AttentionPolicy::Full,
                    1,
                    8,
                    vec![],
                    vec![AppendStreamPolicy::new(
                        7,
                        width,
                        eredu_core::cache::StateTensorDtype::Int32,
                        cadence
                    )
                    .unwrap()],
                )
                .unwrap();
                layers
            ],
        )
        .unwrap(),
    )
    .unwrap()
}

fn fixture() -> (ReplicatedTextRequirements, BackendMechanismCapabilities) {
    let layout = layout(4, 3, 4);
    let graph =
        ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [3]).unwrap();
    let components = (0..3)
        .flat_map(|layer| {
            layout
                .components(layer)
                .unwrap()
                .to_vec()
                .into_iter()
                .map(move |component| {
                    StateComponentMechanism::new(
                        layer,
                        component,
                        Some(StateComponentPlacement::Device),
                        Some(StateComponentPlacement::Paged),
                    )
                })
        })
        .collect::<Vec<_>>();
    let requirements = ReplicatedTextRequirements::new(
        "streams",
        NeuralOperatorCapabilities::NONE,
        graph,
        units,
        vec![ArchitectureGroupTransport {
            placement: crate::ArchitectureGroupPlacement::Pipeline,
            kind: crate::ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: crate::ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        layout,
        ReplicatedTextStateAccess::AttentionWithStreams,
        vec![],
    )
    .unwrap()
    .with_floating_state_source(TensorDtype::F32)
    .with_append_streams(
        (0..3)
            .map(|layer| AppendStreamBinding {
                layer,
                lanes: 2,
                spec: AppendStreamSpec {
                    slot: 7,
                    width: 4,
                    element: eredu_nn::TensorElementType::I32,
                },
                limits: AppendStreamLimits {
                    entries: 16,
                    page_entries: 4,
                    read_entries: 2,
                },
                payload_bytes: 256,
                scratch_bytes: 192,
                catalog_bytes: 4096,
            })
            .collect(),
    )
    .unwrap();
    let capabilities = BackendMechanismCapabilities::new(
        NeuralOperatorCapabilities::NONE,
        vec![],
        vec![WeightResidencyMechanism::Resident],
        StateMechanismCapabilities::new(components)
            .with_floating_state_dtype(TensorDtype::F32, StateStorageDtype::F32)
            .with_transactions(true, true)
            .with_reset(true),
    );
    (requirements, capabilities)
}

fn select(
    requirements: &ReplicatedTextRequirements,
    capabilities: &BackendMechanismCapabilities,
    policy: CacheResidencyPolicy,
) -> Result<SelectedReplicatedTextRealization, ReplicatedTextSelectionError> {
    select_replicated_text_realization(
        requirements,
        &ReplicatedTextSelectionRequest::new(LayerWeightResidency::FullyResident, policy),
        capabilities,
    )
}

#[test]
fn selected_streams_preserve_partition_ownership_geometry_and_replica_allowances() {
    let (requirements, capabilities) = fixture();
    let selected = select(&requirements, &capabilities, CacheResidencyPolicy::Device).unwrap();
    assert_eq!(
        selected.state().append_streams(),
        requirements.append_streams()
    );
    let total = selected.state().append_stream_allowances();
    assert_eq!(
        (
            total.payload_bytes,
            total.scratch_bytes,
            total.catalog_bytes
        ),
        (1536, 1152, 24576)
    );
    let partition = PartitionState::new(selected.state().layout().slice(1..3).unwrap(), 1).unwrap();
    let local = selected.state().for_partition(&partition).unwrap();
    assert_eq!(
        local
            .append_streams()
            .iter()
            .map(|b| b.layer)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(local.append_stream_allowances().scratch_bytes, 768);
    assert_eq!(local.append_streams()[0].lanes, 2);
    let narrow = PartitionState::new(layout(2, 2, 4), 1).unwrap();
    assert!(selected.state().for_partition(&narrow).is_err());
    let local = selected.state().for_partitioned_geometry(&narrow).unwrap();
    assert_eq!(local.append_streams()[0].spec.width, 2);
    // Narrowed TP geometry retains the admitted allowances; no replica budget is refunded.
    assert_eq!(local.append_stream_allowances().payload_bytes, 1024);
    assert!(selected
        .state()
        .for_partitioned_geometry(&PartitionState::new(layout(2, 2, 2), 1).unwrap())
        .is_err());
    assert!(selected
        .state()
        .for_partitioned_geometry(&PartitionState::new(layout(8, 2, 4), 1).unwrap())
        .is_err());
}

#[test]
fn stream_admission_rejects_missing_duplicate_changed_and_overflowing_bounds() {
    let (requirements, capabilities) = fixture();
    for mutate in [
        (|b: &mut Vec<AppendStreamBinding>| {
            b.pop();
        }) as fn(&mut Vec<AppendStreamBinding>),
        |b| {
            b[1].layer = 0;
        },
        |b| {
            b[0].spec.width = 5;
        },
        |b| {
            b[0].spec.element = eredu_nn::TensorElementType::F32;
        },
        |b| {
            b[0].lanes = 1;
        },
        |b| {
            b[0].scratch_bytes -= 1;
        },
        |b| {
            b[0].catalog_bytes = u64::MAX;
        },
    ] {
        let mut bindings = requirements.append_streams().to_vec();
        mutate(&mut bindings);
        assert!(requirements.clone().with_append_streams(bindings).is_err());
    }
    assert!(requirements
        .clone()
        .with_state_layout(layout(2, 3, 4))
        .is_err());
    let mut missing = requirements.clone();
    missing.append_streams.clear();
    assert!(select(&missing, &capabilities, CacheResidencyPolicy::Device).is_err());
    let mut insufficient = requirements.clone();
    insufficient.append_streams[0].payload_bytes -= 1;
    assert!(select(&insufficient, &capabilities, CacheResidencyPolicy::Device).is_err());
}

#[test]
fn paged_stream_admission_checks_page_geometry_and_device_capacity() {
    let (requirements, capabilities) = fixture();
    let policy = |page, budget| {
        CacheResidencyPolicy::Paged(
            crate::PagedCacheOptions::new(page, budget, 8192, 1)
                .unwrap()
                .with_full_attention(true),
        )
    };
    assert!(select(&requirements, &capabilities, policy(2, 64)).is_err());
    assert!(select(&requirements, &capabilities, policy(4, 63)).is_err());
    let selected = select(&requirements, &capabilities, policy(4, 64)).unwrap();
    assert_eq!(
        selected.state().append_streams(),
        requirements.append_streams()
    );
}

#[test]
fn stream_selection_checks_concrete_floating_storage_before_construction() {
    let (mut requirements, mut capabilities) = fixture();
    let policy = eredu_core::cache::LayerCachePolicy::key_value_with_state(
        AttentionPolicy::Full,
        1,
        8,
        vec![],
        vec![
            AppendStreamPolicy::new(7, 4, eredu_core::cache::StateTensorDtype::Floating, 4)
                .unwrap(),
        ],
    )
    .unwrap();
    requirements.state_layout =
        StateLayout::new(LayerSchedule::new(3, vec![policy; 3]).unwrap()).unwrap();
    let bindings = requirements
        .append_streams()
        .iter()
        .cloned()
        .map(|mut binding| {
            binding.spec.element = eredu_nn::TensorElementType::Bf16;
            binding
        })
        .collect();
    requirements = requirements.with_append_streams(bindings).unwrap();
    capabilities.state.components = (0..3)
        .flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .unwrap()
                .to_vec()
                .into_iter()
                .map(move |component| {
                    StateComponentMechanism::new(
                        layer,
                        component,
                        Some(StateComponentPlacement::Device),
                        Some(StateComponentPlacement::Paged),
                    )
                })
        })
        .collect();
    assert!(select(&requirements, &capabilities, CacheResidencyPolicy::Device).is_err());
    capabilities.state = capabilities
        .state
        .with_floating_state_dtype(TensorDtype::F32, StateStorageDtype::Bf16);
    assert!(select(&requirements, &capabilities, CacheResidencyPolicy::Device).is_ok());
}

#[test]
fn standalone_state_selection_preserves_text_admission_and_rejects_missing_lifecycle() {
    let (requirements, capabilities) = fixture();
    let state = requirements.state_requirements();
    let state = StateRealizationRequirements::new(
        state.layout().clone(),
        state.access(),
        state.floating_source().cloned(),
        state.append_streams().to_vec(),
    )
    .unwrap();
    let paged = CacheResidencyPolicy::Paged(
        crate::PagedCacheOptions::new(4, 64, 8192, 1)
            .unwrap()
            .with_full_attention(true),
    );
    for policy in [CacheResidencyPolicy::Device, paged] {
        let request =
            ReplicatedTextSelectionRequest::new(LayerWeightResidency::FullyResident, policy);
        let selected = select_state_realization(&state, &request, capabilities.state()).unwrap();
        let text =
            select_replicated_text_realization(&requirements, &request, &capabilities).unwrap();
        assert_eq!(&selected, text.state());
        assert_eq!(selected.append_streams(), state.append_streams());
        assert_eq!(selected.append_stream_allowances().payload_bytes, 1536);
        let mut missing = capabilities.state().clone();
        missing.components.pop();
        assert!(select_state_realization(&state, &request, &missing).is_err());
        let mut duplicate = capabilities.state().clone();
        duplicate.components.push(duplicate.components[0].clone());
        assert!(select_state_realization(&state, &request, &duplicate).is_err());
        for (facts, expected) in [
            (
                capabilities.state().clone().with_transactions(false, true),
                "state checkpoint",
            ),
            (
                capabilities.state().clone().with_transactions(true, false),
                "state rollback",
            ),
            (
                capabilities.state().clone().with_reset(false),
                "state reset",
            ),
        ] {
            assert!(select_state_realization(&state, &request, &facts)
                .unwrap_err()
                .issues()
                .iter()
                .any(|i| i == expected));
        }
        let observed = request
            .clone()
            .with_session(SessionCapabilities::new(false, true, true));
        assert!(
            select_state_realization(&state, &observed, capabilities.state())
                .unwrap_err()
                .issues()
                .iter()
                .any(|i| i == "state observation retention")
        );
        let persistent = request.clone().with_prompt_cache(true);
        assert!(
            select_state_realization(&state, &persistent, capabilities.state())
                .unwrap_err()
                .issues()
                .iter()
                .any(|i| i == "state prompt-cache persistence")
        );
    }
    let mut missing = state.append_streams().to_vec();
    missing.pop();
    assert!(StateRealizationRequirements::new(
        state.layout().clone(),
        state.access(),
        state.floating_source().cloned(),
        missing
    )
    .is_err());
}
