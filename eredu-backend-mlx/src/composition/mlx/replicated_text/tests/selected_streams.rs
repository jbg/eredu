//! Combined state through the same selected mechanisms used by prepared sessions.
use super::*;
use eredu_core::cache::{
    AppendStreamPolicy, MutableStateResidency, PromptCacheStateSegment, PromptCacheTopology,
    StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
};
use eredu_nn::{AttentionCache, Tensor, TensorElementType};
use eredu_runtime::{
    AppendOnlyStream, AppendStreamBinding, AppendStreamLimits, AppendStreamSpec,
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    RuntimeAppendStreams, RuntimeStateComponents,
};

fn selection(paged: bool, entries: usize) -> SelectedStateRealization {
    let fixed = StateTensorPolicy::new(
        StateTensorRole::IntegerHistory { slot: 0 },
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(2).unwrap(),
        ],
        StateTensorDtype::Int32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value_with_state(
                AttentionPolicy::Full,
                1,
                2,
                vec![fixed],
                vec![AppendStreamPolicy::new(5, 2, StateTensorDtype::Int32, 1).unwrap()],
            )
            .unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let components = layout
        .components(0)
        .unwrap()
        .iter()
        .cloned()
        .map(|component| {
            let paged = if matches!(
                component.role(),
                eredu_core::cache::StateComponentRole::Fixed(_)
            ) {
                StateComponentPlacement::Device
            } else {
                StateComponentPlacement::Paged
            };
            StateComponentMechanism::new(
                0,
                component,
                Some(StateComponentPlacement::Device),
                Some(paged),
            )
        })
        .collect::<Vec<_>>();
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let requirements = ReplicatedTextRequirements::new(
        "combined-stream-state",
        eredu_nn::NeuralOperatorCapabilities::NONE,
        graph,
        units,
        vec![ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        layout,
        ReplicatedTextStateAccess::AttentionWithStreams,
        vec![],
    )
    .unwrap()
    .with_floating_state_source(eredu_core::checkpoint::TensorDtype::F32)
    .with_append_streams(vec![AppendStreamBinding {
        layer: 0,
        lanes: 2,
        spec: AppendStreamSpec {
            slot: 5,
            width: 2,
            element: TensorElementType::I32,
        },
        limits: AppendStreamLimits {
            entries,
            page_entries: 2,
            read_entries: 2,
        },
        payload_bytes: 256,
        scratch_bytes: 64,
        catalog_bytes: 65536,
    }])
    .unwrap();
    let capabilities = BackendMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::NONE,
        vec![],
        vec![WeightResidencyMechanism::Resident],
        complete_state_capabilities(components).with_floating_state_dtype(
            eredu_core::checkpoint::TensorDtype::F32,
            eredu_runtime::StateStorageDtype::F32,
        ),
    );
    let policy = if paged {
        CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(2, 512, 16384, 1)
                .unwrap()
                .with_full_attention(true),
        )
    } else {
        CacheResidencyPolicy::Device
    };
    eredu_runtime::select_replicated_text_realization(
        &requirements,
        &ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            policy,
        ),
        &capabilities,
    )
    .unwrap()
    .state()
    .clone()
}

fn step(state: &mut MlxHybridState, value: i32, stream: &Stream) {
    let layer = &mut state.layers_mut()[0];
    let kv: MlxTensor = Array::from_slice(&[0.5f32, -0.75, 1.25, 2.0], &[2, 1, 1, 2]).into();
    AttentionCache::update_for_attention(layer, kv.clone(), kv, stream).unwrap();
    let records = layer.append_stream(5, 0).unwrap();
    records
        .append(
            records.len(),
            Array::from_slice(&[value, -value], &[1, 2]).into(),
            stream,
        )
        .unwrap();
    *layer
        .fixed_component(StateTensorRole::IntegerHistory { slot: 0 })
        .unwrap() = Some(Array::from_slice(&[value, 17, 23, -value], &[2, 2]).into());
}
fn last(state: &mut MlxHybridState, stream: &Stream) -> Vec<i32> {
    let records = state.layers_mut()[0].append_stream(5, 0).unwrap();
    records
        .read(records.len() - 1..records.len(), stream)
        .unwrap()
        .to_i32_vec(stream)
        .unwrap()
}

#[test]
fn selected_combined_state_realizes_restores_and_forks_retained_stream_bounds() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for paged in [false, true] {
        let selected = selection(paged, 32);
        let mut state =
            <MlxHybridState as MlxStateMechanisms>::realize(&selected, None, 0).unwrap();
        for value in [19, 71, 16_777_217] {
            step(&mut state, value, &stream);
        }
        {
            let records = state.layers_mut()[0].append_stream(5, 0).unwrap();
            assert!(records.read(0..3, &stream).is_err());
            assert!(records
                .append(3, Array::from_slice(&[17i32; 64], &[32, 2]).into(), &stream)
                .is_err());
            assert_eq!(records.len(), 3);
        }
        let checkpoint = MlxStateMechanisms::deep_checkpoint(&state).unwrap();
        step(&mut state, 101, &stream);
        let charged = MlxStateMechanisms::residency_report(&state).unwrap();
        MlxStateMechanisms::restore_checkpoint(&mut state, &checkpoint, &stream).unwrap();
        assert_eq!(state.offset(), 3);
        assert_eq!(last(&mut state, &stream), [16_777_217, -16_777_217]);
        if let Some(before) = charged {
            let after = MlxStateMechanisms::residency_report(&state)
                .unwrap()
                .unwrap();
            assert!(after.append_stream_read_bytes >= before.append_stream_read_bytes);
        }
        let mut branch = MlxStateMechanisms::isolated_snapshot(&state, &stream).unwrap();
        step(&mut branch, 103, &stream);
        assert_eq!(last(&mut branch, &stream), [103, -103]);
        assert_eq!(last(&mut state, &stream), [16_777_217, -16_777_217]);
        assert_eq!(state.layers_mut()[0].append_stream(5, 1).unwrap().len(), 0);
        if paged {
            let descriptor = PromptCacheDescriptor::new(
                "fixture",
                "fixture",
                "weights",
                "prefix",
                "layout",
                1,
                0,
                1,
                2,
                selected.layout().layers().clone(),
                vec![0],
                vec![PromptCacheStateSegment::new("state", 0..1).unwrap()],
                0,
                PromptCacheTopology::default(),
            )
            .unwrap();
            let identity = PromptCacheModelIdentity::new(
                "fixture",
                "fixture",
                "layout",
                1,
                0,
                1,
                0,
                PromptCacheTopology::default(),
                selected.layout().layers().clone(),
                vec![0],
                descriptor.state_segments().to_vec(),
            )
            .unwrap();
            let directory = tempfile::tempdir().unwrap();
            let destination = directory.path().join("state");
            MlxStateMechanisms::save_prompt_cache(
                &mut state,
                &destination,
                descriptor.clone(),
                &[1, 2, 3],
                &PromptCacheOptions::default(),
            )
            .unwrap();
            assert!(
                <MlxHybridState as MlxStateMechanisms>::load_prompt_cache(
                    &selection(true, 2),
                    &destination,
                    &descriptor,
                    &identity,
                    &[1, 2, 3],
                    &stream,
                )
                .is_err(),
                "restore cannot exceed the selected stream history bound"
            );
            let (mut restored, _) = <MlxHybridState as MlxStateMechanisms>::load_prompt_cache(
                &selected,
                &destination,
                &descriptor,
                &identity,
                &[1, 2, 3],
                &stream,
            )
            .unwrap();
            assert_eq!(last(&mut restored, &stream), [16_777_217, -16_777_217]);
            let fixed = restored.layers_mut()[0]
                .fixed_component(StateTensorRole::IntegerHistory { slot: 0 })
                .unwrap()
                .as_ref()
                .unwrap()
                .to_i32_vec(&stream)
                .unwrap();
            assert_eq!(fixed, [16_777_217, 17, 23, -16_777_217]);
            step(&mut restored, 107, &stream);
            assert_eq!(last(&mut restored, &stream), [107, -107]);
        }
    }
}
