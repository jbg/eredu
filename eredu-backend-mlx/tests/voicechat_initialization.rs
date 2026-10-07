//! Native cache initialization consumes the architecture's exact cold selections.

use eredu_architectures::{
    nemotron_h,
    voicechat::{
        config::VoiceChatConfig,
        state::{NeuralStateGeometry, NeuralStateGeometryError},
    },
};
use eredu_backend_mlx::{
    backend::runtime::cache::state::{JointCopyLimits, MlxJointState, MlxKeyValueLayerState},
    native::{initialize_voicechat_neural_state, VoiceChatNeuralInitializationError},
    MlxTensor,
};
use eredu_core::{
    cache::StateResidencyClass, checkpoint::TensorDtype, scheduler::SemanticStateTransaction,
};
use eredu_nn::{AttentionCache, NeuralOperatorCapabilities, Tensor};
use eredu_runtime::{
    select_replicated_text_realization, ArchitectureGroupKind, ArchitectureGroupPlacement,
    ArchitectureGroupTransport, ArchitectureMergeDestination, BackendMechanismCapabilities,
    CacheResidencyPolicy, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    LayerWeightResidency, PagedCacheOptions, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, ReplicatedTextStateAccess, SelectedStateRealization,
    StateComponentMechanism, StateComponentPlacement, StateLayout, StateMechanismCapabilities,
    StateStorageDtype, WeightResidencyMechanism,
};
use safemlx::{Array, Device, DeviceType, Stream};

fn geometry() -> NeuralStateGeometry {
    let backbone = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
        "intermediate_size":24, "num_hidden_layers":3,
        "hybrid_override_pattern":"M*-", "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3, "conv_kernel":3
    }))
    .unwrap();
    let mut config = VoiceChatConfig::from_json(include_str!(
        "../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let tts = &mut config.tts.backbone;
    tts.hidden_size = 4;
    tts.intermediate_size = 8;
    tts.num_hidden_layers = 2;
    tts.num_attention_heads = 1;
    tts.num_key_value_heads = 1;
    tts.head_dim = 4;
    tts.sliding_window = 8;
    tts.layer_types = vec!["sliding_attention".into(), "full_attention".into()];
    NeuralStateGeometry::new(&backbone, tts).unwrap()
}

fn selected(
    layout: &StateLayout,
    access: ReplicatedTextStateAccess,
    policy: CacheResidencyPolicy,
) -> SelectedStateRealization {
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [layout.len()]).unwrap();
    let requirements = ReplicatedTextRequirements::new(
        "test.voicechat-state-initialization",
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
        layout.clone(),
        access,
        Vec::new(),
    )
    .unwrap()
    .with_floating_state_source(TensorDtype::F32);
    let components = (0..layout.len()).flat_map(|layer| {
        layout
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
        .with_floating_state_dtype(TensorDtype::F32, StateStorageDtype::F32)
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

fn limits() -> JointCopyLimits {
    JointCopyLimits {
        max_branch_bytes: 1 << 20,
        max_total_copy_bytes: 1 << 24,
    }
}

#[test]
fn voicechat_initialization_preserves_device_and_paged_state_and_guidance_rows() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let geometry = geometry();
    for paged in [false, true] {
        let policy = if paged {
            CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            )
        } else {
            CacheResidencyPolicy::Device
        };
        let backbone = selected(
            geometry.backbone_layout(),
            ReplicatedTextStateAccess::AttentionWithFixed,
            policy.clone(),
        );
        let tts = selected(
            geometry.tts_layout(),
            ReplicatedTextStateAccess::KeyValue,
            policy,
        );
        let mut owner =
            initialize_voicechat_neural_state(&geometry, &backbone, &tts, limits(), &stream)
                .unwrap();
        assert_eq!(owner.parts().0.residency_manager().is_some(), paged);
        for cache in owner.parts().1.as_ref() {
            assert_eq!(matches!(cache, MlxKeyValueLayerState::Paged(_)), paged);
        }
        for commit in [false, true] {
            let mut branch = owner.branch().unwrap();
            let (_, caches) = branch.parts_mut();
            for cache in caches {
                let keys = MlxTensor::from_array(Array::from_slice(
                    &[1f32, 2., 3., 4., 11., 12., 13., 14.],
                    &[2, 1, 1, 4],
                ));
                let (keys, _) = cache
                    .update_for_attention(keys.clone(), keys, &stream)
                    .unwrap();
                assert_eq!(keys.shape(), vec![2, 1, 1, 4]);
                assert_eq!(
                    keys.as_array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap(),
                    vec![1., 2., 3., 4., 11., 12., 13., 14.]
                );
            }
            safemlx::transforms::async_eval_with_event(branch.retained_arrays())
                .unwrap()
                .synchronize()
                .unwrap();
            assert_eq!(owner.parts().1.offset(), 0);
            if commit {
                owner.commit_branch(branch).unwrap();
                assert_eq!(owner.parts().1.offset(), 1);
            } else {
                MlxJointState::discard_branch(branch).unwrap();
                assert_eq!(owner.parts().1.offset(), 0);
            }
        }
        assert!(owner.copy_bytes_consumed() > 0);
    }
}

#[test]
fn voicechat_initialization_rejects_mismatched_selection() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let geometry = geometry();
    let backbone = selected(
        geometry.backbone_layout(),
        ReplicatedTextStateAccess::AttentionWithFixed,
        CacheResidencyPolicy::Device,
    );
    let tts = selected(
        geometry.tts_layout(),
        ReplicatedTextStateAccess::KeyValue,
        CacheResidencyPolicy::Device,
    );
    let error = initialize_voicechat_neural_state(&geometry, &tts, &tts, limits(), &stream)
        .err()
        .unwrap();
    assert!(matches!(
        error,
        VoiceChatNeuralInitializationError::Geometry(NeuralStateGeometryError::BackboneLayout)
    ));
    let error =
        initialize_voicechat_neural_state(&geometry, &backbone, &backbone, limits(), &stream)
            .err()
            .unwrap();
    assert!(matches!(
        error,
        VoiceChatNeuralInitializationError::Geometry(NeuralStateGeometryError::TtsLayout)
    ));
}
