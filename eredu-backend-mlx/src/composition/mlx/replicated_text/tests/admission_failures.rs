use super::*;

#[test]
fn heterogeneous_requirements_are_invariant_across_caller_policies() {
    for config in [
        lfm2_config(),
        kimi_linear_config(),
        nemotron_h_config(),
        qwen_next_config(),
        qwen_hybrid_config(),
    ] {
        let root = tiny_heterogeneous_artifact(config);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let expected =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let requests = [
            eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                CacheResidencyPolicy::Device,
            ),
            eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::LayerwiseHost(
                    eredu_runtime::LayerwiseLoadOptions::default(),
                ),
                CacheResidencyPolicy::Paged(
                    PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                        .unwrap()
                        .with_full_attention(true),
                ),
            )
            .with_quantization(eredu_core::QuantizationRequest::Affine {
                group_size: 16,
                bits: 4,
            })
            .with_session(eredu_core::SessionCapabilities::new(true, true, true))
            .with_prompt_cache(true)
            .with_exact_completion(true),
            eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::DenseDiskStream(
                    eredu_runtime::DenseDiskStreamLoadOptions::default(),
                ),
                CacheResidencyPolicy::Device,
            )
            .with_quantization(eredu_core::QuantizationRequest::MxFp4),
        ];
        for request in requests {
            assert!(matches!(
                request.residency(),
                eredu_runtime::LayerWeightResidency::FullyResident
                    | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                    | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
            ));
            assert_eq!(
                expected,
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap()
            );
        }
    }
}

#[test]
fn selected_paged_state_controls_generic_construction() {
    let (stream, weights_stream) = execution_streams();
    for model_type in ["llama", "qwen3"] {
        let root = tiny_artifact(model_type, false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let policy = eredu_core::PreparationPolicy::default();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let state = CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        );
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            state.clone(),
        );
        let selected = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities(&requirements, &request),
        )
        .unwrap();
        assert_eq!(selected.state().policy(), &state);
        let plan = eredu_core::plan_model_preparation(
            inspection,
            policy,
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let architecture_plan = plan.inspection().architecture_plan().clone();
        let artifact = plan.into_artifact();
        let eredu_core::ModelArtifact::SafeTensors {
            configuration: _,
            tensors,
            shards,
            ..
        } = artifact
        else {
            panic!("expected SafeTensors fixture")
        };
        let resolution =
            super::super::loading::prepared_safetensors_architecture(&architecture_plan)
                .unwrap()
                .checkpoint_resolution()
                .unwrap()
                .clone();
        let store = eredu_core::artifact::open_prepared_safetensors_artifact(
            &tensors, shards, resolution, 1,
        )
        .unwrap();
        let executable =
            eredu_architectures::replicated_text::visit_replicated_text_architecture::<
                MlxNeuralBackend,
                MlxKeyValueState,
                _,
            >(
                &architecture_plan,
                selected,
                store,
                &stream,
                BindingVisitor {
                    stream: &stream,
                    weights_stream: &weights_stream,
                },
            )
            .unwrap();
        assert!(executable.cache_residency_report().unwrap().is_some());
    }
}

#[test]
fn routed_selection_rejects_each_missing_addressable_storage_tier_before_construction() {
    crate::tests::support::path_instrumentation::reset();
    let root = tiny_artifact("qwen3_moe", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
    let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    );
    let request = eredu_architectures::RoutedTextSelectionRequest::new(
        text,
        eredu_runtime::WeightResidency::with_independent_parameter_banks(
            eredu_runtime::OrdinaryWeightResidency::FullyResident,
            eredu_runtime::ParameterBankLoadOptions::default(),
        ),
    )
    .unwrap();
    let full = capabilities(requirements.text(), request.text());
    for (expected, tiers) in [
        (
            "addressable disk storage",
            eredu_runtime::AddressableStorageTiers::new(true, true, false),
        ),
        (
            "addressable host storage",
            eredu_runtime::AddressableStorageTiers::new(true, false, true),
        ),
        (
            "addressable device storage",
            eredu_runtime::AddressableStorageTiers::new(false, true, true),
        ),
    ] {
        let capabilities = BackendMechanismCapabilities::new(
            full.operators(),
            full.weight_lowerings().to_vec(),
            full.weight_residencies().to_vec(),
            full.state().clone(),
        )
        .with_session(full.session())
        .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
        .with_indexed_movement(true)
        .with_addressable_storage(
            eredu_runtime::AddressableStorageCapabilities::new(true, true, true, u64::MAX)
                .with_tiers(tiers),
        )
        .with_prompt_cache(full.prompt_cache())
        .with_exact_completion(full.exact_completion());
        let error = eredu_architectures::select_routed_text_realization(
            &requirements,
            &request,
            &capabilities,
        )
        .expect_err("missing storage tier was admitted");
        assert!(
            error.issues().iter().any(|issue| issue == expected),
            "{error}"
        );
    }
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn routed_selection_rejects_top_two_when_scratch_holds_only_one_member() {
    crate::tests::support::path_instrumentation::reset();
    let root = tiny_artifact("qwen3_moe", false);
    let config_path = root.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["num_experts_per_tok"] = 2.into();
    std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
    assert_eq!(requirements.routes_per_token(), 2);
    let one_member = requirements
        .catalog()
        .units()
        .iter()
        .filter_map(eredu_architectures::ExpertResidencyUnit::byte_len)
        .max()
        .unwrap();
    let bank = eredu_runtime::ParameterBankLoadOptions::new(
        eredu_core::residency::OffloadConfig::default(),
        one_member,
        one_member,
    )
    .unwrap();
    let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    );
    let request = eredu_architectures::RoutedTextSelectionRequest::new(
        text,
        eredu_runtime::WeightResidency::with_independent_parameter_banks(
            eredu_runtime::OrdinaryWeightResidency::FullyResident,
            bank,
        ),
    )
    .unwrap();
    let error = eredu_architectures::select_routed_text_realization(
        &requirements,
        &request,
        &capabilities(requirements.text(), request.text()),
    )
    .expect_err("top-two route was admitted into one-member scratch");
    assert!(error
        .issues()
        .iter()
        .any(|issue| issue.contains("one routed token row") && issue.contains("2 routes")));
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn routed_selection_aggregates_text_and_addressable_mechanism_denials() {
    crate::tests::support::path_instrumentation::reset();
    let root = tiny_artifact("qwen3_moe", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
    let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    );
    let request = eredu_architectures::RoutedTextSelectionRequest::new(
        text,
        eredu_runtime::WeightResidency::with_independent_parameter_banks(
            eredu_runtime::OrdinaryWeightResidency::FullyResident,
            eredu_runtime::ParameterBankLoadOptions::default(),
        ),
    )
    .unwrap();
    let full = capabilities(requirements.text(), request.text());
    let incomplete = BackendMechanismCapabilities::new(
        full.operators(),
        full.weight_lowerings().to_vec(),
        full.weight_residencies().to_vec(),
        full.state().clone(),
    )
    .with_session(full.session())
    .with_prompt_cache(full.prompt_cache())
    .with_exact_completion(full.exact_completion());
    let error =
        eredu_architectures::select_routed_text_realization(&requirements, &request, &incomplete)
            .expect_err("incomplete addressable mechanisms were admitted");
    let grouped = error
        .issues()
        .iter()
        .position(|issue| issue.contains("grouped operation"))
        .expect("grouped-operation denial");
    let indexed = error
        .issues()
        .iter()
        .position(|issue| issue.contains("indexed selection"))
        .expect("indexed-movement denial");
    let storage = error
        .issues()
        .iter()
        .position(|issue| issue.contains("addressable storage"))
        .expect("addressable-storage denial");
    assert!(grouped < indexed && indexed < storage, "{error}");
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn dense_generic_addressable_request_rejects_before_production_paths() {
    crate::tests::support::path_instrumentation::reset();
    let root = tiny_artifact("llama", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        ),
    );
    let error = super::super::loading::select_preparation_with_grouped_capabilities(
        &inspection,
        options,
        &GROUPED_OPERATION_CAPABILITIES,
    )
    .expect_err("dense replicated text silently discarded addressable residency");
    assert!(matches!(
        error,
        Error::PreparationAdmission(
            eredu_core::PreparationAdmissionError::ArchitectureParameterBanks
        )
    ));
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn invalid_source_and_missing_grouped_mechanism_never_reach_production_paths() {
    crate::tests::support::path_instrumentation::reset();
    let root = tiny_artifact("llama", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    );

    let first = &requirements.parameters()[0];
    let invalid = ReplicatedTextParameterRequirement::new(
        first.name(),
        first.sources().to_vec(),
        first.physical_sources().to_vec(),
        first.aliases().to_vec(),
        Some(SourceTensorEncoding::Safetensors(StoredDtype::U8)),
        first.physical_shape().map(<[usize]>::to_vec),
        first.logical_shape().to_vec(),
        first.native_executable(),
        first.role(),
        first.owner().clone(),
        first.presence().clone(),
        first.transform_constraint(),
    )
    .unwrap();
    let mut parameters = requirements.parameters().to_vec();
    parameters[0] = invalid;
    let invalid_requirements = ReplicatedTextRequirements::new(
        requirements.architecture_identity().to_owned(),
        requirements.operators(),
        requirements.execution_graph().clone(),
        requirements.execution_units().clone(),
        requirements.group_transports().to_vec(),
        requirements.state_layout().clone(),
        requirements.state_access(),
        parameters,
    )
    .unwrap();
    let error = eredu_runtime::select_replicated_text_realization(
        &invalid_requirements,
        &request,
        &capabilities(&invalid_requirements, &request),
    )
    .unwrap_err();
    assert!(error
        .issues()
        .iter()
        .any(|issue| issue.contains("weight lowering")));

    let routed_root = tiny_artifact("qwen3_moe", false);
    let routed = eredu_architectures::configuration::inspect_artifact(routed_root.path()).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let routed_options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    let error = super::super::loading::select_preparation_with_grouped_capabilities(
        &routed,
        routed_options,
        &[GroupedOperationRequirement::GatedProduct],
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("GatedProductTensorParallelPartial"));

    let affine_error = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &request
            .clone()
            .with_quantization(eredu_core::QuantizationRequest::Affine {
                group_size: 128,
                bits: 4,
            }),
        &capabilities(&requirements, &request),
    )
    .unwrap_err();
    assert!(affine_error
        .issues()
        .iter()
        .any(|issue| issue.contains("affine group size")));

    let linear_index = requirements
        .parameters()
        .iter()
        .position(|parameter| {
            matches!(
                parameter.transform_constraint(),
                ParameterTransformConstraint::Linear { .. }
            )
        })
        .unwrap();
    let linear = &requirements.parameters()[linear_index];
    let ParameterTransformConstraint::Linear { packed_axis } = linear.transform_constraint() else {
        unreachable!()
    };
    let mut logical_shape = linear.logical_shape().to_vec();
    logical_shape[packed_axis] = 48;
    let invalid_mxfp4 = ReplicatedTextParameterRequirement::new(
        linear.name(),
        linear.sources().to_vec(),
        linear.physical_sources().to_vec(),
        linear.aliases().to_vec(),
        linear.source_encoding().cloned(),
        Some(logical_shape.clone()),
        logical_shape,
        linear.native_executable(),
        linear.role(),
        linear.owner().clone(),
        linear.presence().clone(),
        linear.transform_constraint(),
    )
    .unwrap();
    let mut parameters = requirements.parameters().to_vec();
    parameters[linear_index] = invalid_mxfp4;
    let invalid_mxfp4_requirements = ReplicatedTextRequirements::new(
        requirements.architecture_identity().to_owned(),
        requirements.operators(),
        requirements.execution_graph().clone(),
        requirements.execution_units().clone(),
        requirements.group_transports().to_vec(),
        requirements.state_layout().clone(),
        requirements.state_access(),
        parameters,
    )
    .unwrap();
    let mxfp4_request = request
        .clone()
        .with_quantization(eredu_core::QuantizationRequest::MxFp4);
    let mxfp4_error = eredu_runtime::select_replicated_text_realization(
        &invalid_mxfp4_requirements,
        &mxfp4_request,
        &capabilities(&invalid_mxfp4_requirements, &mxfp4_request),
    )
    .unwrap_err();
    assert!(mxfp4_error
        .issues()
        .iter()
        .any(|issue| issue.contains("MXFP4 packed extent 48")));

    let full = capabilities(&requirements, &request);
    let only_basic = BackendMechanismCapabilities::new(
        full.operators(),
        full.weight_lowerings().to_vec(),
        vec![WeightResidencyMechanism::Resident],
        StateMechanismCapabilities::new(full.state().components().iter().map(|mechanism| {
            StateComponentMechanism::new(
                mechanism.layer(),
                mechanism.component().clone(),
                Some(StateComponentPlacement::Device),
                None,
            )
        })),
    );
    let paged = CacheResidencyPolicy::Paged(PagedCacheOptions::new(4, 4096, 4096, 1).unwrap());
    let state_error = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            paged,
        ),
        &only_basic,
    )
    .unwrap_err();
    assert!(state_error
        .issues()
        .iter()
        .any(|issue| issue.contains("state component")));
    let session_error = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &request
            .clone()
            .with_session(eredu_core::SessionCapabilities::new(true, false, false)),
        &only_basic,
    )
    .unwrap_err();
    assert!(session_error
        .issues()
        .iter()
        .any(|issue| issue.contains("session capability")));
    let residency_error = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::DenseDiskStream(
                eredu_runtime::DenseDiskStreamLoadOptions::default(),
            ),
            CacheResidencyPolicy::Device,
        ),
        &only_basic,
    )
    .unwrap_err();
    assert!(residency_error
        .issues()
        .iter()
        .any(|issue| issue.contains("weight residency")));
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn deepseek_prediction_artifact_selects_one_neutral_target_before_payload_work() {
    let mut config = routed_deepseek_v3_config();
    config["num_nextn_predict_layers"] = 1.into();
    let artifact = tiny_heterogeneous_artifact(config);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        2,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    crate::tests::support::path_instrumentation::reset();

    let selected = super::super::loading::select_preparation(&inspection, options)
        .expect("prediction target projection must enter neutral routed admission");

    assert!(selected.neutral().communication_manifest().is_some());
    assert!(selected.rank_context().is_some());
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn routed_prediction_and_media_graphs_are_ineligible_without_production_work() {
    use eredu_architectures::replicated_text::ReplicatedTextIneligibility;

    crate::tests::support::path_instrumentation::reset();
    let mut routed = qwen_hybrid_config();
    routed["model_type"] = "qwen3_next".into();
    routed["num_experts"] = 2.into();
    routed["num_experts_per_tok"] = 1.into();

    let mut nemotron_prediction = nemotron_h_config();
    nemotron_prediction["num_nextn_predict_layers"] = 1.into();
    nemotron_prediction["mtp_hybrid_override_pattern"] = "*E".into();

    let mut qwen_prediction = qwen_hybrid_config();
    qwen_prediction["mtp_num_hidden_layers"] = 1.into();

    let text = qwen_hybrid_config();
    let media = serde_json::json!({
        "model_type": "qwen3_5",
        "image_token_id": 60,
        "video_token_id": 61,
        "text_config": text,
        "vision_config": {
            "depth": 1, "hidden_size": 8, "intermediate_size": 16,
            "num_heads": 2, "num_position_embeddings": 16,
            "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "out_hidden_size": 32
        }
    });

    for (name, config, expected) in [
        ("routed", routed, ReplicatedTextIneligibility::Routed),
        (
            "nemotron prediction",
            nemotron_prediction,
            ReplicatedTextIneligibility::EmbeddedPrediction,
        ),
        (
            "qwen prediction",
            qwen_prediction,
            ReplicatedTextIneligibility::EmbeddedPrediction,
        ),
        ("media", media, ReplicatedTextIneligibility::CompositeInput),
    ] {
        let root = tiny_heterogeneous_artifact(config);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
        let error = eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
            .expect_err("excluded graph entered replicated text admission");
        assert!(
            matches!(
                error,
                eredu_architectures::replicated_text::ReplicatedTextRequirementsError::Ineligible(actual)
                    if actual == expected
            ),
            "{name}: {error}"
        );
    }
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn heterogeneous_state_and_operator_gaps_reject_before_any_production_path() {
    use eredu_core::cache::{
        LayerCachePolicy, StateComponentRole, StateTensorDtype, StateTensorPolicy,
    };

    crate::tests::support::path_instrumentation::reset();
    let paged = CacheResidencyPolicy::Paged(
        PagedCacheOptions::new(4, 4096, 4096, 1)
            .unwrap()
            .with_full_attention(true),
    );
    for (name, config) in [
        ("lfm2", lfm2_config()),
        ("kimi_linear", kimi_linear_config()),
        ("nemotron_h", nemotron_h_config()),
        ("qwen3_next", qwen_next_config()),
        ("qwen3_5_text", qwen_hybrid_config()),
    ] {
        let root = tiny_heterogeneous_artifact(config);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let full = capabilities(&requirements, &request);

        let fixed_components = full
            .state()
            .components()
            .iter()
            .filter(|mechanism| {
                matches!(mechanism.component().role(), StateComponentRole::Fixed(_))
            })
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            !fixed_components.is_empty(),
            "{name} fixture has no fixed state"
        );
        for fixed in &fixed_components {
            let missing_fixed = complete_state_capabilities(
                full.state()
                    .components()
                    .iter()
                    .filter(|mechanism| *mechanism != fixed)
                    .cloned(),
            );
            let error = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities_with(&full, full.operators(), missing_fixed),
            )
            .expect_err("missing fixed state was admitted");
            assert!(
                error.issues().iter().any(|issue| {
                    issue.contains(&fixed.component().role().stable_name())
                        && issue.contains("state component")
                }),
                "{name}: {error}"
            );
        }
        if name == "kimi_linear" {
            let without_compressed = complete_state_capabilities(
                full.state()
                    .components()
                    .iter()
                    .filter(|mechanism| {
                        mechanism.component().role() != StateComponentRole::CompressedLatent
                    })
                    .cloned(),
            );
            let error = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities_with(&full, full.operators(), without_compressed),
            )
            .expect_err("missing compressed attention was admitted");
            assert!(error
                .issues()
                .iter()
                .any(|issue| issue.contains("attention.compressed_latent")));
        }

        for fixed in &fixed_components {
            let StateComponentRole::Fixed(role) = fixed.component().role() else {
                unreachable!("fixed component filter changed")
            };
            let wrong_shape =
                LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new_with_residency(
                    role,
                    vec![eredu_core::cache::StateTensorDimension::fixed(999).unwrap()],
                    fixed.component().dtype(),
                    fixed.component().residency(),
                )
                .unwrap()])
                .unwrap()
                .components()
                .pop()
                .unwrap();
            let components = full.state().components().iter().map(|mechanism| {
                if mechanism == fixed {
                    StateComponentMechanism::new(
                        mechanism.layer(),
                        wrong_shape.clone(),
                        Some(StateComponentPlacement::Device),
                        Some(StateComponentPlacement::Device),
                    )
                } else {
                    mechanism.clone()
                }
            });
            let error = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities_with(
                    &full,
                    full.operators(),
                    complete_state_capabilities(components),
                ),
            )
            .expect_err("wrong fixed-state shape was admitted");
            assert!(error
                .issues()
                .iter()
                .any(|issue| issue.contains("shape") && issue.contains("dtype")));

            let alternate_dtype = match fixed.component().dtype() {
                StateTensorDtype::Float32 => StateTensorDtype::Floating,
                _ => StateTensorDtype::Float32,
            };
            let wrong_dtype =
                LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new_with_residency(
                    role,
                    fixed.component().shape().to_vec(),
                    alternate_dtype,
                    fixed.component().residency(),
                )
                .unwrap()])
                .unwrap()
                .components()
                .pop()
                .unwrap();
            let components = full.state().components().iter().map(|mechanism| {
                if mechanism == fixed {
                    StateComponentMechanism::new(
                        mechanism.layer(),
                        wrong_dtype.clone(),
                        Some(StateComponentPlacement::Device),
                        Some(StateComponentPlacement::Device),
                    )
                } else {
                    mechanism.clone()
                }
            });
            assert!(
                eredu_runtime::select_replicated_text_realization(
                    &requirements,
                    &request,
                    &capabilities_with(
                        &full,
                        full.operators(),
                        complete_state_capabilities(components),
                    ),
                )
                .is_err(),
                "{name} admitted an incompatible fixed-state dtype"
            );
        }

        let paged_components = full.state().components().iter().map(|mechanism| {
            StateComponentMechanism::new(
                mechanism.layer(),
                mechanism.component().clone(),
                Some(StateComponentPlacement::Device),
                Some(StateComponentPlacement::Paged),
            )
        });
        let paged_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            paged.clone(),
        )
        .with_prompt_cache(true);
        assert!(
            eredu_runtime::select_replicated_text_realization(
                &requirements,
                &paged_request,
                &capabilities_with(
                    &full,
                    full.operators(),
                    complete_state_capabilities(paged_components),
                ),
            )
            .is_err(),
            "{name} admitted incompatible paged fixed-component placement"
        );

        if requirements.operators() != eredu_nn::NeuralOperatorCapabilities::NONE {
            let error = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities_with(
                    &full,
                    eredu_nn::NeuralOperatorCapabilities::NONE,
                    full.state().clone(),
                ),
            )
            .expect_err("missing semantic neural operations were admitted");
            let operation = match name {
                "nemotron_h" => "selective_state_space_scan",
                "kimi_linear" | "qwen3_next" | "qwen3_5_text" => "gated_delta_scan",
                _ => unreachable!(),
            };
            assert!(
                error.issues().iter().any(|issue| issue.contains(operation)),
                "{name}: {error}"
            );
        }

        let paged_full = capabilities(&requirements, &paged_request);
        for (facility, state) in [
            (
                "checkpoint",
                StateMechanismCapabilities::new(paged_full.state().components().iter().cloned())
                    .with_transactions(false, true)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
            ),
            (
                "rollback",
                StateMechanismCapabilities::new(paged_full.state().components().iter().cloned())
                    .with_transactions(true, false)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
            ),
            (
                "reset",
                StateMechanismCapabilities::new(paged_full.state().components().iter().cloned())
                    .with_transactions(true, true)
                    .with_reset(false)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
            ),
            (
                "prompt-cache",
                StateMechanismCapabilities::new(paged_full.state().components().iter().cloned())
                    .with_transactions(true, true)
                    .with_reset(true)
                    .with_prompt_cache(false)
                    .with_observation_retention(true),
            ),
        ] {
            let error = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &paged_request,
                &capabilities_with(&paged_full, paged_full.operators(), state),
            )
            .unwrap_err();
            assert!(
                error.issues().iter().any(|issue| issue.contains(facility)),
                "{name}: missing {facility} diagnostic: {error}"
            );
        }

        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
    }
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn unsupported_topology_fails_before_checkpoint_payload_or_module_construction() {
    crate::tests::support::path_instrumentation::reset();
    let root = tiny_artifact("llama", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    );
    let request = request.with_topology(topology.topology());
    std::fs::remove_file(root.path().join("model.safetensors")).unwrap();

    let error = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &request,
        &capabilities(&requirements, &request),
    )
    .expect_err("unsupported topology was admitted");
    let message = error.to_string();
    assert!(
        message.contains("replicated execution topology"),
        "{message}"
    );
    assert!(!message.contains("No such file"), "{message}");
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}
