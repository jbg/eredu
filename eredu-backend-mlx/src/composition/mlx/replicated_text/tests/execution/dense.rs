#[test]
fn both_text_architectures_retain_exact_sharded_admission_with_one_cached_payload() {
    let (stream, weights_stream) = execution_streams();
    for model_type in ["mistral", "qwen3"] {
        let root = tiny_sharded_artifact(model_type, false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let streaming =
            eredu_runtime::DenseDiskStreamLoadOptions::default().with_max_cached_shards(1);
        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::dense_disk_stream(streaming),
            ),
        );
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        std::fs::remove_file(root.path().join("model.safetensors.index.json")).unwrap();

        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
        let mut executable = model.into_executable();
        let executable = executable.erased_mut();
        executable
            .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let report = executable.dense_stream_report().unwrap().unwrap();
        assert!(report.residency().weight_store().currently_cached_shards <= 1);
        assert!(!report
            .residency()
            .weight_store()
            .payload_shard_paths
            .is_empty());
    }
}

#[test]
fn alias_backed_packed_safetensors_companions_keep_their_exact_source() {
    let (stream, weights_stream) = execution_streams();
    let config = packed_alias_nemotron_h_config();
    assert_eq!(
        eredu_architectures::nemotron_h::model_args_from_config_value(&config)
            .unwrap()
            .hidden_size,
        32
    );
    let root = tiny_heterogeneous_artifact(config);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let eredu_architectures::configuration::SafetensorsModelConfig::NemotronH(inspected) =
        inspection
            .architecture_plan()
            .safetensors_architecture()
            .unwrap()
            .model()
    else {
        panic!("expected Nemotron-H inspection")
    };
    assert_eq!(inspected.hidden_size, 32);
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    for parameter in requirements.parameters().iter().filter(|parameter| {
        parameter.source_encoding()
            == Some(&eredu_checkpoint::SourceTensorEncoding::Safetensors(
                eredu_checkpoint::StoredDtype::U32,
            ))
    }) {
        let descriptor = parameter
            .lowering_descriptor(parameter.native_executable())
            .unwrap();
        assert!(
            supports_direct(&descriptor),
            "unsupported native packed requirement {} {:?} {:?} logical={:?}: {descriptor:?}",
            parameter.name(),
            parameter.role(),
            parameter.presence(),
            parameter.logical_shape()
        );
    }
    let aliased_companion = requirements
        .parameters()
        .iter()
        .find(|parameter| {
            parameter.role() == eredu_runtime::ReplicatedTextParameterRole::FormatCompanion
                && parameter.name().starts_with("model.")
                && parameter
                    .sources()
                    .first()
                    .is_some_and(|source| source.starts_with("backbone."))
        })
        .unwrap_or_else(|| {
            panic!(
                "Nemotron-H fixture must select an official alias-backed companion: {:?}",
                requirements
                    .parameters()
                    .iter()
                    .filter(|parameter| parameter.role()
                        == eredu_runtime::ReplicatedTextParameterRole::FormatCompanion)
                    .map(|parameter| (parameter.name(), parameter.sources()))
                    .collect::<Vec<_>>()
            )
        });
    assert_ne!(aliased_companion.name(), aliased_companion.sources()[0]);

    let plan = eredu_core::plan_model_preparation(
        inspection,
        eredu_core::PreparationPolicy::default(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    let model = materialize_model_plan(
        plan,
        crate::MlxLoadRequest::default(),
        &stream,
        &weights_stream,
    )
    .unwrap();
    let mut executable = model.into_executable();
    let generic = executable.erased_mut();
    let logits = generic
        .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(logits.len(), 64);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
fn checkpoint_native_packed_safetensors_companions_are_consumed_once() {
    let (stream, weights_stream) = execution_streams();
    for model_type in ["llama", "qwen3"] {
        let root = tiny_packed_safetensors_artifact(model_type);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        assert!(requirements.parameters().iter().any(|parameter| {
            parameter.role() == eredu_runtime::ReplicatedTextParameterRole::FormatCompanion
                && parameter.source_encoding()
                    == Some(&eredu_checkpoint::SourceTensorEncoding::Safetensors(
                        eredu_checkpoint::StoredDtype::F32,
                    ))
        }));
        assert!(requirements.parameters().iter().any(|parameter| {
            parameter.role() == eredu_runtime::ReplicatedTextParameterRole::LinearWeight
                && matches!(
                    parameter.native_executable(),
                    eredu_checkpoint::LinearFormat::Affine(_)
                )
                && parameter.source_encoding()
                    == Some(&eredu_checkpoint::SourceTensorEncoding::Safetensors(
                        eredu_checkpoint::StoredDtype::U32,
                    ))
        }));

        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap_or_else(|error| panic!("{model_type}: {error}"));
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let logits = generic
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(logits.len(), 64);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn fused_qwen_next_safetensors_preserves_both_source_families_into_execution() {
    let (stream, weights_stream) = execution_streams();
    let root = tiny_heterogeneous_artifact_with_layout(qwen_next_config(), true);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let fused = requirements
        .parameters()
        .iter()
        .filter(|parameter| {
            parameter.name().contains("linear_attn.in_proj_")
                && matches!(
                    parameter.presence(),
                    eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
                )
        })
        .collect::<Vec<_>>();
    assert_eq!(fused.len(), 4);
    assert!(fused.iter().all(|parameter| {
        parameter.has_lowering_source()
            && matches!(
                parameter.source_encoding(),
                Some(eredu_checkpoint::SourceTensorEncoding::RecipeOutput(
                    eredu_checkpoint::StoredDtype::F32
                ))
            )
            && parameter.physical_shape().is_some()
            && parameter.physical_sources().len() == 1
    }));
    assert_eq!(
        fused
            .iter()
            .filter(|parameter| parameter.physical_sources()[0].tensor().contains("qkvz"))
            .count(),
        2
    );
    assert_eq!(
        fused
            .iter()
            .filter(|parameter| parameter.physical_sources()[0].tensor().contains("ba"))
            .count(),
        2
    );
    assert!(fused
        .iter()
        .filter(|parameter| parameter.physical_sources()[0].tensor().contains("ba"))
        .all(|parameter| parameter.native_executable() == eredu_checkpoint::LinearFormat::Dense));

    let plan = eredu_core::plan_model_preparation(
        inspection,
        eredu_core::PreparationPolicy::default(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    let model = materialize_model_plan(
        plan,
        crate::MlxLoadRequest::default(),
        &stream,
        &weights_stream,
    )
    .unwrap();
    let mut executable = model.into_executable();
    let generic = executable.erased_mut();
    let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let parts = [input::token_ids_part(&prompt).unwrap()];
    generic
        .prefill(input::ModelInput::new(&parts), &stream)
        .unwrap()
        .evaluated()
        .unwrap();
    let logits = generic
        .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(
        logits.iter().all(|value| value.is_finite())
            && logits.iter().any(|value| value.abs() > 1e-12)
    );
    assert!(generic
        .state_snapshot()
        .iter()
        .all(|(position, _)| *position == 3));
}

#[test]
fn packed_fused_qwen_next_gguf_format_reaches_split_projection_execution() {
    let (stream, weights_stream) = execution_streams();
    let gguf = tiny_heterogeneous_gguf_with_packed_qwen_next(
        "qwen3next",
        Some(eredu_gguf::GgmlType::MxFp4),
        &stream,
    );
    let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let fused_targets = requirements
        .parameters()
        .iter()
        .filter(|parameter| {
            parameter.name().contains("linear_attn.in_proj_qkv.weight")
                || parameter.name().contains("linear_attn.in_proj_z.weight")
        })
        .collect::<Vec<_>>();
    assert_eq!(fused_targets.len(), 2);
    assert!(fused_targets.iter().all(|parameter| {
        matches!(
            parameter.presence(),
            eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
        ) && parameter.has_lowering_source()
            && parameter.native_executable() == eredu_checkpoint::LinearFormat::MxFp4
    }));
    let selection_request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        eredu_runtime::CacheResidencyPolicy::Device,
    );
    let selected = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &selection_request,
        &capabilities(&requirements, &selection_request),
    )
    .unwrap();
    assert!(selected
        .parameters()
        .iter()
        .filter(|parameter| {
            parameter.name().contains("linear_attn.in_proj_qkv.weight")
                || parameter.name().contains("linear_attn.in_proj_z.weight")
        })
        .all(|parameter| parameter.lowering() == eredu_runtime::WeightLoweringKind::Derived));

    let plan = eredu_core::plan_model_preparation(
        inspection,
        eredu_core::PreparationPolicy::default(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    let model = materialize_model_plan(
        plan,
        crate::MlxLoadRequest::default(),
        &stream,
        &weights_stream,
    )
    .unwrap();
    let mut executable = model.into_executable();
    let generic = executable.erased_mut();
    let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let parts = [input::token_ids_part(&prompt).unwrap()];
    generic
        .prefill(input::ModelInput::new(&parts), &stream)
        .unwrap()
        .evaluated()
        .unwrap();
    let logits = generic
        .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(
        logits.iter().all(|value| value.is_finite())
            && logits.iter().any(|value| value.abs() > 1e-12)
    );
    assert!(generic
        .state_snapshot()
        .iter()
        .all(|(position, _)| *position == 3));
}

#[test]
fn gpt_oss_gguf_uses_generic_routed_execution_for_both_residencies() {
    let (stream, weights_stream) = execution_streams();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("gpt-oss.gguf");
    crate::tests::distributed_pipeline_ring::write_gpt_oss_gguf_fixture(&path);
    for addressable in [false, true] {
        let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
        let mut options = crate::MlxLoadRequest::default();
        if addressable {
            options = crate::MlxLoadRequest::from_normalized(
                options.normalized().clone().with_weight_residency(
                    eredu_runtime::WeightResidency::with_independent_parameter_banks(
                        eredu_runtime::OrdinaryWeightResidency::FullyResident,
                        eredu_runtime::ParameterBankLoadOptions::default(),
                    ),
                ),
            );
        }
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"))
            .evaluated()
            .unwrap();
        let logits = generic
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
            .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
        assert_eq!(logits.shape(), &[1, 64]);
        assert!(logits
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .all(|value| value.is_finite()));
    }
}

#[test]
fn heterogeneous_gguf_artifacts_use_the_same_generic_state_contract() {
    crate::tests::support::path_instrumentation::reset();
    let (stream, weights_stream) = execution_streams();
    for (name, gguf_name, config) in [
        ("lfm2", "lfm2", lfm2_config()),
        ("kimi_linear", "kimi_linear", kimi_linear_config()),
        ("nemotron_h", "nemotron_h", nemotron_h_config()),
        ("qwen3_next", "qwen3next", qwen_next_config()),
        ("qwen3_5_text", "qwen35", qwen_hybrid_config()),
    ] {
        let gguf = tiny_heterogeneous_gguf(gguf_name, &stream);
        let inspection = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap_or_else(|error| panic!("{name} GGUF requirements: {error}"));
        let stateful = (0..requirements.state_layout().len())
            .map(|layer| {
                !requirements
                    .state_layout()
                    .components(layer)
                    .unwrap()
                    .is_empty()
            })
            .collect::<Vec<_>>();
        let safe = tiny_heterogeneous_artifact(config);
        let safe_inspection =
            eredu_architectures::configuration::inspect_artifact(safe.path()).unwrap();
        let safe_requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&safe_inspection)
                .unwrap();
        assert_eq!(
            requirements.state_access(),
            safe_requirements.state_access()
        );
        assert_eq!(
            requirements.state_layout(),
            safe_requirements.state_layout()
        );
        assert_eq!(requirements.operators(), safe_requirements.operators());
        assert_eq!(
            requirements.execution_graph(),
            safe_requirements.execution_graph()
        );

        let execute = |token| {
            let fresh = eredu_architectures::configuration::inspect_artifact(gguf.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                fresh,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(
                plan,
                crate::MlxLoadRequest::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{name} GGUF: {error}"));
            let mut executable = model.into_executable();
            let generic = executable.erased_mut();
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let fixed_before = generic.fixed_numeric_state_snapshot().unwrap();
            let logits = generic
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            let fixed_after = generic.fixed_numeric_state_snapshot().unwrap();
            (logits, generic.state_snapshot(), fixed_before, fixed_after)
        };
        let token_three = execute(3_u32);
        let token_four = execute(4_u32);
        for (logits, snapshot, fixed_before, fixed_after) in [&token_three, &token_four] {
            assert!(
                logits.iter().all(|value| value.is_finite())
                    && logits.iter().any(|value| value.abs() > 1e-12),
                "{name} GGUF produced invalid logits: {logits:?}"
            );
            assert!(snapshot
                .iter()
                .zip(&stateful)
                .all(|((position, fixed), stateful)| {
                    *position == if *stateful { 3 } else { 0 }
                        && fixed.iter().all(|(_, present)| *present)
                }));
            assert_ne!(
                fixed_before, fixed_after,
                "{name} fixed state did not consume decode input"
            );
        }
        assert_ne!(
            token_three.0, token_four.0,
            "{name} ignored token identity at an identical state frontier"
        );
    }
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts {
            architecture_constructions: 10,
            state_allocations: 10,
            payload_opens: 10,
            constructors: 10,
            unit_constructions: 24,
            materializations: 0,
            local_static_bindings: 0,
            excluded_local_static_parameters: 0,
            forwards: 20,
            state_publications: 20,
            completions: 20,
        }
    );
}

#[test]
fn homogeneous_state_schedules_execute_with_only_their_exact_mechanisms() {
    let (stream, weights_stream) = execution_streams();
    let mut cases = Vec::new();
    let mut lfm = lfm2_config();
    lfm["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
    cases.push((
        "lfm_attention",
        lfm,
        ReplicatedTextStateAccess::KeyValue,
        None,
    ));
    let mut lfm = lfm2_config();
    lfm["layer_types"] = serde_json::json!(["conv", "conv"]);
    cases.push(("lfm_fixed", lfm, ReplicatedTextStateAccess::Fixed, None));

    let mut kimi = kimi_linear_config();
    kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([1, 2]);
    kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([]);
    cases.push((
        "kimi_kda",
        kimi,
        ReplicatedTextStateAccess::Fixed,
        Some(eredu_nn::NeuralOperatorCapabilities::GATED_DELTA_SCAN),
    ));
    let mut kimi = kimi_linear_config();
    kimi["linear_attn_config"]["kda_layers"] = serde_json::json!([]);
    kimi["linear_attn_config"]["full_attn_layers"] = serde_json::json!([1, 2]);
    cases.push((
        "kimi_mla",
        kimi,
        ReplicatedTextStateAccess::CompressedAttention,
        None,
    ));

    for (name, pattern, access, operator) in [
        (
            "nemotron_attention",
            "****",
            ReplicatedTextStateAccess::KeyValue,
            None,
        ),
        (
            "nemotron_mamba",
            "MMMM",
            ReplicatedTextStateAccess::Fixed,
            Some(eredu_nn::NeuralOperatorCapabilities::SELECTIVE_STATE_SPACE_SCAN),
        ),
        (
            "nemotron_stateless",
            "----",
            ReplicatedTextStateAccess::Stateless,
            None,
        ),
    ] {
        let mut nemo = nemotron_h_config();
        nemo["hybrid_override_pattern"] = pattern.into();
        cases.push((name, nemo, access, operator));
    }
    let mut qwen = qwen_hybrid_config();
    qwen["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
    cases.push((
        "qwen_attention",
        qwen,
        ReplicatedTextStateAccess::KeyValue,
        None,
    ));
    let mut qwen = qwen_hybrid_config();
    qwen["layer_types"] = serde_json::json!(["linear_attention", "linear_attention"]);
    cases.push((
        "qwen_fixed",
        qwen,
        ReplicatedTextStateAccess::Fixed,
        Some(eredu_nn::NeuralOperatorCapabilities::GATED_DELTA_SCAN),
    ));

    for (name, config, access, operator) in cases {
        let root = tiny_heterogeneous_artifact(config);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap_or_else(|error| panic!("{name} requirements: {error}"));
        assert_eq!(requirements.state_access(), access, "{name}");
        if let Some(operator) = operator {
            assert!(requirements.operators().contains(operator), "{name}");
        } else {
            assert_eq!(
                requirements.operators(),
                eredu_nn::NeuralOperatorCapabilities::NONE,
                "{name}"
            );
        }
        let stateful = requirements
            .state_layout()
            .layers()
            .iter()
            .map(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
            .collect::<Vec<_>>();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::default(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let logits = generic
            .decode(&Array::from_slice(&[3_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(
            logits.iter().all(|value| value.is_finite())
                && logits.iter().any(|value| value.abs() > 1e-12),
            "{name}: {logits:?}"
        );
        assert!(generic
            .state_snapshot()
            .iter()
            .zip(&stateful)
            .all(|((position, _), stateful)| *position == if *stateful { 3 } else { 0 }));
    }
}

#[test]
fn generic_handoff_executes_every_replicated_state_profile() {
    crate::tests::support::path_instrumentation::reset();
    let (stream, weights_stream) = execution_streams();
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
                .unwrap_or_else(|error| panic!("{name} requirements: {error}"));
        assert!(requirements
            .state_layout()
            .layers()
            .iter()
            .any(|layer| !layer.fixed_state().is_empty()));
        let stateful_layers = requirements
            .state_layout()
            .layers()
            .iter()
            .map(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
            .collect::<Vec<_>>();
        let policy = eredu_core::PreparationPolicy::default();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            policy,
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(
            plan,
            crate::MlxLoadRequest::default(),
            &stream,
            &weights_stream,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut executable = model.into_executable();
        let executable = executable.erased_mut();
        let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        let logits = executable
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap_or_else(|error| panic!("{name} prefill: {error}"));
        assert_eq!(logits.shape(), &[1, 64], "{name}");
        logits.evaluated().unwrap();
        let snapshot = executable.state_snapshot();
        assert!(
            snapshot
                .iter()
                .zip(&stateful_layers)
                .all(|((position, _), stateful)| *position == if *stateful { 2 } else { 0 }),
            "{name} prefill: {snapshot:?}"
        );
        for (step, token) in [3_u32, 4].into_iter().enumerate() {
            let logits = executable
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("{name} decode {step}: {error}"));
            assert_eq!(logits.shape(), &[1, 64], "{name}");
            logits.evaluated().unwrap();
            let snapshot = executable.state_snapshot();
            assert!(
                snapshot
                    .iter()
                    .zip(&stateful_layers)
                    .all(|((position, _), stateful)| *position
                        == if *stateful { step as i32 + 3 } else { 0 }),
                "{name} step {step}: {snapshot:?}"
            );
            assert!(snapshot
                .iter()
                .flat_map(|(_, fixed)| fixed)
                .all(|(_, present)| *present));
        }
    }
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts {
            architecture_constructions: 5,
            state_allocations: 5,
            payload_opens: 5,
            constructors: 5,
            unit_constructions: 12,
            materializations: 0,
            local_static_bindings: 0,
            excluded_local_static_parameters: 0,
            forwards: 15,
            state_publications: 15,
            completions: 15,
        }
    );
}
