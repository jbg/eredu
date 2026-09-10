#[test]
fn qwen_hybrid_aliased_fp8_experts_execute_resident_and_addressable() {
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

    let mut config = routed_qwen_hybrid_config();
    config["moe_intermediate_size"] = 128.into();
    // Each segmented FP8 projection must end on a complete scale block.
    config["linear_key_head_dim"] = 64.into();
    config["linear_value_head_dim"] = 32.into();
    config["shared_expert_intermediate_size"] = 128.into();
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.embed_tokens"]
    });
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let plan = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let constraints = plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .filter(|group| group.required)
            .flat_map(|group| {
                group
                    .variants
                    .iter()
                    .find(|variant| variant.id == "independent")
                    .unwrap_or(&group.variants[0])
                    .tensors
                    .iter()
            }),
    );
    let tensors = constraints
        .filter(|constraint| {
            constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
        })
        .map(|constraint| {
            let elements = constraint.shape.iter().product::<usize>();
            let (dtype, bytes) = if constraint.dtype
                == eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::F8E4M3,
                ) {
                (Dtype::F8_E4M3, vec![0x18_u8; elements])
            } else {
                let value = if constraint.key.contains("norm")
                    || constraint.role == eredu_checkpoint::schema::TensorRole::Companion
                {
                    1.0_f32
                } else {
                    0.005_f32
                };
                (
                    Dtype::F32,
                    (0..elements).flat_map(|_| value.to_le_bytes()).collect(),
                )
            };
            (
                constraint
                    .key
                    .replacen("model.", "model.language_model.", 1),
                constraint.shape.clone(),
                dtype,
                bytes,
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &root.path().join("model.safetensors"),
    )
    .unwrap();

    let (stream, weights_stream) = execution_streams();
    let mut outputs = Vec::new();
    for addressable in [false, true] {
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
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
        let mut logits = Vec::new();
        for token in [1_u32, 2] {
            let output = executable
                .erased_mut()
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            assert_eq!(output.len(), 64);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(output.iter().any(|value| *value != 0.0));
            logits.extend(output);
        }
        outputs.push(logits);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn routed_deepseek_v4_pooling_state_uses_shared_checkpoint_and_prompt_cache_controls() {
    let (stream, weights_stream) = execution_streams();
    let root = tiny_heterogeneous_artifact(routed_deepseek_v4_config());
    let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default()
            .with_weight_residency(
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    eredu_runtime::OrdinaryWeightResidency::FullyResident,
                    eredu_runtime::ParameterBankLoadOptions::default(),
                ),
            )
            .with_state_residency(CacheResidencyPolicy::Paged(paged)),
    );
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        options.normalized().preparation_policy().unwrap(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
    let mut executable = model.into_executable();
    let generic = executable.erased_mut();
    let prefix = [1_u32, 2, 3, 4, 5];
    let prompt = Array::from_slice(&prefix, &[1, 5]);
    let parts = [input::token_ids_part(&prompt).unwrap()];
    generic
        .prefill(input::ModelInput::new(&parts), &stream)
        .unwrap()
        .evaluated()
        .unwrap();
    let before = generic.state_snapshot();
    let before_numeric = generic.fixed_numeric_state_snapshot().unwrap();
    assert!(before
        .iter()
        .any(|(_, components)| { components.iter().any(|(_, present)| *present) }));
    assert!(!before_numeric.is_empty());

    let continuation = Array::from_slice(&[6_u32], &[1, 1]);
    let probe = generic
        .checkpoint_restore_probe(&continuation, &stream)
        .unwrap();
    assert_eq!(probe.0, probe.2);
    assert_eq!(probe.3, probe.5);

    let descriptor = PromptCacheDescriptor::from_model_identity(
        generic.prompt_cache_model_identity().clone(),
        "deepseek-v4-checkpoint",
        "tokens:1,2,3,4,5",
        1,
    )
    .unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let destination = cache_root.path().join("cache");
    generic
        .save_prompt_cache(
            &destination,
            descriptor.clone(),
            &prefix,
            &PromptCacheOptions::default(),
        )
        .unwrap();
    generic.reset_cache().unwrap();
    assert!(generic.state_snapshot().iter().all(|(offset, components)| {
        *offset == 0 && components.iter().all(|(_, present)| !present)
    }));
    generic
        .load_prompt_cache(&destination, &descriptor, &prefix)
        .unwrap();
    assert_eq!(generic.state_snapshot(), before);
    assert_eq!(
        generic.fixed_numeric_state_snapshot().unwrap(),
        before_numeric
    );
    let restored = generic
        .decode(&continuation, &stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(restored, probe.6);
}

#[test]
fn routed_deepseek_v4_executes_resident_and_addressable_with_pooling_state() {
    let (stream, weights_stream) = execution_streams();
    for addressable in [false, true] {
        let root = tiny_heterogeneous_artifact(routed_deepseek_v4_config());
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
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
        let executable = executable.erased_mut();
        for token in [1_u32, 2, 3, 4, 5] {
            let logits = executable
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            assert_eq!(logits.shape(), &[1, 64]);
            assert!(logits
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .all(|value| value.is_finite()));
        }
        assert_eq!(executable.state_snapshot().len(), 3);
    }
}

#[test]
fn routed_only_default_observation_intervenes_on_provider_output() {
    struct Observer {
        routing_path: Option<String>,
        routed_only: bool,
        intervened: bool,
        stream: Stream,
    }

    impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
        fn observe(&mut self, _: &str, _: &Array) -> Result<(), Exception> {
            Ok(())
        }

        fn observe_routing(
            &mut self,
            observation: eredu_runtime::RoutingObservation<'_, Array>,
        ) -> Result<(), Exception> {
            self.routed_only =
                observation.shared_output.is_none() && observation.combined_output.is_none();
            self.routing_path = Some(observation.path.to_owned());
            Ok(())
        }

        fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
            if self
                .routing_path
                .as_deref()
                .is_some_and(|routing| path == format!("{routing}.output"))
            {
                self.intervened = true;
                Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
            } else {
                Ok(None)
            }
        }
    }

    let (stream, weights_stream) = execution_streams();
    let root = tiny_artifact("qwen3_moe", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            ),
        ),
    );
    let plan = eredu_core::plan_model_preparation(
        inspection,
        options.normalized().preparation_policy().unwrap(),
        eredu_core::SessionCapabilities::default(),
    )
    .unwrap();
    let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
    let mut executable = model.into_executable();
    let generic = executable.erased_mut();
    let tokens = Array::from_slice(&[3_u32], &[1, 1]);
    let baseline = generic
        .decode(&tokens, &stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    generic.reset_cache().unwrap();
    let mut observer = Observer {
        routing_path: None,
        routed_only: false,
        intervened: false,
        stream: stream.clone(),
    };
    let changed = generic
        .forward_with_observer(&tokens, None, &stream, &mut observer)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(observer.routed_only);
    assert!(observer.intervened);
    assert_ne!(baseline, changed);
}

#[test]
fn routed_session_observation_reports_shared_combination_and_intervenes_causally() {
    struct Observer {
        routing_path: Option<String>,
        semantic_outputs: bool,
        intervened: bool,
        stream: Stream,
    }

    impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
        fn observe(&mut self, _: &str, _: &Array) -> Result<(), Exception> {
            Ok(())
        }

        fn observe_routing(
            &mut self,
            observation: eredu_runtime::RoutingObservation<'_, Array>,
        ) -> Result<(), Exception> {
            self.semantic_outputs =
                observation.shared_output.is_some() && observation.combined_output.is_some();
            self.routing_path = Some(observation.path.to_owned());
            Ok(())
        }

        fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
            let routed_output = self
                .routing_path
                .as_deref()
                .is_some_and(|routing| path == format!("{routing}.output"));
            if routed_output {
                self.intervened = true;
                Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
            } else {
                Ok(None)
            }
        }
    }

    let (stream, weights_stream) = execution_streams();
    for addressable in [false, true] {
        let root = tiny_heterogeneous_artifact(routed_kimi_linear_config());
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
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
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        let tokens = Array::from_slice(&[3_u32], &[1, 1]);
        let baseline = generic
            .decode(&tokens, &stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        generic.reset_cache().unwrap();
        let mut observer = Observer {
            routing_path: None,
            semantic_outputs: false,
            intervened: false,
            stream: stream.clone(),
        };
        let changed = generic
            .forward_with_observer(&tokens, None, &stream, &mut observer)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(observer.semantic_outputs, "addressable={addressable}");
        assert!(observer.intervened, "addressable={addressable}");
        assert_ne!(baseline, changed, "addressable={addressable}");
    }
}

#[test]
fn routed_gated_families_execute_resident_and_addressable_with_heterogeneous_state() {
    let (stream, weights_stream) = execution_streams();
    for (name, config) in [
        ("lfm2_moe", routed_lfm2_config()),
        ("kimi_linear", routed_kimi_linear_config()),
        ("qwen3_5_moe_text", routed_qwen_hybrid_config()),
        ("qwen3_next", routed_qwen_next_config()),
        ("deepseek_v3", routed_deepseek_v3_config()),
    ] {
        for addressable in [false, true] {
            let root = tiny_heterogeneous_artifact(config.clone());
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
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
                .unwrap_or_else(|error| panic!("{name} addressable={addressable}: {error}"));
            let mut executable = model.into_executable();
            let executable = executable.erased_mut();
            for token in [1_u32, 2, 3] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("{name} addressable={addressable}: {error}"));
                assert_eq!(logits.shape(), &[1, 64]);
                logits.evaluated().unwrap();
            }
            assert_eq!(
                executable.state_snapshot().len(),
                2,
                "{name} state layout must retain both target layers"
            );
        }
    }
}

#[test]
fn routed_nemotron_relu2_executes_resident_and_addressable_with_mixed_state() {
    let (stream, weights_stream) = execution_streams();
    let mut config = nemotron_h_config();
    config["hybrid_override_pattern"] = "M*EM".into();
    for addressable in [false, true] {
        let root = tiny_heterogeneous_artifact(config.clone());
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
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
        let executable = executable.erased_mut();
        for token in [1_u32, 2, 3] {
            let logits = executable
                .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
            assert_eq!(logits.shape(), &[1, 64]);
            logits.evaluated().unwrap();
        }
        let state = executable.state_snapshot();
        assert_eq!(state.len(), 4);
        assert!(state
            .iter()
            .all(|(_, components)| components.iter().all(|(_, present)| *present)));
    }
}

#[test]
fn gpt_oss_load_time_transform_preserves_native_experts_in_both_residencies() {
    let (stream, weights_stream) = execution_streams();
    for addressable in [false, true] {
        let root = tiny_artifact("gpt_oss", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let requirements = eredu_architectures::routed_text_requirements(&inspection).unwrap();
        let text = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_quantization(eredu_core::QuantizationRequest::Affine {
            group_size: 32,
            bits: 4,
        });
        let weights = if addressable {
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                eredu_runtime::ParameterBankLoadOptions::default(),
            )
        } else {
            eredu_runtime::WeightResidency::fully_resident()
        };
        let request = eredu_architectures::RoutedTextSelectionRequest::new(text, weights).unwrap();
        let selected = eredu_architectures::select_routed_text_realization(
            &requirements,
            &request,
            &capabilities(requirements.text(), request.text()),
        )
        .unwrap();
        let expert_targets = requirements
            .banks()
            .values()
            .flat_map(|bank| bank.catalog().logical_targets())
            .collect::<std::collections::BTreeSet<_>>();
        let selected_experts = requirements
            .text()
            .parameters()
            .iter()
            .filter(|parameter| {
                expert_targets.contains(parameter.name())
                    && parameter.role() == eredu_runtime::ReplicatedTextParameterRole::LinearWeight
            })
            .map(|requirement| {
                selected
                    .text()
                    .parameters()
                    .iter()
                    .find(|parameter| parameter.name() == requirement.name())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(!selected_experts.is_empty());
        assert!(selected_experts
            .iter()
            .all(|parameter| { parameter.executable() == eredu_checkpoint::LinearFormat::MxFp4 }));
        assert!(selected.text().parameters().iter().any(|parameter| {
            !expert_targets.contains(parameter.name())
                && parameter.lowering() == eredu_runtime::WeightLoweringKind::Transform
        }));

        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::with_quantization(
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            )
            .with_weight_residency(weights),
        );
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"));
        assert!(
            model
                .materialization_report()
                .is_some_and(|report| report.transformed_weights > 0),
            "addressable={addressable}"
        );
        let mut complete = model.into_executable();
        let generic = complete.erased_mut();
        generic
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap_or_else(|error| panic!("addressable={addressable}: {error}"))
            .evaluated()
            .unwrap();
    }
}

#[test]
fn routed_addressable_load_time_transform_uses_selected_bank_geometry() {
    let (stream, weights_stream) = execution_streams();
    let mut nemotron = nemotron_h_config();
    nemotron["hybrid_override_pattern"] = "M*EM".into();
    nemotron["hidden_size"] = 32.into();
    nemotron["intermediate_size"] = 32.into();
    nemotron["head_dim"] = 8.into();
    nemotron["mamba_head_dim"] = 8.into();
    nemotron["moe_intermediate_size"] = 32.into();
    nemotron["moe_shared_expert_intermediate_size"] = 32.into();
    let artifacts = [
        ("qwen", tiny_artifact("qwen3_moe", false)),
        ("nemotron", tiny_heterogeneous_artifact(nemotron)),
    ];
    for (name, root) in artifacts {
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let residency = eredu_runtime::WeightResidency::with_independent_parameter_banks(
            eredu_runtime::OrdinaryWeightResidency::FullyResident,
            eredu_runtime::ParameterBankLoadOptions::default(),
        );
        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::with_quantization(
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            )
            .with_weight_residency(residency),
        );
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut complete = model.into_executable();
        {
            let executable = complete.erased_mut();
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .evaluated()
                .unwrap();
        }
        let report = complete
            .parameter_bank_report()
            .unwrap()
            .unwrap_or_else(|| panic!("{name}: no addressable-bank telemetry"));
        let report = &report.banks()[&eredu_runtime::RoutedBankId::new(0)];
        assert_eq!(
            report.weight_quantizations(),
            [eredu_checkpoint::WeightQuantization::Affine(
                eredu_checkpoint::AffineQuantization::new(32, 4).unwrap()
            )],
            "{name}"
        );
        let materialization = report
            .materialization()
            .unwrap_or_else(|| panic!("{name}: no bank materialization telemetry"));
        assert!(materialization.transformed_weights > 0, "{name}");
        assert_eq!(materialization.output_bytes, report.owned_bytes(), "{name}");
        assert!(
            materialization.source_bytes_read > materialization.output_bytes,
            "{name}"
        );
    }
}

#[test]
fn routed_addressable_storage_executes_qwen_and_gpt_oss_repeated_decode() {
    let (stream, weights_stream) = execution_streams();
    for model_type in ["qwen3_moe", "gpt_oss"] {
        let root = tiny_artifact(model_type, false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let residency = eredu_runtime::WeightResidency::with_independent_parameter_banks(
            eredu_runtime::OrdinaryWeightResidency::FullyResident,
            eredu_runtime::ParameterBankLoadOptions::default(),
        );
        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
        );
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.normalized().preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
        let mut complete = model.into_executable();
        {
            let executable = complete.erased_mut();
            let prompt = Array::from_slice(&[1_u32, 2], &[1, 2]);
            let parts = [input::token_ids_part(&prompt).unwrap()];
            executable
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap_or_else(|error| panic!("{model_type}: {error}"))
                .evaluated()
                .unwrap();
            for token in [3_u32, 4, 5] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap_or_else(|error| panic!("{model_type}: {error}"));
                assert_eq!(logits.shape(), &[1, 64]);
                assert!(logits
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .iter()
                    .all(|value| value.is_finite()));
            }
        }
        let report = complete
            .parameter_bank_report()
            .unwrap()
            .unwrap_or_else(|| panic!("{model_type}: no addressable-bank telemetry"));
        assert!(report.owned_entries() > 0, "{model_type}");
        assert!(report.bulk().requested_selections() > 0, "{model_type}");
        assert!(report.bulk().compact_banks() > 0, "{model_type}");
        assert!(
            report.incremental().requested_selections() > 0,
            "{model_type}"
        );
        assert!(report.incremental().compact_banks() > 0, "{model_type}");
    }
}

#[test]
fn routed_qwen_gguf_executes_resident_and_addressable_through_generic_composition() {
    let (stream, weights_stream) = execution_streams();
    let artifact = tiny_qwen_moe_gguf(&stream);
    for addressable in [false, true] {
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
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
        let logits = generic
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 64]);
        logits.evaluated().unwrap();
    }
}
