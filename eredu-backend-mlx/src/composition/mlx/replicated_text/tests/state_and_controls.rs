use super::*;

#[test]
fn heterogeneous_logits_and_fixed_state_match_across_weight_and_state_residency() {
    let (stream, weights_stream) = execution_streams();
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 2).unwrap();
    let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    for (name, config) in [
        ("lfm2", lfm2_config()),
        ("kimi_linear", kimi_linear_config()),
        ("nemotron_h", nemotron_h_config()),
        ("qwen3_next", qwen_next_config()),
        ("qwen3_5_text", qwen_hybrid_config()),
    ] {
        let root = tiny_heterogeneous_artifact(config);
        let mut results = Vec::new();
        for (residency, state) in [
            (
                eredu_runtime::WeightResidency::fully_resident(),
                CacheResidencyPolicy::Device,
            ),
            (
                eredu_runtime::WeightResidency::dense_disk_stream(disk),
                CacheResidencyPolicy::Paged(paged.clone()),
            ),
        ] {
            let options = crate::MlxLoadRequest::default()
                .with_weight_residency(residency)
                .with_state_residency(state);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = materialize_model_plan(plan, options, &stream, &weights_stream)
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
            results.push((
                logits,
                generic.state_snapshot(),
                generic.fixed_numeric_state_snapshot().unwrap(),
            ));
        }
        let (resident_logits, resident_semantics, resident_fixed) = &results[0];
        let (bounded_logits, bounded_semantics, bounded_fixed) = &results[1];
        assert_eq!(resident_semantics, bounded_semantics, "{name}");
        assert_eq!(resident_fixed.len(), bounded_fixed.len(), "{name}");
        for (left, right) in resident_logits.iter().zip(bounded_logits) {
            assert!((left - right).abs() <= 1e-5, "{name}: {left} != {right}");
        }
        for (left, right) in resident_fixed.iter().zip(bounded_fixed) {
            assert_eq!((&left.0, &left.1, &left.2), (&right.0, &right.1, &right.2));
            for (left, right) in left.3.iter().zip(&right.3) {
                assert!((left - right).abs() <= 1e-5, "{name}: {left} != {right}");
            }
        }
    }
}

#[test]
fn heterogeneous_generic_sessions_preserve_every_state_component_across_controls() {
    struct Observer {
        activation: bool,
        logits: bool,
        intervened: bool,
        stream: Stream,
    }
    impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
        fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
            self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
            self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
            Ok(())
        }

        fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
            if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                self.intervened = true;
                Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
            } else {
                Ok(None)
            }
        }
    }

    let (stream, weights_stream) = execution_streams();
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 3).unwrap(),
    )
    .with_max_cached_shards(2);
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 2).unwrap();
    let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let cases = vec![
        (
            "lfm2",
            lfm2_config(),
            eredu_runtime::WeightResidency::fully_resident(),
            CacheResidencyPolicy::Device,
        ),
        (
            "lfm2-paged",
            lfm2_config(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            CacheResidencyPolicy::Paged(paged.clone()),
        ),
        (
            "kimi_linear",
            kimi_linear_config(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            CacheResidencyPolicy::Device,
        ),
        (
            "kimi_linear-paged",
            kimi_linear_config(),
            eredu_runtime::WeightResidency::dense_disk_stream(disk),
            CacheResidencyPolicy::Paged(paged.clone()),
        ),
        (
            "nemotron_h",
            nemotron_h_config(),
            eredu_runtime::WeightResidency::dense_disk_stream(disk),
            CacheResidencyPolicy::Device,
        ),
        (
            "nemotron_h-paged",
            nemotron_h_config(),
            eredu_runtime::WeightResidency::fully_resident(),
            CacheResidencyPolicy::Paged(paged.clone()),
        ),
        (
            "qwen3_next",
            qwen_next_config(),
            eredu_runtime::WeightResidency::fully_resident(),
            CacheResidencyPolicy::Device,
        ),
        (
            "qwen3_next-paged",
            qwen_next_config(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            CacheResidencyPolicy::Paged(paged.clone()),
        ),
        (
            "qwen3_5_text",
            qwen_hybrid_config(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            CacheResidencyPolicy::Device,
        ),
        (
            "qwen3_5_text-paged",
            qwen_hybrid_config(),
            eredu_runtime::WeightResidency::dense_disk_stream(disk),
            CacheResidencyPolicy::Paged(paged),
        ),
    ];

    for (name, config, residency, state_policy) in cases {
        let root = tiny_heterogeneous_artifact(config);
        let options = crate::MlxLoadRequest::default()
            .with_weight_residency(residency)
            .with_state_residency(state_policy.clone());
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(model.residency_report().unwrap().is_some());
        assert_eq!(
            model.dense_stream_report().unwrap().is_some(),
            matches!(
                residency,
                eredu_runtime::WeightResidency::Layers(
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                )
            ),
            "{name}"
        );
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        assert_eq!(generic.selected_residency(), residency.layers(), "{name}");
        assert_eq!(
            generic.cache_residency_report().unwrap().is_some(),
            matches!(state_policy, CacheResidencyPolicy::Paged(_)),
            "{name}"
        );

        let prefix = [1_u32, 2, 3];
        let prompt = Array::from_slice(&prefix, &[1, 3]);
        let parts = [input::token_ids_part(&prompt).unwrap()];
        generic
            .prefill(input::ModelInput::new(&parts), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let saved_snapshot = generic.state_snapshot();
        let saved_numeric = generic
            .fixed_numeric_state_snapshot()
            .unwrap_or_else(|error| panic!("{name} numeric snapshot: {error}"));
        assert!(!saved_numeric.is_empty(), "{name}");
        assert!(saved_snapshot
            .iter()
            .flat_map(|(_, fixed)| fixed)
            .all(|(_, present)| *present));

        let persisted = matches!(state_policy, CacheResidencyPolicy::Paged(_)).then(|| {
            let identity = generic.prompt_cache_model_identity().clone();
            let descriptor = PromptCacheDescriptor::from_model_identity(
                identity,
                format!("{name}-checkpoint"),
                "tokens:1,2,3",
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
            (cache_root, destination, descriptor)
        });
        let continuation_token = Array::from_slice(&[4_u32], &[1, 1]);
        let persistence_baseline = persisted
            .as_ref()
            .map(|_| generic.checkpoint_restore_probe(&continuation_token, &stream))
            .transpose()
            .unwrap_or_else(|error| panic!("{name} persistence baseline: {error}"));
        generic.reset_cache().unwrap();
        assert!(generic.state_snapshot().iter().all(|(position, fixed)| {
            *position == 0 && fixed.iter().all(|(_, present)| !present)
        }));
        if let Some((_cache_root, destination, descriptor)) = persisted {
            let incompatible = descriptor
                .clone()
                .with_architecture_fingerprint(format!(
                    "{}-different",
                    descriptor.architecture_fingerprint()
                ))
                .unwrap();
            assert!(generic
                .load_prompt_cache(&destination, &incompatible, &prefix)
                .is_err());
            generic
                .load_prompt_cache(&destination, &descriptor, &prefix)
                .unwrap();
        } else {
            assert!(generic
                .save_prompt_cache(
                    tempfile::tempdir().unwrap().path(),
                    PromptCacheDescriptor::from_model_identity(
                        generic.prompt_cache_model_identity().clone(),
                        format!("{name}-checkpoint"),
                        "tokens:1,2,3",
                        1,
                    )
                    .unwrap(),
                    &prefix,
                    &PromptCacheOptions::default(),
                )
                .is_err());
            generic
                .prefill(input::ModelInput::new(&parts), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
        assert_eq!(generic.state_snapshot(), saved_snapshot, "{name}");
        assert_eq!(
            generic.fixed_numeric_state_snapshot().unwrap(),
            saved_numeric,
            "{name} fixed tensors changed across prompt-cache restoration"
        );
        let probe = generic
            .checkpoint_restore_probe(&continuation_token, &stream)
            .unwrap_or_else(|error| panic!("{name} checkpoint/restore: {error}"));
        if let Some(baseline) = persistence_baseline {
            assert_eq!(
                probe, baseline,
                "{name} continuation changed after prompt-cache restoration"
            );
        }
        let (
            before,
            advanced,
            restored,
            before_numeric,
            advanced_numeric,
            restored_numeric,
            continuation,
        ) = probe;
        assert_eq!(before, saved_snapshot, "{name}");
        assert_ne!(advanced, before, "{name}");
        assert_eq!(restored, before, "{name}");
        assert_eq!(before_numeric, saved_numeric, "{name}");
        assert_ne!(advanced_numeric, before_numeric, "{name}");
        assert_eq!(restored_numeric, before_numeric, "{name}");
        let replayed = generic.decode(&continuation_token, &stream).unwrap();
        let replayed = replayed.evaluated().unwrap();
        assert_eq!(
            replayed.as_slice::<f32>(),
            continuation.as_slice(),
            "{name}"
        );
        assert_eq!(generic.state_snapshot(), advanced, "{name}");
        assert_eq!(
            generic.fixed_numeric_state_snapshot().unwrap(),
            advanced_numeric,
            "{name}"
        );

        let mut observer = Observer {
            activation: false,
            logits: false,
            intervened: false,
            stream: stream.clone(),
        };
        let replacement = generic
            .forward_with_observer(
                &Array::from_slice(&[5_u32], &[1, 1]),
                None,
                &stream,
                &mut observer,
            )
            .unwrap();
        let replacement = replacement.evaluated().unwrap();
        assert!(observer.activation && observer.logits, "{name}");
        assert!(observer.intervened, "{name}");
        assert!(
            replacement
                .as_slice::<f32>()
                .iter()
                .all(|value| *value == 0.0),
            "{name}"
        );
    }
}

#[test]
fn generic_controls_cover_residency_cache_persistence_and_observation() {
    struct Observer {
        activation: bool,
        logits: bool,
        intervened: bool,
        stream: Stream,
    }
    impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
        fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
            self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
            self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
            Ok(())
        }

        fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
            if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
                self.intervened = true;
                Ok(Some(safemlx::ops::zeros_like(value, &self.stream)?))
            } else {
                Ok(None)
            }
        }
    }

    let (stream, weights_stream) = execution_streams();
    let mut host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 7).unwrap(),
    );
    host = host.with_max_cached_shards(3);
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 5, 4).unwrap();
    for (model_type, residency) in ["llama", "qwen2"].into_iter().flat_map(|family| {
        [
            eredu_runtime::WeightResidency::fully_resident(),
            eredu_runtime::WeightResidency::layerwise_host(host),
            eredu_runtime::WeightResidency::dense_disk_stream(disk),
        ]
        .into_iter()
        .map(move |residency| (family, residency))
    }) {
        let root = tiny_artifact(model_type, false);
        let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true);
        let options = crate::MlxLoadRequest::default()
            .with_weight_residency(residency)
            .with_state_residency(CacheResidencyPolicy::Paged(paged.clone()));
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        assert!(model.residency_report().unwrap().is_some());
        assert_eq!(
            model.dense_stream_report().unwrap().is_some(),
            matches!(
                residency,
                eredu_runtime::WeightResidency::Layers(
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                )
            )
        );
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        assert_eq!(generic.selected_residency(), residency.layers());
        generic
            .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();

        let mut observer = Observer {
            activation: false,
            logits: false,
            intervened: false,
            stream: stream.clone(),
        };
        let replacement = generic
            .forward_with_observer(
                &Array::from_slice(&[3_u32], &[1, 1]),
                None,
                &stream,
                &mut observer,
            )
            .unwrap();
        let replacement = replacement.evaluated().unwrap();
        assert!(observer.logits);
        assert!(observer.activation);
        assert!(observer.intervened);
        assert!(replacement
            .as_slice::<f32>()
            .iter()
            .all(|value| *value == 0.0));

        let identity = generic.prompt_cache_model_identity().clone();
        let descriptor = PromptCacheDescriptor::from_model_identity(
            identity,
            "tiny-checkpoint",
            "tokens:1,2,3",
            1,
        )
        .unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let destination = cache_root.path().join("cache");
        let prefix = [1_u32, 2, 3];
        generic.reset_cache().unwrap();
        generic
            .decode(&Array::from_slice(&prefix, &[1, 3]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
        let manifest = generic
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &prefix,
                &PromptCacheOptions::default(),
            )
            .unwrap();
        assert_eq!(manifest.block_size_tokens, paged.block_size_tokens());
        let incompatible = descriptor
            .clone()
            .with_architecture_fingerprint(format!(
                "{}-different",
                descriptor.architecture_fingerprint()
            ))
            .unwrap();
        assert!(generic
            .load_prompt_cache(&destination, &incompatible, &prefix)
            .is_err());
        generic
            .load_prompt_cache(&destination, &descriptor, &prefix)
            .unwrap();
        assert!(generic.cache_residency_report().unwrap().is_some());
        generic
            .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
    }
}

#[test]
fn both_text_architectures_execute_checkpoint_native_packed_gguf_formats() {
    let (stream, weights_stream) = execution_streams();
    for (architecture, format) in [
        ("llama", eredu_gguf::GgmlType::Q4_0),
        ("llama", eredu_gguf::GgmlType::MxFp4),
        ("llama", eredu_gguf::GgmlType::IQ4NL),
        ("qwen2", eredu_gguf::GgmlType::Q4_0),
        ("qwen2", eredu_gguf::GgmlType::MxFp4),
        ("qwen2", eredu_gguf::GgmlType::IQ4NL),
    ] {
        let artifact = if architecture == "llama" {
            tiny_llama_gguf(architecture, Some(format), &stream)
        } else {
            tiny_qwen_gguf(architecture, Some(format), &stream)
        };
        let checkpoint = eredu_gguf::Checkpoint::open(artifact.path()).unwrap();
        let translated = if architecture == "llama" {
            checkpoint
                .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
                .unwrap()
        } else {
            checkpoint
                .translated_outputs(|name| {
                    eredu_architectures::qwen::translate_gguf_weight_name(name, false)
                })
                .unwrap()
        };
        if matches!(
            format,
            eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::MxFp4
        ) {
            assert!(translated.iter().any(|mapping| {
                mapping.original_name.ends_with(".scales")
                    && mapping.layout.name.ends_with(".scales")
                    && mapping.layout.name.starts_with("model.")
            }));
        }
        if format == eredu_gguf::GgmlType::Q4_0 {
            assert!(translated.iter().any(|mapping| {
                mapping.original_name.ends_with(".biases")
                    && mapping.layout.name.ends_with(".biases")
            }));
        }
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        assert!(requirements.parameters().iter().any(|parameter| {
            matches!(
                (format, parameter.native_executable()),
                (eredu_gguf::GgmlType::Q4_0, LinearFormat::Affine(_))
                    | (eredu_gguf::GgmlType::MxFp4, LinearFormat::MxFp4)
                    | (eredu_gguf::GgmlType::IQ4NL, LinearFormat::GgufIQuant { .. })
            )
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
        .unwrap_or_else(|error| panic!("{architecture} {format:?}: {error}"));
        let mut executable = model.into_executable();
        let executable = executable.erased_mut();
        executable
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
    }
}

#[test]
fn public_handoff_executes_admitted_gguf_mapping() {
    let (stream, weights_stream) = execution_streams();
    let artifacts = [
        tiny_llama_gguf("llama", None, &stream),
        tiny_llama_gguf("mistral", None, &stream),
        tiny_qwen_gguf("qwen2", None, &stream),
        tiny_qwen_gguf("qwen3", None, &stream),
    ];
    for artifact in artifacts {
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
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
        let executable = executable.erased_mut();
        let logits = executable
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 64]);
        logits.evaluated().unwrap();
    }
}

#[test]
fn heterogeneous_generic_handoff_executes_selected_load_time_transforms() {
    let (stream, weights_stream) = execution_streams();
    let mut lfm_affine = lfm2_config();
    lfm_affine["hidden_size"] = 32.into();
    lfm_affine["intermediate_size"] = 32.into();
    lfm_affine["num_key_value_heads"] = 1.into();
    lfm_affine["block_auto_adjust_ff_dim"] = false.into();
    let mut kimi_affine = kimi_linear_config();
    kimi_affine["hidden_size"] = 32.into();
    kimi_affine["intermediate_size"] = 32.into();
    kimi_affine["kv_lora_rank"] = 32.into();
    kimi_affine["moe_intermediate_size"] = 32.into();
    kimi_affine["linear_attn_config"]["num_heads"] = 4.into();
    kimi_affine["linear_attn_config"]["head_dim"] = 32.into();
    kimi_affine["num_attention_heads"] = 4.into();
    kimi_affine["qk_nope_head_dim"] = 24.into();
    kimi_affine["qk_rope_head_dim"] = 8.into();
    kimi_affine["v_head_dim"] = 8.into();
    let mut nemotron_affine = nemotron_h_config();
    nemotron_affine["hidden_size"] = 32.into();
    nemotron_affine["intermediate_size"] = 32.into();
    nemotron_affine["num_attention_heads"] = 8.into();
    nemotron_affine["num_key_value_heads"] = 4.into();
    nemotron_affine["mamba_num_heads"] = 8.into();
    nemotron_affine["moe_intermediate_size"] = 32.into();
    nemotron_affine["moe_shared_expert_intermediate_size"] = 32.into();
    let mut lfm_mxfp4 = lfm2_config();
    lfm_mxfp4["hidden_size"] = 32.into();
    lfm_mxfp4["intermediate_size"] = 64.into();
    lfm_mxfp4["num_key_value_heads"] = 1.into();
    lfm_mxfp4["block_auto_adjust_ff_dim"] = false.into();
    let mut qwen_mxfp4 = qwen_hybrid_config();
    qwen_mxfp4["intermediate_size"] = 64.into();
    for (name, config, request) in [
        (
            "lfm2-affine",
            lfm_affine,
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
        ),
        (
            "kimi-affine",
            kimi_affine,
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
        ),
        (
            "nemotron-affine",
            nemotron_affine,
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
        ),
        (
            "lfm2-mxfp4",
            lfm_mxfp4,
            eredu_core::QuantizationRequest::MxFp4,
        ),
        (
            "qwen-mxfp4",
            qwen_mxfp4,
            eredu_core::QuantizationRequest::MxFp4,
        ),
    ] {
        let root = tiny_heterogeneous_artifact(config);
        let options = crate::MlxLoadRequest::with_quantization(request);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let report = model
            .materialization_report()
            .unwrap_or_else(|| panic!("{name}: no materialization report"));
        assert!(report.transformed_weights > 0, "{name}");
        let mut executable = model.into_executable();
        let generic = executable.erased_mut();
        generic
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .evaluated()
            .unwrap();
    }
}

#[test]
fn public_handoff_executes_selected_load_time_transform() {
    let (stream, weights_stream) = execution_streams();
    for (model_type, request) in [
        (
            "llama",
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
        ),
        ("llama", eredu_core::QuantizationRequest::MxFp4),
        (
            "qwen3",
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
        ),
        ("qwen3", eredu_core::QuantizationRequest::MxFp4),
    ] {
        let root = tiny_artifact(model_type, false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let options = crate::MlxLoadRequest::with_quantization(request);
        let plan = eredu_core::plan_model_preparation(
            inspection,
            options.preparation_policy().unwrap(),
            eredu_core::SessionCapabilities::default(),
        )
        .unwrap();
        let model = materialize_model_plan(plan, options, &stream, &weights_stream).unwrap();
        assert!(model.materialization_report().is_some());
        let mut executable = model.into_executable();
        let executable = executable.erased_mut();
        executable
            .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
            .unwrap()
            .evaluated()
            .unwrap();
    }
}
