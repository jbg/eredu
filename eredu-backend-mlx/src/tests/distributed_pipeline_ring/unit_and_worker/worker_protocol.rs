#[test]
fn pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let family = FixtureFamily::parse(&std::env::var(FIXTURE_FAMILY).unwrap());
    let prompt_cache_root = PathBuf::from(std::env::var_os(PROMPT_CACHE_ROOT).unwrap());
    let native_group = distributed::init(true, Backend::Ring).unwrap();
    let cartesian_axes = std::env::var(CARTESIAN_AXES).ok();
    let (tensor_parallel_size, pipeline_parallel_size, expert_parallel_size) =
        match cartesian_axes.as_deref() {
            None => (1, 2, 1),
            Some("tp") => (2, 1, 1),
            Some("ep") => (1, 1, 2),
            Some("tp-pp") => (2, 2, 1),
            Some("tp-ep") => (2, 1, 2),
            Some("pp-ep") => (1, 2, 2),
            Some("tp-pp-ep") => (2, 2, 2),
            Some(other) => panic!("unexpected Cartesian pipeline axes {other:?}"),
        };
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            1,
        )
        .unwrap(),
        expected_rank,
    )
    .unwrap();
    assert_eq!(topology.global_rank(), expected_rank);
    let pipeline_rank = topology.pipeline_parallel_rank();
    let neutral_gemma_config =
        (family == FixtureFamily::Gemma && std::env::var_os(OPAQUE_SESSION).is_some()).then(|| {
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(checkpoint.join("config.json")).expect("Gemma config"),
            )
            .expect("Gemma JSON config");
            config
        });
    let neutral_gemma_layers = neutral_gemma_config.as_ref().map(|config| {
        config["text_config"]["num_hidden_layers"]
            .as_u64()
            .expect("Gemma text layer count") as usize
    });
    let neutral_prediction_target_layers = ((family == FixtureFamily::Inkling
        && std::env::var_os(OPAQUE_INKLING_MTP).is_some())
        || (family == FixtureFamily::Qwen35Multimodal
            && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some())
        || (family == FixtureFamily::NemotronH
            && std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_some()))
    .then(|| {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(checkpoint.join("config.json")).expect("prediction target config"),
        )
        .expect("prediction target JSON config");
        if family == FixtureFamily::NemotronH {
            config["num_hidden_layers"]
                .as_u64()
                .expect("prediction target layer count") as usize
        } else {
            config["text_config"]["num_hidden_layers"]
                .as_u64()
                .expect("prediction target layer count") as usize
        }
    });
    let neutral_qwen_vl_config =
        (matches!(family, FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe)
            && std::env::var_os(OPAQUE_QWEN3_VL_MEDIA).is_some())
        .then(|| {
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(checkpoint.join("config.json")).expect("Qwen3-VL config"),
            )
            .expect("Qwen3-VL JSON config")
        });
    let local_layer_range =
        if let Some(layers) = neutral_gemma_layers.or(neutral_prediction_target_layers) {
            layers * pipeline_rank / pipeline_parallel_size
                ..layers * (pipeline_rank + 1) / pipeline_parallel_size
        } else if pipeline_parallel_size == 1 {
            0..family.layer_count()
        } else {
            family.stage_range(pipeline_rank)
        };
    let public_output_owner = topology
        .topology()
        .rank_for(eredu_core::ParallelCoordinates::new(
            0,
            pipeline_parallel_size - 1,
            0,
            topology.data_parallel_rank(),
        ))
        .unwrap();
    let owns_public_output = expected_rank == public_output_owner;
    let device = DeviceAssignment::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device.device().unwrap());
    if std::env::var_os(OPAQUE_SESSION).is_some() {
        let dense_composite_neutral = matches!(
            family,
            FixtureFamily::MuseGlimmer
                | FixtureFamily::InklingDense
                | FixtureFamily::InklingDenseMultimodal
                | FixtureFamily::Qwen35ZeroPrediction
        ) && matches!(
            cartesian_axes.as_deref(),
            None | Some("tp") | Some("pp") | Some("tp-pp")
        );
        let dense_composite_neutral = dense_composite_neutral
            || (family == FixtureFamily::Qwen35Multimodal
                && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some()
                && matches!(
                    cartesian_axes.as_deref(),
                    None | Some("tp") | Some("pp") | Some("tp-pp")
                ));
        let dense_composite_auxiliary_units = dense_composite_neutral
            .then(|| {
                serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(checkpoint.join("config.json")).expect("dense composite config"),
                )
                .expect("dense composite JSON config")
            })
            .map_or(0, |config| match family {
                FixtureFamily::MuseGlimmer => {
                    let depth = config["vision_config"]["num_hidden_layers"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                    if pipeline_rank == 0 {
                        depth
                    } else {
                        0
                    }
                }
                FixtureFamily::Qwen35ZeroPrediction | FixtureFamily::Qwen35Multimodal => {
                    let depth = config["vision_config"]["depth"].as_u64().unwrap_or(0) as usize;
                    depth * (pipeline_rank + 1) / pipeline_parallel_size
                        - depth * pipeline_rank / pipeline_parallel_size
                }
                _ => 0,
            });
        let routed_neutral = (matches!(
            family,
            FixtureFamily::Qwen3Moe
                | FixtureFamily::Qwen3MoeGguf
                | FixtureFamily::GptOss
                | FixtureFamily::GptOssGguf
                | FixtureFamily::DeepSeek
                | FixtureFamily::DeepSeekGguf
                | FixtureFamily::NemotronH
                | FixtureFamily::NemotronHGguf
                | FixtureFamily::Lfm2MoeGguf
                | FixtureFamily::KimiLinearGguf
                | FixtureFamily::Qwen3VlMoe
                | FixtureFamily::Inkling
                | FixtureFamily::InklingMultimodal
        ) || (family == FixtureFamily::DeepSeekV4
            && (std::env::var_os(PREDICTION_FREE_TARGET).is_some()
                || std::env::var_os(PREPARED_SPECULATIVE_CAPABILITY).is_some())))
            && matches!(
                cartesian_axes.as_deref(),
                None | Some("tp")
                    | Some("ep")
                    | Some("tp-pp")
                    | Some("tp-ep")
                    | Some("pp-ep")
                    | Some("tp-pp-ep")
            );
        let prove_prepared_communication_lifecycle = dense_composite_neutral
            || routed_neutral
            || (family == FixtureFamily::Qwen3Vl
                && matches!(cartesian_axes.as_deref(), None | Some("tp") | Some("tp-pp")))
            || (matches!(
                family,
                FixtureFamily::Llama
                    | FixtureFamily::Mistral
                    | FixtureFamily::Qwen2
                    | FixtureFamily::Qwen2Gguf
                    | FixtureFamily::Qwen3
                    | FixtureFamily::Qwen3Gguf
                    | FixtureFamily::KimiLinear
                    | FixtureFamily::NemotronH
                    | FixtureFamily::Lfm2
                    | FixtureFamily::Gemma
            ) && matches!(
                cartesian_axes.as_deref(),
                None | Some("tp") | Some("pp") | Some("tp-pp")
            ));
        let prove_direct_expert_communication =
            cartesian_axes.as_deref() == Some("ep") && !routed_neutral;
        if prove_prepared_communication_lifecycle || prove_direct_expert_communication {
            crate::tests::support::path_instrumentation::reset();
        }
        let backend = crate::native::distributed_backend(&stream, &stream, &native_group);
        let selected_paged = PagedCacheOptions::new(1, 32768, 32768, 1)
            .unwrap()
            .with_full_attention(true);
        let dense_stream = std::env::var_os(DENSE_STREAM).is_some();
        let layerwise_host = std::env::var_os(LAYERWISE_HOST).is_some();
        assert!(!(dense_stream && layerwise_host));
        let load_options = if std::env::var_os(REQUANTIZE).is_some() {
            let request = if family == FixtureFamily::NemotronH {
                eredu_core::QuantizationRequest::MxFp4
            } else {
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                }
            };
            eredu_runtime::NormalizedLoadRequest::with_quantization(request)
        } else {
            eredu_runtime::NormalizedLoadRequest::default()
        };
        let load_options = if std::env::var_os(EXPERT_CACHE).is_some() {
            let ordinary = if dense_stream {
                OrdinaryWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
                )
            } else if layerwise_host {
                OrdinaryWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(None, None, 1).unwrap(),
                ))
            } else {
                OrdinaryWeightResidency::FullyResident
            };
            let bank = if std::env::var_os(EXPERT_CACHE_EVICTION).is_some() {
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(12_288), Some(0), 1).unwrap(),
                    u64::MAX,
                    1 << 30,
                )
                .unwrap()
            } else {
                ParameterBankLoadOptions::default()
            };
            load_options.with_weight_residency(WeightResidency::with_independent_parameter_banks(
                ordinary, bank,
            ))
        } else if dense_stream {
            load_options.with_weight_residency(WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
            ))
        } else if layerwise_host {
            load_options.with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            ))
        } else {
            load_options
        }
        .with_state_residency(CacheResidencyPolicy::Paged(selected_paged));
        let load_options = MlxLoadRequest::from_normalized(load_options)
            .with_parallel_topology(
                topology,
                device,
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                4,
                4096,
                ring_completion_policy(),
            )
            .unwrap();
        if std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some()
            && family == FixtureFamily::DeepSeekV4
        {
            let inspection = eredu_architectures::configuration::inspect_artifact(&checkpoint)
                .expect("V4 prediction artifact inspection");
            let selected = crate::composition::mlx::loading::select_preparation(
                &inspection,
                load_options.clone(),
            )
            .expect("V4 prediction target selection");
            assert_eq!(
                selected
                    .neutral()
                    .prediction_extension()
                    .map(|extension| extension.kind()),
                Some(
                    eredu_architectures::configuration::PredictionExtensionKind::DeepSeekV4Embedded
                )
            );
            assert!(selected.neutral().communication_manifest().is_some());
            assert!(selected.rank_context().is_some());
        }
        let model = match load_model(&backend, &checkpoint, load_options) {
            Ok(_) if std::env::var_os(EXPECTED_UNSUPPORTED_DIRECT_PARTITION).is_some() => {
                panic!("unsupported direct partition route unexpectedly loaded")
            }
            Ok(model) => model,
            Err(error) if std::env::var_os(EXPECTED_UNSUPPORTED_DIRECT_PARTITION).is_some() => {
                assert!(
                    error
                        .to_string()
                        .contains("has no neutral production implementation"),
                    "unsupported direct partition failed for an unexpected reason: {error}"
                );
                return;
            }
            Err(error) => panic!("failed to load Ring fixture: {error}"),
        };
        if prove_prepared_communication_lifecycle {
            assert_eq!(
                crate::tests::support::path_instrumentation::snapshot().payload_opens,
                1,
                "included dense decoder must open its admitted payload store exactly once"
            );
            assert_eq!(
                crate::tests::support::path_instrumentation::communication_realization_attempts(),
                1,
                "included dense decoder must realize its prepared communication exactly once before payload construction"
            );
            assert_eq!(
                crate::tests::support::path_instrumentation::manifest_communication_realization_attempts(),
                1,
                "eligible dense-decoder TP/PP must realize its neutral manifest exactly once"
            );
            if matches!(
                cartesian_axes.as_deref(),
                None | Some("tp")
                    | Some("pp")
                    | Some("ep")
                    | Some("tp-pp")
                    | Some("tp-ep")
                    | Some("pp-ep")
                    | Some("tp-pp-ep")
            ) {
                assert_eq!(
                    crate::tests::support::path_instrumentation::neutral_partitioned_constructions(
                    ),
                    1,
                    "eligible dense-decoder TP/PP must construct the neutral partitioned session"
                );
                assert_eq!(
                    crate::tests::support::path_instrumentation::snapshot().unit_constructions,
                    local_layer_range.len()
                        + neutral_gemma_config.as_ref().map_or(0, |config| {
                            if pipeline_rank == 0 {
                                ["vision_config", "audio_config"]
                                    .iter()
                                    .map(|root| {
                                        config[*root]["num_hidden_layers"].as_u64().unwrap_or(0)
                                            as usize
                                    })
                                    .sum::<usize>()
                            } else {
                                0
                            }
                        })
                        + neutral_qwen_vl_config.as_ref().map_or(0, |config| {
                            let depth =
                                config["vision_config"]["depth"].as_u64().unwrap_or(0) as usize;
                            depth * (pipeline_rank + 1) / pipeline_parallel_size
                                - depth * pipeline_rank / pipeline_parallel_size
                        })
                        + dense_composite_auxiliary_units,
                    "neutral partition construction must bind every local unit exactly once"
                );
                assert_eq!(
                    crate::tests::support::path_instrumentation::snapshot().materializations,
                    usize::from(std::env::var_os(REQUANTIZE).is_some()),
                    "neutral construction must execute exactly the selected transform groups"
                );
            }
            if neutral_gemma_layers.is_some() && pipeline_parallel_size > 1 {
                let counts = crate::tests::support::path_instrumentation::snapshot();
                let has_media = neutral_gemma_config.as_ref().is_some_and(|config| {
                    config.get("vision_config").is_some() || config.get("audio_config").is_some()
                });
                assert_eq!(
                    counts.local_static_bindings,
                    if has_media && pipeline_rank == 0 {
                        12
                    } else if pipeline_rank == 0 {
                        1
                    } else {
                        2
                    },
                    "only exact first-owner ingress or last-owner output statics may be bound"
                );
                assert_eq!(
                    counts.excluded_local_static_parameters,
                    if has_media && pipeline_rank != 0 {
                        12
                    } else if pipeline_rank == 0 {
                        2
                    } else {
                        1
                    },
                    "every unowned static definition must remain unbound and unread"
                );
            }
            if neutral_qwen_vl_config.is_some() {
                let counts = crate::tests::support::path_instrumentation::snapshot();
                assert!(
                    counts.local_static_bindings > 0,
                    "Qwen3-VL must bind its selected stage-local static tasks"
                );
                if pipeline_parallel_size == 1 {
                    assert_eq!(
                        counts.excluded_local_static_parameters, 0,
                        "Qwen3-VL pure TP owns every selected static task"
                    );
                } else {
                    assert!(
                        counts.excluded_local_static_parameters > 0,
                        "Qwen3-VL PP must leave non-owned static parameters unbound"
                    );
                }
            }
        }
        if prove_direct_expert_communication {
            assert_eq!(
                crate::tests::support::path_instrumentation::communication_realization_attempts(),
                1
            );
            assert_eq!(
                crate::tests::support::path_instrumentation::manifest_communication_realization_attempts(),
                0,
                "direct expert communication must not realize a partition manifest"
            );
        }
        let expected_effective_model_type = family.effective_model_type();
        assert_eq!(model.effective_model_type(), expected_effective_model_type);
        if dense_stream {
            let report = model.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), local_layer_range.len());
            let streamed = report
                .residency()
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Disk)
                .collect::<Vec<_>>();
            assert_eq!(streamed.len(), local_layer_range.len());
            assert!(streamed
                .iter()
                .all(|unit| !unit.host_resident() && !unit.device_resident()));
        }
        if layerwise_host {
            assert!(model.dense_stream_report().unwrap().is_none());
            let report = model.residency_report().unwrap().unwrap();
            let layerwise = report
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Host)
                .collect::<Vec<_>>();
            assert_eq!(layerwise.len(), local_layer_range.len());
            assert!(layerwise
                .iter()
                .all(|unit| unit.host_resident() && !unit.device_resident()));
        }
        let expected_speculative_capability = model.speculative_capability_for_test();
        if std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some() {
            assert_eq!(
                expected_speculative_capability,
                SpeculativeCapability::Ready {
                    draft_source: eredu_core::SpeculativeDraftSource::Embedded,
                },
                "the complete prediction artifact must retain its embedded-draft capability on the neutral target"
            );
        }
        let mut runtime = eredu_core::ModelRuntime::from_prepared(backend, model).unwrap();
        let distributed_view =
            <MlxBackend<'_> as eredu_core::DistributedBackend>::distributed_session(
                runtime.session(),
            )
            .expect("partitioned MLX session must retain its selected communication view");
        let distributed_descriptor = eredu_core::DistributedSession::descriptor(distributed_view);
        assert_eq!(
            distributed_descriptor.world_size(),
            topology.topology().world_size()
        );
        assert_eq!(distributed_descriptor.rank(), topology.global_rank());
        if prove_prepared_communication_lifecycle {
            assert_eq!(
                crate::tests::support::path_instrumentation::communication_realization_attempts(),
                1,
                "session creation must consume the prepared communicator without recreating it"
            );
        }
        if std::env::var_os(OPAQUE_TEXT_GENERATION).is_some() {
            let sampling = eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
            )
            .unwrap();
            let generated = eredu_core::TextGeneration::new(
                &mut runtime,
                vec![1, 2],
                eredu_core::TextGenerationConfig::new(sampling),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>();
            assert_eq!(generated.len(), 3);
            return;
        }
        assert_eq!(
            runtime.session().effective_model_type(),
            expected_effective_model_type
        );
        assert_eq!(
            <MlxBackend<'_> as eredu_core::SpeculativeGenerationBackend>::speculative_capability(
                &runtime
            ),
            expected_speculative_capability
        );
        use crate::backend::runtime::media::input::{InputPayload, ModelInput};
        let capability_tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let capability_parts = [text_input_part(&capability_tokens)];
        let capability_input = ModelInput::new(&capability_parts).into();
        let capabilities =
            <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::model_capabilities(&runtime)
                .unwrap();
        assert_eq!(
            capabilities.effective_model_type,
            expected_effective_model_type
        );
        let counted = <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::count_prepared_input(
            &runtime,
            &capability_input,
        )
        .unwrap();
        assert_eq!(counted.text_tokens, 2);
        assert_eq!(counted.model_positions, 2);
        let state = <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::estimate_runtime_state(
            &runtime, counted, 2, 1,
        )
        .unwrap();
        assert_eq!(state.assumptions.requested_positions, 4);
        assert!(matches!(
            eredu_core::apply_admission_policy(
                &capabilities,
                eredu_core::AdmissionRequest {
                    input: counted,
                    max_output_tokens: 2,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: None,
                    require_complete_estimate: false,
                },
                state,
                None,
            )
            .unwrap(),
            eredu_core::AdmissionResult::Admitted(_)
        ));
        <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::static_memory(&runtime).unwrap();
        if std::env::var_os(PREPARED_SPECULATIVE_CAPABILITY).is_some() {
            return;
        }
        let image_mode = std::env::var_os(OPAQUE_MUSE_IMAGE).is_some();
        let inkling_media_mode = std::env::var_os(OPAQUE_INKLING_MEDIA).is_some();
        let inkling_mtp_mode = std::env::var_os(OPAQUE_INKLING_MTP).is_some();
        let qwen_hybrid_mtp_mode = std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some();
        let qwen_hybrid_prompt_cache_mode = std::env::var_os(QWEN_HYBRID_PROMPT_CACHE).is_some();
        let nemotron_h_mtp_mode = std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_some();
        let deepseek_mtp_target_mode = std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some();
        let deepseek_dspark_target_mode = std::env::var_os(OPAQUE_DEEPSEEK_DSPARK_TARGET).is_some();
        let gemma4_media_mode = std::env::var_os(OPAQUE_GEMMA4_MEDIA).is_some();
        let qwen3_vl_media_mode = std::env::var_os(OPAQUE_QWEN3_VL_MEDIA).is_some();
        let qwen_conditional_media_mode = std::env::var_os(OPAQUE_QWEN_CONDITIONAL_MEDIA).is_some();
        let prompt = Array::from_slice(&[1u32, 2], &[1, 2]);
        let text_before = Array::from_slice(&[1u32], &[1, 1]);
        let text_after = Array::from_slice(&[2u32], &[1, 1]);
        let image_grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let image_pixels = Array::from_slice(&[0.01f32; 48], &[4, 12]);
        let inkling_image = Array::from_slice(&[0.01f32; 16], &[1, 1, 16]);
        let inkling_audio = Array::from_slice(&[0u32, 1, 2, 3, 4, 5], &[1, 3, 2]);
        let gemma4_patches = Array::from_slice(&[0.01f32; 192], &[1, 4, 48]);
        let gemma4_positions = Array::from_slice(&[0i32, 0, 0, 1, 1, 0, 1, 1], &[1, 4, 2]);
        let gemma4_grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let gemma4_audio = Array::from_slice(&[0.01f32; 512], &[1, 4, 128]);
        let gemma4_audio_mask = Array::from_slice(&[true, true, true, true], &[1, 4]);
        let qwen3_vl_grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        let qwen3_vl_pixels = Array::from_slice(&[0.01f32; 96], &[8, 12]);
        let parts = if image_mode {
            vec![
                text_input_part(&text_before),
                input_part(
                    InputModality::Image,
                    InputPayload::Tensor(image_pixels.clone()),
                    [(InputMetadataKey::PatchGrid, image_grid.clone())],
                    [],
                ),
                text_input_part(&text_after),
            ]
        } else if inkling_media_mode {
            vec![
                text_input_part(&prompt),
                input_part(
                    InputModality::Image,
                    InputPayload::Embeddings(inkling_image.clone()),
                    [],
                    [],
                ),
                input_part(
                    InputModality::Audio,
                    InputPayload::Tensor(inkling_audio.clone()),
                    [],
                    [],
                ),
            ]
        } else if gemma4_media_mode {
            vec![
                text_input_part(&prompt),
                input_part(
                    InputModality::Image,
                    InputPayload::Tensor(gemma4_patches.clone()),
                    [
                        (InputMetadataKey::PatchGrid, gemma4_grid.clone()),
                        (InputMetadataKey::PatchPositions, gemma4_positions.clone()),
                    ],
                    [InputExtent::PatchGrid {
                        time: 1,
                        height: 2,
                        width: 2,
                    }],
                ),
                input_part(
                    InputModality::Audio,
                    InputPayload::Tensor(gemma4_audio.clone()),
                    [(InputMetadataKey::AudioMask, gemma4_audio_mask.clone())],
                    [InputExtent::AudioValidFrames(4)],
                ),
            ]
        } else if qwen3_vl_media_mode || qwen_conditional_media_mode {
            vec![
                text_input_part(&prompt),
                input_part(
                    InputModality::Image,
                    InputPayload::Tensor(qwen3_vl_pixels.clone()),
                    [(InputMetadataKey::PatchGrid, qwen3_vl_grid.clone())],
                    [],
                ),
            ]
        } else {
            vec![text_input_part(&prompt)]
        };
        let prefix_tokens = if image_mode {
            vec![1, 22, 2]
        } else if inkling_media_mode {
            vec![1, 2, 21, 20, 20, 20]
        } else if gemma4_media_mode {
            vec![1, 2, 30, 31]
        } else if qwen3_vl_media_mode || qwen_conditional_media_mode {
            vec![1, 2, 42, 42]
        } else {
            vec![1, 2]
        };
        let reference_input =
            PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap();
        let reference = (((tensor_parallel_size == 2
            && (pipeline_parallel_size == 1
                || (pipeline_parallel_size == 2
                    && matches!(
                        family,
                        FixtureFamily::Llama
                            | FixtureFamily::Mistral
                            | FixtureFamily::Qwen2
                            | FixtureFamily::Qwen2Gguf
                            | FixtureFamily::Qwen3
                            | FixtureFamily::Qwen3Gguf
                            | FixtureFamily::KimiLinear
                            | FixtureFamily::Qwen3Moe
                            | FixtureFamily::GptOss
                            | FixtureFamily::DeepSeek
                    ))))
            || (tensor_parallel_size == 1
                && pipeline_parallel_size == 2
                && matches!(
                    family,
                    FixtureFamily::Llama
                        | FixtureFamily::Mistral
                        | FixtureFamily::Qwen2
                        | FixtureFamily::Qwen2Gguf
                        | FixtureFamily::Qwen3
                        | FixtureFamily::Qwen3Gguf
                        | FixtureFamily::KimiLinear
                        | FixtureFamily::Qwen3Moe
                        | FixtureFamily::GptOss
                        | FixtureFamily::DeepSeek
                )))
            && (expert_parallel_size == 1
                || matches!(
                    family,
                    FixtureFamily::DeepSeek
                        | FixtureFamily::DeepSeekV4
                        | FixtureFamily::Qwen3Moe
                        | FixtureFamily::GptOss
                ))
            && family.needs_opaque_reference()
            && std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_none()
            && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_none()
            && std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_none())
        .then(|| resident_reference_for_prepared(&checkpoint, &reference_input));
        let reference_tolerance = if image_mode || gemma4_media_mode || qwen_conditional_media_mode
        {
            5e-4
        } else {
            family.comparison_tolerance()
        };
        let neutral_forwards_before =
            crate::tests::support::path_instrumentation::snapshot().forwards;
        if std::env::var_os(OPAQUE_INSPECTION).is_some() {
            let identity = runtime.session().prompt_cache_model_identity().unwrap();
            let layer_root = if family == FixtureFamily::Gemma {
                "model.language_model.layers"
            } else {
                "model.layers"
            };
            let expected = format!("{layer_root}.{}.output", identity.global_layer_start());
            let inspected = runtime
                .inspect_prefill(ModelInput::new(&parts).into(), &ObservationRequest::all())
                .unwrap();
            assert!(
                inspected.observations.get(&expected).is_some(),
                "rank {expected_rank} missing canonical {expected:?} in {:?}",
                inspected.observations
            );
            assert!(
                inspected
                    .observations
                    .iter()
                    .all(|(path, _)| !path.starts_with("text_decoder.")),
                "rank {expected_rank} returned a synthetic group/index path: {:?}",
                inspected.observations
            );
            assert_eq!(
                inspected
                    .observations
                    .get(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                    .is_some(),
                owns_public_output
            );
            return;
        }
        if inkling_mtp_mode
            || qwen_hybrid_mtp_mode
            || nemotron_h_mtp_mode
            || deepseek_mtp_target_mode
        {
            if inkling_mtp_mode || qwen_hybrid_mtp_mode || nemotron_h_mtp_mode {
                let identity = runtime.session().prompt_cache_model_identity().unwrap();
                assert!(
                    !identity.layer_prefix_offsets().contains(&-1),
                    "the neutral target cache must not absorb adapter-owned prediction state"
                );
            }
            let max_tokens = 3;
            let proposal_capacity = if deepseek_dspark_target_mode { 2 } else { 1 };
            let output = run_neutral_embedded_mtp(
                &mut runtime,
                synthetic_prediction_input(&parts, &prefix_tokens),
                SpeculativeConfig {
                    max_tokens,
                    max_draft_tokens: proposal_capacity,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
            )
            .unwrap();
            assert_eq!(output.token_ids().len(), max_tokens);
            assert_eq!(output.stats().emitted_tokens(), max_tokens);
            assert!(output.stats().draft_tokens() > 0);
            if deepseek_dspark_target_mode {
                assert!(
                    output.stats().draft_tokens() >= 2,
                    "the fused DSpark block must propose more than one token"
                );
                assert!(output.stats().rounds() > 0);
                assert_eq!(
                    output.stats().accept_lens().len(),
                    output.stats().rounds(),
                    "every DSpark verification round must reach transactional commit"
                );
                assert!(output.stats().target_tokens() > max_tokens);
                assert!(output.stats().scheduler_turns() >= output.stats().rounds());
            }
            if deepseek_mtp_target_mode || qwen_hybrid_mtp_mode || nemotron_h_mtp_mode {
                let replay = run_neutral_embedded_mtp(
                    &mut runtime,
                    synthetic_prediction_input(&parts, &prefix_tokens),
                    SpeculativeConfig {
                        max_tokens,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                )
                .unwrap();
                assert_eq!(
                    replay.token_ids(),
                    output.token_ids(),
                    "a fresh prediction lane must not inherit target or extension cache state"
                );
                assert_eq!(replay.stats().emitted_tokens(), max_tokens);
                assert!(replay.stats().draft_tokens() > 0);
            }
            if deepseek_mtp_target_mode {
                let vocabulary_size = if family == FixtureFamily::DeepSeekV4 {
                    16
                } else {
                    8
                };
                let invalid_prompt = Array::from_slice(&[vocabulary_size], &[1, 1]);
                let invalid_parts = [text_input_part(&invalid_prompt)];
                let error = match run_neutral_embedded_mtp(
                    &mut runtime,
                    synthetic_prediction_input(
                        &invalid_parts,
                        &[u32::try_from(vocabulary_size).unwrap()],
                    ),
                    SpeculativeConfig {
                        max_tokens,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                ) {
                    Ok(_) => panic!("out-of-domain target token unexpectedly entered prediction"),
                    Err(error) => error,
                };
                assert!(
                    error
                        .to_string()
                        .contains(&format!("token ID is outside 0..{vocabulary_size}")),
                    "invalid prediction target failed for an unexpected reason: {error}"
                );
                let retry = match run_neutral_embedded_mtp(
                    &mut runtime,
                    synthetic_prediction_input(&parts, &prefix_tokens),
                    SpeculativeConfig {
                        max_tokens,
                        max_draft_tokens: 1,
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                ) {
                    Ok(_) => panic!("fenced prediction session unexpectedly accepted a retry"),
                    Err(error) => error,
                };
                assert!(
                    retry.to_string().contains("session is fenced"),
                    "prediction retry was not rejected by the causal fence: {retry}"
                );
            }
            return;
        }
        if qwen_hybrid_prompt_cache_mode {
            let prompt_input = synthetic_prediction_input(&parts, &prefix_tokens);
            let descriptor;
            let rank_prompt_cache = prompt_cache_root.join(format!("rank-{expected_rank}"));
            {
                let (backend, session) = runtime.parts_mut();
                session
                    .prefill(backend, prompt_input.clone())
                    .unwrap()
                    .wait()
                    .unwrap();
                descriptor = PromptCacheDescriptor::from_model_identity(
                    session.prompt_cache_model_identity().unwrap(),
                    "opaque-ring-qwen-hybrid-prediction",
                    prompt_input
                        .cache_identity()
                        .expect("Qwen Hybrid prefix has an exact identity")
                        .prefix_content_fingerprint(),
                    1,
                )
                .unwrap();
                session
                    .save_prompt_cache(
                        backend,
                        &rank_prompt_cache,
                        descriptor.clone(),
                        &prefix_tokens,
                        &PromptCacheOptions::default(),
                    )
                    .unwrap();
                session
                    .decode(backend, Array::from_slice(&[3_u32], &[1, 1]))
                    .unwrap()
                    .wait()
                    .unwrap();
                session
                    .load_prompt_cache_for_input(
                        backend,
                        &rank_prompt_cache,
                        &descriptor,
                        &prefix_tokens,
                        &prompt_input,
                    )
                    .unwrap();
            }
            let suffix = [3_u32];
            let suffix_tensor = Array::from_slice(&suffix, &[1, 1]);
            let suffix_parts = [text_input_part(&suffix_tensor)];
            let output = run_neutral_embedded_mtp(
                &mut runtime,
                synthetic_prediction_input(&suffix_parts, &suffix),
                SpeculativeConfig {
                    max_tokens: 3,
                    max_draft_tokens: 1,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                },
            )
            .unwrap();
            assert_eq!(output.token_ids().len(), 3);
            assert!(output.stats().draft_tokens() > 0);
            assert!(output.stats().rounds() > 0);
            return;
        }
        let (backend, session) = runtime.parts_mut();
        if neutral_gemma_layers.is_some() || neutral_qwen_vl_config.is_some() {
            let unsupported_image = Array::from_slice(&[0.0f32; 4], &[1, 1, 4]);
            let malformed_parts = [input_part(
                InputModality::Image,
                InputPayload::Tensor(unsupported_image),
                [],
                [],
            )];
            let before = crate::tests::support::path_instrumentation::snapshot();
            let error = match session.prefill(backend, ModelInput::new(&malformed_parts).into()) {
                Ok(_) => panic!("unselected Gemma image input unexpectedly entered execution"),
                Err(error) => error,
            };
            let expected = if neutral_gemma_layers.is_some() && !gemma4_media_mode {
                "prepared input modality image is outside the selected composite modalities {Text}"
            } else {
                "unsupported prepared input"
            };
            assert!(
                error.to_string().contains(expected),
                "malformed prepared input failed for an unexpected reason: {error}"
            );
            let after = crate::tests::support::path_instrumentation::snapshot();
            assert_eq!(after.forwards, before.forwards);
            assert_eq!(after.completions, before.completions);
        }
        let prompt_input = (dense_composite_neutral
            || neutral_gemma_layers.is_some()
            || neutral_qwen_vl_config.is_some()
            || inkling_media_mode
            || matches!(
                family,
                FixtureFamily::Inkling | FixtureFamily::Qwen35Multimodal
            ))
        .then(|| {
            MlxModelInput::from(ModelInput::new(&parts)).with_semantic_content_fingerprint(
                eredu_core::cache::prompt_cache_token_fingerprint(&prefix_tokens),
            )
        })
        .transpose()
        .unwrap();
        let submitted_input = prompt_input
            .clone()
            .unwrap_or_else(|| ModelInput::new(&parts).into());
        let mut output = session
            .prefill(backend, submitted_input)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(output.logits().is_some(), owns_public_output);
        if let (Some(actual), Some((expected, _))) = (output.logits(), &reference) {
            assert_final_logits_close(actual.as_array(), expected, reference_tolerance);
        }
        let descriptor = PromptCacheDescriptor::from_model_identity(
            session.prompt_cache_model_identity().unwrap(),
            "opaque-ring-fixture",
            prompt_input.as_ref().map_or_else(
                || format!("tokens:{prefix_tokens:?}"),
                |input| {
                    input
                        .cache_identity()
                        .expect("neutral Gemma input carries its exact prepared identity")
                        .prefix_content_fingerprint()
                        .to_owned()
                },
            ),
            1,
        )
        .unwrap();
        let rank_prompt_cache = prompt_cache_root.join(format!("rank-{expected_rank}"));
        if std::env::var_os(PROMPT_CACHE_PREPARE_FAILURE).is_some() {
            if expected_rank == 0 {
                std::fs::write(
                    &rank_prompt_cache,
                    b"block rank-local cache directory creation",
                )
                .unwrap();
            }
            let error = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    descriptor.clone(),
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            if expected_rank == 0 {
                assert!(
                    error
                        .to_string()
                        .contains("create reversible prompt cache parent"),
                    "injected rank-local preparation failed for an unexpected reason: {error}"
                );
            } else {
                assert!(
                    error.to_string().contains("another rank failed")
                        && error.to_string().contains("PromptCacheSavePreparation"),
                    "peer preparation failure was not reported causally: {error}"
                );
                assert!(
                    rank_prompt_cache.is_dir()
                        && std::fs::read_dir(&rank_prompt_cache)
                            .unwrap()
                            .next()
                            .is_none(),
                    "successful peer preparation left a published or staged shard"
                );
            }
            let retry = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    descriptor,
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            assert!(
                retry.to_string().contains("session is fenced"),
                "cache-control retry was not rejected by the causal fence: {retry}"
            );
            return;
        }
        session
            .save_prompt_cache(
                backend,
                &rank_prompt_cache,
                descriptor.clone(),
                &prefix_tokens,
                &PromptCacheOptions::default(),
            )
            .unwrap();
        let continuity_token = if reference.is_some() {
            Array::from_slice(&[0u32], &[1, 1])
        } else {
            session
                .sample_and_synchronize(output.logits(), 1, &mut DefaultSampler, 0.0, None, false)
                .unwrap()
                .token
        };
        let uninterrupted = session
            .decode(backend, continuity_token.clone())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(uninterrupted.logits().is_some(), owns_public_output);
        if let (Some(actual), Some((_, expected))) = (uninterrupted.logits(), &reference) {
            assert_final_logits_close(actual.as_array(), expected, reference_tolerance);
        }
        let uninterrupted_logits = uninterrupted.logits().map(|logits| {
            logits
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        });
        match prompt_input.as_ref() {
            Some(input) => session
                .load_prompt_cache_for_input(
                    backend,
                    &rank_prompt_cache,
                    &descriptor,
                    &prefix_tokens,
                    input,
                )
                .unwrap(),
            None => session
                .load_prompt_cache(backend, &rank_prompt_cache, &descriptor, &prefix_tokens)
                .unwrap(),
        };
        output = session
            .decode(backend, continuity_token)
            .unwrap()
            .wait()
            .unwrap();
        let restored_logits = output.logits().map(|logits| {
            logits
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        });
        assert_eq!(uninterrupted_logits, restored_logits);
        for _ in 0..2 {
            assert_eq!(output.logits().is_some(), owns_public_output);
            let token = session
                .sample_and_synchronize(output.logits(), 1, &mut DefaultSampler, 0.0, None, false)
                .unwrap()
                .token;
            output = session.decode(backend, token).unwrap().wait().unwrap();
        }
        assert_eq!(output.logits().is_some(), owns_public_output);
        if prove_prepared_communication_lifecycle {
            assert_eq!(
                crate::tests::support::path_instrumentation::snapshot().forwards
                    - neutral_forwards_before,
                5,
                "prefill, uninterrupted/restored decode, and two continued decodes must each traverse the neutral session once"
            );
            if dense_stream || layerwise_host {
                assert_eq!(
                    crate::tests::support::path_instrumentation::bounded_unit_acquisitions(),
                    5 * local_layer_range.len(),
                    "each neutral forward must acquire every selected bounded-residency unit exactly once"
                );
            }
            if routed_neutral && expert_parallel_size > 1 {
                let exchanges_per_layer = match family {
                    FixtureFamily::Qwen3Moe
                    | FixtureFamily::Qwen3MoeGguf
                    | FixtureFamily::DeepSeek
                    | FixtureFamily::DeepSeekGguf
                    | FixtureFamily::DeepSeekV4
                    | FixtureFamily::NemotronH
                    | FixtureFamily::NemotronHGguf
                    | FixtureFamily::Lfm2MoeGguf
                    | FixtureFamily::KimiLinearGguf
                    | FixtureFamily::Qwen3VlMoe => 8,
                    FixtureFamily::Inkling | FixtureFamily::InklingMultimodal => 16,
                    FixtureFamily::GptOss | FixtureFamily::GptOssGguf
                        if tensor_parallel_size > 1 =>
                    {
                        9
                    }
                    FixtureFamily::GptOss | FixtureFamily::GptOssGguf => 8,
                    _ => unreachable!("only routed neutral fixtures select expert exchange"),
                };
                let exchanged_layers = if pipeline_parallel_size > 1 {
                    family.expert_layer_count(0..family.layer_count())
                } else {
                    family.expert_layer_count(local_layer_range.clone())
                };
                assert_eq!(
                    crate::tests::support::path_instrumentation::variable_all_to_all_submissions(),
                    5 * exchanged_layers * exchanges_per_layer,
                    "every routed layer forward must exchange global/local expert IDs, route tags, activations, scores, coefficients, and the exact inverse results without fallback"
                );
            }
        }
        if dense_stream {
            let report = session.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.prefill_forwards(), 1);
            assert_eq!(report.decode_forwards(), 4);
            assert_eq!(report.planned_layer_count(), local_layer_range.len());
        }
        if layerwise_host {
            let report = session.residency_report().unwrap().unwrap();
            let expected_units = local_layer_range.len();
            let layerwise = report
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Host)
                .collect::<Vec<_>>();
            assert_eq!(layerwise.len(), expected_units);
            assert!(layerwise.iter().all(|unit| unit.host_resident()));
            assert!(
                layerwise
                    .iter()
                    .filter(|unit| unit.device_resident())
                    .count()
                    <= 1
            );
        }
        if std::env::var_os(EXPERT_CACHE).is_some() {
            let report = session.parameter_bank_report().unwrap();
            let expected_owned = match family {
                FixtureFamily::Qwen3Moe | FixtureFamily::Qwen3MoeGguf => {
                    local_layer_range.len() * 4 / expert_parallel_size
                }
                FixtureFamily::GptOss | FixtureFamily::GptOssGguf => {
                    local_layer_range.len() * 2 / expert_parallel_size
                }
                FixtureFamily::DeepSeek | FixtureFamily::DeepSeekGguf => {
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size
                }
                FixtureFamily::DeepSeekV4 => {
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size
                }
                FixtureFamily::KimiLinearGguf => {
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size
                }
                FixtureFamily::Lfm2MoeGguf
                | FixtureFamily::NemotronH
                | FixtureFamily::NemotronHGguf => {
                    family.expert_layer_count(local_layer_range.clone()) * 2 / expert_parallel_size
                }
                _ => unreachable!("only exact routed addressable fixtures select expert caching"),
            };
            if expected_owned == 0 {
                assert!(
                    report.is_none(),
                    "rank with no routed units must not manufacture bank ownership"
                );
                return;
            }
            let report = report.expect("owned independent expert bank must expose live telemetry");
            assert!(report.owned_entries() > 0);
            assert!(report.owned_bytes() > 0);
            let requests =
                report.bulk().device().requests() + report.incremental().device().requests();
            let misses = report.bulk().device().misses() + report.incremental().device().misses();
            assert_eq!(
                report.owned_entries(),
                expected_owned,
                "the addressable bank must own only this PP×EP rank's exact expert entries"
            );
            assert_eq!(requests == 0, misses == 0);
            if std::env::var_os(EXPERT_CACHE_EVICTION).is_some() {
                let evictions =
                    report.bulk().device().evictions() + report.incremental().device().evictions();
                if requests > 0 {
                    assert!(evictions > 0, "bounded expert bank never evicted an entry");
                    assert!(misses > report.owned_entries() as u64);
                }
            }
        }
        if prompt_input.is_some() {
            let wrong_descriptor = PromptCacheDescriptor::from_model_identity(
                session.prompt_cache_model_identity().unwrap(),
                "opaque-ring-fixture",
                "wrong-prepared-input-identity",
                1,
            )
            .unwrap();
            let error = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    wrong_descriptor,
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("prepared input"),
                "wrong prepared-input descriptor failed for an unexpected reason: {error}"
            );
            let retry = session
                .save_prompt_cache(
                    backend,
                    &rank_prompt_cache,
                    descriptor,
                    &prefix_tokens,
                    &PromptCacheOptions::default(),
                )
                .unwrap_err();
            assert!(
                retry.to_string().contains("session is fenced"),
                "cache-control retry was not rejected by the causal fence: {retry}"
            );
        }
    }
}
