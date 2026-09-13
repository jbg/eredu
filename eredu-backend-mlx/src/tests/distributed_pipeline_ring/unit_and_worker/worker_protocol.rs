// Verify native binding counts against retained semantic ownership. Model
// fixtures can change optional roots, replicas, formats, and per-layer inputs.
fn selected_composite_fixture_counts(
    execution: &eredu_architectures::SelectedExecution,
) -> (usize, usize, usize) {
    use eredu_architectures::replicated_text::SelectedCompositeTextRealization;
    use eredu_architectures::{
        SelectedCompositePartitionedExecution, SelectedDensePartitionedExecution,
        SelectedExecutionDispatcher, SelectedRoutedPartitionedExecution,
        SelectedRoutedTextRealization,
    };
    use eredu_runtime::{ReplicatedTextParameterOwner, SelectedReplicatedTextRealization};
    struct Counts;
    impl SelectedExecutionDispatcher for Counts {
        type Output = (usize, usize, usize);
        type Error = &'static str;
        fn replicated(
            self,
            _: SelectedReplicatedTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            Err("expected a partitioned composite fixture")
        }
        fn routed(self, _: SelectedRoutedTextRealization) -> Result<Self::Output, Self::Error> {
            Err("expected a partitioned composite fixture")
        }
        fn composite(
            self,
            _: SelectedCompositeTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            Err("expected a partitioned composite fixture")
        }
        fn partitioned_dense(
            self,
            _: SelectedDensePartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            Err("expected a partitioned composite fixture")
        }
        fn partitioned_routed(
            self,
            _: SelectedRoutedPartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            Err("expected a partitioned composite fixture")
        }
        fn partitioned_composite(
            self,
            selected: SelectedCompositePartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            let owned = selected
                .requirements()
                .logical_parameter_targets()
                .iter()
                .collect::<std::collections::BTreeSet<_>>();
            let all_static = selected
                .materialization_tasks()
                .iter()
                .filter(|task| {
                    matches!(
                        task.owner(),
                        ReplicatedTextParameterOwner::StaticRole(_)
                            | ReplicatedTextParameterOwner::StaticUnitConsumers { .. }
                    )
                })
                .flat_map(|task| {
                    std::iter::once(task.name()).chain(
                        task.output_companions()
                            .iter()
                            .map(|companion| companion.name()),
                    )
                })
                .collect::<std::collections::BTreeSet<_>>();
            // Output companions inherit the admitted parent task's owner.
            let local = selected
                .materialization_tasks()
                .iter()
                .filter(|task| {
                    matches!(
                        task.owner(),
                        ReplicatedTextParameterOwner::StaticRole(_)
                            | ReplicatedTextParameterOwner::StaticUnitConsumers { .. }
                    )
                })
                .filter(|task| owned.iter().any(|target| target.as_str() == task.name()))
                .flat_map(|task| {
                    std::iter::once(task.name()).chain(
                        task.output_companions()
                            .iter()
                            .map(|companion| companion.name()),
                    )
                })
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let units = selected
                .requirements()
                .groups()
                .iter()
                .map(|group| group.units().len())
                .sum();
            Ok((units, local, all_static.len() - local))
        }
    }
    execution
        .clone()
        .dispatch(Counts)
        .expect("retained composite fixture ownership")
}

#[test]
fn pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let fixture_root = if checkpoint.is_file() {
        checkpoint.parent().unwrap()
    } else {
        checkpoint.as_path()
    };
    let family = FixtureFamily::parse(&std::env::var(FIXTURE_FAMILY).unwrap());
    let prompt_cache_root = PathBuf::from(std::env::var_os(PROMPT_CACHE_ROOT).unwrap());
    let native_group = distributed::init(true, Backend::Ring).unwrap();
    let cartesian_axes = std::env::var(CARTESIAN_AXES).ok();
    let (tensor_parallel_size, pipeline_parallel_size, expert_parallel_size) =
        match cartesian_axes.as_deref() {
            None | Some("pp") => (1, 2, 1),
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
                &std::fs::read(fixture_root.join("config.json")).expect("Gemma config"),
            )
            .expect("Gemma JSON config");
            config
        });
    let neutral_gemma_layers = neutral_gemma_config.as_ref().map(|config| {
        config["text_config"]["num_hidden_layers"]
            .as_u64()
            .expect("Gemma text layer count") as usize
    });
    let neutral_prediction_target_layers =
        ((matches!(family, FixtureFamily::Inkling | FixtureFamily::InklingDense)
            && (std::env::var_os(OPAQUE_INKLING_MTP).is_some()
                || std::env::var_os(OPAQUE_COMPONENT_CAPTURE).is_some()))
            || (family == FixtureFamily::Qwen35Multimodal
                && std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some())
            || (family == FixtureFamily::NemotronH
                && std::env::var_os(OPAQUE_NEMOTRON_H_MTP).is_some()))
        .then(|| {
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(fixture_root.join("config.json")).expect("prediction target config"),
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
                &std::fs::read(fixture_root.join("config.json")).expect("Qwen3-VL config"),
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
    let device_type = match std::env::var("EREDU_TEST_RING_DEVICE").as_deref() {
        Ok("gpu") => DeviceType::Gpu,
        Ok("cpu") | Err(std::env::VarError::NotPresent) => DeviceType::Cpu,
        other => panic!("invalid EREDU_TEST_RING_DEVICE: {other:?}"),
    };
    let device = DeviceAssignment::new(device_type, 0);
    let stream = Stream::new_with_device(&device.device().unwrap());
    let weights_stream = fixture_weights_stream(&stream);
    if std::env::var_os(OPAQUE_SESSION).is_some() {
        let component_media = std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some()
            || (std::env::var_os(OPAQUE_QWEN_HYBRID_MTP).is_some()
                && matches!(
                    family,
                    FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                ));
        let dense_composite_neutral = matches!(
            family,
            FixtureFamily::MuseGlimmer
                | FixtureFamily::Qwen3Vl
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
        let gemma_components =
            neutral_gemma_config.is_some() && std::env::var_os(OPAQUE_COMPONENT_CAPTURE).is_some();
        let gemma_sparse = gemma_components
            && neutral_gemma_config.as_ref().is_some_and(|config| {
                config["text_config"]["enable_moe_block"].as_bool() == Some(true)
            });
        let dense_composite_neutral =
            dense_composite_neutral || (gemma_components && !gemma_sparse);
        let dense_composite_auxiliary_units = (dense_composite_neutral || component_media)
            .then(|| {
                serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(fixture_root.join("config.json"))
                        .expect("dense composite config"),
                )
                .expect("dense composite JSON config")
            })
            .map_or(0, |config| match family {
                FixtureFamily::MuseGlimmer | FixtureFamily::MuseGlimmerMoe => {
                    let depth = config["vision_config"]["num_hidden_layers"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                    if pipeline_rank == 0 {
                        depth
                    } else {
                        0
                    }
                }
                FixtureFamily::Qwen3Vl
                | FixtureFamily::Qwen3VlMoe
                | FixtureFamily::Qwen35ZeroPrediction
                | FixtureFamily::Qwen35Multimodal
                | FixtureFamily::Qwen35MoeMultimodal => {
                    let depth = config["vision_config"]["depth"].as_u64().unwrap_or(0) as usize;
                    depth * (pipeline_rank + 1) / pipeline_parallel_size
                        - depth * pipeline_rank / pipeline_parallel_size
                }
                _ => 0,
            });
        let routed_neutral = (matches!(
            family,
            FixtureFamily::K2Mova
                | FixtureFamily::K2Fp8(_)
                | FixtureFamily::Qwen3Moe
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
                | FixtureFamily::MuseGlimmerMoe
                | FixtureFamily::MuseGlimmerGguf(true)
                | FixtureFamily::Qwen3NextMoe
                | FixtureFamily::Qwen35Moe
                | FixtureFamily::Qwen35MoeMultimodal
                | FixtureFamily::Inkling
                | FixtureFamily::InklingMultimodal
        ) || gemma_sparse
            || (family == FixtureFamily::DeepSeekV4
                && (std::env::var_os(PREDICTION_FREE_TARGET).is_some()
                    || std::env::var_os(PREPARED_SPECULATIVE_CAPABILITY).is_some()
                    || std::env::var_os(OPAQUE_DEEPSEEK_MTP_TARGET).is_some()
                    || std::env::var_os(OPAQUE_COMPONENT_CAPTURE).is_some())))
            && matches!(
                cartesian_axes.as_deref(),
                None | Some("tp")
                    | Some("pp")
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
                FixtureFamily::DeepSeekDense
                    | FixtureFamily::DeepSeekDenseGguf
                    | FixtureFamily::Llama
                    | FixtureFamily::Mistral
                    | FixtureFamily::Nanbeige
                    | FixtureFamily::Qwen2
                    | FixtureFamily::Qwen2Gguf
                    | FixtureFamily::Qwen3
                    | FixtureFamily::Qwen3Gguf
                    | FixtureFamily::KimiLinear
                    | FixtureFamily::NemotronH
                    | FixtureFamily::Lfm2
                    | FixtureFamily::Gemma
                    | FixtureFamily::MuseGlimmerGguf(false)
            ) && matches!(
                cartesian_axes.as_deref(),
                None | Some("tp") | Some("pp") | Some("tp-pp")
            ));
        let prove_expert_communication = cartesian_axes.as_deref() == Some("ep") && !routed_neutral;
        if prove_prepared_communication_lifecycle || prove_expert_communication {
            crate::tests::support::path_instrumentation::reset();
        }
        let backend = crate::native::distributed_backend(&stream, &weights_stream, &native_group);
        // Keep a small device tier while reserving finite host capacity for the
        // independently writable branches exercised by the K2 control matrix
        // and the block-aligned Qwen FP8 prediction fixture.
        let host_cache_bytes = if matches!(
            family,
            FixtureFamily::K2Dense | FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)
        ) || std::env::var_os("EREDU_RING_PREDICTION_FP8").is_some()
            || std::env::var_os("EREDU_RING_DEEPSEEK_FP8").is_some()
        {
            1 << 20
        } else {
            32768
        };
        let device_cache_bytes = if matches!(
            family,
            FixtureFamily::K2Dense | FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)
        ) || std::env::var_os("EREDU_RING_DEEPSEEK_FP8").is_some()
        {
            131072
        } else {
            32768
        };
        let selected_paged = PagedCacheOptions::new(1, device_cache_bytes, host_cache_bytes, 1)
            .unwrap()
            .with_full_attention(true);
        let dense_stream = std::env::var_os(DENSE_STREAM).is_some();
        let layerwise_host = std::env::var_os(LAYERWISE_HOST).is_some();
        assert!(!(dense_stream && layerwise_host));
        let load_options = if std::env::var_os(REQUANTIZE).is_some() {
            let request = if family == FixtureFamily::NemotronH
                || std::env::var(REQUANTIZE).as_deref() == Ok("mxfp4")
            {
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
        let reference_load_options = load_options.clone();
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
            let bank = if family.is_k2_fp8() {
                let config: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(fixture_root.join("config.json")).unwrap(),
                )
                .unwrap();
                let hidden = config["hidden_size"].as_u64().unwrap() as usize;
                let args =
                    eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
                let description =
                    eredu_architectures::k2_horizon::parameter_description(&args).unwrap();
                let layout =
                    eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                        &description,
                        topology,
                    )
                    .unwrap();
                let ffn = layout
                    .tensor("model.layers.1.mlp.experts.gate_up_proj")
                    .unwrap()
                    .local_shape();
                let local_experts = ffn[0];
                let local_units = ffn[1] / 2;
                // A pipeline stage can own value banks with different local
                // head counts and encodings. Admit the largest single-token
                // batch among its actual layers, keeping FFN cache pressure.
                let value_batch_bytes = local_layer_range
                    .clone()
                    .filter(|layer| args.is_mova_layer(*layer))
                    .map(|layer| {
                        let value_name = format!("model.layers.{layer}.self_attn.v_experts.weight");
                        let value = layout.tensor(&value_name).unwrap().local_shape();
                        let value_member_bytes = match args.linear_format_for(&value_name) {
                            eredu_checkpoint::LinearFormat::Dense => value[1] * hidden * 4,
                            eredu_checkpoint::LinearFormat::E4M3BlockFp8(format) => {
                                let scale_bytes = match format.scale_encoding {
                                    eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint => 4,
                                    eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 => 1,
                                };
                                value[1] * hidden
                                    + scale_bytes
                                        * value[1].div_ceil(format.block_rows as usize)
                                        * hidden.div_ceil(format.block_columns as usize)
                            }
                            other => panic!("unexpected value-bank fixture encoding {other:?}"),
                        };
                        value[0].min(config["mova_num_experts_per_tok"].as_u64().unwrap() as usize)
                            * value_member_bytes
                    })
                    .max()
                    .unwrap_or(0);
                let budget = (local_experts
                    * (3 * hidden * local_units
                        + 12 * hidden.div_ceil(128) * local_units.div_ceil(128)))
                .max(value_batch_bytes) as u64;
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(budget), Some(0), 1).unwrap(),
                    2 << 20,
                    budget,
                )
                .unwrap()
            } else if matches!(family, FixtureFamily::K2Mova) {
                let budget = if checkpoint.is_file() {
                    16384
                } else if serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(fixture_root.join("config.json")).unwrap(),
                )
                .unwrap()["quantization_config"]
                    .is_object()
                {
                    393312
                } else {
                    1152
                };
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
                    budget,
                    budget,
                )
                .unwrap()
            } else if std::env::var_os(EXPERT_CACHE_EVICTION).is_some() {
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
            let inspection = component_fixture_inspection(&checkpoint);
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
        let expert_manifest_expected = if prove_expert_communication {
            let inspection = component_fixture_inspection(&checkpoint);
            let selected = crate::composition::mlx::loading::select_preparation(
                &inspection,
                load_options.clone(),
            )
            .expect("expert fixture cold selection");
            selected.neutral().communication_manifest().is_some()
        } else {
            false
        };
        let composite_binding_counts =
            (neutral_gemma_layers.is_some() || component_media).then(|| {
                let inspection = component_fixture_inspection(&checkpoint);
                let selected = crate::composition::mlx::loading::select_preparation(
                    &inspection,
                    load_options.clone(),
                )
                .expect("composite fixture cold selection");
                selected_composite_fixture_counts(selected.neutral().execution())
            });
        let inspection = component_fixture_inspection(&checkpoint);
        let admitted_payload_stores = inspection
            .validated_gguf()
            .map_or(1, |validated| 1 + validated.companions().count());
        let model = match eredu_core::prepare_inspected_model(&backend, inspection, load_options) {
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
                admitted_payload_stores,
                "each admitted primary or companion payload store must open exactly once"
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
                    composite_binding_counts.map_or_else(
                        || local_layer_range.len()
                            + neutral_qwen_vl_config.as_ref().map_or(0, |config| {
                                let depth =
                                    config["vision_config"]["depth"].as_u64().unwrap_or(0) as usize;
                                depth * (pipeline_rank + 1) / pipeline_parallel_size
                                    - depth * pipeline_rank / pipeline_parallel_size
                            })
                            + dense_composite_auxiliary_units,
                        |counts| counts.0
                    ),
                    "neutral partition construction must bind every local unit exactly once"
                );
                assert_eq!(
                    crate::tests::support::path_instrumentation::snapshot().materializations,
                    usize::from(std::env::var_os(REQUANTIZE).is_some()),
                    "neutral construction must execute exactly the selected transform groups"
                );
            }
            if let Some((_, selected, excluded)) = composite_binding_counts {
                let counts = crate::tests::support::path_instrumentation::snapshot();
                assert_eq!(
                    counts.local_static_bindings, selected,
                    "composite static bindings must match exact selected ownership"
                );
                assert_eq!(
                    counts.excluded_local_static_parameters, excluded,
                    "composite unowned static definitions must remain unbound and unread"
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
        if prove_expert_communication {
            assert_eq!(
                crate::tests::support::path_instrumentation::communication_realization_attempts(),
                1
            );
            assert_eq!(
                crate::tests::support::path_instrumentation::manifest_communication_realization_attempts(),
                usize::from(expert_manifest_expected),
                "expert communication realization must match the retained cold selection"
            );
        }
        let expected_effective_model_type = family.effective_model_type();
        // Source formats can retain different official type aliases for the
        // same architecture (for example qwen3_vl and qwen3_vl_text).
        assert_eq!(
            eredu_architectures::configuration::ModelKind::resolve_model_type(
                model.effective_model_type()
            )
            .unwrap(),
            eredu_architectures::configuration::ModelKind::resolve_model_type(
                expected_effective_model_type
            )
            .unwrap(),
        );
        let local_residency_units = composite_binding_counts
            .map(|(units, _, _)| units)
            .unwrap_or(
                local_layer_range.len()
                    + if component_media {
                        dense_composite_auxiliary_units
                    } else {
                        0
                    },
            );
        let expected_speculative_capability = model.speculative_capability_for_test();
        let prediction_units = model
            .residency_report()
            .unwrap()
            .unwrap()
            .unit_sources()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        if matches!(
            expected_speculative_capability,
            SpeculativeCapability::Ready {
                draft_source: eredu_core::SpeculativeDraftSource::Embedded,
            }
        ) {
            assert!(
                !prediction_units.is_empty(),
                "embedded modules share the target weight ledger"
            );
        }
        if dense_stream {
            let report = model.dense_stream_report().unwrap().unwrap();
            assert_eq!(report.planned_layer_count(), local_residency_units);
            let streamed = report
                .residency()
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == eredu_core::residency::MemoryTier::Disk)
                .collect::<Vec<_>>();
            assert_eq!(
                streamed.len(),
                local_residency_units + prediction_units.len()
            );
            assert!(prediction_units
                .iter()
                .all(|id| streamed.iter().any(|unit| unit.id() == id)));
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
            assert_eq!(
                layerwise.len(),
                local_residency_units + prediction_units.len()
            );
            assert!(prediction_units
                .iter()
                .all(|id| layerwise.iter().any(|unit| unit.id() == id)));
            assert!(layerwise
                .iter()
                .all(|unit| unit.host_resident() && !unit.device_resident()));
        }
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
        if std::env::var_os(OPAQUE_PROVIDER_FAILURE).is_some() {
            verify_partition_provider_failure(&mut runtime, family, topology);
            return;
        }
        if std::env::var_os(OPAQUE_COMPONENT_CAPTURE).is_some() {
            let backend = MlxBackend::new(&stream, &weights_stream);
            let reference_path = if family == FixtureFamily::Qwen3MoeGguf {
                checkpoint.parent().unwrap().join("independent-reference")
            } else {
                checkpoint.clone()
            };
            let prepared = eredu_core::prepare_inspected_model(
                &backend,
                component_fixture_inspection(&reference_path),
                MlxLoadRequest::from_normalized(reference_load_options.clone()),
            )
            .unwrap();
            let mut reference = ModelRuntime::from_prepared(backend, prepared).unwrap();
            verify_public_partition_parameter_access(
                &mut runtime,
                &mut reference,
                expected_rank,
                family,
            );
            if family.is_k2_fp8() {
                verify_k2_fp8_source_parameters(&mut runtime);
            }
            verify_public_partition_parameter_overlays(
                &mut runtime,
                &mut reference,
                expected_rank,
                family,
                &checkpoint,
                &stream,
            );
            if matches!(
                family,
                FixtureFamily::DeepSeek | FixtureFamily::DeepSeekGguf
            ) && std::env::var_os(EXPERT_CACHE).is_some()
            {
                let expected_owned =
                    family.expert_layer_count(local_layer_range.clone()) * 4 / expert_parallel_size;
                let report = runtime.session().parameter_bank_report().unwrap();
                let mut active_stages = vec![0_i32; pipeline_parallel_size];
                if expected_owned == 0 {
                    assert!(
                        report.is_none(),
                        "dense PP owner must not create expert banks"
                    );
                } else {
                    let report =
                        report.expect("selected independent experts must retain live banks");
                    assert_eq!(report.owned_entries(), expected_owned);
                    assert!(report.owned_bytes() > 0);
                    let requests = report.bulk().device().requests()
                        + report.incremental().device().requests();
                    let selections = report.bulk().requested_selections()
                        + report.incremental().requested_selections();
                    let compact_bytes = report
                        .banks()
                        .values()
                        .map(|bank| {
                            bank.bulk()
                                .peak_compact_bank_bytes()
                                .max(bank.incremental().peak_compact_bank_bytes())
                        })
                        .max()
                        .unwrap_or(0);
                    // Queries and overlay rollback can populate entries before
                    // an inference acquisition, so those requests may all hit.
                    // An EP owner can receive no routes, even while its parameters
                    // are queried/edited. Require activity per routed PP stage
                    // below, and verify inactive owners did no compact-bank work.
                    if requests == 0 {
                        assert!(
                            expert_parallel_size > 1,
                            "a non-EP routed owner must acquire experts"
                        );
                        assert_eq!(
                            selections, 0,
                            "a selected local route must acquire its bank"
                        );
                        assert_eq!(
                            compact_bytes, 0,
                            "idle EP owner must not construct a compact bank"
                        );
                    } else {
                        assert!(selections > 0);
                        assert!(report.peak_device_resident_bytes() > 0);
                        assert!(
                            compact_bytes > 0,
                            "active expert providers must construct actual compact banks"
                        );
                        active_stages[pipeline_rank] = 1;
                    }
                }
                let activity = distributed::all_sum(
                    &Array::from_slice(&active_stages, &[pipeline_parallel_size as i32]),
                    &native_group,
                    &stream,
                )
                .unwrap();
                let activity = activity.evaluated().unwrap();
                for (stage, active) in activity.as_slice::<i32>().iter().enumerate() {
                    let layers = family.layer_count();
                    let owns_experts = family.expert_layer_count(
                        layers * stage / pipeline_parallel_size
                            ..layers * (stage + 1) / pipeline_parallel_size,
                    ) > 0;
                    assert_eq!(
                        *active > 0,
                        owns_experts,
                        "every routed PP stage must exercise actual expert caching"
                    );
                }
            }
            if family == FixtureFamily::Qwen3 {
                runtime
                    .session_mut()
                    .verify_partition_parameter_owner_for_test(&stream, &mut reference);
            }
            let loop_normalization = (family == FixtureFamily::Nanbeige).then_some((
                "model.layers.0.feed_forward.residual",
                "model.layers.0.output",
            ));
            let required_unit_points: &[&str] = if gemma_components {
                &[
                    "model.language_model.layers.0.attention.input",
                    "model.language_model.layers.0.attention.channels",
                    "model.language_model.layers.0.attention.channels.effective",
                    "model.language_model.layers.3.attention.channels",
                    "model.language_model.layers.3.attention.channels.effective",
                    "model.language_model.layers.1.dense_feed_forward.input",
                    "model.language_model.layers.1.dense_feed_forward.units",
                    "model.language_model.layers.1.dense_feed_forward.units.effective",
                    "model.language_model.layers.0.per_layer.input",
                    "model.language_model.layers.0.per_layer.output.effective",
                    "model.language_model.layers.0.residual.before_scale",
                    "model.language_model.layers.0.residual.scaled.effective",
                ]
            } else if matches!(
                family,
                FixtureFamily::Qwen3NextMoe
                    | FixtureFamily::Qwen35Moe
                    | FixtureFamily::Qwen35MoeMultimodal
            ) {
                &[
                    "model.layers.0.mixer.input",
                    "model.layers.0.mixer.channels",
                    "model.layers.0.mixer.channels.effective",
                    "model.layers.0.mixer.qkv.projected",
                    "model.layers.0.mixer.qkv.convolved",
                    "model.layers.0.mixer.gate.projected",
                    "model.layers.0.mixer.update.projected",
                    "model.layers.0.mixer.decay.projected",
                    "model.layers.0.mixer.output.effective",
                    "model.layers.1.attention.input",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.0.feed_forward.input",
                    "model.layers.1.feed_forward.input",
                    "model.layers.0.mlp.shared_expert.feed_forward.units",
                    "model.layers.0.mlp.shared_expert.feed_forward.units.effective",
                    "model.layers.1.mlp.shared_expert.feed_forward.units",
                    "model.layers.1.mlp.shared_expert.feed_forward.units.effective",
                    "model.layers.1.mlp.shared_expert.feed_forward.write",
                    "model.layers.1.mlp.shared_expert.feed_forward.output.effective",
                    "model.layers.1.mlp.shared_expert.gate.input",
                    "model.layers.1.mlp.shared_expert.gate.projection_input",
                    "model.layers.1.mlp.shared_expert.gate",
                    "model.layers.1.mlp.shared_expert.gate.effective",
                ]
            } else if matches!(
                family,
                FixtureFamily::Qwen3Next | FixtureFamily::Qwen35 | FixtureFamily::Qwen35Multimodal
            ) {
                &[
                    "model.layers.0.mixer.input",
                    "model.layers.0.mixer.channels",
                    "model.layers.0.mixer.channels.effective",
                    "model.layers.0.mixer.qkv.projected",
                    "model.layers.0.mixer.qkv.convolved",
                    "model.layers.0.mixer.gate.projected",
                    "model.layers.0.mixer.update.projected",
                    "model.layers.0.mixer.decay.projected",
                    "model.layers.0.mixer.output.effective",
                    "model.layers.1.attention.input",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.0.feed_forward.input",
                    "model.layers.0.feed_forward.units",
                    "model.layers.1.feed_forward.units",
                    "model.layers.1.feed_forward.units.effective",
                ]
            } else if family == FixtureFamily::NemotronH {
                &[
                    "model.layers.0.mixer.input",
                    "model.layers.0.mixer.output",
                    "model.layers.0.mixer.output.effective",
                    "model.layers.1.feed_forward.input",
                    "model.layers.1.feed_forward.units",
                    "model.layers.1.feed_forward.units.effective",
                    "model.layers.2.feed_forward.input",
                    "model.layers.2.feed_forward.output",
                    "model.layers.2.feed_forward.output.effective",
                    "model.layers.2.shared.feed_forward.input",
                    "model.layers.2.shared.feed_forward.input.effective",
                    "model.layers.2.shared.feed_forward.units",
                    "model.layers.2.shared.feed_forward.units.effective",
                    "model.layers.2.shared.feed_forward.write",
                    "model.layers.2.shared.feed_forward.write.effective",
                    "model.layers.2.shared.feed_forward.output",
                    "model.layers.2.shared.feed_forward.output.effective",
                    "model.layers.3.attention.input",
                    "model.layers.3.attention.channels",
                    "model.layers.3.attention.channels.effective",
                ]
            } else if family == FixtureFamily::NemotronHGguf {
                &[
                    "model.layers.0.mixer.input",
                    "model.layers.0.mixer.output",
                    "model.layers.0.mixer.output.effective",
                    "model.layers.1.feed_forward.input",
                    "model.layers.1.feed_forward.output",
                    "model.layers.1.feed_forward.output.effective",
                    "model.layers.1.shared.feed_forward.input",
                    "model.layers.1.shared.feed_forward.input.effective",
                    "model.layers.1.shared.feed_forward.units",
                    "model.layers.1.shared.feed_forward.units.effective",
                    "model.layers.1.shared.feed_forward.write",
                    "model.layers.1.shared.feed_forward.write.effective",
                    "model.layers.1.shared.feed_forward.output",
                    "model.layers.1.shared.feed_forward.output.effective",
                    "model.layers.2.feed_forward.input",
                    "model.layers.2.feed_forward.output",
                    "model.layers.2.feed_forward.output.effective",
                    "model.layers.2.shared.feed_forward.input",
                    "model.layers.2.shared.feed_forward.input.effective",
                    "model.layers.2.shared.feed_forward.units",
                    "model.layers.2.shared.feed_forward.units.effective",
                    "model.layers.2.shared.feed_forward.write",
                    "model.layers.2.shared.feed_forward.write.effective",
                    "model.layers.2.shared.feed_forward.output",
                    "model.layers.2.shared.feed_forward.output.effective",
                    "model.layers.3.attention.input",
                    "model.layers.3.attention.channels",
                    "model.layers.3.attention.channels.effective",
                ]
            } else if matches!(family, FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)) {
                &[
                    "model.layers.0.feed_forward.units",
                    "model.layers.1.attention.input",
                    "model.layers.1.attention.channels",
                    "model.layers.1.feed_forward.input",
                    "model.layers.1.feed_forward.contribution",
                    "model.layers.1.feed_forward.contribution.effective",
                    "model.layers.1.shared.feed_forward.input",
                    "model.layers.1.shared.feed_forward.units",
                    "model.layers.1.shared.feed_forward.units.effective",
                    "model.layers.1.shared.feed_forward.write",
                    "model.layers.1.shared.feed_forward.write.effective",
                    "model.layers.1.shared.feed_forward.output",
                    "model.layers.1.shared.feed_forward.output.effective",
                ]
            } else if family == FixtureFamily::DeepSeekV4 {
                &[
                    "layers.0.compressed_attention.input",
                    "layers.0.compressed_attention.channels",
                    "layers.0.compressed_attention.channels.effective",
                    "layers.0.feed_forward.input",
                    "layers.0.feed_forward.shared.units",
                    "layers.0.feed_forward.shared.units.effective",
                    "layers.1.compressed_attention.channels",
                    "layers.1.feed_forward.shared.units",
                    "layers.1.feed_forward.shared.write.effective",
                    "layers.1.output.effective",
                ]
            } else if family.is_dense_v3() {
                &[
                    "model.layers.0.attention.input",
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.query.latent",
                    "model.layers.0.attention.query.latent.effective",
                    "model.layers.0.attention.key_value.latent",
                    "model.layers.0.attention.key_value.latent.effective",
                    "model.layers.0.feed_forward.units",
                    "model.layers.0.feed_forward.write.effective",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.query.latent.effective",
                    "model.layers.1.attention.key_value.latent.effective",
                    "model.layers.1.feed_forward.units",
                    "model.layers.1.feed_forward.units.effective",
                    "model.layers.1.feed_forward.write",
                    "model.layers.1.feed_forward.write.effective",
                ]
            } else if matches!(
                family,
                FixtureFamily::DeepSeek | FixtureFamily::DeepSeekGguf
            ) {
                &[
                    "model.layers.0.attention.input",
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.query.latent",
                    "model.layers.0.attention.query.latent.effective",
                    "model.layers.0.attention.key_value.latent",
                    "model.layers.0.attention.key_value.latent.effective",
                    "model.layers.0.feed_forward.units",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.query.latent.effective",
                    "model.layers.1.attention.key_value.latent.effective",
                    "model.layers.1.feed_forward.shared.units",
                    "model.layers.1.feed_forward.shared.units.effective",
                    "model.layers.1.feed_forward.shared.write",
                    "model.layers.1.feed_forward.shared.write.effective",
                    "model.layers.1.feed_forward.contribution",
                    "model.layers.1.feed_forward.contribution.effective",
                ]
            } else if family == FixtureFamily::K2Dense {
                &[
                    "model.layers.0.attention.channels",
                    "model.layers.0.feed_forward.units",
                    "model.layers.1.attention.channels",
                    "model.layers.1.feed_forward.units",
                ]
            } else if family == FixtureFamily::Lfm2Moe {
                &[
                    "model.layers.1.attention.input",
                    "model.layers.1.attention.channels",
                    "model.layers.0.feed_forward.input",
                    "model.layers.0.feed_forward.units",
                    "model.layers.0.mixer.output",
                    "model.layers.0.mixer.output.effective",
                    "model.layers.1.feed_forward.contribution",
                    "model.layers.1.feed_forward.contribution.effective",
                ]
            } else if matches!(
                family,
                FixtureFamily::MuseGlimmerMoe | FixtureFamily::MuseGlimmerGguf(true)
            ) {
                &[
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.channels.effective",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.0.attention.write_input",
                    "model.layers.0.attention.write",
                    "model.layers.0.attention.output",
                    "model.layers.0.feed_forward.input",
                    "model.layers.0.feed_forward.write",
                    "model.layers.0.feed_forward.output",
                    "model.layers.1.feed_forward.write",
                    "model.layers.1.feed_forward.output",
                ]
            } else if matches!(
                family,
                FixtureFamily::MuseGlimmer | FixtureFamily::MuseGlimmerGguf(false)
            ) {
                &[
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.channels.effective",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.0.feed_forward.units",
                    "model.layers.0.feed_forward.units.effective",
                    "model.layers.1.feed_forward.units",
                    "model.layers.1.feed_forward.units.effective",
                    "model.layers.0.attention.write_input",
                    "model.layers.0.attention.write",
                    "model.layers.0.attention.output",
                    "model.layers.0.feed_forward.write_input",
                    "model.layers.0.feed_forward.write",
                    "model.layers.0.feed_forward.output",
                ]
            } else if family == FixtureFamily::InklingDense {
                &[
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.channels.effective",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.0.feed_forward.units",
                    "model.layers.0.feed_forward.units.effective",
                    "model.layers.1.feed_forward.units",
                    "model.layers.1.feed_forward.units.effective",
                    "model.layers.0.feed_forward.projection",
                    "model.layers.0.feed_forward.contribution.effective",
                    "readout.scaled",
                ]
            } else if family == FixtureFamily::Inkling {
                &[
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.channels.effective",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.0.feed_forward.input",
                    "model.layers.0.feed_forward.input.effective",
                    "model.layers.1.feed_forward.contribution",
                    "model.layers.1.feed_forward.contribution.effective",
                    "readout.scaled",
                ]
            } else if matches!(
                family,
                FixtureFamily::KimiLinear | FixtureFamily::KimiLinearGguf
            ) {
                &[
                    "model.layers.0.attention.channels",
                    "model.layers.0.attention.channels.effective",
                    "model.layers.0.attention.query.projected",
                    "model.layers.0.attention.query.convolved",
                    "model.layers.0.attention.key.convolved",
                    "model.layers.0.attention.value.convolved",
                    "model.layers.0.attention.decay.latent",
                    "model.layers.0.attention.gate.latent",
                    "model.layers.1.attention.channels",
                    "model.layers.1.attention.channels.effective",
                    "model.layers.1.attention.key_value.latent",
                    "model.layers.0.feed_forward.units",
                    "model.layers.1.mlp.shared_experts.feed_forward.units",
                    "model.layers.1.feed_forward.contribution",
                ]
            } else if family == FixtureFamily::Qwen3VlMoe {
                &[
                    "model.language_model.layers.0.attention.input",
                    "model.language_model.layers.0.attention.channels",
                    "model.language_model.layers.0.attention.channels.effective",
                    "model.language_model.layers.1.attention.channels",
                    "model.language_model.layers.1.attention.channels.effective",
                    "model.language_model.layers.0.feed_forward.input",
                    "model.language_model.layers.1.feed_forward.input",
                    "model.language_model.layers.1.feed_forward.contribution",
                    "model.language_model.layers.1.feed_forward.contribution.effective",
                ]
            } else if family == FixtureFamily::Qwen3Vl {
                &[
                    "model.language_model.layers.0.attention.input",
                    "model.language_model.layers.0.attention.channels",
                    "model.language_model.layers.0.attention.channels.effective",
                    "model.language_model.layers.1.attention.channels",
                    "model.language_model.layers.1.attention.channels.effective",
                    "model.language_model.layers.0.feed_forward.input",
                    "model.language_model.layers.1.feed_forward.input",
                    "model.language_model.layers.0.feed_forward.units",
                    "model.language_model.layers.0.feed_forward.units.effective",
                    "model.language_model.layers.1.feed_forward.units",
                    "model.language_model.layers.1.feed_forward.units.effective",
                ]
            } else if family == FixtureFamily::Lfm2 {
                &[
                    "model.layers.1.attention.input",
                    "model.layers.1.attention.channels",
                    "model.layers.0.feed_forward.input",
                    "model.layers.0.feed_forward.units",
                    "model.layers.0.mixer.output",
                    "model.layers.0.mixer.output.effective",
                ]
            } else if matches!(
                family,
                FixtureFamily::Qwen3Moe | FixtureFamily::Qwen3MoeGguf | FixtureFamily::GptOss
            ) {
                &[
                    "model.layers.0.attention.input",
                    "model.layers.0.attention.channels",
                    "model.layers.0.feed_forward.input",
                    "model.layers.0.feed_forward.input.effective",
                    "model.layers.1.feed_forward.contribution",
                    "model.layers.1.feed_forward.contribution.effective",
                ]
            } else {
                &[
                    "model.layers.0.attention.input",
                    "model.layers.0.feed_forward.input",
                ]
            };
            if family.is_k2_fp8() && std::env::var_os(EXPERT_CACHE).is_some() {
                let local_experts = eredu_core::balanced_contiguous_range(
                    3,
                    expert_parallel_size,
                    topology.expert_parallel_rank(),
                    false,
                )
                .unwrap()
                .len();
                verify_k2_fp8_bank_query_replay(
                    &mut runtime,
                    &mut reference,
                    2 * local_experts * family.expert_layer_count(local_layer_range.clone()),
                );
            }
            let stream_readout = if family == FixtureFamily::DeepSeekV4 {
                use eredu_core::ModelConfigurationResolver;
                let config = serde_json::from_slice(
                    &std::fs::read(fixture_root.join("config.json")).unwrap(),
                )
                .unwrap();
                eredu_architectures::configuration::MODEL_CONFIGURATIONS
                    .resolve_safetensors(&config)
                    .unwrap()
                    .architecture_plan()
                    .architecture_descriptor()
                    .component_readout
                    .clone()
            } else {
                None
            };
            if family == FixtureFamily::Inkling {
                use eredu_core::ModelConfigurationResolver;
                let config = serde_json::from_slice(
                    &std::fs::read(fixture_root.join("config.json")).unwrap(),
                )
                .unwrap();
                let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
                    .resolve_safetensors(&config)
                    .unwrap()
                    .architecture_plan()
                    .architecture_descriptor();
                assert_eq!(
                    graph.routed_components.len(),
                    4,
                    "two routed and two shared invocations"
                );
                let discovery =
                    <MlxBackend as eredu_core::TextGenerationBackend>::capture_discovery(&runtime)
                        .unwrap();
                for component in &graph.routed_components {
                    for path in [&component.activation, &component.effective_activation] {
                        let point = discovery
                            .catalog
                            .get(path)
                            .expect("architecture-declared expert units must be published");
                        let eredu_core::ObservationValueType::RoutedUnits { geometry, .. } =
                            &point.value_type
                        else {
                            panic!("attributed expert units")
                        };
                        assert_eq!(geometry.experts, component.expert_count as u64);
                        assert_eq!(geometry.units_per_expert, component.units_per_expert as u64);
                    }
                }
            }
            verify_loaded_component_capture(
                &mut runtime,
                &checkpoint,
                &stream,
                &reference_load_options,
                loop_normalization,
                required_unit_points,
                stream_readout.as_ref(),
            );
            return;
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
        if matches!(
            family,
            FixtureFamily::K2Dense | FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)
        ) {
            let (kv_heads, head_dim) = if checkpoint.is_dir() {
                let config: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(fixture_root.join("config.json")).unwrap(),
                )
                .unwrap();
                (
                    config["num_key_value_heads"].as_u64().unwrap(),
                    config["head_dim"].as_u64().unwrap(),
                )
            } else {
                (2, 8)
            };
            let expected = local_layer_range.len() as u64 * 2 * kv_heads
                / tensor_parallel_size as u64
                * head_dim
                * u64::from(state.assumptions.floating_state_dtype_bytes.get());
            assert_eq!(
                state.bytes_per_position_per_batch, expected,
                "rank-local keys and already-mixed values"
            );
            assert_eq!(state.fixed_state_bytes, 0);
        }

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
        let static_memory =
            <MlxBackend<'_> as eredu_core::ModelCapabilityBackend>::static_memory(&runtime)
                .unwrap();
        if matches!(
            family,
            FixtureFamily::K2Dense | FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)
        ) {
            let ordinary = runtime.session().residency_report().unwrap().unwrap();
            let planned = ordinary.offload().planned_bytes();
            let banks = runtime.session().parameter_bank_report().unwrap();
            let bank_bytes = banks.as_ref().map_or(0, |banks| banks.owned_bytes());
            let ordinary_bytes = [
                eredu_core::residency::MemoryTier::Device,
                eredu_core::residency::MemoryTier::Host,
                eredu_core::residency::MemoryTier::Disk,
            ]
            .into_iter()
            .map(|tier| planned.get(tier))
            .sum::<u64>();
            let eredu_core::Observed::Available { value, .. } =
                static_memory.logical_parameter_bytes
            else {
                panic!("missing complete rank-local parameter report")
            };
            assert_eq!(value, ordinary_bytes + bank_bytes);
            let eredu_core::Observed::Available { value, .. } =
                static_memory.planned_disk_backed_bytes
            else {
                panic!("missing disk report")
            };
            assert_eq!(
                value,
                planned.get(eredu_core::residency::MemoryTier::Disk) + bank_bytes
            );
        }

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
        // DSpark must append complete proposal blocks after the local window is
        // full; a short prefix cannot expose mismatched returned-key geometry.
        let text_tokens = if deepseek_dspark_target_mode {
            (0..17).map(|index| 1 + index % 8).collect::<Vec<u32>>()
        } else {
            vec![1, 2]
        };
        let prompt = Array::from_slice(&text_tokens, &[1, text_tokens.len() as i32]);
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
            text_tokens
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
            let max_tokens = if deepseek_dspark_target_mode { 7 } else { 3 };
            let proposal_capacity = if deepseek_dspark_target_mode { 2 } else { 1 };
            if std::env::var_os("EREDU_TEST_SPECULATIVE_PREPARATION_FAILURE").is_some() {
                let before = runtime.text_preparation_usage().unwrap();
                let (result, publications) = execute_neutral_embedded_mtp(
                    &mut runtime,
                    synthetic_prediction_input(&parts, &prefix_tokens),
                    SpeculativeConfig {
                        max_tokens,
                        max_draft_tokens: if expected_rank == 0 {
                            usize::MAX
                        } else {
                            proposal_capacity
                        },
                        temperature: 0.0,
                        eos_token_ids: Vec::new(),
                    },
                );
                let error = result
                    .err()
                    .expect("one invalid rank must reject speculative setup on every rank");
                assert_eq!(publications, 0);
                let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
                let mut found = false;
                while let Some(source) = cause {
                    if expected_rank == 0 {
                        found |= source.downcast_ref::<crate::backend::error::Error>()
                            .is_some_and(|error| matches!(error, crate::backend::error::Error::Speculative(message) if message.contains("lane requests")));
                    } else {
                        found |= source
                            .downcast_ref::<eredu_core::run_preparation::TextPreparationRejected>()
                            .is_some_and(|error| {
                                error.stage
                                    == eredu_core::run_preparation::TextPreparationStage::Sampling
                                    && error.rank == 0
                            });
                    }
                    cause = source.source();
                }
                assert!(
                    found,
                    "rank {expected_rank} lost original preparation cause: {error}"
                );
                let after = runtime.text_preparation_usage().unwrap();
                assert_eq!(after.attempts, before.attempts + 1);
                assert!(after.retained_bytes > before.retained_bytes);
                assert!(after.host_bytes > before.host_bytes);
                // A failed broad native operation has no whole-run restoration
                // witness. Completion may retire its lease but never unfences it.
                assert!(runtime.synchronize().is_err());
                eprintln!(
                    "speculative setup rejection retained cause and usage rank={expected_rank}"
                );
                return;
            }
            if let Ok(case) = std::env::var("EREDU_TEST_SPECULATIVE_CONTROL_DELIVERY") {
                check_speculative_control_delivery(
                    &mut runtime,
                    synthetic_prediction_input(&parts, &prefix_tokens),
                    expected_rank,
                    &case,
                );
                return;
            }
            if let Ok(mode) = std::env::var("EREDU_TEST_SPECULATIVE_SCHEDULER_FAILURE") {
                let before = runtime.text_preparation_usage().unwrap();
                let options = eredu_core::SpeculativeSchedulerOptions {
                    max_in_flight_verifications: if expected_rank == 0 { 0 } else { 1 },
                    ..Default::default()
                };
                let config = SpeculativeConfig {
                    max_tokens,
                    max_draft_tokens: proposal_capacity,
                    temperature: 0.0,
                    eos_token_ids: Vec::new(),
                };
                let input = synthetic_prediction_input(&parts, &prefix_tokens);
                let mut failure = None;
                let (result, publications) = if mode == "controlled" {
                    execute_neutral_embedded_mtp_with(
                        &mut runtime, input, config,
                        eredu_runtime::speculative::DriveControlledSpeculation::new(
                            options, Default::default(),
                            |_: &mut dyn eredu_runtime::speculative::ControlledSpeculativeSession| -> Result<(), eredu_core::speculative::SpeculativeControlError> {
                                panic!("invalid scheduler cannot enter the controlled session")
                            },
                            &mut failure,
                        ),
                    )
                } else {
                    execute_neutral_embedded_mtp_with(
                        &mut runtime,
                        input,
                        config,
                        eredu_runtime::RunSpeculativeGeneration::new(options),
                    )
                };
                let error = result
                    .err()
                    .expect("one scheduler rejection must reach every rank");
                assert_eq!(publications, 0);
                let original: &(dyn std::error::Error + 'static) = match &failure {
                    Some(error) => error,
                    None => &error,
                };
                let mut cause = Some(original);
                let mut found = false;
                while let Some(error) = cause {
                    if expected_rank == 0 {
                        found |= error
                            .downcast_ref::<eredu_core::GenerationError>()
                            .is_some_and(|error| {
                                matches!(
                                    error,
                                    eredu_core::GenerationError::ZeroInFlightVerifications
                                )
                            });
                        // Transparent driver variants retain their typed policy
                        // directly; Error::source may skip the wrapped leaf.
                        found |= error
                            .downcast_ref::<eredu_core::SpeculativeDriverError<safemlx::error::Exception>>()
                            .is_some_and(|error| matches!(error,
                                eredu_core::SpeculativeDriverError::Generation(
                                    eredu_core::GenerationError::ZeroInFlightVerifications)));
                    } else {
                        found |= error.downcast_ref::<eredu_core::run_preparation::TextPreparationRejected>()
                            .is_some_and(|error| error.rank == 0 && error.stage == eredu_core::run_preparation::TextPreparationStage::Request);
                    }
                    cause = error.source();
                }
                assert!(
                    found,
                    "rank {expected_rank} lost {mode} visitor failure: {original}"
                );
                let after = runtime.text_preparation_usage().unwrap();
                assert!(after.attempts > before.attempts);
                assert!(after.retained_bytes > before.retained_bytes);
                assert!(after.host_bytes > before.host_bytes);
                assert!(runtime.synchronize().is_err());
                return;
            }
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
            if inkling_mtp_mode
                || deepseek_mtp_target_mode
                || qwen_hybrid_mtp_mode
                || nemotron_h_mtp_mode
            {
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
            if qwen_hybrid_mtp_mode
                || nemotron_h_mtp_mode
                || std::env::var_os(OPAQUE_INKLING_COMPONENTS).is_some()
            {
                prediction_components::verify_partitioned(
                    &mut runtime,
                    &checkpoint,
                    &stream,
                    &reference_load_options,
                    expected_rank,
                );
                eprintln!(
                    "Prediction components and parameter overlays complete rank={expected_rank}"
                );
            }
            if deepseek_mtp_target_mode {
                if matches!(family, FixtureFamily::DeepSeek | FixtureFamily::DeepSeekV4) {
                    prediction_components::verify_partitioned(
                        &mut runtime,
                        &checkpoint,
                        &stream,
                        &reference_load_options,
                        expected_rank,
                    );
                    eprintln!(
                        "prediction components and parameter overlays complete rank={expected_rank}"
                    );
                    if fixture_root
                        .join("component-independent-experts.json")
                        .exists()
                    {
                        let report = runtime.session().parameter_bank_report().unwrap();
                        let target_experts = family.expert_layer_count(local_layer_range.clone())
                            * 4
                            / expert_parallel_size;
                        if target_experts > 0 {
                            assert!(
                                report
                                    .as_ref()
                                    .is_some_and(|report| report.owned_entries() >= target_experts),
                                "selected target experts must use independent banks"
                            );
                        }
                        if let Some(report) = report {
                            assert!(report.owned_bytes() > 0);
                            assert!(
                                report.bulk().device().requests()
                                    + report.incremental().device().requests()
                                    > 0,
                                "encoded component inference must actually acquire expert banks"
                            );
                        }
                    }
                }
                let fixture_config: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(fixture_root.join("config.json")).unwrap(),
                )
                .unwrap();
                let vocabulary_size = fixture_config["vocab_size"].as_u64().unwrap();
                let invalid_prompt =
                    Array::from_slice(&[u32::try_from(vocabulary_size).unwrap()], &[1, 1]);
                let invalid_parts = [text_input_part(&invalid_prompt)];
                eprintln!("prediction invalid-token rejection rank={expected_rank}");
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
                    if pipeline_rank == 0 {
                        error
                            .to_string()
                            .contains(&format!("token ID is outside 0..{vocabulary_size}"))
                    } else {
                        error
                            .to_string()
                            .contains("another rank reported failure during distributed phase")
                            || error.to_string().contains(
                                "another rank failed during distributed mechanism completion",
                            )
                    },
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
        if family == FixtureFamily::Qwen3 || neutral_gemma_layers.is_some() {
            // One rank's local input failure must reach the common session vote.
            // The next ordinary prefill/reference comparison proves all ranks can
            // reuse their unchanged native session after bounded settlement.
            let failed_input = if expected_rank == 0 {
                ModelInput::new(&[])
            } else {
                ModelInput::new(&parts)
            };
            let error = match runtime.prefill(failed_input.into()) {
                Ok(_) => panic!("asymmetric invalid input entered distributed execution"),
                Err(error) => error,
            };
            assert!(error.model_state_preserved(), "{error}");
            if expected_rank != 0 {
                assert!(
                    error
                        .to_string()
                        .contains("another rank rejected input preparation"),
                    "{error}"
                );
            }
            runtime.synchronize().unwrap();
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
        let neutral_forwards_before =
            crate::tests::support::path_instrumentation::snapshot().forwards;
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
                    FixtureFamily::K2Mova => 16,
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
                FixtureFamily::K2Mova => {
                    family.expert_layer_count(local_layer_range.clone())
                        * [3, 5]
                            .into_iter()
                            .map(|count| {
                                eredu_core::balanced_contiguous_range(
                                    count,
                                    expert_parallel_size,
                                    topology.expert_parallel_rank(),
                                    false,
                                )
                                .unwrap()
                                .len()
                            })
                            .sum::<usize>()
                }
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
            if matches!(family, FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)) {
                assert_eq!(report.banks().len(), 2);
                if checkpoint.is_dir() {
                    let config: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(fixture_root.join("config.json")).unwrap(),
                    )
                    .unwrap();
                    let fp8 = config["quantization_config"].is_object();
                    let budget = if fp8 { 393312 } else { 1152 };
                    assert!(report.peak_device_resident_bytes() <= budget);
                    let local_experts = |count| {
                        eredu_core::balanced_contiguous_range(
                            count,
                            expert_parallel_size,
                            topology.expert_parallel_rank(),
                            false,
                        )
                        .unwrap()
                        .len()
                    };
                    let routed_layers = family.expert_layer_count(local_layer_range.clone());
                    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
                    let value_rows = config["num_key_value_heads"].as_u64().unwrap() as usize
                        * config["head_dim"].as_u64().unwrap() as usize
                        / tensor_parallel_size;
                    let intermediate = config["moe_intermediate_size"].as_u64().unwrap() as usize
                        / tensor_parallel_size;
                    let matrix_bytes = |rows: usize, columns: usize| {
                        if fp8 {
                            rows * columns + rows.div_ceil(128) * columns.div_ceil(128) * 4
                        } else {
                            rows * columns * 4
                        }
                    };
                    let value_bytes = local_experts(3) * matrix_bytes(value_rows, hidden);
                    let feed_forward_bytes = local_experts(5)
                        * (2 * matrix_bytes(intermediate, hidden)
                            + matrix_bytes(hidden, intermediate));
                    assert_eq!(
                        report.owned_bytes(),
                        (routed_layers * (value_bytes + feed_forward_bytes)) as u64
                    );
                } else {
                    assert!(report.peak_device_resident_bytes() <= 16384);
                }
            }
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
        if matches!(
            family,
            FixtureFamily::K2Dense | FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)
        ) {
            verify_distributed_control_branches(&mut runtime, owns_public_output);
        }
    }
}

/// Every rank advances the same serial branch schedule through the public neutral
/// state contract. Copies share weights, while mutable KV storage is independent.
fn verify_distributed_control_branches(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    owns_public_output: bool,
) {
    use eredu_core::execution_control::{ControlSupport, NativeTextStateBackend};
    assert_eq!(
        MlxBackend::native_text_state_support(runtime),
        ControlSupport::Supported
    );
    let estimate = MlxBackend::estimate_native_text_state(runtime, None)
        .unwrap()
        .unwrap();
    assert!(estimate.copy_bytes > 0 && estimate.retained_bytes > 0);
    let saved = MlxBackend::capture_native_text_state(runtime).unwrap();
    let mut first = MlxBackend::copy_native_text_state(runtime, &saved).unwrap();
    let mut second = MlxBackend::copy_native_text_state(runtime, &saved).unwrap();
    assert!(
        MlxBackend::estimate_native_text_growth(runtime, &saved, 4)
            .unwrap()
            .unwrap()
            > 0
    );
    let advance = |runtime: &mut ModelRuntime<MlxBackend<'_>>, tokens: &[u32]| {
        tokens
            .iter()
            .map(|&token| {
                let (backend, session) = runtime.parts_mut();
                let output = session
                    .decode(backend, Array::from_slice(&[token], &[1, 1]))
                    .unwrap()
                    .wait()
                    .unwrap();
                assert_eq!(output.logits().is_some(), owns_public_output);
                output.logits().map(|logits| {
                    logits
                        .as_array()
                        .evaluated()
                        .unwrap()
                        .as_slice::<f32>()
                        .to_vec()
                })
            })
            .collect::<Vec<_>>()
    };
    let baseline = advance(runtime, &[1, 2, 3, 4]);
    MlxBackend::exchange_native_text_state(runtime, &mut first).unwrap();
    drop(first);
    assert_eq!(advance(runtime, &[1, 2]), baseline[..2]);
    let mid = MlxBackend::capture_native_text_state(runtime).unwrap();
    let mut resumed = MlxBackend::copy_native_text_state(runtime, &mid).unwrap();
    drop(mid);
    let _ = advance(runtime, &[5]);
    MlxBackend::exchange_native_text_state(runtime, &mut second).unwrap();
    drop(second);
    assert_eq!(advance(runtime, &[1, 2, 3, 4]), baseline);
    MlxBackend::exchange_native_text_state(runtime, &mut resumed).unwrap();
    drop(resumed);
    assert_eq!(advance(runtime, &[3, 4]), baseline[2..]);
    // Reusing an immutable snapshot after all descendants advanced remains exact.
    let mut restored = MlxBackend::copy_native_text_state(runtime, &saved).unwrap();
    MlxBackend::exchange_native_text_state(runtime, &mut restored).unwrap();
    assert_eq!(advance(runtime, &[1, 2, 3, 4]), baseline);
}

fn verify_public_partition_parameter_access(
    runtime: &mut eredu_core::ModelRuntime<MlxBackend<'_>>,
    reference: &mut eredu_core::ModelRuntime<MlxBackend<'_>>,
    rank: usize,
    family: FixtureFamily,
) {
    use eredu_core::{capture::CaptureUsage, parameters::*};
    let facts = MlxBackend::parameter_discovery(runtime).unwrap();
    let ordinary = MlxBackend::parameter_discovery(reference).unwrap();
    assert_eq!(facts.parameters.len(), ordinary.parameters.len());
    if family.is_v3() || family == FixtureFamily::DeepSeekV4 {
        for parameter in facts
            .parameters
            .iter()
            .filter(|p| component_fixture_weight(family, &p.id))
        {
            assert_eq!(
                parameter.access(),
                ParameterAccess {
                    query: true,
                    projection: true,
                    replacement: true,
                },
                "every component parameter must support effective access: {}: {}",
                parameter.id,
                parameter.condition,
            );
        }
    }
    for (actual, expected) in facts.parameters.iter().zip(&ordinary.parameters) {
        assert_eq!(
            (&actual.id, &actual.shared_id, &actual.shape, actual.dtype),
            (
                &expected.id,
                &expected.shared_id,
                &expected.shape,
                expected.dtype
            )
        );
        assert_eq!(actual.access().query, expected.access().query);
        assert_eq!(actual.access().projection, expected.access().projection);
        assert_eq!(actual.access().replacement, expected.access().replacement);
    }
    // This trial queries every floating V4 slot and contracts every axis. Each
    // attempt also reserves a complete distributed catalogue; price the whole
    // repeated trial, including replica metadata, instead of the smaller
    // representative-parameter trial used by other families. These are
    // cumulative logical charges, not simultaneously allocated host memory.
    let repeated_catalog_factor = if family == FixtureFamily::DeepSeekV4 {
        16
    } else {
        1
    };
    let allowance = CaptureUsage {
        captures: 100_000,
        retained_bytes: repeated_catalog_factor << 30,
        host_bytes: (2 * repeated_catalog_factor) << 30,
        encoded_bytes: repeated_catalog_factor << 30,
    };
    let limits = facts.usage.checked_add(allowance).unwrap();
    let weights = facts
        .parameters
        .iter()
        .filter(|p| p.access().query && component_fixture_weight(family, &p.id))
        .collect::<Vec<_>>();
    assert!(!weights.is_empty());
    let mut targets = if family.is_v3() || family == FixtureFamily::DeepSeekV4 {
        weights
            .iter()
            .map(|parameter| parameter.id.as_str())
            .collect()
    } else if family == FixtureFamily::Qwen3 {
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "lm_head.weight",
        ]
    } else {
        vec![
            weights[0].id.as_str(),
            weights[weights.len() / 2].id.as_str(),
            weights.last().unwrap().id.as_str(),
        ]
    };
    if matches!(
        family,
        FixtureFamily::NemotronH | FixtureFamily::NemotronHGguf
    ) {
        targets = vec![
            "model.embeddings.weight",
            "model.layers.0.mamba.in_proj.weight",
            "model.layers.0.mamba.out_proj.weight",
            if family == FixtureFamily::NemotronH {
                "model.layers.1.mlp.up_proj.weight"
            } else {
                "model.layers.1.moe.shared_experts.up_proj.weight"
            },
            if family == FixtureFamily::NemotronH {
                "model.layers.1.mlp.down_proj.weight"
            } else {
                "model.layers.1.moe.shared_experts.down_proj.weight"
            },
            "model.layers.2.moe.shared_experts.up_proj.weight",
            "model.layers.2.moe.shared_experts.down_proj.weight",
            "model.layers.3.attention.q_proj.weight",
            "model.layers.3.attention.k_proj.weight",
            "model.layers.3.attention.v_proj.weight",
            "model.layers.3.attention.o_proj.weight",
            "model.norm_f.weight",
            "lm_head.weight",
        ];
    }
    if matches!(
        family,
        FixtureFamily::GptOss
            | FixtureFamily::Qwen3Moe
            | FixtureFamily::Qwen3MoeGguf
            | FixtureFamily::Lfm2Moe
            | FixtureFamily::K2Mova
            | FixtureFamily::K2Fp8(_)
            | FixtureFamily::NemotronH
            | FixtureFamily::NemotronHGguf
    ) {
        assert!(
            facts
                .parameters
                .iter()
                .filter(|p| p.access().query && p.shape.len() == 3)
                .count()
                >= 2,
            "released-layout fixture must expose its grouped read and write parameters"
        );
    }
    targets.extend(
        facts
            .parameters
            .iter()
            .filter(|p| p.access().query && p.shape.len() == 3)
            .map(|p| p.id.as_str()),
    );
    let mut seen = std::collections::BTreeSet::new();
    targets.retain(|target| seen.insert(*target));
    let target = targets[0];
    let descriptor = facts.parameters.iter().find(|p| p.id == target).unwrap();
    let selected = ParameterRegion {
        starts: descriptor.shape.iter().map(|_| 0).collect(),
        shape: descriptor.shape.iter().map(|n| (*n).min(2)).collect(),
    };
    // One peer exhausts its caller allowance before local metadata is built.
    assert!(MlxBackend::query_parameter(
        runtime,
        &facts.identity,
        target,
        selected.clone(),
        if rank == 1 { facts.usage } else { limits }
    )
    .is_err());
    // One peer presents foreign local authority while every request intent matches.
    assert!(MlxBackend::query_parameter(
        runtime,
        if rank == 1 {
            "foreign-model"
        } else {
            &facts.identity
        },
        target,
        selected,
        limits
    )
    .is_err());
    for target in targets {
        let descriptor = facts.parameters.iter().find(|p| p.id == target).unwrap();
        let region = ParameterRegion {
            starts: descriptor.shape.iter().map(|n| u64::from(*n > 2)).collect(),
            shape: descriptor
                .shape
                .iter()
                .map(|n| if *n > 2 { *n - 2 } else { *n })
                .collect(),
        };
        let limits = parameter_fixture_limits(runtime, allowance);
        let actual =
            MlxBackend::query_parameter(runtime, &facts.identity, target, region.clone(), limits)
                .unwrap_or_else(|error| panic!("effective query {target}: {error}"));
        let limits = parameter_fixture_limits(reference, allowance);
        let expected = MlxBackend::query_parameter(
            reference,
            &ordinary.identity,
            target,
            region.clone(),
            limits,
        )
        .unwrap();
        assert_eq!(
            actual.values, expected.values,
            "global loaded query {target}"
        );
        assert!(actual.values.iter().any(|v| v.abs() > 1e-5));
        for axis in 0..descriptor.shape.len() {
            let projection = ParameterProjection {
                region: region.clone(),
                axis,
                directions: 2,
                coefficients: (0..region.shape[axis] * 2)
                    .map(|i| if i % 2 == 0 { 0.75 } else { -0.5 })
                    .collect(),
            };
            let limits = parameter_fixture_limits(runtime, allowance);
            let actual = MlxBackend::project_parameter(
                runtime,
                &facts.identity,
                target,
                projection.clone(),
                limits,
            )
            .unwrap();
            let limits = parameter_fixture_limits(reference, allowance);
            let expected = MlxBackend::project_parameter(
                reference,
                &ordinary.identity,
                target,
                projection,
                limits,
            )
            .unwrap();
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(actual.values.len(), expected.values.len());
            for (actual, expected) in actual.values.iter().zip(expected.values) {
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "global projection {target} axis{axis}: {actual} vs {expected}"
                );
            }
        }
    }
    let after = MlxBackend::parameter_discovery(runtime).unwrap();
    assert_eq!(after.identity, facts.identity);
    assert!(after.usage.host_bytes > facts.usage.host_bytes);
    assert!(after.coordination_usage.attempts > facts.coordination_usage.attempts);
}
