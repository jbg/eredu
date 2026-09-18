//! Conditional target components include assembled input and actual DeepStack writes.
use super::*;

#[test]
fn conditional_qwen_prediction_tokens_match_interleaved_target_media() {
    use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
    type Model = qwen::hybrid::ConditionalLayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let context = NumericContext::default();
    for routed in [false, true] {
        let parsed =
            qwen::hybrid::model_args_from_config_value(&conditional_qwen_partition_config(routed))
                .unwrap();
        let text = |ids: &[usize]| {
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(NumericTensor::token_ids(ids)),
                [],
            )
            .unwrap()
        };
        let media = |modality, time: i32| {
            PreparedInputPart::new_with_extents(
                modality,
                PreparedInputPayload::Tensor(NumericTensor::new(
                    [4 * time, 24],
                    (0..96 * time)
                        .map(|index| (index as f32 - 41.0) / 97.0)
                        .collect(),
                )),
                [(
                    InputMetadataKey::PatchGrid,
                    NumericTensor::new([1, 3], vec![time as f32, 2.0, 2.0]),
                )],
                [InputExtent::PatchGrid {
                    time: time as usize,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap()
        };
        let original = PreparedModelInput::new(
            vec![
                text(&[1, 2]),
                media(InputModality::Image, 1),
                text(&[3]),
                media(InputModality::Video, 2),
                PreparedInputPart::new(
                    InputModality::Image,
                    PreparedInputPayload::Embeddings(NumericTensor::new([1, 2, 8], vec![0.25; 16])),
                    [],
                )
                .unwrap(),
                text(&[4]),
            ],
            |value| eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, value),
        )
        .unwrap();
        let admitted =
            eredu_architectures::media_plan::admit_qwen_hybrid_input(
                &parsed,
                &original,
                &NumericInputInspector,
            )
            .unwrap();
        let paired = PreparedCompositeInput::new(&original, &admitted).unwrap();
        let actual =
            <Model as CompositeArchitecture<NumericBackend, State>>::prepared_prediction_token_ids(
                paired, &context,
            )
            .unwrap();
        assert_tensor_exact(
            &actual,
            &NumericTensor::token_ids(&[1, 2, 5, 3, 6, 6, 0, 0, 4]),
            "Qwen prediction preserves text, image, video, and projected segment order",
        );
        assert_eq!(admitted.decoder_positions(), 9);
        let target = qwen::hybrid::prepare_conditional_input(paired, &context).unwrap();
        assert_tensor_exact(
            &actual,
            &target.token_ids(&context).unwrap(),
            "Qwen target and prediction use identical semantic tokens",
        );
    }
}

fn trial(
    parsed: &qwen::hybrid::ParsedHybridConfig,
    descriptor: &eredu_core::ArchitectureDescriptor,
    layout: Option<LocalModelLayout>,
    parallel: Option<&NumericParallelContext>,
    mode: usize,
    observed: bool,
    parameters: Option<&BTreeMap<String, NumericTensor>>,
) -> Vec<(NumericTensor, BTreeMap<String, NumericTensor>)> {
    let mut context = layout
        .clone()
        .map(NumericContext::with_local_layout)
        .unwrap_or_default();
    context.bind_checkpoint_values = parameters.is_some();
    let geometry = layout
        .as_ref()
        .map(|layout| qwen::hybrid::conditional_local_geometry(parsed, layout).unwrap());
    let mut model = match geometry {
        Some(geometry) => qwen::hybrid::ConditionalLayeredModel::<NumericBackend>::new_parallel(
            parsed.clone(),
            geometry,
            &context,
        ),
        None => {
            qwen::hybrid::ConditionalLayeredModel::<NumericBackend>::new(parsed.clone(), &context)
        }
    }
    .unwrap();
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let source = self.0.get(metadata.id().as_str()).unwrap_or_else(|| {
                panic!(
                    "missing conditional reference parameter {}",
                    metadata.id().as_str()
                )
            });
            assert!(
                source.shape == value.shape
                    || (value.shape.len() == 2
                        && source.shape == [value.shape[0], 1, value.shape[1]]),
                "conditional reference geometry {}: {:?} vs {:?}",
                metadata.id().as_str(),
                source.shape,
                value.shape
            );
            value.data.clone_from(&source.data);
        }
    }
    if let Some(parameters) = parameters {
        assert!(
            layout.is_none(),
            "the source oracle binds global serial parameters"
        );
        <qwen::hybrid::ConditionalLayeredModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
        >>::static_modules_mut(&mut model)
        .visit_parameters_mut(&mut Populate(parameters));
    }
    let state_layout = model.state_layout(None).unwrap();
    let units = [(0, 0), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .map(|(group, index)| {
            let mut unit =
                <qwen::hybrid::ConditionalLayeredModel<NumericBackend> as LayeredArchitecture<
                    NumericBackend,
                    DeviceState<NumericBackend, NumericHybridLayerState>,
                >>::build_unit(&model, group, index, &context)
                .unwrap();
            if let Some(parameters) = parameters {
                unit.visit_parameters_mut(&mut Populate(parameters));
            }
            unit
        })
        .collect();
    let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
    let mut state = DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let channel = "model.layers.0.attention.channels";
    let unit = if parsed.text.is_moe() {
        "model.layers.1.mlp.shared_expert.feed_forward.units"
    } else {
        "model.layers.1.feed_forward.units"
    };
    let local = |path: &str, id: u32| -> Option<u32> {
        layout.as_ref().map_or(Some(id), |layout| {
            let component = descriptor
                .components
                .iter()
                .find(|group| group.activation == path)
                .unwrap();
            eredu_architectures::component_partition::derive_component_coordinates(
                component,
                layout.tensor(&component.write_weight).unwrap(),
            )
            .unwrap()
            .global_to_local(id as usize)
            .map(|value| value as u32)
        })
    };
    let mut masks = Vec::new();
    if matches!(mode, 1 | 5) {
        masks.push((channel, local(channel, 1).into_iter().collect(), false));
    }
    if matches!(mode, 2 | 5) {
        masks.push((unit, local(unit, 3).into_iter().collect(), true));
    }
    let pixels = NumericTensor::new(
        [4, 24],
        (0..96).map(|index| (index as f32 - 41.0) / 97.0).collect(),
    );
    let image_tokens = NumericTensor::token_ids(&[5]);
    let grid = [(1, 2, 2)];
    [vec![1, 2], vec![3], vec![4]]
        .into_iter()
        .enumerate()
        .map(|(step, ids)| {
            let tokens = NumericTensor::token_ids(&ids);
            let mut parts = vec![qwen::vl::InputPart::Text(&tokens)];
            if step == 0 {
                parts.push(qwen::vl::InputPart::Image {
                    tokens: &image_tokens,
                    grid: &grid,
                });
            }
            let input = qwen::hybrid::ConditionalInput::Target {
                parts: &parts,
                pixels: (step == 0).then_some(&pixels),
                mask: None,
            };
            let mut observer = GlobalComponentObserver {
                inner: NumericLifecycleObserver {
                    zero_path: match mode {
                        3 => Some("model.layers.0.deepstack.output".into()),
                        4 => Some("readout.embedding".into()),
                        _ => None,
                    },
                    ..Default::default()
                },
                values: BTreeMap::new(),
                layout: None,
                masks: &masks,
                position: if step == 0 { 2 } else { 0 },
                projection_plan: None,
                producer_receipts: None,
                prediction: step as u64,
            };
            let output = match (parallel, observed) {
                (None, false) => runtime.forward(input, &mut state, &context),
                (Some(parallel), false) => {
                    runtime.forward_parallel(input, &mut state, parallel, &context)
                }
                (None, true) => {
                    runtime.forward_with_observer(input, &mut state, &context, &mut observer)
                }
                (Some(parallel), true) => runtime.forward_parallel_with_observer(
                    input,
                    &mut state,
                    parallel,
                    &context,
                    &mut observer,
                ),
            }
            .unwrap();
            (output, observer.values)
        })
        .collect()
}

#[test]
fn conditional_qwen_components_reconstruct_media_and_deepstack_writes_serial_and_tp2() {
    for routed in [false, true] {
        let config = conditional_qwen_partition_config(routed);
        let parsed = qwen::hybrid::model_args_from_config_value(&config).unwrap();
        let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        for suffix in [
            "input",
            "output",
            "output.effective",
            "residual",
            "residual.effective",
        ] {
            let point = descriptor
                .observations
                .get(&format!("model.layers.0.deepstack.{suffix}"))
                .unwrap();
            assert!(point.prefill && !point.decode);
            assert!(point
                .requirements
                .contains(&eredu_core::ObservationRequirement::MediaInput));
        }
        let context = NumericContext::default();
        let model =
            qwen::hybrid::ConditionalLayeredModel::<NumericBackend>::new(parsed.clone(), &context)
                .unwrap();
        let parameters = model.parameter_description(&context).unwrap().into_owned();
        let groups = parameters
            .groups()
            .iter()
            .map(|owned| owned.group().clone())
            .collect::<Vec<_>>();
        let ordinary = trial(&parsed, &descriptor, None, None, 0, false, None);
        for mode in 0..6 {
            let expected = trial(&parsed, &descriptor, None, None, mode, true, None);
            if mode == 0 {
                for (actual, baseline) in expected.iter().zip(&ordinary) {
                    assert_tensor_exact(
                        &actual.0,
                        &baseline.0,
                        "conditional no-op instrumentation",
                    );
                }
                assert!(expected[0].1["model.layers.0.deepstack.output"]
                    .data
                    .iter()
                    .any(|value| value.abs() > 1e-5));
            } else {
                assert!(
                    expected
                        .iter()
                        .zip(&ordinary)
                        .any(|(actual, baseline)| actual
                            .0
                            .data
                            .iter()
                            .zip(&baseline.0.data)
                            .any(|(a, b)| (a - b).abs() > 1e-6)),
                    "conditional nonzero causal trial {mode}"
                );
            }
            for (_, capture) in &expected {
                let mut residual = capture["readout.embedding.effective"].clone();
                for (layer, mixer) in [(0, "attention"), (1, "mixer")] {
                    for operator in [mixer, "feed_forward"] {
                        residual = residual
                            .add(
                                &capture
                                    [&format!("model.layers.{layer}.{operator}.output.effective")],
                                &context,
                            )
                            .unwrap();
                    }
                    if layer == 0 {
                        residual = residual
                            .add(
                                &capture["model.layers.0.deepstack.output.effective"],
                                &context,
                            )
                            .unwrap();
                    }
                }
                assert_tensor_close(
                    &residual,
                    &capture["readout.residual"],
                    "assembled input plus text and DeepStack writes",
                );
            }
            let world = NumericParallelGroup::new(2);
            let actual = std::thread::scope(|scope| {
                let handles = (0..2)
                    .map(|rank| {
                        let parsed = &parsed;
                        let descriptor = &descriptor;
                        let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                        let world = Arc::clone(&world);
                        scope.spawn(move || {
                            let parallel = NumericParallelContext::new(rank, world);
                            let result = trial(
                                parsed,
                                descriptor,
                                Some(layout),
                                Some(&parallel),
                                mode,
                                true,
                                None,
                            );
                            (result, parallel.trace())
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for (steps, _) in &actual {
                for (step, (score, capture)) in steps.iter().enumerate() {
                    assert_tensor_close(score, &expected[step].0, "conditional masked TP2 scores");
                    for path in [
                        "readout.embedding.effective",
                        "model.layers.0.deepstack.output.effective",
                        "model.layers.0.deepstack.residual.effective",
                        "readout.residual",
                        "readout.normalized",
                        "readout.linear",
                    ] {
                        assert_tensor_close(&capture[path], &expected[step].1[path], path);
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_conditional_qwen_component_ownership_and_media_writes_cross_partitions() {
    prepared_conditional_components(false);
}

#[test]
fn prepared_conditional_qwen_independent_banks_preserve_components_and_bound_residency() {
    prepared_conditional_components(true);
}

fn prepared_conditional_components(independent: bool) {
    for tied in [false, true] {
        prepared_conditional_components_with_tying(independent, tied);
    }
}

fn prepared_conditional_components_with_tying(independent: bool, tied: bool) {
    for routed in [false, true] {
        if independent && !routed {
            continue;
        }
        let mut config = conditional_qwen_partition_config(routed);
        config["tie_word_embeddings"] = tied.into();
        config["text_config"]["tie_word_embeddings"] = tied.into();
        // This projector executes on the later vision stage, then travels back
        // to the first target stage where its residual contribution is applied.
        config["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([1]);
        let parsed = qwen::hybrid::model_args_from_config_value(&config).unwrap();
        let (artifact, bits) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let context = NumericContext {
            bind_checkpoint_values: true,
            ..Default::default()
        };
        let store = eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap();
        let mut recipes = qwen::hybrid::static_recipes(&store).unwrap();
        for ordinal in 0..4 {
            recipes
                .extend(qwen::hybrid::conditional_unit_recipes(&store, &parsed, ordinal).unwrap());
        }
        let mut fixture = bits
            .into_iter()
            .map(|(name, (shape, bits))| {
                (
                    name,
                    NumericTensor::new(shape, bits.into_iter().map(f32::from_bits).collect()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (target, recipe) in recipes {
            fixture.insert(
                target,
                payload::recipe_value(&recipe, &store, &context).unwrap(),
            );
        }
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let descriptor = inspection.architecture_plan().architecture_descriptor();
        let parameters = numeric_composite_parameter_description(&config);
        let expected = (0..6)
            .map(|mode| trial(&parsed, &descriptor, None, None, mode, true, Some(&fixture)))
            .collect::<Vec<_>>();
        let image = eredu_runtime::PreparedInputPart::new_with_extents(
            eredu_core::InputModality::Image,
            eredu_runtime::PreparedInputPayload::Tensor(NumericTensor::new(
                [4, 24],
                (0..96).map(|index| (index as f32 - 41.0) / 97.0).collect(),
            )),
            [(
                eredu_core::InputMetadataKey::PatchGrid,
                NumericTensor::new([1, 3], vec![1.0, 2.0, 2.0]),
            )],
            [eredu_core::InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap();
        let prefill = eredu_runtime::PreparedModelInput::new(
            vec![
                eredu_runtime::PreparedInputPart::new(
                    eredu_core::InputModality::Text,
                    eredu_runtime::PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[
                        1, 2,
                    ])),
                    [],
                )
                .unwrap(),
                image,
            ],
            |value| eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, value),
        )
        .unwrap();
        let inputs = [
            prefill,
            numeric_text_prepared_input(&[3]),
            numeric_text_prepared_input(&[4]),
        ];
        let mut topologies = vec![
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ];
        if routed {
            topologies.extend([
                ParallelTopology::new(1, 1, 2, 1).unwrap(),
                ParallelTopology::new(2, 1, 2, 1).unwrap(),
                ParallelTopology::new(1, 2, 2, 1).unwrap(),
                ParallelTopology::new(2, 2, 2, 1).unwrap(),
            ]);
        }
        for residency in [
            eredu_core::ResidencyPlan::FullyResident,
            eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(1 << 20),
                host_budget_bytes: Some(1 << 20),
            },
            eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 1 << 20,
                host_budget_bytes: 1 << 20,
                host_lookahead: 1,
                background_queue: 1,
            },
        ] {
            for &topology in &topologies {
                eprintln!("conditional components routed={routed}, {topology:?}, {residency:?}");
                let world = Arc::new(NumericPartitionWorld::default());
                std::thread::scope(|scope| {
                    let handles = (0..topology.world_size())
                        .map(|rank| {
                            let inspection = &inspection;
                            let parameters = &parameters;
                            let descriptor = &descriptor;
                            let inputs = &inputs;
                            let expected = &expected;
                            let residency = residency.clone();
                            let world = Arc::clone(&world);
                            scope.spawn(move || {
                                let plan = prepared_adapter::plan(None)
                                    .with_topology(topology)
                                    .with_residency(residency);
                                let sources = partitioned_adapter::prepare_plan_with_banks(
                                    inspection,
                                    &plan,
                                    rank,
                                    std::time::Duration::from_secs(10),
                                    independent.then(|| ParameterBankLoadOptions::new(
                                        eredu_core::residency::OffloadConfig::new(Some(384), Some(1 << 20), 1).unwrap(),
                                        384, 384,
                                    ).unwrap()),
                                )
                                .unwrap();
                                let component_layout = sources
                                    .selected()
                                    .execution()
                                    .component_partition_layout(descriptor, parameters)
                                    .unwrap()
                                    .unwrap();
                                let rank_topology =
                                    ParallelRankTopology::new(topology, rank).unwrap();
                                let layout =
                                    eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, rank_topology)
                                        .unwrap();
                                let mut context = NumericContext::with_partition(layout, rank, world);
                                context.bind_checkpoint_values = true;
                                reset_reference_stage_evidence("SafeTensors");
                                let mut executable =
                                    partitioned_adapter::composite(sources, &context).unwrap();
                                for mode in 0..6 {
                                    executable.reset().unwrap();
                                    let unit = if routed {
                                        "model.layers.1.mlp.shared_expert.feed_forward.units"
                                    } else {
                                        "model.layers.1.feed_forward.units"
                                    };
                                    let mut masks = Vec::new();
                                    if matches!(mode, 1 | 5) {
                                        masks.push((
                                            "model.layers.0.attention.channels",
                                            vec![1],
                                            false,
                                        ));
                                    }
                                    if matches!(mode, 2 | 5) {
                                        masks.push((unit, vec![3], true));
                                    }
                                    for (step, input) in inputs.iter().enumerate() {
                                        let mut observer = GlobalComponentObserver {
                                            inner: NumericLifecycleObserver {
                                                zero_path: match mode {
                                                    3 => Some(
                                                        "model.layers.0.deepstack.output".into(),
                                                    ),
                                                    4 => Some("readout.embedding".into()),
                                                    _ => None,
                                                },
                                                ..Default::default()
                                            },
                                            values: BTreeMap::new(),
                                            layout: Some(&component_layout),
                                            masks: &masks,
                                            position: if step == 0 { 2 } else { 0 },
                                            projection_plan: None,
                                            producer_receipts: None,
                                            prediction: step as u64,
                                        };
                                        let output = executable
                                            .forward_observed(input, step == 0, &mut observer)
                                            .unwrap();
                                        for path in &observer.inner.paths {
                                            assert!(
                                                observer.values[path].data.iter().all(|value| value.is_finite()),
                                                "nonfinite conditional capture rank={rank} mode={mode} step={step}: {path}"
                                            );
                                        }
                                        for path in [
                                            "readout.embedding.effective",
                                            "model.layers.0.attention.input.effective",
                                            "model.layers.0.attention.output.effective",
                                            "model.layers.0.feed_forward.input.effective",
                                            "model.layers.0.feed_forward.output.effective",
                                        ] {
                                            if let Some(actual) = observer.values.get(path) {
                                                assert_tensor_close(actual, &expected[mode][step].1[path],
                                                    &format!("rank={rank} mode={mode} step={step} {path}"));
                                            }
                                        }
                                        let baseline = &expected[mode][step].0;
                                        let reference = NumericTensor::new(
                                            [1, 1, 7],
                                            baseline.data[baseline.data.len() - 7..].to_vec(),
                                        );
                                        assert_tensor_close(
                                            &output,
                                            &reference,
                                            "prepared conditional component scores",
                                        );
                                        for path in [
                                            "readout.embedding.effective",
                                            "model.layers.0.deepstack.output.effective",
                                            "model.layers.0.deepstack.residual.effective",
                                            "readout.residual",
                                            "readout.normalized",
                                            "readout.linear",
                                        ] {
                                            let declared = component_layout
                                                .observation(path)
                                                .unwrap_or_else(|| panic!("missing conditional partition declaration {path}"))
                                                .coordinates()
                                                .is_some();
                                            assert_eq!(
                                                observer.values.contains_key(path),
                                                declared,
                                                "rank {rank} ownership of {path}"
                                            );
                                            if let Some(actual) = observer.values.get(path) {
                                                assert_tensor_close(
                                                    actual,
                                                    &expected[mode][step].1[path],
                                                    path,
                                                );
                                            }
                                        }
                                    }
                                }
                                if independent {
                                    let evidence = last_reference_stage_evidence();
                                    assert!(!evidence.bank_acquisitions.is_empty(), "selected independent cache must execute");
                                    assert!(evidence.bank_completions > 0);
                                    assert!(evidence.peak_bank_bytes <= 384);
                                    let owned = balanced_rank_range(2, topology.pipeline(), rank_topology.pipeline_parallel_rank());
                                    assert!(evidence.bank_acquisitions.iter().all(|key| owned.contains(&key.unit())));
                                    if topology.tensor() == 1 && topology.pipeline() == 1 {
                                        assert!(evidence.bank_evictions > 0, "one-entry budget must evict between layers");
                                    }
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            }
        }
    }
}
