//! The actual MLA output and dense-unit seams across both compressed cache modes.
use super::*;

#[path = "deepseek_v3/additive.rs"]
mod additive;

#[path = "deepseek_v3/prediction.rs"]
mod prediction;

fn args(query_rank: Option<i32>) -> deepseek::V3Args {
    deepseek::parse_v3_config(&v3_config(query_rank)).unwrap()
}

fn v3_config(query_rank: Option<i32>) -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v3", "hidden_size": 8,
        "intermediate_size": 10, "moe_intermediate_size": 4,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "vocab_size": 16, "max_position_embeddings": 64,
        "q_lora_rank": query_rank, "kv_lora_rank": 3,
        "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 3,
        "first_k_dense_replace": 1, "n_routed_experts": 2,
        "n_shared_experts": 1, "num_experts_per_tok": 1,
        "n_group": 1, "topk_group": 1, "tie_word_embeddings": false
    })
}

#[derive(Default)]
struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a NumericTensor) {
        self.0.insert(metadata.id().as_str().into(), value.clone());
    }
}

fn describe(config: &serde_json::Value) -> eredu_core::ArchitectureDescriptor {
    eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor()
}

#[test]
fn v3_pipeline_components_match_whole_execution_across_residencies_and_cached_steps() {
    verify_v3_partition_components(true, &[ParallelTopology::new(1, 2, 1, 1).unwrap()]);
}

#[test]
fn v3_mixed_components_match_tensor_pipeline_and_bounded_execution() {
    verify_v3_partition_components(
        true,
        &[
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ],
    );
}

#[test]
fn v3_dense_components_match_tensor_pipeline_and_bounded_execution() {
    verify_v3_partition_components(
        false,
        &[
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
        ],
    );
}

#[test]
fn v3_dense_partition_transforms_preserve_source_geometry_and_cached_execution() {
    verify_v3_partition_transforms(false, false);
}

#[test]
fn v3_mixed_partition_transforms_preserve_source_geometry_and_cached_execution() {
    verify_v3_partition_transforms(true, false);
}

#[test]
fn v3_mixed_cached_partition_transforms_preserve_companion_ownership_and_execution() {
    verify_v3_partition_transforms(true, true);
}

fn verify_v3_partition_transforms(sparse: bool, independent_banks: bool) {
    let mut config = v3_config(Some(16));
    for (name, value) in [
        ("hidden_size", 32),
        ("intermediate_size", 64),
        ("num_hidden_layers", 2),
        ("first_k_dense_replace", if sparse { 1 } else { 2 }),
        (
            "moe_intermediate_size",
            if independent_banks { 64 } else { 32 },
        ),
        ("num_attention_heads", 4),
        ("vocab_size", 32),
        ("kv_lora_rank", 16),
        ("qk_nope_head_dim", 4),
        ("qk_rope_head_dim", 4),
        ("v_head_dim", 8),
    ] {
        config[name] = value.into();
    }
    let (root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let quantization = eredu_core::QuantizationRequest::Affine {
        group_size: 16,
        bits: 4,
    };
    let plan = prepared_adapter::plan(Some(quantization));
    let tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let sources = prepared_adapter::prepare(
        &inspection,
        &plan,
        &prepared_adapter::NumericPreparationProvider { addressable: false },
    )
    .unwrap();
    // Scalar affine operators retain expanded logical weights. Their context
    // therefore uses the source coordinates; construction still derives and
    // validates the real selected packed layout and both companion tensors.
    let parameters =
        deepseek::parallel::v3_parameter_description(&deepseek::parse_v3_config(&config).unwrap())
            .unwrap();
    let target_args = deepseek::v3_with_checkpoint_formats(
        &deepseek::parse_v3_config(&config).unwrap(),
        sources
            .selected()
            .execution()
            .text_realization()
            .parameters()
            .iter()
            .map(|parameter| (parameter.name().to_owned(), parameter.executable()))
            .collect(),
    )
    .unwrap();
    let target_parameters = deepseek::parallel::v3_parameter_description(&target_args).unwrap();
    let expected = if sparse {
        prepared_adapter::routed(sources, &context, &tokens)
    } else {
        prepared_adapter::replicated(sources, &context, &tokens)
    }
    .unwrap();
    assert!(expected.outputs[0]
        .data
        .iter()
        .any(|value| value.abs() > 1e-5));
    for topology in [
        ParallelTopology::new(2, 1, 1, 1).unwrap(),
        ParallelTopology::new(1, 2, 1, 1).unwrap(),
        ParallelTopology::new(2, 2, 1, 1).unwrap(),
    ] {
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
            let world = Arc::new(NumericPartitionWorld::default());
            std::thread::scope(|scope| {
                let handles = (0..topology.world_size())
                .map(|rank| {
                    let world = world.clone();
                    let plan = plan.clone().with_topology(topology).with_residency(residency.clone());
                    let parameters = &parameters;
                    let target_parameters = &target_parameters;
                    let inspection = &inspection;
                    let expected = &expected.outputs;
                    let tokens = &tokens;
                    scope.spawn(move || {
                        let mut layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                            parameters,
                            ParallelRankTopology::new(topology, rank).unwrap(),
                        ).unwrap();
                        let encoded = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                            target_parameters, ParallelRankTopology::new(topology, rank).unwrap(),
                        ).unwrap();
                        for (name, companion) in encoded.tensors() {
                            if !layout.contains(name) {
                                layout.insert(name.to_owned(), companion.clone());
                            }
                        }
                        let mut context = NumericContext::with_partition(layout, rank, world);
                        context.bind_checkpoint_values = true;
                        let sources = partitioned_adapter::prepare_plan_with_banks(
                            inspection, &plan, rank, std::time::Duration::from_secs(10),
                            independent_banks.then(ParameterBankLoadOptions::default),
                        ).unwrap();
                        assert!(sources.selected().execution().text_realization().parameters().iter().any(|parameter|
                            matches!(parameter.lowering(), eredu_runtime::WeightLoweringKind::Transform)));
                        let mut executable = if sparse {
                            partitioned_adapter::routed(sources, &context, Arc::new(AtomicUsize::new(0)), None)
                        } else {
                            partitioned_adapter::dense(sources, &context)
                        }.unwrap();
                        for (step, tokens) in [tokens.clone(), NumericTensor::token_ids(&[4]), NumericTensor::token_ids(&[5])].iter().enumerate() {
                            let actual = executable.forward(tokens, step == 0).unwrap();
                            assert_tensor_close(&actual, &expected[step], "transformed V3 partition");
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

fn verify_v3_partition_components(sparse: bool, topologies: &[ParallelTopology]) {
    for query_rank in [None, Some(3)] {
        let mut config = v3_config(query_rank);
        config["num_hidden_layers"] = 2.into();
        config["first_k_dense_replace"] = if sparse { 1 } else { 2 }.into();
        let args = deepseek::parse_v3_config(&config).unwrap();
        let (root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let reference_sources = prepared_adapter::prepare(
            &inspection,
            &prepared_adapter::plan(None),
            &prepared_adapter::NumericPreparationProvider { addressable: false },
        )
        .unwrap();
        verify_latent_partition_ownership(&inspection, &args);
        let inputs = [
            NumericTensor::token_ids(&[1, 2, 5]),
            NumericTensor::token_ids(&[3]),
            NumericTensor::token_ids(&[4]),
        ];
        let ffn = if sparse {
            "model.layers.1.feed_forward.shared.units"
        } else {
            "model.layers.1.feed_forward.units"
        };
        let mut targets = [
            None,
            Some("readout.embedding"),
            Some("readout.normalized"),
            Some("readout.linear"),
            Some("model.layers.0.attention.channels"),
            Some("model.layers.0.feed_forward.units"),
            Some(ffn),
            Some("model.layers.0.attention.key_value.latent"),
        ]
        .map(|target| (target, vec![]))
        .to_vec();
        if query_rank.is_some() {
            targets.push((Some("model.layers.1.attention.query.latent"), vec![]));
        }
        if sparse {
            targets.extend([
                (Some("model.layers.1.feed_forward.shared.input"), vec![]),
                (Some("model.layers.1.feed_forward.shared.write"), vec![]),
                (Some("model.layers.1.feed_forward.shared.output"), vec![]),
            ]);
        }
        targets.push((
            None,
            vec![
                ("model.layers.0.attention.channels", vec![1, 4], true),
                (ffn, vec![1], true),
                ("model.layers.1.attention.key_value.latent", vec![2], true),
            ],
        ));
        let expected = targets
            .iter()
            .map(|(target, masks)| {
                let context = NumericContext {
                    bind_checkpoint_values: true,
                    ..Default::default()
                };
                let mut architecture =
                    deepseek::v3::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
                let hooks = <deepseek::v3::Model<NumericBackend> as LayeredArchitecture<
                    NumericBackend,
                    DeviceState<NumericBackend, NumericHybridLayerState>,
                >>::observation_hooks(&architecture);
                assert!(hooks.supports(eredu_runtime::inspection::ObservationHookSite::Input));
                assert!(hooks.supports(eredu_runtime::inspection::ObservationHookSite::Readout));
                for tensor_parallel in [false, true] {
                    let partition_hooks = <deepseek::v3::Model<NumericBackend> as eredu_runtime::PartitionedLayeredArchitecture<
                        NumericBackend, DeviceState<NumericBackend, NumericHybridLayerState>,
                    >>::partition_observation_hooks(&architecture, tensor_parallel);
                    assert!(partition_hooks.supports(eredu_runtime::inspection::ObservationHookSite::Input));
                    assert!(partition_hooks.supports(eredu_runtime::inspection::ObservationHookSite::Readout));
                    assert!(partition_hooks.supports(eredu_runtime::inspection::ObservationHookSite::Unit));
                }

                let mut units = (0..2)
                    .map(|index| architecture.construct_unit(0, index, &context).unwrap())
                    .collect::<Vec<_>>();
                payload::bind(
                    &mut architecture,
                    &mut units,
                    reference_sources
                        .selected()
                        .execution()
                        .text_realization()
                        .materialization_tasks(),
                    &[],
                    reference_sources.target().as_ref(),
                    &context,
                )
                .unwrap();
                let mut runtime =
                    LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                let mut state = DeviceState::<NumericBackend, _>::create(
                    deepseek::v3::state_layout(&args).unwrap(),
                    |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                )
                .unwrap();
                inputs
                    .iter()
                    .enumerate()
                    .map(|(step, tokens)| {
                        let mut observer = GlobalComponentObserver {
                            inner: NumericLifecycleObserver {
                                zero_path: target.map(str::to_owned),
                                ..Default::default()
                            },
                            values: BTreeMap::new(),
                            layout: None,
                            masks,
                            position: usize::from(step == 0),
                            projection_plan: None,
                            producer_receipts: None,
                            prediction: step as u64,
                        };
                        let output = runtime
                            .forward_with_observer(
                                deepseek::mtp::EmbeddedInput::target(tokens, None),
                                &mut state,
                                &context,
                                &mut observer,
                            )
                            .unwrap();
                        let output =
                            eredu_runtime::observe_model_logits(&mut observer, &output).unwrap();
                        let vocabulary = *output.shape.last().unwrap() as usize;
                        (
                            NumericTensor::new(
                                [1, 1, vocabulary as i32],
                                output.data[output.data.len() - vocabulary..].to_vec(),
                            ),
                            observer.values,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for trial in &expected[1..] {
            assert!(
                trial.iter().zip(&expected[0]).any(|(a, b)| a
                    .0
                    .data
                    .iter()
                    .zip(&b.0.data)
                    .any(|(a, b)| (a - b).abs() > 1e-5)),
                "nonzero pipeline experiment"
            );
        }
        verify_prepared_component_execution_with_topologies(
            &inspection,
            deepseek::parallel::v3_parameter_description(&args).unwrap(),
            &inputs,
            &targets,
            &expected,
            if sparse {
                |sources, context| {
                    partitioned_adapter::routed(
                        sources,
                        context,
                        Arc::new(AtomicUsize::new(0)),
                        None,
                    )
                }
            } else {
                partitioned_adapter::dense
            },
            topologies,
        );
    }
}

fn verify_latent_partition_ownership(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    args: &deepseek::V3Args,
) {
    use eredu_architectures::component_partition::ComponentCaptureProjectionRequest;
    let parameters = deepseek::parallel::v3_parameter_description(args).unwrap();
    for topology in [
        ParallelTopology::new(2, 1, 1, 1).unwrap(),
        ParallelTopology::new(2, 2, 1, 1).unwrap(),
    ] {
        let sources = partitioned_adapter::prepare_plan(
            inspection,
            &prepared_adapter::plan(None).with_topology(topology),
            0,
            std::time::Duration::from_secs(10),
        )
        .unwrap();
        let descriptor = sources.architecture().architecture_descriptor();
        let selected = sources.selected().execution();
        if !args.has_sparse_moe_layers() {
            assert!(
                selected
                    .text_realization()
                    .requirements()
                    .grouped_operations()
                    .is_empty(),
                "dense partition selection cannot require expert mechanisms"
            );
        }
        let layouts = selected
            .component_partition_layouts(&descriptor, &parameters, topology.world_size())
            .unwrap()
            .unwrap();
        let capture = component_partition_capture_plan(&descriptor);
        for component in &descriptor.components {
            for path in component.write_output.iter().chain(component.output.iter()) {
                use eredu_core::{
                    capture::PartitionCaptureCombination, component::ComponentWritePartition,
                };
                let expected = match component.write_partition {
                    ComponentWritePartition::Complete => PartitionCaptureCombination::Disjoint,
                    ComponentWritePartition::TensorParallelSum => {
                        PartitionCaptureCombination::SumF64ToF32
                    }
                };
                for path in [path.clone(), format!("{path}.effective")] {
                    assert_eq!(layouts.observation_combination(&path).unwrap(), expected);
                    let index = capture
                        .plan()
                        .selections
                        .iter()
                        .position(|selection| selection.path == path)
                        .unwrap();
                    let producers = layouts
                        .capture_producers(ComponentCaptureProjectionRequest {
                            invocation: None,
                            plan: &capture,
                            selection_index: index,
                            phase: eredu_core::capture::CapturePhase::Prefill,
                            prediction: 0,
                            max_producers: 4,
                            max_fragments: 4,
                        })
                        .unwrap();
                    assert_eq!(
                        producers.len(),
                        if expected == PartitionCaptureCombination::SumF64ToF32 {
                            2
                        } else {
                            1
                        }
                    );
                }
            }
            for stage in component
                .reads
                .iter()
                .flat_map(|read| &read.input_projections)
            {
                let members = layouts.capture_hook_members(&stage.output).unwrap();
                assert_eq!(
                    members.len(),
                    2,
                    "replicated latent on both executing tensor ranks"
                );
                for rank in 0..topology.world_size() {
                    let layout = layouts.rank(rank).unwrap();
                    let original = stage.output.strip_suffix(".effective").unwrap();
                    let point = layout.observation(&stage.output).unwrap();
                    assert_eq!(layout.observation(original), Some(point));
                    assert_eq!(
                        point.coordinates().is_some(),
                        layout.group(&component.id).unwrap().coordinates().is_some()
                    );
                    if let Some(map) = point.coordinates() {
                        assert_eq!(map.contiguous_range(), Some(0..stage.rows.len()));
                    }
                }
                let index = capture
                    .plan()
                    .selections
                    .iter()
                    .position(|selection| selection.path == stage.output)
                    .unwrap();
                let producers = layouts
                    .capture_producers(ComponentCaptureProjectionRequest {
                        invocation: None,
                        plan: &capture,
                        selection_index: index,
                        phase: eredu_core::capture::CapturePhase::Prefill,
                        prediction: 0,
                        max_producers: 4,
                        max_fragments: 4,
                    })
                    .unwrap();
                assert_eq!(
                    producers.len(),
                    1,
                    "one publisher for complete replicated latent"
                );
            }
        }
        let mut aliases = descriptor.clone();
        for component in &mut aliases.components {
            for stage in component
                .reads
                .iter_mut()
                .flat_map(|read| &mut read.input_projections)
            {
                stage.shared_weight = "independent.shared.storage.alias".into();
            }
        }
        assert_eq!(
            selected
                .component_partition_layouts(&aliases, &parameters, topology.world_size())
                .unwrap()
                .unwrap(),
            layouts,
            "source storage identity cannot transfer latent invocation ownership"
        );
        let mut invalid = descriptor.clone();
        invalid
            .components
            .iter_mut()
            .flat_map(|group| &mut group.reads)
            .flat_map(|read| &mut read.input_projections)
            .next()
            .unwrap()
            .rows
            .end += 1;
        assert!(
            selected
                .component_partition_layouts(&invalid, &parameters, topology.world_size())
                .is_err(),
            "stage rows must agree with declared observation extent"
        );
    }
}

#[derive(Default)]
struct SharedCapture(Components);
impl ActivationObserver<NumericTensor, Error> for SharedCapture {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.0.observe(path, value)
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        self.0.intervene(path, value)
    }
    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        self.0
            .values
            .insert("routed".into(), routing.routed_output.clone());
        self.0
            .values
            .insert("shared".into(), routing.shared_output.unwrap().clone());
        self.0
            .values
            .insert("combined".into(), routing.combined_output.unwrap().clone());
        Ok(())
    }
}

#[test]
fn v3_shared_units_recompute_declared_reads_and_reconstruct_the_sparse_sum() {
    let context = NumericContext::default();
    let inputs = [3, 1, 1].map(|sequence| {
        NumericTensor::new(
            [1, sequence, 8],
            (0..sequence * 8)
                .map(|i| (i as f32 * 0.31 - 0.4).sin())
                .collect(),
        )
    });
    for shared_count in [1, 2] {
        let mut config = v3_config(Some(3));
        config["num_hidden_layers"] = 2.into();
        config["n_shared_experts"] = shared_count.into();
        let args = deepseek::parse_v3_config(&config).unwrap();
        let descriptor = describe(&config);
        let group = descriptor
            .components
            .iter()
            .find(|group| group.node_id.ends_with(".shared"))
            .unwrap();
        assert_eq!(group.count, shared_count as usize * 4);
        let block = deepseek::block::V3Block::<NumericBackend>::new(&args, 1, &context).unwrap();
        let mut parameters = Parameters::default();
        block.visit_parameters(&mut parameters);
        for paged in [false, true] {
            let cache = || {
                if paged {
                    NumericCompressedCache::paged(2)
                } else {
                    NumericCompressedCache::resident()
                }
            };
            let mut baseline = Vec::new();
            for trial in 0..5 {
                let mut candidate = block.clone();
                let mut state = cache();
                let mut ordinary = block.clone();
                let mut ordinary_state = cache();
                for (step, input) in inputs.iter().enumerate() {
                    let row = input.shape[1] as usize - 1;
                    let mut capture = SharedCapture::default();
                    match trial {
                        1 => {
                            capture.0.zero =
                                Some(("model.layers.1.feed_forward.shared.units", row, 1))
                        }
                        2 => {
                            capture.0.keep =
                                Some(("model.layers.1.feed_forward.shared.units", row, 1))
                        }
                        3 => {
                            capture.0.zero =
                                Some(("model.layers.1.feed_forward.shared.input", row, 0))
                        }
                        4 => {
                            capture.0.zero = Some(("model.layers.1.attention.channels", row, 2));
                            capture.0.keep =
                                Some(("model.layers.1.feed_forward.shared.units", row, 1));
                        }
                        _ => {}
                    }
                    let output = candidate
                        .forward_observed(
                            "model.layers.1",
                            input,
                            None,
                            Some(&mut state),
                            &context,
                            &mut capture,
                        )
                        .unwrap();
                    if trial == 0 {
                        assert_tensor_exact(
                            &output,
                            &ordinary
                                .forward(input, None, Some(&mut ordinary_state), &context)
                                .unwrap(),
                            "shared no-op transparency",
                        );
                        baseline.push((
                            output.clone(),
                            capture.0.values[group.activation.as_str()].clone(),
                        ));
                    } else {
                        assert_ne!(output.data, baseline[step].0.data);
                    }
                    let values = &capture.0.values;
                    let normalized = &values[&format!("{}.effective", group.input)];
                    let gate =
                        linear(normalized, &parameters.0[&group.reads[0].weight], None).unwrap();
                    let up =
                        linear(normalized, &parameters.0[&group.reads[1].weight], None).unwrap();
                    let expected = NumericTensor::new(
                        gate.shape.clone(),
                        gate.data
                            .iter()
                            .zip(&up.data)
                            .map(|(g, u)| {
                                let g = f64::from(*g);
                                (g / (1.0 + (-g).exp()) * f64::from(*u)) as f32
                            })
                            .collect(),
                    );
                    assert_tensor_close(
                        &expected,
                        &values[&group.activation],
                        "shared reads recompute current normalized input",
                    );
                    if trial == 3 || trial == 4 {
                        assert_ne!(values[&group.activation].data, baseline[step].1.data);
                    }
                    let units = &values[&group.effective_activation];
                    if [1, 2, 4].contains(&trial) {
                        for (i, (before, after)) in values[&group.activation]
                            .data
                            .iter()
                            .zip(&units.data)
                            .enumerate()
                        {
                            let zero =
                                i / group.count == row && ((i % group.count == 1) != (trial != 1));
                            assert_eq!(*after, if zero { 0.0 } else { *before });
                        }
                    }
                    assert_tensor_exact(
                        units,
                        &values[group.write_input.as_ref().unwrap()],
                        "actual shared down input",
                    );
                    let down = &parameters.0[&group.write_weight];
                    let write = NumericTensor::new(
                        [1, input.shape[1], 8],
                        units
                            .data
                            .chunks_exact(group.count)
                            .flat_map(|row| {
                                down.data.chunks_exact(group.count).map(move |weight| {
                                    row.iter()
                                        .zip(weight)
                                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                                        .sum::<f64>() as f32
                                })
                            })
                            .collect(),
                    );
                    assert_tensor_close(
                        &write,
                        &values[group.write_output.as_ref().unwrap()],
                        "shared signed write reconstruction",
                    );
                    assert_tensor_exact(
                        &values["shared"],
                        &values[&format!("{}.effective", group.output.as_ref().unwrap())],
                        "effective shared output feeds sparse sum",
                    );
                    assert_tensor_exact(
                        &values["routed"].add(&values["shared"], &context).unwrap(),
                        &values["combined"],
                        "sparse write counts shared constituent once",
                    );
                    assert_tensor_exact(
                        &values["combined"],
                        &values["model.layers.1.feed_forward.contribution"],
                        "complete residual term retains shared contribution",
                    );
                }
            }
        }
    }
}

#[test]
fn shared_expert_instrumentation_preserves_bounded_gating_and_identity_parallel_reduction() {
    let context = NumericContext::default();
    let mut config = v3_config(None);
    config["num_hidden_layers"] = 2.into();
    let args = deepseek::parse_v3_config(&config).unwrap();
    let mut policy = deepseek::v3::moe_policy(&args, 1).unwrap();
    let bound = 0.02_f32;
    policy.shared_limit = Some(eredu_nn::GatedProductPolicy::bounded_silu(bound).unwrap());
    let block = deepseek::moe::RoutedPlusShared::<NumericBackend>::new(&policy, &context).unwrap();
    let mut parameters = Parameters::default();
    block.visit_parameters(&mut parameters);
    let input = NumericTensor::new(
        [1, 3, 8],
        (0..24).map(|i| 100.0 * (i as f32 * 0.7).sin()).collect(),
    );
    let mut capture = SharedCapture::default();
    let observed = block
        .clone()
        .forward_with_provider_observed(
            "sparse",
            &input,
            deepseek::moe::RouteSource::Learned,
            eredu_runtime::ExpertPass::Prefill,
            &mut eredu_runtime::ResidentExpertProvider,
            &context,
            &mut capture,
        )
        .unwrap();
    let ordinary = block
        .clone()
        .forward(&input, deepseek::moe::RouteSource::Learned, &context)
        .unwrap();
    assert_tensor_exact(&observed, &ordinary, "bounded shared observer transparency");
    let parallel = block
        .clone()
        .forward_tensor_parallel_with_provider(
            &input,
            deepseek::moe::RouteSource::Learned,
            eredu_runtime::ExpertPass::Prefill,
            &mut eredu_runtime::ResidentExpertProvider,
            &context,
            |value, _| Ok(value),
        )
        .unwrap();
    assert_tensor_close(&observed, &parallel, "bounded shared identity TP reduction");
    let gate = linear(&input, &parameters.0[&policy.shared_gate], None).unwrap();
    let up = linear(&input, &parameters.0[&policy.shared_up], None).unwrap();
    assert!(gate.data.iter().any(|g| *g > bound));
    assert!(up.data.iter().any(|u| u.abs() > bound));
    let expected = NumericTensor::new(
        gate.shape,
        gate.data
            .iter()
            .zip(&up.data)
            .map(|(g, u)| {
                let g = f64::from(g.min(bound));
                (g / (1.0 + (-g).exp()) * f64::from(u.clamp(-bound, bound))) as f32
            })
            .collect(),
    );
    assert_tensor_close(
        &expected,
        &capture.0.values["sparse.shared.units"],
        "bounded shared equation",
    );
}

#[test]
fn v3_declared_latent_reads_match_current_values_and_interventions_reach_the_cache() {
    let context = NumericContext::default();
    let config = v3_config(Some(3));
    let args = deepseek::parse_v3_config(&config).unwrap();
    let descriptor = describe(&config);
    let group = descriptor
        .components
        .iter()
        .find(|g| g.activation.ends_with("attention.channels"))
        .unwrap();
    let block = deepseek::block::DenseV3Block::<NumericBackend>::new(&args, 0, &context).unwrap();
    let mut parameters = Parameters::default();
    block.visit_parameters(&mut parameters);
    let input = NumericTensor::new(
        [1, 3, 8],
        (0..24).map(|i| (i as f32 * 0.3 - 1.1).sin()).collect(),
    );
    for selected in [
        None,
        Some("model.layers.0.attention.query.latent"),
        Some("model.layers.0.attention.key_value.latent"),
    ] {
        let mut state = NumericCompressedCache::resident();
        let mut capture = Components {
            zero: selected.map(|p| (p, 2, 1)),
            ..Default::default()
        };
        block
            .clone()
            .forward_instrumented(
                &input,
                None,
                Some(&mut state),
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut capture),
            )
            .unwrap();
        for read in &group.reads {
            let mut input = capture.values[&format!("{}.effective", group.input)].clone();
            for stage in &read.input_projections {
                let weight =
                    parameters.0[&stage.weight].axis_slice(0, stage.rows.start, stage.rows.end);
                let projected = linear(&input, &weight, None).unwrap();
                let normalization = stage.normalization.as_ref().unwrap();
                let gain = &parameters.0[normalization.gain.as_ref().unwrap()];
                let mut expected = projected.clone();
                for row in expected.data.chunks_exact_mut(stage.rows.len()) {
                    let denominator = (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
                        / row.len() as f64
                        + f64::from(normalization.epsilon.value()))
                    .sqrt();
                    for (x, gain) in row.iter_mut().zip(&gain.data) {
                        *x = (f64::from(*x) * f64::from(*gain) / denominator) as f32;
                    }
                }
                let original = stage.output.strip_suffix(".effective").unwrap();
                assert_tensor_close(
                    &expected,
                    &capture.values[original],
                    "declared latent read stage",
                );
                input = capture.values[&stage.output].clone();
                assert_eq!(input.shape[2] as usize, stage.rows.len());
            }
            let rows = read.rows.row_range(4).unwrap();
            let weight = parameters.0[&read.weight].axis_slice(0, rows.start, rows.end);
            assert_eq!(
                weight.shape[1], input.shape[2],
                "terminal read consumes the declared stage"
            );
            let value = linear(&input, &weight, None).unwrap();
            assert!(value.data.iter().any(|x| x.abs() > 1e-5));
        }
        assert_tensor_exact(
            &state.state.as_ref().unwrap().latent,
            &capture.values["model.layers.0.attention.key_value.latent.effective"],
            "cache consumes effective latent values, not reconstructed diagnostics",
        );
        if let Some(path) = selected {
            assert_eq!(
                capture.values[&format!("{path}.effective")].data[2 * 3 + 1],
                0.0
            );
        }
    }
}

#[test]
fn v3_whole_model_declared_terms_reconstruct_scores_and_preserve_cached_observation() {
    let context = NumericContext::default();
    let mut config = v3_config(Some(3));
    config["num_hidden_layers"] = 2.into();
    let args = deepseek::parse_v3_config(&config).unwrap();
    let descriptor = describe(&config);
    let readout = descriptor.component_readout.as_ref().unwrap();
    let model = || deepseek::v3::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
    let state = || {
        DeviceState::<NumericBackend, _>::create(
            deepseek::v3::state_layout(&args).unwrap(),
            |_, _| Ok::<_, Error>(NumericCompressedCache::resident()),
        )
        .unwrap()
    };
    let runtime = || {
        let architecture = model();
        let units = (0..2)
            .map(|i| architecture.construct_unit(0, i, &context).unwrap())
            .collect();
        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units))
    };
    let mut ordinary = runtime();
    let mut ordinary_state = state();
    let mut baseline = Vec::new();
    for masked in [false, true] {
        let mut runtime = runtime();
        let mut state = state();
        for (step, ids) in [vec![1, 2, 5], vec![3], vec![4]].into_iter().enumerate() {
            let tokens = NumericTensor::token_ids(&ids);
            let input = || deepseek::mtp::EmbeddedInput::target(&tokens, None);
            let mut capture = Components {
                zero: masked.then_some(("model.layers.0.attention.channels", ids.len() - 1, 4)),
                keep: masked.then_some(("model.layers.1.attention.channels", ids.len() - 1, 2)),
                ..Default::default()
            };
            let logits = runtime
                .forward_with_observer(input(), &mut state, &context, &mut capture)
                .unwrap();
            if !masked {
                let expected = ordinary
                    .forward(input(), &mut ordinary_state, &context)
                    .unwrap();
                assert_tensor_exact(
                    &logits,
                    &expected,
                    "observed target model uses ordinary equations",
                );
                baseline.push(logits.clone());
            } else {
                assert_ne!(logits.data, baseline[step].data);
            }
            let mut residual = capture.values[&format!("{}.effective", readout.embedding)].clone();
            for layer in 0..2 {
                for suffix in [
                    "compressed_attention.output.effective",
                    if layer == 0 {
                        "feed_forward.output.effective"
                    } else {
                        "feed_forward.contribution.effective"
                    },
                ] {
                    residual = residual
                        .add(
                            &capture.values[&format!("model.layers.{layer}.{suffix}")],
                            &context,
                        )
                        .unwrap();
                }
            }
            assert_tensor_exact(
                &residual,
                &capture.values[&format!("{}.effective", readout.residual)],
                "declared embedding and writes reconstruct residual",
            );
            for whole in &readout.other_writes {
                assert!(descriptor
                    .observations
                    .get(&whole.effective_output)
                    .is_some());
                assert!(capture.values.contains_key(&whole.effective_output));
            }
            let modules = model();
            let gain = &modules.static_modules().norm.weight;
            let head = &modules.static_modules().lm_head.as_ref().unwrap().weight;
            let width = args.hidden_size as usize;
            for (row, output) in residual
                .data
                .chunks_exact(width)
                .zip(logits.data.chunks_exact(args.vocab_size as usize))
            {
                let denominator = (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
                    / width as f64
                    + f64::from(readout.normalization.epsilon.value()))
                .sqrt();
                let score = |token: usize| {
                    row.iter()
                        .zip(&gain.data)
                        .zip(&head.data[token * width..(token + 1) * width])
                        .map(|((x, g), w)| {
                            f64::from(*x) * f64::from(*g) * f64::from(*w) / denominator
                        })
                        .sum::<f64>()
                };
                assert!((score(3) - f64::from(output[3])).abs() < 2e-4);
                assert!((score(3) - score(7) - f64::from(output[3] - output[7])).abs() < 2e-4);
            }
        }
    }
}

fn assert_write(
    capture: &Components,
    parameters: &Parameters,
    units: &str,
    write: &str,
    weight: &str,
) {
    let units = &capture.values[&format!("model.layers.0.{units}")];
    let weight = &parameters.0[weight];
    let width = weight.shape[1] as usize;
    let hidden = weight.shape[0] as usize;
    let mut values = Vec::new();
    for row in units.data.chunks_exact(width) {
        for weights in weight.data.chunks_exact(width) {
            values.push(
                row.iter()
                    .zip(weights)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>() as f32,
            );
        }
    }
    let expected = NumericTensor::new([1, units.shape[1], hidden as i32], values);
    assert_tensor_close(
        &expected,
        &capture.values[&format!("model.layers.0.{write}")],
        "MLA/dense signed write reconstruction",
    );
}

#[test]
fn v3_rotary_positions_are_independent_of_heads_and_prefill_chunking() {
    struct IdenticalHeads;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for IdenticalHeads {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let name = metadata.id().as_str();
            if value.shape.len() == 2 {
                for value in &mut value.data {
                    *value *= 6.0;
                }
            }
            if name.ends_with("q_proj.weight")
                || name.ends_with("q_b_proj.weight")
                || name.ends_with("kv_b_proj.weight")
            {
                let half = value.data.len() / 2;
                let (first, second) = value.data.split_at_mut(half);
                second.copy_from_slice(first);
            }
        }
    }
    let context = NumericContext::default();
    let input = NumericTensor::new(
        [1, 5, 8],
        (0..40).map(|i| (i as f32 * 0.7 - 1.4).sin()).collect(),
    );
    for query_rank in [None, Some(3)] {
        let mut attention = deepseek::attention::v3::Attention::<NumericBackend>::new(
            &args(query_rank),
            0,
            &context,
        )
        .unwrap();
        attention.visit_parameters_mut(&mut IdenticalHeads);
        for paged in [false, true] {
            let cache = if paged {
                NumericCompressedCache::paged(2)
            } else {
                NumericCompressedCache::resident()
            };
            let mut whole = attention.clone();
            let mut whole_cache = cache.clone();
            let mut full = Components::default();
            whole
                .forward_instrumented(
                    &input,
                    None,
                    Some(&mut whole_cache),
                    &context,
                    &mut ComponentInstrumentation::new("layer", &mut full),
                )
                .unwrap();
            let values = &full.values["layer.attention.channels"];
            let width = values.shape[2] as usize;
            for row in values.data.chunks_exact(width) {
                for (first, second) in row[..width / 2].iter().zip(&row[width / 2..]) {
                    assert!(
                        (first - second).abs() < 1e-6,
                        "identical heads must use the same token position: {first} vs {second}"
                    );
                }
            }
            let mut chunked = attention.clone();
            let mut chunked_cache = cache;
            for position in 0..5 {
                let mut capture = Components::default();
                chunked
                    .forward_instrumented(
                        &input.axis_slice(1, position, position + 1),
                        None,
                        Some(&mut chunked_cache),
                        &context,
                        &mut ComponentInstrumentation::new("layer", &mut capture),
                    )
                    .unwrap();
                let expected = values.axis_slice(1, position, position + 1);
                let actual = &capture.values["layer.attention.channels"];
                assert_eq!(actual.shape, expected.shape);
                for (actual, expected) in actual.data.iter().zip(&expected.data) {
                    assert!((actual - expected).abs() < 2e-6,
                        "MLA token {position} must be independent of prefill chunking: {actual} vs {expected}");
                }
            }
        }
    }
}

#[test]
fn v3_parallel_observation_preserves_ordinary_reductions_and_cached_values() {
    let context = NumericContext::default();
    for (query_rank, sparse) in [None, Some(3)]
        .into_iter()
        .flat_map(|rank| [false, true].map(|sparse| (rank, sparse)))
    {
        for paged in [false, true] {
            let mut config = v3_config(query_rank);
            config["first_k_dense_replace"] = if sparse { 0 } else { 1 }.into();
            let args = deepseek::parse_v3_config(&config).unwrap();
            let mut ordinary =
                deepseek::block::V3Block::<NumericBackend>::new(&args, 0, &context).unwrap();
            let mut observed = ordinary.clone();
            let mut ordinary_cache = if paged {
                NumericCompressedCache::paged(2)
            } else {
                NumericCompressedCache::resident()
            };
            let mut observed_cache = ordinary_cache.clone();
            for (step, length) in [3, 1, 1].into_iter().enumerate() {
                let input = NumericTensor::new(
                    [1, length, 8],
                    (0..length * 8)
                        .map(|i| (i as f32 * 0.4 + step as f32 - 2.0).sin())
                        .collect(),
                );
                let mut ordinary_reductions = 0;
                let expected = ordinary
                    .forward_parallel(
                        &input,
                        None,
                        Some(&mut ordinary_cache),
                        &context,
                        |value, _| {
                            ordinary_reductions += 1;
                            Ok(value)
                        },
                    )
                    .unwrap();
                let mut observed_reductions = 0;
                let mut capture = Components::default();
                let actual = observed
                    .forward_parallel_observed(
                        "model.layers.0",
                        &input,
                        None,
                        Some(&mut observed_cache),
                        &context,
                        &mut capture,
                        |value, _| {
                            observed_reductions += 1;
                            Ok(value)
                        },
                    )
                    .unwrap();
                assert_eq!(ordinary_reductions, 2);
                assert_eq!(observed_reductions, ordinary_reductions);
                assert_tensor_exact(&actual, &expected, "resident TP observation transparency");
                for path in [
                    "attention.channels",
                    if sparse {
                        "feed_forward.shared.units"
                    } else {
                        "feed_forward.units"
                    },
                ] {
                    assert!(capture.values[&format!("model.layers.0.{path}")]
                        .data
                        .iter()
                        .any(|value| value.abs() > 1e-5));
                }
            }
        }
    }
}

#[test]
fn v3_component_writes_masks_and_survivors_match_across_compressed_cache_modes() {
    let context = NumericContext::default();
    let inputs = [
        NumericTensor::new(
            [1, 3, 8],
            (0..24).map(|i| (i as f32 * 0.4 - 2.0).sin()).collect(),
        ),
        NumericTensor::new(
            [1, 1, 8],
            (0..8).map(|i| (i as f32 * 0.7 + 0.3).cos()).collect(),
        ),
        NumericTensor::new(
            [1, 1, 8],
            (0..8).map(|i| (i as f32 * 0.5 - 0.8).sin()).collect(),
        ),
    ];
    for query_rank in [None, Some(3)] {
        let args = args(query_rank);
        let block =
            deepseek::block::DenseV3Block::<NumericBackend>::new(&args, 0, &context).unwrap();
        let general = deepseek::block::V3Block::<NumericBackend>::new(&args, 0, &context).unwrap();
        let mut parameters = Parameters::default();
        block.visit_parameters(&mut parameters);
        let mut cache_results = Vec::new();
        for paged in [false, true] {
            let cache = if paged {
                NumericCompressedCache::paged(2)
            } else {
                NumericCompressedCache::resident()
            };
            let mut mode_results = Vec::new();
            // Every trial starts before prefill. No-op, single deletion, keep-only,
            // and a dense-unit edit all use independently owned cache state.
            for trial in 0..4 {
                let mut candidate = block.clone();
                let mut general = general.clone();
                let mut ordinary = block.clone();
                let mut state = cache.clone();
                let mut general_state = cache.clone();
                let mut ordinary_state = cache.clone();
                for (prediction, input) in inputs.iter().enumerate() {
                    let row = input.shape[1] as usize - 1;
                    let mut capture = Components::default();
                    match trial {
                        1 => capture.zero = Some(("model.layers.0.attention.channels", row, 4)),
                        2 => capture.keep = Some(("model.layers.0.attention.channels", row, 4)),
                        3 => capture.zero = Some(("model.layers.0.feed_forward.units", row, 3)),
                        _ => {}
                    }
                    let mut general_capture = Components {
                        zero: capture.zero,
                        keep: capture.keep,
                        ..Default::default()
                    };
                    let mut baseline = Components::default();
                    let expected = ordinary
                        .forward_instrumented(
                            input,
                            None,
                            Some(&mut ordinary_state),
                            &context,
                            &mut ComponentInstrumentation::new("model.layers.0", &mut baseline),
                        )
                        .unwrap();
                    let result = candidate
                        .forward_instrumented(
                            input,
                            None,
                            Some(&mut state),
                            &context,
                            &mut ComponentInstrumentation::new("model.layers.0", &mut capture),
                        )
                        .unwrap();
                    let general_result = general
                        .forward_observed(
                            "model.layers.0",
                            input,
                            None,
                            Some(&mut general_state),
                            &context,
                            &mut general_capture,
                        )
                        .unwrap();
                    assert_tensor_exact(
                        &result,
                        &general_result,
                        "dense/routed-capable V3 block shares observed equations",
                    );
                    for (path, value) in &capture.values {
                        assert_tensor_exact(value, &general_capture.values[path], path);
                    }
                    if trial == 0 {
                        assert_tensor_exact(&expected, &result, "MLA observation transparency");
                        let mut ordinary_cache = if prediction == 0 {
                            cache.clone()
                        } else {
                            let mut previous = cache.clone();
                            let mut unobserved = block.clone();
                            for earlier in &inputs[..prediction] {
                                unobserved
                                    .forward(earlier, None, Some(&mut previous), &context)
                                    .unwrap();
                            }
                            previous
                        };
                        let unobserved = block
                            .clone()
                            .forward(input, None, Some(&mut ordinary_cache), &context)
                            .unwrap();
                        assert_tensor_exact(
                            &unobserved,
                            &result,
                            "disabled instrumentation agrees with observed execution",
                        );
                    } else {
                        assert!(
                            result
                                .data
                                .iter()
                                .zip(&expected.data)
                                .any(|(a, b)| (a - b).abs() > 1e-6),
                            "nonzero intervention must affect output"
                        );
                    }
                    for (units, write, weight) in [
                        (
                            "attention.write_input",
                            "attention.write",
                            "model.layers.0.self_attn.o_proj.weight",
                        ),
                        (
                            "feed_forward.write_input",
                            "feed_forward.write",
                            "model.layers.0.mlp.down_proj.weight",
                        ),
                    ] {
                        assert_write(&capture, &parameters, units, write, weight);
                    }
                    let selected = capture.zero.or(capture.keep);
                    if let Some((path, row, component)) = selected {
                        let before = &capture.values[path];
                        let after = &capture.values[&format!("{path}.effective")];
                        let width = before.shape[2] as usize;
                        for (index, (a, b)) in before.data.iter().zip(&after.data).enumerate() {
                            let zero = index / width == row
                                && ((index % width == component) != capture.keep.is_some());
                            assert_eq!(*b, if zero { 0.0 } else { *a });
                        }
                    }
                    if trial == 1 || trial == 2 {
                        let effective_input =
                            &capture.values["model.layers.0.feed_forward.input.effective"];
                        let baseline_input =
                            &baseline.values["model.layers.0.feed_forward.input.effective"];
                        assert_ne!(
                            effective_input.data, baseline_input.data,
                            "surviving FFN must read changed residual"
                        );
                        let gate = linear(
                            effective_input,
                            &parameters.0["model.layers.0.mlp.gate_proj.weight"],
                            None,
                        )
                        .unwrap();
                        let up = linear(
                            effective_input,
                            &parameters.0["model.layers.0.mlp.up_proj.weight"],
                            None,
                        )
                        .unwrap();
                        let recomputed = NumericTensor::new(
                            gate.shape.clone(),
                            gate.data
                                .iter()
                                .zip(&up.data)
                                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                                .collect(),
                        );
                        assert_tensor_close(
                            &recomputed,
                            &capture.values["model.layers.0.feed_forward.units"],
                            "survivors recompute gate and value reads",
                        );
                    }
                    assert_eq!(
                        state.offset(),
                        if prediction == 0 {
                            3
                        } else {
                            3 + prediction as i32
                        }
                    );
                    mode_results.push(result);
                }
            }
            cache_results.push(mode_results);
        }
        for (resident, paged) in cache_results[0].iter().zip(&cache_results[1]) {
            assert_tensor_close(
                resident,
                paged,
                "resident/paged compressed component execution",
            );
        }
    }
}
