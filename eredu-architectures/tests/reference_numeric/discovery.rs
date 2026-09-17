//! Discovery conformance against the existing scalar executable fixtures.
use super::*;
use eredu_core::{
    ArchitectureDescriptor, ObservationRequest, ObservationSelector, SymbolicDimension,
};

struct Capture {
    request: ObservationRequest,
    values: BTreeMap<String, NumericTensor>,
}

impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Capture {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if self.request.matches(path) {
            assert!(
                self.values.insert(path.into(), value.clone()).is_none(),
                "duplicate capture {path}"
            );
        }
        Ok(())
    }
    fn observe_routing(
        &mut self,
        event: eredu_runtime::RoutingObservation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        event.for_each_tensor(|path, value| {
            self.observe(&path, value).unwrap();
        });
        Ok(())
    }
}

fn descriptor(config: &serde_json::Value) -> ArchitectureDescriptor {
    eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor()
}

#[test]
fn k2_components_declare_real_routed_reads_and_capture_shared_and_sparse_units() {
    use eredu_architectures::k2_horizon as family;
    use eredu_core::{component::*, parameters::*};
    struct Observe {
        capture: Capture,
        units: BTreeMap<String, super::routed_units::Capture>,
        zero: Option<&'static str>,
    }
    impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observe {
        fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
            self.capture.observe(path, value)
        }
        fn observe_routing(
            &mut self,
            event: eredu_runtime::RoutingObservation<'_, NumericTensor>,
        ) -> Result<(), Error> {
            self.capture.observe_routing(event)
        }
        fn intervene(
            &mut self,
            path: &str,
            value: &NumericTensor,
        ) -> Result<Option<NumericTensor>, Error> {
            if self.zero != Some(path) {
                return Ok(None);
            }
            let mut changed = value.clone();
            let width = *value.shape.last().unwrap() as usize;
            changed.data[value.data.len() - width] = 0.;
            Ok(Some(changed))
        }
        fn routed_unit_observer(
            &mut self,
            path: &str,
        ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error>
        {
            Ok(Some(self.units.entry(path.into()).or_default()))
        }
    }
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/k2_horizon/reference.json")).unwrap();
    for case in ["dense", "moe", "mova"] {
        let config = &fixture[case]["config"];
        let graph = descriptor(config);
        let roundtrip: ArchitectureDescriptor =
            serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
        assert_eq!(graph, roundtrip);
        let args = family::model_args_from_config_value(config).unwrap();
        assert_eq!(
            graph.routed_components.len(),
            if case == "dense" { 0 } else { 2 }
        );
        let context = NumericContext::default();
        let mut baseline: Vec<Vec<f32>> = Vec::new();
        let targets: &[Option<&str>] = if case == "dense" {
            &[None]
        } else {
            &[
                None,
                Some("model.layers.1.shared.feed_forward.units"),
                Some("model.layers.1.attention.channels"),
            ]
        };
        for (zero, provider) in targets
            .iter()
            .flat_map(|zero| [(*zero, false), (*zero, true)])
        {
            let architecture =
                family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            let mut state = DeviceState::<NumericBackend, _>::create(
                architecture.state_layout().unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            let units: Vec<_> = (0..3)
                .map(|i| architecture.construct_unit(i, &context).unwrap())
                .collect();
            let mut weights = BTreeMap::new();
            struct Weights<'a>(&'a mut BTreeMap<String, NumericTensor>);
            impl<'a> ParameterVisitor<'a, NumericTensor> for Weights<'_> {
                fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
                    self.0.insert(metadata.id.to_string(), value.clone());
                }
            }
            for unit in &units {
                unit.visit_parameters(&mut Weights(&mut weights));
            }
            architecture
                .static_modules()
                .visit_parameters(&mut Weights(&mut weights));
            let loaded = ParameterDiscovery {
                identity: "numeric-loaded".into(),
                artifact_identity: "fixture".into(),
                overlay_identity: None,
                usage: Default::default(),
                coordination_usage: Default::default(),
                parameters: weights
                    .iter()
                    .map(|(id, value)| LoadedParameter {
                        id: id.clone(),
                        shared_id: id.clone(),
                        shape: value.shape.iter().map(|n| *n as u64).collect(),
                        dtype: Some(eredu_core::intervention::InterventionDtype::Float32),
                        supported: true,
                        access: None,
                        condition: "actual numeric parameter".into(),
                        input_transform: ProjectionInputTransform::Identity,
                    })
                    .collect(),
            };
            for group in &graph.components {
                for read in &group.reads {
                    assert!(weights.contains_key(&read.weight), "{}", read.weight);
                }
                for read in &group.routed_reads {
                    assert_eq!(case, "mova");
                    let component = ComponentId {
                        group: group.id.clone(),
                        index: group.count - 1,
                    };
                    let selected = read.read_weight(group, &component, 2, &loaded).unwrap();
                    assert_eq!(selected.region.starts, [2, 7, 0]);
                    assert_eq!(selected.region.shape, [1, 1, 8]);
                    assert!(read
                        .read_bias(group, &component, 2, &loaded)
                        .unwrap()
                        .is_none());
                    assert!(read
                        .read_weight(group, &component, read.expert_count, &loaded)
                        .is_err());
                    let mut foreign = component.clone();
                    foreign.group = "foreign".into();
                    assert!(read.read_weight(group, &foreign, 2, &loaded).is_err());
                    let mut malformed = loaded.clone();
                    malformed
                        .parameters
                        .iter_mut()
                        .find(|p| p.id == selected.parameter.id)
                        .unwrap()
                        .shape[0] += 1;
                    assert!(read.read_weight(group, &component, 2, &malformed).is_err());
                    assert!(weights.contains_key(&read.selector_weight));
                    assert!(read
                        .selector_correction_bias
                        .as_ref()
                        .is_some_and(|id| weights.contains_key(id)));
                }
            }
            let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
            for (step, tokens) in [vec![1, 3, 2], vec![4], vec![5]].into_iter().enumerate() {
                let input = NumericTensor::token_ids(&tokens);
                let mut observer = Observe {
                    capture: Capture {
                        request: ObservationRequest::all(),
                        values: BTreeMap::new(),
                    },
                    units: BTreeMap::new(),
                    zero,
                };
                let logits = if provider {
                    runtime
                        .forward_with_provider_and_observer(
                            decoder::LayeredInput {
                                tokens: &input,
                                mask: None,
                            },
                            &mut state,
                            if step == 0 {
                                ExpertPass::Prefill
                            } else {
                                ExpertPass::Decode
                            },
                            &mut eredu_runtime::ResidentExpertProvider,
                            &context,
                            &mut observer,
                        )
                        .unwrap()
                } else {
                    runtime
                        .forward_with_observer(
                            decoder::LayeredInput {
                                tokens: &input,
                                mask: None,
                            },
                            &mut state,
                            &context,
                            &mut observer,
                        )
                        .unwrap()
                };
                eredu_runtime::observe_model_logits(&mut observer, &logits).unwrap();
                check_captures(&graph, &observer.capture, tokens.len());
                assert_eq!(observer.units.len(), graph.routed_components.len());
                for units in observer.units.values() {
                    assert!(!units.values.is_empty());
                    assert_eq!(units.values, units.effective);
                    assert!(units.values.values().flatten().any(|v| *v != 0.));
                }
                for group in &graph.components {
                    let values = &observer.capture.values[group.write_input.as_ref().unwrap()];
                    let write = linear(values, &weights[&group.write_weight], None).unwrap();
                    let output_path = group.write_output.as_ref().unwrap();
                    assert_tensor_close(&write, &observer.capture.values[output_path], output_path);
                }
                // Constituent shared writes are already inside the complete sparse
                // contribution. Include that term once when reconstructing readout.
                let readout = graph.component_readout.as_ref().unwrap();
                let values = &observer.capture.values;
                let mut terms = vec![&values[&format!("{}.effective", readout.embedding)]];
                for layer in 0..args.num_hidden_layers as usize {
                    let attention = graph
                        .components
                        .iter()
                        .find(|group| {
                            group.layer_index == layer
                                && matches!(
                                    group.activation_equation,
                                    ComponentActivation::Attention { .. }
                                )
                        })
                        .unwrap();
                    terms.push(
                        &values[&format!("{}.effective", attention.output.as_ref().unwrap())],
                    );
                    if let Some(whole) = readout
                        .other_writes
                        .iter()
                        .find(|write| write.layer_index == layer)
                    {
                        terms.push(&values[&whole.effective_output]);
                    } else {
                        let feed_forward = graph
                            .components
                            .iter()
                            .find(|group| {
                                group.layer_index == layer
                                    && matches!(
                                        group.activation_equation,
                                        ComponentActivation::Gated { .. }
                                    )
                            })
                            .unwrap();
                        terms.push(
                            &values
                                [&format!("{}.effective", feed_forward.output.as_ref().unwrap())],
                        );
                    }
                }
                let residual = &values[&readout.residual];
                let reconstructed = NumericTensor::new(
                    residual.shape.clone(),
                    (0..residual.data.len())
                        .map(|index| {
                            terms
                                .iter()
                                .map(|term| term.data[index] as f64)
                                .sum::<f64>() as f32
                        })
                        .collect(),
                );
                assert_tensor_close(
                    &reconstructed,
                    residual,
                    "K2 embedding and complete residual writes",
                );
                assert_eq!(readout.normalization.kind, ComponentNormalizationKind::Rms);
                assert_eq!(readout.output_transform, ComponentOutputTransform::Identity);
                assert!(readout.bias.is_none());
                let width = args.hidden_size as usize;
                let group_width = width / readout.normalization.groups;
                let gain = &weights[readout.normalization.gain.as_ref().unwrap()].data;
                let head = &weights[&readout.weight].data;
                let vocabulary = args.vocab_size as usize;
                for (row_index, row) in residual.data.chunks_exact(width).enumerate() {
                    let denominators = row
                        .chunks_exact(group_width)
                        .map(|group| {
                            (group
                                .iter()
                                .map(|value| (*value as f64).powi(2))
                                .sum::<f64>()
                                / group_width as f64
                                + readout.normalization.epsilon.value() as f64)
                                .sqrt()
                        })
                        .collect::<Vec<_>>();
                    let mut scores = [0.0; 2];
                    for (slot, token) in [2usize, 7].into_iter().enumerate() {
                        scores[slot] = terms
                            .iter()
                            .map(|term| {
                                (0..width)
                                    .map(|column| {
                                        term.data[row_index * width + column] as f64
                                            * head[token * width + column] as f64
                                            * (gain[column] as f64
                                                + readout.normalization.gain_offset.value() as f64)
                                            / denominators[column / group_width]
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        let actual = logits.data[row_index * vocabulary + token] as f64;
                        assert!(
                            (scores[slot] - actual).abs() < 2e-5,
                            "K2 signed score: {case}, {step}, {token}"
                        );
                    }
                    let actual_margin = (logits.data[row_index * vocabulary + 2]
                        - logits.data[row_index * vocabulary + 7])
                        as f64;
                    assert!(
                        ((scores[0] - scores[1]) - actual_margin).abs() < 2e-5,
                        "K2 signed margin: {case}, {step}"
                    );
                }
                if let Some(path) = zero {
                    let effective = &observer.capture.values[&format!("{path}.effective")];
                    assert_eq!(
                        effective.data
                            [effective.data.len() - *effective.shape.last().unwrap() as usize],
                        0.
                    );
                    assert!(logits
                        .data
                        .iter()
                        .zip(&baseline[step])
                        .any(|(a, b)| (a - b).abs() > 1e-7));
                } else if !provider {
                    baseline.push(logits.data.clone());
                } else {
                    assert_eq!(baseline[step], logits.data);
                }
            }
        }
    }
}

#[test]
fn cold_artifact_report_retains_the_architecture_through_selection_and_source_preparation() {
    let (directory, _) = prepared_adapter::payload_fixture_config(&config("llama", false), 1.0);
    let artifact = eredu_architectures::configuration::inspect_artifact(directory.path()).unwrap();
    let expected = artifact.architecture_plan().architecture_descriptor();
    let outcome = eredu_architectures::inspect_selected_model(
        artifact,
        &eredu_runtime::NormalizedLoadRequest::default(),
        &prepared_adapter::NumericPreparationProvider { addressable: false },
        eredu_core::MediaFeatureAvailability {
            image: false,
            audio: false,
        },
    );
    assert_eq!(
        outcome.report().architecture_descriptor.as_ref(),
        Some(&expected)
    );
    let support = outcome.report().observation_support.as_ref().unwrap();
    // This fixture provider declares no host collector, despite constructing
    // the same logical architecture as an instrumented provider.
    assert!(support.points.iter().all(|p| matches!(
        p.prefill,
        eredu_core::ObservationSupportStatus::Unsupported(_)
    )));
    let sources = outcome.into_selected().unwrap().prepare_sources().unwrap();
    assert_eq!(sources.architecture().architecture_descriptor(), expected);
    assert_eq!(
        sources
            .complete()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        0
    );
}

fn check_captures(graph: &ArchitectureDescriptor, capture: &Capture, sequence: usize) {
    let support = eredu_runtime::inspection::observation_support(
        &graph.observations,
        eredu_runtime::inspection::ObservationExecutionContext {
            activation_inspection: true,
            selected: true,
            prediction_inspection: false,
            partitioned: false,
            mechanisms: eredu_core::ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                routed_unit_tensors: false,
                floating_to_f32: true,
            },
        },
    );
    for supported in &support.points {
        let point = graph.observations.get(&supported.path).unwrap();
        if matches!(
            point.value_type,
            eredu_core::ObservationValueType::RoutedUnits { .. }
        ) {
            assert!(matches!(
                supported.prefill,
                eredu_core::ObservationSupportStatus::Unsupported(_)
            ));
            assert!(matches!(
                supported.decode,
                eredu_core::ObservationSupportStatus::Unsupported(_)
            ));
            assert!(!capture.values.contains_key(&supported.path));
            continue;
        }
        assert_eq!(
            supported.prefill,
            eredu_core::ObservationSupportStatus::Supported
        );
        assert_eq!(
            supported.decode,
            eredu_core::ObservationSupportStatus::Supported
        );
        if !capture.request.matches(&supported.path) {
            continue;
        }
        let value = capture.values.get(&supported.path).unwrap_or_else(|| {
            panic!(
                "advertised point {} missing from {:?}",
                supported.path,
                capture.values.keys()
            )
        });
        if let Some(axes) = &point.axes {
            assert_eq!(axes.len(), value.shape.len(), "rank at {}", point.path);
            for (axis, actual) in axes.iter().zip(&value.shape) {
                let expected = match axis.dimension {
                    SymbolicDimension::Known(n) => Some(n),
                    SymbolicDimension::Batch => Some(1),
                    SymbolicDimension::Sequence => Some(sequence),
                    _ => None,
                };
                if let Some(expected) = expected {
                    assert_eq!(
                        *actual as usize, expected,
                        "axis {} at {}",
                        axis.name, point.path
                    );
                }
            }
        }
    }
    for path in capture.values.keys() {
        assert!(
            graph.observations.get(path).is_some(),
            "emitted path omitted by complete catalog: {path}"
        );
    }
}

#[test]
fn advertised_dense_paths_capture_prefill_and_decode_values() {
    let config = config("llama", false);
    let graph = descriptor(&config);
    let args = llama::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let architecture = llama::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
    let mut state = DeviceState::<NumericBackend, _>::create(
        llama::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    for tokens in [vec![1, 4, 2], vec![3]] {
        let input = NumericTensor::token_ids(&tokens);
        let mut capture = Capture {
            request: ObservationRequest::all(),
            values: BTreeMap::new(),
        };
        let logits = runtime
            .forward_with_observer(
                decoder::LayeredInput {
                    tokens: &input,
                    mask: None,
                },
                &mut state,
                &context,
                &mut capture,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut capture, &logits).unwrap();
        check_captures(&graph, &capture, tokens.len());
        assert_tensor_exact(
            &capture.values[eredu_core::MODEL_LOGITS_OBSERVATION_PATH],
            &logits,
            "advertised logits capture",
        );
    }
}

#[test]
fn nanbeige_discovery_describes_logical_invocations_and_executable_captures() {
    let config = super::nanbeige::tiny_config(false);
    let graph = descriptor(&config);
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.layer_index.is_some())
            .count(),
        4
    );
    assert_ne!(
        graph.node("decoder.layers.0").unwrap().parameter_groups,
        graph.node("decoder.layers.2").unwrap().parameter_groups
    );
    assert_ne!(
        graph
            .node("decoder.layers.1.output_norm")
            .unwrap()
            .parameter_groups,
        graph.node("output.norm").unwrap().parameter_groups
    );
    let group = &graph.layer_groups[0];
    assert_eq!(group.physical_layer_count, 2);
    assert_eq!(group.passes.len(), 2);
    assert_eq!(
        group.weight_sharing,
        eredu_core::LayerWeightSharing::SharedAcrossPasses
    );
    // Expand the group and select the output of physical layer zero in each pass.
    // These must remain two distinct observations despite sharing weights.
    let selected_paths = group
        .passes
        .iter()
        .map(|pass| {
            let execution = &pass.executions[0];
            assert_eq!(execution.physical_layer_index, 0);
            let node = graph.node(&execution.node_id).unwrap();
            let path = eredu_core::UnitObservation::Output
                .path(&format!("model.layers.{}", node.layer_index.unwrap()));
            assert!(node.observation_paths.contains(&path));
            path
        })
        .collect::<Vec<_>>();
    assert_eq!(
        selected_paths,
        ["model.layers.0.output", "model.layers.2.output"]
    );
    let args = eredu_architectures::nanbeige::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let model =
        eredu_architectures::nanbeige::LayeredModel::<NumericBackend>::new(args.clone(), &context)
            .unwrap();
    let mut runtime = LayerwiseRuntime::new(model, super::nanbeige::FixturePolicy::default());
    let selected_model =
        eredu_architectures::nanbeige::LayeredModel::<NumericBackend>::new(args.clone(), &context)
            .unwrap();
    let mut selected_runtime =
        LayerwiseRuntime::new(selected_model, super::nanbeige::FixturePolicy::default());
    let mut state = DeviceState::<NumericBackend, _>::create(
        eredu_architectures::nanbeige::state_layout(&args).unwrap(),
        |_, p| Ok::<_, Error>(NumericHybridLayerState::new(p)),
    )
    .unwrap();
    let mut selected_state = DeviceState::<NumericBackend, _>::create(
        eredu_architectures::nanbeige::state_layout(&args).unwrap(),
        |_, p| Ok::<_, Error>(NumericHybridLayerState::new(p)),
    )
    .unwrap();
    for ids in [vec![1, 3, 2], vec![4], vec![2]] {
        let tokens = NumericTensor::token_ids(&ids);
        let mut capture = Capture {
            request: ObservationRequest::all(),
            values: BTreeMap::new(),
        };
        let logits = runtime
            .forward_with_observer(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                &context,
                &mut capture,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut capture, &logits).unwrap();
        check_captures(&graph, &capture, ids.len());
        let mut selected = Capture {
            request: ObservationRequest::selected(
                selected_paths
                    .iter()
                    .cloned()
                    .map(ObservationSelector::Exact),
            ),
            values: BTreeMap::new(),
        };
        let selected_logits = selected_runtime
            .forward_with_observer(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut selected_state,
                &context,
                &mut selected,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut selected, &selected_logits).unwrap();
        check_captures(&graph, &selected, ids.len());
        assert_eq!(selected.values.len(), 2);
        assert_tensor_exact(
            &selected_logits,
            &logits,
            "capture selection preserves execution",
        );
        for path in &selected_paths {
            assert_tensor_exact(
                &selected.values[path],
                &capture.values[path],
                "expanded execution capture",
            );
        }
        assert_ne!(
            selected.values[&selected_paths[0]].data, selected.values[&selected_paths[1]].data,
            "passes over shared weights retain distinct activations"
        );
    }
}

#[test]
fn advertised_hybrid_shared_moe_paths_capture_both_phases_and_exact_selection() {
    let mut config = heterogeneous_replicated_configs()
        .into_iter()
        .find(|v| v["model_type"] == "qwen3_next")
        .unwrap();
    config["num_experts"] = 4.into();
    config["num_experts_per_tok"] = 2.into();
    config["moe_intermediate_size"] = 6.into();
    config["shared_expert_intermediate_size"] = 8.into();
    config["norm_topk_prob"] = true.into();
    let graph = descriptor(&config);
    let args = qwen::hybrid::model_args_from_config_value(&config)
        .unwrap()
        .text;
    let context = NumericContext::default();
    let architecture =
        qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
    let mut state = DeviceState::<NumericBackend, _>::create(
        qwen::hybrid::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let selected = graph
        .observations
        .points
        .iter()
        .find(|p| p.path.ends_with("routing.shared_output"))
        .unwrap()
        .path
        .clone();
    for (step, tokens) in [vec![1, 4, 2], vec![3], vec![2]].into_iter().enumerate() {
        let input = NumericTensor::token_ids(&tokens);
        let request = if step < 2 {
            ObservationRequest::all()
        } else {
            ObservationRequest::selected([ObservationSelector::Exact(selected.clone())])
        };
        let mut capture = Capture {
            request,
            values: BTreeMap::new(),
        };
        let logits = runtime
            .forward_with_provider_and_observer(
                qwen::hybrid::EmbeddedInput::target(&input, None),
                &mut state,
                if step == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                },
                &mut eredu_runtime::ResidentExpertProvider,
                &context,
                &mut capture,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut capture, &logits).unwrap();
        check_captures(&graph, &capture, tokens.len());
        if step == 2 {
            assert_eq!(capture.values.len(), 1);
            assert!(capture.values.contains_key(&selected));
        }
    }
}

#[test]
fn nemotron_non_gated_discovery_matches_actual_parameters_and_hooks() {
    let config = components::nemotron_config();
    let graph = descriptor(&config);
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let model = nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let description = model.parameter_description(&context).unwrap();
    let mut runtime = LayerwiseRuntime::new(model, RebuildingUnitPolicy::default());
    let mut state = DeviceState::<NumericBackend, _>::create(
        nemotron_h::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    assert_eq!(graph.components.len(), 2);
    let readout = graph
        .component_readout
        .as_ref()
        .expect("declared final readout");
    assert_eq!(readout.embedding_weight, "model.embeddings.weight");
    assert_eq!(
        readout.normalization.gain.as_deref(),
        Some("model.norm_f.weight")
    );
    let declarations: std::collections::BTreeSet<_> = description
        .groups()
        .iter()
        .flat_map(|group| group.members().iter().map(|member| member.target()))
        .collect();
    for group in &graph.components {
        assert!(
            declarations.contains(group.write_weight.as_str()),
            "{}",
            group.write_weight
        );
        for read in &group.reads {
            assert!(
                declarations.contains(read.weight.as_str()),
                "{}",
                read.weight
            );
        }
    }
    assert!(matches!(
        graph.components[1].activation_equation,
        eredu_core::component::ComponentActivation::Unary {
            activation: eredu_core::component::ComponentNonlinearity::ReluSquared
        }
    ));
    for ids in [vec![1, 4, 2], vec![3], vec![5]] {
        let tokens = NumericTensor::token_ids(&ids);
        let mut captured = Capture {
            request: ObservationRequest::all(),
            values: BTreeMap::new(),
        };
        let logits = runtime
            .forward_with_observer(
                nemotron_h::EmbeddedInput::target(&tokens, None),
                &mut state,
                &context,
                &mut captured,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut captured, &logits).unwrap();
        check_captures(&graph, &captured, ids.len());
    }
}

#[test]
fn lfm2_discovery_matches_mixed_component_and_routed_observations() {
    for sparse in [false, true] {
        let mut config = components::lfm2_component_config();
        if sparse {
            config["model_type"] = "lfm2_moe".into();
            config["num_dense_layers"] = 1.into();
            config["moe_intermediate_size"] = 6.into();
            config["num_experts"] = 2.into();
            config["num_experts_per_tok"] = 1.into();
        }
        let graph = descriptor(&config);
        let args = lfm2::model_args_from_config_value(&config).unwrap();
        let context = NumericContext::default();
        let model = lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        let description = model.parameter_description(&context).unwrap();
        let parameters: std::collections::BTreeSet<_> = description
            .groups()
            .iter()
            .flat_map(|group| group.members().iter().map(|member| member.target()))
            .collect();
        for group in &graph.components {
            assert!(graph.node(&group.node_id).is_some());
            assert!(parameters.contains(group.write_weight.as_str()));
            for read in &group.reads {
                assert!(parameters.contains(read.weight.as_str()));
                if let Some(norm) = &read.head_normalization {
                    assert!(parameters.contains(norm.normalization.gain.as_deref().unwrap()));
                }
            }
        }
        let readout = graph.component_readout.as_ref().unwrap();
        assert_eq!(readout.other_writes.len(), if sparse { 4 } else { 2 });
        for term in &readout.other_writes {
            assert!(graph.node(&term.node_id).is_some(), "{}", term.node_id);
            assert!(graph.observations.get(&term.output).is_some());
            assert!(graph.observations.get(&term.effective_output).is_some());
            assert_eq!(term.residual_scale.value(), 1.0);
        }
        let mut runtime = LayerwiseRuntime::new(model, RebuildingUnitPolicy::default());
        let mut state = DeviceState::<NumericBackend, _>::create(
            lfm2::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap();
        for ids in [vec![1, 4, 2], vec![3], vec![5]] {
            let tokens = NumericTensor::token_ids(&ids);
            let mut captured = Capture {
                request: ObservationRequest::all(),
                values: BTreeMap::new(),
            };
            let logits = runtime
                .forward_with_observer(
                    decoder::LayeredInput {
                        tokens: &tokens,
                        mask: None,
                    },
                    &mut state,
                    &context,
                    &mut captured,
                )
                .unwrap();
            eredu_runtime::observe_model_logits(&mut captured, &logits).unwrap();
            check_captures(&graph, &captured, ids.len());
        }
    }
}

#[test]
fn nemotron_mixed_discovery_includes_complete_mamba_and_routed_writes() {
    let config = components::nemotron_mixed_config();
    let graph = descriptor(&config);
    let args = nemotron_h::model_args_from_config_value(&config).unwrap();
    let context = NumericContext::default();
    let model = nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut runtime = LayerwiseRuntime::new(model, components::MixedComponentUnitPolicy);
    let mut state = DeviceState::<NumericBackend, _>::create(
        nemotron_h::state_layout(&args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let terms = &graph.component_readout.as_ref().unwrap().other_writes;
    assert_eq!(terms.len(), 2);
    assert_eq!(
        terms
            .iter()
            .map(|term| term.layer_index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    for term in terms {
        assert!(graph.nodes.iter().any(|node| node.id == term.node_id));
        assert!(graph.observations.get(&term.output).is_some());
        assert!(graph.observations.get(&term.effective_output).is_some());
    }
    for ids in [vec![1, 4, 2], vec![3], vec![5]] {
        let tokens = NumericTensor::token_ids(&ids);
        let mut capture = Capture {
            request: ObservationRequest::all(),
            values: BTreeMap::new(),
        };
        let logits = runtime
            .forward_with_observer(
                nemotron_h::EmbeddedInput::target(&tokens, None),
                &mut state,
                &context,
                &mut capture,
            )
            .unwrap();
        eredu_runtime::observe_model_logits(&mut capture, &logits).unwrap();
        check_captures(&graph, &capture, ids.len());
        for term in terms {
            assert!(capture.values[&term.effective_output]
                .data
                .iter()
                .any(|value| value.abs() > 1e-7));
        }
    }
}
