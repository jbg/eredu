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
            partitioned: false,
            mechanisms: eredu_core::ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                floating_to_f32: true,
            },
        },
    );
    for supported in &support.points {
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
        let point = graph.observations.get(&supported.path).unwrap();
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
