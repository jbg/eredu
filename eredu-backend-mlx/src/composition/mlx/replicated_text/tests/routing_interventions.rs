use super::*;
use crate::MlxTensor;
use eredu_nn::routing_intervention::{
    GroupScoreStage, GroupSelectionAction, GroupSelectionControl,
};
use eredu_nn::{GroupedNeuralBackend, Parameterized, Tensor};
use eredu_runtime::{
    ActivationObserver, RoutedExpertProvider, RoutedExpertRequest, RoutingDecision,
};
use std::{cell::Cell, rc::Rc};

type B = crate::composition::MlxNeuralBackend;

struct Fill;
impl<'a> eredu_nn::ParameterVisitorMut<'a, MlxTensor> for Fill {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadata, tensor: &'a mut MlxTensor) {
        let shape = tensor.shape().to_vec();
        let count = shape.iter().product::<i32>() as usize;
        let values = if metadata.id.as_str() == "model.layers.0.mlp.gate.weight" {
            (0..count).map(|i| (i / 32 + 1) as f32 / 32.0).collect()
        } else {
            vec![0.01f32; count]
        };
        *tensor = MlxTensor::from_array(Array::from_slice(&values, &shape));
    }
}

struct Probe {
    calls: usize,
    applied: Rc<Cell<bool>>,
    ids: Vec<u32>,
    coefficients: Vec<f32>,
}

impl RoutedExpertProvider<B> for Probe {
    type Error = eredu_nn::Error;
    fn forward_grouped(
        &mut self,
        _: &mut <B as GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        assert!(
            self.applied.get(),
            "effective decision must be reported before expert provider execution"
        );
        self.calls += 1;
        self.ids = request
            .routes
            .group_indices()
            .as_array()
            .as_dtype(Dtype::Uint32, stream)
            .unwrap()
            .contiguous(false, stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        self.coefficients = request.routes.coefficients().to_f32_vec(stream).unwrap();
        let top_k = request.routes.group_indices().shape()[1] as usize;
        let width = *request.input.shape().last().unwrap() as usize;
        let mut data = Vec::new();
        // Every expert has a distinguishable contribution. These are the exact
        // IDs handed to the provider, not inferred from final generated text.
        for (ids, weights) in self.ids.chunks(top_k).zip(self.coefficients.chunks(top_k)) {
            let output = ids
                .iter()
                .zip(weights)
                .map(|(id, weight)| 10.0f32.powi(*id as i32) * weight)
                .sum::<f32>();
            data.extend(std::iter::repeat_n(output, width));
        }
        Ok(MlxTensor::from_array(Array::from_slice(
            &data,
            &request.input.shape(),
        )))
    }
    fn forward_relu2_routed(
        &mut self,
        _: &mut <B as GroupedNeuralBackend>::Relu2Groups,
        _: RoutedExpertRequest<'_, MlxTensor>,
        _: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        unreachable!()
    }
}

struct Observer {
    control: Option<GroupSelectionControl>,
    applied: Rc<Cell<bool>>,
    stream: Stream,
    original_ids: Vec<u32>,
    original_coefficients: Vec<f32>,
    shared: Vec<f32>,
}
impl ActivationObserver<MlxTensor, eredu_nn::Error> for Observer {
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), eredu_nn::Error> {
        Ok(())
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<GroupSelectionControl>, eredu_nn::Error> {
        assert_eq!(path, "model.layers.0.mlp");
        assert_eq!(rows, 2);
        Ok(self.control.clone())
    }
    fn routing_applied(
        &mut self,
        _: &str,
        original: Option<RoutingDecision<'_, MlxTensor>>,
        _: RoutingDecision<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        let original = original.unwrap();
        self.original_ids = original
            .ids
            .as_array()
            .as_dtype(Dtype::Uint32, &self.stream)
            .unwrap()
            .contiguous(false, &self.stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        self.original_coefficients = original.coefficients.to_f32_vec(&self.stream).unwrap();
        self.applied.set(true);
        Ok(())
    }
    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.shared = routing
            .shared_output
            .unwrap()
            .to_f32_vec(&self.stream)
            .unwrap();
        Ok(())
    }
}

#[test]
fn routing_intervention_shared_hybrid_controls_actual_dispatch_and_preserves_shared_branch() {
    let (stream, _) = execution_streams();
    let mut config = routed_qwen_hybrid_config();
    config["num_experts"] = 4.into();
    config["num_experts_per_tok"] = 2.into();
    config["norm_topk_prob"] = true.into();
    let config = eredu_architectures::qwen::hybrid::model_args_from_config_value(&config)
        .unwrap()
        .text;
    let mut block =
        eredu_architectures::qwen::hybrid::Block::<B>::new(&config, 0, &stream).unwrap();
    block.visit_parameters_mut(&mut Fill);
    let eredu_architectures::qwen::hybrid::FeedForward::Routed(moe) = &mut block.feed_forward
    else {
        panic!()
    };
    let input = MlxTensor::from_array(Array::from_slice(&[1.0f32; 64], &[1, 2, 32]));
    let mut baseline_shared = Vec::new();
    let mut baseline_ids = Vec::new();
    let mut baseline_weights = Vec::new();
    let mut rank_bias_weights = Vec::new();
    let actions = [
        None,
        Some(GroupSelectionAction::Exclude(vec![3])),
        Some(GroupSelectionAction::ZeroContribution(vec![3])),
        Some(GroupSelectionAction::Bias {
            stage: GroupScoreStage::RankingScores,
            ids: vec![0],
            values: vec![10.0],
        }),
        Some(GroupSelectionAction::Bias {
            stage: GroupScoreStage::RawLogits,
            ids: vec![0],
            values: vec![10.0],
        }),
        Some(GroupSelectionAction::Force(vec![0, 1])),
        Some(GroupSelectionAction::Bias {
            stage: GroupScoreStage::TransformedScores,
            ids: vec![0],
            values: vec![1.0],
        }),
    ];
    for (experiment, action) in actions.into_iter().enumerate() {
        let applied = Rc::new(Cell::new(action.is_none()));
        let control = action.map(|action| GroupSelectionControl {
            expected: eredu_nn::TopKGroupSelectionSpec::new(
                4,
                2,
                eredu_nn::GroupScoring::Softmax,
                true,
            )
            .unwrap(),
            learned_coefficient_scale: false,
            first_row: 1,
            end_row: 2,
            row_stride: 1,
            action,
            capture_original: true,
        });
        let mut observer = Observer {
            control,
            applied: applied.clone(),
            stream: stream.clone(),
            original_ids: vec![],
            original_coefficients: vec![],
            shared: vec![],
        };
        let mut provider = Probe {
            calls: 0,
            applied,
            ids: vec![],
            coefficients: vec![],
        };
        let output = moe
            .forward_observed_with_provider(
                eredu_runtime::RoutedObservationPoint::new("model.layers.0.mlp", 4),
                &input,
                &stream,
                &mut provider,
                &mut observer,
            )
            .unwrap();
        assert_eq!(provider.calls, 1);
        let values = output.to_f32_vec(&stream).unwrap();
        for row in 0..2 {
            let weighted = provider.ids[row * 2..row * 2 + 2]
                .iter()
                .zip(&provider.coefficients[row * 2..row * 2 + 2])
                .map(|(id, w)| 10.0f32.powi(*id as i32) * w)
                .sum::<f32>();
            assert!((values[row * 32] - observer.shared[row * 32] - weighted).abs() < 0.001);
        }
        if experiment == 0 {
            baseline_shared = observer.shared;
            baseline_ids = provider.ids;
            baseline_weights = provider.coefficients;
            continue;
        }
        assert_eq!(
            observer.shared, baseline_shared,
            "routed control must not mutate shared-expert behavior"
        );
        assert_eq!(observer.original_ids, baseline_ids);
        assert_eq!(observer.original_coefficients, baseline_weights);
        assert_eq!(
            provider.ids[..2],
            baseline_ids[..2],
            "unselected prefill rows are unchanged"
        );
        assert_eq!(provider.coefficients[..2], baseline_weights[..2]);
        let ids = &provider.ids[2..];
        let weights = &provider.coefficients[2..];
        match experiment {
            1 => {
                assert!(!ids.contains(&3), "excluded IDs reached provider: {ids:?}");
                assert!(ids.contains(&1) && ids.contains(&2));
            }
            2 => {
                assert_eq!(provider.ids, baseline_ids);
                for (slot, id) in ids.iter().enumerate() {
                    assert_eq!(
                        weights[slot],
                        if *id == 3 {
                            0.0
                        } else {
                            baseline_weights[slot + 2]
                        }
                    );
                }
            }
            3 => {
                assert!(ids.contains(&0));
                rank_bias_weights = weights.to_vec();
            }
            4 => {
                assert!(ids.contains(&0));
                assert_ne!(
                    weights, rank_bias_weights,
                    "raw-logit and ranking-only bias have distinct weight semantics"
                );
            }
            5 => assert_eq!(ids, [0, 1]),
            6 => {
                assert!(ids.contains(&0));
                assert_ne!(weights, rank_bias_weights);
            }
            _ => unreachable!(),
        }
        if experiment != 2 {
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
    assert!(baseline_shared.iter().any(|v| *v != 0.0));
    for action in [
        GroupSelectionAction::Force(vec![4, 1]),
        GroupSelectionAction::Force(vec![1, 1]),
        GroupSelectionAction::Force(vec![1]),
        GroupSelectionAction::Exclude(vec![0, 1, 2]),
    ] {
        let applied = Rc::new(Cell::new(false));
        let mut observer = Observer {
            control: Some(GroupSelectionControl {
                expected: eredu_nn::TopKGroupSelectionSpec::new(
                    4,
                    2,
                    eredu_nn::GroupScoring::Softmax,
                    true,
                )
                .unwrap(),
                learned_coefficient_scale: false,
                first_row: 1,
                end_row: 2,
                row_stride: 1,
                action,
                capture_original: false,
            }),
            applied: applied.clone(),
            stream: stream.clone(),
            original_ids: vec![],
            original_coefficients: vec![],
            shared: vec![],
        };
        let mut provider = Probe {
            calls: 0,
            applied,
            ids: vec![],
            coefficients: vec![],
        };
        assert!(moe
            .forward_observed_with_provider(
                eredu_runtime::RoutedObservationPoint::new("model.layers.0.mlp", 4),
                &input,
                &stream,
                &mut provider,
                &mut observer
            )
            .is_err());
        assert_eq!(
            provider.calls, 0,
            "invalid requests must never fall back to ordinary dispatch"
        );
    }
}

#[test]
fn routing_intervention_loaded_discovery_admission_and_evidence_agree_for_qwen_families() {
    use eredu_core::{
        capture::*, intervention::*, ControlledTextGeneration, ModelRuntime, TextGenerationBackend,
        TextGenerationConfig,
    };
    struct Open;
    impl eredu_core::TokenFilterController for Open {
        type Error = std::convert::Infallible;
        fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
            Ok(eredu_core::TokenFilter::All)
        }
        fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    for root in [
        tiny_artifact("qwen3_moe", false),
        tiny_heterogeneous_artifact(routed_qwen_next_config()),
        tiny_heterogeneous_artifact(routed_qwen_hybrid_config()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let model = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
            .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let discovery = crate::backend::MlxBackend::intervention_discovery(&runtime).unwrap();
        let captures = crate::backend::MlxBackend::capture_discovery(&runtime).unwrap();
        let request = CaptureRequestShape {
            batch: 1,
            prompt_tokens: 2,
            max_predictions: 3,
        };
        let routes: Vec<_> = discovery
            .points
            .iter()
            .filter(|p| p.routing.is_some())
            .collect();
        assert!(!routes.is_empty());
        let mut operations = Vec::new();
        for (i, point) in routes.iter().enumerate() {
            assert_eq!(
                point.prefill,
                eredu_core::ObservationSupportStatus::Supported
            );
            assert_eq!(
                point.decode,
                eredu_core::ObservationSupportStatus::Supported
            );
            for prefill in [true, false] {
                operations.push(InterventionOperation {
                    id: format!("route-{i}-{prefill}"),
                    target: point.path.clone(),
                    schedule: CaptureSchedule {
                        prefill,
                        decode: !prefill,
                        ..Default::default()
                    },
                    slices: if prefill {
                        vec![CaptureSlice {
                            axis: "token".into(),
                            start: 1,
                            end: 2,
                            stride: 1,
                        }]
                    } else {
                        vec![]
                    },
                    action: InterventionAction::ForceExperts {
                        shape: [1, 1],
                        expert_ids: vec![u32::from(!prefill)],
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 2 },
                });
            }
        }
        let admitted = InterventionPlan {
            schema_version: 1,
            operations,
        }
        .admit(&discovery, request, "loaded-routes")
        .unwrap();
        let budget = CaptureUsage {
            captures: 1000,
            retained_bytes: 1_000_000_000,
            host_bytes: 10_000_000,
            encoded_bytes: 10_000_000,
        };
        let capture = CapturePlan {
            schema_version: 1,
            selections: vec![],
            limits: CaptureLimits {
                per_step: budget,
                cumulative: budget,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &captures.catalog,
            &captures.support,
            &captures.support.capture,
            request,
        )
        .unwrap();
        crate::backend::MlxBackend::validate_text_interventions(&runtime, &capture, &admitted)
            .unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(3),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        let mut generator = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 2],
            TextGenerationConfig::new(sampling),
            Open,
        )
        .unwrap();
        generator.enable_interventions(capture, admitted).unwrap();
        let mut count = 0;
        while let Some(token) = generator.next() {
            token.unwrap();
            let step = generator.take_captured_step().unwrap().unwrap();
            assert_eq!(step.prediction_index, count);
            for record in step.interventions {
                if record.outcome == InterventionOutcome::Inactive {
                    continue;
                }
                assert_eq!(record.outcome, InterventionOutcome::Applied);
                assert_eq!(record.evidence.len(), 4);
                assert!(record.evidence.iter().all(|e| e.payload.is_some()));
                assert_eq!(
                    record.evidence[0].position,
                    eredu_core::ObservationPosition::BeforeIntervention
                );
                assert_eq!(
                    record.evidence[2].position,
                    eredu_core::ObservationPosition::AfterIntervention
                );
                let Some(CapturePayload::Tensor(tensor)) = &record.evidence[2].payload else {
                    panic!("effective route IDs missing")
                };
                let ids: Vec<u32> = match tensor.data() {
                    eredu_core::TensorObservationData::U64(v) => {
                        v.iter().map(|n| *n as u32).collect()
                    }
                    eredu_core::TensorObservationData::I64(v) => {
                        v.iter().map(|n| *n as u32).collect()
                    }
                    other => panic!("route IDs are not exact integers: {other:?}"),
                };
                assert_eq!(ids, [u32::from(count != 0)]);
            }
            count += 1;
        }
        assert_eq!(count, 3);
        drop(generator);
        runtime.parts_mut().1.reset().unwrap();
    }
}
