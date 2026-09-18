//! Typed extensions borrow the target's embedding while owning prediction state.
use super::*;
use eredu_core::{component::ComponentActivation, ObservationValueType};
use eredu_runtime::ActivationObserver;

type V3State = DeviceState<NumericBackend, NumericCompressedCache>;

struct ParameterEdit<'a> {
    target: &'a str,
    replacement: Option<NumericTensor>,
    saved: Option<NumericTensor>,
    modules: Vec<usize>,
}
impl<'a> eredu_nn::ParameterVisitorMut<'a, NumericTensor> for ParameterEdit<'_> {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
        if metadata.id().as_str() == self.target {
            assert!(
                self.saved.replace(value.clone()).is_none(),
                "one authoritative prediction slot"
            );
            *value = self
                .replacement
                .take()
                .unwrap_or_else(|| value.map(|x| 2.0 * x));
        }
    }
}
impl PredictionModuleVisitor<NumericBackend, Materializer> for ParameterEdit<'_> {
    type Error = std::convert::Infallible;
    fn visit<M: Parameterized<NumericTensor>>(
        &mut self,
        ordinal: usize,
        module: &mut Module<M>,
    ) -> Result<(), Self::Error> {
        self.modules.push(ordinal);
        module.as_mut().visit_parameters_mut(self);
        Ok(())
    }
}
struct Invoke<'a> {
    target: &'a mut deepseek::v3::Model<NumericBackend>,
    state: &'a mut V3State,
    parallel: Option<&'a NumericParallelContext>,
    context: &'a NumericContext,
    calls: usize,
}
impl PredictionOperationInvoker<deepseek::v3::Model<NumericBackend>, NumericBackend, V3State>
    for Invoke<'_>
{
    type Error = Error;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Error>
    where
        O: eredu_runtime::PredictionTargetOperation<
            deepseek::v3::Model<NumericBackend>,
            NumericBackend,
            V3State,
        >,
    {
        self.calls += 1;
        operation.apply(self.target, self.state, self.parallel, self.context)
    }
    fn invalid(message: String) -> Error {
        Error::backend(message)
    }
}
#[derive(Default)]
struct Observe {
    values: BTreeMap<String, NumericTensor>,
    zero: Option<String>,
    fail: Option<String>,
    active_units: Option<eredu_core::component::ComponentCoordinateMap>,
    unit_starts: usize,
    unit_finishes: Vec<bool>,
    unit_callbacks: usize,
    fail_units: bool,
}
impl ActivationObserver<NumericTensor, Error> for Observe {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.values.insert(path.into(), value.clone());
        if self.fail.as_deref() == Some(path) {
            return Err(Error::backend("injected extension observation failure"));
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        if self.zero.as_deref() != Some(path) {
            return Ok(None);
        }
        let mut result = value.clone();
        let width = *result.shape.last().unwrap() as usize;
        let index = result.data.len() - width + 1;
        result.data[index] = 0.0;
        Ok(Some(result))
    }
    fn routed_unit_observer(
        &mut self,
        _: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error> {
        Ok(Some(self))
    }
    fn observe_routing(
        &mut self,
        event: eredu_runtime::RoutingObservation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        event.for_each_tensor(|path, value| {
            self.values.insert(path, value.clone());
        });
        Ok(())
    }
}

impl eredu_runtime::RoutedUnitObserver<NumericTensor> for Observe {
    fn begin_invocation(
        &mut self,
        invocation: &eredu_runtime::RoutedUnitInvocation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(self.active_units.is_none());
        assert!(invocation.origins.is_none());
        let coordinates = invocation
            .unit_coordinates
            .expect("prepared resident columns");
        assert_eq!(coordinates.contiguous_range(), Some(0..4));
        self.active_units = Some(coordinates.clone());
        self.unit_starts += 1;
        Ok(())
    }
    fn invocation_active(&self) -> bool {
        self.active_units.is_some()
    }
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(self.active_units.is_some(), "active provider");
        assert!(
            batch.unit_coordinates.is_none(),
            "serial source uses local capture"
        );
        assert!(batch.origins.is_none());
        assert_eq!(batch.units.values.shape[1], 4);
        self.unit_callbacks += 1;
        if self.fail_units {
            return Err(Error::backend("injected sparse prediction failure"));
        }
        Ok(())
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert!(batch.unit_coordinates.is_none());
        assert!(self.active_units.is_some());
        Ok(())
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        assert!(self.active_units.take().is_some());
        self.unit_finishes.push(success);
        Ok(())
    }
}

#[test]
fn v3_typed_extension_hooks_preserve_scope_state_and_retry() {
    let context = NumericContext::default();
    for query in [None, Some(3)] {
        let config = serde_json::json!({
            "model_type":"deepseek_v3", "hidden_size":8, "vocab_size":16,
            "num_hidden_layers":1, "num_attention_heads":2, "intermediate_size":10,
            "moe_intermediate_size":4, "q_lora_rank":query, "kv_lora_rank":3,
            "qk_nope_head_dim":2, "qk_rope_head_dim":2, "v_head_dim":3,
            "first_k_dense_replace":1, "n_routed_experts":2, "n_shared_experts":1,
            "num_experts_per_tok":1, "n_group":1, "topk_group":1,
            "max_position_embeddings":64, "num_nextn_predict_layers":2
        });
        let args = deepseek::parse_v3_config(&config).unwrap();
        let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let (_root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection =
            eredu_architectures::configuration::inspect_artifact(_root.path()).unwrap();
        let plan = prepared_adapter::plan(None).with_drafting(eredu_core::DraftingPlan::Embedded {
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
        let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
            &plan,
            eredu_runtime::ResidencyDiagnostics::new(false, false),
            None,
        )
        .unwrap();
        let selected =
            eredu_architectures::select_preparation(&inspection, &request, &Provider).unwrap();
        let extension_plan = selected.prediction_extension().unwrap();
        let tasks = selected
            .text_realization()
            .auxiliary_materialization_tasks();
        for parallel in [false, true] {
            let mut target_args = args.clone();
            target_args.num_nextn_predict_layers = 0;
            let model = if parallel {
                let mut groups =
                    deepseek::parallel::v3_static_parameter_groups(&target_args).unwrap();
                groups.extend(
                    deepseek::parallel::v3_layer_parameter_groups(&target_args, 0).unwrap(),
                );
                let layout = numeric_local_layout(&groups, 1, 0).unwrap();
                let geometry =
                    deepseek::parallel::v3_local_geometry(&target_args, &layout).unwrap();
                deepseek::v3::Model::<NumericBackend>::new_parallel(
                    target_args.clone(),
                    geometry,
                    &context,
                )
                .unwrap()
            } else {
                deepseek::v3::Model::<NumericBackend>::new(target_args.clone(), &context).unwrap()
            };
            let mut model = model;
            let fresh = || {
                let prepared = prepare_replicated_prediction_extension::<NumericBackend>(
                    extension_plan,
                    tasks,
                    &context,
                    &context,
                )
                .unwrap();
                let PreparedPredictionExtension::DeepSeekV3 { units, .. } = prepared else {
                    unreachable!()
                };
                MaterializedDeepSeekV3Prediction::<NumericBackend, Materializer> {
                    units: units
                        .into_iter()
                        .map(|unit| Module(unit.into_parts().1))
                        .collect(),
                }
            };
            let mut extension = fresh();
            retained_resources::verify::<deepseek::v3::Model<NumericBackend>, _>(&mut extension, 0);
            let mut ordinary = fresh();
            let mut state =
                V3State::create(deepseek::v3::state_layout(&target_args).unwrap(), |_, _| {
                    Ok::<_, Error>(NumericCompressedCache::resident())
                })
                .unwrap();
            let parallel_context = NumericParallelContext::new(0, NumericParallelGroup::new(1));
            let mut invoker = Invoke {
                target: &mut model,
                state: &mut state,
                parallel: parallel.then_some(&parallel_context),
                context: &context,
                calls: 0,
            };
            let mut lane = extension.new_state();
            let mut ordinary_lane = ordinary.new_state();
            let hidden = NumericTensor::new(
                [1, 3, 8],
                (0..24).map(|i| (i as f32 * 0.43 - 0.8).sin()).collect(),
            );
            let tokens = NumericTensor::token_ids(&[1, 3, 2]);
            let mut capture = Observe::default();
            extension
                .prefill_observed::<V3State, _>(
                    &mut invoker,
                    &hidden,
                    &hidden,
                    &tokens,
                    &mut lane,
                    Some(&mut capture),
                )
                .unwrap();
            assert_eq!(capture.unit_starts, 2);
            assert_eq!(capture.unit_finishes, [true, true]);
            assert!(capture.unit_callbacks >= 2);
            assert!(capture.active_units.is_none());
            ordinary
                .prefill::<V3State, _>(&mut invoker, &hidden, &hidden, &tokens, &mut ordinary_lane)
                .unwrap();
            // Serial execution emits the ordinary tensor catalog. This direct
            // TP fixture has no provider-output collector; verify its component,
            // outer-unit and generated-input hooks independently of route events.
            for scope in &graph.component_scopes {
                for point in &graph.observations.points {
                    let mut node = graph.node(&point.node_id).unwrap();
                    while node.id != scope.node_id {
                        let Some(parent) = &node.parent else {
                            break;
                        };
                        node = graph.node(parent).unwrap();
                    }
                    if node.id == scope.node_id
                        && matches!(point.value_type, ObservationValueType::Tensor)
                        && !(parallel
                            && point
                                .requirements
                                .contains(&eredu_core::ObservationRequirement::RoutingEvents))
                    {
                        assert!(
                            capture.values.contains_key(&point.path),
                            "missing {} (parallel={parallel})",
                            point.path
                        );
                    }
                }
            }
            let replay_hidden = hidden.axis_slice(1, 1, 3);
            let replay_tokens = NumericTensor::token_ids(&[3, 2]);
            let mut replay = Observe::default();
            extension
                .advance_observed::<V3State, _>(
                    &mut invoker,
                    &replay_hidden,
                    &replay_tokens,
                    &mut lane,
                    Some(&mut replay),
                )
                .unwrap();
            ordinary
                .advance::<V3State, _>(
                    &mut invoker,
                    &replay_hidden,
                    &replay_tokens,
                    &mut ordinary_lane,
                )
                .unwrap();
            for scope in &graph.component_scopes {
                assert_eq!(replay.values[&scope.readout.logits].shape[1], 2);
            }
            for depth in 0..2 {
                let scope = &graph.component_scopes[depth];
                let channel = scope
                    .components
                    .iter()
                    .find(|g| {
                        matches!(g.activation_equation, ComponentActivation::Attention { .. })
                    })
                    .unwrap();
                for step in 0..3 {
                    let checkpoint = lane.clone();
                    let input = hidden.axis_slice(1, 2, 3);
                    let token = NumericTensor::token_ids(&[4 + step]);
                    let mut capture = Observe::default();
                    let actual = extension
                        .logits_observed::<V3State, _>(
                            &mut invoker,
                            &input,
                            &token,
                            depth,
                            &mut lane,
                            Some(&mut capture),
                        )
                        .unwrap();
                    let expected = ordinary
                        .logits::<V3State, _>(
                            &mut invoker,
                            &input,
                            &token,
                            depth,
                            &mut ordinary_lane,
                        )
                        .unwrap();
                    assert_tensor_exact(&actual.0, &expected.0, "typed extension no-op logits");
                    assert_tensor_exact(&actual.1, &expected.1, "typed extension no-op hidden");
                    assert_tensor_exact(
                        &actual.0,
                        &capture.values[&scope.readout.logits],
                        "typed scope output head",
                    );
                    let completed = lane.clone();
                    let mut edit = ParameterEdit {
                        target: &scope.readout.weight,
                        replacement: None,
                        saved: None,
                        modules: Vec::new(),
                    };
                    extension.visit_modules(&mut edit).unwrap();
                    assert_eq!(edit.modules, [0, 1]);
                    let original_head = edit.saved.unwrap();
                    lane = checkpoint.clone();
                    let edited = extension
                        .logits::<V3State, _>(&mut invoker, &input, &token, depth, &mut lane)
                        .unwrap();
                    assert_tensor_exact(
                        &edited.0,
                        &actual.0.map(|x| 2.0 * x),
                        "visited prediction head changes actual logits",
                    );
                    assert_tensor_exact(
                        &edited.1,
                        &actual.1,
                        "head edit preserves decoder state output",
                    );
                    let mut restore = ParameterEdit {
                        target: &scope.readout.weight,
                        replacement: Some(original_head),
                        saved: None,
                        modules: Vec::new(),
                    };
                    extension.visit_modules(&mut restore).unwrap();
                    assert!(restore.saved.is_some());
                    lane = checkpoint.clone();
                    let mut masked = Observe {
                        zero: Some(channel.activation.clone()),
                        ..Default::default()
                    };
                    let changed = extension
                        .logits_observed::<V3State, _>(
                            &mut invoker,
                            &input,
                            &token,
                            depth,
                            &mut lane,
                            Some(&mut masked),
                        )
                        .unwrap();
                    assert_ne!(
                        changed.0.data, actual.0.data,
                        "typed extension consumes the channel edit"
                    );
                    lane = checkpoint.clone();
                    let mut failure = Observe {
                        fail: Some(scope.readout.normalized.clone()),
                        ..Default::default()
                    };
                    assert!(extension
                        .logits_observed::<V3State, _>(
                            &mut invoker,
                            &input,
                            &token,
                            depth,
                            &mut lane,
                            Some(&mut failure)
                        )
                        .unwrap_err()
                        .to_string()
                        .contains("injected extension observation failure"));
                    assert_eq!(
                        lane[depth].offset(),
                        completed[depth].offset(),
                        "failure followed real state work"
                    );
                    lane = checkpoint.clone();
                    let mut sparse_failure = Observe {
                        fail_units: true,
                        ..Default::default()
                    };
                    let error = extension
                        .logits_observed::<V3State, _>(
                            &mut invoker,
                            &input,
                            &token,
                            depth,
                            &mut lane,
                            Some(&mut sparse_failure),
                        )
                        .unwrap_err();
                    assert!(error
                        .to_string()
                        .contains("injected sparse prediction failure"));
                    assert_eq!(sparse_failure.unit_starts, 1);
                    assert_eq!(sparse_failure.unit_finishes, [false]);
                    assert!(sparse_failure.active_units.is_none());
                    // The enclosing speculative owner restores its prediction lane
                    // before retry; the target's own cache was never advanced.
                    lane = checkpoint;
                    let retry = extension
                        .logits_observed::<V3State, _>(
                            &mut invoker,
                            &input,
                            &token,
                            depth,
                            &mut lane,
                            None,
                        )
                        .unwrap();
                    assert_tensor_exact(&retry.0, &actual.0, "typed extension restored retry");
                    assert_eq!(invoker.state.layer(0).unwrap().offset(), 0);
                    for other in 0..2 {
                        assert_eq!(lane[other].offset(), ordinary_lane[other].offset());
                    }
                }
            }
            let calls = invoker.calls;
            let mut capture = Observe::default();
            assert!(extension
                .logits_observed::<V3State, _>(
                    &mut invoker,
                    &hidden,
                    &tokens,
                    2,
                    &mut lane,
                    Some(&mut capture)
                )
                .is_err());
            assert_eq!(
                invoker.calls, calls,
                "invalid depth rejected before invocation"
            );
            assert!(capture.values.is_empty());
        }
    }
}

#[test]
fn v3_prepared_prediction_formats_preserve_sources_and_packed_layouts() {
    let config = serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":64, "vocab_size":32,
        "num_hidden_layers":2, "num_attention_heads":4, "intermediate_size":128,
        "moe_intermediate_size":64, "q_lora_rank":32, "kv_lora_rank":32,
        "qk_nope_head_dim":8, "qk_rope_head_dim":4, "v_head_dim":16,
        "first_k_dense_replace":1, "n_routed_experts":4, "n_shared_experts":1,
        "num_experts_per_tok":2, "n_group":2, "topk_group":1,
        "max_position_embeddings":64, "num_nextn_predict_layers":1
    });
    let (root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let context = NumericContext::default();
    struct Geometry(BTreeMap<String, (Vec<i32>, eredu_core::checkpoint::TensorDtype)>);
    impl<'a> eredu_nn::ParameterVisitor<'a, NumericTensor> for Geometry {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a NumericTensor) {
            assert!(self
                .0
                .insert(
                    metadata.id().to_string(),
                    (value.shape.clone(), value.dtype.clone())
                )
                .is_none());
        }
    }
    for (quantization, companions) in [
        (None, 0),
        (
            Some(eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            }),
            2,
        ),
        (Some(eredu_core::QuantizationRequest::MxFp4), 1),
    ] {
        let packed = quantization.is_some();
        let request = quantization.map_or_else(
            Default::default,
            eredu_runtime::NormalizedLoadRequest::with_quantization,
        );
        let selected =
            eredu_architectures::select_preparation(&inspection, &request, &Provider).unwrap();
        let prepared = prepare_replicated_prediction_extension::<NumericBackend>(
            selected.prediction_extension().unwrap(),
            selected
                .text_realization()
                .auxiliary_materialization_tasks(),
            &context,
            &context,
        )
        .unwrap();
        let PreparedPredictionExtension::DeepSeekV3 { layout, units, .. } = prepared else {
            unreachable!()
        };
        let mut transformed = 0;
        for unit in units {
            assert_eq!(unit.source_layout().is_some(), packed);
            let (source, target, tasks) = unit.into_parts();
            let mut source_shapes = Geometry(Default::default());
            let mut target_shapes = Geometry(Default::default());
            source.visit_parameters(&mut source_shapes);
            target.visit_parameters(&mut target_shapes);
            for task in tasks {
                let target = &target_shapes.0[task.name()];
                // The scalar reference backend keeps decoded logical matrices;
                // selected physical storage is represented by the retained layout.
                assert_eq!(
                    target.0,
                    task.logical_shape()
                        .iter()
                        .map(|n| *n as i32)
                        .collect::<Vec<_>>()
                );
                if matches!(
                    task.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                ) {
                    transformed += 1;
                    assert_eq!(
                        source_shapes.0[task.name()].0,
                        task.logical_shape()
                            .iter()
                            .map(|n| *n as i32)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        source_shapes.0[task.name()].1,
                        eredu_core::checkpoint::TensorDtype::F32
                    );
                    let mut packed_shape = task.logical_shape().to_vec();
                    *packed_shape.last_mut().unwrap() /= 8;
                    assert_eq!(
                        layout.tensor(task.name()).unwrap().local_shape(),
                        packed_shape
                    );

                    assert_eq!(task.output_companions().len(), companions);
                    for companion in task.output_companions() {
                        assert_eq!(
                            target_shapes.0[companion.name()].0,
                            layout
                                .tensor(companion.name())
                                .unwrap()
                                .local_shape()
                                .iter()
                                .map(|n| *n as i32)
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
        assert_eq!(transformed > 8, packed);
    }
}
