//! Typed Inkling prediction operations preserve effective values and lane ownership.
use super::*;
use eredu_architectures::inkling;
use eredu_runtime::ActivationObserver;

type Model = inkling::LayeredModel<NumericBackend>;
type Extension = MaterializedInklingPrediction<NumericBackend, Materializer>;

struct Invoke<'a> {
    model: &'a mut Model,
    state: &'a mut State,
    context: &'a NumericContext,
}
impl PredictionOperationInvoker<Model, NumericBackend, State> for Invoke<'_> {
    type Error = String;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, String>
    where
        O: eredu_runtime::PredictionTargetOperation<Model, NumericBackend, State>,
    {
        operation
            .apply(self.model, self.state, None, self.context)
            .map_err(|error| error.to_string())
    }
    fn invalid(message: String) -> String {
        message
    }
}

#[derive(Default)]
struct Observe {
    values: BTreeMap<String, NumericTensor>,
    zero: Option<String>,
    fail: Option<String>,
}
impl ActivationObserver<NumericTensor, Error> for Observe {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        assert!(self.values.insert(path.into(), value.clone()).is_none());
        if self.fail.as_deref() == Some(path) {
            return Err(Error::backend(
                "injected Inkling prediction observer failure",
            ));
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok((self.zero.as_deref() == Some(path)).then(|| {
            let mut edited = value.clone();
            let width = *edited.shape.last().unwrap() as usize;
            let selected = edited.data.len() - width + 1;
            assert_ne!(edited.data[selected], 0.0);
            edited.data[selected] = 0.0;
            edited
        }))
    }
}

// Learned global scales are order-one multipliers, not small matrix entries.
// Distinct causal taps keep every selected component numerically visible.
struct Initialize;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Initialize {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        if metadata.id.as_str().ends_with("global_scale") {
            value.data.fill(1.3);
        }
        if metadata.id.as_str().contains("sconv") {
            for (i, value) in value.data.iter_mut().enumerate() {
                *value = [0.11, -0.07, 0.23][i % 3] + (i / 3) as f32 * 0.002;
            }
        }
    }
}

fn hidden(sequence: i32, step: i32) -> NumericTensor {
    NumericTensor::new(
        [1, sequence, 8],
        (0..sequence * 8)
            .map(|i| ((i * 7 + step * 3) % 23) as f32 / 13.0 - 0.5)
            .collect(),
    )
}

#[test]
fn inkling_materialized_prediction_observations_preserve_scoring_and_replay() {
    for chain_norm in [false, true] {
        let mut config = routed_inkling_partition_fixture();
        config.as_object_mut().unwrap().remove("vision_config");
        config["text_config"]["layer_types"] =
            serde_json::json!(["full_attention", "sliding_attention"]);
        config["text_config"]["logits_mup_width_multiplier"] = serde_json::json!(1.7);
        config["text_config"]["unpadded_vocab_size"] = serde_json::json!(13);
        config["mtp_config"] = serde_json::json!({
            "num_nextn_predict_layers": 2, "local_layer_ids": [1],
            "chain_hidden_post_norm": chain_norm,
        });
        let args = inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
        let context = NumericContext {
            bind_checkpoint_values: true,
            ..Default::default()
        };
        let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        assert_eq!(descriptor.component_scopes.len(), 2);
        let mut target_args = args.clone();
        target_args.mtp_config = None;
        let mut model = Model::new(target_args, &context).unwrap();
        let mut target_state = State::create(model.ingress_state_layout().unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
        let target_before = target_state.clone();
        let state = State::create(
            inkling::mtp_state_layout(&args).unwrap().unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap();
        let mut prediction = inkling::MtpModel::new(&args, &context).unwrap().unwrap();
        prediction.visit_parameters_mut(&mut Initialize);
        let mut extension = Extension {
            units: prediction.layers.into_iter().map(Module).collect(),
            shared: prediction
                .chain_norm
                .map(|chain_norm| Module(inkling::MtpShared { chain_norm })),
            state: PredictionState {
                state,
                snapshots: Rc::new(RefCell::new(Vec::new())),
            },
        };
        let output_weight =
            <Model as LayeredArchitecture<NumericBackend, State>>::static_modules(&model)
                .output
                .weight
                .clone();
        let mut parameters = Parameters::default();
        <Extension as MaterializedPredictionExecutor<Model, NumericBackend, Materializer>>::visit_modules(&mut extension, &mut parameters).unwrap();
        <Model as LayeredArchitecture<NumericBackend, State>>::static_modules(&model)
            .visit_parameters(&mut parameters);
        let mut invoker = Invoke {
            model: &mut model,
            state: &mut target_state,
            context: &context,
        };
        assert!(<Extension as MaterializedPredictionExecutor<
            Model,
            NumericBackend,
            Materializer,
        >>::supports_internal_observations(&extension));
        let empty = <Extension as MaterializedPredictionExecutor<
            Model,
            NumericBackend,
            Materializer,
        >>::new_state(&extension);
        let mut ordinary = empty.clone();
        let mut observed = empty.clone();
        let prefix = hidden(3, 0);
        let tokens = NumericTensor::token_ids(&[1, 3, 2]);
        extension
            .prefill::<State, _>(&mut invoker, &prefix, &prefix, &tokens, &mut ordinary)
            .unwrap();
        let mut observer = Observe::default();
        extension
            .prefill_observed::<State, _>(
                &mut invoker,
                &prefix,
                &prefix,
                &tokens,
                &mut observed,
                Some(&mut observer),
            )
            .unwrap();
        assert_state_exact(
            &ordinary.state,
            &observed.state,
            2,
            "Inkling observed adapter prefill",
        );
        for depth in 0..2 {
            let root = format!("model.mtp.layers.{depth}");
            assert_eq!(
                observer.values[&format!("{root}.prediction.readout.linear")].shape,
                [1, 3, 13]
            );
            assert_eq!(
                observer.values[&format!("{root}.transformer_block.feed_forward.units")].shape[1],
                3
            );
        }
        for depth in 0..2 {
            let root = format!("model.mtp.layers.{depth}");
            let trial = |extension: &mut Extension,
                         invoker: &mut Invoke<'_>,
                         mode: usize,
                         instrumented: bool| {
                let mut lane = ordinary.clone();
                let suffix = match mode {
                    0 => None,
                    1 => Some("prediction.hidden.first_normalized"),
                    2 => Some("prediction.hidden.normalized"),
                    3 => Some("prediction.embedding.normalized"),
                    4 => Some("prediction.fusion.output"),
                    5 => Some("transformer_block.attention.channels"),
                    6 => Some("transformer_block.feed_forward.units"),
                    7 => Some("prediction.readout.normalized"),
                    8 => Some("prediction.readout.linear"),
                    _ => unreachable!(),
                };
                let mut scores = Vec::new();
                for step in 1..=2 {
                    let mut observer = Observe {
                        zero: suffix.map(|suffix| format!("{root}.{suffix}")),
                        ..Default::default()
                    };
                    let hidden = hidden(1, step);
                    let tokens = NumericTensor::token_ids(&[4 + step as usize]);
                    let (logits, capture) = if instrumented {
                        extension.logits_observed::<State, _>(
                            invoker,
                            &hidden,
                            &tokens,
                            depth,
                            &mut lane,
                            Some(&mut observer),
                        )
                    } else {
                        extension.logits::<State, _>(invoker, &hidden, &tokens, depth, &mut lane)
                    }
                    .unwrap();
                    if instrumented {
                        verify_scope(&descriptor, depth, &observer, &parameters, &tokens);
                        let get = |suffix: &str| &observer.values[&format!("{root}.{suffix}")];
                        assert_tensor_exact(
                            &capture,
                            get("prediction.readout.normalized.effective"),
                            "Inkling prediction capture",
                        );
                        let scaled = capture.multiply_scalar(1.0 / 1.7, &context).unwrap();
                        assert_tensor_exact(
                            &scaled,
                            get("prediction.readout.scaled"),
                            "Inkling muP scale before projection input arithmetic",
                        );
                        let scores = linear(
                            get("prediction.readout.projection_input"),
                            &output_weight,
                            None,
                        )
                        .unwrap()
                        .axis_slice(2, 0, 13);
                        assert_tensor_close(
                            &scores,
                            get("prediction.readout.linear"),
                            "Inkling adapter score reconstruction",
                        );
                        assert_tensor_exact(
                            &logits,
                            get("prediction.readout.linear.effective"),
                            "Inkling effective scores",
                        );
                    }
                    scores.push(logits);
                }
                assert_eq!(lane.state.layer(depth).unwrap().position(), 5);
                assert_eq!(lane.state.layer(1 - depth).unwrap().position(), 3);
                (scores, lane)
            };
            let (baseline, baseline_lane) = trial(&mut extension, &mut invoker, 0, false);
            let (captured, captured_lane) = trial(&mut extension, &mut invoker, 0, true);
            assert_state_exact(
                &baseline_lane.state,
                &captured_lane.state,
                2,
                "Inkling adapter no-op cache",
            );
            for (a, b) in baseline.iter().zip(&captured) {
                assert_tensor_exact(a, b, "Inkling adapter no-op scores");
            }
            for mode in 1..=8 {
                let (masked, _) = trial(&mut extension, &mut invoker, mode, true);
                for (a, b) in baseline.iter().zip(&masked) {
                    assert_ne!(a.data, b.data, "causal adapter mask {mode}, depth {depth}");
                }
            }
            let mut failed = ordinary.clone();
            let mut observer = Observe {
                fail: Some(format!("{root}.prediction.readout.linear")),
                ..Default::default()
            };
            assert!(extension
                .logits_observed::<State, _>(
                    &mut invoker,
                    &hidden(1, 1),
                    &NumericTensor::token_ids(&[5]),
                    depth,
                    &mut failed,
                    Some(&mut observer)
                )
                .unwrap_err()
                .contains("injected Inkling"));
            assert_eq!(failed.state.layer(depth).unwrap().position(), 4);
            let (replay, _) = trial(&mut extension, &mut invoker, 0, true);
            for (a, b) in baseline.iter().zip(&replay) {
                assert_tensor_exact(a, b, "Inkling replay after masks and failed sibling");
            }
        }
        assert_state_exact(
            &target_before,
            invoker.state,
            target_before.layout().len(),
            "prediction preserves target state",
        );
        assert!(extension
            .logits::<State, _>(
                &mut invoker,
                &hidden(1, 1),
                &NumericTensor::token_ids(&[5]),
                2,
                &mut observed
            )
            .unwrap_err()
            .contains("exceeds"));
    }
}

#[derive(Default)]
struct Modules(BTreeMap<usize, BTreeSet<String>>);
impl PredictionModuleVisitor<NumericBackend, Materializer> for Modules {
    type Error = std::convert::Infallible;
    fn visit<M: Parameterized<NumericTensor>>(
        &mut self,
        ordinal: usize,
        module: &mut Module<M>,
    ) -> Result<(), Self::Error> {
        let names = eredu_nn::validate_parameter_topology(module.as_mut())
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.id.to_string())
            .collect();
        assert!(self.0.insert(ordinal, names).is_none());
        Ok(())
    }
}

#[test]
fn inkling_prepared_prediction_depths_bind_exact_selected_formats() {
    for chain_norm in [false, true] {
        for quantized in [false, true] {
            let mut config = routed_inkling_partition_fixture();
            config.as_object_mut().unwrap().remove("vision_config");
            config["text_config"]["model_max_length"] = 128.into();
            config["text_config"]["hidden_size"] = 32.into();
            config["text_config"]["head_dim"] = 16.into();
            config["text_config"]["dense_intermediate_size"] = 32.into();
            config["text_config"]["mlp_layer_types"] = serde_json::json!(["dense", "dense"]);
            config["text_config"]["layer_types"] =
                serde_json::json!(["full_attention", "sliding_attention"]);
            config["mtp_config"] = serde_json::json!({
                "num_nextn_predict_layers": 2, "local_layer_ids": [1],
                "chain_hidden_post_norm": chain_norm,
            });
            let (root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = prepared_adapter::plan(quantized.then_some(
                eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                },
            ))
            .with_drafting(eredu_core::DraftingPlan::Embedded {
                max_draft_tokens: 2,
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
            let admitted = eredu_core::ModelPreparationPlan::from_retained_admission(
                inspection,
                selected.admission(),
            )
            .unwrap();
            let sources =
                eredu_architectures::prepared_sources::prepare_model_sources(admitted, selected)
                    .unwrap();
            let extension = sources.prediction_extension().unwrap();
            let tasks = sources
                .selected()
                .text_realization()
                .auxiliary_materialization_tasks();
            for task in tasks
                .iter()
                .filter(|task| task.executable() != eredu_checkpoint::LinearFormat::Dense)
            {
                assert_eq!(
                    task.role(),
                    eredu_runtime::ReplicatedTextParameterRole::LinearWeight,
                    "{}",
                    task.name()
                );
            }
            let relative_tables = tasks
                .iter()
                .filter(|task| task.name().ends_with(".self_attn.rel_proj"))
                .collect::<Vec<_>>();
            assert_eq!(relative_tables.len(), 2);
            for table in relative_tables {
                assert_eq!(
                    table.role(),
                    eredu_runtime::ReplicatedTextParameterRole::Other
                );
                assert_eq!(table.executable(), eredu_checkpoint::LinearFormat::Dense);
                assert!(table.output_companions().is_empty());
            }
            let context = NumericContext {
                bind_checkpoint_values: true,
                ..Default::default()
            };
            let prepare = || {
                prepare_replicated_prediction_extension::<NumericBackend>(
                    extension, tasks, &context, &context,
                )
                .unwrap()
            };
            let PreparedPredictionExtension::Inkling { units, shared, .. } = prepare() else {
                panic!("Inkling extension")
            };
            assert_eq!(units.len(), 2);
            assert_eq!(shared.is_some(), chain_norm);
            let mut expected = BTreeMap::new();
            let mut selected_names = BTreeSet::new();
            for (depth, unit) in units.into_iter().enumerate() {
                let (source, local, tasks) = unit.into_parts();
                assert!(source.input_projection.format_companions.is_empty());
                assert_eq!(
                    local.input_projection.format_companions.len(),
                    if quantized { 2 } else { 0 }
                );
                let names = eredu_nn::validate_parameter_topology(&local)
                    .unwrap()
                    .into_iter()
                    .map(|parameter| parameter.id.to_string())
                    .collect::<BTreeSet<_>>();
                let owned = tasks
                    .iter()
                    .flat_map(|task| {
                        std::iter::once(task.name()).chain(
                            task.output_companions()
                                .iter()
                                .map(|companion| companion.name()),
                        )
                    })
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>();
                assert_eq!(owned, names);
                assert!(selected_names.is_disjoint(&owned));
                selected_names.extend(owned);
                expected.insert(depth + 1, names);
            }
            if let Some(shared) = shared {
                let (_, local, tasks) = shared.into_parts();
                assert_eq!(tasks.len(), 1);
                let names = eredu_nn::validate_parameter_topology(&local)
                    .unwrap()
                    .into_iter()
                    .map(|parameter| parameter.id.to_string())
                    .collect::<BTreeSet<_>>();
                assert!(selected_names.is_disjoint(&names));
                selected_names.extend(names.iter().cloned());
                expected.insert(0, names);
            }
            let all_tasks = tasks
                .iter()
                .flat_map(|task| {
                    std::iter::once(task.name()).chain(
                        task.output_companions()
                            .iter()
                            .map(|companion| companion.name()),
                    )
                })
                .map(str::to_owned)
                .collect::<BTreeSet<_>>();
            assert_eq!(selected_names, all_tasks);
            let materialized = prepare()
                .materialize::<Materializer>(&mut Materialization {
                    source: sources.extension().unwrap().clone(),
                    context: &context,
                    snapshots: Rc::new(RefCell::new(Vec::new())),
                })
                .unwrap();
            let mut extension =
                <Model as MaterializedPredictionTarget<NumericBackend>>::pair_prediction_extension(
                    materialized,
                )
                .unwrap();
            let mut modules = Modules::default();
            <Extension as MaterializedPredictionExecutor<Model, NumericBackend, Materializer>>::visit_modules(&mut extension, &mut modules).unwrap();
            assert_eq!(modules.0, expected);
            let args =
                inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
            let mut model = Model::new(args, &context).unwrap();
            let mut target_state =
                State::create(model.ingress_state_layout().unwrap(), |_, policy| {
                    Ok::<_, Error>(NumericHybridLayerState::new(policy))
                })
                .unwrap();
            let mut invoker = Invoke {
                model: &mut model,
                state: &mut target_state,
                context: &context,
            };
            let empty = <Extension as MaterializedPredictionExecutor<
                Model,
                NumericBackend,
                Materializer,
            >>::new_state(&extension);
            let mut lane = empty.clone();
            let prefix = NumericTensor::new(
                [1, 3, 32],
                (0..96)
                    .map(|i| ((i * 7) % 23) as f32 / 13.0 - 0.5)
                    .collect(),
            );
            let tokens = NumericTensor::token_ids(&[1, 3, 2]);
            extension
                .prefill::<State, _>(&mut invoker, &prefix, &prefix, &tokens, &mut lane)
                .unwrap();
            let hidden = NumericTensor::new([1, 1, 32], prefix.data[64..].to_vec());
            let token = NumericTensor::token_ids(&[4]);
            let mut advanced = lane.clone();
            let mut observed_advanced = lane.clone();
            let mut advance_trace = Observe::default();
            extension
                .advance::<State, _>(&mut invoker, &hidden, &token, &mut advanced)
                .unwrap();
            extension
                .advance_observed::<State, _>(
                    &mut invoker,
                    &hidden,
                    &token,
                    &mut observed_advanced,
                    Some(&mut advance_trace),
                )
                .unwrap();
            let point = advance_trace
                .values
                .keys()
                .find(|path| path.ends_with("attention.input"))
                .unwrap()
                .clone();
            let mut masked_advanced = lane.clone();
            let mut mask_trace = Observe {
                zero: Some(point),
                ..Default::default()
            };
            extension
                .advance_observed::<State, _>(
                    &mut invoker,
                    &hidden,
                    &token,
                    &mut masked_advanced,
                    Some(&mut mask_trace),
                )
                .unwrap();
            let mut changed_after_replay = false;
            for depth in 0..2 {
                let (expected, expected_hidden) = extension
                    .logits::<State, _>(&mut invoker, &hidden, &token, depth, &mut advanced)
                    .unwrap();
                let (actual, actual_hidden) = extension
                    .logits::<State, _>(
                        &mut invoker,
                        &hidden,
                        &token,
                        depth,
                        &mut observed_advanced,
                    )
                    .unwrap();
                assert_tensor_exact(
                    &actual,
                    &expected,
                    "observed Inkling accepted-token replay state",
                );
                assert_tensor_exact(
                    &actual_hidden,
                    &expected_hidden,
                    "Inkling replay causal histories",
                );
                let (masked, _) = extension
                    .logits::<State, _>(&mut invoker, &hidden, &token, depth, &mut masked_advanced)
                    .unwrap();
                changed_after_replay |= masked
                    .data
                    .iter()
                    .zip(&expected.data)
                    .any(|(a, b)| (a - b).abs() > 1e-6);
            }
            assert!(
                changed_after_replay,
                "replay interventions affect later prediction through retained state"
            );
            for depth in 0..2 {
                let mut ordinary = lane.clone();
                let mut replay = lane.clone();
                let (logits, capture) = extension
                    .logits::<State, _>(&mut invoker, &hidden, &token, depth, &mut ordinary)
                    .unwrap();
                let mut observer = Observe::default();
                let (observed, observed_capture) = extension
                    .logits_observed::<State, _>(
                        &mut invoker,
                        &hidden,
                        &token,
                        depth,
                        &mut replay,
                        Some(&mut observer),
                    )
                    .unwrap();
                assert!(logits.data.iter().all(|value| value.is_finite()));
                assert!(logits.data.iter().any(|value| value.abs() > 1e-4));
                assert_tensor_exact(&logits, &observed, "prepared Inkling depth replay");
                assert_tensor_exact(
                    &capture,
                    &observed_capture,
                    "prepared Inkling continuation replay",
                );
                assert!(!observer.values.is_empty());
            }
        }
    }
}

#[derive(Default)]
struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> eredu_nn::ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
        assert!(self
            .0
            .insert(metadata.id.to_string(), value.clone())
            .is_none());
    }
}
impl PredictionModuleVisitor<NumericBackend, Materializer> for Parameters {
    type Error = std::convert::Infallible;
    fn visit<M: Parameterized<NumericTensor>>(
        &mut self,
        _: usize,
        module: &mut Module<M>,
    ) -> Result<(), Self::Error> {
        module.as_mut().visit_parameters(self);
        Ok(())
    }
}

fn normalize(
    value: &NumericTensor,
    norm: &eredu_core::component::ComponentNormalization,
    parameters: &Parameters,
) -> NumericTensor {
    use eredu_core::component::ComponentNormalizationKind;
    assert_eq!(norm.groups, 1);
    let width = *value.shape.last().unwrap() as usize;
    let data = value
        .data
        .chunks_exact(width)
        .flat_map(|row| {
            let denominator = match norm.kind {
                ComponentNormalizationKind::Identity => 1.0,
                ComponentNormalizationKind::Rms => {
                    (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / width as f64
                        + norm.epsilon.value() as f64)
                        .sqrt()
                }
                _ => panic!("Inkling normalization"),
            };
            row.iter().enumerate().map(move |(i, value)| {
                let gain = norm.gain.as_ref().map_or(1.0, |name| {
                    parameters.0[name].data[i] as f64 + norm.gain_offset.value() as f64
                });
                let bias = norm
                    .bias
                    .as_ref()
                    .map_or(0.0, |name| parameters.0[name].data[i] as f64);
                (*value as f64 / denominator * gain + bias) as f32
            })
        })
        .collect();
    NumericTensor::new(value.shape.clone(), data)
}

fn verify_scope(
    descriptor: &eredu_core::ArchitectureDescriptor,
    depth: usize,
    observer: &Observe,
    parameters: &Parameters,
    tokens: &NumericTensor,
) {
    use eredu_core::component::*;
    let scope = &descriptor.component_scopes[depth];
    assert_eq!(scope.components.len(), 2);
    assert!(scope.execution_groups.is_empty());
    assert!(scope
        .static_parameter_roles
        .iter()
        .any(|role| role == "embedding_norm"));
    for component in &scope.components {
        assert!(observer.values.contains_key(&component.activation));
        assert!(observer
            .values
            .contains_key(&component.effective_activation));
        assert!(parameters.0.contains_key(&component.write_weight));
    }
    for transform in &descriptor.component_transforms {
        if transform.node_id.starts_with(&scope.node_id) {
            assert!(
                observer.values.contains_key(&transform.input),
                "{}",
                transform.input
            );
            assert!(
                observer.values.contains_key(&transform.output),
                "{}",
                transform.output
            );
            if let ComponentTensorTransformEquation::Normalization { normalization } =
                &transform.equation
            {
                let expected = normalize(
                    &observer.values[&transform.input],
                    normalization,
                    parameters,
                );
                assert_tensor_close(
                    &expected,
                    &observer.values[&transform.output],
                    "declared first hidden normalization",
                );
            }
        }
    }
    let ComponentResidualBase::LinearFusion {
        inputs,
        weight,
        projection_input,
        output,
        effective_output,
        ..
    } = &scope.residual_base
    else {
        panic!("Inkling fusion")
    };
    let width = parameters.0[weight].shape[0] as usize;
    let mut combined = vec![0.0; width * 2];
    for input in inputs {
        let source = match &input.source {
            ComponentFusionSource::Observation { path } => observer.values[path].clone(),
            ComponentFusionSource::TokenEmbedding {
                weight,
                scale,
                normalization,
                ..
            } => {
                let token = tokens.data[0] as usize;
                let table = &parameters.0[weight];
                let lookup = NumericTensor::new(
                    [1, 1, width as i32],
                    table.data[token * width..(token + 1) * width]
                        .iter()
                        .map(|v| v * scale.value())
                        .collect(),
                );
                normalize(
                    &lookup,
                    normalization
                        .as_ref()
                        .expect("target embedding normalization"),
                    parameters,
                )
            }
        };
        let expected = normalize(&source, &input.normalization, parameters);
        assert_tensor_close(
            &expected,
            &observer.values[input.output.trim_end_matches(".effective")],
            "declared prediction fusion normalization",
        );
        combined[input.columns.clone()].copy_from_slice(&observer.values[&input.output].data);
    }
    let combined = NumericTensor::new([1, 1, (width * 2) as i32], combined);
    assert_tensor_exact(
        &combined,
        &observer.values[projection_input],
        "declared hidden-first fusion input",
    );
    let fused = linear(&combined, &parameters.0[weight], None).unwrap();
    assert_tensor_close(
        &fused,
        &observer.values[output],
        "declared fusion projection",
    );
    let base = &observer.values[effective_output];
    let mut residual = base.data.iter().map(|v| *v as f64).collect::<Vec<_>>();
    assert_eq!(scope.readout.other_writes.len(), 2);
    for write in &scope.readout.other_writes {
        for (sum, value) in residual
            .iter_mut()
            .zip(&observer.values[&write.effective_output].data)
        {
            *sum += *value as f64 * write.residual_scale.value() as f64;
        }
    }
    let residual = NumericTensor::new(
        base.shape.clone(),
        residual.into_iter().map(|v| v as f32).collect(),
    );
    assert_tensor_close(
        &residual,
        &observer.values[&scope.readout.residual],
        "declared causal prediction residual sum",
    );
    let normalized = normalize(
        &observer.values[&format!("{}.effective", scope.readout.residual)],
        &scope.readout.normalization,
        parameters,
    );
    assert_tensor_close(
        &normalized,
        &observer.values[&scope.readout.normalized],
        "declared optional chain normalization",
    );
}
