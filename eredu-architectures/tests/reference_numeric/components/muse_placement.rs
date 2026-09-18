//! Component evidence and masks use retained composite placement and bank authority.
use super::*;

struct Observer<'a> {
    dense: GlobalComponentObserver<'a>,
    path: String,
    sparse_mask: bool,
    original: Option<NumericTensor>,
    rows: usize,
    parameters: Option<&'a BTreeMap<String, NumericTensor>>,
    projected: BTreeMap<String, Vec<f64>>,
}
impl ActivationObserver<NumericTensor, Error> for Observer<'_> {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if let Some(sum) = self.projected.remove(path) {
            assert_tensor_close(
                &NumericTensor::new(
                    value.shape.clone(),
                    sum.into_iter().map(|v| v as f32).collect(),
                ),
                value,
                "Muse selected sparse components reconstruct the complete write before postnorm",
            );
        }
        self.dense.observe(path, value)
    }
    fn observe_replica(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.dense.observe_replica(path, value)
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        self.dense.intervene(path, value)
    }
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<NumericTensor>>, Error> {
        self.path = path.into();
        if let Some(layout) = self.dense.layout {
            assert!(layout
                .routed_observation(&format!("{path}.units"))
                .unwrap()
                .ownership()
                .is_some());
        }
        Ok(Some(self))
    }
}
impl Observer<'_> {
    fn edit(
        &self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Option<NumericTensor> {
        if !self.sparse_mask || self.path != "model.layers.1.routing" {
            return None;
        }
        let mut value = batch.units.values.clone();
        let width = value.shape[1] as usize;
        let selected_width = *batch.units.coefficients.shape.last().unwrap() as usize;
        let coordinates = batch.unit_coordinates;
        for row in 0..value.shape[0] as usize {
            let token = batch.units.token_indices.data[row] as usize;
            let slot = batch.units.selection_indices.data[row] as usize % selected_width;
            if batch.route_origin(token, slot).unwrap().token != self.dense.position {
                continue;
            }
            for column in 0..width {
                if coordinates.map_or(Some(column), |map| map.local_to_global(column)) != Some(1) {
                    value.data[row * width + column] = 0.0;
                }
            }
        }
        Some(value)
    }
}
impl eredu_runtime::RoutedUnitObserver<NumericTensor> for Observer<'_> {
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        self.rows += batch.units.values.shape[0] as usize;
        assert!(self.original.replace(batch.units.values.clone()).is_none());
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok(self.edit(batch))
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, NumericTensor>,
    ) -> Result<(), Error> {
        let original = self.original.take().unwrap();
        let before = eredu_runtime::RoutedUnitBatch {
            units: batch.units.with_values(&original),
            ..*batch
        };
        let expected = self.edit(&before).unwrap_or(original);
        assert_tensor_exact(
            batch.units.values,
            &expected,
            "Muse sparse effective values follow global token/unit coordinates",
        );
        if let Some(parameters) = self.parameters {
            let prefix = self.path.strip_suffix(".routing").unwrap();
            let weight = &parameters[&format!("{prefix}.mlp.experts.down_proj")];
            let hidden = weight.shape[1] as usize;
            let width = weight.shape[2] as usize;
            assert_eq!(batch.units.values.shape[1] as usize, width);
            let sum = self
                .projected
                .entry(format!("{prefix}.feed_forward.write"))
                .or_insert_with(|| vec![0.0; batch.units.total_token_count * hidden]);
            let routes = *batch.units.coefficients.shape.last().unwrap() as usize;
            let source_routes = *batch.source_groups.shape.last().unwrap() as usize;
            for (row, units) in batch.units.values.data.chunks_exact(width).enumerate() {
                let native_token = batch.units.token_indices.data[row] as usize;
                let selected = batch.units.selection_indices.data[row] as usize;
                let slot = selected % routes;
                let source_token = batch.source_token(native_token).unwrap();
                let expert = batch
                    .global_group(
                        batch.source_groups.data[source_token * source_routes + slot] as usize,
                    )
                    .unwrap();
                let token = batch.route_origin(native_token, slot).unwrap().token;
                let coefficient = f64::from(batch.units.coefficients.data[selected]);
                for channel in 0..hidden {
                    sum[token * hidden + channel] += coefficient
                        * units
                            .iter()
                            .zip(
                                &weight.data[(expert * hidden + channel) * width
                                    ..(expert * hidden + channel + 1) * width],
                            )
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>();
                }
            }
        }
        Ok(())
    }
}

type Trial = Vec<Vec<(NumericTensor, BTreeMap<String, NumericTensor>)>>;

#[test]
fn muse_components_follow_media_residency_and_all_parallel_bank_placements() {
    for tied in [false, true] {
        muse_components_follow_media_residency_and_all_parallel_bank_placements_with_tying(tied);
    }
}

fn muse_components_follow_media_residency_and_all_parallel_bank_placements_with_tying(tied: bool) {
    for sparse in [false, true] {
        let mut config = if sparse {
            routed_muse_partition_fixture()
        } else {
            dense_muse_partition_fixture()
        };
        // Force PP to cut inside vision, then transport only projected media
        // back to the decoder. Patches, media positions and text extent differ.
        config["vision_config"]["num_hidden_layers"] = 3.into();
        config["vision_config"]["layer_types"] =
            serde_json::json!(["full_attention", "full_attention", "full_attention"]);
        config["text_config"]["output_multiplier"] = 1.9.into();
        config["text_config"]["tie_word_embeddings"] = tied.into();
        let (artifact, bits) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                let count = shape.iter().map(|n| *n as usize).product();
                Some(NumericTensor::new(
                    shape.to_vec(),
                    (0..count)
                        .map(|i| {
                            let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                            if name.contains("norm") && name.ends_with("weight") {
                                0.9 + delta * 0.002
                            } else {
                                delta * 0.025
                            }
                        })
                        .collect(),
                ))
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let descriptor = inspection.architecture_plan().architecture_descriptor();
        let parameters = numeric_composite_parameter_description(&config);
        let inputs = [
            muse_partition_image_input(),
            numeric_text_prepared_input(&[3]),
            numeric_text_prepared_input(&[4]),
        ];
        let reference = serial_reference(&config, artifact.path(), bits, &inputs, sparse);
        let mut cases = 0;
        for independent in [false, true] {
            if independent && !sparse {
                continue;
            }
            for (tp, pp, ep) in [
                (2, 1, 1),
                (1, 2, 1),
                (2, 2, 1),
                (1, 1, 2),
                (2, 1, 2),
                (1, 2, 2),
                (2, 2, 2),
            ] {
                if !sparse && ep > 1 {
                    continue;
                }
                let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
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
                    cases += 1;
                    eprintln!("Muse components tied={tied} sparse={sparse} independent={independent} {topology:?} {residency:?}");
                    let plan = prepared_adapter::plan(None)
                        .with_topology(topology)
                        .with_residency(residency);
                    let world = Arc::new(NumericPartitionWorld::default());
                    let results = std::thread::scope(|scope| {
                        let workers = (0..topology.world_size()).map(|rank| {
                            let (inspection, parameters, inputs, plan, descriptor) = (&inspection, &parameters, &inputs, &plan, &descriptor);
                            let world = world.clone();
                            scope.spawn(move || {
                                let sources = partitioned_adapter::prepare_plan_with_banks(inspection, plan, rank, std::time::Duration::from_secs(30),
                                    independent.then(|| ParameterBankLoadOptions::new(
                                        eredu_core::residency::OffloadConfig::new(Some(1152), Some(1<<20), 1).unwrap(), 1152, 1152).unwrap())).unwrap();
                                let component_layout = sources.selected().execution().component_partition_layout(descriptor, parameters).unwrap().unwrap();
                                if sparse && pp == 2 && rank == 0 {
                                    for wrong_shape in [false, true] {
                                        let mut invalid = descriptor.clone();
                                        let transform = invalid.component_transforms.iter_mut().find(|t| t.input == "model.layers.0.feed_forward.write.effective").unwrap();
                                        if wrong_shape {
                                            let input = transform.input.clone();
                                            invalid.observations.points.iter_mut().find(|p| p.path == input).unwrap().axes.as_mut().unwrap()[2].dimension = eredu_core::SymbolicDimension::Known(7);
                                        } else {
                                            let eredu_core::component::ComponentTensorTransformEquation::Normalization { normalization } = &mut transform.equation else { unreachable!() };
                                            normalization.gain = Some("model.layers.1.post_feedforward_layernorm.weight".into());
                                        }
                                        assert!(sources.selected().execution().component_partition_layout(&invalid, parameters).is_err(), "postnorm cannot borrow foreign geometry or ownership");
                                    }
                                }

                                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, ParallelRankTopology::new(topology,rank).unwrap()).unwrap();
                                let mut context = NumericContext::with_partition(layout, rank, world);
                                context.bind_checkpoint_values = true;
                                reset_reference_stage_evidence("SafeTensors");
                                let mut executable = partitioned_adapter::composite(sources, &context).unwrap();
                                let mut rows = 0;
                                let trials: Trial = (0..6).map(|mode| {
                                    executable.reset().unwrap();
                                    let mut masks = vec![];
                                    if matches!(mode,1|3) { masks.push(("model.layers.0.attention.channels",vec![1],false)); }
                                    if !sparse && matches!(mode,2|3) { masks.push(("model.layers.1.feed_forward.units",vec![1],true)); }
                                    inputs.iter().enumerate().map(|(step,input)| {
                                        let mut observer = Observer {
                                            dense: GlobalComponentObserver {
                                                inner: NumericLifecycleObserver { zero_path: match mode { 4 => Some("readout.embedding".into()), 5 => Some("readout.normalized".into()), _ => None }, ..Default::default() },
                                                values: BTreeMap::new(), layout: Some(&component_layout), masks: &masks,
                                                position: if step == 0 { 2 } else { 0 }, projection_plan: None, producer_receipts: None, prediction: step as u64,
                                            },
                                            path: String::new(), sparse_mask: matches!(mode,2|3), original: None, rows: 0, parameters: None, projected: BTreeMap::new(),
                                        };
                                        let output = executable.forward_observed(input,step==0,&mut observer).unwrap();
                                        rows += observer.rows;
                                        for path in ["readout.embedding.effective", "readout.residual", "readout.normalized", "readout.projection_input", "readout.linear"] {
                                            assert_eq!(observer.dense.values.contains_key(path), component_layout.observation(path).unwrap().coordinates().is_some(), "rank {rank} ownership {path}");
                                        }
                                        for group in &descriptor.components {
                                            for path in [&group.activation, &group.effective_activation] {
                                                assert_eq!(observer.dense.values.contains_key(path), component_layout.observation(path).unwrap().coordinates().is_some(), "rank {rank} ownership {path}");
                                            }
                                        }
                                        (output, observer.dense.values)
                                    }).collect()
                                }).collect();
                                if sparse { assert!(rows > 0); }
                                if independent {
                                    let evidence = last_reference_stage_evidence();
                                    assert!(!evidence.bank_acquisitions.is_empty());
                                    assert!(evidence.bank_completions > 0);
                                    assert!(evidence.peak_bank_bytes <= 1152);
                                    if tp == 1 && pp == 1 { assert!(evidence.bank_evictions > 0); }
                                }
                                trials
                            })
                        }).collect::<Vec<_>>();
                        workers
                            .into_iter()
                            .map(|w| w.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    for trials in &results {
                        for (mode, trial) in trials.iter().enumerate() {
                            for (step, (output, values)) in trial.iter().enumerate() {
                                assert_tensor_close(
                                    output,
                                    &reference[mode][step].0,
                                    "Muse media component trial matches serial resident scores",
                                );
                                for path in [
                                    "readout.embedding.effective",
                                    "readout.residual",
                                    "readout.normalized",
                                    "readout.linear",
                                    "model.layers.0.attention.output.effective",
                                    "model.layers.0.feed_forward.output.effective",
                                    "model.layers.1.feed_forward.output.effective",
                                ] {
                                    if let Some(actual) = values.get(path) {
                                        assert_tensor_close(
                                            actual,
                                            &reference[mode][step].1[path],
                                            path,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, if sparse { 42 } else { 9 });
        for mode in 1..6 {
            assert!(
                reference[mode]
                    .iter()
                    .zip(&reference[0])
                    .any(|(trial, base)| trial
                        .0
                        .data
                        .iter()
                        .zip(&base.0.data)
                        .any(|(a, b)| (a - b).abs() > 1e-5)),
                "nonzero Muse causal trial sparse={sparse} mode={mode}"
            );
        }
    }
}

fn serial_reference(
    config: &serde_json::Value,
    artifact: &std::path::Path,
    bits: prepared_adapter::ParameterBits,
    inputs: &[eredu_runtime::PreparedModelInput<NumericTensor>],
    sparse: bool,
) -> Trial {
    use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
    type Model = muse_glimmer::LayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let args = muse_glimmer::DecoderConfig::from_hf_value(config).unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let store = eredu_checkpoint::store::SafetensorsWeightStore::open(artifact).unwrap();
    let mut values = bits
        .into_iter()
        .map(|(name, (shape, bits))| {
            (
                name,
                NumericTensor::new(shape, bits.into_iter().map(f32::from_bits).collect()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (target, recipe) in muse_glimmer::safetensors_recipes(&args, &store).unwrap() {
        values.insert(
            target,
            payload::recipe_value(&recipe, &store, &context).unwrap(),
        );
    }
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            let source = self
                .0
                .get(metadata.id().as_str())
                .unwrap_or_else(|| panic!("missing Muse reference parameter {}", metadata.id()));
            assert_eq!(value.shape, source.shape, "{}", metadata.id());
            value.data.clone_from(&source.data);
        }
    }
    (0..6).map(|mode| {
        let mut model = Model::new(args.clone(),&context).unwrap();
        <Model as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut Populate(&values));
        let units = [(0,args.vision_config.as_ref().unwrap().layer_count()),(1,args.num_hidden_layers as usize)]
            .into_iter().flat_map(|(group,count)| (0..count).map(move |index| (group,index))).map(|(group,index)| {
            let mut unit = <Model as LayeredArchitecture<NumericBackend,State>>::build_unit(&model,group,index,&context).unwrap();
            unit.visit_parameters_mut(&mut Populate(&values));
            unit
        }).collect();
        let mut runtime = LayerwiseRuntime::new(model,ResidentUnitWindow::new(units));
        let mut state = State::create(muse_glimmer::state_layout(&args).unwrap(),|_,policy| Ok::<_,Error>(NumericHybridLayerState::new(policy))).unwrap();
        let mut masks = vec![];
        if matches!(mode,1|3) { masks.push(("model.layers.0.attention.channels",vec![1],false)); }
        if !sparse && matches!(mode,2|3) { masks.push(("model.layers.1.feed_forward.units",vec![1],true)); }
        inputs.iter().enumerate().map(|(step,input)| {
            let admitted = <Model as CompositeArchitecture<NumericBackend,State>>::admit_prepared_input(&args,input,&NumericInputInspector).unwrap();
            let ingress = muse_glimmer::prepare_composite_ingress::<NumericBackend>(PreparedCompositeInput::new(input,&admitted).unwrap(),&context).unwrap();
            let parts = ingress.decoder_parts();
            let mut observer = Observer {
                dense: GlobalComponentObserver {
                    inner: NumericLifecycleObserver { zero_path: match mode { 4 => Some("readout.embedding".into()), 5 => Some("readout.normalized".into()), _ => None }, ..Default::default() },
                    values: BTreeMap::new(), layout: None, masks: &masks,
                    position: if step == 0 { 2 } else { 0 }, projection_plan: None, producer_receipts: None, prediction: step as u64,
                },
                path: String::new(), sparse_mask: matches!(mode,2|3), original: None, rows: 0, parameters: Some(&values), projected: BTreeMap::new(),
            };
            let output = runtime.forward_with_observer(muse_glimmer::ModelInput { parts: &parts, vision: ingress.vision_input(), mask: None },&mut state,&context,&mut observer).unwrap();
            if sparse { assert!(observer.rows > 0); }
            assert!(observer.projected.is_empty());
            reconstruct_score(&observer.dense.values, &values, &args);
            let width = args.vocab_size as usize;
            let selected = NumericTensor::new([1,1,args.vocab_size],output.data[output.data.len()-width..].to_vec());
            (selected,observer.dense.values)
        }).collect()
    }).collect()
}

fn reconstruct_score(
    captures: &BTreeMap<String, NumericTensor>,
    parameters: &BTreeMap<String, NumericTensor>,
    args: &muse_glimmer::DecoderConfig,
) {
    let mut terms = vec![&captures["readout.embedding.effective"]];
    for layer in 0..2 {
        for branch in ["attention", "feed_forward"] {
            terms.push(&captures[&format!("model.layers.{layer}.{branch}.output.effective")]);
        }
    }
    let residual = &captures["readout.residual"];
    let actual_input = &captures["readout.projection_input"];
    let linear = &captures["readout.linear"];
    let gain = &parameters["model.norm.weight"];
    let weight = &parameters[if args.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    }];
    let width = args.hidden_size as usize;
    let mut reconstructed = vec![];
    for (token, row) in residual.data.chunks_exact(width).enumerate() {
        let denominator = (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / width as f64
            + f64::from(args.rms_norm_eps))
        .sqrt();
        let mut scores = [0.0; 2];
        for vocabulary in 0..2 {
            for channel in 0..width {
                let base = terms
                    .iter()
                    .map(|term| f64::from(term.data[token * width + channel]))
                    .sum::<f64>();
                assert!(
                    (base - f64::from(row[channel])).abs() < 2e-5,
                    "Muse residual terms reconstruct"
                );
                let direction = f64::from(weight.data[vocabulary * width + channel]);
                let factor = f64::from(gain.data[channel]) / denominator;
                scores[vocabulary] += base * factor * direction;
                // Explicit effective-input correction also accounts for a readout mask.
                scores[vocabulary] += (f64::from(actual_input.data[token * width + channel])
                    - f64::from(row[channel]) * factor)
                    * direction;
            }
            assert!(
                (scores[vocabulary]
                    - f64::from(linear.data[token * args.vocab_size as usize + vocabulary]))
                .abs()
                    < 2e-5,
                "Muse signed contributions reconstruct affine score"
            );
        }
        let actual_difference = linear.data[token * args.vocab_size as usize]
            - linear.data[token * args.vocab_size as usize + 1];
        assert!(
            (scores[0] - scores[1] - f64::from(actual_difference)).abs() < 3e-5,
            "Muse target-alternative score difference"
        );
        reconstructed.extend(scores);
    }
    assert!(reconstructed.iter().all(|score| score.is_finite()));
}

#[test]
fn muse_decoder_ingress_remains_pending_through_vision_execution() {
    use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
    type Model = muse_glimmer::LayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let args = muse_glimmer::DecoderConfig::from_hf_value(&dense_muse_partition_fixture()).unwrap();
    let mut groups = muse_glimmer::static_parameter_groups(&args).unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        groups.extend(muse_glimmer::layer_parameter_groups(&args, layer).unwrap());
    }
    let layout = numeric_local_layout(&groups, 1, 0).unwrap();
    let context = NumericContext::with_local_layout(layout.clone());
    let geometry = muse_glimmer::local_geometry(&args, &layout).unwrap();
    let mut state = State::create(geometry.state_layout().clone(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let mut model = Model::new_parallel(args.clone(), geometry, &context).unwrap();
    let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let input = muse_partition_image_input();
    let admitted = <Model as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
        &args,
        &input,
        &NumericInputInspector,
    )
    .unwrap();
    let mut forward =
        <Model as CompositeArchitecture<NumericBackend, State>>::begin_composite_forward_parallel(
            &mut model,
            PreparedCompositeInput::new(&input, &admitted).unwrap(),
            &mut state,
            &parallel,
            &context,
        )
        .unwrap();
    let pending = |model: &Model, forward: &muse_glimmer::ForwardContext<NumericTensor>| {
        <Model as CompositeArchitecture<NumericBackend, State>>::primary_ingress_collectives_pending(
            model, forward,
        )
    };
    assert!(pending(&model, &forward.context));
    let mut vision = <Model as ParallelLayeredArchitecture<NumericBackend, State>>::begin_execution_group_parallel(
        &mut model, 0, &forward.hidden, &[], &mut state, &mut forward.context, &parallel, &context,
    ).unwrap();
    for unit_index in 0..args.vision_config.as_ref().unwrap().layer_count() {
        let mut unit = <Model as LayeredArchitecture<NumericBackend, State>>::build_unit(
            &model, 0, unit_index, &context,
        )
        .unwrap();
        vision =
            <Model as ParallelLayeredArchitecture<NumericBackend, State>>::forward_unit_parallel(
                &mut model,
                0,
                unit_index,
                &mut unit,
                &vision,
                &mut state,
                &mut forward.context,
                &parallel,
                &context,
            )
            .unwrap();
    }
    let vision = <Model as ParallelLayeredArchitecture<NumericBackend, State>>::complete_execution_group_parallel(
        &mut model, 0, &vision, &mut state, &mut forward.context, &parallel, &context,
    ).unwrap();
    assert!(
        pending(&model, &forward.context),
        "vision completion does not perform deferred token lookups"
    );
    let decoder = <Model as ParallelLayeredArchitecture<NumericBackend, State>>::begin_execution_group_parallel(
        &mut model, 1, &forward.hidden, &[&vision], &mut state, &mut forward.context, &parallel, &context,
    ).unwrap();
    assert!(!pending(&model, &forward.context));
    assert_eq!(decoder.shape(), [1, 3, args.hidden_size]);
    assert!(decoder.data.iter().all(|value| value.is_finite()));
    assert!(decoder.data.iter().any(|value| value.abs() > 1e-5));
}

#[test]
fn muse_embedding_waves_follow_token_segments_and_exclude_media_positions() {
    use eredu_architectures::composite_execution::{
        CompositeArchitecture, CompositeTensorCollective, PreparedCompositeInput,
    };
    type Model = muse_glimmer::LayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    for sparse in [false, true] {
        let args = muse_glimmer::DecoderConfig::from_hf_value(&if sparse {
            routed_muse_partition_fixture()
        } else {
            dense_muse_partition_fixture()
        })
        .unwrap();
        let architecture = Model::new(args.clone(), &NumericContext::default()).unwrap();
        let mixed = muse_partition_image_input();
        for (parts, expected) in [
            (mixed.parts().to_vec(), vec![1, 1]),
            (vec![mixed.parts()[1].clone()], vec![]),
            (vec![mixed.parts()[0].clone()], vec![1]),
        ] {
            let input = eredu_runtime::PreparedModelInput::new(parts, |tensor| {
                eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
            })
            .unwrap();
            let admitted =
                <Model as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
                    &args,
                    &input,
                    &NumericInputInspector,
                )
                .unwrap();
            let prepared = PreparedCompositeInput::new(&input, &admitted).unwrap();
            let sums = <Model as CompositeArchitecture<NumericBackend, State>>::prepared_primary_ingress_collectives(
                &architecture, prepared, 2, None,
            ).unwrap().unwrap();
            assert_eq!(
                sums,
                expected
                    .into_iter()
                    .map(|positions| CompositeTensorCollective::Sum {
                        shape: vec![1, positions, args.hidden_size],
                    })
                    .collect::<Vec<_>>()
            );
            assert!(<Model as CompositeArchitecture<NumericBackend, State>>::prepared_primary_ingress_collectives(
                &architecture, prepared, 1, None,
            ).unwrap().is_none());
        }
    }
}

#[test]
fn muse_selected_transforms_retain_source_geometry_and_cached_execution() {
    for sparse in [false, true] {
        let mut config = if sparse {
            routed_muse_partition_fixture()
        } else {
            dense_muse_partition_fixture()
        };
        for (field, value) in [
            ("hidden_size", 64),
            ("intermediate_size", 64),
            ("moe_intermediate_size", if sparse { 64 } else { 0 }),
            ("vocab_size", 64),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 2),
            ("head_dim", 16),
        ] {
            config["text_config"][field] = value.into();
        }
        config["text_config"]["output_multiplier"] = 1.9.into();
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                Some(NumericTensor::new(
                    shape.to_vec(),
                    (0..shape.iter().product::<i32>() as usize)
                        .map(|i| {
                            let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                            if name.contains("norm") && name.ends_with("weight") {
                                0.9 + delta * 0.002
                            } else {
                                delta * 0.008
                            }
                        })
                        .collect(),
                ))
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let args = muse_glimmer::DecoderConfig::from_hf_value(&config).unwrap();
        let parameters = numeric_composite_parameter_description(&config);
        // This scalar payload binder models affine transforms. Native MXFP4
        // behavior is covered by the separate MLX component matrix.
        for quantization in [eredu_core::QuantizationRequest::Affine {
            group_size: 32,
            bits: 4,
        }] {
            let mut reference: Option<Vec<NumericTensor>> = None;
            for (tp, pp, ep) in [(2, 1, 1), (2, 2, if sparse { 2 } else { 1 })] {
                let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
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
                    eprintln!("Muse source handoff sparse={sparse} {quantization:?} {topology:?} {residency:?}");
                    let plan = prepared_adapter::plan(Some(quantization))
                        .with_topology(topology)
                        .with_residency(residency);
                    let world = Arc::new(NumericPartitionWorld::default());
                    let results = std::thread::scope(|scope| {
                        let workers = (0..topology.world_size()).map(|rank| {
                            let (inspection,plan,args,parameters) = (&inspection,&plan,&args,&parameters);
                            let world = world.clone();
                            scope.spawn(move || {
                                let sources = partitioned_adapter::prepare_plan(inspection,plan,rank,std::time::Duration::from_secs(30)).unwrap();
                                let selected = sources.selected().execution().text_realization();
                                assert!(selected.parameters().iter().any(|p| matches!(p.lowering(),eredu_runtime::WeightLoweringKind::Transform|eredu_runtime::WeightLoweringKind::DerivedTransform)));
                                let target_args = muse_glimmer::with_checkpoint_formats(args,selected.parameters().iter().filter_map(|p| p.executable().weight_quantization().map(|format|(p.name().to_owned(),format))).collect()).unwrap();
                                let target = muse_glimmer::LayeredModel::<NumericBackend>::new(target_args,&NumericContext::default()).unwrap();
                                let target_parameters = target.parameter_description(&NumericContext::default()).unwrap().into_owned();
                                let rank_topology = ParallelRankTopology::new(topology,rank).unwrap();
                                let source = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters,rank_topology).unwrap();
                                let encoded = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(&target_parameters,rank_topology).unwrap();
                                let mut layout = eredu_runtime::derive_transform_source_layout(&source, &encoded).unwrap();
                                for (name,companion) in encoded.tensors() { if !layout.contains(name) { layout.insert(name.to_owned(),companion.clone()); } }
                                let mut context = NumericContext::with_partition(layout,rank,world);
                                context.bind_checkpoint_values = true;
                                // The common visitor asserts that every selected transform retains
                                // a typed source architecture before any materialization work.
                                let mut executable = partitioned_adapter::composite(sources,&context).unwrap();
                                [muse_partition_image_input(),numeric_text_prepared_input(&[3]),numeric_text_prepared_input(&[4])].iter().enumerate().map(|(step,input)| {
                                    executable.forward_observed(input,step==0,&mut NumericLifecycleObserver::default()).unwrap()
                                }).collect::<Vec<_>>()
                            })
                        }).collect::<Vec<_>>();
                        workers
                            .into_iter()
                            .map(|w| w.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    for outputs in results {
                        assert!(outputs.iter().all(|v| v.data.iter().all(|x| x.is_finite())));
                        assert!(outputs
                            .iter()
                            .any(|v| v.data.iter().any(|x| x.abs() > 1e-5)));
                        if let Some(reference) = &reference {
                            for (actual, expected) in outputs.iter().zip(reference) {
                                assert_tensor_close(actual,expected,"Muse selected source/target prefill and cached decode across residency and pipeline/expert cuts");
                            }
                        } else {
                            reference = Some(outputs);
                        }
                    }
                }
            }
        }
    }
}
