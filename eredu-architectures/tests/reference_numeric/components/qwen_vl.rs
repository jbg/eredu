//! Component evidence and masks use retained composite placement and bank authority.
use super::*;

use super::composite_routed::RoutedComponentObserver as Observer;

type Trial = Vec<Vec<(NumericTensor, BTreeMap<String, NumericTensor>)>>;

#[test]
fn qwen_vl_components_follow_media_residency_and_all_parallel_bank_placements() {
    for tied in [false, true] {
        qwen_vl_components_follow_media_residency_and_all_parallel_bank_placements_with_tying(tied);
    }
}

fn qwen_vl_components_follow_media_residency_and_all_parallel_bank_placements_with_tying(
    tied: bool,
) {
    for sparse in [false, true] {
        let mut config = qwen_vl_partition_config(sparse);
        config["tie_word_embeddings"] = tied.into();
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
        for suffix in [
            "input",
            "output",
            "output.effective",
            "residual",
            "residual.effective",
        ] {
            let point = descriptor
                .observations
                .get(&format!("model.language_model.layers.0.deepstack.{suffix}"))
                .unwrap();
            assert!(point.prefill && !point.decode);
            assert!(point
                .requirements
                .contains(&eredu_core::ObservationRequirement::MediaInput));
        }
        let parameters = numeric_composite_parameter_description(&config);
        let inputs = [
            qwen_partition_image_input(),
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
                    eprintln!("Qwen3-VL components tied={tied} sparse={sparse} independent={independent} {topology:?} {residency:?}");
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
                                        eredu_core::residency::OffloadConfig::new(Some(4608), Some(1<<20), 1).unwrap(), 4608, 4608).unwrap())).unwrap();
                                let component_layout = sources.selected().execution().component_partition_layout(descriptor, parameters).unwrap().unwrap();

                                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, ParallelRankTopology::new(topology,rank).unwrap()).unwrap();
                                let mut context = NumericContext::with_partition(layout, rank, world);
                                context.bind_checkpoint_values = true;
                                reset_reference_stage_evidence("SafeTensors");
                                let mut executable = partitioned_adapter::composite(sources, &context).unwrap();
                                let mut rows = 0;
                                let trials: Trial = (0..7).map(|mode| {
                                    executable.reset().unwrap();
                                    let mut masks = vec![];
                                    if matches!(mode,1|3) { masks.push(("model.language_model.layers.0.attention.channels",vec![1],false)); }
                                    if !sparse && matches!(mode,2|3) { masks.push(("model.language_model.layers.1.feed_forward.units",vec![1],true)); }
                                    inputs.iter().enumerate().map(|(step,input)| {
                                        let mut observer = Observer {
                                            dense: GlobalComponentObserver {
                                                inner: NumericLifecycleObserver { zero_path: match mode { 4 => Some("readout.embedding".into()), 5 => Some("readout.normalized".into()), 6 => Some("model.language_model.layers.0.deepstack.output".into()), _ => None }, ..Default::default() },
                                                values: BTreeMap::new(), layout: Some(&component_layout), masks: &masks,
                                                position: if step == 0 { 1 } else { 0 }, projection_plan: None, producer_receipts: None, prediction: step as u64,
                                            },
                                            masked_path: "model.language_model.layers.1.mlp", route_suffix: ".mlp", weight_suffix: ".mlp.experts.down_proj", write_suffix: ".feed_forward.write", path: String::new(), sparse_mask: matches!(mode,2|3), original: None, rows: 0, parameters: None, projected: BTreeMap::new(),
                                        };
                                        let output = executable.forward_observed(input,step==0,&mut observer).unwrap();
                                        rows += observer.rows;
                                        for path in ["readout.embedding.effective", "readout.residual", "readout.normalized", "readout.linear"] {
                                            assert_eq!(observer.dense.values.contains_key(path), component_layout.observation(path).unwrap().coordinates().is_some(), "rank {rank} ownership {path}");
                                        }
                                        let path = "model.language_model.layers.0.deepstack.output.effective";
                                        assert_eq!(observer.dense.values.contains_key(path), component_layout.observation(path).unwrap().coordinates().is_some(), "DeepStack owner rank={rank} step={step}");
                                        // The ordinary transport retains zero placeholders on text-only
                                        // decode. Public plans advertise media/prefill captures only.
                                        if step > 0 { if let Some(value) = observer.dense.values.get(path) { assert!(value.data.iter().all(|v| *v == 0.0)); } }
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
                                    assert!(evidence.peak_bank_bytes <= 4608);
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
                                    "Qwen3-VL media component trial matches serial resident scores",
                                );
                                for path in [
                                    "readout.embedding.effective",
                                    "readout.residual",
                                    "readout.normalized",
                                    "readout.linear",
                                    "model.language_model.layers.0.attention.output.effective",
                                    if sparse {
                                        "model.language_model.layers.0.feed_forward.contribution.effective"
                                    } else {
                                        "model.language_model.layers.0.feed_forward.output.effective"
                                    },
                                    if sparse {
                                        "model.language_model.layers.1.feed_forward.contribution.effective"
                                    } else {
                                        "model.language_model.layers.1.feed_forward.output.effective"
                                    },
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
        for mode in 1..7 {
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
                "nonzero Qwen3-VL causal trial sparse={sparse} mode={mode}"
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
    type Model = qwen::vl::LayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let args = qwen::vl::model_args_from_config_value(config).unwrap();
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
    let mut recipes = qwen::vl::static_recipes(&store);
    for flat in 0..4 {
        recipes.extend(qwen::vl::unit_recipes(&store, &args, flat).unwrap());
    }
    for (target, recipe) in recipes {
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
                .unwrap_or_else(|| panic!("missing Qwen3-VL reference parameter {}", metadata.id()));
            assert_eq!(value.shape, source.shape, "{}", metadata.id());
            value.data.clone_from(&source.data);
        }
    }
    let trials: Trial = (0..8).map(|mode| {
        let mut model = Model::new(args.clone(),&context).unwrap();
        <Model as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut Populate(&values));
        let units = [(0,0),(0,1),(1,0),(1,1)].into_iter().map(|(group,index)| {
            let mut unit = <Model as LayeredArchitecture<NumericBackend,State>>::build_unit(&model,group,index,&context).unwrap();
            unit.visit_parameters_mut(&mut Populate(&values));
            unit
        }).collect();
        let mut runtime = LayerwiseRuntime::new(model,ResidentUnitWindow::new(units));
        let mut state = State::create(qwen::vl::state_layout(&args).unwrap(),|_,policy| Ok::<_,Error>(NumericHybridLayerState::new(policy))).unwrap();
        let mut masks = vec![];
        if matches!(mode,1|3) { masks.push(("model.language_model.layers.0.attention.channels",vec![1],false)); }
        if !sparse && matches!(mode,2|3) { masks.push(("model.language_model.layers.1.feed_forward.units",vec![1],true)); }
        inputs.iter().enumerate().map(|(step,input)| {
            let admitted = eredu_architectures::media_plan::admit_qwen_vl_input(&args,input,&NumericInputInspector).unwrap();
            let ingress = qwen::vl::prepare_input(PreparedCompositeInput::new(input,&admitted).unwrap(),&context).unwrap();
            let mut observer = Observer {
                dense: GlobalComponentObserver {
                    inner: NumericLifecycleObserver { zero_path: match mode { 4 => Some("readout.embedding".into()), 5 => Some("readout.normalized".into()), 6 => Some("model.language_model.layers.0.deepstack.output".into()), _ => None }, ..Default::default() },
                    values: BTreeMap::new(), layout: None, masks: &masks,
                    position: if step == 0 { 1 } else { 0 }, projection_plan: None, producer_receipts: None, prediction: step as u64,
                },
                masked_path: "model.language_model.layers.1.mlp", route_suffix: ".mlp", weight_suffix: ".mlp.experts.down_proj", write_suffix: ".feed_forward.write", path: String::new(), sparse_mask: matches!(mode,2|3), original: None, rows: 0, parameters: Some(&values), projected: BTreeMap::new(),
            };
            let output = ingress.with_model_input(|input| if mode == 7 {
                runtime.forward(input,&mut state,&context)
            } else { runtime.forward_with_observer(input,&mut state,&context,&mut observer) }).unwrap();
            if sparse && mode != 7 { assert!(observer.rows > 0); }
            assert!(observer.projected.is_empty());
            if mode != 7 { reconstruct_score(&observer.dense.values, &values, &args.text); }
            let width = args.text.vocab_size as usize;
            let selected = NumericTensor::new([1,1,args.text.vocab_size],output.data[output.data.len()-width..].to_vec());
            (selected,observer.dense.values)
        }).collect()
    }).collect();
    for (observed, ordinary) in trials[0].iter().zip(&trials[7]) {
        assert_tensor_exact(
            &observed.0,
            &ordinary.0,
            "Qwen3-VL no-op observation preserves ordinary cached inference",
        );
    }
    assert!(
        trials[0][0].1["model.language_model.layers.0.deepstack.output"]
            .data
            .iter()
            .any(|v| v.abs() > 1e-5)
    );
    trials
}

fn reconstruct_score(
    captures: &BTreeMap<String, NumericTensor>,
    parameters: &BTreeMap<String, NumericTensor>,
    args: &qwen::ModelArgs,
) {
    let mut terms = vec![&captures["readout.embedding.effective"]];
    for layer in 0..2 {
        for branch in ["attention", "feed_forward"] {
            let seam = if args.is_moe() && branch == "feed_forward" {
                "contribution"
            } else {
                "output"
            };
            terms.push(
                &captures
                    [&format!("model.language_model.layers.{layer}.{branch}.{seam}.effective")],
            );
        }
    }
    if let Some(deepstack) =
        captures.get("model.language_model.layers.0.deepstack.output.effective")
    {
        terms.push(deepstack);
    }
    for layer in 0..2 {
        let path = format!("model.language_model.layers.{layer}");
        for (seam, weight, target) in [
            (
                "attention.channels",
                "self_attn.o_proj.weight",
                "attention.write",
            ),
            (
                "feed_forward.units",
                "mlp.down_proj.weight",
                "feed_forward.write",
            ),
        ] {
            if args.is_moe() && seam == "feed_forward.units" {
                continue;
            }
            let units = &captures[&format!("{path}.{seam}.effective")];
            let weight = &parameters[&format!("{path}.{weight}")];
            let projected = numeric_linear(units, weight);
            assert_tensor_close(
                &projected,
                &captures[&format!("{path}.{target}")],
                "Qwen3-VL signed component sums reconstruct the actual affine write",
            );
        }
    }
    let residual = &captures["readout.residual"];
    let actual_input = &captures["readout.projection_input"];
    let linear = &captures["readout.linear"];
    let gain = &parameters["model.language_model.norm.weight"];
    let weight = &parameters[&if args.tie_word_embeddings {
        format!("{}.embed_tokens.weight", args.parameter_root)
    } else {
        "lm_head.weight".into()
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
                    "Qwen3-VL residual terms reconstruct"
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
                "Qwen3-VL signed contributions reconstruct affine score"
            );
        }
        let actual_difference = linear.data[token * args.vocab_size as usize]
            - linear.data[token * args.vocab_size as usize + 1];
        assert!(
            (scores[0] - scores[1] - f64::from(actual_difference)).abs() < 3e-5,
            "Qwen3-VL target-alternative score difference"
        );
        reconstructed.extend(scores);
    }
    assert!(reconstructed.iter().all(|score| score.is_finite()));
}

fn numeric_linear(input: &NumericTensor, weight: &NumericTensor) -> NumericTensor {
    let input_width = weight.shape[1] as usize;
    let output_width = weight.shape[0] as usize;
    let mut shape = input.shape.clone();
    *shape.last_mut().unwrap() = output_width as i32;
    let values = input
        .data
        .chunks_exact(input_width)
        .flat_map(|row| {
            weight.data.chunks_exact(input_width).map(move |direction| {
                row.iter()
                    .zip(direction)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>() as f32
            })
        })
        .collect();
    NumericTensor::new(shape, values)
}

#[test]
fn qwen_vl_selected_transforms_retain_source_geometry_and_cached_execution() {
    for sparse in [false, true] {
        let mut config = qwen_vl_partition_config(sparse);
        for (field, value) in [
            ("hidden_size", 64),
            ("intermediate_size", if sparse { 0 } else { 64 }),
            ("moe_intermediate_size", if sparse { 64 } else { 0 }),
            ("vocab_size", 64),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 2),
            ("head_dim", 16),
        ] {
            config["text_config"][field] = value.into();
        }
        config["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!([4, 2, 2]);
        for (field, value) in [
            ("hidden_size", 64),
            ("intermediate_size", 64),
            ("out_hidden_size", 64),
        ] {
            config["vision_config"][field] = value.into();
        }
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
        let args = qwen::vl::model_args_from_config_value(&config).unwrap();
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
                    eprintln!("Qwen3-VL source handoff sparse={sparse} {quantization:?} {topology:?} {residency:?}");
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
                                let (vision,text) = selected.parameters().iter().filter_map(|p| p.executable().weight_quantization().map(|format|(p.name().to_owned(),format))).partition(|(name,_)| name.starts_with("model.visual."));
                                let target_args = qwen::vl::with_checkpoint_formats(args,text,vision).unwrap();
                                let target = qwen::vl::LayeredModel::<NumericBackend>::new(target_args,&NumericContext::default()).unwrap();
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
                                [qwen_partition_image_input(),numeric_text_prepared_input(&[3]),numeric_text_prepared_input(&[4])].iter().enumerate().map(|(step,input)| {
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
                                assert_tensor_close(actual,expected,"Qwen3-VL selected source/target prefill and cached decode across residency and pipeline/expert cuts");
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
