//! Shared-KV component trials across composite residency, TP, PP and expert banks.
//! Partitioned trials expose only submitted K/V; the serial oracle exposes full history.
use super::composite_routed::RoutedComponentObserver as Observer;
use super::*;

type Trial = Vec<Vec<(NumericTensor, BTreeMap<String, NumericTensor>)>>;

fn fixture(sparse: bool) -> serde_json::Value {
    serde_json::json!({
        "model_type":"gemma4_unified", "tie_word_embeddings":false, "image_token_id":5,
        "text_config":{
            "model_type":"gemma4_text", "hidden_size":8, "num_hidden_layers":4,
            "intermediate_size":12, "num_attention_heads":2, "num_key_value_heads":2,
            "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":19,
            "max_position_embeddings":64, "attention_bias":true, "attention_k_eq_v":sparse,
            "num_kv_shared_layers":2,
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
            "sliding_window":4, "enable_moe_block":sparse, "num_experts":sparse.then_some(4),
            "top_k_experts":sparse.then_some(2), "moe_intermediate_size":sparse.then_some(6),
            "hidden_size_per_layer_input":4, "vocab_size_per_layer_input":19,
            "final_logit_softcapping":7.0
        },
        "vision_config":{
            "hidden_size":8, "intermediate_size":12, "num_hidden_layers":2,
            "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4, "patch_size":2,
            "pooling_kernel_size":2, "position_embedding_size":2, "rms_norm_eps":0.00001
        }
    })
}

fn image_input() -> eredu_runtime::PreparedModelInput<NumericTensor> {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
    PreparedModelInput::new(
        vec![
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(NumericTensor::token_ids(&[1, 5])),
                [],
            )
            .unwrap(),
            PreparedInputPart::new_with_extents(
                InputModality::Image,
                PreparedInputPayload::Tensor(NumericTensor::new(
                    [1, 4, 12],
                    (0..48).map(|i| (i as f32 - 24.0) / 100.0).collect(),
                )),
                [
                    (
                        InputMetadataKey::PatchGrid,
                        NumericTensor::new([1, 3], vec![1.0, 2.0, 2.0]),
                    ),
                    (
                        InputMetadataKey::PatchPositions,
                        NumericTensor::new([1, 4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
                    ),
                ],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap(),
        ],
        |tensor| eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor),
    )
    .unwrap()
}

#[test]
fn gemma4_components_transport_padded_image_and_audio_batches() {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
    for sparse in [false, true] {
        let mut config = fixture(sparse);
        config["audio_token_id"] = 6.into();
        config["audio_config"] = serde_json::json!({
            "hidden_size":8, "num_hidden_layers":2, "num_attention_heads":2,
            "output_proj_dims":8, "conv_kernel_size":3, "attention_chunk_size":4,
            "attention_context_left":5, "attention_context_right":0,
            "attention_invalid_logits_value":-1000000000.0, "attention_logit_cap":50.0,
            "residual_weight":0.5, "rms_norm_eps":0.00001, "subsampling_conv_channels":[4,8]
        });
        let mut parts = image_input().parts().to_vec();
        parts.push(
            PreparedInputPart::new_with_extents(
                InputModality::Image,
                PreparedInputPayload::Tensor(NumericTensor::new(
                    [1, 8, 12],
                    (0..96)
                        .map(|i| {
                            if i < 48 {
                                (i as f32 - 20.0) * 0.013
                            } else {
                                9.0
                            }
                        })
                        .collect(),
                )),
                [
                    (
                        InputMetadataKey::PatchGrid,
                        NumericTensor::new([1, 3], vec![1.0, 2.0, 2.0]),
                    ),
                    (
                        InputMetadataKey::PatchPositions,
                        NumericTensor::new(
                            [1, 8, 2],
                            vec![
                                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
                                -1.0, -1.0, -1.0, -1.0,
                            ],
                        ),
                    ),
                ],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap(),
        );
        for (valid, padded) in [(4, 4), (5, 8)] {
            parts.push(
                PreparedInputPart::new_with_extents(
                    InputModality::Audio,
                    PreparedInputPayload::Tensor(NumericTensor::new(
                        [1, padded, 128],
                        (0..padded * 128)
                            .map(|i| {
                                if i < valid * 128 {
                                    (i as f32 % 17.0 - 8.0) * 0.019
                                } else {
                                    7.0
                                }
                            })
                            .collect(),
                    )),
                    [(
                        InputMetadataKey::AudioMask,
                        NumericTensor::new(
                            [1, padded],
                            (0..padded)
                                .map(|i| if i < valid { 1.0 } else { 0.0 })
                                .collect(),
                        ),
                    )],
                    [InputExtent::AudioValidFrames(valid as usize)],
                )
                .unwrap(),
            );
        }
        let input = PreparedModelInput::new(parts, |tensor| {
            eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
        })
        .unwrap();
        // Two encoder batches of eight padded patches require sixteen wire
        // positions, even though the pooled decoder sequence is shorter.
        run_placements(
            config,
            sparse,
            &[(2, 2, if sparse { 2 } else { 1 })],
            input,
            16,
        );
    }
}

fn masks(mode: usize) -> Vec<(&'static str, Vec<u32>, bool)> {
    let mut masks = vec![];
    if matches!(mode, 1 | 3) {
        masks.push((
            "model.language_model.layers.0.attention.channels",
            vec![1],
            false,
        ));
    }
    if matches!(mode, 2 | 3) {
        masks.push((
            "model.language_model.layers.1.dense_feed_forward.units",
            vec![1],
            true,
        ));
    }
    masks
}

fn observer<'a>(
    mode: usize,
    step: usize,
    masks: &'a [(&'static str, Vec<u32>, bool)],
    layout: Option<&'a eredu_architectures::component_partition::ComponentPartitionLayout>,
    parameters: Option<&'a BTreeMap<String, NumericTensor>>,
) -> Observer<'a> {
    Observer {
        dense: GlobalComponentObserver {
            inner: NumericLifecycleObserver {
                zero_path: match mode {
                    4 => Some("readout.embedding".into()),
                    5 => Some("readout.normalized".into()),
                    6 => Some("model.language_model.layers.0.per_layer.output".into()),
                    _ => None,
                },
                ..Default::default()
            },
            values: BTreeMap::new(),
            layout,
            masks,
            position: if step == 0 { 1 } else { 0 },
            projection_plan: None,
            producer_receipts: None,
            prediction: step as u64,
        },
        masked_path: "model.language_model.layers.1.routing",
        route_suffix: ".routing",
        weight_suffix: ".experts.switch_glu.down_proj",
        write_suffix: ".routed_feed_forward.write",
        path: String::new(),
        sparse_mask: matches!(mode, 2 | 3),
        original: None,
        rows: 0,
        parameters,
        projected: BTreeMap::new(),
    }
}

#[test]
fn gemma4_components_cross_shared_kv_cuts_media_and_all_parallel_bank_placements() {
    for sparse in [false, true] {
        run_placements(
            fixture(sparse),
            sparse,
            &[
                (2, 1, 1),
                (1, 2, 1),
                (2, 2, 1),
                (1, 1, 2),
                (2, 1, 2),
                (1, 2, 2),
                (2, 2, 2),
            ],
            image_input(),
            8,
        );
    }
}

#[test]
fn gemma4_components_cross_early_and_shared_pipeline_cuts() {
    for sparse in [false, true] {
        let mut config = fixture(sparse);
        config["text_config"]["num_hidden_layers"] = 8.into();
        config["text_config"]["num_kv_shared_layers"] = 4.into();
        config["text_config"]["layer_types"] = serde_json::json!([
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ]);
        config["vision_config"]["num_hidden_layers"] = 4.into();
        run_placements(
            config,
            sparse,
            &[(1, 4, 1), (2, 4, 1), (1, 4, 2), (2, 4, 2)],
            image_input(),
            8,
        );
    }
}

fn run_placements(
    config: serde_json::Value,
    sparse: bool,
    axes: &[(usize, usize, usize)],
    prefill: eredu_runtime::PreparedModelInput<NumericTensor>,
    maximum_sequence: i32,
) {
    let (artifact, bits) =
        prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
            let seed = name
                .bytes()
                .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            Some(NumericTensor::new(
                shape.to_vec(),
                (0..shape.iter().product::<i32>() as usize)
                    .map(|i| {
                        let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        if name.ends_with("layer_scalar") {
                            1.15
                        } else if name.ends_with("input_min") || name.ends_with("output_min") {
                            -4.0
                        } else if name.ends_with("input_max") || name.ends_with("output_max") {
                            4.0
                        } else if name.contains("norm") && name.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else {
                            delta * 0.025
                        }
                    })
                    .collect(),
            ))
        });
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let descriptor = inspection.architecture_plan().architecture_descriptor();
    let parameters = numeric_composite_parameter_description(&config);
    let inputs = [
        prefill,
        numeric_text_prepared_input(&[3]),
        numeric_text_prepared_input(&[4]),
    ];
    let reference = serial_reference(&config, artifact.path(), bits, &inputs, sparse);
    let bank_budget = if sparse {
        let requirements =
            eredu_architectures::replicated_text::composite_text_requirements(&inspection).unwrap();
        let routed = requirements.routed_execution().unwrap();
        routed
            .bank(eredu_runtime::RoutedBankId::new(0))
            .unwrap()
            .catalog()
            .units()
            .iter()
            .filter_map(ExpertResidencyUnit::byte_len)
            .max()
            .unwrap()
            * routed.routes_per_token() as u64
    } else {
        0
    };
    let mut cases = 0;
    for independent in [false, true] {
        if independent && !sparse {
            continue;
        }
        for &(tp, pp, ep) in axes {
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
                eprintln!("Gemma4 components sparse={sparse} independent={independent} {topology:?} {residency:?}");
                let plan = prepared_adapter::plan(None)
                    .with_topology(topology)
                    .with_residency(residency);
                let world = Arc::new(NumericPartitionWorld::default());
                let results = std::thread::scope(|scope| {
                    let workers=(0..topology.world_size()).map(|rank| {
                            let (inspection,parameters,inputs,plan,descriptor)=(&inspection,&parameters,&inputs,&plan,&descriptor);
                            let world=world.clone();
                            scope.spawn(move || {
                                let sources=partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection,plan,rank,std::time::Duration::from_secs(30),
                                    independent.then(||ParameterBankLoadOptions::new(eredu_core::residency::OffloadConfig::new(Some(bank_budget),Some(1<<20),1).unwrap(),bank_budget,bank_budget).unwrap()),maximum_sequence).unwrap();
                                let component_layout=sources.selected().execution().component_partition_layout(descriptor,parameters).unwrap().unwrap();
                                let layout=eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters,ParallelRankTopology::new(topology,rank).unwrap()).unwrap();
                                let mut context=NumericContext::with_partition(layout,rank,world); context.bind_checkpoint_values=true; context.cache_owned_attention=true;
                                reset_reference_stage_evidence("SafeTensors");
                                let mut executable=partitioned_adapter::composite(sources,&context).unwrap();
                                let mut rows=0;
                                let trials:Trial=(0..7).map(|mode| {
                                    executable.reset().unwrap(); let masks=masks(mode);
                                    inputs.iter().enumerate().map(|(step,input)| {
                                        let mut observer=observer(mode,step,&masks,Some(&component_layout),None);
                                        let output=executable.forward_observed(input,step==0,&mut observer).unwrap(); rows+=observer.rows;
                                        for group in &descriptor.components { for path in [&group.activation,&group.effective_activation] {
                                            assert_eq!(observer.dense.values.contains_key(path),component_layout.observation(path).unwrap().coordinates().is_some(),"Gemma rank={rank} {path}");
                                        } }
                                        (output,observer.dense.values)
                                    }).collect()
                                }).collect();
                                if sparse {assert!(rows>0);}
                                if independent {
                                    let evidence=last_reference_stage_evidence();
                                    assert!(!evidence.bank_acquisitions.is_empty()); assert!(evidence.bank_completions>0);
                                    assert!(evidence.peak_bank_bytes<=bank_budget);
                                    if tp==1 && pp==1 {assert!(evidence.bank_evictions>0);}
                                }
                                trials
                            })
                        }).collect::<Vec<_>>();
                    workers
                        .into_iter()
                        .map(|w| w.join().unwrap())
                        .collect::<Vec<_>>()
                });
                for (rank, trials) in results.into_iter().enumerate() {
                    for (mode, trial) in trials.iter().enumerate() {
                        for (step, (output, values)) in trial.iter().enumerate() {
                            for (path, value) in values {
                                assert!(value.data.iter().all(|value| value.is_finite()),
                                        "Gemma rank={rank} mode={mode} step={step} non-finite capture {path}");
                            }
                            assert_tensor_close(
                                    output,
                                    &reference[mode][step].0,
                                    &format!("Gemma rank={rank} mode={mode} step={step} media and shared-KV scores"),
                                );
                            for (path, actual) in values {
                                if path == "readout.residual"
                                    || path == "readout.linear"
                                    || path.ends_with(".output.effective")
                                    || path.ends_with(".residual.scaled.effective")
                                {
                                    if let Some(expected) = reference[mode][step].1.get(path) {
                                        assert_tensor_close(actual, expected, path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        cases,
        axes.iter().filter(|(_, _, ep)| sparse || *ep == 1).count()
            * 3
            * if sparse { 2 } else { 1 }
    );
    for mode in 1..7 {
        assert!(
            reference[mode].iter().zip(&reference[0]).any(|(a, b)| a
                .0
                .data
                .iter()
                .zip(&b.0.data)
                .any(|(a, b)| (a - b).abs() > 1e-5)),
            "nonzero Gemma trial {mode}"
        );
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
    type Model = gemma4::LayeredModel<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    let mut args =
        gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(config).unwrap()).unwrap();
    if let Some(audio) = args.audio.as_mut() {
        audio.output_projection_bias = bits.contains_key("model.audio_tower.output_proj.bias");
    }
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let store = eredu_checkpoint::store::SafetensorsWeightStore::open(artifact).unwrap();
    let mut parameters = bits
        .into_iter()
        .map(|(name, (shape, bits))| {
            (
                name,
                NumericTensor::new(shape, bits.into_iter().map(f32::from_bits).collect()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let counts = [
        args.vision
            .as_ref()
            .map_or(0, |v| v.num_hidden_layers as usize),
        args.audio
            .as_ref()
            .map_or(0, |a| a.num_hidden_layers as usize),
        args.text.num_hidden_layers(),
    ];
    for flat in 0..counts.iter().sum::<usize>() {
        for (target, recipe) in gemma4::unit_recipes(&store, &args, flat).unwrap() {
            parameters.insert(
                target,
                payload::recipe_value(&recipe, &store, &context).unwrap(),
            );
        }
    }
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let source = self
                .0
                .get(metadata.id.as_str())
                .unwrap_or_else(|| panic!("missing Gemma parameter {}", metadata.id));
            assert_eq!(value.shape, source.shape, "{}", metadata.id);
            value.data.clone_from(&source.data);
        }
    }
    let trials:Trial=(0..9).map(|mode| {
        let context=NumericContext { cache_owned_attention:mode==8, ..context.clone() };
        let mut model=Model::new(args.clone(),&context).unwrap();
        <Model as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut model).visit_parameters_mut(&mut Populate(&parameters));
        let units=counts.iter().enumerate().flat_map(|(group,count)| (0..*count).map(move |index| (group,index))).map(|(group,index)| {
            let mut unit=<Model as LayeredArchitecture<NumericBackend,State>>::build_unit(&model,group,index,&context).unwrap();
            unit.visit_parameters_mut(&mut Populate(&parameters)); unit
        }).collect();
        let mut runtime=LayerwiseRuntime::new(model,ResidentUnitWindow::new(units));
        let mut state=State::create(gemma4::state_layout(&args.text).unwrap(),|_,policy|Ok::<_,Error>(NumericHybridLayerState::new(policy))).unwrap();
        let masks=masks(mode);
        inputs.iter().enumerate().map(|(step,input)| {
            let admitted=<Model as CompositeArchitecture<NumericBackend,State>>::admit_prepared_input(&args,input,&NumericInputInspector).unwrap();
            let ingress=gemma4::prepare_composite_ingress::<NumericBackend>(PreparedCompositeInput::new(input,&admitted).unwrap(),&context).unwrap();
            let mut observer=observer(mode,step,&masks,None,Some(&parameters));
            let parts=ingress.decoder_parts();
            let input=gemma4::ModelInput { parts:&parts,vision:ingress.vision_input(),audio:ingress.audio_input(),per_layer_tokens:None,mask:None };
            let output=if mode>=7 {runtime.forward(input,&mut state,&context)} else {runtime.forward_with_observer(input,&mut state,&context,&mut observer)}.unwrap();
            if sparse && mode<7 {assert!(observer.rows>0);}
            assert!(observer.projected.is_empty());
            if mode<7 {reconstruct(&observer.dense.values,&parameters,&args.text);}
            let width=args.text.vocab_size as usize;
            (NumericTensor::new([1,1,args.text.vocab_size],output.data[output.data.len()-width..].to_vec()),observer.dense.values)
        }).collect()
    }).collect();
    for (observed, ordinary) in trials[0].iter().zip(&trials[7]) {
        assert_tensor_exact(
            &observed.0,
            &ordinary.0,
            "Gemma no-op observed cached execution",
        );
    }
    for (resident, owned) in trials[7].iter().zip(&trials[8]) {
        assert_tensor_close(
            &resident.0,
            &owned.0,
            "Gemma ordinary cache-owned shared history",
        );
    }
    trials
}

fn reconstruct(
    captures: &BTreeMap<String, NumericTensor>,
    parameters: &BTreeMap<String, NumericTensor>,
    args: &gemma4::ModelArgs,
) {
    let width = args.hidden_size as usize;
    let mut terms = vec![captures["readout.embedding.effective"]
        .data
        .iter()
        .map(|v| f64::from(*v))
        .collect::<Vec<_>>()];
    for layer in 0..args.num_hidden_layers() {
        let prefix = format!("model.language_model.layers.{layer}");
        for branch in ["attention", "feed_forward", "per_layer"] {
            terms.push(
                captures[&format!("{prefix}.{branch}.output.effective")]
                    .data
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect(),
            );
        }
        let scale = f64::from(parameters[&format!("{prefix}.layer_scalar")].data[0]);
        assert!((scale - 1.0).abs() > 0.05);
        for term in &mut terms {
            for value in term {
                *value *= scale;
            }
        }
    }
    let residual = &captures["readout.residual"];
    let head = &parameters["lm_head.weight"];
    let gain = &parameters["model.language_model.norm.weight"];
    let input = &captures["readout.projection_input"];
    let linear = &captures["readout.linear"];
    for (token, row) in residual.data.chunks_exact(width).enumerate() {
        let denominator = (row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / width as f64
            + f64::from(args.rms_norm_eps))
        .sqrt();
        let mut scores = [0.0; 2];
        for channel in 0..width {
            let base = terms
                .iter()
                .map(|t| t[token * width + channel])
                .sum::<f64>();
            assert!(
                (base - f64::from(row[channel])).abs() < 5e-5,
                "Gemma whole-residual scale includes all prior writes"
            );
            let factor = f64::from(gain.data[channel]) / denominator;
            for target in 0..2 {
                let direction = f64::from(head.data[target * width + channel]);
                scores[target] += base * factor * direction;
                scores[target] += (f64::from(input.data[token * width + channel])
                    - f64::from(row[channel]) * factor)
                    * direction;
            }
        }
        for target in 0..2 {
            assert!(
                (scores[target]
                    - f64::from(linear.data[token * args.vocab_size as usize + target]))
                .abs()
                    < 5e-5
            );
        }
        assert!(
            (scores[0]
                - scores[1]
                - f64::from(
                    linear.data[token * args.vocab_size as usize]
                        - linear.data[token * args.vocab_size as usize + 1]
                ))
            .abs()
                < 7e-5
        );
    }
}

#[test]
fn gemma4_attention_bias_schemas_follow_projection_and_shared_state_ownership() {
    for sparse in [false, true] {
        for enabled in [false, true] {
            let mut config = fixture(sparse);
            config["text_config"]["attention_bias"] = enabled.into();
            config["text_config"]["num_key_value_heads"] = 1.into();
            let family =
                gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
            let safe = gemma4::safetensors_plan(&family).unwrap();
            let gguf = gemma4::gguf_plan(&family.text).unwrap();
            let safe = safe
                .common_tensors
                .iter()
                .filter(|t| {
                    t.key.starts_with("model.language_model.layers.") && t.key.ends_with(".bias")
                })
                .map(|t| {
                    (
                        t.key
                            .replace("model.language_model.layers.", "model.layers."),
                        t.shape.clone(),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let gguf = gguf
                .common_tensors
                .iter()
                .filter(|t| t.key.ends_with(".bias"))
                .map(|t| (gemma4::translate_gguf_weight_name(&t.key), t.shape.clone()))
                .collect::<BTreeMap<_, _>>();
            let mut expected = BTreeMap::new();
            if enabled {
                for layer in 0..4 {
                    for (field, width) in
                        [("q_proj", 8), ("o_proj", 8), ("k_proj", 4), ("v_proj", 4)]
                    {
                        if (layer >= 2 && matches!(field, "k_proj" | "v_proj"))
                            || (sparse && layer == 1 && field == "v_proj")
                        {
                            continue;
                        }
                        expected.insert(
                            format!("model.layers.{layer}.self_attn.{field}.bias"),
                            vec![width],
                        );
                    }
                }
            }
            assert_eq!(safe, expected, "SafeTensors actual bias ownership");
            assert_eq!(gguf, expected, "GGUF actual bias ownership");
        }
    }
}

#[test]
fn gemma4_selected_transforms_retain_source_geometry_and_cached_execution() {
    for sparse in [false, true] {
        let mut config = fixture(sparse);
        for (field, value) in [
            ("hidden_size", 64),
            ("intermediate_size", 128),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 2),
            ("head_dim", 16),
            ("hidden_size_per_layer_input", 64),
        ] {
            config["text_config"][field] = value.into();
        }
        if sparse {
            config["text_config"]["moe_intermediate_size"] = 64.into();
        }
        for (field, value) in [
            ("hidden_size", 64),
            ("intermediate_size", 128),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 2),
            ("head_dim", 16),
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
                            if name.ends_with("layer_scalar") {
                                1.15
                            } else if name.ends_with("input_min") || name.ends_with("output_min") {
                                -4.0
                            } else if name.ends_with("input_max") || name.ends_with("output_max") {
                                4.0
                            } else if name.contains("norm") && name.ends_with("weight") {
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
        let args =
            gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
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
                    eprintln!("Gemma4 source handoff sparse={sparse} {quantization:?} {topology:?} {residency:?}");
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
                                let formats = selected.parameters().iter().filter_map(|p| p.executable().weight_quantization().map(|format|(p.name().to_owned(),format))).collect();
                                let target_args = gemma4::with_checkpoint_formats(args,formats).unwrap();
                                let target = gemma4::LayeredModel::<NumericBackend>::new(target_args,&NumericContext::default()).unwrap();
                                let target_parameters = target.parameter_description(&NumericContext::default()).unwrap();
                                let rank_topology = ParallelRankTopology::new(topology,rank).unwrap();
                                let source = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters,rank_topology).unwrap();
                                let encoded = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(&target_parameters,rank_topology).unwrap();
                                let mut layout = eredu_runtime::derive_transform_source_layout(&source, &encoded).unwrap();
                                for (name,companion) in encoded.tensors() { if !layout.contains(name) { layout.insert(name.to_owned(),companion.clone()); } }
                                let mut context = NumericContext::with_partition(layout,rank,world);
                                context.bind_checkpoint_values = true; context.cache_owned_attention = true;
                                // The common visitor asserts that every selected transform retains
                                // a typed source architecture before any materialization work.
                                let mut executable = partitioned_adapter::composite(sources,&context).unwrap();
                                [image_input(),numeric_text_prepared_input(&[3]),numeric_text_prepared_input(&[4])].iter().enumerate().map(|(step,input)| {
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
                                assert_tensor_close(actual,expected,"Gemma4 selected source/target prefill and cached decode across residency and pipeline/expert cuts");
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
