//! Genuine inverse-rotated V4 channels, before either grouped write factor.
use super::*;
use eredu_runtime::{PartitionedLayeredArchitecture, ResidentExpertProvider};

#[path = "deepseek_v4/dspark.rs"]
mod dspark;
#[path = "deepseek_v4/prediction.rs"]
mod prediction;

#[derive(Default)]
struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
        self.0.insert(metadata.id.as_str().into(), value.clone());
    }
}

impl eredu_runtime::StaticParameterVisitor<NumericBackend> for Parameters {
    type Error = Error;
    fn visit<M: Parameterized<NumericTensor>>(&mut self, _: &str, module: &M) -> Result<(), Error> {
        module.visit_parameters(self);
        Ok(())
    }
}

#[test]
fn v4_measured_stream_equation_reconstructs_target_scores_and_differences() {
    let config = tiny_v4_config();
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let readout = graph.component_readout.as_ref().unwrap();
    let mixing = readout.stream_residual.as_ref().unwrap();
    let args = tiny_v4_args();
    let context = NumericContext::default();
    let layout = deepseek::v4::state_layout(&args).unwrap();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(layout.clone(), |layer, _| {
            let ratios = match args.attention_policy(layer).unwrap() {
                deepseek::V4AttentionPolicy::Local => vec![],
                deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
                deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
            };
            Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &ratios))
        })
        .unwrap()
    };
    let model = deepseek::v4::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
    type V4State = DeviceState<NumericBackend, NumericPoolingCache>;
    let hooks = [
        <deepseek::v4::Model<NumericBackend> as LayeredArchitecture<NumericBackend, V4State>>::observation_hooks(&model),
        <deepseek::v4::Model<NumericBackend> as eredu_runtime::ParallelLayeredArchitecture<NumericBackend, V4State>>::parallel_observation_hooks(&model),
        <deepseek::v4::Model<NumericBackend> as PartitionedLayeredArchitecture<NumericBackend, V4State>>::partition_observation_hooks(&model, false),
        <deepseek::v4::Model<NumericBackend> as PartitionedLayeredArchitecture<NumericBackend, V4State>>::partition_observation_hooks(&model, true),
    ];
    for hooks in hooks {
        for site in [
            eredu_runtime::inspection::ObservationHookSite::Input,
            eredu_runtime::inspection::ObservationHookSite::Unit,
            eredu_runtime::inspection::ObservationHookSite::Readout,
        ] {
            assert!(hooks.supports(site), "V4 target {site:?}");
        }
    }
    let mut parameters = Parameters::default();
    eredu_runtime::ArchitectureParameters::visit_static_parameters(&model, &mut parameters)
        .unwrap();
    let mut runtime = ResidentRuntime::new(model, &context).unwrap();
    let mut ordinary = ResidentRuntime::new(
        deepseek::v4::Model::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    let mut state = make_state();
    let mut original_state = make_state();
    for ids in [&[1, 2, 3, 4, 5, 6, 7, 8][..], &[9][..], &[10][..]] {
        let tokens = NumericTensor::token_ids(ids);
        let mut capture = Components::strict();
        let (scores, _) = runtime
            .forward_with_traversal_hook(
                deepseek::mtp::EmbeddedInput::target(&tokens, None),
                &mut state,
                &context,
                &mut capture,
            )
            .unwrap();
        let original = ordinary
            .forward(
                deepseek::mtp::EmbeddedInput::target(&tokens, None),
                &mut original_state,
                &context,
            )
            .unwrap();
        assert_tensor_exact(&scores, &original, "V4 complete observed target readout");
        let base = match &mixing.base {
            eredu_core::component::ComponentStreamBase::Broadcast { input } => {
                &capture.values[input]
            }
            _ => panic!("broadcast target embedding"),
        };
        let hidden = args.hidden_size as usize;
        let streams = mixing.streams;
        let rows = ids.len();
        // Keep base and each whole sublayer write as independent signed terms;
        // channel/FFN tests separately prove each write's scalar decomposition.
        let mut terms = vec![(0..rows * streams * hidden)
            .map(|i| base.data[(i / (streams * hidden)) * hidden + i % hidden] as f64)
            .collect::<Vec<_>>()];
        for cycle in &mixing.cycles {
            let coefficients = &capture.values[&cycle.combination];
            for term in &mut terms {
                let mut mixed = vec![0.0; term.len()];
                for row in 0..rows {
                    for to in 0..streams {
                        for h in 0..hidden {
                            mixed[(row * streams + to) * hidden + h] = (0..streams)
                                .map(|from| {
                                    term[(row * streams + from) * hidden + h]
                                        * coefficients.data[(row * streams + from) * streams + to]
                                            as f64
                                })
                                .sum();
                        }
                    }
                }
                *term = mixed;
            }
            let post = &capture.values[&cycle.post];
            let write = &capture.values[&cycle.write];
            terms.push(
                (0..rows * streams * hidden)
                    .map(|i| {
                        let row = i / (streams * hidden);
                        post.data[i / hidden] as f64 * write.data[row * hidden + i % hidden] as f64
                    })
                    .collect(),
            );
            let actual = &capture.values[&cycle.output];
            for i in 0..actual.data.len() {
                assert!(
                    (terms.iter().map(|t| t[i]).sum::<f64>() - actual.data[i] as f64).abs() < 2e-4,
                    "stream propagation {}",
                    cycle.output
                );
            }
        }
        let collapse = &capture.values[&mixing.head.coefficients];
        let residual = &capture.values[&readout.residual];
        let norm = &parameters.0[readout.normalization.gain.as_ref().unwrap()];
        let head = &parameters.0[&readout.weight];
        for row in 0..rows {
            let rms = (residual.data[row * hidden..(row + 1) * hidden]
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                / hidden as f64
                + args.rms_norm_eps as f64)
                .sqrt();
            let mut reconstructed = [0.0; 2];
            for term in &terms {
                for (selected, token) in [2usize, 7].into_iter().enumerate() {
                    let contribution: f64 = (0..hidden)
                        .map(|h| {
                            let collapsed: f64 = (0..streams)
                                .map(|s| {
                                    term[(row * streams + s) * hidden + h]
                                        * collapse.data[row * streams + s] as f64
                                })
                                .sum();
                            collapsed / rms
                                * norm.data[h] as f64
                                * head.data[token * hidden + h] as f64
                        })
                        .sum();
                    reconstructed[selected] += contribution;
                }
            }
            let expected = [
                scores.data[row * 16 + 2] as f64,
                scores.data[row * 16 + 7] as f64,
            ];
            for (actual, expected) in reconstructed.iter().zip(expected) {
                assert!((actual - expected).abs() < 3e-4);
            }
            assert!(
                ((reconstructed[0] - reconstructed[1]) - (expected[0] - expected[1])).abs() < 3e-4
            );
        }
    }
}

fn reconstruct(
    capture: &Components,
    first: &NumericTensor,
    last: &NumericTensor,
    groups: usize,
    rank: usize,
) {
    let channels = &capture.values["attention.channels.effective"];
    let input = &capture.values["attention.grouped_input"];
    let projection = &capture.values["attention.projection"];
    let tokens = channels.shape[1] as usize;
    let count = channels.shape[2] as usize;
    let per_group = count / groups;
    assert_eq!(
        input.shape,
        [1, groups as i32, tokens as i32, per_group as i32]
    );
    assert_eq!(projection.shape, [1, tokens as i32, (groups * rank) as i32]);
    let hidden = last.shape[0] as usize;
    let mut output = vec![0.0; tokens * hidden];
    for token in 0..tokens {
        for group in 0..groups {
            for channel in 0..per_group {
                let activation = channels.data[token * count + group * per_group + channel];
                assert_eq!(
                    input.data[(group * tokens + token) * per_group + channel],
                    activation
                );
                for out in 0..hidden {
                    let column: f64 = (0..rank)
                        .map(|k| {
                            last.data[out * groups * rank + group * rank + k] as f64
                                * first.data[(group * rank + k) * per_group + channel] as f64
                        })
                        .sum();
                    output[token * hidden + out] += activation as f64 * column;
                }
            }
        }
    }
    assert_tensor_exact(
        projection,
        &capture.values["attention.projection_input"],
        "dense second-factor input",
    );
    assert_tensor_close(
        &NumericTensor::new(
            vec![1, tokens as i32, hidden as i32],
            output.into_iter().map(|n| n as f32).collect(),
        ),
        &capture.values["attention.write"],
        "grouped scalar columns reconstruct V4 attention write",
    );
}

#[test]
fn v4_grouped_channels_reconstruct_and_mask_all_attention_policies_with_cached_steps() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    for (layer, length, ratios) in [(0, 9, vec![]), (1, 9, vec![4, 4]), (2, 130, vec![128])] {
        let mut attention =
            deepseek::attention::v4::Attention::<NumericBackend>::new(&args, layer, &context)
                .unwrap();
        let mut parameters = Parameters::default();
        attention.visit_parameters(&mut parameters);
        let first = &parameters.0[&format!("layers.{layer}.attn.wo_a.weight")];
        let last = &parameters.0[&format!("layers.{layer}.attn.wo_b.weight")];
        let mut cache = NumericPoolingCache::new(args.sliding_window, &ratios);
        for (step, tokens) in [(0, length), (1, 1), (2, 1)] {
            let input = NumericTensor::new(
                vec![1, tokens, 4],
                (0..tokens * 4)
                    .map(|i| ((i + step * 5) as f32 * 0.37 - 1.3).sin())
                    .collect(),
            );
            let mut before = cache.clone();
            let ordinary = attention
                .forward(&input, None, Some(&mut cache), &context)
                .unwrap();
            let mut baseline = Components::default();
            let observed = attention
                .forward_observed(
                    "attention",
                    &input,
                    None,
                    Some(&mut before),
                    &context,
                    &mut baseline,
                )
                .unwrap();
            assert_tensor_exact(&observed, &ordinary, "V4 ordinary/observed attention");
            assert_eq!(cache.offset, before.offset);
            assert!(baseline.values["attention.channels"]
                .data
                .iter()
                .any(|n| n.abs() > 1e-4));
            reconstruct(
                &baseline,
                first,
                last,
                args.o_groups as usize,
                args.o_lora_rank as usize,
            );
            if step != 0 {
                continue;
            }
            for keep in [false, true] {
                let selection = Some(("attention.channels", 2, 5));
                let mut trial = Components {
                    zero: if keep { None } else { selection },
                    keep: if keep { selection } else { None },
                    ..Default::default()
                };
                let mut fresh = NumericPoolingCache::new(args.sliding_window, &ratios);
                let changed = attention
                    .forward_observed(
                        "attention",
                        &input,
                        None,
                        Some(&mut fresh),
                        &context,
                        &mut trial,
                    )
                    .unwrap();
                assert_tensor_exact(
                    &trial.values["attention.channels"],
                    &baseline.values["attention.channels"],
                    "pre-intervention channels",
                );
                let pre = &trial.values["attention.channels"];
                let post = &trial.values["attention.channels.effective"];
                let width = pre.shape[2] as usize;
                for i in 0..pre.data.len() {
                    let removed = i / width == 2 && ((i % width == 5) != keep);
                    assert_eq!(post.data[i], if removed { 0.0 } else { pre.data[i] });
                }
                assert_eq!(&changed.data[..8], &ordinary.data[..8]);
                assert_eq!(&changed.data[12..], &ordinary.data[12..]);
                assert!(changed.data[8..12]
                    .iter()
                    .zip(&ordinary.data[8..12])
                    .any(|(a, b)| (a - b).abs() > 1e-6));
                reconstruct(
                    &trial,
                    first,
                    last,
                    args.o_groups as usize,
                    args.o_lora_rank as usize,
                );
            }
        }
    }
}

fn reconstruct_cycle(capture: &Components, kind: &str, residual: &str, write: &str, output: &str) {
    let base = &capture.values[residual];
    let pre = &capture.values[&format!("block.hyper.{kind}.pre")];
    let post = &capture.values[&format!("block.hyper.{kind}.post")];
    let combination = &capture.values[&format!("block.hyper.{kind}.combination")];
    let write = &capture.values[write];
    let tokens = base.shape[1] as usize;
    let streams = base.shape[2] as usize;
    let hidden = base.shape[3] as usize;
    let mut collapsed = vec![0.0; tokens * hidden];
    let mut expanded = vec![0.0; base.data.len()];
    for t in 0..tokens {
        for h in 0..hidden {
            collapsed[t * hidden + h] = (0..streams)
                .map(|s| {
                    pre.data[t * streams + s] as f64
                        * base.data[(t * streams + s) * hidden + h] as f64
                })
                .sum::<f64>() as f32;
            for to in 0..streams {
                let mixed: f64 = (0..streams)
                    .map(|from| {
                        combination.data[(t * streams + from) * streams + to] as f64
                            * base.data[(t * streams + from) * hidden + h] as f64
                    })
                    .sum();
                expanded[(t * streams + to) * hidden + h] = (mixed
                    + post.data[t * streams + to] as f64 * write.data[t * hidden + h] as f64)
                    as f32;
            }
        }
    }
    assert_tensor_close(
        &NumericTensor::new(vec![1, tokens as i32, hidden as i32], collapsed),
        &capture.values[&format!("block.hyper.{kind}.collapsed")],
        "actual hyper collapse coefficients",
    );
    assert_tensor_close(
        &NumericTensor::new(base.shape.clone(), expanded),
        &capture.values[output],
        "actual hyper injection and input-to-output stream mixing",
    );
}

#[test]
fn v4_hyper_block_reconstructs_mixing_and_recomputes_keep_only_shared_units() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    for layer in [0, 1] {
        let block =
            deepseek::block::V4Block::<NumericBackend>::new(&args, layer, &context).unwrap();
        let input = NumericTensor::new(
            vec![1, 9, 2, 4],
            (0..72).map(|i| (i as f32 * 0.23 + 0.6).cos()).collect(),
        );
        let tokens = NumericTensor::token_ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let ordinary = block
            .clone()
            .forward(
                &input,
                &tokens,
                None,
                None::<&mut NumericPoolingCache>,
                &context,
            )
            .unwrap();
        let mut baseline = Components::default();
        let observed = block
            .clone()
            .forward_observed(
                "block",
                &input,
                &tokens,
                None,
                None::<&mut NumericPoolingCache>,
                &context,
                &mut baseline,
            )
            .unwrap();
        assert_tensor_exact(&observed, &ordinary, "unified V4 block observer parity");
        let mut parallel = Components::default();
        let mut reductions = 0;
        let result = block
            .clone()
            .forward_parallel_observed(
                "block",
                &input,
                &tokens,
                None,
                None::<&mut NumericPoolingCache>,
                &context,
                &mut parallel,
                |value, _| {
                    reductions += 1;
                    Ok(value)
                },
            )
            .unwrap();
        assert_eq!(reductions, 2);
        assert_tensor_exact(
            &result,
            &ordinary,
            "single-rank V4 collective driver parity",
        );
        for (path, value) in &baseline.values {
            assert_tensor_exact(value, &parallel.values[path], path);
        }
        let mut trial = Components {
            zero: Some(("block.compressed_attention.channels", 2, 5)),
            keep: Some(("block.feed_forward.shared.units", 2, 1)),
            ..Default::default()
        };
        let changed = block
            .clone()
            .forward_observed(
                "block",
                &input,
                &tokens,
                None,
                None::<&mut NumericPoolingCache>,
                &context,
                &mut trial,
            )
            .unwrap();
        let units = &trial.values["block.feed_forward.shared.units"];
        let effective = &trial.values["block.feed_forward.shared.units.effective"];
        let width = units.shape[2] as usize;
        assert_ne!(
            units.data[2 * width + 1],
            baseline.values["block.feed_forward.shared.units"].data[2 * width + 1]
        );
        for i in 0..units.data.len() {
            assert_eq!(
                effective.data[i],
                if i / width == 2 && i % width != 1 {
                    0.0
                } else {
                    units.data[i]
                }
            );
        }
        assert!(changed
            .data
            .iter()
            .zip(&ordinary.data)
            .any(|(a, b)| (a - b).abs() > 1e-5));
        for capture in [&baseline, &parallel, &trial] {
            reconstruct_cycle(
                capture,
                "attention",
                "block.input.effective",
                "block.compressed_attention.output.effective",
                "block.hyper.attention.streams",
            );
            reconstruct_cycle(
                capture,
                "feed_forward",
                "block.hyper.attention.streams.effective",
                "block.feed_forward.contribution.effective",
                "block.output",
            );
        }
    }
}

#[test]
fn v4_partition_readout_matches_resident_across_cuts_providers_and_embedding_interventions() {
    type Model = deepseek::v4::Model<NumericBackend>;
    let args = tiny_v4_args();
    let layout = deepseek::v4::state_layout(&args).unwrap();
    let mut groups = deepseek::parallel::v4_static_parameter_groups(&args).unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        groups.extend(deepseek::parallel::v4_layer_parameter_groups(&args, layer).unwrap());
    }
    let local = numeric_local_layout(&groups, 1, 0).unwrap();
    let context = NumericContext::with_local_layout(local.clone());
    let make_model = || {
        Model::new_parallel(
            args.clone(),
            deepseek::parallel::v4_local_geometry(&args, &local).unwrap(),
            &context,
        )
        .unwrap()
    };
    let make_state = |range: std::ops::Range<usize>| {
        let start = range.start;
        DeviceState::<NumericBackend, _>::create(layout.slice(range).unwrap(), |index, _| {
            let ratios = match args.attention_policy(start + index).unwrap() {
                deepseek::V4AttentionPolicy::Local => vec![],
                deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
                deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
            };
            Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &ratios))
        })
        .unwrap()
    };
    for cut in [1, 2] {
        for provider in [false, true] {
            for tensor_parallel in [false, true] {
                for intervention in [false, true] {
                    let mut reference = ResidentRuntime::new(make_model(), &context).unwrap();
                    let mut reference_state = make_state(0..3);
                    let mut models = [make_model(), make_model()];
                    models[1].set_partition_target_start(cut);
                    let mut states = [make_state(0..cut), make_state(cut..3)];
                    let ranges = [0..cut, cut..3];
                    let mut units = ranges
                        .iter()
                        .enumerate()
                        .map(|(stage, range)| {
                            range
                                .clone()
                                .map(|layer| {
                                    models[stage].construct_unit(0, layer, &context).unwrap()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
                    for (step, ids) in [vec![1, 2, 3, 4, 5, 6, 7, 8], vec![9], vec![10]]
                        .into_iter()
                        .enumerate()
                    {
                        let tokens = NumericTensor::token_ids(&ids);
                        let capture = || Components {
                            reject_duplicates: true,
                            zero: intervention.then_some(("readout.embedding", ids.len() - 1, 1)),
                            keep: intervention.then_some((
                                "layers.1.feed_forward.shared.units",
                                ids.len() - 1,
                                2,
                            )),
                            ..Default::default()
                        };
                        let mut expected_capture = capture();
                        let (expected, _) = reference
                            .forward_with_traversal_hook(
                                deepseek::mtp::EmbeddedInput::target(&tokens, None),
                                &mut reference_state,
                                &context,
                                &mut expected_capture,
                            )
                            .unwrap();
                        let mut actual_capture = capture();
                        let mut boundary: Option<(
                            NumericTensor,
                            deepseek::v4::TargetBoundary<NumericTensor>,
                        )> = None;
                        for stage in 0..2 {
                            let input = match boundary.take() {
                                None => LayeredPartitionInput::Tokens(&tokens),
                                Some((hidden, auxiliary)) => {
                                    LayeredPartitionInput::Hidden { hidden, auxiliary }
                                }
                            };
                            let model = &mut models[stage];
                            let state = &mut states[stage];
                            let expected_layout = state.layout().clone();
                            let parallel = tensor_parallel.then_some(&parallel);
                            let mut forward = model
                                .begin_partition_observed(
                                    input,
                                    None,
                                    state,
                                    &expected_layout,
                                    ranges[stage].start,
                                    parallel,
                                    &context,
                                    &mut actual_capture,
                                )
                                .unwrap();
                            let mut hidden = model
                                .enter_partition_group(
                                    0,
                                    &forward.hidden,
                                    state,
                                    &mut forward.context,
                                    parallel,
                                    &context,
                                )
                                .unwrap();
                            for (unit, layer) in units[stage].iter_mut().zip(ranges[stage].clone())
                            {
                                let pass = if step == 0 {
                                    ExpertPass::Prefill
                                } else {
                                    ExpertPass::Decode
                                };
                                hidden = match (provider, parallel) {
                                    (false, None) => model.forward_unit_observed(
                                        0,
                                        layer,
                                        unit,
                                        &hidden,
                                        state,
                                        &mut forward.context,
                                        &context,
                                        &mut actual_capture,
                                    ),
                                    (true, None) => model.forward_unit_observed_with_provider(
                                        0,
                                        layer,
                                        unit,
                                        &hidden,
                                        state,
                                        &mut forward.context,
                                        pass,
                                        &mut ResidentExpertProvider,
                                        &context,
                                        &mut actual_capture,
                                    ),
                                    (false, Some(parallel)) => model
                                        .forward_unit_parallel_observed(
                                            0,
                                            layer,
                                            unit,
                                            &hidden,
                                            state,
                                            &mut forward.context,
                                            parallel,
                                            &context,
                                            &mut actual_capture,
                                        ),
                                    (true, Some(parallel)) => model
                                        .forward_unit_parallel_observed_with_provider(
                                            0,
                                            layer,
                                            unit,
                                            &hidden,
                                            state,
                                            &mut forward.context,
                                            pass,
                                            &mut ResidentExpertProvider,
                                            parallel,
                                            &context,
                                            &mut actual_capture,
                                        ),
                                }
                                .unwrap();
                            }
                            hidden = model
                                .leave_partition_group(
                                    0,
                                    &hidden,
                                    state,
                                    &mut forward.context,
                                    parallel,
                                    &context,
                                )
                                .unwrap();
                            match model
                                .finish_partition_observed(
                                    &hidden,
                                    state,
                                    &forward.context,
                                    stage == 1,
                                    parallel,
                                    &context,
                                    &mut actual_capture,
                                )
                                .unwrap()
                            {
                                LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
                                    assert_eq!(stage, 0);
                                    boundary = Some((hidden, auxiliary));
                                }
                                LayeredPartitionOutput::Final { output, .. } => {
                                    assert_eq!(stage, 1);
                                    assert_tensor_close(
                                        &output,
                                        &expected,
                                        "V4 observed partition score",
                                    );
                                }
                            }
                        }
                        for path in [
                            "readout.embedding.effective",
                            "layers.0.output.effective",
                            "layers.1.output.effective",
                            "layers.2.output.effective",
                            "readout.stream_coefficients",
                            "readout.residual",
                            "readout.linear",
                        ] {
                            assert_tensor_close(
                                &actual_capture.values[path],
                                &expected_capture.values[path],
                                path,
                            );
                        }
                        for (stage, state) in states.iter().enumerate() {
                            for (local, global) in ranges[stage].clone().enumerate() {
                                assert_eq!(
                                    state.as_ref()[local].offset,
                                    reference_state.as_ref()[global].offset
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn v4_custom_plain_executor_preserves_outer_boundary_capture() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    let architecture = deepseek::v4::Model::<NumericBackend>::new(args.clone(), &context).unwrap();
    let units = (0..3)
        .map(|layer| architecture.construct_unit(0, layer, &context).unwrap())
        .collect();
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(
            deepseek::v4::state_layout(&args).unwrap(),
            |layer, _| {
                let ratios = match args.attention_policy(layer).unwrap() {
                    deepseek::V4AttentionPolicy::Local => vec![],
                    deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
                    deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
                };
                Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &ratios))
            },
        )
        .unwrap()
    };
    let tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let mut capture = Components::strict();
    let actual = runtime
        .forward_with_unit_executor_and_observer(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut make_state(),
            &context,
            |model, group, layer, unit, hidden, state, forward, context| {
                LayeredArchitecture::forward_unit(
                    model, group, layer, unit, hidden, state, forward, context,
                )
            },
            &mut capture,
        )
        .unwrap();
    let expected = runtime
        .forward(
            deepseek::mtp::EmbeddedInput::target(&tokens, None),
            &mut make_state(),
            &context,
        )
        .unwrap();
    assert_tensor_exact(&actual, &expected, "plain custom unit executor");
    for layer in 0..3 {
        for seam in ["input", "output"] {
            let path = format!("layers.{layer}.{seam}");
            assert_eq!(capture.values[&path].shape, [1, 3, 2, 4]);
            // Generic outer hooks expose the original unit boundary. Detailed
            // effective companions belong to the architecture's observed call.
            assert!(!capture.values.contains_key(&format!("{path}.effective")));
        }
    }
}
