//! Independent scalar QSA+GQA equations and combined cache lifecycle.
use super::qwen4_recurrent::Parameters;
use super::*;
use eredu_architectures::{
    decoder::ComponentInstrumentation,
    qwen4_exp::{
        config::Config,
        indexed::{IndexedSublayer, IndexedSublayerInput, IndexedSublayerSpec},
        qsa::{QsaError, QsaExecutionLimits, QsaStreamStateSpec},
    },
};
use eredu_nn::TensorElementType;
use eredu_runtime::{AppendOnlyStream, AppendStreamLimits, ResidentAppendStream};

fn config() -> Config {
    let mut c = super::qwen4_recurrent::config();
    c.hidden_size = 2;
    c.attention.heads = 2;
    c.attention.kv_heads = 1;
    c.attention.head_dim = 2;
    c.attention.rotary_dim = 2;
    c.attention.rotary.dimensions = 2;
    c.attention.index_heads = 2;
    c.attention.index_head_dim = 2;
    c.attention.ratio = 3;
    c.attention.budget = 6;
    c
}
fn spec() -> IndexedSublayerSpec {
    IndexedSublayerSpec::from_config(
        &config(),
        "model.layers.0",
        2,
        16384,
        QsaExecutionLimits {
            batch: 2,
            tokens: 32,
            workspace_bytes: 1024 * 1024,
        },
        TensorElementType::F32,
        TensorElementType::F32,
        |_| Ok(dense_linear_format()),
    )
    .unwrap()
}
fn state(spec: &IndexedSublayerSpec) -> NumericHybridLayerState {
    let mut state = NumericHybridLayerState::new(
        &spec
            .state
            .cache_policy(spec.kv_heads, spec.head_dim)
            .unwrap(),
    );
    for lane in 0..2 {
        for stream in spec.state.streams() {
            let limits = AppendStreamLimits {
                entries: 64,
                page_entries: 2,
                read_entries: if stream.slot == QsaStreamStateSpec::KEYS {
                    2
                } else {
                    1
                },
            };
            state.streams.push((
                stream.slot,
                lane,
                ResidentAppendStream::new(stream, limits, 65536, 65536, 65536).unwrap(),
            ));
        }
    }
    state
}
fn sig(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn project(x: &[f64], w: &[f32]) -> Vec<f64> {
    w.chunks_exact(x.len())
        .map(|w| x.iter().zip(w).map(|(x, w)| x * *w as f64).sum())
        .collect()
}
fn norm(x: &[f64], w: &[f32], epsilon: f64) -> Vec<f64> {
    let scale = (x.iter().map(|x| x * x).sum::<f64>() / x.len() as f64 + epsilon)
        .sqrt()
        .recip();
    x.iter()
        .zip(w)
        .map(|(x, w)| x * scale * (1. + *w as f64))
        .collect()
}
fn rotate(x: &[f64], angle: f64) -> [f64; 2] {
    [
        x[0] * angle.cos() - x[1] * angle.sin(),
        x[1] * angle.cos() + x[0] * angle.sin(),
    ]
}
fn input() -> NumericTensor {
    NumericTensor::new(
        [2, 19, 2, 2],
        (0..152)
            .map(|i| ((i * 11 % 43) as f32 - 21.) / 17.)
            .collect(),
    )
}
fn visibility() -> Vec<bool> {
    (0..2)
        .flat_map(|lane| {
            (0..19).map(move |t| {
                if lane == 0 {
                    ![0, 1, 7, 11].contains(&t)
                } else {
                    ![2, 3, 4, 9, 18].contains(&t)
                }
            })
        })
        .collect()
}
fn angle(lane: usize, token: usize, media: bool) -> f64 {
    (token + if media { lane * 3 } else { 0 }) as f64
}

fn oracle(
    input: &NumericTensor,
    visible: &[bool],
    p: &Parameters,
    media: bool,
) -> (NumericTensor, Vec<i32>) {
    let weight = |name: &str| &p.0[&format!("model.layers.0.{name}")].data;
    let epsilon = config().norm_epsilon as f64;
    let mut output = vec![];
    let mut selections = vec![];
    for lane in 0..2 {
        let mut raw = Vec::<Vec<f64>>::new();
        let mut keys = Vec::<[f64; 2]>::new();
        let mut values = Vec::<Vec<f64>>::new();
        let mut valid = vec![];
        for t in 0..19 {
            let row = &input.data[(lane * 19 + t) * 4..(lane * 19 + t + 1) * 4];
            let normalized: Vec<_> = row
                .chunks_exact(2)
                .enumerate()
                .flat_map(|(s, x)| {
                    norm(
                        &x.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                        &weight("attn_hyper_connection.hc_norm.weight")[s * 2..s * 2 + 2],
                        epsilon,
                    )
                })
                .collect();
            let low: Vec<_> = project(
                &normalized,
                weight("attn_hyper_connection.input_mix_weight_down.weight"),
            )
            .into_iter()
            .map(|x| {
                let x = x / 2.;
                x * sig(x)
            })
            .collect();
            let mixing = project(
                &low,
                weight("attn_hyper_connection.input_mix_weight_up.weight"),
            );
            let mixed: Vec<_> = (0..2)
                .map(|i| {
                    (normalized[i] * sig(mixing[i]) + normalized[i + 2] * sig(mixing[i + 2])) / 2.
                })
                .collect();
            let injection = project(
                &normalized,
                weight("attn_hyper_connection.block_inject_weight.weight"),
            );
            let index = project(&mixed, weight("self_attn.indexer.index_qk_proj.weight"));
            let iq: Vec<_> = index[..4]
                .chunks_exact(2)
                .map(|q| {
                    rotate(
                        &norm(q, weight("self_attn.indexer.q_layernorm.weight"), epsilon),
                        angle(lane, t, media),
                    )
                })
                .collect();
            raw.push(index[4..].to_vec());
            if visible[lane * 19 + t] {
                valid.push(t);
            }
            let mut blocks = vec![];
            for (block, positions) in valid.chunks_exact(3).enumerate() {
                let pooled: Vec<_> = (0..2)
                    .map(|d| positions.iter().map(|&t| raw[t][d]).sum::<f64>() / 3.)
                    .collect();
                let key = rotate(
                    &norm(
                        &pooled,
                        weight("self_attn.indexer.k_layernorm.weight"),
                        epsilon,
                    ),
                    angle(lane, positions[0], media),
                );
                let score = iq
                    .iter()
                    .map(|q| (q[0] * key[0] + q[1] * key[1]).max(0.))
                    .sum::<f64>()
                    / 2f64.sqrt();
                blocks.push((block, score));
            }
            blocks.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut selected: Vec<usize> = blocks
                .iter()
                .take(2)
                .flat_map(|(b, _)| valid[b * 3..b * 3 + 3].iter().copied())
                .chain(valid[valid.len() / 3 * 3..].iter().copied())
                .collect();
            selected.sort();
            let mut slots: Vec<_> = selected.iter().map(|&x| x as i32).collect();
            slots.resize(8, -1);
            selections.extend(slots);
            let qg = project(&mixed, weight("self_attn.q_proj.weight"));
            let key = project(&mixed, weight("self_attn.k_proj.weight"));
            keys.push(rotate(
                &norm(&key, weight("self_attn.k_norm.weight"), epsilon),
                angle(lane, t, media),
            ));
            values.push(project(&mixed, weight("self_attn.v_proj.weight")));
            let mut channels = vec![];
            for head in 0..2 {
                let q = rotate(
                    &norm(
                        &qg[head * 4..head * 4 + 2],
                        weight("self_attn.q_norm.weight"),
                        epsilon,
                    ),
                    angle(lane, t, media),
                );
                let scores: Vec<_> = selected
                    .iter()
                    .map(|&k| (q[0] * keys[k][0] + q[1] * keys[k][1]) / 2f64.sqrt())
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
                for d in 0..2 {
                    let v = if selected.is_empty() {
                        0.
                    } else {
                        selected
                            .iter()
                            .zip(&scores)
                            .map(|(&k, s)| values[k][d] * (s - max).exp() / denominator)
                            .sum::<f64>()
                    };
                    channels.push(v * sig(qg[head * 4 + 2 + d]));
                }
            }
            let written = project(&channels, weight("self_attn.o_proj.weight"));
            for s in 0..2 {
                for d in 0..2 {
                    output.push(
                        (row[s * 2 + d] as f64 + 2. * sig(injection[s] / 2.) * written[d]) as f32,
                    );
                }
            }
        }
    }
    (NumericTensor::new(input.shape.clone(), output), selections)
}
#[derive(Default)]
struct Capture {
    selected: Option<NumericTensor>,
    zero: bool,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Capture {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        if path == "model.layers.0.attention.selected_positions" {
            assert!(self.selected.replace(value.clone()).is_none());
        }
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok((self.zero && path == "model.layers.0.attention.write")
            .then(|| NumericTensor::zeros(value.shape.clone())))
    }
}
fn media_position(start: usize, end: usize) -> (NumericTensor, NumericTensor) {
    let data = |cos: bool| {
        (0..2)
            .flat_map(|lane| {
                (start..end).flat_map(move |t| {
                    let a = angle(lane, t, true);
                    [if cos { a.cos() as f32 } else { a.sin() as f32 }; 2]
                })
            })
            .collect()
    };
    (
        NumericTensor::new([2, (end - start) as i32, 2], data(true)),
        NumericTensor::new([2, (end - start) as i32, 2], data(false)),
    )
}
fn compare_state(
    a: &NumericHybridLayerState,
    b: &NumericHybridLayerState,
    context: &NumericContext,
) {
    assert_eq!(a.position(), b.position());
    for (role, value) in &a.fixed {
        assert_tensor_close(
            value.as_ref().unwrap(),
            b.fixed[role].as_ref().unwrap(),
            "QSA fixed tails",
        );
    }
    assert_tensor_close(
        a.attention.as_ref().unwrap().keys.as_ref().unwrap(),
        b.attention.as_ref().unwrap().keys.as_ref().unwrap(),
        "ordinary full K/V prefix",
    );
    assert_tensor_close(
        a.attention.as_ref().unwrap().values.as_ref().unwrap(),
        b.attention.as_ref().unwrap().values.as_ref().unwrap(),
        "ordinary full value prefix",
    );
    for ((slot, lane, sa), (other_slot, other_lane, sb)) in a.streams.iter().zip(&b.streams) {
        assert_eq!((slot, lane, sa.len()), (other_slot, other_lane, sb.len()));
        for t in 0..sa.len() {
            assert_tensor_close(
                &sa.clone().read(t..t + 1, context).unwrap(),
                &sb.clone().read(t..t + 1, context).unwrap(),
                "paired summary/position streams",
            );
        }
    }
}
#[test]
fn qwen4_indexed_scalar_ragged_media_chunks_and_state_replay() {
    let context = NumericContext::default();
    let input = input();
    let visible = visibility();
    for media in [false, true] {
        let spec = spec();
        let mut layer = IndexedSublayer::<NumericBackend>::new(spec.clone(), &context).unwrap();
        let mut parameters = Parameters::default();
        layer.visit_parameters_mut(&mut parameters);
        let (expected, selection) = oracle(&input, &visible, &parameters, media);
        let (cos, sin) = media_position(0, 19);
        let mut whole = state(&spec);
        let mut capture = Capture::default();
        let full = layer
            .forward(
                IndexedSublayerInput {
                    residual: &input,
                    visible: Some(&visible),
                    rotary: media.then_some(RotaryPosition::Embeddings {
                        cosine: &cos,
                        sine: &sin,
                    }),
                },
                &mut whole,
                None,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut capture),
            )
            .unwrap();
        assert_tensor_close(
            &full,
            &expected,
            "independent QSA, GQA, gate and residual equations",
        );
        assert_eq!(
            capture.selected.unwrap().to_i32_vec(&context).unwrap(),
            selection
        );
        assert_eq!(whole.position(), 19);
        assert_eq!(
            whole
                .attention
                .as_ref()
                .unwrap()
                .keys
                .as_ref()
                .unwrap()
                .shape,
            [2, 1, 19, 2]
        );
        for chunks in [vec![2, 1, 4, 3, 1, 8], vec![1; 19]] {
            let mut split = state(&spec);
            let mut start = 0;
            let mut outputs = vec![];
            for length in chunks {
                let end = start + length;
                let chunk = input.axis_slice(1, start, end);
                let mask: Vec<_> = (0..2)
                    .flat_map(|lane| visible[lane * 19 + start..lane * 19 + end].iter().copied())
                    .collect();
                let (cos, sin) = media_position(start, end);
                let position = media.then_some(RotaryPosition::Embeddings {
                    cosine: &cos,
                    sine: &sin,
                });
                let checkpoint = split.clone();
                let output = layer
                    .forward(
                        IndexedSublayerInput {
                            residual: &chunk,
                            visible: Some(&mask),
                            rotary: position,
                        },
                        &mut split,
                        None,
                        &context,
                        &mut ComponentInstrumentation::disabled(),
                    )
                    .unwrap();
                let mut restored = checkpoint.clone();
                let replay = layer
                    .forward(
                        IndexedSublayerInput {
                            residual: &chunk,
                            visible: Some(&mask),
                            rotary: position,
                        },
                        &mut restored,
                        None,
                        &context,
                        &mut ComponentInstrumentation::disabled(),
                    )
                    .unwrap();
                assert_tensor_exact(&output, &replay, "forked/replayed indexed unit");
                compare_state(&split, &restored, &context);
                outputs.push(output);
                start = end;
            }
            assert_tensor_close(
                &NumericTensor::concatenate(&outputs, 1, &context).unwrap(),
                &full,
                "indexed chunk invariance",
            );
            compare_state(&whole, &split, &context);
        }
        let mut reset = whole.clone();
        reset.reset().unwrap();
        let repeated = layer
            .forward(
                IndexedSublayerInput {
                    residual: &input,
                    visible: Some(&visible),
                    rotary: media.then_some(RotaryPosition::Embeddings {
                        cosine: &cos,
                        sine: &sin,
                    }),
                },
                &mut reset,
                None,
                &context,
                &mut ComponentInstrumentation::disabled(),
            )
            .unwrap();
        assert_tensor_exact(&repeated, &full, "all indexed state resets together");
        let mut zero = Capture {
            zero: true,
            ..Default::default()
        };
        let mut masked = state(&spec);
        let out = layer
            .forward(
                IndexedSublayerInput {
                    residual: &input,
                    visible: Some(&visible),
                    rotary: media.then_some(RotaryPosition::Embeddings {
                        cosine: &cos,
                        sine: &sin,
                    }),
                },
                &mut masked,
                None,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut zero),
            )
            .unwrap();
        assert_tensor_exact(&out, &input, "write edit preserves complete residuals");
        compare_state(&masked, &whole, &context);
    }
}
#[test]
fn qwen4_indexed_geometry_scratch_and_prefix_drift_fail_closed() {
    let context = NumericContext::default();
    let good = spec();
    let mut bad = good.clone();
    bad.projections[0].output -= 1;
    assert!(IndexedSublayer::<NumericBackend>::new(bad, &context).is_err());
    let mut bad = good.clone();
    bad.limits.workspace_bytes = 0;
    assert!(matches!(
        IndexedSublayer::<NumericBackend>::new(bad, &context),
        Err(QsaError::Workspace { .. })
    ));
    let mut bad = good.clone();
    bad.indexer.rotary.base *= 2.;
    assert!(IndexedSublayer::<NumericBackend>::new(bad, &context).is_err());
    let mut layer = IndexedSublayer::<NumericBackend>::new(good.clone(), &context).unwrap();
    layer.visit_parameters_mut(&mut Parameters::default());
    let mut state = state(&good);
    let mut bad_visibility = visibility();
    bad_visibility.pop();
    assert!(layer
        .forward(
            IndexedSublayerInput {
                residual: &input(),
                visible: Some(&bad_visibility),
                rotary: None
            },
            &mut state,
            None,
            &context,
            &mut ComponentInstrumentation::disabled()
        )
        .is_err());
    assert_eq!(state.position(), 0);
    assert!(state.streams.iter().all(|(_, _, s)| s.is_empty()));
    state.attention.as_mut().unwrap().offset = 1;
    assert!(layer
        .forward(
            IndexedSublayerInput {
                residual: &input(),
                visible: None,
                rotary: None
            },
            &mut state,
            None,
            &context,
            &mut ComponentInstrumentation::disabled()
        )
        .is_err());
    assert!(state.streams.iter().all(|(_, _, s)| s.is_empty()));
}

#[test]
fn qwen4_indexed_tp2_tp4_replicates_kv_and_qsa_with_partial_rotary() {
    let context = NumericContext::default();
    let mut c = config();
    c.attention.heads = 4;
    c.attention.kv_heads = 2;
    c.attention.head_dim = 4;
    c.attention.index_head_dim = 4;
    let spec = IndexedSublayerSpec::from_config(
        &c,
        "model.layers.0",
        2,
        16384,
        spec().limits,
        TensorElementType::F32,
        TensorElementType::F32,
        |_| Ok(dense_linear_format()),
    )
    .unwrap();
    let mut ordinary = IndexedSublayer::<NumericBackend>::new(spec.clone(), &context).unwrap();
    let mut parameters = Parameters::default();
    ordinary.visit_parameters_mut(&mut parameters);
    let input = input();
    let visible = visibility();
    let (cos, sin) = media_position(0, 19);
    let mut full_state = state(&spec);
    let expected = ordinary
        .forward(
            IndexedSublayerInput {
                residual: &input,
                visible: Some(&visible),
                rotary: Some(RotaryPosition::Embeddings {
                    cosine: &cos,
                    sine: &sin,
                }),
            },
            &mut full_state,
            None,
            &context,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
    struct Shards<'a> {
        rank: usize,
        ranks: usize,
        parameters: &'a Parameters,
    }
    impl<'a, 'b> ParameterVisitorMut<'b, NumericTensor> for Shards<'a> {
        fn visit_mut(&mut self, meta: ParameterMetadata, value: &'b mut NumericTensor) {
            let name = meta.id.as_str();
            let full = &self.parameters.0[name];
            *value = if name.ends_with("self_attn.q_proj.weight") {
                full.axis_slice(
                    0,
                    self.rank * 32 / self.ranks,
                    (self.rank + 1) * 32 / self.ranks,
                )
            } else if name.ends_with("self_attn.k_proj.weight")
                || name.ends_with("self_attn.v_proj.weight")
            {
                let kv_rank = self.rank / (self.ranks / 2);
                full.axis_slice(0, kv_rank * 4, (kv_rank + 1) * 4)
            } else if name.ends_with("self_attn.o_proj.weight") {
                full.axis_slice(
                    1,
                    self.rank * 16 / self.ranks,
                    (self.rank + 1) * 16 / self.ranks,
                )
            } else {
                full.clone()
            };
        }
    }
    for ranks in [2, 4] {
        let group = NumericParallelGroup::new(ranks);
        let results = std::thread::scope(|scope| {
            (0..ranks)
                .map(|rank| {
                    let mut local = spec.clone();
                    local.heads /= ranks as i32;
                    local.kv_heads = 1;
                    local.projections[0].output /= ranks as i32;
                    local.projections[1].output = 4;
                    local.projections[2].output = 4;
                    local.projections[3].input /= ranks as i32;
                    let group = group.clone();
                    let parameters = &parameters;
                    let input = &input;
                    let visible = &visible;
                    scope.spawn(move || {
                        let context = NumericContext::default();
                        let parallel = NumericParallelContext::new(rank, group);
                        let mut state = state(&local);
                        let mut layer =
                            IndexedSublayer::<NumericBackend>::new(local, &context).unwrap();
                        layer.visit_parameters_mut(&mut Shards {
                            rank,
                            ranks,
                            parameters,
                        });
                        let pieces: Vec<_> = [(0, 2), (2, 7), (7, 8), (8, 19)]
                            .into_iter()
                            .map(|(a, b)| {
                                let visibility: Vec<_> = (0..2)
                                    .flat_map(|lane| {
                                        visible[lane * 19 + a..lane * 19 + b].iter().copied()
                                    })
                                    .collect();
                                let (cos, sin) = media_position(a, b);
                                layer
                                    .forward(
                                        IndexedSublayerInput {
                                            residual: &input.axis_slice(1, a, b),
                                            visible: Some(&visibility),
                                            rotary: Some(RotaryPosition::Embeddings {
                                                cosine: &cos,
                                                sine: &sin,
                                            }),
                                        },
                                        &mut state,
                                        Some(&parallel),
                                        &context,
                                        &mut ComponentInstrumentation::disabled(),
                                    )
                                    .unwrap()
                            })
                            .collect();
                        (
                            NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
                            state,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (rank, (out, local)) in results.iter().enumerate() {
            assert_tensor_close(out, &expected, "TP partial-rotary QSA output");
            let mut expected_state = full_state.clone();
            let kv_rank = rank / (ranks / 2);
            let cache = expected_state.attention.as_mut().unwrap();
            cache.keys = Some(
                cache
                    .keys
                    .as_ref()
                    .unwrap()
                    .axis_slice(1, kv_rank, kv_rank + 1),
            );
            cache.values = Some(
                cache
                    .values
                    .as_ref()
                    .unwrap()
                    .axis_slice(1, kv_rank, kv_rank + 1),
            );
            compare_state(local, &expected_state, &context);
        }
    }
}

#[test]
fn qwen4_indexed_unit_failure_restores_summaries_tail_and_kv_together() {
    struct Fail(&'static str);
    impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Fail {
        fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
            if path == self.0 {
                return Err(Error::backend("injected unit failure"));
            }
            Ok(())
        }
    }
    let context = NumericContext::default();
    let spec = spec();
    let mut layer = IndexedSublayer::<NumericBackend>::new(spec.clone(), &context).unwrap();
    layer.visit_parameters_mut(&mut Parameters::default());
    let input = input();
    let mut committed = state(&spec);
    let prefix = input.axis_slice(1, 0, 2);
    layer
        .forward(
            IndexedSublayerInput {
                residual: &prefix,
                visible: None,
                rotary: None,
            },
            &mut committed,
            None,
            &context,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
    let next = input.axis_slice(1, 2, 9);
    let mut reference = committed.clone();
    let expected = layer
        .forward(
            IndexedSublayerInput {
                residual: &next,
                visible: None,
                rotary: None,
            },
            &mut reference,
            None,
            &context,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
    for path in [
        "model.layers.0.attention.selected_positions",
        "model.layers.0.attention.write",
    ] {
        let checkpoint = committed.clone();
        let mut tentative = checkpoint.clone();
        assert!(layer
            .forward(
                IndexedSublayerInput {
                    residual: &next,
                    visible: None,
                    rotary: None
                },
                &mut tentative,
                None,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut Fail(path))
            )
            .is_err());
        assert!(tentative.streams[0].2.len() > checkpoint.streams[0].2.len());
        if path.ends_with("write") {
            assert_eq!(tentative.position(), 9);
        } else {
            assert_eq!(tentative.position(), 2);
        }
        // The surrounding unit transaction restores the complete declared state.
        tentative = checkpoint;
        compare_state(&tentative, &committed, &context);
        let replay = layer
            .forward(
                IndexedSublayerInput {
                    residual: &next,
                    visible: None,
                    rotary: None,
                },
                &mut tentative,
                None,
                &context,
                &mut ComponentInstrumentation::disabled(),
            )
            .unwrap();
        assert_tensor_exact(&replay, &expected, "failed unit replay");
        compare_state(&tentative, &reference, &context);
    }
}
