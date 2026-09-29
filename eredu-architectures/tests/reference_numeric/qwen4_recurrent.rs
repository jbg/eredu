//! Independent scalar composition, then chunk/fork and state-geometry conformance.
use super::*;
use eredu_architectures::{
    decoder::ComponentInstrumentation,
    qwen4_exp::{
        config::Config,
        recurrent::{RecurrentSublayer, RecurrentSublayerSpec},
    },
};

pub(super) fn config() -> Config {
    let mut config = Config::from_json(
        &serde_json::from_str(include_str!("../../src/qwen4_exp/config/released.json")).unwrap(),
    )
    .unwrap();
    config.hidden_size = 3;
    config.residual.streams = 2;
    config.residual.rank = 2;
    config.norm_epsilon = 1e-4;
    config.recurrent.key_heads = 1;
    config.recurrent.value_heads = 2;
    config.recurrent.key_dim = 2;
    config.recurrent.value_dim = 2;
    config.recurrent.kernel = 3;
    config
}
#[derive(Default)]
pub(super) struct Parameters(pub(super) BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Parameters {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = meta.id.as_str();
        let seed = name
            .bytes()
            .fold(0usize, |s, b| (s * 17 + b as usize) % 101);
        for (i, x) in value.data.iter_mut().enumerate() {
            let delta = ((i * 7 + seed) % 29) as f32 - 14.;
            *x = if name.ends_with("linear_attn.norm.weight") {
                1. + delta * 0.01
            } else if name.ends_with("A_log") {
                -0.5 + delta * 0.01
            } else {
                delta * 0.03
            };
        }
        self.0.insert(name.into(), value.clone());
    }
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn dot(a: &[f64], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * *b as f64).sum()
}
fn linear(x: &[f64], weight: &NumericTensor) -> Vec<f64> {
    weight
        .data
        .chunks_exact(x.len())
        .map(|w| dot(x, w))
        .collect()
}

// F64 scalar equations deliberately do not invoke any portable neural primitive.
fn oracle(
    input: &NumericTensor,
    padding: Option<&NumericTensor>,
    spec: &RecurrentSublayerSpec,
    p: &Parameters,
) -> NumericTensor {
    let (batch, sequence, streams, hidden) = (
        input.dim(0) as usize,
        input.dim(1) as usize,
        input.dim(2) as usize,
        input.dim(3) as usize,
    );
    let r = &spec.mixer;
    let (kh, vh, kd, vd) = (
        r.key_heads as usize,
        r.value_heads as usize,
        r.key_head_dim as usize,
        r.value_head_dim as usize,
    );
    let kw = kh * kd;
    let vw = vh * vd;
    let cw = kw * 2 + vw;
    let weight = |suffix: &str| &p.0[&format!("model.layers.0.{suffix}")];
    let project = |x: &[f64], suffix: &str| linear(x, weight(suffix));
    let mut expected = Vec::new();
    for lane in 0..batch {
        let mut history = Vec::<Vec<f64>>::new();
        let mut state = vec![0.; vh * kd * vd];
        for token in 0..sequence {
            let begin = (lane * sequence + token) * streams * hidden;
            let x = &input.data[begin..begin + streams * hidden];
            let mut normalized = Vec::new();
            for (stream, row) in x.chunks_exact(hidden).enumerate() {
                let rms = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hidden as f64
                    + spec.residual.normalization.epsilon as f64)
                    .sqrt();
                normalized.extend(row.iter().enumerate().map(|(i, v)| {
                    *v as f64 / rms
                        * (1.
                            + weight("attn_hyper_connection.hc_norm.weight").data
                                [stream * hidden + i] as f64)
                }));
            }
            let low: Vec<_> = project(
                &normalized,
                "attn_hyper_connection.input_mix_weight_down.weight",
            )
            .into_iter()
            .map(|v| {
                let v = v / streams as f64;
                v * sigmoid(v)
            })
            .collect();
            let mix_weights = project(&low, "attn_hyper_connection.input_mix_weight_up.weight");
            let mut mixed: Vec<_> = (0..hidden)
                .map(|i| {
                    (0..streams)
                        .map(|s| {
                            let j = s * hidden + i;
                            normalized[j] * sigmoid(mix_weights[j])
                        })
                        .sum::<f64>()
                        / streams as f64
                })
                .collect();
            if let Some(mask) = padding {
                for value in &mut mixed {
                    *value *= mask.data[lane * sequence + token] as f64;
                }
            }
            let injections: Vec<_> = project(
                &normalized,
                "attn_hyper_connection.block_inject_weight.weight",
            )
            .into_iter()
            .map(|v| 2. * sigmoid(v / streams as f64))
            .collect();
            history.push(project(&mixed, "linear_attn.in_proj_qkv.weight"));
            let kernel = r.convolution.kernel_size as usize;
            let dilation = r.convolution.dilation as usize;
            let conv: Vec<_> = (0..cw)
                .map(|channel| {
                    let v = (0..kernel)
                        .filter_map(|tap| {
                            token.checked_sub((kernel - 1 - tap) * dilation).map(|t| {
                                history[t][channel]
                                    * weight("linear_attn.conv1d.weight").data
                                        [channel * kernel + tap]
                                        as f64
                            })
                        })
                        .sum::<f64>();
                    v * sigmoid(v)
                })
                .collect();
            let beta = project(&mixed, "linear_attn.in_proj_b.weight");
            let decay = project(&mixed, "linear_attn.in_proj_a.weight");
            let gates = project(&mixed, "linear_attn.in_proj_z.weight");
            let mut channels = Vec::new();
            for h in 0..vh {
                let head = h / (vh / kh);
                let normalize = |start| {
                    let x = &conv[start..start + kd];
                    let norm = (x.iter().map(|v| v * v).sum::<f64>() + r.l2_epsilon as f64).sqrt();
                    x.iter().map(|v| v / norm).collect::<Vec<_>>()
                };
                let q = normalize(head * kd);
                let k = normalize(kw + head * kd);
                let d = decay[h] + weight("linear_attn.dt_bias").data[h] as f64;
                let factor = (-(weight("linear_attn.A_log").data[h] as f64).exp()
                    * (d.max(0.) + (-d.abs()).exp().ln_1p()))
                .exp();
                let s = &mut state[h * kd * vd..(h + 1) * kd * vd];
                for v in s.iter_mut() {
                    *v *= factor;
                }
                for v in 0..vd {
                    let delta = (conv[2 * kw + h * vd + v]
                        - (0..kd).map(|j| k[j] * s[j * vd + v]).sum::<f64>())
                        * sigmoid(beta[h]);
                    for j in 0..kd {
                        s[j * vd + v] += k[j] * delta;
                    }
                }
                let y: Vec<_> = (0..vd)
                    .map(|v| {
                        (0..kd).map(|j| q[j] * s[j * vd + v]).sum::<f64>() / (kd as f64).sqrt()
                    })
                    .collect();
                let rms = (y.iter().map(|v| v * v).sum::<f64>() / vd as f64
                    + r.output_epsilon as f64)
                    .sqrt();
                channels.extend((0..vd).map(|v| {
                    y[v] / rms
                        * weight("linear_attn.norm.weight").data[v] as f64
                        * sigmoid(gates[h * vd + v])
                }));
            }
            let output = project(&channels, "linear_attn.out_proj.weight");
            for s in 0..streams {
                for i in 0..hidden {
                    expected.push((x[s * hidden + i] as f64 + injections[s] * output[i]) as f32);
                }
            }
        }
    }
    NumericTensor::new(input.shape.clone(), expected)
}

#[test]
fn qwen4_recurrent_sublayer_scalar_chunk_and_fork_parity() {
    let context = NumericContext::default();
    for (kernel, dilation, padded) in [(3, 1, false), (3, 3, false), (1, 1, false), (3, 1, true)] {
        let mut c = config();
        c.recurrent.kernel = kernel;
        let mut spec =
            RecurrentSublayerSpec::from_config(&c, "model.layers.0", |_| Ok(dense_linear_format()))
                .unwrap();
        spec.mixer.convolution.dilation = dilation;
        let policy = spec.mixer.state_policy().unwrap();
        assert_eq!(policy.fixed_state().len(), if kernel == 1 { 1 } else { 2 });
        let mut layer = RecurrentSublayer::<NumericBackend>::new(spec.clone(), &context).unwrap();
        let mut p = Parameters::default();
        layer.visit_parameters_mut(&mut p);
        let input = NumericTensor::new(
            [2, 7, 2, 3],
            (0..84)
                .map(|i| ((i * 11 % 37) as f32 - 18.) / 13.)
                .collect(),
        );
        let padding = padded.then(|| {
            NumericTensor::new(
                [2, 7],
                vec![0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
            )
        });
        let expected = oracle(&input, padding.as_ref(), &spec, &p);
        let mut whole = NumericHybridLayerState::new(&policy);
        let actual = layer
            .forward(
                &input,
                padding.as_ref(),
                &mut whole,
                &context,
                &mut ComponentInstrumentation::disabled(),
            )
            .unwrap();
        assert_tensor_close(
            &actual,
            &expected,
            "released recurrent residual composition",
        );
        assert_ne!(actual.data, input.data);
        let mut split = NumericHybridLayerState::new(&policy);
        let mut pieces = Vec::new();
        for (start, end) in [(0, 2), (2, 3), (3, 4), (4, 7)] {
            let snapshot = split.clone();
            let chunk = input.axis_slice(1, start, end);
            let mask = padding.as_ref().map(|mask| mask.axis_slice(1, start, end));
            let output = layer
                .forward(
                    &chunk,
                    mask.as_ref(),
                    &mut split,
                    &context,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap();
            let mut fork = snapshot.clone();
            let fork_output = layer
                .forward(
                    &chunk,
                    mask.as_ref(),
                    &mut fork,
                    &context,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap();
            assert_tensor_exact(&output, &fork_output, "fork and restored state");
            assert_eq!(snapshot.fixed_offset, start as i32);
            pieces.push(output);
        }
        assert_tensor_close(
            &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
            &actual,
            "arbitrary recurrent chunk boundaries",
        );
        assert_eq!(split.fixed_offset, 7);
        for (role, value) in &whole.fixed {
            assert_tensor_close(
                value.as_ref().unwrap(),
                split.fixed[role].as_ref().unwrap(),
                "identical retained recurrence and dilated history",
            );
        }
        let old = split.clone();
        assert!(layer
            .forward(
                &input.axis_slice(3, 0, 2),
                None,
                &mut split,
                &context,
                &mut ComponentInstrumentation::disabled()
            )
            .is_err());
        assert_eq!(old.fixed_offset, split.fixed_offset);
        split.reset().unwrap();
        assert_eq!(split.fixed_offset, 0);
        let repeated = layer
            .forward(
                &input,
                padding.as_ref(),
                &mut split,
                &context,
                &mut ComponentInstrumentation::disabled(),
            )
            .unwrap();
        assert_tensor_exact(&repeated, &actual, "reset replays recurrence");
    }
}

#[test]
fn qwen4_recurrent_sublayer_rejects_inconsistent_composition() {
    let context = NumericContext::default();
    let spec = RecurrentSublayerSpec::from_config(&config(), "model.layers.0", |_| {
        Ok(dense_linear_format())
    })
    .unwrap();
    let mut invalid = spec.clone();
    invalid.residual.injection = None;
    assert!(RecurrentSublayer::<NumericBackend>::new(invalid, &context).is_err());
    let mut invalid = spec.clone();
    invalid.mixer.input_qkv.input += 1;
    assert!(RecurrentSublayer::<NumericBackend>::new(invalid, &context).is_err());
    let mut invalid = spec;
    invalid.mixer.convolution.dilation = i32::MAX;
    assert!(invalid.mixer.state_policy().is_err());
}

#[test]
fn qwen4_recurrent_sublayer_tp2_tp4_preserves_global_output_and_local_state() {
    let context = NumericContext::default();
    let mut c = config();
    c.recurrent.key_heads = 4;
    c.recurrent.value_heads = 8;
    let spec =
        RecurrentSublayerSpec::from_config(&c, "model.layers.0", |_| Ok(dense_linear_format()))
            .unwrap();
    let mut ordinary = RecurrentSublayer::<NumericBackend>::new(spec.clone(), &context).unwrap();
    let mut parameters = Parameters::default();
    ordinary.visit_parameters_mut(&mut parameters);
    let input = NumericTensor::new(
        [2, 7, 2, 3],
        (0..84).map(|i| ((i * 7 % 29) as f32 - 14.) / 17.).collect(),
    );
    let mut state = NumericHybridLayerState::new(&spec.mixer.state_policy().unwrap());
    let expected = ordinary
        .forward(
            &input,
            None,
            &mut state,
            &context,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
    let local_qkv = |tensor: &NumericTensor, axis, rank, ranks| {
        let kw = 8;
        let vw = 16;
        NumericTensor::concatenate(
            &[
                tensor.axis_slice(axis, rank * kw / ranks, (rank + 1) * kw / ranks),
                tensor.axis_slice(axis, kw + rank * kw / ranks, kw + (rank + 1) * kw / ranks),
                tensor.axis_slice(
                    axis,
                    2 * kw + rank * vw / ranks,
                    2 * kw + (rank + 1) * vw / ranks,
                ),
            ],
            axis as i32,
            &NumericContext::default(),
        )
        .unwrap()
    };
    struct Shards<'a> {
        rank: usize,
        ranks: usize,
        parameters: &'a Parameters,
    }
    impl<'a, 'b> ParameterVisitorMut<'b, NumericTensor> for Shards<'a> {
        fn visit_mut(&mut self, meta: ParameterMetadata, value: &'b mut NumericTensor) {
            let name = meta.id.as_str();
            let full = &self.parameters.0[name];
            let slice = |axis: usize| {
                let width = full.dim(axis) as usize;
                full.axis_slice(
                    axis,
                    self.rank * width / self.ranks,
                    (self.rank + 1) * width / self.ranks,
                )
            };
            *value = if name.ends_with("in_proj_qkv.weight") || name.ends_with("conv1d.weight") {
                NumericTensor::concatenate(
                    &[
                        full.axis_slice(
                            0,
                            self.rank * 8 / self.ranks,
                            (self.rank + 1) * 8 / self.ranks,
                        ),
                        full.axis_slice(
                            0,
                            8 + self.rank * 8 / self.ranks,
                            8 + (self.rank + 1) * 8 / self.ranks,
                        ),
                        full.axis_slice(
                            0,
                            16 + self.rank * 16 / self.ranks,
                            16 + (self.rank + 1) * 16 / self.ranks,
                        ),
                    ],
                    0,
                    &NumericContext::default(),
                )
                .unwrap()
            } else if name.ends_with("out_proj.weight") {
                slice(1)
            } else if name.ends_with("in_proj_z.weight")
                || name.ends_with("in_proj_a.weight")
                || name.ends_with("in_proj_b.weight")
                || name.ends_with("A_log")
                || name.ends_with("dt_bias")
            {
                slice(0)
            } else {
                full.clone()
            };
        }
    }
    for ranks in [2, 4] {
        let group = NumericParallelGroup::new(ranks);
        let outputs = std::thread::scope(|scope| {
            (0..ranks)
                .map(|rank| {
                    let mut local = spec.clone();
                    let r = &mut local.mixer;
                    r.key_heads /= ranks as i32;
                    r.value_heads /= ranks as i32;
                    r.input_qkv.output /= ranks as i32;
                    r.input_gate.output /= ranks as i32;
                    r.input_beta.output /= ranks as i32;
                    r.input_decay.output /= ranks as i32;
                    r.output.input /= ranks as i32;
                    r.convolution.channels /= ranks as i32;
                    let group = group.clone();
                    let parameters = &parameters;
                    let input = &input;
                    scope.spawn(move || {
                        let context = NumericContext::default();
                        let parallel = NumericParallelContext::new(rank, group);
                        let mut state =
                            NumericHybridLayerState::new(&local.mixer.state_policy().unwrap());
                        let mut layer =
                            RecurrentSublayer::<NumericBackend>::new(local, &context).unwrap();
                        layer.visit_parameters_mut(&mut Shards {
                            rank,
                            ranks,
                            parameters,
                        });
                        let pieces: Vec<_> = [(0, 3), (3, 4), (4, 7)]
                            .into_iter()
                            .map(|(a, b)| {
                                layer
                                    .forward_parallel(
                                        &input.axis_slice(1, a, b),
                                        None,
                                        &mut state,
                                        &parallel,
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
        for (rank, (output, local)) in outputs.iter().enumerate() {
            assert_tensor_close(
                output,
                &expected,
                "TP replicated residual with reduced recurrent output",
            );
            assert_eq!(local.fixed_offset, 7);
            assert_tensor_close(
                local.fixed[&StateTensorRole::Recurrent].as_ref().unwrap(),
                &state.fixed[&StateTensorRole::Recurrent]
                    .as_ref()
                    .unwrap()
                    .axis_slice(1, rank * 8 / ranks, (rank + 1) * 8 / ranks),
                "rank-local recurrent matrices",
            );
            assert_tensor_close(
                local.fixed[&StateTensorRole::Convolution { slot: 0 }]
                    .as_ref()
                    .unwrap(),
                &local_qkv(
                    state.fixed[&StateTensorRole::Convolution { slot: 0 }]
                        .as_ref()
                        .unwrap(),
                    2,
                    rank,
                    ranks,
                ),
                "rank-local component-major convolution history",
            );
        }
    }
}
