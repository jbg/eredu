use super::qwen4_recurrent::Parameters;
use super::*;
use eredu_architectures::{
    decoder::ComponentInstrumentation,
    qwen4_exp::{
        config::Config,
        feed_forward::{FeedForwardSublayer, FeedForwardSublayerSpec},
    },
};
use eredu_nn::{GatedProductGroupLayout, GroupReduction, GroupedProjectionSpec, RoutingPrecision};
use eredu_runtime::{ResidentExpertProvider, RoutedBankId, RoutedObservationPoints};

fn config() -> Config {
    let mut config = super::qwen4_recurrent::config();
    config.experts.count = 3;
    config.experts.selected = 2;
    config.experts.intermediate = 2;
    config.experts.shared_intermediate = 4;
    config
}
fn spec(c: &Config) -> FeedForwardSublayerSpec {
    let projection = |name| {
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(),
            None,
            dense_linear_format(),
        )
        .unwrap()
    };
    let experts = GroupedGatedProductSpec::new(
        c.experts.count,
        c.hidden_size,
        c.experts.intermediate,
        c.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: projection("model.layers.0.mlp.experts.gate_up_proj"),
            down: projection("model.layers.0.mlp.experts.down_proj"),
        },
    )
    .unwrap();
    FeedForwardSublayerSpec::from_config(c, 7, "model.layers.0", experts, |_| {
        Ok(dense_linear_format())
    })
    .unwrap()
}
fn points() -> RoutedObservationPoints {
    RoutedObservationPoints::new(RoutedBankId::new(0), "model.layers.0.mlp", 3)
}
fn sig(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn projection(x: &[f64], weight: &[f32]) -> Vec<f64> {
    weight
        .chunks_exact(x.len())
        .map(|w| x.iter().zip(w).map(|(x, w)| x * *w as f64).sum())
        .collect()
}
fn oracle(input: &NumericTensor, c: &Config, p: &Parameters) -> NumericTensor {
    let weight = |suffix: &str| &p.0[&format!("model.layers.0.{suffix}")].data;
    let mut expected = vec![];
    for row in input.data.chunks_exact(6) {
        let norm: Vec<_> = row
            .chunks_exact(3)
            .enumerate()
            .flat_map(|(s, x)| {
                let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 3.
                    + c.norm_epsilon as f64)
                    .sqrt();
                (0..3).map(move |i| {
                    x[i] as f64 / rms
                        * (1. + weight("mlp_hyper_connection.hc_norm.weight")[s * 3 + i] as f64)
                })
            })
            .collect();
        let low: Vec<_> = projection(
            &norm,
            weight("mlp_hyper_connection.input_mix_weight_down.weight"),
        )
        .into_iter()
        .map(|x| {
            let x = x / 2.;
            x * sig(x)
        })
        .collect();
        let mix_weights = projection(
            &low,
            weight("mlp_hyper_connection.input_mix_weight_up.weight"),
        );
        let mixed: Vec<_> = (0..3)
            .map(|i| (norm[i] * sig(mix_weights[i]) + norm[i + 3] * sig(mix_weights[i + 3])) / 2.)
            .collect();
        let gates = projection(
            &norm,
            weight("mlp_hyper_connection.block_inject_weight.weight"),
        );
        let logits = projection(&mixed, weight("mlp.gate.weight"));
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = logits.iter().map(|v| (v - max).exp()).sum();
        let scores: Vec<_> = logits
            .iter()
            .map(|v| (v - max).exp() / denominator)
            .collect();
        let mut selected: Vec<_> = (0..3).collect();
        selected.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        assert!(
            (scores[selected[1]] - scores[selected[2]]).abs() > 1e-5,
            "unambiguous routing fixture"
        );
        selected.truncate(2);
        let normalization = if c.experts.renormalize {
            selected.iter().map(|&i| scores[i]).sum()
        } else {
            1.
        };
        selected.sort();
        let mut output = vec![0.; 3];
        for expert in selected {
            let both = projection(
                &mixed,
                &weight("mlp.experts.gate_up_proj")[expert * 12..(expert + 1) * 12],
            );
            let activation: Vec<_> = (0..2)
                .map(|i| both[i] * sig(both[i]) * both[i + 2])
                .collect();
            let write = projection(
                &activation,
                &weight("mlp.experts.down_proj")[expert * 6..(expert + 1) * 6],
            );
            for i in 0..3 {
                output[i] += scores[expert] / normalization * write[i];
            }
        }
        let gate = projection(&mixed, weight("mlp.shared_expert.gate_proj.weight"));
        let up = projection(&mixed, weight("mlp.shared_expert.up_proj.weight"));
        let product: Vec<_> = gate.iter().zip(up).map(|(g, u)| g * sig(*g) * u).collect();
        let shared = projection(&product, weight("mlp.shared_expert.down_proj.weight"));
        let shared_gate = sig(projection(&mixed, weight("mlp.shared_expert_gate.weight"))[0]);
        for s in 0..2 {
            for i in 0..3 {
                expected.push(
                    (row[s * 3 + i] as f64
                        + 2. * sig(gates[s] / 2.) * (output[i] + shared_gate * shared[i]))
                        as f32,
                );
            }
        }
    }
    NumericTensor::new(input.shape.clone(), expected)
}
#[derive(Default)]
struct Observer {
    values: BTreeMap<String, NumericTensor>,
    routes: usize,
    zero_write: bool,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observer {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        assert!(
            self.values.insert(path.into(), value.clone()).is_none(),
            "duplicate {path}"
        );
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok(
            (self.zero_write && path == "model.layers.0.feed_forward.write")
                .then(|| NumericTensor::zeros(value.shape.clone())),
        )
    }
    fn observe_routing(
        &mut self,
        event: eredu_runtime::RoutingObservation<'_, NumericTensor>,
    ) -> Result<(), Error> {
        assert_eq!(event.path, "model.layers.0.mlp");
        assert!(event.shared_output.is_some());
        self.routes += 1;
        Ok(())
    }
}
#[test]
fn qwen4_feed_forward_scalar_provider_and_intervention_parity() {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    for renormalize in [true, false] {
        let mut c = config();
        c.experts.renormalize = renormalize;
        let spec = spec(&c);
        assert_eq!(
            spec.feed_forward.router.arithmetic().scores,
            RoutingPrecision::Float32
        );
        assert_eq!(
            spec.feed_forward.experts.reduction(),
            GroupReduction::SequentialGroupOrder
        );
        let mut layer = FeedForwardSublayer::<NumericBackend>::new(spec, &context).unwrap();
        let mut p = Parameters::default();
        layer.visit_parameters_mut(&mut p);
        let input = NumericTensor::new(
            [2, 3, 2, 3],
            (0..36)
                .map(|i| ((i * 11 % 37) as f32 - 18.) / 13.)
                .collect(),
        );
        let expected = oracle(&input, &c, &p);
        let mut provider = RecordingNumericExpertProvider::default();
        let actual = layer
            .forward(
                &input,
                points(),
                &mut provider,
                &context,
                &mut ComponentInstrumentation::disabled(),
            )
            .unwrap();
        assert_tensor_close(&actual, &expected, "routed and shared residual equation");
        assert_eq!(provider.calls.len(), 1);
        assert_eq!(
            (provider.calls[0].0, provider.calls[0].1),
            (7, ExpertPass::Prefill)
        );
        assert_eq!(provider.calls[0].2.len(), 12);
        let mut observer = Observer::default();
        let observed = layer
            .forward(
                &input,
                points(),
                &mut ResidentExpertProvider,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut observer),
            )
            .unwrap();
        assert_tensor_exact(
            &observed,
            &actual,
            "observations retain routing and residual values",
        );
        assert_eq!(observer.routes, 1);
        let mut zero = Observer {
            zero_write: true,
            ..Default::default()
        };
        let masked = layer
            .forward(
                &input,
                points(),
                &mut ResidentExpertProvider,
                &context,
                &mut ComponentInstrumentation::new("model.layers.0", &mut zero),
            )
            .unwrap();
        assert_tensor_exact(&masked, &input, "zero write preserves all original streams");
        let pieces: Vec<_> = (0..3)
            .map(|t| {
                layer
                    .forward(
                        &input.axis_slice(1, t, t + 1),
                        points(),
                        &mut provider,
                        &context,
                        &mut ComponentInstrumentation::disabled(),
                    )
                    .unwrap()
            })
            .collect();
        assert_tensor_exact(
            &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
            &actual,
            "feed-forward decode/chunk parity",
        );
        assert!(provider.calls[1..]
            .iter()
            .all(|call| call.1 == ExpertPass::Decode));
    }
}
#[test]
fn qwen4_feed_forward_rejects_projection_and_residual_mismatch() {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut s = spec(&config());
    s.residual.injection = None;
    assert!(FeedForwardSublayer::<NumericBackend>::new(s, &context).is_err());
    let mut s = spec(&config());
    s.feed_forward.shared_gate.input += 1;
    assert!(FeedForwardSublayer::<NumericBackend>::new(s, &context).is_err());
}

#[test]
fn qwen4_feed_forward_tp2_tp4_reduces_before_residual_injection() {
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut c = config();
    c.experts.intermediate = 4;
    c.experts.shared_intermediate = 8;
    let global = spec(&c);
    let mut ordinary =
        FeedForwardSublayer::<NumericBackend>::new(global.clone(), &context).unwrap();
    let mut parameters = Parameters::default();
    ordinary.visit_parameters_mut(&mut parameters);
    let input = NumericTensor::new(
        [2, 3, 2, 3],
        (0..36)
            .map(|i| ((i * 11 % 37) as f32 - 18.) / 13.)
            .collect(),
    );
    let expected = ordinary
        .forward(
            &input,
            points(),
            &mut ResidentExpertProvider,
            &context,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap();
    struct Shards<'a> {
        parameters: &'a Parameters,
        rank: usize,
        ranks: usize,
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
            *value = if name.ends_with("experts.gate_up_proj") {
                NumericTensor::concatenate(
                    &[
                        full.axis_slice(
                            1,
                            self.rank * 4 / self.ranks,
                            (self.rank + 1) * 4 / self.ranks,
                        ),
                        full.axis_slice(
                            1,
                            4 + self.rank * 4 / self.ranks,
                            4 + (self.rank + 1) * 4 / self.ranks,
                        ),
                    ],
                    1,
                    &NumericContext::default(),
                )
                .unwrap()
            } else if name.ends_with("experts.down_proj") {
                slice(2)
            } else if name.ends_with("shared_expert.gate_proj.weight")
                || name.ends_with("shared_expert.up_proj.weight")
            {
                slice(0)
            } else if name.ends_with("shared_expert.down_proj.weight") {
                slice(1)
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
                    let mut local = global.clone();
                    let ff = &mut local.feed_forward;
                    ff.experts = ff
                        .experts
                        .clone()
                        .with_group_geometry(3, 4 / ranks as i32)
                        .unwrap();
                    ff.shared[0].output /= ranks as i32;
                    ff.shared[1].output /= ranks as i32;
                    ff.shared[2].input /= ranks as i32;
                    let group = group.clone();
                    let parameters = &parameters;
                    let input = &input;
                    scope.spawn(move || {
                        let context = NumericContext {
                            bind_checkpoint_values: true,
                            ..Default::default()
                        };
                        let parallel = NumericParallelContext::new(rank, group);
                        let mut layer =
                            FeedForwardSublayer::<NumericBackend>::new(local, &context).unwrap();
                        layer.visit_parameters_mut(&mut Shards {
                            parameters,
                            rank,
                            ranks,
                        });
                        let plain = layer
                            .forward_parallel(
                                input,
                                points(),
                                &mut ResidentExpertProvider,
                                &parallel,
                                &context,
                                &mut ComponentInstrumentation::disabled(),
                            )
                            .unwrap();
                        let mut observer = Observer::default();
                        let observed = layer
                            .forward_parallel(
                                input,
                                points(),
                                &mut ResidentExpertProvider,
                                &parallel,
                                &context,
                                &mut ComponentInstrumentation::new("model.layers.0", &mut observer),
                            )
                            .unwrap();
                        assert_tensor_exact(&plain, &observed, "TP component observations");
                        plain
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        for output in outputs {
            assert_tensor_close(&output, &expected, "TP routed plus shared injection");
        }
    }
}
