//! Recurrent component equations use the declared unequal head and decay geometry.
use super::*;
use eredu_core::component::*;

#[derive(Default)]
struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a NumericTensor) {
        self.0.insert(metadata.id().to_string(), value.clone());
    }
}
struct Initialize;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Initialize {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
        let name = metadata.id().as_str();
        let seed = name
            .bytes()
            .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
        for (i, x) in value.data.iter_mut().enumerate() {
            let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
            *x = if name.ends_with("norm.weight") {
                1.0 + delta * 0.003
            } else if name.ends_with("A_log") {
                -0.5 + delta * 0.01
            } else {
                delta * 0.02
            };
        }
    }
}
fn affine(input: &NumericTensor, weight: &NumericTensor) -> NumericTensor {
    let input_width = input.shape[2] as usize;
    let output_width = weight.shape[0] as usize;
    NumericTensor::new(
        [1, input.shape[1], output_width as i32],
        input
            .data
            .chunks_exact(input_width)
            .flat_map(|row| {
                weight.data.chunks_exact(input_width).map(move |w| {
                    row.iter()
                        .zip(w)
                        .map(|(x, w)| *x as f64 * *w as f64)
                        .sum::<f64>() as f32
                })
            })
            .collect(),
    )
}

#[test]
fn qwen_recurrent_channels_reconstruct_declared_recurrence_and_cached_masks() {
    for mut config in heterogeneous_replicated_configs().into_iter().filter(|c| {
        matches!(
            c["model_type"].as_str(),
            Some("qwen3_next" | "qwen3_5_text")
        )
    }) {
        for (kh, vh, kd, vd) in [(2, 2, 3, 4), (2, 4, 3, 5)] {
            config["linear_num_key_heads"] = kh.into();
            config["linear_num_value_heads"] = vh.into();
            config["linear_key_head_dim"] = kd.into();
            config["linear_value_head_dim"] = vd.into();
            let args = qwen::hybrid::model_args_from_config_value(&config)
                .unwrap()
                .text;
            let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&config)
                .unwrap()
                .architecture_plan()
                .architecture_descriptor();
            let group = descriptor
                .components
                .iter()
                .find(|g| {
                    matches!(
                        g.activation_equation,
                        ComponentActivation::GatedDeltaAttention { .. }
                    )
                })
                .unwrap();
            let ComponentActivation::GatedDeltaAttention {
                key_heads,
                value_heads,
                key_head_width,
                value_head_width,
                decay_layout,
                query_scale,
                key_scale,
                decay_rate,
                decay_bias,
                channel_normalization,
                output_gate,
            } = &group.activation_equation
            else {
                unreachable!()
            };
            assert_eq!(
                (*key_heads, *value_heads, *key_head_width, *value_head_width),
                (kh as usize, vh as usize, kd as usize, vd as usize)
            );
            assert_eq!(*decay_layout, ComponentDeltaDecay::Head);
            assert_eq!(
                *output_gate,
                ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0)
                }
            );
            let (kh, vh, kd, vd) = (*key_heads, *value_heads, *key_head_width, *value_head_width);
            let (key_width, channels) = (kh * kd, vh * vd);
            assert_eq!(group.count, channels);
            let context = NumericContext::default();
            let mut layer =
                qwen::hybrid::LinearAttention::<NumericBackend>::new(&args, 0, &context).unwrap();
            layer.visit_parameters_mut(&mut Initialize);
            let mut parameters = Parameters::default();
            layer.visit_parameters(&mut parameters);
            let layout = qwen::hybrid::state_layout(&args).unwrap();
            let mut state = NumericHybridLayerState::new(layout.layer(0).unwrap());
            let mut history = Vec::<f32>::new();
            let mut reference_state = vec![0.0_f64; vh * kd * vd];
            for (step, sequence) in [3, 1, 2].into_iter().enumerate() {
                let input = NumericTensor::new(
                    [1, sequence, 8],
                    (0..sequence * 8)
                        .map(|i| ((i * 11 + step as i32 * 7) % 31) as f32 / 13.0 - 1.0)
                        .collect(),
                );
                let before = state.clone();
                let ordinary = layer.forward(&input, &mut state, &context).unwrap();
                let mut observed_state = before.clone();
                let mut captured = Components::strict();
                let actual = layer
                    .forward_instrumented(
                        &input,
                        &mut observed_state,
                        &context,
                        &mut ComponentInstrumentation::new("model.layers.0", &mut captured),
                    )
                    .unwrap();
                assert_tensor_exact(
                    &actual,
                    &ordinary,
                    "recurrent observation preserves ordinary output",
                );
                for read in &group.reads {
                    let expected = affine(&input, &parameters.0[&read.weight]);
                    assert_tensor_close(
                        &expected,
                        &captured.values[read.projection_output.as_ref().unwrap()],
                        "declared recurrent read projection",
                    );
                    for component in 0..channels {
                        assert!(
                            read.rows.row_range(component).unwrap().end
                                <= expected.shape[2] as usize
                        );
                    }
                    if matches!(read.role, ComponentReadRole::Query | ComponentReadRole::Key) {
                        let norm = read.head_normalization.as_ref().unwrap();
                        assert_eq!(norm.normalization.kind, ComponentNormalizationKind::L2);
                        assert_eq!(norm.normalization.epsilon.value(), 1e-6);
                    }
                }
                let transform = &descriptor.component_transforms[0];
                let ComponentTensorTransformEquation::CausalDepthwiseConvolution {
                    kernel,
                    channels: fused_width,
                    taps,
                    residual,
                    activation,
                } = &transform.equation
                else {
                    panic!("causal fused QKV")
                };
                assert!(!residual);
                assert_eq!(*fused_width, 2 * key_width + channels);
                assert_eq!(
                    *activation,
                    Some(ComponentNonlinearity::Silu {
                        multiplier: ComponentScalar::new(1.0)
                    })
                );
                let projected = &captured.values[&transform.input];
                let offset = history.len() / fused_width;
                history.extend_from_slice(&projected.data);
                let convolved = NumericTensor::new(
                    projected.shape.clone(),
                    (0..sequence as usize * fused_width)
                        .map(|i| {
                            let time = offset + i / fused_width;
                            let channel = i % fused_width;
                            let z = (0..*taps)
                                .filter_map(|tap| {
                                    (time + tap + 1).checked_sub(*taps).map(|t| {
                                        history[t * fused_width + channel] as f64
                                            * parameters.0[&kernel.parameter].data
                                                [channel * taps + tap]
                                                as f64
                                    })
                                })
                                .sum::<f64>();
                            (z / (1.0 + (-z).exp())) as f32
                        })
                        .collect(),
                );
                assert_tensor_close(
                    &convolved,
                    &captured.values[&transform.output],
                    "causal recurrent convolution retains preceding tokens",
                );
                let update = &captured.values["model.layers.0.mixer.update.projected"];
                let decay = &captured.values["model.layers.0.mixer.decay.projected"];
                let gate = &captured.values["model.layers.0.mixer.gate.projected"];
                let gain =
                    &parameters.0[channel_normalization.normalization.gain.as_ref().unwrap()];
                let mut expected_channels = vec![];
                for t in 0..sequence as usize {
                    let row = &convolved.data[t * fused_width..(t + 1) * fused_width];
                    for h in 0..vh {
                        let key_head = h / (vh / kh);
                        let normalize = |start: usize, scale: f32| {
                            let x = &row[start..start + kd];
                            let denominator =
                                (x.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() + 1e-6).sqrt();
                            x.iter()
                                .map(|x| *x as f64 / denominator * scale as f64)
                                .collect::<Vec<_>>()
                        };
                        let q = normalize(key_head * kd, query_scale.value());
                        let k = normalize(key_width + key_head * kd, key_scale.value());
                        let log_rate = parameters.0[&decay_rate.parameter].data[h] as f64;
                        let x = decay.data[t * vh + h] as f64
                            + parameters.0[&decay_bias.parameter].data[h] as f64;
                        let factor =
                            (-log_rate.exp() * (x.max(0.0) + (-x.abs()).exp().ln_1p())).exp();
                        let beta = 1.0 / (1.0 + (-(update.data[t * vh + h] as f64)).exp());
                        let s = &mut reference_state[h * kd * vd..(h + 1) * kd * vd];
                        for x in s.iter_mut() {
                            *x *= factor;
                        }
                        for v in 0..vd {
                            let delta = (row[2 * key_width + h * vd + v] as f64
                                - (0..kd).map(|kdim| k[kdim] * s[kdim * vd + v]).sum::<f64>())
                                * beta;
                            for kdim in 0..kd {
                                s[kdim * vd + v] += k[kdim] * delta;
                            }
                        }
                        let y = (0..vd)
                            .map(|v| (0..kd).map(|kdim| q[kdim] * s[kdim * vd + v]).sum::<f64>())
                            .collect::<Vec<_>>();
                        let rms = (y.iter().map(|x| x * x).sum::<f64>() / vd as f64
                            + channel_normalization.normalization.epsilon.value() as f64)
                            .sqrt();
                        expected_channels.extend((0..vd).map(|v| {
                            let z = gate.data[t * channels + h * vd + v] as f64;
                            (y[v] / rms * gain.data[v] as f64 * z / (1.0 + (-z).exp())) as f32
                        }));
                    }
                }
                let values = &captured.values[&group.activation];
                assert!(values.data.iter().any(|x| x.abs() > 1e-5));
                assert_tensor_close(
                    &NumericTensor::new(values.shape.clone(), expected_channels),
                    values,
                    "declared gated-delta equation with unequal heads and cached state",
                );
                assert_tensor_exact(
                    values,
                    &captured.values[group.write_input.as_ref().unwrap()],
                    "consumed recurrent projection input",
                );
                assert_tensor_close(
                    &affine(values, &parameters.0[&group.write_weight]),
                    &actual,
                    "signed recurrent channel write reconstruction",
                );
                for keep in [false, true] {
                    let selection = Some((
                        "model.layers.0.mixer.channels",
                        sequence as usize - 1,
                        channels - 1,
                    ));
                    let mut mask = Components {
                        zero: (!keep).then_some(selection).flatten(),
                        keep: keep.then_some(selection).flatten(),
                        ..Components::strict()
                    };
                    let mut masked_state = before.clone();
                    let masked = layer
                        .forward_instrumented(
                            &input,
                            &mut masked_state,
                            &context,
                            &mut ComponentInstrumentation::new("model.layers.0", &mut mask),
                        )
                        .unwrap();
                    let effective = &mask.values[&group.effective_activation];
                    assert_tensor_close(
                        &affine(effective, &parameters.0[&group.write_weight]),
                        &masked,
                        "masked channel projection uses effective values",
                    );
                    assert_eq!(
                        &masked.data[..(sequence as usize - 1) * 8],
                        &actual.data[..(sequence as usize - 1) * 8]
                    );
                    assert_ne!(
                        &masked.data[(sequence as usize - 1) * 8..],
                        &actual.data[(sequence as usize - 1) * 8..]
                    );
                    for channel in 0..channels {
                        if (channel == channels - 1) != keep {
                            assert_eq!(
                                effective.data[(sequence as usize - 1) * channels + channel],
                                0.0
                            );
                        }
                    }
                    let next = NumericTensor::new(
                        [1, 1, 8],
                        vec![0.13, -0.22, 0.4, 0.6, -0.1, 0.35, -0.21, 0.08],
                    );
                    assert_tensor_exact(
                        &layer.forward(&next, &mut masked_state, &context).unwrap(),
                        &layer.forward(&next, &mut state.clone(), &context).unwrap(),
                        "output-channel masks do not rewrite recurrent state",
                    );
                }
            }
        }
    }
}
