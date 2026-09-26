use super::*;

struct Parameters;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Parameters {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let seed = metadata.id.as_str().bytes().map(usize::from).sum::<usize>();
        for (index, item) in value.data.iter_mut().enumerate() {
            let delta = ((index * 17 + seed) % 97) as f32 - 48.0;
            *item = if metadata.id.as_str().contains("norm") {
                1.0 + delta * 0.001
            } else {
                delta * 0.003
            };
        }
    }
}

#[test]
fn lfm2_dense_and_routed_chunked_state_and_decode_match_full_prefill() {
    for sparse in [false, true] {
        for schedule in [
            vec!["conv", "full_attention", "conv"],
            vec!["conv", "conv", "conv"],
            vec!["full_attention", "full_attention", "full_attention"],
        ] {
            let args = lfm2::model_args_from_config_value(&serde_json::json!({
                "model_type": if sparse { "lfm2_moe" } else { "lfm2" },
                "vocab_size": 16, "hidden_size": 8, "intermediate_size": 12,
                "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                "max_position_embeddings": 64, "layer_types": schedule,
                "conv_L_cache": 4, "conv_bias": true,
                "block_auto_adjust_ff_dim": false, "tie_word_embeddings": false,
                "num_dense_layers": if sparse { 1 } else { 0 },
                "moe_intermediate_size": if sparse { 6 } else { 0 },
                "num_experts": if sparse { 2 } else { 0 },
                "num_experts_per_tok": if sparse { 2 } else { 0 },
                "norm_topk_prob": sparse, "use_expert_bias": sparse
            }))
            .unwrap();
            let context = NumericContext::default();
            let make = || {
                let model =
                    lfm2::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
                let state = DeviceState::<NumericBackend, _>::create(
                    model.state_layout().unwrap(),
                    |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                )
                .unwrap();
                let mut runtime = ResidentRuntime::new(model, &context).unwrap();
                runtime
                    .architecture_mut()
                    .static_modules_mut()
                    .visit_parameters_mut(&mut Parameters);
                for unit in runtime.units_mut().iter_mut().flatten() {
                    unit.visit_parameters_mut(&mut Parameters);
                }
                (runtime, state)
            };
            for length in [1, 2, 3, 4, 11] {
                let ids: Vec<_> = (0..length).map(|i| (i * 3 + 1) % 16).collect();
                let (mut reference, mut expected_state) = make();
                let tokens = NumericTensor::token_ids(&ids);
                let full = reference
                    .forward(
                        decoder::LayeredInput {
                            tokens: &tokens,
                            mask: None,
                        },
                        &mut expected_state,
                        &context,
                    )
                    .unwrap();
                assert!(full.data.iter().any(|v| v.abs() > 0.01));
                let prefill_state = expected_state.clone();
                let decodes: Vec<_> = [5, 8, 13]
                    .into_iter()
                    .map(|token| {
                        reference
                            .forward(
                                decoder::LayeredInput {
                                    tokens: &NumericTensor::token_ids(&[token]),
                                    mask: None,
                                },
                                &mut expected_state,
                                &context,
                            )
                            .unwrap()
                    })
                    .collect();
                for chunk in [1, 2, 3, 4, 32] {
                    let (runtime, mut state) = make();
                    let (architecture, units) = runtime.into_parts();
                    // The same portable traversal used by host/disk execution,
                    // holding exactly the supplied unit window.
                    let mut runtime =
                        LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
                    let outputs: Vec<_> = ids
                        .chunks(chunk)
                        .map(|ids| {
                            runtime
                                .forward(
                                    decoder::LayeredInput {
                                        tokens: &NumericTensor::token_ids(ids),
                                        mask: None,
                                    },
                                    &mut state,
                                    &context,
                                )
                                .unwrap()
                        })
                        .collect();
                    assert_tensor_close(
                        &NumericTensor::concatenate(&outputs, 1, &context).unwrap(),
                        &full,
                        "chunked prompt logits",
                    );
                    for layer in 0..3 {
                        assert_eq!(state.layer(layer).unwrap().position(), length as i32);
                        if schedule[layer] == "conv" {
                            let role = StateTensorRole::Convolution { slot: 0 };
                            let actual = state
                                .layer(layer)
                                .unwrap()
                                .fixed_component(role)
                                .unwrap()
                                .as_ref()
                                .unwrap()
                                .clone();
                            let mut expected = prefill_state.clone();
                            let history = expected
                                .layer(layer)
                                .unwrap()
                                .fixed_component(role)
                                .unwrap()
                                .as_ref()
                                .unwrap();
                            assert_eq!(actual.shape, [1, 3, 8]);
                            assert_tensor_close(&actual, history, "chunked convolution history");
                        }
                    }
                    for (token, expected) in [5, 8, 13].into_iter().zip(&decodes) {
                        let output = runtime
                            .forward(
                                decoder::LayeredInput {
                                    tokens: &NumericTensor::token_ids(&[token]),
                                    mask: None,
                                },
                                &mut state,
                                &context,
                            )
                            .unwrap();
                        assert_tensor_close(
                            &output,
                            expected,
                            "cached decode after chunked prefill",
                        );
                    }
                }
            }
        }
    }
}
