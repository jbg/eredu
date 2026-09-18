//! Disabled observation uses the same routed block equations and cache updates.
use super::*;
use eredu_runtime::ResidentExpertProvider;

struct Observer {
    enabled: bool,
    paths: Vec<String>,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observer {
    fn observes_activations(&self) -> bool {
        self.enabled
    }
    fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
        self.paths.push(path.to_owned());
        Ok(())
    }
}

#[test]
fn v3_disabled_observation_preserves_routed_serial_and_tp_equations() {
    let args = deepseek::parse_v3_config(&serde_json::json!({
        "model_type":"deepseek_v3", "hidden_size":4, "intermediate_size":8,
        "moe_intermediate_size":4, "num_hidden_layers":1, "num_attention_heads":2,
        "vocab_size":8, "max_position_embeddings":64, "q_lora_rank":2,
        "kv_lora_rank":2, "qk_nope_head_dim":2, "qk_rope_head_dim":2,
        "v_head_dim":2, "first_k_dense_replace":0, "n_routed_experts":2,
        "n_shared_experts":1, "num_experts_per_tok":1, "n_group":1,
        "topk_group":1, "tie_word_embeddings":false
    }))
    .unwrap();
    let context = NumericContext::default();
    let inputs = [
        NumericTensor::new(
            vec![1, 2, 4],
            vec![0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3],
        ),
        NumericTensor::new(vec![1, 1, 4], vec![0.1, 0.3, -0.4, 0.2]),
    ];
    // Ordinary, disabled-observation, and actual observed entry points; each
    // route uses independent mutable cache and module owners.
    for route in 0..3 {
        let mut expected = None;
        for mode in 0..3 {
            let mut block =
                deepseek::block::V3Block::<NumericBackend>::new(&args, 0, &context).unwrap();
            let mut cache = NumericCompressedCache::resident();
            let mut observer = Observer {
                enabled: mode == 2,
                paths: Vec::new(),
            };
            let mut outputs = Vec::new();
            for (step, input) in inputs.iter().enumerate() {
                let pass = if step == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let output = if mode == 0 {
                    match route {
                        0 => block.forward_with_provider(
                            input,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                        ),
                        1 => block.forward_parallel(
                            input,
                            None,
                            Some(&mut cache),
                            &context,
                            |v, _| Ok(v),
                        ),
                        _ => block.forward_parallel_with_provider(
                            input,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                            |v, _| Ok(v),
                        ),
                    }
                } else {
                    match route {
                        0 => block.forward_observed_with_provider(
                            "layers.0",
                            input,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                            &mut observer,
                        ),
                        1 => block.forward_parallel_observed(
                            "layers.0",
                            input,
                            None,
                            Some(&mut cache),
                            &context,
                            &mut observer,
                            |v, _| Ok(v),
                        ),
                        _ => block.forward_parallel_observed_with_provider(
                            "layers.0",
                            input,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                            &mut observer,
                            |v, _| Ok(v),
                        ),
                    }
                }
                .unwrap();
                assert!(output.data.iter().any(|v| v.abs() > 1e-6));
                assert_eq!(cache.offset(), if step == 0 { 2 } else { 3 });
                outputs.push(output);
            }
            if let Some(expected) = &expected {
                for (actual, expected) in outputs.iter().zip(expected) {
                    assert_tensor_exact(actual, expected, "V3 routed observation parity");
                }
            } else {
                expected = Some(outputs);
            }
            assert_eq!(observer.paths.is_empty(), mode != 2);
            if mode == 2 {
                assert!(
                    observer
                        .paths
                        .iter()
                        .any(|path| path.contains("feed_forward.shared"))
                );
            }
        }
    }
}

#[test]
fn v4_disabled_observation_preserves_routed_serial_and_tp_equations() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    let inputs = [
        NumericTensor::new(
            vec![1, 8, 2, 4],
            (0..64).map(|i| i as f32 / 64.0 - 0.5).collect(),
        ),
        NumericTensor::new(
            vec![1, 1, 2, 4],
            vec![0.1, 0.3, -0.4, 0.2, 0.4, -0.3, 0.1, 0.2],
        ),
    ];
    for route in 0..3 {
        let mut expected = None;
        for mode in 0..3 {
            let mut block =
                deepseek::block::V4Block::<NumericBackend>::new(&args, 1, &context).unwrap();
            let mut cache = NumericPoolingCache::new(args.sliding_window, &[4, 4]);
            let mut observer = Observer {
                enabled: mode == 2,
                paths: Vec::new(),
            };
            let mut outputs = Vec::new();
            for (step, input) in inputs.iter().enumerate() {
                let tokens = if step == 0 {
                    NumericTensor::token_ids(&[1, 2, 3, 4, 5, 6, 7, 8])
                } else {
                    NumericTensor::token_ids(&[9])
                };
                let pass = if step == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let output = if mode == 0 {
                    match route {
                        0 => block.forward_with_provider(
                            input,
                            &tokens,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                        ),
                        1 => block.forward_parallel(
                            input,
                            &tokens,
                            None,
                            Some(&mut cache),
                            &context,
                            |v, _| Ok(v),
                        ),
                        _ => block.forward_parallel_with_provider(
                            input,
                            &tokens,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                            |v, _| Ok(v),
                        ),
                    }
                } else {
                    match route {
                        0 => block.forward_observed_with_provider(
                            "layers.1",
                            input,
                            &tokens,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                            &mut observer,
                        ),
                        1 => block.forward_parallel_observed(
                            "layers.1",
                            input,
                            &tokens,
                            None,
                            Some(&mut cache),
                            &context,
                            &mut observer,
                            |v, _| Ok(v),
                        ),
                        _ => block.forward_parallel_observed_with_provider(
                            "layers.1",
                            input,
                            &tokens,
                            None,
                            Some(&mut cache),
                            pass,
                            &mut ResidentExpertProvider,
                            &context,
                            &mut observer,
                            |v, _| Ok(v),
                        ),
                    }
                }
                .unwrap();
                assert!(output.data.iter().any(|v| v.abs() > 1e-6));
                assert_eq!(cache.offset(), if step == 0 { 8 } else { 9 });
                outputs.push(output);
            }
            if let Some(expected) = &expected {
                for (actual, expected) in outputs.iter().zip(expected) {
                    assert_tensor_exact(actual, expected, "V4 routed observation parity");
                }
            } else {
                expected = Some(outputs);
            }
            assert_eq!(observer.paths.is_empty(), mode != 2);
            if mode == 2 {
                assert!(
                    observer
                        .paths
                        .iter()
                        .any(|path| path.contains("feed_forward.shared"))
                );
            }
        }
    }
}
