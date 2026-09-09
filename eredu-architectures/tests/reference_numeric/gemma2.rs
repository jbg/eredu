use super::*;
use eredu_architectures::gemma2 as family;
use std::collections::BTreeMap;

pub(super) fn tiny_config() -> serde_json::Value {
    let mut value = config("gemma2", true);
    value["num_hidden_layers"] = 4.into();
    value["sliding_window"] = 2.into();
    value["query_pre_attn_scalar"] = 7.0.into();
    value["attn_logit_softcapping"] = 0.3.into();
    value["final_logit_softcapping"] = 0.7.into();
    value
}

#[test]
fn gemma2_matches_scalar_oracle_across_windows_chunked_prefill_and_cached_decode() {
    let oracle: Vec<Vec<f32>> =
        serde_json::from_str(include_str!("../fixtures/gemma2/logits.json")).unwrap();
    let expected = NumericTensor::new(vec![1, 7, 17], oracle.into_iter().flatten().collect());
    let args = family::model_args_from_config_value(&tiny_config()).unwrap();
    let context = NumericContext::default();
    let state = || {
        DeviceState::<NumericBackend, _>::create(family::state_layout(&args).unwrap(), |_, p| {
            Ok::<_, Error>(NumericHybridLayerState::new(p))
        })
        .unwrap()
    };
    let mut model = ResidentRuntime::new(
        family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
        &context,
    )
    .unwrap();
    for chunks in [
        vec![vec![1, 3, 2, 4, 5, 6, 7]],
        vec![vec![1, 3, 2], vec![4], vec![5], vec![6], vec![7]],
        vec![vec![1, 3], vec![2, 4, 5], vec![6, 7]],
    ] {
        let mut cache = state();
        let mut output = Vec::new();
        for ids in chunks {
            output.extend(
                model
                    .forward(
                        decoder::LayeredInput {
                            tokens: &NumericTensor::token_ids(&ids),
                            mask: None,
                        },
                        &mut cache,
                        &context,
                    )
                    .unwrap()
                    .data,
            );
        }
        assert_tensor_close(
            &NumericTensor::new(vec![1, 7, 17], output),
            &expected,
            "Gemma 2 scalar oracle",
        );
        assert_eq!(
            cache
                .layer(0)
                .unwrap()
                .attention
                .as_ref()
                .unwrap()
                .keys
                .as_ref()
                .unwrap()
                .shape[2],
            2
        );
        assert_eq!(
            cache
                .layer(1)
                .unwrap()
                .attention
                .as_ref()
                .unwrap()
                .keys
                .as_ref()
                .unwrap()
                .shape[2],
            7
        );
        let snapshot = cache.clone();
        let token = NumericTensor::token_ids(&[8]);
        let next = model
            .forward(
                decoder::LayeredInput {
                    tokens: &token,
                    mask: None,
                },
                &mut cache,
                &context,
            )
            .unwrap();
        let mut restored = snapshot;
        let replay = model
            .forward(
                decoder::LayeredInput {
                    tokens: &token,
                    mask: None,
                },
                &mut restored,
                &context,
            )
            .unwrap();
        assert_tensor_exact(&next, &replay, "snapshot/fork across sliding window");
    }
}

#[test]
fn gemma2_prepared_session_and_bounded_units_agree() {
    let value = tiny_config();
    let args = family::model_args_from_config_value(&value).unwrap();
    let (root, _) = prepared_adapter::payload_fixture_config(&value, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let prepared = execute_numeric_replicated_inspection(
        &inspection,
        &context,
        &NumericTensor::token_ids(&[1, 3, 2]),
        None,
    );
    assert_eq!(last_reference_stage_evidence().family, "gemma2");
    let context = NumericContext::default();
    let model = family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let mut bounded = LayerwiseRuntime::new(model, RebuildingUnitPolicy::default());
    let mut state =
        DeviceState::<NumericBackend, _>::create(family::state_layout(&args).unwrap(), |_, p| {
            Ok::<_, Error>(NumericHybridLayerState::new(p))
        })
        .unwrap();
    for (ids, expected) in [&[1, 3, 2][..], &[4][..], &[5][..]]
        .into_iter()
        .zip(&prepared.outputs)
    {
        let out = bounded
            .forward(
                decoder::LayeredInput {
                    tokens: &NumericTensor::token_ids(ids),
                    mask: None,
                },
                &mut state,
                &context,
            )
            .unwrap();
        assert_tensor_close(
            &NumericTensor::new(vec![1, 1, 17], out.data[out.data.len() - 17..].to_vec()),
            expected,
            "Gemma 2 bounded/prepared",
        );
    }
    assert_state_exact(&state, &prepared.state, 4, "bounded KV state");
}

#[test]
fn gemma2_actual_payload_tp_pp_and_combined_match_replicated() {
    for quantization in {
        [
            None,
            Some(eredu_core::QuantizationRequest::Affine {
                bits: 4,
                group_size: 16,
            }),
        ]
    } {
        let mut value = tiny_config();
        value["hidden_size"] = 32.into();
        value["intermediate_size"] = 64.into();
        value["head_dim"] = 8.into();
        value["num_attention_heads"] = 8.into();
        value["num_key_value_heads"] = 2.into();
        let args = family::model_args_from_config_value(&value).unwrap();
        let (root, _) = prepared_adapter::payload_fixture_config(&value, 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let reference_context = NumericContext {
            bind_checkpoint_values: true,
            ..Default::default()
        };
        let inputs = [
            NumericTensor::token_ids(&[1, 3, 2]),
            NumericTensor::token_ids(&[4]),
            NumericTensor::token_ids(&[5]),
        ];
        let expected = execute_numeric_replicated_inspection(
            &inspection,
            &reference_context,
            &inputs[0],
            quantization,
        );
        for topology in [
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            ParallelTopology::new(1, 2, 1, 1).unwrap(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
            ParallelTopology::new(1, 4, 1, 1).unwrap(),
        ] {
            let world = Arc::new(NumericPartitionWorld::default());
            let actual = std::thread::scope(|scope| {
                let threads = (0..topology.world_size()).map(|rank| {
                    let world = Arc::clone(&world);
                    let args = &args;
                    let inspection = &inspection;
                    let inputs = &inputs;
                    scope.spawn(move || {
                        use eredu_architectures::partitioned_execution::derive_partitioned_local_layout;
                        let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                        let plan = prepared_adapter::plan(quantization)
                            .with_topology(topology)
                            .with_prompt_cache_persistence(true);
                        let sources = partitioned_adapter::prepare_plan(
                            inspection, &plan, rank, std::time::Duration::from_secs(10),
                        ).unwrap();
                        let formats = sources.selected().text_realization().parameters()
                            .iter().filter_map(|p| {
                                p.executable().weight_quantization()
                                    .map(|format| (p.name().to_owned(), format))
                            }).collect();
                        let selected_args = family::with_checkpoint_formats(args, formats).unwrap();
                        let description = decoder::dense_parameter_description(&selected_args).unwrap();
                        let layout = derive_partitioned_local_layout(&description, rank_topology).unwrap();
                        let mut context = NumericContext::with_partition(layout, rank, world);
                        if quantization.is_some() {
                            let expanded = decoder::dense_parameter_description(args).unwrap();
                            context.expanded_weight_layout = Some(Arc::new(
                                derive_partitioned_local_layout(&expanded, rank_topology).unwrap(),
                            ));
                        }
                        context.bind_checkpoint_values = true;
                        let mut executable = partitioned_adapter::dense(sources, &context).unwrap();
                        let outputs = inputs.iter().enumerate()
                            .map(|(i, tokens)| executable.forward(tokens, i == 0).unwrap())
                            .collect::<Vec<_>>();
                        assert!(executable.positions().unwrap().iter().all(|p| *p == 5));
                        executable.reset().unwrap();
                        let replay = executable.forward(&inputs[0], true).unwrap();
                        assert_tensor_exact(&replay, &outputs[0], "partition reset");
                        outputs
                    })
                }).collect::<Vec<_>>();
                threads
                    .into_iter()
                    .map(|thread| thread.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for rank in actual {
                for (output, expected) in rank.iter().zip(&expected.outputs) {
                    assert_tensor_close(output, expected, "Gemma 2 TP/PP post-norm reductions");
                }
            }
        }
    }
}

#[test]
fn gemma2_published_gguf_layout_preserves_logits_norm_convention_and_quantization() {
    use eredu_gguf::{GgmlType, MetadataValue as M, TensorInput, Writer};
    let mut value = tiny_config();
    value["hidden_size"] = 32.into();
    value["intermediate_size"] = 64.into();
    value["head_dim"] = 8.into();
    value["query_pre_attn_scalar"] = 8.into();
    value["vocab_size"] = 32.into();
    value["num_attention_heads"] = 4.into();
    value["num_key_value_heads"] = 2.into();
    let (root, physical) = prepared_adapter::payload_fixture_config(&value, 1.0);
    let safe = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let input = NumericTensor::token_ids(&[1, 3, 2]);
    let expected = execute_numeric_replicated_inspection(&safe, &context, &input, None);
    let names = |name: &str| {
        name.replace("model.embed_tokens", "token_embd")
            .replace("model.norm", "output_norm")
            .replace("model.layers.", "blk.")
            .replace("self_attn.q_proj", "attn_q")
            .replace("self_attn.k_proj", "attn_k")
            .replace("self_attn.v_proj", "attn_v")
            .replace("self_attn.o_proj", "attn_output")
            .replace("mlp.gate_proj", "ffn_gate")
            .replace("mlp.up_proj", "ffn_up")
            .replace("mlp.down_proj", "ffn_down")
            .replace("input_layernorm", "attn_norm")
            .replace("post_attention_layernorm", "attn_post_norm")
            .replace("pre_feedforward_layernorm", "ffn_norm")
            .replace("post_feedforward_layernorm", "ffn_post_norm")
    };
    let tensors = physical
        .into_iter()
        .map(|(name, (shape, bits))| {
            let values = bits.into_iter().map(|v| {
                if name.contains("norm") {
                    (f32::from_bits(v) + 1.0).to_bits()
                } else {
                    v
                }
            });
            (
                names(&name),
                shape
                    .into_iter()
                    .rev()
                    .map(|d| d as u64)
                    .collect::<Vec<_>>(),
                values.flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let metadata = BTreeMap::from([
        ("general.architecture".into(), M::String("gemma2".into())),
        ("gemma2.embedding_length".into(), M::Uint32(32)),
        ("gemma2.block_count".into(), M::Uint32(4)),
        ("gemma2.attention.head_count".into(), M::Uint32(4)),
        ("gemma2.attention.head_count_kv".into(), M::Uint32(2)),
        ("gemma2.attention.key_length".into(), M::Uint32(8)),
        ("gemma2.attention.value_length".into(), M::Uint32(8)),
        ("gemma2.feed_forward_length".into(), M::Uint32(64)),
        (
            "gemma2.attention.layer_norm_rms_epsilon".into(),
            M::Float32(1e-5),
        ),
        ("gemma2.rope.freq_base".into(), M::Float32(10000.0)),
        ("gemma2.vocab_size".into(), M::Uint32(32)),
        ("gemma2.context_length".into(), M::Uint32(64)),
        ("gemma2.attention.sliding_window".into(), M::Uint32(2)),
        ("gemma2.attn_logit_softcapping".into(), M::Float32(0.3)),
        ("gemma2.final_logit_softcapping".into(), M::Float32(0.7)),
    ]);
    let path = root.path().join("model.gguf");
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &metadata,
            &tensors
                .iter()
                .map(|(name, dimensions, data)| TensorInput {
                    name,
                    dimensions,
                    data,
                    ggml_type: GgmlType::F32,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let gguf = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
    assert_eq!(
        gguf.architecture_plan().architecture_descriptor(),
        safe.architecture_plan().architecture_descriptor()
    );
    let actual = execute_numeric_replicated_inspection(&gguf, &context, &input, None);
    for (a, e) in actual.outputs.iter().zip(&expected.outputs) {
        assert_tensor_close(a, e, "Gemma 2 GGUF/HF scale conventions");
    }
    let packed = tensors
        .iter()
        .map(|(name, shape, data)| {
            let (encoding, bytes) = if shape.len() == 2 {
                let block = [vec![0x00, 0x3c], (0..32).collect::<Vec<u8>>()].concat();
                (
                    GgmlType::Q8_0,
                    block.repeat(shape.iter().product::<u64>() as usize / 32),
                )
            } else {
                (GgmlType::F32, data.clone())
            };
            (name, shape, encoding, bytes)
        })
        .collect::<Vec<_>>();
    let path = root.path().join("packed.gguf");
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &metadata,
            &packed
                .iter()
                .map(|(name, dimensions, encoding, data)| TensorInput {
                    name,
                    dimensions,
                    data,
                    ggml_type: *encoding,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let parameter = requirements
        .parameters()
        .iter()
        .find(|p| p.name() == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert!(matches!(
        parameter.source_encoding(),
        Some(eredu_checkpoint::SourceTensorEncoding::Gguf {
            ggml_type: GgmlType::Q8_0,
            ..
        })
    ));
    assert_eq!(parameter.logical_shape(), &[32, 32]);
}
