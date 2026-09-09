use super::*;
use eredu_architectures::nanbeige as family;

pub(super) fn tiny_config(skip_norm: bool) -> serde_json::Value {
    let mut value = config("nanbeige", false);
    value["num_hidden_layers"] = 2.into();
    value["num_attention_heads"] = 4.into();
    value["num_key_value_heads"] = 2.into();
    value["num_loops"] = 2.into();
    value["skip_loop_final_norm"] = skip_norm.into();
    value["rope_theta"] = 70_000_000.into();
    value
}

// Bind the independent fixture's physical tensors into logical invocations.
// This deliberately does not call the production alias planner.
struct FixtureAliases;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for FixtureAliases {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let Some((layer, suffix)) = name
            .strip_prefix("model.layers.")
            .and_then(|s| s.split_once('.'))
        else {
            return;
        };
        let layer: usize = layer.parse().unwrap();
        let source = if suffix == "output_norm.weight" {
            "model.norm.weight".to_owned()
        } else {
            format!("model.layers.{}.{suffix}", layer % 2)
        };
        *value = parameter(
            &ParameterSpec::trainable(source.clone()).unwrap(),
            value.shape.clone(),
            source.contains("norm"),
        );
    }
}

#[derive(Default)]
pub(super) struct FixturePolicy {
    acquired: Vec<usize>,
}
impl<U: Parameterized<NumericTensor>> LayerwisePolicy<NumericBackend, U> for FixturePolicy {
    type Lease = RebuiltUnitLease<U>;
    type Error = &'static str;
    fn begin(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Self::Error> {
        Ok(())
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        _: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&NumericContext) -> Result<U, E>,
    {
        let mut unit = build(context).map_err(LayerwiseAcquireError::Architecture)?;
        unit.visit_parameters_mut(&mut FixtureAliases);
        self.acquired.push(ordinal);
        Ok(RebuiltUnitLease(unit))
    }
    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        _: usize,
        _: ExecutionUnitAddress,
        _: Self::Lease,
        _: &'a NumericTensor,
        _: StateValues,
        _: ContextValues,
        _: &NumericContext,
    ) -> Result<(), Self::Error>
    where
        StateValues: Iterator<Item = &'a NumericTensor>,
        ContextValues: Iterator<Item = &'a NumericTensor>,
    {
        Ok(())
    }
    fn finish(&mut self, _: &NumericTensor, _: &NumericContext) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[test]
fn nanbeige_logits_match_independent_scalar_oracle_and_incremental_cache() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/nanbeige/logits.json")).unwrap();
    let context = NumericContext::default();
    for loops in [1, 2, 3] {
        for skip in [false, true] {
            let mut value = tiny_config(skip);
            value["num_loops"] = loops.into();
            let args = family::model_args_from_config_value(&value).unwrap();
            let layout = family::state_layout(&args).unwrap();
            let state = || {
                DeviceState::<NumericBackend, _>::create(layout.clone(), |_, p| {
                    Ok::<_, Error>(NumericHybridLayerState::new(p))
                })
                .unwrap()
            };
            let mut incremental_state = state();
            let mut resident = ResidentRuntime::new(
                family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap(),
                &context,
            )
            .unwrap();
            for unit in resident.units_mut().iter_mut().flatten() {
                unit.visit_parameters_mut(&mut FixtureAliases);
            }
            let key = format!("{loops}_{}", if skip { "True" } else { "False" });
            let expected = NumericTensor::new(
                vec![1, 5, 17],
                oracle[&key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                    })
                    .collect(),
            );
            let tokens = NumericTensor::token_ids(&[1, 3, 2, 4, 5]);
            let full = resident
                .forward(
                    decoder::LayeredInput {
                        tokens: &tokens,
                        mask: None,
                    },
                    &mut state(),
                    &context,
                )
                .unwrap();
            assert_tensor_close(&full, &expected, &key);
            let mut incremental = Vec::new();
            for token_ids in [&[1, 3, 2][..], &[4][..], &[5][..]] {
                let tokens = NumericTensor::token_ids(token_ids);
                incremental.extend(
                    resident
                        .forward(
                            decoder::LayeredInput {
                                tokens: &tokens,
                                mask: None,
                            },
                            &mut incremental_state,
                            &context,
                        )
                        .unwrap()
                        .data,
                );
            }
            assert_tensor_close(
                &NumericTensor::new(vec![1, 5, 17], incremental),
                &expected,
                "incremental logical caches",
            );
            for layer in 0..layout.len() {
                assert_eq!(incremental_state.layer(layer).unwrap().position(), 5);
            }
            if loops > 1 {
                let first = incremental_state
                    .layer(0)
                    .unwrap()
                    .attention
                    .as_ref()
                    .unwrap()
                    .keys
                    .as_ref()
                    .unwrap()
                    .clone();
                let second = incremental_state
                    .layer(2)
                    .unwrap()
                    .attention
                    .as_ref()
                    .unwrap()
                    .keys
                    .as_ref()
                    .unwrap()
                    .clone();
                assert_ne!(
                    first.data, second.data,
                    "different passes must not alias their KV state"
                );
            }
        }
    }
}

#[test]
fn nanbeige_prepared_session_and_block_streaming_share_checkpoint_values_and_state() {
    let value = tiny_config(false);
    let args = family::model_args_from_config_value(&value).unwrap();
    let context = NumericContext::default();
    let tokens = NumericTensor::token_ids(&[1, 3, 2]);
    let (root, _) = prepared_adapter::payload_fixture_config(&value, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let bound_context = NumericContext {
        bind_checkpoint_values: true,
        ..NumericContext::default()
    };
    let prepared =
        execute_numeric_replicated_inspection(&inspection, &bound_context, &tokens, None);
    assert_eq!(last_reference_stage_evidence().family, "nanbeige");
    let model = family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let parameters = model.parameter_description(&context).unwrap();
    assert_eq!(parameters.unit_layout().group_range(0).unwrap().len(), 4);
    let mut streamed = LayerwiseRuntime::new(model, FixturePolicy::default());
    let mut state =
        DeviceState::<NumericBackend, _>::create(family::state_layout(&args).unwrap(), |_, p| {
            Ok::<_, Error>(NumericHybridLayerState::new(p))
        })
        .unwrap();
    for (ids, expected) in [&[1, 3, 2][..], &[4][..], &[5][..]]
        .into_iter()
        .zip(&prepared.outputs)
    {
        let tokens = NumericTensor::token_ids(ids);
        let actual = streamed
            .forward(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                &mut state,
                &context,
            )
            .unwrap();
        let last = NumericTensor::new(
            vec![1, 1, 17],
            actual.data[actual.data.len() - 17..].to_vec(),
        );
        assert_tensor_close(&last, expected, "prepared/streamed repetition");
    }
    assert_eq!(streamed.policy().acquired, [0, 1, 2, 3].repeat(3));
    assert_state_exact(&state, &prepared.state, 4, "prepared/streamed loop caches");
    let snapshot = state.clone();
    let tokens = NumericTensor::token_ids(&[6]);
    let expected = streamed
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    let mut restored = snapshot;
    let actual = streamed
        .forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut restored,
            &context,
        )
        .unwrap();
    assert_tensor_exact(&actual, &expected, "all loop states restored");
    assert_state_exact(&restored, &state, 4, "restored loop caches");
}

#[test]
fn nanbeige_actual_payload_tp_pp_and_combined_match_replicated() {
    for (skip, quantization) in [false, true].into_iter().flat_map(|skip| {
        [
            None,
            Some(eredu_core::QuantizationRequest::Affine {
                bits: 4,
                group_size: 16,
            }),
        ]
        .map(|quantization| (skip, quantization))
    }) {
        let mut value = tiny_config(skip);
        value["hidden_size"] = 32.into();
        value["intermediate_size"] = 64.into();
        value["head_dim"] = 8.into();
        value["num_attention_heads"] = 8.into();
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
            ParallelTopology::new(1, 3, 1, 1).unwrap(),
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
                    assert_tensor_close(output, expected, "actual-payload repeated partition");
                }
            }
        }
    }
}

#[test]
fn nanbeige_official_gguf_layout_matches_safetensors_with_permuted_query_key_rows() {
    use eredu_gguf::{GgmlType, MetadataValue as M, TensorInput, Writer};
    for skip in [false, true] {
        let mut value = tiny_config(skip);
        value["hidden_size"] = 32.into();
        value["intermediate_size"] = 64.into();
        value["head_dim"] = 8.into();
        value["vocab_size"] = 32.into();
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
                .replace("lm_head", "output")
                .replace("model.layers.", "blk.")
                .replace("self_attn.q_proj", "attn_q")
                .replace("self_attn.k_proj", "attn_k")
                .replace("self_attn.v_proj", "attn_v")
                .replace("self_attn.o_proj", "attn_output")
                .replace("mlp.gate_proj", "ffn_gate")
                .replace("mlp.up_proj", "ffn_up")
                .replace("mlp.down_proj", "ffn_down")
                .replace("input_layernorm", "attn_norm")
                .replace("post_attention_layernorm", "ffn_norm")
        };
        let tensors = physical
            .into_iter()
            .map(|(name, (shape, bits))| {
                let mut values = bits.clone();
                if name.ends_with("q_proj.weight") || name.ends_with("k_proj.weight") {
                    let width = shape[1] as usize;
                    for row in 0..shape[0] as usize {
                        let source_row = row / 8 * 8 + (row % 8) / 2 + (row % 2) * 4;
                        values[row * width..(row + 1) * width]
                            .copy_from_slice(&bits[source_row * width..(source_row + 1) * width]);
                    }
                }
                (
                    names(&name),
                    shape
                        .into_iter()
                        .rev()
                        .map(|d| d as u64)
                        .collect::<Vec<_>>(),
                    values
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let metadata = BTreeMap::from([
            ("general.architecture".into(), M::String("nanbeige".into())),
            ("nanbeige.embedding_length".into(), M::Uint32(32)),
            ("nanbeige.block_count".into(), M::Uint32(2)),
            ("nanbeige.num_loops".into(), M::Uint32(2)),
            ("nanbeige.skip_loop_final_norm".into(), M::Bool(skip)),
            ("nanbeige.attention.head_count".into(), M::Uint32(4)),
            ("nanbeige.attention.head_count_kv".into(), M::Uint32(2)),
            ("nanbeige.attention.key_length".into(), M::Uint32(8)),
            ("nanbeige.feed_forward_length".into(), M::Uint32(64)),
            (
                "nanbeige.attention.layer_norm_rms_epsilon".into(),
                M::Float32(1e-5),
            ),
            ("nanbeige.rope.freq_base".into(), M::Float32(70_000_000.0)),
            ("nanbeige.vocab_size".into(), M::Uint32(32)),
            ("nanbeige.context_length".into(), M::Uint32(64)),
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
                        ggml_type: GgmlType::F32,
                        data,
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let gguf = eredu_architectures::configuration::inspect_artifact(&path).unwrap();
        assert_eq!(
            gguf.architecture_plan().architecture_descriptor(),
            safe.architecture_plan().architecture_descriptor(),
            "source formats expose the same physical layers, passes, sharing and captures"
        );
        let actual = execute_numeric_replicated_inspection(&gguf, &context, &input, None);
        for (a, e) in actual.outputs.iter().zip(&expected.outputs) {
            assert_tensor_close(a, e, "official GGUF row permutation and repeated cache");
        }

        // Packed aliases must retain GGUF block encoding and physical provenance,
        // rather than becoming U8 recipe outputs. Nonzero blocks make accidental
        // payload coercion observable to native consumers.
        let packed = tensors
            .iter()
            .map(|(name, shape, data)| {
                let (encoding, bytes) = if shape.len() == 2 {
                    let blocks = shape.iter().product::<u64>() as usize / 32;
                    let block = [vec![0x00, 0x3c], (0..32).collect::<Vec<u8>>()].concat();
                    (GgmlType::Q8_0, block.repeat(blocks))
                } else {
                    (GgmlType::F32, data.clone())
                };
                (name, shape, encoding, bytes)
            })
            .collect::<Vec<_>>();
        let packed_path = root.path().join("packed.gguf");
        Writer::default()
            .write(
                std::fs::File::create(&packed_path).unwrap(),
                &metadata,
                &packed
                    .iter()
                    .map(|(name, dimensions, ggml_type, data)| TensorInput {
                        name,
                        dimensions,
                        ggml_type: *ggml_type,
                        data,
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let packed_inspection =
            eredu_architectures::configuration::inspect_artifact(&packed_path).unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&packed_inspection)
                .unwrap();
        let first = requirements
            .parameters()
            .iter()
            .find(|p| p.name() == "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let second = requirements
            .parameters()
            .iter()
            .find(|p| p.name() == "model.layers.2.self_attn.q_proj.weight")
            .unwrap();
        assert!(matches!(
            second.source_encoding(),
            Some(eredu_checkpoint::SourceTensorEncoding::Gguf {
                ggml_type: GgmlType::Q8_0,
                ..
            })
        ));
        assert!(matches!(
            second.native_executable(),
            eredu_checkpoint::LinearFormat::GgufIQuant { .. }
        ));
        assert_eq!(second.logical_shape(), &[32, 32]);
        assert_eq!(first.physical_sources(), second.physical_sources());
        assert_eq!(first.sources(), second.sources());
        assert_ne!(first.owner(), second.owner());
    }
}
