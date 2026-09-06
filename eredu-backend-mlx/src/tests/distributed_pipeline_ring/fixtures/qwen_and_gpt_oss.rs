fn qwen_config(model_type: &str) -> serde_json::Value {
    let is_moe = model_type == "qwen3_moe";
    let mut config = serde_json::json!({
        "architectures": [match model_type {
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "qwen3_moe" => "Qwen3MoeForCausalLM",
            _ => panic!("unsupported Qwen pipeline fixture model type {model_type}"),
        }],
        "model_type": model_type,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": if is_moe { 0 } else { 64 },
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 0.000001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "attention_bias": model_type == "qwen2",
        "mlp_bias": false,
        "moe_intermediate_size": if is_moe { 32 } else { 0 },
        "num_experts": if is_moe { 4 } else { 0 },
        "num_experts_per_tok": if is_moe { 2 } else { 0 },
        "norm_topk_prob": is_moe
    });
    if model_type == "qwen2" {
        config["use_sliding_window"] = serde_json::json!(true);
        config["sliding_window"] = serde_json::json!(3);
        config["max_window_layers"] = serde_json::json!(1);
    }
    config
}

fn write_qwen_fixture(directory: &Path, model_type: &str) {
    write_qwen_fixture_with_tied_head(directory, model_type, false);
}

fn write_indexed_qwen_fixture(directory: &Path, model_type: &str) {
    write_qwen_fixture(directory, model_type);
    let source = directory.join("model.safetensors");
    let shard = directory.join("model-00001-of-00001.safetensors");
    std::fs::rename(source, &shard).unwrap();
    let bytes = std::fs::read(&shard).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let weight_map = tensors
        .names()
        .into_iter()
        .map(|name| (name.to_owned(), "model-00001-of-00001.safetensors"))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
}

fn write_qwen_fixture_with_tied_head(directory: &Path, model_type: &str, tied: bool) {
    let mut config = qwen_config(model_type);
    config["tie_word_embeddings"] = serde_json::json!(tied);
    write_qwen_config_fixture(directory, config);
}

fn write_qwen_requantized_tp_fixture(directory: &Path) {
    let mut config = qwen_config("qwen3");
    config["hidden_size"] = serde_json::json!(64);
    config["num_attention_heads"] = serde_json::json!(8);
    config["num_key_value_heads"] = serde_json::json!(4);
    config["intermediate_size"] = serde_json::json!(128);
    config["vocab_size"] = serde_json::json!(64);
    write_qwen_config_fixture(directory, config);
}

fn qwen_fixture_arrays(
    args: &eredu_architectures::qwen::ModelArgs,
    stream: &Stream,
) -> Vec<(String, Array)> {
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let name = metadata.id.to_string();
            let shape = parameter.as_array().shape().to_vec();
            let value = if name.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, self.stream).unwrap()
            } else {
                let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 17;
                Array::full::<f32>(
                    &shape,
                    Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                    self.stream,
                )
                .unwrap()
            };
            self.arrays.push((name, value));
        }
    }

    let architecture = eredu_architectures::qwen::RoutedLayeredModel::<MlxNeuralBackend>::new(
        args.clone(),
        stream,
    )
    .unwrap();
    let mut collector = Collector {
        stream,
        arrays: Vec::new(),
    };
    architecture
        .static_modules()
        .visit_parameters(&mut collector);
    for layer in 0..args.num_hidden_layers as usize {
        eredu_architectures::qwen::new_routed_block::<MlxNeuralBackend>(args, layer, stream)
            .unwrap()
            .visit_parameters(&mut collector);
    }
    collector.arrays
}

fn write_qwen_config_fixture(directory: &Path, config: serde_json::Value) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::qwen::model_args_from_config_value(&config).unwrap();
    let arrays = qwen_fixture_arrays(&args, stream);
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
}

fn write_qwen3_moe_gguf_fixture(path: &Path) {
    write_qwen_gguf_fixture(path, "qwen3_moe");
}

fn write_qwen_gguf_fixture(path: &Path, model_type: &str) {
    let config = qwen_config(model_type);
    std::fs::write(
        path.parent().unwrap().join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::qwen::model_args_from_config_value(&config).unwrap();
    let arrays = qwen_fixture_arrays(&args, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in &arrays {
        let runtime_name = runtime_name.to_string();
        if let Some(prefix) = runtime_name.strip_suffix(".mlp.experts.gate_up_proj") {
            let gate = value
                .try_index_device((.., ..args.moe_intermediate_size, ..), stream)
                .unwrap();
            let up = value
                .try_index_device((.., args.moe_intermediate_size.., ..), stream)
                .unwrap();
            let prefix = prefix.replace("model.layers.", "blk.");
            specs.push(gguf_tensor_from_array(
                format!("{prefix}.ffn_gate_exps.weight"),
                &gate,
            ));
            specs.push(gguf_tensor_from_array(
                format!("{prefix}.ffn_up_exps.weight"),
                &up,
            ));
            continue;
        }
        if let Some(prefix) = runtime_name.strip_suffix(".mlp.experts.down_proj") {
            specs.push(gguf_tensor_from_array(
                format!(
                    "{}.ffn_down_exps.weight",
                    prefix.replace("model.layers.", "blk.")
                ),
                value,
            ));
            continue;
        }
        let name = runtime_name
            .replace("model.layers.", "blk.")
            .replace("self_attn.q_norm", "attn_q_norm")
            .replace("self_attn.k_norm", "attn_k_norm")
            .replace("self_attn.q_proj", "attn_q")
            .replace("self_attn.k_proj", "attn_k")
            .replace("self_attn.v_proj", "attn_v")
            .replace("self_attn.o_proj", "attn_output")
            .replace("input_layernorm", "attn_norm")
            .replace("post_attention_layernorm", "ffn_norm")
            .replace("mlp.gate.weight", "ffn_gate_inp.weight")
            .replace("mlp.gate_proj", "ffn_gate")
            .replace("mlp.up_proj", "ffn_up")
            .replace("mlp.down_proj", "ffn_down")
            .replace("model.embed_tokens", "token_embd")
            .replace("model.norm", "output_norm")
            .replace("lm_head", "output");
        specs.push(gguf_tensor_from_array(name, value));
    }
    let architecture = match model_type {
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        "qwen3_moe" => "qwen3moe",
        _ => panic!("unsupported Qwen GGUF fixture model type {model_type}"),
    };
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            key("embedding_length"),
            GgufMetadataValue::Uint32(args.hidden_size as u32),
        ),
        (
            key("block_count"),
            GgufMetadataValue::Uint32(args.num_hidden_layers as u32),
        ),
        (
            key("attention.head_count"),
            GgufMetadataValue::Uint32(args.num_attention_heads as u32),
        ),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Uint32(args.num_key_value_heads as u32),
        ),
        (
            key("attention.key_length"),
            GgufMetadataValue::Uint32(args.head_dim as u32),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(args.rms_norm_eps),
        ),
        (
            key("context_length"),
            GgufMetadataValue::Uint32(args.max_position_embeddings as u32),
        ),
        (
            key("rope.freq_base"),
            GgufMetadataValue::Float32(args.rope_theta),
        ),
        (
            key("vocab_size"),
            GgufMetadataValue::Uint32(args.vocab_size as u32),
        ),
    ]);
    if args.is_moe() {
        metadata.insert(
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(args.moe_intermediate_size as u32),
        );
        metadata.insert(
            key("expert_count"),
            GgufMetadataValue::Uint32(args.num_experts as u32),
        );
        metadata.insert(
            key("expert_used_count"),
            GgufMetadataValue::Uint32(args.num_experts_per_tok as u32),
        );
    } else {
        metadata.insert(
            key("feed_forward_length"),
            GgufMetadataValue::Uint32(args.intermediate_size as u32),
        );
    }
    let tensors = specs
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn write_gpt_oss_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001,
        "sliding_window": 3,
        "max_position_embeddings": 128,
        "rope_theta": 150000.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "quantization_config": {"quant_method": "mxfp4"},
        "swiglu_limit": 7.0
    });
    let args = gpt_oss::model_args_from_config_value(&config).unwrap();
    let plan = gpt_oss::safetensors_plan(&args).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let arrays = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let shape = tensor
                .shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            let value = match &tensor.dtype {
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U8,
                ) if tensor.key.ends_with("_scales") => {
                    Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
                }
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U8,
                ) => Array::full::<u8>(&shape, Array::from_slice(&[0x11u8], &[]), stream).unwrap(),
                _ if tensor.key.ends_with("norm.weight") => {
                    Array::ones::<f32>(&shape, stream).unwrap()
                }
                _ => {
                    let ordinal = tensor
                        .key
                        .bytes()
                        .fold(0u32, |sum, byte| sum + u32::from(byte))
                        % 17;
                    Array::full::<f32>(
                        &shape,
                        Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                        stream,
                    )
                    .unwrap()
                }
            };
            (tensor.key.clone(), value)
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn replicated_inspection_dispatches_gpt_oss_and_nemotron_h_observers() {
    fn inspect(
        write_fixture: impl FnOnce(&Path) -> PathBuf,
        options: MlxLoadRequest,
        expected_observation: &str,
    ) {
        let checkpoint = tempfile::tempdir().unwrap();
        let artifact = write_fixture(checkpoint.path());
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let backend = crate::native::backend(stream, stream);
        let model = load_model(&backend, artifact, options).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let parts = [text_input_part(&tokens)];
        let input = crate::composition::mlx::MlxModelInput::from(
            crate::backend::runtime::media::input::ModelInput::new(&parts),
        );
        let inspected = runtime
            .inspect_prefill(input, &ObservationRequest::all())
            .unwrap();
        assert!(
            inspected.observations.get(expected_observation).is_some(),
            "missing {expected_observation:?} in {:?}",
            inspected.observations
        );
        let decode = Array::from_slice(&[3_u32], &[1, 1]);
        let inspected = runtime
            .inspect_decode(decode, &ObservationRequest::all())
            .unwrap();
        assert!(
            inspected.observations.get(expected_observation).is_some(),
            "decode missing {expected_observation:?} in {:?}",
            inspected.observations
        );
    }

    inspect(
        |directory| {
            write_gpt_oss_fixture(directory);
            directory.to_path_buf()
        },
        MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                WeightResidency::with_independent_parameter_banks(
                    OrdinaryWeightResidency::FullyResident,
                    ParameterBankLoadOptions::default(),
                ),
            ),
        ),
        "model.layers.0.output",
    );
    inspect(
        |directory| {
            write_nemotron_fixture(directory);
            directory.to_path_buf()
        },
        MlxLoadRequest::default(),
        "model.layers.0.output",
    );
}

struct QuantizedGgufFixtureTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

fn mxfp4_payload(elements: u64, phase: usize) -> Vec<u8> {
    assert_eq!(elements % 32, 0);
    let mut data = Vec::with_capacity((elements / 32) as usize * 17);
    for block in 0..elements / 32 {
        data.push(127 + ((block as usize + phase) % 3) as u8);
        data.extend((0..16).map(|index| {
            let low = ((index + phase) % 7 + 1) as u8;
            let high = ((index * 3 + phase) % 7 + 1) as u8;
            low | (high << 4)
        }));
    }
    data
}

pub(crate) fn write_gpt_oss_gguf_fixture(path: &Path) {
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("gpt-oss".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(39)),
        (
            "gpt-oss.embedding_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        ("gpt-oss.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "gpt-oss.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(96),
        ),
        (
            "gpt-oss.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "gpt-oss.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "gpt-oss.attention.key_length".into(),
            GgufMetadataValue::Uint32(32),
        ),
        (
            "gpt-oss.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            "gpt-oss.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "gpt-oss.context_length".into(),
            GgufMetadataValue::Uint32(128),
        ),
        (
            "gpt-oss.rope.freq_base".into(),
            GgufMetadataValue::Float32(150000.0),
        ),
        ("gpt-oss.expert_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "gpt-oss.expert_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        ("gpt-oss.vocab_size".into(), GgufMetadataValue::Uint32(64)),
    ]);
    let f32_tensor = |name: String, dimensions: Vec<u64>, phase: usize| {
        let values = patterned_values(
            usize::try_from(dimensions.iter().product::<u64>()).unwrap(),
            0.003,
            phase,
        );
        QuantizedGgufFixtureTensor {
            name,
            dimensions,
            ggml_type: GgmlType::F32,
            data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        }
    };
    let mxfp4_tensor = |name: String, dimensions: Vec<u64>, phase: usize| {
        let elements = dimensions.iter().product();
        QuantizedGgufFixtureTensor {
            name,
            dimensions,
            ggml_type: GgmlType::MxFp4,
            data: mxfp4_payload(elements, phase),
        }
    };
    let mut tensors = vec![f32_tensor("token_embd.weight".into(), vec![64, 64], 0)];
    for layer in 0..2 {
        let prefix = format!("blk.{layer}");
        let phase = layer * 20;
        tensors.extend([
            f32_tensor(format!("{prefix}.attn_norm.weight"), vec![64], phase + 1),
            f32_tensor(
                format!("{prefix}.attn_post_norm.weight"),
                vec![64],
                phase + 2,
            ),
            f32_tensor(format!("{prefix}.attn_q.weight"), vec![64, 128], phase + 3),
            f32_tensor(format!("{prefix}.attn_q.bias"), vec![128], phase + 4),
            f32_tensor(format!("{prefix}.attn_k.weight"), vec![64, 64], phase + 5),
            f32_tensor(format!("{prefix}.attn_k.bias"), vec![64], phase + 6),
            f32_tensor(format!("{prefix}.attn_v.weight"), vec![64, 64], phase + 7),
            f32_tensor(format!("{prefix}.attn_v.bias"), vec![64], phase + 8),
            f32_tensor(
                format!("{prefix}.attn_output.weight"),
                vec![128, 64],
                phase + 9,
            ),
            f32_tensor(format!("{prefix}.attn_output.bias"), vec![64], phase + 10),
            f32_tensor(format!("{prefix}.attn_sinks.weight"), vec![4], phase + 11),
            f32_tensor(
                format!("{prefix}.ffn_gate_inp.weight"),
                vec![64, 2],
                phase + 12,
            ),
            f32_tensor(format!("{prefix}.ffn_gate_inp.bias"), vec![2], phase + 13),
            mxfp4_tensor(
                format!("{prefix}.ffn_gate_exps.weight"),
                vec![64, 96, 2],
                phase + 14,
            ),
            f32_tensor(
                format!("{prefix}.ffn_gate_exps.bias"),
                vec![96, 2],
                phase + 15,
            ),
            mxfp4_tensor(
                format!("{prefix}.ffn_up_exps.weight"),
                vec![64, 96, 2],
                phase + 16,
            ),
            f32_tensor(
                format!("{prefix}.ffn_up_exps.bias"),
                vec![96, 2],
                phase + 17,
            ),
            mxfp4_tensor(
                format!("{prefix}.ffn_down_exps.weight"),
                vec![96, 64, 2],
                phase + 18,
            ),
            f32_tensor(
                format!("{prefix}.ffn_down_exps.bias"),
                vec![64, 2],
                phase + 19,
            ),
        ]);
    }
    tensors.extend([
        f32_tensor("output_norm.weight".into(), vec![64], 41),
        f32_tensor("output.weight".into(), vec![64, 64], 42),
    ]);
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: tensor.ggml_type,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}
