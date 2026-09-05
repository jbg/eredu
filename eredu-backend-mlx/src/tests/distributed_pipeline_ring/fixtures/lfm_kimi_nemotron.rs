fn write_lfm2_pipeline_fixture(directory: &Path, moe: bool) {
    let config = serde_json::json!({
        "model_type": if moe { "lfm2_moe" } else { "lfm2" },
        "architectures": [if moe { "Lfm2MoeForCausalLM" } else { "Lfm2ForCausalLM" }],
        "vocab_size": 16,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 2,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "max_position_embeddings": 64,
        "norm_eps": 0.00001,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_bias": true,
        "block_auto_adjust_ff_dim": false,
        "tie_word_embeddings": false,
        "moe_intermediate_size": if moe { 9 } else { 0 },
        "num_dense_layers": if moe { 1 } else { 0 },
        "num_experts": if moe { 2 } else { 0 },
        "num_experts_per_tok": if moe { 1 } else { 0 },
        "norm_topk_prob": moe,
        "use_expert_bias": moe
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::lfm2::model_args_from_config_value(&config).unwrap();
    let mut model =
        MlxModule::new(checkpoint_fixtures::Lfm2CheckpointTemplate::new(args, stream).unwrap());
    for (name, parameter) in neutral_parameter_refs_mut(&mut model).flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 17;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0003),
                stream,
            )
            .unwrap()
        };
    }
    let arrays = neutral_parameter_refs(&model, false)
        .flatten()
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let weight_map = arrays
        .iter()
        .map(|(name, _)| (name.clone(), "model-00001-of-00001.safetensors".to_owned()))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn write_lfm2_moe_gguf_fixture(path: &Path) {
    let config = serde_json::json!({
        "model_type": "lfm2_moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "vocab_size": 16,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 2,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "max_position_embeddings": 64,
        "norm_eps": 0.00001,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_bias": true,
        "block_auto_adjust_ff_dim": false,
        "tie_word_embeddings": false,
        "moe_intermediate_size": 9,
        "num_dense_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "norm_topk_prob": true,
        "use_expert_bias": true
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::lfm2::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::Lfm2CheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in neutral_parameter_refs(&model, false).flatten() {
        let runtime_name = runtime_name.to_string();
        let layer_name = |name: &str| {
            name.replace("model.layers.", "blk.")
                .replace(".conv.conv.", ".shortconv.conv.")
                .replace(".conv.in_proj.", ".shortconv.in_proj.")
                .replace(".conv.out_proj.", ".shortconv.out_proj.")
                .replace(".self_attn.q_layernorm.", ".attn_q_norm.")
                .replace(".self_attn.k_layernorm.", ".attn_k_norm.")
                .replace(".self_attn.q_proj.", ".attn_q.")
                .replace(".self_attn.k_proj.", ".attn_k.")
                .replace(".self_attn.v_proj.", ".attn_v.")
                .replace(".self_attn.out_proj.", ".attn_output.")
                .replace(".operator_norm.", ".attn_norm.")
                .replace(".feed_forward.gate.", ".ffn_gate_inp.")
                .replace(".feed_forward.experts.down_proj", ".ffn_down_exps.weight")
                .replace(".feed_forward.w1.", ".ffn_gate.")
                .replace(".feed_forward.w2.", ".ffn_down.")
                .replace(".feed_forward.w3.", ".ffn_up.")
        };
        if runtime_name == "model.embed_tokens.weight" {
            specs.push(gguf_tensor_from_array("token_embd.weight", value));
        } else if runtime_name == "model.embedding_norm.weight" {
            specs.push(gguf_tensor_from_array("token_embd_norm.weight", value));
        } else if runtime_name == "lm_head.weight" {
            specs.push(gguf_tensor_from_array("output.weight", value));
        } else if let Some(prefix) = runtime_name.strip_suffix("feed_forward.experts.gate_up_proj")
        {
            let width = value.dim(1) / 2;
            let gate = value.try_index_device((.., ..width, ..), stream).unwrap();
            let up = value.try_index_device((.., width.., ..), stream).unwrap();
            specs.push(gguf_tensor_from_array(
                layer_name(&format!("{prefix}ffn_gate_exps.weight")),
                &gate,
            ));
            specs.push(gguf_tensor_from_array(
                layer_name(&format!("{prefix}ffn_up_exps.weight")),
                &up,
            ));
        } else if let Some(prefix) = runtime_name.strip_suffix("feed_forward.expert_bias") {
            specs.push(gguf_tensor_from_array(
                layer_name(&format!("{prefix}ffn_exp_probs_b.bias")),
                value,
            ));
        } else if runtime_name.ends_with(".conv.conv.weight") {
            let reshaped = value
                .reshape(&[value.shape()[0], value.shape()[2]], stream)
                .unwrap();
            specs.push(gguf_tensor_from_array(layer_name(&runtime_name), &reshaped));
        } else {
            specs.push(gguf_tensor_from_array(layer_name(&runtime_name), value));
        }
    }
    let key = |suffix: &str| format!("lfm2moe.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("lfm2moe".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(2)),
        (key("embedding_length"), GgufMetadataValue::Uint32(12)),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(17)),
        (
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(9),
        ),
        (
            key("leading_dense_block_count"),
            GgufMetadataValue::Uint32(1),
        ),
        (key("expert_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_used_count"), GgufMetadataValue::Uint32(1)),
        (key("expert_weights_norm"), GgufMetadataValue::Uint32(1)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(6)),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 3])),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(0.00001),
        ),
        (key("context_length"), GgufMetadataValue::Uint32(64)),
        (key("shortconv.l_cache"), GgufMetadataValue::Uint32(3)),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10_000.0)),
        (key("vocab_size"), GgufMetadataValue::Uint32(16)),
    ]);
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

fn initialize_fixture(model: &mut impl Parameterized<MlxTensor>, stream: &Stream) {
    for (name, parameter) in neutral_parameter_refs_mut(model).flatten() {
        let shape = parameter.shape().to_vec();
        let value = if name.ends_with("norm.weight")
            || name.ends_with("layernorm.weight")
            || name.ends_with("o_norm.weight")
            || name.ends_with("global_scale")
            || name.as_ref() == "model.norm_f.weight"
        {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else if name.ends_with("A_log") {
            Array::full::<f32>(&shape, Array::from_f32(-0.2), stream).unwrap()
        } else {
            let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 29;
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.002 + ordinal as f32 * 0.0002),
                stream,
            )
            .unwrap()
        };
        *parameter = value.as_dtype(parameter.dtype(), stream).unwrap();
    }
}

fn save_parameter_fixture(
    directory: &Path,
    config: &serde_json::Value,
    model: &impl Parameterized<MlxTensor>,
) {
    let arrays = neutral_parameter_refs(model, false)
        .flatten()
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect::<Vec<_>>();
    save_indexed_pipeline_fixture(directory, &arrays, "model.layers.", 2);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(config).unwrap(),
    )
    .unwrap();
}

fn save_indexed_pipeline_fixture(
    directory: &Path,
    arrays: &[(String, Array)],
    layer_prefix: &str,
    layer_count: usize,
) {
    let mut weight_map = serde_json::Map::new();
    for layer in 0..layer_count {
        let prefix = format!("{layer_prefix}{layer}.");
        let selected = arrays
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .collect::<Vec<_>>();
        assert!(!selected.is_empty(), "fixture layer {layer} has no tensors");
        let shard = format!("layer-{layer}.safetensors");
        Array::save_safetensors(
            selected.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            directory.join(&shard),
        )
        .unwrap();
        for (name, _) in selected {
            weight_map.insert(name.clone(), serde_json::json!(shard));
        }
    }
    let static_tensors = arrays
        .iter()
        .filter(|(name, _)| {
            !(0..layer_count).any(|layer| name.starts_with(&format!("{layer_prefix}{layer}.")))
        })
        .collect::<Vec<_>>();
    assert!(!static_tensors.is_empty());
    Array::save_safetensors(
        static_tensors
            .iter()
            .map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("static.safetensors"),
    )
    .unwrap();
    for (name, _) in static_tensors {
        weight_map.insert(name.clone(), serde_json::json!("static.safetensors"));
    }
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
    let store = open_safetensors_weight_store(directory, 1).unwrap();
    for layer in 0..layer_count {
        let prefix = format!("{layer_prefix}{layer}.");
        for (name, _) in arrays.iter().filter(|(name, _)| name.starts_with(&prefix)) {
            let backing = store.source_metadata(name).unwrap().backing_shard.unwrap();
            assert_eq!(
                backing.file_name().unwrap().to_string_lossy(),
                format!("layer-{layer}.safetensors"),
                "fixture tensor {name} was indexed to the wrong shard"
            );
        }
    }
}

fn kimi_linear_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "kimi_linear",
        "vocab_size": 13,
        "hidden_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "intermediate_size": 17,
        "head_dim": 4,
        "model_max_length": 64,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 3,
            "head_dim": 4,
            "short_conv_kernel_size": 2
        },
        "num_experts": 4,
        "moe_intermediate_size": 9,
        "kv_lora_rank": 4,
        "q_lora_rank": null,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "mla_use_nope": true,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": true,
        "routed_scaling_factor": 1.0,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "use_grouped_topk": true,
        "num_expert_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 0
    })
}

fn write_kimi_linear_fixture(directory: &Path) {
    write_kimi_linear_fixture_from_config(directory, kimi_linear_config());
}

fn write_kimi_linear_dense_fixture(directory: &Path) {
    let mut config = kimi_linear_config();
    config["first_k_dense_replace"] = config["num_hidden_layers"].clone();
    write_kimi_linear_fixture_from_config(directory, config);
}

fn write_kimi_linear_fixture_from_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::kimi_linear::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::KimiLinearCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in neutral_parameter_refs(&model, false).flatten() {
        if name.as_ref() == "model.layers.1.mlp.experts.gate_up_proj" {
            for expert in 0..args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w1.weight"),
                    value
                        .try_index_device((expert, ..args.moe_intermediate_size, ..), stream)
                        .unwrap(),
                ));
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w3.weight"),
                    value
                        .try_index_device((expert, args.moe_intermediate_size.., ..), stream)
                        .unwrap(),
                ));
            }
            continue;
        }
        if name.as_ref() == "model.layers.1.mlp.experts.down_proj" {
            for expert in 0..args.num_experts {
                arrays.push((
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.w2.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
            continue;
        }
        let checkpoint_name =
            if args.has_sparse_moe_layers() && name.starts_with("model.layers.1.mlp.") {
                name.replacen("model.layers.1.mlp.", "model.layers.1.block_sparse_moe.", 1)
            } else {
                name.to_string()
            };
        let value = if checkpoint_name.ends_with("_conv1d.weight") {
            value
                .reshape(
                    &[
                        args.kda_config.num_heads * args.kda_config.head_dim,
                        args.kda_config.short_conv_kernel_size,
                    ],
                    stream,
                )
                .unwrap()
        } else {
            value.clone()
        };
        arrays.push((checkpoint_name, value));
    }
    save_indexed_pipeline_fixture(directory, &arrays, "model.layers.", 2);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn nemotron_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": 13,
        "hidden_size": 12,
        "intermediate_size": 17,
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "M-E*",
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 2,
        "max_position_embeddings": 64,
        "sliding_window": 3,
        "layer_norm_epsilon": 0.00001,
        "norm_eps": 0.00001,
        "mamba_num_heads": 6,
        "mamba_head_dim": 2,
        "n_groups": 3,
        "ssm_state_size": 2,
        "conv_kernel": 3,
        "chunk_size": 2,
        "moe_intermediate_size": 5,
        "moe_shared_expert_intermediate_size": 7,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "torch_dtype": "float32"
    })
}

fn nemotron_quantizable_config() -> serde_json::Value {
    let mut value = nemotron_config();
    value["vocab_size"] = 64.into();
    value["hidden_size"] = 64.into();
    value["intermediate_size"] = 64.into();
    value["num_attention_heads"] = 8.into();
    value["num_key_value_heads"] = 4.into();
    value["head_dim"] = 8.into();
    value["mamba_num_heads"] = 8.into();
    value["mamba_head_dim"] = 8.into();
    value["n_groups"] = 2.into();
    value["moe_intermediate_size"] = 64.into();
    value["moe_shared_expert_intermediate_size"] = 64.into();
    value
}

fn nemotron_public_name(
    runtime: &str,
    args: &eredu_architectures::nemotron_h::ModelArgs,
) -> String {
    if let Some(rest) = runtime.strip_prefix("model.embeddings.") {
        return format!("backbone.embeddings.{rest}");
    }
    if let Some(rest) = runtime.strip_prefix("model.norm_f.") {
        return format!("backbone.norm_f.{rest}");
    }
    for index in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{index}.");
        let Some(rest) = runtime.strip_prefix(&prefix) else {
            continue;
        };
        if rest.starts_with("norm.") {
            return format!("backbone.layers.{index}.{rest}");
        }
        let field = match args.layer_schedule.get(index).unwrap() {
            eredu_architectures::nemotron_h::LayerPolicy::Mamba => "mamba",
            eredu_architectures::nemotron_h::LayerPolicy::SelfAttention(_) => "attention",
            eredu_architectures::nemotron_h::LayerPolicy::DenseMlp => "mlp",
            eredu_architectures::nemotron_h::LayerPolicy::SparseMoe => "moe",
        };
        let rest = rest.strip_prefix(&format!("{field}.")).unwrap_or(rest);
        return format!("backbone.layers.{index}.mixer.{rest}");
    }
    runtime.to_string()
}

fn write_nemotron_fixture(directory: &Path) {
    write_nemotron_fixture_with_config(directory, nemotron_config());
}

fn write_nemotron_dense_fixture(directory: &Path) {
    let mut config = nemotron_config();
    config["hybrid_override_pattern"] = serde_json::json!("M-**");
    config["intermediate_size"] = 18.into();
    config["num_key_value_heads"] = 2.into();
    config["n_groups"] = 2.into();
    write_nemotron_fixture_with_config(directory, config);
}

fn write_nemotron_mtp_fixture(directory: &Path) {
    let mut config = nemotron_config();
    config["hybrid_override_pattern"] = serde_json::json!("M-**");
    config["intermediate_size"] = 18.into();
    config["num_key_value_heads"] = 2.into();
    config["n_groups"] = 2.into();
    config["num_nextn_predict_layers"] = 1.into();
    config["mtp_hybrid_override_pattern"] = serde_json::json!("*");
    write_nemotron_fixture_with_config(directory, config);
}

fn write_nemotron_quantizable_fixture(directory: &Path) {
    write_nemotron_fixture_with_config(directory, nemotron_quantizable_config());
}

fn write_nemotron_dense_quantizable_fixture(directory: &Path) {
    let mut config = nemotron_quantizable_config();
    config["hybrid_override_pattern"] = serde_json::json!("M-**");
    write_nemotron_fixture_with_config(directory, config);
}

fn write_nemotron_fixture_with_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::NemotronHCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in neutral_parameter_refs(&model, false).flatten() {
        let runtime = name.to_string();
        if runtime.ends_with("moe.experts.up_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".up_proj"), &args);
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.up_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else if runtime.ends_with("moe.experts.down_proj") {
            let prefix = nemotron_public_name(runtime.trim_end_matches(".down_proj"), &args);
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.{expert}.down_proj.weight"),
                    value.try_index_device((expert, .., ..), stream).unwrap(),
                ));
            }
        } else {
            arrays.push((nemotron_public_name(&runtime, &args), value.clone()));
        }
    }
    save_indexed_pipeline_fixture(directory, &arrays, "backbone.layers.", 4);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn write_nemotron_h_moe_gguf_fixture(path: &Path) {
    let mut config = nemotron_config();
    config["hybrid_override_pattern"] = serde_json::json!("MEE*");
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::nemotron_h::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        checkpoint_fixtures::NemotronHCheckpointTemplate::new(args.clone(), stream).unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let mut specs = Vec::new();
    for (runtime_name, value) in neutral_parameter_refs(&model, false).flatten() {
        let runtime_name = runtime_name.to_string();
        let gguf_name = if runtime_name == "model.embeddings.weight" {
            "token_embd.weight".to_string()
        } else if runtime_name == "model.norm_f.weight" {
            "output_norm.weight".to_string()
        } else if runtime_name == "lm_head.weight" {
            "output.weight".to_string()
        } else if let Some(rest) = runtime_name.strip_prefix("model.layers.") {
            let (layer, parameter) = rest.split_once('.').unwrap();
            let parameter = parameter
                .strip_prefix("norm.")
                .map_or_else(|| parameter.to_string(), |rest| format!("attn_norm.{rest}"));
            let parameter = parameter
                .replace("mamba.norm.", "ssm_norm.")
                .replace("mamba.in_proj.", "ssm_in.")
                .replace("mamba.conv1d.", "ssm_conv1d.")
                .replace("mamba.dt_bias", "ssm_dt.bias")
                .replace("mamba.A_log", "ssm_a")
                .replace("mamba.D", "ssm_d")
                .replace("mamba.out_proj.", "ssm_out.")
                .replace("attention.q_proj.", "attn_q.")
                .replace("attention.k_proj.", "attn_k.")
                .replace("attention.v_proj.", "attn_v.")
                .replace("attention.o_proj.", "attn_output.")
                .replace("moe.gate.e_score_correction_bias", "exp_probs_b.bias")
                .replace("moe.gate.", "ffn_gate_inp.")
                .replace("moe.experts.up_proj", "ffn_up_exps.weight")
                .replace("moe.experts.down_proj", "ffn_down_exps.weight")
                .replace("moe.shared_experts.up_proj.", "ffn_up_shexp.")
                .replace("moe.shared_experts.down_proj.", "ffn_down_shexp.");
            format!("blk.{layer}.{parameter}")
        } else {
            panic!("unmapped Nemotron-H GGUF fixture tensor {runtime_name}")
        };
        if gguf_name.ends_with(".ssm_conv1d.weight") {
            let reshaped = value
                .reshape(&[value.shape()[0], value.shape()[2]], stream)
                .unwrap();
            specs.push(gguf_tensor_from_array(gguf_name, &reshaped));
        } else if gguf_name.ends_with(".ssm_a") {
            let negative =
                Array::full::<f32>(value.shape(), Array::from_f32(-0.8), stream).unwrap();
            specs.push(gguf_tensor_from_array(gguf_name, &negative));
        } else {
            specs.push(gguf_tensor_from_array(gguf_name, value));
        }
    }
    let key = |suffix: &str| format!("nemotron_h_moe.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("nemotron_h_moe".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(4)),
        (key("embedding_length"), GgufMetadataValue::Uint32(12)),
        (
            key("feed_forward_length"),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 17, 17, 0])),
        ),
        (
            key("attention.head_count_kv"),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![0, 0, 0, 3])),
        ),
        (key("attention.head_count"), GgufMetadataValue::Uint32(6)),
        (key("attention.key_length"), GgufMetadataValue::Uint32(2)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(0.00001),
        ),
        (
            key("attention.sliding_window"),
            GgufMetadataValue::Uint32(3),
        ),
        (key("context_length"), GgufMetadataValue::Uint32(64)),
        (key("ssm.inner_size"), GgufMetadataValue::Uint32(12)),
        (key("ssm.time_step_rank"), GgufMetadataValue::Uint32(6)),
        (key("ssm.state_size"), GgufMetadataValue::Uint32(2)),
        (key("ssm.group_count"), GgufMetadataValue::Uint32(3)),
        (key("ssm.conv_kernel"), GgufMetadataValue::Uint32(3)),
        (key("expert_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_shared_count"), GgufMetadataValue::Uint32(1)),
        (
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(5),
        ),
        (
            key("expert_shared_feed_forward_length"),
            GgufMetadataValue::Uint32(7),
        ),
        (key("expert_used_count"), GgufMetadataValue::Uint32(2)),
        (key("expert_weights_norm"), GgufMetadataValue::Uint32(1)),
        (key("expert_group_count"), GgufMetadataValue::Uint32(1)),
        (key("expert_group_used_count"), GgufMetadataValue::Uint32(1)),
        (key("vocab_size"), GgufMetadataValue::Uint32(13)),
    ]);
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
