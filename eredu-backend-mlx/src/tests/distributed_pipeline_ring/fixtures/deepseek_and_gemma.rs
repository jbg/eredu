fn write_deepseek_fixture(directory: &Path, layers: i32) {
    write_deepseek_fixture_with_prediction(directory, layers, 0);
}

fn write_deepseek_fixture_with_prediction(directory: &Path, layers: i32, prediction_layers: i32) {
    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "q_lora_rank": null,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "num_nextn_predict_layers": prediction_layers,
        "split_kv_b": false,
        "tie_word_embeddings": false
    });
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = eredu_architectures::deepseek::parse_v3_config(&config).unwrap();
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
                Array::full::<f32>(&shape, Array::from_f32(0.01), self.stream).unwrap()
            };
            self.arrays.push((name, value));
        }
    }
    type Backend = MlxNeuralBackend;
    let architecture =
        eredu_architectures::deepseek::v3::Model::<Backend>::new(args.clone(), stream).unwrap();
    let mut collector = Collector {
        stream,
        arrays: Vec::new(),
    };
    architecture
        .static_modules()
        .visit_parameters(&mut collector);
    for layer in 0..usize::try_from(layers).unwrap() {
        eredu_architectures::deepseek::block::V3Block::<Backend>::new(&args, layer, stream)
            .unwrap()
            .visit_parameters(&mut collector);
    }
    for depth in 0..usize::try_from(prediction_layers).unwrap() {
        eredu_architectures::deepseek::mtp::V3PredictionLayer::<Backend>::new(&args, depth, stream)
            .unwrap()
            .visit_parameters(&mut collector);
    }
    let mut arrays = Vec::new();
    let width = args.moe_intermediate_size;
    for (name, value) in collector.arrays {
        if let Some(prefix) = name.strip_suffix(".experts.gate_up_proj") {
            for expert in 0..args.n_routed_experts {
                let packed = value.try_index_device(expert, stream).unwrap();
                arrays.push((
                    format!("{prefix}.experts.{expert}.gate_proj.weight"),
                    packed.try_index_device((0..width, ..), stream).unwrap(),
                ));
                arrays.push((
                    format!("{prefix}.experts.{expert}.up_proj.weight"),
                    packed
                        .try_index_device((width..2 * width, ..), stream)
                        .unwrap(),
                ));
            }
        } else if let Some(prefix) = name.strip_suffix(".experts.down_proj") {
            for expert in 0..args.n_routed_experts {
                arrays.push((
                    format!("{prefix}.experts.{expert}.down_proj.weight"),
                    value.try_index_device(expert, stream).unwrap(),
                ));
            }
        } else {
            arrays.push((name.to_string(), value));
        }
    }
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
    assert!(arrays
        .iter()
        .all(|(_, value)| value.dtype() == MlxDtype::Float32));
}

pub(crate) fn write_deepseek_v4_fixture(directory: &Path, prediction_layers: u64) {
    write_deepseek_v4_fixture_kind(directory, prediction_layers, false)
}

pub(crate) fn write_deepseek_v4_dspark_fixture(directory: &Path) {
    write_deepseek_v4_fixture_kind(directory, 1, true)
}

fn write_deepseek_v4_fixture_kind(directory: &Path, prediction_layers: u64, dspark: bool) {
    let compress_ratios = if prediction_layers == 0 {
        vec![0, 4]
    } else {
        vec![0, 4, 0]
    };
    let mut config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "qk_rope_head_dim": 4,
        "q_lora_rank": 8,
        "o_lora_rank": 8,
        "o_groups": 2,
        "vocab_size": 16,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "compress_ratios": compress_ratios,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 0.000001,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "num_hash_layers": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "num_nextn_predict_layers": prediction_layers
    });
    if dspark {
        config["dspark_block_size"] = 2.into();
        config["dspark_noise_token_id"] = 0.into();
        config["dspark_target_layer_ids"] = serde_json::json!([0, 1]);
        config["dspark_markov_rank"] = 4.into();
    }
    let args = eredu_architectures::deepseek::parse_v4_config(&config).unwrap();
    let plan = eredu_architectures::deepseek::v4_safetensors_plan(&args).unwrap();
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
            let value = if matches!(
                tensor.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::I32
                )
            ) {
                Array::zeros::<i32>(&shape, stream).unwrap()
            } else if tensor.key.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
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

fn gemma_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "pad_token_id": 0,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "head_dim": 4,
            "attention_bias": false,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 32,
            "num_kv_shared_layers": 1,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "sliding_window": 8,
            "final_logit_softcapping": 4.0
        }
    })
}

fn write_gemma_fixture(directory: &Path) {
    let config = gemma_config();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture =
        eredu_architectures::gemma4::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let parameter = parameter.as_array();
            let value = if metadata.id.as_str().ends_with("norm.weight") {
                Array::ones::<f32>(parameter.shape(), self.stream).unwrap()
            } else {
                Array::full::<f32>(parameter.shape(), Array::from_f32(0.01), self.stream).unwrap()
            };
            self.arrays.push((metadata.id.to_string(), value));
        }
    }
    let architecture = Architecture::new(args, stream).unwrap();
    let mut collector = Collector {
        stream,
        arrays: Vec::new(),
    };
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules(&architecture)
    .visit_parameters(&mut collector);
    for group in 0..3 {
        let count = <Architecture as eredu_runtime::LayeredArchitecture<
            crate::backend::nn::shared::MlxNeuralBackend,
            State,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap()
            .visit_parameters(&mut collector);
        }
    }
    let arrays = collector.arrays;
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
    assert!(arrays
        .iter()
        .all(|(_, value)| value.dtype() == MlxDtype::Float32));
}

fn write_gemma_assistant_fixture(directory: &Path) {
    let config = serde_json::json!({
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 8,
        "use_ordered_embeddings": false,
        "tie_word_embeddings": true,
        "block_size": 3,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "attention_k_eq_v": false,
            "layer_types": ["full_attention"]
        }
    });
    let config_bytes = serde_json::to_vec_pretty(&config).unwrap();
    let assistant = eredu_architectures::gemma4::AssistantConfig::from_json(&config_bytes).unwrap();
    let plan = eredu_architectures::gemma4::assistant_safetensors_plan(&assistant).unwrap();
    assert!(plan.layout_groups.is_empty());

    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let arrays = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let shape = tensor
                .shape
                .iter()
                .copied()
                .map(i32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let value = if tensor.key.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
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
    std::fs::write(directory.join("config.json"), config_bytes).unwrap();
}
