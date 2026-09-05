fn qwen_hybrid_config(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": [if model_type == "qwen3_next" { "Qwen3NextForCausalLM" } else { "Qwen3_5ForCausalLM" }],
        "model_type": model_type,
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 0.00001,
        "intermediate_size": 32,
        "num_experts": 0,
        "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": false
    })
}

fn qwen_hybrid_moe_config(model_type: &str) -> serde_json::Value {
    let mut config = qwen_hybrid_config(model_type);
    config["architectures"] = serde_json::json!([if model_type == "qwen3_next" {
        "Qwen3NextForCausalLM"
    } else {
        "Qwen3_5MoeForCausalLM"
    }]);
    config["intermediate_size"] = serde_json::json!(0);
    config["moe_intermediate_size"] = serde_json::json!(8);
    config["shared_expert_intermediate_size"] = serde_json::json!(8);
    config["num_experts"] = serde_json::json!(2);
    config["num_experts_per_tok"] = serde_json::json!(1);
    config["norm_topk_prob"] = serde_json::json!(true);
    config
}

fn write_qwen_hybrid_fixture(directory: &Path, model_type: &str) {
    let config = qwen_hybrid_config(model_type);
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let parsed = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::tests::support::checkpoint_fixtures::QwenHybridCheckpointTemplate::new(
            parsed.text,
            stream,
        )
        .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen_hybrid_moe_fixture(directory: &Path, model_type: &str) {
    let config = qwen_hybrid_moe_config(model_type);
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let parsed = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::tests::support::checkpoint_fixtures::QwenHybridCheckpointTemplate::new(
            parsed.text,
            stream,
        )
        .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen35_multimodal_fixture(directory: &Path, moe: bool) {
    write_qwen35_multimodal_fixture_with_prediction(directory, moe, 1);
}

fn write_qwen35_zero_prediction_fixture(directory: &Path) {
    write_qwen35_multimodal_fixture_with_prediction(directory, false, 0);
}

fn write_qwen35_multimodal_fixture_with_prediction(
    directory: &Path,
    moe: bool,
    prediction_layers: usize,
) {
    let mut text_config = if moe {
        qwen_hybrid_moe_config("qwen3_5_moe_text")
    } else {
        qwen_hybrid_config("qwen3_5_text")
    };
    text_config["mtp_num_hidden_layers"] = prediction_layers.into();
    let config = serde_json::json!({
        "architectures": [if moe { "Qwen3_5MoeForConditionalGeneration" } else { "Qwen3_5ForConditionalGeneration" }],
        "model_type": if moe { "qwen3_5_moe" } else { "qwen3_5" },
        "image_token_id": 42,
        "video_token_id": 43,
        "text_config": text_config,
        "vision_config": {
            "depth": 2,
            "hidden_size": 8,
            "hidden_act": "silu",
            "intermediate_size": 8,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "window_size": 8,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [0, 1],
            "deepstack_visual_indexes": []
        }
    });
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let parsed = qwen_hybrid::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::tests::support::checkpoint_fixtures::QwenConditionalCheckpointTemplate::new(
            parsed, stream,
        )
        .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    save_parameter_fixture(directory, &config, &model);
}

fn write_qwen3_vl_fixture(directory: &Path, moe: bool) {
    let config = serde_json::json!({
        "architectures": [if moe { "Qwen3VLMoeForConditionalGeneration" } else { "Qwen3VLForConditionalGeneration" }],
        "model_type": if moe { "qwen3_vl_moe" } else { "qwen3_vl" },
        "image_token_id": 42,
        "video_token_id": 43,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": if moe { "qwen3_vl_moe_text" } else { "qwen3_vl_text" },
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": if moe { 0 } else { 32 },
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "moe_intermediate_size": if moe { 8 } else { 0 },
            "num_experts": if moe { 2 } else { 0 },
            "num_experts_per_tok": if moe { 1 } else { 0 },
            "norm_topk_prob": moe,
            "rope_scaling": { "mrope_section": [2, 1, 1] }
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "deepstack_visual_indexes": [0, 1]
        }
    });
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::qwen::vl::model_args_from_config_value(&config).unwrap();
    let mut model = MlxModule::new(
        crate::tests::support::checkpoint_fixtures::QwenVlCheckpointTemplate::new(args, stream)
            .unwrap(),
    );
    initialize_fixture(&mut model, stream);
    let arrays = neutral_parameter_refs(&model, false)
        .flatten()
        .into_iter()
        .map(|(name, value)| {
            let canonical = name.to_string();
            let canonical = canonical
                .strip_prefix("model.language_model.model.language_model.")
                .map_or(canonical.clone(), |suffix| {
                    format!("model.language_model.{suffix}")
                });
            (canonical, value.clone())
        })
        .collect::<Vec<_>>();
    save_indexed_pipeline_fixture(directory, &arrays, "model.language_model.layers.", 2);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn inkling_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "inkling_mm_model",
        "eos_token_id": 1,
        "text_config": {
            "torch_dtype": "float32",
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "model_max_length": 32,
            "vocab_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 4,
            "sliding_window_size": 4,
            "layer_types": ["full_attention", "sliding_attention", "full_attention"],
            "dense_mlp_idx": 1,
            "sconv_kernel_size": 3,
            "d_rel": 4,
            "rel_extent": 8,
            "intermediate_size": 8,
            "dense_intermediate_size": 16,
            "moe_intermediate_size": 8,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "route_scale": 1.0,
            "use_sconv": true,
            "use_embed_norm": true,
            "shared_expert_sink": true,
            "use_gate_bias": true,
            "norm_after_topk": true,
            "use_global_scale": true,
            "gate_activation": "sigmoid",
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "q_bias": false,
            "o_bias": false,
            "logits_mup_width_multiplier": 2.0,
            "unpadded_vocab_size": 30
        }
    })
}

fn inkling_quantizable_config() -> serde_json::Value {
    let mut value = inkling_config();
    value["text_config"]["hidden_size"] = 32.into();
    value["text_config"]["vocab_size"] = 64.into();
    value["text_config"]["num_attention_heads"] = 4.into();
    value["text_config"]["num_key_value_heads"] = 2.into();
    value["text_config"]["head_dim"] = 8.into();
    value["text_config"]["swa_num_attention_heads"] = 4.into();
    value["text_config"]["swa_num_key_value_heads"] = 2.into();
    value["text_config"]["swa_head_dim"] = 8.into();
    value["text_config"]["d_rel"] = 32.into();
    value["text_config"]["rel_extent"] = 32.into();
    value["text_config"]["intermediate_size"] = 32.into();
    value["text_config"]["dense_intermediate_size"] = 32.into();
    value["text_config"]["moe_intermediate_size"] = 32.into();
    value
}

fn inkling_multimodal_config() -> serde_json::Value {
    let mut config = inkling_config();
    config["audio_config"] = serde_json::json!({
        "text_hidden_size": 16,
        "num_codebooks": 2,
        "codebook_size": 8,
        "bias": false,
        "use_audio_norm": true,
        "audio_mode": "dmel",
        "rms_norm_eps": 1e-6
    });
    config["image_token_id"] = serde_json::json!(21);
    config["audio_token_id"] = serde_json::json!(20);
    config
}

fn inkling_released_name(runtime: &str) -> String {
    if runtime == "lm_head.weight" {
        return "model.llm.unembed.weight".into();
    }
    if let Some(rest) = runtime.strip_prefix("audio.") {
        return format!("model.audio.{rest}");
    }
    if let Some(rest) = runtime.strip_prefix("visual.") {
        return format!("model.visual.{rest}");
    }
    let rest = runtime.strip_prefix("model.").unwrap();
    let mut raw = format!("model.llm.{rest}");
    raw = raw
        .replace("model.llm.embed_tokens.weight", "model.llm.embed.weight")
        .replace(".input_layernorm.weight", ".attn_norm.weight")
        .replace(".post_attention_layernorm.weight", ".mlp_norm.weight")
        .replace(".self_attn.q_proj.weight", ".attn.wq_du.weight")
        .replace(".self_attn.k_proj.weight", ".attn.wk_dv.weight")
        .replace(".self_attn.v_proj.weight", ".attn.wv_dv.weight")
        .replace(".self_attn.r_proj.weight", ".attn.wr_du.weight")
        .replace(".self_attn.o_proj.weight", ".attn.wo_ud.weight")
        .replace(".self_attn.q_norm.weight", ".attn.q_norm.weight")
        .replace(".self_attn.k_norm.weight", ".attn.k_norm.weight")
        .replace(".self_attn.rel_proj", ".attn.rel_logits_proj.proj")
        .replace(".self_attn.k_sconv.weight", ".attn.k_sconv.weight")
        .replace(".self_attn.v_sconv.weight", ".attn.v_sconv.weight")
        .replace(".dense.down_proj.weight", ".mlp.w2_md.weight")
        .replace(".dense_global_scale", ".mlp.global_scale")
        .replace(".moe.router.weight", ".mlp.gate.weight")
        .replace(".moe.router.bias", ".mlp.gate.bias")
        .replace(".moe.router.global_scale", ".mlp.gate.global_scale")
        .replace(".moe.experts.down_proj", ".mlp.experts.w2_weight")
        .replace(
            ".moe.shared_experts.down_proj",
            ".mlp.shared_experts.shared_w2_weight",
        );
    raw
}

fn interleave(gate: &Array, up: &Array, axis: i32, stream: &Stream) -> Array {
    let stacked = stack_axis(&[gate.clone(), up.clone()], axis, stream).unwrap();
    let mut shape = gate.shape().to_vec();
    let row_axis = shape.len() - 2;
    shape[row_axis] *= 2;
    stacked.reshape(&shape, stream).unwrap()
}

fn write_inkling_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_config());
}

fn write_inkling_dense_fixture(directory: &Path) {
    let mut config = inkling_config();
    config["text_config"]["dense_mlp_idx"] = config["text_config"]["num_hidden_layers"].clone();
    write_inkling_fixture_with_config(directory, config);
}

fn write_inkling_dense_multimodal_fixture(directory: &Path) {
    let mut config = inkling_multimodal_config();
    config["text_config"]["dense_mlp_idx"] = config["text_config"]["num_hidden_layers"].clone();
    write_inkling_fixture_with_config(directory, config);
}

fn write_inkling_mtp_fixture(directory: &Path) {
    write_inkling_mtp_fixture_for_pipeline(directory, false);
}

fn write_inkling_pipeline_mtp_fixture(directory: &Path) {
    write_inkling_mtp_fixture_for_pipeline(directory, true);
}

fn write_inkling_mtp_fixture_for_pipeline(directory: &Path, pipeline: bool) {
    let mut config = inkling_config();
    if pipeline {
        config["text_config"]["num_hidden_layers"] = 2.into();
        config["text_config"]["layer_types"] =
            serde_json::json!(["sliding_attention", "full_attention"]);
        config["text_config"]["dense_mlp_idx"] = 1.into();
        config["text_config"]["model_max_length"] = 32.into();
    } else {
        config["text_config"]["num_hidden_layers"] = 1.into();
        config["text_config"]["layer_types"] = serde_json::json!(["sliding_attention"]);
        config["text_config"]["dense_mlp_idx"] = 0.into();
    }
    config["mtp_config"] = serde_json::json!({
        "num_nextn_predict_layers": 2,
        "local_layer_ids": [1],
        "chain_hidden_post_norm": true,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 4,
        "dense_intermediate_size": 16,
        "sconv_kernel_size": 3
    });
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::inkling::ModelArgs::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture =
        eredu_architectures::inkling::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;
    let architecture = Architecture::new(args, stream).unwrap();
    let mut arrays = Vec::<(String, Array)>::new();
    struct Collector<'a> {
        stream: &'a Stream,
        arrays: &'a mut Vec<(String, Array)>,
    }
    impl<'tensor> ParameterVisitor<'tensor, MlxTensor> for Collector<'_> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'tensor MlxTensor) {
            let parameter = parameter.as_array();
            self.arrays.push((
                metadata.id.to_string(),
                safemlx::ops::zeros_dtype(parameter.shape(), parameter.dtype(), self.stream)
                    .unwrap(),
            ));
        }
    }
    let mut collector = Collector {
        stream,
        arrays: &mut arrays,
    };
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules(&architecture)
    .visit_parameters(&mut collector);
    let graph = <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::execution_graph(&architecture)
    .unwrap();
    for group in 0..graph.groups().len() {
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
    Array::save_safetensors(
        arrays.iter().map(|(name, array)| (name.as_str(), array)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
}
