fn write_gemma4_tensor_parallel_fixture(directory: &Path) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, false, false);
}

fn write_gemma4_tensor_parallel_fixture_with_tied_embeddings(directory: &Path, tied: bool) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, tied, false);
}

fn write_gemma4_multimodal_tensor_parallel_fixture(directory: &Path) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, false, true);
}

fn write_gemma4_tied_multimodal_tensor_parallel_fixture(directory: &Path) {
    write_gemma4_tensor_parallel_fixture_with_options(directory, true, true);
}

fn write_gemma4_tensor_parallel_fixture_with_options(
    directory: &Path,
    tied: bool,
    multimodal: bool,
) {
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": tied,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": tied,
            "attention_k_eq_v": false,
            "layer_types": ["full_attention", "sliding_attention"],
            "sliding_window": 8
        }
    });
    if multimodal {
        config["model_type"] = "gemma4_unified".into();
        config["image_token_id"] = 30.into();
        config["audio_token_id"] = 31.into();
        config["vision_config"] = serde_json::json!({
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "patch_size": 4,
            "pooling_kernel_size": 2,
            "position_embedding_size": 4,
            "rms_norm_eps": 0.00001
        });
        config["audio_config"] = serde_json::json!({
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "output_proj_dims": 8,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 5,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1000000000.0,
            "attention_logit_cap": 50.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 0.00001,
            "subsampling_conv_channels": [4, 8]
        });
    }
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::gemma4::FamilyConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture =
        eredu_architectures::gemma4::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
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
    // MLX's SafeTensors writer promotes rank-zero arrays to `[1]`, while the
    // released clipped media bounds and their neutral schema are true
    // scalars. Preserve the authoritative parameter shapes in this fixture.
    let tensors = arrays
        .iter()
        .map(|(name, array)| {
            let shape = if ["input_min", "input_max", "output_min", "output_max"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
            {
                Vec::new()
            } else {
                array
                    .shape()
                    .iter()
                    .map(|dimension| usize::try_from(*dimension).unwrap())
                    .collect()
            };
            (name.as_str(), shape, 0.0)
        })
        .collect::<Vec<_>>();
    write_f32_shard(&directory.join("model.safetensors"), &tensors);
}

fn write_muse_glimmer_tensor_parallel_fixture(directory: &Path) {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 22,
        "video_token_id": 23,
        "out_hidden_size": 32,
        "projector_hidden_size": 16,
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "post_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [10000.0, 0.0],
            "sliding_window": 8,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "qk_scale_factor": 1.0,
            "output_multiplier": 1.0,
            "final_logit_softcapping": 30.0
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "patch_size": 2,
            "patch_temporal": 1,
            "merge_size": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "max_position_embeddings": 4,
            "layer_norm_eps": 0.00001,
            "hidden_act": "gelu",
            "layer_types": ["full_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        }
    });
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = eredu_architectures::muse_glimmer::DecoderConfig::from_hf_json(
        &serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    type Architecture = eredu_architectures::muse_glimmer::LayeredModel<
        crate::backend::nn::shared::MlxNeuralBackend,
    >;
    type State = crate::backend::runtime::cache::state::MlxKeyValueState;
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
    for group in 0..2 {
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

fn write_inkling_quantizable_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_quantizable_config());
}

fn write_inkling_multimodal_fixture(directory: &Path) {
    write_inkling_fixture_with_config(directory, inkling_multimodal_config());
}

fn initialized_inkling_parameters(
    config: &serde_json::Value,
    stream: &Stream,
) -> (
    eredu_architectures::inkling::ModelArgs,
    BTreeMap<String, Array>,
) {
    type Architecture =
        eredu_architectures::inkling::LayeredModel<crate::backend::nn::shared::MlxNeuralBackend>;
    type State = crate::backend::runtime::cache::state::MlxHybridState;

    struct Initializer<'a> {
        stream: &'a Stream,
        parameters: &'a mut BTreeMap<String, Array>,
    }

    impl<'tensor> ParameterVisitorMut<'tensor, MlxTensor> for Initializer<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, parameter: &'tensor mut MlxTensor) {
            let name = metadata.id.to_string();
            let shape = parameter.as_array().shape().to_vec();
            let dtype = parameter.as_array().dtype();
            let value = if name.ends_with("norm.weight")
                || name.ends_with("layernorm.weight")
                || name.ends_with("o_norm.weight")
                || name.ends_with("global_scale")
                || name == "model.norm_f.weight"
            {
                Array::ones::<f32>(&shape, self.stream).unwrap()
            } else if name.ends_with("A_log") {
                Array::full::<f32>(&shape, Array::from_f32(-0.2), self.stream).unwrap()
            } else {
                let ordinal = name.bytes().fold(0u32, |sum, byte| sum + u32::from(byte)) % 29;
                Array::full::<f32>(
                    &shape,
                    Array::from_f32(0.002 + ordinal as f32 * 0.0002),
                    self.stream,
                )
                .unwrap()
            }
            .as_dtype(dtype, self.stream)
            .unwrap();
            *parameter = MlxTensor::from_array(value);
            self.parameters.insert(name, parameter.as_array().clone());
        }
    }

    let args =
        eredu_architectures::inkling::ModelArgs::from_hf_json(&serde_json::to_vec(config).unwrap())
            .unwrap();
    let mut architecture = Architecture::new(args.clone(), stream).unwrap();
    let mut parameters = BTreeMap::new();
    <Architecture as eredu_runtime::LayeredArchitecture<
        crate::backend::nn::shared::MlxNeuralBackend,
        State,
    >>::static_modules_mut(&mut architecture)
    .visit_parameters_mut(&mut Initializer {
        stream,
        parameters: &mut parameters,
    });
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
            let mut unit = <Architecture as eredu_runtime::LayeredArchitecture<
                crate::backend::nn::shared::MlxNeuralBackend,
                State,
            >>::build_unit(&architecture, group, index, stream)
            .unwrap();
            unit.visit_parameters_mut(&mut Initializer {
                stream,
                parameters: &mut parameters,
            });
        }
    }
    (args, parameters)
}

fn write_inkling_fixture_with_config(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let (args, parameters) = initialized_inkling_parameters(&config, stream);
    let mut arrays = Vec::<(String, Array)>::new();
    for (name, value) in &parameters {
        let name = name.as_str();
        if name.ends_with(".dense.up_proj.weight") {
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".dense.gate_proj.weight") {
            let up = parameters
                .get(format!("{prefix}.dense.up_proj.weight").as_str())
                .unwrap();
            arrays.push((
                format!("model.llm.{}.mlp.w13_dn.weight", &prefix["model.".len()..]),
                interleave(value, up, 1, stream),
            ));
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".moe.experts.gate_up_proj") {
            let intermediate = args.text_config.moe_intermediate_size.unwrap();
            let gate = value
                .try_index_device((.., ..intermediate, ..), stream)
                .unwrap();
            let up = value
                .try_index_device((.., intermediate.., ..), stream)
                .unwrap();
            arrays.push((
                format!(
                    "model.llm.{}.mlp.experts.w13_weight",
                    &prefix["model.".len()..]
                ),
                interleave(&gate, &up, 2, stream),
            ));
            continue;
        }
        if let Some(prefix) = name.strip_suffix(".moe.shared_experts.gate_up_proj") {
            let intermediate = args.text_config.moe_intermediate_size.unwrap();
            let gate = value
                .try_index_device((.., ..intermediate, ..), stream)
                .unwrap();
            let up = value
                .try_index_device((.., intermediate.., ..), stream)
                .unwrap();
            arrays.push((
                format!(
                    "model.llm.{}.mlp.shared_experts.shared_w13_weight",
                    &prefix["model.".len()..]
                ),
                interleave(&gate, &up, 2, stream),
            ));
            continue;
        }
        let raw = inkling_released_name(name);
        arrays.push((raw, (*value).clone()));
    }
    save_indexed_pipeline_fixture(directory, &arrays, "model.llm.layers.", 3);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn inkling_gguf_metadata() -> BTreeMap<String, GgufMetadataValue> {
    BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("inkling".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        ("inkling.block_count".into(), GgufMetadataValue::Uint32(3)),
        (
            "inkling.embedding_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "inkling.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "inkling.attention.head_count_kv".into(),
            GgufMetadataValue::Array(MetadataArray::Uint32(vec![2, 2, 2])),
        ),
        (
            "inkling.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(MetadataArray::Bool(vec![false, true, false])),
        ),
        (
            "inkling.attention.key_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        ("inkling.vocab_size".into(), GgufMetadataValue::Uint32(32)),
        (
            "inkling.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "inkling.dense_block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "inkling.shortconv_kernel".into(),
            GgufMetadataValue::Uint32(3),
        ),
        ("inkling.rel_extent".into(), GgufMetadataValue::Uint32(8)),
        ("inkling.d_rel".into(), GgufMetadataValue::Uint32(4)),
        (
            "inkling.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        ),
        (
            "inkling.unpadded_vocab_size".into(),
            GgufMetadataValue::Uint32(30),
        ),
        (
            "inkling.logit_scale_denom".into(),
            GgufMetadataValue::Float32(2.0),
        ),
        (
            "inkling.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "inkling.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        ("inkling.expert_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "inkling.expert_used_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "inkling.expert_shared_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "inkling.expert_weights_scale".into(),
            GgufMetadataValue::Float32(1.0),
        ),
        (
            "inkling.context_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
    ])
}

fn inkling_gguf_layer_name(runtime: &str) -> Option<String> {
    for (source, target) in [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.embed_norm.weight", "token_embd_norm.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ] {
        if runtime == source {
            return Some(target.into());
        }
    }
    let rest = runtime.strip_prefix("model.layers.")?;
    let (layer, parameter) = rest.split_once('.')?;
    let target = match parameter {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.r_proj.weight" => "attn_r.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "self_attn.rel_proj" => "attn_rel_proj",
        "self_attn.k_sconv.weight" => "shortconv_k.weight",
        "self_attn.v_sconv.weight" => "shortconv_v.weight",
        "attn_sconv.weight" => "shortconv_attn.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "dense.gate_proj.weight" => "ffn_gate.weight",
        "dense.up_proj.weight" => "ffn_up.weight",
        "dense.down_proj.weight" => "ffn_down.weight",
        "dense_global_scale" | "moe.router.global_scale" => "ffn_gscale",
        "moe.router.weight" => "ffn_gate_inp.weight",
        "moe.router.bias" => "exp_probs_b.bias",
        "moe.experts.down_proj" => "ffn_down_exps.weight",
        "moe.shared_experts.down_proj" => "ffn_down_shexp.weight",
        "mlp_sconv.weight" => "shortconv_mlp.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{target}"))
}

fn gguf_tensor_from_array(name: impl Into<String>, array: &Array) -> GgufFixtureTensor {
    let evaluated = array.evaluated().unwrap();
    let mut dimensions = array
        .shape()
        .iter()
        .rev()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if dimensions.is_empty() {
        dimensions.push(1);
    }
    f32_gguf_tensor(name, dimensions, evaluated.as_slice::<f32>().to_vec())
}

fn write_inkling_gguf_fixture(path: &Path) {
    let config = inkling_config();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let (_args, parameters) = initialized_inkling_parameters(&config, stream);
    let mut specs = Vec::new();
    for (runtime, value) in &parameters {
        let runtime = runtime.as_str();
        for (suffix, gate_name, up_name) in [
            (
                ".moe.experts.gate_up_proj",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
            ),
            (
                ".moe.shared_experts.gate_up_proj",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
            ),
        ] {
            if let Some(prefix) = runtime.strip_suffix(suffix) {
                let layer = prefix.strip_prefix("model.layers.").unwrap();
                let gate = value.try_index_device((.., ..8, ..), stream).unwrap();
                let up = value.try_index_device((.., 8.., ..), stream).unwrap();
                specs.push(gguf_tensor_from_array(
                    format!("blk.{layer}.{gate_name}"),
                    &gate,
                ));
                specs.push(gguf_tensor_from_array(
                    format!("blk.{layer}.{up_name}"),
                    &up,
                ));
                break;
            }
        }
        if runtime.ends_with(".moe.experts.gate_up_proj")
            || runtime.ends_with(".moe.shared_experts.gate_up_proj")
        {
            continue;
        }
        let name = inkling_gguf_layer_name(runtime)
            .unwrap_or_else(|| panic!("missing Inkling GGUF name for {runtime}"));
        specs.push(gguf_tensor_from_array(name, value));
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
        .write(
            std::fs::File::create(path).unwrap(),
            &inkling_gguf_metadata(),
            &tensors,
        )
        .unwrap();
}

struct GgufFixtureTensor {
    name: String,
    dimensions: Vec<u64>,
    data: Vec<u8>,
}

fn patterned_values(length: usize, scale: f32, phase: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let centered = ((index * 17 + phase * 11) % 29) as f32 - 14.0;
            centered * scale
        })
        .collect()
}

fn f32_gguf_tensor(
    name: impl Into<String>,
    dimensions: Vec<u64>,
    values: Vec<f32>,
) -> GgufFixtureTensor {
    assert_eq!(dimensions.iter().product::<u64>() as usize, values.len());
    GgufFixtureTensor {
        name: name.into(),
        dimensions,
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}
