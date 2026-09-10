fn write_f32_shard(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
    let buffers = tensors
        .iter()
        .map(|(_, shape, value)| {
            let count = shape.iter().product::<usize>();
            (0..count)
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&buffers)
        .map(|((name, shape, _), bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        });
    serialize_to_file(views, None, path).unwrap();
}

fn write_fixture(directory: &Path) {
    write_llama_compatible_fixture(directory, "llama");
}

fn write_mistral_fixture(directory: &Path) {
    write_llama_compatible_fixture(directory, "mistral");
}

fn write_unindexed_llama_compatible_fixture(directory: &Path, model_type: &str) {
    write_llama_compatible_fixture(directory, model_type);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut arrays = BTreeMap::new();
    for shard in [
        "input.safetensors",
        "layer-0.safetensors",
        "layer-1.safetensors",
        "output.safetensors",
    ] {
        arrays.extend(Array::load_safetensors(directory.join(shard), &stream).unwrap());
    }
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::remove_file(directory.join("model.safetensors.index.json")).unwrap();
}

fn write_llama_compatible_gguf(path: &Path, architecture: &str) {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(2)),
        (key("embedding_length"), GgufMetadataValue::Uint32(4)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(2)),
        (key("attention.head_count_kv"), GgufMetadataValue::Uint32(2)),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(4)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(1e-5),
        ),
        (key("vocab_size"), GgufMetadataValue::Uint32(4)),
        (key("context_length"), GgufMetadataValue::Uint32(32)),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10_000.0)),
    ]);
    let vector = vec![0_u8; 4 * std::mem::size_of::<f32>()];
    let matrix = vec![0_u8; 16 * std::mem::size_of::<f32>()];
    let vector_dimensions: &[u64] = &[4];
    let matrix_dimensions: &[u64] = &[4, 4];
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
    ];
    for layer in 0..2 {
        names.extend(
            [
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            .map(|suffix| format!("blk.{layer}.{suffix}")),
        );
    }
    let tensors = names
        .iter()
        .map(|name| {
            let vector_tensor = name.ends_with("norm.weight");
            TensorInput {
                name,
                dimensions: if vector_tensor {
                    vector_dimensions
                } else {
                    matrix_dimensions
                },
                ggml_type: GgmlType::F32,
                data: if vector_tensor { &vector } else { &matrix },
            }
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

fn write_llama_compatible_fixture(directory: &Path, model_type: &str) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": model_type,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false,
            "attention_schedule": ["full", {"sliding": {"window": 2}}]
        }))
        .unwrap(),
    )
    .unwrap();
    write_f32_shard(
        &directory.join("input.safetensors"),
        &[("model.embed_tokens.weight", vec![64, 64], 0.01)],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let names = [
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![64, 64],
                0.01,
            ),
            (format!("{prefix}.mlp.gate_proj.weight"), vec![64, 64], 0.01),
            (format!("{prefix}.mlp.up_proj.weight"), vec![64, 64], 0.01),
            (format!("{prefix}.mlp.down_proj.weight"), vec![64, 64], 0.01),
            (format!("{prefix}.input_layernorm.weight"), vec![64], 1.0),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![64],
                1.0,
            ),
        ];
        let borrowed = names
            .iter()
            .map(|(name, shape, value)| (name.as_str(), shape.clone(), *value))
            .collect::<Vec<_>>();
        write_f32_shard(
            &directory.join(format!("layer-{layer}.safetensors")),
            &borrowed,
        );
    }
    write_f32_shard(
        &directory.join("output.safetensors"),
        &[
            ("model.norm.weight", vec![64], 1.0),
            ("lm_head.weight", vec![64, 64], 0.01),
        ],
    );
    let mut weight_map = serde_json::Map::new();
    weight_map.insert(
        "model.embed_tokens.weight".into(),
        serde_json::json!("input.safetensors"),
    );
    for layer in 0..2 {
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            weight_map.insert(
                format!("model.layers.{layer}.{suffix}"),
                serde_json::json!(format!("layer-{layer}.safetensors")),
            );
        }
    }
    weight_map.insert(
        "model.norm.weight".into(),
        serde_json::json!("output.safetensors"),
    );
    weight_map.insert(
        "lm_head.weight".into(),
        serde_json::json!("output.safetensors"),
    );
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_k2_fixture(directory: &Path, mova: bool) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let mut config = fixture[if mova { "mova" } else { "dense" }]["config"].clone();
    config["num_hidden_layers"] = 3.into();
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let tensors = eredu_architectures::k2_horizon::parameter_shapes(&args, false)
        .unwrap()
        .into_iter()
        .map(|(name, shape)| {
            let phase = name
                .bytes()
                .fold(0_usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
            let bytes = (0..shape.iter().product::<usize>())
                .flat_map(|i| {
                    (if name.contains("norm") {
                        1.0 + (i % 7) as f32 * 0.01
                    } else {
                        ((i * 17 + phase) % 101) as f32 * 0.003 - 0.15
                    })
                    .to_le_bytes()
                })
                .collect::<Vec<_>>();
            (name, shape, bytes)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        tensors.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.join("model.safetensors"),
    )
    .unwrap();
}
