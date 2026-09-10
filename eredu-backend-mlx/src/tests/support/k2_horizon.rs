//! Nonzero published GGUF layouts with independent block arithmetic expansion.

pub(crate) fn gguf(format: eredu_gguf::GgmlType) -> tempfile::TempDir {
    use eredu_gguf::{GgmlType, MetadataValue as V, TensorInput, Writer};
    use std::collections::BTreeMap;
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let mut config = fixture["mova"]["config"].clone();
    config["hidden_size"] = 64.into();
    config["head_dim"] = 8.into();
    config["intermediate_size"] = 64.into();
    config["moe_intermediate_size"] = 64.into();
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    let plan = eredu_architectures::k2_horizon::gguf_plan(&args).unwrap();
    let mut metadata = BTreeMap::from([(
        "general.architecture".into(),
        V::String("k2-horizon".into()),
    )]);
    for (key, value) in [
        ("block_count", 3),
        ("embedding_length", 64),
        ("feed_forward_length", 64),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 8),
        ("context_length", 256),
        ("vocab_size", 17),
        ("attention.group_norm_groups", 2),
        ("expert_count", 5),
        ("expert_used_count", 2),
        ("expert_shared_count", 2),
        ("expert_feed_forward_length", 64),
        ("attention.value_expert_count", 3),
        ("attention.value_expert_used_count", 2),
        ("leading_dense_block_count", 1),
        ("expert_gating_func", 2),
        ("rope.dimension_count", 2),
        ("rope.scaling.original_context_length", 16),
    ] {
        metadata.insert(format!("k2-horizon.{key}"), V::Uint32(value));
    }
    for (key, value) in [
        ("attention.layer_norm_rms_epsilon", 1e-5),
        ("rope.freq_base", 1000.),
        ("expert_weights_scale", 1.7),
        ("rope.scaling.factor", 4.),
        ("rope.scaling.yarn_beta_fast", 8.),
        ("rope.scaling.yarn_beta_slow", 1.),
        ("rope.scaling.yarn_attn_factor", 1.35),
    ] {
        metadata.insert(format!("k2-horizon.{key}"), V::Float32(value));
    }
    metadata.insert(
        "k2-horizon.rope.scaling.type".into(),
        V::String("yarn".into()),
    );
    metadata.insert("k2-horizon.expert_weights_norm".into(), V::Bool(false));
    let root = tempfile::tempdir().unwrap();
    let packed_path = root.path().join("packed.gguf");
    let expanded_path = root.path().join("expanded.gguf");
    let mut encoded = Vec::new();
    let mut expanded = Vec::new();
    for tensor in &plan.common_tensors {
        let name = &tensor.key;
        let shape = &tensor.shape;
        let count = shape.iter().product::<usize>();
        let phase = name
            .bytes()
            .fold(0_usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
        let packed = name.contains("_exps.weight");
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        if packed {
            assert_eq!(count % 32, 0);
            for block in 0..count / 32 {
                let code = |i: usize| ((i * 7 + block * 3 + phase) % 16) as u8;
                let scale = 2.0_f32.powi(-9 + (block % 3) as i32);
                if format == GgmlType::MxFp4 {
                    bytes.push((118 + block % 3) as u8);
                } else {
                    bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                }
                for i in 0..16 {
                    bytes.push(code(i) | (code(i + 16) << 4));
                }
                for i in 0..32 {
                    let c = code(i) as usize;
                    let value = match format {
                        GgmlType::Q4_0 => c as f32 - 8.,
                        GgmlType::MxFp4 => [
                            0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4.,
                            -6.,
                        ][c],
                        GgmlType::IQ4NL => [
                            -127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38.,
                            53., 69., 89., 113.,
                        ][c],
                        _ => unreachable!(),
                    };
                    values.push(scale * value);
                }
            }
        } else {
            values = (0..count)
                .map(|i| {
                    if name.contains("norm") {
                        1.0 + (i % 7) as f32 * 0.01
                    } else {
                        ((i * 17 + phase) % 101) as f32 * 0.0006 - 0.03
                    }
                })
                .collect();
        }
        let dense_bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        if !packed {
            bytes = dense_bytes.clone();
        }
        let dimensions = shape.iter().rev().map(|v| *v as u64).collect::<Vec<_>>();
        encoded.push((
            name.clone(),
            dimensions.clone(),
            if packed { format } else { GgmlType::F32 },
            bytes,
        ));
        expanded.push((name.clone(), dimensions, GgmlType::F32, dense_bytes));
    }
    for (path, tensors) in [(&packed_path, &encoded), (&expanded_path, &expanded)] {
        let inputs = tensors
            .iter()
            .map(|(name, dimensions, ggml_type, data)| TensorInput {
                name,
                dimensions,
                ggml_type: *ggml_type,
                data,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
            .unwrap();
    }
    root
}

pub(crate) fn fp8() -> (tempfile::TempDir, tempfile::TempDir) {
    use safetensors::{
        tensor::{serialize_to_file, TensorView},
        Dtype,
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let mut config = fixture["mova"]["config"].clone();
    config["hidden_size"] = 256.into();
    config["head_dim"] = 128.into();
    config["rope_head_dim"] = 128.into();
    config["intermediate_size"] = 256.into();
    config["moe_intermediate_size"] = 256.into();
    config["query_key_norm"] = false.into();
    let dense = tempfile::tempdir().unwrap();
    std::fs::write(
        dense.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let excluded = [
        "model.embed_tokens",
        "lm_head",
        "model.layers.0",
        "model.layers.1.mlp.gate",
        "model.layers.2.mlp.gate",
        "model.layers.1.self_attn.v_router",
        "model.layers.2.self_attn.v_router",
        "model.layers.1.mlp.shared_experts",
        "model.layers.2.mlp.shared_experts",
        "model.layers.1.self_attn.gate_proj",
        "model.layers.2.self_attn.gate_proj",
    ];
    config["quantization_config"] = serde_json::json!({"quant_method":"fp8", "activation_scheme":"dynamic", "weight_block_size":[128,128], "ignored_layers":excluded});
    let encoded = tempfile::tempdir().unwrap();
    std::fs::write(
        encoded.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let plan = resolved.architecture.checkpoint();
    let constraints = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .filter(|g| g.required)
                .flat_map(|g| &g.variants[0].tensors),
        )
        .filter(|c| c.requirement == eredu_checkpoint::schema::TensorRequirement::Required);
    let hash = |name: &str| {
        name.bytes()
            .fold(0_u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32))
    };
    let scale = |name: &str, block: usize| (1 + (hash(name) as usize + block) % 5) as f32 * 0.125;
    let mut packed = Vec::new();
    let mut expanded = Vec::new();
    for tensor in constraints {
        let count = tensor.shape.iter().product::<usize>();
        if tensor.role == eredu_checkpoint::schema::TensorRole::Companion {
            let weight = tensor.key.strip_suffix("_scale_inv").unwrap();
            let bytes = (0..count)
                .flat_map(|i| scale(weight, i).to_le_bytes())
                .collect::<Vec<_>>();
            packed.push((tensor.key.clone(), tensor.shape.clone(), Dtype::F32, bytes));
            continue;
        }
        let fp8 = tensor.dtype
            == eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                eredu_checkpoint::StoredDtype::F8E4M3,
            );
        let mut bytes = Vec::new();
        let mut dense_bytes = Vec::new();
        for index in 0..count {
            let value = if fp8 {
                // E4M3 exponent 2, mantissa 0..7: +/- 2^-5 * (1 + m/8).
                let mantissa = (index + hash(&tensor.key) as usize) % 8;
                let negative = (index / 7 + hash(&tensor.key) as usize) % 2 == 0;
                bytes.push(0x10 | mantissa as u8 | if negative { 0x80 } else { 0 });
                let rows = tensor.shape[tensor.shape.len() - 2];
                let columns = tensor.shape[tensor.shape.len() - 1];
                let row = index / columns % rows;
                let column = index % columns;
                let block = row / 128 * columns.div_ceil(128) + column / 128;
                (if negative { -1. } else { 1. })
                    * 0.03125
                    * (1. + mantissa as f32 / 8.)
                    * scale(&tensor.key, block)
            } else if tensor.key.contains("norm") {
                1.0
            } else {
                ((index * 17 + hash(&tensor.key) as usize) % 101) as f32 * 0.0006 - 0.03
            };
            dense_bytes.extend_from_slice(&value.to_le_bytes());
        }
        if !fp8 {
            bytes = dense_bytes.clone();
        }
        packed.push((
            tensor.key.clone(),
            tensor.shape.clone(),
            if fp8 { Dtype::F8_E4M3 } else { Dtype::F32 },
            bytes,
        ));
        expanded.push((
            tensor.key.clone(),
            tensor.shape.clone(),
            Dtype::F32,
            dense_bytes,
        ));
    }
    for (root, tensors) in [(&encoded, &packed), (&dense, &expanded)] {
        serialize_to_file(
            tensors.iter().map(|(name, shape, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            }),
            None,
            &root.path().join("model.safetensors"),
        )
        .unwrap();
    }
    (dense, encoded)
}
