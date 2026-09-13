// Full shared-KV text model plus its released sibling vision projector schema.
fn write_gemma4_gguf_component_fixture(path: &Path, sparse: bool) {
    use eredu_architectures::gemma4;
    let key = |suffix: &str| format!("gemma4.{suffix}");
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("gemma4".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(4)),
        (key("embedding_length"), GgufMetadataValue::Uint32(64)),
        (
            key("embedding_length_per_layer_input"),
            GgufMetadataValue::Uint32(32),
        ),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(128)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(4)),
        (key("attention.head_count_kv"), GgufMetadataValue::Uint32(2)),
        (key("attention.key_length"), GgufMetadataValue::Uint32(16)),
        (
            key("attention.shared_kv_layers"),
            GgufMetadataValue::Uint32(2),
        ),
        (
            key("attention.sliding_window"),
            GgufMetadataValue::Uint32(8),
        ),
        (
            key("attention.sliding_window_pattern"),
            GgufMetadataValue::Array(MetadataArray::Bool(vec![true, false, true, false])),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(1e-5),
        ),
        (key("vocab_size"), GgufMetadataValue::Uint32(64)),
        (key("context_length"), GgufMetadataValue::Uint32(128)),
        (
            key("final_logit_softcapping"),
            GgufMetadataValue::Float32(7.0),
        ),
        (key("image_token_id"), GgufMetadataValue::Uint32(42)),
    ]);
    if sparse {
        metadata.insert(key("expert_count"), GgufMetadataValue::Uint32(4));
        metadata.insert(key("expert_used_count"), GgufMetadataValue::Uint32(2));
        metadata.insert(
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(64),
        );
    }
    struct Catalog {
        reuse_key: bool,
    }
    impl eredu_architectures::GgufTensorCatalog for Catalog {
        fn contains(&self, name: &str) -> bool {
            matches!(
                name,
                "output.weight" | "blk.0.attn_q.bias" | "blk.1.attn_k.weight"
            ) || (!self.reuse_key && name == "blk.1.attn_v.weight")
        }
        fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
            [
                "output.weight",
                "blk.0.attn_q.bias",
                "blk.1.attn_k.weight",
                "blk.1.attn_v.weight",
            ]
            .into_iter()
            .any(|name| self.contains(name) && predicate(name))
        }
    }
    let text = gemma4::ModelArgs::from_gguf_metadata(
        &Catalog { reuse_key: sparse },
        &metadata.clone().into_iter().collect(),
    )
    .unwrap();
    let projector = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("clip".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            "clip.has_vision_encoder".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "clip.vision.projector_type".into(),
            GgufMetadataValue::String("gemma4".into()),
        ),
        (
            "clip.vision.embedding_length".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "clip.vision.feed_forward_length".into(),
            GgufMetadataValue::Uint32(128),
        ),
        (
            "clip.vision.attention.head_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "clip.vision.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.attention.key_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "clip.vision.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.patch_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.pooling_kernel_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.position_embedding_size".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "clip.vision.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-5),
        ),
    ]);
    let family = gemma4::family_from_gguf_metadata(
        text,
        &metadata.clone().into_iter().collect(),
        Some(&projector.clone().into_iter().collect()),
    )
    .unwrap();
    for (destination, metadata, plan) in [
        (
            path.to_path_buf(),
            metadata,
            gemma4::gguf_plan(&family.text).unwrap(),
        ),
        (
            path.parent().unwrap().join("mmproj.gguf"),
            projector,
            gemma4::mmproj_gguf_plan(&family).unwrap(),
        ),
    ] {
        let tensors = plan
            .common_tensors
            .iter()
            .chain(
                plan.layout_groups
                    .iter()
                    .filter_map(|group| group.variants.first())
                    .flat_map(|variant| &variant.tensors),
            )
            .map(|tensor| {
                let seed = tensor
                    .key
                    .bytes()
                    .fold(0u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
                let values = (0..tensor.shape.iter().product())
                    .map(|index| {
                        let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        if tensor.key.ends_with("layer_output_scale.weight") {
                            1.15
                        } else if tensor.key.ends_with("input_min")
                            || tensor.key.ends_with("output_min")
                        {
                            -4.0
                        } else if tensor.key.ends_with("input_max")
                            || tensor.key.ends_with("output_max")
                        {
                            4.0
                        } else if tensor.key.contains("norm") && tensor.key.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else {
                            delta * 0.008
                        }
                    })
                    .collect();
                f32_gguf_tensor(
                    tensor.key.clone(),
                    tensor.shape.iter().rev().map(|&d| d as u64).collect(),
                    values,
                )
            })
            .collect::<Vec<_>>();
        let inputs = tensors
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
                std::fs::File::create(destination).unwrap(),
                &metadata,
                &inputs,
            )
            .unwrap();
    }
    // Only the worker's expected execution-unit counts use this sidecar; model
    // admission always receives the explicit GGUF path and its sibling projector.
    std::fs::write(path.parent().unwrap().join("config.json"), serde_json::to_vec(&serde_json::json!({
        "text_config": {"num_hidden_layers": 4, "enable_moe_block": sparse}, "vision_config": {"num_hidden_layers": 2}
    })).unwrap()).unwrap();
}
