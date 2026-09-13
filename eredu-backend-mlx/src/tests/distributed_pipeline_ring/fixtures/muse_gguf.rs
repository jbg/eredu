fn write_muse_glimmer_gguf_component_fixture(path: &Path, routed: bool) {
    let key = |suffix: &str| format!("muse-glimmer.{suffix}");
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("muse-glimmer".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(2)),
        (key("embedding_length"), GgufMetadataValue::Uint32(64)),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(128)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(4)),
        (key("attention.head_count_kv"), GgufMetadataValue::Uint32(2)),
        (key("attention.key_length"), GgufMetadataValue::Uint32(16)),
        (
            key("attention.sliding_window"),
            GgufMetadataValue::Uint32(8),
        ),
        (
            key("attention.sliding_window_pattern"),
            GgufMetadataValue::Array(MetadataArray::Bool(vec![true, false])),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(1e-5),
        ),
        (
            key("attention.post_norm_rms_epsilon"),
            GgufMetadataValue::Float32(2e-5),
        ),
        (key("vocab_size"), GgufMetadataValue::Uint32(64)),
        (key("context_length"), GgufMetadataValue::Uint32(64)),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10000.0)),
        (key("logit_scale"), GgufMetadataValue::Float32(1.9)),
        (
            key("final_logit_softcapping"),
            GgufMetadataValue::Float32(7.0),
        ),
        (key("image_token_id"), GgufMetadataValue::Uint32(42)),
        (key("video_token_id"), GgufMetadataValue::Uint32(43)),
    ]);
    if routed {
        metadata.insert(key("expert_count"), GgufMetadataValue::Uint32(4));
        metadata.insert(key("expert_used_count"), GgufMetadataValue::Uint32(2));
        metadata.insert(
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(64),
        );
    }
    struct Untied;
    impl eredu_architectures::GgufTensorCatalog for Untied {
        fn contains(&self, name: &str) -> bool {
            name == "output.weight"
        }
        fn any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
            predicate("output.weight")
        }
    }
    let args = eredu_architectures::muse_glimmer::DecoderConfig::from_gguf_catalog(
        &Untied,
        &metadata.clone().into_iter().collect(),
    )
    .unwrap();
    let plan = eredu_architectures::muse_glimmer::gguf_plan(&args).unwrap();
    let tensors = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let seed = tensor
                .key
                .bytes()
                .fold(0u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
            let values = (0..tensor.shape.iter().product())
                .map(|index| {
                    let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                    if tensor.key.contains("norm") {
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
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}
