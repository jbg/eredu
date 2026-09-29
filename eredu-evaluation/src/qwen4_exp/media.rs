//! Independent miniature media checkpoint fixtures shared by portable and native tests.
use std::collections::BTreeMap;

fn vision_values(depth: usize) -> Vec<(String, Vec<usize>, Vec<u8>)> {
    assert!(depth > 0);
    let block = [
        ("attn.proj.bias", vec![8]),
        ("attn.proj.weight", vec![8, 8]),
        ("attn.qkv.bias", vec![24]),
        ("attn.qkv.weight", vec![24, 8]),
        ("mlp.linear_fc1.bias", vec![12]),
        ("mlp.linear_fc1.weight", vec![12, 8]),
        ("mlp.linear_fc2.bias", vec![8]),
        ("mlp.linear_fc2.weight", vec![8, 12]),
        ("norm1.bias", vec![8]),
        ("norm1.weight", vec![8]),
        ("norm2.bias", vec![8]),
        ("norm2.weight", vec![8]),
    ];
    let mut shapes = BTreeMap::from([
        ("merger.linear_fc1.bias".to_owned(), vec![32]),
        ("merger.linear_fc1.weight".to_owned(), vec![32, 32]),
        ("merger.linear_fc2.bias".to_owned(), vec![32]),
        ("merger.linear_fc2.weight".to_owned(), vec![32, 32]),
        ("merger.norm.bias".to_owned(), vec![8]),
        ("merger.norm.weight".to_owned(), vec![8]),
        ("patch_embed.proj.bias".to_owned(), vec![8]),
        ("patch_embed.proj.weight".to_owned(), vec![8, 3, 2, 2, 2]),
        ("pos_embed.weight".to_owned(), vec![16, 8]),
    ]);
    for index in 0..depth {
        for (name, shape) in &block {
            shapes.insert(format!("blocks.{index}.{name}"), shape.clone());
        }
    }
    let mut tensors = Vec::<(String, Vec<usize>, Vec<u8>)>::new();
    for (name, shape) in shapes {
        let canonical = format!("model.visual.{name}");
        let seed = canonical.bytes().map(usize::from).sum::<usize>() % 17;
        let data: Vec<u8> = (0..shape.iter().product())
            .flat_map(|i| {
                let v = ((i * 7 + seed) % 37) as f32 / 256. - 18. / 256.
                    + if name.contains("norm") && name.ends_with("weight") {
                        1.
                    } else {
                        0.
                    };
                v.to_le_bytes()
            })
            .collect();
        tensors.push((canonical, shape, data));
    }
    tensors
}

/// Writes a miniature published-format vision projector with deterministic nonzero weights.
/// `depth` controls the encoder block count for pipeline placement fixtures.
pub fn write_vision_projector(path: &std::path::Path, depth: usize) {
    use eredu_gguf::{GgmlType, MetadataArray, MetadataValue as V, TensorInput, Writer};
    let mut tensors = Vec::<(String, Vec<usize>, Vec<u8>)>::new();
    for (canonical, shape, data) in vision_values(depth) {
        let name = canonical
            .strip_prefix("model.visual.")
            .unwrap()
            .replace("pos_embed", "v.position_embd")
            .replace("patch_embed.proj", "v.patch_embd")
            .replace("merger.norm", "v.post_ln")
            .replace("merger.linear_fc1", "mm.0")
            .replace("merger.linear_fc2", "mm.2")
            .replace("blocks.", "v.blk.")
            .replace(".norm1.", ".ln1.")
            .replace(".norm2.", ".ln2.")
            .replace(".attn.qkv.", ".attn_qkv.")
            .replace(".attn.proj.", ".attn_out.")
            .replace(".mlp.linear_fc1.", ".ffn_up.")
            .replace(".mlp.linear_fc2.", ".ffn_down.");
        if name == "v.patch_embd.weight" {
            for t in 0..2 {
                let bytes = data
                    .chunks_exact(32)
                    .flat_map(|pair| pair[t * 16..t * 16 + 16].iter().copied())
                    .collect();
                tensors.push((
                    if t == 0 {
                        name.clone()
                    } else {
                        format!("{name}.1")
                    },
                    vec![8, 3, 2, 2],
                    bytes,
                ));
            }
        } else {
            tensors.push((name, shape, data));
        }
    }
    let mut metadata = BTreeMap::from([
        ("general.architecture".into(), V::String("clip".into())),
        (
            "clip.projector_type".into(),
            V::String("qwen3vl_merger".into()),
        ),
        ("clip.use_gelu".into(), V::Bool(true)),
        (
            "clip.vision.attention.layer_norm_epsilon".into(),
            V::Float32(1e-6),
        ),
        (
            "clip.vision.is_deepstack_layers".into(),
            V::Array(MetadataArray::Bool(vec![false; depth])),
        ),
    ]);
    for (name, n) in [
        ("embedding_length", 8),
        ("feed_forward_length", 12),
        ("attention.head_count", 2),
        ("block_count", u32::try_from(depth).unwrap()),
        ("patch_size", 2),
        ("spatial_merge_size", 2),
        ("projection_dim", 32),
    ] {
        metadata.insert(format!("clip.vision.{name}"), V::Uint32(n));
    }
    metadata.insert(
        "clip.vision.image_mean".into(),
        V::Array(MetadataArray::Float32(vec![0.; 3])),
    );
    metadata.insert(
        "clip.vision.image_std".into(),
        V::Array(MetadataArray::Float32(vec![1.; 3])),
    );
    metadata.insert("clip.vision.image_min_pixels".into(), V::Uint32(16));
    metadata.insert("clip.vision.image_max_pixels".into(), V::Uint32(64));
    let dimensions: Vec<Vec<u64>> = tensors
        .iter()
        .map(|(_, s, _)| s.iter().rev().map(|v| *v as u64).collect())
        .collect();
    let inputs: Vec<_> = tensors
        .iter()
        .zip(&dimensions)
        .map(|((name, _, data), dimensions)| TensorInput {
            name,
            dimensions,
            data,
            ggml_type: GgmlType::F32,
        })
        .collect();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}

/// Adds the miniature vision tower and processor configuration to a text checkpoint.
/// Existing text configuration, including MTP and context length, is retained.
/// `depth` controls the encoder block count and must match the projector fixture.
pub fn add_vision_weights(path: &std::path::Path, depth: usize) {
    let weights_path = path.join("model.safetensors");
    let original = std::fs::read(&weights_path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&original).unwrap();
    let vision = vision_values(depth);
    let mut views = tensors.tensors();
    views.extend(vision.iter().map(|(name, shape, data)| {
        (
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), data)
                .unwrap(),
        )
    }));
    safetensors::tensor::serialize_to_file(views, None, &weights_path).unwrap();
    let text_config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let config = serde_json::json!({
        "model_type": "qwen4_exp", "text_config": text_config,
        "image_token_id":12, "video_token_id":13, "vision_start_token_id":14, "vision_end_token_id":15,
        "vision_config": {"depth":depth,"hidden_size":8,"hidden_act":"gelu_pytorch_tanh","intermediate_size":12,"num_heads":2,
            "num_position_embeddings":16,"in_channels":3,"patch_size":2,"spatial_merge_size":2,
            "temporal_patch_size":2,"out_hidden_size":32,"deepstack_visual_indexes":[]}
    });
    std::fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let processor = serde_json::json!({
        "size":{"shortest_edge":16,"longest_edge":64},
        "patch_size":2,"temporal_patch_size":2,"merge_size":2,
        "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0],
        "fps":2.0,"min_frames":4,"max_frames":768
    });
    for name in ["preprocessor_config.json", "video_preprocessor_config.json"] {
        std::fs::write(path.join(name), serde_json::to_vec(&processor).unwrap()).unwrap();
    }
}
