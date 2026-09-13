// Use the published projector geometry with a bounded prepared patch grid.
// The small decoder isolates component experiments from fixture storage size.
fn write_muse_projector_gguf_component_fixture(path: &Path, routed: bool) {
    use eredu_architectures::muse_glimmer;
    use eredu_checkpoint::schema::{GgufTypeConstraint, TensorOperation};

    write_muse_glimmer_gguf_component_fixture(path, routed);
    let primary = eredu_gguf::Checkpoint::open(path).unwrap();
    let args = muse_glimmer::DecoderConfig::from_gguf_catalog(
        &primary,
        &primary.metadata().clone().into_iter().collect(),
    )
    .unwrap();
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("clip".into()),
        ),
        (
            "general.type".into(),
            GgufMetadataValue::String("mmproj".into()),
        ),
        (
            "clip.projector_type".into(),
            GgufMetadataValue::String("muse-glimmer".into()),
        ),
        (
            "clip.has_vision_encoder".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "clip.vision.attention.layer_norm_epsilon".into(),
            GgufMetadataValue::Float32(1e-5),
        ),
    ]);
    for (name, value) in [
        ("embedding_length", 1536),
        ("feed_forward_length", 8960),
        ("block_count", 50),
        ("attention.head_count", 16),
        ("patch_size", 14),
        ("spatial_merge_size", 2),
        ("image_size", 896),
        ("projection_dim", 64),
    ] {
        metadata.insert(
            format!("clip.vision.{name}"),
            GgufMetadataValue::Uint32(value),
        );
    }
    for name in ["clip.vision.image_mean", "clip.vision.image_std"] {
        metadata.insert(
            name.into(),
            GgufMetadataValue::Array(MetadataArray::Float32(vec![0.5; 3])),
        );
    }
    let args = args
        .with_gguf_projector_metadata(&metadata.clone().into_iter().collect(), Default::default())
        .unwrap();
    let plan = muse_glimmer::projector_gguf_plan(&args).unwrap();
    let mut encodings = BTreeMap::new();
    let tensors = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            let seed = tensor
                .key
                .bytes()
                .fold(0u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
            let count = tensor.shape.iter().product::<usize>();
            let packed = matches!(
                tensor.encoding,
                GgufTypeConstraint::OperationClass(TensorOperation::Matrix)
            );
            let data = if packed {
                assert_eq!(tensor.shape.last().unwrap() % 32, 0);
                let ty = if seed % 2 == 0 {
                    GgmlType::Q8_0
                } else {
                    GgmlType::IQ4NL
                };
                encodings.insert(tensor.key.clone(), ty);
                let (_, block_bytes) = ty.block_and_bytes().unwrap();
                let mut bytes = Vec::with_capacity(count / 32 * block_bytes as usize);
                for block in 0..count / 32 {
                    let exponent = (block % 3) as i32 - if ty == GgmlType::Q8_0 { 11 } else { 14 };
                    bytes.extend((((exponent + 15) as u16) << 10).to_le_bytes());
                    let code = |column: usize| block * 7 + column * 11 + seed as usize;
                    if ty == GgmlType::Q8_0 {
                        for column in 0..32 {
                            bytes.push(((code(column) % 31) as i8 - 15) as u8);
                        }
                    } else {
                        for column in 0..16 {
                            bytes.push(
                                (code(column) % 16) as u8 | ((code(column + 16) % 16) as u8) << 4,
                            );
                        }
                    }
                }
                bytes
            } else {
                (0..count)
                    .flat_map(|index| {
                        let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                        let value = if tensor.key.contains("ln") && tensor.key.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else {
                            delta * 0.002
                        };
                        value.to_le_bytes()
                    })
                    .collect()
            };
            GgufFixtureTensor {
                name: tensor.key.clone(),
                dimensions: tensor.shape.iter().rev().map(|&d| d as u64).collect(),
                data,
            }
        })
        .collect::<Vec<_>>();
    assert!(encodings.values().any(|&ty| ty == GgmlType::Q8_0));
    assert!(encodings.values().any(|&ty| ty == GgmlType::IQ4NL));
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: encodings
                .get(&tensor.name)
                .copied()
                .unwrap_or(GgmlType::F32),
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    let root = path.parent().unwrap();
    Writer::default()
        .write(
            std::fs::File::create(root.join("mmproj.gguf")).unwrap(),
            &metadata,
            &inputs,
        )
        .unwrap();
    // Only test accounting and prepared-input construction read these sidecars.
    // Production admission consumes the explicitly selected GGUF and projector.
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "text_config": {"num_hidden_layers": 2},
            "vision_config": {"num_hidden_layers": 50},
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("component-media-fixture.json"),
        br#"{"patch_width":588,"case_timeout_seconds":1800}"#,
    )
    .unwrap();
}
