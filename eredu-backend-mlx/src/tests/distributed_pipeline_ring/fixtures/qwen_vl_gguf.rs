// Primary text and mandatory DeepStack projector use their released GGUF schemas.
fn write_qwen_vl_gguf_component_fixture(path: &Path, routed: bool) {
    write_qwen_vl_gguf_component_fixture_encoded(path, routed, false);
}

fn write_qwen_vl_gguf_component_fixture_encoded(path: &Path, routed: bool, packed: bool) {
    use eredu_architectures::qwen::{self, TextConfigContext};
    let architecture = if routed { "qwen3vlmoe" } else { "qwen3vl" };
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (key("block_count"), GgufMetadataValue::Uint32(2)),
        (key("embedding_length"), GgufMetadataValue::Uint32(64)),
        (key("feed_forward_length"), GgufMetadataValue::Uint32(128)),
        (key("attention.head_count"), GgufMetadataValue::Uint32(4)),
        (key("attention.head_count_kv"), GgufMetadataValue::Uint32(2)),
        (key("attention.key_length"), GgufMetadataValue::Uint32(16)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufMetadataValue::Float32(1e-6),
        ),
        (key("vocab_size"), GgufMetadataValue::Uint32(64)),
        (key("context_length"), GgufMetadataValue::Uint32(128)),
        (key("rope.freq_base"), GgufMetadataValue::Float32(10000.0)),
        (
            key("rope.dimension_sections"),
            GgufMetadataValue::Array(MetadataArray::Int32(vec![4, 2, 2])),
        ),
        (key("n_deepstack_layers"), GgufMetadataValue::Uint32(2)),
    ]);
    if routed {
        metadata.insert(key("expert_count"), GgufMetadataValue::Uint32(4));
        metadata.insert(key("expert_used_count"), GgufMetadataValue::Uint32(2));
        metadata.insert(
            key("expert_feed_forward_length"),
            GgufMetadataValue::Uint32(64),
        );
    }
    struct Tied;
    impl eredu_architectures::GgufTensorCatalog for Tied {
        fn contains(&self, _: &str) -> bool {
            false
        }
        fn any(&self, _: impl FnMut(&str) -> bool) -> bool {
            false
        }
    }
    let text = qwen::model_args_from_gguf_catalog_with_context(
        &Tied,
        &metadata.clone().into_iter().collect(),
        if routed {
            TextConfigContext::Qwen3VlMoe
        } else {
            TextConfigContext::Qwen3Vl
        },
    )
    .unwrap();
    let projector_metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("clip".into()),
        ),
        ("general.file_type".into(), GgufMetadataValue::Uint32(0)),
        (
            "clip.projector_type".into(),
            GgufMetadataValue::String("qwen3vl_merger".into()),
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
            "clip.vision.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.patch_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.spatial_merge_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.projection_dim".into(),
            GgufMetadataValue::Uint32(64),
        ),
        (
            "clip.vision.image_mean".into(),
            GgufMetadataValue::Array(MetadataArray::Float32(vec![0.5; 3])),
        ),
        (
            "clip.vision.image_std".into(),
            GgufMetadataValue::Array(MetadataArray::Float32(vec![0.5; 3])),
        ),
        (
            "clip.vision.is_deepstack_layers".into(),
            GgufMetadataValue::Array(MetadataArray::Bool(vec![true, true])),
        ),
    ]);
    struct Vision;
    impl qwen::vision::VisionGgufCatalog for Vision {
        fn shape(&self, name: &str) -> Option<Vec<usize>> {
            (name == "v.position_embd.weight").then_some(vec![16, 64])
        }
    }
    let vision = qwen::vl::vision_config_from_gguf_catalog(
        &Vision,
        &projector_metadata.clone().into_iter().collect(),
    )
    .unwrap();
    let args =
        qwen::vl::model_args_from_gguf_parts(text, &metadata.clone().into_iter().collect(), vision)
            .unwrap()
            .with_media_token_ids(42, 43)
            .unwrap();
    for (path, mut metadata, plan) in [
        (
            path.to_path_buf(),
            metadata,
            qwen::gguf_plan(&args.text).unwrap(),
        ),
        (
            path.parent().unwrap().join("mmproj.gguf"),
            projector_metadata,
            qwen::vl::projector_gguf_plan(&args).unwrap(),
        ),
    ] {
        if packed {
            metadata.remove("general.file_type");
        }
        let mut encodings = BTreeMap::new();
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
                        if tensor.key.contains("norm") && tensor.key.ends_with("weight") {
                            0.9 + delta * 0.002
                        } else {
                            delta * 0.008
                        }
                    })
                    .collect();
                let mut stored = f32_gguf_tensor(
                    tensor.key.clone(),
                    tensor.shape.iter().rev().map(|&d| d as u64).collect(),
                    values,
                );
                if packed
                    && matches!(
                        tensor.encoding,
                        eredu_checkpoint::schema::GgufTypeConstraint::OperationClass(
                            eredu_checkpoint::schema::TensorOperation::Matrix
                        )
                    )
                {
                    let width = *tensor.shape.last().unwrap();
                    assert_eq!(width % 32, 0);
                    let ty = if seed % 2 == 0 {
                        GgmlType::Q8_0
                    } else {
                        GgmlType::IQ4NL
                    };
                    let mut bytes = Vec::new();
                    for block in 0..tensor.shape.iter().product::<usize>() / 32 {
                        let exponent =
                            (block % 3) as i32 - if ty == GgmlType::Q8_0 { 8 } else { 11 };
                        bytes.extend((((exponent + 15) as u16) << 10).to_le_bytes());
                        let code = |column: usize| block * 7 + column * 11 + seed as usize;
                        if ty == GgmlType::Q8_0 {
                            for column in 0..32 {
                                bytes.push(((code(column) % 31) as i8 - 15) as u8);
                            }
                        } else {
                            for column in 0..16 {
                                bytes.push(
                                    (code(column) % 16) as u8
                                        | ((code(column + 16) % 16) as u8) << 4,
                                );
                            }
                        }
                    }
                    stored.data = bytes;
                    encodings.insert(tensor.key.clone(), ty);
                }
                stored
            })
            .collect::<Vec<_>>();
        if packed {
            assert!(encodings.values().any(|&ty| ty == GgmlType::Q8_0));
            assert!(encodings.values().any(|&ty| ty == GgmlType::IQ4NL));
        }
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
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
            .unwrap();
    }
    // Sidecar geometry is used only by the native test's expected unit counts;
    // the explicitly selected .gguf file remains the production admission source.
    std::fs::write(
        path.parent().unwrap().join("config.json"),
        serde_json::to_vec(
            &serde_json::json!({"text_config": {"num_hidden_layers": args.text.num_hidden_layers},
            "vision_config": {"depth": args.vision.layer_schedule.len()}}),
        )
        .unwrap(),
    )
    .unwrap();
    // The native low-level consumer binds this explicit fixture protocol through
    // the same neutral inspected-artifact API that the facade's tokenizer uses.
    std::fs::write(
        path.parent().unwrap().join("component-media-fixture.json"),
        serde_json::to_vec(&serde_json::json!({
            "patch_width": 24,
            "image_token_id": 42, "video_token_id": 43,
            "vision_start_token_id": 44, "vision_end_token_id": 45,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn component_fixture_inspection(
    path: &Path,
) -> eredu_core::ArtifactInspection<eredu_architectures::processor_plan::ArtifactArchitecturePlan> {
    let mut inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    if let Some(eredu_architectures::processor_plan::GgufSpecialTokenKind::Qwen) = inspection
        .architecture_plan()
        .required_gguf_special_tokens()
    {
        let protocol: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.parent().unwrap().join("component-media-fixture.json")).unwrap(),
        )
        .unwrap();
        let token = |name: &str| u32::try_from(protocol[name].as_u64().unwrap()).unwrap();
        inspection
            .architecture_plan_mut()
            .bind_gguf_special_token_ids(
                eredu_architectures::processor_plan::GgufSpecialTokenIds::Qwen {
                    image_token_id: token("image_token_id"),
                    video_token_id: token("video_token_id"),
                    vision_start_token_id: token("vision_start_token_id"),
                    vision_end_token_id: token("vision_end_token_id"),
                },
            )
            .unwrap();
    }
    inspection
}
