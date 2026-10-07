use super::*;
use eredu_checkpoint::recipe::RecipeDtype;
use eredu_checkpoint::store::MemoryWeightStore;
use eredu_gguf::{Checkpoint, CheckpointHeader, GgmlType, Limits};
use serde_json::Value;

fn config() -> VoiceChatConfig {
    VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json")).unwrap()
}

// Encode only the public GGUF header format. Declared payload extents establish
// metadata provenance without allocating, writing, or authorizing model weights.
pub(in crate::voicechat) fn header_checkpoint(layout: GgufLayout) -> Checkpoint {
    Checkpoint::from_headers_with_representation(
        &[checkpoint_header(layout)],
        Limits::default(),
        eredu_gguf::QuantizedTensorRepresentation::DecodedF32,
    )
    .unwrap()
}

fn checkpoint_header(layout: GgufLayout) -> CheckpointHeader {
    let (directory, metadata) = match layout {
        GgufLayout::Complete => (
            include_str!("../../tests/fixtures/voicechat/gguf-q4-tensors.json"),
            include_str!("../../tests/fixtures/voicechat/gguf-q4-metadata.json"),
        ),
        GgufLayout::FunctionHead => (
            include_str!("../../tests/fixtures/voicechat/gguf-1-tensors.json"),
            include_str!("../../tests/fixtures/voicechat/gguf-1-metadata.json"),
        ),
        GgufLayout::Backbone => (
            include_str!("../../tests/fixtures/voicechat/gguf-2-tensors.json"),
            include_str!("../../tests/fixtures/voicechat/gguf-2-metadata.json"),
        ),
        _ => panic!("this fixture only needs complete, function and backbone containers"),
    };
    let directory: BTreeMap<String, Value> = serde_json::from_str(directory).unwrap();
    let mut metadata: BTreeMap<String, Value> = serde_json::from_str(metadata).unwrap();
    if layout == GgufLayout::Backbone {
        // Header fixtures omit large tokenizer arrays. These synthetic entries
        // satisfy structural admission only; retained-asset digest validation
        // deliberately cannot authorize them as a real tokenizer.
        let config = config();
        let mut tokens = vec![Value::String(String::new()); 131072];
        let mut types = vec![Value::from(1); 131072];
        for (index, token) in [
            (1, config.tokens.bos),
            (2, config.tokens.eos),
            (12, config.tokens.pad),
            (20, config.tokens.tool_call_start),
            (21, config.tokens.tool_call_end),
            (22, config.tokens.tool_response_end),
        ] {
            tokens[index] = Value::String(token);
            types[index] = Value::from(3);
        }
        metadata.insert("tokenizer.ggml.tokens".into(), Value::Array(tokens));
        metadata.insert("tokenizer.ggml.token_type".into(), Value::Array(types));
        metadata.insert(
            "tokenizer.ggml.merges".into(),
            Value::Array(vec![Value::String(String::new()); 269443]),
        );
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((directory.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    for (key, value) in metadata {
        string(&mut bytes, &key);
        match value {
            Value::String(value) => {
                bytes.extend(8u32.to_le_bytes());
                string(&mut bytes, &value);
            }
            Value::Number(value) if value.is_f64() => {
                bytes.extend(6u32.to_le_bytes());
                bytes.extend((value.as_f64().unwrap() as f32).to_le_bytes());
            }
            Value::Number(value) => {
                bytes.extend(11u32.to_le_bytes());
                bytes.extend(value.as_i64().unwrap().to_le_bytes());
            }
            Value::Bool(value) => {
                bytes.extend(7u32.to_le_bytes());
                bytes.push(u8::from(value));
            }
            Value::Array(values) => {
                bytes.extend(9u32.to_le_bytes());
                let strings = values.first().is_some_and(Value::is_string);
                bytes.extend(if strings { 8u32 } else { 11u32 }.to_le_bytes());
                bytes.extend((values.len() as u64).to_le_bytes());
                for value in values {
                    if strings {
                        string(&mut bytes, value.as_str().unwrap());
                    } else {
                        bytes.extend(value.as_i64().unwrap().to_le_bytes());
                    }
                }
            }
            _ => panic!("unexpected metadata kind"),
        }
    }
    let mut extent = 0u64;
    for (key, value) in directory {
        let shape: Vec<u64> = serde_json::from_value(value["shape"].clone()).unwrap();
        let code = value["type"].as_u64().unwrap() as u32;
        let (block, width) = GgmlType::from_code(code).block_and_bytes().unwrap();
        let len = shape.iter().product::<u64>() / block * width;
        extent = extent.div_ceil(32) * 32;
        string(&mut bytes, &key);
        bytes.extend((shape.len() as u32).to_le_bytes());
        for dimension in shape.into_iter().rev() {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(code.to_le_bytes());
        bytes.extend(extent.to_le_bytes());
        extent += len;
    }
    let file_len = (bytes.len() as u64).div_ceil(32) * 32 + extent;
    CheckpointHeader {
        member: "release.gguf".into(),
        bytes,
        file_len,
    }
}

pub(in crate::voicechat) fn admitted(layout: GgufLayout) -> AdmittedVoiceChatSource {
    AdmittedVoiceChatSource::from_gguf_headers(header_checkpoint(layout), &config(), layout)
        .unwrap()
}

pub(in crate::voicechat) fn official_parameters() -> BTreeMap<String, ParameterAdmission> {
    struct Catalog(eredu_checkpoint::safetensors::SafetensorsHeaderCatalog);
    impl RecipeCatalog for Catalog {
        fn tensor_metadata(
            &self,
            key: &str,
        ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
            self.0
                .tensors()
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }
    let catalog = Catalog(checkpoint::tests::official());
    let plan = checkpoint::safetensors_plan(&config()).unwrap();
    resolve_safetensors_plan(&catalog.0, &plan).unwrap();
    let bytes = catalog
        .0
        .tensors()
        .iter()
        .map(|(key, value)| (key.clone(), value.encoded_byte_len))
        .collect();
    admit_recipes(&config(), None, &catalog, &bytes, &BTreeMap::new()).unwrap()
}

#[test]
fn missing_rnnt_requires_an_explicit_companion_and_preserves_primary_function_weights() {
    let function = admitted(GgufLayout::FunctionHead);
    let required = BTreeSet::from([Component::FunctionHead, Component::Rnnt]);
    let error =
        VoiceChatArtifactPlan::new(&config(), vec![function.clone()], None, required.clone())
            .unwrap_err();
    let AdmissionError::MissingComponent { component, keys } = error else {
        panic!("expected a typed missing component")
    };
    assert_eq!(component, Component::Rnnt);
    assert_eq!(keys.len(), 15);
    assert!(keys.contains(&"stt_model.rnnt_decoder.prediction.embed.weight".to_owned()));

    let plan = VoiceChatArtifactPlan::new(
        &config(),
        vec![function],
        Some(admitted(GgufLayout::Complete)),
        required,
    )
    .unwrap();
    let function = &plan.parameters()["stt_model.function_head.weight"];
    assert_eq!(function.artifact, PhysicalArtifactId::Primary);
    assert_eq!(function.recipe.source_keys(), ["function_head.weight"]);
    let companion =
        PhysicalArtifactId::Companion(GgufCompanionRole::Named("voicechat/companion".into()));
    assert_eq!(
        plan.parameters()["stt_model.rnnt_decoder.prediction.embed.weight"].artifact,
        companion
    );
    assert_eq!(plan.sources().artifacts().len(), 2);
    let rnnt_view = plan
        .sources()
        .role(ArtifactSourceRole::Component("voicechat/rnnt"))
        .unwrap();
    assert_eq!(rnnt_view[&companion].len(), 15);
    assert!(!rnnt_view[&companion].contains("stt_model.function_head.weight"));
    assert!(
        plan.sources()
            .role(ArtifactSourceRole::Component("voicechat/backbone"))
            .is_none()
    );
}

#[test]
fn overlapping_primary_sources_are_rejected_and_header_only_sources_cannot_open_payloads() {
    let function = admitted(GgufLayout::FunctionHead);
    assert!(
        matches!(VoiceChatArtifactPlan::new(&config(), vec![function.clone(),function.clone()], None, BTreeSet::from([Component::FunctionHead])), Err(AdmissionError::DuplicateParameter(key)) if key == "stt_model.function_head.weight")
    );
    let plan = VoiceChatArtifactPlan::new(
        &config(),
        vec![function],
        None,
        BTreeSet::from([Component::FunctionHead]),
    )
    .unwrap();
    assert!(plan.sources().bind(1).is_err());
}

#[test]
fn quantized_storage_accounts_for_dense_conversion_without_allocating_the_parameter() {
    let source = admitted(GgufLayout::FunctionHead);
    let parameter = &source.parameters()["stt_model.function_head.weight"];
    // Published Q4_0: each 32-value block occupies 18 encoded bytes. The
    // canonical recipe receives F32 values, not a packed linear operator.
    let elements = 131072u64 * 4480;
    assert_eq!(parameter.stored_bytes, elements / 32 * 18);
    assert_eq!(parameter.output.dtype, RecipeDtype::F32);
    assert_eq!(parameter.output.shape, [131072, 4480]);
    assert_eq!(parameter.output.byte_len, elements * 4);
    // The portable decoder holds its f32 value vector while collecting the
    // canonical byte buffer, in addition to the encoded block storage.
    assert!(parameter.materialization_peak_bytes >= parameter.stored_bytes + elements * 8);
}

#[test]
fn bounded_integer_reads_preserve_large_integers_sign_and_requested_range() {
    let large = (1i64 << 53) + 1;
    let store = MemoryWeightStore::from_safetensors([
        (
            "wide".into(),
            safetensors::Dtype::I64,
            vec![3],
            [i64::MIN, large, i64::MAX]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        ),
        (
            "narrow".into(),
            safetensors::Dtype::I32,
            vec![4],
            [17i32, -9, i32::MIN, 23]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
        ),
    ])
    .unwrap();
    assert_eq!(
        read_integer_range(&store, "wide", 1..3).unwrap(),
        [large, i64::MAX]
    );
    assert_eq!(
        read_integer_range(&store, "narrow", 1..3).unwrap(),
        [-9, i64::from(i32::MIN)]
    );
    assert!(read_integer_range(&store, "wide", 2..4).is_err());
}

#[test]
fn retained_gguf_component_reads_exact_integer_ranges_and_cannot_access_other_roles() {
    use eredu_checkpoint::schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    };
    use eredu_gguf::{Endian, TensorInput, Writer, WriterOptions};

    // This small physical source exercises the integer-lease and restricted-view
    // seam. Full released tensor geometry is exercised by the header fixtures.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("controls.gguf");
    let large = (1i64 << 53) + 1;
    let wide: Vec<u8> = [i64::MIN, large, i64::MAX]
        .into_iter()
        .flat_map(i64::to_be_bytes)
        .collect();
    let narrow: Vec<u8> = [17i32, -9, i32::MIN, 23]
        .into_iter()
        .flat_map(i32::to_be_bytes)
        .collect();
    let unrelated = 11i64.to_be_bytes();
    Writer::new(WriterOptions {
        endian: Endian::Big,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(
        std::fs::File::create(&path).unwrap(),
        &BTreeMap::new(),
        &[
            TensorInput {
                name: "wide",
                dimensions: &[3],
                ggml_type: GgmlType::I64,
                data: &wide,
            },
            TensorInput {
                name: "narrow",
                dimensions: &[4],
                ggml_type: GgmlType::I32,
                data: &narrow,
            },
            TensorInput {
                name: "unrelated",
                dimensions: &[1],
                ggml_type: GgmlType::I64,
                data: &unrelated,
            },
        ],
    )
    .unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    let schema = GgufCheckpointPlan::new(
        "bounded integer control fixture",
        [
            ("wide", 3, GgmlType::I64),
            ("narrow", 4, GgmlType::I32),
            ("unrelated", 1, GgmlType::I64),
        ]
        .into_iter()
        .map(|(name, count, ty)| {
            GgufTensorConstraint::required(name, vec![count], GgufTypeConstraint::Exact(ty))
        })
        .collect(),
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .unwrap();
    let resolution = resolve_gguf_plan(&checkpoint, &schema).unwrap();
    let mapping = checkpoint
        .tensors()
        .flat_map(|tensor| {
            tensor
                .outputs()
                .iter()
                .map(move |output| eredu_gguf::TranslatedTensorLayout {
                    physical_name: tensor.descriptor().name.clone(),
                    original_name: output.name.clone(),
                    layout: output.clone(),
                })
        })
        .collect();
    let role = ArtifactSourceRole::Component("voicechat/controls");
    let declarations = ArtifactSourceDeclarations {
        artifacts: BTreeMap::from([(
            PhysicalArtifactId::Primary,
            PhysicalArtifactPreparation::Gguf {
                checkpoint,
                mapping,
                resolution,
            },
        )]),
        roles: BTreeMap::from([(
            role,
            BTreeMap::from([(
                PhysicalArtifactId::Primary,
                BTreeSet::from(["wide".into(), "narrow".into()]),
            )]),
        )]),
    };
    let sources = declarations.bind(1).unwrap();
    assert_eq!(sources.physical.len(), 1);
    let controls = &sources.roles[&role];
    assert_eq!(controls.source_keys(), ["narrow", "wide"]);
    assert_eq!(
        read_integer_range(controls.as_ref(), "wide", 1..3).unwrap(),
        [large, i64::MAX]
    );
    assert_eq!(
        read_integer_range(controls.as_ref(), "narrow", 1..3).unwrap(),
        [-9, i64::from(i32::MIN)]
    );
    assert!(matches!(
        controls.source_metadata("unrelated"),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    assert_eq!(
        sources.physical[&PhysicalArtifactId::Primary].source_keys(),
        ["narrow", "unrelated", "wide"]
    );
}

#[test]
fn voicechat_complete_gguf_uses_generic_configuration_and_discovery() {
    use crate::{GgufArchitecture, ModelKind, configuration::MODEL_CONFIGURATIONS};
    use eredu_core::artifact::{ArtifactMetadata, CheckpointMetadata, MetadataProvenance};
    let metadata = ArtifactMetadata {
        provenance: MetadataProvenance {
            source: "hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF".into(),
            revision: "89883a05a031557729771f94abb9998e4facdd45".into(),
        },
        checkpoint: CheckpointMetadata::Gguf {
            headers: vec![checkpoint_header(GgufLayout::Complete)],
            companions: Vec::new(),
        },
        sidecars: BTreeMap::new(),
    };
    let inspected =
        eredu_core::artifact::inspect_artifact_metadata(&metadata, &MODEL_CONFIGURATIONS).unwrap();
    let plan = inspected.architecture_plan();
    assert_eq!(plan.model_kind(), ModelKind::VoiceChat);
    assert_eq!(
        plan.gguf_plan().unwrap().architecture(),
        GgufArchitecture::VoiceChat
    );
    assert!(
        plan.architecture_descriptor()
            .nodes
            .iter()
            .any(|node| node.id == "transcription.joint")
    );
}
