use super::*;
use crate::decoder::Config;
use eredu_checkpoint::{
    StoredDtype,
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
};
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
struct HeaderEntry {
    shape: Vec<usize>,
    dtype: String,
}
struct Header(BTreeMap<String, HeaderEntry>);
impl SafetensorsCatalog for Header {
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        let entry = self.0.get(key).ok_or_else(|| format!("missing {key}"))?;
        let stored_dtype = match entry.dtype.as_str() {
            "BF16" => StoredDtype::BF16,
            "F16" => StoredDtype::F16,
            "F32" => StoredDtype::F32,
            "U32" => StoredDtype::U32,
            "F8_E4M3" => StoredDtype::F8E4M3,
            other => return Err(format!("unexpected released dtype {other}")),
        };
        Ok(CatalogTensorMetadata {
            shape: entry.shape.clone(),
            stored_dtype,
        })
    }
}

#[test]
fn every_pinned_official_header_admits_exact_text_and_vision_geometry() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/nemotron_labs_diffusion");
    for (release, hidden, layers, vocabulary) in [
        ("3B", 3072, 26, 131072),
        ("3B-Base", 3072, 26, 131072),
        ("8B", 4096, 34, 131072),
        ("8B-Base", 4096, 34, 131072),
        ("14B", 5120, 40, 131072),
        ("14B-Base", 5120, 40, 131072),
        ("VLM-8B", 4096, 34, 131073),
    ] {
        let config = FamilyConfig::from_hf_json(
            &std::fs::read(fixtures.join(format!("{release}-config.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(
            (
                config.hidden_size(),
                config.num_hidden_layers(),
                config.vocabulary_size()
            ),
            (hidden, layers, vocabulary)
        );
        assert_eq!(config.num_attention_heads() * config.head_dim(), 4096);
        assert_eq!(config.num_key_value_heads() * config.head_dim(), 1024);
        let mut header = Header(
            serde_json::from_slice(
                &std::fs::read(fixtures.join(format!("{release}-header.json"))).unwrap(),
            )
            .unwrap(),
        );
        let plan = safetensors_plan(&config).unwrap();
        eredu_checkpoint::validation::resolve_safetensors_plan(&header, &plan).unwrap();
        if hidden != 4096 {
            header
                .0
                .get_mut("encoder.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .shape = vec![hidden as usize, hidden as usize];
            assert!(
                eredu_checkpoint::validation::resolve_safetensors_plan(&header, &plan).is_err(),
                "head width must not be inferred from hidden width for {release}"
            );
        }
    }
}

#[test]
fn configuration_rejects_missing_head_width_and_conflicting_rotary_aliases() {
    let source: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/3B-config.json"
    ))
    .unwrap();
    let mut missing = source.clone();
    missing.as_object_mut().unwrap().remove("head_dim");
    assert!(FamilyConfig::from_hf_value(&missing).is_err());
    let mut conflicting = source;
    conflicting["rope_theta"] = 10000.into();
    assert!(FamilyConfig::from_hf_value(&conflicting).is_err());
}

#[test]
fn image_processor_matches_published_float_cubic_and_marker_sequences() {
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/pixtral-numeric.json"
    ))
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    let processor = processing::ImageProcessor::new(&config, 16).unwrap();
    for case in fixture["processor"].as_array().unwrap() {
        let pixels: Vec<u8> = case["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let width = case["width"].as_u64().unwrap() as u32;
        let height = case["height"].as_u64().unwrap() as u32;
        let prepared = if case["channels"].as_u64() == Some(4) {
            processor
                .prepare_rgba(&pixels, width, height, width as usize * 4)
                .unwrap()
        } else {
            processor
                .prepare(eredu_media::image::RgbImageView::packed(&pixels, width, height).unwrap())
                .unwrap()
        };
        assert_eq!(
            prepared.grid,
            (
                case["grid"][0].as_u64().unwrap() as u32,
                case["grid"][1].as_u64().unwrap() as u32
            )
        );
        assert_eq!(
            prepared.token_ids,
            case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        );
        let expected = case["patches"]["data"].as_array().unwrap();
        assert_eq!(prepared.patches.len(), expected.len());
        let max_error = prepared
            .patches
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (f64::from(*actual) - expected.as_f64().unwrap()).abs())
            .fold(0.0, f64::max);
        assert!(
            max_error < 2e-5,
            "published OpenCV floating cubic resize maximum error {max_error}"
        );
    }
}

#[test]
fn official_draft_adapter_has_separate_rank_parameters_and_exact_ancestry() {
    let config = FamilyConfig::from_hf_json(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/3B-config.json"
    ))
    .unwrap();
    let bytes =
        include_bytes!("../../tests/fixtures/nemotron_labs_diffusion/3B-adapter-config.json");
    let adapter =
        adapter::DraftAdapterConfig::from_json(bytes, "nvidia/Nemotron-Labs-Diffusion-3B").unwrap();
    let header = Header(
        serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/nemotron_labs_diffusion/3B-adapter-header.json"
        ))
        .unwrap(),
    );
    eredu_checkpoint::validation::resolve_safetensors_plan(
        &header,
        &adapter::safetensors_plan(&config, &adapter).unwrap(),
    )
    .unwrap();
    assert!(
        adapter::DraftAdapterConfig::from_json(bytes, "nvidia/Nemotron-Labs-Diffusion-8B").is_err()
    );
    let mut wrong: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    wrong["use_rslora"] = true.into();
    assert!(
        adapter::DraftAdapterConfig::from_json(
            &serde_json::to_vec(&wrong).unwrap(),
            "nvidia/Nemotron-Labs-Diffusion-3B"
        )
        .is_err()
    );
}

#[test]
fn mlx_affine_conversion_admits_independent_projection_widths_and_companions() {
    let config = FamilyConfig::from_hf_json(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/mlx-3B-4bit-config.json"
    ))
    .unwrap();
    let mut header = Header(
        serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/nemotron_labs_diffusion/mlx-3B-4bit-header.json"
        ))
        .unwrap(),
    );
    let plan = safetensors_plan(&config).unwrap();
    eredu_checkpoint::validation::resolve_safetensors_plan(&header, &plan).unwrap();
    header
        .0
        .remove("language_model.model.layers.0.self_attn.o_proj.scales");
    assert!(
        eredu_checkpoint::validation::resolve_safetensors_plan(&header, &plan).is_err(),
        "packed independent-width projections require their exact affine companions"
    );
}

#[test]
fn tensor_fp8_header_preserves_weight_only_semantics_and_requires_scalar_companions() {
    let source: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/FP8-W8A16-config.json"
    ))
    .unwrap();
    let config = FamilyConfig::from_hf_value(&source).unwrap();
    assert!(config.tensor_fp8());
    let plan = safetensors_plan(&config).unwrap();
    let mut header = Header(
        serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/nemotron_labs_diffusion/FP8-W8A16-header.json"
        ))
        .unwrap(),
    );
    eredu_checkpoint::validation::resolve_safetensors_plan(&header, &plan).unwrap();
    use eredu_core::{
        ModelConfigurationResolver,
        checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype},
    };
    let catalog = TensorCatalog::new(header.0.iter().map(|(name, tensor)| TensorDescriptor {
        name: name.clone(),
        shape: tensor.shape.clone(),
        storage: None,
        dtype: match tensor.dtype.as_str() {
            "BF16" => TensorDtype::Bf16,
            "F32" => TensorDtype::F32,
            other => TensorDtype::Encoded(other.into()),
        },
    }))
    .unwrap();
    let resolved = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&source)
        .unwrap();
    let dtype = crate::preparation::prepared_safetensors_floating_state_dtype_source(
        resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap(),
        &catalog,
    )
    .unwrap();
    assert_eq!(
        dtype.dtype(),
        &TensorDtype::F32,
        "bounded FP8 decoding produces F32 cache projections despite BF16 embeddings"
    );
    assert_eq!(
        dtype.checkpoint_tensor(),
        "encoder.layers.0.self_attn.q_proj.weight_scale"
    );
    header
        .0
        .get_mut("encoder.layers.0.self_attn.q_proj.weight_scale")
        .unwrap()
        .shape = vec![1];
    assert!(eredu_checkpoint::validation::resolve_safetensors_plan(&header, &plan).is_err());
    let mut activation_quantized = source.clone();
    activation_quantized["quantization_config"]["config_groups"]["group_0"]["input_activations"] =
        serde_json::json!({"num_bits":8,"type":"float","dynamic":false});
    assert!(FamilyConfig::from_hf_value(&activation_quantized).is_err());
    let mut wrong_exclusion = source;
    wrong_exclusion["quantization_config"]["ignore"][0] = "encoder.layers.0".into();
    assert!(FamilyConfig::from_hf_value(&wrong_exclusion).is_err());
}

#[test]
fn discovery_expands_pixtral_spatial_projector_and_conditional_draft_parameters() {
    use eredu_core::{ArchitectureNodeKind, DescriptionCompleteness, ModelConfigurationResolver};
    let value: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/VLM-8B-config.json"
    ))
    .unwrap();
    let resolved = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&value)
        .unwrap();
    let descriptor = resolved.architecture_plan().architecture_descriptor();
    assert_eq!(descriptor.completeness, DescriptionCompleteness::Complete);
    let vision = descriptor
        .layer_groups
        .iter()
        .find(|group| group.id == "vision")
        .unwrap();
    assert_eq!(vision.physical_layer_count, 24);
    let attention = descriptor
        .node("vision.layers.23.attention")
        .unwrap()
        .attention
        .as_ref()
        .unwrap();
    assert_eq!(
        (
            attention.query_heads,
            attention.key_head_dimension,
            attention.causal,
            attention.recurrent
        ),
        (Some(16), Some(64), Some(false), Some(false))
    );
    assert_eq!(
        descriptor.node("vision.projector.merge").unwrap().kind,
        ArchitectureNodeKind::Projector
    );
    assert!(
        descriptor
            .parameter_groups
            .iter()
            .any(|group| group.canonical_prefix
                == "model.multi_modal_projector.patch_merger.merging_layer")
    );
    assert!(
        descriptor
            .edges
            .iter()
            .any(|edge| edge.from == "vision.projector" && edge.to == "assembly")
    );
    assert!(
        descriptor
            .observations
            .points
            .iter()
            .all(|point| !point.path.starts_with("vision.layers.")),
        "logical discovery does not promise absent capture hooks"
    );
    let text: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/3B-config.json"
    ))
    .unwrap();
    let resolved = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&text)
        .unwrap();
    let adapter = adapter::DraftAdapterConfig::from_json(
        include_bytes!("../../tests/fixtures/nemotron_labs_diffusion/3B-adapter-config.json"),
        "nvidia/Nemotron-Labs-Diffusion-3B",
    )
    .unwrap();
    let plan = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .clone()
        .with_draft_adapter(adapter)
        .unwrap();
    let descriptor =
        crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(plan)
            .architecture_descriptor();
    assert_eq!(
        descriptor
            .node("draft_adapter.layers.25.o_proj.down")
            .unwrap()
            .output_axes
            .as_ref()
            .unwrap()[2]
            .dimension,
        eredu_core::SymbolicDimension::Known(128)
    );
    assert_eq!(descriptor.layer_groups[0].physical_layer_count, 26);
    assert_eq!(
        descriptor
            .node("decoder.layers.0.attention")
            .unwrap()
            .attention
            .as_ref()
            .unwrap()
            .causal,
        None
    );
}

#[test]
fn adapter_admission_rejects_changed_sidecars_and_header_preserving_weight_replacement() {
    use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/pixtral-numeric.json"
    ))
    .unwrap();
    let config = FamilyConfig::from_hf_value(&fixture["config"]).unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/nemotron_labs_diffusion/3B-adapter-config.json"
    ))
    .unwrap();
    json["r"] = 2.into();
    json["lora_alpha"] = 4.into();
    let bytes = serde_json::to_vec(&json).unwrap();
    let root = tempfile::tempdir().unwrap();
    let config_path = root.path().join("adapter_config.json");
    let weight_path = root.path().join("adapter_model.safetensors");
    std::fs::write(&config_path, &bytes).unwrap();
    let adapter =
        adapter::DraftAdapterConfig::from_json(&bytes, "nvidia/Nemotron-Labs-Diffusion-3B")
            .unwrap();
    let schema = adapter::safetensors_plan(&config, &adapter).unwrap();
    let values = schema
        .common_tensors
        .iter()
        .map(|tensor| {
            (
                tensor.key.clone(),
                tensor.shape.clone(),
                (0..tensor.shape.iter().product::<usize>())
                    .flat_map(|_| 0.25f32.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let write = |path: &std::path::Path| {
        serialize_to_file(
            values.iter().map(|(key, shape, bytes)| {
                (
                    key.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            }),
            None,
            path,
        )
        .unwrap()
    };
    write(&weight_path);
    let source =
        eredu_core::DraftAdapterSource::new(root.path(), "nvidia/Nemotron-Labs-Diffusion-3B");
    let admitted = adapter::AdmittedDraftAdapter::admit(&config, &source).unwrap();
    admitted.verify_configuration().unwrap();
    json["lora_alpha"] = 8.into();
    std::fs::write(&config_path, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(admitted.verify_configuration().is_err());
    std::fs::write(&config_path, &bytes).unwrap();
    let admitted = adapter::AdmittedDraftAdapter::admit(&config, &source).unwrap();
    let replacement = root.path().join("replacement.safetensors");
    write(&replacement);
    std::fs::rename(replacement, weight_path).unwrap();
    assert!(
        admitted.verify_configuration().is_err(),
        "identical tensor geometry does not confer authority on a replacement file"
    );
}
