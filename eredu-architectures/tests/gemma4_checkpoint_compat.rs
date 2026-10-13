//! Admission of published MLX conversion metadata without tensor allocation.
use std::collections::BTreeMap;

use eredu_architectures::{gemma4, processor_plan::Gemma4ProcessorPlan};
use eredu_checkpoint::{
    validation::{resolve_safetensors_plan, CatalogTensorMetadata, SafetensorsCatalog},
    StoredDtype,
};

struct Catalog(BTreeMap<String, CatalogTensorMetadata>);

impl SafetensorsCatalog for Catalog {
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.0.get(key).cloned().ok_or_else(|| key.into())
    }
}

#[test]
fn gemma4_mlx_qat_admits_published_packing_and_retains_strict_validation() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gemma4_mlx_qat_header.json")).unwrap();
    let bytes = serde_json::to_vec(&fixture["config"]).unwrap();
    let family = gemma4::FamilyConfig::from_hf_json(&bytes).unwrap();
    assert!(Gemma4ProcessorPlan::from_hf_json(&bytes, None, None)
        .unwrap()
        .is_none());
    let plan = gemma4::safetensors_plan(&family).unwrap();
    let mut catalog = Catalog(
        fixture["headers"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, header)| {
                (
                    name.clone(),
                    CatalogTensorMetadata {
                        shape: serde_json::from_value(header["shape"].clone()).unwrap(),
                        stored_dtype: match header["dtype"].as_str().unwrap() {
                            "U32" => StoredDtype::U32,
                            "BF16" => StoredDtype::BF16,
                            other => panic!("unexpected fixture encoding {other}"),
                        },
                    },
                )
            })
            .collect(),
    );
    let resolved = resolve_safetensors_plan(&catalog, &plan).unwrap();
    assert!(resolved
        .source_keys()
        .contains("language_model.model.layers.0.mlp.down_proj.weight"));
    assert!(!resolved
        .source_keys()
        .contains("embed_vision.embedding_projection.weight"));
    // The QAT MLP uses 8-bit words, while attention retains the 4-bit default.
    assert_eq!(
        family
            .text
            .linear_format_for("model.layers.0.mlp.down_proj.weight")
            .weight_quantization()
            .unwrap()
            .bits(),
        8
    );
    assert_eq!(
        family
            .text
            .linear_format_for("model.language_model.layers.0.self_attn.q_proj.weight")
            .weight_quantization()
            .unwrap()
            .bits(),
        4
    );

    let scale = "language_model.model.layers.0.mlp.down_proj.scales";
    let saved = catalog.0.remove(scale).unwrap();
    assert!(resolve_safetensors_plan(&catalog, &plan).is_err());
    catalog.0.insert(scale.into(), saved);
    let weight = "language_model.model.layers.0.mlp.down_proj.weight";
    let saved = catalog.0[weight].clone();
    catalog.0.get_mut(weight).unwrap().shape[1] /= 2;
    assert!(resolve_safetensors_plan(&catalog, &plan).is_err());
    catalog.0.insert(weight.into(), saved.clone());
    catalog
        .0
        .insert("model.layers.0.mlp.down_proj.weight".into(), saved);
    assert!(resolve_safetensors_plan(&catalog, &plan).is_err());
}
