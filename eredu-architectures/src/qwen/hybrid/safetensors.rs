//! Admission and canonical recipes for the converted MLX-VLM checkpoint layout.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    schema::SafetensorsCheckpointPlan,
    store::{CheckpointSource, TensorSelection, WeightStoreBackend},
    validation::SafetensorsCatalog,
};

use super::{composite_safetensors_plan, HybridVariant, ParsedHybridConfig};

/// Selects a physical layout from the catalog, before execution capabilities or
/// prediction extensions are derived. Official HF schemas remain strict.
pub(crate) fn catalog_plan(
    config: &ParsedHybridConfig,
    catalog: &impl SafetensorsCatalog,
) -> Result<(ParsedHybridConfig, SafetensorsCheckpointPlan), String> {
    let keys = catalog.keys();
    let converted = config.text.variant != HybridVariant::Qwen3Next
        && keys
            .iter()
            .any(|key| key.starts_with("language_model.model."));
    let mut config = config.clone();
    if converted {
        if keys.iter().any(|key| key.starts_with("model.")) {
            return Err("Qwen checkpoint mixes official and MLX-VLM tensor namespaces".into());
        }
        // MLX-VLM releases retain the upstream MTP config but omit the entire
        // optional draft. A partial draft must still fail strict admission.
        if !keys
            .iter()
            .any(|key| key.split('.').any(|part| part == "mtp"))
        {
            config.text.mtp_num_hidden_layers = 0;
        }
    }
    let mut plan = composite_safetensors_plan(&config)?;
    if converted {
        for tensor in plan
            .common_tensors
            .iter_mut()
            .chain(plan.layout_groups.iter_mut().flat_map(|group| {
                group
                    .variants
                    .iter_mut()
                    .flat_map(|variant| variant.tensors.iter_mut())
            }))
        {
            if let Some(physical) = physical_name(&tensor.key) {
                tensor.aliases = vec![physical.clone()];
                let mut shape = tensor.shape.clone();
                if tensor.key.ends_with(".linear_attn.conv1d.weight") {
                    shape.swap(1, 2);
                } else if tensor.key == "model.visual.patch_embed.proj.weight" {
                    shape = vec![shape[0], shape[2], shape[3], shape[4], shape[1]];
                }
                if shape != tensor.shape {
                    if let Ok(metadata) = catalog.metadata(&physical) {
                        if metadata.shape != shape {
                            return Err(format!(
                                "MLX-VLM tensor {physical:?} expected shape {shape:?}, got {:?}",
                                metadata.shape
                            ));
                        }
                    }
                    // Preserve canonical geometry for materialization sizing;
                    // only the selected source has the alternate axis order.
                    tensor.alternate_shapes = vec![shape];
                }
            }
        }
    }
    Ok((config, plan))
}

fn physical_name(canonical: &str) -> Option<String> {
    if let Some(rest) = canonical.strip_prefix("model.visual.") {
        Some(format!("vision_tower.{rest}"))
    } else if canonical.starts_with("model.") || canonical.starts_with("lm_head.") {
        Some(format!("language_model.{canonical}"))
    } else {
        None
    }
}

fn canonical_name(physical: &str) -> Option<String> {
    if let Some(rest) = physical.strip_prefix("vision_tower.") {
        Some(format!("model.visual.{rest}"))
    } else {
        physical
            .strip_prefix("language_model.")
            .filter(|rest| rest.starts_with("model.") || rest.starts_with("lm_head."))
            .map(str::to_owned)
    }
}

/// These recipes consume exact admitted sources; they never reinterpret GGUF.
/// Filtering by logical owner keeps layerwise and partitioned loads bounded.
pub(super) fn recipes(
    store: &dyn CheckpointSource,
    owns: impl Fn(&str) -> bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    if store
        .source_diagnostics()
        .map_err(|error| error.to_string())?
        .backend
        == WeightStoreBackend::Gguf
    {
        return Ok(recipes);
    }
    for physical in store.source_keys() {
        let Some(canonical) = canonical_name(&physical) else {
            continue;
        };
        if !owns(&canonical) {
            continue;
        }
        let source = Box::new(DerivedWeightRecipe::source(physical, TensorSelection::Full));
        let recipe = if canonical.ends_with(".linear_attn.conv1d.weight") {
            DerivedWeightRecipe::Transpose {
                input: source,
                axes: vec![0, 2, 1],
            }
        } else if canonical == "model.visual.patch_embed.proj.weight" {
            DerivedWeightRecipe::Transpose {
                input: source,
                axes: vec![0, 4, 1, 2, 3],
            }
        } else if !canonical.starts_with("model.visual.")
            && (canonical == "model.norm.weight"
                || [
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".self_attn.q_norm.weight",
                    ".self_attn.k_norm.weight",
                ]
                .iter()
                .any(|suffix| canonical.ends_with(suffix)))
        {
            DerivedWeightRecipe::SubtractOne { input: source }
        } else {
            // Unchanged matrices and their companions retain native encoding
            // through the schema's ordinary physical-alias binding.
            continue;
        };
        recipes.insert(canonical, recipe);
    }
    Ok(recipes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{
        validation::{resolve_safetensors_plan, CatalogTensorMetadata},
        StoredDtype,
    };

    struct Catalog(BTreeMap<String, CatalogTensorMetadata>);

    impl SafetensorsCatalog for Catalog {
        fn keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| format!("missing {key}"))
        }
    }

    fn released() -> (ParsedHybridConfig, Catalog) {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/configs/qwen3.5-0.8b-mlx-87e768fb.json"
        ))
        .unwrap();
        let config = super::super::model_args_from_config_value(&fixture["config"]).unwrap();
        let catalog = Catalog(
            fixture["tensors"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| {
                    let stored_dtype = match value["dtype"].as_str().unwrap() {
                        "BF16" => StoredDtype::BF16,
                        "F32" => StoredDtype::F32,
                        "U32" => StoredDtype::U32,
                        dtype => panic!("unexpected released dtype {dtype}"),
                    };
                    (
                        key.clone(),
                        CatalogTensorMetadata {
                            shape: serde_json::from_value(value["shape"].clone()).unwrap(),
                            stored_dtype,
                        },
                    )
                })
                .collect(),
        );
        (config, catalog)
    }

    #[test]
    fn released_mlx_vlm_catalog_admits_all_847_tensors_without_phantom_mtp() {
        let (config, catalog) = released();
        assert_eq!(config.text.mtp_num_hidden_layers, 1);
        let (selected, plan) = catalog_plan(&config, &catalog).unwrap();
        assert_eq!(selected.text.mtp_num_hidden_layers, 0);
        let resolved = resolve_safetensors_plan(&catalog, &plan).unwrap();
        assert_eq!(resolved.source_keys().len(), 847);
        assert!(resolved.unclaimed_keys().is_empty());
        let conv = plan
            .common_tensors
            .iter()
            .find(|t| t.key == "model.layers.0.linear_attn.conv1d.weight")
            .unwrap();
        assert_eq!(conv.shape, [6144, 1, 4]);
        assert_eq!(conv.alternate_shapes, [vec![6144, 4, 1]]);
    }

    #[test]
    fn converted_checkpoint_rejects_partial_mtp_missing_companions_and_mixed_layouts() {
        let (config, mut catalog) = released();
        catalog.0.insert(
            "mtp.norm.weight".into(),
            CatalogTensorMetadata {
                shape: vec![1024],
                stored_dtype: StoredDtype::BF16,
            },
        );
        let (selected, plan) = catalog_plan(&config, &catalog).unwrap();
        assert_eq!(selected.text.mtp_num_hidden_layers, 1);
        assert!(resolve_safetensors_plan(&catalog, &plan).is_err());
        catalog.0.remove("mtp.norm.weight");
        let scales = catalog
            .0
            .remove("language_model.model.embed_tokens.scales")
            .unwrap();
        let (_, plan) = catalog_plan(&config, &catalog).unwrap();
        assert!(resolve_safetensors_plan(&catalog, &plan).is_err());
        catalog.0.insert("model.embed_tokens.scales".into(), scales);
        assert!(catalog_plan(&config, &catalog).is_err());
    }

    #[test]
    fn converted_namespace_does_not_accept_unconverted_convolution_geometry() {
        let (config, mut catalog) = released();
        catalog
            .0
            .get_mut("language_model.model.layers.0.linear_attn.conv1d.weight")
            .unwrap()
            .shape = vec![6144, 1, 4];
        assert!(catalog_plan(&config, &catalog)
            .unwrap_err()
            .contains("expected shape"));
    }

    #[test]
    fn official_schema_and_declared_mtp_remain_required() {
        let (config, _) = released();
        let official = composite_safetensors_plan(&config).unwrap();
        let catalog = Catalog(BTreeMap::new());
        let (selected, plan) = catalog_plan(&config, &catalog).unwrap();
        assert_eq!(selected.text.mtp_num_hidden_layers, 1);
        assert_eq!(plan, official);
        assert!(resolve_safetensors_plan(&catalog, &plan).is_err());
    }
}
