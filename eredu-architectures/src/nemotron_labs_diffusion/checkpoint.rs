use crate::decoder::Config;
use eredu_checkpoint::schema::{
    CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
};
use eredu_checkpoint::StoredDtype;

use super::FamilyConfig;

/// Normalizes only the family's exact published shared-backbone namespaces.
pub fn canonical_parameter_name(source: &str) -> String {
    if let Some(rest) = source.strip_prefix("nemotron.gguf.pixtral.layers.") {
        if let Some((index, field)) = rest.split_once('.') {
            if index.parse::<usize>().is_ok() && matches!(field, "q_proj.weight" | "k_proj.weight")
            {
                return format!("model.vision_tower.transformer.layers.{index}.attention.{field}");
            }
        }
    }
    if let Some(name) = source.strip_prefix("base_model.model.encoder.layers.") {
        if let Some((layer, suffix)) = name.split_once(".self_attn.o_proj.") {
            if layer.parse::<usize>().is_ok() && matches!(suffix, "lora_A.weight" | "lora_B.weight")
            {
                return format!("draft_adapter.layers.{layer}.o_proj.{suffix}");
            }
        }
    }
    if let Some(name) = source.strip_prefix("language_model.model.") {
        return format!("model.{name}");
    }
    if let Some(name) = source.strip_prefix("language_model.diffusion_head.") {
        return format!("lm_head.{name}");
    }
    if let Some(name) = source.strip_prefix("encoder.") {
        format!("model.{name}")
    } else if let Some(name) = source.strip_prefix("diffusion_head.") {
        format!("lm_head.{name}")
    } else {
        source.to_owned()
    }
}

/// Exact source-only recipes normalize admitted physical names without a second
/// base model or a copy-producing layout transformation.
pub fn source_recipes(
    source: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> Result<std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String>
{
    let mut recipes = std::collections::BTreeMap::new();
    let mut names = std::collections::BTreeSet::new();
    let keys = source.source_keys();
    let mut conversion_inputs = std::collections::BTreeSet::new();
    for key in &keys {
        if source
            .source_metadata(key)
            .map_err(|e| e.to_string())?
            .stored_dtype
            == StoredDtype::F8E4M3
        {
            let prefix = key
                .strip_suffix(".weight")
                .ok_or_else(|| "FP8 source is not a projection weight".to_owned())?;
            conversion_inputs.insert(format!("{prefix}.weight_scale"));
        }
    }
    for key in keys {
        // Scales are dependencies of DecodeTensorFp8, not independently
        // normalized outputs or additional resident parameter aliases.
        if conversion_inputs.contains(&key) {
            continue;
        }
        let canonical = canonical_parameter_name(&key);
        if !names.insert(canonical.clone()) {
            return Err(format!("ambiguous shared-backbone parameter {canonical}"));
        }
        if key.starts_with("nemotron.gguf.pixtral.layers.") && canonical != key {
            use eredu_checkpoint::recipe::{DerivedWeightRecipe as Recipe, RecipeDtype};
            let metadata = source.source_metadata(&key).map_err(|e| e.to_string())?;
            if metadata.stored_dtype != StoredDtype::F16 || metadata.logical_shape != [1024, 1024] {
                return Err(
                    "Pixtral GGUF Q/K differs from admitted half-head conversion geometry".into(),
                );
            }
            let recipe = Recipe::Reshape {
                input: Box::new(Recipe::Transpose {
                    input: Box::new(Recipe::Reshape {
                        input: Box::new(Recipe::Cast {
                            input: Box::new(Recipe::source(
                                &key,
                                eredu_checkpoint::store::TensorSelection::Full,
                            )),
                            dtype: RecipeDtype::F32,
                        }),
                        shape: vec![16, 32, 2, 1024],
                    }),
                    axes: vec![0, 2, 1, 3],
                }),
                shape: vec![1024, 1024],
            };
            recipe.infer(source).map_err(|e| e.to_string())?;
            recipes.insert(canonical, recipe);
        } else if source
            .source_metadata(&key)
            .map_err(|e| e.to_string())?
            .stored_dtype
            == StoredDtype::F8E4M3
        {
            let prefix = key
                .strip_suffix(".weight")
                .ok_or_else(|| "FP8 source is not a projection weight".to_owned())?;
            let recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::DecodeTensorFp8 {
                weight: Box::new(eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                    &key,
                    eredu_checkpoint::store::TensorSelection::Full,
                )),
                scale: Box::new(eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                    format!("{prefix}.weight_scale"),
                    eredu_checkpoint::store::TensorSelection::Full,
                )),
            };
            recipe.infer(source).map_err(|e| e.to_string())?;
            recipes.insert(canonical, recipe);
        } else if canonical != key {
            recipes.insert(
                canonical,
                eredu_checkpoint::recipe::DerivedWeightRecipe::Source {
                    key,
                    selection: eredu_checkpoint::store::TensorSelection::Full,
                },
            );
        }
    }
    Ok(recipes)
}

pub(crate) fn module_source_recipes<T: 'static, M: eredu_nn::Parameterized<T>>(
    module: &M,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, String>
{
    struct Names(std::collections::BTreeSet<String>);
    impl<'a, T: 'a> eredu_nn::ParameterVisitor<'a, T> for Names {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadata, _: &'a T) {
            self.0.insert(metadata.id.to_string());
        }
    }
    let mut names = Names(std::collections::BTreeSet::new());
    module.visit_parameters(&mut names);
    Ok(source_recipes(source)?
        .into_iter()
        .filter(|(name, _)| names.0.contains(name))
        .collect())
}

/// Complete text tensor geometry, including independently encoded affine companions.
pub fn safetensors_plan(config: &FamilyConfig) -> Result<SafetensorsCheckpointPlan, String> {
    let dense =
        crate::llama::safetensors_plan(config.dense()).map_err(|error| error.to_string())?;
    let mut tensors = dense.common_tensors;
    if let Some(vision) = &config.vision {
        let width = vision.hidden_size as usize;
        let intermediate = vision.intermediate_size as usize;
        let text = config.hidden_size() as usize;
        let merged = width
            .checked_mul(config.spatial_merge_size as usize)
            .and_then(|x| x.checked_mul(config.spatial_merge_size as usize))
            .ok_or_else(|| "Pixtral merge width overflow".to_string())?;
        let mut add = |key: String, shape: Vec<usize>| {
            tensors.push(SafetensorsTensorConstraint::required(
                key,
                shape,
                StoredDtypeConstraint::OneOf(vec![
                    StoredDtype::F32,
                    StoredDtype::F16,
                    StoredDtype::BF16,
                ]),
            ));
        };
        for (key, shape) in [
            (
                "model.vision_tower.patch_conv.weight",
                vec![
                    width,
                    vision.num_channels as usize,
                    vision.patch_size as usize,
                    vision.patch_size as usize,
                ],
            ),
            ("model.vision_tower.ln_pre.weight", vec![width]),
            (
                "model.multi_modal_projector.patch_merger.merging_layer.weight",
                vec![width, merged],
            ),
            ("model.multi_modal_projector.norm.weight", vec![width]),
            (
                "model.multi_modal_projector.linear_1.weight",
                vec![text, width],
            ),
            (
                "model.multi_modal_projector.linear_2.weight",
                vec![text, text],
            ),
        ] {
            add(key.into(), shape);
        }
        for layer in 0..vision.num_hidden_layers {
            let prefix = format!("model.vision_tower.transformer.layers.{layer}");
            for field in ["attention_norm", "ffn_norm"] {
                add(format!("{prefix}.{field}.weight"), vec![width]);
            }
            for field in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                add(
                    format!("{prefix}.attention.{field}.weight"),
                    vec![width, width],
                );
            }
            for (field, shape) in [
                ("gate_proj", vec![intermediate, width]),
                ("up_proj", vec![intermediate, width]),
                ("down_proj", vec![width, intermediate]),
            ] {
                add(format!("{prefix}.feed_forward.{field}.weight"), shape);
            }
        }
    }
    if config.tensor_fp8() {
        let mut scales = Vec::new();
        for tensor in &mut tensors {
            if tensor.key.starts_with("model.layers.") && tensor.key.ends_with("_proj.weight") {
                tensor.dtype = StoredDtypeConstraint::Exact(StoredDtype::F8E4M3);
                let prefix = tensor
                    .key
                    .strip_suffix(".weight")
                    .expect("projection weight");
                let mut scale = SafetensorsTensorConstraint::required(
                    format!("{prefix}.weight_scale"),
                    vec![],
                    StoredDtypeConstraint::Exact(StoredDtype::F32),
                );
                // The bounded decode recipe consumes this admitted source
                // companion. It is not a separately bound model parameter.
                scale.role = eredu_checkpoint::schema::TensorRole::RecipeInput;
                scales.push(scale);
            }
        }
        tensors.extend(scales);
    }
    for tensor in &mut tensors {
        let source = if let Some(name) = tensor.key.strip_prefix("model.") {
            Some(format!("encoder.{name}"))
        } else {
            tensor
                .key
                .strip_prefix("lm_head.")
                .map(|name| format!("diffusion_head.{name}"))
        };
        if let Some(source) = source {
            tensor.aliases.push(source);
        }
        if let Some(name) = tensor.key.strip_prefix("model.") {
            tensor.aliases.push(format!("language_model.model.{name}"));
        } else if let Some(name) = tensor.key.strip_prefix("lm_head.") {
            tensor
                .aliases
                .push(format!("language_model.diffusion_head.{name}"));
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new(
        "Nemotron-Labs-Diffusion shared text SafeTensors",
        tensors,
        Vec::new(),
        policy,
    )
    .map_err(|error| error.to_string())
}
