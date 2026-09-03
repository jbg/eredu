//! Composite SafeTensors contract for Qwen3-VL text and shared vision.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{GgufCheckpointPlan, SafetensorsCheckpointPlan},
    store::TensorSelection,
    WeightQuantization,
};

use super::ModelArgs;
use crate::qwen::{self, vision};

/// Derives a Qwen3-VL configuration whose text and aligned vision matrix
/// formats reflect load-time quantization.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
    target.text = qwen::load_time_quantization(&args.text, quantization)?;
    target.vision.apply_load_time_quantization(quantization);
    target
        .vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical text and projector checkpoint formats to a complete
/// Qwen3-VL configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut text_formats: HashMap<String, WeightQuantization>,
    vision_formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
    normalize_text_weight_formats(&args.text, &mut text_formats);
    target.text = qwen::with_checkpoint_formats(&args.text, text_formats)?;
    target.vision.linear_formats = vision_formats
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect();
    target
        .vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    Ok(target)
}

/// Builds the projector checkpoint contract from an admitted Qwen3-VL composite.
///
/// Family mode and decoder/projector width compatibility are revalidated here
/// so a backend cannot construct a projector plan from unrelated parts.
pub fn projector_gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    args.vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    if args.vision.out_hidden_size != args.text.hidden_size {
        return Err(format!(
            "Qwen3-VL projector output {} does not match text hidden size {}",
            args.vision.out_hidden_size, args.text.hidden_size
        ));
    }
    vision::gguf_plan(&args.vision, args.text.hidden_size)
}

/// Builds one strict catalog for the ordinary Qwen decoder plus shared vision tower.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    args.vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    let mut text = qwen::safetensors_plan_with_root(&args.text, &args.text.parameter_root, true)?;
    let vision = vision::safetensors_plan(&args.vision, "model.visual")?;
    text.common_tensors.extend(vision.common_tensors);
    text.layout_groups.extend(vision.layout_groups);
    let policy = text.catalog_policy;
    SafetensorsCheckpointPlan::new(
        format!("{} composite SafeTensors", args.model_type),
        text.common_tensors,
        text.layout_groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

/// Translates one Qwen3-VL text GGUF name into the composite runtime namespace.
pub fn translate_text_gguf_weight_name(name: &str, is_moe: bool) -> String {
    let name = qwen::translate_gguf_weight_name(name, is_moe);
    name.strip_prefix("model.")
        .map(|name| format!("model.language_model.{name}"))
        .unwrap_or(name)
}

/// Translates one Qwen3-VL projector GGUF name, including split patch weights.
pub fn translate_vision_gguf_weight_name(name: &str, deepstack: &[i32]) -> String {
    match name {
        "v.patch_embd.weight" => "model.visual.patch_embed.proj.weight.0".into(),
        "v.patch_embd.weight.1" => "model.visual.patch_embed.proj.weight.1".into(),
        _ => vision::translate_gguf_weight_name(name, deepstack),
    }
}

/// Rehomes split expert GGUF formats onto their canonical fused runtime weights.
pub fn normalize_text_weight_formats<V>(args: &qwen::ModelArgs, formats: &mut HashMap<String, V>) {
    if !args.is_moe() {
        return;
    }
    for layer in 0..args.num_hidden_layers {
        let root = format!("model.language_model.layers.{layer}.mlp.experts");
        if let Some(format) = formats.remove(&format!("{root}.gate_proj.weight")) {
            formats.remove(&format!("{root}.up_proj.weight"));
            formats.insert(format!("{root}.gate_up_proj"), format);
        }
    }
}

/// Returns all derived recipes owned by the composite static modules.
pub fn static_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
) -> BTreeMap<String, DerivedWeightRecipe> {
    let first = "model.visual.patch_embed.proj.weight.0";
    let second = "model.visual.patch_embed.proj.weight.1";
    if catalog.tensor_metadata(first).is_err() || catalog.tensor_metadata(second).is_err() {
        return BTreeMap::new();
    }
    BTreeMap::from([(
        "model.visual.patch_embed.proj.weight".into(),
        DerivedWeightRecipe::Stack {
            axis: 2,
            inputs: vec![
                DerivedWeightRecipe::source(first, TensorSelection::Full),
                DerivedWeightRecipe::source(second, TensorSelection::Full),
            ],
        },
    )])
}

/// Returns all derived recipes for one flat vision/text execution unit.
pub fn unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    flat: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let vision_layers = args.vision.layer_count();
    if flat < vision_layers || !args.text.is_moe() {
        return Ok(BTreeMap::new());
    }
    let resolved = qwen::expert_recipes(catalog, &args.text, flat - vision_layers)?;
    Ok(BTreeMap::from([
        (resolved.target_gate_up, resolved.gate_up),
        (resolved.target_down, resolved.down),
    ]))
}

/// Returns rank-local recipes for one flat Qwen3-VL text execution unit.
pub fn rank_local_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    flat: usize,
    group_indices: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let vision_layers = args.vision.layer_count();
    if flat < vision_layers {
        return Err(format!(
            "Qwen3-VL unit {flat} is a vision unit and has no expert bank"
        ));
    }
    qwen::rank_local_expert_recipes(catalog, &args.text, flat - vision_layers, group_indices)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eredu_checkpoint::{
        recipe::RecipeCatalog,
        store::{StoreError, TensorMetadata},
        StoredDtype,
    };
    use serde_json::json;

    use super::*;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 2,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F16,
            backing_shard: None,
        }
    }

    fn model_args() -> ModelArgs {
        crate::qwen::vl::model_args_from_config_value(&json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap()
    }

    #[test]
    fn composite_plan_contains_one_text_and_one_shared_vision_namespace() {
        let args = model_args();
        let plan = safetensors_plan(&args).unwrap();
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "model.language_model.embed_tokens.weight"));
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "model.visual.patch_embed.proj.weight"));
        assert!(!plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key.starts_with("patch_embed.")));
    }

    #[test]
    fn text_gguf_translation_uses_the_composite_language_namespace() {
        assert_eq!(
            translate_text_gguf_weight_name("token_embd.weight", false),
            "model.language_model.embed_tokens.weight"
        );
        assert_eq!(
            translate_text_gguf_weight_name("blk.0.attn_q.weight", false),
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn projector_plan_owns_family_mode_and_width_compatibility() {
        let admitted = model_args();
        assert!(projector_gguf_plan(&admitted).is_ok());

        let mut wrong_width = admitted.clone();
        wrong_width.vision.out_hidden_size = 16;
        assert!(projector_gguf_plan(&wrong_width)
            .unwrap_err()
            .contains("does not match text hidden size"));

        let mut wrong_mode = admitted;
        wrong_mode.vision.mode = vision::VisionMode::WindowScheduled;
        assert!(projector_gguf_plan(&wrong_mode)
            .unwrap_err()
            .contains("does not match required"));
    }

    #[test]
    fn load_time_quantization_normalizes_text_and_vision_formats() {
        let args = crate::qwen::vl::model_args_from_config_value(&json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":32,"intermediate_size":64,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap();
        let quantization =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());

        let target = load_time_quantization(&args, quantization).unwrap();

        assert_eq!(target.text.quantization, Some(quantization));
        assert!(!target.vision.linear_formats.is_empty());
        target.text.validate().unwrap();
        target
            .vision
            .validate_for(vision::VisionMode::DeepStack)
            .unwrap();
    }

    #[test]
    fn neutral_catalog_owns_split_patch_translation_and_stacking() {
        assert_eq!(
            translate_vision_gguf_weight_name("v.patch_embd.weight", &[0]),
            "model.visual.patch_embed.proj.weight.0"
        );
        let first = "model.visual.patch_embed.proj.weight.0";
        let second = "model.visual.patch_embed.proj.weight.1";
        let catalog = Catalog(BTreeMap::from([
            (first.into(), metadata(first, vec![2, 3, 4])),
            (second.into(), metadata(second, vec![2, 3, 4])),
        ]));
        let recipes = static_recipes(&catalog);
        assert!(matches!(
            recipes.get("model.visual.patch_embed.proj.weight"),
            Some(DerivedWeightRecipe::Stack { axis: 2, inputs }) if inputs.len() == 2
        ));
    }

    #[test]
    fn unit_recipes_use_the_composite_architecture_ordinal() {
        let args = crate::qwen::vl::model_args_from_config_value(&json!({
            "model_type":"qwen3_vl_moe", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_moe_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":0, "moe_intermediate_size":16,
                "num_experts":4, "num_experts_per_tok":2, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap();
        let root = "model.language_model.layers.0.mlp.experts";
        let catalog = Catalog(BTreeMap::from([
            (
                format!("{root}.gate_up_proj"),
                metadata(&format!("{root}.gate_up_proj"), vec![4, 32, 32]),
            ),
            (
                format!("{root}.down_proj"),
                metadata(&format!("{root}.down_proj"), vec![4, 32, 16]),
            ),
        ]));

        assert!(unit_recipes(&catalog, &args, 0).unwrap().is_empty());
        let decoder = unit_recipes(&catalog, &args, 1).unwrap();
        assert!(decoder.contains_key(&format!("{root}.gate_up_proj")));
        assert!(decoder.contains_key(&format!("{root}.down_proj")));
        let local = rank_local_unit_recipes(&catalog, &args, 1, &[3, 1]).unwrap();
        assert_eq!(
            local[&format!("{root}.gate_up_proj")]
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[2, 32, 32]
        );
    }
}
