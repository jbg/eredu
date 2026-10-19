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
        crate::qwen::vision::temporal_patch_recipe([
            DerivedWeightRecipe::source(first, TensorSelection::Full),
            DerivedWeightRecipe::source(second, TensorSelection::Full),
        ]),
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
