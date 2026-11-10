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

/// Projects text and aligned vision storage through retained composite declarations.
pub fn load_time_quantization(
    declaration: &super::Declaration,
    quantization: WeightQuantization,
) -> Result<super::Declaration, String> {
    let mut target = declaration.args().clone();
    target.text = qwen::load_time_quantization(&target.text, quantization)?;
    let vision = declaration
        .vision_declaration()
        .with_quantization(quantization)
        .map_err(|error| error.to_string())?;
    target.vision = vision.config().clone();
    super::Declaration::from_components(
        target.clone(),
        std::sync::Arc::new(
            declaration
                .text_declaration()
                .with_formats(target.text)
                .map_err(|error| error.to_string())?,
        ),
        std::sync::Arc::new(vision),
    )
    .map_err(|error| error.to_string())
}

/// Applies canonical text and projector checkpoint formats to a complete
/// Qwen3-VL configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    text_formats: HashMap<String, WeightQuantization>,
    vision_formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
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
pub fn projector_gguf_plan(declaration: &super::Declaration) -> Result<GgufCheckpointPlan, String> {
    let args = declaration.args();
    args.vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    if args.vision.out_hidden_size != args.text.hidden_size {
        return Err(format!(
            "Qwen3-VL projector output {} does not match text hidden size {}",
            args.vision.out_hidden_size, args.text.hidden_size
        ));
    }
    vision::gguf_plan(declaration.vision_declaration())
}

/// Builds one strict catalog for the ordinary Qwen decoder plus shared vision tower.
pub fn safetensors_plan(
    declaration: &super::Declaration,
) -> Result<SafetensorsCheckpointPlan, String> {
    let args = declaration.args();
    args.vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    let mut text = qwen::safetensors_plan(declaration.text_declaration(), true)?;
    let vision = vision::safetensors_plan(declaration.vision_declaration())?;
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

/// Projects text and vision source matrix geometry from the corresponding
/// construction declarations and their explicit source-layout extensions.
pub(crate) fn source_linear_shapes(
    declaration: &super::Declaration,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let args = declaration.args();
    args.vision
        .validate_for(vision::VisionMode::DeepStack)
        .map_err(|error| error.to_string())?;
    let mut shapes =
        super::super::checkpoint::source_linear_shapes(declaration.text_declaration())?;
    shapes.extend(vision::source_linear_shapes(
        declaration.vision_declaration(),
    )?);
    Ok(shapes)
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
    declaration: &super::Declaration,
    flat: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let vision_layers = args.vision.layer_count();
    if flat < vision_layers || !args.text.is_moe() {
        return Ok(BTreeMap::new());
    }
    let resolved = qwen::expert_recipes(
        catalog,
        declaration.text_declaration(),
        flat - vision_layers,
    )?;
    Ok(BTreeMap::from([
        (resolved.target_gate_up, resolved.gate_up),
        (resolved.target_down, resolved.down),
    ]))
}

/// Returns rank-local recipes for one flat Qwen3-VL text execution unit.
pub fn rank_local_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    flat: usize,
    group_indices: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let vision_layers = args.vision.layer_count();
    if flat < vision_layers {
        return Err(format!(
            "Qwen3-VL unit {flat} is a vision unit and has no expert bank"
        ));
    }
    qwen::rank_local_expert_recipes(
        catalog,
        declaration.text_declaration(),
        flat - vision_layers,
        group_indices,
    )
}
