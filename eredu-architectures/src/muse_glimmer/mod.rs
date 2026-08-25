//! Neutral Muse-Glimmer multimodal decoder family.

pub mod assistant;
pub mod checkpoint;
pub mod config;
pub mod graph;
pub mod model;
pub mod parallel;
pub mod text;
pub mod vision;

pub use assistant::{
    dflash_gguf_plan, translate_dflash_gguf_weight_name, DFlash, DFlashCompatibility,
    DFlashCompatibilityError, DFlashConfig, DFlashConfigError, DFlashContext, DFlashLayerContext,
};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    projector_gguf_plan, safetensors_parameter_names, safetensors_plan, safetensors_recipes,
    translate_projector_gguf_name, translate_text_gguf_name, ArtifactConfig,
};
pub use config::{
    ConfigError, DecoderConfig, GgufTensorCatalog, VisionAttentionPolicy, VisionConfig,
    WeightConvention,
};
pub use graph::{component_graph, state_layout};
pub use model::{
    DecoderInputPart, ForwardContext, LayeredModel, ModelInput, TextPartitionInput, Unit,
};
pub use parallel::{
    layer_parameter_groups, local_decoder_config, local_geometry, static_parameter_groups,
    vision_layer_parameter_groups, vision_parameter_groups, vision_static_parameter_groups,
    LocalGeometry,
};
pub use text::{
    Attention, CenteredRmsNorm, FeedForward, Mlp, SparseMoe, StaticModules, TransformerBlock,
};
pub use vision::{VisionBlock, VisionInput, VisionState, VisionStatic, VisionTower};

/// Declares Muse-Glimmer cache identity independently of concrete state storage.
pub fn state_identity(
    args: &DecoderConfig,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, ConfigError> {
    args.validate()?;
    topology
        .validate()
        .map_err(|error| ConfigError::Invalid(error.to_string()))?;
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ConfigError::Invalid("Muse-Glimmer layer count exceeds usize".into()))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| ConfigError::Invalid("Muse-Glimmer owned state range overflowed".into()))?;
    if global_layer_end > layer_count {
        return Err(ConfigError::Invalid(format!(
            "Muse-Glimmer owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(eredu_runtime::ModelStateIdentity {
        model_family: "muse_glimmer".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        topology,
    })
}
