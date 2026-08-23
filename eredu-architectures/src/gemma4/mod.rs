//! Neutral Gemma 4 family configuration and checkpoint policy.

pub mod assistant;
pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod family;
pub mod graph;
pub mod ingress;
pub mod media;
pub mod model;
pub mod parallel;
pub mod projector;
pub mod text;
pub mod vision;

pub use assistant::{
    assistant_gguf_plan, translate_assistant_gguf_weight_name, Assistant, AssistantCompatibility,
    AssistantCompatibilityError, AssistantConfig, AssistantConfigError, AssistantOutput,
    AssistantState,
};
pub use audio::{AudioConfig, AudioConfigError, AudioInput, AudioLayer, AudioStatic, AudioTower};
pub use checkpoint::{
    expert_recipes, gguf_plan, mmproj_gguf_plan, safetensors_plan, translate_gguf_weight_name,
    Gemma4ArtifactConfig,
};
pub use config::{ConfigError, FeedForwardPolicy, GgufTensorCatalog, LayerPolicy, ModelArgs};
pub use family::{FamilyConfig, FamilyConfigError};
pub use graph::{component_graph, state_layout, ComponentOptions};
pub use ingress::{
    AudioIngressBatchPlan, AudioIngressPartPlan, IngressPlanError, VisionIngressBatchPlan,
    VisionIngressPartPlan, VISION_ATTENTION_INVALID_LOGIT,
};
pub use media::ModalityProjector;
pub use model::{
    DecoderInputPart, ForwardContext, LayeredModel, ModelInput, StaticModules, TextBoundary,
    TextBoundarySchema, Unit,
};
pub use parallel::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    local_block_args, local_geometry, modality_projection_parameter_groups,
    static_parameter_groups, vision_layer_parameter_groups, vision_static_parameter_groups,
    LocalGeometry,
};
pub use projector::{
    family_from_gguf_metadata, translate_mmproj_weight_name, validate_projector_identity,
};
pub use text::{Attention, AttentionInput, BlockInput, DenseBlock, SharedAttentionStates};
pub use vision::{
    ClippedLinear, PatchEmbedder, VisionConfig, VisionConfigError, VisionInput, VisionLayer,
    VisionState, VisionStatic, VisionTower,
};

/// Declares Gemma 4 cache identity independently of concrete state storage.
pub fn state_identity(
    args: &FamilyConfig,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, FamilyConfigError> {
    args.validate()?;
    topology
        .validate()
        .map_err(|error| FamilyConfigError::Invalid(error.to_string()))?;
    let layer_count = args.text.num_hidden_layers();
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| FamilyConfigError::Invalid("Gemma 4 owned state range overflowed".into()))?;
    if global_layer_end > layer_count {
        return Err(FamilyConfigError::Invalid(format!(
            "Gemma 4 owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(eredu_runtime::ModelStateIdentity {
        model_family: "gemma4".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        topology,
    })
}
