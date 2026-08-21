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
    dflash_gguf_plan, translate_dflash_gguf_weight_name, DFlash, DFlashConfig, DFlashConfigError,
    DFlashContext, DFlashLayerContext,
};
pub use checkpoint::{
    expert_recipes, gguf_plan, projector_gguf_plan, safetensors_parameter_names, safetensors_plan,
    translate_projector_gguf_name, translate_text_gguf_name, ArtifactConfig,
};
pub use config::{
    ConfigError, DecoderConfig, VisionAttentionPolicy, VisionConfig, WeightConvention,
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
