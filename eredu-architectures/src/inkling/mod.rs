//! Neutral Inkling multimodal decoder family.

pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod graph;
pub mod model;
pub mod mtp;
pub mod parallel;
pub mod text;
pub mod vision;

pub use audio::{AudioInput, AudioTower};
pub use checkpoint::{
    dense_w13_recipes, expert_w13_recipe, gguf_plan, mmproj_gguf_plan, safetensors_aliases,
    safetensors_plan, safetensors_recipes, translate_gguf_weight_name,
    translate_gguf_weight_name_for_model, translate_mmproj_weight_name, DenseW13Recipes,
    ParameterAlias,
};
pub use config::{
    AudioConfig, ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs, MtpConfig, TextArgs,
    VisionConfig,
};
pub use graph::{component_graph, mtp_state_layout, parallel_state_layout, state_layout};
pub use model::{
    state_identity, DecoderInputPart, ForwardContext, LayeredModel, ModelInput, PartitionMtpOutput,
    StaticModules, TextPartitionInput, Unit,
};
pub use mtp::{MtpDepth, MtpModel, MtpOutput};
pub use parallel::{
    layer_parameter_groups, local_geometry, local_text_args, mtp_parameter_groups,
    static_parameter_groups, vision_layer_parameter_groups, LocalGeometry,
};
pub use text::{
    convolution_history_shape, Attention, ConvolutionState, DecoderLayer, FeedForward, LayerState,
    TextModel,
};
pub use vision::{VisionLayer, VisionStatic, VisionTower};
