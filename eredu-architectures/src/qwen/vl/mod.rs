//! Neutral Qwen3-VL configuration, state, and multimodal position policy.

mod checkpoint;
mod config;
mod model;
mod positions;

pub use checkpoint::safetensors_plan;
pub use config::{
    model_args_from_config_value, model_args_from_gguf_parts,
    prompt_cache_architecture_fingerprint, state_identity, state_layout,
    state_layout_with_key_value_heads, ModelArgs, VlConfigError,
};
pub use model::{
    ForwardContext, InputPart, LayeredModel, ModelInput, PipelinePrepared, PipelineVisionState,
    StaticModules, Unit,
};
pub use positions::{
    mrope_embeddings, mrope_values, multimodal_position_ids, position_ids_tensor, PositionPart,
};
