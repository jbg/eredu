//! Neutral Qwen3-VL configuration, state, and multimodal position policy.

mod checkpoint;
mod config;
mod model;
mod parallel;
mod positions;

pub use checkpoint::{
    load_time_quantization, normalize_text_weight_formats, projector_gguf_plan,
    rank_local_unit_recipes, safetensors_plan, static_recipes, translate_text_gguf_weight_name,
    translate_vision_gguf_weight_name, unit_recipes,
};
pub use config::{
    model_args_from_config_value, model_args_from_gguf_parts,
    prompt_cache_architecture_fingerprint, state_identity, state_layout,
    state_layout_with_key_value_heads, vision_config_from_gguf_catalog, ModelArgs, VlConfigError,
};
pub use model::{
    ForwardContext, InputPart, LayeredModel, ModelInput, PipelineBoundary, PipelineBoundarySchema,
    PipelinePartitionInput, PipelinePrepared, PipelineVisionState, StaticModules, Unit,
};
pub use parallel::{local_geometry, LocalGeometry};
pub use positions::{
    mrope_embeddings, mrope_values, multimodal_position_ids, position_ids_tensor, PositionPart,
};
