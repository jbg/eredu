//! One backend-neutral hybrid decoder shared by Qwen3-Next and Qwen3.5.

mod block;
mod checkpoint;
mod conditional;
mod config;
mod linear_attention;
mod model;
mod mtp;
mod parallel;

pub use block::{expert_bank_spec, Block, FeedForward, SharedRoutedGatedProduct, TokenMixer};
pub use checkpoint::{
    conditional_load_time_quantization, expert_recipes, expert_residency_catalog, gguf_plan,
    load_time_quantization, qwen3_next_fused_recipes, safetensors_plan, static_recipes,
    translate_gguf_weight_name, translate_vision_gguf_weight_name, unit_recipes,
};
pub use conditional::{
    ConditionalForwardContext, ConditionalInput, ConditionalLayeredModel,
    ConditionalPartitionInput, ConditionalPipelineBoundary, ConditionalPipelineBoundarySchema,
    ConditionalPipelinePrepared, ConditionalPipelineVisionState, ConditionalStaticModules,
    ConditionalUnit,
};
pub use config::{
    fp8_block_row_widths, fused_projection_widths, model_args_from_config_value,
    model_args_from_gguf_catalog, prompt_cache_architecture_fingerprint, state_layout,
    state_layout_with_geometry, with_gguf_vision_projector, HybridConfig, HybridConfigError,
    HybridLayerPolicy, HybridStateGeometry, HybridVariant, ParsedHybridConfig,
    QwenFp8QuantizationConfig, PREDICTION_STATE_SEGMENT, TARGET_STATE_SEGMENT,
};
pub use linear_attention::LinearAttention;
pub use model::{state_identity, ForwardContext, LayeredModel, TargetPartitionInput, Unit};
pub use mtp::{EmbeddedInput, ForwardMode, PredictionUnit};
pub use parallel::{
    conditional_local_geometry, local_block_config, local_geometry, local_unit_config,
    unit_parallel_parameter_groups, ConditionalLocalGeometry, LocalGeometry,
};
