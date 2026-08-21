//! One backend-neutral hybrid decoder shared by Qwen3-Next and Qwen3.5.

mod block;
mod checkpoint;
mod conditional;
mod config;
mod linear_attention;
mod model;
mod mtp;
mod parallel;

pub use block::{Block, FeedForward, SharedRoutedSwiGlu, TokenMixer};
pub use checkpoint::{
    gguf_plan, qwen3_next_fused_recipes, safetensors_plan, translate_gguf_weight_name,
};
pub use conditional::{
    ConditionalForwardContext, ConditionalInput, ConditionalLayeredModel,
    ConditionalPipelinePrepared, ConditionalPipelineVisionState, ConditionalStaticModules,
    ConditionalUnit,
};
pub use config::{
    fp8_block_row_widths, fused_projection_widths, model_args_from_config_value,
    model_args_from_gguf_catalog, prompt_cache_architecture_fingerprint, state_layout,
    state_layout_with_geometry, with_gguf_vision_projector, HybridConfig, HybridConfigError,
    HybridLayerPolicy, HybridStateGeometry, HybridVariant, ParsedHybridConfig,
    QwenFp8QuantizationConfig,
};
pub use linear_attention::LinearAttention;
pub use model::{state_identity, ForwardContext, LayeredModel, Unit};
pub use mtp::{EmbeddedInput, ForwardMode, PredictionUnit};
pub use parallel::{local_block_config, local_unit_config, unit_parallel_parameter_groups};
