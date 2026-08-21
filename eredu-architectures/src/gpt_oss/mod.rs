//! Backend-neutral OpenAI GPT-OSS causal decoder architecture.

pub mod block;
pub mod checkpoint;
pub mod config;
pub mod model;
pub mod moe;
pub mod parallel;

pub use block::{new_block, GptOssBlockFactory, TransformerBlock};
pub use checkpoint::{
    expert_recipes, gguf_plan, safetensors_plan, translate_gguf_weight_name, validate_gguf,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_identity, state_layout, ConfigError, ModelArgs,
    MxFp4Config,
};
pub use model::{new_layered_model, LayerUnit, LayeredModel};
pub use moe::RoutedMlp;
pub use parallel::{
    layer_parallel_parameter_groups, local_block_args, local_key_value_heads,
    static_parameter_groups,
};

pub use crate::decoder::{
    cache_layout, cache_layout_with_key_value_heads, create_caches, validate_caches,
    AttentionInput, ForwardContext, LayeredInput, StaticModules,
};
