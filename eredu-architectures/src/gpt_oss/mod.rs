//! Backend-neutral OpenAI GPT-OSS causal decoder architecture.

pub mod block;
pub mod checkpoint;
pub mod config;
mod declarations;
pub use declarations::Declaration;
pub mod model;
pub mod moe;
pub mod parallel;
pub(crate) use parallel::parameter_description;

pub use block::{new_block, GptOssBlockFactory, TransformerBlock};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_expert_quantization_targets, gguf_plan,
    load_time_quantization, rank_local_expert_recipes, safetensors_expert_tensors,
    safetensors_plan, translate_gguf_weight_name, validate_gguf, with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_identity, ConfigError, ModelArgs, MxFp4Config,
};
pub use model::{LayeredModel, PartitionedLayeredModel};
pub use moe::{expert_realization_plan, RoutedMlp};
pub use parallel::{
    local_block_args, local_geometry, partition_local_routed_geometry, LocalGeometry,
};

pub use crate::decoder::{
    cache_layout, cache_layout_with_attention_geometry, create_caches, validate_caches,
    AttentionInput, ForwardContext, LayeredInput, PartitionLocalGeometry, PartitionStaticModules,
    StaticModules,
};
