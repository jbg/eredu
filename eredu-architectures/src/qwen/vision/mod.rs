//! Backend-neutral shared Qwen vision encoder policy and components.

mod checkpoint;
mod config;
mod model;
mod parallel;
mod tables;
pub(crate) use tables::encoder_table_spec;

pub(crate) use checkpoint::gguf_plan;
pub use checkpoint::{safetensors_plan, translate_gguf_weight_name};
pub(crate) use config::config_from_gguf_catalog;
pub(crate) use config::prompt_cache_architecture_fingerprint_with_metadata;
pub use config::{
    prompt_cache_architecture_fingerprint, VisionAttentionPolicy, VisionConfig, VisionConfigError,
    VisionConfigSource, VisionGgufCatalog, VisionLayerPolicy, VisionMode,
};
pub use model::{VisionBlock, VisionInput, VisionOutput, VisionState, VisionStatic, VisionTower};
pub use parallel::{
    block_parallel_parameter_groups, local_block_geometry, local_merger_widths,
    owned_static_parallel_parameter_groups, static_parallel_parameter_groups,
};

pub(crate) use parallel::{
    block_parallel_parameter_groups_with_metadata,
    owned_static_parallel_parameter_groups_with_metadata,
};
