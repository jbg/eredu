//! Backend-neutral shared Qwen vision encoder policy and components.

mod checkpoint;
mod config;
mod memory;
mod model;
mod parallel;

pub(crate) use checkpoint::{gguf_plan, temporal_patch_recipe};
pub use checkpoint::{safetensors_plan, translate_gguf_weight_name};
pub(crate) use config::config_from_gguf_catalog;
pub use config::{
    prompt_cache_architecture_fingerprint, VisionAttentionPolicy, VisionConfig, VisionConfigError,
    VisionConfigSource, VisionGgufCatalog, VisionLayerPolicy, VisionMode,
};
pub use memory::{VisionMemoryInvocation, VisionMemoryReport};
pub use model::{
    VisionBlock, VisionInput, VisionOutput, VisionOutputTensorRole, VisionState,
    VisionStateTensorRole, VisionStatic, VisionTower,
};
pub use parallel::{
    block_parallel_parameter_groups, local_block_geometry, local_merger_widths,
    owned_static_parallel_parameter_groups, static_parallel_parameter_groups,
};
