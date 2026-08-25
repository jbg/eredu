//! Shared backend-neutral DeepSeek architecture policy.

/// Stable segment identity for target decoder state.
pub const TARGET_STATE_SEGMENT: &str = "target";
/// Stable segment identity for checkpoint-embedded prediction state.
pub const PREDICTION_STATE_SEGMENT: &str = "prediction";

/// Family-specific compressed-attention equations.
pub mod attention;
/// Shared normalization, residual, and feed-forward sequencing.
pub mod block;
/// Pure SafeTensors/GGUF schemas and canonical name translation.
pub mod checkpoint;
/// Strict V3/R1 and V4 configuration normalization.
pub mod config;
/// Shared routed-plus-shared expert block.
pub mod moe;
/// Shared embedded-prediction layers and outputs.
pub mod mtp;
/// Semantic tensor/expert/hyper-stream placement plans.
pub mod parallel;
/// Shared normalized low-rank projection assembly.
pub mod projection;
/// Thin V3/R1 architecture policy.
pub mod v3;
/// Thin V4 architecture policy.
pub mod v4;

pub use checkpoint::{
    expert_unit_recipes, normalize_v3_weight_formats, normalize_v4_weight_formats,
    translate_v3_gguf_weight_name, translate_v4_gguf_weight_name, v3_expert_recipes,
    v3_expert_residency_catalog, v3_gguf_kv_b_recipe, v3_gguf_plan, v3_load_time_quantization,
    v3_safetensors_plan, v3_unit_recipes, v3_with_checkpoint_formats, v4_expert_recipes,
    v4_expert_residency_catalog, v4_gguf_plan, v4_load_time_quantization, v4_safetensors_plan,
    v4_with_checkpoint_formats, ExpertUnitRecipes,
};
pub use config::{
    parse_v3_config, parse_v3_gguf, parse_v4_config, parse_v4_gguf, v3_architecture_fingerprint,
    v3_uses_split_kv, v4_architecture_fingerprint, ConfigError, DeepSeekQuantizationConfig,
    DsparkConfig, ExpertFormat, Fp8QuantizationConfig, GgufTensorCatalog, LayerPolicy, V3Args,
    V4Args, V4AttentionPolicy, YarnConfig,
};
