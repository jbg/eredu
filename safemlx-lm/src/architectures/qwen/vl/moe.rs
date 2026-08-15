//! Qwen3-VL-MoE multimodal conditional-generation support.
//!
//! The architecture shares Qwen3-VL's vision encoder, DeepStack integration,
//! multimodal RoPE, and runtime input preparation. Its language decoder uses
//! the sparse Qwen3 feed-forward blocks selected by the nested text config.

use crate::error::Error;

pub(crate) use super::model::Cache;

pub(crate) fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    super::model::validate_model_config_value(config)
}
