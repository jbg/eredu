//! Shared Qwen vision-language implementations.

/// Bounded-residency vision-language execution.
pub mod layerwise;
/// Qwen3-VL implementation.
pub mod model;
/// Qwen3-VL-MoE specialization.
pub mod moe;
#[cfg(feature = "mlx-image")]
pub(crate) mod processor;
pub(crate) mod vision;
