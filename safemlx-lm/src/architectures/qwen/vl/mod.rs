//! Shared Qwen vision-language implementations.

/// Architecture-owned text, vision, and projector checkpoint contracts.
pub(crate) mod checkpoint;
/// Bounded-residency vision-language execution.
pub mod layerwise;
/// Qwen3-VL implementation.
pub mod model;
/// Qwen3-VL-MoE specialization.
pub(crate) mod moe;
#[cfg(feature = "image-processing")]
pub(crate) mod processor;
pub(crate) mod vision;
