//! Shared Qwen3-Next and Qwen3.5 hybrid decoder implementation.

/// Bounded-residency execution shared by the hybrid family.
pub mod layerwise;
pub(crate) mod mtp;
/// Qwen3.5 dense and MoE implementation.
pub mod qwen3_5;
/// Qwen3-Next specialization.
pub mod qwen3_next;
