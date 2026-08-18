//! Facade execution infrastructure and MLX-specific runtime implementations.

/// Validated decoder-layer schedules and attention geometry.
pub mod attention {
    pub use safemlx_lm_core::attention::*;
}
/// Attention-cache storage and residency.
#[cfg(feature = "mlx")]
pub mod cache;
/// Chat preparation, output dialects, and constrained decoding.
pub mod chat;
/// Checkpoint loading, storage, binding, and conversion.
#[cfg(feature = "mlx")]
pub mod checkpoint;
/// Distributed topology, placement, routing, and transport.
#[cfg(feature = "mlx")]
pub mod distributed;
/// Generic execution engines and observability hooks.
#[cfg(feature = "mlx")]
pub mod execution;
/// Sampling, streaming, and speculative decoding.
pub mod generation;
/// Typed multimodal inputs and media preprocessing.
#[cfg(feature = "mlx")]
pub mod media;
/// Weight-residency planning and execution.
#[cfg(feature = "mlx")]
pub mod residency;
