//! MLX tensor, checkpoint, execution, and residency infrastructure.

/// Attention-cache storage and residency.
pub mod cache;
/// Checkpoint loading, storage, binding, and conversion.
pub mod checkpoint;
/// MLX distributed placement, collectives, and transport.
pub mod distributed;
/// MLX model execution engines and observability hooks.
pub mod execution;
/// MLX token sampling.
pub mod generation;
/// MLX tensor inputs and media preprocessing.
pub mod media;
/// MLX weight-residency execution.
pub mod residency;
