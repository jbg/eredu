//! Architecture-independent model execution infrastructure.

/// Attention-cache storage and residency.
pub mod cache;
/// Chat preparation, output dialects, and constrained decoding.
pub mod chat;
/// Checkpoint loading, storage, binding, and conversion.
pub mod checkpoint;
/// Distributed topology, placement, routing, and transport.
pub mod distributed;
/// Generic execution engines and observability hooks.
pub mod execution;
/// Sampling, streaming, and speculative decoding.
pub mod generation;
/// Typed multimodal inputs and media preprocessing.
pub mod media;
/// Weight-residency planning and execution.
pub mod residency;
