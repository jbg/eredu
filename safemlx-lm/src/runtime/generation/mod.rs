//! Sampling, semantic streaming, and speculative decoding.

/// Token sampling policies.
pub mod sampler;
/// Multi-token prediction and speculative decoding.
pub mod speculative;
/// Protocol-independent semantic streaming.
pub mod streaming;
