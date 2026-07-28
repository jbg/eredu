//! Compatibility paths for architecture-neutral neural-network components.

/// Compatibility re-export for shared attention components.
pub use crate::nn::attention;
/// Compatibility re-export for causal convolution layers.
pub use crate::nn::convolution;
/// Compatibility re-export for block-scaled FP8 projections.
pub use crate::nn::fp8 as block_fp8;
/// Compatibility re-export for causal generation traits.
pub use crate::nn::generation;
/// Compatibility re-export for activations and feed-forward layers.
pub use crate::nn::layers;
/// Compatibility re-export for linear layers and embeddings.
pub use crate::nn::linear;
/// Compatibility re-export for mixture-of-experts components.
pub use crate::nn::moe;
