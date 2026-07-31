//! Architecture-neutral neural-network building blocks.

/// Shared attention inputs, transforms, cache updates, and kernels.
pub mod attention;
/// Causal convolution layers and generation caches.
pub mod convolution;
/// Architecture-neutral block-scaled FP8 projections.
pub mod fp8;
/// Gated-delta recurrent scans shared by linear-attention architectures.
pub mod gated_delta;
/// Causal language-model generation traits used by architecture implementations.
pub mod generation;
/// Activation functions and feed-forward layers.
pub mod layers;
/// Linear layers, embeddings, and language-model heads.
pub mod linear;
/// Mixture-of-experts routing and packed expert implementations.
pub mod moe;
/// Rotary position-embedding variants and initialization.
pub mod rope;
/// Shared tensor and attention-mask helpers.
pub mod tensor;
