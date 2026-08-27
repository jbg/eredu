//! Architecture-neutral neural-network building blocks.

/// Shared attention inputs, transforms, cache updates, and kernels.
pub mod attention;
/// Architecture-neutral block-scaled FP8 projections.
pub mod fp8;
/// Gated-delta recurrent scans shared by linear-attention architectures.
pub mod gated_delta;
/// Multi-stream residual mixing shared by hyper-connection architectures.
pub mod hyper_connections;
/// Activation functions.
pub mod layers;
/// Linear layers, embeddings, and language-model heads.
pub mod linear;
/// Mixture-of-experts routing and packed expert implementations.
pub mod moe;
/// Rotary position-embedding variants and initialization.
pub mod rope;
/// Backend-neutral architecture operators specialized to MLX.
pub mod shared;
/// Shared tensor and attention-mask helpers.
pub mod tensor;
