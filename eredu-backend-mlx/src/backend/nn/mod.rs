//! Architecture-neutral neural-network building blocks.

/// Shared attention inputs, transforms, cache updates, and kernels.
pub mod attention;
/// Architecture-neutral block-scaled FP8 projections.
pub mod fp8;
/// Gated-delta recurrent scans shared by linear-attention architectures.
pub mod gated_delta;
pub(crate) mod hyper_connections;
/// Activation functions.
pub mod layers;
pub(crate) mod linear;
pub(crate) mod module;
pub(crate) mod moe;
pub(crate) mod native_quantization;
pub(crate) mod nested;
pub(crate) mod primitives;
pub(crate) mod rope;
pub(crate) mod routing;
/// Backend-neutral architecture operators specialized to MLX.
pub mod shared;
/// Shared tensor and attention-mask helpers.
pub mod tensor;
