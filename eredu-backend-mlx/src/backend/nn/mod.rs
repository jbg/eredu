//! Architecture-neutral neural-network building blocks.

pub(crate) mod arithmetic;
pub(crate) mod boundary_frame;
pub(crate) mod logical_collective;
pub(crate) mod expert_movement;
/// Shared attention inputs, transforms, cache updates, and kernels.
pub mod attention;
pub(crate) mod convolution;
/// Architecture-neutral block-scaled FP8 projections.
pub mod fp8;
/// Gated-delta recurrent scans shared by linear-attention architectures.
pub mod gated_delta;
pub(crate) mod grouped;
pub(crate) mod grouping;
pub(crate) mod hyper_connections;
/// Activation functions.
pub mod layers;
pub(crate) mod linear;
pub(crate) mod matrix;
pub(crate) mod module;
pub(crate) mod native_quantization;
pub(crate) mod nested;
pub(crate) mod normalization;
pub(crate) mod primitives;
pub(crate) mod relative_attention;
pub(crate) mod rope;
pub(crate) mod selective_scan;
/// Backend-neutral architecture operators specialized to MLX.
pub mod shared;
/// Shared tensor and attention-mask helpers.
pub mod tensor;
/// Cold native allocation and operator bounds for inference admission.
pub mod workspace;
