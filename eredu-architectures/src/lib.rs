//! Backend-neutral text model architectures.
//!
//! Architecture code is monomorphized over an [`eredu_nn::NeuralBackend`]. Concrete
//! backends retain their native tensors, packed weights, lazy graphs, fused
//! kernels, caches, and collective implementations.

#![warn(missing_docs)]

/// Shared decoder mechanics used by backend-neutral text architectures.
pub mod decoder;

/// Llama and Mistral-compatible decoder architecture.
pub mod llama;

/// Qwen2, Qwen3, and Qwen3-MoE text decoder architecture.
pub mod qwen;
