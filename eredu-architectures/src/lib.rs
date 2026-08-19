//! Backend-neutral text model architectures.
//!
//! Architecture code is monomorphized over an [`eredu_nn::NeuralBackend`]. Concrete
//! backends retain their native tensors, packed weights, lazy graphs, fused
//! kernels, caches, and collective implementations.

#![warn(missing_docs)]

/// Llama and Mistral-compatible decoder architecture.
pub mod llama;
