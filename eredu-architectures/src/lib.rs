//! Backend-neutral text model architectures.
//!
//! Architecture code is monomorphized over an [`eredu_nn::NeuralBackend`]. Concrete
//! backends retain their native tensors, packed weights, lazy graphs, fused
//! kernels, caches, and collective implementations.

#![warn(missing_docs)]

/// Shared decoder mechanics used by backend-neutral text architectures.
pub mod decoder;
/// Shared assembly for heterogeneous stateful text decoders.
pub mod hybrid_decoder;

/// Inkling multimodal routed decoder family.
pub mod inkling;
pub mod muse_glimmer;

/// DeepSeek V3/R1 and V4 compressed-attention decoder family.
pub mod deepseek;
/// Neutral Gemma 4 family implementation.
pub mod gemma4;

/// Llama and Mistral-compatible decoder architecture.
pub mod llama;

/// Kimi Linear hybrid KDA/MLA decoder family.
pub mod kimi_linear;
/// LFM2 and LFM2-MoE hybrid decoder architecture.
pub mod lfm2;
pub mod nemotron_h;

/// Qwen2, Qwen3, and Qwen3-MoE text decoder architecture.
pub mod qwen;
