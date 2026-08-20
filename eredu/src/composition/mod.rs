//! Cold-path architecture/backend composition selected by public loaders.

#[cfg(feature = "mlx")]
pub(crate) mod llama;
#[cfg(feature = "mlx")]
pub(crate) mod llama_checkpoint;
#[cfg(feature = "mlx")]
pub(crate) mod llama_mlx;
#[cfg(feature = "mlx")]
pub mod mlx;
#[cfg(feature = "mlx")]
pub mod mlx_architectures;
#[cfg(feature = "mlx")]
pub(crate) mod qwen;
#[cfg(feature = "mlx")]
pub(crate) mod qwen_expert;
