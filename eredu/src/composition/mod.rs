//! Cold-path architecture/backend composition selected by public loaders.

#[cfg(feature = "mlx")]
pub(crate) mod deepseek;
#[cfg(feature = "mlx")]
pub(crate) mod deepseek_expert;
#[cfg(feature = "mlx")]
pub(crate) mod kimi_linear;
pub(crate) mod lfm2;
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
pub mod nemotron_h;
#[cfg(feature = "mlx")]
pub(crate) mod qwen;
#[cfg(feature = "mlx")]
pub(crate) mod qwen_expert;
