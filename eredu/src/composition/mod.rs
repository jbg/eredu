//! Cold-path architecture/backend composition selected by public loaders.

#[cfg(feature = "mlx")]
pub(crate) mod deepseek;
#[cfg(feature = "mlx")]
pub(crate) mod deepseek_expert;
#[cfg(feature = "mlx")]
pub(crate) mod gemma4;
#[cfg(feature = "mlx")]
pub(crate) mod gemma4_expert;
#[cfg(feature = "mlx-media")]
pub(crate) mod gemma4_processor;
#[cfg(feature = "mlx")]
pub(crate) mod inkling;
#[cfg(feature = "mlx")]
pub(crate) mod inkling_expert;
#[cfg(feature = "mlx-media")]
pub(crate) mod inkling_processor;
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
pub(crate) mod muse_glimmer;
#[cfg(feature = "mlx")]
pub(crate) mod muse_glimmer_expert;
#[cfg(feature = "mlx-image")]
pub(crate) mod muse_glimmer_processor;
#[cfg(feature = "mlx")]
pub mod nemotron_h;
#[cfg(feature = "mlx")]
pub(crate) mod qwen;
#[cfg(feature = "mlx")]
pub(crate) mod qwen_expert;
#[cfg(all(test, feature = "mlx"))]
mod stage5_neutral_tests;
