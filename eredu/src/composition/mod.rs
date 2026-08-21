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
pub(crate) mod gpt_oss;
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
pub mod mlx;
#[cfg(feature = "mlx")]
pub(crate) mod moshi;
#[cfg(feature = "mlx")]
pub(crate) mod moshi_parallel;
#[cfg(feature = "mlx")]
pub(crate) mod muse_glimmer;
#[cfg(feature = "mlx")]
pub(crate) mod muse_glimmer_expert;
#[cfg(feature = "mlx-image")]
pub(crate) mod muse_glimmer_processor;
#[cfg(feature = "mlx")]
pub(crate) mod nemotron_h;
#[cfg(feature = "mlx")]
pub(crate) mod qwen;

#[cfg(all(test, feature = "mlx"))]
#[path = "tests/mlx_architecture_conformance.rs"]
mod mlx_architecture_conformance;
