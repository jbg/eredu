//! Cold-path architecture/backend composition selected by public loaders.

use ref_cast::RefCast;
use safemlx::Array;

pub(crate) fn tensor_ref(array: &Array) -> &crate::MlxTensor {
    crate::MlxTensor::ref_cast(array)
}

pub(crate) fn tensor_opt(array: Option<&Array>) -> Option<&crate::MlxTensor> {
    array.map(tensor_ref)
}

pub mod deepseek;
pub mod deepseek_expert;
pub mod gemma4;
pub mod gemma4_expert;
#[cfg(feature = "media")]
pub mod gemma4_processor;
pub mod gpt_oss;
pub mod inkling;
pub mod inkling_expert;
#[cfg(feature = "media")]
pub mod inkling_processor;
pub mod kimi_linear;
// MLX adapter only; the neutral family is always available from
// `eredu_architectures::lfm2`.
pub mod lfm2;
pub mod llama;
pub mod mlx;
pub mod moshi;
pub mod moshi_parallel;
pub mod muse_glimmer;
pub mod muse_glimmer_expert;
#[cfg(feature = "image")]
pub mod muse_glimmer_processor;
pub mod nemotron_h;
pub mod qwen;

#[cfg(test)]
#[path = "tests/mlx_architecture_conformance.rs"]
mod mlx_architecture_conformance;
