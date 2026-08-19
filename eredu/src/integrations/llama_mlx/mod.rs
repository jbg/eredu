//! Llama and Mistral-compatible model family.

pub(crate) mod checkpoint;
/// Bounded-residency execution and unified loading.
pub mod layerwise;
/// MLX model configuration and resident binding; shared operators live in
/// `eredu-architectures`.
pub mod model;
