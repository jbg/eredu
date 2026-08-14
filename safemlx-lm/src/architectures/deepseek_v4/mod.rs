//! DeepSeek-V4 model family.

pub(crate) mod attention;
mod layers;
/// Generalized bounded-residency execution.
pub mod layerwise;
/// Configuration and decoder implementation.
pub mod model;

pub use model::{DsparkConfig, ModelArgs};
