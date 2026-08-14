//! DeepSeek-V4 model family.

pub(crate) mod attention;
pub(crate) mod checkpoint;
mod layers;
/// Generalized bounded-residency execution.
pub mod layerwise;
/// Configuration and decoder implementation.
pub mod model;

pub use model::{DsparkConfig, ModelArgs};
