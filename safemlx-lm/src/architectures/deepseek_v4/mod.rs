//! DeepSeek-V4 model family.

mod attention;
mod layers;
/// Configuration and decoder implementation.
pub mod model;

pub use model::{DsparkConfig, ModelArgs};
