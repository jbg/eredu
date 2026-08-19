//! Gemma 4 text and multimodal model family.

pub(crate) mod assistant;
pub(crate) mod audio;
pub(crate) mod checkpoint;
/// Bounded-residency multimodal execution.
pub mod layerwise;
/// Fully resident model implementation.
pub mod model;
pub(crate) mod mtp;
pub(crate) mod multimodal;
pub(crate) mod processor;
pub(crate) mod vision;
