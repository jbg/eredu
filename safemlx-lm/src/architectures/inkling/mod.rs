//! Inkling multimodal model family.

/// Bounded-residency multimodal execution.
pub mod layerwise;
/// Fully resident multimodal implementation.
pub mod model;
#[cfg(feature = "media-processing")]
pub(crate) mod processor;
