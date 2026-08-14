//! Inkling multimodal model family.

pub(crate) mod checkpoint;
/// Bounded-residency multimodal execution.
pub mod layerwise;
/// Fully resident multimodal implementation.
pub mod model;
pub(crate) mod processor;
