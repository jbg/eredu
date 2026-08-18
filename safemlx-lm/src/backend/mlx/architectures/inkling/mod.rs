//! Inkling multimodal model family.

pub(crate) mod checkpoint;
/// Bounded-residency multimodal execution.
pub mod layerwise;
/// Reusable multimodal operators; checkpoint loading is exposed only by [`layerwise`].
pub mod model;
pub(crate) mod processor;
