//! Llama and Mistral-compatible model family.

pub(crate) mod checkpoint;
/// Bounded-residency execution and unified loading.
pub mod layerwise;
/// Reusable decoder operators; checkpoint loading is exposed only by [`layerwise`].
pub mod model;
