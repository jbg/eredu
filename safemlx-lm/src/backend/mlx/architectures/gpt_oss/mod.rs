//! GPT-OSS model family.

pub(crate) mod checkpoint;
/// Bounded and unified residency execution.
pub mod layerwise;
/// Reusable decoder operators; checkpoint loading is exposed only by [`layerwise`].
pub mod model;
