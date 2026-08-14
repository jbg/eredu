//! GPT-OSS model family.

pub(crate) mod checkpoint;
pub(crate) mod format;
/// Bounded and unified residency execution.
pub mod layerwise;
/// Fully resident decoder implementation.
pub mod model;
