//! DeepSeek-V3 and DeepSeek-R1 model family.

/// Architecture-owned physical checkpoint contracts.
pub(crate) mod checkpoint;
/// Bounded-residency execution.
pub mod layerwise;
/// Fully resident decoder implementation.
pub mod model;
