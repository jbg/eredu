//! Moshi and PersonaPlex realtime-token model family.

/// Bounded-residency execution.
pub mod layerwise;
/// Fully resident Moshi implementation.
pub mod model;
/// PersonaPlex specialization.
pub mod personaplex;
/// Architecture-owned PersonaPlex checkpoint contracts.
pub(crate) mod personaplex_checkpoint;
