//! Generic model execution engines and observability hooks.

pub mod generic;
/// Lightweight activation inspection hooks.
pub mod inspection;
/// Generic architecture adapters and host-backed layerwise execution.
pub mod layerwise;
/// Native binding and exact completion for selected component pipelines.
pub(crate) mod pipeline;
