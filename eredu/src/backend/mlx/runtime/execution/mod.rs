//! Generic model execution engines and observability hooks.

pub(crate) mod generic;
/// Lightweight activation inspection hooks.
pub mod inspection;
/// Generic model-family adapters and host-backed layerwise execution.
pub mod layerwise;
