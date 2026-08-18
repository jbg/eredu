//! Checkpoint loading, event-owned materialization, binding, and bounded
//! double-buffered conversion.

/// Stable identities for loaded checkpoint artifacts.
pub(crate) mod artifact;
/// Canonical unloaded-module checkpoint binding.
pub mod binding;
/// Out-of-core transformation of dense bindings into packed weight stores.
pub mod bounded_quantization;
/// GGUF tokenizer metadata conversion.
pub(crate) mod gguf;
/// Strict checkpoint loading and validation.
pub mod load;
/// Generic affine checkpoint quantization and conversion.
pub mod quantization;
/// Composable checkpoint-derived weight recipes.
pub mod recipe;
/// Persistent lazy checkpoint tensor storage.
pub mod store;
