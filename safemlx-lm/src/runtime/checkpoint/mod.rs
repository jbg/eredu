//! Checkpoint loading, storage, binding, and conversion.

/// Canonical unloaded-module checkpoint binding.
pub mod binding;
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
