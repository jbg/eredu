//! MLX checkpoint loading and bounded native materialization.
//!
//! Architecture modules own normalized geometry and emit complete physical
//! tensor constraints, layout alternatives, aliases, exclusions, and derived
//! recipes. Neutral crates own deterministic constraint evaluation, strict
//! catalog enforcement, recipe inference, and binding orchestration. This
//! module realizes selected recipes as MLX arrays. Source storage constraints deliberately remain separate
//! from runtime binding dtypes: an encoded FP8 source can, for example,
//! materialize as a `u8` runtime array.
//!
//! Aliases and packed/split/fused choices are represented as mutually
//! exclusive `eredu_checkpoint::schema::AlternativeLayoutGroup` variants. Companion tensors
//! are ordinary constraints supplied by the architecture; validation contains
//! no companion naming convention. Architecture adapters feed the same
//! physical plan to structural admission and its
//! [`BindingPlan`](eredu_runtime::BindingPlan)
//! to resident, layerwise, or streamed loading.

/// Canonical unloaded-module checkpoint binding.
pub mod binding;
/// Out-of-core transformation of dense bindings into packed weight stores.
pub mod bounded_quantization;
/// GGUF checkpoint access and bounded MLX tensor materialization.
pub mod gguf;
/// Strict checkpoint loading and validation.
pub(crate) mod load;
/// Generic affine checkpoint quantization and conversion.
pub mod quantization;
/// Composable checkpoint-derived weight recipes.
pub mod recipe;
/// Persistent lazy checkpoint tensor storage.
pub mod store;
