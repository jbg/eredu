//! Architecture-neutral checkpoint planning, validation, binding, loading,
//! and bounded materialization.
//!
//! Architecture modules own normalized geometry and emit complete physical
//! tensor constraints, layout alternatives, aliases, exclusions, and derived
//! recipes. This module owns inspection of checkpoint metadata, deterministic
//! constraint evaluation, strict catalog enforcement, recipe inference, and
//! materialization. Source storage constraints deliberately remain separate
//! from runtime binding dtypes: an encoded FP8 source can, for example,
//! materialize as a `u8` runtime array.
//!
//! Aliases and packed/split/fused choices are represented as mutually
//! exclusive `eredu_checkpoint::schema::AlternativeLayoutGroup` variants. Companion tensors
//! are ordinary constraints supplied by the architecture; validation contains
//! no companion naming convention. Architecture adapters feed the same
//! physical plan to structural admission and its [`binding_plan::BindingPlan`]
//! to resident, layerwise, or streamed loading.

/// Stable identities for loaded checkpoint artifacts.
pub(crate) mod artifact;
/// Canonical unloaded-module checkpoint binding.
pub mod binding;
/// Declarative target bindings backed by checkpoint-derived recipes.
pub(crate) mod binding_plan;
/// Out-of-core transformation of dense bindings into packed weight stores.
pub mod bounded_quantization;
/// Strict checkpoint loading and validation.
pub mod load;
/// Generic affine checkpoint quantization and conversion.
pub mod quantization;
/// Composable checkpoint-derived weight recipes.
pub mod recipe;
/// Persistent lazy checkpoint tensor storage.
pub mod store;
