//! DiffusionGemma geometry and shared encoder/decoder parameter construction.

pub mod boundary;
mod checkpoint;
pub mod composite;
mod config;
pub mod model;
pub mod parallel;
pub mod text;
pub(crate) mod topology;
pub mod vision;

/// Shared-stack invocation equations, distinct from parameter and state ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextInvocationPhase {
    /// Appends authoritative context.
    Context,
    /// Reads context while refining an uncommitted canvas.
    Refinement,
}

pub use checkpoint::{
    canonical_safetensors_target, expert_recipes, expert_residency_catalog, gguf_plan,
    safetensors_plan, translate_gguf_weight_name, unit_recipes,
};
pub use config::{ConfigError, EncoderAttention, FamilyConfig};

/// Declares one authoritative context history per shared text layer. Refinement
/// invocations borrow these histories and own no second key/value state.
pub fn state_layout(
    config: &FamilyConfig,
) -> Result<eredu_runtime::StateLayout, eredu_runtime::StateError> {
    let layers = config
        .text
        .layer_schedule
        .iter()
        .map(|policy| {
            eredu_core::cache::LayerCachePolicy::key_value(
                policy.attention,
                policy.num_key_value_heads.get() as i32,
                policy.head_dim.get() as i32,
            )
            .map_err(|error| eredu_runtime::StateError::InvalidResidency(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    eredu_runtime::StateLayout::new(
        eredu_core::LayerSchedule::new(layers.len(), layers)
            .map_err(|error| eredu_runtime::StateError::InvalidResidency(error.to_string()))?,
    )
}
