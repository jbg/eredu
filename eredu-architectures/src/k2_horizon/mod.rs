//! Backend-neutral K2 Horizon dense, routed SwiGLU, and routed-value policy.

mod checkpoint;
mod config;
mod gguf;
mod routing;
pub use checkpoint::{gguf_plan, parameter_shapes, safetensors_plan, translate_gguf_weight_name};
pub use config::{
    model_args_from_config_value, prompt_cache_architecture_fingerprint, ConfigError,
    Configuration, ExpertBank, ModelArgs,
};
pub use gguf::model_args_from_gguf_catalog;
pub use routing::{feed_forward_expert_spec, new_router, value_expert_spec};

mod model;
mod parallel;
pub use model::{
    new_block, BlockFactory, DenseBlockFactory, DenseLayeredModel, FeedForward, LayeredModel,
    PartitionedLayeredModel, Projections, RoutedValues, TransformerBlock,
};
pub use parallel::{
    block_parameter_groups, expert_realization_plans, local_block_args, parameter_description,
    partition_local_routed_geometry,
};

mod recipes;
pub use recipes::{
    expert_recipes, expert_residency_catalog, expert_unit_recipes, linear_companion_recipes,
};

mod formats;
pub(crate) use formats::normalize_gguf_formats;

/// Applies the retained executable format to each selected logical parameter.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    formats: impl IntoIterator<Item = (String, eredu_checkpoint::LinearFormat)>,
) -> Result<ModelArgs, String> {
    let mut selected = args.clone();
    selected.formats.extend(formats);
    selected.validate().map_err(|e| e.to_string())?;
    Ok(selected)
}
