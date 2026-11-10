//! Backend-neutral K2 Horizon dense, routed SwiGLU, and routed-value policy.

mod checkpoint;
mod config;
mod declarations;
pub use declarations::Declaration;
mod gguf;
mod routing;
/// Shared logical construction declarations for all K2 parameter owners.
pub mod specs;
pub use checkpoint::{
    gguf_plan, safetensors_plan, source_parameters, translate_gguf_weight_name, SourceLayout,
    SourceParameter,
};
pub use config::{
    model_args_from_config_value, prompt_cache_architecture_fingerprint, ConfigError,
    Configuration, ExpertBank, ModelArgs,
};
pub use gguf::model_args_from_gguf_catalog;

mod model;
mod parallel;
pub use model::{
    new_block, BlockFactory, DenseBlockFactory, DenseLayeredModel, FeedForward, LayeredModel,
    PartitionedLayeredModel, ProjectionParameters, Projections, RoutedValues, TransformerBlock,
};
pub use parallel::{
    expert_realization_plans, local_block_args, parameter_description,
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
    let source = Declaration::new(args.clone()).map_err(|e| e.to_string())?;
    let mut selected = args.clone();
    selected.formats.extend(formats);
    let selected = source.with_formats(selected).map_err(|e| e.to_string())?;
    Ok(selected.args().clone())
}
