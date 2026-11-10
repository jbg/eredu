//! Neutral Qwen3-VL configuration, state, and multimodal position policy.

mod checkpoint;
mod config;
mod declarations;
pub use declarations::Declaration;
mod model;
mod parallel;
mod parameters;
mod positions;

pub(crate) use checkpoint::source_linear_shapes;

pub use checkpoint::{
    load_time_quantization, projector_gguf_plan, rank_local_unit_recipes, safetensors_plan,
    static_recipes, translate_text_gguf_weight_name, translate_vision_gguf_weight_name,
    unit_recipes, with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_value, model_args_from_gguf_parts,
    prompt_cache_architecture_fingerprint, state_identity, state_layout_with_text_state,
    vision_config_from_gguf_catalog, GgufModelArgs, ModelArgs, VlConfigError,
};
pub use model::{
    prepare_input, vision_dependency_boundary_schema, vision_partition_boundary_schema,
    ForwardContext, InputPart, LayeredModel, ModelInput, PipelineBoundary, PipelineBoundarySchema,
    PipelinePartitionInput, PipelinePrepared, PipelineVisionState, PreparedInput, StaticModules,
    Unit, VlKernel, TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
pub use parallel::{
    local_geometry, partition_local_geometry, partition_local_routed_geometry, LocalGeometry,
    PartitionLocalFoundation, PartitionLocalGeometry,
};
pub(crate) use parameters::select_definition;
pub use parameters::{component_definition, parameter_description, project_parameters};

pub use positions::{
    mrope_embeddings, mrope_spec, mrope_values, multimodal_position_ids, position_ids_tensor,
    PositionPart,
};

/// Derives complete expert ownership and rank-local text-bank geometry from Qwen3-VL.
pub fn expert_realization_plan(
    declaration: &Declaration,
    geometry: Option<&LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    if let Some(geometry) = geometry {
        geometry
            .validate_for(declaration)
            .map_err(eredu_nn::Error::backend_source)?;
    }
    super::expert_realization_plan(
        declaration.text_declaration(),
        geometry.map(LocalGeometry::text),
        topology,
    )
}
