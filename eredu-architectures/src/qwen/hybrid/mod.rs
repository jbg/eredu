//! One backend-neutral hybrid decoder shared by Qwen3-Next and Qwen3.5.

/// Target-only causal text passes preserve convolution, recurrent and attention state.
pub(crate) const CHUNKED_TEXT_PREFILL: bool = true;

pub(crate) mod block;
mod checkpoint;
mod conditional;
mod config;
mod linear_attention;
mod model;
mod mtp;
mod parallel;
mod parameters;
mod safetensors;
pub(crate) mod topology;

pub(crate) use safetensors::catalog_plan as safetensors_catalog_plan;

/// Checkpoint spellings shared by tensor admission and FP8 exclusion policy.
fn text_checkpoint_aliases(name: &str) -> Vec<String> {
    name.strip_prefix("model.")
        .map(|rest| {
            vec![
                format!("model.language_model.{rest}"),
                format!("language_model.{rest}"),
                format!("model.model.{rest}"),
            ]
        })
        .unwrap_or_default()
}

pub use block::{expert_bank_spec, Block, FeedForward, TokenMixer};
pub use checkpoint::{
    composite_safetensors_plan, conditional_load_time_quantization,
    conditional_projector_gguf_plan, conditional_unit_recipes, conditional_with_checkpoint_formats,
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    qwen3_next_fused_recipes, safetensors_plan, static_recipes, translate_gguf_weight_name,
    translate_vision_gguf_weight_name, unit_recipes,
};
pub use conditional::{
    ConditionalForwardContext, ConditionalInput, ConditionalKernel, ConditionalLayeredModel,
    ConditionalPartitionInput, ConditionalPipelineBoundary, ConditionalPipelineBoundarySchema,
    ConditionalPipelinePrepared, ConditionalPipelineVisionState, ConditionalStaticModules,
    ConditionalUnit, VISION_EXECUTION_GROUP,
};
pub use config::{
    conditional_prompt_cache_architecture_fingerprint, fp8_block_row_widths,
    fused_projection_widths, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry,
    vision_config_from_gguf_catalog, with_gguf_vision_projector, with_media_token_ids,
    HybridConfig, HybridConfigError, HybridLayerPolicy, HybridStateGeometry, HybridVariant,
    ParsedHybridConfig, QwenFp8QuantizationConfig, PREDICTION_STATE_SEGMENT, TARGET_STATE_SEGMENT,
};
pub use linear_attention::recurrent_spec;
pub use model::{
    state_identity, ForwardContext, LayeredModel, QwenKernel, QwenStaticModules,
    TargetPartitionInput, Unit,
};
pub use mtp::{
    prompt_token_identity, EmbeddedInput, ForwardMode, PredictionShared, PredictionUnit,
};
pub(crate) use parallel::routed_conditional_partition_local_geometry;
pub use parallel::{
    conditional_local_geometry, conditional_partition_local_geometry, local_block_config,
    local_geometry, local_unit_config, ConditionalLocalGeometry,
    ConditionalPartitionLocalFoundation, ConditionalPartitionLocalGeometry, LocalGeometry,
};
pub(crate) use parameters::{select_conditional_definition, select_definition};
pub use parameters::{
    component_definition, conditional_component_definition, conditional_parameter_description,
    parameter_description, project_conditional_parameters, project_parameters, unit_parallel_parameter_groups,
};

/// Derives the complete replicated target/MTP expert plan from normalized geometry.
pub fn replicated_expert_realization_plan(
    config: &HybridConfig,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error> {
    expert_realization_plan(
        config,
        None,
        eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(eredu_nn::Error::backend)?,
            0,
        )
        .map_err(eredu_nn::Error::backend)?,
    )?
    .ok_or_else(|| eredu_nn::Error::backend("Qwen hybrid routed text requires sparse units"))
}

/// Derives conditional text/MTP expert ownership from neutral policy and optional local geometry.
pub fn conditional_expert_realization_plan(
    parsed: &ParsedHybridConfig,
    geometry: Option<&ConditionalLocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    parsed
        .vision
        .as_ref()
        .ok_or_else(|| {
            eredu_nn::Error::backend("conditional Qwen expert planning requires vision config")
        })?
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(eredu_nn::Error::backend)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_for(parsed)
            .map_err(eredu_nn::Error::backend)?;
    }
    expert_realization_plan(
        &parsed.text,
        geometry.map(ConditionalLocalGeometry::text),
        topology,
    )
}

/// Derives the selected text expert bank without constructing native modules.
pub(crate) fn partition_expert_realization_plan(
    config: &HybridConfig,
    geometry: &LocalGeometry,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>, eredu_nn::Error> {
    expert_realization_plan(config, Some(geometry), topology)?
        .ok_or_else(|| eredu_nn::Error::backend("Qwen partition has no routed units"))
}

/// Derives target and MTP bank ownership without a backend type or native resources.
pub fn expert_realization_plan(
    config: &HybridConfig,
    geometry: Option<&LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    config.validate().map_err(eredu_nn::Error::backend)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_for(config)
            .map_err(eredu_nn::Error::backend)?;
    }
    if !config.is_moe() {
        return Ok(None);
    }
    let global_experts = usize::try_from(config.num_experts).map_err(eredu_nn::Error::backend)?;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(eredu_nn::Error::backend)?
        .len(),
    )
    .map_err(eredu_nn::Error::backend)?;
    let targets = usize::try_from(config.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let predictions =
        usize::try_from(config.mtp_num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let target_group =
        eredu_runtime::ExecutionGroupId::new("target").map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for layer in 0..targets {
        let local = geometry
            .and_then(|geometry| geometry.target(layer))
            .unwrap_or(config);
        unit_specs.insert(
            (target_group.clone(), layer),
            block::localized_expert_bank_spec(
                config,
                layer,
                local_experts,
                local.moe_intermediate_size,
            )?,
        );
    }
    for depth in 0..predictions {
        let local = geometry
            .and_then(|geometry| geometry.prediction(depth))
            .unwrap_or(config);
        let group = eredu_runtime::ExecutionGroupId::new(format!("mtp.{depth}"))
            .map_err(eredu_nn::Error::backend)?;
        unit_specs.insert(
            (group, 0),
            block::localized_expert_bank_spec(
                config,
                targets + depth,
                local_experts,
                local.moe_intermediate_size,
            )?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}
