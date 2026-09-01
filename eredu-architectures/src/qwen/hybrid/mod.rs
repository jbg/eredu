//! One backend-neutral hybrid decoder shared by Qwen3-Next and Qwen3.5.

mod block;
mod checkpoint;
mod conditional;
mod config;
mod linear_attention;
mod model;
mod mtp;
mod parallel;

pub use block::{expert_bank_spec, Block, FeedForward, SharedRoutedGatedProduct, TokenMixer};
pub use checkpoint::{
    composite_safetensors_plan, conditional_load_time_quantization,
    conditional_projector_gguf_plan, conditional_unit_recipes, conditional_with_checkpoint_formats,
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    qwen3_next_fused_recipes, safetensors_plan, static_recipes, translate_gguf_weight_name,
    translate_vision_gguf_weight_name, unit_recipes,
};
pub use conditional::{
    ConditionalForwardContext, ConditionalInput, ConditionalLayeredModel,
    ConditionalPartitionInput, ConditionalPipelineBoundary, ConditionalPipelineBoundarySchema,
    ConditionalPipelinePrepared, ConditionalPipelineVisionState, ConditionalStaticModules,
    ConditionalUnit, VISION_EXECUTION_GROUP,
};
pub use config::{
    fp8_block_row_widths, fused_projection_widths, model_args_from_config_value,
    model_args_from_gguf_catalog, prompt_cache_architecture_fingerprint, state_layout,
    state_layout_with_geometry, vision_config_from_gguf_catalog, with_gguf_vision_projector,
    with_media_token_ids, HybridConfig, HybridConfigError, HybridLayerPolicy, HybridStateGeometry,
    HybridVariant, ParsedHybridConfig, QwenFp8QuantizationConfig, PREDICTION_STATE_SEGMENT,
    TARGET_STATE_SEGMENT,
};
pub use linear_attention::LinearAttention;
pub use model::{state_identity, ForwardContext, LayeredModel, TargetPartitionInput, Unit};
pub use mtp::{EmbeddedInput, ForwardMode, PredictionUnit};
pub use parallel::{
    conditional_local_geometry, local_block_config, local_geometry, local_unit_config,
    unit_parallel_parameter_groups, ConditionalLocalGeometry, LocalGeometry,
};

/// Derives complete expert ownership and local bank geometry for Qwen hybrid text/MTP units.
pub fn expert_realization_plan<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    architecture: &LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    let geometry = architecture.shared_parallel_geometry();
    realization_plan(architecture.config(), geometry.as_deref(), topology)
}

/// Derives complete expert ownership and local bank geometry for conditional Qwen hybrid units.
pub fn conditional_expert_realization_plan<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    architecture: &ConditionalLayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    let geometry = architecture.shared_parallel_geometry();
    realization_plan(
        &architecture.parsed().text,
        geometry.as_deref().map(ConditionalLocalGeometry::text),
        topology,
    )
}

fn realization_plan(
    config: &HybridConfig,
    geometry: Option<&LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
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
