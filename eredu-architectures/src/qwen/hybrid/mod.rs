//! One backend-neutral hybrid decoder shared by Qwen3-Next and Qwen3.5.

mod declarations;
pub use declarations::HybridConfig;
mod specs;
pub use specs::BlockParameters;
pub(crate) use specs::{FeedForwardParameters, MixerParameters};
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

pub(crate) use checkpoint::{conditional_source_linear_shapes, source_linear_shapes};
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
pub(crate) use checkpoint::qwen3_next_global_sources;
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
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_block_configs,
    vision_config_from_gguf_catalog, with_gguf_vision_projector, with_media_token_ids,
    HybridConfigError, HybridLayerPolicy, HybridPolicy, HybridVariant, ParsedHybridConfig,
    QwenFp8QuantizationConfig, PREDICTION_STATE_SEGMENT, TARGET_STATE_SEGMENT,
};
pub use model::{
    state_identity, ForwardContext, LayeredModel, QwenKernel, QwenStaticModules,
    TargetPartitionInput, Unit,
};
pub use mtp::{
    prompt_token_identity, EmbeddedInput, ForwardMode, PredictionShared, PredictionSharedSpec,
    PredictionUnit,
};
pub(crate) use parallel::routed_conditional_partition_local_geometry;
pub use parallel::{
    conditional_local_geometry, conditional_partition_local_geometry, local_block_config,
    local_geometry, local_unit_config, ConditionalLocalGeometry,
    ConditionalPartitionLocalFoundation, ConditionalPartitionLocalGeometry, LocalGeometry,
};
pub use parameters::{
    component_definition, conditional_component_definition, conditional_parameter_description,
    parameter_description, project_conditional_parameters, project_parameters,
    unit_parallel_parameter_groups,
};
pub(crate) use parameters::{select_conditional_definition, select_definition};

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
    let banks = block::expert_bank_specs(config)?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| eredu_nn::Error::backend("Qwen hybrid has no declared expert banks"))?
        .group_count() as usize;
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
    let (schedule, _, layout) = parameters::execution_schedule(config)?;
    let targets = schedule.unit_count(0)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for (layer, bank) in banks {
        let address = layout
            .address(layer)
            .ok_or_else(|| eredu_nn::Error::backend("unresolved Qwen hybrid bank invocation"))?;
        let local = if address.group() == 0 {
            geometry.and_then(|geometry| geometry.target(layer))
        } else {
            geometry.and_then(|geometry| geometry.prediction(layer - targets))
        };
        let width = local.map_or(bank.intermediate_dimensions(), |local| {
            local.moe_intermediate_size
        });
        unit_specs.insert(
            (
                layout
                    .group_id(address.group())
                    .ok_or_else(|| eredu_nn::Error::backend("unresolved Qwen hybrid bank owner"))?
                    .clone(),
                address.index(),
            ),
            bank.with_group_geometry(local_experts, width)?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}

/// Shared static construction consumed by ordinary execution and cold topology.
pub(super) fn static_module_spec(config: &HybridPolicy) -> crate::decoder::StaticModuleSpec {
    crate::decoder::StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: "model.embed_tokens.weight".into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: config.vocab_size,
        hidden_size: config.hidden_size,
        normalization_epsilon: config.rms_norm_eps,
        normalization_offset: 1.0,
        embedding_format: config
            .linear_format("model.embed_tokens.weight")
            .weight_quantization()
            .into(),
        head_format: config.linear_format("lm_head.weight"),
        tied_head: config.tie_word_embeddings,
    }
}
