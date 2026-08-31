//! Neutral Qwen3-VL configuration, state, and multimodal position policy.

mod checkpoint;
mod config;
mod model;
mod parallel;
mod positions;

pub use checkpoint::{
    load_time_quantization, normalize_text_weight_formats, projector_gguf_plan,
    rank_local_unit_recipes, safetensors_plan, static_recipes, translate_text_gguf_weight_name,
    translate_vision_gguf_weight_name, unit_recipes, with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_value, model_args_from_gguf_parts,
    prompt_cache_architecture_fingerprint, state_identity, state_layout,
    state_layout_with_key_value_heads, vision_config_from_gguf_catalog, GgufModelArgs, ModelArgs,
    VlConfigError,
};
pub use model::{
    ForwardContext, InputPart, LayeredModel, ModelInput, PipelineBoundary, PipelineBoundarySchema,
    PipelinePartitionInput, PipelinePrepared, PipelineVisionState, StaticModules, Unit,
    TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
pub use parallel::{local_geometry, LocalGeometry};
pub use positions::{
    mrope_embeddings, mrope_values, multimodal_position_ids, position_ids_tensor, PositionPart,
};

/// Derives complete expert ownership and rank-local text-bank geometry from Qwen3-VL.
pub fn expert_realization_plan<B: eredu_nn::RoutedNeuralBackend>(
    architecture: &LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<
    Option<crate::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>>,
    eredu_nn::Error,
> {
    let args = architecture.args();
    if !args.text.is_moe() {
        return Ok(None);
    }
    let global_experts =
        usize::try_from(args.text.num_experts).map_err(eredu_nn::Error::backend)?;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
            false,
        )
        .map_err(eredu_nn::Error::backend)?
        .len(),
    )
    .map_err(eredu_nn::Error::backend)?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP)
        .map_err(eredu_nn::Error::backend)?;
    let geometry = architecture.shared_parallel_geometry();
    let mut unit_specs = std::collections::BTreeMap::new();
    for layer in
        0..usize::try_from(args.text.num_hidden_layers).map_err(eredu_nn::Error::backend)?
    {
        let local = geometry
            .as_deref()
            .and_then(|geometry| geometry.text().block(layer))
            .unwrap_or(&args.text);
        unit_specs.insert(
            (owner_group.clone(), layer),
            super::moe::localized_expert_bank_spec(
                &args.text,
                layer,
                local_experts,
                local.moe_intermediate_size,
            )?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}
