//! Neutral Inkling multimodal decoder family.

pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod graph;
pub mod model;
pub mod mtp;
pub mod parallel;
pub mod text;
pub mod vision;

pub use audio::{AudioInput, AudioTower};
pub use checkpoint::{
    dense_w13_recipes, expert_residency_catalog, expert_w13_recipe, gguf_plan,
    load_time_quantization, mmproj_gguf_plan, normalize_gguf_weight_formats, safetensors_aliases,
    safetensors_plan, safetensors_recipes, static_safetensors_recipes, translate_gguf_weight_name,
    translate_gguf_weight_name_for_model, translate_mmproj_weight_name, unit_safetensors_recipes,
    with_checkpoint_formats, DenseW13Recipes, ParameterAlias,
};
pub use config::{
    AudioConfig, ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs, MtpConfig, TextArgs,
    VisionConfig,
};
pub use graph::{
    component_graph, composite_state_layout, mtp_state_layout, parallel_state_layout, state_layout,
    PREDICTION_STATE_SEGMENT, TARGET_STATE_SEGMENT,
};
pub use model::{
    state_identity, DecoderInputPart, ForwardContext, InklingStateLayouts, LayeredModel,
    ModelInput, PartitionMtpOutput, StaticModules, TextPartitionInput, Unit, MTP_STATIC_ROLE,
    TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
pub use mtp::{MtpDepth, MtpModel, MtpOutput};
pub use parallel::{
    layer_parameter_groups, local_geometry, local_text_args, mtp_parameter_groups,
    static_parameter_groups, vision_layer_parameter_groups, LocalGeometry,
};
pub use text::{
    convolution_history_shape, Attention, ConvolutionState, DecoderLayer, FeedForward, LayerState,
    TextModel,
};
pub use vision::{VisionLayer, VisionStatic, VisionTower};

/// Rank-local construction specifications for one Inkling sparse unit.
#[derive(Debug, Clone)]
pub struct ExpertBankRealization {
    /// Selectable experts partitioned over the expert axis.
    pub routed: eredu_nn::GroupedGatedProductSpec,
    /// Always-on shared experts replicated over the expert axis.
    pub shared: eredu_nn::GroupedGatedProductSpec,
}

/// Derives complete routed-expert ownership and both rank-local bank geometries.
pub fn expert_realization_plan<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    architecture: &LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<ExpertBankRealization>>, eredu_nn::Error> {
    let args = architecture.args();
    let sparse = args
        .text_config
        .layer_schedule
        .iter()
        .any(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe);
    if !sparse {
        return Ok(None);
    }
    let global_experts =
        usize::try_from(args.text_config.n_routed_experts).map_err(eredu_nn::Error::backend)?;
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
    let owner_group = eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP)
        .map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for (layer, policy) in args.text_config.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let local = architecture
            .shared_parallel_geometry()
            .and_then(|geometry| geometry.text_layer(layer).cloned())
            .unwrap_or_else(|| args.text_config.clone());
        let (routed, shared) =
            text::localized_expert_bank_specs(args, layer, &local, local_experts)?;
        unit_specs.insert(
            (owner_group.clone(), layer),
            ExpertBankRealization { routed, shared },
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}
