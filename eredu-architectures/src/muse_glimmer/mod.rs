//! Neutral Muse-Glimmer multimodal decoder family.

pub mod assistant;
pub mod checkpoint;
pub mod config;
pub mod graph;
pub mod model;
pub mod parallel;
pub mod text;
pub mod vision;

pub use assistant::{
    dflash_gguf_plan, dflash_safetensors_plan, translate_dflash_gguf_weight_name, DFlash,
    DFlashCompatibility, DFlashCompatibilityError, DFlashConfig, DFlashConfigError, DFlashContext,
    DFlashLayerContext,
};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    projector_gguf_plan, safetensors_parameter_names, safetensors_plan, safetensors_recipes,
    static_safetensors_recipes, translate_projector_gguf_name, translate_text_gguf_name,
    unit_safetensors_recipes, ArtifactConfig,
};
pub use config::{
    ConfigError, DecoderConfig, VisionAttentionPolicy, VisionConfig, WeightConvention,
};
pub use graph::{component_graph, state_layout};
pub use model::{
    DecoderInputPart, ForwardContext, LayeredModel, ModelInput, TextPartitionInput, Unit,
};
pub use parallel::{
    layer_parameter_groups, local_decoder_config, local_geometry, static_parameter_groups,
    vision_layer_parameter_groups, vision_parameter_groups, vision_static_parameter_groups,
    LocalGeometry,
};
pub use text::{
    Attention, CenteredRmsNorm, FeedForward, Mlp, SparseMoe, StaticModules, TransformerBlock,
};
pub use vision::{VisionBlock, VisionInput, VisionState, VisionStatic, VisionTower};

/// Derives complete expert ownership and rank-local bank geometry from Muse-Glimmer.
pub fn expert_realization_plan<B: eredu_nn::RoutedNeuralBackend>(
    architecture: &LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<
    Option<crate::ExpertRealizationPlan<eredu_nn::GatedProductExpertBankSpec>>,
    eredu_nn::Error,
> {
    let args = architecture.args();
    if !args.is_moe() {
        return Ok(None);
    }
    let global_experts = usize::try_from(args.num_experts).map_err(eredu_nn::Error::backend)?;
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
    let layers = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for layer in 0..layers {
        let local = architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.text_block(layer))
            .unwrap_or(args);
        unit_specs.insert(
            (owner_group.clone(), layer),
            text::localized_expert_bank_spec(
                args,
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

/// Declares Muse-Glimmer cache identity independently of concrete state storage.
pub fn state_identity(
    args: &DecoderConfig,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, ConfigError> {
    args.validate()?;
    topology
        .validate()
        .map_err(|error| ConfigError::Invalid(error.to_string()))?;
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| ConfigError::Invalid("Muse-Glimmer layer count exceeds usize".into()))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| ConfigError::Invalid("Muse-Glimmer owned state range overflowed".into()))?;
    if global_layer_end > layer_count {
        return Err(ConfigError::Invalid(format!(
            "Muse-Glimmer owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(eredu_runtime::ModelStateIdentity {
        model_family: "muse_glimmer".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        topology,
    })
}
