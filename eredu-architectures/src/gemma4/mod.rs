//! Neutral Gemma 4 family configuration and checkpoint policy.

pub mod assistant;
pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod family;
pub mod graph;
pub mod ingress;
pub mod media;
pub mod model;
pub mod parallel;
pub mod projector;
pub mod text;
pub mod vision;

pub use assistant::{
    assistant_gguf_plan, assistant_safetensors_plan, translate_assistant_gguf_weight_name,
    Assistant, AssistantCompatibility, AssistantCompatibilityError, AssistantConfig,
    AssistantConfigError, AssistantOutput, AssistantState,
};
pub use audio::{AudioConfig, AudioConfigError, AudioInput, AudioLayer, AudioStatic, AudioTower};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization, mmproj_gguf_plan,
    safetensors_plan, translate_gguf_weight_name, unit_recipes, with_checkpoint_formats,
    Gemma4ArtifactConfig,
};
pub use config::{ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs};
pub use family::{FamilyConfig, FamilyConfigError};
pub use graph::{component_graph, state_layout, ComponentOptions};
pub use ingress::{
    AudioIngressBatchPlan, AudioIngressPartPlan, IngressPlanError, VisionIngressBatchPlan,
    VisionIngressPartPlan, VISION_ATTENTION_INVALID_LOGIT,
};
pub use media::ModalityProjector;
pub use model::{
    DecoderInputPart, ForwardContext, LayeredModel, ModelInput, StaticModules, TextBoundary,
    TextBoundarySchema, Unit, AUDIO_EXECUTION_GROUP, TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
pub use parallel::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    local_block_args, local_geometry, modality_projection_parameter_groups,
    static_parameter_groups, vision_layer_parameter_groups, vision_static_parameter_groups,
    LocalGeometry,
};
pub use projector::{
    family_from_gguf_metadata, translate_mmproj_weight_name, validate_projector_identity,
};
pub use text::{Attention, AttentionInput, BlockInput, DenseBlock, SharedAttentionStates};
pub use vision::{
    ClippedLinear, PatchEmbedder, VisionConfig, VisionConfigError, VisionInput, VisionLayer,
    VisionState, VisionStatic, VisionTower,
};

/// Derives complete expert ownership and rank-local bank geometry from Gemma 4.
pub fn expert_realization_plan<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    architecture: &LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    let args = architecture.args();
    let sparse_layers = args
        .text
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| {
            (policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe).then_some(layer)
        })
        .collect::<Vec<_>>();
    if sparse_layers.is_empty() {
        return Ok(None);
    }
    let global_experts =
        usize::try_from(args.text.num_experts.ok_or_else(|| {
            eredu_nn::Error::backend("Gemma 4 sparse config has no expert count")
        })?)
        .map_err(eredu_nn::Error::backend)?;
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
    for layer in sparse_layers {
        let local = architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.text_block(layer))
            .unwrap_or(&args.text);
        let width = local
            .moe_intermediate_size
            .ok_or_else(|| eredu_nn::Error::backend("Gemma 4 sparse config has no expert width"))?;
        unit_specs.insert(
            (owner_group.clone(), layer),
            text::localized_expert_bank_spec(&args.text, layer, local_experts, width)?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}

/// Declares Gemma 4 cache identity independently of concrete state storage.
pub fn state_identity(
    args: &FamilyConfig,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, FamilyConfigError> {
    args.validate()?;
    topology
        .validate()
        .map_err(|error| FamilyConfigError::Invalid(error.to_string()))?;
    let layer_count = args.text.num_hidden_layers();
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| FamilyConfigError::Invalid("Gemma 4 owned state range overflowed".into()))?;
    if global_layer_end > layer_count {
        return Err(FamilyConfigError::Invalid(format!(
            "Gemma 4 owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "gemma4",
        args.effective_model_type(),
        args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(|error| FamilyConfigError::Invalid(error.to_string()))
}
