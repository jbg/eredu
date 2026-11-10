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
pub(crate) mod pipeline;
pub mod projector;
/// Architecture-owned external-assistant speculative lifecycle.
pub mod speculative;
pub mod text;
pub mod vision;

pub use assistant::{
    assistant_gguf_plan, assistant_safetensors_plan, Assistant, AssistantCompatibility,
    AssistantCompatibilityError, AssistantConfig, AssistantConfigError, AssistantDefinition,
    AssistantOutput, AssistantState,
};
pub use audio::{
    AudioConfig, AudioConfigError, AudioInput, AudioLayer, AudioPolicy, AudioStatic, AudioTower,
};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization, mmproj_gguf_plan,
    safetensors_plan, unit_recipes, with_checkpoint_formats, Gemma4ArtifactConfig, TextGgufNames,
};
pub use config::{ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs};
pub use family::{FamilyConfig, FamilyConfigError, FamilyPolicy};
pub use graph::{component_graph, state_layout, ComponentOptions};
pub use ingress::{
    AudioIngressBatchPlan, AudioIngressPartPlan, IngressPlanError, VisionIngressBatchPlan,
    VisionIngressPartPlan, VISION_ATTENTION_INVALID_LOGIT,
};
pub use media::ModalityProjector;
pub use model::{
    component_definition, parameter_description, prepare_composite_ingress, DecoderInputPart,
    ForwardContext, GemmaKernel, LayeredModel, ModelInput, PreparedCompositeIngress, StaticModules,
    TextBoundary, TextBoundarySchema, Unit, AUDIO_EXECUTION_GROUP, TEXT_EXECUTION_GROUP,
    VISION_EXECUTION_GROUP,
};
pub use parallel::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    local_block_args, local_geometry, partition_local_geometry, static_parameter_groups,
    vision_layer_parameter_groups, vision_static_parameter_groups, LocalGeometry,
    PartitionLocalFoundation, PartitionLocalGeometry,
};
pub use projector::{
    family_from_gguf_metadata, translate_mmproj_weight_name, validate_projector_identity,
};
pub use text::{Attention, AttentionInput, BlockInput, DenseBlock, SharedAttentionStates};
pub use vision::{
    MediaLinear, PatchEmbedder, VisionConfig, VisionConfigError, VisionInput, VisionLayer,
    VisionOwner, VisionPolicy, VisionState, VisionStatic, VisionTower,
};

/// Derives complete expert ownership and rank-local bank geometry from Gemma 4.
pub fn expert_realization_plan(
    args: &FamilyConfig,
    geometry: Option<&LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, eredu_nn::Error>
{
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_for(args)
            .map_err(eredu_nn::Error::backend_source)?;
    }
    let declarations = args.routed_layers();
    let Some((_, first)) = args.routed_layers().next() else {
        return Ok(None);
    };
    let global_experts = first.routed.experts.group_count() as usize;
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
    for (layer, declaration) in declarations {
        let bank = &declaration.routed.experts;
        let width = match geometry.and_then(|geometry| geometry.text_block(layer)) {
            Some(local) => local.moe_intermediate_size.ok_or_else(|| {
                eredu_nn::Error::backend("Gemma 4 sparse config has no expert width")
            })?,
            None => bank.intermediate_dimensions(),
        };
        unit_specs.insert(
            (owner_group.clone(), layer),
            declaration.local_bank(local_experts, width)?,
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
