//! Shared neutral Muse decoder, vision, preparation, and placement mechanisms.

pub mod assistant;
pub mod audio;
pub mod checkpoint;
pub mod config;
pub mod equations;
pub mod graph;
pub mod media_companion;
pub mod model;
pub mod parallel;
/// Architecture-owned external DFlash speculative lifecycle.
pub mod speculative;
pub mod text;
pub mod vision;

pub use assistant::{
    translate_dflash_gguf_weight_name, DFlash, DFlashCompatibility, DFlashCompatibilityError,
    DFlashConfig, DFlashConfigError, DFlashContext, DFlashLayerContext,
};
pub use audio::{
    AudioConfig, AudioConvBlock, AudioPatchEmbedding, AudioPolicy, AudioProjection, AudioPyramid,
    AudioStage, AudioState, AudioUnit, AudioUnitKind,
};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    projector_gguf_name_mapping, projector_gguf_plan, resolve_safetensors_config, safetensors_plan,
    safetensors_recipes, static_safetensors_recipes, unit_safetensors_recipes,
    with_checkpoint_formats, ArtifactConfig, TextGgufNames,
};
pub use config::{
    AttentionEquation, BlockNormalization, ConfigError, DecoderConfig, DecoderPolicy,
    MediaEmbeddingPrecision, MediaInputPrecision, MuseIdentity, PerceptionNormalization,
    ResidualEquation, RoutedFeedForwardEquation, VisionArithmetic, VisionAttentionPolicy,
    VisionConfig, VisionInputPolicy, VisionPolicy, VisionPositionLayout, VocabularyArithmetic,
    WeightConvention,
};
pub use equations::{
    BranchNorm, LearnedResidualCombination, LearnedRmsNorm, ResidualCombination, RmsGain,
    RmsNormEquation, StoredResidualCombination,
};
pub use graph::{component_graph, state_layout};
pub use model::{
    component_definition, prepare_composite_ingress, DecoderInputPart, ForwardContext,
    LayeredModel, ModelInput, MuseKernel, PreparedCompositeIngress, TextPartitionInput, Unit,
    TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
pub use parallel::{
    layer_parameter_groups, local_decoder_config, local_geometry, partition_local_geometry,
    routed_tensor_reductions, static_parameter_groups, vision_layer_parameter_groups,
    vision_parameter_groups, vision_static_parameter_groups, LocalGeometry,
    PartitionLocalFoundation, PartitionLocalGeometry,
};
pub use text::{Attention, FeedForward, Mlp, SparseMoe, StaticModules, TransformerBlock};
pub use vision::{
    VisionBlock, VisionInput, VisionState, VisionStatic, VisionStaticSelection, VisionTower,
};

/// Declares canonical expert ownership and TP/EP-local bank geometry without native resources.
pub fn expert_realization_plan(
    args: &DecoderConfig,
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
    if !args.is_moe() {
        return Ok(None);
    }
    let declarations = args.routed_layers();
    let global_experts = declarations
        .values()
        .next()
        .ok_or_else(|| eredu_nn::Error::backend("routed decoder has no declared expert banks"))?
        .bank()
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
    let owner_group = eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP)
        .map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for (layer, declaration) in declarations {
        let bank = declaration.bank().clone();
        let width = geometry
            .and_then(|geometry| geometry.text_block(*layer))
            .map_or(bank.intermediate_dimensions(), |local| {
                local.moe_intermediate_size
            });
        unit_specs.insert(
            (owner_group.clone(), *layer),
            bank.with_group_geometry(local_experts, width)?,
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
    let layer_count = state_layout(args)
        .map_err(|error| ConfigError::Invalid(error.to_string()))?
        .len();
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| ConfigError::Invalid("Muse-Glimmer owned state range overflowed".into()))?;
    if global_layer_end > layer_count {
        return Err(ConfigError::Invalid(format!(
            "Muse-Glimmer owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        args.identity.model_type(),
        args.identity.text_model_type(),
        args.architecture_fingerprint(),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(|error| ConfigError::Invalid(error.to_string()))
}
