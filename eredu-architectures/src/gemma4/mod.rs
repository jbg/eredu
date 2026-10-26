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
    assistant_gguf_plan, assistant_safetensors_plan, translate_assistant_gguf_weight_name,
    Assistant, AssistantCompatibility, AssistantCompatibilityError, AssistantConfig,
    AssistantConfigError, AssistantOutput, AssistantState,
};
pub use audio::{AudioConfig, AudioConfigError, AudioInput, AudioLayer, AudioStatic, AudioTower};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization, mmproj_gguf_plan,
    safetensors_plan, translate_family_gguf_weight_name, translate_gguf_weight_name, unit_recipes,
    with_checkpoint_formats, Gemma4ArtifactConfig,
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
    component_definition, parameter_description, prepare_composite_ingress, DecoderInputPart,
    ForwardContext, GemmaKernel, LayeredModel, ModelInput, PreparedCompositeIngress, StaticModules,
    TextBoundary, TextBoundarySchema, Unit, AUDIO_EXECUTION_GROUP, TEXT_EXECUTION_GROUP,
    VISION_EXECUTION_GROUP,
};
pub use parallel::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    local_block_args, local_geometry, modality_projection_parameter_groups,
    partition_local_geometry, static_parameter_groups, vision_layer_parameter_groups,
    vision_static_parameter_groups, LocalGeometry, PartitionLocalFoundation,
    PartitionLocalGeometry,
};
pub use projector::{
    family_from_gguf_metadata, translate_mmproj_weight_name, validate_projector_identity,
};
pub use text::{Attention, AttentionInput, BlockInput, DenseBlock, SharedAttentionStates};
pub use vision::{
    MediaLinear, PatchEmbedder, VisionConfig, VisionConfigError, VisionInput, VisionLayer,
    VisionState, VisionStatic, VisionTower,
};

/// Ordinary pinned text semantics shared by catalogs and native construction.
pub(crate) fn static_construction_specs(
    args: &ModelArgs,
) -> Result<crate::decoder::StaticModuleConstruction, eredu_nn::Error> {
    use eredu_nn::{EmbeddingSpec, Error, LinearSpec, NormalizationConstructionSpec, ParameterSpec};
    let embedding_weight = "model.language_model.embed_tokens.weight";
    let head_weight = "lm_head.weight";
    let embedding = EmbeddingSpec {
        vocabulary: args.vocab_size,
        dimensions: args.hidden_size,
        weight: ParameterSpec::trainable(embedding_weight).map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            embedding_weight,
            args.linear_format_for(embedding_weight),
        )?,
    };
    let head = (!args.tie_word_embeddings)
        .then(|| -> Result<_, Error> {
            Ok(LinearSpec {
                input: args.hidden_size,
                output: args.vocab_size,
                weight: ParameterSpec::trainable(head_weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    head_weight,
                    args.linear_format_for(head_weight),
                )?,
            })
        })
        .transpose()?;
    let normalization = NormalizationConstructionSpec::learned(
        args.hidden_size,
        args.rms_norm_eps,
        ParameterSpec::trainable("model.language_model.norm.weight").map_err(Error::backend)?,
    );
    crate::decoder::StaticModuleConstruction::new(embedding, head, normalization)
}

/// Derives complete expert ownership and rank-local bank geometry from Gemma 4.
pub fn expert_realization_plan<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    architecture: &GemmaKernel<B>,
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
