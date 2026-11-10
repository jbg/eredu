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
    expert_residency_catalog, gguf_plan, load_time_quantization, mmproj_gguf_plan,
    mtp_safetensors_recipes, normalize_gguf_weight_formats, safetensors_aliases, safetensors_plan,
    safetensors_recipes, static_safetensors_recipes, translate_gguf_weight_name,
    translate_gguf_weight_name_for_model, translate_mmproj_weight_name, unit_safetensors_recipes,
    with_checkpoint_formats, ParameterAlias,
};
pub use config::{
    AudioConfig, AudioPolicy, ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs, ModelPolicy,
    MtpConfig, TextArgs, VisionConfig, VisionPolicy,
};
pub use graph::{
    component_graph, composite_state_layout, mtp_state_layout, parallel_state_layout, state_layout,
    PREDICTION_STATE_SEGMENT, TARGET_STATE_SEGMENT,
};
pub use model::{
    component_definition, parameter_description, prepare_input, state_identity, DecoderInputPart,
    ForwardContext, InklingKernel, InklingStateLayouts, LayeredModel, ModelInput,
    PartitionMtpOutput, PreparedInput, StaticModules, TextPartitionInput, Unit,
    AUDIO_EXECUTION_GROUP, MTP_STATIC_ROLE, TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
pub use mtp::{MtpDepth, MtpModel, MtpOutput, MtpShared};
pub use parallel::{
    layer_parameter_groups, local_geometry, local_text_args, mtp_parameter_groups,
    partition_local_geometry, static_parameter_groups, vision_layer_parameter_groups,
    LocalGeometry, PartitionLocalFoundation, PartitionLocalGeometry,
};
pub use text::{Attention, ConvolutionState, DecoderLayer, FeedForward, LayerState, TextModel};
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
pub fn expert_realization_plan(
    args: &ModelArgs,
    geometry: Option<&LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<ExpertBankRealization>>, eredu_nn::Error> {
    args.validate().map_err(eredu_nn::Error::backend_source)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_for(args)
            .map_err(eredu_nn::Error::backend_source)?;
    }
    let Some((_, first)) = args.sparse_layers().next() else {
        return Ok(None);
    };
    let global_experts = first.routed.group_count() as usize;
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
    for (layer, parameters) in args.sparse_layers() {
        let width = geometry
            .and_then(|geometry| geometry.text_layer(layer))
            .map_or(parameters.routed.intermediate_dimensions(), |local| {
                local.moe_intermediate_size()
            });
        let shared_count = parameters.shared.group_count();
        let routed = parameters
            .routed
            .clone()
            .with_group_geometry(local_experts, width)?;
        let shared = parameters
            .shared
            .clone()
            .with_group_geometry(shared_count, width)?;
        unit_specs.insert(
            (owner_group.clone(), layer),
            ExpertBankRealization { routed, shared },
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}

/// Projects routed and shared cache-layer identities for replicated expert residency.
pub(crate) fn replicated_expert_realization_plan(
    args: &ModelArgs,
) -> Result<
    (
        crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
        std::collections::BTreeMap<usize, usize>,
    ),
    eredu_nn::Error,
> {
    let global_experts = args
        .sparse_layers()
        .next()
        .ok_or_else(|| {
            eredu_nn::Error::backend("Inkling expert realization requires a sparse invocation")
        })?
        .1
        .routed
        .group_count() as usize;
    let owner_group = eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP)
        .map_err(eredu_nn::Error::backend)?;
    let mut specs = std::collections::BTreeMap::new();
    let mut routes = std::collections::BTreeMap::new();
    for (layer, parameters) in args.sparse_layers() {
        let shared_unit = text::shared_expert_cache_layer(&args.text_config, layer)?;
        routes.insert(layer, parameters.router.selection().top_k() as usize);
        routes.insert(
            shared_unit,
            parameters.router.selection().always_on_groups() as usize,
        );
        specs.insert((owner_group.clone(), layer), parameters.routed.clone());
        specs.insert(
            (owner_group.clone(), shared_unit),
            parameters.shared.clone(),
        );
    }
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(eredu_nn::Error::backend)?,
        0,
    )
    .map_err(eredu_nn::Error::backend)?;
    crate::ExpertRealizationPlan::balanced(global_experts, topology, specs)
        .map(|plan| (plan, routes))
        .map_err(eredu_nn::Error::backend)
}
