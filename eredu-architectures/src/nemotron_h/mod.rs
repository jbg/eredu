//! Backend-neutral Nemotron-H architecture.

pub mod attention;
pub mod block;
pub mod checkpoint;
pub mod config;
pub mod mamba;
pub mod mlp;
pub mod model;
pub mod mtp;
pub mod parallel;

pub use attention::{new_attention, new_attention_at};
pub use block::{Block, Operator};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    normalized_checkpoint_keys, safetensors_plan, static_recipes, translate_gguf_weight_name,
    unit_recipes, unit_recipes_flat, with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry, ConfigError,
    LayerGeometry, LayerPolicy, ModelArgs, WeightDtype, PREDICTION_STATE_SEGMENT,
    TARGET_STATE_SEGMENT,
};
pub use mamba::Mamba2;
pub use mlp::{expert_bank_spec, localized_expert_bank_spec, DenseMlp, SparseMoe};
pub use model::{
    ForwardContext, LayeredModel, TargetBoundary, TargetBoundarySchema, TargetPartitionInput, Unit,
};
pub use mtp::{EmbeddedInput, ForwardMode, PredictionUnit, RetainedValues};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_geometry, local_geometry, local_state_geometry,
    static_parallel_parameter_groups, unit_parallel_parameter_groups, LocalGeometry,
};

/// Derives complete expert ownership and rank-local ReLU-squared bank geometry.
pub fn expert_realization_plan<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    architecture: &LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>>, eredu_nn::Error> {
    let args = architecture.args();
    if !args.has_sparse_moe_layers() {
        return Ok(None);
    }
    let global_experts =
        usize::try_from(args.n_routed_experts).map_err(eredu_nn::Error::backend)?;
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
    let target_group =
        eredu_runtime::ExecutionGroupId::new("target").map_err(eredu_nn::Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        if policy != LayerPolicy::SparseMoe {
            continue;
        }
        let width = match architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.target_unit(layer))
        {
            Some(LayerGeometry::SparseMoe { routed, .. }) => *routed,
            Some(_) => {
                return Err(eredu_nn::Error::backend(
                    "Nemotron-H sparse schedule has non-sparse local geometry",
                ))
            }
            None => args.moe_intermediate_size,
        };
        unit_specs.insert(
            (target_group.clone(), layer),
            localized_expert_bank_spec(args, layer, local_experts, width)?,
        );
    }
    let target_layers =
        usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let prediction_steps =
        usize::try_from(args.num_nextn_predict_layers).map_err(eredu_nn::Error::backend)?;
    let prediction_policies = args.mtp_policies().map_err(eredu_nn::Error::backend)?;
    let pattern = if prediction_steps == 0 {
        0
    } else {
        prediction_policies
            .len()
            .checked_div(prediction_steps)
            .filter(|pattern| *pattern > 0)
            .ok_or_else(|| {
                eredu_nn::Error::backend("Nemotron-H MTP operator pattern cannot be empty")
            })?
    };
    for (physical, policy) in prediction_policies.into_iter().enumerate() {
        if policy != LayerPolicy::SparseMoe {
            continue;
        }
        let depth = physical / pattern;
        let unit = physical % pattern;
        let group = eredu_runtime::ExecutionGroupId::new(format!("mtp.{depth}"))
            .map_err(eredu_nn::Error::backend)?;
        let width = match architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.prediction_unit(physical))
        {
            Some(LayerGeometry::SparseMoe { routed, .. }) => *routed,
            Some(_) => {
                return Err(eredu_nn::Error::backend(
                    "Nemotron-H sparse MTP schedule has non-sparse local geometry",
                ))
            }
            None => args.moe_intermediate_size,
        };
        unit_specs.insert(
            (group, unit),
            localized_expert_bank_spec(args, target_layers + physical, local_experts, width)?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}

/// Resolves an identity-layer callback to its architecture-owned local bank spec.
pub fn realized_expert_bank_spec<'a>(
    plan: &'a crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
    args: &ModelArgs,
    identity_layer: usize,
) -> Option<&'a eredu_nn::GroupedRelu2Spec> {
    let target = usize::try_from(args.num_hidden_layers).ok()?;
    if identity_layer < target {
        return plan.unit_spec("target", identity_layer);
    }
    let physical = identity_layer.checked_sub(target)?;
    let prediction_steps = usize::try_from(args.num_nextn_predict_layers).ok()?;
    let policies = args.mtp_policies().ok()?;
    let pattern = policies
        .len()
        .checked_div(prediction_steps)
        .filter(|value| *value > 0)?;
    plan.unit_spec(&format!("mtp.{}", physical / pattern), physical % pattern)
}

/// Declares cache identity independently of its backend realization.
pub fn state_identity(
    args: &ModelArgs,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
    let target = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let total = target
        .checked_add(args.mtp_policies().map_err(eredu_nn::Error::backend)?.len())
        .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H state layer count overflowed"))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H owned state range overflowed"))?;
    if global_layer_end > total {
        return Err(eredu_nn::Error::backend(format!(
            "Nemotron-H owns state layers {global_layer_start}..{global_layer_end}, outside {total} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "nemotron_h",
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        total,
        global_layer_start,
        0,
        topology,
    )
    .map_err(eredu_nn::Error::backend)
}
