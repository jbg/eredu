//! Backend-neutral Nemotron-H architecture.

pub mod attention;
pub mod block;
pub mod checkpoint;
pub mod config;
pub mod declarations;
pub mod mamba;
pub mod mlp;
pub mod model;
pub mod mtp;
pub mod parallel;
mod parameters;
pub use declarations::Declaration;
mod partitioned;

pub use block::{Block, Operator};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, load_time_quantization, normalized_checkpoint_keys,
    static_recipes, translate_gguf_weight_name, unit_recipes, unit_recipes_flat,
    with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, ConfigError, LayerGeometry, LayerPolicy, ModelArgs,
    WeightDtype, PREDICTION_STATE_SEGMENT, TARGET_STATE_SEGMENT,
};
pub use mamba::Mamba2;
pub use mlp::{DenseMlp, SparseMoe};
pub use model::{
    CompleteNemotronKernel, ForwardContext, LayeredModel, TargetBoundary, TargetBoundarySchema,
    TargetPartitionInput, Unit,
};
pub use mtp::{EmbeddedInput, ForwardMode, PredictionUnit, RetainedValues, TargetReadout};
pub use parallel::{
    local_block_geometry, local_geometry, local_state_geometry, partition_local_geometry,
    partition_local_routed_geometry, partitioned_state_layout, LocalGeometry,
    PartitionExpertBankOwnership, PartitionLocalGeometry,
};
pub use partitioned::{NemotronKernel, PartitionedForwardContext, PartitionedLayeredModel};

/// Derives complete expert ownership and rank-local ReLU-squared bank geometry
/// from retained bank declarations and optional validated local geometry.
pub fn expert_realization_plan(
    declaration: &Declaration,
    geometry: Option<&LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>>, eredu_nn::Error> {
    let args = declaration.args();
    if let Some(geometry) = geometry {
        geometry
            .validate_for(declaration)
            .map_err(eredu_nn::Error::backend)?;
    }
    if !args.has_sparse_moe_layers() {
        return Ok(None);
    }
    let banks = declaration.expert_banks();
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H sparse schedule has no declared bank"))?
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
    let layout = declaration.definition().parameters().unit_layout();
    let target_layers = declaration.blocks.len();
    let mut unit_specs = std::collections::BTreeMap::new();
    for (identity, bank) in banks {
        let address = layout
            .address(identity)
            .ok_or_else(|| eredu_nn::Error::backend("unresolved Nemotron-H expert invocation"))?;
        let local = if address.group() == 0 {
            geometry.and_then(|geometry| geometry.target_unit(address.index()))
        } else {
            geometry.and_then(|geometry| geometry.prediction_unit(identity - target_layers))
        };
        let width = match local {
            Some(LayerGeometry::SparseMoe { routed, .. }) => *routed,
            Some(_) => {
                return Err(eredu_nn::Error::backend(
                    "Nemotron-H sparse schedule has non-sparse local geometry",
                ));
            }
            None => bank.intermediate_dimensions(),
        };
        let owner = layout
            .group_id(address.group())
            .ok_or_else(|| eredu_nn::Error::backend("unresolved Nemotron-H expert owner"))?
            .clone();
        unit_specs.insert(
            (owner, address.index()),
            bank.clone().with_group_geometry(local_experts, width)?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(eredu_nn::Error::backend)
}

/// Opaque identity-indexed expert provider descriptor selected by Nemotron-H.
#[derive(Clone)]
pub struct ExpertProviderDescriptor {
    specs: std::collections::BTreeMap<usize, eredu_nn::GroupedRelu2Spec>,
}

impl ExpertProviderDescriptor {
    /// Returns the selected local grouped specification for one routed identity.
    pub fn grouped_spec(&self, identity_layer: usize) -> Option<&eredu_nn::GroupedRelu2Spec> {
        self.specs.get(&identity_layer)
    }
}

fn realized_expert_bank_spec<'a>(
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

/// Projects the architecture realization into an opaque provider descriptor.
pub(crate) fn expert_provider_descriptor(
    plan: &crate::ExpertRealizationPlan<eredu_nn::GroupedRelu2Spec>,
    args: &ModelArgs,
) -> Result<ExpertProviderDescriptor, eredu_nn::Error> {
    let target = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let total = target
        .checked_add(args.mtp_policies().map_err(eredu_nn::Error::backend)?.len())
        .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H expert identity count overflowed"))?;
    let specs = (0..total)
        .filter_map(|identity| {
            realized_expert_bank_spec(plan, args, identity)
                .cloned()
                .map(|spec| (identity, spec))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    if specs.len() != plan.unit_specs().len() {
        return Err(eredu_nn::Error::backend(
            "Nemotron-H expert provider descriptor omitted a realized unit",
        ));
    }
    Ok(ExpertProviderDescriptor { specs })
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

pub(crate) fn static_spec(args: &ModelArgs) -> crate::decoder::StaticModuleSpec {
    let embedding_name = "model.embeddings.weight";
    crate::decoder::StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: embedding_name.into(),
        normalization_weight: "model.norm_f.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.layer_norm_epsilon,
        normalization_offset: 0.0,
        embedding_format: args.weight_quantization_for(embedding_name).into(),
        head_format: args.weight_quantization_for("lm_head.weight").into(),
        tied_head: args.tie_word_embeddings,
    }
}
