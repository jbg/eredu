//! Semantic TP/PP/EP placement for neutral GPT-OSS blocks.

use eredu_nn::RoutedNeuralBackend;
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

use crate::decoder::{block_common_parallel_parameter_groups, static_parallel_parameter_groups};

use super::{block::TransformerBlock, config::ModelArgs};

fn expert_member_sharding(
    name: &str,
    shape: &[usize],
    intermediate: usize,
) -> Result<MemberSharding, ParallelPlanError> {
    if name.contains("gate_up_proj") {
        if shape.len() < 2 {
            return Err(ParallelPlanError::InvalidTensor(format!(
                "GPT-OSS gate/up expert member {name} has rank {}",
                shape.len()
            )));
        }
        return Ok(MemberSharding::PartitionedSegments {
            axis: 1,
            segments: vec![0..intermediate, intermediate..2 * intermediate],
        });
    }
    if name.contains("down_proj") {
        return match shape.len() {
            2 if name.ends_with("down_proj_bias") => Ok(MemberSharding::Replicated),
            rank if rank >= 3 => Ok(MemberSharding::Partitioned { axis: 2 }),
            _ => Err(ParallelPlanError::InvalidTensor(format!(
                "GPT-OSS down expert member {name} has unsupported shape {shape:?}"
            ))),
        };
    }
    Err(ParallelPlanError::InvalidTensor(format!(
        "unexpected GPT-OSS expert-bank parameter {name}"
    )))
}

/// Declares attention, sinks, router, biased expert bank, and norms for a layer.
pub fn layer_parallel_parameter_groups<B: RoutedNeuralBackend>(
    block: &TransformerBlock<B>,
    args: &ModelArgs,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = block_common_parallel_parameter_groups(block, args, layer)?;
    let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
    groups.push(module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.router"),
        ParameterRole::Replicated,
        &block.mlp.router,
        |_, _| Ok(MemberSharding::Replicated),
    )?);

    let intermediate = usize::try_from(args.intermediate_size).map_err(|_| {
        ParallelPlanError::InvalidGroup("GPT-OSS expert width exceeds usize".into())
    })?;
    let units = aligned_partition_units(&format!("{prefix}.experts"), intermediate, 1, 32)?;
    groups.push(partitioned_module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.experts.intermediate"),
        ParameterRole::ExpertIntermediate,
        units,
        &block.mlp.experts,
        |metadata, shape| expert_member_sharding(metadata.id.as_str(), shape, intermediate),
    )?);
    Ok(groups)
}

/// Declares pinned embedding, final norm, and separate head placement.
pub fn static_parameter_groups<B: RoutedNeuralBackend>(
    modules: &crate::decoder::StaticModules<B>,
    args: &ModelArgs,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    static_parallel_parameter_groups::<B>(
        &modules.embeddings,
        &modules.norm,
        modules.lm_head.as_ref(),
        &args.parameter_root,
    )
}

/// Derives TP-local attention and expert geometry from semantic placement.
pub fn local_block_args(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", args.parameter_root);
    let tensor = |identity: &str| {
        layout.tensor(identity).ok_or_else(|| {
            ParallelPlanError::InvalidTensor(format!("missing local layout for {identity}"))
        })
    };
    let query = tensor(&format!("{prefix}.self_attn.q_proj.weight"))?;
    let key = tensor(&format!("{prefix}.self_attn.k_proj.weight"))?;
    let query_width = *query.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local GPT-OSS query projection is scalar".into())
    })?;
    let key_width = *key.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local GPT-OSS key projection is scalar".into())
    })?;
    let head_dim = usize::try_from(args.head_dim)
        .map_err(|_| ParallelPlanError::InvalidTensor("GPT-OSS head width exceeds usize".into()))?;
    if head_dim == 0 || query_width % head_dim != 0 || key_width % head_dim != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local GPT-OSS attention widths q={query_width}, k={key_width} split head width {head_dim}"
        )));
    }

    let gate_up = tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))?;
    let gate_up_shape = gate_up.local_shape();
    if gate_up_shape.len() < 2 || gate_up_shape[1] % 2 != 0 {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local GPT-OSS gate/up shape {gate_up_shape:?} is not [experts, 2*intermediate, ...]"
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = i32::try_from(query_width / head_dim).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS query-head count exceeds i32".into())
    })?;
    local.num_key_value_heads = i32::try_from(key_width / head_dim).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS key/value-head count exceeds i32".into())
    })?;
    local.num_local_experts = i32::try_from(gate_up_shape[0]).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS expert count exceeds i32".into())
    })?;
    local.intermediate_size = i32::try_from(gate_up_shape[1] / 2).map_err(|_| {
        ParallelPlanError::InvalidTensor("local GPT-OSS expert width exceeds i32".into())
    })?;
    Ok(local)
}

/// Returns TP-local key/value head geometry in global layer order.
pub fn local_key_value_heads(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<i32>, ParallelPlanError> {
    (0..args.num_hidden_layers as usize)
        .map(|layer| Ok(local_block_args(args, layer, layout)?.num_key_value_heads))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biased_packed_experts_use_semantic_axes() {
        assert_eq!(
            expert_member_sharding("gate_up_proj_bias", &[8, 128], 64).unwrap(),
            MemberSharding::PartitionedSegments {
                axis: 1,
                segments: vec![0..64, 64..128],
            }
        );
        assert_eq!(
            expert_member_sharding("down_proj_bias", &[8, 32], 64).unwrap(),
            MemberSharding::Replicated
        );
        assert_eq!(
            expert_member_sharding("down_proj_scales", &[8, 32, 2], 64).unwrap(),
            MemberSharding::Partitioned { axis: 2 }
        );
    }
}
