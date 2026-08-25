//! Semantic tensor-parallel placement for the shared Qwen vision tower.

use eredu_nn::NeuralBackend;
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_module_parameter_group,
    partitioned_projection_group, MemberSharding, ParallelPlanError, ParameterGroupSpec,
    ParameterRole, ProjectionSharding,
};

use super::{VisionBlock, VisionConfig, VisionStatic};

fn tensor<'a>(
    layout: &'a eredu_runtime::LocalModelLayout,
    name: &str,
) -> Result<&'a eredu_runtime::LocalTensorLayout, ParallelPlanError> {
    layout.tensor(name).ok_or_else(|| {
        ParallelPlanError::InvalidTensor(format!("missing local vision layout for {name}"))
    })
}

/// Rank-local head and MLP widths resolved from one vision block layout.
pub fn local_block_geometry(
    config: &VisionConfig,
    root: &str,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<(i32, i32), ParallelPlanError> {
    let prefix = rooted(root, &format!("blocks.{layer}"));
    let qkv = tensor(layout, &format!("{prefix}.attn.qkv.weight"))?;
    let fc1 = tensor(layout, &format!("{prefix}.mlp.linear_fc1.weight"))?;
    let qkv_rows = i32::try_from(qkv.local_shape()[0])
        .map_err(|_| ParallelPlanError::InvalidTensor("vision QKV width exceeds i32".into()))?;
    let intermediate = i32::try_from(fc1.local_shape()[0])
        .map_err(|_| ParallelPlanError::InvalidTensor("vision MLP width exceeds i32".into()))?;
    if config.num_heads <= 0 {
        return Err(ParallelPlanError::InvalidTensor(
            "vision head count must be positive".into(),
        ));
    }
    let head_dim = config.hidden_size / config.num_heads;
    if qkv_rows <= 0 || intermediate <= 0 || head_dim <= 0 || qkv_rows % (3 * head_dim) != 0 {
        return Err(ParallelPlanError::InvalidTensor(
            "local vision geometry is zero or QKV splits a head".into(),
        ));
    }
    Ok((qkv_rows / (3 * head_dim), intermediate))
}

/// Declares one vision block's attention, MLP, and replicated norms.
pub fn block_parallel_parameter_groups<B: NeuralBackend>(
    block: &VisionBlock<B>,
    config: &VisionConfig,
    root: &str,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let prefix = rooted(root, &format!("blocks.{layer}"));
    let hidden = usize::try_from(config.hidden_size)
        .map_err(|_| ParallelPlanError::InvalidGroup("vision hidden width exceeds usize".into()))?;
    let heads = usize::try_from(config.num_heads)
        .map_err(|_| ParallelPlanError::InvalidGroup("vision heads exceed usize".into()))?;
    let segments = vec![0..hidden, hidden..2 * hidden, 2 * hidden..3 * hidden];
    Ok(vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.norm1"),
            ParameterRole::Replicated,
            &block.norm1,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
        module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.norm2"),
            ParameterRole::Replicated,
            &block.norm2,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
        partitioned_module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.attention.heads"),
            ParameterRole::AttentionHeads,
            heads,
            &block.attention,
            |metadata, shape| {
                let name = metadata.id.as_str();
                if name.contains("qkv") {
                    Ok(MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: segments.clone(),
                    })
                } else if name.contains("proj") && shape.len() >= 2 {
                    Ok(MemberSharding::Partitioned { axis: 1 })
                } else {
                    Ok(MemberSharding::Replicated)
                }
            },
        )?,
        partitioned_projection_group::<B::Tensor, B::Linear>(
            format!("{prefix}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            &[
                (&block.fc1, ProjectionSharding::Column),
                (&block.fc2, ProjectionSharding::Row),
            ],
            aligned_partition_units(
                &prefix,
                usize::try_from(config.intermediate_size).map_err(|_| {
                    ParallelPlanError::InvalidGroup(
                        "vision intermediate width exceeds usize".into(),
                    )
                })?,
                1,
                1,
            )?,
        )?,
    ])
}

fn merger_groups<B: NeuralBackend>(
    merger: &super::model::Merger<B>,
    root: &str,
    width: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{root}.norm"),
            ParameterRole::Replicated,
            &merger.norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
        partitioned_projection_group::<B::Tensor, B::Linear>(
            format!("{root}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            &[
                (&merger.fc1, ProjectionSharding::Column),
                (&merger.fc2, ProjectionSharding::Row),
            ],
            aligned_partition_units(root, width, 1, 1)?,
        )?,
    ])
}

/// Declares replicated patch/position modules and TP merger channels.
pub fn static_parallel_parameter_groups<B: NeuralBackend>(
    modules: &VisionStatic<B>,
    config: &VisionConfig,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{root}.position"),
            ParameterRole::Replicated,
            &modules.position,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
        module_parameter_group::<B::Tensor, _>(
            format!("{root}.patch"),
            ParameterRole::Replicated,
            &modules.patch,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    let width =
        usize::try_from(config.hidden_size * config.spatial_merge_size * config.spatial_merge_size)
            .map_err(|_| {
                ParallelPlanError::InvalidGroup("vision merger width exceeds usize".into())
            })?;
    groups.extend(merger_groups(
        &modules.merger,
        &rooted(root, "merger"),
        width,
    )?);
    for (index, merger) in modules.deepstack_mergers.iter().enumerate() {
        groups.extend(merger_groups(
            merger,
            &rooted(root, &format!("deepstack_merger_list.{index}")),
            width,
        )?);
    }
    Ok(groups)
}

/// Resolves local main/deepstack merger intermediate widths.
pub fn local_merger_widths(
    config: &VisionConfig,
    root: &str,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<i32>, ParallelPlanError> {
    std::iter::once(rooted(root, "merger"))
        .chain(
            (0..config.deepstack_layer_count())
                .map(|index| rooted(root, &format!("deepstack_merger_list.{index}"))),
        )
        .map(|prefix| {
            let value = tensor(layout, &format!("{prefix}.linear_fc1.weight"))?;
            let width = i32::try_from(value.local_shape()[0]).map_err(|_| {
                ParallelPlanError::InvalidTensor("local vision merger width exceeds i32".into())
            })?;
            if width <= 0 {
                return Err(ParallelPlanError::InvalidTensor(
                    "local vision merger width must be positive".into(),
                ));
            }
            Ok(width)
        })
        .collect()
}

fn rooted(root: &str, relative: &str) -> String {
    if root.is_empty() {
        relative.to_owned()
    } else {
        format!("{root}.{relative}")
    }
}
