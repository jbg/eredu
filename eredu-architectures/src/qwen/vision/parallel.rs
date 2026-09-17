//! Semantic tensor-parallel placement for the shared Qwen vision tower.

use eredu_nn::NeuralBackend;
use eredu_runtime::{
    MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole, ProjectionSharding,
};

use super::{VisionBlock, VisionConfig, VisionStatic};
use crate::decoder::parameter_metadata::{
    DeclarationDestination as Destination, ParameterGroupError as GroupError,
};

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
pub fn block_parallel_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    block: &VisionBlock<B>,
    config: &VisionConfig,
    root: &str,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    block_groups(block, config, root, layer, Destination(None)).map_err(GroupError::ordinary)
}

pub(crate) fn block_parallel_parameter_groups_with_metadata<
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    block: &VisionBlock<B>,
    config: &VisionConfig,
    root: &str,
    layer: usize,
    context: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<Vec<ParameterGroupSpec>, eredu_nn::Error> {
    block_groups(block, config, root, layer, Destination(context)).map_err(GroupError::into_neural)
}

fn block_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    block: &VisionBlock<B>,
    config: &VisionConfig,
    root: &str,
    layer: usize,
    destination: Destination<'_>,
) -> Result<Vec<ParameterGroupSpec>, GroupError> {
    destination.controls::<(
        String,
        Vec<std::ops::Range<usize>>,
        Vec<ParameterGroupSpec>,
        [(&B::Linear, ProjectionSharding); 2],
    )>()?;
    let prefix = if root.is_empty() {
        destination.text(format_args!("blocks.{layer}"))?
    } else {
        destination.text(format_args!("{root}.blocks.{layer}"))?
    };
    let hidden = usize::try_from(config.hidden_size)
        .map_err(|_| destination.group_error(format_args!("vision hidden width exceeds usize")))?;
    let heads = usize::try_from(config.num_heads)
        .map_err(|_| destination.group_error(format_args!("vision heads exceed usize")))?;
    let mut segments = destination.vector(3)?;
    segments.extend([0..hidden, hidden..2 * hidden, 2 * hidden..3 * hidden]);
    let mut groups = destination.vector(4)?;
    groups.push(destination.module::<B::Tensor, _>(
        format_args!("{prefix}.norm1"),
        ParameterRole::Replicated,
        &block.norm1,
        |_| Ok(MemberSharding::Replicated),
    )?);
    groups.push(destination.module::<B::Tensor, _>(
        format_args!("{prefix}.norm2"),
        ParameterRole::Replicated,
        &block.norm2,
        |_| Ok(MemberSharding::Replicated),
    )?);
    groups.push(destination.partitioned_module_named::<B::Tensor, _>(
        format_args!("{prefix}.attention.heads"),
        ParameterRole::AttentionHeads,
        heads,
        &block.attention,
        |name, shape| {
            if name.contains("qkv") {
                let mut owned = destination.vector(segments.len())?;
                owned.extend(segments.iter().cloned());
                Ok(MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: owned,
                })
            } else if name.contains("proj") && shape.len() >= 2 {
                Ok(MemberSharding::Partitioned { axis: 1 })
            } else {
                Ok(MemberSharding::Replicated)
            }
        },
    )?);
    groups.push(destination.projections::<B::Tensor, B::Linear>(
        format_args!("{prefix}.mlp.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        &[
            (&block.fc1, ProjectionSharding::Column),
            (&block.fc2, ProjectionSharding::Row),
        ],
        destination.aligned(
            &prefix,
            usize::try_from(config.intermediate_size).map_err(|_| {
                destination.group_error(format_args!("vision intermediate width exceeds usize"))
            })?,
            1,
            1,
        )?,
    )?);
    Ok(groups)
}

fn merger_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    merger: &super::model::Merger<B>,
    root: &str,
    width: usize,
    destination: Destination<'_>,
) -> Result<Vec<ParameterGroupSpec>, GroupError> {
    destination.controls::<(
        Vec<ParameterGroupSpec>,
        [(&B::Linear, ProjectionSharding); 2],
    )>()?;
    let mut groups = destination.vector(2)?;
    groups.push(destination.module::<B::Tensor, _>(
        format_args!("{root}.norm"),
        ParameterRole::Replicated,
        &merger.norm,
        |_| Ok(MemberSharding::Replicated),
    )?);
    groups.push(destination.projections::<B::Tensor, B::Linear>(
        format_args!("{root}.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        &[
            (&merger.fc1, ProjectionSharding::Column),
            (&merger.fc2, ProjectionSharding::Row),
        ],
        destination.aligned(root, width, 1, 1)?,
    )?);
    Ok(groups)
}

/// Declares replicated patch/position modules and TP merger channels.
pub fn static_parallel_parameter_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    modules: &VisionStatic<B>,
    config: &VisionConfig,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(owned_static_parallel_parameter_groups(
        modules,
        config,
        root,
        eredu_runtime::ExecutionGroupId::new("vision")
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?,
        "vision",
    )?
    .into_iter()
    .map(eredu_runtime::OwnedParameterGroupSpec::into_group)
    .collect())
}

/// Declares pinned vision storage at its actual patch, final projection and
/// per-layer DeepStack consumers. A shared consumer list allocates one copy per
/// owning partition and never transfers these parameters into a block's payload.
pub fn owned_static_parallel_parameter_groups<
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    modules: &VisionStatic<B>,
    config: &VisionConfig,
    root: &str,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
) -> Result<Vec<eredu_runtime::OwnedParameterGroupSpec>, ParallelPlanError> {
    owned_static_groups(
        modules,
        config,
        root,
        execution_group,
        role,
        Destination(None),
    )
    .map_err(GroupError::ordinary)
}

pub(crate) fn owned_static_parallel_parameter_groups_with_metadata<
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    modules: &VisionStatic<B>,
    config: &VisionConfig,
    root: &str,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
    context: Option<&eredu_nn::workspace::WorkspaceContext>,
) -> Result<Vec<eredu_runtime::OwnedParameterGroupSpec>, eredu_nn::Error> {
    owned_static_groups(
        modules,
        config,
        root,
        execution_group,
        role,
        Destination(context),
    )
    .map_err(GroupError::into_neural)
}

fn owned_static_groups<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    modules: &VisionStatic<B>,
    config: &VisionConfig,
    root: &str,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
    destination: Destination<'_>,
) -> Result<Vec<eredu_runtime::OwnedParameterGroupSpec>, GroupError> {
    use eredu_runtime::{OwnedParameterGroupSpec, ParameterGroupOwner};
    destination.controls::<(
        Vec<ParameterGroupSpec>,
        Vec<OwnedParameterGroupSpec>,
        ParameterGroupOwner,
        String,
    )>()?;
    let last_unit = config.layer_count().checked_sub(1).ok_or_else(|| {
        destination.group_error(format_args!("vision projection has no consuming unit"))
    })?;
    let mut ingress = destination.vector(2)?;
    ingress.push(destination.module::<B::Tensor, _>(
        format_args!("{root}.position"),
        ParameterRole::Replicated,
        &modules.position,
        |_| Ok(MemberSharding::Replicated),
    )?);
    ingress.push(destination.module::<B::Tensor, _>(
        format_args!("{root}.patch"),
        ParameterRole::Replicated,
        &modules.patch,
        |_| Ok(MemberSharding::Replicated),
    )?);
    let mut groups = destination.vector(ingress.len())?;
    for group in ingress {
        let owner = ParameterGroupOwner::static_role(destination.text(format_args!("{role}"))?);
        groups.push(OwnedParameterGroupSpec::new(owner, group));
    }
    let width =
        usize::try_from(config.hidden_size * config.spatial_merge_size * config.spatial_merge_size)
            .map_err(|_| {
                destination.group_error(format_args!("vision merger width exceeds usize"))
            })?;
    let name = rooted_with(root, format_args!("merger"), destination)?;
    let emitted = merger_groups(&modules.merger, &name, width, destination)?;
    append_consumers(
        &mut groups,
        emitted,
        role,
        &execution_group,
        last_unit,
        destination,
    )?;
    for (index, merger) in modules.deepstack_mergers.iter().enumerate() {
        let unit = usize::try_from(config.deepstack_layers()[index])
            .map_err(|_| destination.group_error(format_args!("negative DeepStack consumer")))?;
        let name = rooted_with(
            root,
            format_args!("deepstack_merger_list.{index}"),
            destination,
        )?;
        let emitted = merger_groups(merger, &name, width, destination)?;
        append_consumers(
            &mut groups,
            emitted,
            role,
            &execution_group,
            unit,
            destination,
        )?;
    }
    Ok(groups)
}

fn append_consumers(
    groups: &mut Vec<eredu_runtime::OwnedParameterGroupSpec>,
    emitted: Vec<ParameterGroupSpec>,
    role: &str,
    execution_group: &eredu_runtime::ExecutionGroupId,
    unit: usize,
    destination: Destination<'_>,
) -> Result<(), GroupError> {
    use eredu_runtime::{ExecutionGroupId, OwnedParameterGroupSpec, ParameterGroupOwner};
    destination.controls::<(
        ParameterGroupOwner,
        Vec<(ExecutionGroupId, usize)>,
        OwnedParameterGroupSpec,
    )>()?;
    destination.reserve(groups, emitted.len())?;
    for group in emitted {
        let mut consumers = destination.vector(1)?;
        let id =
            ExecutionGroupId::new(destination.text(format_args!("{}", execution_group.as_str()))?)
                .map_err(|cause| destination.group_error(format_args!("{cause}")))?;
        consumers.push((id, unit));
        let owner = ParameterGroupOwner::StaticUnitConsumers {
            role: destination.text(format_args!("{role}"))?,
            consumers,
        };
        groups.push(OwnedParameterGroupSpec::new(owner, group));
    }
    Ok(())
}

fn rooted_with(
    root: &str,
    relative: std::fmt::Arguments<'_>,
    destination: Destination<'_>,
) -> Result<String, GroupError> {
    if root.is_empty() {
        destination.text(relative)
    } else {
        destination.text(format_args!("{root}.{relative}"))
    }
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
