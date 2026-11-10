//! Semantic tensor-parallel placement for the shared Qwen vision tower.

use eredu_runtime::{MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole};

use super::specs::rooted;
use eredu_nn::LogicalParameterRole;

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
    declaration: &super::Declaration,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<(i32, i32), ParallelPlanError> {
    let spec = declaration.block(layer).map_err(invalid)?.clone();
    let qkv = tensor(layout, spec.attention.qkv.weight.id.as_str())?;
    let fc1 = tensor(layout, spec.feed_forward.up.weight.id.as_str())?;
    let qkv_rows = i32::try_from(qkv.local_shape()[0])
        .map_err(|_| ParallelPlanError::InvalidTensor("vision QKV width exceeds i32".into()))?;
    let intermediate = i32::try_from(fc1.local_shape()[0])
        .map_err(|_| ParallelPlanError::InvalidTensor("vision MLP width exceeds i32".into()))?;
    let stride = spec.attention.layout.output_width() / spec.attention.geometry.query_heads();
    if qkv_rows <= 0 || intermediate <= 0 || qkv_rows % stride != 0 {
        return Err(ParallelPlanError::InvalidTensor(
            "local vision geometry is zero or QKV splits a head".into(),
        ));
    }
    Ok((qkv_rows / stride, intermediate))
}

fn norm(
    prefix: &str,
    spec: &eredu_nn::LayerNormalizationConstructionSpec,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    eredu_runtime::logical_parameter_group(prefix, ParameterRole::Replicated, None, spec, |_| {
        MemberSharding::Replicated
    })
}

/// The paired projections share a semantic partition domain. Their identities,
/// geometry and storage extensions remain with the neural construction specs.
fn projections(
    name: String,
    role: ParameterRole,
    units: usize,
    specs: [eredu_nn::LinearSpec; 2],
    column_weight: MemberSharding,
    column_bias: MemberSharding,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let column = &specs[0];
    eredu_runtime::logical_parameter_group(name, role, Some(units), specs.as_slice(), |d| {
        if d.parameter.id == column.weight.id {
            column_weight.clone()
        } else if column
            .bias
            .as_ref()
            .is_some_and(|bias| bias.id == d.parameter.id)
        {
            column_bias.clone()
        } else if d.role == LogicalParameterRole::LinearWeight {
            MemberSharding::Partitioned { axis: 1 }
        } else {
            MemberSharding::Replicated
        }
    })
}
fn merger(
    declaration: &super::Declaration,
    index: usize,
    prefix: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let (normalization, [up, down]) = declaration.merger(index).map_err(invalid)?;
    Ok(vec![
        norm(&format!("{prefix}.norm"), normalization)?,
        projections(
            format!("{prefix}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            up.output as usize,
            [up.clone(), down.clone()],
            MemberSharding::Partitioned { axis: 0 },
            MemberSharding::Partitioned { axis: 0 },
        )?,
    ])
}
fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

/// Declares one canonical vision block without native resources.
pub fn block_parallel_parameter_groups(
    declaration: &super::Declaration,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let prefix = rooted(declaration.root(), &format!("blocks.{layer}"));
    let spec = declaration.block(layer).map_err(invalid)?.clone();
    let segments = MemberSharding::PartitionedSegments {
        axis: 0,
        segments: spec.attention.segments(),
    };
    let heads = spec.attention.geometry.query_heads() as usize;
    Ok(vec![
        norm(&format!("{prefix}.norm1"), &spec.norms[0])?,
        norm(&format!("{prefix}.norm2"), &spec.norms[1])?,
        projections(
            format!("{prefix}.attention.heads"),
            ParameterRole::AttentionHeads,
            heads,
            [spec.attention.qkv, spec.attention.output],
            segments.clone(),
            segments,
        )?,
        projections(
            format!("{prefix}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            spec.feed_forward.up.output as usize,
            [spec.feed_forward.up, spec.feed_forward.down],
            MemberSharding::Partitioned { axis: 0 },
            MemberSharding::Partitioned { axis: 0 },
        )?,
    ])
}

/// Projects replicated patch/position storage and TP merger channels.
pub fn static_parallel_parameter_groups(
    declaration: &super::Declaration,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(owned_static_parallel_parameter_groups(
        declaration,
        eredu_runtime::ExecutionGroupId::new("vision").map_err(invalid)?,
        "vision",
    )?
    .into_iter()
    .map(|group| group.group().clone())
    .collect())
}

/// Declares pinned vision storage at its patch, final projection and DeepStack consumers.
/// Shared storage has one physical copy on each partition containing a consumer.
pub fn owned_static_parallel_parameter_groups(
    declaration: &super::Declaration,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
) -> Result<Vec<eredu_runtime::OwnedParameterGroupSpec>, ParallelPlanError> {
    Ok(static_resource_groups(declaration, execution_group, role)?
        .into_iter()
        .map(|(_, group)| group)
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum StaticResource {
    Ingress,
    Merger,
    DeepStack(usize),
}

/// Exact cold selection of vision ingress and merger banks from canonical bindings.
#[derive(Debug, Clone)]
pub struct VisionStaticSelection {
    pub(super) declaration: std::sync::Arc<super::Declaration>,
    pub(super) ingress: bool,
    pub(super) merger: bool,
    pub(super) deepstack: Vec<bool>,
}
impl VisionStaticSelection {
    /// Selects all pinned resources for complete-tower construction.
    pub fn all(declaration: std::sync::Arc<super::Declaration>) -> Result<Self, ParallelPlanError> {
        let deepstack = vec![true; declaration.config().deepstack_layer_count()];
        Ok(Self {
            declaration,
            ingress: true,
            merger: true,
            deepstack,
        })
    }

    /// Complete-model merger widths from the same checked native construction geometry.
    pub fn unsharded_merger_intermediates(&self) -> Result<Vec<i32>, ParallelPlanError> {
        Ok((0..=self.declaration.config().deepstack_layer_count())
            .map(|index| {
                self.declaration
                    .merger(index)
                    .map(|(_, [up, _])| up.output)
                    .map_err(invalid)
            })
            .collect::<Result<_, _>>()?)
    }

    /// Resolves complete native resources from selected canonical parameter groups.
    /// Encoded companions and normalization/projection groups remain indivisible.
    pub fn from_parameter_bindings(
        declaration: std::sync::Arc<super::Declaration>,
        execution_group: eredu_runtime::ExecutionGroupId,
        role: &str,
        bindings: &[eredu_runtime::OwnedParameterGroupSpec],
    ) -> Result<Self, ParallelPlanError> {
        let declared = static_resource_groups(&declaration, execution_group, role)?;
        let resources = eredu_runtime::select_complete_pinned_resources(
            &format!("vision {:?}", declaration.root()),
            declared.iter().map(|(resource, group)| (*resource, group)),
            bindings,
            &[role],
        )
        .map_err(|error| invalid(error.to_string()))?;
        let owns = |resource| resources.contains(&resource);
        let deepstack = (0..declaration.config().deepstack_layer_count())
            .map(|bank| owns(StaticResource::DeepStack(bank)))
            .collect();
        Ok(Self {
            declaration,
            ingress: owns(StaticResource::Ingress),
            merger: owns(StaticResource::Merger),
            deepstack,
        })
    }
}

fn static_resource_groups(
    declaration: &super::Declaration,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
) -> Result<Vec<(StaticResource, eredu_runtime::OwnedParameterGroupSpec)>, ParallelPlanError> {
    use eredu_runtime::{OwnedParameterGroupSpec, ParameterGroupOwner};
    let config = declaration.config();
    let root = declaration.root();
    let last = config
        .layer_count()
        .checked_sub(1)
        .ok_or_else(|| invalid("vision projection has no consuming unit"))?;
    let position = declaration.position();
    let patch = declaration.patch();
    let mut groups = vec![
        (
            StaticResource::Ingress,
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role(role),
                eredu_runtime::logical_parameter_group(
                    format!("{root}.position"),
                    ParameterRole::Replicated,
                    None,
                    position,
                    |_| MemberSharding::Replicated,
                )?,
            ),
        ),
        (
            StaticResource::Ingress,
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role(role),
                eredu_runtime::logical_parameter_group(
                    format!("{root}.patch"),
                    ParameterRole::Replicated,
                    None,
                    patch,
                    |_| MemberSharding::Replicated,
                )?,
            ),
        ),
    ];
    for (resource, prefix, unit, index) in
        std::iter::once((StaticResource::Merger, rooted(root, "merger"), last, 0)).chain(
            config
                .deepstack_layers()
                .into_iter()
                .enumerate()
                .map(|(index, unit)| {
                    (
                        StaticResource::DeepStack(index),
                        rooted(root, &format!("deepstack_merger_list.{index}")),
                        unit as usize,
                        index + 1,
                    )
                }),
        )
    {
        let owner =
            ParameterGroupOwner::static_unit_consumers(role, [(execution_group.clone(), unit)]);
        groups.extend(
            merger(declaration, index, &prefix)?
                .into_iter()
                .map(|group| (resource, OwnedParameterGroupSpec::new(owner.clone(), group))),
        );
    }
    Ok(groups)
}

/// Resolves local main/deepstack merger intermediate widths.
pub fn local_merger_widths(
    declaration: &super::Declaration,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<i32>, ParallelPlanError> {
    let config = declaration.config();
    let root = declaration.root();
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
