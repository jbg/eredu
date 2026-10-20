//! Semantic tensor-parallel placement for the shared Qwen vision tower.

use eredu_runtime::{MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole};

use super::{specs::rooted, VisionConfig};

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

#[derive(Default)]
struct Declaration {
    formats: std::collections::BTreeMap<String, eredu_nn::LinearFormatSpec>,
}
impl Declaration {
    fn linear(
        &mut self,
        spec: eredu_nn::LinearSpec,
        weight: MemberSharding,
        bias: MemberSharding,
    ) -> Vec<eredu_runtime::ParameterMemberSpec> {
        self.formats
            .insert(spec.weight.id.as_str().into(), spec.format);
        let mut members = vec![eredu_runtime::ParameterMemberSpec::new(
            spec.weight.id.as_str(),
            vec![spec.output as usize, spec.input as usize],
            weight,
        )];
        if let Some(parameter) = spec.bias {
            members.push(eredu_runtime::ParameterMemberSpec::new(
                parameter.id.as_str(),
                vec![spec.output as usize],
                bias,
            ));
        }
        members
    }
    fn norm(&self, prefix: &str, width: i32) -> Result<ParameterGroupSpec, ParallelPlanError> {
        let spec = super::specs::norm_spec(prefix, width).map_err(invalid)?;
        ParameterGroupSpec::new(
            prefix,
            ParameterRole::Replicated,
            [spec.weight, spec.bias].into_iter().map(|parameter| {
                eredu_runtime::ParameterMemberSpec::new(
                    parameter.id.as_str(),
                    vec![spec.dimensions as usize],
                    MemberSharding::Replicated,
                )
            }),
        )
    }
    fn projections(
        &mut self,
        name: String,
        role: ParameterRole,
        units: usize,
        specs: [eredu_nn::LinearSpec; 2],
    ) -> Result<ParameterGroupSpec, ParallelPlanError> {
        let [column, row] = specs;
        let mut members = self.linear(
            column,
            MemberSharding::Partitioned { axis: 0 },
            MemberSharding::Partitioned { axis: 0 },
        );
        members.extend(self.linear(
            row,
            MemberSharding::Partitioned { axis: 1 },
            MemberSharding::Replicated,
        ));
        ParameterGroupSpec::partitioned(name, role, units, members)
    }
    fn merger(
        &mut self,
        config: &VisionConfig,
        prefix: &str,
        postshuffle: bool,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        let (_, width, norm) =
            super::specs::merger_geometry(config, postshuffle).map_err(invalid)?;
        Ok(vec![
            self.norm(&format!("{prefix}.norm"), norm)?,
            self.projections(
                format!("{prefix}.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                width as usize,
                super::specs::merger_projection_specs(config, prefix, width).map_err(invalid)?,
            )?,
        ])
    }
    fn expand(
        &self,
        groups: Vec<ParameterGroupSpec>,
    ) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
        eredu_runtime::expand_linear_format_parameter_groups(groups, |member| {
            Ok(self.formats.get(member.target()).cloned())
        })
    }
}
fn invalid(error: impl std::fmt::Display) -> ParallelPlanError {
    ParallelPlanError::InvalidGroup(error.to_string())
}

/// Declares one canonical vision block without native resources.
pub fn block_parallel_parameter_groups(
    config: &VisionConfig,
    root: &str,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    config.validate().map_err(invalid)?;
    if config.layer_policy(layer).is_none() {
        return Err(invalid(format!("vision has no block {layer}")));
    }
    let prefix = rooted(root, &format!("blocks.{layer}"));
    let hidden = config.hidden_size as usize;
    let segments = MemberSharding::PartitionedSegments {
        axis: 0,
        segments: vec![0..hidden, hidden..2 * hidden, 2 * hidden..3 * hidden],
    };
    let mut declaration = Declaration::default();
    let [qkv, output, fc1, fc2] = super::specs::block_projection_specs(
        config,
        root,
        layer,
        config.num_heads,
        config.intermediate_size,
    )
    .map_err(invalid)?;
    let mut attention = declaration.linear(qkv, segments.clone(), segments);
    attention.extend(declaration.linear(
        output,
        MemberSharding::Partitioned { axis: 1 },
        MemberSharding::Replicated,
    ));
    let groups = vec![
        declaration.norm(&format!("{prefix}.norm1"), config.hidden_size)?,
        declaration.norm(&format!("{prefix}.norm2"), config.hidden_size)?,
        ParameterGroupSpec::partitioned(
            format!("{prefix}.attention.heads"),
            ParameterRole::AttentionHeads,
            config.num_heads as usize,
            attention,
        )?,
        declaration.projections(
            format!("{prefix}.mlp.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            config.intermediate_size as usize,
            [fc1, fc2],
        )?,
    ];
    declaration.expand(groups)
}

/// Projects replicated patch/position storage and TP merger channels.
pub fn static_parallel_parameter_groups(
    config: &VisionConfig,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(owned_static_parallel_parameter_groups(
        config,
        root,
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
    config: &VisionConfig,
    root: &str,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
) -> Result<Vec<eredu_runtime::OwnedParameterGroupSpec>, ParallelPlanError> {
    Ok(static_resource_groups(config, root, execution_group, role)?
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
    pub(super) config: VisionConfig,
    pub(super) root: String,
    pub(super) ingress: bool,
    pub(super) merger: bool,
    pub(super) deepstack: Vec<bool>,
}
impl VisionStaticSelection {
    /// Selects all pinned resources for complete-tower construction.
    pub fn all(config: VisionConfig, root: &str) -> Result<Self, ParallelPlanError> {
        config.validate().map_err(invalid)?;
        let deepstack = vec![true; config.deepstack_layer_count()];
        Ok(Self {
            config,
            root: root.to_owned(),
            ingress: true,
            merger: true,
            deepstack,
        })
    }

    /// Complete-model merger widths from the same checked native construction geometry.
    pub fn unsharded_merger_intermediates(&self) -> Result<Vec<i32>, ParallelPlanError> {
        let (_, width, _) = super::specs::merger_geometry(&self.config, false).map_err(invalid)?;
        Ok(vec![width; self.config.deepstack_layer_count() + 1])
    }

    /// Resolves complete native resources from selected canonical parameter groups.
    /// Encoded companions and normalization/projection groups remain indivisible.
    pub fn from_parameter_bindings(
        config: VisionConfig,
        root: &str,
        execution_group: eredu_runtime::ExecutionGroupId,
        role: &str,
        bindings: &[eredu_runtime::OwnedParameterGroupSpec],
    ) -> Result<Self, ParallelPlanError> {
        let declared = static_resource_groups(&config, root, execution_group, role)?;
        let mut selected = std::collections::BTreeSet::new();
        for binding in bindings
            .iter()
            .filter(|group| group.owner().accepts_static_role(role))
        {
            if !declared.iter().any(|(_, group)| group == binding) {
                return Err(invalid(format!(
                    "vision {root:?}: incompatible pinned group {}",
                    binding.group().logical_name()
                )));
            }
            if !selected.insert(binding.group().logical_name()) {
                return Err(invalid(format!(
                    "vision {root:?}: duplicate pinned group {}",
                    binding.group().logical_name()
                )));
            }
        }
        let mut resources = std::collections::BTreeMap::<_, (usize, usize)>::new();
        for (resource, group) in declared {
            let counts = resources.entry(resource).or_default();
            counts.0 += usize::from(selected.contains(group.group().logical_name()));
            counts.1 += 1;
        }
        for (resource, (present, total)) in &resources {
            if *present != 0 && present != total {
                return Err(invalid(format!(
                    "vision {root:?}: incomplete pinned resource {resource:?}"
                )));
            }
        }
        let owns = |resource| {
            resources
                .get(&resource)
                .is_some_and(|(present, _)| *present > 0)
        };
        let deepstack = (0..config.deepstack_layer_count())
            .map(|bank| owns(StaticResource::DeepStack(bank)))
            .collect();
        Ok(Self {
            config,
            root: root.to_owned(),
            ingress: owns(StaticResource::Ingress),
            merger: owns(StaticResource::Merger),
            deepstack,
        })
    }
}

fn static_resource_groups(
    config: &VisionConfig,
    root: &str,
    execution_group: eredu_runtime::ExecutionGroupId,
    role: &str,
) -> Result<Vec<(StaticResource, eredu_runtime::OwnedParameterGroupSpec)>, ParallelPlanError> {
    use eredu_runtime::{OwnedParameterGroupSpec, ParameterGroupOwner, ParameterMemberSpec};
    config.validate().map_err(invalid)?;
    let last = config
        .layer_count()
        .checked_sub(1)
        .ok_or_else(|| invalid("vision projection has no consuming unit"))?;
    let position = super::specs::position_spec(config, root).map_err(invalid)?;
    let patch = super::specs::patch_spec(config, root).map_err(invalid)?;
    let patch_shape = patch
        .weight_shape()
        .into_iter()
        .map(|width| width as usize)
        .collect::<Vec<_>>();
    let mut declaration = Declaration::default();
    declaration
        .formats
        .insert(position.weight.id.as_str().into(), position.format);
    let mut groups = vec![
        (
            StaticResource::Ingress,
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role(role),
                ParameterGroupSpec::new(
                    format!("{root}.position"),
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::new(
                        position.weight.id.as_str(),
                        vec![position.vocabulary as usize, position.dimensions as usize],
                        MemberSharding::Replicated,
                    )],
                )?,
            ),
        ),
        (
            StaticResource::Ingress,
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role(role),
                ParameterGroupSpec::new(
                    format!("{root}.patch"),
                    ParameterRole::Replicated,
                    [
                        ParameterMemberSpec::new(
                            patch.weight.id.as_str(),
                            patch_shape,
                            MemberSharding::Replicated,
                        ),
                        ParameterMemberSpec::new(
                            patch.bias.id.as_str(),
                            vec![patch.geometry.output as usize],
                            MemberSharding::Replicated,
                        ),
                    ],
                )?,
            ),
        ),
    ];
    for (resource, prefix, unit, postshuffle) in
        std::iter::once((StaticResource::Merger, rooted(root, "merger"), last, false)).chain(
            config
                .deepstack_layers()
                .into_iter()
                .enumerate()
                .map(|(index, unit)| {
                    (
                        StaticResource::DeepStack(index),
                        rooted(root, &format!("deepstack_merger_list.{index}")),
                        unit as usize,
                        true,
                    )
                }),
        )
    {
        let owner =
            ParameterGroupOwner::static_unit_consumers(role, [(execution_group.clone(), unit)]);
        groups.extend(
            declaration
                .merger(config, &prefix, postshuffle)?
                .into_iter()
                .map(|group| (resource, OwnedParameterGroupSpec::new(owner.clone(), group))),
        );
    }
    let mut expanded = Vec::with_capacity(groups.len());
    for (resource, group) in groups {
        expanded.extend(
            declaration
                .expand(vec![group.group().clone()])?
                .into_iter()
                .map(|expanded| {
                    (
                        resource,
                        OwnedParameterGroupSpec::new(group.owner().clone(), expanded),
                    )
                }),
        );
    }
    Ok(expanded)
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
