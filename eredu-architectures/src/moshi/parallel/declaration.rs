//! One semantic declaration producer, with ordinary or admitted metadata storage.
use super::*;
use crate::decoder::parameter_metadata::{
    DeclarationDestination as Destination, ParameterGroupError as Failure,
};
use eredu_nn::{Error, workspace::WorkspaceContext};
use eredu_runtime::{ExecutionGroupId, ExecutionGroupSpec};

type Declaration = (
    ExecutionGraph,
    ExecutionUnitLayout,
    Vec<OwnedParameterGroupSpec>,
);

fn dimension(value: i32, label: &str, destination: Destination<'_>) -> Result<usize, Failure> {
    destination.controls::<(i32, &str, Result<usize, std::num::TryFromIntError>)>()?;
    usize::try_from(value)
        .map_err(|_| destination.tensor_error(format_args!("Moshi {label} exceeds usize")))
}
fn values<T, const N: usize>(
    values: [T; N],
    destination: Destination<'_>,
) -> Result<Vec<T>, Failure> {
    destination.controls::<([T; N], Vec<T>)>()?;
    let mut out = destination.vector(N)?;
    out.extend(values);
    Ok(out)
}
fn member<const N: usize>(
    name: std::fmt::Arguments<'_>,
    shape: [usize; N],
    sharding: MemberSharding,
    destination: Destination<'_>,
) -> Result<ParameterMemberSpec, Failure> {
    destination.controls::<(
        ParameterMemberSpec,
        std::fmt::Arguments<'_>,
        [usize; N],
        MemberSharding,
    )>()?;
    Ok(ParameterMemberSpec::new(
        destination.text(name)?,
        values(shape, destination)?,
        sharding,
    ))
}
fn group<const N: usize>(
    name: std::fmt::Arguments<'_>,
    role: ParameterRole,
    units: Option<usize>,
    members: [ParameterMemberSpec; N],
    destination: Destination<'_>,
) -> Result<ParameterGroupSpec, Failure> {
    destination.controls::<(
        ParameterGroupSpec,
        std::fmt::Arguments<'_>,
        ParameterRole,
        Option<usize>,
        [ParameterMemberSpec; N],
        Vec<ParameterMemberSpec>,
    )>()?;
    let name = destination.text(name)?;
    let members = values(members, destination)?;
    match destination.0 {
        Some(context) => {
            ParameterGroupSpec::from_owned_with_metadata(name, role, units, members, context)
                .map_err(Failure::Metadata)
        }
        None => match units {
            Some(units) => ParameterGroupSpec::partitioned(name, role, units, members),
            None => ParameterGroupSpec::new(name, role, members),
        }
        .map_err(Failure::Ordinary),
    }
}
fn owner(
    group: &str,
    unit: usize,
    destination: Destination<'_>,
) -> Result<ParameterGroupOwner, Failure> {
    destination.controls::<(ParameterGroupOwner, ExecutionGroupId, &str, usize)>()?;
    let id = ExecutionGroupId::new(destination.text(format_args!("{group}"))?)
        .map_err(|error| destination.group_error(format_args!("{error}")))?;
    Ok(ParameterGroupOwner::execution_unit(id, unit))
}
fn push(
    owned: &mut Vec<OwnedParameterGroupSpec>,
    owner: ParameterGroupOwner,
    group: ParameterGroupSpec,
    destination: Destination<'_>,
) -> Result<(), Failure> {
    destination.controls::<(
        &mut Vec<OwnedParameterGroupSpec>,
        ParameterGroupOwner,
        ParameterGroupSpec,
    )>()?;
    destination.reserve(owned, 1)?;
    owned.push(OwnedParameterGroupSpec::new(owner, group));
    Ok(())
}
fn static_group(
    owned: &mut Vec<OwnedParameterGroupSpec>,
    role: &str,
    name: std::fmt::Arguments<'_>,
    parameter_role: ParameterRole,
    parameter: ParameterMemberSpec,
    destination: Destination<'_>,
) -> Result<(), Failure> {
    destination.controls::<(
        &mut Vec<OwnedParameterGroupSpec>,
        &str,
        std::fmt::Arguments<'_>,
        ParameterRole,
        ParameterMemberSpec,
        ParameterGroupOwner,
        ParameterGroupSpec,
    )>()?;
    let owner = ParameterGroupOwner::static_role(destination.text(format_args!("{role}"))?);
    push(
        owned,
        owner,
        group(name, parameter_role, None, [parameter], destination)?,
        destination,
    )
}

pub(super) fn groups(
    config: &MoshiConfig,
    destination: Destination<'_>,
) -> Result<Declaration, Failure> {
    destination.controls::<(
        &MoshiConfig,
        Declaration,
        Vec<ExecutionGroupSpec>,
        Vec<String>,
        [usize; 2],
        usize,
        usize,
        usize,
        usize,
        usize,
        String,
        Vec<ParameterGroupSpec>,
        ParameterGroupOwner,
        ParameterGroupSpec,
        [ParameterGroupSpec; 3],
        std::ops::Range<usize>,
        std::vec::IntoIter<ParameterGroupSpec>,
    )>()?;
    let temporal_layers = dimension(
        config.temporal().num_hidden_layers(),
        "temporal layers",
        destination,
    )?;
    let depth_slices = config.frame_schedule().depth_audio_codebooks();
    let mut specs = destination.vector(2)?;
    specs.push(ExecutionGroupSpec::root(
        destination.text(format_args!("temporal_transformer"))?,
    ));
    specs.push(ExecutionGroupSpec::from_parts(
        destination.text(format_args!("depth_codebook_slices"))?,
        values(
            [destination.text(format_args!("temporal_transformer"))?],
            destination,
        )?,
    ));
    let graph = match destination.0 {
        Some(context) => ExecutionGraph::new_with_metadata(specs, "depth_codebook_slices", context)
            .map_err(Failure::Metadata)?,
        None => ExecutionGraph::new(specs, "depth_codebook_slices")
            .map_err(|error| destination.group_error(format_args!("{error}")))?,
    };
    let counts = [temporal_layers, depth_slices];
    let layout = match destination.0 {
        Some(context) => ExecutionUnitLayout::new_with_metadata(&graph, &counts, context)
            .map_err(Failure::Metadata)?,
        None => ExecutionUnitLayout::new(&graph, counts)
            .map_err(|error| destination.group_error(format_args!("{error}")))?,
    };
    let temporal_hidden = dimension(
        config.temporal().hidden_size(),
        "temporal hidden width",
        destination,
    )?;
    let text_vocabulary = dimension(
        config.text_vocabulary_size(),
        "text vocabulary",
        destination,
    )?;
    let audio_vocabulary = dimension(
        config.audio_vocabulary_size(),
        "audio vocabulary",
        destination,
    )?;
    let padded = |value: usize| {
        value.checked_add(1).ok_or_else(|| {
            destination.tensor_error(format_args!("Moshi input vocabulary overflowed"))
        })
    };
    let mut owned = destination.vector(0)?;
    static_group(
        &mut owned,
        "embedding",
        format_args!("text_emb"),
        ParameterRole::Vocabulary,
        member(
            format_args!("text_emb.weight"),
            [padded(text_vocabulary)?, temporal_hidden],
            MemberSharding::Balanced { axis: 0 },
            destination,
        )?,
        destination,
    )?;
    for codebook in 0..config.frame_schedule().total_audio_codebooks() {
        static_group(
            &mut owned,
            "embedding",
            format_args!("audio_embs.{codebook}"),
            ParameterRole::Vocabulary,
            member(
                format_args!("audio_embs.{codebook}.weight"),
                [padded(audio_vocabulary)?, temporal_hidden],
                MemberSharding::Balanced { axis: 0 },
                destination,
            )?,
            destination,
        )?;
    }
    static_group(
        &mut owned,
        "norm",
        format_args!("out_norm"),
        ParameterRole::Replicated,
        member(
            format_args!("out_norm.weight"),
            [temporal_hidden],
            MemberSharding::Replicated,
            destination,
        )?,
        destination,
    )?;
    static_group(
        &mut owned,
        "output",
        format_args!("text_linear"),
        ParameterRole::Vocabulary,
        member(
            format_args!("text_linear.weight"),
            [text_vocabulary, temporal_hidden],
            MemberSharding::Balanced { axis: 0 },
            destination,
        )?,
        destination,
    )?;
    for layer in 0..temporal_layers {
        for group in block_groups(
            config.temporal(),
            config.temporal().parameter_root(),
            layer,
            destination,
        )? {
            push(
                &mut owned,
                owner("temporal_transformer", layer, destination)?,
                group,
                destination,
            )?;
        }
    }
    // Only namespaces vary by depth slice. Geometry and format policy are borrowed
    // from the same validated template used by depth_transformer; no config copy.
    for slice in 0..depth_slices {
        let transformer =
            config.depth_template_for(slice, |error| destination.group_error(error))?;
        let hidden = dimension(transformer.hidden_size(), "depth hidden width", destination)?;
        let prefix = destination.text(format_args!("depformer.slices.{slice}"))?;
        let input = padded(if slice == 0 {
            text_vocabulary
        } else {
            audio_vocabulary
        })?;
        let first = [
            group(
                format_args!("{prefix}.emb"),
                ParameterRole::Vocabulary,
                None,
                [member(
                    format_args!("{prefix}.emb.weight"),
                    [input, hidden],
                    MemberSharding::Balanced { axis: 0 },
                    destination,
                )?],
                destination,
            )?,
            group(
                format_args!("{prefix}.linear_in"),
                ParameterRole::Replicated,
                None,
                [member(
                    format_args!("{prefix}.linear_in.weight"),
                    [hidden, temporal_hidden],
                    MemberSharding::Replicated,
                    destination,
                )?],
                destination,
            )?,
            group(
                format_args!("{prefix}.linear_out"),
                ParameterRole::Vocabulary,
                None,
                [member(
                    format_args!("{prefix}.linear_out.weight"),
                    [audio_vocabulary, hidden],
                    MemberSharding::Balanced { axis: 0 },
                    destination,
                )?],
                destination,
            )?,
        ];
        for group in first {
            push(
                &mut owned,
                owner("depth_codebook_slices", slice, destination)?,
                group,
                destination,
            )?;
        }
        // The per-slice transformer is rooted at `<slice>.transformer`.
        let root = destination.text(format_args!("{prefix}.transformer"))?;
        for layer in 0..dimension(transformer.num_hidden_layers(), "depth layers", destination)? {
            for group in block_groups(transformer, &root, layer, destination)? {
                push(
                    &mut owned,
                    owner("depth_codebook_slices", slice, destination)?,
                    group,
                    destination,
                )?;
            }
        }
    }
    Ok((graph, layout, owned))
}

fn block_groups(
    config: &MoshiTransformerConfig,
    prefix: &str,
    layer: usize,
    destination: Destination<'_>,
) -> Result<Vec<ParameterGroupSpec>, Failure> {
    destination.controls::<(
        &MoshiTransformerConfig,
        &str,
        usize,
        String,
        Vec<ParameterGroupSpec>,
        [ParameterGroupSpec; 4],
        [ParameterMemberSpec; 2],
        Vec<Range<usize>>,
        [Range<usize>; 3],
        [Range<usize>; 2],
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )>()?;
    let hidden = dimension(
        config.hidden_size(),
        "transformer hidden width",
        destination,
    )?;
    let heads = dimension(config.num_attention_heads(), "attention heads", destination)?;
    let head = dimension(config.head_dim(), "head width", destination)?;
    let gated = dimension(config.gated_hidden_size(), "gated width", destination)?;
    let root = destination.text(format_args!("{prefix}.layers.{layer}"))?;
    let out = destination.text(format_args!("{root}.self_attn.out_proj.weight"))?;
    let format = destination.linear_format(config, &out)?;
    let attention_units = crate::linear_format::input_partition_units_with(
        &destination.text(format_args!("{root}.self_attn"))?,
        heads,
        head,
        format,
        destination,
    )?;
    let gating_units = crate::linear_format::input_partition_units_with(
        &destination.text(format_args!("{root}.gating"))?,
        gated,
        1,
        format,
        destination,
    )?;
    let k_end = hidden.checked_mul(2).ok_or_else(|| {
        destination.tensor_error(format_args!("Moshi fused attention width overflowed"))
    })?;
    let v_end = hidden.checked_mul(3).ok_or_else(|| {
        destination.tensor_error(format_args!("Moshi fused attention width overflowed"))
    })?;
    let up_end = gated.checked_mul(2).ok_or_else(|| {
        destination.tensor_error(format_args!("Moshi fused gating width overflowed"))
    })?;
    values(
        [
            group(
                format_args!("{root}.self_attn.projections"),
                ParameterRole::AttentionHeads,
                Some(attention_units),
                [
                    member(
                        format_args!("{root}.self_attn.in_proj.weight"),
                        [v_end, hidden],
                        MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: values(
                                [0..hidden, hidden..k_end, k_end..v_end],
                                destination,
                            )?,
                        },
                        destination,
                    )?,
                    member(
                        format_args!("{root}.self_attn.out_proj.weight"),
                        [hidden, hidden],
                        MemberSharding::Partitioned { axis: 1 },
                        destination,
                    )?,
                ],
                destination,
            )?,
            group(
                format_args!("{root}.norm1"),
                ParameterRole::Replicated,
                None,
                [member(
                    format_args!("{root}.norm1.weight"),
                    [hidden],
                    MemberSharding::Replicated,
                    destination,
                )?],
                destination,
            )?,
            group(
                format_args!("{root}.norm2"),
                ParameterRole::Replicated,
                None,
                [member(
                    format_args!("{root}.norm2.weight"),
                    [hidden],
                    MemberSharding::Replicated,
                    destination,
                )?],
                destination,
            )?,
            group(
                format_args!("{root}.gating.projections"),
                ParameterRole::FeedForwardIntermediate,
                Some(gating_units),
                [
                    member(
                        format_args!("{root}.gating.linear_in.weight"),
                        [up_end, hidden],
                        MemberSharding::PartitionedSegments {
                            axis: 0,
                            segments: values([0..gated, gated..up_end], destination)?,
                        },
                        destination,
                    )?,
                    member(
                        format_args!("{root}.gating.linear_out.weight"),
                        [hidden, gated],
                        MemberSharding::Partitioned { axis: 1 },
                        destination,
                    )?,
                ],
                destination,
            )?,
        ],
        destination,
    )
}

pub(in crate::moshi) fn description(
    config: &MoshiConfig,
    context: Option<&WorkspaceContext>,
) -> Result<ArchitectureParameterDescription, Error> {
    crate::decoder::identity::Metadata::new(context).controls::<(
        &MoshiConfig, Option<&WorkspaceContext>, ArchitectureParameterDescription,
        Result<ArchitectureParameterDescription, Failure>,
    )>()?;
    description_with(
        config,
        Destination(context.filter(|context| context.uses_checked_metadata())),
    )
    .map_err(Failure::into_neural)
}
pub(super) fn description_with(
    config: &MoshiConfig,
    destination: Destination<'_>,
) -> Result<ArchitectureParameterDescription, Failure> {
    destination.controls::<(
        &MoshiConfig,
        Declaration,
        Vec<OwnedParameterGroupSpec>,
        OwnedParameterGroupSpec,
        ParameterGroupOwner,
        Vec<ParameterGroupSpec>,
        ParameterGroupSpec,
        ArchitectureParameterDescription,
    )>()?;
    let (graph, layout, owned) = groups(config, destination)?;
    let mut expanded = destination.vector(owned.len())?;
    for tagged in owned {
        let retained_owner = match tagged.owner() {
            ParameterGroupOwner::StaticRole(role) => {
                ParameterGroupOwner::static_role(destination.text(format_args!("{role}"))?)
            }
            ParameterGroupOwner::ExecutionUnit { group, global_unit, .. } => {
                owner(group.as_str(), *global_unit, destination)?
            }
            _ => unreachable!("Moshi emits static or execution-unit ownership"),
        };
        let one = values([tagged.into_group()], destination)?;
        let mut groups = match destination.0 {
            Some(context) => eredu_runtime::expand_linear_format_parameter_groups_with_metadata(
                one,
                |member| {
                    crate::linear_format::standard_parallel_linear_format_with(
                        member,
                        destination
                            .linear_format(config.temporal(), member.target())
                            .map_err(Failure::into_neural)?,
                        destination,
                    )
                    .map_err(Failure::into_neural)
                },
                context,
            )
            .map_err(Failure::Metadata)?,
            None => eredu_runtime::expand_linear_format_parameter_groups(one, |member| {
                crate::linear_format::standard_parallel_linear_format_with(
                    member,
                    destination
                        .linear_format(config.temporal(), member.target())
                        .map_err(Failure::ordinary)?,
                    destination,
                )
                .map_err(Failure::ordinary)
            })
            .map_err(Failure::Ordinary)?,
        };
        let group = groups.pop().expect("one Moshi group expands to one group");
        debug_assert!(groups.is_empty());
        expanded.push(OwnedParameterGroupSpec::new(retained_owner, group));
    }
    match destination.0 {
        Some(context) => ArchitectureParameterDescription::from_owned_with_metadata(
            graph, layout, expanded, context,
        )
        .map_err(Failure::Metadata),
        None => ArchitectureParameterDescription::from_owned(graph, layout, expanded)
            .map_err(|error| destination.group_error(format_args!("{error}"))),
    }
}
