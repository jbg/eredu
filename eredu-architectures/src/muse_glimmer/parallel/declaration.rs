use super::*;
use crate::decoder::parameter_metadata::{DeclarationDestination, ParameterGroupError};
fn values<T, const N: usize>(
    values: [T; N],
    destination: DeclarationDestination<'_>,
) -> Result<Vec<T>, ParameterGroupError> {
    destination.controls::<([T; N], Vec<T>)>()?;
    let mut out = destination.vector(N)?;
    out.extend(values);
    Ok(out)
}
fn push<T>(
    out: &mut Vec<T>,
    value: T,
    destination: DeclarationDestination<'_>,
) -> Result<(), ParameterGroupError> {
    destination.controls::<(&mut Vec<T>, T)>()?;
    destination.reserve(out, 1)?;
    out.push(value);
    Ok(())
}
fn dim(value: i32, destination: DeclarationDestination<'_>) -> Result<usize, ParameterGroupError> {
    destination.controls::<(i32, usize, Result<usize, std::num::TryFromIntError>)>()?;
    usize::try_from(value)
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| {
            invalid(
                format_args!("invalid Muse-Glimmer dimension {value}"),
                destination,
            )
        })
}
fn invalid(
    message: impl std::fmt::Display,
    destination: DeclarationDestination<'_>,
) -> ParameterGroupError {
    destination.group_error(format_args!("{message}"))
}
fn member(
    name: impl std::fmt::Display,
    shape: Vec<usize>,
    sharding: MemberSharding,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterMemberSpec, ParameterGroupError> {
    destination.controls::<(Vec<usize>, MemberSharding, ParameterMemberSpec)>()?;
    destination.controls_of(&name)?;
    Ok(ParameterMemberSpec::new(
        destination.text(format_args!("{name}"))?,
        shape,
        sharding,
    ))
}
fn finish(
    name: impl std::fmt::Display,
    role: ParameterRole,
    units: Option<usize>,
    members: impl IntoIterator<Item = ParameterMemberSpec>,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    destination.controls::<(
        ParameterRole,
        Option<usize>,
        ParameterGroupSpec,
        Vec<ParameterMemberSpec>,
    )>()?;
    destination.controls_of(&name)?;
    destination.controls_of(&members)?;
    let iter = members.into_iter();
    let mut members = destination.vector(iter.size_hint().0)?;
    for value in iter {
        push(&mut members, value, destination)?;
    }
    let name = destination.text(format_args!("{name}"))?;
    match destination.0 {
        Some(context) => {
            ParameterGroupSpec::from_owned_with_metadata(name, role, units, members, context)
                .map_err(ParameterGroupError::Metadata)
        }
        None => match units {
            Some(units) => ParameterGroupSpec::partitioned(name, role, units, members),
            None => ParameterGroupSpec::new(name, role, members),
        }
        .map_err(ParameterGroupError::Ordinary),
    }
}
fn group_members(
    name: impl std::fmt::Display,
    role: ParameterRole,
    members: impl IntoIterator<Item = ParameterMemberSpec>,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    finish(name, role, None, members, destination)
}
fn partitioned_group(
    name: impl std::fmt::Display,
    role: ParameterRole,
    units: usize,
    members: impl IntoIterator<Item = ParameterMemberSpec>,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    finish(name, role, Some(units), members, destination)
}
fn group(
    name: impl std::fmt::Display,
    role: ParameterRole,
    members: impl IntoIterator<Item = (impl std::fmt::Display, Vec<usize>, MemberSharding)>,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    destination.controls_of(&name)?;
    destination.controls_of(&members)?;
    let iter = members.into_iter();
    let mut output = destination.vector(iter.size_hint().0)?;
    for (name, shape, sharding) in iter {
        push(
            &mut output,
            member(name, shape, sharding, destination)?,
            destination,
        )?;
    }
    finish(name, role, None, output, destination)
}
fn replicated(
    name: impl std::fmt::Display,
    members: impl IntoIterator<Item = (impl std::fmt::Display, Vec<usize>)>,
    destination: DeclarationDestination<'_>,
) -> Result<ParameterGroupSpec, ParameterGroupError> {
    group(
        name,
        ParameterRole::Replicated,
        members
            .into_iter()
            .map(|(name, shape)| (name, shape, MemberSharding::Replicated)),
        destination,
    )
}
fn expand(
    groups: Vec<ParameterGroupSpec>,
    format: impl Fn(&ParameterMemberSpec) -> eredu_checkpoint::LinearFormat,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls_of(&format)?;
    destination.controls::<(
        Vec<ParameterGroupSpec>,
        &ParameterMemberSpec,
        Result<Option<eredu_nn::LinearFormatSpec>, ParameterGroupError>,
    )>()?;
    match destination.0 {
        Some(context) => eredu_runtime::expand_linear_format_parameter_groups_with_metadata(
            groups,
            |member| {
                crate::linear_format::standard_parallel_linear_format_with(
                    member,
                    format(member),
                    destination,
                )
                .map_err(ParameterGroupError::into_neural)
            },
            context,
        )
        .map_err(ParameterGroupError::Metadata),
        None => eredu_runtime::expand_linear_format_parameter_groups(groups, |member| {
            crate::linear_format::standard_parallel_linear_format_with(
                member,
                format(member),
                destination,
            )
            .map_err(ParameterGroupError::ordinary)
        })
        .map_err(ParameterGroupError::Ordinary),
    }
}
const fn partitioned(axis: usize) -> MemberSharding {
    MemberSharding::Partitioned { axis }
}
/// Declares pinned text embeddings, final normalization, and vocabulary output.
pub(crate) fn static_parameter_groups(
    args: &DecoderConfig,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls::<(
        &DecoderConfig,
        usize,
        usize,
        usize,
        usize,
        String,
        String,
        Vec<ParameterGroupSpec>,
        [ParameterMemberSpec; 8],
        [(String, Vec<usize>); 4],
        [Range<usize>; 2],
        Result<Vec<ParameterGroupSpec>, ParameterGroupError>,
    )>()?;

    let hidden = dim(args.hidden_size, destination)?;
    let vocabulary = dim(args.vocab_size, destination)?;
    let mut groups = values(
        [group(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            [(
                "model.embed_tokens.weight",
                values([vocabulary, hidden], destination)?,
                MemberSharding::Balanced { axis: 0 },
            )],
            destination,
        )?],
        destination,
    )?;
    push(
        &mut groups,
        replicated(
            "model.norm",
            [("model.norm.weight", values([hidden], destination)?)],
            destination,
        )?,
        destination,
    )?;
    if !args.tie_word_embeddings {
        push(
            &mut groups,
            group(
                "lm_head",
                ParameterRole::Vocabulary,
                [(
                    "lm_head.weight",
                    values([vocabulary, hidden], destination)?,
                    MemberSharding::Balanced { axis: 0 },
                )],
                destination,
            )?,
            destination,
        )?;
    }
    expand(
        groups,
        |member| args.linear_format_for(member.target()),
        destination,
    )
}

/// Declares one decoder block's head, dense/expert, and replicated groups.
pub(crate) fn layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls::<(
        &DecoderConfig,
        usize,
        usize,
        usize,
        usize,
        String,
        String,
        Vec<ParameterGroupSpec>,
        [ParameterMemberSpec; 8],
        [(String, Vec<usize>); 4],
        [Range<usize>; 2],
        Result<Vec<ParameterGroupSpec>, ParameterGroupError>,
    )>()?;

    if layer >= args.num_hidden_layers as usize {
        return Err(invalid(
            format_args!("Muse-Glimmer layer {layer} is out of range"),
            destination,
        ));
    }
    let hidden = dim(args.hidden_size, destination)?;
    let query_heads = dim(args.num_attention_heads, destination)?;
    let key_value_heads = dim(args.num_key_value_heads, destination)?;
    let head = dim(args.head_dim, destination)?;
    let query_width = query_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Muse-Glimmer query width overflow", destination))?;
    let key_value_width = key_value_heads
        .checked_mul(head)
        .ok_or_else(|| invalid("Muse-Glimmer key/value width overflow", destination))?;
    let root = destination.text(format_args!("model.layers.{layer}"))?;
    let attention = destination.text(format_args!("{root}.self_attn"))?;
    let mut groups = values(
        [partitioned_group(
            format_args!("{attention}.query_heads"),
            ParameterRole::AttentionHeads,
            query_heads,
            [
                member(
                    format_args!("{attention}.q_proj.weight"),
                    values([query_width, hidden], destination)?,
                    partitioned(0),
                    destination,
                )?,
                member(
                    format_args!("{attention}.gate_proj.weight"),
                    values([query_width, hidden], destination)?,
                    partitioned(0),
                    destination,
                )?,
                member(
                    format_args!("{attention}.o_proj.weight"),
                    values([hidden, query_width], destination)?,
                    partitioned(1),
                    destination,
                )?,
            ],
            destination,
        )?],
        destination,
    )?;
    push(
        &mut groups,
        partitioned_group(
            format_args!("{attention}.key_value_heads"),
            ParameterRole::AttentionHeads,
            key_value_heads,
            [
                member(
                    format_args!("{attention}.k_proj.weight"),
                    values([key_value_width, hidden], destination)?,
                    partitioned(0),
                    destination,
                )?,
                member(
                    format_args!("{attention}.v_proj.weight"),
                    values([key_value_width, hidden], destination)?,
                    partitioned(0),
                    destination,
                )?,
            ],
            destination,
        )?,
        destination,
    )?;
    if args.weight_convention == crate::muse_glimmer::WeightConvention::Gguf {
        push(
            &mut groups,
            replicated(
                format_args!("{attention}.qk_norm"),
                [
                    (
                        format_args!("{attention}.q_norm.weight"),
                        values([head], destination)?,
                    ),
                    (
                        format_args!("{attention}.k_norm.weight"),
                        values([head], destination)?,
                    ),
                ],
                destination,
            )?,
            destination,
        )?;
    }
    if args.is_moe() {
        let experts = dim(args.num_experts, destination)?;
        let intermediate = dim(args.moe_intermediate_size, destination)?;
        push(
            &mut groups,
            partitioned_group(
                format_args!("{root}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                intermediate,
                [
                    member(
                        format_args!("{root}.mlp.experts.gate_up_proj"),
                        values([experts, 2 * intermediate, hidden], destination)?,
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: values(
                                [0..intermediate, intermediate..2 * intermediate],
                                destination,
                            )?,
                        },
                        destination,
                    )?,
                    member(
                        format_args!("{root}.mlp.experts.down_proj"),
                        values([experts, hidden, intermediate], destination)?,
                        partitioned(2),
                        destination,
                    )?,
                ],
                destination,
            )?,
            destination,
        )?;
        push(
            &mut groups,
            replicated(
                format_args!("{root}.mlp.router"),
                [(
                    format_args!("{root}.mlp.gate.weight"),
                    values([experts, hidden], destination)?,
                )],
                destination,
            )?,
            destination,
        )?;
    } else {
        let intermediate = dim(args.intermediate_size, destination)?;
        push(
            &mut groups,
            partitioned_group(
                format_args!("{root}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                intermediate,
                [
                    member(
                        format_args!("{root}.mlp.gate_proj.weight"),
                        values([intermediate, hidden], destination)?,
                        partitioned(0),
                        destination,
                    )?,
                    member(
                        format_args!("{root}.mlp.up_proj.weight"),
                        values([intermediate, hidden], destination)?,
                        partitioned(0),
                        destination,
                    )?,
                    member(
                        format_args!("{root}.mlp.down_proj.weight"),
                        values([hidden, intermediate], destination)?,
                        partitioned(1),
                        destination,
                    )?,
                ],
                destination,
            )?,
            destination,
        )?;
    }
    push(
        &mut groups,
        replicated(
            format_args!("{root}.norms"),
            [
                (
                    format_args!("{root}.input_layernorm.weight"),
                    values([hidden], destination)?,
                ),
                (
                    format_args!("{root}.post_attention_layernorm.weight"),
                    values([hidden], destination)?,
                ),
                (
                    format_args!("{root}.pre_feedforward_layernorm.weight"),
                    values([hidden], destination)?,
                ),
                (
                    format_args!("{root}.post_feedforward_layernorm.weight"),
                    values([hidden], destination)?,
                ),
            ],
            destination,
        )?,
        destination,
    )?;
    expand(
        groups,
        |member| args.linear_format_for(member.target()),
        destination,
    )
}

/// Declares the patch/position roots, vision blocks, merge adapter, and projection.
pub(crate) fn vision_parameter_groups(
    args: &DecoderConfig,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls::<(
        &DecoderConfig,
        usize,
        usize,
        usize,
        usize,
        String,
        String,
        Vec<ParameterGroupSpec>,
        [ParameterMemberSpec; 8],
        [(String, Vec<usize>); 4],
        [Range<usize>; 2],
        Result<Vec<ParameterGroupSpec>, ParameterGroupError>,
    )>()?;

    let Some(config) = &args.vision_config else {
        return Ok(Vec::new());
    };
    let hidden = dim(config.hidden_size, destination)?;
    let _ = dim(config.num_heads, destination)?;
    let patch_input = dim(
        config.temporal_patch_size * 3 * config.patch_size * config.patch_size,
        destination,
    )?;
    let positions = dim(config.position_height * config.position_width, destination)?;
    let mut groups = values(
        [group_members(
            "model.vision_tower.patch_channels",
            ParameterRole::Channels,
            [
                member(
                    "model.vision_tower.patch_embedder.patch_embedding.weight",
                    values([hidden, patch_input], destination)?,
                    MemberSharding::Replicated,
                    destination,
                )?,
                member(
                    "model.vision_tower.patch_embedder.position_embedding_table.weight",
                    values([positions, hidden], destination)?,
                    MemberSharding::Replicated,
                    destination,
                )?,
            ],
            destination,
        )?],
        destination,
    )?;
    for layer in 0..config.layer_count() {
        let root = destination.text(format_args!("model.vision_tower.layers.{layer}"))?;
        push(
            &mut groups,
            group_members(
                format_args!("{root}.attention_heads"),
                ParameterRole::AttentionHeads,
                [
                    member(
                        format_args!("{root}.attn.q_proj.weight"),
                        values([hidden, hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.q_proj.bias"),
                        values([hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.k_proj.weight"),
                        values([hidden, hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.k_proj.bias"),
                        values([hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.v_proj.weight"),
                        values([hidden, hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.v_proj.bias"),
                        values([hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.proj.weight"),
                        values([hidden, hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.attn.proj.bias"),
                        values([hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                ],
                destination,
            )?,
            destination,
        )?;
        let intermediate = dim(config.intermediate_size, destination)?;
        push(
            &mut groups,
            group_members(
                format_args!("{root}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                [
                    member(
                        format_args!("{root}.mlp.fc1.weight"),
                        values([intermediate, hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.mlp.fc1.bias"),
                        values([intermediate], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.mlp.fc2.weight"),
                        values([hidden, intermediate], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                    member(
                        format_args!("{root}.mlp.fc2.bias"),
                        values([hidden], destination)?,
                        MemberSharding::Replicated,
                        destination,
                    )?,
                ],
                destination,
            )?,
            destination,
        )?;
        push(
            &mut groups,
            replicated(
                format_args!("{root}.norms"),
                [
                    (
                        format_args!("{root}.norm1.weight"),
                        values([hidden], destination)?,
                    ),
                    (
                        format_args!("{root}.norm1.bias"),
                        values([hidden], destination)?,
                    ),
                    (
                        format_args!("{root}.norm2.weight"),
                        values([hidden], destination)?,
                    ),
                    (
                        format_args!("{root}.norm2.bias"),
                        values([hidden], destination)?,
                    ),
                ],
                destination,
            )?,
            destination,
        )?;
    }
    let projector = dim(config.projector_hidden_size, destination)?;
    let shuffled = hidden
        .checked_mul(dim(config.merge_size * config.merge_size, destination)?)
        .ok_or_else(|| invalid("Muse-Glimmer shuffled width overflow", destination))?;
    push(
        &mut groups,
        group_members(
            "model.vision_adapter.intermediate",
            ParameterRole::FeedForwardIntermediate,
            [
                member(
                    "model.vision_adapter.fc1.weight",
                    values([projector, shuffled], destination)?,
                    MemberSharding::Replicated,
                    destination,
                )?,
                member(
                    "model.vision_adapter.fc2.weight",
                    values([projector, projector], destination)?,
                    MemberSharding::Replicated,
                    destination,
                )?,
            ],
            destination,
        )?,
        destination,
    )?;
    push(
        &mut groups,
        group_members(
            "model.vision_projection",
            ParameterRole::ColumnProjection,
            [member(
                "model.vision_projection.weight",
                values(
                    [dim(config.language_hidden_size, destination)?, projector],
                    destination,
                )?,
                MemberSharding::Replicated,
                destination,
            )?],
            destination,
        )?,
        destination,
    )?;
    push(
        &mut groups,
        replicated(
            "model.vision_tower.static_norms",
            [
                (
                    "model.vision_tower.ln_pre.weight",
                    values([hidden], destination)?,
                ),
                (
                    "model.vision_tower.ln_pre.bias",
                    values([hidden], destination)?,
                ),
                (
                    "model.vision_tower.ln_post.weight",
                    values([hidden], destination)?,
                ),
                (
                    "model.vision_tower.ln_post.bias",
                    values([hidden], destination)?,
                ),
            ],
            destination,
        )?,
        destination,
    )?;
    expand(
        groups,
        |member| config.linear_format_for(member.target()),
        destination,
    )
}

/// Declares only the pinned patch/position, merge, projection, and norm groups.
pub(crate) fn vision_static_parameter_groups(
    args: &DecoderConfig,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls::<(
        &DecoderConfig,
        usize,
        usize,
        usize,
        usize,
        String,
        String,
        Vec<ParameterGroupSpec>,
        [ParameterMemberSpec; 8],
        [(String, Vec<usize>); 4],
        [Range<usize>; 2],
        Result<Vec<ParameterGroupSpec>, ParameterGroupError>,
    )>()?;

    let Some(vision) = &args.vision_config else {
        return Ok(Vec::new());
    };
    let all = vision_parameter_groups(args, destination)?;
    let layer_groups = vision.layer_count() * 3;
    let mut selected = destination.vector(4)?;
    selected.extend(
        all.into_iter()
            .enumerate()
            .filter_map(|(index, group)| (index == 0 || index > layer_groups).then_some(group)),
    );
    Ok(selected)
}

/// Declares exactly one architecture-global Muse vision execution unit.
pub(crate) fn vision_layer_parameter_groups(
    args: &DecoderConfig,
    layer: usize,
    destination: DeclarationDestination<'_>,
) -> Result<Vec<ParameterGroupSpec>, ParameterGroupError> {
    destination.controls::<(
        &DecoderConfig,
        usize,
        usize,
        usize,
        usize,
        String,
        String,
        Vec<ParameterGroupSpec>,
        [ParameterMemberSpec; 8],
        [(String, Vec<usize>); 4],
        [Range<usize>; 2],
        Result<Vec<ParameterGroupSpec>, ParameterGroupError>,
    )>()?;

    let count = args
        .vision_config
        .as_ref()
        .map_or(0, |vision| vision.layer_count());
    if layer >= count {
        return Err(invalid(
            format_args!("Muse-Glimmer vision layer {layer} is outside {count} layers"),
            destination,
        ));
    }
    let all = vision_parameter_groups(args, destination)?;
    let start = 1 + layer * 3;
    let mut selected = destination.vector(3)?;
    selected.extend(all.into_iter().skip(start).take(3));
    Ok(selected)
}
