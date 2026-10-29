//! Backend-free ownership and transforms for a shared base and separate draft LoRA.
use super::FamilyConfig;
use crate::decoder::{self, Config};
use eredu_runtime::{
    ArchitectureParameterDescription, MemberSharding, OwnedParameterGroupSpec, ParallelPlanError,
    ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
};

/// Optional Pixtral traversal precedes the same canonical text group.
pub fn composite_graph(
    config: &FamilyConfig,
) -> Result<
    (
        eredu_runtime::ExecutionGraph,
        eredu_runtime::ExecutionUnitLayout,
    ),
    ParallelPlanError,
> {
    config
        .validate_config()
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let vision = config.vision.as_ref().ok_or_else(|| {
        ParallelPlanError::InvalidGroup("composite graph requires Pixtral".into())
    })?;
    let graph = eredu_runtime::ExecutionGraph::new(
        vec![
            eredu_runtime::ExecutionGroupSpec::root("vision"),
            eredu_runtime::ExecutionGroupSpec::with_dependencies("vision_projection", ["vision"]),
            eredu_runtime::ExecutionGroupSpec::with_dependencies(
                decoder::TEXT_DECODER_EXECUTION_GROUP,
                ["vision_projection"],
            ),
        ],
        decoder::TEXT_DECODER_EXECUTION_GROUP,
    )
    .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let layout = eredu_runtime::ExecutionUnitLayout::new(
        &graph,
        [
            vision.num_hidden_layers as usize,
            1,
            config.num_hidden_layers() as usize,
        ],
    )
    .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    Ok((graph, layout))
}

/// Source-free transport and static ownership shared by admission and construction.
pub fn group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
    if group == 2 {
        let mut transport = crate::transport::decoder();
        transport.placement = eredu_runtime::ArchitectureGroupPlacement::PipelineSequence;
        return transport;
    }
    eredu_runtime::ArchitectureGroupTransport {
        placement: eredu_runtime::ArchitectureGroupPlacement::PipelineSequence,
        kind: if group == 0 {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder
        } else {
            eredu_runtime::ArchitectureGroupKind::Projector
        },
        first_owner_static_roles: if group == 0 {
            vec!["vision_embedding".into()]
        } else {
            vec![]
        },
        last_owner_static_roles: vec![],
        merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: (group == 0)
            .then_some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
        request_optional: group == 0,
    }
}

/// Complete cold catalog, with independently owned image ingress, projector,
/// and head/channel-sharded tower layers. Text storage identities are unchanged.
pub fn composite_parameter_description(
    config: &FamilyConfig,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let (graph, layout) = composite_graph(config)?;
    let vision = config.vision.as_ref().expect("validated composite graph");
    let width = vision.hidden_size as usize;
    let text = config.hidden_size() as usize;
    let merge = config.spatial_merge_size as usize;
    let mut owned = parameter_description(config)?.groups().to_vec();
    for (role, name, shape) in [
        (
            "vision_embedding",
            "model.vision_tower.patch_conv",
            vec![
                width,
                vision.num_channels as usize,
                vision.patch_size as usize,
                vision.patch_size as usize,
            ],
        ),
        ("vision_embedding", "model.vision_tower.ln_pre", vec![width]),
        (
            "vision_projection",
            "model.multi_modal_projector.norm",
            vec![width],
        ),
        (
            "vision_projection",
            "model.multi_modal_projector.patch_merger.merging_layer",
            vec![width, width * merge * merge],
        ),
        (
            "vision_projection",
            "model.multi_modal_projector.linear_1",
            vec![text, width],
        ),
        (
            "vision_projection",
            "model.multi_modal_projector.linear_2",
            vec![text, text],
        ),
    ] {
        owned.push(OwnedParameterGroupSpec::new(
            if role == "vision_projection" {
                ParameterGroupOwner::execution_unit(layout.group_id(1).unwrap().clone(), 0)
            } else {
                ParameterGroupOwner::static_role(role)
            },
            ParameterGroupSpec::new(
                name,
                ParameterRole::Replicated,
                [ParameterMemberSpec::new(
                    format!("{name}.weight"),
                    shape,
                    MemberSharding::Replicated,
                )],
            )?,
        ));
    }
    for index in 0..vision.num_hidden_layers as usize {
        for group in vision_layer_groups(vision, index)? {
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), index),
                group,
            ));
        }
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))
}

/// Counts shared base groups once and gives each adapter matrix the same layer owner.
pub fn parameter_description(
    config: &FamilyConfig,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let base = decoder::dense_parameter_description(config)?;
    let Some(adapter) = config.draft_adapter() else {
        return Ok(base);
    };
    let mut owned = base.groups().to_vec();
    let group = base
        .unit_layout()
        .group_id(0)
        .expect("validated text graph")
        .clone();
    for index in 0..config.num_hidden_layers() as usize {
        let name = format!("draft_adapter.layers.{index}.o_proj");
        let parameters = ParameterGroupSpec::partitioned(
            &name,
            ParameterRole::AttentionHeads,
            config.num_attention_heads() as usize,
            [
                ParameterMemberSpec::new(
                    format!("{name}.lora_A.weight"),
                    vec![
                        adapter.rank() as usize,
                        (config.num_attention_heads() * config.head_dim()) as usize,
                    ],
                    MemberSharding::Partitioned { axis: 1 },
                ),
                ParameterMemberSpec::new(
                    format!("{name}.lora_B.weight"),
                    vec![config.hidden_size() as usize, adapter.rank() as usize],
                    MemberSharding::Replicated,
                ),
            ],
        )?;
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(group.clone(), index),
            parameters,
        ));
    }
    ArchitectureParameterDescription::from_owned(base.graph(), base.unit_layout(), owned)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))
}

/// Exact Pixtral head/channel transforms. Both branch norms remain replicated
/// because each normalizes the complete residual width before projection.
pub fn vision_layer_groups(
    config: &super::PixtralConfig,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    config.validate().map_err(ParallelPlanError::InvalidGroup)?;
    if index >= config.num_hidden_layers as usize {
        return Err(ParallelPlanError::InvalidGroup(
            "Pixtral layer outside tower".into(),
        ));
    }
    let root = format!("model.vision_tower.transformer.layers.{index}");
    let width = config.hidden_size as usize;
    let intermediate = config.intermediate_size as usize;
    let mut groups = Vec::new();
    for field in ["attention_norm", "ffn_norm"] {
        groups.push(ParameterGroupSpec::new(
            format!("{root}.{field}"),
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                format!("{root}.{field}.weight"),
                vec![width],
                MemberSharding::Replicated,
            )],
        )?);
    }
    groups.push(ParameterGroupSpec::partitioned(
        format!("{root}.attention"),
        ParameterRole::AttentionHeads,
        config.num_attention_heads as usize,
        ["q_proj", "k_proj", "v_proj", "o_proj"].map(|field| {
            ParameterMemberSpec::new(
                format!("{root}.attention.{field}.weight"),
                vec![width, width],
                MemberSharding::Partitioned {
                    axis: if field == "o_proj" { 1 } else { 0 },
                },
            )
        }),
    )?);
    groups.push(ParameterGroupSpec::partitioned(
        format!("{root}.feed_forward"),
        ParameterRole::FeedForwardIntermediate,
        intermediate,
        ["gate_proj", "up_proj", "down_proj"].map(|field| {
            ParameterMemberSpec::new(
                format!("{root}.feed_forward.{field}.weight"),
                if field == "down_proj" {
                    vec![width, intermediate]
                } else {
                    vec![intermediate, width]
                },
                MemberSharding::Partitioned {
                    axis: if field == "down_proj" { 1 } else { 0 },
                },
            )
        }),
    )?);
    Ok(groups)
}

/// Proves aligned head/channel selections before any native layer is declared.
pub(crate) fn vision_local_widths(
    config: &super::PixtralConfig,
    index: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<(i32, i32), ParallelPlanError> {
    use eredu_runtime::TensorPlacement;
    let invalid = |message: &str| ParallelPlanError::InvalidGroup(message.into());
    let mut heads = None;
    let mut channels = None;
    for group in vision_layer_groups(config, index)? {
        let mut selected = None;
        for member in group.members() {
            let tensor = layout
                .tensor(member.target())
                .ok_or_else(|| invalid("missing selected Pixtral parameter"))?;
            if tensor.global_shape() != member.global_shape()
                || !tensor.additional_placements().is_empty()
            {
                return Err(invalid(
                    "Pixtral placement differs from admitted tensor geometry",
                ));
            }
            let MemberSharding::Partitioned { axis } = member.sharding() else {
                if tensor.local_shape() != member.global_shape()
                    || *tensor.placement() != TensorPlacement::Replicated
                {
                    return Err(invalid(
                        "Pixtral RMS scale requires the complete residual width",
                    ));
                }
                continue;
            };
            let extent = member.global_shape()[*axis];
            let range = match tensor.placement() {
                TensorPlacement::Replicated => 0..extent,
                TensorPlacement::Range {
                    axis: placed,
                    start,
                    end,
                } if placed == axis && start < end && *end <= extent => *start..*end,
                TensorPlacement::Shard {
                    axis: placed,
                    index,
                    parts,
                } if placed == axis && *parts > 0 && index < parts && extent % parts == 0 => {
                    index * (extent / parts)..(index + 1) * (extent / parts)
                }
                _ => {
                    return Err(invalid(
                        "Pixtral projection has no exact uniform axis selection",
                    ));
                }
            };
            let mut expected = member.global_shape().to_vec();
            expected[*axis] = range.len();
            if tensor.local_shape() != expected {
                return Err(invalid(
                    "Pixtral projection local shape differs from its selected range",
                ));
            }
            if let Some(previous) = &selected {
                if previous != &range {
                    return Err(invalid(
                        "Pixtral branch projections select different head or channel ranges",
                    ));
                }
            } else {
                selected = Some(range);
            }
        }
        if let Some(range) = selected {
            match group.role() {
                ParameterRole::AttentionHeads => {
                    if range.start % config.head_dim as usize != 0
                        || range.end % config.head_dim as usize != 0
                    {
                        return Err(invalid("Pixtral selection splits a rotary attention head"));
                    }
                    heads = Some(
                        i32::try_from(range.len() / config.head_dim as usize)
                            .map_err(|_| invalid("Pixtral local head count exceeds i32"))?,
                    );
                }
                ParameterRole::FeedForwardIntermediate => {
                    channels = Some(
                        i32::try_from(range.len())
                            .map_err(|_| invalid("Pixtral intermediate width exceeds i32"))?,
                    )
                }
                _ => return Err(invalid("invalid Pixtral partition domain")),
            }
        }
    }
    Ok((
        heads.ok_or_else(|| invalid("missing Pixtral attention selection"))?,
        channels.ok_or_else(|| invalid("missing Pixtral intermediate selection"))?,
    ))
}

/// Selects independently addressable text layers, static ownership, exact local
/// cache geometry and request-semantic transport before backend construction.
pub fn partition(
    config: &FamilyConfig,
    definition: &eredu_runtime::ComponentDefinition,
    layout: &eredu_runtime::LocalModelLayout,
    units: std::ops::Range<usize>,
    ownership: eredu_runtime::PartitionOwnership,
) -> Result<
    eredu_runtime::ArchitecturePartition<
        decoder::PartitionLocalGeometry<FamilyConfig>,
        super::partitioned::TextBoundarySchema,
    >,
    ParallelPlanError,
> {
    super::definition::validate_text_identity(definition)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    definition
        .validate_parameter_topology(&parameter_description(config)?)
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    let parameters = definition.parameters();
    let geometry =
        decoder::partition_local_geometry(config, parameters.invocations(), layout, units.clone())?;
    let definition = definition
        .clone()
        .with_state_layout(geometry.complete_state_layout().clone())
        .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
    eredu_runtime::ArchitecturePartition::from_description(
        parameters,
        [(decoder::TEXT_DECODER_EXECUTION_GROUP, units)],
        ownership,
        definition.state_layout(),
        definition.state_partition_plan(),
        geometry.clone(),
        super::partitioned::TextBoundarySchema::new(config.hidden_size()),
    )
    .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))
}
