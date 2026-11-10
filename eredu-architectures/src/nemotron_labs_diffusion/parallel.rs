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
    use eredu_nn::DeclareLogicalParameters;
    let declaration = super::vision::specs::VisionSpec::new(config)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
    let mut owned = parameter_description(config)?.groups().to_vec();
    for (spec, owner) in [
        (
            &declaration.embedding as &dyn DeclareLogicalParameters,
            ParameterGroupOwner::static_role("vision_embedding"),
        ),
        (
            &declaration.projector as &dyn DeclareLogicalParameters,
            ParameterGroupOwner::execution_unit(layout.group_id(1).unwrap().clone(), 0),
        ),
    ] {
        for parameter in spec
            .logical_parameters()
            .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        {
            owned.push(OwnedParameterGroupSpec::new(
                owner.clone(),
                ParameterGroupSpec::new(
                    parameter
                        .parameter
                        .group
                        .as_ref()
                        .expect("declared image parameter owner"),
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::from_logical_declaration(
                        &parameter,
                        MemberSharding::Replicated,
                    )],
                )?,
            ));
        }
    }
    for (index, layer) in declaration.layers.iter().enumerate() {
        for group in layer.parameter_groups()? {
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
    let base = config
        .text_declaration()
        .parameter_description(config, decoder::ParameterEncoding::Stored)?;
    config.extend_common_parameters(base)
}

pub(crate) fn with_draft_parameters(
    base: ArchitectureParameterDescription,
    adapter: Option<&super::adapter::DraftAdapterDeclaration>,
) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
    let Some(adapter) = adapter else {
        return Ok(base);
    };
    let mut owned = base.groups().to_vec();
    let group = base
        .unit_layout()
        .group_id(0)
        .expect("validated text graph")
        .clone();
    for (index, layer) in adapter.layers().iter().enumerate() {
        let parameters = layer.parameter_group()?;
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
    config
        .tower()
        .layer(index)
        .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?
        .parameter_groups()
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
    let geometry = decoder::partition_local_geometry(
        config,
        config.text_declaration().as_ref(),
        parameters.invocations(),
        layout,
        units.clone(),
    )?;
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
