//! Source-free Inkling invocation, storage, state and boundary declarations.
use super::{
    InklingStateLayouts, AUDIO_EXECUTION_GROUP, MTP_STATIC_ROLE, TEXT_EXECUTION_GROUP,
    VISION_EXECUTION_GROUP,
};
use crate::inkling::{
    layer_parameter_groups, mtp_parameter_groups, mtp_state_layout, state_layout,
    vision_layer_parameter_groups, LocalGeometry, ModelArgs,
};
use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, OwnedParameterGroupSpec,
    ParameterGroupOwner, PartitionState,
};

fn declare_parameters(args: &ModelArgs) -> Result<ArchitectureParameterDescription, Error> {
    let graph = ExecutionGraph::new(
        vec![
            eredu_runtime::ExecutionGroupSpec::root(VISION_EXECUTION_GROUP),
            eredu_runtime::ExecutionGroupSpec::root(AUDIO_EXECUTION_GROUP),
            eredu_runtime::ExecutionGroupSpec::with_dependencies(
                TEXT_EXECUTION_GROUP,
                [VISION_EXECUTION_GROUP, AUDIO_EXECUTION_GROUP],
            ),
        ],
        TEXT_EXECUTION_GROUP,
    )
    .map_err(Error::backend)?;
    let counts = [
        args.vision_config
            .as_ref()
            .map_or(0, |vision| vision.num_hidden_layers as usize),
        0,
        args.text_config.num_hidden_layers as usize,
    ];
    let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
    let static_groups =
        crate::inkling::parallel::pinned_parameter_groups(args).map_err(Error::backend)?;
    let mtp_groups = mtp_parameter_groups(args).map_err(Error::backend)?;
    let mut expected = static_groups
        .iter()
        .map(|(_, group)| group)
        .chain(&mtp_groups)
        .cloned()
        .collect::<Vec<_>>();
    let mut owned = static_groups
        .into_iter()
        .map(|(role, group)| {
            let owner = if role == "embedding" {
                ParameterGroupOwner::static_any_of(["embedding", MTP_STATIC_ROLE])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .chain(mtp_groups.into_iter().map(|group| {
            OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role(MTP_STATIC_ROLE), group)
        }))
        .collect::<Vec<_>>();
    for (group_index, &count) in counts.iter().enumerate() {
        let owner_group = layout
            .group_id(group_index)
            .expect("Inkling layout group")
            .clone();
        for index in 0..count {
            let groups = if group_index == 0 {
                let vision = args
                    .vision_config
                    .as_ref()
                    .expect("vision group configured");
                vision_layer_parameter_groups(vision, index)
            } else {
                layer_parameter_groups(args, index)
            }
            .map_err(Error::backend)?;
            expected.extend(groups.iter().cloned());
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                    group,
                )
            }));
        }
    }
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned).map_err(Error::backend)
}

pub(super) fn group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
    match group {
        0 => eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            first_owner_static_roles: vec!["vision".into()],
            last_owner_static_roles: Vec::new(),
            merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
            parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
            request_optional: true,
        },
        1 => eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::AudioEncoder,
            first_owner_static_roles: vec!["audio".into()],
            last_owner_static_roles: Vec::new(),
            merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
            parallel_subgroup: None,
            request_optional: true,
        },
        _ => eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec!["embedding".into(), "embedding_norm".into()],
            last_owner_static_roles: vec!["norm".into(), "output".into(), MTP_STATIC_ROLE.into()],
            merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::Decoder),
            request_optional: false,
        },
    }
}

pub(super) fn state_partition_plan(
    layout: &eredu_runtime::StateLayout,
) -> eredu_runtime::ArchitectureStatePartitionPlan {
    let target_layers = layout
        .segments()
        .iter()
        .find(|segment| segment.id().as_str() == crate::inkling::TARGET_STATE_SEGMENT)
        .map(|segment| segment.layers().end)
        .unwrap_or(layout.len());
    crate::transport::pipeline_with_output_state(2, target_layers, layout)
}

/// Validates Inkling resource ownership before selection or native construction.
pub fn component_definition(
    args: &ModelArgs,
    geometry: Option<&LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    args.validate().map_err(Error::backend_source)?;
    let parameters = declare_parameters(args)?;
    let target = match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            geometry.state_layout().clone()
        }
        None => state_layout(args).map_err(Error::backend_source)?,
    };
    let prediction = match geometry {
        Some(geometry) => geometry.prediction_state().cloned(),
        None => mtp_state_layout(args)
            .map_err(Error::backend_source)?
            .map(|layout| PartitionState::new(layout, target.len()))
            .transpose()
            .map_err(Error::backend_source)?,
    };
    let state = InklingStateLayouts::new(target, prediction)?.composite;
    let units = (0..parameters.unit_layout().len())
        .map(|ordinal| {
            let address = parameters
                .unit_layout()
                .address(ordinal)
                .expect("declared Inkling unit");
            let index = address.index();
            match address.group() {
                0 => eredu_runtime::ComponentUnit {
                    path: format!("visual.layers.{index}"),
                    state: 0..0,
                },
                2 => eredu_runtime::ComponentUnit {
                    path: format!("model.layers.{index}"),
                    state: index..index + 1,
                },
                _ => unreachable!("Inkling audio has pinned resources and no execution units"),
            }
        })
        .collect();
    let ownership = state_partition_plan(&state);
    eredu_runtime::ComponentDefinition::new(
        "inkling",
        parameters,
        units,
        (0..3).map(group_transport).collect(),
        TEXT_EXECUTION_GROUP,
        vec![],
        state,
        ownership,
    )
    .map_err(Error::from)
}

/// Projects the canonical parameter catalog without allocating a backend resource.
pub fn parameter_description(args: &ModelArgs) -> Result<ArchitectureParameterDescription, Error> {
    component_definition(args, None).map(eredu_runtime::ComponentDefinition::into_parameters)
}
