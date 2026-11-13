//! Composite invocation and parameter ownership for LFM3 vision-language.

use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ComponentDefinition,
    ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec, ParameterRole,
};

use super::{Declaration, VisionUnit};

pub(super) const VISION_GROUP: &str = "vision";

pub(super) fn definition(
    language: &crate::lfm3::Declaration,
    vision: &Declaration,
) -> Result<ComponentDefinition, Error> {
    let decoder = language.definition();
    let text_group = crate::decoder::TARGET_EXECUTION_GROUP;
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(VISION_GROUP),
            ExecutionGroupSpec::with_dependencies(text_group, [VISION_GROUP]),
        ],
        text_group,
    )
    .map_err(Error::backend_source)?;
    let layout = ExecutionUnitLayout::new(&graph, [vision.units().count(), decoder.unit_count(0)?])
        .map_err(Error::backend_source)?;
    let mut groups = decoder.parameters().groups().to_vec();
    let mut units = Vec::with_capacity(layout.len());
    let vision_id = layout.group_id(0).expect("declared vision group");
    for (ordinal, unit) in vision.units().enumerate() {
        let path = match unit {
            VisionUnit::Patchifier => "model.vision_tower.patchifier".to_owned(),
            VisionUnit::Block(layer) => format!("model.vision_tower.layers.{layer}"),
            VisionUnit::Projector => "model.multi_modal_projector".to_owned(),
        };
        let members = vision.unit_members(unit)?;
        let group = ParameterGroupSpec::new(&path, ParameterRole::Replicated, members)
            .map_err(Error::backend_source)?;
        groups.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(vision_id.clone(), ordinal),
            group,
        ));
        units.push(ComponentUnit { path, state: 0..0 });
    }
    for index in 0..decoder.unit_count(0)? {
        units.push(decoder.unit(0, index)?.clone());
    }
    let parameters = ArchitectureParameterDescription::from_owned(&graph, &layout, groups)
        .map_err(Error::backend_source)?;
    let state = decoder.state_layout().clone();
    let state_partition = crate::transport::pipeline_state(1, &state);
    ComponentDefinition::new(
        "lfm3-vl",
        parameters,
        units,
        vec![
            ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::VisionEncoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: None,
                request_optional: true,
            },
            decoder.group_transport(0)?.clone(),
        ],
        text_group,
        Vec::new(),
        state,
        state_partition,
    )
    .map_err(Error::from)
}
