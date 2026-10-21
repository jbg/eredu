//! Backend-neutral Qwen3-VL ownership composed from shared text and vision specifications.

use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureParameterDescription, ComponentDefinition, ComponentUnit, ExecutionGraph,
    ExecutionGroupId, ExecutionGroupSpec, ExecutionUnitLayout, OwnedParameterGroupSpec,
    ParameterGroupOwner,
};

use super::{LocalGeometry, ModelArgs, TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP};

/// Declares checkpoint-global parameters without constructing native modules.
pub fn parameter_description(args: &ModelArgs) -> Result<ArchitectureParameterDescription, Error> {
    args.vision
        .validate_for(crate::qwen::vision::VisionMode::DeepStack)
        .map_err(Error::backend)?;
    let text =
        crate::qwen::parallel::parameter_description(&args.text).map_err(Error::backend_source)?;
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(VISION_EXECUTION_GROUP),
            ExecutionGroupSpec::with_dependencies(TEXT_EXECUTION_GROUP, [VISION_EXECUTION_GROUP]),
        ],
        TEXT_EXECUTION_GROUP,
    )
    .map_err(Error::backend_source)?;
    let layout = ExecutionUnitLayout::new(
        &graph,
        [
            args.vision.layer_count(),
            usize::try_from(args.text.num_hidden_layers).map_err(Error::backend_source)?,
        ],
    )
    .map_err(Error::backend_source)?;
    let vision_id = ExecutionGroupId::new(VISION_EXECUTION_GROUP).map_err(Error::backend_source)?;
    let text_id = ExecutionGroupId::new(TEXT_EXECUTION_GROUP).map_err(Error::backend_source)?;
    let mut owned = text
        .groups()
        .iter()
        .filter(|group| !matches!(group.owner(), ParameterGroupOwner::ExecutionUnit { .. }))
        .cloned()
        .collect::<Vec<_>>();
    owned.extend(
        crate::qwen::vision::owned_static_parallel_parameter_groups(
            &args.vision,
            "model.visual",
            vision_id.clone(),
            "vision",
        )
        .map_err(Error::backend_source)?,
    );
    for index in 0..args.vision.layer_count() {
        owned.extend(
            crate::qwen::vision::block_parallel_parameter_groups(
                &args.vision,
                "model.visual",
                index,
            )
            .map_err(Error::backend_source)?
            .into_iter()
            .map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(vision_id.clone(), index),
                    group,
                )
            }),
        );
    }
    owned.extend(
        text.groups()
            .iter()
            .filter_map(|group| match group.owner() {
                ParameterGroupOwner::ExecutionUnit { global_unit, .. } => {
                    Some(OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(text_id.clone(), *global_unit),
                        group.group().clone(),
                    ))
                }
                _ => None,
            }),
    );
    let expected = owned
        .iter()
        .map(|group| group.group().clone())
        .collect::<Vec<_>>();
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
        .map_err(Error::backend_source)
}

/// Validates one semantic definition before native construction or execution selection.
pub fn component_definition(
    args: &ModelArgs,
    geometry: Option<&LocalGeometry>,
) -> Result<ComponentDefinition, Error> {
    let parameters = parameter_description(args)?;
    let state = match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            geometry.state_layout().clone()
        }
        None => super::state_layout(args).map_err(Error::backend_source)?,
    };
    let layout = parameters.unit_layout();
    let units = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared Qwen3-VL unit");
            let index = address.index();
            match address.group() {
                0 => ComponentUnit {
                    path: format!("model.visual.blocks.{index}"),
                    state: 0..0,
                },
                _ => ComponentUnit {
                    path: format!("{}.layers.{index}", args.text.parameter_root),
                    state: index..index + 1,
                },
            }
        })
        .collect();
    let transport = vec![
        eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            first_owner_static_roles: vec!["vision".into()],
            last_owner_static_roles: Vec::new(),
            merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
            parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
            request_optional: true,
        },
        crate::transport::decoder(),
    ];
    let partition = crate::transport::pipeline_state(1, &state);
    ComponentDefinition::new(
        "qwen-vl",
        parameters,
        units,
        transport,
        TEXT_EXECUTION_GROUP,
        Vec::new(),
        state,
        partition,
    )
    .map_err(Error::from)
}
