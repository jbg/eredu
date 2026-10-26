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
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned)
        .map_err(Error::backend_source)
}

/// Validates one semantic definition before native construction or execution selection.
pub fn component_definition(
    args: &ModelArgs,
    geometry: Option<&LocalGeometry>,
) -> Result<ComponentDefinition, Error> {
    let parameters = parameter_description(args)?;
    let state = component_state(args, geometry)?;
    let units = component_units(args, &parameters);
    let transport = component_transport();
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

fn component_state(
    args: &ModelArgs,
    geometry: Option<&LocalGeometry>,
) -> Result<eredu_runtime::StateLayout, Error> {
    match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            Ok(geometry.state_layout().clone())
        }
        None => super::state_layout(args).map_err(Error::backend_source),
    }
}

fn component_units(
    args: &ModelArgs,
    parameters: &ArchitectureParameterDescription,
) -> Vec<ComponentUnit> {
    let layout = parameters.unit_layout();
    (0..layout.len())
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
        .collect()
}

fn component_transport() -> Vec<eredu_runtime::ArchitectureGroupTransport> {
    vec![
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
    ]
}

fn invalid(definition: &ComponentDefinition, resource: impl Into<String>) -> Error {
    eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    }
    .into()
}
fn validate_identity(definition: &ComponentDefinition) -> Result<(), Error> {
    if definition.id() != "qwen-vl" {
        return Err(invalid(definition, "Qwen3-VL component identity"));
    }
    Ok(())
}

/// Selects physical formats while retaining canonical consumers, state and observation seams.
pub fn project_parameters(
    args: &ModelArgs,
    definition: &std::sync::Arc<ComponentDefinition>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    validate_identity(definition)?;
    let parameters = parameter_description(args)?;
    definition
        .project_parameters(parameters)
        .map_err(Error::from)
}

pub(crate) fn select_definition(
    args: &ModelArgs,
    definition: std::sync::Arc<ComponentDefinition>,
    geometry: Option<&LocalGeometry>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    validate_identity(&definition)?;
    definition.validate_parameter_topology(&parameter_description(args)?)?;
    let state = component_state(args, geometry)?;
    let global = component_state(args, None)?;
    if definition.state_layout() != &global && definition.state_layout() != &state {
        return Err(invalid(
            &definition,
            "Qwen3-VL retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained = super::local_geometry(args, layout).map_err(Error::backend_source)?;
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: args.text.vocab_size as usize,
            local: 0..args.text.vocab_size as usize,
        };
        let output = if args.text.tie_word_embeddings {
            None
        } else {
            Some(
                geometry
                    .and_then(|geometry| geometry.text().output_range())
                    .unwrap_or(&full),
            )
        };
        let unsharded =
            crate::qwen::vision::VisionStaticSelection::all(args.vision.clone(), "model.visual")
                .and_then(|selection| selection.unsharded_merger_intermediates())
                .map_err(Error::backend_source)?;
        if retained.state_layout() != &state
            || retained.text().embedding_range()
                != geometry.map_or(&full, |geometry| geometry.text().embedding_range())
            || retained.text().output_range() != output
            || retained.merger_widths()
                != geometry.map_or(unsharded.as_slice(), LocalGeometry::merger_widths)
            || (0..args.vision.layer_count()).any(|index| {
                retained.vision_block(index)
                    != geometry.map_or(
                        Some((args.vision.num_heads, args.vision.intermediate_size)),
                        |geometry| geometry.vision_block(index),
                    )
            })
            || retained
                .text()
                .blocks()
                .iter()
                .enumerate()
                .any(|(index, block)| {
                    crate::qwen::prompt_cache_architecture_fingerprint(block)
                        != crate::qwen::prompt_cache_architecture_fingerprint(
                            geometry
                                .and_then(|geometry| geometry.text().block(index))
                                .unwrap_or(&args.text),
                        )
                })
        {
            return Err(invalid(
                &definition,
                "Qwen3-VL retained placement differs from selected geometry",
            ));
        }
    }
    definition.validate_execution_contract(
        TEXT_EXECUTION_GROUP,
        &[],
        &component_units(args, definition.parameters()),
        &component_transport(),
        &crate::transport::pipeline_state(1, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    if definition.state_layout() == &state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state)
            .map(std::sync::Arc::new)
            .map_err(Error::from)
    }
}
