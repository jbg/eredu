//! Source-free Gemma invocation, storage, state and transport declarations.
use crate::gemma4::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    modality_projection_parameter_groups, state_layout, vision_layer_parameter_groups,
    vision_static_parameter_groups, FamilyConfig, LocalGeometry, AUDIO_EXECUTION_GROUP,
    TEXT_EXECUTION_GROUP, VISION_EXECUTION_GROUP,
};
use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    OwnedParameterGroupSpec, ParallelPlanError, ParameterGroupOwner,
};
fn text_static_parameter_ownership(
    args: &crate::gemma4::ModelArgs,
) -> Result<Vec<OwnedParameterGroupSpec>, ParallelPlanError> {
    Ok(crate::gemma4::parallel::pinned_parameter_groups(args)?
        .into_iter()
        .map(|(role, group)| {
            let owner = if role == "embedding" && args.tie_word_embeddings {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect())
}
fn declare_parameters(args: &FamilyConfig) -> Result<ArchitectureParameterDescription, Error> {
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(VISION_EXECUTION_GROUP),
            ExecutionGroupSpec::root(AUDIO_EXECUTION_GROUP),
            ExecutionGroupSpec::with_dependencies(
                TEXT_EXECUTION_GROUP,
                [VISION_EXECUTION_GROUP, AUDIO_EXECUTION_GROUP],
            ),
        ],
        TEXT_EXECUTION_GROUP,
    )
    .map_err(Error::backend)?;
    let counts = [
        args.vision
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize),
        args.audio
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize),
        args.text.num_hidden_layers(),
    ];
    let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
    let mut owned = text_static_parameter_ownership(&args.text).map_err(Error::backend)?;
    {
        let mut add_static = |role: &'static str, groups: Vec<_>| {
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role(role), group)
            }));
        };
        if let Some(vision) = &args.vision {
            add_static(
                "vision",
                vision_static_parameter_groups(vision).map_err(Error::backend)?,
            );
        }
        if let Some(config) = &args.vision {
            add_static(
                "vision_projection",
                modality_projection_parameter_groups(
                    "model.vision_projector",
                    "embed_vision",
                    config.hidden_size,
                    args.text.hidden_size,
                    config.linear_format_for(
                        "model.embed_vision.embedding_projection.weight",
                        config.hidden_size,
                    ),
                )
                .map_err(Error::backend)?,
            );
        }
        if let Some(audio) = &args.audio {
            add_static(
                "audio",
                audio_static_parameter_groups(audio).map_err(Error::backend)?,
            );
        }
        if let Some(config) = &args.audio {
            add_static(
                "audio_projection",
                modality_projection_parameter_groups(
                    "model.audio_projector",
                    "embed_audio",
                    config.output_proj_dims,
                    args.text.hidden_size,
                    config.linear_format_for(
                        "model.embed_audio.embedding_projection.weight",
                        config.output_proj_dims,
                    ),
                )
                .map_err(Error::backend)?,
            );
        }
    }
    for (group_index, &count) in counts.iter().enumerate() {
        let owner_group = layout
            .group_id(group_index)
            .expect("Gemma layout group")
            .clone();
        for index in 0..count {
            let groups = match group_index {
                0 => vision_layer_parameter_groups(
                    args.vision.as_ref().expect("vision group configured"),
                    index,
                ),
                1 => audio_layer_parameter_groups(
                    args.audio.as_ref().expect("audio group configured"),
                    index,
                ),
                _ => layer_parameter_groups(&args.text, index),
            }
            .map_err(Error::backend)?;
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                    group,
                )
            }));
        }
    }
    ArchitectureParameterDescription::from_owned(&graph, &layout, owned).map_err(Error::backend)
}
pub(super) fn group_transport(
    args: &FamilyConfig,
    group: usize,
) -> eredu_runtime::ArchitectureGroupTransport {
    match group {
        0 | 1 => crate::gemma4::pipeline::media_transport(args, group),
        _ => eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec![
                "embedding".into(),
                "per_layer_embedding".into(),
                "per_layer_projection".into(),
                "per_layer_norm".into(),
            ],
            last_owner_static_roles: if args.text.tie_word_embeddings {
                vec!["norm".into(), "embedding".into()]
            } else {
                vec!["norm".into(), "output".into()]
            },
            merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::Decoder),
            request_optional: false,
        },
    }
}

pub(super) fn state_partition_plan(
    layout: &eredu_runtime::StateLayout,
) -> eredu_runtime::ArchitectureStatePartitionPlan {
    crate::transport::pipeline_state(2, layout)
}

/// Validates canonical Gemma resources and local state before native construction.
pub fn component_definition(
    args: &FamilyConfig,
    geometry: Option<&LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    args.validate().map_err(Error::backend_source)?;
    definition_with_state(args, component_state(args, geometry)?)
}

fn component_state(
    args: &FamilyConfig,
    geometry: Option<&LocalGeometry>,
) -> Result<eredu_runtime::StateLayout, Error> {
    match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            Ok(geometry.state_layout().clone())
        }
        None => state_layout(&args.text).map_err(Error::backend_source),
    }
}

fn partition_state(
    args: &FamilyConfig,
    local: &LocalGeometry,
    geometry: &crate::gemma4::PartitionLocalGeometry,
) -> Result<eredu_runtime::StateLayout, Error> {
    args.validate().map_err(Error::backend_source)?;
    local.validate_for(args).map_err(Error::backend_source)?;
    geometry.validate_for(args).map_err(Error::backend_source)?;
    let expected = crate::gemma4::pipeline::partition_state_layout(
        &args.text,
        local.state_layout(),
        geometry.text_units(),
    )
    .map_err(Error::backend_source)?;
    if &expected != geometry.complete_state_layout() {
        return Err(Error::backend(
            "Gemma partition receiver state differs from its tensor geometry",
        ));
    }
    Ok(expected)
}

fn definition_with_state(
    args: &FamilyConfig,
    state: eredu_runtime::StateLayout,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let parameters = declare_parameters(args)?;
    let units = component_units(&parameters);
    let ownership = state_partition_plan(&state);
    eredu_runtime::ComponentDefinition::new(
        "gemma4",
        parameters,
        units,
        (0..3).map(|group| group_transport(args, group)).collect(),
        TEXT_EXECUTION_GROUP,
        vec![],
        state,
        ownership,
    )
    .map_err(Error::from)
}

/// Projects the canonical catalog without allocating native tensors or operators.
pub fn parameter_description(
    args: &FamilyConfig,
) -> Result<ArchitectureParameterDescription, Error> {
    args.validate().map_err(Error::backend_source)?;
    component_state(args, None)?;
    declare_parameters(args)
}

fn component_units(
    parameters: &ArchitectureParameterDescription,
) -> Vec<eredu_runtime::ComponentUnit> {
    (0..parameters.unit_layout().len())
        .map(|ordinal| {
            let address = parameters
                .unit_layout()
                .address(ordinal)
                .expect("declared Gemma unit");
            let index = address.index();
            let (root, state) = match address.group() {
                0 => ("model.vision_tower.encoder.layers", 0..0),
                1 => ("model.audio_tower.layers", 0..0),
                2 => ("model.language_model.layers", index..index + 1),
                _ => unreachable!("validated Gemma group"),
            };
            eredu_runtime::ComponentUnit {
                path: format!("{root}.{index}"),
                state,
            }
        })
        .collect()
}

fn invalid(definition: &eredu_runtime::ComponentDefinition, resource: impl Into<String>) -> Error {
    eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    }
    .into()
}
fn validate_identity(definition: &eredu_runtime::ComponentDefinition) -> Result<(), Error> {
    if definition.id() != "gemma4" {
        return Err(invalid(definition, "Gemma component identity"));
    }
    Ok(())
}

/// Selects exact physical formats while retaining logical consumers and observation seams.
pub fn project_parameters(
    args: &FamilyConfig,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    validate_identity(definition)?;
    let parameters = parameter_description(args)?;
    definition
        .project_parameters(parameters)
        .map_err(Error::from)
}

fn validate_definition(
    args: &FamilyConfig,
    definition: &eredu_runtime::ComponentDefinition,
    allowed: &[&eredu_runtime::StateLayout],
) -> Result<(), Error> {
    validate_identity(definition)?;
    definition.validate_parameter_topology(&declare_parameters(args)?)?;
    if !allowed.contains(&definition.state_layout()) {
        return Err(invalid(
            definition,
            "Gemma retained state differs from selected geometry",
        ));
    }
    definition.validate_execution_contract(
        TEXT_EXECUTION_GROUP,
        &[],
        &component_units(definition.parameters()),
        &(0..3)
            .map(|group| group_transport(args, group))
            .collect::<Vec<_>>(),
        &state_partition_plan(definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    Ok(())
}

fn validate_placement(
    args: &FamilyConfig,
    definition: &eredu_runtime::ComponentDefinition,
    geometry: Option<&LocalGeometry>,
) -> Result<(), Error> {
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained =
            crate::gemma4::local_geometry(args, layout).map_err(Error::backend_source)?;
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: args.text.vocab_size as usize,
            local: 0..args.text.vocab_size as usize,
        };
        let per_layer = 0..args.text.hidden_size_per_layer_input;
        let output = if args.text.tie_word_embeddings {
            None
        } else {
            Some(
                geometry
                    .and_then(LocalGeometry::output_range)
                    .unwrap_or(&full),
            )
        };
        if retained.state_layout() != &component_state(args, geometry)?
            || retained.embedding_range() != geometry.map_or(&full, LocalGeometry::embedding_range)
            || retained.output_range() != output
            || retained.per_layer_range()
                != geometry.map_or(&per_layer, LocalGeometry::per_layer_range)
            || retained
                .text_blocks()
                .iter()
                .enumerate()
                .any(|(index, block)| {
                    block.architecture_fingerprint()
                        != geometry
                            .and_then(|geometry| geometry.text_block(index))
                            .unwrap_or(&args.text)
                            .architecture_fingerprint()
                })
        {
            return Err(invalid(
                definition,
                "Gemma retained placement differs from selected geometry",
            ));
        }
    }
    Ok(())
}

fn project_state(
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    state: eredu_runtime::StateLayout,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    definition
        .select_state_layout(&state)
        .map_err(Error::from)
}

pub(super) fn select_definition(
    args: &FamilyConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: Option<&LocalGeometry>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    args.validate().map_err(Error::backend_source)?;
    let state = component_state(args, geometry)?;
    let global = component_state(args, None)?;
    validate_definition(args, &definition, &[&global, &state])?;
    validate_placement(args, &definition, geometry)?;
    project_state(definition, state)
}

pub(super) fn select_partition_definition(
    args: &FamilyConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    local: &LocalGeometry,
    geometry: &crate::gemma4::PartitionLocalGeometry,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    let state = partition_state(args, local, geometry)?;
    let global = component_state(args, None)?;
    validate_definition(args, &definition, &[&global, local.state_layout(), &state])?;
    validate_placement(args, &definition, Some(local))?;
    if definition.state_layout() == &state {
        return Ok(definition);
    }
    let receivers = crate::gemma4::pipeline::receiver_caches(&args.text, geometry.text_units())
        .into_iter()
        .map(|(receiver, _)| eredu_runtime::StateSlotId(receiver));
    definition
        .as_ref()
        .clone()
        .with_materialized_state_slots(state, receivers)
        .map(std::sync::Arc::new)
        .map_err(Error::from)
}
