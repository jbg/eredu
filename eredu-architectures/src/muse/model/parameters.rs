//! Shared source-free Muse parameter declarations for cold and live construction.
use super::*;

pub(crate) fn describe(args: &DecoderConfig) -> Result<ArchitectureParameterDescription, Error> {
    let graph = crate::muse::graph::execution_graph(args)?;
    let mut counts = vec![
        args.vision_config
            .as_ref()
            .map_or(0, |vision| vision.layer_count()),
        args.num_hidden_layers as usize,
    ];
    if let Some(audio) = &args.audio_config {
        counts.push(audio.execution_units()?.len());
    }
    let layout =
        ExecutionUnitLayout::new(&graph, counts.iter().copied()).map_err(Error::backend)?;
    let text_static = crate::muse::parallel::static_resource_parameter_groups(args)
        .map_err(Error::backend)?;
    let vision_static = crate::muse::parallel::vision_static_resource_groups(args)
        .map_err(Error::backend_source)?;
    let mut owned = text_static
        .into_iter()
        .map(|(role, group)| {
            OwnedParameterGroupSpec::new(
                if role == "embedding" && args.tie_word_embeddings {
                    ParameterGroupOwner::static_any_of(["embedding", "output"])
                } else {
                    ParameterGroupOwner::static_role(role)
                },
                group,
            )
        })
        .chain(vision_static.into_iter().map(|(resource, group)| {
            let consumers = resource.consumers(
                args.vision_config
                    .as_ref()
                    .expect("declared vision")
                    .layer_count(),
            );
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_unit_consumers(
                    "vision",
                    consumers
                        .map(|unit| (layout.group_id(0).expect("Muse vision group").clone(), unit)),
                ),
                group,
            )
        }))
        .collect::<Vec<_>>();
    for (group_index, &count) in counts.iter().enumerate() {
        let owner_group = layout
            .group_id(group_index)
            .expect("Muse layout group")
            .clone();
        for index in 0..count {
            let groups = match group_index {
                0 => vision_layer_parameter_groups(args, index).map_err(Error::backend)?,
                1 => layer_parameter_groups(args, index).map_err(Error::backend)?,
                2 => {
                    let audio = args.audio_config.as_ref().expect("declared audio group");
                    crate::muse::audio::execution::unit_parameter_groups(
                        audio,
                        audio.execution_units()?[index],
                    )?
                }
                _ => unreachable!(),
            };
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

/// Validates Muse's logical units, canonical storage and independent mutable state
/// before execution selection or native materialization.
pub fn component_definition(
    args: &DecoderConfig,
    geometry: Option<&LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    args.validate().map_err(Error::backend_source)?;
    let parameters = describe(args)?;
    let state = match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            geometry.state_layout().clone()
        }
        None => state_layout(args).map_err(Error::backend_source)?,
    };
    let units = component_units(args, &parameters)?;
    let transport = component_transport(args);
    let partition = state_partition_plan(args, &state);
    eredu_runtime::ComponentDefinition::new(
        args.identity.model_type(),
        parameters,
        units,
        transport,
        TEXT_EXECUTION_GROUP,
        Vec::new(),
        state,
        partition,
    )
    .and_then(|definition| {
        let text_group = definition
            .parameters()
            .graph()
            .group_index(definition.primary_group())
            .expect("validated primary group");
        definition.map_unit_observation_boundaries(|address, boundaries| {
            if address.group() == text_group {
                boundaries.clone().with_kernel_ownership()
            } else {
                boundaries.clone()
            }
        })
    })
    .map_err(Error::from)
}

fn component_units(
    args: &DecoderConfig,
    parameters: &ArchitectureParameterDescription,
) -> Result<Vec<eredu_runtime::ComponentUnit>, Error> {
    let audio = args
        .audio_config
        .as_ref()
        .map(|audio| audio.execution_units())
        .transpose()?;
    let layout = parameters.unit_layout();
    let units = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared Muse unit");
            let index = address.index();
            match address.group() {
                0 => eredu_runtime::ComponentUnit {
                    path: format!("model.vision_tower.layers.{index}"),
                    state: 0..0,
                },
                1 => eredu_runtime::ComponentUnit {
                    path: format!("model.layers.{index}"),
                    state: index..index + 1,
                },
                _ => {
                    let slot = args.num_hidden_layers as usize + index;
                    eredu_runtime::ComponentUnit {
                        path: audio.as_ref().expect("declared Muse audio group")[index].path(),
                        state: slot..slot + 1,
                    }
                }
            }
        })
        .collect();
    Ok(units)
}

fn component_transport(args: &DecoderConfig) -> Vec<eredu_runtime::ArchitectureGroupTransport> {
    let mut transport = vec![vision_group_transport(args), crate::transport::decoder()];
    if args.audio_config.is_some() {
        transport.push(audio_group_transport());
    }
    transport
}

/// Selects executable parameter formats while retaining logical consumers and seams.
pub fn project_parameters(
    args: &DecoderConfig,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    args.validate().map_err(Error::backend_source)?;
    validate_identity(args, definition)?;
    let parameters = describe(args)?;
    definition
        .project_parameters(parameters)
        .map_err(Error::from)
}

fn invalid(definition: &eredu_runtime::ComponentDefinition, resource: impl Into<String>) -> Error {
    eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    }
    .into()
}
fn validate_identity(
    args: &DecoderConfig,
    definition: &eredu_runtime::ComponentDefinition,
) -> Result<(), Error> {
    if definition.id() != args.identity.model_type() {
        return Err(invalid(definition, "Muse component identity"));
    }
    Ok(())
}

pub(crate) fn select_definition(
    args: &DecoderConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: Option<&LocalGeometry>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    args.validate().map_err(Error::backend_source)?;
    validate_identity(args, &definition)?;
    definition.validate_parameter_topology(&describe(args)?)?;
    let state = match geometry {
        Some(geometry) => {
            geometry.validate_for(args).map_err(Error::backend_source)?;
            geometry.state_layout().clone()
        }
        None => state_layout(args).map_err(Error::backend_source)?,
    };
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained = super::super::local_geometry(args, layout).map_err(Error::backend_source)?;
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: args.vocab_size as usize,
            local: 0..args.vocab_size as usize,
        };
        let embedding = geometry.map_or(&full, LocalGeometry::embedding_range);
        let output = if args.tie_word_embeddings {
            None
        } else {
            Some(
                geometry
                    .and_then(LocalGeometry::output_range)
                    .unwrap_or(&full),
            )
        };
        if retained.state_layout() != &state
            || retained.embedding_range() != embedding
            || retained.output_range() != output
            || retained
                .text_blocks()
                .iter()
                .enumerate()
                .any(|(index, block)| {
                    block.architecture_fingerprint()
                        != geometry
                            .and_then(|geometry| geometry.text_block(index))
                            .unwrap_or(args)
                            .architecture_fingerprint()
                })
        {
            return Err(invalid(
                &definition,
                "Muse retained placement differs from selected geometry",
            ));
        }
    }
    let global = state_layout(args).map_err(Error::backend_source)?;
    let retained = definition.state_layout();
    if retained.len() != global.len() || retained.segments() != global.segments() {
        return Err(invalid(&definition, "Muse complete state geometry"));
    }
    for (index, (local, global)) in retained
        .layers()
        .iter()
        .zip(global.layers().iter())
        .enumerate()
    {
        let compatible = match (local, global) {
            (
                eredu_core::cache::LayerCachePolicy::KeyValue {
                    attention,
                    num_key_value_heads,
                    head_dim,
                },
                eredu_core::cache::LayerCachePolicy::KeyValue {
                    attention: global_attention,
                    num_key_value_heads: global_heads,
                    head_dim: global_dim,
                },
            ) => {
                attention == global_attention
                    && head_dim == global_dim
                    && num_key_value_heads <= global_heads
            }
            _ => local == global,
        };
        if !compatible {
            return Err(invalid(
                &definition,
                format!("Muse state slot {index} geometry"),
            ));
        }
    }
    definition.validate_execution_contract(
        TEXT_EXECUTION_GROUP,
        &[],
        &component_units(args, definition.parameters())?,
        &component_transport(args),
        &state_partition_plan(args, &state),
        |address, unit, boundaries| {
            if address.group() == 1 {
                boundaries
                    == &eredu_runtime::UnitObservationBoundaries::around(&unit.path)
                        .with_kernel_ownership()
            } else {
                !boundaries.is_kernel_owned()
            }
        },
    )?;
    definition
        .select_state_layout(&state)
        .map_err(Error::from)
}
