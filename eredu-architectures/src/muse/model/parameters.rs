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
    let text_static = static_parameter_groups(args).map_err(Error::backend)?;
    let vision_static = vision_static_parameter_groups(args).map_err(Error::backend)?;
    let mut expected = text_static
        .iter()
        .chain(&vision_static)
        .cloned()
        .collect::<Vec<_>>();
    let mut owned = text_static
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            OwnedParameterGroupSpec::new(
                if index == 0 && args.tie_word_embeddings {
                    ParameterGroupOwner::static_any_of(["embedding", "output"])
                } else {
                    ParameterGroupOwner::static_role(match index {
                        0 => "embedding",
                        1 => "norm",
                        _ => "output",
                    })
                },
                group,
            )
        })
        .chain(vision_static.into_iter().map(|group| {
            let consumers = crate::muse::parallel::vision_static_consumer_units(
                args,
                group.members()[0].target(),
            )
            .expect("declared Muse vision static group has consumers");
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
    let mut transport = vec![vision_group_transport(args), crate::transport::decoder()];
    if audio.is_some() {
        transport.push(audio_group_transport());
    }
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
    .map_err(Error::from)
}
