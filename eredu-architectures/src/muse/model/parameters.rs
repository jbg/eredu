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
