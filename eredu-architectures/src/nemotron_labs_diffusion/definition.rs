//! Source-free declarations for shared invocation storage and independent context state.
use super::FamilyConfig;
use crate::decoder::{self, Config};
use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentDefinition, ComponentDefinitionError, ComponentUnit,
    StateLayout,
};

const TEXT_COMPONENT_ID: &str = "nemotron_labs_diffusion_text";

/// Retains text semantics while selecting the family's exact executable parameter formats.
pub fn project_text_parameters(
    config: &FamilyConfig,
    definition: &std::sync::Arc<ComponentDefinition>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    validate_text_identity(definition)?;
    let parameters =
        super::parallel::parameter_description(config).map_err(Error::backend_source)?;
    if definition.parameters() == &parameters {
        return Ok(definition.clone());
    }
    let uses = definition.parameter_uses().clone();
    definition
        .as_ref()
        .clone()
        .with_parameters(parameters, uses)
        .map(std::sync::Arc::new)
        .map_err(Error::from)
}

pub(crate) fn validate_text_identity(definition: &ComponentDefinition) -> Result<(), Error> {
    if definition.id() != TEXT_COMPONENT_ID {
        return Err(ComponentDefinitionError::Invalid {
            component: definition.id().into(),
            resource: "shared-backbone text component identity".into(),
        }
        .into());
    }
    Ok(())
}

/// Declares a complete shared text stack, optionally at retained tensor-local geometry.
pub fn text(
    config: &FamilyConfig,
    geometry: Option<&decoder::LocalGeometry<FamilyConfig>>,
) -> Result<ComponentDefinition, Error> {
    let state = complete_state(config, geometry)?;
    build(config, false, None, state)
}

/// Declares the same text resources with exact retained placement and global context ordinals.
pub fn text_partition(
    config: &FamilyConfig,
    parameters: &ArchitectureParameterDescription,
    geometry: &decoder::PartitionLocalGeometry<FamilyConfig>,
) -> Result<ComponentDefinition, Error> {
    build(
        config,
        false,
        Some(parameters),
        geometry.complete_state_layout().clone(),
    )
}

/// Declares Pixtral ingress, independently leased projector and the same shared text storage.
pub fn composite(
    config: &FamilyConfig,
    geometry: Option<&decoder::LocalGeometry<FamilyConfig>>,
) -> Result<ComponentDefinition, Error> {
    let state = complete_state(config, geometry)?;
    build(config, true, None, state)
}

/// Retains selected composite placement without reconstructing image or text modules.
pub fn composite_partition(
    config: &FamilyConfig,
    parameters: &ArchitectureParameterDescription,
    geometry: &super::composite::partition::Geometry,
) -> Result<ComponentDefinition, Error> {
    if parameters
        .partition_layout()
        .is_some_and(|(_, layout)| layout != geometry.local_layout())
    {
        return Err(ComponentDefinitionError::Invalid {
            component: "nemotron_labs_diffusion_composite".into(),
            resource: "retained placement differs from selected geometry".into(),
        }
        .into());
    }
    build(
        config,
        true,
        Some(parameters),
        geometry.complete_state().clone(),
    )
}

fn complete_state(
    config: &FamilyConfig,
    geometry: Option<&decoder::LocalGeometry<FamilyConfig>>,
) -> Result<StateLayout, Error> {
    config.validate_config()?;
    match geometry {
        Some(geometry) => {
            geometry
                .validate_for(config)
                .map_err(Error::backend_source)?;
            Ok(geometry.state_layout().clone())
        }
        None => decoder::state_layout(config),
    }
}

fn build(
    config: &FamilyConfig,
    composite: bool,
    retained: Option<&ArchitectureParameterDescription>,
    state: StateLayout,
) -> Result<ComponentDefinition, Error> {
    config.validate_config()?;
    let id = if composite {
        "nemotron_labs_diffusion_composite"
    } else {
        TEXT_COMPONENT_ID
    };
    let invalid = |resource: &str| {
        Error::from(ComponentDefinitionError::Invalid {
            component: id.into(),
            resource: resource.into(),
        })
    };
    let canonical = if composite {
        super::parallel::composite_parameter_description(config)
    } else {
        super::parallel::parameter_description(config)
    }
    .map_err(Error::backend_source)?;
    let parameters = match retained {
        Some(parameters) => {
            if parameters.graph() != canonical.graph()
                || parameters.unit_layout() != canonical.unit_layout()
                || parameters.groups() != canonical.groups()
            {
                return Err(invalid("canonical parameter topology"));
            }
            parameters.clone()
        }
        None => canonical,
    };
    let layers = config.num_hidden_layers() as usize;
    if state.len() != layers {
        return Err(invalid("complete context slot count"));
    }
    let expected = decoder::state_layout(config)?;
    if state.segments() != expected.segments() {
        return Err(invalid("context segment lifetime or ownership"));
    }
    for (index, (local, global)) in state
        .layers()
        .iter()
        .zip(expected.layers().iter())
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
            return Err(invalid(&format!("context slot {index} geometry")));
        }
    }
    let text_group = if composite { 2 } else { 0 };
    let layout = parameters.unit_layout();
    let units = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("declared invocation");
            let index = address.index();
            if address.group() == text_group {
                ComponentUnit {
                    path: format!("{}.layers.{index}", config.parameter_root()),
                    state: index..index + 1,
                }
            } else if address.group() == 0 {
                ComponentUnit {
                    path: format!("model.vision_tower.transformer.layers.{index}"),
                    state: 0..0,
                }
            } else {
                ComponentUnit {
                    path: "model.multi_modal_projector".into(),
                    state: 0..0,
                }
            }
        })
        .collect();
    let transport = if composite {
        (0..3).map(super::parallel::group_transport).collect()
    } else {
        vec![crate::transport::decoder()]
    };
    ComponentDefinition::new(
        id,
        parameters,
        units,
        transport,
        decoder::TEXT_DECODER_EXECUTION_GROUP,
        vec![],
        state,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
            text_group,
            0..layers,
        )]),
    )
    .map_err(Error::from)
}
