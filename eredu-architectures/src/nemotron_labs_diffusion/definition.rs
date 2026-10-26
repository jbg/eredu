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
const COMPOSITE_COMPONENT_ID: &str = "nemotron_labs_diffusion_composite";

/// Retains text semantics while selecting the family's exact executable parameter formats.
pub fn project_text_parameters(
    config: &FamilyConfig,
    definition: &std::sync::Arc<ComponentDefinition>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    validate_text_identity(definition)?;
    let parameters =
        super::parallel::parameter_description(config).map_err(Error::backend_source)?;
    definition
        .project_parameters(parameters)
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

pub(crate) fn text_from_composite(
    definition: &ComponentDefinition,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    let graph = eredu_runtime::ExecutionGraph::chain([decoder::TEXT_DECODER_EXECUTION_GROUP])
        .map_err(Error::backend_source)?;
    definition
        .project_groups(TEXT_COMPONENT_ID, graph, ["embedding", "norm", "output"])
        .map(std::sync::Arc::new)
        .map_err(Error::from)
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
            component: COMPOSITE_COMPONENT_ID.into(),
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
        COMPOSITE_COMPONENT_ID
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
    validate_state(config, &state, id)?;
    let layout = parameters.unit_layout();
    let units = (0..layout.len())
        .map(|ordinal| {
            unit(
                config,
                composite,
                layout.address(ordinal).expect("declared invocation"),
            )
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
        state_partition(config, composite),
    )
    .map_err(Error::from)
}

fn validate_state(config: &FamilyConfig, state: &StateLayout, id: &str) -> Result<(), Error> {
    let invalid = |resource: &str| {
        Error::from(ComponentDefinitionError::Invalid {
            component: id.into(),
            resource: resource.into(),
        })
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
    Ok(())
}

fn unit(
    config: &FamilyConfig,
    composite: bool,
    address: eredu_runtime::ExecutionUnitAddress,
) -> ComponentUnit {
    let index = address.index();
    if address.group() == if composite { 2 } else { 0 } {
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
}

fn state_partition(config: &FamilyConfig, composite: bool) -> ArchitectureStatePartitionPlan {
    ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
        if composite { 2 } else { 0 },
        0..config.num_hidden_layers() as usize,
    )])
}

/// Projects exact composite executable formats while retaining semantic identities.
pub fn project_composite_parameters(
    config: &FamilyConfig,
    definition: &std::sync::Arc<ComponentDefinition>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    validate_composite_identity(definition)?;
    let parameters =
        super::parallel::composite_parameter_description(config).map_err(Error::backend_source)?;
    definition
        .project_parameters(parameters)
        .map_err(Error::from)
}

fn validate_composite_identity(definition: &ComponentDefinition) -> Result<(), Error> {
    if definition.id() != COMPOSITE_COMPONENT_ID {
        return Err(ComponentDefinitionError::Invalid {
            component: definition.id().into(),
            resource: "shared composite identity".into(),
        }
        .into());
    }
    Ok(())
}

pub(crate) fn select_composite(
    config: &FamilyConfig,
    definition: std::sync::Arc<ComponentDefinition>,
    state: &StateLayout,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<std::sync::Arc<ComponentDefinition>, Error> {
    config.validate_config()?;
    validate_composite_identity(&definition)?;
    definition.validate_parameter_topology(
        &super::parallel::composite_parameter_description(config).map_err(Error::backend_source)?,
    )?;
    if definition
        .parameters()
        .partition_layout()
        .is_some_and(|(_, retained)| Some(retained) != layout)
    {
        return Err(ComponentDefinitionError::Invalid {
            component: definition.id().into(),
            resource: "retained placement differs from selected geometry".into(),
        }
        .into());
    }
    let units = (0..definition.parameters().unit_layout().len())
        .map(|ordinal| {
            unit(
                config,
                true,
                definition
                    .parameters()
                    .unit_layout()
                    .address(ordinal)
                    .expect("declared invocation"),
            )
        })
        .collect::<Vec<_>>();
    definition.validate_execution_contract(
        decoder::TEXT_DECODER_EXECUTION_GROUP,
        &[],
        &units,
        &(0..3)
            .map(super::parallel::group_transport)
            .collect::<Vec<_>>(),
        &state_partition(config, true),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    validate_state(config, definition.state_layout(), definition.id())?;
    validate_state(config, state, definition.id())?;
    if definition.state_layout() == state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state.clone())
            .map(std::sync::Arc::new)
            .map_err(Error::from)
    }
}
