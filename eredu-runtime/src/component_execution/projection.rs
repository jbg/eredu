//! Identity-based child declarations with caller-owned ingress boundaries.
use super::*;
use crate::{ArchitectureStatePartitionRule, ArchitectureStatePlacement};

impl ComponentDefinition {
    /// Projects complete named groups and pinned resources into a child component.
    /// The child graph explicitly cuts omitted dependencies at caller-owned ingress.
    /// Included dependencies, canonical storage owners and all state-slot identities
    /// remain unchanged. Every mutable-state owner must remain in the child.
    pub fn project_groups(
        &self,
        id: impl Into<String>,
        graph: ExecutionGraph,
        pinned_roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ComponentDefinitionError> {
        let id = id.into();
        let invalid = |resource: String| ComponentDefinitionError::Invalid {
            component: id.clone(),
            resource,
        };
        let parameters = self
            .parameters
            .project_groups(&graph, pinned_roles)
            .map_err(|source| ComponentDefinitionError::Parameters {
                component: id.clone(),
                source,
            })?;
        let layout = parameters.unit_layout();
        let parent_groups = graph
            .groups()
            .iter()
            .map(|group| {
                self.parameters
                    .graph()
                    .group_index(group.id())
                    .expect("projected group")
            })
            .collect::<Vec<_>>();
        let remap = |address: ExecutionUnitAddress| {
            let group = self.parameters.unit_layout().group_id(address.group())?;
            let group = graph.group_index(group.as_str())?;
            layout
                .ordinal(group, address.index())
                .and_then(|ordinal| layout.address(ordinal))
        };
        let mut rules = Vec::new();
        for rule in self.state_partition.rules() {
            rules.push(match rule.placement() {
                ArchitectureStatePlacement::GroupUnits { group } => {
                    let name = self.parameters.graph().groups()[group].id();
                    let selected = graph
                        .group_index(name)
                        .ok_or_else(|| invalid(format!("excluded state owner {name:?}")))?;
                    ArchitectureStatePartitionRule::group_units(selected, rule.layers())
                }
                ArchitectureStatePlacement::Invocation { address } => {
                    let selected = remap(address)
                        .ok_or_else(|| invalid(format!("excluded state invocation {address:?}")))?;
                    ArchitectureStatePartitionRule::invocation(selected, rule.layers())
                }
                ArchitectureStatePlacement::OutputOwner => {
                    if graph.groups()[graph.output()].id()
                        != self.parameters.graph().groups()[self.parameters.graph().output()].id()
                    {
                        return Err(invalid("projected output-owned state changes owner".into()));
                    }
                    ArchitectureStatePartitionRule::output_owner(rule.layers())
                }
            });
        }
        let units = parent_groups
            .iter()
            .flat_map(|&group| {
                self.parameters
                    .unit_layout()
                    .group_range(group)
                    .expect("declared group")
                    .map(|ordinal| self.units[ordinal].clone())
            })
            .collect();
        let observations = parent_groups
            .iter()
            .flat_map(|&group| {
                self.parameters
                    .unit_layout()
                    .group_range(group)
                    .expect("declared group")
                    .map(|ordinal| self.unit_observations[ordinal].clone())
            })
            .collect();
        let transport = parent_groups
            .iter()
            .map(|&group| self.transport[group].clone())
            .collect();
        let primary = graph.groups()[graph.output()].id().to_owned();
        let prediction = self
            .prediction
            .iter()
            .filter(|group| graph.group_index(group).is_some())
            .cloned()
            .collect();
        let uses = self
            .parameter_uses
            .iter()
            .filter_map(|(path, parameter)| {
                remap(parameter.invocation).map(|invocation| {
                    (
                        path.clone(),
                        ComponentParameterUse {
                            invocation,
                            target: parameter.target.clone(),
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut selected = Self::new(
            id,
            parameters,
            units,
            transport,
            primary,
            prediction,
            self.state.clone(),
            ArchitectureStatePartitionPlan::new(rules).with_packing(self.state_partition.packing()),
        )?
        .with_parameter_uses(uses)?;
        selected.unit_observations = observations;
        Ok(selected)
    }
}
