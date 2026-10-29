//! Unit-relative logical uses over retained canonical storage ownership.
use super::*;

impl ComponentDefinition {
    /// Declares logical uses for execution-unit storage whose targets are namespaced
    /// by the canonical owner's unit path. Every consumer receives the same suffix
    /// under its own declared path, including physical encoding companions.
    /// Other naming schemes supply explicit uses through `with_parameter_uses`.
    pub fn with_unit_parameter_paths(self) -> Result<Self, ComponentDefinitionError> {
        let mut uses = Vec::new();
        for owned in self.parameters.groups() {
            let crate::ParameterGroupOwner::ExecutionUnit {
                group,
                global_unit,
                consumers,
                ..
            } = owned.owner()
            else {
                continue;
            };
            let group_index = self
                .parameters
                .graph()
                .group_index(group.as_str())
                .expect("validated parameter group");
            let canonical = &self.unit(group_index, *global_unit)?.path;
            for &consumer in consumers {
                let prefix = &self.unit(group_index, consumer)?.path;
                let invocation = self.unit_address(group_index, consumer)?;
                for member in owned.members() {
                    let suffix = member
                        .target()
                        .strip_prefix(canonical)
                        .filter(|suffix| suffix.starts_with('.'))
                        .ok_or_else(|| ComponentDefinitionError::Invalid {
                            component: self.id.clone(),
                            resource: format!(
                                "parameter {} outside canonical unit path {canonical}",
                                member.target()
                            ),
                        })?;
                    uses.push((
                        format!("{prefix}{suffix}"),
                        ComponentParameterUse {
                            invocation,
                            target: member.target().into(),
                        },
                    ));
                }
            }
        }
        self.with_parameter_uses(uses)
    }

    /// Projects exact physical resources and rebuilds unit-relative logical uses
    /// from retained invocation paths. Ownership and companion validation precede
    /// alias projection; identical catalogs reuse the retained definition.
    pub fn project_parameters_with_unit_paths(
        self: &std::sync::Arc<Self>,
        parameters: ArchitectureParameterDescription,
    ) -> Result<std::sync::Arc<Self>, ComponentDefinitionError> {
        if self.parameters == parameters {
            return Ok(self.clone());
        }
        self.as_ref()
            .clone()
            .with_parameters(parameters, [])?
            .with_unit_parameter_paths()
            .map(std::sync::Arc::new)
    }
}
