//! Parameter ownership projected from the complete target construction declaration.
use super::*;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, MemberSharding,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterRole,
};

impl TargetSpec {
    /// Projects canonical storage from the validated target declaration.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        Ok(self.definition()?.parameters().clone())
    }

    /// Validates target invocations, storage, state and transport without native resources.
    pub fn definition(&self) -> Result<eredu_runtime::ComponentDefinition, Error> {
        self.definition_with_parameters(self.declare_parameters()?)
    }

    fn definition_with_parameters(
        &self,
        parameters: ArchitectureParameterDescription,
    ) -> Result<eredu_runtime::ComponentDefinition, Error> {
        eredu_runtime::ComponentDefinition::new(
            "qwen4_exp_target",
            parameters,
            self.units
                .iter()
                .enumerate()
                .map(|(index, unit)| eredu_runtime::ComponentUnit {
                    path: unit.path(),
                    state: index..index + 1,
                })
                .collect(),
            vec![crate::transport::decoder()],
            crate::decoder::TARGET_EXECUTION_GROUP,
            vec![],
            self.state_layout()?,
            eredu_runtime::ArchitectureStatePartitionPlan::new([
                eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..self.units.len()),
            ]),
        )
        .map_err(Error::from)
    }

    pub(super) fn declare_parameters(&self) -> Result<ArchitectureParameterDescription, Error> {
        self.limits.validate_for_config(&self.config)?;
        self.logical_parameters()?;
        let graph = ExecutionGraph::chain([crate::decoder::TARGET_EXECUTION_GROUP])
            .map_err(Error::backend_source)?;
        let layout =
            ExecutionUnitLayout::new(&graph, [self.units.len()]).map_err(Error::backend_source)?;
        let group = |name: &str, role, spec: &dyn DeclareLogicalParameters| {
            eredu_runtime::logical_parameter_group(name, role, None, spec, |_| {
                MemberSharding::Replicated
            })
            .map_err(Error::backend_source)
        };
        let mut groups = vec![OwnedParameterGroupSpec::new(
            if self.static_parameters.head.is_none() {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role("embedding")
            },
            group(
                "embedding",
                ParameterRole::Vocabulary,
                &self.static_parameters.embedding,
            )?,
        )];
        groups.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("norm"),
            group(
                "readout",
                ParameterRole::Replicated,
                &self.static_parameters.readout,
            )?,
        ));
        if let Some(head) = &self.static_parameters.head {
            groups.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("output"),
                group("output", ParameterRole::Vocabulary, head)?,
            ));
        }
        for (index, unit) in self.units.iter().enumerate() {
            if let UnitSpec::Decoder { feed_forward, .. } = unit {
                if feed_forward.feed_forward.layer != index {
                    return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                        component: "qwen4_exp_target".into(),
                        resource: format!("expert owner for {}", unit.path()),
                    }
                    .into());
                }
            }
            groups.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(0).expect("target group").clone(),
                    index,
                ),
                group(&unit.path(), ParameterRole::Replicated, unit)?,
            ));
        }
        ArchitectureParameterDescription::from_owned(&graph, &layout, groups)
            .map_err(Error::backend_source)
    }
}
