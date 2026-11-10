//! Explicit replicated scoring-head ownership alongside a sharded DSpark body.
use super::DsparkHeadDefinition;
use eredu_nn::Error;
use eredu_runtime::{
    logical_parameter_group, ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupId,
    MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterRole,
};

impl DsparkHeadDefinition {
    /// Declares complete scoring replicas at their independent acquisition units.
    /// Exact I64 token mapping remains separate from floating-point projections.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        let graph = ExecutionGraph::chain(self.units.iter().map(|unit| unit.group))
            .map_err(Error::backend_source)?;
        let mut owned = Vec::new();
        for unit in &self.units {
            let group = logical_parameter_group(
                unit.group,
                ParameterRole::Replicated,
                None,
                &unit.module,
                |_| MemberSharding::Replicated,
            )
            .map_err(Error::backend_source)?;
            let owner = ParameterGroupOwner::execution_unit(
                ExecutionGroupId::new(unit.group).map_err(Error::backend_source)?,
                0,
            );
            owned.push(OwnedParameterGroupSpec::new(owner, group));
        }
        ArchitectureParameterDescription::from_owned(&graph, &self.layout, owned)
            .map_err(Error::backend_source)
    }
}
