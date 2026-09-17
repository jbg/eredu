//! Owning parameter declarations from the existing composite traversal.
use super::graph::Destination;
use eredu_nn::Error;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupId, ExecutionUnitLayout,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec,
};

impl Destination<'_> {
    pub(crate) fn reserve<T>(self, values: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        match self.0 {
            Some(context) => context.reserve_metadata_vec(values, additional),
            None => {
                values.reserve(additional);
                Ok(())
            }
        }
    }

    pub(crate) fn group_id(self, name: &str) -> Result<ExecutionGroupId, Error> {
        self.controls::<ExecutionGroupId>()?;
        ExecutionGroupId::new(self.text(format_args!("{name}"))?)
            .map_err(|cause| self.error(format_args!("{cause}")))
    }

    pub(crate) fn static_owner(self, roles: &[&str]) -> Result<ParameterGroupOwner, Error> {
        self.controls::<ParameterGroupOwner>()?;
        if let [role] = roles {
            return Ok(ParameterGroupOwner::static_role(
                self.text(format_args!("{role}"))?,
            ));
        }
        let mut owned = self.vector(roles.len())?;
        for role in roles {
            owned.push(self.text(format_args!("{role}"))?);
        }
        Ok(ParameterGroupOwner::StaticAnyOf(owned))
    }

    pub(crate) fn append_owned(
        self,
        destination: &mut Vec<OwnedParameterGroupSpec>,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<(), Error> {
        self.controls::<Vec<OwnedParameterGroupSpec>>()?;
        self.reserve(destination, groups.len())?;
        destination.extend(groups);
        Ok(())
    }

    pub(crate) fn append_unit(
        self,
        destination: &mut Vec<OwnedParameterGroupSpec>,
        groups: Vec<ParameterGroupSpec>,
        group: &ExecutionGroupId,
        index: usize,
    ) -> Result<(), Error> {
        self.controls::<(OwnedParameterGroupSpec, Vec<ParameterGroupSpec>)>()?;
        self.reserve(destination, groups.len())?;
        for parameter_group in groups {
            let owner = ParameterGroupOwner::execution_unit(self.group_id(group.as_str())?, index);
            destination.push(OwnedParameterGroupSpec::new(owner, parameter_group));
        }
        Ok(())
    }

    pub(crate) fn description(
        self,
        graph: ExecutionGraph,
        layout: ExecutionUnitLayout,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<ArchitectureParameterDescription, Error> {
        match self.0 {
            Some(context) => ArchitectureParameterDescription::from_owned_with_metadata(
                graph, layout, groups, context,
            ),
            None => ArchitectureParameterDescription::from_owned(graph, layout, groups)
                .map_err(Error::backend),
        }
    }
}
