//! Prediction parameter ownership projected without backend allocation.
use super::{PredictionSpec, PREDICTION_EXECUTION_GROUP};
use eredu_nn::{DeclareLogicalParameters, Error};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, MemberSharding,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterRole,
};

impl PredictionSpec {
    /// Complete shared and depth-local storage, including retained encoding companions.
    /// Target embedding and vocabulary parameters remain with their target owner.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        self.logical_parameters()?;
        let graph =
            ExecutionGraph::chain([PREDICTION_EXECUTION_GROUP]).map_err(Error::backend_source)?;
        let layout =
            ExecutionUnitLayout::new(&graph, [self.units.len()]).map_err(Error::backend_source)?;
        let group = |name: &str, spec: &dyn DeclareLogicalParameters| {
            eredu_runtime::logical_parameter_group(
                name,
                ParameterRole::Replicated,
                None,
                spec,
                |_| MemberSharding::Replicated,
            )
            .map_err(Error::backend_source)
        };
        let mut groups = vec![OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("prediction_shared"),
            group("prediction.shared", &self.shared)?,
        )];
        for unit in &self.units {
            groups.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(0).expect("prediction group").clone(),
                    unit.depth,
                ),
                group(&format!("mtp.layers.{}", unit.depth), unit)?,
            ));
        }
        ArchitectureParameterDescription::from_owned(&graph, &layout, groups)
            .map_err(Error::backend_source)
    }
}
