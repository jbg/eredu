//! Cold selected resource coverage for the retained target and optional prediction.
use super::SelectedTargetExecution;
use eredu_core::resources::{ResourceCoverage, ResourceDescription, ResourceDescriptionError};
use eredu_runtime::execution_resources::{
    describe_prepared_resources, PreparedResourceQuery, PreparedStateResource,
};

/// Selected payload geometry and configured allowances, kept distinct from a
/// physical memory forecast. Unknown coverage cannot establish aggregate fit.
#[derive(Debug, Clone)]
pub struct TargetResourceReport {
    /// Ordinary/prediction state together, with explicit missing physical facts.
    pub resources: ResourceDescription,
    /// Resident stream payload plus all lane-local scratch/catalog allowances.
    /// Paged payload is excluded: its shared cache pool is selected separately.
    pub stream_allowances: eredu_runtime::AppendStreamAllowances,
    /// Actual selected row workspace and companion requirements, if present.
    pub rows: Option<eredu_runtime::SelectedRowLookupRequirements>,
    /// Selected maximum-acquisition decoder contracts, with explicit unknowns.
    /// Physical invocation/pool/lifetime binding uses the shared runtime producers.
    pub row_decoders: std::collections::BTreeMap<
        eredu_nn::ParameterId,
        eredu_nn::mechanism_memory::MechanismMemoryContract,
    >,
    /// The retained shared row cache and scratch policy, never a whole-table load.
    pub row_policy: Option<eredu_runtime::ParameterBankLoadOptions>,
}
impl SelectedTargetExecution {
    /// Reports recipe/conversion workspace from the retained selected tasks and
    /// source. Native temporary geometry is supplied by the backend's cold facts;
    /// table lookup buffers remain in the separate row resource contract.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        crate::SelectedExecution::routed(self.selected.clone()).parameter_materialization_workspace(
            self.plan.target.artifact.as_ref(),
            None,
            mechanisms,
        )
    }

    /// Describes the complete selected state pair without payload reads or native
    /// allocation. The query supplies a common hypothetical prefix/horizon envelope,
    /// not independently observed live predictor frontiers. Weights remain in their
    /// ordinary retained residency owner;
    /// logical embedding/output sharing cannot invent physical allocation credit.
    pub fn describe_prepared_resources(
        &self,
        query: &PreparedResourceQuery,
    ) -> Result<TargetResourceReport, ResourceDescriptionError> {
        if query.batch_size > self.plan.target.spec.limits.qsa.batch as u64
            || self
                .plan
                .prediction
                .as_ref()
                .is_some_and(|(prediction, _)| {
                    query.batch_size > prediction.spec().limits.qsa.batch as u64
                })
        {
            return Err(ResourceDescriptionError::Invalid(
                "resource query exceeds selected execution lanes".into(),
            ));
        }
        let mut states = vec![PreparedStateResource {
            owner: "target",
            state: self.selected.text().state(),
        }];
        if let Some(state) = &self.prediction_state {
            states.push(PreparedStateResource {
                owner: "prediction",
                state,
            });
        }
        let mut resources =
            describe_prepared_resources(self.selected.text(), &states, &[], &[], query)?;
        let ResourceCoverage::Partial { reasons } = &mut resources.coverage else {
            unreachable!("cold prepared resources retain missing backing facts")
        };
        for (id, bank) in self.selected.banks() {
            reasons.push(format!("selected expert bank {id:?} in {}: member allocations, shared cache backing and dispatch scratch are not described", bank.owner_group().as_str()));
        }
        let rows = self.selected.row_lookups();
        if rows.is_some_and(|r| !r.descriptors().entries().is_empty()) {
            reasons.push("selected row tables: encoded sources are addressable, not resident allocations; cache/scalar backing, transfer buffers and completion retention are not described".into());
        }
        if self.prediction_state.is_some() {
            reasons.push("prediction target-feature retention, fusion/residual workspace and transaction copies lack physical backing/lifetime facts".into());
        }
        resources.validate()?;
        Ok(TargetResourceReport {
            resources,
            stream_allowances: self.stream_allowances,
            rows: rows.map(|r| r.requirements()),
            row_policy: rows.map(|r| r.options()),
            row_decoders: rows.map(|r| r.decode_memory().clone()).unwrap_or_default(),
        })
    }
}
