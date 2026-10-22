//! Exact source projection from RNNT bindings into one recurrent pipeline stage.
use super::{
    rnnt::RnntDimensions,
    rnnt_predictor::rnnt_predictor_parameter_description,
    rnnt_predictor_partition::{validate_rnnt_predictor_partition, RnntPredictorBoundarySchema},
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{
    ArchitecturePartition, ExecutionUnitAddress, ExecutionUnitLayout, LocalModelLayout,
    PreparedComponent, WeightBinding,
};
/// Retained canonical LSTM recipes and explicit optional replica placement.
pub struct PreparedRnntPredictorBindings {
    dimensions: RnntDimensions,
    prepared: PreparedComponent,
}
impl PreparedRnntPredictorBindings {
    /// Selects only a stage's predictor units; embedding and joint sources are excluded.
    pub fn from_rnnt<G>(
        bindings: PreparedRnntBindings,
        partition: &ArchitecturePartition<G, RnntPredictorBoundarySchema>,
    ) -> Result<Self, RnntPreparationError> {
        let dimensions = bindings.dimensions();
        let range = validate_rnnt_predictor_partition(dimensions, partition)?;
        let description = rnnt_predictor_parameter_description(dimensions)?;
        let selected = range
            .map(|index| {
                description
                    .unit_layout()
                    .address(index)
                    .expect("validated predictor unit")
            })
            .collect::<Vec<_>>();
        let prepared = bindings
            .prepared
            .project("voicechat/rnnt/predictor-stage", description)?
            .select_units(selected)?;
        Ok(Self {
            dimensions,
            prepared,
        })
    }

    /// Selects explicit tensor replicas for a combined TP/PP execution cohort.
    /// Recurrent parameters retain their complete payload on every participant.
    pub fn select_parallel(
        mut self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RnntPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(RnntPreparationError::Parallel(
                "predictor replica selection requires a tensor-only topology".into(),
            ));
        }
        let layout = eredu_runtime::derive_partitioned_local_layout(
            &rnnt_predictor_parameter_description(self.dimensions)?,
            topology,
        )
        .map_err(RnntPreparationError::Parallel)?;
        self.prepared = self.prepared.select(&layout)?;
        Ok((self, layout))
    }
    /// Exact completed prediction tensor payload, including the separately retained
    /// final readout on non-final stages in addition to phase-local recurrent carry.
    pub fn prediction_state_bytes(&self) -> Result<u64, RnntPreparationError> {
        let units = self.prepared.unit_layout();
        let range = units.address(0).expect("validated nonempty stage").index()
            ..units
                .address(units.len() - 1)
                .expect("validated nonempty stage")
                .index()
                + 1;
        Ok(super::rnnt_predictor::rnnt_predictor_prediction_state_bytes(self.dimensions, range)?)
    }
    /// Complete RNNT geometry describing these recurrent layers.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Exact optional full-replica declaration used by cold pipeline validation.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }
    /// Exact canonical stage units and their complete physical parameter bindings.
    pub fn unit_bindings(
        &self,
    ) -> impl ExactSizeIterator<Item = (ExecutionUnitAddress, &[WeightBinding])> {
        self.prepared.addressed_unit_bindings()
    }
    /// Restricted retained sources; no non-predictor or neighboring-stage tensor is visible.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Transfers prepared recipes in the same globally addressed order as execution.
    pub fn into_parts(
        self,
    ) -> (
        RnntDimensions,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
    ) {
        let (_, layout, source, units) = self
            .prepared
            .into_addressed_unit_parts()
            .expect("predictor declares only independently leased units");
        (self.dimensions, layout, source, units)
    }
}
