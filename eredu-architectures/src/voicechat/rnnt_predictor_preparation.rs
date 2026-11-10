//! Exact source projection from RNNT bindings into one recurrent pipeline stage.
use super::{
    rnnt::RnntDimensions,
    rnnt_predictor_partition::{validate_rnnt_predictor_partition, RnntPredictorBoundarySchema},
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    rnnt_specs::RnntPredictorDeclaration,
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{
    ArchitecturePartition, ExecutionUnitAddress, LocalModelLayout, PreparedComponent, WeightBinding,
};
/// Retained canonical LSTM recipes and explicit optional replica placement.
pub struct PreparedRnntPredictorBindings {
    prepared: PreparedComponent<RnntPredictorDeclaration>,
}
impl PreparedRnntPredictorBindings {
    /// Selects only a stage's predictor units; embedding and joint sources are excluded.
    pub fn from_rnnt<G>(
        bindings: PreparedRnntBindings,
        partition: &ArchitecturePartition<G, RnntPredictorBoundarySchema>,
    ) -> Result<Self, RnntPreparationError> {
        let declaration = bindings.declaration().predictor()?;
        validate_rnnt_predictor_partition(&declaration, partition)?;
        let prepared = bindings
            .prepared
            .project("voicechat/rnnt/predictor-stage", declaration)?
            .select_units(partition.units())?;
        Ok(Self { prepared })
    }

    /// Selects explicit tensor-cohort replicas for a recurrent pipeline stage.
    /// Pipeline rank selection remains separate. Recurrent parameters retain their
    /// complete payload on every tensor participant.
    pub fn select_parallel(
        mut self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RnntPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
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
        Ok(super::rnnt_predictor::rnnt_predictor_prediction_state_bytes(self.dimensions(), range)?)
    }
    /// Global recurrent scope retained alongside the selected stage's exact sources.
    pub fn declaration(&self) -> &RnntPredictorDeclaration {
        self.prepared.declaration()
    }
    /// Complete RNNT geometry describing these recurrent layers.
    pub fn dimensions(&self) -> RnntDimensions {
        self.prepared.declaration().dimensions()
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
    /// Complete predictor definition, including global recurrent carry ownership.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }
    /// Transfers prepared recipes in the same globally addressed order as execution.
    pub fn into_parts(
        self,
    ) -> (
        RnntPredictorDeclaration,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
    ) {
        let (declaration, _, layout, source, units) = self
            .prepared
            .into_addressed_unit_parts()
            .expect("predictor declares only independently leased units");
        (declaration, layout, source, units)
    }
}
