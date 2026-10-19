//! Exact source projection from RNNT bindings into one recurrent pipeline stage.
use super::{
    rnnt::{RnntDimensions, RnntError},
    rnnt_predictor::rnnt_predictor_parameter_description,
    rnnt_predictor_partition::{validate_rnnt_predictor_partition, RnntPredictorBoundarySchema},
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
};
use eredu_checkpoint::store::{RestrictedCheckpointSource, SharedCheckpointSource};
use eredu_runtime::{
    ArchitecturePartition, ExecutionUnitAddress, ExecutionUnitLayout, LocalModelLayout,
    WeightBinding,
};
use std::{collections::BTreeSet, sync::Arc};
/// Retained canonical LSTM recipes and explicit optional replica placement.
pub struct PreparedRnntPredictorBindings {
    dimensions: RnntDimensions,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
    local_layout: Option<LocalModelLayout>,
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
        let (_, _, full_layout, source, full_units) = bindings.into_parts();
        let mut units = Vec::new();
        for (ordinal, bindings) in full_units.into_iter().enumerate() {
            for index in range.clone() {
                if full_layout.ordinal(1, index) == Some(ordinal) {
                    units.push((
                        description
                            .unit_layout()
                            .address(index)
                            .expect("validated predictor unit"),
                        bindings,
                    ));
                    break;
                }
            }
        }
        if units.len() != range.len() {
            return Err(RnntError::InvalidState(
                "RNNT predictor retained source omits selected layers",
            )
            .into());
        }
        let inputs = units
            .iter()
            .flat_map(|(_, bindings)| bindings)
            .flat_map(|binding| {
                binding
                    .source_recipe()
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeSet<_>>();
        let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
            source,
            "voicechat/rnnt/predictor-stage",
            inputs,
        )?);
        Ok(Self {
            dimensions,
            layout: description.unit_layout().clone(),
            source,
            units,
            local_layout: None,
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
        if self
            .local_layout
            .as_ref()
            .is_some_and(|selected| selected != &layout)
        {
            return Err(RnntPreparationError::Parallel(
                "predictor replica placement differs from retained selection".into(),
            ));
        }
        self.local_layout = Some(layout.clone());
        Ok((self, layout))
    }
    /// Exact completed prediction tensor payload, including the separately retained
    /// final readout on non-final stages in addition to phase-local recurrent carry.
    pub fn prediction_state_bytes(&self) -> Result<u64, RnntPreparationError> {
        let range = self
            .units
            .first()
            .expect("validated nonempty stage")
            .0
            .index()
            ..self
                .units
                .last()
                .expect("validated nonempty stage")
                .0
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
        self.local_layout.as_ref()
    }
    /// Exact canonical stage units and their complete physical parameter bindings.
    pub fn unit_bindings(&self) -> &[(ExecutionUnitAddress, Vec<WeightBinding>)] {
        &self.units
    }
    /// Restricted retained sources; no non-predictor or neighboring-stage tensor is visible.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
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
        (self.dimensions, self.layout, self.source, self.units)
    }
}
