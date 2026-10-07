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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voicechat::{
        artifact::tests::official_parameters, checkpoint, config::VoiceChatConfig,
        rnnt_predictor::rnnt_predictor_state_layout,
    };
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        WeightStoreDiagnostics,
    };
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    use eredu_runtime::{
        ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, PartitionOwnership,
        TensorPlacement,
    };
    use std::collections::BTreeMap;
    struct Headers(BTreeMap<String, TensorMetadata>);
    impl CheckpointSource for Headers {
        fn source_keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("cold recurrent stage selection must not acquire weights")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal("metadata-only fixture".into()))
        }
    }
    #[test]
    fn released_predictor_stage_retains_only_owned_lstm_sources_and_complete_replica_bytes() {
        let config =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let dimensions = RnntDimensions::from_config(&config.rnnt).unwrap();
        let description = rnnt_predictor_parameter_description(dimensions).unwrap();
        let state = rnnt_predictor_state_layout(dimensions).unwrap();
        let parameters = official_parameters();
        let source: SharedCheckpointSource =
            Arc::new(Headers(checkpoint::tests::official().tensors().clone()));
        for stage in 0..2 {
            let partition = ArchitecturePartition::from_description(
                &description,
                [(description.graph().groups()[0].id(), stage..stage + 1)],
                PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>())
                    .unwrap(),
                &state,
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..2),
                ]),
                (),
                RnntPredictorBoundarySchema::new(dimensions).unwrap(),
            )
            .unwrap();
            let rnnt =
                PreparedRnntBindings::from_parameters(dimensions, source.clone(), &parameters)
                    .unwrap();
            let prepared = PreparedRnntPredictorBindings::from_rnnt(rnnt, &partition).unwrap();
            assert_eq!(prepared.unit_bindings().len(), 1);
            assert_eq!(prepared.unit_bindings()[0].0.index(), stage);
            assert_eq!(prepared.unit_bindings()[0].1.len(), 4);
            assert_eq!(prepared.source().source_keys().len(), 4);
            let retained = prepared.source().clone();
            let topology =
                ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1).unwrap();
            let (prepared, layout) = prepared.select_parallel(topology).unwrap();
            assert!(Arc::ptr_eq(&retained, prepared.source()));
            let mut bytes = 0;
            for binding in &prepared.unit_bindings()[0].1 {
                assert!(binding.name().ends_with(&format!("_l{stage}")));
                assert!(matches!(
                    layout.tensor(binding.name()).unwrap().placement(),
                    TensorPlacement::Replicated
                ));
                bytes += binding.expected_bytes();
            }
            assert_eq!(
                bytes, 13_127_680,
                "each tensor replica retains complete recurrent weights"
            );
            assert_eq!(
                prepared.prediction_state_bytes().unwrap(),
                if stage == 0 { 3 * 640 * 4 } else { 2 * 640 * 4 }
            );
            for denied in [
                "stt_model.rnnt_joint.enc.weight".to_owned(),
                format!(
                    "stt_model.rnnt_decoder.prediction.dec_rnn.lstm.weight_ih_l{}",
                    1 - stage
                ),
            ] {
                assert!(matches!(
                    prepared.source().source_metadata(&denied),
                    Err(StoreError::UnauthorizedTensor { .. })
                ));
            }
            let (prepared, again) = prepared.select_parallel(topology).unwrap();
            assert_eq!(layout, again);
            assert!(Arc::ptr_eq(&retained, prepared.source()));
        }
    }
}
