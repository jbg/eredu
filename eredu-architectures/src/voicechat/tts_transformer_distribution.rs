//! Cold authority for a retained transformer component pipeline.
use super::{
    tts_transformer_layered::{
        tts_transformer_parameter_description, validate_transformer_partition,
        TransformerBoundarySchema, TransformerError,
    },
    tts_transformer_preparation::PreparedTtsTransformerPartitionBindings,
};
use eredu_core::CollectiveGroupId;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    PartitionExecutionError, PartitionedExecutionPlan, PipelineActivationDtype,
};

/// An inconsistent component selection is rejected before native resources exist.
#[derive(Debug, thiserror::Error)]
pub enum TransformerPipelinePreparationError {
    /// Family geometry or parameter/state authority differs.
    #[error(transparent)]
    Transformer(#[from] TransformerError),
    /// Generic communication selection is inconsistent.
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
    /// The tensor cohort cannot propagate local preparation or lease failures.
    #[error(transparent)]
    LocalAgreement(#[from] eredu_runtime::PipelineLocalAgreementError),
    /// Selected stage, transport, or completion authority differs.
    #[error("invalid TTS pipeline selection: {0}")]
    Contract(&'static str),
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::voicechat::{
        artifact::tests::official_parameters, checkpoint, config::VoiceChatConfig,
        tts_transformer::TransformerConfig,
        tts_transformer_preparation::PreparedTtsTransformerBindings,
    };
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, SharedCheckpointSource, StoreError, TensorMetadata,
        TensorReadRequest, WeightStoreDiagnostics,
    };
    use eredu_core::{checkpoint::TensorDtype, CompletionCancellationMode};
    use eredu_runtime::{
        ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule,
        CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationOperation, CommunicationOperationRequirement,
        CommunicationTensorLimits, LayeredPartitionDriver, PartitionOutputPublication,
        PartitionOwnership, PipelineWireContract,
    };
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    struct HeaderSource(BTreeMap<String, TensorMetadata>);
    impl CheckpointSource for HeaderSource {
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
            panic!("cold pipeline admission must not read weight payloads")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Err(StoreError::Internal(
                "header fixture has no payload storage".into(),
            ))
        }
    }

    fn admission(
        norm_only: bool,
        publication_dtype: TensorDtype,
    ) -> Result<PreparedTtsTransformerPipelineBindings, TransformerPipelinePreparationError> {
        let released =
            VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
                .unwrap();
        let config = TransformerConfig::new(&released.tts.backbone).unwrap();
        let description = tts_transformer_parameter_description(&config).unwrap();
        let state = crate::decoder::state_layout(&config).unwrap();
        let blocks = released.tts.backbone.num_hidden_layers;
        let mut selected = Vec::new();
        if !norm_only {
            selected.push((description.graph().groups()[0].id(), 0..blocks));
        }
        selected.push((description.graph().groups()[1].id(), 0..1));
        let partition = ArchitecturePartition::from_description(
            &description,
            selected,
            PartitionOwnership::new(!norm_only, true, std::iter::empty::<&str>()).unwrap(),
            &state,
            &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..blocks,
            )]),
            (),
            TransformerBoundarySchema::new(&config),
        )
        .unwrap();
        let source: SharedCheckpointSource = Arc::new(HeaderSource(
            checkpoint::tests::official().tensors().clone(),
        ));
        let bindings =
            PreparedTtsTransformerBindings::from_parameters(config, source, &official_parameters())
                .unwrap()
                .into_partition(&partition)
                .unwrap();
        let mut drivers = vec![None, None];
        for group in partition.groups() {
            drivers[group.group_index()] = Some(
                LayeredPartitionDriver::new_with_state_ownership(
                    &partition,
                    group.group_index(),
                    group.global_units(),
                    group.group_index() == 0,
                )
                .unwrap(),
            );
        }
        let (world, rank) = if norm_only { (2, 1) } else { (1, 0) };
        let session = CollectiveGroupId::new(1);
        let group = CommunicationGroupDescriptor::new(
            session,
            0,
            (0..world).collect(),
            Some(rank),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    vec![publication_dtype],
                    CommunicationTensorLimits::new(1, 3, 1 << 20, None).unwrap(),
                    true,
                )
                .unwrap(),
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap();
        let manifest = CommunicationManifest::new(world, rank, vec![group], vec![])
            .unwrap()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let plan = PartitionedExecutionPlan::new(
            description.graph().clone(),
            vec![
                (ArchitectureGroupKind::Decoder, false),
                (ArchitectureGroupKind::Projector, false),
            ],
            drivers,
            vec![],
            Some(PartitionOutputPublication {
                group: session,
                owner_rank: rank,
            }),
            Some(session),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        PreparedTtsTransformerPipelineBindings::new(
            bindings,
            partition,
            manifest,
            plan,
            None,
            PipelineActivationDtype::Float32,
        )
    }

    #[test]
    fn released_pipeline_admission_rejects_missing_ingress_and_wrong_publication_dtype() {
        admission(false, TensorDtype::F32).unwrap();
        assert!(matches!(
            admission(true, TensorDtype::F32),
            Err(TransformerPipelinePreparationError::Contract(
                "routes differ from the stage's single ingress and egress"
            ))
        ));
        assert!(matches!(
            admission(false, TensorDtype::I32),
            Err(TransformerPipelinePreparationError::Contract(
                "publication does not support the selected activation precision"
            ))
        ));
    }
}

/// Exact component sources, stage ownership and transport selection.
///
/// This low-level constructor accepts an explicit portable partition selection;
/// it does not advertise a coupled VoiceChat execution class or discover sources.
/// Construction reads only retained metadata and allocates no native resources.
pub struct PreparedTtsTransformerPipelineBindings {
    bindings: PreparedTtsTransformerPartitionBindings,
    partition: ArchitecturePartition<(), TransformerBoundarySchema>,
    manifest: CommunicationManifest,
    plan: PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
}
impl PreparedTtsTransformerPipelineBindings {
    /// Validates complete retained stage and communication authority.
    pub fn new(
        bindings: PreparedTtsTransformerPartitionBindings,
        partition: ArchitecturePartition<(), TransformerBoundarySchema>,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, TransformerPipelinePreparationError> {
        use TransformerPipelinePreparationError::Contract;
        validate_transformer_partition(bindings.config(), bindings.local_layout(), &partition)?;
        let description = tts_transformer_parameter_description(bindings.config())?;
        let kind = if bindings.config().geometry().causal {
            ArchitectureGroupKind::Decoder
        } else {
            ArchitectureGroupKind::AudioEncoder
        };
        for route in plan.routes() {
            if route.source_group != 0 || !matches!(route.destination_group, 0 | 1) {
                return Err(Contract(
                    "route does not carry the transformer block boundary",
                ));
            }
        }
        eredu_runtime::validate_component_pipeline(
            &description,
            &partition,
            bindings.unit_bindings().iter().map(|(address, _)| *address),
            bindings.state_layout(),
            bindings.local_layout(),
            &manifest,
            &plan,
            tensor_group,
            activation_dtype,
            &[(kind, false), (ArchitectureGroupKind::Projector, false)],
            0,
            bindings.config().geometry().max_position_embeddings,
            |_| {
                partition.boundary_schema().wire_schema().map_err(|_| {
                    eredu_runtime::PipelineBindingError::Contract(
                        "invalid component boundary schema",
                    )
                })
            },
        )
        .map_err(|error| match error {
            eredu_runtime::PipelineBindingError::Communication(error) => {
                TransformerPipelinePreparationError::Communication(error)
            }
            eredu_runtime::PipelineBindingError::LocalAgreement(error) => {
                TransformerPipelinePreparationError::LocalAgreement(error)
            }
            eredu_runtime::PipelineBindingError::Contract(message) => Contract(message),
        })?;
        Ok(Self {
            bindings,
            partition,
            manifest,
            plan,
            tensor_group,
            activation_dtype,
        })
    }

    /// Exact retained manifest for native communicator realization.
    pub fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }
    /// Selected tensor collective identity, when projections are partitioned.
    pub fn tensor_group(&self) -> Option<CollectiveGroupId> {
        self.tensor_group
    }
    /// Exact stage geometry and source authority for inspection.
    pub fn bindings(&self) -> &PreparedTtsTransformerPartitionBindings {
        &self.bindings
    }
    /// Consumes the validated selection without reopening artifacts.
    pub fn into_parts(
        self,
    ) -> (
        PreparedTtsTransformerPartitionBindings,
        ArchitecturePartition<(), TransformerBoundarySchema>,
        CommunicationManifest,
        PartitionedExecutionPlan,
        Option<CollectiveGroupId>,
        PipelineActivationDtype,
    ) {
        (
            self.bindings,
            self.partition,
            self.manifest,
            self.plan,
            self.tensor_group,
            self.activation_dtype,
        )
    }
}
