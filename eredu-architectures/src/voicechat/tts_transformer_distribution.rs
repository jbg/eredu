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
