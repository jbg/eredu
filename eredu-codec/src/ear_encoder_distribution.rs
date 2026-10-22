//! Cold finite encoder stage, source and transport authority.
use crate::{
    ear_encoder_layered::encoder_parameter_description,
    ear_encoder_partition::{
        validate_encoder_partition, EncoderBoundarySchema, EncoderPipelineCuts,
    },
    ear_encoder_preparation::PreparedEncoderPartitionBindings,
};
use eredu_core::CollectiveGroupId;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    PartitionExecutionError, PartitionedExecutionPlan, PipelineActivationDtype,
    PipelineBindingError,
};

/// Invalid retained encoder geometry or communication selection, before native allocation.
#[derive(Debug, thiserror::Error)]
pub enum EncoderPipelinePreparationError {
    /// Codec-owned geometry, state or boundary validation failed.
    #[error(transparent)]
    Encoder(#[from] crate::Error),
    /// Shared pipeline authority or completion validation failed.
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
    /// The tensor cohort cannot propagate local preparation or lease failures.
    #[error(transparent)]
    LocalAgreement(#[from] eredu_runtime::PipelineLocalAgreementError),
    /// The selected source, stage or route contract differs from encoder ownership.
    #[error("invalid EAR encoder pipeline selection: {0}")]
    Contract(&'static str),
}

/// Exact source recipes, global cuts and distributed authority for one finite stage.
/// This does not select the surrounding waveform preparation or conversational scheduler.
pub struct PreparedEncoderPipelineBindings {
    bindings: PreparedEncoderPartitionBindings,
    partition: ArchitecturePartition<(), EncoderBoundarySchema>,
    cuts: EncoderPipelineCuts,
    manifest: CommunicationManifest,
    plan: PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
}
impl PreparedEncoderPipelineBindings {
    /// Checks stage ownership and every route without reading payloads or allocating tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bindings: PreparedEncoderPartitionBindings,
        partition: ArchitecturePartition<(), EncoderBoundarySchema>,
        cuts: EncoderPipelineCuts,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, EncoderPipelinePreparationError> {
        use EncoderPipelinePreparationError::Contract;
        validate_encoder_partition(
            bindings.config(),
            bindings.prefix(),
            bindings.local_layout(),
            &partition,
        )?;
        if cuts.config() != bindings.config()
            || activation_dtype != PipelineActivationDtype::Float32
        {
            return Err(Contract("encoder cuts or F32 activation precision differ"));
        }
        eredu_runtime::validate_component_pipeline_ranges(
            bindings.config().unit_specs()?.len(),
            partition.groups()[0].global_units(),
            0,
            cuts.routes(),
            &manifest,
            &plan,
            tensor_group,
        )
        .map_err(|error| match error {
            PipelineBindingError::Communication(error) => {
                EncoderPipelinePreparationError::Communication(error)
            }
            PipelineBindingError::LocalAgreement(error) => {
                EncoderPipelinePreparationError::LocalAgreement(error)
            }
            PipelineBindingError::Contract(message) => Contract(message),
        })?;
        let description = encoder_parameter_description(bindings.config(), bindings.prefix())?;
        eredu_runtime::validate_component_pipeline(
            &description,
            &partition,
            bindings.unit_bindings().map(|(address, _)| address),
            None,
            bindings.local_layout(),
            &manifest,
            &plan,
            tensor_group,
            activation_dtype,
            &[(ArchitectureGroupKind::AudioEncoder, false)],
            0,
            i32::MAX as usize,
            |route| {
                cuts.schema(route.id())
                    .map_err(|_| PipelineBindingError::Contract("invalid encoder route cut"))?
                    .wire_schema()
                    .map_err(|_| PipelineBindingError::Contract("invalid encoder route geometry"))
            },
        )
        .map_err(|error| match error {
            PipelineBindingError::Communication(error) => {
                EncoderPipelinePreparationError::Communication(error)
            }
            PipelineBindingError::LocalAgreement(error) => {
                EncoderPipelinePreparationError::LocalAgreement(error)
            }
            PipelineBindingError::Contract(message) => Contract(message),
        })?;
        Ok(Self {
            bindings,
            partition,
            cuts,
            manifest,
            plan,
            tensor_group,
            activation_dtype,
        })
    }
    /// Exact retained parameter and source selection.
    pub fn bindings(&self) -> &PreparedEncoderPartitionBindings {
        &self.bindings
    }
    /// Exact distributed manifest, with bounded completion and publication authority.
    pub fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }
    /// Selected tensor collective identity, when combined TP/PP is used.
    pub fn tensor_group(&self) -> Option<CollectiveGroupId> {
        self.tensor_group
    }
    /// Consumes the complete validated authority without reconstructing topology or sources.
    pub fn into_parts(
        self,
    ) -> (
        PreparedEncoderPartitionBindings,
        ArchitecturePartition<(), EncoderBoundarySchema>,
        EncoderPipelineCuts,
        CommunicationManifest,
        PartitionedExecutionPlan,
        Option<CollectiveGroupId>,
        PipelineActivationDtype,
    ) {
        (
            self.bindings,
            self.partition,
            self.cuts,
            self.manifest,
            self.plan,
            self.tensor_group,
            self.activation_dtype,
        )
    }
}
