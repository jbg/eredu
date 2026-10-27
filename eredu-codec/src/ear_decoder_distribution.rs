//! Cold streaming decoder stage, source and transport authority.
use crate::{
    ear_decoder_partition::{
        validate_decoder_partition, DecoderBoundarySchema, DecoderPipelineCuts,
    },
    ear_decoder_preparation::PreparedDecoderPartitionBindings,
};
use eredu_core::CollectiveGroupId;
use eredu_runtime::{
    ComponentPipelinePreparationError, ArchitectureBoundary, ArchitectureGroupKind,
    ArchitecturePartition, CommunicationManifest, PartitionedExecutionPlan,
    PipelineActivationDtype, PipelineBindingError,
};

/// Exact source recipes, global cuts and distributed authority for one streaming stage.
/// This does not select the surrounding waveform preparation or conversational scheduler.
pub struct PreparedDecoderPipelineBindings {
    bindings: PreparedDecoderPartitionBindings,
    partition: ArchitecturePartition<(), DecoderBoundarySchema>,
    cuts: DecoderPipelineCuts,
    manifest: CommunicationManifest,
    plan: PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
}
impl PreparedDecoderPipelineBindings {
    /// Checks stage ownership and every route without reading payloads or allocating tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bindings: PreparedDecoderPartitionBindings,
        partition: ArchitecturePartition<(), DecoderBoundarySchema>,
        cuts: DecoderPipelineCuts,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, ComponentPipelinePreparationError<crate::Error>> {
        let contract = |reason| ComponentPipelinePreparationError::Contract {
            component: "EAR decoder",
            reason,
        };
        validate_decoder_partition(
            bindings.definition(),
            bindings.config(),
            bindings.prefix(),
            bindings.local_layout(),
            &partition,
        )
        .map_err(ComponentPipelinePreparationError::Component)?;
        if cuts.config() != bindings.config()
            || activation_dtype != PipelineActivationDtype::Float32
        {
            return Err(contract("decoder cuts or F32 activation precision differ"));
        }
        eredu_runtime::validate_component_pipeline_ranges(
            bindings.definition().parameters().unit_layout().len(),
            partition.groups()[0].global_units(),
            0,
            cuts.routes(),
            &manifest,
            &plan,
            tensor_group,
        )
        .map_err(|error| error.into_preparation("EAR decoder"))?;
        eredu_runtime::validate_component_pipeline(
            bindings.definition().parameters(),
            &partition,
            bindings.unit_bindings().map(|(address, _)| address),
            partition.state().map(|state| state.layout()),
            bindings.local_layout(),
            &manifest,
            &plan,
            tensor_group,
            activation_dtype,
            &[(ArchitectureGroupKind::AudioDecoder, false)],
            0,
            i32::MAX as usize,
            |route| {
                cuts.schema(route.id())
                    .map_err(|_| PipelineBindingError::Contract("invalid decoder route cut"))?
                    .wire_schema()
                    .map_err(|_| PipelineBindingError::Contract("invalid decoder route geometry"))
            },
        )
        .map_err(|error| error.into_preparation("EAR decoder"))?;
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
    pub fn bindings(&self) -> &PreparedDecoderPartitionBindings {
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
        PreparedDecoderPartitionBindings,
        ArchitecturePartition<(), DecoderBoundarySchema>,
        DecoderPipelineCuts,
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
