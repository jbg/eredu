//! Cold finite encoder stage, source and transport authority.
use crate::ear_encoder_layered::EncoderKernel;
use crate::{
    ear_encoder_partition::{
        validate_encoder_partition, EncoderBoundarySchema, EncoderPipelineCuts,
    },
    ear_encoder_preparation::PreparedEncoderPartitionBindings,
};
use eredu_core::{BackendFailure, CollectiveGroupId};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    ComponentPipelinePreparationError, ComponentRankMechanisms, LayerWeightResidency,
    LayerwiseModelMetadata, PartitionedExecutionPlan, PipelineActivationDtype,
    PipelineBindingError,
};

use crate::ear_encoder_stage::{EarEncoderPipeline, EarEncoderPipelineArchitecture};

type Pipeline<B> = eredu_runtime::PreparedComponentPipeline<EarEncoderPipelineArchitecture<B>>;

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
    ) -> Result<Self, ComponentPipelinePreparationError<crate::Error>> {
        let contract = |reason| ComponentPipelinePreparationError::Contract {
            component: "EAR encoder",
            reason,
        };
        validate_encoder_partition(bindings.declaration(), bindings.local_layout(), &partition)
            .map_err(ComponentPipelinePreparationError::Component)?;
        if cuts.config() != bindings.config()
            || activation_dtype != PipelineActivationDtype::Float32
        {
            return Err(contract("encoder cuts or F32 activation precision differ"));
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
        .map_err(|error| error.into_preparation("EAR encoder"))?;
        eredu_runtime::validate_component_pipeline(
            bindings.definition().parameters(),
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
        .map_err(|error| error.into_preparation("EAR encoder"))?;
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
    /// Binds the retained plan and constructs the portable stage.
    pub fn bind<B, C, H>(
        self,
        residency: LayerWeightResidency,
        communication: &C,
        mechanisms: &H,
    ) -> Result<
        (
            EarEncoderPipeline<H::Pipeline<EarEncoderPipelineArchitecture<B>>>,
            LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend + 'static,
        B::Tensor: 'static,
        H: ComponentRankMechanisms<B, eredu_runtime::StatelessComponentState, C>,
    {
        let pipeline = self.into_pipeline::<B>(residency)?;
        let (execution, metadata) = mechanisms.bind_pipeline(pipeline, communication)?;
        Ok((EarEncoderPipeline::new(execution), metadata))
    }

    /// Builds the typed stage from its retained source, ownership and communication plan.
    fn into_pipeline<B: NeuralBackend>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<Pipeline<B>, BackendFailure> {
        let (declaration, local_layout, invocations, source, units) = self.bindings.into_parts();
        let architecture = EncoderKernel::<B>::new_partition(
            declaration,
            local_layout.as_ref(),
            &self.partition,
            &self.cuts,
        )
        .map_err(BackendFailure::from_error)?;
        let pipeline = eredu_runtime::PreparedComponentPipeline {
            architecture,
            source,
            invocations,
            pinned: Vec::new(),
            units,
            manifest: self.manifest,
            plan: self.plan,
            tensor_group: self.tensor_group,
            activation_dtype: self.activation_dtype,
            residency,
        };
        Ok(pipeline)
    }
}
