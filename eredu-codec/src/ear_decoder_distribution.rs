//! Cold streaming decoder stage, source and transport authority.
use crate::ear_decoder_layered::{DecoderKernel, DecoderRuntimeState};
use crate::{
    ear_decoder_partition::{
        validate_decoder_partition, DecoderBoundarySchema, DecoderPipelineCuts,
    },
    ear_decoder_preparation::PreparedDecoderPartitionBindings,
};
use eredu_core::{BackendFailure, CollectiveGroupId};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    ComponentPipelinePreparationError, ComponentRankMechanisms, LayerWeightResidency,
    LayerwiseModelMetadata, PartitionedExecutionPlan, PipelineActivationDtype,
    PipelineBindingError,
};

use crate::ear_decoder_stage::{EarDecoderPipeline, EarDecoderPipelineArchitecture};

type Pipeline<B> = eredu_runtime::PreparedComponentPipeline<EarDecoderPipelineArchitecture<B>>;

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
        validate_decoder_partition(bindings.declaration(), bindings.local_layout(), &partition)
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
    /// Binds the retained plan and constructs the portable stage.
    pub fn bind<B, C, H>(
        self,
        residency: LayerWeightResidency,
        communication: &C,
        mechanisms: &H,
    ) -> Result<
        (
            EarDecoderPipeline<B, H::Pipeline<EarDecoderPipelineArchitecture<B>>>,
            LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend + 'static,
        B::Tensor: 'static,
        H: ComponentRankMechanisms<B, DecoderRuntimeState<B::Tensor>, C>,
    {
        let (pipeline, layout, state) = self.into_pipeline::<B>(residency)?;
        let (execution, metadata) = mechanisms.bind_pipeline(pipeline, communication)?;
        Ok((EarDecoderPipeline::new(execution, layout, state), metadata))
    }

    /// Builds the typed stage from its retained source, ownership and communication plan.
    fn into_pipeline<B: NeuralBackend>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<
        (
            Pipeline<B>,
            eredu_runtime::StateLayout,
            DecoderRuntimeState<B::Tensor>,
        ),
        BackendFailure,
    > {
        let layout = self
            .partition
            .state()
            .ok_or_else(|| {
                BackendFailure::from_error(eredu_nn::Error::backend(
                    "pipeline stage has no retained state layout",
                ))
            })?
            .layout()
            .clone();
        let (declaration, local_layout, invocations, source, units) = self.bindings.into_parts();
        let architecture = DecoderKernel::<B>::new_partition(
            declaration,
            local_layout.as_ref(),
            &self.partition,
            &self.cuts,
        )
        .map_err(BackendFailure::from_error)?;
        let state = DecoderRuntimeState::for_partition(architecture.kernel(), None)
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
        Ok((pipeline, layout, state))
    }
}
