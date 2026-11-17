//! Cold recurrent stage, source and communication authority.
use super::rnnt_predictor::{RnntPredictorKernel, RnntPredictorRuntimeState};
use super::{
    rnnt::RnntError,
    rnnt_predictor_partition::{
        validate_rnnt_predictor_partition, RnntPredictorBoundarySchema, RnntPredictorPipelineCuts,
    },
    rnnt_predictor_preparation::PreparedRnntPredictorBindings,
};
use eredu_core::{BackendFailure, CollectiveGroupId};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    ComponentPipelinePreparationError, ComponentRankMechanisms, LayerWeightResidency,
    LayerwiseModelMetadata, PartitionedExecutionPlan, PipelineActivationDtype,
    PipelineBindingError,
};

use super::rnnt_predictor_stage::{RnntPredictorPipeline, RnntPredictorPipelineArchitecture};

type Pipeline<B> = eredu_runtime::PreparedComponentPipeline<RnntPredictorPipelineArchitecture<B>>;

/// Exact source recipes, global cuts and distributed authority for one streaming stage.
/// This does not select the surrounding transcription policy or conversational scheduler.
pub struct PreparedRnntPredictorPipelineBindings {
    bindings: PreparedRnntPredictorBindings,
    partition: ArchitecturePartition<(), RnntPredictorBoundarySchema>,
    cuts: RnntPredictorPipelineCuts,
    manifest: CommunicationManifest,
    plan: PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
}
impl PreparedRnntPredictorPipelineBindings {
    /// Checks stage ownership and every route without reading payloads or allocating tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bindings: PreparedRnntPredictorBindings,
        partition: ArchitecturePartition<(), RnntPredictorBoundarySchema>,
        cuts: RnntPredictorPipelineCuts,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, ComponentPipelinePreparationError<RnntError>> {
        let contract = |reason| ComponentPipelinePreparationError::Contract {
            component: "RNNT predictor",
            reason,
        };
        validate_rnnt_predictor_partition(bindings.declaration(), &partition)
            .map_err(ComponentPipelinePreparationError::Component)?;
        if cuts.dimensions() != bindings.dimensions()
            || activation_dtype != PipelineActivationDtype::Float32
        {
            return Err(contract(
                "predictor cuts or F32 activation precision differ",
            ));
        }
        eredu_runtime::validate_component_pipeline_ranges(
            bindings.dimensions().predictor_layers,
            partition.groups()[0].global_units(),
            0,
            cuts.routes(),
            &manifest,
            &plan,
            tensor_group,
        )
        .map_err(|error| error.into_preparation("RNNT predictor"))?;
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
            &[(ArchitectureGroupKind::Recurrent, false)],
            0,
            1,
            |route| {
                cuts.schema(route.id())
                    .map_err(|_| PipelineBindingError::Contract("invalid predictor route cut"))?
                    .wire_schema()
                    .map_err(|_| PipelineBindingError::Contract("invalid predictor route geometry"))
            },
        )
        .map_err(|error| error.into_preparation("RNNT predictor"))?;
        // Route-role bounds are admitted by their exact boundary contracts.
        // The published row is separate and needs its own concrete payload bound.
        let publication = plan.publication().expect("validated predictor publication");
        let group = manifest
            .groups()
            .iter()
            .find(|group| group.id() == publication.group)
            .expect("validated predictor publication group");
        let capabilities = eredu_runtime::CommunicationCapabilities::new(
            group.requirements().operations().iter().cloned(),
        )?;
        capabilities.validate_requirement(
            &eredu_runtime::CommunicationOperationRequirement::tensors(
                eredu_runtime::CommunicationOperation::Broadcast,
                [eredu_core::checkpoint::TensorDtype::F32],
                eredu_runtime::CommunicationTensorLimits::new(
                    1,
                    2,
                    bindings.dimensions().predictor_width as usize,
                    None,
                )?,
                true,
            )?,
        )?;
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
    pub fn bindings(&self) -> &PreparedRnntPredictorBindings {
        &self.bindings
    }
    /// Exact distributed manifest, with bounded completion and publication authority.
    pub fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }
    /// Exact retained routes, output publication and commit authority.
    pub fn plan(&self) -> &PartitionedExecutionPlan {
        &self.plan
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
            RnntPredictorPipeline<H::Pipeline<RnntPredictorPipelineArchitecture<B>>>,
            LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend + 'static,
        B::Tensor: 'static,
        H: ComponentRankMechanisms<B, RnntPredictorRuntimeState<B::Tensor>, C>,
    {
        let (pipeline, layout) = self.into_pipeline::<B>(residency)?;
        let (execution, metadata) = mechanisms.bind_pipeline(pipeline, communication)?;
        Ok((RnntPredictorPipeline::new(execution, layout), metadata))
    }

    /// Builds the typed stage from its retained source, ownership and communication plan.
    fn into_pipeline<B: NeuralBackend>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<(Pipeline<B>, eredu_runtime::StateLayout), BackendFailure> {
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
        let (declaration, invocations, source, units) = self.bindings.into_parts();
        let architecture =
            RnntPredictorKernel::<B>::new_partition(declaration, &self.partition, &self.cuts)
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
        Ok((pipeline, layout))
    }
}
