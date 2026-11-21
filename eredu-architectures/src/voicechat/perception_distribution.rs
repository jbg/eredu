//! Exact cold authority for a retained perception component pipeline.
use super::perception_layered::{PerceptionKernel, PerceptionRuntimeState};
use super::{
    perception::PerceptionError,
    perception_layered::{validate_perception_partition, PerceptionBoundarySchema},
    perception_preparation::PreparedPerceptionPartitionBindings,
};
use eredu_core::communication::CommunicationManifest;
use eredu_core::communication::PipelineActivationDtype;
use eredu_core::{BackendFailure, CollectiveGroupId};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition,
    ComponentPipelinePreparationError, ComponentRankMechanisms, LayerWeightResidency,
    LayerwiseModelMetadata, PartitionedExecutionPlan,
};

use super::perception_stage::{PerceptionPipeline, PerceptionPipelineArchitecture};

type Pipeline<B> = eredu_runtime::PreparedComponentPipeline<PerceptionPipelineArchitecture<B>>;

/// Retained source recipes and exact partition/transport authority.
/// This component selection does not advertise a coupled distributed session.
pub struct PreparedPerceptionPipelineBindings {
    bindings: PreparedPerceptionPartitionBindings,
    partition: ArchitecturePartition<(), PerceptionBoundarySchema>,
    manifest: CommunicationManifest,
    plan: PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
}
impl PreparedPerceptionPipelineBindings {
    /// Validates the complete stage selection without opening payloads or native resources.
    pub fn new(
        bindings: PreparedPerceptionPartitionBindings,
        partition: ArchitecturePartition<(), PerceptionBoundarySchema>,
        manifest: CommunicationManifest,
        plan: PartitionedExecutionPlan,
        tensor_group: Option<CollectiveGroupId>,
        activation_dtype: PipelineActivationDtype,
    ) -> Result<Self, ComponentPipelinePreparationError<PerceptionError>> {
        let contract = |reason| ComponentPipelinePreparationError::Contract {
            component: "perception",
            reason,
        };
        validate_perception_partition(bindings.declaration(), bindings.local_layout(), &partition)
            .map_err(ComponentPipelinePreparationError::Component)?;
        if activation_dtype != PipelineActivationDtype::Float32 {
            return Err(contract("perception requires Float32 activation transport"));
        }
        for route in plan.routes() {
            if !matches!(
                (route.source_group, route.destination_group),
                (0, 1) | (1, 1) | (1, 2)
            ) {
                return Err(contract(
                    "route does not carry a subsampled perception boundary",
                ));
            }
        }
        eredu_runtime::validate_component_pipeline(
            bindings.definition().parameters(),
            &partition,
            bindings.unit_bindings().map(|(address, _)| address),
            bindings.state_layout(),
            bindings.local_layout(),
            &manifest,
            &plan,
            tensor_group,
            activation_dtype,
            &[
                (ArchitectureGroupKind::AudioEncoder, false),
                (ArchitectureGroupKind::AudioEncoder, false),
                (ArchitectureGroupKind::Projector, false),
            ],
            1,
            i32::MAX as usize,
            |_| {
                partition.boundary_schema().wire_schema().map_err(|_| {
                    eredu_runtime::PipelineBindingError::Contract(
                        "invalid component boundary schema",
                    )
                })
            },
        )
        .map_err(|error| error.into_preparation("perception"))?;
        Ok(Self {
            bindings,
            partition,
            manifest,
            plan,
            tensor_group,
            activation_dtype,
        })
    }
    /// Exact manifest retained for native realization.
    pub fn manifest(&self) -> &CommunicationManifest {
        &self.manifest
    }
    /// Exact retained routes, output publication and commit authority.
    pub fn plan(&self) -> &PartitionedExecutionPlan {
        &self.plan
    }
    /// Selected tensor collective identity, when projections are partitioned.
    pub fn tensor_group(&self) -> Option<CollectiveGroupId> {
        self.tensor_group
    }
    /// Exact component geometry, sources and addressed parameter recipes.
    pub fn bindings(&self) -> &PreparedPerceptionPartitionBindings {
        &self.bindings
    }
    /// Binds the retained plan and constructs the portable stage.
    pub fn bind<B, C, H>(
        self,
        residency: LayerWeightResidency,
        communication: &C,
        mechanisms: &H,
    ) -> Result<
        (
            PerceptionPipeline<B, H::Pipeline<PerceptionPipelineArchitecture<B>>>,
            LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend + 'static,
        B::Tensor: 'static,
        H: ComponentRankMechanisms<B, PerceptionRuntimeState<B::Tensor>, C>,
    {
        let (pipeline, dimensions, layout, state) = self.into_pipeline::<B>(residency)?;
        let (execution, metadata) = pipeline.bind(mechanisms, communication)?;
        Ok((
            PerceptionPipeline::new(execution, dimensions, layout, state),
            metadata,
        ))
    }

    /// Builds the typed stage from its retained source, ownership and communication plan.
    fn into_pipeline<B: NeuralBackend>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<
        (
            Pipeline<B>,
            super::perception::PerceptionDimensions,
            Option<eredu_runtime::StateLayout>,
            PerceptionRuntimeState<B::Tensor>,
        ),
        BackendFailure,
    > {
        let (declaration, local_layout, invocations, source, units, layout) =
            self.bindings.into_parts();
        let dimensions = declaration.dimensions();
        let architecture = PerceptionKernel::<B>::new_partition(
            declaration,
            local_layout.as_ref(),
            &self.partition,
        )
        .map_err(BackendFailure::from_error)?;
        let state = PerceptionRuntimeState::for_partition(architecture.kernel(), None)
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
        Ok((pipeline, dimensions, layout, state))
    }
}
