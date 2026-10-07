//! Exact cold authority for a retained perception component pipeline.
use super::{
    perception::PerceptionError,
    perception_layered::{
        perception_parameter_description, validate_perception_partition, PerceptionBoundarySchema,
    },
    perception_preparation::PreparedPerceptionPartitionBindings,
};
use eredu_core::CollectiveGroupId;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    PartitionExecutionError, PartitionedExecutionPlan, PipelineActivationDtype,
};

/// Invalid stage, source geometry or communication selection, before native allocation.
#[derive(Debug, thiserror::Error)]
pub enum PerceptionPipelinePreparationError {
    /// Family equations or local state geometry differ.
    #[error(transparent)]
    Perception(#[from] PerceptionError),
    /// Generic communication selection is inconsistent.
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
    /// The tensor cohort cannot propagate local preparation or lease failures.
    #[error(transparent)]
    LocalAgreement(#[from] eredu_runtime::PipelineLocalAgreementError),
    /// Selected source, stage or transport authority differs.
    #[error("invalid perception pipeline selection: {0}")]
    Contract(&'static str),
}

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
    ) -> Result<Self, PerceptionPipelinePreparationError> {
        use PerceptionPipelinePreparationError::Contract;
        validate_perception_partition(bindings.dimensions(), bindings.local_layout(), &partition)?;
        if activation_dtype != PipelineActivationDtype::Float32 {
            return Err(Contract("perception requires Float32 activation transport"));
        }
        for route in plan.routes() {
            if !matches!(
                (route.source_group, route.destination_group),
                (0, 1) | (1, 1) | (1, 2)
            ) {
                return Err(Contract(
                    "route does not carry a subsampled perception boundary",
                ));
            }
        }
        let description = perception_parameter_description(bindings.dimensions())?;
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
        .map_err(|error| match error {
            eredu_runtime::PipelineBindingError::Communication(error) => {
                PerceptionPipelinePreparationError::Communication(error)
            }
            eredu_runtime::PipelineBindingError::LocalAgreement(error) => {
                PerceptionPipelinePreparationError::LocalAgreement(error)
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
    /// Consumes validated authority without source discovery or artifact reopening.
    pub fn into_parts(
        self,
    ) -> (
        PreparedPerceptionPartitionBindings,
        ArchitecturePartition<(), PerceptionBoundarySchema>,
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
