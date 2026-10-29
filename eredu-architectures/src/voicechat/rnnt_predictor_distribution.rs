//! Cold recurrent stage, source and communication authority.
use super::{
    rnnt::RnntError,
    rnnt_predictor_partition::{
        validate_rnnt_predictor_partition, RnntPredictorBoundarySchema, RnntPredictorPipelineCuts,
    },
    rnnt_predictor_preparation::PreparedRnntPredictorBindings,
};
use eredu_core::CollectiveGroupId;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitecturePartition, CommunicationManifest,
    ComponentPipelinePreparationError, PartitionedExecutionPlan, PipelineActivationDtype,
    PipelineBindingError,
};

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
        validate_rnnt_predictor_partition(bindings.definition(), bindings.dimensions(), &partition)
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
    /// Consumes the complete validated authority without reconstructing topology or sources.
    pub fn into_parts(
        self,
    ) -> (
        PreparedRnntPredictorBindings,
        ArchitecturePartition<(), RnntPredictorBoundarySchema>,
        RnntPredictorPipelineCuts,
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
