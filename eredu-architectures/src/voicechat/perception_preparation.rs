//! Cold exact binding plans for independently leased perception units.

use std::collections::BTreeMap;

use super::perception_layered::{BoundPerception, PerceptionKernel, PerceptionUnit};
use eredu_checkpoint::{recipe::RecipeError, store::SharedCheckpointSource};
use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    ComponentPreparationError, LocalModelLayout, PreparedComponent, RecipeBindingError,
    WeightBinding,
};
use eredu_runtime::{LayeredTensorCohort, ParameterUnitMechanisms, SubmissionBackend};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{admitted_member, execution_parameter, SpeechParameterError},
    perception::{PerceptionDimensions, PerceptionError},
    perception_specs::{released_declaration, PerceptionDeclaration},
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum PerceptionPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Invalid floating execution profile or attempted precision reselection.
    #[error("invalid speech precision: {0}")]
    Precision(&'static str),
    /// Perception geometry cannot be constructed.
    #[error(transparent)]
    Perception(#[from] PerceptionError),
    /// A required parameter is missing from the selected component.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
}

/// Cold bindings for subsampling, each Conformer block, and final projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedPerceptionBindings {
    prepared: PreparedComponent<PerceptionDeclaration>,
}

impl PreparedPerceptionBindings {
    /// Selects portable execution precision and exact converted bindings before materialization.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, PerceptionPreparationError> {
        if self.precision() != Default::default() && self.precision() != precision {
            return Err(PerceptionPreparationError::Precision(
                "precision is already selected",
            ));
        }
        precision
            .validate()
            .map_err(PerceptionPreparationError::Precision)?;
        let declaration = super::precision::SpeechPrecisionDeclaration::from_groups(
            self.definition().parameters().groups(),
        );
        self.prepared = self.prepared.transform_units(|source, units| {
            precision
                .bindings(source, units, &declaration)
                .map_err(PerceptionPreparationError::from)
        })?;
        let declaration = self.declaration().clone().with_precision(precision)?;
        self.prepared = self
            .prepared
            .project("voicechat/perception/leased-units", declaration)?;
        Ok(self)
    }
    /// Retained floating storage and arithmetic policy.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.declaration().precision()
    }

    /// Projects complete perception bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, PerceptionPreparationError> {
        let source = artifacts.component_source(Component::Perception).ok_or(
            SpeechParameterError::MissingComponent(Component::Perception),
        )?;
        Self::from_parameters(
            released_declaration()?.clone(),
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component and preprocessing payloads are excluded.
    pub fn from_parameters(
        declaration: PerceptionDeclaration,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, PerceptionPreparationError> {
        let prepared = PreparedComponent::prepare(
            "voicechat/perception/leased-units",
            declaration,
            source.clone(),
            |member| {
                let parameter =
                    admitted_member(parameters, &source, Component::Perception, member)?;
                let parameter = execution_parameter(parameter, &source)?;
                Ok::<_, PerceptionPreparationError>(parameter.recipe)
            },
        )?;
        let result = Self { prepared };
        let precision = result.precision();
        result.with_precision(precision)
    }

    /// Binds prepared units locally or to an explicitly selected tensor cohort.
    pub fn bind<B, H, C>(
        self,
        residency: eredu_runtime::LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
        mechanisms: &H,
        cohort: Option<C>,
    ) -> Result<
        (
            BoundPerception<B, H::UnitPolicy<PerceptionUnit<B>>, C>,
            eredu_runtime::LayerwiseModelMetadata,
        ),
        BackendFailure,
    >
    where
        B: DistributedNeuralBackend
            + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
            + 'static,
        H: ParameterUnitMechanisms<B>,
        C: LayeredTensorCohort<B>,
    {
        let selected = if let Some(cohort) = &cohort {
            let (manifest, group) = cohort.selection();
            let width = self.dimensions().block.width as usize;
            let sum = eredu_runtime::CommunicationOperationRequirement::tensors(
                eredu_runtime::CommunicationOperation::AllReduceSum,
                [eredu_core::checkpoint::TensorDtype::F32],
                eredu_runtime::CommunicationTensorLimits::new(1, 3, width, None)
                    .map_err(BackendFailure::from_error)?,
                true,
            )
            .map_err(BackendFailure::from_error)?;
            eredu_runtime::replicated_execution::validate_collective_coordination(
                manifest,
                group,
                [sum],
            )
            .map_err(BackendFailure::from_error)?;
            self.select_parallel(cohort.tensor_topology()?)
                .map_err(BackendFailure::from_error)?
                .0
        } else {
            if self.local_layout().is_some() {
                return Err(BackendFailure::from_error(PerceptionError::State(
                    "tensor-local parameter placement requires its selected cohort",
                )));
            }
            self
        };
        let (declaration, local, invocations, source, units) = selected.into_parts();
        let definition = declaration.definition().clone();
        let kernel = match local.as_ref() {
            Some(layout) => PerceptionKernel::<B>::new_parallel(declaration, layout),
            None => PerceptionKernel::<B>::new(declaration),
        }
        .map_err(BackendFailure::from_error)?;
        let architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
            .map_err(BackendFailure::from_error)?;
        eredu_runtime::BoundLayerwiseComponent::bind(
            architecture,
            source,
            invocations,
            units,
            residency,
            context,
            mechanisms,
            cohort,
        )
    }

    /// Selects complete-stack tensor placement through the shared physical planner.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), PerceptionPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }
    /// Applies exact selected tensor placements without changing retained source authority.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, PerceptionPreparationError> {
        self.declaration().local_dimensions(layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }
    /// Narrows retained source authority and exact bindings to one selected stage.
    /// Unit addresses remain canonical; this result is not a zero-based complete
    /// provider layout and must be consumed by a partition traversal.
    pub fn into_partition<G>(
        self,
        partition: &eredu_runtime::ArchitecturePartition<
            G,
            super::perception_layered::PerceptionBoundarySchema,
        >,
    ) -> Result<PreparedPerceptionPartitionBindings, PerceptionPreparationError> {
        super::perception_layered::validate_perception_partition(
            self.declaration(),
            self.prepared.local_layout(),
            partition,
        )?;
        let prepared = self.prepared.select_units(partition.units())?;
        Ok(PreparedPerceptionPartitionBindings {
            prepared,
            state_layout: partition.state().map(|state| state.layout().clone()),
        })
    }

    /// Exact retained tensor placement, present only after parallel selection.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Typed scope retained through precision and placement selection.
    pub fn declaration(&self) -> &PerceptionDeclaration {
        self.prepared.declaration()
    }

    /// Validated perception geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.declaration().dimensions()
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Complete semantic declaration retained through precision and placement selection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    fn into_parts(
        self,
    ) -> (
        PerceptionDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (declaration, selected, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (declaration, selected, layout, source, units)
    }
}

/// Exact canonical units and restricted retained sources for a pipeline stage.
/// A parameter-only stage has no state and owns only its selected recipes.
pub struct PreparedPerceptionPartitionBindings {
    prepared: PreparedComponent<PerceptionDeclaration>,
    state_layout: Option<eredu_runtime::StateLayout>,
}
impl PreparedPerceptionPartitionBindings {
    /// Global scope retained alongside exact selected stage resources.
    pub fn declaration(&self) -> &PerceptionDeclaration {
        self.prepared.declaration()
    }

    /// Retained floating policy for partition construction.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.declaration().precision()
    }
    /// Global family equations; physical TP geometry is retained separately.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.declaration().dimensions()
    }
    /// Exact selected physical layout when tensor parallelism is also selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }
    /// Source authority limited to the actual selected unit recipes.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Canonical addresses and their exact resident/bounded materialization recipes.
    pub fn unit_bindings(
        &self,
    ) -> impl ExactSizeIterator<Item = (eredu_runtime::ExecutionUnitAddress, &[WeightBinding])>
    {
        self.prepared.addressed_unit_bindings()
    }
    /// Selected block cache schedule; absent for a parameter-only stage.
    pub fn state_layout(&self) -> Option<&eredu_runtime::StateLayout> {
        self.state_layout.as_ref()
    }
    /// Complete definition with global cache identities and parameter ownership.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }
    /// Consumes exact stage authority without rebasing canonical unit addresses.
    pub fn into_parts(
        self,
    ) -> (
        PerceptionDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
        Option<eredu_runtime::StateLayout>,
    ) {
        let (declaration, local_layout, invocations, source, units) = self
            .prepared
            .into_addressed_unit_parts()
            .expect("partition declares only independently leased units");
        (
            declaration,
            local_layout,
            invocations,
            source,
            units,
            self.state_layout,
        )
    }
}
