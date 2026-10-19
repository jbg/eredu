//! Cold exact binding plans for independently leased perception units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::RecipeError,
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    BindingPlacementError, ComponentPreparationError, ExecutionUnitLayout, LocalModelLayout,
    PreparedComponent, RecipeBindingError, WeightBinding,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{execution_parameter, SpeechParameterError},
    perception::{PerceptionDimensions, PerceptionError},
    perception_layered::perception_parameter_description,
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
    /// Source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// Exact recipes cannot realize the selected tensor placement.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// The shared planner rejected the selected tensor topology.
    #[error("invalid perception tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat perception parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for subsampling, each Conformer block, and final projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedPerceptionBindings {
    precision: super::precision::SpeechPrecision,
    dimensions: PerceptionDimensions,
    prepared: PreparedComponent,
}

impl PreparedPerceptionBindings {
    /// Selects portable execution precision and exact converted bindings before materialization.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, PerceptionPreparationError> {
        if self.precision != Default::default() && self.precision != precision {
            return Err(PerceptionPreparationError::Precision(
                "precision is already selected",
            ));
        }
        precision
            .validate()
            .map_err(PerceptionPreparationError::Precision)?;
        self.prepared = self.prepared.transform_units(|source, units| {
            precision
                .bindings(source, units)
                .map_err(PerceptionPreparationError::from)
        })?;
        self.precision = precision;
        Ok(self)
    }
    /// Retained floating storage and arithmetic policy.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }

    /// Projects complete perception bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, PerceptionPreparationError> {
        let source = artifacts.component_source(Component::Perception).ok_or(
            SpeechParameterError::MissingComponent(Component::Perception),
        )?;
        Self::from_parameters(
            PerceptionDimensions::from_config(&artifacts.plan().config().perception)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component and preprocessing payloads are excluded.
    pub fn from_parameters(
        dimensions: PerceptionDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, PerceptionPreparationError> {
        let description = perception_parameter_description(dimensions)?;
        let prepared = PreparedComponent::prepare(
            "voicechat/perception/leased-units",
            description,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter = parameters
                    .get(name)
                    .filter(|p| p.component == Component::Perception)
                    .ok_or_else(|| SpeechParameterError::MissingParameter {
                        component: Component::Perception,
                        name: name.into(),
                    })?;
                if parameter.output.shape != member.global_shape()
                    || parameter.recipe.infer(source.as_ref())? != parameter.output
                {
                    return Err(PerceptionPreparationError::Geometry(name.into()));
                }
                let parameter = execution_parameter(parameter, &source)?;
                Ok::<_, PerceptionPreparationError>(parameter.recipe)
            },
        )?;
        Ok(Self {
            precision: Default::default(),
            dimensions,
            prepared,
        })
    }

    /// Selects complete-stack tensor placement through the shared physical planner.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), PerceptionPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(PerceptionPreparationError::Parallel(
                "complete-stack perception selection requires a tensor-only topology".into(),
            ));
        }
        let description = perception_parameter_description(self.dimensions)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(PerceptionPreparationError::Parallel)?;
        Ok((self.into_parallel(&layout)?, layout))
    }
    /// Applies exact selected tensor placements without changing retained source authority.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, PerceptionPreparationError> {
        super::perception_layered::perception_local_dimensions(self.dimensions, layout)?;
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
            self.dimensions,
            self.prepared.local_layout(),
            partition,
        )?;
        let prepared = self.prepared.select_units(partition.units())?;
        let (local_layout, layout, source, bindings) = prepared.into_unit_parts()?;
        let units = bindings
            .into_iter()
            .enumerate()
            .map(|(ordinal, bindings)| {
                (
                    layout.address(ordinal).expect("selected unit address"),
                    bindings,
                )
            })
            .collect();
        Ok(PreparedPerceptionPartitionBindings {
            precision: self.precision,
            dimensions: self.dimensions,
            local_layout,
            source,
            units,
            state_layout: partition.state().map(|state| state.layout().clone()),
        })
    }

    /// Exact retained tensor placement, present only after parallel selection.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated perception geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        PerceptionDimensions,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (_, layout, source, units) = self
            .prepared
            .into_unit_parts()
            .expect("component declares only independently leased units");
        (self.dimensions, layout, source, units)
    }
}

/// Exact canonical units and restricted retained sources for a pipeline stage.
/// A parameter-only stage has no state and owns only its selected recipes.
pub struct PreparedPerceptionPartitionBindings {
    precision: super::precision::SpeechPrecision,
    dimensions: PerceptionDimensions,
    local_layout: Option<LocalModelLayout>,
    source: SharedCheckpointSource,
    units: Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
    state_layout: Option<eredu_runtime::StateLayout>,
}
impl PreparedPerceptionPartitionBindings {
    /// Retained floating policy for partition construction.
    pub fn precision(&self) -> super::precision::SpeechPrecision {
        self.precision
    }
    /// Global family equations; physical TP geometry is retained separately.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions
    }
    /// Exact selected physical layout when tensor parallelism is also selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }
    /// Source authority limited to the actual selected unit recipes.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Canonical addresses and their exact resident/bounded materialization recipes.
    pub fn unit_bindings(&self) -> &[(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)] {
        &self.units
    }
    /// Selected block cache schedule; absent for a parameter-only stage.
    pub fn state_layout(&self) -> Option<&eredu_runtime::StateLayout> {
        self.state_layout.as_ref()
    }
    /// Consumes exact stage authority without rebasing canonical unit addresses.
    pub fn into_parts(
        self,
    ) -> (
        PerceptionDimensions,
        Option<LocalModelLayout>,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
        Option<eredu_runtime::StateLayout>,
    ) {
        (
            self.dimensions,
            self.local_layout,
            self.source,
            self.units,
            self.state_layout,
        )
    }
}
