//! Restricted RNNT embedding and joint recipes without recurrent allocation.
use super::{
    rnnt::RnntDimensions,
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    rnnt_projection_layered::{
        rnnt_projection_local_dimensions, rnnt_projection_parameter_description,
    },
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{ExecutionUnitLayout, LocalModelLayout, PreparedComponent, WeightBinding};

/// Exact four-unit source authority, including vocabulary placement when selected.
/// Recurrent recipes and replica declarations are excluded from this owner.
pub struct PreparedRnntProjectionBindings {
    dimensions: RnntDimensions,
    prepared: PreparedComponent,
}
impl PreparedRnntProjectionBindings {
    /// Projects existing retained RNNT recipes without reading parameter payloads.
    /// Existing tensor selection is retained exactly; recipes are never sliced twice.
    pub fn from_rnnt(bindings: PreparedRnntBindings) -> Result<Self, RnntPreparationError> {
        let dimensions = bindings.dimensions();
        let description = rnnt_projection_parameter_description(dimensions)?;
        let prepared = bindings
            .prepared
            .project("voicechat/rnnt/projection-units", description)?;
        if let Some(layout) = prepared.local_layout() {
            rnnt_projection_local_dimensions(dimensions, layout)?;
        }
        Ok(Self {
            dimensions,
            prepared,
        })
    }

    /// Selects vocabulary shards and exact encoder/prediction replicas for one cohort.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RnntPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(RnntPreparationError::Parallel(
                "RNNT projection selection requires a tensor-only topology".into(),
            ));
        }
        let layout = self.prepared.parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }
    /// Applies architecture-owned vocabulary transforms without widening source access.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntPreparationError> {
        rnnt_projection_local_dimensions(self.dimensions, layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }
    /// Global geometry used by projection equations and the shared greedy driver.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Exact optional vocabulary placement with no recurrent replica declarations.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }
    /// Architecture-ordered embedding, encoder, prediction and output bindings.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }
    /// Retained source restricted to the selected nonrecurrent parameters.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Transfers compact layout and exactly the authority needed by its four units.
    pub fn into_parts(
        self,
    ) -> (
        RnntDimensions,
        Option<LocalModelLayout>,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (local, layout, source, units) = self
            .prepared
            .into_unit_parts()
            .expect("RNNT projection declares only independently leased units");
        (self.dimensions, local, layout, source, units)
    }
}

/// Validates only projection tensor collectives. Decision and failure agreement
/// are admitted separately by the shared greedy driver's selected coordinators.
pub fn validate_projection_parallel_communication(
    dimensions: RnntDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), RnntPreparationError> {
    super::rnnt_preparation::validate_neural_communication(dimensions, manifest, group, false)
}
