//! Restricted RNNT embedding and joint recipes without recurrent allocation.
use super::{
    rnnt::RnntDimensions,
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    rnnt_specs::RnntDeclaration,
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{LocalModelLayout, PreparedComponent, WeightBinding};

/// Exact four-unit source authority, including vocabulary placement when selected.
/// Recurrent recipes and replica declarations are excluded from this owner.
pub struct PreparedRnntProjectionBindings {
    prepared: PreparedComponent<RnntDeclaration<false>>,
}
impl PreparedRnntProjectionBindings {
    /// Projects existing retained RNNT recipes without reading parameter payloads.
    /// Existing tensor selection is retained exactly; recipes are never sliced twice.
    pub fn from_rnnt(bindings: PreparedRnntBindings) -> Result<Self, RnntPreparationError> {
        let declaration = bindings.declaration().projections()?;
        let prepared = bindings
            .prepared
            .project("voicechat/rnnt/projection-units", declaration)?;
        if let Some(layout) = prepared.local_layout() {
            prepared.declaration().local_dimensions(layout)?;
        }
        Ok(Self { prepared })
    }

    /// Selects vocabulary shards and exact encoder/prediction replicas for one cohort.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RnntPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }
    /// Applies architecture-owned vocabulary transforms without widening source access.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntPreparationError> {
        self.prepared.declaration().local_dimensions(layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }
    /// Global geometry used by projection equations and the shared greedy driver.
    pub fn dimensions(&self) -> RnntDimensions {
        self.prepared.declaration().dimensions()
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
    /// Projection semantic declaration retained independently of excluded recurrent resources.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }
    /// Transfers compact layout and exactly the authority needed by its four units.
    pub fn into_parts(
        self,
    ) -> (
        RnntDeclaration<false>,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (declaration, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("RNNT projection declares only independently leased units");
        (declaration, local, layout, source, units)
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
