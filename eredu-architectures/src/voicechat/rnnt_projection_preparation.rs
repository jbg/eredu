//! Restricted RNNT embedding and joint recipes without recurrent allocation.
use super::{
    rnnt::RnntDimensions,
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    rnnt_projection_layered::{
        rnnt_projection_local_dimensions, rnnt_projection_parameter_description,
    },
};
use eredu_checkpoint::store::{RestrictedCheckpointSource, SharedCheckpointSource};
use eredu_runtime::{place_weight_bindings, ExecutionUnitLayout, LocalModelLayout, WeightBinding};
use std::{collections::BTreeSet, sync::Arc};

/// Exact four-unit source authority, including vocabulary placement when selected.
/// Recurrent recipes and replica declarations are excluded from this owner.
pub struct PreparedRnntProjectionBindings {
    dimensions: RnntDimensions,
    local_layout: Option<LocalModelLayout>,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
}
impl PreparedRnntProjectionBindings {
    /// Projects existing retained RNNT recipes without reading parameter payloads.
    /// Existing tensor selection is retained exactly; recipes are never sliced twice.
    pub fn from_rnnt(bindings: PreparedRnntBindings) -> Result<Self, RnntPreparationError> {
        let (dimensions, full_local, full_layout, source, full_units) = bindings.into_parts();
        let description = rnnt_projection_parameter_description(dimensions)?;
        let mut units = Vec::new();
        for (ordinal, bindings) in full_units.into_iter().enumerate() {
            if [0, 2, 3, 4]
                .into_iter()
                .any(|group| full_layout.ordinal(group, 0) == Some(ordinal))
            {
                units.push(bindings);
            }
        }
        if units.len() != 4 {
            return Err(RnntPreparationError::Geometry(
                "RNNT projection units are incomplete".into(),
            ));
        }
        let local_layout = full_local
            .map(|full| {
                let mut selected = LocalModelLayout::default();
                for member in description
                    .groups()
                    .iter()
                    .flat_map(|group| group.members())
                {
                    let tensor = full.tensor(member.target()).ok_or_else(|| {
                        RnntPreparationError::Parallel(format!(
                            "RNNT projection placement omits {}",
                            member.target()
                        ))
                    })?;
                    selected.insert(member.target().to_owned(), tensor.clone());
                }
                rnnt_projection_local_dimensions(dimensions, &selected)?;
                Ok::<_, RnntPreparationError>(selected)
            })
            .transpose()?;
        let inputs = units
            .iter()
            .flatten()
            .flat_map(|binding| {
                binding
                    .source_recipe()
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeSet<_>>();
        let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
            source,
            "voicechat/rnnt/projection-units",
            inputs,
        )?);
        Ok(Self {
            dimensions,
            local_layout,
            layout: description.unit_layout().clone(),
            source,
            units,
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
        let layout = eredu_runtime::derive_partitioned_local_layout(
            &rnnt_projection_parameter_description(self.dimensions)?,
            topology,
        )
        .map_err(RnntPreparationError::Parallel)?;
        if let Some(selected) = self.local_layout.as_ref() {
            if selected != &layout {
                return Err(RnntPreparationError::Parallel(
                    "retained RNNT projection placement differs from the tensor topology".into(),
                ));
            }
            return Ok((self, layout));
        }
        Ok((self.into_parallel(&layout)?, layout))
    }
    /// Applies architecture-owned vocabulary transforms without widening source access.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntPreparationError> {
        if self.local_layout.is_some() {
            return Err(RnntPreparationError::Parallel(
                "RNNT projection bindings already have a tensor partition".into(),
            ));
        }
        rnnt_projection_local_dimensions(self.dimensions, layout)?;
        self.units = self
            .units
            .into_iter()
            .map(|unit| place_weight_bindings(unit, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.local_layout = Some(layout.clone());
        Ok(self)
    }
    /// Global geometry used by projection equations and the shared greedy driver.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Exact optional vocabulary placement with no recurrent replica declarations.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }
    /// Architecture-ordered embedding, encoder, prediction and output bindings.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        &self.units
    }
    /// Retained source restricted to the selected nonrecurrent parameters.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
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
        (
            self.dimensions,
            self.local_layout,
            self.layout,
            self.source,
            self.units,
        )
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
