//! Cold exact binding plans for independently leased EAR neural encoder units.

use eredu_checkpoint::{recipe::RecipeDtype, store::SharedCheckpointSource};
use eredu_runtime::{
    ComponentPreparationError, ComponentRecipeConversion, LocalModelLayout, PreparedComponent,
    WeightBinding,
};

use crate::ear_encoder::EncoderConfig;

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum EncoderPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Encoder geometry cannot be constructed.
    #[error(transparent)]
    Encoder(#[from] crate::Error),
    /// A required parameter is missing from the selected component.
    #[error("EAR encoder parameter {0} is missing")]
    MissingParameter(String),
}

/// Cold bindings for each input-projection, ConvNeXt and downsampling unit.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no waveform or conversational execution capability.
pub struct PreparedEncoderBindings {
    prepared: PreparedComponent<crate::declarations::EncoderDeclaration>,
}

impl PreparedEncoderBindings {
    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_recipes<'a>(
        declaration: crate::declarations::EncoderDeclaration,
        source: SharedCheckpointSource,
        parameters: impl FnMut(&str) -> Option<&'a eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> Result<Self, EncoderPreparationError> {
        let prepared = PreparedComponent::prepare_physical_recipes(
            "ear/encoder/leased-units",
            declaration,
            source,
            parameters,
            Some(ComponentRecipeConversion {
                dtype: RecipeDtype::F32,
                from: &[RecipeDtype::F16, RecipeDtype::BF16],
            }),
            |name| EncoderPreparationError::MissingParameter(name.into()),
        )?;
        Ok(Self { prepared })
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), EncoderPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, EncoderPreparationError> {
        crate::ear_encoder_layered::declared_local_dimensions(
            self.prepared.declaration().spec(),
            layout,
        )?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Retains only the exact sources and canonical units owned by one pipeline stage.
    pub fn into_partition<G>(
        self,
        partition: &eredu_runtime::ArchitecturePartition<
            G,
            crate::ear_encoder_partition::EncoderBoundarySchema,
        >,
    ) -> Result<PreparedEncoderPartitionBindings, EncoderPreparationError> {
        crate::ear_encoder_partition::validate_encoder_partition(
            self.prepared.declaration(),
            self.prepared.local_layout(),
            partition,
        )?;
        let prepared = self.prepared.select_units(partition.units())?;
        Ok(PreparedEncoderPartitionBindings { prepared })
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated neural encoder geometry.
    pub fn config(&self) -> &EncoderConfig {
        self.prepared.declaration().config()
    }
    /// Canonical parameter namespace retained from preparation.
    pub fn prefix(&self) -> &str {
        self.prepared.declaration().prefix()
    }

    /// Exact prepared unit bindings for cold resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }

    /// Validated semantic definition retained through source and placement selection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        crate::declarations::EncoderDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (definition, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (definition, local, layout, source, units)
    }
}

/// Canonically addressed finite encoder units and their restricted retained sources.
/// This selection has no persistent cache allocation.
pub struct PreparedEncoderPartitionBindings {
    prepared: PreparedComponent<crate::declarations::EncoderDeclaration>,
}
impl PreparedEncoderPartitionBindings {
    /// Retained typed construction and execution declaration.
    pub fn declaration(&self) -> &crate::declarations::EncoderDeclaration {
        self.prepared.declaration()
    }

    /// Global finite encoder geometry.
    pub fn config(&self) -> &EncoderConfig {
        self.prepared.declaration().config()
    }
    /// Exact canonical parameter namespace.
    pub fn prefix(&self) -> &str {
        self.prepared.declaration().prefix()
    }
    /// Complete definition retained independently of the selected pipeline units.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration().definition()
    }
    /// Global TP declaration retained independently of selected pipeline units.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }
    /// Source authority limited to owned recipes.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Global unit addresses and exact materialization recipes.
    pub fn unit_bindings(
        &self,
    ) -> impl ExactSizeIterator<Item = (eredu_runtime::ExecutionUnitAddress, &[WeightBinding])>
    {
        self.prepared.addressed_unit_bindings()
    }
    /// Consumes the retained stage selection without reopening artifacts.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        crate::declarations::EncoderDeclaration,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
    ) {
        let (definition, local_layout, invocations, source, units) = self
            .prepared
            .into_addressed_unit_parts()
            .expect("partition declares only independently leased units");
        (definition, local_layout, invocations, source, units)
    }
}
