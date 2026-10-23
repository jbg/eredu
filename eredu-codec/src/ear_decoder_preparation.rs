//! Cold exact binding plans for independently leased EAR neural decoder units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    BindingPlacementError, ComponentPreparationError, ComponentRecipe, ComponentRecipeConversion,
    LocalModelLayout, PreparedComponent, RecipeBindingError, WeightBinding,
};

use crate::{ear_decoder::DecoderConfig, ear_decoder_layered::decoder_component_definition};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum DecoderPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Decoder geometry cannot be constructed.
    #[error(transparent)]
    Decoder(#[from] crate::Error),
    /// A required parameter is missing from the selected component.
    #[error("EAR decoder parameter {0} is missing")]
    MissingParameter(String),
    /// Source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// Shared physical placement cannot be applied to an exact recipe.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Tensor topology or retained placement is inconsistent.
    #[error("invalid EAR decoder tensor partition: {0}")]
    Parallel(String),
}

/// Cold bindings for each upsampling, ConvNeXt and output-projection unit.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no waveform or conversational execution capability.
pub struct PreparedDecoderBindings {
    config: DecoderConfig,
    prefix: String,
    prepared: PreparedComponent<eredu_runtime::ComponentDefinition>,
}

impl PreparedDecoderBindings {
    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_recipes(
        config: DecoderConfig,
        prefix: &str,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> Result<Self, DecoderPreparationError> {
        let definition = decoder_component_definition(&config, prefix)?;
        let prepared =
            PreparedComponent::prepare("ear/decoder/leased-units", definition, source, |member| {
                let name = member.target();
                let recipe = parameters
                    .get(name)
                    .cloned()
                    .ok_or_else(|| DecoderPreparationError::MissingParameter(name.into()))?;
                Ok::<_, DecoderPreparationError>(ComponentRecipe::Physical {
                    recipe,
                    conversion: Some(ComponentRecipeConversion {
                        dtype: RecipeDtype::F32,
                        from: &[RecipeDtype::F16, RecipeDtype::BF16],
                    }),
                })
            },
        )?;
        Ok(Self {
            config,
            prefix: prefix.into(),
            prepared,
        })
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), DecoderPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(DecoderPreparationError::Parallel(
                "EAR decoder component selection requires a tensor-only topology".into(),
            ));
        }
        let layout = self.prepared.parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, DecoderPreparationError> {
        crate::ear_decoder_layered::decoder_local_dimensions(&self.config, &self.prefix, layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Retains only the exact sources and canonical units owned by one pipeline stage.
    pub fn into_partition<G>(
        self,
        partition: &eredu_runtime::ArchitecturePartition<
            G,
            crate::ear_decoder_partition::DecoderBoundarySchema,
        >,
    ) -> Result<PreparedDecoderPartitionBindings, DecoderPreparationError> {
        crate::ear_decoder_partition::validate_decoder_partition(
            self.prepared.declaration(),
            &self.config,
            &self.prefix,
            self.prepared.local_layout(),
            partition,
        )?;
        let prepared = self.prepared.select_units(partition.units())?;
        Ok(PreparedDecoderPartitionBindings {
            config: self.config,
            prefix: self.prefix,
            prepared,
        })
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated neural decoder geometry.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }
    /// Canonical parameter namespace retained from preparation.
    pub fn prefix(&self) -> &str {
        &self.prefix
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
        self.prepared.declaration()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        DecoderConfig,
        String,
        eredu_runtime::ComponentDefinition,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (definition, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("component declares only independently leased units");
        (
            self.config,
            self.prefix,
            definition,
            local,
            layout,
            source,
            units,
        )
    }
}

/// Canonically addressed streaming decoder units and their restricted retained sources.
/// Mutable convolution state is selected separately by the architecture partition.
pub struct PreparedDecoderPartitionBindings {
    config: DecoderConfig,
    prefix: String,
    prepared: PreparedComponent<eredu_runtime::ComponentDefinition>,
}
impl PreparedDecoderPartitionBindings {
    /// Global streaming decoder geometry.
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }
    /// Exact canonical parameter namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    /// Complete definition retained independently of the selected pipeline units.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration()
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
        DecoderConfig,
        String,
        eredu_runtime::ComponentDefinition,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
    ) {
        let (definition, local_layout, invocations, source, units) = self
            .prepared
            .into_addressed_unit_parts()
            .expect("partition declares only independently leased units");
        (
            self.config,
            self.prefix,
            definition,
            local_layout,
            invocations,
            source,
            units,
        )
    }
}
