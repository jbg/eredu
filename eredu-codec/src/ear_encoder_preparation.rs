//! Cold exact binding plans for independently leased EAR neural encoder units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{RecipeDtype, RecipeError},
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    BindingPlacementError, ComponentPreparationError, ExecutionUnitLayout, LocalModelLayout,
    PreparedComponent, RecipeBindingError, WeightBinding,
};

use crate::{ear_encoder::EncoderConfig, ear_encoder_layered::encoder_parameter_description};

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
    /// Source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// A recipe disagrees with its exact source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact recipe binding cannot be represented.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// Shared physical placement cannot be applied to an exact recipe.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Tensor topology or retained placement is inconsistent.
    #[error("invalid EAR encoder tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("EAR encoder parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for each input-projection, ConvNeXt and downsampling unit.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no waveform or conversational execution capability.
pub struct PreparedEncoderBindings {
    config: EncoderConfig,
    prefix: String,
    prepared: PreparedComponent,
}

impl PreparedEncoderBindings {
    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_recipes(
        config: EncoderConfig,
        prefix: &str,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> Result<Self, EncoderPreparationError> {
        let description = encoder_parameter_description(&config, prefix)?;
        let prepared = PreparedComponent::prepare(
            "ear/encoder/leased-units",
            description,
            source.clone(),
            |member| {
                let name = member.target();
                let mut recipe = parameters
                    .get(name)
                    .cloned()
                    .ok_or_else(|| EncoderPreparationError::MissingParameter(name.into()))?;
                let metadata = recipe.infer(source.as_ref())?;
                if metadata.shape != member.global_shape() {
                    return Err(EncoderPreparationError::Geometry(name.into()));
                }
                if matches!(metadata.dtype, RecipeDtype::F16 | RecipeDtype::BF16) {
                    recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::Cast {
                        input: Box::new(recipe),
                        dtype: RecipeDtype::F32,
                    };
                }
                if recipe.infer(source.as_ref())?.dtype != RecipeDtype::F32 {
                    return Err(EncoderPreparationError::Geometry(name.into()));
                }
                Ok::<_, EncoderPreparationError>(recipe)
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
    ) -> Result<(Self, LocalModelLayout), EncoderPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(EncoderPreparationError::Parallel(
                "EAR encoder component selection requires a tensor-only topology".into(),
            ));
        }
        let description = encoder_parameter_description(&self.config, &self.prefix)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(EncoderPreparationError::Parallel)?;
        if let Some(selected) = self.prepared.local_layout() {
            if selected != &layout {
                return Err(EncoderPreparationError::Parallel(
                    "retained EAR encoder placement differs from the tensor topology".into(),
                ));
            }
            return Ok((self, layout));
        }
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, EncoderPreparationError> {
        crate::ear_encoder_layered::encoder_local_dimensions(&self.config, &self.prefix, layout)?;
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
            &self.config,
            &self.prefix,
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
        Ok(PreparedEncoderPartitionBindings {
            config: self.config,
            prefix: self.prefix,
            local_layout,
            source,
            units,
        })
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated neural encoder geometry.
    pub fn config(&self) -> &EncoderConfig {
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

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        EncoderConfig,
        String,
        Option<LocalModelLayout>,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (local, layout, source, units) = self
            .prepared
            .into_unit_parts()
            .expect("component declares only independently leased units");
        (self.config, self.prefix, local, layout, source, units)
    }
}

/// Canonically addressed finite encoder units and their restricted retained sources.
/// This selection has no persistent cache allocation.
pub struct PreparedEncoderPartitionBindings {
    config: EncoderConfig,
    prefix: String,
    local_layout: Option<LocalModelLayout>,
    source: SharedCheckpointSource,
    units: Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
}
impl PreparedEncoderPartitionBindings {
    /// Global finite encoder geometry.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
    /// Exact canonical parameter namespace.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    /// Global TP declaration retained independently of selected pipeline units.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }
    /// Source authority limited to owned recipes.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Global unit addresses and exact materialization recipes.
    pub fn unit_bindings(&self) -> &[(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)] {
        &self.units
    }
    /// Consumes the retained stage selection without reopening artifacts.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        EncoderConfig,
        String,
        Option<LocalModelLayout>,
        SharedCheckpointSource,
        Vec<(eredu_runtime::ExecutionUnitAddress, Vec<WeightBinding>)>,
    ) {
        (
            self.config,
            self.prefix,
            self.local_layout,
            self.source,
            self.units,
        )
    }
}
