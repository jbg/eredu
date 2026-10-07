//! Cold exact binding plans for independently leased EAR neural encoder units.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeDtype, RecipeError},
    store::{RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    bindings_from_recipe_set, place_weight_bindings, BindingPlacementError, ExecutionUnitLayout,
    LocalModelLayout, ParameterGroupOwner, RecipeBindingError, WeightBinding,
};

use crate::{ear_encoder::EncoderConfig, ear_encoder_layered::encoder_parameter_description};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum EncoderPreparationError {
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
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
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
        let layout = description.unit_layout().clone();
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(EncoderPreparationError::Geometry(
                    "unexpected static encoder owner".into(),
                ));
            };
            let group = description
                .graph()
                .group_index(group.as_str())
                .expect("validated parameter ownership");
            let ordinal = layout
                .ordinal(group, *global_unit)
                .expect("validated parameter unit");
            for member in owned.members() {
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
                unit_recipes[ordinal].insert(name.into(), recipe);
            }
        }
        let inputs = unit_recipes
            .iter()
            .flat_map(|unit| unit.values())
            .flat_map(|recipe| recipe.source_keys().into_iter().map(str::to_owned))
            .collect();
        let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
            source,
            "ear/encoder/leased-units",
            inputs,
        )?);
        let units = unit_recipes
            .into_iter()
            .map(|recipes| {
                let targets = recipes
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>();
                let recipes = AtomicRecipeSet::new(source.as_ref(), recipes)?;
                bindings_from_recipe_set(source.as_ref(), recipes)?
                    .into_iter()
                    .map(|binding| {
                        let target = binding.name().to_owned();
                        if !targets.contains(&target) {
                            return Err(EncoderPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(EncoderPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, EncoderPreparationError>>()?;
        Ok(Self {
            config,
            prefix: prefix.into(),
            layout,
            source,
            units,
            local_layout: None,
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
        if let Some(selected) = self.local_layout.as_ref() {
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
        if self.local_layout.is_some() {
            return Err(EncoderPreparationError::Parallel(
                "EAR encoder bindings already have a tensor partition".into(),
            ));
        }
        crate::ear_encoder_layered::encoder_local_dimensions(&self.config, &self.prefix, layout)?;
        self.units = self
            .units
            .into_iter()
            .map(|unit| place_weight_bindings(unit, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.local_layout = Some(layout.clone());
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
            self.local_layout.as_ref(),
            partition,
        )?;
        let selected = partition
            .units()
            .map(|address| {
                self.layout
                    .ordinal(address.group(), address.index())
                    .map(|ordinal| (ordinal, address))
                    .ok_or_else(|| {
                        EncoderPreparationError::Geometry("partition unit outside encoder".into())
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let units = self
            .units
            .into_iter()
            .enumerate()
            .filter_map(|(ordinal, bindings)| {
                selected
                    .get(&ordinal)
                    .copied()
                    .map(|address| (address, bindings))
            })
            .collect::<Vec<_>>();
        let inputs = units
            .iter()
            .flat_map(|(_, bindings)| bindings)
            .flat_map(|binding| {
                binding
                    .source_recipe()
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect();
        let source = Arc::new(RestrictedCheckpointSource::including(
            self.source,
            "ear/encoder/partition-units",
            inputs,
        )?);
        Ok(PreparedEncoderPartitionBindings {
            config: self.config,
            prefix: self.prefix,
            local_layout: self.local_layout,
            source,
            units,
        })
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
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
        &self.units
    }

    /// Restricted retained source used by these exact bindings.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
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
        (
            self.config,
            self.prefix,
            self.local_layout,
            self.layout,
            self.source,
            self.units,
        )
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
