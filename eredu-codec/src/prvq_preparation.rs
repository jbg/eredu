//! Cold exact binding plans for independently leased PRVQ codebook units.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeDtype, RecipeError},
    store::{RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    bindings_from_recipe_set, place_weight_bindings, BindingPlacementError, ExecutionUnitLayout,
    LocalModelLayout, ParameterGroupOwner, RecipeBindingError, WeightBinding,
};

use crate::{prvq::Geometry, prvq_layered::quantizer_parameter_description};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum QuantizerPreparationError {
    /// A retained communicator lacks required membership, completion or operations.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// A communication requirement is malformed.
    #[error(transparent)]
    Communication(#[from] eredu_runtime::CommunicationManifestError),

    /// Quantizer geometry cannot be constructed.
    #[error(transparent)]
    Quantizer(#[from] crate::Error),
    /// A required parameter is missing from the selected component.
    #[error("PRVQ parameter {0} is missing")]
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
    #[error("invalid PRVQ tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("PRVQ parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for each canonical codebook mean table and retained variance.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared checkpoint recipe machinery. This component
/// contract declares no waveform or conversational execution capability.
pub struct PreparedQuantizerBindings {
    geometry: Geometry,
    prefix: String,
    layout: ExecutionUnitLayout,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedQuantizerBindings {
    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring component payloads are excluded.
    pub fn from_recipes(
        geometry: Geometry,
        prefix: &str,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    ) -> Result<Self, QuantizerPreparationError> {
        let description = quantizer_parameter_description(geometry, prefix)?;
        let layout = description.unit_layout().clone();
        let mut unit_recipes = vec![BTreeMap::new(); layout.len()];
        for owned in description.groups() {
            let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owned.owner()
            else {
                return Err(QuantizerPreparationError::Geometry(
                    "unexpected static quantizer owner".into(),
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
                    .ok_or_else(|| QuantizerPreparationError::MissingParameter(name.into()))?;
                let metadata = recipe.infer(source.as_ref())?;
                if metadata.shape != member.global_shape() {
                    return Err(QuantizerPreparationError::Geometry(name.into()));
                }
                if matches!(metadata.dtype, RecipeDtype::F16 | RecipeDtype::BF16) {
                    recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::Cast {
                        input: Box::new(recipe),
                        dtype: RecipeDtype::F32,
                    };
                }
                if recipe.infer(source.as_ref())?.dtype != RecipeDtype::F32 {
                    return Err(QuantizerPreparationError::Geometry(name.into()));
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
            "prvq/codebooks/leased-units",
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
                            return Err(QuantizerPreparationError::Geometry(target));
                        }
                        binding
                            .with_logical_target(target)
                            .map_err(RecipeBindingError::from)
                            .map_err(QuantizerPreparationError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, QuantizerPreparationError>>()?;
        Ok(Self {
            geometry,
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
    ) -> Result<(Self, LocalModelLayout), QuantizerPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(QuantizerPreparationError::Parallel(
                "PRVQ component selection requires a tensor-only topology".into(),
            ));
        }
        let description = quantizer_parameter_description(self.geometry, &self.prefix)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(QuantizerPreparationError::Parallel)?;
        if let Some(selected) = self.local_layout.as_ref() {
            if selected != &layout {
                return Err(QuantizerPreparationError::Parallel(
                    "retained PRVQ placement differs from the tensor topology".into(),
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
    ) -> Result<Self, QuantizerPreparationError> {
        if self.local_layout.is_some() {
            return Err(QuantizerPreparationError::Parallel(
                "PRVQ bindings already have a tensor partition".into(),
            ));
        }
        crate::prvq_layered::quantizer_local_dimensions(self.geometry, &self.prefix, layout)?;
        self.units = self
            .units
            .into_iter()
            .map(|unit| place_weight_bindings(unit, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.local_layout = Some(layout.clone());
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }

    /// Validated neural quantizer geometry.
    pub fn geometry(&self) -> Geometry {
        self.geometry
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
        Geometry,
        String,
        Option<LocalModelLayout>,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        (
            self.geometry,
            self.prefix,
            self.local_layout,
            self.layout,
            self.source,
            self.units,
        )
    }
}

/// Admits exact PRVQ sums and local failure agreement before native parameter reads.
/// One-row limits are the minimum viable component; the retained communicator
/// validates every requested batch/frame extent before each actual collective.
pub fn validate_parallel_communication(
    geometry: Geometry,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), QuantizerPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        replicated_execution::validate_collective_coordination, CommunicationOperation,
        CommunicationOperationRequirement, CommunicationTensorLimits,
    };
    geometry.validate()?;
    let selected = validate_collective_coordination(
        manifest,
        group,
        [CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(
                1,
                3,
                geometry.channels.max(geometry.cardinality) as usize,
                None,
            )?,
            true,
        )?],
    )?;
    if selected.members().len() > geometry.channels as usize {
        return Err(QuantizerPreparationError::Parallel(
            "PRVQ tensor degree exceeds channels".into(),
        ));
    }
    Ok(())
}
