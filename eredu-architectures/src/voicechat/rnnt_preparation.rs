//! Exact cold recipes for independently leased RNNT prediction and joint units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeError},
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    BindingPlacementError, ComponentPreparationError, ComponentRecipe, ComponentRecipeConversion,
    ExecutionUnitLayout, LocalModelLayout, PreparedComponent, RecipeBindingError, WeightBinding,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::SpeechParameterError,
    rnnt::{RnntDimensions, RnntError},
    rnnt_layered::rnnt_parameter_description,
};

/// Invalid RNNT geometry, retained source authority, or execution recipe.
#[derive(Debug, thiserror::Error)]
pub enum RnntPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Architecture geometry cannot be constructed.
    #[error(transparent)]
    Rnnt(#[from] RnntError),
    /// A required retained component or parameter is absent.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// The restricted source could not be retained.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// An exact recipe disagrees with source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A validated recipe cannot be represented by a runtime binding.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// A recipe cannot realize the selected physical tensor placement.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Tensor topology or a repeated placement selection is inconsistent.
    #[error("invalid RNNT tensor partition: {0}")]
    Parallel(String),
    /// The selected communication group cannot cover an exact neural operation.
    #[error(transparent)]
    Communication(#[from] eredu_runtime::CommunicationCapabilityError),
    /// A communication requirement or descriptor is malformed.
    #[error(transparent)]
    Manifest(#[from] eredu_runtime::CommunicationManifestError),
    /// A required canonical parameter has no recipe.
    #[error("VoiceChat RNNT parameter {0} is missing")]
    MissingParameter(String),
    /// A parameter disagrees with its architecture or admitted metadata.
    #[error("VoiceChat RNNT parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Restricted checkpoint authority and F32 recipes in architecture unit order.
///
/// Preparation reads metadata only. Learned parameters are distinct from recurrent
/// state and can be materialized independently through shared residency providers.
pub struct PreparedRnntBindings {
    dimensions: RnntDimensions,
    pub(super) prepared: PreparedComponent,
}
impl PreparedRnntBindings {
    /// Projects the admitted component without resolving or reopening artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, RnntPreparationError> {
        let source = artifacts
            .component_source(Component::Rnnt)
            .ok_or(SpeechParameterError::MissingComponent(Component::Rnnt))?;
        Self::from_parameters(
            RnntDimensions::from_config(&artifacts.plan().config().rnnt)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Checks admitted component metadata before normalizing execution recipes.
    pub fn from_parameters(
        dimensions: RnntDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, RnntPreparationError> {
        let description = rnnt_parameter_description(dimensions)?;
        let mut recipes = BTreeMap::new();
        for owned in description.groups() {
            for member in owned.members() {
                let name = member.target();
                let parameter = parameters
                    .get(name)
                    .filter(|parameter| parameter.component == Component::Rnnt)
                    .ok_or_else(|| SpeechParameterError::MissingParameter {
                        component: Component::Rnnt,
                        name: name.into(),
                    })?;
                if parameter.output.shape != member.global_shape()
                    || parameter.recipe.infer(source.as_ref())? != parameter.output
                {
                    return Err(RnntPreparationError::Geometry(name.into()));
                }
                recipes.insert(name.into(), parameter.recipe.clone());
            }
        }
        Self::from_recipes(dimensions, source, &recipes)
    }

    /// Selects exact component recipes from a low-level retained source.
    /// F16/BF16 recipes use bounded F32 casts; integer execution weights fail cold.
    pub fn from_recipes(
        dimensions: RnntDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, DerivedWeightRecipe>,
    ) -> Result<Self, RnntPreparationError> {
        let description = rnnt_parameter_description(dimensions)?;
        let prepared = PreparedComponent::prepare(
            "voicechat/rnnt/leased-units",
            description,
            source,
            |member| {
                let name = member.target();
                let recipe = parameters
                    .get(name)
                    .cloned()
                    .ok_or_else(|| RnntPreparationError::MissingParameter(name.into()))?;
                Ok::<_, RnntPreparationError>(ComponentRecipe::Physical {
                    recipe,
                    conversion: Some(ComponentRecipeConversion {
                        dtype: RecipeDtype::F32,
                        from: &[RecipeDtype::F16, RecipeDtype::BF16],
                    }),
                })
            },
        )?;
        Ok(Self {
            dimensions,
            prepared,
        })
    }

    /// Selects vocabulary rows while retaining replicated recurrent and joint
    /// parameters. Repeated selection validates the exact retained topology.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RnntPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(RnntPreparationError::Parallel(
                "RNNT component selection requires a tensor-only topology".into(),
            ));
        }
        let description = rnnt_parameter_description(self.dimensions)?;
        let layout = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(RnntPreparationError::Parallel)?;
        if let Some(selected) = self.prepared.local_layout() {
            if selected != &layout {
                return Err(RnntPreparationError::Parallel(
                    "retained RNNT placement differs from the tensor topology".into(),
                ));
            }
            return Ok((self, layout));
        }
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Applies exact rank-local recipes without expanding retained source authority.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntPreparationError> {
        super::rnnt_layered::rnnt_local_dimensions(self.dimensions, layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained vocabulary placement, if tensor parallelism was selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Exact prediction and joint geometry retained during preparation.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Canonical unit bindings for selected-task resource accounting.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
    }
    /// Restricted retained source used by every execution recipe.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Transfers source authority and architecture-ordered unit bindings.
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
            .expect("component declares only independently leased units");
        (self.dimensions, local, layout, source, units)
    }
}

/// Validates RNNT's exact logical tensor and decision communication before any
/// parameter materialization. Backend-private gather padding is not a second
/// architecture operation; its storage is owned by the selected gather mechanism.
pub fn validate_parallel_communication(
    dimensions: RnntDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), RnntPreparationError> {
    validate_neural_communication(dimensions, manifest, group, true)
}

/// Shared exact neural admission; projection-only owners omit greedy decisions.
pub(crate) fn validate_neural_communication(
    dimensions: RnntDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
    include_decisions: bool,
) -> Result<(), RnntPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        CommunicationCapabilities, CommunicationOperation as Op,
        CommunicationOperationRequirement as Requirement, CommunicationTensorLimits,
    };
    dimensions.validate()?;
    let selected = manifest
        .groups()
        .iter()
        .find(|item| item.id() == group)
        .ok_or_else(|| {
            RnntPreparationError::Parallel("RNNT communication group is absent".into())
        })?;
    if selected.local_index().is_none() {
        return Err(RnntPreparationError::Parallel(
            "local rank is not an RNNT communication member".into(),
        ));
    }
    if manifest.completion_policy().is_none() {
        return Err(RnntPreparationError::Parallel(
            "RNNT communication requires bounded completion".into(),
        ));
    }
    let vocabulary = (dimensions.vocab_size + 1) as usize;
    let degree = selected.members().len();
    if degree > vocabulary {
        return Err(RnntPreparationError::Parallel(
            "RNNT tensor degree exceeds vocabulary rows".into(),
        ));
    }
    let rows = vocabulary.div_ceil(degree);
    let width = dimensions.predictor_width as usize;
    let admitted =
        CommunicationCapabilities::new(selected.requirements().operations().iter().cloned())?;
    if include_decisions {
        admitted.validate_requirement(&Requirement::failure_agreement(true))?;
        admitted.validate_requirement(&Requirement::tensors(
            Op::Broadcast,
            [TensorDtype::I32],
            CommunicationTensorLimits::new(1, 1, 1, None)?,
            true,
        )?)?;
    }
    for requirement in [
        Requirement::tensors(
            Op::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 2, width, None)?,
            true,
        )?,
        Requirement::tensors(
            Op::AllGatherUneven,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 2, rows, None)?
                .with_output_tensor_elements(vocabulary)?,
            true,
        )?,
    ] {
        admitted.validate_requirement(&requirement)?;
    }
    Ok(())
}
