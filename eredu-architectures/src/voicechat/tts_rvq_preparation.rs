//! Cold exact binding plans for independently leased TTS RVQ units.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{RecipeDtype, RecipeError},
    store::{SharedCheckpointSource, StoreError},
};
use eredu_runtime::{
    BindingPlacementError, ComponentPreparationError, LocalModelLayout, PreparedComponent,
    RecipeBindingError, WeightBinding,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
    parameters::{admitted_member, execution_parameter, SpeechParameterError},
    tts_mog::TtsError,
    tts_rvq::RvqDimensions,
    tts_rvq_layered::rvq_component_definition,
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum RvqPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Selected communication lacks membership, bounded completion or required operations.
    #[error(transparent)]
    Coordination(#[from] eredu_runtime::replicated_execution::CollectiveCoordinationError),
    /// An exact communication requirement is malformed.
    #[error(transparent)]
    Communication(#[from] eredu_runtime::CommunicationManifestError),
    /// RVQ geometry cannot be constructed.
    #[error(transparent)]
    Rvq(#[from] TtsError),
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
    /// Shared physical placement cannot be applied to an exact recipe.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Tensor topology or retained placement is inconsistent.
    #[error("invalid RVQ tensor partition: {0}")]
    Parallel(String),
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS RVQ parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for the stacked feedback codebook bank and shared latent projection.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedRvqBindings {
    dimensions: RvqDimensions,
    prepared: PreparedComponent<eredu_runtime::ComponentDefinition>,
}

impl PreparedRvqBindings {
    /// Projects exact RVQ bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, RvqPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        Self::from_parameters(
            RvqDimensions::from_config(&artifacts.plan().config().tts)?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        dimensions: RvqDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, RvqPreparationError> {
        let definition = rvq_component_definition(dimensions)?;
        let prepared = PreparedComponent::prepare(
            "voicechat/tts_rvq/leased-units",
            definition,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter = admitted_member(parameters, &source, Component::Tts, member)?;
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(RvqPreparationError::Geometry(name.into()));
                }
                Ok(parameter.recipe)
            },
        )?;
        Ok(Self {
            dimensions,
            prepared,
        })
    }

    /// Selects exact tensor placement or validates a retained selection without
    /// applying its recipes a second time. Source authority remains unchanged.
    pub fn select_parallel(
        self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<(Self, LocalModelLayout), RvqPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(RvqPreparationError::Parallel(
                "RVQ component selection requires a tensor-only topology".into(),
            ));
        }
        let layout = self.prepared.parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(mut self, layout: &LocalModelLayout) -> Result<Self, RvqPreparationError> {
        super::tts_rvq_layered::rvq_local_dimensions(self.dimensions, layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated RVQ geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> RvqDimensions {
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

    /// Validated semantic declaration retained by this exact preparation.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        RvqDimensions,
        eredu_runtime::ComponentDefinition,
        Option<LocalModelLayout>,
        eredu_runtime::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
    ) {
        let (definition, local, layout, source, units) = self
            .prepared
            .into_declared_unit_parts()
            .expect("RVQ declares only independently leased units");
        (self.dimensions, definition, local, layout, source, units)
    }
}

/// Validates the exact operations needed by guarded latent-sharded feedback.
/// Actual batch/time limits are checked against retained authority at execution.
pub fn validate_parallel_communication(
    dimensions: RvqDimensions,
    manifest: &eredu_runtime::CommunicationManifest,
    group: eredu_core::CollectiveGroupId,
) -> Result<(), RvqPreparationError> {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{
        CommunicationOperation, CommunicationOperationRequirement, CommunicationTensorLimits,
    };
    dimensions.validate()?;
    let selected = eredu_runtime::replicated_execution::validate_collective_coordination(
        manifest,
        group,
        [CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(
                1,
                3,
                dimensions
                    .cardinality
                    .max(dimensions.latent)
                    .max(dimensions.hidden) as usize,
                None,
            )?,
            true,
        )?],
    )?;
    if selected.members().len() > dimensions.latent as usize {
        return Err(RvqPreparationError::Parallel(
            "tensor degree exceeds latent width".into(),
        ));
    }
    Ok(())
}
