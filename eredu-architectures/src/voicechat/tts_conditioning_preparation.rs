//! Cold exact binding plans for independently leased TTS conditioning units.

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
    tts_conditioning_layered::{tts_conditioning_component_definition, ConditioningDimensions},
    tts_mog::TtsError,
};

/// Invalid component geometry or exact retained parameter contract.
#[derive(Debug, thiserror::Error)]
pub enum ConditioningPreparationError {
    /// Shared exact preparation failed.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
    /// Conditioning geometry cannot be constructed.
    #[error(transparent)]
    Conditioning(#[from] TtsError),
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
    /// An admitted parameter disagrees with architecture or recipe geometry.
    #[error("VoiceChat TTS conditioning parameter {0} has inconsistent geometry")]
    Geometry(String),
}

/// Cold bindings for character embeddings, ingress parameters, and gated fusion.
///
/// This reads no payload and constructs no native module. Only the admitted
/// physical inputs used by these units remain visible. Execution F32 casts and
/// conversion bounds use the shared speech recipe projection. This component
/// contract declares no full VoiceChat execution capability.
pub struct PreparedConditioningBindings {
    dimensions: ConditioningDimensions,
    prepared: PreparedComponent<eredu_runtime::ComponentDefinition>,
}

impl PreparedConditioningBindings {
    /// Projects exact conditioning bindings from selected retained artifacts.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, ConditioningPreparationError> {
        let source = artifacts
            .component_source(Component::Tts)
            .ok_or(SpeechParameterError::MissingComponent(Component::Tts))?;
        let dimension = |name: &str, axis: usize| -> Result<i32, ConditioningPreparationError> {
            let parameter = artifacts.plan().parameters().get(name).ok_or_else(|| {
                SpeechParameterError::MissingParameter {
                    component: Component::Tts,
                    name: name.into(),
                }
            })?;
            let value = parameter
                .output
                .shape
                .get(axis)
                .copied()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| ConditioningPreparationError::Geometry(name.into()))?;
            Ok(value)
        };
        let characters = dimension("tts_model.tts_model.embed_subword.embed_tokens.weight", 0)?
            .checked_sub(1)
            .ok_or_else(|| {
                ConditioningPreparationError::Geometry("character padding row".into())
            })?;
        let speaker_frames = dimension("tts_model.audio_prompt_latents.Aria", 1)?;
        Self::from_parameters(
            ConditioningDimensions::from_config(
                &artifacts.plan().config().tts,
                characters,
                speaker_frames,
            )?,
            source.clone(),
            artifacts.plan().parameters(),
        )
    }

    /// Projects explicit low-level admitted component recipes without artifact
    /// discovery or native allocation. Shapes follow the symbolic architecture;
    /// neighboring TTS component payloads are excluded.
    pub fn from_parameters(
        dimensions: ConditioningDimensions,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, ConditioningPreparationError> {
        let definition = tts_conditioning_component_definition(dimensions)?;
        let prepared = PreparedComponent::prepare(
            "voicechat/tts_conditioning/leased-units",
            definition,
            source.clone(),
            |member| {
                let name = member.target();
                let parameter = admitted_member(parameters, &source, Component::Tts, member)?;
                let parameter = execution_parameter(parameter, &source)?;
                if parameter.output.dtype != RecipeDtype::F32 {
                    return Err(ConditioningPreparationError::Geometry(name.into()));
                }
                Ok::<_, ConditioningPreparationError>(parameter.recipe)
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
    ) -> Result<(Self, LocalModelLayout), ConditioningPreparationError> {
        let layout = self.prepared.tensor_parallel_layout(topology)?;
        Ok((self.into_parallel(&layout)?, layout))
    }

    /// Lowers each canonical binding through the selected shared physical layout.
    pub fn into_parallel(
        mut self,
        layout: &LocalModelLayout,
    ) -> Result<Self, ConditioningPreparationError> {
        super::tts_conditioning_layered::conditioning_local_dimensions(self.dimensions, layout)?;
        self.prepared = self.prepared.select(layout)?;
        Ok(self)
    }

    /// Exact retained tensor placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.prepared.local_layout()
    }

    /// Validated conditioning geometry; no other speech stack is selected.
    pub fn dimensions(&self) -> ConditioningDimensions {
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

    /// Complete semantic declaration retained through precision and placement selection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        self.prepared.declaration()
    }

    /// Consumes exact source authority and the canonical ordered binding plan.
    pub fn into_parts(
        self,
    ) -> (
        ConditioningDimensions,
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
        (self.dimensions, definition, local, layout, source, units)
    }
}
