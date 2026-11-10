//! Exact learned-parameter units projected from retained VoiceChat artifacts.
//!
//! A unit projects a semantic parameter declaration. Conformer, Gemma, MoG and
//! codec blocks can be bound independently. This contract does not select a
//! residency schedule or construct a realtime executable.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeError},
    store::{SharedCheckpointSource, StoreError},
};
use eredu_nn::Parameterized;
use eredu_runtime::{
    bind_materialized_unit, materialize_selected_bindings, select_bindings,
    ArchitectureParameterDescription, ComponentPreparationError, ParameterBackend,
    ParameterMemberSpec, ParameterOrchestrationError, PreparedComponent, RecipeBindingError,
    SelectedBindingPlan, WeightBinding,
};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
};

/// Failure while projecting a learned module onto the retained source contract.
#[derive(Debug, thiserror::Error)]
pub enum SpeechParameterError {
    /// The task did not retain the requested component.
    #[error("VoiceChat component {0:?} was not retained")]
    MissingComponent(Component),
    /// Backbone names use the existing Nemotron projection; controls are host data.
    #[error("VoiceChat component {0:?} does not use speech neural parameter units")]
    ComponentRole(Component),
    /// A module requests a canonical identity absent from the selected component.
    #[error("VoiceChat component {component:?} has no learned parameter {name}")]
    MissingParameter {
        /// Selected component.
        component: Component,
        /// Module's exact stable identity.
        name: String,
    },
    /// A declared member disagrees with its retained identity or recipe output metadata.
    #[error("VoiceChat component {component:?} parameter {name} disagrees with admitted metadata")]
    InconsistentAdmission {
        /// Selected source component.
        component: Component,
        /// Declared canonical identity.
        name: String,
    },
    /// Host preprocessing and integer policy buffers cannot become neural weights.
    #[error("VoiceChat parameter {0} is owned by host preparation")]
    HostParameter(String),
    /// Exact source projection failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// Canonical recipes cannot be represented by the retained source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Generic runtime binding declarations are inconsistent.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// A semantic declaration disagrees with retained recipes or geometry.
    #[error(transparent)]
    Preparation(#[from] ComponentPreparationError),
}

/// One exact module's canonical recipes and restricted physical source authority.
///
/// Preparation reads metadata only. Source identities, transforms, conversion
/// bounds and backing leases are retained from artifact admission. Perception,
/// TTS and codec half-precision outputs receive an explicit F32 execution cast;
/// its input/output overlap is included in the materialization bound. Host
/// buffers and neighboring blocks are absent from the resulting source view.
pub struct PreparedSpeechParameterUnit {
    component: Component,
    source: SharedCheckpointSource,
    parameters: BTreeMap<String, ParameterAdmission>,
    bindings: Vec<WeightBinding>,
}

impl PreparedSpeechParameterUnit {
    pub(super) fn from_description(
        artifacts: &RetainedVoiceChatArtifacts,
        component: Component,
        description: ArchitectureParameterDescription,
    ) -> Result<Self, SpeechParameterError> {
        check_component(component)?;
        let source = artifacts
            .component_source(component)
            .ok_or(SpeechParameterError::MissingComponent(component))?
            .clone();
        project_description(
            artifacts.plan().parameters(),
            source,
            component,
            description,
        )
    }

    /// Architecture component retaining this unit's physical source.
    pub const fn component(&self) -> Component {
        self.component
    }

    /// Only the physical recipe inputs needed by the selected module.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }

    /// Exact canonical identities, original provenance and conversion bounds.
    pub fn parameters(&self) -> &BTreeMap<String, ParameterAdmission> {
        &self.parameters
    }

    /// Runtime bindings, including explicit shared-storage aliases.
    pub fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }

    /// Admits backend mechanisms before any weight payload is read.
    pub fn select<B: ParameterBackend>(
        self,
    ) -> Result<SelectedSpeechParameterUnit<B>, ParameterOrchestrationError<B::ParameterError>>
    {
        Ok(SelectedSpeechParameterUnit {
            selected: select_bindings::<B>(self.source, self.bindings)?,
        })
    }
}

/// A unit whose exact source and recipes passed generic backend admission.
pub struct SelectedSpeechParameterUnit<B: ParameterBackend> {
    selected: SelectedBindingPlan<B>,
}

impl<B: ParameterBackend> SelectedSpeechParameterUnit<B> {
    /// Materializes through exact completion and atomically replaces the module's
    /// matching slots. A mismatched or incomplete module receives no replacements.
    pub fn bind<M: Parameterized<B::Parameter>>(
        self,
        module: &mut M,
        context: &B::MaterializationContext,
    ) -> Result<(), ParameterOrchestrationError<B::ParameterError>> {
        let materialized = materialize_selected_bindings::<B>(self.selected, context)?;
        bind_materialized_unit::<B, _>(module, materialized)
    }
}

fn check_component(component: Component) -> Result<(), SpeechParameterError> {
    if matches!(component, Component::Backbone | Component::Controls) {
        return Err(SpeechParameterError::ComponentRole(component));
    }
    Ok(())
}

fn host_parameter(name: &str) -> bool {
    name.starts_with("stt_model.perception.preprocessor.")
        || super::controls::VOICECHAT_INTEGER_BUFFERS
            .iter()
            .any(|buffer| buffer.key == name)
}

/// Integer scale storage belongs to a declared scale companion of an admitted MXFP4 primary.
pub(super) fn is_mxfp4_scale(
    parameters: &BTreeMap<String, ParameterAdmission>,
    member: &ParameterMemberSpec,
) -> bool {
    if member.linear_companion() != Some(eredu_nn::LinearCompanionRole::Scale) {
        return false;
    }
    let Some(primary_id) = member.linear_companion_of() else {
        return false;
    };
    let (Some(primary), Some(companion)) = (
        parameters.get(primary_id.as_str()),
        parameters.get(member.target()),
    ) else {
        return false;
    };
    primary.name == primary_id.as_str()
        && companion.name == member.target()
        && primary.component == companion.component
        && primary.linear_format == eredu_checkpoint::LinearFormat::MxFp4
}

pub(super) fn execution_parameter(
    parameter: &ParameterAdmission,
    source: &SharedCheckpointSource,
) -> Result<ParameterAdmission, RecipeError> {
    let mut parameter = parameter.clone();
    // Sensitive backbone normalization, convolution and recurrent parameters
    // accumulate in F32. Projection matrices retain their admitted format.
    // RNNT consumes its joint parameter dtype; other learned speech components
    // require F32 at their boundary before the selected projection precision.
    let requires_float32 = (parameter.component == Component::Backbone
        && parameter.output.shape.len() != 2)
        || matches!(
            parameter.component,
            Component::Perception
                | Component::Tts
                | Component::CodecEncoder
                | Component::CodecDecoder
                | Component::CodecQuantizer
        );
    if requires_float32 && matches!(parameter.output.dtype, RecipeDtype::F16 | RecipeDtype::BF16) {
        parameter.recipe = DerivedWeightRecipe::Cast {
            input: Box::new(parameter.recipe),
            dtype: RecipeDtype::F32,
        };
        parameter.output = parameter.recipe.infer(source.as_ref())?;
        // The converter workspace describes physical GGUF decoding and stays
        // separate from the cast's recipe temporaries; neither replaces the
        // retained encoded storage charge.
        parameter.materialization_peak_bytes = parameter
            .recipe
            .peak_materialization_bytes(source.as_ref())?
            .checked_add(parameter.conversion_workspace_bytes)
            .and_then(|bytes| bytes.checked_add(parameter.stored_bytes))
            .ok_or(RecipeError::ArithmeticOverflow(
                "VoiceChat speech execution parameter peak bytes",
            ))?;
    }
    Ok(parameter)
}

fn project_description(
    admitted: &BTreeMap<String, ParameterAdmission>,
    source: SharedCheckpointSource,
    component: Component,
    description: ArchitectureParameterDescription,
) -> Result<PreparedSpeechParameterUnit, SpeechParameterError> {
    check_component(component)?;
    let mut parameters = BTreeMap::new();
    let prepared = PreparedComponent::prepare(
        format!("voicechat/{component:?}/module"),
        description,
        source.clone(),
        |member| -> Result<_, SpeechParameterError> {
            let name = member.target();
            let parameter = admitted_parameter(admitted, &source, component, name)?;
            let recipe = parameter.recipe.clone();
            parameters.insert(name.to_owned(), parameter);
            Ok(recipe)
        },
    )?;
    let (_, _, _, source, units) = prepared.into_declared_unit_parts()?;
    let mut bindings = units.into_iter().flatten().collect::<Vec<_>>();
    bindings.sort_by(|a, b| a.name().cmp(b.name()));
    Ok(PreparedSpeechParameterUnit {
        component,
        source,
        parameters,
        bindings,
    })
}

fn parameter_for_component<'a>(
    admitted: &'a BTreeMap<String, ParameterAdmission>,
    component: Component,
    name: &str,
) -> Result<&'a ParameterAdmission, SpeechParameterError> {
    let parameter = admitted
        .get(name)
        .filter(|parameter| parameter.component == component)
        .ok_or_else(|| SpeechParameterError::MissingParameter {
            component,
            name: name.into(),
        })?;
    if parameter.name != name {
        return Err(SpeechParameterError::InconsistentAdmission {
            component,
            name: name.into(),
        });
    }
    Ok(parameter)
}

/// Checks retained admission before any execution conversion or accounting change.
pub(super) fn admitted_member<'a>(
    admitted: &'a BTreeMap<String, ParameterAdmission>,
    source: &SharedCheckpointSource,
    component: Component,
    member: &ParameterMemberSpec,
) -> Result<&'a ParameterAdmission, SpeechParameterError> {
    let name = member.target();
    let parameter = parameter_for_component(admitted, component, name)?;
    if parameter.output.shape != member.global_shape()
        || parameter.recipe.infer(source.as_ref())? != parameter.output
    {
        return Err(SpeechParameterError::InconsistentAdmission {
            component,
            name: name.into(),
        });
    }
    Ok(parameter)
}

fn admitted_parameter(
    admitted: &BTreeMap<String, ParameterAdmission>,
    source: &SharedCheckpointSource,
    component: Component,
    name: &str,
) -> Result<ParameterAdmission, SpeechParameterError> {
    if host_parameter(name) {
        return Err(SpeechParameterError::HostParameter(name.into()));
    }
    let parameter = parameter_for_component(admitted, component, name)?;
    Ok(execution_parameter(parameter, source)?)
}
