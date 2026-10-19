//! Exact learned-parameter units projected from retained VoiceChat artifacts.
//!
//! A unit follows the supplied module's ordinary parameter traversal, so a
//! Conformer, Gemma, MoG or codec block can be bound independently. This contract
//! does not select a residency schedule or construct a realtime executable.

use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, DerivedWeightRecipe, RecipeAlias, RecipeDtype, RecipeError},
    store::{RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
};
use eredu_nn::{validate_parameter_topology, ParameterTopologyError, Parameterized};
use eredu_runtime::{
    bind_materialized_unit, bindings_from_recipe_set, materialize_selected_bindings,
    select_bindings, ParameterBackend, ParameterOrchestrationError, RecipeBindingError,
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
    /// Host preprocessing and integer policy buffers cannot become neural weights.
    #[error("VoiceChat parameter {0} is owned by host preparation")]
    HostParameter(String),
    /// Module aliases must describe identical admitted recipes.
    #[error("VoiceChat alias {alias} disagrees with its owner {owner}")]
    AliasMismatch {
        /// Alias destination identity.
        alias: String,
        /// Shared storage owner identity.
        owner: String,
    },
    /// A module's parameter traversal is malformed.
    #[error(transparent)]
    Topology(#[from] ParameterTopologyError),
    /// Exact source projection failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// Canonical recipes cannot be represented by the retained source metadata.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Generic runtime binding declarations are inconsistent.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
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
    /// Selects the exact identities visited by one already constructed module.
    /// The module may be an individual layered block or a small static component.
    /// No payload is read and no native parameter is published here.
    pub fn from_retained<T: 'static, M: Parameterized<T>>(
        artifacts: &RetainedVoiceChatArtifacts,
        component: Component,
        module: &M,
    ) -> Result<Self, SpeechParameterError> {
        check_component(component)?;
        let source = artifacts
            .component_source(component)
            .ok_or(SpeechParameterError::MissingComponent(component))?
            .clone();
        project_unit(artifacts.plan().parameters(), source, component, module)
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

/// Integer scale storage belongs only to an explicitly encoded MXFP4 projection.
pub(super) fn is_byte_scale(parameters: &BTreeMap<String, ParameterAdmission>, name: &str) -> bool {
    name.strip_suffix(".scales")
        .and_then(|prefix| parameters.get(&format!("{prefix}.weight")))
        .is_some_and(|weight| weight.linear_format == eredu_checkpoint::LinearFormat::MxFp4)
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

fn project_unit<T: 'static, M: Parameterized<T>>(
    admitted: &BTreeMap<String, ParameterAdmission>,
    source: SharedCheckpointSource,
    component: Component,
    module: &M,
) -> Result<PreparedSpeechParameterUnit, SpeechParameterError> {
    check_component(component)?;
    let topology = validate_parameter_topology(module)?;
    let mut parameters = BTreeMap::new();
    for metadata in &topology {
        let name = metadata.id.as_str();
        if host_parameter(name) {
            return Err(SpeechParameterError::HostParameter(name.into()));
        }
        let parameter = admitted
            .get(name)
            .filter(|parameter| parameter.component == component)
            .ok_or_else(|| SpeechParameterError::MissingParameter {
                component,
                name: name.into(),
            })?;
        parameters.insert(name.to_owned(), execution_parameter(parameter, &source)?);
    }
    let mut recipes = BTreeMap::new();
    let mut aliases = Vec::new();
    for metadata in &topology {
        let name = metadata.id.as_str();
        let parameter = &parameters[name];
        if let Some(owner) = &metadata.alias_of {
            let owner_parameter = &parameters[owner.as_str()];
            if parameter.recipe != owner_parameter.recipe
                || parameter.output != owner_parameter.output
            {
                return Err(SpeechParameterError::AliasMismatch {
                    alias: name.into(),
                    owner: owner.as_str().into(),
                });
            }
            aliases.push(RecipeAlias::new(name, owner.as_str()));
        } else {
            recipes.insert(name.to_owned(), parameter.recipe.clone());
        }
    }
    let inputs = recipes
        .values()
        .flat_map(|recipe| recipe.source_keys().into_iter().map(str::to_owned))
        .collect();
    let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        source,
        format!("voicechat/{component:?}/module"),
        inputs,
    )?);
    let recipes = AtomicRecipeSet::new_with_aliases(source.as_ref(), recipes, aliases)?;
    let bindings = bindings_from_recipe_set(source.as_ref(), recipes)?;
    Ok(PreparedSpeechParameterUnit {
        component,
        source,
        parameters,
        bindings,
    })
}
