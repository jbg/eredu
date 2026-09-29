//! Source-free joint vision/target selection, followed by exact role binding.
use super::*;
use crate::qwen4_exp::{
    media::{processor::SelectedMediaProcessor, MediaAdmissionConfig},
    mtp::PredictionSpec,
};
use crate::routed_text::{
    RoutedTextRequirements, RoutedTextSelectionRequest, SelectedRoutedTextRealization,
};
use eredu_checkpoint::recipe::RecipeCatalog;

pub(super) struct JointCatalog<'a> {
    pub(super) target: &'a dyn RecipeCatalog,
    pub(super) vision: &'a VisionPlan,
}
impl RecipeCatalog for JointCatalog<'_> {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        if self.vision.physical_sources().contains_key(key) {
            self.vision.tensor_metadata(key)
        } else {
            self.target.tensor_metadata(key)
        }
    }
}

/// Exact target and projector headers plus combined layer-sized requirements.
/// This authority contains no readable source, native context or backend tensor.
#[derive(Clone)]
pub struct ConditionalHeaderExecutionPlan {
    target: TargetPreparationPlan,
    vision: VisionPlan,
    policy: MediaAdmissionConfig,
    requirements: RoutedTextRequirements,
    capability: crate::capability::CapabilityEstimate,
    load_processor: Option<eredu_runtime::ProcessorSelectionRequest>,
    processor_budget: Option<eredu_runtime::processor_resources::ProcessorRequestBudget>,
}

/// All joint mechanisms selected before sources or integer controls are acquired.
#[derive(Clone)]
pub struct SelectedConditionalHeaderExecution {
    plan: Box<ConditionalHeaderExecutionPlan>,
    selected: SelectedRoutedTextRealization,
    processor: SelectedMediaProcessor,
    prediction_state: Option<eredu_runtime::SelectedStateRealization>,
}

impl ConditionalHeaderExecutionPlan {
    pub(super) fn target_header(&self) -> &TargetPreparationPlan {
        &self.target
    }
    pub(super) fn new(
        target: TargetPreparationPlan,
        vision: VisionPlan,
    ) -> Result<Self, PreparationError> {
        let invalid = |error: crate::qwen4_exp::media::MediaInputError| {
            PreparationError::Contract(error.to_string())
        };
        let spec = target.target_spec()?;
        let reset_token = target.vision_reset_token_id()?;
        vision.validate_target(&spec.config, reset_token)?;
        let policy = MediaAdmissionConfig::new(&spec, vision.config(), vision.media_tokens())
            .map_err(invalid)?;
        let blocks = (0..vision.config().layer_count())
            .map(|i| vision.block_recipes(i).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let requirements = super::conditional::joint_requirements(
            target.requirements(),
            &spec,
            vision.config(),
            vision.static_recipes(),
            &blocks,
            &JointCatalog {
                target: &target,
                vision: &vision,
            },
            vision.physical_sources(),
            policy.identity(),
        )?;
        let capability = crate::capability::qwen4_exp_conditional(&spec, target.prediction_spec())?;
        Ok(Self {
            target,
            vision,
            policy,
            requirements,
            capability,
            load_processor: None,
            processor_budget: None,
        })
    }

    pub(crate) fn with_load_processor(
        mut self,
        processor: eredu_runtime::ProcessorSelectionRequest,
        budget: Option<eredu_runtime::processor_resources::ProcessorRequestBudget>,
    ) -> Self {
        self.load_processor = Some(processor);
        self.processor_budget = budget;
        self
    }

    /// Optional coarse raw-request limits retained from ordinary load policy.
    pub fn processor_budget(
        &self,
    ) -> Option<eredu_runtime::processor_resources::ProcessorRequestBudget> {
        self.processor_budget
    }

    /// Exact input intent retained from normalized conditional preparation.
    pub fn load_processor_request(&self) -> Option<&eredu_runtime::ProcessorSelectionRequest> {
        self.load_processor.as_ref()
    }

    /// Combined target, vision, expert-bank and row contracts.
    pub fn requirements(&self) -> &RoutedTextRequirements {
        &self.requirements
    }
    /// Header-derived persistent-state accounting; request buffers are separate.
    pub fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability
    }
    /// Exact admitted projector role and its source construction declaration.
    pub fn vision_plan(&self) -> &VisionPlan {
        &self.vision
    }
    /// Common artifact authority retained by the conditional target.
    pub fn target_artifact(&self) -> &TargetArtifactDeclaration {
        self.target.artifact()
    }
    /// Independent mutable predictor state, selected alongside the conditional target.
    pub fn prediction_state_requirements(
        &self,
    ) -> Option<&eredu_runtime::StateRealizationRequirements> {
        self.target.prediction_state_requirements()
    }
    /// Retained predictor geometry before parameter transformations.
    pub fn prediction_spec(&self) -> Option<&PredictionSpec> {
        self.target.prediction_spec()
    }
    /// Separate GGUF predictor headers, when admitted.
    pub fn prediction_header(&self) -> Option<&SafetensorsPredictionPlan> {
        self.target.prediction_header()
    }
    /// Source-independent geometry, original-ID and sequence-limit policy.
    pub fn media_policy(&self) -> &MediaAdmissionConfig {
        &self.policy
    }
    /// Normalized target policy retained before adding the vision role.
    pub fn load_selection_request(&self) -> Option<&RoutedTextSelectionRequest> {
        self.target.load_selection_request()
    }
    /// Raw, prepared and projected input requirements from retained processor artifacts.
    pub fn processor_requirements(
        &self,
    ) -> Result<eredu_runtime::ProcessorExecutionRequirements, eredu_runtime::ProcessorSelectionError>
    {
        self.policy.processor_requirements(self.vision.processor())
    }
    /// Selects combined weight/state and processor mechanisms with no source access.
    pub fn select(
        self,
        request: &RoutedTextSelectionRequest,
        mechanisms: &eredu_runtime::BackendMechanismCapabilities,
        prediction_mechanisms: Option<&eredu_runtime::StateMechanismCapabilities>,
        inputs: &eredu_runtime::ProcessorSelectionRequest,
        media: &eredu_runtime::MediaPrimitiveCapabilities,
    ) -> Result<SelectedConditionalHeaderExecution, TargetSelectionError> {
        if self
            .load_processor
            .as_ref()
            .is_some_and(|retained| retained != inputs)
        {
            return Err(TargetSelectionError::ProcessorRequestMismatch);
        }
        if self
            .target
            .load_selection_request()
            .is_some_and(|retained| retained != request)
        {
            return Err(TargetSelectionError::LoadRequestMismatch);
        }
        let processor = self
            .policy
            .select_processor(self.vision.processor(), inputs, media)?;
        let prediction_state = match (self.prediction_state_requirements(), prediction_mechanisms) {
            (Some(state), Some(mechanisms)) => Some(eredu_runtime::select_state_realization(
                state,
                request.text(),
                mechanisms,
            )?),
            (None, None) => None,
            _ => return Err(TargetSelectionError::PredictionStatePresence),
        };
        let selected = crate::routed_text::select_routed_text_realization(
            &self.requirements,
            request,
            mechanisms,
        )?;
        super::execution::selected_stream_allowances(&selected, prediction_state.as_ref())?;
        Ok(SelectedConditionalHeaderExecution {
            plan: Box::new(self),
            selected,
            processor,
            prediction_state,
        })
    }
}
impl SelectedConditionalHeaderExecution {
    /// Exact selected joint graph and parameter-bank mechanisms.
    pub fn realization(&self) -> &SelectedRoutedTextRealization {
        &self.selected
    }
    /// Independent predictor state retained through ordinary source binding.
    pub fn prediction_state(&self) -> Option<&eredu_runtime::SelectedStateRealization> {
        self.prediction_state.as_ref()
    }
    /// Predictor geometry after selected parameter transformations.
    pub fn prediction_spec(&self) -> Result<PredictionSpec, PreparationError> {
        self.plan.target.selected_prediction_spec(&self.selected)
    }
    pub(crate) fn prediction_descriptor(
        &self,
    ) -> Result<eredu_core::ArchitectureDescriptor, PreparationError> {
        let target = self.plan.target.target_spec()?;
        let prediction = self.prediction_spec()?;
        super::graph::Graph::new(&target, &prediction)
            .with_prediction_observations(&target, &prediction)
    }
    /// Exact selected input representations and processor policy.
    pub fn processor(&self) -> &SelectedMediaProcessor {
        &self.processor
    }
    /// Retained source-free admission for ordinary source-role construction.
    pub fn header_plan(&self) -> &ConditionalHeaderExecutionPlan {
        &self.plan
    }
    /// Conservative recipe workspace from both header catalogs, with no payload methods.
    pub fn parameter_materialization_workspace(
        &self,
        mechanisms: &impl crate::PreparationMechanismProvider,
    ) -> Result<eredu_core::ParameterMaterializationWorkspace, String> {
        crate::SelectedExecution::routed(self.selected.clone()).parameter_materialization_workspace(
            &JointCatalog {
                target: &self.plan.target,
                vision: &self.plan.vision,
            },
            None,
            mechanisms,
        )
    }
    /// Pins exact retained target/projector sources and consumes the selected authority.
    /// Only bounded target integer controls are read; vision and table weights remain lazy.
    pub fn bind(
        self,
        target_source: SharedCheckpointSource,
        vision_source: SharedCheckpointSource,
        prediction_source: Option<SharedCheckpointSource>,
    ) -> Result<SelectedConditionalExecution, PreparationError> {
        // Prove projector identity before target binding may read integer controls.
        let vision = self.plan.vision.bind(vision_source)?;
        let bound = self
            .plan
            .target
            .bind(target_source, prediction_source)?
            .with_vision(vision)?;
        bound
            .bind_selected(self.selected, self.processor, self.prediction_state)
            .map_err(|error| PreparationError::Contract(error.to_string()))
    }
}
