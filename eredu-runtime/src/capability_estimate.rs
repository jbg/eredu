//! Retained model capabilities and state layout after portable selection.

use crate::StateLayout as RuntimeStateLayout;
use eredu_core::{CapabilityError, ModelCapabilities, SpeculativeDraftSource, StateMemoryLayout};

/// Complete portable capability and runtime-state estimate for one architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityEstimate {
    capabilities: ModelCapabilities,
    state_layout: StateMemoryLayout,
    missing_media_companions: Vec<(eredu_core::InputModality, &'static str)>,
    draft_source: Option<SpeculativeDraftSource>,
    shared_backbone: Option<crate::speculative::shared_backbone::SharedBackboneExecutionPolicy>,
    ordinary_refinement:
        Option<crate::generation::refinement::architecture::RefinementExecutionPolicy>,
}

impl CapabilityEstimate {
    /// Retains a validated capability report and its executable state geometry.
    pub fn new(capabilities: ModelCapabilities, state_layout: StateMemoryLayout) -> Self {
        Self {
            capabilities,
            state_layout,
            missing_media_companions: Vec::new(),
            draft_source: None,
            shared_backbone: None,
            ordinary_refinement: None,
        }
    }

    /// Updates the declared capabilities while building the family report.
    pub fn capabilities_mut(&mut self) -> &mut ModelCapabilities {
        &mut self.capabilities
    }

    /// Replaces state geometry with the exact admitted layout.
    pub fn set_state_layout(&mut self, layout: StateMemoryLayout) {
        self.state_layout = layout;
    }

    /// Records the separately admitted components required by each modality.
    pub fn set_missing_media_companions(
        &mut self,
        companions: Vec<(eredu_core::InputModality, &'static str)>,
    ) {
        self.missing_media_companions = companions;
    }

    /// Records the checkpoint role of the selected speculative graph.
    pub fn with_speculative_draft_source(mut self, source: Option<SpeculativeDraftSource>) -> Self {
        self.draft_source = source;
        self
    }

    /// Records the retained ordinary equations for block refinement.
    pub fn with_ordinary_refinement(
        mut self,
        refinement: crate::generation::refinement::architecture::RefinementExecutionPolicy,
    ) -> Self {
        self.ordinary_refinement = Some(refinement);
        self
    }

    /// Attaches an admitted shared-backbone draft policy to the family declaration.
    pub fn with_shared_backbone_draft(
        mut self,
        policy: crate::speculative::shared_backbone::SharedBackboneExecutionPolicy,
    ) -> Self {
        self.ordinary_refinement = Some(
            crate::generation::refinement::architecture::RefinementExecutionPolicy {
                geometry: policy.geometry(),
                policy: policy
                    .diffusion_policy(0.0)
                    .expect("validated family policy"),
                causal_sampling: true,
                seed_first: policy.seed_first(),
            },
        );
        self.shared_backbone = Some(policy);
        self.draft_source = Some(SpeculativeDraftSource::SharedBackbone);
        self
    }
    /// Source-free masked draft defaults consumed by native construction adapters.
    pub const fn shared_backbone_draft(
        &self,
    ) -> Option<crate::speculative::shared_backbone::SharedBackboneExecutionPolicy> {
        self.shared_backbone
    }
    /// Retained ordinary numerical equations, independent of any draft source.
    pub const fn ordinary_refinement(
        &self,
    ) -> Option<crate::generation::refinement::architecture::RefinementExecutionPolicy> {
        self.ordinary_refinement
    }
    /// Complete target-stack observation semantics for a retained shared draft.
    /// All modes reuse ordinary target hook identities and producer ownership.
    pub fn shared_backbone_activation_execution(
        &self,
    ) -> Option<crate::speculative::SpeculativeActivationExecution> {
        self.shared_backbone
            .map(|_| crate::speculative::SpeculativeActivationExecution::shared_backbone())
    }
    /// Projects memory accounting onto the exact selected rank-local state.
    /// Family capabilities and context limits remain global model properties.
    pub fn for_selected_state(
        mut self,
        layout: Option<&RuntimeStateLayout>,
    ) -> Result<Self, CapabilityError> {
        self.state_layout = StateMemoryLayout::new(
            layout.map_or_else(eredu_core::LayerSchedule::empty, |layout| {
                layout.layers().clone()
            }),
            layout.map_or_else(Vec::new, |layout| layout.layer_prefix_offsets()),
            self.state_layout.hidden_size,
            self.state_layout.allocation_granularity,
            self.state_layout.completeness,
        )?;
        Ok(self)
    }

    /// Reuses an exact mechanism/state estimate for an architecture extension
    /// that implements the same validated neutral geometry under a new identity.
    pub fn for_architecture_extension(mut self, effective_model_type: impl Into<String>) -> Self {
        self.capabilities.effective_model_type = effective_model_type.into();
        self
    }

    /// Returns validated portable model capabilities.
    pub const fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Admits decoded media against selected component coverage before processor lookup.
    /// Token-only segments require no media components or processor assets.
    pub fn admit_raw_media_request(
        &self,
        request: &eredu_core::TokenizedMultimodalRequest,
        processor_available: bool,
    ) -> Result<(), CapabilityError> {
        use eredu_core::{InputModality, Media, TokenizedMultimodalSegment};
        for segment in request.segments() {
            let modality = match segment {
                TokenizedMultimodalSegment::TokenIds(_) => continue,
                TokenizedMultimodalSegment::Media(Media::Image(_)) => InputModality::Image,
                TokenizedMultimodalSegment::Media(Media::Video(_)) => InputModality::Video,
                TokenizedMultimodalSegment::Media(Media::Audio(_)) => InputModality::Audio,
            };
            if let Some((_, component)) = self
                .missing_media_companions
                .iter()
                .find(|(missing, _)| *missing == modality)
            {
                return Err(CapabilityError::MissingCompanion {
                    architecture: self.capabilities.effective_model_type.clone(),
                    modality,
                    component,
                });
            }
            let supported = match modality {
                InputModality::Text => self.capabilities.modalities.text,
                InputModality::Image => self.capabilities.modalities.image,
                InputModality::Video => self.capabilities.modalities.video,
                InputModality::Audio => self.capabilities.modalities.audio,
                _ => false,
            };
            if !supported {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: self.capabilities.effective_model_type.clone(),
                    reason: format!(
                        "selected architecture does not accept {} input",
                        modality.as_str()
                    ),
                });
            }
            if !processor_available {
                return Err(CapabilityError::MissingCompanion {
                    architecture: self.capabilities.effective_model_type.clone(),
                    modality,
                    component: "processor assets",
                });
            }
        }
        Ok(())
    }

    /// Returns memory metadata around the exact executable layer schedule.
    pub const fn state_layout(&self) -> &StateMemoryLayout {
        &self.state_layout
    }

    /// Architecture-declared checkpoint form for executable draft weights.
    ///
    /// `None` means that this exact normalized configuration exposes no
    /// speculative drafting graph. Concrete backends decide whether they
    /// implement the declared graph; they do not infer family policy themselves.
    pub const fn speculative_draft_source(&self) -> Option<SpeculativeDraftSource> {
        self.draft_source
    }

    /// Splits the estimate into its portable capability and state values.
    pub fn into_parts(self) -> (ModelCapabilities, StateMemoryLayout) {
        (self.capabilities, self.state_layout)
    }
}
