use super::*;

/// Opaque MLX model configuration selected before payloads are opened.
pub struct MlxModelConfig {
    pub(crate) sources: PreparedModelSources,
    pub(crate) rank_context: Option<crate::backend::MlxRankContext>,
}

impl MlxModelConfig {
    pub(crate) fn new(
        selected: eredu_core::SelectedModelPreparation<crate::backend::MlxBackend<'_>>,
    ) -> Result<Self, Error> {
        let (plan, selected) = selected.into_parts();
        let (sources, rank_context) = prepare_selected_sources(plan, selected)?;
        Ok(Self {
            sources,
            rank_context,
        })
    }
}

pub(crate) fn prepare_selected_sources(
    plan: eredu_core::ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected: MlxSelectedPreparation,
) -> Result<(PreparedModelSources, Option<crate::backend::MlxRankContext>), Error> {
    let MlxSelectedPreparation {
        selected,
        rank_context,
    } = selected;
    let sources = prepare_model_sources(plan, selected)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    #[cfg(test)]
    {
        // Every admitted primary source is prepared once, independent of its
        // container format; target/extension views share that same source.
        crate::tests::support::path_instrumentation::payload_open();
        for _ in sources.companions() {
            crate::tests::support::path_instrumentation::payload_open();
        }
    }
    Ok((sources, rank_context))
}

/// Opaque, authoritative MLX construction policy selected before payloads are opened.
#[derive(Debug, Clone)]
pub struct MlxSelectedPreparation {
    selected: eredu_architectures::SelectedPreparation,
    rank_context: Option<crate::backend::MlxRankContext>,
}

impl MlxSelectedPreparation {
    const fn new(
        selected: eredu_architectures::SelectedPreparation,
        rank_context: Option<crate::backend::MlxRankContext>,
    ) -> Self {
        Self {
            selected,
            rank_context,
        }
    }

    pub(crate) const fn session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.selected.session_capabilities()
    }

    pub(crate) const fn neutral(&self) -> &eredu_architectures::SelectedPreparation {
        &self.selected
    }

    #[cfg(test)]
    pub(crate) const fn rank_context(&self) -> Option<crate::backend::MlxRankContext> {
        self.rank_context
    }
}

pub(crate) fn select_preparation(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_mechanisms(
        inspection,
        options,
        &MlxPreparationMechanisms::new(
            &super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
        ),
    )
}

pub(crate) struct MlxPreparationMechanisms<'a> {
    grouped: &'a [eredu_runtime::GroupedOperationRequirement],
    communication: Option<&'a eredu_runtime::CommunicationCapabilities>,
}

impl<'a> MlxPreparationMechanisms<'a> {
    pub(crate) const fn new(grouped: &'a [eredu_runtime::GroupedOperationRequirement]) -> Self {
        Self {
            grouped,
            communication: None,
        }
    }

    #[cfg(test)]
    const fn with_communication(
        mut self,
        communication: &'a eredu_runtime::CommunicationCapabilities,
    ) -> Self {
        self.communication = Some(communication);
        self
    }
}

impl eredu_architectures::PreparationMechanismProvider for MlxPreparationMechanisms<'_> {
    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            floating_to_f32: true,
        }
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        super::super::session::bounded_capture::capabilities()
    }

    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        super::super::structural::preparation_mechanism_capabilities()
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        self.grouped.contains(&requirement)
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        super::super::replicated_text::capabilities(requirements, request)
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        super::super::processor::capabilities()
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        super::super::speculative::speculative_mechanism_capabilities()
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        self.communication.cloned().unwrap_or_else(|| {
            crate::backend::runtime::distributed::topology::mlx_communication_capabilities()
        })
    }
}

#[cfg(test)]
pub(crate) fn select_preparation_with_grouped_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_mechanisms(
        inspection,
        options,
        &MlxPreparationMechanisms::new(grouped_capabilities),
    )
}

#[cfg(test)]
pub(super) fn select_preparation_with_mechanism_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
    communication: &eredu_runtime::CommunicationCapabilities,
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_mechanisms(
        inspection,
        options,
        &MlxPreparationMechanisms::new(grouped_capabilities).with_communication(communication),
    )
}

fn select_preparation_with_mechanisms(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    mechanisms: &impl eredu_architectures::PreparationMechanismProvider,
) -> Result<MlxSelectedPreparation, Error> {
    let (request, rank_context) = options.checked_normalized()?;
    let selected = eredu_architectures::preparation_selection::select_preparation(
        inspection, request, mechanisms,
    )
    .map_err(preparation_selection_error)?;
    Ok(MlxSelectedPreparation::new(selected, rank_context))
}

fn preparation_selection_error(error: eredu_architectures::PreparationSelectionError) -> Error {
    match error {
        eredu_architectures::PreparationSelectionError::Admission(error) => {
            Error::PreparationAdmission(error)
        }
        error => Error::ArchitectureModel(error.to_string()),
    }
}
