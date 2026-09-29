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
    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<eredu_runtime::StateStorageDtype> {
        super::super::replicated_text::floating_state_storage_dtype(source)
    }

    fn row_lookup_storage(&self) -> Option<eredu_runtime::AddressableStorageCapabilities> {
        eredu_runtime::RowLookupMechanismSupport::storage(
            &crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport,
        )
    }

    fn row_lookup_workspace(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<eredu_runtime::RowLookupWorkspace>, eredu_runtime::RowLookupError> {
        eredu_runtime::RowLookupMechanismSupport::workspace(
            &crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport,
            descriptor,
        )
    }

    fn row_lookup_decode_memory(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_runtime::RowLookupError>
    {
        eredu_runtime::RowLookupMechanismSupport::decode_memory(
            &crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport,
            descriptor,
        )
    }

    fn input_score_attention_workspace(
        &self,
    ) -> Option<eredu_runtime::memory_estimation::InputScoreAttentionMechanism> {
        Some(crate::backend::nn::attention::INPUT_SCORE_WORKSPACE)
    }

    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &C,
    ) -> Result<u64, String> {
        crate::backend::runtime::checkpoint::recipe::native_recipe_workspace(recipe, source)
    }

    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            routed_unit_tensors: true,
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

    fn state_capabilities(
        &self,
        requirements: &eredu_runtime::StateRealizationRequirements,
        policy: &eredu_runtime::CacheResidencyPolicy,
    ) -> eredu_runtime::StateMechanismCapabilities {
        super::super::replicated_text::state_capabilities(requirements, policy)
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
