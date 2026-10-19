//! Architecture-owned total cold selection from normalized policy and backend facts.

use std::num::NonZeroUsize;

use eredu_core::{
    admit_preparation, ArchitecturePreparationCapabilities, ArtifactFormat, ArtifactInspection,
    ParallelRankTopology, ParallelTopology, PreparationAdmissionError, PreparationAdmissionRequest,
    PreparationMechanismCapabilities, SessionCapabilities,
};
use eredu_runtime::{
    BackendMechanismCapabilities, CommunicationCapabilities, DraftingLoadRequest,
    GroupedOperationRequirement, MediaPrimitiveCapabilities, NormalizedLoadRequest,
    NormalizedLoadRequestError, ProcessorSelectionError, ReplicatedTextRequirements,
    ReplicatedTextSelectionError, ReplicatedTextSelectionRequest, SelectedProcessorExecution,
    SelectedReplicatedTextRealization, SpeculativeMechanismCapabilities, SpeculativeSelectionError,
};

use crate::{
    composite_partitioned::{
        composite_partitioned_production_decision, CompositePartitionedProductionDecision,
    },
    configuration::PredictionExtensionPlan,
    partitioned_execution::{
        dense_decoder_partitioned_production_route, dispatch_partitioned_admission,
        routed_partitioned_production_supported, select_composite_partitioned_admission,
        select_direct_partitioned_admission, select_routed_partitioned_admission,
        CompositePartitionedAdmission, DenseDecoderPartitionedProductionRoute,
        DirectPartitionedAdmission, PartitionedAdmissionDispatcher, PartitionedAdmissionError,
        PartitionedAdmissionSelectionError, PartitionedSelectionRequest,
        RoutedPartitionedAdmission,
    },
    prediction_extension::{
        embedded_prediction_capacity, embedded_speculative_contract,
        prediction_extension_capability, validate_partitioned_prediction_extension,
        EmbeddedSpeculativeContractRequest,
    },
    preparation::{
        prepared_gguf_capabilities, prepared_safetensors_capabilities, ArchitectureCapabilities,
        PreparationCapabilityError,
    },
    processor_plan::{ArtifactArchitecturePlan, GgufSpecialTokenKind},
    replicated_text::{
        composite_processor_execution_requirements, dispatch_replicated_text_execution_class,
        select_composite_text_realization_with_processor, CompositeTextRequirements,
        CompositeTextSelectionError, ReplicatedTextDispatchError,
        ReplicatedTextExecutionClassDispatcher, ReplicatedTextRequirementsError,
        SelectedCompositeTextRealization,
    },
    select_routed_text_realization, RoutedTextRequirements, RoutedTextSelectionError,
    SelectedExecution, SelectedPreparation, SelectedRoutedTextRealization,
};

/// Side-effect-free backend facts consumed by total portable preparation selection.
///
/// Implementations must only describe mechanisms which are already implemented.
/// They must not allocate native resources, open checkpoint payloads, select an
/// execution branch, or inspect model-family identity. Architecture code supplies
/// exact neutral requirements and retains every selection decision.
pub trait PreparationMechanismProvider {
    /// Floating activation and state representation produced by materializing the
    /// declared embedding dtype. This is an immutable native mechanism fact.
    fn floating_state_dtype(
        &self,
        _source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<eredu_runtime::StateStorageDtype> {
        None
    }

    /// Storage tiers and completion ownership for addressable table rows.
    fn row_lookup_storage(&self) -> Option<eredu_runtime::AddressableStorageCapabilities> {
        None
    }

    /// Exact row decoder and scalar-recipe support, using immutable headers only.
    fn row_lookup_workspace(
        &self,
        _descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<eredu_runtime::RowLookupWorkspace>, eredu_runtime::RowLookupError> {
        Ok(None)
    }

    /// Native row decoder storage and retention facts for one bounded acquisition.
    fn row_lookup_decode_memory(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_runtime::RowLookupError>
    {
        descriptor.decode_memory()
    }

    /// Native temporary geometry for explicit input-score attention. No family
    /// identity, tensor, stream or device is consulted.
    fn input_score_attention_workspace(
        &self,
    ) -> Option<eredu_runtime::memory_estimation::InputScoreAttentionMechanism> {
        None
    }

    /// Conservative live buffers for one physical recipe, including native
    /// copies and index buffers. This must inspect metadata only.
    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        _recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        _source: &C,
    ) -> Result<u64, String> {
        Err("backend did not report native recipe workspace".into())
    }

    /// Host observation collector facts, independent of model family and devices.
    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms::default()
    }

    /// Bounded native transformations, without allocating or selecting a device.
    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        Default::default()
    }

    /// Portable artifact, residency, topology, modality, completion, and session mechanisms.
    fn preparation_capabilities(&self) -> PreparationMechanismCapabilities;

    /// Whether one exact grouped-compute semantic is implemented.
    fn supports_grouped_operation(&self, requirement: GroupedOperationRequirement) -> bool;

    /// Mechanisms applicable to one exact neutral text requirement and request.
    fn replicated_text_capabilities(
        &self,
        requirements: &ReplicatedTextRequirements,
        request: &ReplicatedTextSelectionRequest,
    ) -> BackendMechanismCapabilities;

    /// Mechanisms for an independently owned auxiliary state role.
    fn state_capabilities(
        &self,
        _requirements: &eredu_runtime::StateRealizationRequirements,
        _policy: &eredu_runtime::CacheResidencyPolicy,
    ) -> eredu_runtime::StateMechanismCapabilities {
        eredu_runtime::StateMechanismCapabilities::new([])
    }

    /// Portable media and prepared-tensor primitives.
    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities;

    /// Portable speculative execution mechanisms.
    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities;

    /// Portable communication operations, bounds, framing, and completion support.
    fn communication_capabilities(&self) -> CommunicationCapabilities;
}

/// Projects preparation facts into the shared metadata-only row selector.
/// The borrowed provider can record every query for cached-selection agreement.
pub struct PreparationRowLookupMechanisms<'a, P: ?Sized> {
    provider: &'a P,
}

impl<'a, P: PreparationMechanismProvider + ?Sized> PreparationRowLookupMechanisms<'a, P> {
    /// Borrows the same provider used by ordinary preparation selection.
    pub const fn new(provider: &'a P) -> Self {
        Self { provider }
    }
}

impl<P: PreparationMechanismProvider + ?Sized> eredu_runtime::RowLookupMechanismSupport
    for PreparationRowLookupMechanisms<'_, P>
{
    fn storage(&self) -> Option<eredu_runtime::AddressableStorageCapabilities> {
        self.provider.row_lookup_storage()
    }

    fn workspace(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<eredu_runtime::RowLookupWorkspace>, eredu_runtime::RowLookupError> {
        self.provider.row_lookup_workspace(descriptor)
    }

    fn decode_memory(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_runtime::RowLookupError>
    {
        self.provider.row_lookup_decode_memory(descriptor)
    }
}

pub(crate) mod qwen4;

/// Structured failure from total cold preparation selection.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum PreparationSelectionError {
    /// Explicit media-source geometry, processor or retained file admission failed.
    #[error("media companion admission failed: {0}")]
    MediaCompanion(#[from] std::sync::Arc<crate::muse::media_companion::MediaCompanionError>),
    /// Separately supplied draft parameters failed ancestry, geometry or file admission.
    #[error("draft adapter admission failed: {0}")]
    DraftAdapter(String),
    /// Request-specific target bounds cannot be lowered from admitted headers.
    #[error(transparent)]
    Qwen4TargetLoad(#[from] std::sync::Arc<crate::qwen4_exp::prepared::TargetLoadError>),
    /// Exact retained target mechanisms are unavailable.
    #[error(transparent)]
    Qwen4TargetSelection(#[from] std::sync::Arc<crate::qwen4_exp::prepared::TargetSelectionError>),
    /// The selected family/container has no ordinary companion binding route.
    #[error("separate prediction-source loading is not implemented for this family/container")]
    PredictionSourceUnsupported,
    /// The separately supplied prediction artifact failed header admission.
    #[error(transparent)]
    PredictionSource(#[from] std::sync::Arc<eredu_core::artifact::ArtifactError>),
    /// The normalized request is internally contradictory.
    #[error(transparent)]
    Request(#[from] NormalizedLoadRequestError),
    /// Explicit media intent needs the retained conditional source and cursor route.
    #[error("ordinary loading of explicit media execution requires retained conditional source and prefill cursor wiring, which is not yet implemented")]
    MediaPreparationRequired,
    /// The selected composite cannot currently be projected to a text-only constructor.
    #[error("media-disabled loading requires a text-only projection of this composite execution, which is not yet implemented")]
    MediaDisabledForComposite,
    /// Facade-owned GGUF tokenizer identities have not been resolved yet.
    #[error("GGUF {0:?} media token IDs must be resolved before preparation selection")]
    UnresolvedGgufSpecialTokens(GgufSpecialTokenKind),
    /// The inspected container does not carry its corresponding architecture plan.
    #[error("inspected {format:?} artifact omitted its validated architecture plan")]
    MissingArchitecturePlan {
        /// Inspected artifact format.
        format: ArtifactFormat,
    },
    /// The architecture could not derive exact admission capabilities.
    #[error(transparent)]
    ArchitectureCapabilities(#[from] PreparationCapabilityError),
    /// Portable architecture/backend admission rejected the request.
    #[error(transparent)]
    Admission(#[from] PreparationAdmissionError),
    /// Required grouped operations are absent from the backend facts.
    #[error("backend mechanisms are missing grouped operations {0:?}")]
    MissingGroupedOperations(Vec<GroupedOperationRequirement>),
    /// Architecture-owned target/extension projection failed.
    #[error("prediction target projection failed: {0}")]
    PredictionProjection(#[source] std::sync::Arc<eredu_core::artifact::ArtifactError>),
    /// Embedded drafting was requested for an artifact without an extension.
    #[error("embedded drafting requires an admitted prediction extension")]
    MissingPredictionExtension,
    /// Architecture-owned embedded prediction selection failed.
    #[error("embedded prediction selection failed: {0}")]
    Prediction(#[source] std::sync::Arc<eredu_core::artifact::ArtifactError>),
    /// Generic speculative mechanism selection rejected embedded prediction.
    #[error(transparent)]
    Speculative(#[from] SpeculativeSelectionError),
    /// Composite processor requirements could not be derived.
    #[error("composite processor requirements are invalid: {0}")]
    ProcessorRequirements(#[source] ReplicatedTextRequirementsError),
    /// Generic media mechanisms cannot satisfy the processor request.
    #[error(transparent)]
    Processor(#[from] ProcessorSelectionError),
    /// A composite execution class did not receive its selected processor proof.
    #[error("composite execution has no selected processor realization")]
    MissingProcessor,
    /// Ordinary replicated-text selection failed.
    #[error(transparent)]
    Replicated(#[from] ReplicatedTextSelectionError),
    /// Ordinary routed-text selection failed.
    #[error(transparent)]
    Routed(#[from] RoutedTextSelectionError),
    /// Ordinary composite-text selection failed.
    #[error(transparent)]
    Composite(#[from] CompositeTextSelectionError),
    /// Architecture execution-class dispatch failed before backend selection.
    #[error("execution-class selection failed: {0}")]
    ExecutionClass(String),
    /// Architecture/topology partition admission failed.
    #[error("partitioned execution admission failed: {0}")]
    PartitionedAdmission(String),
    /// Backend communication mechanisms rejected a selected partition.
    #[error(transparent)]
    PartitionedMechanisms(#[from] PartitionedAdmissionSelectionError),
    /// The selected class has no exact architecture-owned production route.
    #[error("selected {class} execution has no neutral production implementation: {reason}")]
    UnsupportedProductionRoute {
        /// Selected execution class.
        class: &'static str,
        /// Architecture-owned rejection detail.
        reason: String,
    },
}

struct ExecutionClassSelection<'a, P> {
    request: &'a NormalizedLoadRequest,
    admitted_session: SessionCapabilities,
    processor: Option<SelectedProcessorExecution>,
    partitioned_base: bool,
    mechanisms: &'a P,
}

impl<P> ExecutionClassSelection<'_, P> {
    fn text_request(&self) -> ReplicatedTextSelectionRequest {
        self.request
            .validate_model_preparation()
            .expect("execution class selection retains a validated load request")
            .text_selection_request(self.admitted_session, self.partitioned_base)
    }
}

impl<P> ReplicatedTextExecutionClassDispatcher for ExecutionClassSelection<'_, P>
where
    P: PreparationMechanismProvider,
{
    type Replicated = SelectedReplicatedTextRealization;
    type Routed = SelectedRoutedTextRealization;
    type Composite = SelectedCompositeTextRealization;
    type Error = PreparationSelectionError;

    fn replicated(
        self,
        requirements: ReplicatedTextRequirements,
    ) -> Result<Self::Replicated, Self::Error> {
        let request = self.text_request();
        let capabilities = self
            .mechanisms
            .replicated_text_capabilities(&requirements, &request);
        eredu_runtime::select_replicated_text_realization(&requirements, &request, &capabilities)
            .map_err(Into::into)
    }

    fn routed(self, requirements: RoutedTextRequirements) -> Result<Self::Routed, Self::Error> {
        let text = self.text_request();
        let request =
            crate::RoutedTextSelectionRequest::new(text, self.request.weight_residency())?;
        let capabilities = self
            .mechanisms
            .replicated_text_capabilities(requirements.text(), request.text());
        select_routed_text_realization(&requirements, &request, &capabilities).map_err(Into::into)
    }

    fn composite(
        self,
        requirements: CompositeTextRequirements,
    ) -> Result<Self::Composite, Self::Error> {
        if matches!(
            self.request.media_execution(),
            eredu_runtime::MediaLoadRequest::Disabled
        ) {
            return Err(PreparationSelectionError::MediaDisabledForComposite);
        }
        let request = self.text_request();
        let processor = self
            .processor
            .ok_or(PreparationSelectionError::MissingProcessor)?;
        let capabilities = self
            .mechanisms
            .replicated_text_capabilities(requirements.execution(), &request);
        select_composite_text_realization_with_processor(
            &requirements,
            &request,
            self.request.weight_residency(),
            &capabilities,
            processor,
        )
        .map_err(Into::into)
    }
}

struct PartitionedSelection<'a, P> {
    inspection: &'a ArtifactInspection<ArtifactArchitecturePlan>,
    base: ExecutionClassSelection<'a, P>,
    communication: CommunicationCapabilities,
}

impl<P> PartitionedAdmissionDispatcher for PartitionedSelection<'_, P>
where
    P: PreparationMechanismProvider,
{
    type Output = SelectedExecution;
    type Error = PreparationSelectionError;

    fn direct(self, requirements: DirectPartitionedAdmission) -> Result<Self::Output, Self::Error> {
        let selected = self.base.replicated(requirements.execution().clone())?;
        let selected =
            select_direct_partitioned_admission(requirements, selected, &self.communication)?;
        match dense_decoder_partitioned_production_route(self.inspection, &selected) {
            DenseDecoderPartitionedProductionRoute::NeutralPartitioned => {
                Ok(SelectedExecution::partitioned_dense(selected))
            }
            DenseDecoderPartitionedProductionRoute::Unsupported(reason) => {
                Err(PreparationSelectionError::UnsupportedProductionRoute {
                    class: "direct partitioned",
                    reason: format!("{reason:?}"),
                })
            }
        }
    }

    fn routed(self, requirements: RoutedPartitionedAdmission) -> Result<Self::Output, Self::Error> {
        let selected = self.base.routed(requirements.execution().clone())?;
        let selected =
            select_routed_partitioned_admission(requirements, selected, &self.communication)?;
        if !routed_partitioned_production_supported(self.inspection, &selected) {
            return Err(PreparationSelectionError::UnsupportedProductionRoute {
                class: "routed partitioned",
                reason: "architecture has no admitted neutral routed constructor".into(),
            });
        }
        Ok(SelectedExecution::partitioned_routed(selected))
    }

    fn composite(
        self,
        requirements: CompositePartitionedAdmission,
    ) -> Result<Self::Output, Self::Error> {
        let selected = self.base.composite(requirements.execution().clone())?;
        let selected =
            select_composite_partitioned_admission(requirements, selected, &self.communication)?;
        match composite_partitioned_production_decision(&selected) {
            CompositePartitionedProductionDecision::Resident => {
                Ok(SelectedExecution::partitioned_composite(selected))
            }
            CompositePartitionedProductionDecision::Unsupported(reason) => {
                Err(PreparationSelectionError::UnsupportedProductionRoute {
                    class: "composite partitioned",
                    reason: reason.into(),
                })
            }
        }
    }
}

/// Selects complete neutral preparation exactly once before native construction.
pub fn select_preparation<P>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
) -> Result<SelectedPreparation, PreparationSelectionError>
where
    P: PreparationMechanismProvider,
{
    use crate::inspection_validation::{RecordingMechanisms, ValidatedSelection};
    let validation = inspection
        .architecture_plan()
        .validation(inspection.admission_token());
    let mut selections = validation
        .selections
        .lock()
        .expect("preparation validation poisoned");
    if let Some(retained) = selections.iter().find(|retained| {
        retained.request == *request
            && retained
                .mechanisms
                .iter()
                .all(|facts| facts.matches(mechanisms))
    }) {
        return retained.selected.clone();
    }
    let recording = RecordingMechanisms {
        provider: mechanisms,
        observations: Default::default(),
    };
    let selected = select_preparation_once(inspection, request, &recording);
    selections.push(ValidatedSelection {
        request: request.clone(),
        mechanisms: recording.observations.into_inner(),
        selected: selected.clone(),
    });
    selected
}

fn select_preparation_once<P>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
) -> Result<SelectedPreparation, PreparationSelectionError>
where
    P: PreparationMechanismProvider,
{
    let adapted_inspection = request
        .draft_adapter()
        .map(|source| {
            inspection
                .architecture_plan()
                .clone()
                .with_draft_adapter(source)
                .map(|plan| inspection.clone().map_architecture_plan(|_| plan))
                .map_err(PreparationSelectionError::DraftAdapter)
        })
        .transpose()?;
    let inspection = adapted_inspection.as_ref().unwrap_or(inspection);
    let media_inspection = if request.media_companions().is_empty() {
        None
    } else {
        let architecture = inspection
            .architecture_plan()
            .clone()
            .with_media_companions(request.media_companions())
            .map_err(|e| PreparationSelectionError::MediaCompanion(std::sync::Arc::new(e)))?;
        Some(inspection.clone().map_architecture_plan(|_| architecture))
    };
    let inspection = media_inspection.as_ref().unwrap_or(inspection);
    let validated = request.validate_model_preparation()?;
    if matches!(
        request.media_execution(),
        eredu_runtime::MediaLoadRequest::Required(_)
    ) && inspection.architecture_plan().model_kind() != crate::ModelKind::Qwen4Exp
    {
        return Err(PreparationSelectionError::MediaPreparationRequired);
    }
    if let Some(kind) = inspection
        .architecture_plan()
        .required_gguf_special_tokens()
    {
        return Err(PreparationSelectionError::UnresolvedGgufSpecialTokens(kind));
    }
    let policy = validated.preparation_policy();
    let admission =
        admit_inspected_preparation(inspection, policy, mechanisms.preparation_capabilities())?;

    let missing_grouped = inspection
        .architecture_plan()
        .grouped_operation_requirements(policy.topology())
        .into_iter()
        .filter(|required| !mechanisms.supports_grouped_operation(*required))
        .collect::<Vec<_>>();
    if !missing_grouped.is_empty() {
        return Err(PreparationSelectionError::MissingGroupedOperations(
            missing_grouped,
        ));
    }

    if request.prediction_source().is_some()
        && (inspection.architecture_plan().model_kind() != crate::ModelKind::Qwen4Exp
            || inspection.format() != ArtifactFormat::Gguf)
    {
        return Err(PreparationSelectionError::PredictionSourceUnsupported);
    }
    if inspection.architecture_plan().model_kind() == crate::ModelKind::Qwen4Exp {
        return qwen4::select(inspection, request, mechanisms, admission);
    }

    let projection = inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| PreparationSelectionError::PredictionProjection(error.into()))?;
    let discovered_extension = projection.as_ref().map(|(_, extension)| extension.clone());
    let prediction_extension = match request.drafting() {
        DraftingLoadRequest::ArchitectureDefault => discovered_extension,
        DraftingLoadRequest::Disabled | DraftingLoadRequest::ExternalTarget => None,
        DraftingLoadRequest::Embedded { .. } => Some(
            discovered_extension.ok_or(PreparationSelectionError::MissingPredictionExtension)?,
        ),
    };
    let prediction_realization = prediction_extension
        .as_ref()
        .map(|extension| {
            select_embedded_prediction_realization(inspection, extension, request, mechanisms)
        })
        .transpose()?;
    let projected_inspection = projection.map(|(target, _)| {
        let target = if prediction_extension.is_some() {
            target
        } else {
            target.without_prediction_extension()
        };
        inspection.clone().map_architecture_plan(|_complete| target)
    });
    let execution_inspection = projected_inspection.as_ref().unwrap_or(inspection);

    let processor = select_processor(
        execution_inspection,
        &policy,
        request.media_execution(),
        mechanisms,
    )?;
    let parallel = request
        .parallel_topology()
        .filter(|topology| !topology.is_replicated());
    if let (Some(extension), Some(topology)) = (prediction_extension.as_ref(), parallel) {
        validate_partitioned_prediction_extension(extension, topology)
            .map_err(|error| PreparationSelectionError::Prediction(error.into()))?;
    }
    let base = ExecutionClassSelection {
        request,
        admitted_session: admission.session_capabilities(),
        processor,
        partitioned_base: parallel.is_some(),
        mechanisms,
    };
    let mut execution = match parallel {
        Some(topology) => {
            let (maximum_batch_size, maximum_sequence_length) = request
                .partitioned_invocation_limits()
                .expect("parallel topology and invocation limits are atomic");
            let activation_dtype = request
                .pipeline_wire_contract()
                .expect("parallel topology and activation wire are atomic")
                .activation_dtype();
            let completion = request
                .communication_completion_policy()?
                .expect("parallel topology and completion policy are atomic");
            let partitioned = PartitionedSelectionRequest::new(
                topology.topology(),
                topology.global_rank(),
                maximum_batch_size,
                maximum_sequence_length,
                activation_dtype,
            )
            .map_err(PreparationSelectionError::PartitionedAdmission)?
            .with_completion_policy(completion);
            match dispatch_partitioned_admission(
                execution_inspection,
                partitioned,
                PartitionedSelection {
                    inspection: execution_inspection,
                    base,
                    communication: mechanisms.communication_capabilities(),
                },
            ) {
                Ok(selected) => selected,
                Err(PartitionedAdmissionError::Unsupported(reason)) => {
                    let reason = if prediction_extension.is_some() {
                        format!("prediction target: {reason}")
                    } else {
                        reason
                    };
                    return Err(PreparationSelectionError::PartitionedAdmission(reason));
                }
                Err(PartitionedAdmissionError::Dispatch(error)) => return Err(error),
            }
        }
        None => {
            let selected = dispatch_replicated_text_execution_class(
                execution_inspection,
                policy.topology(),
                base,
            )
            .map_err(map_execution_class_error)?;
            SelectedExecution::ordinary(selected)
        }
    };
    execution.input_score_attention_workspace = mechanisms.input_score_attention_workspace();
    Ok(SelectedPreparation::new(
        inspection.admission_token(),
        execution,
        admission,
        prediction_extension,
        prediction_realization,
    )
    .with_retained_architecture(inspection.architecture_plan().clone()))
}

fn admit_inspected_preparation(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    policy: eredu_core::PreparationPolicy,
    mechanisms: PreparationMechanismCapabilities,
) -> Result<eredu_core::PreparationAdmission, PreparationSelectionError> {
    let capabilities = match inspection.format() {
        ArtifactFormat::SafeTensors => prepared_safetensors_capabilities(
            inspection
                .architecture_plan()
                .safetensors_architecture()
                .ok_or(PreparationSelectionError::MissingArchitecturePlan {
                    format: ArtifactFormat::SafeTensors,
                })?,
        )?,
        ArtifactFormat::Gguf => {
            prepared_gguf_capabilities(inspection.architecture_plan().gguf_plan().ok_or(
                PreparationSelectionError::MissingArchitecturePlan {
                    format: ArtifactFormat::Gguf,
                },
            )?)
        }
        format => {
            return Err(PreparationSelectionError::MissingArchitecturePlan { format });
        }
    };
    let request = PreparationAdmissionRequest::new(
        inspection.configuration().loading_protocol(),
        inspection.format(),
        policy,
        architecture_admission_capabilities(capabilities),
    )
    .with_exact_completion(true);
    admit_preparation(request, mechanisms).map_err(Into::into)
}

fn architecture_admission_capabilities(
    capabilities: ArchitectureCapabilities,
) -> ArchitecturePreparationCapabilities {
    let parallel = capabilities.parallel_plan();
    ArchitecturePreparationCapabilities::new(
        capabilities.independently_addressable_experts(),
        capabilities.nonresident_safetensors_quantization(),
        parallel.tensor_parallel(),
        parallel.pipeline_parallel(),
        parallel.expert_parallel(),
        capabilities.input_modalities(),
    )
}

fn select_processor<P>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    policy: &eredu_core::PreparationPolicy,
    media: &eredu_runtime::MediaLoadRequest,
    mechanisms: &P,
) -> Result<Option<SelectedProcessorExecution>, PreparationSelectionError>
where
    P: PreparationMechanismProvider,
{
    composite_processor_execution_requirements(inspection.architecture_plan())
        .map_err(PreparationSelectionError::ProcessorRequirements)?
        .map(|requirements| {
            if matches!(media, eredu_runtime::MediaLoadRequest::Disabled) {
                return Err(PreparationSelectionError::MediaDisabledForComposite);
            }
            let request = eredu_runtime::ProcessorSelectionRequest::new(
                requirements
                    .modalities()
                    .iter()
                    .map(eredu_runtime::ModalityProcessorRequirements::modality),
            )
            .with_prepared_tensors(true)
            .with_projected_modalities(
                requirements
                    .modalities()
                    .iter()
                    .filter(|requirement| requirement.projected_embeddings())
                    .map(eredu_runtime::ModalityProcessorRequirements::modality),
            )
            .with_available_raw_media(
                policy
                    .topology()
                    .is_none_or(ParallelTopology::is_replicated)
                    && inspection.architecture_plan().has_processor(),
            );
            eredu_runtime::select_processor_execution(
                &requirements,
                &request,
                &mechanisms.processor_capabilities(),
            )
            .map_err(Into::into)
        })
        .transpose()
}

fn select_embedded_prediction_realization<P>(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    extension: &PredictionExtensionPlan,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
) -> Result<eredu_runtime::SelectedSpeculativeRealization, PreparationSelectionError>
where
    P: PreparationMechanismProvider,
{
    let identity =
        |value: String| eredu_runtime::SpeculativeIdentity::new(value).map_err(prediction_error);
    let topology = request.parallel_topology().map_or_else(
        || {
            ParallelTopology::new(1, 1, 1, 1)
                .map_err(prediction_error)
                .and_then(|topology| {
                    ParallelRankTopology::new(topology, 0).map_err(prediction_error)
                })
        },
        Ok,
    )?;
    let (maximum_batch_size, maximum_sequence_length) = match request
        .partitioned_invocation_limits()
    {
        Some(limits) => limits,
        None => {
            let capability = prediction_extension_capability(extension)
                .map_err(|error| PreparationSelectionError::Prediction(error.into()))?;
            let maximum_sequence_length = capability
                .capabilities()
                .effective_max_context
                .value()
                .copied()
                .ok_or_else(|| {
                    prediction_error("embedded prediction target has no effective context bound")
                })?;
            let maximum_sequence_length = i32::try_from(maximum_sequence_length).map_err(|_| {
                prediction_error(
                    "embedded prediction effective context exceeds portable dimensions",
                )
            })?;
            (1, maximum_sequence_length)
        }
    };
    let maximum_batch_size = positive_bound("maximum batch size", maximum_batch_size)?;
    let maximum_sequence_length =
        positive_bound("maximum sequence length", maximum_sequence_length)?;
    let maximum_draft_tokens = request
        .drafting()
        .embedded_capacity()
        .map_or_else(|| embedded_prediction_capacity(extension), Ok)
        .map_err(|error| PreparationSelectionError::Prediction(error.into()))?;
    // Cold selection is deliberately header-only. The opaque admission token
    // protects the exact artifact pairing and the prepared-source factory later
    // computes the content fingerprint once, before native materialization.
    let artifact = match inspection.format() {
        ArtifactFormat::SafeTensors | ArtifactFormat::Gguf => {
            format!("header/{:?}/{:?}", inspection.format(), extension.kind())
        }
        format => return Err(PreparationSelectionError::MissingArchitecturePlan { format }),
    };
    let contract = embedded_speculative_contract(
        extension,
        EmbeddedSpeculativeContractRequest::new(
            identity(format!("target/{:?}", extension.kind()))?,
            identity(format!("artifact/{artifact}"))?,
            identity(format!("format/{:?}", inspection.format()))?,
            topology,
            identity("prepared-input/text-token-ids/v1".into())?,
            maximum_batch_size,
            maximum_sequence_length,
            maximum_draft_tokens,
        ),
    )
    .map_err(|error| PreparationSelectionError::Prediction(error.into()))?;
    eredu_runtime::select_speculative_realization(
        contract.requirements(),
        &contract.selection_request(eredu_runtime::SpeculativePlacementRequest::Single),
        &mechanisms.speculative_capabilities(),
    )
    .map_err(Into::into)
}

fn positive_bound(
    name: &'static str,
    value: i32,
) -> Result<NonZeroUsize, PreparationSelectionError> {
    usize::try_from(value)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| prediction_error(format!("{name} must be positive")))
}

fn prediction_error(error: impl std::fmt::Display) -> PreparationSelectionError {
    PreparationSelectionError::Prediction(
        eredu_core::artifact::ArtifactError::InvalidArchitecturePlan(error.to_string()).into(),
    )
}

fn map_execution_class_error(
    error: ReplicatedTextDispatchError<PreparationSelectionError>,
) -> PreparationSelectionError {
    match error {
        ReplicatedTextDispatchError::Backend(error) => error,
        ReplicatedTextDispatchError::Ineligible(error) => {
            PreparationSelectionError::ExecutionClass(error.to_string())
        }
        ReplicatedTextDispatchError::Architecture(error) => {
            PreparationSelectionError::ExecutionClass(error)
        }
    }
}
