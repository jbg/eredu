//! Architecture-owned total cold selection from normalized policy and backend facts.

use std::num::NonZeroUsize;

use eredu_core::{
    admit_preparation, ArchitecturePreparationCapabilities, ArtifactFormat, ArtifactInspection,
    ParallelRankTopology, ParallelTopology, PreparationAdmissionError, PreparationAdmissionRequest,
    PreparationMechanismCapabilities, SessionCapabilities,
};
use eredu_runtime::{
    BackendMechanismCapabilities, CacheResidencyPolicy, CommunicationCapabilities,
    DraftingLoadRequest, GroupedOperationRequirement, MediaPrimitiveCapabilities,
    NormalizedLoadRequest, NormalizedLoadRequestError, ProcessorSelectionError,
    ReplicatedTextRequirements, ReplicatedTextSelectionError, ReplicatedTextSelectionRequest,
    SelectedProcessorExecution, SelectedReplicatedTextRealization,
    SpeculativeMechanismCapabilities, SpeculativeSelectionError,
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
    /// Conservative live buffers for one physical recipe, including native
    /// copies and index buffers. This must inspect metadata only.
    fn recipe_materialization_workspace(
        &self,
        _recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        _source: &dyn eredu_checkpoint::store::CheckpointSource,
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

    /// Portable media and prepared-tensor primitives.
    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities;

    /// Portable speculative execution mechanisms.
    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities;

    /// Portable communication operations, bounds, framing, and completion support.
    fn communication_capabilities(&self) -> CommunicationCapabilities;
}

/// Structured failure from total cold preparation selection.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum PreparationSelectionError {
    /// The normalized request is internally contradictory.
    #[error(transparent)]
    Request(#[from] NormalizedLoadRequestError),
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
    policy: eredu_core::PreparationPolicy,
    admitted_session: SessionCapabilities,
    processor: Option<SelectedProcessorExecution>,
    partitioned_base: bool,
    mechanisms: &'a P,
}

impl<P> ExecutionClassSelection<'_, P> {
    fn text_request(&self) -> ReplicatedTextSelectionRequest {
        let mut request = ReplicatedTextSelectionRequest::new(
            self.request.weight_residency().layers(),
            self.request.state_residency().clone(),
        )
        .with_max_cached_shards(
            std::num::NonZeroUsize::new(self.request.max_cached_shards())
                .expect("validated source reader limit is positive"),
        )
        .with_session(self.admitted_session)
        .with_prompt_cache(
            self.request.prompt_cache_persistence()
                || matches!(
                    self.request.state_residency(),
                    CacheResidencyPolicy::Paged(_)
                ),
        )
        .with_exact_completion(true);
        if !self.partitioned_base {
            if let Some(topology) = self.policy.topology() {
                request = request.with_topology(topology);
            }
        }
        if let Some(quantization) = self.policy.quantization() {
            request = request.with_quantization(quantization);
        }
        request
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
    #[cfg(test)]
    validation
        .selection_runs
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    let validated = request.validate_model_preparation()?;
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
    let projected_inspection =
        projection.map(|(target, _)| inspection.clone().map_architecture_plan(|_complete| target));
    let execution_inspection = projected_inspection.as_ref().unwrap_or(inspection);

    let processor = select_processor(execution_inspection, &policy, mechanisms)?;
    let parallel = request
        .parallel_topology()
        .filter(|topology| !topology.is_replicated());
    if let (Some(extension), Some(topology)) = (prediction_extension.as_ref(), parallel) {
        validate_partitioned_prediction_extension(extension, topology)
            .map_err(|error| PreparationSelectionError::Prediction(error.into()))?;
    }
    let base = ExecutionClassSelection {
        request,
        policy,
        admitted_session: admission.session_capabilities(),
        processor,
        partitioned_base: parallel.is_some(),
        mechanisms,
    };
    let execution = match parallel {
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
    Ok(SelectedPreparation::new(
        inspection.admission_token(),
        execution,
        admission,
        prediction_extension,
        prediction_realization,
    ))
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
    mechanisms: &P,
) -> Result<Option<SelectedProcessorExecution>, PreparationSelectionError>
where
    P: PreparationMechanismProvider,
{
    composite_processor_execution_requirements(inspection.architecture_plan())
        .map_err(PreparationSelectionError::ProcessorRequirements)?
        .map(|requirements| {
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

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;
    use std::time::Duration;

    use eredu_core::{CompletionCancellationMode, ModelConfigurationResolver};
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

    use super::*;

    #[derive(Default)]
    struct IndependentCounters {
        preparation_queries: Cell<usize>,
        grouped_queries: Cell<usize>,
        text_queries: Cell<usize>,
        processor_queries: Cell<usize>,
        speculative_queries: Cell<usize>,
        communication_queries: Cell<usize>,
        payload_reads: Cell<usize>,
        native_realizations: Cell<usize>,
    }

    #[derive(Clone, Copy, Default)]
    enum IndependentFailure {
        #[default]
        None,
        Processor,
        Grouped,
        Communication,
    }

    #[derive(Default)]
    pub(crate) struct BoundedIndependentAdapter {
        counters: IndependentCounters,
        failure: IndependentFailure,
        fp8: bool,
        transforms: bool,
    }

    impl BoundedIndependentAdapter {
        pub(crate) fn with_transforms() -> Self {
            Self {
                transforms: true,
                ..Self::default()
            }
        }
        pub(crate) fn with_fp8() -> Self {
            Self {
                fp8: true,
                ..Self::default()
            }
        }

        fn failing(failure: IndependentFailure) -> Self {
            Self {
                failure,
                ..Self::default()
            }
        }

        fn assert_cold_only(&self) {
            assert_eq!(self.counters.payload_reads.get(), 0);
            assert_eq!(self.counters.native_realizations.get(), 0);
        }
    }

    impl eredu_runtime::ReplicatedTextMechanismSupport for BoundedIndependentAdapter {
        fn facts(
            &self,
            _: &eredu_runtime::CacheResidencyPolicy,
        ) -> eredu_runtime::BackendMechanismFacts {
            eredu_runtime::BackendMechanismFacts::new(
                eredu_nn::NeuralOperatorCapabilities::ALL,
                [
                    eredu_runtime::WeightResidencyMechanism::Resident,
                    eredu_runtime::WeightResidencyMechanism::Windowed,
                    eredu_runtime::WeightResidencyMechanism::DiskStreamed,
                ],
                eredu_runtime::StateLifecycleCapabilities::new()
                    .with_transactions(true, true)
                    .with_reset(true)
                    .with_prompt_cache(true)
                    .with_observation_retention(true),
            )
            .with_session(SessionCapabilities::new(true, true, true))
            .with_prompt_cache(true)
            .with_exact_completion(true)
            .with_grouped_operations([
                GroupedOperationRequirement::Linear,
                GroupedOperationRequirement::GatedProduct,
                GroupedOperationRequirement::GatedProductTensorParallelPartial,
                GroupedOperationRequirement::Relu2,
            ])
            .with_indexed_movement(true)
            .with_addressable_storage(
                eredu_runtime::AddressableStorageCapabilities::new(true, true, true, 1 << 20),
            )
        }

        fn supports_direct(&self, descriptor: &eredu_runtime::WeightLoweringDescriptor) -> bool {
            use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
            if self.fp8
                && matches!(
                    descriptor.source(),
                    SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3)
                        | SourceTensorEncoding::RecipeOutput(StoredDtype::F8E4M3)
                )
            {
                return matches!(descriptor.executable(), LinearFormat::E4M3BlockFp8(_));
            }
            descriptor.executable() == LinearFormat::Dense
                && matches!(
                    descriptor.source(),
                    SourceTensorEncoding::Safetensors(
                        StoredDtype::F32 | StoredDtype::I32 | StoredDtype::U8
                    ) | SourceTensorEncoding::RecipeOutput(
                        StoredDtype::F32 | StoredDtype::I32 | StoredDtype::U8
                    ) | SourceTensorEncoding::Gguf {
                        ggml_type: eredu_gguf::GgmlType::F32,
                        ..
                    }
                )
        }

        fn supports_transform(&self, descriptor: &eredu_runtime::WeightLoweringDescriptor) -> bool {
            self.transforms
                && matches!(
                    descriptor.executable(),
                    eredu_checkpoint::LinearFormat::Affine(_)
                )
        }

        fn floating_state_dtype(
            &self,
            source: &eredu_core::checkpoint::TensorDtype,
        ) -> Option<eredu_runtime::StateStorageDtype> {
            use eredu_core::checkpoint::TensorDtype;
            use eredu_runtime::StateStorageDtype;
            match source {
                TensorDtype::F16 => Some(StateStorageDtype::F16),
                TensorDtype::Bf16 => Some(StateStorageDtype::Bf16),
                TensorDtype::F32 | TensorDtype::U32 | TensorDtype::Encoded(_) => {
                    Some(StateStorageDtype::F32)
                }
                _ => None,
            }
        }

        fn supports_state_component(
            &self,
            _: &eredu_core::cache::StateComponentPolicy,
            _storage_dtype: eredu_runtime::StateStorageDtype,
            placement: eredu_runtime::StateComponentPlacement,
        ) -> bool {
            matches!(
                placement,
                eredu_runtime::StateComponentPlacement::Device
                    | eredu_runtime::StateComponentPlacement::Paged
            )
        }
    }

    impl PreparationMechanismProvider for BoundedIndependentAdapter {
        fn recipe_materialization_workspace(
            &self,
            recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
            source: &dyn eredu_checkpoint::store::CheckpointSource,
        ) -> Result<u64, String> {
            recipe
                .peak_materialization_bytes(source)
                .map_err(|error| error.to_string())
        }

        fn preparation_capabilities(&self) -> PreparationMechanismCapabilities {
            self.counters
                .preparation_queries
                .set(self.counters.preparation_queries.get() + 1);
            let mut capabilities = PreparationMechanismCapabilities::new(true, true)
                .with_safetensors_quantization(true, true)
                .with_gguf_quantized_loading(true)
                .with_input_modalities(eredu_core::InputModalities {
                    text: true,
                    image: true,
                    audio: true,
                    video: true,
                })
                .with_exact_completion(true)
                .with_session(SessionCapabilities::new(true, true, true));
            for residency in [
                eredu_core::ResidencyRequest::FullyResident,
                eredu_core::ResidencyRequest::LayerwiseHost,
                eredu_core::ResidencyRequest::DenseDiskStream,
                eredu_core::ResidencyRequest::AddressableParameterBanks,
            ] {
                capabilities = capabilities.with_residency(residency, true);
            }
            for axis in [
                eredu_core::ParallelAxis::Tensor,
                eredu_core::ParallelAxis::Pipeline,
                eredu_core::ParallelAxis::Expert,
                eredu_core::ParallelAxis::Data,
            ] {
                capabilities = capabilities.with_parallel_axis(axis, true);
            }
            capabilities
        }

        fn supports_grouped_operation(&self, _: GroupedOperationRequirement) -> bool {
            self.counters
                .grouped_queries
                .set(self.counters.grouped_queries.get() + 1);
            !matches!(self.failure, IndependentFailure::Grouped)
        }

        fn replicated_text_capabilities(
            &self,
            requirements: &ReplicatedTextRequirements,
            request: &ReplicatedTextSelectionRequest,
        ) -> BackendMechanismCapabilities {
            self.counters
                .text_queries
                .set(self.counters.text_queries.get() + 1);
            eredu_runtime::synthesize_replicated_text_capabilities(requirements, request, self)
        }

        fn processor_capabilities(&self) -> MediaPrimitiveCapabilities {
            self.counters
                .processor_queries
                .set(self.counters.processor_queries.get() + 1);
            let modalities = [
                eredu_core::InputModality::Text,
                eredu_core::InputModality::Image,
                eredu_core::InputModality::Audio,
                eredu_core::InputModality::Video,
            ];
            let available = !matches!(self.failure, IndependentFailure::Processor);
            MediaPrimitiveCapabilities::new(
                available.then_some(modalities).into_iter().flatten(),
                available.then_some(modalities).into_iter().flatten(),
                available.then_some(modalities).into_iter().flatten(),
                [],
                if available { u64::MAX } else { 1 },
            )
        }

        fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities {
            self.counters
                .speculative_queries
                .set(self.counters.speculative_queries.get() + 1);
            use eredu_runtime::SpeculativeMechanism::*;
            SpeculativeMechanismCapabilities::new([
                TensorOperations,
                NeuralOperations,
                GroupedNeuralOperations,
                HyperNeuralOperations,
                PayloadMaterialization,
                LogitsProcessing,
                Sampling,
                Randomness,
                StateStorage,
                StorageResidency,
                ExactCompletion,
                Observation,
                Timing,
                QueueBinding,
                Communication,
                Agreement,
                Publication,
                SameDeviceHandoff,
                CrossDeviceTransfer,
            ])
        }

        fn communication_capabilities(&self) -> CommunicationCapabilities {
            self.counters
                .communication_queries
                .set(self.counters.communication_queries.get() + 1);
            if matches!(self.failure, IndependentFailure::Communication) {
                return CommunicationCapabilities::new([]).unwrap();
            }
            use eredu_runtime::{
                BoundaryFramingProtocol, CommunicationCompletionCapabilities,
                CommunicationOperation, CommunicationOperationRequirement,
                CommunicationTensorLimits,
            };
            let dtypes = [
                eredu_core::checkpoint::TensorDtype::Bool,
                eredu_core::checkpoint::TensorDtype::F32,
                eredu_core::checkpoint::TensorDtype::F16,
                eredu_core::checkpoint::TensorDtype::Bf16,
                eredu_core::checkpoint::TensorDtype::I32,
                eredu_core::checkpoint::TensorDtype::U32,
            ];
            let limits = CommunicationTensorLimits::new(64, 16, usize::MAX / 4, None)
                .unwrap()
                .with_output_tensor_elements(usize::MAX / 2)
                .unwrap();
            let operations = [
                CommunicationOperation::AllReduceSum,
                CommunicationOperation::AllGatherEven,
                CommunicationOperation::AllGatherUneven,
                CommunicationOperation::SendReceive,
                CommunicationOperation::Broadcast,
            ]
            .into_iter()
            .map(|operation| {
                CommunicationOperationRequirement::tensors(operation, dtypes.clone(), limits, true)
                    .unwrap()
            })
            .chain([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::VariableAllToAll,
                    dtypes.clone(),
                    CommunicationTensorLimits::new(64, 16, usize::MAX / 4, Some(usize::MAX / 4))
                        .unwrap()
                        .with_output_tensor_elements(usize::MAX / 2)
                        .unwrap(),
                    true,
                )
                .unwrap(),
                CommunicationOperationRequirement::barrier(true),
                CommunicationOperationRequirement::failure_agreement(true),
            ]);
            CommunicationCapabilities::new(operations)
                .unwrap()
                .with_boundary_framing([BoundaryFramingProtocol::RoleExactV1])
                .unwrap()
                .with_completion_capabilities(
                    CommunicationCompletionCapabilities::new([
                        CompletionCancellationMode::QuarantineUntilComplete,
                    ])
                    .unwrap(),
                )
        }
    }

    struct UnreachableMechanisms;

    impl PreparationMechanismProvider for UnreachableMechanisms {
        fn preparation_capabilities(&self) -> PreparationMechanismCapabilities {
            panic!("invalid request reached backend preparation facts")
        }

        fn supports_grouped_operation(&self, _: GroupedOperationRequirement) -> bool {
            panic!("invalid request reached grouped-operation facts")
        }

        fn replicated_text_capabilities(
            &self,
            _: &ReplicatedTextRequirements,
            _: &ReplicatedTextSelectionRequest,
        ) -> BackendMechanismCapabilities {
            panic!("invalid request reached text mechanism facts")
        }

        fn processor_capabilities(&self) -> MediaPrimitiveCapabilities {
            panic!("invalid request reached processor facts")
        }

        fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities {
            panic!("invalid request reached speculative facts")
        }

        fn communication_capabilities(&self) -> CommunicationCapabilities {
            panic!("invalid request reached communication facts")
        }
    }

    pub(crate) fn inspected_llama() -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        inspected_config(serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }))
    }

    pub(crate) fn inspected_config(
        config: serde_json::Value,
    ) -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        inspected_config_layout(config, None)
    }

    fn inspected_config_layout(
        config: serde_json::Value,
        layout: Option<&str>,
    ) -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let plan = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let tensors = plan
            .common_tensors
            .iter()
            .chain(
                plan.layout_groups
                    .iter()
                    .filter_map(|group| {
                        group
                            .variants
                            .iter()
                            .find(|variant| Some(variant.id.as_str()) == layout)
                            .or_else(|| group.variants.first())
                    })
                    .flat_map(|variant| variant.tensors.iter()),
            )
            .map(|constraint| {
                let dtype = match constraint.dtype {
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::U8,
                    ) => Dtype::U8,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::F8E4M3,
                    ) => Dtype::F8_E4M3,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::BF16,
                    ) => Dtype::BF16,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::F16,
                    ) => Dtype::F16,
                    _ => Dtype::F32,
                };
                let bytes_per_element = dtype.bitsize() / 8;
                let bytes = vec![0; constraint.shape.iter().product::<usize>() * bytes_per_element];
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    dtype,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            tensors.iter().map(|(name, shape, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes.as_slice()).unwrap(),
                )
            }),
            None,
            &root.path().join("model.safetensors"),
        )
        .unwrap();
        let inspection = crate::configuration::inspect_artifact(root.path()).unwrap();
        (root, inspection)
    }

    pub(crate) fn routed_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        })
    }

    pub(crate) fn composite_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"gemma4", "tie_word_embeddings":false,
            "text_config": {
                "model_type":"gemma4_text", "hidden_size":16,
                "num_hidden_layers":2, "intermediate_size":32,
                "num_attention_heads":4, "num_key_value_heads":2,
                "head_dim":4, "rms_norm_eps":0.000001, "vocab_size":64,
                "max_position_embeddings":128,
                "layer_types":["full_attention","full_attention"],
                "enable_moe_block":true, "num_experts":2,
                "top_k_experts":1, "moe_intermediate_size":8
            }
        })
    }

    pub(crate) fn prediction_config() -> serde_json::Value {
        serde_json::json!({
            "architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3",
            "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":8,
            "num_hidden_layers":2,"num_attention_heads":2,"vocab_size":32,
            "max_position_embeddings":128,"q_lora_rank":4,"kv_lora_rank":4,
            "qk_nope_head_dim":6,"qk_rope_head_dim":2,"v_head_dim":8,
            "first_k_dense_replace":1,"moe_layer_freq":1,"n_routed_experts":2,
            "n_shared_experts":1,"num_experts_per_tok":1,"n_group":1,"topk_group":1,
            "topk_method":"noaux_tc","scoring_func":"sigmoid","norm_topk_prob":true,
            "routed_scaling_factor":1.0,"tie_word_embeddings":false,"attention_dropout":0.0,
            "hidden_act":"silu","num_nextn_predict_layers":1
        })
    }

    struct SemanticExecutionProbe {
        partitioned: bool,
        routed: bool,
        processor: bool,
    }

    impl SemanticExecutionProbe {
        fn verify(self, partitioned: bool, routed: bool, processor: bool) {
            assert_eq!(self.partitioned, partitioned);
            assert_eq!(self.routed, routed);
            assert_eq!(self.processor, processor);
        }
    }

    impl crate::SelectedExecutionDispatcher for SemanticExecutionProbe {
        type Output = ();
        type Error = std::convert::Infallible;

        fn replicated(
            self,
            _: eredu_runtime::SelectedReplicatedTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            self.verify(false, false, false);
            Ok(())
        }

        fn routed(
            self,
            _: crate::SelectedRoutedTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            self.verify(false, true, false);
            Ok(())
        }

        fn composite(
            self,
            _: crate::replicated_text::SelectedCompositeTextRealization,
        ) -> Result<Self::Output, Self::Error> {
            self.verify(false, false, true);
            Ok(())
        }

        fn partitioned_dense(
            self,
            _: crate::SelectedDensePartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            self.verify(true, false, false);
            Ok(())
        }

        fn partitioned_routed(
            self,
            _: crate::SelectedRoutedPartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            self.verify(true, true, false);
            Ok(())
        }

        fn partitioned_composite(
            self,
            _: crate::SelectedCompositePartitionedExecution,
        ) -> Result<Self::Output, Self::Error> {
            self.verify(true, false, true);
            Ok(())
        }
    }

    pub(crate) fn parallel_request() -> NormalizedLoadRequest {
        let topology = ParallelTopology::new(1, 2, 1, 1).unwrap();
        let rank = ParallelRankTopology::new(topology, 0).unwrap();
        let completion = eredu_runtime::CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let parallel = eredu_runtime::ParallelLoadRequest::new(
            rank,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            32,
            completion,
        )
        .unwrap();
        NormalizedLoadRequest::default()
            .with_parallel_execution(parallel)
            .unwrap()
    }

    #[test]
    #[ignore = "manual scaling measurement with a full expert count and reduced matrix dimensions"]
    fn qwen_fp8_expert_validation_scaling() {
        for experts in [128, 256] {
            let config = serde_json::json!({
                "model_type": "qwen3_5_moe",
                "quantization_config": {
                    "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128], "modules_to_not_convert": ["model.embed_tokens", "lm_head", "mtp.fc"]
                },
                "text_config": {
                    "model_type": "qwen3_5_moe_text", "vocab_size": 64, "hidden_size": 32,
                    "num_hidden_layers": 40, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 8, "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                    "moe_intermediate_size": 128, "shared_expert_intermediate_size": 16,
                    "num_experts": experts, "num_experts_per_tok": 8, "mtp_num_hidden_layers": 1,
                    "layer_types": (0..40).map(|layer| if layer % 4 == 3 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>()
                }
            });
            let started = std::time::Instant::now();
            let (_root, inspection) = inspected_config_layout(config, Some("independent"));
            let inspected = started.elapsed();
            eprintln!("{experts} experts: fixture+inspection={inspected:.3?}");
            let mechanisms = BoundedIndependentAdapter {
                fp8: true,
                ..Default::default()
            };
            let request = NormalizedLoadRequest::default();
            let started = std::time::Instant::now();
            select_preparation(&inspection, &request, &mechanisms).unwrap();
            let first = started.elapsed();
            eprintln!("{experts} experts: first selection={first:.3?}");
            let started = std::time::Instant::now();
            let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
            let repeated = started.elapsed();
            let started = std::time::Instant::now();
            let plan = eredu_core::ModelPreparationPlan::from_retained_admission(
                inspection,
                selected.admission(),
            )
            .unwrap();
            crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
            eprintln!("40 layers x {experts} FP8 experts: fixture+inspection={inspected:.3?}, first selection={first:.3?}, repeated selection={repeated:.3?}, source preparation={:.3?}", started.elapsed());
        }
    }

    #[test]
    fn expert_projection_shares_complete_tasks_between_members() {
        let (_root, inspection) = inspected_config_layout(routed_config(), Some("independent"));
        let requirements = crate::routed_text_requirements(&inspection).unwrap();
        let selected = select_preparation(
            &inspection,
            &NormalizedLoadRequest::default(),
            &BoundedIndependentAdapter::default(),
        )
        .unwrap();
        let members = crate::routed_text::project_addressable_members(
            requirements
                .bank(eredu_runtime::RoutedBankId::new(0))
                .unwrap()
                .catalog(),
            selected.text_realization(),
        )
        .unwrap();
        let mut tasks = std::collections::BTreeMap::new();
        let mut shared = 0;
        for parameter in members.iter().flat_map(|member| member.parameters()) {
            if let Some(previous) = tasks.insert(parameter.task().name(), parameter.task()) {
                assert!(std::ptr::eq(previous, parameter.task()));
                shared += 1;
            }
        }
        assert!(
            shared > 0,
            "fixture must contain multiple members per target"
        );
    }

    #[test]
    fn validation_is_reused_through_inspection_selection_and_source_preparation() {
        use std::sync::{atomic::Ordering, Arc};

        for config in [routed_config(), composite_config(), prediction_config()] {
            let (_root, inspection) = inspected_config(config);
            let validation = inspection
                .architecture_plan()
                .validation(inspection.admission_token());
            let weak = Arc::downgrade(&validation);
            let mechanisms = BoundedIndependentAdapter::default();
            let request = NormalizedLoadRequest::default();
            let report = crate::inspect_selected_model(
                inspection.clone(),
                &request,
                &mechanisms,
                eredu_core::MediaFeatureAvailability {
                    image: false,
                    audio: false,
                },
            );
            assert!(report.selected().is_some());
            let selected = select_preparation(&inspection.clone(), &request, &mechanisms).unwrap();
            assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 1);
            let plan = eredu_core::ModelPreparationPlan::from_retained_admission(
                inspection,
                selected.admission(),
            )
            .unwrap();
            let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
            assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 1);
            assert!(!sources.graph().source_identity().is_resolved());
            let discovery = sources.prepare_discovery(Default::default(), Default::default());
            assert!(!discovery.identity_is_resolved());
            let captured = discovery.capture().unwrap();
            assert!(sources.graph().source_identity().is_resolved());
            assert_eq!(
                captured.artifact_identity,
                sources.source_identity().unwrap().to_string()
            );
            assert_eq!(
                sources.execution_identity(),
                sources
                    .selected()
                    .text_realization()
                    .requirements()
                    .architecture_identity()
            );
            drop(sources);
            drop(report);
            drop(validation);
            assert!(
                weak.upgrade().is_none(),
                "retained requirements must not form an ownership cycle"
            );
        }
    }

    #[test]
    fn validation_reuse_requires_identical_policy_mechanisms_and_artifact_admission() {
        use std::sync::{atomic::Ordering, Arc};

        let (root, inspection) = inspected_config(routed_config());
        let request = NormalizedLoadRequest::default();
        let mechanisms = BoundedIndependentAdapter::default();
        let validation = inspection
            .architecture_plan()
            .validation(inspection.admission_token());
        select_preparation(&inspection, &request, &mechanisms).unwrap();

        let unsupported = BoundedIndependentAdapter::failing(IndependentFailure::Grouped);
        for _ in 0..2 {
            assert!(matches!(
                select_preparation(&inspection, &request, &unsupported),
                Err(PreparationSelectionError::MissingGroupedOperations(_))
            ));
        }
        assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 2);
        select_preparation(&inspection, &request, &mechanisms).unwrap();
        assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 2);

        let changed = request.clone().with_drafting(DraftingLoadRequest::Disabled);
        select_preparation(&inspection, &changed, &mechanisms).unwrap();
        assert_eq!(validation.selection_runs.load(Ordering::Relaxed), 3);

        // Even transplanting a plan into a different inspection cannot reuse
        // proofs established against the first inspection's tensor catalog.
        let other = crate::configuration::inspect_artifact(root.path())
            .unwrap()
            .map_architecture_plan(|_| inspection.architecture_plan().clone());
        let other_validation = other
            .architecture_plan()
            .validation(other.admission_token());
        assert!(!Arc::ptr_eq(&validation, &other_validation));
        select_preparation(&other, &request, &mechanisms).unwrap();
        assert_eq!(other_validation.selection_runs.load(Ordering::Relaxed), 1);

        let changed_architecture = inspection
            .clone()
            .map_architecture_plan(ArtifactArchitecturePlan::without_prediction_extension);
        let changed_validation = changed_architecture
            .architecture_plan()
            .validation(changed_architecture.admission_token());
        assert!(!Arc::ptr_eq(&validation, &changed_validation));
    }

    #[test]
    fn contradictory_request_fails_before_backend_facts_are_queried() {
        let (_root, inspection) = inspected_llama();
        let completion = eredu_runtime::CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let request =
            NormalizedLoadRequest::default().with_communication_completion_policy(completion);

        let error = select_preparation(&inspection, &request, &UnreachableMechanisms).unwrap_err();
        assert!(matches!(
            error,
            PreparationSelectionError::Request(NormalizedLoadRequestError::OrphanedModelCompletion)
        ));
    }

    #[test]
    fn independent_adapter_proves_each_execution_semantic_through_the_total_driver() {
        let dense = serde_json::json!({
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8, "num_hidden_layers": 2, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "rms_norm_eps": 0.00001, "vocab_size": 16,
            "max_position_embeddings": 32, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        });
        for (config, probe) in [
            (
                dense.clone(),
                SemanticExecutionProbe {
                    partitioned: false,
                    routed: false,
                    processor: false,
                },
            ),
            (
                routed_config(),
                SemanticExecutionProbe {
                    partitioned: false,
                    routed: true,
                    processor: false,
                },
            ),
            (
                composite_config(),
                SemanticExecutionProbe {
                    partitioned: false,
                    routed: false,
                    processor: true,
                },
            ),
            (
                dense,
                SemanticExecutionProbe {
                    partitioned: true,
                    routed: false,
                    processor: false,
                },
            ),
            (
                routed_config(),
                SemanticExecutionProbe {
                    partitioned: true,
                    routed: true,
                    processor: false,
                },
            ),
            (
                composite_config(),
                SemanticExecutionProbe {
                    partitioned: true,
                    routed: false,
                    processor: true,
                },
            ),
        ] {
            let partitioned = probe.partitioned;
            let (_root, inspection) = inspected_config(config);
            let mechanisms = BoundedIndependentAdapter::default();
            let request = if partitioned {
                parallel_request()
            } else {
                NormalizedLoadRequest::default()
            };
            let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
            assert_eq!(selected.communication_manifest().is_some(), partitioned);
            selected.execution().clone().dispatch(probe).unwrap();
            assert_eq!(mechanisms.counters.preparation_queries.get(), 1);
            assert!(mechanisms.counters.text_queries.get() > 0);
            mechanisms.assert_cold_only();
        }
    }

    #[test]
    fn independent_adapter_exercises_all_four_drafting_modes_and_prepared_roles() {
        use std::collections::BTreeSet;

        for (drafting, expects_extension) in [
            (DraftingLoadRequest::ArchitectureDefault, true),
            (DraftingLoadRequest::Disabled, false),
            (DraftingLoadRequest::embedded(1).unwrap(), true),
            (DraftingLoadRequest::ExternalTarget, false),
        ] {
            let (_root, inspection) = inspected_config(prediction_config());
            let request = NormalizedLoadRequest::default().with_drafting(drafting);
            let mechanisms = BoundedIndependentAdapter::default();
            let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
            assert_eq!(selected.prediction_extension().is_some(), expects_extension);
            assert_eq!(
                selected.prediction_realization().is_some(),
                expects_extension
            );
            assert_eq!(
                mechanisms.counters.speculative_queries.get(),
                usize::from(expects_extension)
            );

            let plan = eredu_core::plan_model_preparation(
                inspection,
                request.preparation_policy().unwrap(),
                selected.session_capabilities(),
            )
            .unwrap();
            let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
            assert_eq!(sources.format(), ArtifactFormat::SafeTensors);
            assert_eq!(
                sources.execution_identity(),
                sources
                    .selected()
                    .text_realization()
                    .requirements()
                    .architecture_identity()
            );
            assert_ne!(sources.source_identity().unwrap().digest(), [0; 32]);
            assert!(sources.companions().next().is_none());
            assert_eq!(sources.prediction_extension().is_some(), expects_extension);
            let primary = sources
                .primary()
                .source_keys()
                .into_iter()
                .collect::<BTreeSet<_>>();
            let complete = sources
                .complete()
                .source_keys()
                .into_iter()
                .collect::<BTreeSet<_>>();
            let target = sources
                .target()
                .source_keys()
                .into_iter()
                .collect::<BTreeSet<_>>();
            assert_eq!(primary, complete);
            assert!(target.is_subset(&complete));
            if let Some(extension) = sources.extension() {
                let extension = extension.source_keys().into_iter().collect::<BTreeSet<_>>();
                assert!(!extension.is_empty());
                assert!(target.is_disjoint(&extension));
                assert_eq!(
                    target.union(&extension).cloned().collect::<BTreeSet<_>>(),
                    complete
                );
            } else {
                assert_eq!(target, complete);
            }
            assert_eq!(
                sources
                    .primary()
                    .source_diagnostics()
                    .unwrap()
                    .physical_reads,
                0
            );
            assert_eq!(
                sources
                    .target()
                    .source_diagnostics()
                    .unwrap()
                    .physical_reads,
                0
            );
            mechanisms.assert_cold_only();
        }
    }

    #[test]
    fn independent_adapter_failures_are_causal_and_stop_before_payload_or_native_work() {
        let (_root, composite) = inspected_config(composite_config());
        let processor = BoundedIndependentAdapter::failing(IndependentFailure::Processor);
        let error = select_preparation(&composite, &NormalizedLoadRequest::default(), &processor)
            .unwrap_err();
        assert!(matches!(error, PreparationSelectionError::Processor(_)));
        assert_eq!(processor.counters.processor_queries.get(), 1);
        assert_eq!(processor.counters.text_queries.get(), 0);
        processor.assert_cold_only();

        let (_root, routed) = inspected_config(routed_config());
        let grouped = BoundedIndependentAdapter::failing(IndependentFailure::Grouped);
        let error =
            select_preparation(&routed, &NormalizedLoadRequest::default(), &grouped).unwrap_err();
        assert!(matches!(
            error,
            PreparationSelectionError::MissingGroupedOperations(_)
        ));
        assert_eq!(grouped.counters.text_queries.get(), 0);
        grouped.assert_cold_only();

        let (_root, llama) = inspected_llama();
        let communication = BoundedIndependentAdapter::failing(IndependentFailure::Communication);
        let error = select_preparation(&llama, &parallel_request(), &communication).unwrap_err();
        assert!(
            matches!(error, PreparationSelectionError::PartitionedMechanisms(_)),
            "{error:?}"
        );
        assert_eq!(communication.counters.communication_queries.get(), 1);
        communication.assert_cold_only();

        let (_root, llama) = inspected_llama();
        let mechanisms = BoundedIndependentAdapter::default();
        let request = NormalizedLoadRequest::default()
            .with_drafting(DraftingLoadRequest::embedded(1).unwrap());
        let error = select_preparation(&llama, &request, &mechanisms).unwrap_err();
        assert!(matches!(
            error,
            PreparationSelectionError::MissingPredictionExtension
        ));
        assert_eq!(mechanisms.counters.speculative_queries.get(), 0);
        assert_eq!(mechanisms.counters.text_queries.get(), 0);
        mechanisms.assert_cold_only();
    }

    #[test]
    fn prepared_sources_reject_same_session_selection_from_another_dense_artifact() {
        let (_llama_root, llama) = inspected_llama();
        let mechanisms = BoundedIndependentAdapter::default();
        let request = NormalizedLoadRequest::default();
        let selected = select_preparation(&llama, &request, &mechanisms).unwrap();

        let (_qwen_root, qwen) = inspected_config(serde_json::json!({
            "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8, "num_hidden_layers": 2, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "rms_norm_eps": 0.00001, "vocab_size": 16,
            "max_position_embeddings": 32, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }));
        let plan = eredu_core::plan_model_preparation(
            qwen,
            request.preparation_policy().unwrap(),
            selected.session_capabilities(),
        )
        .unwrap();
        let error = match crate::prepared_sources::prepare_model_sources(plan, selected) {
            Ok(_) => panic!("cross-artifact source/selection substitution was admitted"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::prepared_sources::PreparedModelSourcesError::InvalidSelection(detail)
                if detail.contains("different artifact inspection")
        ));
        mechanisms.assert_cold_only();
    }

    #[test]
    fn prepared_sources_reject_same_artifact_selection_from_another_policy() {
        let (_root, inspection) = inspected_llama();
        let mechanisms = BoundedIndependentAdapter::default();
        let bounded_request = NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::layerwise_host(Default::default()),
        );
        let selected = select_preparation(&inspection, &bounded_request, &mechanisms).unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            NormalizedLoadRequest::default()
                .preparation_policy()
                .unwrap(),
            selected.session_capabilities(),
        )
        .unwrap();

        let error = match crate::prepared_sources::prepare_model_sources(plan, selected) {
            Ok(_) => panic!("cross-policy source/selection substitution was admitted"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::prepared_sources::PreparedModelSourcesError::InvalidSelection(detail)
                if detail.contains("policy or route")
        ));
        mechanisms.assert_cold_only();
    }
}
