use safemlx::error::Exception;

mod cold_workspace;
pub use cold_workspace::{WorkspaceCandidateRefusal, WorkspaceQuoteComponents};

mod ordinary_capture;
pub use ordinary_capture::OrdinaryCaptureFailure;
mod output_observation;
mod retained_original;
pub(crate) mod scoped_snapshots;
pub use output_observation::OutputObservationFailure;
pub use retained_original::RetainedOriginalFailure;

/// Backend-produced evidence that a failed model call did not advance state.
/// The payload has no public constructor; native completion must still be
/// proven independently before a session can accept another operation.
#[derive(Debug, thiserror::Error)]
#[error("architecture model error: {source}")]
pub struct ModelStatePreservedError {
    #[source]
    source: Box<dyn std::error::Error + Send + Sync>,
}

/// One original-operation error whose complete concrete source retires through
/// the neutral owned source boundary. The preservation witness is private and
/// does not establish native completion.
#[derive(Debug)]
pub struct OriginalControlFailure {
    source: eredu_core::BackendFailure,
    state_preserved: bool,
}
impl std::fmt::Display for OriginalControlFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(
            std::error::Error::source(&self.source).expect("original source"),
            f,
        )
    }
}
impl std::error::Error for OriginalControlFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.source)
    }
}

fn format_keys(keys: &[String]) -> String {
    const LIMIT: usize = 50;
    if keys.is_empty() {
        return "  <none>".to_string();
    }
    let mut lines = keys
        .iter()
        .take(LIMIT)
        .map(|key| format!("  {key}"))
        .collect::<Vec<_>>();
    if keys.len() > LIMIT {
        lines.push(format!("  ... and {} more", keys.len() - LIMIT));
    }
    lines.join("\n")
}

#[derive(Debug, thiserror::Error)]
/// Error type used by MLX model loading and execution.
#[non_exhaustive]
pub enum Error {
    /// Canonical parameter traversal could not establish complete source coverage.
    #[error(transparent)]
    ParameterSource(#[from] eredu_nn::ParameterSourceError),

    /// Exact shared embedded protocol diagnostic, with no formatted allocation.
    #[error("{0}")]
    EmbeddedPredictionContract(#[from] eredu_architectures::speculative_execution::EmbeddedPredictionContractError),
    /// Fixed mechanism validation diagnostic without a formatted allocation.
    #[error("{0}")]
    InvalidOperation(&'static str),
    /// Exact neutral selected-occurrence refusal before native dispatch.
    #[error("{0}")]
    SpeculativeOccurrence(
        #[source] eredu_runtime::speculative::autoregressive::AutoregressiveOccurrenceError,
    ),
    /// Fixed borrowed native descriptor refusal.
    #[error("{0}")]
    ArrayDescriptor(#[from] safemlx::ArrayDescriptorError),
    /// Fixed host-planning admission refusal; no diagnostic allocation.
    #[error("{0}")]
    WorkspacePlanning(#[from] eredu_nn::workspace::HostMetadataFundingError),
    /// Exact stream source/constructor refusal, retaining failed native prefixes.
    #[error(transparent)]
    GpuStreamOwnership(#[from] crate::backend::managed_memory::gpu_stream::MlxStreamOwnershipError),
    /// A previously allocated original diagnostic snapshot; retaining and public
    /// conversion preserve the same source without a new error-source Box.
    #[error(transparent)]
    RetainedOriginal(RetainedOriginalFailure),
    /// Fixed original Event/task mode refusal, without a second boxed shell.
    #[error(transparent)]
    OriginalNativeControl(#[from] safemlx::OriginalNativeControlError),
    /// Fixed shared-session inspection boundary; no formatted or boxed wrapper.
    #[error("runtime inspection: {0}")]
    RuntimeInspection(#[source] eredu_runtime::replicated_session::RuntimeInspectionBoundary),
    /// The concrete state visitor's cause without a second boxed error shell.
    #[error("prefill state roots: {0}")]
    PrefillState(#[source] eredu_runtime::StateError),
    /// Fixed root refusal or the unchanged native submission/completion cause.
    #[error(transparent)]
    PrefillRoots(#[from] safemlx::PrefillRootsError),
    /// Exact source/request/role check, without allocating a boxed fixed cause.
    #[error("original prefill control mismatch: {0}")]
    PrefillControl(#[source] eredu_runtime::working_memory::WorkingMemoryError),
    /// Exact original source boundary and unchanged neutral identity refusal.
    #[error("original source {stage}: {cause}")]
    OriginalSourceContract {
        /// Mechanism boundary which rejected the retained source.
        stage: &'static str,
        /// Neutral source/account validation cause.
        #[source]
        cause: eredu_runtime::working_memory::WorkingMemoryError,
    },
    /// Fixed phase attribution before original speculative native entry. The
    /// actual neutral cause remains a source; no diagnostic allocation is made.
    #[error("original speculative {stage}: {cause}")]
    SpeculativePrerequisite {
        /// Actual producer or validation boundary which refused.
        stage: &'static str,
        /// The unchanged neutral accounting or source-identity refusal.
        #[source]
        cause: eredu_runtime::working_memory::WorkingMemoryError,
    },
    /// Exact retained equation prerequisite which refused a neural frontier.
    /// Existing paid detail moves from the recipe; this adds no formatted copy.
    #[error("original neural boundary {boundary}, record {record}, {requirement} (operation {operation:?}, detail {detail:?}): {cause}")]
    NeuralBoundarySource {
        /// Selected shared completion worker.
        boundary: &'static str,
        /// Original equation row in the retained request recipe.
        record: usize,
        /// Missing source or inconsistent population, without an allocated label.
        requirement: &'static str,
        /// Actual missing operation ordinal, if the equation producer recorded it.
        operation: Option<usize>,
        /// Previously paid diagnostic; never a fresh source inspection or copy.
        detail: Option<String>,
        /// Unchanged typed source refusal.
        #[source]
        cause: eredu_runtime::working_memory::WorkingMemoryError,
    },
    /// Fixed native text-step attribution; the original neutral refusal stays
    /// inline, with a static call-site and no allocated diagnostic wrapper.
    #[error("original text admission at {at}: {cause}")]
    TextAdmission {
        /// Exact native admission boundary which returned this refusal.
        at: &'static std::panic::Location<'static>,
        /// Unchanged accounting, request or source-identity refusal.
        #[source]
        cause: eredu_runtime::working_memory::WorkingMemoryError,
    },
    /// Fixed retained observation-source validation, without a diagnostic box.
    #[error("{0}")]
    PreparedObservation(#[source] eredu_runtime::PreparedSessionObservationError),
    /// No original Scope accepted; the same prepared role remains retained.
    #[error("original prefill scope could not begin: {0}")]
    PrefillScope(#[source] safemlx::SubmissionScopeOwnerCause),
    /// Reentrant access cannot issue or replace a native prefill role.
    #[error("original prefill scope is already in flight")]
    PrefillScopeReentrant,
    /// Missing or consumed roles cannot fall back to an ordinary Scope.
    #[error("original prefill scope role is unavailable")]
    PrefillScopeUnavailable,
    /// Ordinary capture error and its existing host custody; no completion grant.
    #[error(transparent)]
    OrdinaryCapture(OrdinaryCaptureFailure),
    /// Contradictory private prepared-capture input without ordinary custody.
    /// This inline rejection allocates no error shell or replacement authority.
    #[error("ordinary prepared capture rebinding: {0}")]
    PreparedCaptureRebind(#[source] eredu_core::PreparedControlInputError),
    /// Ordinary storage-source construction retained its original source and
    /// genuine unquoted participant through closed concrete error retirement.
    #[error("{0}")]
    StorageSource(#[source] eredu_core::BackendFailure),
    /// Ordinary pending-input copy failure, retaining the same unquoted owner
    /// through closed concrete error retirement. It certifies no model rollback.
    #[error("{0}")]
    PendingInputCopy(#[source] eredu_core::BackendFailure),
    /// Saved-copy collector construction refused before native entry. The
    /// closed source retains the original allocator cause and copy custody.
    #[error("{0}")]
    SavedCopyConstructor(#[source] eredu_core::BackendFailure),
    /// Fixed saved-copy completion evidence. This is a refusal, never proof
    /// that failed or unobservable native work can release its resources.
    #[error(
        "saved copy completion unavailable (settled={settled}, failed={failed}, blocked={blocked})"
    )]
    SavedCopyCompletion {
        /// Whether the retained native scope established terminal settlement.
        settled: bool,
        /// Whether the same scope reported a native or callback failure.
        failed: bool,
        /// Whether completion remained unobservable or otherwise blocked.
        blocked: bool,
    },
    /// The enclosing originally funded model operation did not prove settlement.
    #[error("original operation completion unavailable (settled={settled}, failed={failed}, blocked={blocked})")]
    OriginalOperationCompletion {
        /// Whether the retained native scope established terminal settlement.
        settled: bool,
        /// Whether the same scope reported a native or callback failure.
        failed: bool,
        /// Whether completion remained unobservable or otherwise blocked.
        blocked: bool,
    },
    /// A consumed original prediction role could not begin its native Scope.
    #[error("original prediction scope could not begin: {0}")]
    PredictionScope(#[source] safemlx::SubmissionScopeOwnerCause),
    /// A promised original prediction role is absent or already consumed.
    #[error("original prediction scope role is unavailable")]
    PredictionScopeUnavailable,
    /// The original role bank is already borrowed by the current operation.
    #[error("original prediction scope role is already in flight")]
    PredictionScopeReentrant,
    /// No Scope was accepted; the original quote retains the same pending attempt.
    #[error("original preparation scope could not begin: {0}")]
    PreparationScope(#[source] safemlx::SubmissionScopeOwnerCause),
    /// The same preparation is currently checked out by an existing call.
    #[error("original preparation scope is already in flight")]
    PreparationScopeReentrant,
    /// A consumed or interrupted role cannot be replaced.
    #[error("original preparation scope role is unavailable")]
    PreparationScopeUnavailable,
    /// A retry cannot replace the originally retained prompt input.
    #[error("original preparation retry input differs from its retained source")]
    PreparationScopeInputMismatch,

    /// The exact first observation failure, including its original source custody.
    #[error(transparent)]
    OutputObservation(OutputObservationFailure),

    /// A prepared sampling result was not a valid borrowed U32 scalar.
    #[error("original sampling scalar storage failed: {0}")]
    OriginalSamplingData(#[source] safemlx::error::AsSliceError),
    /// Preparing the retained native alias for the advanced sampling key failed.
    #[error("original sampling key alias: {0}")]
    OriginalSamplingClone(#[source] safemlx::PreparedArrayCloneCause),

    /// The same output is already being observed by a reentrant call.
    #[error("output observation is already in flight")]
    OutputObservationReentrant,

    /// An earlier output observation unwound without returning a result.
    #[error("output observation was interrupted")]
    OutputObservationInterrupted,

    /// A complete original native operation failure with closed source retirement.
    #[error(transparent)]
    OriginalControl(OriginalControlFailure),

    /// Cold text-run agreement or its selected transport failed.
    #[error("text run preparation failed: {0}")]
    RunPreparation(#[from] eredu_runtime::run_preparation::TextPreparationAgreementError),

    /// Initial manifest agreement failed with its original transport cause.
    #[error("parallel placement error: communication manifest consensus failed: {0}")]
    CommunicationManifestConsensus(
        #[source] Box<eredu_runtime::CommunicationManifestConsensusError<Error>>,
    ),

    /// Rank-local materialization or policy construction failed with its cause.
    #[error("architecture model error: {0}")]
    PartitionedRuntimeConstruction(
        #[source] Box<eredu_runtime::PartitionedRuntimeConstructionError<eredu_nn::Error, Error>>,
    ),

    /// The prepared partition factory failed with its exact handoff error.
    #[error("architecture model error: {0}")]
    PartitionedSessionPreparation(
        #[source] Box<eredu_runtime::PartitionedSessionPreparationError<Error>>,
    ),

    /// Backend capability discovery, preparation, execution, or completion failed.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendError),

    /// Backend-neutral artifact inspection or preparation planning failed.
    #[error(transparent)]
    Artifact(#[from] eredu_core::artifact::ArtifactError),

    /// Portable architecture, artifact, request, and mechanism admission failed.
    #[error(transparent)]
    PreparationAdmission(#[from] eredu_core::PreparationAdmissionError),

    /// The selected realtime session lacks a required capability.
    #[error(transparent)]
    SessionCapability(#[from] eredu_core::SessionCapabilityError),

    /// Invalid backend-neutral cache identity, geometry, or state policy.
    #[error(transparent)]
    CachePolicy(#[from] eredu_core::cache::CachePolicyError),

    /// Invalid reusable prompt-cache identity, schema, or catalog.
    #[error(transparent)]
    PromptCache(#[from] eredu_core::cache::PromptCacheError),

    /// Invalid dense disk streaming configuration or background work.
    #[error(transparent)]
    DenseStream(#[from] crate::backend::runtime::residency::dense_stream::DenseStreamError),

    /// Invalid backend-neutral dense-stream telemetry lifecycle.
    #[error(transparent)]
    DenseStreamTelemetry(#[from] eredu_runtime::DenseStreamTelemetryError),

    /// Invalid backend-neutral immutable-weight residency policy.
    #[error(transparent)]
    WeightResidencyPolicy(#[from] eredu_runtime::WeightResidencyPolicyError),

    /// Invalid backend-neutral dense transfer-window transition.
    #[error(transparent)]
    DenseTransferSchedule(#[from] eredu_runtime::DenseTransferScheduleError),

    /// Invalid composed architecture configuration or state usage.
    #[error("architecture model error: {0}")]
    ArchitectureModel(String),

    /// Exact cold architecture projection failure before source publication.
    #[error("prepared parameter source: {0}")]
    PreparedParameterSource(
        #[source] eredu_architectures::prepared_execution::PreparedExecutionError<eredu_nn::Error>,
    ),

    /// A typed model adapter proved preflight rejection or successful rollback.
    #[error(transparent)]
    ModelStatePreserved(ModelStatePreservedError),

    /// Invalid or failed layerwise model execution.
    #[error(transparent)]
    LayerwiseModel(#[from] crate::backend::runtime::execution::layerwise::LayerwiseModelError),

    /// Invalid backend-neutral execution-group graph.
    #[error(transparent)]
    ExecutionGraph(#[from] eredu_runtime::ExecutionGraphError),

    /// Invalid backend-neutral pipeline execution-group lifecycle.
    #[error(transparent)]
    LayeredPipelineSchedule(#[from] eredu_runtime::LayeredPipelineScheduleError),

    /// Backend-neutral neural operator or parameter-topology failure.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),

    /// Invalid backend-neutral weight binding or offload-unit declaration.
    #[error(transparent)]
    ResidencyDeclaration(#[from] eredu_runtime::ResidencyDeclarationError),

    /// Invalid backend-neutral bounded selection for a weight binding.
    #[error(transparent)]
    WeightBindingSelection(#[from] eredu_runtime::WeightBindingSelectionError),

    /// Invalid backend-neutral ordered weight-residency window.
    #[error(transparent)]
    ResidencyWindow(#[from] eredu_runtime::ResidencyWindowError),

    /// Invalid module-to-checkpoint or resident-lease binding.
    #[error(transparent)]
    ModuleBinding(#[from] crate::backend::runtime::checkpoint::binding::ModuleBindingError),

    /// Backend-neutral checkpoint catalog, mapping, or I/O failure.
    #[error(transparent)]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),

    /// Backend-neutral SafeTensors shard discovery or path admission failed.
    #[error(transparent)]
    CheckpointShards(#[from] eredu_checkpoint::safetensors::SafetensorsShardError),

    /// MLX checkpoint tensor conversion or materialization failure.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),

    /// Invalid checkpoint-derived weight recipe.
    #[error(transparent)]
    WeightRecipe(#[from] crate::backend::runtime::checkpoint::recipe::WeightRecipeError),

    /// Invalid architecture-independent offload planning request.
    #[error(transparent)]
    Offload(#[from] eredu_core::residency::OffloadError),

    /// Invalid or failed weight residency operation.
    #[error(transparent)]
    Residency(#[from] crate::backend::runtime::residency::manager::ResidencyError),

    /// Invalid addressable parameter-bank catalog, capacity, or execution request.
    #[error(transparent)]
    AddressableParameterBank(
        #[from] crate::backend::runtime::residency::parameter_bank::AddressableParameterBankError,
    ),

    /// Invalid runtime parallel topology, tensor placement, or partition request.
    #[error("parallel placement error: {0}")]
    Parallel(String),

    /// Invalid or unsatisfied automatic execution-planning request.
    #[error("automatic planning error: {0}")]
    AutomaticPlanning(String),

    /// Invalid backend-neutral generation configuration or lifecycle state.
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),

    /// Invalid or unsupported checkpoint quantization request.
    #[error("checkpoint quantization error: {0}")]
    Quantization(String),

    /// Media processor configuration or input error.
    #[error("media processor error: {0}")]
    Processor(String),

    /// MLX speculative execution failed.
    #[error("MLX speculative generation failed: {0}")]
    Speculative(String),

    /// Strict weight loading found missing parameters or unused checkpoint tensors.
    #[error("strict weight-load validation failed: {missing_count} missing parameters, {unused_count} unused weights\nmissing:\n{missing}\nunused:\n{unused}", missing_count = .missing.len(), unused_count = .unused.len(), missing = format_keys(.missing), unused = format_keys(.unused))]
    StrictLoadValidation {
        /// Model parameters that were not populated from the checkpoint.
        missing: Vec<String>,
        /// Checkpoint tensors that were not consumed by the model.
        unused: Vec<String>,
    },

    /// Observation policy, coordination, or collection failed with a retained cause.
    #[error("observation failed: {0}")]
    Observation(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// Error reported by the underlying MLX bindings.
    #[error("{0}")]
    Exception(#[from] Exception),

    /// Filesystem I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON configuration deserialization error.
    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    /// Safetensors loading error from `safemlx`.
    #[error(transparent)]
    LoadWeights(#[from] safemlx::error::IoError),

    /// Boxed failure retaining the original error, including source-less leaves.
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    #[track_caller]
    pub(crate) fn text_admission(cause: eredu_runtime::working_memory::WorkingMemoryError) -> Self {
        Self::TextAdmission { at: std::panic::Location::caller(), cause }
    }

    /// Reuse the existing boxed neutral error, if present, without allocating
    /// a second source or discarding custody on other concrete error variants.
    #[track_caller]
    pub(crate) fn at_text_admission(self) -> Self {
        match self {
            Self::Other(source) => match source.downcast::<eredu_runtime::working_memory::WorkingMemoryError>() {
                Ok(cause) => Self::text_admission(*cause),
                Err(source) => Self::Other(source),
            },
            Self::PrefillControl(cause) => Self::text_admission(cause),
            error => error,
        }
    }

    pub(crate) fn at_speculative_stage(self, stage: &'static str) -> Self {
        match self {
            Self::PrefillControl(cause) => Self::SpeculativePrerequisite { stage, cause },
            other => other,
        }
    }

    pub(crate) fn retained_original(
        source: eredu_core::SharedBackendFailure,
        state_preserved: bool,
    ) -> Self {
        Self::RetainedOriginal(RetainedOriginalFailure {
            source,
            state_preserved,
        })
    }
    /// Borrows an existing public classification without allocating or
    /// classifying an unconverted native diagnostic. Closed aliases remain
    /// closed; the temporary shared alias allocates no new error source.
    pub(crate) fn retained_backend_failure_kind(&self) -> Option<eredu_core::BackendFailureKind> {
        match self {
            Self::RetainedOriginal(error) => Some(error.source.retained().into_failure().kind()),
            Self::OriginalControl(error) => Some(error.source.kind()),
            Self::SavedCopyConstructor(error) => Some(error.kind()),
            Self::WorkspacePlanning(error) => Some(
                Self::WorkspacePlanning(*error)
                    .into_backend_failure()
                    .kind(),
            ),
            _ => None,
        }
    }

    /// Transfers only an existing neutral source or the exact inline funding
    /// refusal. Other errors return unchanged for their caller's ordinary path.
    pub(crate) fn take_retained_backend_failure(self) -> Result<eredu_core::BackendFailure, Self> {
        match self {
            Self::Neural(error) => match error.into_metadata_funding_error() {
                Ok(cause) => Ok(cause.into_backend_failure()),
                Err(error) => Err(Self::Neural(error)),
            },
            Self::StorageSource(error) => Ok(error),
            error @ (Self::WorkspacePlanning(_)
            | Self::RetainedOriginal(_)
            | Self::OriginalControl(_)
            | Self::SavedCopyConstructor(_)) => Ok(error.into_backend_failure()),
            error => Err(error),
        }
    }

    pub(crate) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        match self {
            // This is already the retained neutral envelope. Preserve its exact
            // source/custody without allocating a second backend error box.
            Self::StorageSource(error) => error,
            Self::WorkspacePlanning(error) => error.into_backend_failure(),
            Self::RetainedOriginal(error) => error.source.into_failure(),
            Self::OutputObservation(error) => error.into_backend_failure(),
            Self::OriginalControl(error) => error.source,
            Self::SavedCopyConstructor(error) => error,
            ordinary => eredu_core::BackendFailure::from_error(ordinary),
        }
    }

    // Only native original claim/installation composition supplies this already
    // closed source. This creates neither a guard nor an allocation.
    pub(crate) fn with_original_control_source(
        source: eredu_core::BackendFailure,
        state_preserved: bool,
    ) -> Self {
        Self::OriginalControl(OriginalControlFailure {
            source,
            state_preserved,
        })
    }

    pub(crate) fn observation(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Observation(Box::new(error))
    }

    pub(crate) fn before_model_mutation(
        error: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::ModelStatePreserved(ModelStatePreservedError {
            source: error.into(),
        })
    }

    pub(crate) fn after_model_call(
        error: impl Into<Box<dyn std::error::Error + Send + Sync>>,
        before: Option<u64>,
        after: Option<u64>,
    ) -> Self {
        if matches!((before, after), (Some(before), Some(after)) if after > before) {
            Self::before_model_mutation(error)
        } else {
            Self::Other(error.into())
        }
    }

    /// The shared runtime alone establishes whether admission preceded mutable
    /// state access. A rejection witness never substitutes for native completion.
    pub(crate) fn after_replicated_model_call<A, P, M>(
        error: eredu_runtime::ReplicatedTextSessionError<A, P, M>,
        before: Option<u64>,
        after: Option<u64>,
    ) -> Self
    where
        A: std::error::Error + Send + Sync + 'static,
        P: std::error::Error + Send + Sync + 'static,
        M: std::error::Error + Send + Sync + 'static,
    {
        if matches!(
            error,
            eredu_runtime::ReplicatedTextSessionError::BeforeStateMutation(_)
        ) {
            Self::before_model_mutation(error)
        } else {
            Self::after_model_call(error, before, after)
        }
    }

    pub(crate) const fn model_state_preserved(&self) -> bool {
        match self {
            Self::ModelStatePreserved(_) => true,
            Self::OriginalControl(failure) => failure.state_preserved,
            Self::RetainedOriginal(failure) => failure.state_preserved,
            Self::OutputObservation(failure) => failure.state_preserved(),
            Self::OrdinaryCapture(failure) => failure.state_preserved(),
            _ => false,
        }
    }
}

impl From<eredu_runtime::PartitionedRuntimeConstructionError<eredu_nn::Error, Error>> for Error {
    fn from(
        error: eredu_runtime::PartitionedRuntimeConstructionError<eredu_nn::Error, Error>,
    ) -> Self {
        Self::PartitionedRuntimeConstruction(Box::new(error))
    }
}

impl From<eredu_runtime::PartitionedSessionPreparationError<Error>> for Error {
    fn from(error: eredu_runtime::PartitionedSessionPreparationError<Error>) -> Self {
        Self::PartitionedSessionPreparation(Box::new(error))
    }
}

impl From<eredu_checkpoint::validation::StrictLoadFailure> for Error {
    fn from(error: eredu_checkpoint::validation::StrictLoadFailure) -> Self {
        Self::StrictLoadValidation {
            missing: error.missing,
            unused: error.unused,
        }
    }
}

impl From<eredu_checkpoint::recipe::RecipeError> for Error {
    fn from(error: eredu_checkpoint::recipe::RecipeError) -> Self {
        Self::WeightRecipe(
            crate::backend::runtime::checkpoint::recipe::WeightRecipeError::Neutral(error),
        )
    }
}

impl From<eredu_runtime::ParallelPlanError> for Error {
    fn from(error: eredu_runtime::ParallelPlanError) -> Self {
        Self::Parallel(error.to_string())
    }
}

impl From<eredu_checkpoint::Error> for Error {
    fn from(error: eredu_checkpoint::Error) -> Self {
        Self::Quantization(error.to_string())
    }
}

impl From<eredu_core::TopologyError> for Error {
    fn from(error: eredu_core::TopologyError) -> Self {
        Self::Parallel(error.to_string())
    }
}

impl From<eredu_core::scheduler::SchedulerError> for Error {
    fn from(error: eredu_core::scheduler::SchedulerError) -> Self {
        Self::Parallel(error.to_string())
    }
}
