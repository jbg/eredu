use safemlx::error::Exception;

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

    /// Error reported by the underlying MLX bindings.
    #[error(transparent)]
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

    /// Boxed error used for third-party loader failures.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
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
