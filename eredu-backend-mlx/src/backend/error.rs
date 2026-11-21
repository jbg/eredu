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

    /// Invalid composed architecture configuration or state usage.
    #[error("architecture model error: {0}")]
    ArchitectureModel(String),

    /// An operation reported a neutral failure with its exact cause.
    #[error("{0}")]
    Operation(#[from] eredu_core::BackendFailure),

    /// Backend-neutral neural operator or parameter-topology failure.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),

    /// Invalid backend-neutral weight binding or offload-unit declaration.
    #[error(transparent)]
    ResidencyDeclaration(
        #[from] eredu_core::residency::parameter_binding::ResidencyDeclarationError,
    ),

    /// Invalid backend-neutral bounded selection for a weight binding.
    #[error(transparent)]
    WeightBindingSelection(
        #[from] eredu_core::residency::parameter_binding::WeightBindingSelectionError,
    ),

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

    /// The MLX build omits distributed model execution.
    #[error("distributed MLX model execution requires the Cargo feature `distributed`")]
    DistributedModelExecutionDisabled,

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
    pub(crate) fn observation(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Observation(Box::new(error))
    }

    pub(crate) fn before_model_mutation(
        error: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        eredu_core::BackendFailure::from_boxed_error(error.into())
            .with_model_state_preserved(true)
            .into()
    }

    pub(crate) const fn model_state_preserved(&self) -> bool {
        matches!(self, Self::Operation(error) if error.model_state_preserved())
    }
}

impl From<eredu_core::parameters::ParameterError> for Error {
    fn from(error: eredu_core::parameters::ParameterError) -> Self {
        eredu_nn::Error::backend_source(error).into()
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

impl From<eredu_core::speculative::SpeculativeDriverError<Exception>> for Error {
    fn from(error: eredu_core::speculative::SpeculativeDriverError<Exception>) -> Self {
        Self::Exception(Exception::from_source(error))
    }
}
