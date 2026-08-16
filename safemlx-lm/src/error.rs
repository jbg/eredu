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
/// Error type used by `safemlx-lm` loaders and tokenizer helpers.
pub enum Error {
    /// Invalid dense disk streaming configuration or background work.
    #[error(transparent)]
    DenseStream(#[from] crate::runtime::residency::dense_stream::DenseStreamError),

    /// Invalid unified Llama model configuration or cache usage.
    #[error(transparent)]
    LlamaModel(#[from] crate::architectures::llama::layerwise::LlamaModelError),

    /// Invalid or failed layerwise model execution.
    #[error(transparent)]
    LayerwiseModel(#[from] crate::runtime::execution::layerwise::LayerwiseModelError),

    /// Invalid module-to-checkpoint or resident-lease binding.
    #[error(transparent)]
    ModuleBinding(#[from] crate::runtime::checkpoint::binding::ModuleBindingError),

    /// Persistent checkpoint catalog, mapping, or materialization failure.
    #[error(transparent)]
    WeightStore(#[from] crate::runtime::checkpoint::store::WeightStoreError),

    /// Invalid checkpoint-derived weight recipe.
    #[error(transparent)]
    WeightRecipe(#[from] crate::runtime::checkpoint::recipe::WeightRecipeError),

    /// Invalid architecture-independent offload planning request.
    #[error(transparent)]
    Offload(#[from] crate::runtime::residency::policy::OffloadError),

    /// Invalid or failed weight residency operation.
    #[error(transparent)]
    Residency(#[from] crate::runtime::residency::manager::ResidencyError),

    /// Invalid sparse expert catalog, routing, capacity, or execution request.
    #[error(transparent)]
    ExpertCache(#[from] crate::runtime::residency::expert_cache::ExpertCacheError),

    /// Invalid runtime parallel topology, tensor placement, or partition request.
    #[error("parallel placement error: {0}")]
    Parallel(String),

    /// Invalid or unsatisfied automatic execution-planning request.
    #[error("automatic planning error: {0}")]
    AutomaticPlanning(String),

    /// Invalid checkpoint or request-level text generation settings.
    #[error("generation configuration error: {0}")]
    GenerationConfig(String),

    /// Invalid or unsupported checkpoint quantization request.
    #[error("checkpoint quantization error: {0}")]
    Quantization(String),

    /// The `model_type` in `config.json` is not supported by this crate.
    #[error("unsupported model type: {0}")]
    UnsupportedModelType(String),

    /// The model family is recognized but this specific architecture is unsupported.
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Media processor configuration or input error.
    #[error("media processor error: {0}")]
    Processor(String),

    /// Embedded GGUF tokenizer metadata is invalid or cannot be reconstructed.
    #[error("GGUF tokenizer error: {0}")]
    GgufTokenizer(String),

    /// The loaded checkpoint does not provide a chat template.
    #[error("the loaded model does not provide a chat template")]
    MissingChatTemplate,

    /// A native tool definition or its generation grammar is invalid.
    #[error("native tool constraint error: {0}")]
    ToolConstraint(String),

    /// An ordinary prepared-chat generation request or runtime failed.
    #[error("prepared chat generation error: {0}")]
    PreparedChatGeneration(String),

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

    /// Chat-template or tokenizer utility error.
    #[error(transparent)]
    Template(#[from] safemlx_lm_utils::error::Error),

    /// Boxed error used for third-party loader failures.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl From<safemlx_lm_core::scheduler::SchedulerError> for Error {
    fn from(error: safemlx_lm_core::scheduler::SchedulerError) -> Self {
        Self::Parallel(error.to_string())
    }
}
