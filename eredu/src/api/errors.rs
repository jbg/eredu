//! Portable errors for device selection and diagnostic workflows.

use eredu_core::BackendFailure;

/// Failure to map a facade device choice to the selected local backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum DevicePlanError {
    /// This build contains no native accelerator family for the selected adapter.
    #[error("no local accelerator family is compiled for this target")]
    AcceleratorNotCompiled,
}

/// Failure while running the facade-owned expert-cache benchmark workflow.
#[derive(Debug, thiserror::Error)]
pub enum ExpertCacheBenchmarkError {
    /// The benchmark needs a non-empty prompt for prefill and cached decode.
    #[error("expert-cache benchmark requires at least one prompt token")]
    EmptyPrompt,
    /// The selected model does not expose sparse expert-cache telemetry.
    #[error("sparse expert-cache benchmark requires an expert-cache model")]
    ExpertCacheUnavailable,
    /// The local rank did not produce logits needed to complete a benchmark phase.
    #[error("expert-cache benchmark requires logits on the local rank")]
    LogitsUnavailable,
    /// The selected backend failed while preparing or executing the benchmark.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}
