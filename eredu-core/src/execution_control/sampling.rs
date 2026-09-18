//! Closed sampling-control contract over the existing native generation state.
use crate::{ModelRuntime, TextGenerationBackend};
use serde::{Deserialize, Serialize};

/// Supported changes to future sampling only. Omitting `reseed` retains the exact
/// inherited RNG stream. Sampler strategy, adaptive counters, penalties and token
/// history remain unchanged; incompatible strategy transitions are not exposed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingOverride {
    /// New temperature; zero selects greedy standard sampling.
    pub temperature: Option<f32>,
    /// Explicit new native RNG seed. Does not reset adaptive or penalty history.
    pub reseed: Option<u64>,
}

/// Native facts consumed by shared validation; contains no mutable native handle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplingStateFacts {
    /// Current effective temperature.
    pub temperature: f32,
    /// The retained sampler requires strictly positive temperature (Mirostat).
    pub requires_positive_temperature: bool,
    /// An exact resumable RNG stream exists, including while temporarily greedy.
    pub has_rng: bool,
}

/// Invalid policy is rejected before any native operation or state change.
#[derive(Debug, thiserror::Error)]
pub enum SamplingOverrideError<E: std::error::Error + 'static> {
    /// Invalid or incompatible temperature/RNG policy.
    #[error("invalid sampling override: {0}")]
    Invalid(&'static str),
    /// Native replacement preparation failed, preserving the prior logical state.
    #[error("native sampling override failed: {0}")]
    Backend(#[source] E),
}

/// A completed-boundary sampling operation. Implementations validate policy
/// through the shared runtime worker, admit all replacement/future work before
/// mutation, and preserve the previous logical sampling state on error.
pub trait TextSamplingControlBackend: TextGenerationBackend {
    /// Read-only facts from the installed sampling state.
    fn sampling_control_facts(state: &Self::TextGenerationState) -> SamplingStateFacts;
    /// Changes future sampling without submitting a model prediction, consuming
    /// an RNG draw, resetting history or weakening the retained input/controller
    /// admission. Success returns the actual installed facts.
    fn apply_sampling_override(
        runtime: &mut ModelRuntime<Self>,
        state: &mut Self::TextGenerationState,
        context: Option<&crate::TextStepContext>,
        request: SamplingOverride,
    ) -> Result<SamplingStateFacts, SamplingOverrideError<Self::Error>>;
}
