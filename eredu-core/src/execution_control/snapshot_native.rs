//! Native storage observations and independent value copies for snapshots.

use crate::{
    capture::CaptureError, intervention::InterventionEstimator, sampling::SeededRandomKey,
    TextGenerationBackend,
};
use std::sync::Arc;

/// Native allocation bounds for the stored values of a sampler.
pub struct SamplingGrowthCosts {
    /// Retained size of one random key.
    pub random_key_bytes: u64,
    /// Upper-bound non-payload bytes for an additional retained array.
    pub array_overhead: u64,
}

/// Storage facts and value copies for an already admitted execution scope.
pub trait TextSnapshotNative: TextGenerationBackend {
    /// Reports physical facts for an owned prompt tensor.
    fn prompt_tensor_identity(tensor: &Self::InputTensor) -> Option<crate::InputTensorIdentity>;

    /// Bounds native storage of one retained prompt tensor.
    fn prompt_tensor_bytes(tensor: &Self::InputTensor) -> Option<u64>;

    /// Bounds native storage of one scalar decode token.
    fn token_bytes(token: &Self::Token) -> Option<u64>;

    /// Copies one scalar decode token while the portable scope is active.
    fn copy_token(backend: &Self, token: &Self::Token) -> Result<Self::Token, Self::Error>;

    /// Copies one prompt tensor while the portable scope is active.
    fn copy_prompt_tensor(
        backend: &Self,
        tensor: &Self::InputTensor,
    ) -> Result<Self::InputTensor, Self::Error>;

    /// Settles immutable prompt tensors before their storage is shared.
    fn complete_prompt_tensors<'a>(
        tensors: impl Iterator<Item = &'a Self::InputTensor>,
    ) -> Result<(), Self::Error>
    where
        Self::InputTensor: 'a;
}

/// Mode-specific storage for advanced continuations.
pub trait TextModeSnapshotNative: TextSnapshotNative + SeededRandomKey {
    /// Native causal score row.
    type CausalScores;
    /// Native token block.
    type RefinementTokens;
    /// Native position scores.
    type RefinementScores;
    /// Native boolean position mask.
    type RefinementMask;

    /// Reports native allocation bounds and retained sampler geometry for continuation growth.
    fn sampling_growth_costs() -> SamplingGrowthCosts;

    /// Materializes and settles the supplied subkey indices.
    fn split_random(
        backend: &Self,
        root: &Self::Key,
        indices: [usize; 2],
    ) -> Result<[Self::Key; 2], Self::Error>;
    /// Native storage bound for a random key.
    fn random_bytes(random: &Self::Key) -> Option<u64>;
    /// Native storage bound for retained next-token scores.
    fn score_bytes(scores: &Self::CausalScores) -> Option<u64>;
    /// Native storage bound for a token block.
    fn refinement_token_bytes(value: &Self::RefinementTokens) -> Option<u64>;
    /// Native storage bound for position scores.
    fn refinement_score_bytes(value: &Self::RefinementScores) -> Option<u64>;
    /// Native storage bound for a position mask.
    fn refinement_mask_bytes(value: &Self::RefinementMask) -> Option<u64>;
    /// Deep copies one independent random key.
    fn copy_random(backend: &Self, random: &Self::Key) -> Result<Self::Key, Self::Error>;
    /// Deep copies one retained next-token score tensor.
    fn copy_scores(
        backend: &Self,
        scores: &Self::CausalScores,
    ) -> Result<Self::CausalScores, Self::Error>;

    /// Native workspace facts for child-capture admission.
    fn child_capture_workspace() -> crate::capture::CaptureWorkspaceModel;
    /// Native intervention resource estimator.
    fn child_intervention_estimator() -> Result<Arc<dyn InterventionEstimator>, CaptureError>;
}
