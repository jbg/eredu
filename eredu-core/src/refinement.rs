//! Portable policy and progress records for iterative token-block generation.

use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

/// Initial token positions and immutable positions of an iterative canvas.
/// Missing seed tokens receive ordinary full-domain noise. Pins require a seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementCanvas {
    /// Complete canvas, in model token IDs.
    pub seed_tokens: Vec<Option<u32>>,
    /// Unique zero-based positions held fixed throughout refinement.
    pub pinned_positions: Vec<u32>,
}

impl RefinementCanvas {
    /// Validates geometry and identity before backend allocation or random draws.
    pub fn validate(&self, length: u32, vocabulary: usize) -> Result<(), RefinementPolicyError> {
        let pins: std::collections::BTreeSet<_> = self.pinned_positions.iter().copied().collect();
        if self.seed_tokens.len() != length as usize
            || self
                .seed_tokens
                .iter()
                .flatten()
                .any(|id| *id as usize >= vocabulary)
            || pins.len() != self.pinned_positions.len()
            || pins.iter().any(|p| {
                self.seed_tokens
                    .get(*p as usize)
                    .is_none_or(Option::is_none)
            })
        {
            return Err(RefinementPolicyError::Canvas);
        }
        Ok(())
    }

    /// Immutable host storage retained by request and snapshot owners.
    pub fn host_bytes(&self) -> u64 {
        (self.seed_tokens.len() * std::mem::size_of::<Option<u32>>()
            + self.pinned_positions.len() * std::mem::size_of::<u32>()) as u64
    }
}

/// Validated entropy-bounded refinement controls, independent of model geometry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "EntropyRefinementOptions",
    into = "EntropyRefinementOptions"
)]
pub struct EntropyRefinementConfig {
    options: EntropyRefinementOptions,
}
// All floating fields are finite after construction and deserialization.
impl Eq for EntropyRefinementConfig {}

/// Request or artifact values accepted by entropy-bounded refinement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EntropyRefinementOptions {
    /// Maximum decoder invocations for each block.
    pub max_denoising_steps: u32,
    /// Initial logit temperature.
    pub t_max: f32,
    /// Lower endpoint of the linear temperature schedule.
    pub t_min: f32,
    /// Joint mutual-information upper bound used for position acceptance.
    pub entropy_bound: f32,
    /// Mean entropy must be strictly below this threshold for early stopping.
    pub confidence_threshold: f32,
    /// Number of preceding identical argmax predictions required for stopping.
    pub stability_threshold: u32,
}

impl Default for EntropyRefinementOptions {
    fn default() -> Self {
        Self {
            max_denoising_steps: 48,
            t_max: 0.8,
            t_min: 0.4,
            entropy_bound: 0.1,
            confidence_threshold: 0.005,
            stability_threshold: 1,
        }
    }
}

/// A refinement request rejected before execution or random draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RefinementPolicyError {
    /// Seed geometry, token identities or pinned positions are invalid.
    #[error(
        "refinement canvas requires exact geometry, admitted token IDs and unique seeded pins"
    )]
    Canvas,
    /// No decoder invocation is allowed.
    #[error("refinement requires at least one denoising step")]
    ZeroSteps,
    /// Temperatures must be finite, positive and ordered.
    #[error("refinement temperatures must be finite and satisfy 0 < t_min <= t_max")]
    Temperature,
    /// Entropy thresholds must be finite and strictly positive.
    #[error("refinement entropy thresholds must be finite and positive")]
    Entropy,
    /// The requested iteration is outside the selected schedule.
    #[error("refinement iteration lies outside the selected schedule")]
    Iteration,
    /// Confidence thresholds lie in the probability interval.
    #[error("masked refinement confidence must be finite and lie in [0, 1]")]
    Confidence,
    /// Masked sampling allows greedy selection or finite positive noise.
    #[error("masked refinement temperature must be finite and nonnegative")]
    MaskedTemperature,
}

impl TryFrom<EntropyRefinementOptions> for EntropyRefinementConfig {
    type Error = RefinementPolicyError;

    fn try_from(options: EntropyRefinementOptions) -> Result<Self, Self::Error> {
        if options.max_denoising_steps == 0 {
            return Err(RefinementPolicyError::ZeroSteps);
        }
        if !options.t_min.is_finite()
            || !options.t_max.is_finite()
            || options.t_min <= 0.0
            || options.t_max < options.t_min
        {
            return Err(RefinementPolicyError::Temperature);
        }
        if [options.entropy_bound, options.confidence_threshold]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(RefinementPolicyError::Entropy);
        }
        Ok(Self { options })
    }
}

impl From<EntropyRefinementConfig> for EntropyRefinementOptions {
    fn from(config: EntropyRefinementConfig) -> Self {
        config.options
    }
}

impl Default for EntropyRefinementConfig {
    fn default() -> Self {
        EntropyRefinementOptions::default()
            .try_into()
            .expect("the default refinement policy is valid")
    }
}

impl EntropyRefinementConfig {
    /// Returns the immutable validated controls.
    pub const fn options(self) -> EntropyRefinementOptions {
        self.options
    }

    /// Maximum decoder invocations for each block.
    pub const fn steps(self) -> NonZeroU32 {
        // Construction and deserialization both validate this invariant.
        NonZeroU32::new(self.options.max_denoising_steps).unwrap()
    }

    /// Temperature for a zero-based completed-step count.
    ///
    /// The remaining-step schedule uses N, N-1, ..., 1. Its final invocation
    /// therefore remains one schedule increment above the lower endpoint.
    pub fn temperature(self, completed: u32) -> Result<f32, RefinementPolicyError> {
        let remaining = self
            .options
            .max_denoising_steps
            .checked_sub(completed)
            .filter(|remaining| *remaining > 0)
            .ok_or(RefinementPolicyError::Iteration)?;
        Ok(self.options.t_min
            + (self.options.t_max - self.options.t_min)
                * (remaining as f32 / self.options.max_denoising_steps as f32))
    }
}

/// Numerical selection used while unmasking a tentative block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskedSampling {
    /// Select `argmax(exp(logits) / (-ln(U))^temperature)` in float64.
    GumbelPower,
    /// Categorical selection from temperature-scaled position scores.
    Categorical,
}

/// Request values for confidence-based masked-token unmasking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaskedRefinementOptions {
    /// Maximum numerical invocations for one block.
    pub max_denoising_steps: u32,
    /// Zero selects greedy tokens; positive values control proposal noise.
    pub temperature: f32,
    /// Absent uses the even transfer schedule; present forces at least one transfer.
    pub confidence_threshold: Option<f32>,
    /// Explicit position-selection equations.
    pub sampling: MaskedSampling,
}

impl Default for MaskedRefinementOptions {
    fn default() -> Self {
        Self {
            max_denoising_steps: 32,
            temperature: 0.0,
            confidence_threshold: None,
            sampling: MaskedSampling::GumbelPower,
        }
    }
}

/// Validated unmasking policy, independent of the architecture's mask identity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MaskedRefinementOptions", into = "MaskedRefinementOptions")]
pub struct MaskedRefinementConfig {
    options: MaskedRefinementOptions,
}
// Validation excludes NaN, preserving reflexive equality for retained policies.
impl Eq for MaskedRefinementConfig {}

impl TryFrom<MaskedRefinementOptions> for MaskedRefinementConfig {
    type Error = RefinementPolicyError;
    fn try_from(options: MaskedRefinementOptions) -> Result<Self, Self::Error> {
        if options.max_denoising_steps == 0 {
            return Err(RefinementPolicyError::ZeroSteps);
        }
        if !options.temperature.is_finite() || options.temperature < 0.0 {
            return Err(RefinementPolicyError::MaskedTemperature);
        }
        if options
            .confidence_threshold
            .is_some_and(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
        {
            return Err(RefinementPolicyError::Confidence);
        }
        Ok(Self { options })
    }
}

impl From<MaskedRefinementConfig> for MaskedRefinementOptions {
    fn from(config: MaskedRefinementConfig) -> Self {
        config.options
    }
}

impl Default for MaskedRefinementConfig {
    fn default() -> Self {
        MaskedRefinementOptions::default()
            .try_into()
            .expect("valid masked defaults")
    }
}

impl MaskedRefinementConfig {
    /// Complete validated numerical controls.
    pub const fn options(self) -> MaskedRefinementOptions {
        self.options
    }

    /// Even transfer count with remainder assigned to the earliest invocations.
    pub fn transfer_tokens(
        self,
        initial_masked: u32,
        completed: u32,
    ) -> Result<u32, RefinementPolicyError> {
        let steps = self.options.max_denoising_steps;
        if completed >= steps {
            return Err(RefinementPolicyError::Iteration);
        }
        Ok(initial_masked / steps + u32::from(completed < initial_masked % steps))
    }
}

/// Independent numerical policies sharing one refinement lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", content = "options", rename_all = "snake_case")]
pub enum RefinementPolicy {
    /// Entropy-bounded feedback and final argmax selection.
    EntropyFeedback(EntropyRefinementConfig),
    /// Confidence-based unmasking; selected tokens remain fixed.
    MaskedUnmasking {
        /// Validated schedule and numerical selection.
        config: MaskedRefinementConfig,
        /// Architecture-admitted mask-token identity.
        mask_token_id: u32,
    },
}

impl RefinementPolicy {
    /// Maximum number of preceding token canvases retained by the numerical policy.
    pub const fn stability_history_length(self) -> u32 {
        match self {
            Self::EntropyFeedback(config) => config.options().stability_threshold,
            Self::MaskedUnmasking { .. } => 0,
        }
    }
}

impl From<EntropyRefinementConfig> for RefinementPolicy {
    fn from(config: EntropyRefinementConfig) -> Self {
        Self::EntropyFeedback(config)
    }
}

impl Default for RefinementPolicy {
    fn default() -> Self {
        Self::EntropyFeedback(EntropyRefinementConfig::default())
    }
}

impl TryFrom<EntropyRefinementOptions> for RefinementPolicy {
    type Error = RefinementPolicyError;
    fn try_from(options: EntropyRefinementOptions) -> Result<Self, Self::Error> {
        Ok(EntropyRefinementConfig::try_from(options)?.into())
    }
}

/// Meaning of a reported prediction score.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionScoreKind {
    /// Probability conditioned on the committed causal prefix.
    #[default]
    CausalConditional,
    /// Position score conditioned on a complete tentative block.
    RefinementPosition,
    /// Refinement score restricted by the evolving committed output constraints.
    ConstrainedFinalization,
}

/// Physical operation completed by a generation advancement.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPhase {
    /// Encode prompt or authoritative committed context.
    Context,
    /// Revise tentative candidates without changing retained context.
    Refinement,
    /// Finalize and commit one candidate position.
    Finalization,
}

/// Independent execution and output coordinates of an iterative generation run.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GenerationPosition {
    /// Completed physical advancement count.
    pub execution_step: u64,
    /// Current block index.
    pub block: u64,
    /// Completed refinement invocations in the current block.
    pub iteration: u32,
    /// Total authoritative generated token commitments.
    pub committed_tokens: u64,
}

/// A replaceable tentative block, separate from committed semantic output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftRevision {
    /// Independent branch identity.
    pub branch: u64,
    /// Monotone output reconciliation epoch, never restored backwards.
    pub epoch: u64,
    /// Block containing these tentative positions.
    pub block: u64,
    /// Revision within the current epoch and block.
    pub revision: u32,
    /// Generated-token position at which the block begins.
    pub committed_start: u64,
    /// Complete replacement token IDs for the active draft.
    pub token_ids: Vec<u32>,
}

/// Tentative-output lifecycle. These events never enter the committed parser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DraftEvent {
    /// Replace the entire active tentative block.
    Updated {
        /// Complete revision and identity.
        draft: DraftRevision,
    },
    /// Remove the active tentative block from this branch and epoch.
    Cleared {
        /// Independent branch identity.
        branch: u64,
        /// Monotone reconciliation epoch.
        epoch: u64,
        /// Block whose draft is removed.
        block: u64,
    },
}
