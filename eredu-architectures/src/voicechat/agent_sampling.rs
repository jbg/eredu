//! Released VoiceChat agent-text sampling over shared backend mechanisms.
//!
//! Selection precedes function/turn overrides. The caller records the resulting
//! effective feedback with [`AgentSampler::accept_effective`]; the raw selection
//! does not enter repetition history on its own. Randomness belongs to the shared
//! decision driver and its unpublished generation branch.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{SnapshotReservation, SnapshotValue};
use eredu_runtime::{PenaltyConfig, Sampler, SamplingBackend, TokenDomain};

use super::channels::{TEXT_BOS_ID, TEXT_EOS_ID, TEXT_PAD_ID, TEXT_VOCABULARY_SIZE as VOCABULARY};

/// Explicit released agent-text controls and finite feedback-history capacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentSamplingPolicy {
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    max_history: usize,
}

impl AgentSamplingPolicy {
    /// Creates explicit overrides for the released sampling policy.
    pub fn new(
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        max_history: usize,
    ) -> Result<Self, AgentSamplingError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(AgentSamplingError::Configuration(
                "temperature must be finite and nonnegative",
            ));
        }
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(AgentSamplingError::Configuration(
                "top-p must be finite and in [0, 1]",
            ));
        }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
            return Err(AgentSamplingError::Configuration(
                "repetition penalty must be finite and positive",
            ));
        }
        if max_history == 0 {
            return Err(AgentSamplingError::Configuration(
                "history capacity must be positive",
            ));
        }
        Ok(Self {
            temperature,
            top_p,
            repetition_penalty,
            max_history,
        })
    }

    /// Uses the pinned streaming configuration: temperature zero, nucleus mass
    /// one, and repetition multiplier one. Selection is greedy and consumes no
    /// randomness. History capacity remains an explicit admission choice.
    pub fn streaming_defaults(max_history: usize) -> Result<Self, AgentSamplingError> {
        Self::new(0.0, 1.0, 1.0, max_history)
    }

    /// Temperature supplied to the shared decision driver.
    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    /// Nucleus mass; one disables truncation.
    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    /// Sign-dependent repetition multiplier, applied once per distinct token.
    pub const fn repetition_penalty(self) -> f32 {
        self.repetition_penalty
    }

    /// Maximum effective feedback tokens retained before typed capacity failure.
    pub const fn max_history(self) -> usize {
        self.max_history
    }
}

/// One stream's effective agent feedback, independent of its function channel.
///
/// Each sampling invocation receives exactly one prediction row; batched streams
/// retain separate policies and histories.
///
/// Clone this alongside the shared branch RNG before tentative execution. A
/// failed or cancelled branch must not publish either the selected token or its
/// advanced random state. Prompt prefill tokens are not generated feedback.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentSampler {
    policy: AgentSamplingPolicy,
    history: Vec<u32>,
}

impl SnapshotValue for AgentSamplingPolicy {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = std::mem::size_of::<Self>() as u64;
        Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(*self)
    }
}

impl SnapshotValue for AgentSampler {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = (self.history.capacity() as u64)
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(|| {
                BackendFailure::from_error(std::io::Error::other(
                    "VoiceChat agent history snapshot estimate overflows",
                ))
            })?;
        Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut history = Vec::new();
        history
            .try_reserve_exact(self.history.len())
            .map_err(BackendFailure::from_error)?;
        history.extend_from_slice(&self.history);
        Ok(Self {
            policy: self.policy,
            history,
        })
    }
}

impl AgentSampler {
    /// Begins with no generated feedback.
    pub fn new(policy: AgentSamplingPolicy) -> Self {
        Self {
            policy,
            history: Vec::new(),
        }
    }

    /// Retained immutable sampling policy.
    pub const fn policy(&self) -> AgentSamplingPolicy {
        self.policy
    }

    /// Effective feedback after any function-channel or turn-taking override.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Restores complete effective history, validating before replacing state.
    pub fn set_history(&mut self, history: &[u32]) -> Result<(), AgentSamplingError> {
        if history.len() > self.policy.max_history {
            return Err(AgentSamplingError::HistoryCapacity {
                capacity: self.policy.max_history,
            });
        }
        for &token in history {
            validate_id(token)?;
        }
        self.history = history.to_vec();
        Ok(())
    }

    /// Rewrites the interrupted position's effective feedback without consuming
    /// another history slot. Cache positions and the retained random stream do
    /// not change when host publication ends a conversational turn.
    pub(crate) fn reset_last_effective_for_interruption(&mut self) {
        if let Some(last) = self.history.last_mut() {
            *last = super::channels::TEXT_EOS_ID as u32;
        }
    }

    /// Checks capacity before model invocation or random draws for the next step.
    pub fn preflight(&self) -> Result<(), AgentSamplingError> {
        if self.history.len() >= self.policy.max_history {
            return Err(AgentSamplingError::HistoryCapacity {
                capacity: self.policy.max_history,
            });
        }
        Ok(())
    }

    /// Records one effective feedback token after raw selection and overrides.
    /// PAD/BOS/EOS count toward capacity but never receive repetition penalties.
    pub fn accept_effective(&mut self, token: u32) -> Result<(), AgentSamplingError> {
        self.preflight()?;
        validate_id(token)?;
        self.history.push(token);
        Ok(())
    }
}

impl<B: SamplingBackend> Sampler<B> for AgentSampler {
    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        self.preflight()
            .map_err(|error| B::error(error.to_string()))?;
        if temperature != self.policy.temperature {
            return Err(B::error(
                "VoiceChat driver temperature differs from retained agent policy".into(),
            ));
        }

        // This original-logit decision must precede penalties and filters. A
        // special winner bypasses all processing and consumes no random draw.
        let greedy = B::sample_raw(logits, 0.0, None, context)?;
        let greedy = B::validate_token(&greedy, TokenDomain::new(VOCABULARY), context)?;
        if temperature == 0.0
            || (temperature == 1.0
                && self.policy.top_p == 1.0
                && self.policy.repetition_penalty == 1.0)
        {
            return Ok(greedy);
        }
        if is_special(B::token_id(&greedy, context)?) {
            return Ok(greedy);
        }
        let random = random.ok_or_else(|| {
            B::error("VoiceChat stochastic agent selection requires retained random state".into())
        })?;

        let unique = self
            .history
            .iter()
            .copied()
            .filter(|&token| !is_special(token))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let penalties = PenaltyConfig {
            repeat_penalty: self.policy.repetition_penalty,
            repeat_last_n: -1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };
        let processed = B::apply_penalties(logits, &unique, penalties, context)?;
        // VoiceChat truncates probabilities after temperature scaling. General
        // GenerationSampler ordering is intentionally not substituted here.
        let processed = if temperature == 1.0 {
            processed
        } else {
            B::scale_temperature(&processed, temperature, context)?
        };
        let processed = B::apply_top_p(processed, self.policy.top_p, context)?;
        let selected = B::sample_processed(&processed, 1.0, Some(random), context)?;
        B::validate_token(&selected, TokenDomain::new(VOCABULARY), context)
    }
}

fn is_special(token: u32) -> bool {
    [TEXT_BOS_ID as u32, TEXT_EOS_ID as u32, TEXT_PAD_ID as u32].contains(&token)
}

fn validate_id(token: u32) -> Result<(), AgentSamplingError> {
    if token as usize >= VOCABULARY {
        Err(AgentSamplingError::Token { token })
    } else {
        Ok(())
    }
}

/// Portable policy and retained-history validation failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AgentSamplingError {
    /// Sampling controls do not describe a valid released-policy override.
    #[error("invalid VoiceChat agent sampling: {0}")]
    Configuration(&'static str),
    /// History is full; no earlier feedback is silently evicted.
    #[error("VoiceChat effective agent history capacity {capacity} is exhausted")]
    HistoryCapacity {
        /// Maximum retained effective feedback tokens.
        capacity: usize,
    },
    /// Effective feedback lies outside the admitted released vocabulary.
    #[error("VoiceChat agent feedback token {token} is outside the released vocabulary")]
    Token {
        /// Invalid feedback token.
        token: u32,
    },
}
