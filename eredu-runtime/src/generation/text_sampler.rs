mod workspace_projection;
pub use workspace_projection::{SamplerProjectionError, SamplerWorkspaceProjection};

use super::*;
use eredu_core::{TextGenerationConfig, TextSamplingStrategy};

/// Configured sampling policy shared by native execution and metadata inspection.
#[derive(Debug)]
pub enum ConfiguredTextSampler {
    /// Standard filters and history penalties.
    Standard(GenerationSampler),
    /// Adaptive surprise cutoff with its retained mutable state.
    MirostatV2(MirostatV2Sampler),
}

/// Invalid extent for independently copying a configured sampler's host payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SamplerCopyError {
    /// The boxed history or aggregate managed payload exceeds its size type.
    #[error("sampler copy payload exceeds the supported host extent")]
    Overflow,
}

/// Checked copy of one borrowed sampler, including its complete history box.
///
/// This move-only plan holds the source immutably until consumed or dropped, so
/// neither its controls nor its history can change between inspection and copy.
/// It performs no allocation during preparation and grants no memory funding.
#[derive(Debug)]
#[must_use = "inspect the copy cost, then consume the plan under the caller's allocation authority"]
pub struct SamplerCopyPlan<'a> {
    source: &'a ConfiguredTextSampler,
    history: history::TokenHistoryCopy<'a>,
    retained_bytes: u64,
}

impl SamplerCopyPlan<'_> {
    /// Number of accepted tokens preserved by the copy.
    pub fn history_len(&self) -> usize {
        self.source.history_len()
    }

    /// Exact destination box length, including cleared or unused history slots.
    pub fn history_capacity(&self) -> usize {
        self.source.history_capacity()
    }

    /// Exact managed U32 destination payload, excluding the inline descriptor.
    pub fn history_bytes(&self) -> u64 {
        self.history.bytes()
    }

    /// Managed inline sampler plus boxed history payload only. This excludes
    /// allocator metadata, native RNG/input/state, and any enclosing owner.
    pub fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    /// Copies the fixed-size history box and all standard/adaptive controls.
    /// The caller must establish destination allocation authority first; this
    /// operation does not reserve storage or certify a complete snapshot.
    pub fn copy(self) -> ConfiguredTextSampler {
        let history = self.history.copy();
        match self.source {
            ConfiguredTextSampler::Standard(sampler) => {
                ConfiguredTextSampler::Standard(sampler.copy_with_history(history))
            }
            ConfiguredTextSampler::MirostatV2(sampler) => {
                let MirostatV2Sampler {
                    tau,
                    eta,
                    mu,
                    penalties,
                } = sampler;
                ConfiguredTextSampler::MirostatV2(MirostatV2Sampler {
                    tau: *tau,
                    eta: *eta,
                    mu: *mu,
                    penalties: penalties.copy_with_history(history),
                })
            }
        }
    }
}

impl GenerationSampler {
    pub(super) fn copy_with_history(&self, history: history::TokenHistory) -> Self {
        let Self {
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            frequency_penalty,
            presence_penalty,
            generated_tokens: _,
        } = self;
        Self {
            top_k: *top_k,
            top_p: *top_p,
            min_p: *min_p,
            repeat_penalty: *repeat_penalty,
            repeat_last_n: *repeat_last_n,
            frequency_penalty: *frequency_penalty,
            presence_penalty: *presence_penalty,
            generated_tokens: history,
        }
    }
}

impl Clone for ConfiguredTextSampler {
    fn clone(&self) -> Self {
        self.prepare_copy()
            .expect("existing sampler has a representable managed payload extent")
            .copy()
    }
}

impl ConfiguredTextSampler {
    /// Borrows exact policy and history extents without cloning or reading the
    /// history payload. The projection grants no allocation or run authority.
    pub fn workspace_projection(&self) -> SamplerWorkspaceProjection<'_> {
        SamplerWorkspaceProjection::new(self)
    }

    /// Compares only static sampler policy. A future output allowance, seed,
    /// temperature/RNG, and inference limits require their enclosing checks;
    /// adaptive mu and accepted history are retained state, never reset here.
    pub(crate) fn matches_config_policy(&self, config: TextGenerationConfig) -> bool {
        let sampling = config.sampling();
        let penalties = PenaltyConfig {
            repeat_penalty: sampling.repetition_penalty,
            repeat_last_n: sampling.repeat_last_n,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
        };
        match (self, config.strategy()) {
            (Self::Standard(actual), TextSamplingStrategy::Standard) => {
                actual.top_k == sampling.top_k
                    && actual.top_p == sampling.top_p
                    && actual.min_p == sampling.min_p
                    && actual.penalty_config() == penalties
            }
            (Self::MirostatV2(actual), TextSamplingStrategy::MirostatV2 { tau, eta }) => {
                actual.tau == tau
                    && actual.eta == eta
                    && actual.penalties.penalty_config() == penalties
            }
            _ => false,
        }
    }

    /// Inspects the exact fixed-size history copy without allocating or changing
    /// the sampler. Both standard controls and Mirostat adaptive state are bound
    /// by the returned source borrow; no RNG draw or sampling step occurs.
    pub fn prepare_copy(&self) -> Result<SamplerCopyPlan<'_>, SamplerCopyError> {
        let history = self
            .history()
            .prepare_copy()
            .ok_or(SamplerCopyError::Overflow)?;
        let retained_bytes = u64::try_from(std::mem::size_of::<Self>())
            .ok()
            .and_then(|bytes| bytes.checked_add(history.bytes()))
            .ok_or(SamplerCopyError::Overflow)?;
        Ok(SamplerCopyPlan {
            source: self,
            history,
            retained_bytes,
        })
    }

    /// Accepted-token count, independently of allocated history capacity.
    pub fn history_len(&self) -> usize {
        self.history().as_slice().len()
    }

    /// Exact managed U32 payload capacity retained by the shared history owner.
    pub fn history_capacity(&self) -> usize {
        self.history().capacity()
    }

    fn history(&self) -> &history::TokenHistory {
        match self {
            Self::Standard(sampler) => &sampler.generated_tokens,
            Self::MirostatV2(sampler) => &sampler.penalties.generated_tokens,
        }
    }

    /// Selects the shared standard or adaptive policy from one resolved request.
    pub fn from_config(config: TextGenerationConfig) -> Result<Self, SamplingConfigurationError> {
        let sampling = config.sampling();
        Ok(match config.strategy() {
            TextSamplingStrategy::Standard => {
                Self::Standard(GenerationSampler::from_resolved(sampling))
            }
            TextSamplingStrategy::MirostatV2 { tau, eta } => {
                Self::MirostatV2(MirostatV2Sampler::new(tau, eta)?.penalties(
                    sampling.repetition_penalty,
                    sampling.repeat_last_n,
                    sampling.frequency_penalty,
                    sampling.presence_penalty,
                ))
            }
        })
    }
}

impl<B: SamplingBackend> Sampler<B> for ConfiguredTextSampler {
    fn sample(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        stream: &B::Context,
    ) -> Result<B::Token, B::Error> {
        match self {
            Self::Standard(sampler) => {
                Sampler::<B>::sample(sampler, logits, temperature, random, stream)
            }
            Self::MirostatV2(sampler) => {
                Sampler::<B>::sample(sampler, logits, temperature, random, stream)
            }
        }
    }
}

impl<B: SamplingBackend> SpeculativeSampler<B> for ConfiguredTextSampler {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn prepared_host_copy(&self) -> Option<PreparedSpeculativeSamplerCopy<'_, Self>> {
        speculative_copy::configured(self)
    }

    fn prepared_categorical_policy(&self) -> Option<PreparedCategoricalPolicy<'_>> {
        match self {
            Self::Standard(sampler) => Some(PreparedCategoricalPolicy::standard(sampler)),
            Self::MirostatV2(sampler) => Some(PreparedCategoricalPolicy::adaptive(sampler)),
        }
    }

    fn prepared_adaptive_commit(&self) -> Option<PreparedAdaptiveCommit<'_,Self>> {
        adaptive_commit::configured(self)
    }

    fn prepared_greedy_policy(&self) -> Option<PreparedGreedyPolicy<'_>> {
        match self {
            Self::Standard(sampler) => Some(PreparedGreedyPolicy::standard(sampler)),
            Self::MirostatV2(_) => None,
        }
    }

    fn prepared_logit_policy(&self) -> Option<PreparedLogitPolicy<'_>> {
        match self {
            Self::Standard(sampler) => Some(PreparedLogitPolicy::standard(sampler)),
            Self::MirostatV2(sampler) => Some(PreparedLogitPolicy::adaptive(sampler)),
        }
    }

    fn control_requires_positive_temperature(&self) -> Option<bool> {
        Some(matches!(self, Self::MirostatV2(_)))
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        Some(self.prepare_copy().ok()?.retained_bytes())
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        match self {
            Self::Standard(sampler) => {
                SpeculativeSampler::<B>::supports_exact_optimistic_promotion(sampler)
            }
            Self::MirostatV2(sampler) => {
                SpeculativeSampler::<B>::supports_exact_optimistic_promotion(sampler)
            }
        }
    }

    fn process_logits(
        &mut self,
        logits: &B::Logits,
        temperature: f32,
        history: &[u32],
        stream: &B::Context,
    ) -> Result<B::Logits, B::Error> {
        match self {
            Self::Standard(sampler) => SpeculativeSampler::<B>::process_logits(
                sampler,
                logits,
                temperature,
                history,
                stream,
            ),
            Self::MirostatV2(sampler) => SpeculativeSampler::<B>::process_logits(
                sampler,
                logits,
                temperature,
                history,
                stream,
            ),
        }
    }

    fn commit_token(
        &mut self,
        processed_logits: &B::Logits,
        token: u32,
        stream: &B::Context,
    ) -> Result<(), B::Error> {
        match self {
            Self::Standard(sampler) => {
                SpeculativeSampler::<B>::commit_token(sampler, processed_logits, token, stream)
            }
            Self::MirostatV2(sampler) => {
                SpeculativeSampler::<B>::commit_token(sampler, processed_logits, token, stream)
            }
        }
    }
}

#[cfg(test)]
mod copy_tests;
