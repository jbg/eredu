//! Resolved text sampling policy shared by ordinary and speculative requests.

use super::{
    GenerationSampler, MirostatV2Sampler, Sampler, SamplingBackend, SamplingConfigurationError,
    SpeculativeSampler,
};
use eredu_core::{TextGenerationConfig, TextSamplingStrategy, TokenFilter};

/// Sampling policy selected from a resolved text generation request.
#[derive(Clone)]
pub enum TextSampler {
    /// Configured penalties and standard truncation.
    Standard(GenerationSampler),
    /// Adaptive Mirostat V2 strategy with its own history and state.
    MirostatV2(MirostatV2Sampler),
}

impl TextSampler {
    pub(crate) fn requires_positive_temperature(&self) -> bool {
        matches!(self, Self::MirostatV2(_))
    }

    pub(crate) fn history_bytes(&self) -> Option<u64> {
        let tokens = match self {
            Self::Standard(sampler) => sampler.generated_tokens(),
            Self::MirostatV2(sampler) => sampler.generated_tokens(),
        };
        u64::try_from(tokens.len())
            .ok()?
            .checked_mul(std::mem::size_of::<u32>() as u64)
    }

    /// Configures sampling and its penalty history.
    pub fn from_config(config: &TextGenerationConfig) -> Result<Self, SamplingConfigurationError> {
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

    /// Samples through the configured policy.
    pub fn sample<B: SamplingBackend>(
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

impl<B: SamplingBackend> SpeculativeSampler<B> for TextSampler {
    fn refinement_filter(&mut self, _history: &[u32]) -> Result<TokenFilter, B::Error> {
        Ok(TokenFilter::All)
    }
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        Some(matches!(self, Self::MirostatV2(_)))
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        let history = match self {
            Self::Standard(sampler) => sampler.generated_tokens(),
            Self::MirostatV2(sampler) => sampler.generated_tokens(),
        };
        (history.len() as u64)
            .checked_mul(4)?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn continuation_storage_bytes(&self, additional_tokens: u64) -> Option<u64> {
        // Retain a generous Vec growth/replacement allowance for history.
        <Self as SpeculativeSampler<B>>::control_snapshot_bytes(self)?
            .checked_mul(2)?
            .checked_add(additional_tokens.checked_mul(8)?)
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
