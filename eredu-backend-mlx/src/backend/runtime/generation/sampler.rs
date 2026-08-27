//! Backend-neutral sampling policies specialized to MLX primitives.

use eredu_runtime::{
    Sampler as RuntimeSampler, SamplingBackend, SpeculativeSampler as RuntimeSpeculativeSampler,
};
use safemlx::{error::Exception, random::RandomState, Array, Stream};

pub use super::backend::MlxSamplingBackend;
use crate::MlxTensor;
use eredu_core::TokenFilter;

/// MLX-specialized token selection policy.
pub trait Sampler {
    /// Whether loaded checkpoint defaults should wrap this policy.
    fn uses_checkpoint_defaults(&self) -> bool {
        false
    }

    /// Selects one token from raw MLX logits.
    fn sample(
        &mut self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception>;
}

impl<T> Sampler for T
where
    T: RuntimeSampler<MlxSamplingBackend>,
{
    fn uses_checkpoint_defaults(&self) -> bool {
        RuntimeSampler::<MlxSamplingBackend>::uses_checkpoint_defaults(self)
    }

    fn sample(
        &mut self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        RuntimeSampler::<MlxSamplingBackend>::sample(
            self,
            &MlxTensor::from_array(logits.clone()),
            temperature,
            random,
            stream,
        )
        .map(MlxTensor::into_array)
    }
}

/// MLX-specialized lossless speculative sampling policy.
pub trait SpeculativeSampler {
    /// Whether loaded checkpoint defaults should wrap this policy.
    fn uses_checkpoint_defaults(&self) -> bool {
        false
    }

    /// Whether optimistic draft work is an exact discardable fork.
    fn supports_exact_optimistic_promotion(&self) -> bool {
        false
    }

    /// Whether the committed generation grammar is complete.
    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        Ok(false)
    }

    /// Whether an uncommitted logical prefix completes the grammar.
    fn prefix_is_complete(&self, _history: &[u32]) -> Result<bool, Exception> {
        Ok(false)
    }

    /// Applies penalties, filters, and temperature.
    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception>;

    /// Selects from already processed logits.
    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        MlxSamplingBackend::sample_processed(
            &MlxTensor::from_array(logits.clone()),
            temperature,
            random,
            stream,
        )
        .map(MlxTensor::into_array)
    }

    /// Commits an emitted token from a processed target distribution.
    fn commit_token(
        &mut self,
        _processed_logits: &Array,
        _token: u32,
        _stream: &Stream,
    ) -> Result<(), Exception> {
        Ok(())
    }
}

impl<T> SpeculativeSampler for T
where
    T: RuntimeSpeculativeSampler<MlxSamplingBackend>,
{
    fn uses_checkpoint_defaults(&self) -> bool {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::uses_checkpoint_defaults(self)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(self)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::grammar_is_complete(self)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::prefix_is_complete(self, history)
    }

    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::process_logits(
            self,
            &MlxTensor::from_array(logits.clone()),
            temperature,
            history,
            stream,
        )
        .map(MlxTensor::into_array)
    }

    fn sample_processed(
        &self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::sample_processed(
            self,
            &MlxTensor::from_array(logits.clone()),
            temperature,
            random,
            stream,
        )
        .map(MlxTensor::into_array)
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        RuntimeSpeculativeSampler::<MlxSamplingBackend>::commit_token(
            self,
            &MlxTensor::from_array(processed_logits.clone()),
            token,
            stream,
        )
    }
}

/// Applies a portable vocabulary filter to MLX logits.
pub fn apply_token_filter(
    logits: &Array,
    filter: &TokenFilter,
    stream: &Stream,
) -> Result<Array, Exception> {
    MlxSamplingBackend::apply_token_filter(&MlxTensor::from_array(logits.clone()), filter, stream)
        .map(MlxTensor::into_array)
}
