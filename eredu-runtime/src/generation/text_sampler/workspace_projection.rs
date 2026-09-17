//! Source-bound scalar progress for cold sampling workspace inspection.

use super::ConfiguredTextSampler;
use crate::generation::{next_history_capacity, SamplingConfigurationError};

/// Invalid metadata progress for a configured sampler workspace projection.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SamplerProjectionError {
    /// A history length, rounded capacity, or managed byte extent overflowed.
    #[error("sampling projection history exceeds the supported host extent")]
    HistoryOverflow,
    /// The shared adaptive policy rejected a probability witness.
    #[error("sampling projection adaptive state: {0}")]
    Sampling(#[source] SamplingConfigurationError),
}

/// Exact policy borrow plus history extent and adaptive scalar witnesses.
///
/// Construction copies no token history, numerical value, RNG state or source
/// owner. The actual immutable sampler borrow prevents policy/history mutation
/// while the projection is used. Workspace tracing advances only scalar extent
/// and adaptive metadata; no native sampler or inference permission is created.
/// Numerical primitive facts must cover every token/probability compatible with
/// their descriptors, independently of the workspace backend's scalar witnesses.
pub struct SamplerWorkspaceProjection<'a> {
    source: &'a ConfiguredTextSampler,
    history_len: usize,
    history_capacity: usize,
    mu: Option<f32>,
}

impl std::fmt::Debug for SamplerWorkspaceProjection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerWorkspaceProjection")
            .field("history_len", &self.history_len)
            .field("history_capacity", &self.history_capacity)
            .field("mu", &self.mu)
            .finish_non_exhaustive()
    }
}

impl<'a> SamplerWorkspaceProjection<'a> {
    pub(super) fn new(source: &'a ConfiguredTextSampler) -> Self {
        Self {
            source,
            history_len: source.history_len(),
            history_capacity: source.history_capacity(),
            mu: match source {
                ConfiguredTextSampler::Standard(_) => None,
                ConfiguredTextSampler::MirostatV2(sampler) => Some(sampler.mu()),
            },
        }
    }

    /// Accepted source history length, without reading any token value.
    pub fn history_len(&self) -> usize {
        self.history_len
    }

    /// Exact retained source box capacity, including cleared or unused slots.
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }

    /// The source's current adaptive surprise limit, if applicable.
    pub fn mirostat_mu(&self) -> Option<f32> {
        self.mu
    }

    /// Checks all rounded history growth before tracing any operation. This is
    /// metadata validation only, not a host byte reservation or source proof.
    pub fn validate_steps(&self, steps: u64) -> Result<(), SamplerProjectionError> {
        let additional =
            usize::try_from(steps).map_err(|_| SamplerProjectionError::HistoryOverflow)?;
        let len = self
            .history_len
            .checked_add(additional)
            .ok_or(SamplerProjectionError::HistoryOverflow)?;
        capacity_for(self.history_capacity, len)?;
        Ok(())
    }

    pub(crate) fn source(&self) -> &'a ConfiguredTextSampler {
        self.source
    }

    // Only the closed workspace driver calls this; the public projection has no
    // mutation, arbitrary history setter, or method that samples native values.
    pub(crate) fn advance(
        &mut self,
        probability: Option<f32>,
    ) -> Result<(), SamplerProjectionError> {
        let len = self
            .history_len
            .checked_add(1)
            .ok_or(SamplerProjectionError::HistoryOverflow)?;
        let capacity = capacity_for(self.history_capacity, len)?;
        let mu = match self.source {
            ConfiguredTextSampler::Standard(_) => None,
            ConfiguredTextSampler::MirostatV2(sampler) => Some(
                sampler
                    .next_mu(
                        self.mu.expect("adaptive source has scalar progress"),
                        probability.expect("adaptive workspace commit supplies probability"),
                    )
                    .map_err(SamplerProjectionError::Sampling)?,
            ),
        };
        self.history_len = len;
        self.history_capacity = capacity;
        self.mu = mu;
        Ok(())
    }
}

fn capacity_for(mut capacity: usize, len: usize) -> Result<usize, SamplerProjectionError> {
    let valid = |capacity: usize| {
        capacity
            .checked_mul(std::mem::size_of::<u32>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SamplerProjectionError::HistoryOverflow)
    };
    valid(capacity)?;
    valid(len)?;
    while capacity < len {
        capacity =
            next_history_capacity(capacity).ok_or(SamplerProjectionError::HistoryOverflow)?;
        valid(capacity)?;
    }
    Ok(capacity)
}
