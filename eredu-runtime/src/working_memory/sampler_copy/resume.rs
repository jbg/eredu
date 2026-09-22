//! Source-bound host sampler construction under a fresh inference preparation.

use super::*;

/// Closed host copy and future history growth for one actual funded sampler.
/// This source borrow does not create a request, reset source attempts, or grant
/// native execution. Consume it only through the new request's sampling stage.
#[derive(Debug)]
#[must_use = "a resume plan needs a fresh admitted preparation before copying"]
pub struct SamplerResumePlan<'a> {
    source: BorrowedFundedSampler<'a>,
    copy: crate::generation::SamplerCopyPlan<'a>,
    config: TextGenerationConfig,
    max_samples: u64,
    max_history_capacity: usize,
    host_bytes: u64,
}

impl<'a> BorrowedFundedSampler<'a> {
    /// Plans a preserved sampler with a finite new output allowance. History
    /// values and adaptive controls remain borrowed; preparation copies no
    /// history or numerical payload. Static sampler policy must be unchanged.
    /// Native temperature, PRNG and executable compatibility remain separate.
    pub fn prepare_resume(
        self,
        config: TextGenerationConfig,
    ) -> Result<SamplerResumePlan<'a>, WorkingMemoryError> {
        if !self.sampler.matches_config_policy(config.clone()) {
            return Err(WorkingMemoryError::PreparationConfigurationMismatch);
        }
        let maximum = config
            .sampling()
            .max_new_tokens
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let max_samples = u64::try_from(maximum).map_err(|_| WorkingMemoryError::Overflow)?;
        let copy = self
            .prepare_copy()
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let (max_history_capacity, host_bytes) =
            history_extent(copy.history_len(), copy.history_capacity(), maximum)?;
        Ok(SamplerResumePlan {
            source: self,
            copy,
            config,
            max_samples,
            max_history_capacity,
            host_bytes,
        })
    }
}

pub(super) fn history_extent(
    length: usize,
    capacity: usize,
    additional: usize,
) -> Result<(usize, u64), WorkingMemoryError> {
    if length > capacity {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    let maximum = length
        .checked_add(additional)
        .ok_or(WorkingMemoryError::Overflow)?;
    let mut final_capacity = capacity;
    // Even cleared history or a zero-output resume copies every retained slot.
    let mut peak_capacity = capacity;
    while final_capacity < maximum {
        let next = crate::generation::next_history_capacity(final_capacity)
            .ok_or(WorkingMemoryError::Overflow)?;
        peak_capacity = peak_capacity.max(
            final_capacity
                .checked_add(next)
                .ok_or(WorkingMemoryError::Overflow)?,
        );
        final_capacity = next;
    }
    if final_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(WorkingMemoryError::Overflow);
    }
    let history_bytes = peak_capacity
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    let bytes = u64::try_from(std::mem::size_of::<ConfiguredTextSampler>())
        .ok()
        .and_then(|inline| inline.checked_add(history_bytes))
        .ok_or(WorkingMemoryError::Overflow)?;
    Ok((final_capacity, bytes))
}

impl SamplerResumePlan<'_> {
    /// Inline sampler and peak boxed history, including replacement overlap.
    /// Native state, PRNG and enclosing continuation payload are not included.
    pub fn required_host_bytes(&self) -> u64 {
        self.host_bytes
    }

    /// Actual history entries retained by the source.
    pub fn history_len(&self) -> usize {
        self.copy.history_len()
    }

    /// Actual source payload slots, including unused or cleared entries.
    pub fn history_capacity(&self) -> usize {
        self.copy.history_capacity()
    }

    /// Largest fixed history box this new sampler may construct.
    pub fn max_history_capacity(&self) -> usize {
        self.max_history_capacity
    }

    /// New sampling attempts, independent of preserved source history length.
    pub fn max_samples(&self) -> u64 {
        self.max_samples
    }

    pub(in crate::working_memory) fn construct(
        self,
        config: TextGenerationConfig,
        max_samples: u64,
        execution: InferenceExecutionIdentity,
        mut host_scope: WorkingMemorySamplerScope,
    ) -> Result<RunOwnedTextSampler, WorkingMemoryError> {
        if self.config != config || self.max_samples != max_samples {
            return Err(WorkingMemoryError::PreparationConfigurationMismatch);
        }
        // Source health and the new protected hold are checked under one pool
        // lock. No source account or previous output allowance is repurposed.
        host_scope.hold_resumed_sampler_payload(
            self.host_bytes,
            self.source.source(),
            self.source.execution(),
        )?;
        #[cfg(test)]
        super::tests::before_copy();
        let sampler = self.copy.copy();
        Ok(RunOwnedTextSampler {
            sampler,
            execution,
            max_samples,
            issued_samples: 0,
            max_history_capacity: self.max_history_capacity,
            custody: host_scope,
        })
    }
}
