//! Additive counts from actual accepted capture callbacks, not a second schedule.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CaptureNativePopulation {
    pub(crate) publications: usize,
    pub(crate) completions: usize,
    pub(crate) controls: usize,
    pub(crate) retained_roots: usize,
}
impl CaptureNativePopulation {
    /// Actual five-source sparse carrier: each source is completed and published
    /// once; no conversion or route-index tensor is created by host extraction.
    pub(crate) fn routed_prefill() -> Option<Self> {
        Some(Self {
            publications: 5,
            completions: 5,
            retained_roots: 5,
            controls: super::capture_original_control_bytes()?.checked_mul(5)?
                .checked_add(super::CompletedRoutedCaptureSource::control_bytes()?)?,
        })
    }
    /// Same five retained sources already belong to this original model Q.
    /// Their host extraction needs five completions and no scheduled publication.
    pub(crate) fn routed_model() -> Option<Self> {
        Some(Self { publications: 0, completions: 5, retained_roots: 5,
            controls: super::speculative_routed_capture_control_bytes()? })
    }
    /// Same five-source worker with original partition-coordinate reads.
    pub(crate) fn routed_partition() -> Option<Self> {
        let mut population=Self::routed_prefill()?;
        population.controls=population.controls.checked_sub(super::CompletedRoutedCaptureSource::control_bytes()?)?
            .checked_add(super::CompletedPartitionRoutedCaptureSource::control_bytes()?)?;
        Some(population)
    }
    pub(crate) fn raw(count: usize) -> Option<Self> {
        Some(Self {
            publications: count,
            completions: count.checked_mul(2)?,
            retained_roots: count.checked_mul(6)?,
            controls: super::capture_original_control_bytes()?.checked_mul(count)?,
        })
    }
    pub(crate) fn token_scores(program: super::TokenScoreProgram<'_>) -> Option<Self> {
        let population = program.population().ok()?;
        Some(Self {
            publications: 1,
            retained_roots: population.retained_outputs.checked_add(1)?,
            completions: population.scalar_completions.checked_add(1)?,
            controls: super::capture_original_control_bytes()?
                .checked_add(program.control_bytes()?)?,
        })
    }
    pub(crate) fn candidates() -> Option<Self> {
        Some(Self {
            publications: 1,
            completions: 1 + super::CandidateExtraction::COMPLETIONS,
            retained_roots: super::CandidateExtraction::ROOTS.checked_add(1)?,
            controls: super::capture_original_control_bytes()?
                .checked_add(super::CandidateExtraction::control_bytes()?)?,
        })
    }
    /// Same selected capture worker inside an already admitted original scope.
    /// No separate source publication/completion is performed in this path.
    pub(crate) fn within_raw() -> Option<Self> {
        Some(Self {
            publications: 0,
            completions: 1,
            retained_roots: 5,
            controls: super::speculative_capture_control_bytes()?,
        })
    }
    pub(crate) fn within_candidates() -> Option<Self> {
        Some(Self {
            publications: 0,
            completions: super::CandidateExtraction::COMPLETIONS,
            retained_roots: super::CandidateExtraction::ROOTS.checked_add(1)?,
            controls: super::speculative_candidate_control_bytes()?,
        })
    }
    pub(crate) fn within_token_scores(program: super::TokenScoreProgram<'_>) -> Option<Self> {
        let population = program.population().ok()?;
        Some(Self {
            publications: 0,
            completions: population.scalar_completions,
            retained_roots: population.retained_outputs.checked_add(1)?,
            controls: super::speculative_token_score_control_bytes(program)?,
        })
    }
    pub(crate) fn within_summary(program: &super::PreparedCaptureSummary) -> Option<Self> {
        let population = program.population()?;
        Some(Self {
            publications: 0,
            completions: population.completions.checked_sub(1)?,
            retained_roots: population.retained_roots,
            controls: super::speculative_summary_control_bytes(program)?,
        })
    }
    pub(crate) fn within_histogram(program: &super::PreparedCaptureHistogram<'_>) -> Option<Self> {
        let population = program.population()?;
        Some(Self {
            publications: 0,
            completions: population.completions.checked_sub(1)?,
            retained_roots: population.retained_roots,
            controls: super::speculative_histogram_control_bytes(program)?,
        })
    }
    /// Preflight is required for each spent generated claim, even when its
    /// slice is empty or an earlier selection already constructed the value.
    pub(crate) fn generated_preparation() -> Option<Self> {
        Some(Self {
            controls: super::GeneratedCaptureRetention::preparation_control_bytes()?,
            ..Self::default()
        })
    }
    /// Exactly one actual compact-source/seven-output factory invocation.
    pub(crate) fn generated_retention() -> Option<Self> {
        Some(Self {
            controls: super::GeneratedCaptureRetention::retention_control_bytes()?,
            retained_roots: super::GeneratedCaptureRetention::ROOTS,
            ..Self::default()
        })
    }
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            publications: self.publications.checked_add(other.publications)?,
            completions: self.completions.checked_add(other.completions)?,
            controls: self.controls.checked_add(other.controls)?,
            retained_roots: self.retained_roots.checked_add(other.retained_roots)?,
        })
    }
}
