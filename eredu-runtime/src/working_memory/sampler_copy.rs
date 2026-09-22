//! Concrete sampler payload ownership and independent destination-copy funding.

use super::{
    InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError, WorkingMemoryFundingRun,
    WorkingMemorySamplerScope, funding::FundingSource,
};
use crate::{ConfiguredTextSampler, Sampler, SamplingBackend, generation::SamplerCopyError};
use eredu_core::TextGenerationConfig;

mod resume;
pub use resume::SamplerResumePlan;

/// Limits for one independent sampler copy, separate from logical snapshot
/// counts and from any future native or sampling execution allowance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SamplerCopyLimits {
    /// Total live-charge limits in every physical domain.
    pub memory_limits: eredu_core::MemoryLimitDeclarations,
    /// Additional domain-attributed charge retained until this copy retires.
    pub additional_headroom: eredu_core::MemoryHeadroomDeclarations,
}

impl SamplerCopyLimits {
    /// Selects domain limits without additional headroom.
    pub const fn new(memory_limits: eredu_core::MemoryLimitDeclarations) -> Self {
        Self {
            memory_limits,
            additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
        }
    }
}

/// Rejection before publishing an independently funded sampler copy.
#[derive(Debug, thiserror::Error)]
pub enum SamplerCopyAdmissionError {
    /// The concrete sampler's managed copy extent cannot be represented.
    #[error("{0}")]
    Plan(#[from] SamplerCopyError),
    /// Source custody or shared-domain accounting rejected the copy.
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
}

/// One configured sampler constructed by its original admitted sampling stage.
/// Its exact payload and source funding remain together; borrowing or moving
/// this owner cannot transfer native execution permission.
#[derive(Debug)]
pub struct RunOwnedTextSampler {
    sampler: ConfiguredTextSampler,
    execution: InferenceExecutionIdentity,
    max_samples: u64,
    issued_samples: u64,
    max_history_capacity: usize,
    // Ordinary field destruction retires the sampler BEFORE host certification.
    // This owner deliberately has no custom Drop that could reverse that order.
    custody: WorkingMemorySamplerScope,
}

impl RunOwnedTextSampler {
    pub(super) fn construct(
        config: TextGenerationConfig,
        max_samples: u64,
        execution: InferenceExecutionIdentity,
        mut host_scope: WorkingMemorySamplerScope,
    ) -> Result<Self, WorkingMemoryError> {
        let maximum = usize::try_from(max_samples).map_err(|_| WorkingMemoryError::Overflow)?;
        let mut max_history_capacity = 0;
        let mut peak_history_capacity = 0;
        while max_history_capacity < maximum {
            let next = crate::generation::next_history_capacity(max_history_capacity)
                .ok_or(WorkingMemoryError::Overflow)?;
            peak_history_capacity = max_history_capacity
                .checked_add(next)
                .ok_or(WorkingMemoryError::Overflow)?;
            max_history_capacity = next;
        }
        if max_history_capacity
            .checked_mul(std::mem::size_of::<u32>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(WorkingMemoryError::Overflow);
        }
        let history_bytes = peak_history_capacity
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let host_bytes = u64::try_from(std::mem::size_of::<ConfiguredTextSampler>())
            .ok()
            .and_then(|inline| inline.checked_add(history_bytes))
            .ok_or(WorkingMemoryError::Overflow)?;
        host_scope.hold_sampler_payload(host_bytes)?;
        let sampler = ConfiguredTextSampler::from_config(config)
            .map_err(|_| WorkingMemoryError::PreparationConfigurationMismatch)?;
        Ok(Self {
            sampler,
            execution,
            max_samples,
            issued_samples: 0,
            max_history_capacity,
            custody: host_scope,
        })
    }

    /// Borrows controls and history. Cloning this raw view allocates a separate
    /// caller-owned sampler and carries no funding or managed source proof.
    pub fn as_sampler(&self) -> &ConfiguredTextSampler {
        &self.sampler
    }

    /// Borrows the actual payload together with its original funding custody.
    pub fn borrow_funded(&self) -> BorrowedFundedSampler<'_> {
        BorrowedFundedSampler {
            sampler: &self.sampler,
            execution: &self.execution,
            source: self.custody.source(),
        }
    }

    /// Claims one sampling attempt before any backend filtering or sampling.
    /// Dropping the returned permit does not refund the attempt. This bounds
    /// this sampler's history; the native caller still needs its execution grant.
    pub fn prepare_sample(&mut self) -> Result<PreparedRunSample<'_>, WorkingMemoryError> {
        self.custody.validate_source(&self.execution)?;
        if self.issued_samples >= self.max_samples {
            return Err(WorkingMemoryError::TextOutputAllowanceExceeded {
                issued: self.issued_samples,
                limit: self.max_samples,
            });
        }
        let next_capacity = if self.sampler.history_len() == self.sampler.history_capacity() {
            crate::generation::next_history_capacity(self.sampler.history_capacity())
                .ok_or(WorkingMemoryError::Overflow)?
        } else {
            self.sampler.history_capacity()
        };
        if next_capacity > self.max_history_capacity {
            return Err(WorkingMemoryError::Overflow);
        }
        self.issued_samples = self
            .issued_samples
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(PreparedRunSample {
            sampler: &mut self.sampler,
        })
    }
}

/// A single sampling attempt borrowing the exact admitted sampler exclusively.
#[must_use = "consume this permit once after the enclosing execution grant is checked"]
#[derive(Debug)]
pub struct PreparedRunSample<'a> {
    sampler: &'a mut ConfiguredTextSampler,
}

impl PreparedRunSample<'_> {
    /// Executes the existing configured sampler exactly once. No snapshot,
    /// native submission or future allocation authority is created here.
    pub fn sample<B: SamplingBackend>(
        self,
        logits: &B::Logits,
        temperature: f32,
        random: Option<&mut B::RandomState>,
        context: &B::Context,
    ) -> Result<B::Token, B::Error> {
        Sampler::<B>::sample(self.sampler, logits, temperature, random, context)
    }
}

/// Borrowed evidence available only from an actual owned funded sampler.
/// It cannot be constructed from a raw sampler, scalar execution identity or
/// historical request. The source stays borrowed through admission and copying.
#[derive(Debug)]
pub struct BorrowedFundedSampler<'a> {
    sampler: &'a ConfiguredTextSampler,
    execution: &'a InferenceExecutionIdentity,
    source: FundingSource<'a>,
}

impl<'a> BorrowedFundedSampler<'a> {
    pub(super) fn prepare_copy(
        &self,
    ) -> Result<crate::generation::SamplerCopyPlan<'a>, SamplerCopyError> {
        self.sampler.prepare_copy()
    }

    pub(super) fn source(&self) -> FundingSource<'a> {
        self.source
    }

    pub(super) fn execution(&self) -> &'a InferenceExecutionIdentity {
        self.execution
    }
}

#[derive(Debug)]
enum SamplerCopyCustody {
    IndependentRun(WorkingMemoryFundingRun),
    SharedHost(WorkingMemorySamplerScope),
}

impl SamplerCopyCustody {
    fn source(&self) -> FundingSource<'_> {
        match self {
            Self::IndependentRun(run) => FundingSource::CopyRun(run),
            Self::SharedHost(scope) => scope.source(),
        }
    }
}

/// A frozen sampler component with an independently admitted destination.
/// This is neither a complete snapshot nor an advanceable child run. It has no
/// mutable/owned raw export; another managed duplicate needs another admission.
#[derive(Debug)]
pub struct FundedSamplerCopy {
    sampler: ConfiguredTextSampler,
    execution: InferenceExecutionIdentity,
    bytes: u64,
    // Retire the actual host payload before releasing this account's remainder.
    funding: SamplerCopyCustody,
}

impl FundedSamplerCopy {
    pub(super) fn from_shared_account(
        sampler: ConfiguredTextSampler,
        execution: InferenceExecutionIdentity,
        bytes: u64,
        scope: WorkingMemorySamplerScope,
    ) -> Self {
        Self {
            sampler,
            execution,
            bytes,
            funding: SamplerCopyCustody::SharedHost(scope),
        }
    }

    /// Borrows the copied controls and history without transferring custody.
    pub fn as_sampler(&self) -> &ConfiguredTextSampler {
        &self.sampler
    }

    /// Managed inline/payload charge. Standalone copies also include their
    /// retained safety reserve; an aggregate operation owns its reserve jointly.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Uses this exact frozen component as the source of another admitted copy.
    pub fn borrow_funded(&self) -> BorrowedFundedSampler<'_> {
        BorrowedFundedSampler {
            sampler: &self.sampler,
            execution: &self.execution,
            source: self.funding.source(),
        }
    }
}

impl MemoryLedger {
    /// Quotes the complete independent copy, including protected account controls.
    /// The source remains borrowed; this does not reserve or copy its payload.
    pub fn sampler_copy_requirements(
        &self,
        source: &BorrowedFundedSampler<'_>,
        limits: &SamplerCopyLimits,
    ) -> Result<eredu_core::DomainMemoryRequirements, SamplerCopyAdmissionError> {
        if !self.same_ledger(source.source.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let plan = source.sampler.prepare_copy()?;
        let mut projection = super::transaction_buffers::RequirementProjection {
            parts: &[],
            headroom: &limits.additional_headroom,
            host_bytes: plan.retained_bytes(),
        };
        projection.host_bytes = projection
            .host_bytes
            .checked_add(sampler_controls(self, &projection)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(projection.materialize(self.topology())?)
    }
    /// Admits and performs one concrete sampler copy against the same pool as
    /// its authenticated source. Source, baseline, registered storage and all
    /// other accounts remain charged throughout. No unquoted owner is removed.
    ///
    /// The managed sampler, exact history box and account controls are charged
    /// in the host domain, plus domain-attributed headroom. Native RNG/input/state, allocator metadata and
    /// full snapshots are outside this component contract.
    pub fn copy_sampler(
        &self,
        source: BorrowedFundedSampler<'_>,
        limits: SamplerCopyLimits,
    ) -> Result<FundedSamplerCopy, SamplerCopyAdmissionError> {
        let plan = source.sampler.prepare_copy()?;
        let projection = super::transaction_buffers::RequirementProjection {
            parts: &[],
            headroom: &limits.additional_headroom,
            host_bytes: plan.retained_bytes(),
        };
        let controls = sampler_controls(self, &projection)?;
        let accepted = super::funding::PreparedCopyAccount::accept(
            self,
            source.execution,
            projection,
            &limits.memory_limits,
            controls,
            super::funding::CopyHostHolds::None,
            |usage| {
                if !self.same_ledger(source.source.pool()) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                source.source.validate(usage, source.execution)
            },
        )?;
        let bytes = plan.retained_bytes();
        let execution = source.execution.clone();
        let (_, funding, scope) = accepted.workspace(&execution, None)?;
        #[cfg(test)]
        tests::before_copy();
        let sampler = plan.copy();
        let copied = FundedSamplerCopy {
            sampler,
            execution,
            bytes,
            funding: SamplerCopyCustody::IndependentRun(funding),
        };
        // Only this synchronous host copy has completed. Its independent run
        // stays open until the copied payload dies; no native scope is certified.
        scope.certify()?;
        Ok(copied)
    }
}

fn sampler_controls(
    pool: &MemoryLedger,
    projection: &super::transaction_buffers::RequirementProjection<'_>,
) -> Result<u64, WorkingMemoryError> {
    super::funding::copy_domain_controls(pool, projection)?
        .checked_add(
            u64::try_from(super::funding::copy_account_control_bytes(false, 0, false)?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
        )
        .ok_or(WorkingMemoryError::Overflow)
}

#[cfg(test)]
mod tests;
