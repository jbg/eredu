//! One destination account for a frozen sampler and isolated native copies.

use super::{
    residual::RegisteredStoragePin, AdmittedWorkspaceCopy, BorrowedFundedSampler,
    FundedSamplerCopy, InferenceExecutionIdentity, RegisteredWorkspaceCopy, WorkingMemoryError,
    WorkingMemoryPool, WorkingMemoryStorage, WorkspaceCopyLimits,
};
use crate::generation::{SamplerCopyError, SamplerCopyPlan};

/// A concrete immutable sampler borrow joined to a sealed native copy program.
/// Both sources remain accounted independently throughout copying. No scalar
/// byte estimate, edited report or execution identity can create this proof.
/// Native providers retain the exact physical witnesses and settled-access
/// authority required by `RegisteredWorkspaceCopy` until work completes.
#[derive(Debug)]
pub struct RegisteredSamplingCopy<'a, K: Ord + Send + 'static> {
    pub(super) sampler: BorrowedFundedSampler<'a>,
    pub(super) sampler_plan: SamplerCopyPlan<'a>,
    pub(super) arrays: RegisteredWorkspaceCopy<K>,
    pub(super) host_bytes: u64,
    bytes: u64,
}

impl<'a, K: Clone + Ord + Send + Sync + 'static> RegisteredSamplingCopy<'a, K> {
    /// Prepares the actual history copy and checked sum of both closed plans.
    /// This performs neither destination allocation nor pool admission.
    pub fn prepare(
        sampler: BorrowedFundedSampler<'a>,
        arrays: RegisteredWorkspaceCopy<K>,
    ) -> Result<Self, SamplingCopyAdmissionError> {
        let sampler_plan = sampler.prepare_copy()?;
        let host_bytes = sampler_plan.retained_bytes();
        let bytes = host_bytes
            .checked_add(arrays.incremental_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            sampler,
            sampler_plan,
            arrays,
            host_bytes,
            bytes,
        })
    }

    /// Retains additional already registered source roots through native copy
    /// settlement or quarantine. The native provider must supply its complete
    /// actual source inventory and retain the corresponding physical owners.
    /// A registration handle alone does not prove semantic completeness.
    ///
    /// This adds no destination bytes, decoder table, host hold or allocation
    /// authority. Every supplied origin is revalidated atomically at admission.
    pub fn with_complete_source(
        self,
        complete_source: WorkingMemoryStorage<K>,
    ) -> RegisteredSamplingCopyWithSource<'a, K> {
        RegisteredSamplingCopyWithSource {
            sampling: self,
            complete_source,
        }
    }

    /// Incremental managed host payload plus native closed-program demand,
    /// excluding any safety reserve selected at admission.
    pub fn required_bytes(&self) -> u64 {
        self.bytes
    }
}

/// A sealed sampling copy with mandatory additional source custody.
/// The provider binds this inventory to its actual source representation;
/// runtime validates the exact registered origins, not transitive native roots.
/// Uncopied roots remain separately charged and never become copy operands.
/// This supplies no decoder allocation or inference-construction authority.
#[derive(Debug)]
#[must_use = "the joined source inventories require admission before copying"]
pub struct RegisteredSamplingCopyWithSource<'a, K: Ord + Send + 'static> {
    sampling: RegisteredSamplingCopy<'a, K>,
    complete_source: WorkingMemoryStorage<K>,
}

impl<K: Clone + Ord + Send + Sync + 'static> RegisteredSamplingCopyWithSource<'_, K> {
    /// Incremental sampler and native demand, excluding the existing source
    /// inventory and any safety reserve selected at admission.
    pub fn required_bytes(&self) -> u64 {
        self.sampling.required_bytes()
    }
}

/// Rejection before the destination sampler or native operation is published.
#[derive(Debug, thiserror::Error)]
pub enum SamplingCopyAdmissionError {
    /// The borrowed sampler's exact fixed destination payload is unsupported.
    #[error("{0}")]
    Sampler(#[from] SamplerCopyError),
    /// Source custody, arithmetic or shared-domain admission failed.
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
    /// Combined destination demand and safety exceed the application limit.
    #[error("sampling copy needs {required_bytes} bytes; application limit is {budget_bytes}")]
    ApplicationBudgetExceeded {
        /// Both component bounds plus safety reserve.
        required_bytes: u64,
        /// Requested per-copy application allowance.
        budget_bytes: u64,
    },
}

impl WorkingMemoryPool {
    /// Atomically admits both components, then copies the exact host sampler.
    /// The returned native operation is already admitted and must enter the
    /// existing completion/publication path without reserving another account.
    ///
    /// Native adoption cannot spend the sampler's held inline/boxed payload.
    /// Both outputs share one destination account identity; neither carries a
    /// runnable text grant. `FundedSamplerCopy::bytes` reports only its host
    /// component, while `AdmittedWorkspaceCopy::bytes` reports this aggregate
    /// account, including safety. Do not add those overlapping diagnostics.
    /// Allocator metadata and a complete inference snapshot remain outside this
    /// component contract.
    pub fn copy_sampling_components<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: RegisteredSamplingCopy<'_, K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<(FundedSamplerCopy, AdmittedWorkspaceCopy), SamplingCopyAdmissionError> {
        self.copy_sampling_components_inner(copy, None, limits)
    }

    /// Admits the same sampler/native program while retaining every supplied
    /// complete-source origin alongside the operand origins in the native scope.
    /// The sampler result may retire first without releasing those source pins.
    /// Uncertified native work quarantines the entire bundle.
    ///
    /// This is the no-decoder-allocation route: no placeholder table or zero-byte
    /// decoder scope is created. The native provider must bind the supplied
    /// inventory to its actual source and retain physical witnesses through work.
    /// Neither the result nor the source inventory grants a future inference step.
    pub fn copy_sampling_components_with_source<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: RegisteredSamplingCopyWithSource<'_, K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<(FundedSamplerCopy, AdmittedWorkspaceCopy), SamplingCopyAdmissionError> {
        self.copy_sampling_components_inner(copy.sampling, Some(copy.complete_source), limits)
    }

    fn copy_sampling_components_inner<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: RegisteredSamplingCopy<'_, K>,
        complete_source: Option<WorkingMemoryStorage<K>>,
        limits: WorkspaceCopyLimits,
    ) -> Result<(FundedSamplerCopy, AdmittedWorkspaceCopy), SamplingCopyAdmissionError> {
        let bytes = copy
            .bytes
            .checked_add(limits.safety_reserve_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        if let Some(budget_bytes) = limits.application_memory_budget_bytes {
            if bytes > budget_bytes {
                return Err(SamplingCopyAdmissionError::ApplicationBudgetExceeded {
                    required_bytes: bytes,
                    budget_bytes,
                });
            }
        }
        let source = copy.arrays.source().registration();
        let preparation = complete_source
            .as_ref()
            .and_then(WorkingMemoryStorage::source_preparation)
            .or_else(|| source.source_preparation());
        let execution = match preparation {
            Some(preparation) => {
                super::WorkspaceCopyAccountLayout::sampling()?.execution(preparation)
            }
            None => InferenceExecutionIdentity::default(),
        };
        // Construct the accounting-only bundle outside the usage lock. The
        // native scope owns it; host retirement cannot certify or release it.
        let operand_pin = RegisteredStoragePin::new(source.clone());
        let pin = match &complete_source {
            Some(complete) => {
                let pins = [operand_pin, RegisteredStoragePin::new(complete.clone())];
                if complete.has_source_preparation() {
                    RegisteredStoragePin::aggregate_counted(pins, 2)?
                } else {
                    RegisteredStoragePin::aggregate(pins)
                }
            }
            None => operand_pin,
        };
        let (funding, host_scope, scope) = self.open_sampling_copy_account(
            copy.sampler.source(),
            copy.sampler.execution(),
            source,
            complete_source.as_ref(),
            pin,
            &execution,
            bytes,
            copy.host_bytes,
            limits.capacity_bytes,
        )?;
        #[cfg(test)]
        tests::before_copy();
        let sampler = copy.sampler_plan.copy();
        let sampler = FundedSamplerCopy::from_shared_account(
            sampler,
            execution.clone(),
            copy.host_bytes,
            host_scope,
        );
        let arrays = AdmittedWorkspaceCopy::from_account(execution, bytes, funding, scope);
        Ok((sampler, arrays))
    }
}

#[cfg(test)]
mod tests;
