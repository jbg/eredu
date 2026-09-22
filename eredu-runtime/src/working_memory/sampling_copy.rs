//! One destination account for a frozen sampler and isolated native copies.

use super::{
    AdmittedWorkspaceCopy, BorrowedFundedSampler, FundedSamplerCopy, InferenceExecutionIdentity,
    MemoryLedger, RegisteredWorkspaceCopy, WorkingMemoryError, WorkingMemoryStorage,
    WorkspaceCopyLimits, residual::RegisteredStoragePin,
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
    bytes: Option<u64>,
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
        let bytes = arrays
            .incremental_bytes()
            .and_then(|bytes| bytes.checked_add(host_bytes));
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
    pub fn required_bytes(&self) -> Option<u64> {
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
    pub fn required_bytes(&self) -> Option<u64> {
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
}

impl MemoryLedger {
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
        let source = copy.arrays.source().registration();
        let preparation = complete_source
            .as_ref()
            .and_then(WorkingMemoryStorage::source_preparation)
            .or_else(|| source.source_preparation());
        let direct_controls =
            sampling_controls::<K>(preparation.is_some(), complete_source.is_some())?;
        let accepted = super::workspace_copy::prepare_copy_account(
            self,
            copy.sampler.execution(),
            copy.arrays.incremental_requirements(),
            copy.host_bytes,
            &limits,
            direct_controls,
            super::funding::CopyHostHolds::Sampler(copy.host_bytes),
            |usage| {
                if !self.same_ledger(copy.sampler.source().pool()) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                copy.sampler
                    .source()
                    .validate(usage, copy.sampler.execution())?;
                source.validate_copy_source(self, usage)?;
                if let Some(complete) = &complete_source {
                    complete.validate_copy_source(self, usage)?;
                }
                Ok(())
            },
        )?;
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
                    RegisteredStoragePin::pair(pins[0].clone(), pins[1].clone())
                }
            }
            None => operand_pin,
        };
        let (requirements, funding, host_scope, scope) =
            accepted.sampling(&execution, pin, copy.host_bytes)?;
        #[cfg(test)]
        tests::before_copy();
        let sampler = copy.sampler_plan.copy();
        let sampler = FundedSamplerCopy::from_shared_account(
            sampler,
            execution.clone(),
            copy.host_bytes,
            host_scope,
        );
        let arrays = AdmittedWorkspaceCopy::from_account(execution, requirements, funding, scope);
        Ok((sampler, arrays))
    }
}

fn sampling_controls<K: Ord + Send + Sync + 'static>(
    prepared: bool,
    complete: bool,
) -> Result<usize, WorkingMemoryError> {
    Ok(if !prepared {
        let mut bytes = super::WorkspaceCopyAccountLayout::sampling()?
            .requested_bytes()
            .checked_add(RegisteredStoragePin::single_control_bytes::<K>(false)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        if complete {
            bytes = bytes
                .checked_add(RegisteredStoragePin::single_control_bytes::<K>(false)?)
                .and_then(|n| n.checked_add(RegisteredStoragePin::pair_control_bytes(false).ok()?))
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        bytes
    } else {
        0
    })
}
impl MemoryLedger {
    /// Complete incremental domains for the ordinary sampler and workspace copy.
    pub fn sampling_copy_requirements<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: &RegisteredSamplingCopy<'_, K>,
        limits: &WorkspaceCopyLimits,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError> {
        self.sampling_requirements_inner(copy, None, limits)
    }
    /// Complete domains including the supplied complete-source pin constructor.
    pub fn sampling_copy_with_source_requirements<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: &RegisteredSamplingCopyWithSource<'_, K>,
        limits: &WorkspaceCopyLimits,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError> {
        self.sampling_requirements_inner(&copy.sampling, Some(&copy.complete_source), limits)
    }
    fn sampling_requirements_inner<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: &RegisteredSamplingCopy<'_, K>,
        complete: Option<&WorkingMemoryStorage<K>>,
        limits: &WorkspaceCopyLimits,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError> {
        let prepared = complete
            .and_then(WorkingMemoryStorage::source_preparation)
            .or_else(|| copy.arrays.source().registration().source_preparation())
            .is_some();
        let controls = sampling_controls::<K>(prepared, complete.is_some())?;
        super::workspace_copy::with_copy_projection(
            self,
            copy.arrays.incremental_requirements(),
            copy.host_bytes,
            limits,
            controls,
            |mut projection, controls| {
                projection.host_bytes = projection
                    .host_bytes
                    .checked_add(controls)
                    .ok_or(WorkingMemoryError::Overflow)?;
                projection.materialize(self.topology())
            },
        )
    }
}

#[cfg(test)]
mod tests;
