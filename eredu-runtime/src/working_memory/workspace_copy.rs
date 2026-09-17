//! Closed isolated-copy demand associated with already registered source roots.

use super::{
    InferenceExecutionIdentity, RegisteredWorkspaceStorage, WorkingMemoryError,
    WorkingMemoryFundingRun, WorkingMemoryFundingScope, WorkingMemoryPool,
    residual::RegisteredStoragePin,
};
use eredu_nn::workspace::{WorkspaceIsolatedCopyPlan, WorkspaceTraceReport};

mod account;
pub(in crate::working_memory) mod completed;
mod numerical;
pub use account::WorkspaceCopyAccountLayout;
pub use completed::{
    CompletedWorkspaceSourceAccount, CompletedWorkspaceSourceLayout, CompletedWorkspaceStorage, CompletedWorkspaceStorageLayout,
    OriginalCompletedWorkspaceCopy, OriginalCompletedWorkspaceSource,
};
pub use numerical::OriginalNumericalWorkspaceCopy;

/// A sealed isolated-copy program paired with exact, already charged sources.
/// Neither a mutable diagnostic report nor a scalar byte count can construct
/// this association. Native callers must retain the corresponding physical
/// witnesses and establish exclusive, settled source access independently.
#[derive(Debug)]
pub struct RegisteredWorkspaceCopy<K: Ord + Send + 'static> {
    plan: WorkspaceIsolatedCopyPlan,
    source: RegisteredWorkspaceStorage<K>,
    bytes: u64,
}

impl<K: Clone + Ord + Send + Sync + 'static> RegisteredWorkspaceCopy<K> {
    /// Associates the plan's original borrowed-root selection with its exact
    /// registry pin. The plan's private rebased trace is not caller evidence.
    /// Missing operation or host facts reject rather than admitting a partial
    /// numerical component. Existing source bytes remain separately charged.
    pub fn bind(
        plan: WorkspaceIsolatedCopyPlan,
        source: RegisteredWorkspaceStorage<K>,
    ) -> Result<Self, WorkspaceCopyAdmissionError> {
        if !plan
            .source_storage()
            .same_identity(source.borrowed_storage())
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let bytes = plan
            .incremental_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(Self {
            plan,
            source,
            bytes,
        })
    }

    /// Immutable diagnostics of the actual closed copy program. A cloned and
    /// edited report cannot change the retained admission bound.
    pub fn report(&self) -> &WorkspaceTraceReport {
        self.plan.report()
    }

    pub(super) fn source(&self) -> &RegisteredWorkspaceStorage<K> {
        &self.source
    }

    pub(super) fn incremental_bytes(&self) -> u64 {
        self.bytes
    }
}

/// A copy whose source also retains genuine original prepared-input B roots.
/// This cannot enter legacy sampler/decoder aggregate constructors, whose pins
/// describe registered origins only. It uses the same isolated-copy account.
///
/// ```compile_fail
/// use eredu_runtime::working_memory::{BorrowedFundedSampler, RegisteredPreparedWorkspaceCopy, RegisteredSamplingCopy};
/// fn cannot_erase_b<'a>(sampler: BorrowedFundedSampler<'a>, copy: RegisteredPreparedWorkspaceCopy<u64>) {
///     let _ = RegisteredSamplingCopy::prepare(sampler, copy);
/// }
/// ```
#[derive(Debug)]
pub struct RegisteredPreparedWorkspaceCopy<K: Ord + Send + 'static> {
    copy: RegisteredWorkspaceCopy<K>,
    source: crate::input::OriginalPreparedWorkspaceSource,
}
impl<K: Clone + Ord + Send + Sync + 'static> RegisteredPreparedWorkspaceCopy<K> {
    /// Bind exact source-root identity and the actual B account, never a raw
    /// native allocation or a scalar credit. H only retains constructor storage.
    pub fn bind(
        plan: WorkspaceIsolatedCopyPlan,
        source: super::RegisteredPreparedWorkspaceStorage<K>,
    ) -> Result<Self, WorkspaceCopyAdmissionError> {
        let (registered, source) = source.into_copy_parts();
        if registered.registration().source_preparation().is_none()
            || !registered.pool().same_domain(source.pool())
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(Self {
            copy: RegisteredWorkspaceCopy::bind(plan, registered)?,
            source,
        })
    }
    /// Fixed closed transport, in addition to source binding and account layout.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, WorkspaceCopyAdmissionError>>(),
            size_of::<Option<crate::input::OriginalPreparedWorkspaceSource>>(),
            size_of::<(
                RegisteredWorkspaceCopy<K>,
                crate::input::OriginalPreparedWorkspaceSource,
            )>(),
            size_of::<(&WorkingMemoryPool, WorkspaceCopyLimits)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// Capacity policy for one independently funded isolated-copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceCopyLimits {
    /// Ceiling on the complete shared domain, including sources and all other
    /// accounts. This never raises a tighter existing descendant ceiling.
    pub capacity_bytes: u64,
    /// Optional limit on this copy's incremental demand plus safety reserve.
    pub application_memory_budget_bytes: Option<u64>,
    /// Conservative additional charge retained with destination custody.
    pub safety_reserve_bytes: u64,
}

impl WorkspaceCopyLimits {
    /// Selects a domain ceiling without an additional application limit/reserve.
    pub const fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            application_memory_budget_bytes: None,
            safety_reserve_bytes: 0,
        }
    }
}

/// Rejection before publishing destination-copy funding.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceCopyAdmissionError {
    /// Missing numerical facts, mismatched source custody or shared accounting.
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
    /// Complete copy demand and safety exceed its application allowance.
    #[error("workspace copy needs {required_bytes} bytes; application limit is {budget_bytes}")]
    ApplicationBudgetExceeded {
        /// Incremental closed-program demand plus safety reserve.
        required_bytes: u64,
        /// Explicit per-copy application allowance.
        budget_bytes: u64,
    },
}

/// One admitted copy with exactly one work scope. Dropping it uncertified
/// conservatively quarantines its demand and source pins, even before work.
/// This grants no inference request, future step or capacity handoff.
#[derive(Debug)]
#[must_use = "retain this operation through copy completion and publication"]
pub struct AdmittedWorkspaceCopy {
    // Quarantine before closing the run if an unsplit operation is abandoned.
    scope: WorkingMemoryFundingScope,
    custody: WorkspaceCopyCustody,
}

impl AdmittedWorkspaceCopy {
    pub(super) fn from_account(
        execution: InferenceExecutionIdentity,
        bytes: u64,
        funding: WorkingMemoryFundingRun,
        scope: WorkingMemoryFundingScope,
    ) -> Self {
        Self {
            scope,
            custody: WorkspaceCopyCustody {
                retention: WorkspaceCopyRetention(Some(std::sync::Arc::new(CopyAccount {
                    bytes,
                    execution,
                    funding,
                }))),
            },
        }
    }

    /// Complete admitted operation demand plus the requested safety reserve.
    /// An aggregate sampling copy also includes its protected host component.
    pub fn bytes(&self) -> u64 {
        self.custody.bytes()
    }

    /// Transfers the single work scope into native completion/recovery and
    /// destination custody into the saved component. Retain both through native
    /// work. Certify the scope only after settlement and complete publication;
    /// its source pin then retires independently of destination lifetime.
    pub fn into_parts(self) -> (WorkspaceCopyCustody, WorkingMemoryFundingScope) {
        (self.custody, self.scope)
    }
}

/// Destination lifetime only, with no scope factory or execution permission.
/// Put this after the saved component's managed payload fields. Once every
/// retention alias drops, certified work releases the remaining envelope;
/// published allocations retain their own charges and capacity ceiling.
#[derive(Debug)]
#[must_use = "retain custody until the destination component's payload retires"]
pub struct WorkspaceCopyCustody {
    retention: WorkspaceCopyRetention,
}

#[derive(Debug)]
struct CopyAccount {
    bytes: u64,
    // This same identity is cloned by sampler/table host owners. Each closed
    // clone retains its preparation until the final strong/ledger Weak dies.
    execution: InferenceExecutionIdentity,
    funding: WorkingMemoryFundingRun,
}

/// Accounting-only retention of one already admitted copy. This retains no
/// numerical payload and grants no source credit, native scope or new budget.
/// Put it after the completion, quota, error or buffer resources it covers.
#[derive(Debug)]
#[must_use = "retain until this alias's covered copy resources retire"]
pub struct WorkspaceCopyRetention(Option<std::sync::Arc<CopyAccount>>);
impl Clone for WorkspaceCopyRetention {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for WorkspaceCopyRetention {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // No raw Arc or Weak escapes. The final alias frees the shared shell
            // before the identity or funding run can return the host account.
            drop(std::sync::Arc::into_inner(owner));
        }
    }
}
impl WorkspaceCopyRetention {
    /// Check this admitted copy's exact managed pool without granting source
    /// credit or certifying completion. Native published-source owners must
    /// independently authenticate their actual allocation and completed stream.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.account().funding.pool().same_domain(pool) { Ok(()) }
        else { Err(WorkingMemoryError::IdentityMismatch) }
    }
    fn account(&self) -> &CopyAccount {
        self.0.as_deref().expect("live copy retention")
    }
    pub(in crate::working_memory) fn preparation(
        &self,
    ) -> Option<&eredu_core::HostPreparationAuthority> {
        self.account().execution.1.as_ref()
    }
    pub(in crate::working_memory) fn validate_publication(
        &self,
        scope: &WorkingMemoryFundingScope,
        usage: &super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        let run = super::funding::FundingSource::CopyRun(&self.account().funding);
        if !run.same_account(super::funding::FundingSource::NativeScope(scope)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        scope.validate_native_purpose()?;
        run.validate(usage, &self.account().execution)
    }
}

impl WorkspaceCopyCustody {
    pub(super) fn funding_source(&self) -> super::funding::FundingSource<'_> {
        super::funding::FundingSource::CopyRun(&self.retention.account().funding)
    }

    pub(super) fn execution(&self) -> &InferenceExecutionIdentity {
        &self.retention.account().execution
    }

    /// Allocation-free closed alias for native completion and resource owners.
    /// It keeps this exact account alive and exposes no funding operations.
    pub fn retention(&self) -> WorkspaceCopyRetention {
        self.retention.clone()
    }

    /// The exact managed domain of this destination account.
    pub fn pool(&self) -> &WorkingMemoryPool {
        self.retention.account().funding.pool()
    }

    /// Original incremental demand plus safety, not current remaining balance.
    pub fn bytes(&self) -> u64 {
        self.retention.account().bytes
    }
}

impl WorkingMemoryPool {
    /// Admits a closed isolated-copy program against its registered sources.
    /// Source-origin health and destination capacity are validated atomically.
    /// This performs no native copying, evaluation, completion or publication.
    ///
    /// The numerical boundary is the workspace contract: tensor storage and
    /// operation-owned host workspace. Runtime bookkeeping and unrelated host
    /// snapshot/controller/decoder payloads are not covered by this component.
    /// Native callers must execute exactly the prepared slots under their own
    /// checked source binding and retain every partial result through recovery.
    pub fn admit_workspace_copy<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: RegisteredWorkspaceCopy<K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<AdmittedWorkspaceCopy, WorkspaceCopyAdmissionError> {
        self.admit_workspace_copy_inner(copy, None, limits)
    }
    /// Same capacity comparison and sole copy scope, with the original B pin.
    /// The closed prepared source cannot be stripped by another copy consumer.
    pub fn admit_prepared_workspace_copy<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: RegisteredPreparedWorkspaceCopy<K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<AdmittedWorkspaceCopy, WorkspaceCopyAdmissionError> {
        self.admit_workspace_copy_inner(copy.copy, Some(copy.source), limits)
    }
    fn admit_workspace_copy_inner<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: RegisteredWorkspaceCopy<K>,
        prepared: Option<crate::input::OriginalPreparedWorkspaceSource>,
        limits: WorkspaceCopyLimits,
    ) -> Result<AdmittedWorkspaceCopy, WorkspaceCopyAdmissionError> {
        if !self.same_domain(copy.source.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let bytes = copy_requirement(copy.bytes, limits)?;
        // Metadata and erased pin construction happen before accounting locks.
        // The public caller cannot use this identity to fabricate source proof.
        let execution = match copy.source.registration().source_preparation() {
            Some(preparation) => WorkspaceCopyAccountLayout::workspace()?.execution(preparation),
            None => InferenceExecutionIdentity::default(),
        };
        let pin = RegisteredStoragePin::new(copy.source.registration().clone());
        let pin = match prepared.as_ref() {
            Some(source) => RegisteredStoragePin::pair(
                pin,
                RegisteredStoragePin::PreparedInput(source.account_pin()),
            ),
            None => pin,
        };
        let (funding, scope) = self.open_workspace_copy_account(
            copy.source.registration(),
            pin,
            &execution,
            bytes,
            limits.capacity_bytes,
        )?;
        Ok(AdmittedWorkspaceCopy::from_account(
            execution, bytes, funding, scope,
        ))
    }
}

fn copy_requirement(
    bytes: u64,
    limits: WorkspaceCopyLimits,
) -> Result<u64, WorkspaceCopyAdmissionError> {
    let bytes = bytes
        .checked_add(limits.safety_reserve_bytes)
        .ok_or(WorkingMemoryError::Overflow)?;
    if let Some(budget_bytes) = limits.application_memory_budget_bytes {
        if bytes > budget_bytes {
            return Err(WorkspaceCopyAdmissionError::ApplicationBudgetExceeded {
                required_bytes: bytes,
                budget_bytes,
            });
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
