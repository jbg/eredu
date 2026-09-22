//! Closed isolated-copy demand associated with already registered source roots.

use super::{
    InferenceExecutionIdentity, MemoryLedger, RegisteredWorkspaceStorage, WorkingMemoryError,
    WorkingMemoryFundingRun, WorkingMemoryFundingScope, residual::RegisteredStoragePin,
};
use eredu_nn::workspace::{WorkspaceIsolatedCopyPlan, WorkspaceTraceReport};

mod account;
pub(in crate::working_memory) mod completed;
mod numerical;
pub use account::WorkspaceCopyAccountLayout;
pub use completed::{
    CompletedWorkspaceSourceAccount, CompletedWorkspaceSourceLayout, CompletedWorkspaceStorage,
    CompletedWorkspaceStorageLayout, OriginalCompletedWorkspaceCopy,
    OriginalCompletedWorkspaceSource,
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
    requirements: eredu_core::DomainMemoryRequirements,
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
        let requirements = plan
            .incremental_requirements()
            .ok_or(WorkingMemoryError::UnknownBound)?
            .clone();
        Ok(Self {
            plan,
            source,
            requirements,
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

    pub(super) fn incremental_requirements(&self) -> &eredu_core::DomainMemoryRequirements {
        &self.requirements
    }
    pub(super) fn incremental_bytes(&self) -> Option<u64> {
        diagnostic_bytes(&self.requirements)
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
            || !registered.pool().same_ledger(source.pool())
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
            size_of::<(&MemoryLedger, WorkspaceCopyLimits)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// Capacity policy for one independently funded isolated-copy operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkspaceCopyLimits {
    /// Total live-charge limits in each physical domain, including sources and
    /// all other accounts. Tighter existing constraints remain effective.
    pub memory_limits: eredu_core::MemoryLimitDeclarations,
    /// Additional domain-attributed charge retained with destination custody.
    pub additional_headroom: eredu_core::MemoryHeadroomDeclarations,
    /// Exact additional host metadata retained by the selected copy mechanism.
    pub additional_host_metadata_bytes: u64,
    /// Additional placed allowances certified by the selected copy mechanism.
    pub additional_requirements: Option<std::sync::Arc<eredu_core::DomainMemoryRequirements>>,
}

impl WorkspaceCopyLimits {
    /// Selects domain limits without additional headroom.
    pub const fn new(memory_limits: eredu_core::MemoryLimitDeclarations) -> Self {
        Self {
            memory_limits,
            additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
            additional_host_metadata_bytes: 0,
            additional_requirements: None,
        }
    }
}

/// Rejection before publishing destination-copy funding.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceCopyAdmissionError {
    /// Missing numerical facts, mismatched source custody or shared accounting.
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
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
        requirements: eredu_core::DomainMemoryRequirements,
        funding: WorkingMemoryFundingRun,
        scope: WorkingMemoryFundingScope,
    ) -> Self {
        Self {
            scope,
            custody: WorkspaceCopyCustody {
                retention: WorkspaceCopyRetention(Some(std::sync::Arc::new(CopyAccount {
                    requirements,
                    execution,
                    funding,
                }))),
            },
        }
    }

    /// Complete domain-attributed charge retained by this operation.
    pub fn requirements(&self) -> &eredu_core::DomainMemoryRequirements {
        self.custody.requirements()
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
    requirements: eredu_core::DomainMemoryRequirements,
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
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.account().funding.pool().same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
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
    pub fn pool(&self) -> &MemoryLedger {
        self.retention.account().funding.pool()
    }

    /// Complete domain-attributed charge retained by this operation.
    pub fn requirements(&self) -> &eredu_core::DomainMemoryRequirements {
        &self.retention.account().requirements
    }
}

impl MemoryLedger {
    /// Admits a closed isolated-copy program against its registered sources.
    /// Source-origin health and destination capacity are validated atomically.
    /// This performs no native copying, evaluation, completion or publication.
    ///
    /// The domain charge covers tensor storage, operation-owned host workspace,
    /// and the account, report and source-pin controls. Unrelated host snapshot,
    /// controller and decoder payloads require their own attributed producer.
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
        if !self.same_ledger(copy.source.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let preparation = copy.source.registration().source_preparation();
        let direct_controls = workspace_controls::<K>(preparation.is_some(), prepared.is_some())?;
        let accepted = prepare_copy_account(
            self,
            &self.construction_identity(),
            copy.incremental_requirements(),
            0,
            &limits,
            direct_controls,
            super::funding::CopyHostHolds::None,
            |usage| copy.source.registration().validate_copy_source(self, usage),
        )?;
        let execution = match preparation {
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
        let (requirements, funding, scope) = accepted.workspace(&execution, Some(pin))?;
        Ok(AdmittedWorkspaceCopy::from_account(
            execution,
            requirements,
            funding,
            scope,
        ))
    }
}

fn workspace_controls<K: Ord + Send + Sync + 'static>(
    prepared: bool,
    attached: bool,
) -> Result<usize, WorkingMemoryError> {
    if prepared {
        return Ok(0);
    }
    WorkspaceCopyAccountLayout::workspace()?
        .requested_bytes()
        .checked_add(RegisteredStoragePin::single_control_bytes::<K>(false)?)
        .and_then(|n| {
            n.checked_add(if attached {
                RegisteredStoragePin::pair_control_bytes(true).ok()?
            } else {
                0
            })
        })
        .ok_or(WorkingMemoryError::Overflow)
}
impl MemoryLedger {
    /// Quotes this source-qualified copy's entire incremental domain charge,
    /// including its account and pin construction. This grants no allocation authority.
    pub fn workspace_copy_requirements<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: &RegisteredWorkspaceCopy<K>,
        limits: &WorkspaceCopyLimits,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError> {
        if !self.same_ledger(copy.source.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let controls =
            workspace_controls::<K>(copy.source.registration().has_source_preparation(), false)?;
        with_copy_projection(
            self,
            copy.incremental_requirements(),
            0,
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

pub(super) fn prepare_copy_account(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    native: &eredu_core::DomainMemoryRequirements,
    host_bytes: u64,
    limits: &WorkspaceCopyLimits,
    direct_controls: usize,
    holds: super::funding::CopyHostHolds,
    validate: impl FnOnce(&super::Usage) -> Result<(), WorkingMemoryError>,
) -> Result<super::funding::PreparedCopyAccount, WorkingMemoryError> {
    with_copy_projection(
        pool,
        native,
        host_bytes,
        limits,
        direct_controls,
        |projection, controls| {
            super::funding::PreparedCopyAccount::accept(
                pool,
                execution,
                projection,
                &limits.memory_limits,
                controls,
                holds,
                validate,
            )
        },
    )
}

pub(super) fn with_copy_projection<T>(
    pool: &MemoryLedger,
    native: &eredu_core::DomainMemoryRequirements,
    host_bytes: u64,
    limits: &WorkspaceCopyLimits,
    direct_controls: usize,
    use_projection: impl FnOnce(
        super::transaction_buffers::RequirementProjection<'_>,
        u64,
    ) -> Result<T, WorkingMemoryError>,
) -> Result<T, WorkingMemoryError> {
    let parts = [
        native,
        limits.additional_requirements.as_deref().unwrap_or(native),
    ];
    let parts = &parts[..if limits.additional_requirements.is_some() {
        2
    } else {
        1
    }];
    let additional_metadata = limits
        .additional_requirements
        .as_ref()
        .map(|additional| {
            additional
                .backing_bytes()?
                .checked_add(super::qualified_storage::shared_bytes::<
                    eredu_core::DomainMemoryRequirements,
                >()?)
                .ok_or(WorkingMemoryError::Overflow)
        })
        .transpose()?
        .unwrap_or(0);
    let projection = super::transaction_buffers::RequirementProjection {
        parts,
        headroom: &limits.additional_headroom,
        host_bytes: host_bytes
            .checked_add(limits.additional_host_metadata_bytes)
            .and_then(|n| n.checked_add(additional_metadata))
            .ok_or(WorkingMemoryError::Overflow)?,
    };
    let controls = super::funding::copy_domain_controls(pool, &projection)?
        .checked_add(u64::try_from(direct_controls).map_err(|_| WorkingMemoryError::Overflow)?)
        .ok_or(WorkingMemoryError::Overflow)?;
    use_projection(projection, controls)
}

fn diagnostic_bytes(requirements: &eredu_core::DomainMemoryRequirements) -> Option<u64> {
    requirements.iter().try_fold(0_u64, |sum, (_, charge)| {
        sum.checked_add(charge.total().ok()?)
    })
}

#[cfg(test)]
mod tests;
