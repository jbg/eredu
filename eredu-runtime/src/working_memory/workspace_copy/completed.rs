//! Completed original allocation sources for workspace bindings and copies.
use super::super::{
    OriginalNumericalBudgetCustody, OriginalSpeculativeBudgetCustody,
    OriginalSpeculativeNumericalBudgetCustody, Usage, qualified_storage,
};
use super::*;
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceContext, WorkspaceExistingStorage};
use std::{
    mem::{size_of, size_of_val},
    sync::Arc,
};

/// One actual completed allocation account. The tag preserves the original
/// model invocation, numerical phase, or standalone operation. None can
/// impersonate another source role.
#[derive(Debug, Clone)]
pub enum CompletedWorkspaceSourceAccount {
    /// Exact completed model-role allocation custody.
    Model(OriginalSpeculativeBudgetCustody),
    /// Exact completed numerical-phase allocation custody.
    Numerical(OriginalSpeculativeNumericalBudgetCustody),
    /// Completed standalone numerical work; this carries no speculative role.
    Standalone(OriginalNumericalBudgetCustody),
}
impl CompletedWorkspaceSourceAccount {
    /// Borrow original provenance without issuing a role. Standalone sources
    /// never belong to a speculative request.
    pub fn source(&self) -> crate::working_memory::SpeculativeNumericalSource<'_> {
        match self {
            Self::Standalone(value) => {
                crate::working_memory::SpeculativeNumericalSource::Standalone(value)
            }
            Self::Model(value) => crate::working_memory::SpeculativeNumericalSource::Model(value),
            Self::Numerical(value) => {
                crate::working_memory::SpeculativeNumericalSource::Numerical(value)
            }
        }
    }
    /// Exact account equality; sharing a request or pool is insufficient.
    pub fn same_account(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Standalone(a), Self::Standalone(b)) => a.same_account(b),
            (Self::Model(a), Self::Model(b)) => a.same_account(b),
            (Self::Numerical(a), Self::Numerical(b)) => a.same_account(b),
            _ => false,
        }
    }
    fn pool(&self) -> &MemoryLedger {
        match self {
            Self::Standalone(a) => a.pool(),
            Self::Model(a) => a.pool(),
            Self::Numerical(a) => a.pool(),
        }
    }
    fn validate_roots<'a>(
        &self,
        roots: impl Iterator<Item = &'a WorkspaceExistingStorage> + Clone,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Standalone(account) => account.validate_completed_roots(roots),
            Self::Model(account) => {
                validate_capacity(roots, account.physical_bytes(), account.placement())
            }
            Self::Numerical(account) => {
                validate_capacity(roots, account.physical_bytes(), account.placement())
            }
        }
    }
    fn validate_copy_source(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Standalone(a) => a.validate_copy_source(pool, usage),
            Self::Model(a) => a.validate_copy_source(pool, usage),
            Self::Numerical(a) => a.validate_copy_source(pool, usage),
        }
    }
}
impl From<OriginalSpeculativeBudgetCustody> for CompletedWorkspaceSourceAccount {
    fn from(value: OriginalSpeculativeBudgetCustody) -> Self {
        Self::Model(value)
    }
}
trait AccountInput: Into<CompletedWorkspaceSourceAccount> {
    fn pool(&self) -> &MemoryLedger;
    fn same_account(&self, other: &Self) -> bool;
    fn validate_roots<'a>(
        &self,
        roots: impl Iterator<Item = &'a WorkspaceExistingStorage> + Clone,
    ) -> Result<(), WorkingMemoryError>;
}
impl AccountInput for OriginalSpeculativeBudgetCustody {
    fn pool(&self) -> &MemoryLedger {
        self.pool()
    }
    fn same_account(&self, other: &Self) -> bool {
        self.same_account(other)
    }
    fn validate_roots<'a>(
        &self,
        roots: impl Iterator<Item = &'a WorkspaceExistingStorage> + Clone,
    ) -> Result<(), WorkingMemoryError> {
        validate_capacity(roots, self.physical_bytes(), self.placement())
    }
}
impl AccountInput for CompletedWorkspaceSourceAccount {
    fn pool(&self) -> &MemoryLedger {
        self.pool()
    }
    fn same_account(&self, other: &Self) -> bool {
        self.same_account(other)
    }
    fn validate_roots<'a>(
        &self,
        roots: impl Iterator<Item = &'a WorkspaceExistingStorage> + Clone,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_roots(roots)
    }
}

fn validate_capacity<'a>(
    mut roots: impl Iterator<Item = &'a WorkspaceExistingStorage>,
    capacity: u64,
    placement: &eredu_core::MemoryPlacement,
) -> Result<(), WorkingMemoryError> {
    let total = roots.try_fold(0u64, |sum, root| {
        if root.placement() != Some(placement) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        sum.checked_add(
            root.capacity_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
        )
        .ok_or(WorkingMemoryError::Overflow)
    })?;
    if total > capacity {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    Ok(())
}

#[derive(Debug)]
struct AccountSources {
    accounts: Vec<CompletedWorkspaceSourceAccount>,
    host: HostPreparationAuthority,
}
// This is the existing shared account bundle, not a new allocation or grant.
// Erasing its concrete Arc stops account/pool/pin auto-trait expansion at the
// closed custody boundary. The compiler checks the concrete Send + Sync proof
// where that same Arc is unsized after all source checks have passed.
pub(in crate::working_memory) trait CompletedAccountCustody:
    std::fmt::Debug + Send + Sync
{
    fn host(&self) -> &HostPreparationAuthority;
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError>;
    fn matches_pool(&self, pool: &MemoryLedger) -> bool;
    // Recover the concrete Arc before retirement so its shell dies before the
    // vector, account aliases and host authority that paid their storage.
    fn retire(self: Arc<Self>);
}
impl CompletedAccountCustody for AccountSources {
    fn host(&self) -> &HostPreparationAuthority {
        &self.host
    }
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError> {
        for account in &self.accounts {
            account.validate_copy_source(pool, usage)?;
        }
        Ok(())
    }
    fn matches_pool(&self, pool: &MemoryLedger) -> bool {
        self.accounts
            .iter()
            .all(|account| account.pool().same_ledger(pool))
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug)]
pub(in crate::working_memory) enum CompletedSourceCustody {
    Numerical {
        account: OriginalSpeculativeNumericalBudgetCustody,
        host: HostPreparationAuthority,
    },
    Accounts(Option<Arc<dyn CompletedAccountCustody>>),
}
impl Clone for CompletedSourceCustody {
    fn clone(&self) -> Self {
        match self {
            Self::Numerical { account, host } => Self::Numerical {
                account: account.clone(),
                host: host.clone(),
            },
            Self::Accounts(owner) => Self::Accounts(owner.clone()),
        }
    }
}
impl Drop for CompletedSourceCustody {
    fn drop(&mut self) {
        if let Self::Accounts(owner) = self {
            if let Some(owner) = owner.take() {
                owner.retire();
            }
        }
    }
}
impl CompletedSourceCustody {
    pub(in crate::working_memory) fn host(&self) -> &HostPreparationAuthority {
        match self {
            Self::Numerical { host, .. } => host,
            Self::Accounts(owner) => owner.as_deref().expect("live completed sources").host(),
        }
    }
    pub(in crate::working_memory) fn validate(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Numerical { account, .. } => account.validate_copy_source(pool, usage),
            Self::Accounts(owner) => owner
                .as_deref()
                .expect("live completed sources")
                .validate(pool, usage),
        }
    }
    pub(in crate::working_memory) fn matches_pool(&self, pool: &MemoryLedger) -> bool {
        match self {
            Self::Numerical { account, .. } => account.pool().same_ledger(pool),
            Self::Accounts(owner) => owner
                .as_deref()
                .expect("live completed sources")
                .matches_pool(pool),
        }
    }
}

/// Exact completed roots plus their original account-only custody. A backend
/// must authenticate each full backing against its actual originating native
/// budget after successful completion. This creates no registry row or grant.
#[derive(Debug)]
pub struct OriginalCompletedWorkspaceSource {
    pub(in crate::working_memory) roots: WorkspaceBorrowedStorage,
    pub(in crate::working_memory) custody: CompletedSourceCustody,
}
impl OriginalCompletedWorkspaceSource {
    pub(super) fn numerical(
        roots: &WorkspaceBorrowedStorage,
        account: OriginalSpeculativeNumericalBudgetCustody,
        host: HostPreparationAuthority,
    ) -> Result<Self, WorkingMemoryError> {
        if host.is_unmanaged() || roots.roots().is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        validate_capacity(
            roots.roots().iter(),
            account.physical_bytes(),
            account.placement(),
        )?;
        Ok(Self {
            roots: roots.clone(),
            custody: CompletedSourceCustody::Numerical { account, host },
        })
    }
    /// Source metadata only; account custody remains closed in this owner.
    pub fn borrowed_storage(&self) -> &WorkspaceBorrowedStorage {
        &self.roots
    }
}

/// Constructor layout from the actual unique completed-root population. Input
/// row storage is the caller's paid destination; this prices the consumed root
/// selection, account vector, closed shared shell and fixed transports.
#[derive(Debug)]
pub struct CompletedWorkspaceSourceLayout {
    slots: usize,
    bytes: usize,
    tagged: bool,
}
impl CompletedWorkspaceSourceLayout {
    /// No source, native allocation or account is constructed by this query.
    pub fn new(slots: usize) -> Result<Self, WorkingMemoryError> {
        Self::new_for::<OriginalSpeculativeBudgetCustody>(slots, false)
    }
    /// Same destination for exact mixed model/numerical account rows.
    pub fn new_accounts(slots: usize) -> Result<Self, WorkingMemoryError> {
        Self::new_for::<CompletedWorkspaceSourceAccount>(slots, true)
    }
    fn new_for<A: AccountInput>(slots: usize, tagged: bool) -> Result<Self, WorkingMemoryError> {
        let parts = [
            WorkspaceBorrowedStorage::construction_bytes(slots)
                .ok_or(WorkingMemoryError::Overflow)?,
            std::alloc::Layout::array::<CompletedWorkspaceSourceAccount>(slots)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            usize::try_from(qualified_storage::vector_control_bytes::<
                CompletedWorkspaceSourceAccount,
            >()?)
            .map_err(|_| WorkingMemoryError::Overflow)?,
            usize::try_from(qualified_storage::shared_bytes::<AccountSources>()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<AccountSources>(),
            size_of::<Option<AccountSources>>(),
            size_of::<Arc<AccountSources>>(),
            size_of::<Arc<dyn CompletedAccountCustody>>(),
            size_of::<Option<Arc<dyn CompletedAccountCustody>>>(),
            size_of::<&dyn CompletedAccountCustody>(),
            size_of::<(&dyn CompletedAccountCustody, &MemoryLedger, &Usage)>(),
            size_of::<(&dyn CompletedAccountCustody, &MemoryLedger)>(),
            size_of::<&HostPreparationAuthority>(),
            size_of::<std::slice::Iter<'_, CompletedWorkspaceSourceAccount>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<CompletedSourceCustody>(),
            size_of::<OriginalCompletedWorkspaceSource>(),
            size_of::<Result<OriginalCompletedWorkspaceSource, WorkingMemoryError>>(),
            size_of::<Vec<(WorkspaceExistingStorage, A)>>(),
            size_of::<std::vec::IntoIter<(WorkspaceExistingStorage, A)>>(),
            size_of::<(usize, usize, u64)>(),
            size_of::<A>(),
            size_of::<CompletedWorkspaceSourceAccount>(),
            size_of::<(&A, &A)>(),
            size_of::<(
                &CompletedWorkspaceSourceAccount,
                &CompletedWorkspaceSourceAccount,
            )>(),
            size_of::<bool>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<(&WorkspaceContext, &HostPreparationAuthority)>(),
            size_of::<std::slice::Iter<'_, (WorkspaceExistingStorage, A)>>(),
            size_of::<Result<u64, WorkingMemoryError>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            slots,
            bytes,
            tagged,
        })
    }
    /// Host bytes which must be reserved before construction.
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }
    /// Consumes exact provider-authenticated root/account rows. Multiple roots
    /// may share an account; their unique full capacities must fit that actual
    /// account. Duplicate or contradictory root rows reject before allocation.
    pub fn construct(
        self,
        context: &WorkspaceContext,
        entries: Vec<(WorkspaceExistingStorage, OriginalSpeculativeBudgetCustody)>,
        host: &HostPreparationAuthority,
    ) -> Result<OriginalCompletedWorkspaceSource, WorkingMemoryError> {
        if self.tagged {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.construct_entries(context, entries, host)
    }
    /// Consumes provider-authenticated tagged rows through the same full-root
    /// uniqueness, capacity, pool and later admission checks as model rows.
    pub fn construct_accounts(
        self,
        context: &WorkspaceContext,
        entries: Vec<(WorkspaceExistingStorage, CompletedWorkspaceSourceAccount)>,
        host: &HostPreparationAuthority,
    ) -> Result<OriginalCompletedWorkspaceSource, WorkingMemoryError> {
        if !self.tagged {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.construct_entries(context, entries, host)
    }
    fn construct_entries<A: AccountInput>(
        self,
        context: &WorkspaceContext,
        entries: Vec<(WorkspaceExistingStorage, A)>,
        host: &HostPreparationAuthority,
    ) -> Result<OriginalCompletedWorkspaceSource, WorkingMemoryError> {
        if host.is_unmanaged() || entries.len() > self.slots || entries.is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        for (i, (root, account)) in entries.iter().enumerate() {
            if entries[..i]
                .iter()
                .any(|(prior, _)| prior.same_storage(root))
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            if !account.pool().same_ledger(entries[0].1.pool()) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            account.validate_roots(
                entries
                    .iter()
                    .filter(|(_, a)| a.same_account(account))
                    .map(|(root, _)| root),
            )?;
        }
        let roots = WorkspaceBorrowedStorage::new_finite(
            context,
            entries.iter().map(|(root, _)| root),
            self.slots,
        )
        .map_err(root_error)?;
        let mut accounts = qualified_storage::vector(self.slots, true)?;
        for (_, account) in entries {
            accounts.push(account.into());
        }
        Ok(OriginalCompletedWorkspaceSource {
            roots,
            custody: CompletedSourceCustody::Accounts(Some(Arc::new(AccountSources {
                accounts,
                host: host.clone(),
            }))),
        })
    }
}
fn root_error(cause: eredu_nn::workspace::WorkspaceBorrowedStorageError) -> WorkingMemoryError {
    match cause {
        eredu_nn::workspace::WorkspaceBorrowedStorageError::Reserve(cause) => {
            WorkingMemoryError::ControlStorageReserve(cause)
        }
        eredu_nn::workspace::WorkspaceBorrowedStorageError::Overflow => {
            WorkingMemoryError::Overflow
        }
        _ => WorkingMemoryError::IdentityMismatch,
    }
}

/// Same registered-source binding with additional exact completed role roots.
/// The inner registered pin cannot be extracted independently of those roots.
#[derive(Debug)]
pub struct CompletedWorkspaceStorage<K: Ord + Send + 'static> {
    registered: RegisteredWorkspaceStorage<K>,
    source: OriginalCompletedWorkspaceSource,
}
impl<K: Clone + Ord + Send + Sync + 'static> CompletedWorkspaceStorage<K> {
    /// Complete selection used by the shared isolated-copy equations.
    pub fn borrowed_storage(&self) -> &WorkspaceBorrowedStorage {
        self.registered.borrowed_storage()
    }
}
/// Uses the existing registered-storage construction with a supplemental source.
#[derive(Debug)]
pub struct CompletedWorkspaceStorageLayout<K: Ord + Send + 'static> {
    layout: super::super::RegisteredWorkspaceStorageLayout<K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> CompletedWorkspaceStorageLayout<K> {
    /// Counts only already registered roots in `registered`; role roots remain
    /// separately charged by their actual completed source accounts.
    pub fn new(registered: usize, completed: usize) -> Result<Self, WorkingMemoryError> {
        Ok(Self {
            layout: super::super::RegisteredWorkspaceStorageLayout::with_completed_roots(
                registered, completed,
            )?,
        })
    }
    /// Includes the same actual mixed-pin account shell and fixed transport.
    pub fn requested_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let frames = [
            self.layout.requested_bytes(),
            size_of::<Self>(),
            size_of::<(
                Self,
                &MemoryLedger,
                &WorkspaceContext,
                OriginalCompletedWorkspaceSource,
                &mut super::super::RetainedOriginalStorageSources,
                bool,
            )>(),
            size_of::<Result<CompletedWorkspaceStorage<K>, WorkingMemoryError>>(),
            size_of::<CompletedWorkspaceStorage<K>>(),
            size_of::<Result<CompletedWorkspaceStorage<K>, WorkingMemoryError>>(),
            OriginalCompletedWorkspaceCopy::<K>::control_bytes()?,
            RegisteredStoragePin::single_control_bytes::<K>(true)?,
            RegisteredStoragePin::pair_control_bytes(true)?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// No new physical registration: only existing rows go to the original pin
    /// worker. Every additional root remains bound to the supplied closed source.
    pub fn construct(
        self,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
        registered: impl IntoIterator<Item = impl Into<super::super::RegisteredWorkspaceStorageRow<K>>>,
        source: OriginalCompletedWorkspaceSource,
        carrier: &mut super::super::RetainedOriginalStorageSources,
    ) -> Result<CompletedWorkspaceStorage<K>, WorkingMemoryError>
    where
        K: super::super::HostSlotStorageKey,
    {
        self.construct_inner(pool, context, registered, source, carrier, true)
    }
    /// Performs the same exact source, capacity, pool and existing-pin checks,
    /// retaining every account while leaving context selection to the caller's
    /// single finite union. The one-time context guard is unchanged.
    pub fn construct_unselected(
        self,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
        registered: impl IntoIterator<Item = impl Into<super::super::RegisteredWorkspaceStorageRow<K>>>,
        source: OriginalCompletedWorkspaceSource,
        carrier: &mut super::super::RetainedOriginalStorageSources,
    ) -> Result<CompletedWorkspaceStorage<K>, WorkingMemoryError>
    where
        K: super::super::HostSlotStorageKey,
    {
        self.construct_inner(pool, context, registered, source, carrier, false)
    }
    fn construct_inner(
        self,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
        registered: impl IntoIterator<Item = impl Into<super::super::RegisteredWorkspaceStorageRow<K>>>,
        source: OriginalCompletedWorkspaceSource,
        carrier: &mut super::super::RetainedOriginalStorageSources,
        select: bool,
    ) -> Result<CompletedWorkspaceStorage<K>, WorkingMemoryError>
    where
        K: super::super::HostSlotStorageKey,
    {
        if !source.custody.matches_pool(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let registered = self.layout.construct_with_completed_roots(
            pool,
            context,
            registered,
            &source.roots,
            select,
        )?;
        // This adds H to the existing-only pin; it supplies no source bytes.
        let registered = registered.with_retained_original_sources(carrier)?;
        Ok(CompletedWorkspaceStorage { registered, source })
    }
}

/// A sealed shared copy with completed original sources and optional existing
/// registry roots. The source account remains through the normal copy scope.
#[derive(Debug)]
pub struct OriginalCompletedWorkspaceCopy<K: Ord + Send + 'static> {
    plan: WorkspaceIsolatedCopyPlan,
    source: OriginalCompletedWorkspaceSource,
    registered: Option<RegisteredWorkspaceStorage<K>>,
}
impl<K: Clone + Ord + Send + Sync + 'static> OriginalCompletedWorkspaceCopy<K> {
    /// Consumes the complete mixed source binding, preserving its exact identity.
    pub fn bind(
        plan: WorkspaceIsolatedCopyPlan,
        storage: CompletedWorkspaceStorage<K>,
    ) -> Result<Self, WorkspaceCopyAdmissionError> {
        if !plan
            .source_storage()
            .same_identity(storage.borrowed_storage())
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        plan.incremental_requirements()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(Self {
            plan,
            source: storage.source,
            registered: Some(storage.registered),
        })
    }
    pub(super) fn only_completed(
        plan: WorkspaceIsolatedCopyPlan,
        source: OriginalCompletedWorkspaceSource,
    ) -> Result<Self, WorkspaceCopyAdmissionError> {
        if !plan.source_storage().same_identity(&source.roots) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        plan.incremental_requirements()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(Self {
            plan,
            source,
            registered: None,
        })
    }
    /// Fixed copy/source transports; row and shared-owner populations are priced
    /// by their actual source/storage layouts before construction.
    pub fn control_bytes() -> Result<usize, WorkingMemoryError> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, WorkspaceCopyAdmissionError>>(),
            size_of::<OriginalCompletedWorkspaceSource>(),
            size_of::<CompletedSourceCustody>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(&MemoryLedger, WorkspaceCopyLimits)>(),
            size_of::<Option<&super::super::WorkingMemoryStorage<K>>>(),
            size_of::<(&CompletedSourceCustody, &MemoryLedger)>(),
            size_of::<std::slice::Iter<'_, OriginalSpeculativeBudgetCustody>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
}
impl MemoryLedger {
    /// Same independently admitted copy account and capacity comparison as the
    /// registered/numerical routes; all source tickets validate under its lock.
    pub fn admit_completed_workspace_copy<K: Clone + Ord + Send + Sync + 'static>(
        &self,
        copy: OriginalCompletedWorkspaceCopy<K>,
        limits: WorkspaceCopyLimits,
    ) -> Result<AdmittedWorkspaceCopy, WorkspaceCopyAdmissionError> {
        if !copy.source.custody.matches_pool(self) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let accepted = super::prepare_copy_account(
            self,
            &self.construction_identity(),
            copy.plan
                .incremental_requirements()
                .ok_or(WorkingMemoryError::UnknownBound)?,
            0,
            &limits,
            0,
            super::super::funding::CopyHostHolds::None,
            |usage| {
                copy.source.custody.validate(self, usage)?;
                if let Some(source) = &copy.registered {
                    source.registration().validate_copy_source(self, usage)?;
                }
                Ok(())
            },
        )?;
        let execution =
            WorkspaceCopyAccountLayout::workspace()?.execution(copy.source.custody.host());
        let completed = RegisteredStoragePin::Completed(copy.source.custody.clone());
        let pin = match &copy.registered {
            Some(registered) => RegisteredStoragePin::pair(
                RegisteredStoragePin::new(registered.registration().clone()),
                completed,
            ),
            None => completed,
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
