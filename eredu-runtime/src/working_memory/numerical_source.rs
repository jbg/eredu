//! Standalone numerical work uses the existing account and source-bank lifecycle.
use super::{
    HostSourceConstructionFacts, InferenceExecutionIdentity, MemoryLedger, OriginalHostSourceBank,
    PreparedAccountCommit, Usage, WorkingMemoryError,
    funding::{AccountNode, AccountTicket, PendingAccount, PendingOriginal},
    qualified_storage,
    transaction_buffers::RequirementProjection,
};
use eredu_core::{
    DomainMemoryCharge, DomainMemoryRequirements, HostMetadataAccount, HostMetadataFunding,
    HostMetadataFundingError, MemoryLimits, MemoryPlacement, MemoryPlacementKind, MemoryTopology,
};
use std::{
    mem::{size_of, size_of_val},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Complete physical and Host source requirements for one numerical operation.
/// Native mechanism facts and completion evidence remain backend-owned.
#[derive(Debug)]
pub struct NumericalSourceRequirements {
    native: DomainMemoryRequirements,
    native_metadata: u64,
    controls: u64,
    sources: HostSourceConstructionFacts,
}
impl NumericalSourceRequirements {
    /// Native backing, native wrapper metadata, independent source controls and
    /// the actual finite source population have disjoint allowances.
    pub fn new(
        native: DomainMemoryRequirements,
        native_metadata: Option<u64>,
        controls: Option<u64>,
        sources: HostSourceConstructionFacts,
    ) -> Result<Self, WorkingMemoryError> {
        let value = Self {
            native,
            native_metadata: native_metadata.ok_or(WorkingMemoryError::UnknownBound)?,
            controls: controls.ok_or(WorkingMemoryError::UnknownBound)?,
            sources,
        };
        value.host_bytes()?;
        Ok(value)
    }
    fn host_bytes(&self) -> Result<u64, WorkingMemoryError> {
        [
            self.native_metadata,
            self.controls,
            self.sources.protected_bytes(),
            account_controls()?,
            funding_controls()?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }
    /// The one atomic reservation includes staging/source construction and all
    /// candidate native domains before either kind of payload is allocated.
    pub fn requirements(
        &self,
        topology: &MemoryTopology,
    ) -> Result<DomainMemoryRequirements, WorkingMemoryError> {
        self.native.validate(topology)?;
        let mut domains = self.native.clone();
        let host = self
            .host_bytes()?
            .checked_add(domain_controls(topology, &self.native)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        domains.add_allocation(
            host,
            &MemoryPlacement::fixed(topology, topology.host_domain())?,
        )?;
        Ok(domains)
    }
}

#[derive(Debug)]
struct Charge {
    native: DomainMemoryRequirements,
    native_metadata: u64,
    controls: u64,
    spent: AtomicU64,
    native_released: AtomicU64,
    execution: InferenceExecutionIdentity,
    ticket: AccountTicket,
}
#[derive(Debug)]
struct Account(Option<Arc<Charge>>);
impl Account {
    fn value(&self) -> &Charge {
        self.0.as_deref().expect("live numerical source account")
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live account"),
            other.0.as_ref().expect("live account"),
        )
    }
}
impl Clone for Account {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live numerical source account"),
        )))
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            if std::thread::panicking() {
                value.ticket.quarantine();
            }
            drop(Arc::into_inner(value));
        }
    }
}

/// A failed accepted constructor retains its actual account until retirement.
#[derive(Debug)]
pub struct NumericalSourceAdmissionError {
    cause: WorkingMemoryError,
    _ticket: Option<AccountTicket>,
}
impl From<WorkingMemoryError> for NumericalSourceAdmissionError {
    fn from(cause: WorkingMemoryError) -> Self {
        Self {
            cause,
            _ticket: None,
        }
    }
}
impl std::fmt::Display for NumericalSourceAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for NumericalSourceAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl NumericalSourceAdmissionError {
    /// Original typed failure, without detaching retained accounting custody.
    pub fn cause(&self) -> &WorkingMemoryError {
        &self.cause
    }
}

/// One issuer for an accepted numerical operation. Source construction, native
/// work and metadata each consume their own once-only claim.
#[derive(Debug)]
pub struct OriginalNumericalSource {
    account: Account,
    sources: Option<HostSourceConstructionFacts>,
    native_issued: bool,
    metadata_issued: bool,
}
/// Move-only native allowance. It authenticates physical allocation capacity;
/// the backend must independently establish mechanism and completion evidence.
#[derive(Debug)]
pub struct OriginalNumericalNative {
    account: Account,
}
/// Accounting-only custody, with no source-bank or execution authority.
#[derive(Debug, Clone)]
pub struct OriginalNumericalBudgetCustody {
    account: Account,
}

impl MemoryLedger {
    /// Accepts all physical domains through one coordinator transaction.
    pub fn reserve_numerical_source(
        &self,
        execution: &InferenceExecutionIdentity,
        requirements: NumericalSourceRequirements,
        limits: MemoryLimits,
    ) -> Result<OriginalNumericalSource, NumericalSourceAdmissionError> {
        let floor = requirements
            .host_bytes()?
            .checked_add(domain_controls(self.topology(), &requirements.native)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = requirements
            .controls
            .checked_add(funding_controls()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        // Borrow the caller's descriptor into the coordinator's funded scratch
        // vectors. Account storage is constructed only after atomic admission.
        let pending = self.accept_projected(
            execution,
            RequirementProjection {
                parts: &[&requirements.native],
                headroom: &eredu_core::MemoryHeadroomDeclarations::none(),
                host_bytes: floor,
            },
            &eredu_core::MemoryLimitDeclarations::default(),
            Some(&limits),
            &[],
            floor,
            |_| Ok(()),
        )?;
        let ticket = pending.publish();
        if let Err(cause) = ticket.status() {
            return Err(NumericalSourceAdmissionError {
                cause,
                _ticket: Some(ticket),
            });
        }
        let NumericalSourceRequirements {
            native,
            native_metadata,
            sources,
            ..
        } = requirements;
        Ok(OriginalNumericalSource {
            account: Account(Some(Arc::new(Charge {
                native,
                native_metadata,
                controls,
                spent: AtomicU64::new(0),
                native_released: AtomicU64::new(0),
                execution: execution.clone(),
                ticket,
            }))),
            sources: Some(sources),
            native_issued: false,
            metadata_issued: false,
        })
    }
}
impl OriginalNumericalSource {
    /// Reject a different executable identity before any construction claim.
    pub fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.account.value().ticket.status()?;
        if Arc::ptr_eq(&self.account.value().execution.0, &execution.0) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Extract the exact finite source population once, using the shared bank.
    pub fn take_source_constructions(
        &mut self,
    ) -> Result<OriginalHostSourceBank, WorkingMemoryError> {
        self.account.value().ticket.status()?;
        let facts = self
            .sources
            .take()
            .ok_or(WorkingMemoryError::AlreadyStarted)?;
        Ok(OriginalHostSourceBank::new_with_custody(
            facts,
            None,
            self.budget_custody().into(),
        ))
    }
    /// Consume the separately admitted physical/native-metadata allowance once.
    pub fn claim_native(&mut self) -> Result<OriginalNumericalNative, WorkingMemoryError> {
        self.account.value().ticket.status()?;
        if self.native_issued {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.native_issued = true;
        Ok(OriginalNumericalNative {
            account: self.account.clone(),
        })
    }
    /// One cumulative Host counter over the independently quoted source controls.
    pub fn metadata_funding(&mut self) -> Result<HostMetadataFunding, HostMetadataFundingError> {
        if self.metadata_issued {
            return Err(HostMetadataFundingError::Unavailable);
        }
        self.metadata_issued = true;
        HostMetadataFunding::new(Metadata(self.budget_custody()))
    }
    /// Retention alone cannot extract another source or native claim.
    pub fn budget_custody(&self) -> OriginalNumericalBudgetCustody {
        OriginalNumericalBudgetCustody {
            account: self.account.clone(),
        }
    }
}
/// Move-only lifetime of one validated native allocation mechanism. Its backend
/// reports closed occupancy only after every producer and submission is safe;
/// remaining backings can then retire independently of inspection aliases.
#[derive(Debug)]
pub struct OriginalNumericalLifetime {
    claim: OriginalNumericalNative,
    capacity: u64,
    placement: Arc<MemoryPlacement>,
}
impl OriginalNumericalLifetime {
    /// Original account-only custody, with no new allocation authority.
    pub fn budget_custody(&self) -> OriginalNumericalBudgetCustody {
        self.claim.budget_custody()
    }
    /// Original separately admitted wrapper/control capacity.
    pub fn metadata_bytes(&self) -> u64 {
        self.claim.metadata_bytes()
    }
    /// Backend-certified occupied backing after producer closure. Reports may
    /// arrive out of order; only decreases retire charges. Invalid state or
    /// unwinding quarantines the account and preserves its remaining charges.
    pub fn retire_completed_occupancy(&self, bytes: u64) {
        let account = self.claim.account.value();
        account.ticket.retire_completed_native_occupancy(
            self.capacity,
            bytes,
            &self.placement,
            &account.native_released,
        );
    }
}
impl OriginalNumericalNative {
    /// Consumes this sole native claim after authenticating its full uniform
    /// placement vector. The supplied placement owner already belongs to the
    /// backend's paid mechanism; this binding allocates no descriptor storage.
    pub fn bind_lifetime(
        self,
        capacity: u64,
        placement: Arc<MemoryPlacement>,
    ) -> Result<OriginalNumericalLifetime, WorkingMemoryError> {
        self.validate_allocation(capacity, &placement)?;
        Ok(OriginalNumericalLifetime {
            claim: self,
            capacity,
            placement,
        })
    }
    /// Complete accepted native backing vector, including placement allowances.
    pub fn requirements(&self) -> &DomainMemoryRequirements {
        &self.account.value().native
    }
    /// Actual independently quoted native wrapper/control capacity.
    pub fn metadata_bytes(&self) -> u64 {
        self.account.value().native_metadata
    }
    /// Validate the real allocator's placement and full assigned capacity. A
    /// device cannot substitute spare Host capacity on separate-memory hardware.
    pub fn validate_allocation(
        &self,
        bytes: u64,
        placement: &MemoryPlacement,
    ) -> Result<(), WorkingMemoryError> {
        self.account.value().ticket.status()?;
        let pool = self.account.value().ticket.pool();
        placement.validate(pool.topology())?;
        let native = &self.account.value().native;
        if !native.overhead_estimates().is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        match placement.kind() {
            MemoryPlacementKind::Fixed(_) if !native.placement_allowances().is_empty() => {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            MemoryPlacementKind::Possible { .. } => {
                let [allowance] = native.placement_allowances() else {
                    return Err(WorkingMemoryError::IdentityMismatch);
                };
                if allowance.backing_bytes != bytes || allowance.placement != *placement {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
            _ => (),
        }
        for (domain, charge) in native.iter() {
            let selected = placement.domains().contains(&domain);
            let expected = match placement.kind() {
                MemoryPlacementKind::Fixed(_) => DomainMemoryCharge {
                    accounted_bytes: if selected { bytes } else { 0 },
                    ..Default::default()
                },
                MemoryPlacementKind::Possible { .. } => DomainMemoryCharge {
                    placement_allowance_bytes: if selected { bytes } else { 0 },
                    ..Default::default()
                },
            };
            if charge != expected {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        Ok(())
    }
    /// Retain the same accepted account through native buffers and completion.
    pub fn budget_custody(&self) -> OriginalNumericalBudgetCustody {
        OriginalNumericalBudgetCustody {
            account: self.account.clone(),
        }
    }
}
impl OriginalNumericalBudgetCustody {
    /// Exact accepted owner identity, independent of equal capacity or geometry.
    pub fn same_account(&self, other: &Self) -> bool {
        self.account.same(&other.account)
    }
    pub(in crate::working_memory) fn validate_completed_roots<'a>(
        &self,
        roots: impl Iterator<Item = &'a eredu_nn::workspace::WorkspaceExistingStorage> + Clone,
    ) -> Result<(), WorkingMemoryError> {
        self.account.value().ticket.status()?;
        let mut controls = 0u64;
        for root in roots.clone() {
            root.placement().ok_or(WorkingMemoryError::UnknownBound)?;
            root.capacity_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?;
            controls = controls
                .checked_add(
                    root.host_control_bytes()
                        .ok_or(WorkingMemoryError::UnknownBound)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        if controls > self.account.value().native_metadata {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_completed_allocations(roots.map(|root| {
            (
                root.capacity_bytes().expect("validated capacity"),
                root.placement().expect("validated placement"),
            )
        }))
    }
    pub(in crate::working_memory) fn validate_completed_metadata(
        &self,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        if bytes > self.account.value().native_metadata {
            Err(WorkingMemoryError::IdentityMismatch)
        } else {
            Ok(())
        }
    }

    // Pure backing comparison; callers validate health under their existing
    // coordinator lock. It neither locks recursively nor issues allocation work.
    pub(in crate::working_memory) fn validate_completed_allocations<'a>(
        &self,
        allocations: impl Iterator<Item = (u64, &'a MemoryPlacement)> + Clone,
    ) -> Result<(), WorkingMemoryError> {
        let native = &self.account.value().native;
        let released = self.account.value().native_released.load(Ordering::Acquire);
        for (_, placement) in allocations.clone() {
            placement.validate(self.pool().topology())?;
            if matches!(placement.kind(), MemoryPlacementKind::Possible { .. }) {
                let available = native
                    .placement_allowances()
                    .iter()
                    .filter(|allowance| allowance.placement == *placement)
                    .try_fold(0u64, |sum, allowance| {
                        sum.checked_add(allowance.backing_bytes)
                            .ok_or(WorkingMemoryError::Overflow)
                    })?
                    .checked_sub(released)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                let required = allocations
                    .clone()
                    .filter(|(_, other)| *other == placement)
                    .try_fold(0u64, |sum, (bytes, _)| {
                        sum.checked_add(bytes).ok_or(WorkingMemoryError::Overflow)
                    })?;
                if required > available {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
        }
        for (domain, available) in native.iter() {
            let mut accounted = 0u64;
            let mut allowance = 0u64;
            for (capacity, placement) in allocations.clone() {
                if !placement.domains().contains(&domain) {
                    continue;
                }
                let sum = match placement.kind() {
                    MemoryPlacementKind::Fixed(_) => &mut accounted,
                    MemoryPlacementKind::Possible { .. } => &mut allowance,
                };
                *sum = sum
                    .checked_add(capacity)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
            let remaining = |amount: u64| {
                if amount == 0 {
                    Ok(0)
                } else {
                    amount
                        .checked_sub(released)
                        .ok_or(WorkingMemoryError::IdentityMismatch)
                }
            };
            if accounted > remaining(available.accounted_bytes)?
                || allowance > remaining(available.placement_allowance_bytes)?
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        Ok(())
    }
    pub(in crate::working_memory) fn execution(&self) -> &InferenceExecutionIdentity {
        &self.account.value().execution
    }
    pub(in crate::working_memory) fn account_id(&self) -> u64 {
        self.account.value().ticket.id()
    }
    pub(in crate::working_memory) fn pool(&self) -> &MemoryLedger {
        self.account.value().ticket.pool()
    }
    pub(in crate::working_memory) fn quarantine(&self) {
        self.account.value().ticket.quarantine();
    }
    pub(in crate::working_memory) fn validate_copy_source(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.pool().same_ledger(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.account.value().ticket.validate_in(usage)
    }
}
#[derive(Debug)]
struct Metadata(OriginalNumericalBudgetCustody);
impl HostMetadataAccount for Metadata {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let value = self.0.account.value();
        value
            .ticket
            .status()
            .map_err(|_| HostMetadataFundingError::Unavailable)?;
        let bytes = u64::try_from(bytes).map_err(|_| HostMetadataFundingError::Overflow)?;
        value
            .spent
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |spent| {
                spent
                    .checked_add(bytes)
                    .filter(|next| *next <= value.controls)
            })
            .map(|_| ())
            .map_err(|spent| {
                if spent.checked_add(bytes).is_none() {
                    HostMetadataFundingError::Overflow
                } else {
                    match value.controls.checked_sub(spent) {
                        Some(available) => HostMetadataFundingError::Capacity {
                            required: bytes,
                            available,
                        },
                        None => HostMetadataFundingError::Unavailable,
                    }
                }
            })
    }
}
fn funding_controls() -> Result<u64, WorkingMemoryError> {
    HostMetadataFunding::constructor_bytes::<Metadata>()
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
fn account_controls() -> Result<u64, WorkingMemoryError> {
    let frames = [
        size_of::<AccountNode>(),
        size_of::<AccountTicket>(),
        size_of::<PendingAccount>(),
        size_of::<PendingOriginal>(),
        size_of::<PreparedAccountCommit<'_>>(),
        size_of::<RequirementProjection<'_>>(),
        size_of::<eredu_core::MemoryLimitDeclarations>(),
        size_of::<eredu_core::MemoryHeadroomDeclarations>(),
        size_of::<[&DomainMemoryRequirements; 1]>(),
        size_of::<Account>(),
        size_of::<Charge>(),
        size_of::<NumericalSourceRequirements>(),
        size_of::<OriginalNumericalSource>(),
        size_of::<OriginalNumericalNative>(),
        size_of::<OriginalNumericalLifetime>(),
        size_of::<OriginalNumericalBudgetCustody>(),
        size_of::<NumericalSourceAdmissionError>(),
        size_of::<Result<OriginalNumericalSource, NumericalSourceAdmissionError>>(),
        size_of::<Result<OriginalNumericalNative, WorkingMemoryError>>(),
        size_of::<Result<OriginalNumericalLifetime, WorkingMemoryError>>(),
        size_of::<Result<OriginalHostSourceBank, WorkingMemoryError>>(),
        size_of::<(
            &MemoryLedger,
            &InferenceExecutionIdentity,
            NumericalSourceRequirements,
            MemoryLimits,
        )>(),
        size_of::<(&OriginalNumericalSource, &InferenceExecutionIdentity)>(),
        size_of::<(&OriginalNumericalNative, u64, &MemoryPlacement)>(),
        size_of::<(OriginalNumericalNative, u64, Arc<MemoryPlacement>)>(),
        size_of::<(&OriginalNumericalLifetime, u64)>(),
        size_of::<(&AccountTicket, u64, u64, &MemoryPlacement, &AtomicU64)>(),
        size_of::<Result<Option<(u64, u64)>, WorkingMemoryError>>(),
        size_of::<(&Metadata, usize)>(),
        size_of::<Result<u64, u64>>(),
        size_of::<[u64; 3]>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<WorkingMemoryError>(),
    ];
    let bytes = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    bytes
        .checked_add(qualified_storage::shared_bytes::<Charge>()?)
        .ok_or(WorkingMemoryError::Overflow)
}
fn domain_controls(
    topology: &MemoryTopology,
    native: &DomainMemoryRequirements,
) -> Result<u64, WorkingMemoryError> {
    let limits = topology
        .len()
        .checked_mul(size_of::<eredu_core::MemoryLimit>())
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    native
        .backing_bytes()?
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(limits))
        .and_then(|bytes| bytes.checked_add(super::funding::domain_balance_bytes(topology).ok()?))
        .ok_or(WorkingMemoryError::Overflow)
}

#[cfg(test)]
mod tests;
