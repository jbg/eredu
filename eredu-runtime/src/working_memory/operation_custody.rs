//! Account-only lifetime for already-priced shared native operation storage.
use super::{
    OriginalRealtimeBudgetCustody, OriginalSpeculativeBudgetCustody, OriginalSpeculativeRole,
    OriginalTextMetadataCustody, WorkingMemoryError,
};

/// Retains the actual accepted operation's metadata account. It carries no
/// equation plan, source payload, native object, mutable slot or allocation
/// permission. The consuming bank must separately authenticate its exact role
/// and consume its finite construction claim before allocating anything.
#[derive(Debug, Clone)]
pub struct OriginalOperationMetadataCustody(Custody);
#[derive(Debug, Clone)]
pub(in crate::working_memory) enum Custody {
    Text(OriginalTextMetadataCustody),
    Speculative(OriginalSpeculativeBudgetCustody),
    Realtime(OriginalRealtimeBudgetCustody),
    Numerical(super::OriginalNumericalBudgetCustody),
}
impl From<OriginalTextMetadataCustody> for OriginalOperationMetadataCustody {
    fn from(value: OriginalTextMetadataCustody) -> Self {
        Self(Custody::Text(value))
    }
}
impl From<OriginalSpeculativeBudgetCustody> for OriginalOperationMetadataCustody {
    fn from(value: OriginalSpeculativeBudgetCustody) -> Self {
        Self(Custody::Speculative(value))
    }
}
impl From<OriginalRealtimeBudgetCustody> for OriginalOperationMetadataCustody {
    fn from(value: OriginalRealtimeBudgetCustody) -> Self {
        Self(Custody::Realtime(value))
    }
}
impl From<super::OriginalNumericalBudgetCustody> for OriginalOperationMetadataCustody {
    fn from(value: super::OriginalNumericalBudgetCustody) -> Self {
        Self(Custody::Numerical(value))
    }
}
impl OriginalOperationMetadataCustody {
    pub(in crate::working_memory) fn same_host_account(
        &self,
        other: &super::OriginalHostMetadataCustody,
    ) -> bool {
        other.matches_operation(&self.0)
    }

    pub(in crate::working_memory) fn host_account_control_bytes() -> Option<usize> {
        super::OriginalHostMetadataCustody::operation_origin_control_bytes()?
            .checked_add(std::mem::size_of::<(
                &Self,
                &super::OriginalHostMetadataCustody,
            )>())?
            .checked_add(std::mem::size_of::<&Custody>())?
            .checked_add(std::mem::size_of::<bool>())
    }

    /// Exact retained accounting owner, not merely the shared pool.
    pub fn same_account(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Custody::Text(a), Custody::Text(b)) => a.same_account(b),
            (Custody::Speculative(a), Custody::Speculative(b)) => a.same_account(b),
            (Custody::Realtime(a), Custody::Realtime(b)) => a.same_account(b),
            (Custody::Numerical(a), Custody::Numerical(b)) => a.same_account(b),
            _ => false,
        }
    }
    /// Read-only check of the exact accepted role, never equal geometry or size.
    pub fn belongs_to_speculative_role(&self, role: &OriginalSpeculativeRole) -> bool {
        matches!(&self.0, Custody::Speculative(value) if value.belongs_to(role))
    }
    /// Reject a shared source initialized in another pool. This read-only
    /// comparison creates no source identity, residual credit, or bank claim.
    pub fn validate_initialization(
        &self,
        source: &super::SharedNativeInitializationCustody,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(value) => value.validate_initialization(source),
            Custody::Speculative(value) => source.validate_pool(value.pool()),
            Custody::Realtime(value) => source.validate_pool(value.pool()),
            Custody::Numerical(value) => source.validate_pool(value.pool()),
        }
    }
    /// Existing shared accounting domain only; no source or funding is created.
    pub fn matches_accounting_owner(&self, domain: &eredu_core::SharedStorageAccountingId) -> bool {
        match &self.0 {
            Custody::Text(value) => value.matches_accounting_owner(domain),
            Custody::Speculative(value) => value
                .pool()
                .shared_storage_accounting_id()
                .same_identity(domain),
            Custody::Realtime(value) => value
                .pool()
                .shared_storage_accounting_id()
                .same_identity(domain),
            Custody::Numerical(value) => value
                .pool()
                .shared_storage_accounting_id()
                .same_identity(domain),
        }
    }
    /// Read-only health and pool validation for an already retained origin.
    /// Closed healthy accounts remain valid while their actual source custody
    /// survives. This creates no storage, receipt, registration or permission.
    pub fn validate_retained_origin(
        &self,
        pool: &super::MemoryLedger,
    ) -> Result<(), WorkingMemoryError> {
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        match &self.0 {
            Custody::Text(value) => value.validate_retained_origin_locked(pool, &usage),
            Custody::Speculative(value) => value.validate_copy_source(pool, &usage),
            Custody::Realtime(value) => value.validate_copy_source(pool, &usage),
            Custody::Numerical(value) => value.validate_copy_source(pool, &usage),
        }
    }

    /// Concrete synchronous origin-validation frames, with no new owner.
    pub fn retained_origin_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(&Self, &super::MemoryLedger)>(),
            size_of::<std::sync::MutexGuard<'static, super::Usage>>(),
            size_of::<
                Result<
                    std::sync::MutexGuard<'static, super::Usage>,
                    std::sync::PoisonError<std::sync::MutexGuard<'static, super::Usage>>,
                >,
            >(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(
                &OriginalTextMetadataCustody,
                &super::MemoryLedger,
                &super::Usage,
            )>(),
            size_of::<(
                &OriginalSpeculativeBudgetCustody,
                &super::MemoryLedger,
                &super::Usage,
            )>(),
            size_of::<(
                &OriginalRealtimeBudgetCustody,
                &super::MemoryLedger,
                &super::Usage,
            )>(),
            size_of::<(
                &super::OriginalNumericalBudgetCustody,
                &super::MemoryLedger,
                &super::Usage,
            )>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Validate the already-attached metadata against this retained account.
    pub fn validate_metadata(
        &self,
        value: &crate::SharedHostMetadata,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(custody) => custody.validate_metadata(value),
            Custody::Speculative(custody) => {
                value.validate_original_attachment(custody.pool().shared_storage_accounting_id())
            }
            Custody::Realtime(custody) => {
                value.validate_original_attachment(custody.pool().shared_storage_accounting_id())
            }
            Custody::Numerical(custody) => {
                value.validate_original_attachment(custody.pool().shared_storage_accounting_id())
            }
        }
    }
    /// Read-only validation of an existing fixed metadata attachment.
    pub fn validate_slot_metadata(
        &self,
        value: &crate::HostSlotMetadata,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(custody) => custody.validate_slot_metadata(value),
            Custody::Speculative(custody) => {
                value.validate_original_attachment(custody.pool().shared_storage_accounting_id())
            }
            Custody::Realtime(custody) => {
                value.validate_original_attachment(custody.pool().shared_storage_accounting_id())
            }
            Custody::Numerical(custody) => {
                value.validate_original_attachment(custody.pool().shared_storage_accounting_id())
            }
        }
    }
    /// Validate actual retained source inventory without granting residual credit.
    pub fn validate_source_inventory(
        &self,
        identity: &eredu_checkpoint::store::SourceStorageIdentity,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(custody) => custody.validate_source_inventory(identity, bytes),
            Custody::Speculative(custody) => custody
                .pool()
                .validate_original_source_inventory(identity, bytes),
            Custody::Realtime(custody) => custody
                .pool()
                .validate_original_source_inventory(identity, bytes),
            Custody::Numerical(custody) => custody
                .pool()
                .validate_original_source_inventory(identity, bytes),
        }
    }
}
