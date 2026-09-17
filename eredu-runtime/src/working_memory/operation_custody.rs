//! Account-only lifetime for already-priced shared native operation storage.
use super::{
    OriginalRealtimeBudgetCustody, OriginalSpeculativeBudgetCustody, OriginalSpeculativeRole, OriginalTextMetadataCustody,
    WorkingMemoryError,
};

/// Retains the actual accepted operation's metadata account. It carries no
/// equation plan, source payload, native object, mutable slot or allocation
/// permission. The consuming bank must separately authenticate its exact role
/// and consume its finite construction claim before allocating anything.
#[derive(Debug, Clone)]
pub struct OriginalOperationMetadataCustody(Custody);
#[derive(Debug, Clone)]
enum Custody {
    Text(OriginalTextMetadataCustody),
    Speculative(OriginalSpeculativeBudgetCustody),
    Realtime(OriginalRealtimeBudgetCustody),
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
    fn from(value:OriginalRealtimeBudgetCustody)->Self {Self(Custody::Realtime(value))}
}
impl OriginalOperationMetadataCustody {
    /// Exact retained accounting owner, not merely the shared pool.
    pub fn same_account(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Custody::Text(a), Custody::Text(b)) => a.same_account(b),
            (Custody::Speculative(a), Custody::Speculative(b)) => a.same_account(b),
            (Custody::Realtime(a), Custody::Realtime(b)) => a.same_account(b),
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
        &self, source: &super::SharedNativeInitializationCustody,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(value) => value.validate_initialization(source),
            Custody::Speculative(value) => source.validate_pool(value.pool()),
            Custody::Realtime(value) => source.validate_pool(value.pool()),
        }
    }
    /// Existing shared accounting domain only; no source or funding is created.
    pub fn matches_domain(&self, domain: &eredu_core::SharedStorageDomain) -> bool {
        match &self.0 {
            Custody::Text(value) => value.matches_domain(domain),
            Custody::Speculative(value) => value.pool().shared_storage_domain().same_identity(domain),
            Custody::Realtime(value) => value.pool().shared_storage_domain().same_identity(domain),
        }
    }
    /// Validate the already-attached metadata against this retained account.
    pub fn validate_metadata(
        &self,
        value: &crate::SharedHostMetadata,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(custody) => custody.validate_metadata(value),
            Custody::Speculative(custody) => value.validate_original_attachment(custody.pool().shared_storage_domain()),
            Custody::Realtime(custody) => value.validate_original_attachment(custody.pool().shared_storage_domain()),
        }
    }
    /// Read-only validation of an existing fixed metadata attachment.
    pub fn validate_slot_metadata(
        &self,
        value: &crate::HostSlotMetadata,
    ) -> Result<(), WorkingMemoryError> {
        match &self.0 {
            Custody::Text(custody) => custody.validate_slot_metadata(value),
            Custody::Speculative(custody) => value.validate_original_attachment(custody.pool().shared_storage_domain()),
            Custody::Realtime(custody) => value.validate_original_attachment(custody.pool().shared_storage_domain()),
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
            Custody::Realtime(custody) => custody.pool().validate_original_source_inventory(identity,bytes),
        }
    }
}
