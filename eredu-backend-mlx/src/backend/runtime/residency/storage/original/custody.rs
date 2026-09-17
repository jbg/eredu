//! Distinct destination custody for one shared finite inventory constructor.
use super::*;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::working_memory::WorkingMemoryPool;

pub(in crate::backend::runtime::residency::storage) enum Custody {
    Text(OriginalTextMetadataCustody),
    SnapshotPublication {
        pool: WorkingMemoryPool,
        _host: HostPreparationAuthority,
    },
    Snapshot {
        // Source validation borrows this exact pool; this is not a funding scope.
        pool: WorkingMemoryPool,
        // Last: rows, clone shells and the source pool retire before host custody.
        _host: HostPreparationAuthority,
    },
}
impl Custody {
    pub(in crate::backend::runtime::residency::storage) fn validate_metadata(
        &self,
        metadata: &eredu_runtime::SharedHostMetadata,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Text(custody) => custody.validate_metadata(metadata),
            Self::SnapshotPublication { .. } => Err(WorkingMemoryError::IdentityMismatch),
            Self::Snapshot { pool, .. } => {
                metadata.validate_original_attachment(pool.shared_storage_domain())
            }
        }
    }
    pub(in crate::backend::runtime::residency::storage) fn validate_slot_metadata(
        &self,
        metadata: &eredu_runtime::HostSlotMetadata,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Text(custody) => custody.validate_slot_metadata(metadata),
            Self::SnapshotPublication { pool, .. } => {
                metadata.prepare_copy_attachment(pool.shared_storage_domain())
            }
            Self::Snapshot { pool, .. } => {
                metadata.validate_original_attachment(pool.shared_storage_domain())
            }
        }
    }
    pub(in crate::backend::runtime::residency::storage) fn validate_source_inventory(
        &self,
        key: &StorageIdentity,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Text(custody) => custody.validate_retained_source_inventory(key, bytes),
            Self::SnapshotPublication { .. } => Err(WorkingMemoryError::IdentityMismatch),
            Self::Snapshot { pool, .. } => pool.validate_retained_source_inventory(key, bytes),
        }
    }
    pub(in crate::backend::runtime::residency::storage) fn snapshot_publication(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::SnapshotPublication { pool: expected, .. } if expected.same_domain(pool) => {
                Ok(())
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(in crate::backend::runtime::residency::storage) fn publication(
        &self,
    ) -> Result<OriginalTextMetadataCustody, WorkingMemoryError> {
        match self {
            Self::Text(custody) => Ok(custody.clone()),
            // A snapshot's accepted host constructor has no native publication role.
            Self::Snapshot { .. } | Self::SnapshotPublication { .. } => {
                Err(WorkingMemoryError::IdentityMismatch)
            }
        }
    }
}
