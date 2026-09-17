//! Actual retained-row pin construction, separate from native copy admission.
use super::*;
use eredu_runtime::working_memory::{ExistingStoragePinLayout, OriginalStorageSourcesLayout};
use std::{alloc::Layout, mem::size_of};

pub(crate) struct SourcePinPlan {
    maximum: usize,
    sources: OriginalStorageSourcesLayout,
    pin: ExistingStoragePinLayout<StorageIdentity>,
    bytes: usize,
}
impl SourcePinPlan {
    /// Count comes from the exact retained census. Every row may be a source;
    /// final deduplication can reduce storage but supplies no extra slot credit.
    pub(crate) fn new(rows: usize) -> Result<Self, WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        let sources = OriginalStorageSourcesLayout::new(rows).ok_or_else(overflow)?;
        let pin = ExistingStoragePinLayout::new(rows)?;
        let keys = usize::try_from(
            eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes().ok_or_else(overflow)?,
        )
        .map_err(|_| overflow())?
        .checked_mul(rows)
        .ok_or_else(overflow)?;
        let bytes =
            [
                sources.requested_bytes(),
                pin.requested_bytes(),
                keys,
                Layout::array::<(StorageIdentity, u64)>(rows)
                    .map_err(|_| overflow())?
                    .size(),
                size_of::<Self>(),
                size_of::<Vec<(StorageIdentity, u64)>>(),
                size_of::<StorageIdentity>(),
                size_of::<Option<u64>>(),
                size_of::<Result<Vec<(StorageIdentity, u64)>, Error>>(),
                size_of::<Result<WorkingMemoryStorage<StorageIdentity>, Error>>(),
                size_of::<(
                    &RetainedStorage,
                    &WorkingMemoryPool,
                    &eredu_core::HostPreparationAuthority,
                )>(),
                eredu_core::BackendFailure::source_retention_peak_bytes::<
                    PreparedOriginalSourceFailure,
                >()
                .ok_or_else(overflow)?,
                eredu_core::BackendFailure::source_retention_peak_bytes::<
                    eredu_runtime::working_memory::OriginalStorageSourcesError,
                >()
                .ok_or_else(overflow)?,
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(overflow)?;
        Ok(Self {
            maximum: rows,
            sources,
            pin,
            bytes,
        })
    }
    pub(crate) fn requested_bytes(&self) -> usize {
        self.bytes
    }
    pub(super) fn pin(
        self,
        inventory: &RetainedStorage,
        pool: &WorkingMemoryPool,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, Error> {
        let mut original = self
            .sources
            .construct(pool, host)
            .map_err(|cause| Error::StorageSource(eredu_core::BackendFailure::from_error(cause)))?;
        let result = (|| {
            for (_, (_, token)) in inventory.slot_entries() {
                if pool
                    .classify_host_slot_source(token)
                    .map_err(Error::PrefillControl)?
                    .registered()
                    .is_none()
                {
                    original
                        .retain_table(token)
                        .map_err(Error::PrefillControl)?;
                }
            }
            for (_, (_, metadata)) in inventory.metadata_entries() {
                if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
                    if input.original_residence(pool).is_some() {
                        original
                            .retain_input(input)
                            .map_err(Error::PrefillControl)?;
                    }
                }
            }
            let mut entries = inventory.storage_entries_with_capacity(Some(self.maximum))?;
            entries.retain(|(key,_)|!matches!(key,StorageIdentity::HostMetadata(key) if original.contains(key)));
            self.pin
                .construct(pool, entries)
                .and_then(|storage| storage.with_retained_original_sources(&mut original))
                .map_err(Error::PrefillControl)
        })();
        result.map_err(|cause| {
            Error::StorageSource(eredu_core::BackendFailure::from_error(
                PreparedOriginalSourceFailure {
                    cause,
                    _sources: original,
                },
            ))
        })
    }
}
impl RetainedStorage {
    pub(crate) fn snapshot_census_pin_plan(&self) -> Result<SourcePinPlan, WorkingMemoryError> {
        SourcePinPlan::new(
            self.snapshot_census_rows()
                .ok_or(WorkingMemoryError::UnknownBound)?,
        )
    }

    /// Owning source pin/carrier contribution for the actual count-only census.
    /// The caller separately composes decoder-specific source wrapper topology.
    pub(crate) fn source_pin_plan(&self, pool: &WorkingMemoryPool) -> Result<SourcePinPlan, Error> {
        let rows = match self.original_maximum_rows() {
            Some(rows) => rows,
            None => self
                .original_publication_rows(pool)?
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?,
        };
        SourcePinPlan::new(rows).map_err(Error::PrefillControl)
    }
}
