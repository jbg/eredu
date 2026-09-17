//! Count and fill the same retained decoder inventory under distinct host custody.
use super::*;
use crate::backend::runtime::cache::state::{
    SnapshotArraySources, SnapshotOperand, SnapshotProjectionCause,
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::{mem::size_of, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub(super) enum StoragePreparationCause {
    #[error(transparent)]
    Source(#[from] crate::backend::runtime::cache::state::SnapshotProjectionCause),
    #[error(transparent)]
    Native(#[from] crate::backend::runtime::residency::manager::ResidencyError),
    #[error(transparent)]
    Inspection(#[from] crate::backend::runtime::residency::storage::SnapshotStorageInspectionError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
}

/// Borrowing the actual owner prevents a plan from being applied to another
/// decoder. The enclosing idle submission lease keeps its storage unchanged.
/// This describes only collector construction, not private trace/native copy fit.
pub(super) struct DecoderStoragePlan<'a> {
    source: &'a DecoderCopyOwner,
    pool: &'a WorkingMemoryPool,
    rows: usize,
    key: Option<&'a Array>,
    pending: Option<&'a Array>,
    known_bytes: usize,
}
impl<'a> DecoderStoragePlan<'a> {
    fn new(
        source: &'a DecoderCopyOwner,
        prepared: &PreparedResidentDecoderCopy<'_>,
        pool: &'a WorkingMemoryPool,
        key: Option<&'a Array>,
        pending: Option<&'a Array>,
    ) -> Result<Self, StoragePreparationCause> {
        let mut census = RetainedStorage::original_census(pool);
        source.collect_storage_cold(prepared, &mut census)?;
        for array in key.into_iter().chain(pending) {
            census.include_array(array).map_err(
                crate::backend::runtime::residency::storage::SnapshotStorageInspectionError::from,
            )?;
        }
        let rows = census
            .snapshot_census_rows()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let collector = RetainedStorage::original_collector_control_bytes(rows)
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let source_pin = census.snapshot_census_pin_plan()?.requested_bytes();
        let wrappers = prepared.prepared_source_pin_control_bytes()?;
        let known_bytes =
            [
                collector,
                source_pin,
                wrappers,
                size_of::<Self>(),
                size_of::<Result<Self, StoragePreparationCause>>(),
                size_of::<StoragePreparationCause>(),
                size_of::<
                    crate::backend::runtime::residency::storage::SnapshotStorageInspectionError,
                >(),
                size_of::<Option<crate::backend::runtime::residency::manager::ResidencyError>>(),
                BackendFailure::source_retention_peak_bytes::<StoragePreparationFailure>()
                    .ok_or(WorkingMemoryError::Overflow)?,
                size_of::<RetainedStorage>(),
                size_of::<RetainedStorage>(),
                size_of::<Result<RetainedStorage, Error>>(),
                size_of::<Option<usize>>(),
                size_of::<Option<Error>>(),
                size_of::<Option<crate::backend::runtime::residency::manager::ResidencyError>>(),
                size_of::<PreparedResidentDecoderCopy<'static>>(),
                size_of::<SnapshotOperand<'_>>(),
                size_of::<&mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>>(),
                size_of::<Result<(), SnapshotProjectionCause>>(),
                size_of::<Arc<safemlx::ImmutableHostTransferBuffer>>(),
                size_of::<(&DecoderCopyOwner, &WorkingMemoryPool, &mut RetainedStorage)>(),
                size_of::<HostPreparationAuthority>(),
                size_of::<
                    std::iter::Chain<std::option::IntoIter<&Array>, std::option::IntoIter<&Array>>,
                >(),
                size_of::<Result<(), crate::backend::runtime::residency::manager::ResidencyError>>(
                ),
                BackendFailure::source_retention_peak_bytes::<InventoryFailure>()
                    .ok_or(WorkingMemoryError::Overflow)?,
                size_of::<BackendFailure>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            source,
            pool,
            rows,
            key,
            pending,
            known_bytes,
        })
    }

    /// Known collector, complete-source pin and actual source-wrapper contribution
    /// for aggregate pre-grant planning. Source visitors, registered table binding,
    /// private graphs and native destination-copy roles remain separate.
    pub(super) fn known_control_bytes(&self) -> usize {
        self.known_bytes
    }

    pub(super) fn construct(
        self,
        host: &HostPreparationAuthority,
    ) -> Result<RetainedStorage, Error> {
        // The original snapshot gateway must compose known_bytes before it
        // hands this worker its accepted host authority. The native hook stays
        // unavailable until that complete aggregate query exists.
        let result = (|| {
            let mut storage =
                RetainedStorage::prepare_snapshot_collector(self.rows, self.pool, host)?;
            self.source.collect_storage(&mut storage)?;
            for array in self.key.into_iter().chain(self.pending) {
                storage.include_array(array)?;
            }
            storage.byte_bound()?.ok_or_else(unknown)?;
            Ok(storage)
        })();
        result.map_err(|cause| retain_failure(cause, host))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct InventoryFailure {
    #[source]
    cause: Error,
    // Core's closed source Box retires before this payload and its final token.
    _host: HostPreparationAuthority,
}

fn retain_failure(cause: Error, host: &HostPreparationAuthority) -> Error {
    Error::StorageSource(BackendFailure::from_error(InventoryFailure {
        cause,
        _host: host.clone(),
    }))
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct StoragePreparationFailure {
    #[source]
    cause: StoragePreparationCause,
    _host: HostPreparationAuthority,
}
fn retain_preparation_failure(
    cause: StoragePreparationCause,
    host: &HostPreparationAuthority,
) -> Error {
    Error::StorageSource(BackendFailure::from_error(StoragePreparationFailure {
        cause,
        _host: host.clone(),
    }))
}

impl DecoderCopyOwner {
    fn collect_storage_cold(
        &self,
        prepared: &PreparedResidentDecoderCopy<'_>,
        storage: &mut RetainedStorage,
    ) -> Result<(), StoragePreparationCause> {
        match self {
            Self::Live(payload) => payload
                .model
                .erased()
                .collect_snapshot_storage_fixed(storage)
                .map_err(Into::into),
            Self::Saved(saved) => {
                Self::collect_saved_storage(prepared, saved.input.as_ref(), storage)
                    .map_err(Into::into)
            }
        }
    }

    pub(super) fn collect_storage(&self, storage: &mut RetainedStorage) -> Result<(), Error> {
        match self {
            Self::Live(payload) => {
                payload
                    .model
                    .erased()
                    .collect_retained_decoder_state_storage(storage)?;
                payload
                    .model
                    .erased()
                    .collect_retained_idle_auxiliary_storage(storage)
            }
            Self::Saved(saved) => {
                let prepared = saved.native.prepare_copy()?;
                Self::collect_saved_storage(&prepared, saved.input.as_ref(), storage)
                    .map_err(|cause| Error::Other(Box::new(cause)))
            }
        }
    }

    fn collect_saved_storage(
        prepared: &PreparedResidentDecoderCopy<'_>,
        input: Option<&SharedPreparedInputCacheIdentity>,
        storage: &mut RetainedStorage,
    ) -> Result<(), StoragePreparationCause> {
        let mut failure = None;
        prepared.visit_registered_child_metadata_borrowed(&mut |metadata| {
            if failure.is_none() {
                failure = storage.include_slot_metadata(metadata.clone()).err();
            }
        });
        if let Some(cause) = failure {
            return Err(cause.into());
        }
        if prepared.is_paged() {
            SnapshotArraySources::visit_operands(prepared, &mut |source| {
                if failure.is_none() {
                    failure = match source {
                        SnapshotOperand::Array(array) => storage.include_array(array),
                        SnapshotOperand::Host(host) => storage.include_host(Arc::clone(host)),
                    }
                    .err();
                }
                Ok(())
            })?;
        } else {
            prepared.visit_retained_arrays(&mut |array| {
                if failure.is_none() {
                    failure = storage.include_array(array).err();
                }
            })?;
        }
        if let Some(cause) = failure {
            return Err(cause.into());
        }
        if let Some(layout) = prepared.shared_layout() {
            storage.include_metadata(SharedHostMetadata::Layout(layout.clone()))?;
        }
        if let Some(input) = input {
            storage.include_metadata(SharedHostMetadata::Input(input.clone()))?;
        }
        // Saved tables retain their private host hold; no registry row is minted.
        Ok(())
    }

    pub(super) fn complete_storage(&self) -> Result<RetainedStorage, Error> {
        let mut storage = RetainedStorage::default();
        self.collect_storage(&mut storage)?;
        Ok(storage)
    }

    pub(super) fn complete_storage_with_host(
        &self,
        pool: &WorkingMemoryPool,
        host: &HostPreparationAuthority,
    ) -> Result<RetainedStorage, Error> {
        let prepared = self
            .prepare()
            .map_err(|cause| retain_failure(cause, host))?;
        self.storage_plan(&prepared, pool)
            .map_err(|cause| retain_preparation_failure(cause, host))?
            .construct(host)
    }

    pub(super) fn storage_plan<'a>(
        &'a self,
        prepared: &PreparedResidentDecoderCopy<'_>,
        pool: &'a WorkingMemoryPool,
    ) -> Result<DecoderStoragePlan<'a>, StoragePreparationCause> {
        self.storage_plan_with_sampling(prepared, pool, None, None)
    }

    /// Counts and fills the same full saved source, including its exact optional
    /// sampling roots. These rows authenticate source custody; the separate
    /// isolated-copy plan still selects which roots receive copy credit.
    pub(super) fn storage_plan_with_sampling<'a>(
        &'a self,
        prepared: &PreparedResidentDecoderCopy<'_>,
        pool: &'a WorkingMemoryPool,
        key: Option<&'a Array>,
        pending: Option<&'a Array>,
    ) -> Result<DecoderStoragePlan<'a>, StoragePreparationCause> {
        DecoderStoragePlan::new(self, prepared, pool, key, pending)
    }
}
