//! Exact nonblocking source loans; no tier transition, source copy, or grant.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, HostMetadataFunding},
};
use eredu_runtime::CacheBlockSelection;
use std::{
    mem::{size_of, size_of_val},
    sync::TryLockError,
};

/// Fixed source rejection, preserved by the caller's metadata owner.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CacheSourceError {
    #[error("cache source manager is busy")]
    Busy,
    #[error("cache source manager is poisoned")]
    Poisoned,
    #[error("cache source retains incomplete worker storage")]
    PendingStorage,
    #[error("cache source identity or frontier differs from its retained owner")]
    Identity,
    #[error("paged Host source identity differs during {0}")]
    HostIdentity(&'static str),
    #[error("cache source block or tail geometry is inconsistent")]
    Geometry,
    #[error("cache source current visible history is incomplete")]
    MissingHistory,
    #[error("cache source metadata size overflows")]
    Overflow,
    #[error(transparent)]
    Append(#[from] eredu_runtime::cache::PagedAppendError),
    #[error(transparent)]
    Scan(#[from] eredu_runtime::cache::PagedScanError),
    #[error(transparent)]
    Visible(#[from] eredu_runtime::cache::PagedVisibleError),
    #[error(transparent)]
    AbsoluteMask(#[from] eredu_nn::operation_geometry::AbsoluteAttentionMaskError),
    #[error(transparent)]
    Broadcast(#[from] eredu_nn::workspace::WorkspaceShapeError),
    #[error(transparent)]
    Descriptor(#[from] safemlx::ArrayDescriptorError),
    #[error(transparent)]
    Clone(#[from] safemlx::PreparedArrayCloneCause),
    #[error(transparent)]
    HostDescriptor(#[from] safemlx::HostTransferMetadataError),
    #[error(transparent)]
    ArrayMetadata(safemlx::ArrayMetadataError),
    #[error(transparent)]
    HostInput(safemlx::PreparedInputCause),
    #[error(transparent)]
    HostQuota(safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)]
    HostStore(safemlx::PreparedHostCopyError),
    #[error(transparent)]
    DeviceRetirementPreparation(safemlx::PreparedAllocationOwnerCause),
    #[error(transparent)]
    DeviceRetirementAttachment(safemlx::OriginalBufferCause),
    #[error(transparent)]
    HostPublication(host_demotion::HostPublicationError),
    #[error(transparent)]
    Lifecycle(#[from] CacheLifecycleError),
    #[error("cache block source requires a separately admitted tier promotion")]
    PromotionRequired,
    #[error(transparent)]
    DiskWorker(eredu_runtime::cache::CacheIoRegistryRefusal),
    #[error(transparent)]
    Shard(eredu_runtime::cache::CacheShardError),
    #[error(transparent)]
    DiskReservation(eredu_runtime::cache::CachePoolReservationPreparationFailure),
    #[error(transparent)]
    DiskAdmission(eredu_runtime::cache::CachePoolReservationAdmissionFailure),
    #[error("original paged append window removal is not yet qualified")]
    AppendWindow,
    #[error("original paged relative-bias source is not yet qualified")]
    RelativeScan,
}

/// Typed cause with no formatted source reconstruction.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CacheSourceFailureCause {
    #[error(transparent)]
    Source(CacheSourceError),
    #[error(transparent)]
    Metadata(Error),
    #[error(transparent)]
    Projection(crate::backend::nn::workspace::ProjectionSourceError),
}
/// Inline failure whose paying host owner outlives the error payload. No extra
/// native or dynamic error allocation is required to cross the lexical loan.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct CacheSourceFailure {
    #[source]
    cause: CacheSourceFailureCause,
    _funding: Option<HostMetadataFunding>,
}
impl CacheSourceFailure {
    pub(crate) fn source(cause: CacheSourceError, context: &WorkspaceContext) -> Self {
        Self {
            cause: CacheSourceFailureCause::Source(cause),
            _funding: context.metadata_funding(),
        }
    }
    pub(crate) fn metadata(cause: Error, context: &WorkspaceContext) -> Self {
        Self {
            cause: CacheSourceFailureCause::Metadata(cause),
            _funding: context.metadata_funding(),
        }
    }
    pub(crate) fn projection(
        cause: crate::backend::nn::workspace::ProjectionSourceError,
        context: &WorkspaceContext,
    ) -> Self {
        Self {
            cause: CacheSourceFailureCause::Projection(cause),
            _funding: context.metadata_funding(),
        }
    }
    /// Fixed source-phase attribution preserves the exact paid failure owner.
    pub(crate) fn at_host(mut self, stage: &'static str) -> Self {
        if matches!(
            &self.cause,
            CacheSourceFailureCause::Source(CacheSourceError::Identity)
        ) {
            self.cause = CacheSourceFailureCause::Source(CacheSourceError::HostIdentity(stage));
        }
        self
    }
    pub(crate) fn cause(&self) -> &CacheSourceFailureCause {
        &self.cause
    }
}

/// Actual durable backing borrowed under the manager loan. Paths and cached
/// payloads are neither reopened nor copied; a future reader must retain its
/// own exact source and completion/copy accounts.
#[derive(Clone, Copy)]
pub(crate) struct CacheDiskSource<'a> {
    location: &'a DiskLocation,
}
impl<'a> CacheDiskSource<'a> {
    pub(crate) fn path(self) -> &'a Path {
        &self.location.path
    }
    pub(crate) fn names(self) -> [&'a str; 2] {
        [&self.location.first_name, &self.location.second_name]
    }
    pub(crate) fn buffered(self) -> Option<&'a Arc<[u8]>> {
        self.location.buffered.as_ref()
    }
    pub(crate) fn payload_digest(self) -> Option<&'a str> {
        self.location.payload_sha256.as_deref()
    }
    pub(crate) fn persistent(self) -> bool {
        self.location.persistent
    }
    /// Actual live-file retirement owner. A path or digest alone is not this
    /// source; persistent prompt shards retain their separate artifact owner.
    pub(crate) fn live_file(self) -> Option<&'a LiveCacheBlockSource> {
        self.location.live_source.as_ref()
    }
}

/// One actual canonical catalog entry, never an identity supplied by a caller.
#[derive(Clone, Copy)]
pub(crate) struct CacheBlockSource<'a> {
    id: &'a CacheBlockId,
    record: &'a CacheBlockRecord,
}
impl<'a> CacheBlockSource<'a> {
    pub(crate) fn id(self) -> &'a CacheBlockId {
        self.id
    }
    pub(crate) fn logical_bytes(self) -> u64 {
        self.record.bytes
    }
    pub(crate) fn phase(self) -> CacheStoragePhase {
        self.record.physical.phase()
    }
    pub(crate) fn shapes(self) -> [&'a [i32]; 2] {
        [&self.record.shapes[0], &self.record.shapes[1]]
    }
    pub(crate) fn dtypes(self) -> [&'a str; 2] {
        [&self.record.dtypes[0], &self.record.dtypes[1]]
    }
    pub(crate) fn device(self) -> Option<[&'a Array; 2]> {
        self.record
            .physical
            .device_resource()
            .map(CacheBlockArrays::arrays)
    }
    pub(crate) fn host(self) -> Option<[&'a Arc<ImmutableHostTransferBuffer>; 2]> {
        self.record.host_block().map(|host| match host {
            HostCacheBlock::KeyValue { keys, values } => [keys, values],
            HostCacheBlock::CompressedLatentRotary { latent, rotary_key } => [latent, rotary_key],
        })
    }
    pub(crate) fn disk(self) -> Option<CacheDiskSource<'a>> {
        self.record
            .disk()
            .map(|location| CacheDiskSource { location })
    }
    pub(crate) fn imported(self) -> bool {
        self.record.imported
    }
    fn validate(self, session: u64) -> Result<(), CacheSourceError> {
        if self.id.session_id != session || self.record.physical.id() != self.id {
            return Err(CacheSourceError::Identity);
        }
        if self.id.start < 0 || self.id.end <= self.id.start {
            return Err(CacheSourceError::Geometry);
        }
        if self
            .record
            .physical
            .device_resource()
            .is_some_and(|value| value.representation() != self.id.representation)
            || self
                .record
                .host_block()
                .is_some_and(|value| value.representation() != self.id.representation)
        {
            return Err(CacheSourceError::Geometry);
        }
        let exact = match self.phase() {
            CacheStoragePhase::Device => self.device().is_some(),
            CacheStoragePhase::HostUnbacked => self.host().is_some() && self.disk().is_none(),
            CacheStoragePhase::HostBacked => self.host().is_some() && self.disk().is_some(),
            CacheStoragePhase::DiskReady => self.disk().is_some(),
            _ => false,
        };
        if !exact {
            return Err(CacheSourceError::PendingStorage);
        }
        Ok(())
    }
}

/// Lexical source view. Every reference remains inside the same manager guard;
/// neither block storage nor generation may be changed while it is borrowed.
/// The pool remains the actual manager's pool, not an inference-role account.
pub(crate) struct CacheBlockSourceLoan<'a> {
    manager: &'a CacheResidencyManager,
    generation: u64,
    records: &'a eredu_runtime::cache::CacheRecordTable<CacheBlockId, CacheBlockRecord>,
    lifecycle: &'a mut CacheBlockLifecycle,
    telemetry: &'a CacheResidencyTelemetry,
    publication_controls: usize,
    selection: CacheBlockSelection,
}
impl<'a> CacheBlockSourceLoan<'a> {
    /// Reborrows another descriptive layer selection under this same guard.
    /// No catalog/source reference can outlive the enclosing lexical loan.
    pub(crate) fn selected(&mut self, selection: CacheBlockSelection) -> CacheBlockSourceLoan<'_> {
        CacheBlockSourceLoan {
            manager: self.manager,
            generation: self.generation,
            records: self.records,
            lifecycle: self.lifecycle,
            telemetry: self.telemetry,
            publication_controls: self.publication_controls,
            selection,
        }
    }
    /// Read-only ownership facts for a row in this exact lexical selection.
    /// Counts alone never grant mutation or authorize ignoring another owner.
    pub(crate) fn source_ownership(
        &self,
        id: &CacheBlockId,
    ) -> Result<(usize, bool), CacheSourceError> {
        if !self.blocks().any(|row| row.id() == id) {
            return Err(CacheSourceError::Identity);
        }
        Ok((
            self.lifecycle.source_pin_count(id)?,
            self.lifecycle.is_device_leased(id)?,
        ))
    }
    /// Fixed controls of this actual manager publication worker; no future
    /// block/row allowance is inferred from the retained scalar.
    pub(crate) fn publication_control_bytes(&self) -> usize { self.publication_controls }
    pub(crate) fn catalog_population(&self) -> (usize, usize) {
        self.lifecycle.catalog_population()
    }

    pub(crate) fn manager(&self) -> &'a CacheResidencyManager {
        self.manager
    }
    pub(crate) fn session_id(&self) -> u64 {
        self.manager.session_id
    }
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
    pub(crate) fn pool(&self) -> &'a CacheResidencyPool {
        self.manager.pool()
    }
    pub(crate) fn selection(&self) -> CacheBlockSelection {
        self.selection
    }
    pub(crate) fn tail(&self) -> Option<MutableCacheTail> {
        self.lifecycle.tail(self.selection.layer())
    }
    pub(crate) fn blocks(&self) -> impl Iterator<Item = CacheBlockSource<'a>> + '_ {
        self.all_blocks()
            .filter(|value| self.selection.includes(value.id()))
    }
    /// Complete manager storage is available for the enclosing multi-layer
    /// physical census; selected blocks alone are not that whole-source bound.
    pub(crate) fn all_blocks(&self) -> impl Iterator<Item = CacheBlockSource<'a>> + '_ {
        self.records
            .iter()
            .map(|(id, record)| CacheBlockSource { id, record })
    }
}
impl CacheResidencyManager {
    pub(crate) fn same_catalog(&self, other: &Self) -> bool {
        self.session_id == other.session_id && Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Own controls of the exact lexical loan. Consumer closures price their
    /// own work; callback storage and its result are supplied by actual types.
    pub(crate) fn source_loan_control_bytes<R>(callback_bytes: usize) -> Option<usize> {
        Self::source_loan_fixed_bytes::<R>(callback_bytes)
    }
    fn source_loan_fixed_bytes<R>(callback_bytes: usize) -> Option<usize> {
        let parts = [
            size_of::<CacheBlockSourceLoan<'_>>(),
            size_of::<CacheBlockSource<'_>>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<&CacheBlockSourceLoan<'_>>(),
            size_of::<Result<(), CacheSourceError>>(),
            size_of::<CacheBlockSelection>(),
            size_of::<(&Self, CacheBlockSelection, Option<usize>)>(),
            size_of::<&WorkspaceContext>(),
            size_of::<Option<usize>>(),
            size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(
            ),
            size_of::<Result<R, CacheSourceFailure>>(),
            size_of::<R>(),
            size_of::<bool>(),
            size_of::<CacheSourceFailure>(),
            size_of::<Option<HostMetadataFunding>>(),
            callback_bytes,
            reporting::report_query_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Validates the actual canonical source before lending any entries. The
    /// callback may inspect descriptors and construct paid metadata, but must
    /// not reenter this manager, promote/read blocks, or clone/drop raw native
    /// owners. Typed callback errors are returned unchanged, after unlocking.
    /// No reference to the loan can escape in `R`.
    pub(crate) fn with_source_loan<R>(
        &self,
        selection: CacheBlockSelection,
        context: &WorkspaceContext,
        consume: impl for<'loan> FnOnce(CacheBlockSourceLoan<'loan>) -> Result<R, CacheSourceFailure>,
    ) -> Result<R, CacheSourceFailure> {
        let controls = Self::source_loan_fixed_bytes::<R>(size_of_val(&consume))
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        context
            .charge_metadata(controls)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        self.with_source_loan_inner(
            selection,
            None,
            |cause| CacheSourceFailure::source(cause, context),
            consume,
        )
    }
    pub(crate) fn inspect_copy_source<R, E>(
        &self,
        selection: CacheBlockSelection,
        fail: impl Fn(CacheSourceError) -> E,
        consume: impl for<'loan> FnOnce(CacheBlockSourceLoan<'loan>) -> Result<R, E>,
    ) -> Result<R, E> {
        self.with_source_loan_inner(selection, None, fail, consume)
    }
    pub(super) fn with_source_loan_inner<R, E>(
        &self,
        selection: CacheBlockSelection,
        prepared_publication_controls: Option<usize>,
        fail: impl Fn(CacheSourceError) -> E,
        consume: impl for<'loan> FnOnce(CacheBlockSourceLoan<'loan>) -> Result<R, E>,
    ) -> Result<R, E> {
        let mut state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => {
                return Err(fail(CacheSourceError::Busy));
            }
            Err(TryLockError::Poisoned(_)) => {
                return Err(fail(CacheSourceError::Poisoned));
            }
        };
        if !self.borrowed_storage_complete(&state) {
            drop(state);
            return Err(fail(CacheSourceError::PendingStorage));
        }
        let publication_controls = match prepared_publication_controls {
            Some(value) => value,
            None => reporting::report_control_bytes(&state)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        };
        let CacheManagerState {
            generation,
            blocks,
            lifecycle,
            telemetry,
            ..
        } = &mut *state;
        let loan = CacheBlockSourceLoan {
            manager: self,
            generation: *generation,
            records: blocks,
            lifecycle,
            telemetry,
            publication_controls,
            selection,
        };
        let validated = loan
            .all_blocks()
            .try_for_each(|source| source.validate(self.session_id));
        if let Err(cause) = validated {
            drop(loan);
            drop(state);
            return Err(fail(cause));
        }
        let result = consume(loan);
        let complete = self.borrowed_storage_complete(&state);
        drop(state);
        // A returned value may own native handles; retire it only after unlock.
        match result {
            Err(cause) => Err(cause),
            Ok(value) if complete => Ok(value),
            Ok(_) => Err(fail(CacheSourceError::PendingStorage)),
        }
    }
}

#[path = "source/pin.rs"]
mod pin;
pub(crate) use pin::PinnedCacheSource;

#[cfg(test)]
#[path = "source/tests.rs"]
mod tests;

#[path = "source/catalog.rs"]
mod catalog;
pub(crate) use catalog::{CatalogInstallFailure, InstalledManagerCatalog, PreparedManagerCatalog};

#[path = "source/block_pin.rs"]
mod block_pin;
pub(crate) use block_pin::{PinnedCacheBlock, PinnedCacheBlockLease};

#[path = "source/independent.rs"]
mod independent;
pub(crate) use independent::{IndependentCacheManagerPlan, PreparedIndependentCacheManager};

#[path = "source/registered_copy.rs"]
mod registered_copy;

pub(crate) use registered_copy::{PagedArrayCopyLayout, PreparedPagedArrayCopy};

#[path = "source/host_promotion.rs"]
mod host_promotion;
pub(crate) use host_promotion::{PreparedCacheHostPromotion, PreparedCacheHostPromotionSlots};

#[path = "source/host_demotion.rs"]
mod host_demotion;
pub(crate) use host_demotion::{PreparedCacheHostDemotion, StoredCacheHostSource};

#[path = "source/disk_write.rs"]
mod disk_write;
pub(crate) use disk_write::{
    DiskWriteOccupancy, DiskWriteOperation, DiskWriteOperationFailure, PreparedDiskWrite,
    PreparedDiskWriteOutput,
};

#[path = "source/disk_read.rs"]
mod disk_read;
pub(crate) use disk_read::{
    CompletedDiskRead, DiskReadFinishFailure, DiskReadOccupancy, DiskReadOperation,
    DiskReadOperationFailure, PreparedDiskRead, PreparedDiskReadOutput, PreparedDiskReadSource,
};

pub(crate) use disk_write::PreparedDiskWriteDestination;

pub(crate) use disk_write::{InstalledDiskWorker, PreparedDiskWorker};

pub(crate) use disk_read::{DiskReadBinding, PreparedDiskReadDestination, disk_read_source_facts};

pub(crate) use host_promotion::PreparedInitialDiskReturn;

#[path = "source/disk_backing.rs"]
mod disk_backing;

#[path = "source/device_retirement.rs"]
mod device_retirement;
