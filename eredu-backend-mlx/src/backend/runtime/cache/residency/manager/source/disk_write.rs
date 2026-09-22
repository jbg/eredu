//! One real Host source and paid file destination through the existing worker.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataError;
use eredu_runtime::cache::{
    CacheShardError, CacheShardLayout, CacheShardMetadata, PreparedCacheIoTask,
    PreparedCacheIoTaskSlot, PreparedLiveCachePublication, PreparedLiveCachePublicationFailure,
};
use safemlx::{HostTransferDescriptor, error::Exception};

/// One-use writer. Construction does not mutate canonical state or enqueue I/O.
/// All fallible source preparation precedes its final canonical source pin.
pub(crate) struct PreparedDiskWrite {
    task: Option<PreparedCacheIoTaskSlot<DiskTask, DiskResult>>,
    body: WriteBody,
    output: PreparedDiskWriteOutput,
    worker: Arc<DiskWorker>,
}
struct WriteBody {
    file: Option<File>,
    failed_source: Option<LiveCacheBlockSource>,
    manager: CacheResidencyManager,
    publication: Option<PreparedLiveCachePublication>,
    layout: CacheShardLayout,
    location: DiskLocation,
    host: Option<HostCacheBlock>,
    descriptors: [HostTransferDescriptor<4>; 2],
    id: CacheBlockId,
    generation: u64,
    source_pins: usize,
    // Actual file/native Host resources precede source and pool retirement.
    pin: PinnedCacheBlock,
    transfer: DiskWriteOccupancy,
    disk: Option<CachePoolReservation>,
    funding: HostMetadataFunding,
}
struct Occupancy {
    reservation: Mutex<Option<CachePoolReservation>>,
    host_bytes: u64,
}
/// Closed shared ownership of this source's actual transfer reservation. The
/// canonical pending row only borrows its fixed capacity; it cannot spend it.
#[derive(Clone)]
pub(crate) struct DiskWriteOccupancy {
    inner: Arc<Occupancy>,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for DiskWriteOccupancy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskWriteOccupancy")
            .field("host_bytes", &self.inner.host_bytes)
            .finish_non_exhaustive()
    }
}
impl DiskWriteOccupancy {
    pub(crate) fn host_bytes(&self) -> u64 {
        self.inner.host_bytes
    }
}
struct WriteCompletion {
    result: Result<DiskLocation, DiskWriteFailure>,
    host: Mutex<Option<HostCacheBlock>>,
    body: WriteBody,
}
/// Creator-side retirement nodes never enter the Send I/O task.
pub(crate) struct PreparedDiskWriteHostRetirement {
    retirement: super::device_retirement::DeviceRetirement,
    context: WorkspaceContext,
    consumed: bool,
    publication_controls: usize,
}
/// Closed evidence of this writer's committed file and exact Host source.
/// The immutable result and source pin remain with the operation.
pub(crate) struct OrdinaryWrittenCacheHostSource {
    host: HostCacheBlock,
    file: LiveCacheBlockSource,
    id: CacheBlockId,
    context: WorkspaceContext,
    funding: HostMetadataFunding,
}
/// Preallocated output shell. Both success and failure retain the exact source,
/// partial destination, transfer charge and H; clones share that same owner.
#[derive(Clone)]
pub(crate) struct PreparedDiskWriteOutput {
    inner: Arc<OnceLock<WriteCompletion>>,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum DiskWriteFailure {
    #[error("prepared cache writer host source differs from its retained descriptor")]
    Source,
    #[error(transparent)]
    Host(Exception),
    #[error(transparent)]
    Io(std::io::Error),
    #[error(transparent)]
    Shard(CacheShardError),
    #[error(transparent)]
    Publication(PreparedLiveCachePublicationFailure),
}
impl std::fmt::Debug for PreparedDiskWriteOutput {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("PreparedDiskWriteOutput")
            .field("settled", &self.inner.get().is_some())
            .finish_non_exhaustive()
    }
}
#[path = "disk_write/destination.rs"]
mod destination;
pub(crate) use destination::{PreparedCacheDiskWriteSource, PreparedDiskWriteDestination};

impl CacheBlockSourceLoan<'_> {
    /// Selects an actual stable Host row and the manager's configured directory.
    /// Source facts alone grant no promotion, native execution or canonical commit.
    pub(crate) fn prepare_disk_write(
        &mut self,
        id: &CacheBlockId,
        context: &WorkspaceContext,
    ) -> Result<PreparedDiskWrite, CacheSourceFailure> {
        self.prepare_disk_write_with_pins(id, context, 0)
    }
    pub(crate) fn prepare_disk_write_from_source(
        &mut self,
        source: &crate::backend::nn::workspace::OriginalPagedDiskWriteSource<'_, '_>,
    ) -> Result<PreparedDiskWrite, CacheSourceFailure> {
        let context = source.source().context();
        let pins = source.validate(self).map_err(|cause| {
            CacheSourceFailure::metadata(context.metadata_source(cause), context)
        })?;
        self.prepare_disk_write_with_pins(source.id(), context, pins)
    }
    fn prepare_disk_write_with_pins(
        &mut self,
        id: &CacheBlockId,
        context: &WorkspaceContext,
        source_pins: usize,
    ) -> Result<PreparedDiskWrite, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(
                PreparedDiskWrite::fixed_control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let row = self
            .blocks()
            .find(|row| row.id() == id)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if row.phase() != CacheStoragePhase::HostUnbacked || row.disk().is_some() {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let host = row.host().ok_or_else(|| fail(CacheSourceError::Identity))?;
        let descriptors = [
            host[0]
                .try_fixed_descriptor::<4>()
                .map_err(|e| fail(e.into()))?,
            host[1]
                .try_fixed_descriptor::<4>()
                .map_err(|e| fail(e.into()))?,
        ];
        let shapes = [
            descriptors[0]
                .shape()
                .try_into()
                .map_err(|_| fail(CacheSourceError::Geometry))?,
            descriptors[1]
                .shape()
                .try_into()
                .map_err(|_| fail(CacheSourceError::Geometry))?,
        ];
        if self
            .lifecycle
            .lease_count(id)
            .map_err(|cause| fail(cause.into()))?
            != source_pins
            || self
                .lifecycle
                .source_pin_count(id)
                .map_err(|cause| fail(cause.into()))?
                != source_pins
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let plan = destination::prepare(
            self,
            id,
            shapes,
            descriptors.map(|d| d.dtype()),
            descriptors.map(|d| d.nbytes()),
            descriptors.map(|d| d.allocation().bytes()),
            2,
            context,
        )?;
        plan.bind_with_pins(self, source_pins)
    }
}
impl PreparedDiskWrite {
    /// Fixed native/file/callback controls. Shared metadata, path, pool-table,
    /// task and output-Arc constructors charge their own actual populations.
    pub(crate) fn fixed_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<(
                &mut CacheBlockSourceLoan<'_>,
                &CacheBlockId,
                &WorkspaceContext,
                usize,
            )>(),
            size_of::<(
                &mut CacheBlockSourceLoan<'_>,
                &crate::backend::nn::workspace::OriginalPagedDiskWriteSource<'_, '_>,
            )>(),
            size_of::<Result<usize, Exception>>(),
            size_of::<WriteBody>(),
            size_of::<WriteCompletion>(),
            initialized_mutex_control_bytes::<Option<HostCacheBlock>>()?,
            size_of::<DiskWriteOccupancy>(),
            size_of::<Occupancy>(),
            initialized_mutex_control_bytes::<Option<CachePoolReservation>>()?,
            size_of::<MutexGuard<'_, Option<CachePoolReservation>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, Option<CachePoolReservation>>,
                    TryLockError<MutexGuard<'_, Option<CachePoolReservation>>>,
                >,
            >(),
            size_of::<PreparedDiskWriteOutput>(),
            size_of::<DiskWriteFailure>(),
            size_of::<Result<DiskLocation, DiskWriteFailure>>(),
            size_of::<Result<(), LiveCacheBlockSource>>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<[HostTransferDescriptor<4>; 2]>(),
            size_of::<[[usize; 4]; 2]>(),
            size_of::<[&[u8]; 2]>(),
            size_of::<Result<&[u8], Exception>>(),
            size_of::<Result<File, std::io::Error>>(),
            size_of::<Result<(), std::io::Error>>(),
            size_of::<fs::OpenOptions>(),
            size_of::<Result<(), DiskWriteFailure>>(),
            size_of::<Option<PreparedLiveCachePublication>>(),
            size_of::<Result<LiveCacheBlockSource, PreparedLiveCachePublicationFailure>>(),
            size_of::<Result<(), WriteCompletion>>(),
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            PinnedCacheBlock::fixed_controls()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Binds the actual source into its cold task/completion shells after the
    /// source loan ended. The finite registry and queue must be installed.
    pub(crate) fn prepare_task(
        mut self,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheIoTask<DiskTask, DiskResult>, Error> {
        if context
            .metadata_funding()
            .as_ref()
            .is_none_or(|funding| !funding.same_account(&self.body.funding))
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let task = self
            .task
            .take()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        Ok(task.bind(DiskTask::PreparedWrite(self)))
    }
    /// Existing physical worker entry. It never formats ordinary error text.
    pub(crate) fn run(mut self) -> PreparedDiskWriteOutput {
        let result = self.body.write();
        let host = Mutex::new(self.body.host.take());
        drop(host.lock().expect("new unshared completion mutex"));
        // Once-only construction is private and immutable; no caller can replace
        // another result or substitute source identity after submission.
        if self
            .output
            .inner
            .set(WriteCompletion {
                result,
                host,
                body: self.body,
            })
            .is_err()
        {
            unreachable!("one prepared disk task completion");
        }
        self.output
    }
}
impl WriteBody {
    fn write(&mut self) -> Result<DiskLocation, DiskWriteFailure> {
        let [first, second] = self.host.as_ref().expect("one writer source").buffers();
        let bytes = [
            first.as_bytes().map_err(DiskWriteFailure::Host)?,
            second.as_bytes().map_err(DiskWriteFailure::Host)?,
        ];
        if bytes
            .iter()
            .zip(self.descriptors)
            .any(|(v, d)| v.len() != d.nbytes())
        {
            return Err(DiskWriteFailure::Source);
        }
        self.file = Some(
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(
                    self.publication
                        .as_ref()
                        .expect("one writer")
                        .staging_path(),
                )
                .map_err(DiskWriteFailure::Io)?,
        );
        let file = self.file.as_mut().expect("opened writer");
        self.layout
            .write_to(file, bytes)
            .map_err(DiskWriteFailure::Shard)?;
        file.sync_all().map_err(DiskWriteFailure::Io)?;
        drop(self.file.take());
        let source = self
            .publication
            .take()
            .expect("one publication")
            .commit_with_storage(
                self.layout.clone(),
                self.disk.take().expect("admitted file storage"),
            )
            .map_err(DiskWriteFailure::Publication)?;
        // The unique descriptor was paid before submission. Binding and the
        // returned alias allocate no path, verification map or source metadata.
        if let Err(source) = self.location.bind_live(source) {
            // The actual file owner must precede any source failure retirement.
            // This branch is unreachable for this writer's private destination.
            self.failed_source = Some(source);
            return Err(DiskWriteFailure::Source);
        }
        Ok(self.location.clone())
    }
}
impl PreparedDiskWriteOutput {
    /// The typed result is borrowed from its actual immutable task owner. A
    /// successful file alias independently retains the exact Disk reservation.
    pub(crate) fn result(&self) -> Option<Result<&LiveCacheBlockSource, &DiskWriteFailure>> {
        self.inner.get().map(|completion| {
            completion.result.as_ref().map(|location| {
                location
                    .live_source
                    .as_ref()
                    .expect("bound own publication")
            })
        })
    }
    pub(crate) fn key(&self) -> Option<CacheIoOperationKey> {
        self.inner.get().map(|c| CacheIoOperationKey {
            generation: c.body.generation,
            id: c.body.id.clone(),
            kind: CacheIoOperationKind::Write,
        })
    }
}

#[path = "disk_write/worker.rs"]
mod worker;
pub(crate) use worker::{
    DiskWriteOperation, DiskWriteOperationFailure, InstalledDiskWorker, PreparedDiskWorker,
};

use eredu_nn::workspace::WorkspaceMetadataAllocation;
