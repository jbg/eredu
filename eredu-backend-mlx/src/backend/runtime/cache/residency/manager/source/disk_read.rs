//! Exact published cache file into paid final Host destinations on the shared worker.
use super::*;
use crate::backend::runtime::residency::storage::filled_host;
use eredu_nn::workspace::WorkspaceMetadataError;
use eredu_runtime::working_memory::WorkingMemoryError;
use eredu_runtime::{
    cache::{CacheShardLayout, PreparedCacheIoTask},
    working_memory::{
        HostSourceConstructionFacts, OriginalHostSourceBank, OriginalHostSourceCustody,
    },
};
use safemlx::{PreparedHostTransferPlan, PreparedInputRuntime};

#[path = "disk_read/filling.rs"]
mod filling;
#[path = "disk_read/output.rs"]
mod output;
use filling::Filling;
pub(crate) use output::{CompletedDiskRead, DiskReadFinishFailure};
#[path = "disk_read/destination.rs"]
mod destination;
pub(crate) use destination::{
    DiskReadBinding, PreparedDiskReadDestination, disk_read_source_facts,
};

/// Creator-thread source/destination preparation. Neither a canonical read
/// transition nor an I/O task has started while this descriptor is retained.
pub(crate) struct PreparedDiskReadSource {
    task: PreparedDiskRead,
    context: WorkspaceContext,
}
/// Only this Send payload enters the existing I/O worker. It cannot freeze,
/// register, promote or submit a native buffer on the worker thread.
pub(crate) struct PreparedDiskRead {
    task: Option<eredu_runtime::cache::PreparedCacheIoTaskSlot<DiskTask, DiskResult>>,
    operation: Option<operation::CompletionSlots>,
    body: ReadBody,
    output: PreparedDiskReadOutput,
    worker: Arc<DiskWorker>,
}
struct ReadBody {
    filling: [Option<Filling>; 2],
    ready: [Option<Arc<ImmutableHostTransferBuffer>>; 2],
    host: Option<HostCacheBlock>,
    bytes: Vec<u8>,
    read: Option<PreparedCacheFileRead>,
    location: DiskLocation,
    layout: CacheShardLayout,
    source: CacheFileSource,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    capacities: [usize; 2],
    source_bytes: u64,
    source_custody: Option<OriginalHostSourceCustody>,
    // Only the ordinary constructor can retain its authentic cold source here.
    ordinary_identity: Option<Arc<()>>,
    ordinary_placements: Option<[safemlx::AllocationPlacement; 2]>,
    manager: CacheResidencyManager,
    id: CacheBlockId,
    generation: u64,
    // Native Host/file prefixes retire before their canonical pin and occupancy.
    pin: PinnedCacheBlock,
    reservation: DiskReadOccupancy,
    transfer: CachePoolReservation,
    funding: HostMetadataFunding,
}
/// Actual physical Host reservation shared by source arenas, task and pending row.
#[derive(Clone)]
pub(crate) struct DiskReadOccupancy {
    inner: Arc<Mutex<Option<CachePoolReservation>>>,
    host_bytes: u64,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for DiskReadOccupancy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskReadOccupancy")
            .field("host_bytes", &self.host_bytes)
            .finish_non_exhaustive()
    }
}
impl DiskReadOccupancy {
    pub(crate) fn host_bytes(&self) -> u64 {
        self.host_bytes
    }
}
/// Final native source arenas retain the real Host reservation, including
/// worker-side deferred writer teardown. I/O transfer occupancy is separate.
struct DiskReadPermit(DiskReadOccupancy);
impl filled_host::HostFillPermit for DiskReadPermit {
    type Owner = (
        eredu_runtime::working_memory::OriginalHostSourceReceipt,
        DiskReadOccupancy,
    );
    fn validate(
        &self,
        _: eredu_runtime::working_memory::HostSourcePeakSelection,
        _: usize,
        _: u64,
    ) -> Result<(), WorkingMemoryError> {
        Err(WorkingMemoryError::IdentityMismatch)
    }
    fn create_quota(
        self,
        bytes: usize,
        receipt: eredu_runtime::working_memory::OriginalHostSourceReceipt,
    ) -> Result<safemlx::PreparedSubmissionGraphQuota<Self::Owner>, filled_host::SourceCause> {
        safemlx::PreparedSubmissionGraphQuota::try_new(bytes, (receipt, self.0))
            .map_err(|error| filled_host::SourceCause::Arena(error.cause()))
    }
}
struct ReadCompletion {
    result: Result<(), DiskReadFailure>,
    body: Mutex<Option<ReadBody>>,
}
#[derive(Clone)]
pub(crate) struct PreparedDiskReadOutput {
    inner: Arc<OnceLock<ReadCompletion>>,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum DiskReadFailure {
    #[error("prepared cache read source or destination geometry differs")]
    Source,
    #[error(transparent)]
    Open(std::io::Error),
    #[error(transparent)]
    Read(CacheFileReadFailure),
    #[error(transparent)]
    Shard(eredu_runtime::cache::CacheShardError),
}
impl std::fmt::Debug for PreparedDiskReadOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedDiskReadOutput")
            .field("settled", &self.inner.get().is_some())
            .finish_non_exhaustive()
    }
}
impl CacheBlockSourceLoan<'_> {
    /// Selects the authenticated retained file and its exact schema/version.
    /// All dynamic preparation precedes the final canonical source pin.
    pub(crate) fn prepare_disk_read(
        &mut self,
        id: &CacheBlockId,
        runtime: &PreparedInputRuntime,
        context: &WorkspaceContext,
    ) -> Result<PreparedDiskReadSource, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let row = self
            .blocks()
            .find(|row| row.id() == id)
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        let disk = row
            .disk()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        let source = disk
            .file_source()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        let layout = source
            .layout()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?
            .clone();
        let destination = destination::prepare(self, id, &layout, runtime, 2, context)?;
        let binding = destination.bind_source(self)?;
        destination.bind(binding)
    }
}
impl PreparedDiskReadSource {
    pub(crate) fn source_facts(&self) -> Result<HostSourceConstructionFacts, WorkingMemoryError> {
        if self.task.body.ordinary_identity.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        HostSourceConstructionFacts::new(self.task.body.source_bytes, 2, 0)
    }
    /// Caller-thread allocation/binding only. Failure retains every actual
    /// native prefix; queue/task publication occurs through the existing worker.
    pub(crate) fn construct(
        mut self,
        runtime: &PreparedInputRuntime,
        bank: &mut OriginalHostSourceBank,
        custody: &OriginalHostSourceCustody,
    ) -> Result<PreparedDiskRead, DiskReadFinishFailure> {
        let result = (|| {
            let body = &mut self.task.body;
            if body.ordinary_identity.is_some() {
                return Err(output::FinishCause::Identity);
            }
            destination::begin_filling(
                destination::Initialization {
                    filling: &mut body.filling,
                    source_custody: &mut body.source_custody,
                    shapes: &body.shapes,
                    dtypes: body.dtypes,
                    capacities: body.capacities,
                    reservation: &body.reservation,
                },
                runtime,
                bank,
                custody,
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => self.open(),
            Err(cause) => Err(DiskReadFinishFailure::retain(
                cause,
                self.task.body,
                self.task.output,
            )),
        }
    }
    /// Opens and binds only the actual published file after native destinations
    /// are retained outside the manager loan. Failed prefixes retain their source.
    pub(crate) fn open(mut self) -> Result<PreparedDiskRead, DiskReadFinishFailure> {
        let result = (|| {
            let body = &mut self.task.body;
            if (body.source_custody.is_some() == body.ordinary_identity.is_some())
                || body.filling.iter().any(Option::is_none)
                || body.read.is_some()
            {
                return Err(output::FinishCause::Identity);
            }
            let file = File::open(body.source.path())
                .map_err(DiskReadFailure::Open)
                .map_err(output::FinishCause::Read)?;
            body.read = Some(
                body.source
                    .prepare_read_from(file, &self.context)
                    .map_err(DiskReadFailure::Read)
                    .map_err(output::FinishCause::Read)?,
            );
            Ok(())
        })();
        match result {
            Ok(()) => Ok(self.task),
            Err(cause) => Err(DiskReadFinishFailure::retain(
                cause,
                self.task.body,
                self.task.output,
            )),
        }
    }
}
impl PreparedDiskRead {
    pub(crate) fn output(&self) -> PreparedDiskReadOutput {
        self.output.clone()
    }
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
        Ok(task.bind(DiskTask::PreparedRead(self)))
    }
    pub(crate) fn run(mut self) -> PreparedDiskReadOutput {
        let result = self.body.read_payload();
        let body = Mutex::new(Some(self.body));
        drop(body.lock().expect("new unshared read-body mutex"));
        if self
            .output
            .inner
            .set(ReadCompletion { result, body })
            .is_err()
        {
            unreachable!("one prepared disk read completion");
        }
        self.output
    }
    fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<DiskReadOccupancy>(),
            WorkspaceContext::metadata_arc_bytes::<Mutex<Option<CachePoolReservation>>>()?,
            initialized_mutex_control_bytes::<Option<CachePoolReservation>>()?,
            size_of::<PreparedDiskReadSource>(),
            size_of::<ReadBody>(),
            size_of::<ReadCompletion>(),
            initialized_mutex_control_bytes::<Option<ReadBody>>()?,
            size_of::<PreparedDiskReadOutput>(),
            size_of::<DiskReadFailure>(),
            size_of::<DiskReadFinishFailure>(),
            size_of::<Result<PreparedDiskReadSource, CacheSourceFailure>>(),
            size_of::<Result<Self, DiskReadFinishFailure>>(),
            size_of::<Result<(), DiskReadFailure>>(),
            size_of::<Result<(), output::FinishCause>>(),
            size_of::<Result<(), ReadCompletion>>(),
            size_of::<[(&[usize], StoredDtype, usize); 2]>(),
            size_of::<[[i32; 4]; 2]>(),
            size_of::<[Dtype; 2]>(),
            size_of::<[eredu_runtime::cache::CacheShardTensor<'_>; 2]>(),
            size_of::<
                Result<
                    [eredu_runtime::cache::CacheShardTensor<'_>; 2],
                    eredu_runtime::cache::CacheShardError,
                >,
            >(),
            size_of::<Result<File, std::io::Error>>(),
            size_of::<fs::OpenOptions>(),
            size_of::<PreparedHostTransferPlan<'_>>(),
            size_of::<Result<PreparedHostTransferPlan<'_>, safemlx::PreparedInputCause>>(),
            size_of::<Result<u64, WorkingMemoryError>>(),
            size_of::<Result<HostSourceConstructionFacts, WorkingMemoryError>>(),
            size_of::<(&PreparedDiskReadSource,)>(),
            size_of::<(
                &mut CacheBlockSourceLoan<'_>,
                &CacheBlockId,
                &PreparedInputRuntime,
                &WorkspaceContext,
            )>(),
            size_of::<(
                PreparedDiskReadSource,
                &PreparedInputRuntime,
                &mut OriginalHostSourceBank,
                &OriginalHostSourceCustody,
            )>(),
            size_of::<(
                PreparedDiskRead,
                &WorkspaceContext,
                CacheIoOperationKey,
                Arc<DiskWorker>,
            )>(),
            size_of::<(&mut ReadBody, [usize; 2], [&mut [u8]; 2])>(),
            size_of::<(&mut Filling, &mut [u8])>().checked_mul(2)?,
            WorkspaceContext::metadata_arc_bytes::<OnceLock<ReadCompletion>>()?,
            WorkspaceContext::metadata_arc_bytes::<ImmutableHostTransferBuffer>()?
                .checked_mul(2)?,
            PinnedCacheBlock::fixed_controls()?,
            output::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
impl ReadBody {
    fn read_payload(&mut self) -> Result<(), DiskReadFailure> {
        self.read
            .take()
            .ok_or(DiskReadFailure::Source)?
            .read_into(&mut self.bytes)
            .map_err(DiskReadFailure::Read)?;
        let tensors = self
            .layout
            .tensors(&self.bytes)
            .map_err(DiskReadFailure::Shard)?;
        // Check both exact destinations before writing either tensor payload.
        for (index, tensor) in tensors.iter().enumerate() {
            let destination = self.filling[index]
                .as_mut()
                .ok_or(DiskReadFailure::Source)?;
            if destination.bytes_mut().len() != tensor.data().len() {
                return Err(DiskReadFailure::Source);
            }
        }
        for (index, tensor) in tensors.iter().enumerate() {
            self.filling[index]
                .as_mut()
                .expect("validated destination")
                .bytes_mut()
                .copy_from_slice(tensor.data());
        }
        Ok(())
    }
}

#[path = "disk_read/operation.rs"]
mod operation;
pub(super) use operation::ReadCacheHostSource;
pub(super) use operation::prepare_operation;
pub(crate) use operation::{DiskReadOperation, DiskReadOperationFailure};

pub(crate) use operation::{OrdinaryDiskReadSource, OrdinaryReadCacheHostSource};
