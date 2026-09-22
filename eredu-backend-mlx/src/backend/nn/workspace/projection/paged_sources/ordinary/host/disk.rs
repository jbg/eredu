//! Durable writes and completed reads use the retained ordinary itinerary.
use super::runtime::{Promotion, selection};
use super::*;
use crate::backend::runtime::cache::residency::{
    CacheBlockSourceLoan, DiskReadBinding, DiskReadFinishFailure, DiskReadOperationFailure,
    DiskWriteOperationFailure, PreparedCacheDiskWriteSource, PreparedDiskReadSource,
    PreparedDiskWrite,
};

struct WriteSource<'a, 'source> {
    source: &'a TransferSource<'source>,
    id: &'a CacheBlockId,
    pins: usize,
}
impl PreparedCacheDiskWriteSource for WriteSource<'_, '_> {
    fn id(&self) -> &CacheBlockId {
        self.id
    }
    fn context(&self) -> &WorkspaceContext {
        &self.source.work.program.inner.context
    }
    fn validate(&self, loan: &CacheBlockSourceLoan<'_>) -> Result<usize, Exception> {
        self.source
            .validate_manager(loan.manager(), loan.generation())?;
        Ok(self.pins)
    }
}

fn transport<E: std::error::Error + Send + Sync + 'static>(
    source: &TransferSource<'_>,
    cause: E,
) -> Exception {
    let context = &source.work.program.inner.context;
    source.error(CacheSourceFailure::metadata(context.metadata_source(cause), context).into())
}

impl HostProgram {
    pub(in super::super) fn install_disk_workers(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        super::super::super::host_program::disk::install_workers(&mut self.disk_workers, context)
    }
    fn worker(&self, source: &TransferSource<'_>) -> Result<usize, Exception> {
        let sources = source.work.program.inner.sources.sources();
        self.disk_workers
            .iter()
            .position(|worker| {
                sources[worker.source_index()]
                    .manager()
                    .same_catalog(source.source.manager())
            })
            .ok_or_else(|| source.error(CacheSourceError::Identity.into()))
    }
    pub(super) fn make_disk_room(
        &mut self,
        source: &TransferSource<'_>,
        ordinal: usize,
        required: Option<&CacheBlockId>,
        additional_host: u64,
        proactive: bool,
    ) -> Result<(), Exception> {
        loop {
            let Some(id) = source.source.manager().prepared_disk_write_victim(
                source,
                required,
                additional_host,
                proactive,
            )?
            else {
                return Ok(());
            };
            let sources = source.work.program.inner.sources.sources();
            let index = self
                .stores
                .iter()
                .position(|store| {
                    store.id == id
                        && store.first_ordinal <= ordinal
                        && sources[store.source]
                            .manager()
                            .same_catalog(source.source.manager())
                })
                .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?;
            if self.stores[index].write.is_some() {
                return Err(source.error(CacheSourceError::Identity.into()));
            }
            let worker = self.worker(source)?;
            let pins = self
                .pin_count(source.work, source.source.manager(), &id)
                .map_err(|cause| source.error(cause.into()))?;
            let selected = TransferSource {
                source: &sources[self.stores[index].source],
                ..*source
            };
            let proof = WriteSource {
                source: &selected,
                id: &id,
                pins,
            };
            let destination = self.stores[index]
                .disk
                .take()
                .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?;
            let write = source
                .installed
                .with_source(selection(&id), |mut loan| {
                    Ok(destination.bind(&mut loan, &proof))
                })
                .map_err(|cause| source.error(cause.into()))?
                .map_err(|cause| source.error(cause.into()))?;
            let operation = self.disk_workers[worker]
                .installed()
                .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?
                .prepare_write(write, &source.work.program.inner.context)
                .map_err(|cause| source.error(cause.into()))?;
            self.stores[index].write = Some(operation);
            let operation = self.stores[index]
                .write
                .as_mut()
                .expect("retained before submission");
            operation
                .submit()
                .map_err(|cause| transport(source, cause))?;
            operation
                .finish()
                .map_err(|cause| transport(source, cause))?;
            let pins = self
                .pin_count(source.work, source.source.manager(), &id)
                .map_err(|cause| source.error(cause.into()))?;
            let proof = WriteSource {
                source: &selected,
                id: &id,
                pins,
            };
            let store = &mut self.stores[index];
            let completed = store
                .write
                .as_mut()
                .expect("completed writer")
                .retire_ordinary_host_source(
                    &proof,
                    store
                        .write_retirement
                        .as_mut()
                        .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?,
                )
                .map_err(|cause| transport(source, cause))?;
            if let Some(mover) = &mut store.mover {
                if mover.buffers().is_some() {
                    mover
                        .release_written_host_source(&completed, &source.work.program.inner.context)
                        .map_err(|cause| source.error(cause.into()))?;
                }
            }
            if let Some(host) = &store.initial_host {
                completed
                    .validate_buffers(
                        &id,
                        [host[0].as_ref(), host[1].as_ref()],
                        &source.work.program.inner.context,
                    )
                    .map_err(|cause| source.error(cause.into()))?;
            }
            store.initial_host = None;
            for load in self
                .loads
                .iter_mut()
                .filter(|load| load.store == index && load.ordinal <= ordinal)
            {
                load.promotion
                    .release_written_host_source(&completed)
                    .map_err(|cause| source.error(cause.into()))?;
            }
            drop(completed);
            self.stores[index]
                .write_retirement
                .as_mut()
                .expect("retained staging source")
                .reclaim();
            if proactive {
                return Ok(());
            }
        }
    }
    pub(super) fn read_disk_source(
        &mut self,
        source: &TransferSource<'_>,
        ordinal: usize,
        load: usize,
    ) -> Result<(), Exception> {
        let id = self.stores[self.loads[load].store].id.clone();
        let capacity = self.loads[load]
            .disk_read
            .as_ref()
            .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?
            .host_capacity();
        self.make_disk_room(source, ordinal, Some(&id), capacity, false)?;
        let worker = self.worker(source)?;
        let proof = Promotion { source, id: &id };
        let destination = self.loads[load]
            .disk_read
            .as_mut()
            .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?;
        let binding = source
            .installed
            .with_source(selection(&id), |mut loan| {
                Ok(destination.bind_source(&mut loan))
            })
            .map_err(|cause| source.error(cause.into()))?
            .map_err(|cause| source.error(cause.into()))?;
        // The authentic binding retains the canonical pin before any physical
        // destination is created. The row retains each attempted writer prefix.
        destination.construct_ordinary(&proof, &binding)?;
        let read = self.loads[load]
            .disk_read
            .take()
            .expect("constructed destination")
            .bind(binding)
            .map_err(|cause| source.error(cause.into()))?
            .open()
            .map_err(|cause| transport(source, cause))?;
        let operation = self.disk_workers[worker]
            .installed()
            .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?
            .prepare_read(read, &source.work.program.inner.context)
            .map_err(|cause| source.error(cause.into()))?;
        self.loads[load].read_operation = Some(operation);
        let operation = self.loads[load]
            .read_operation
            .as_mut()
            .expect("retained before submission");
        operation
            .submit()
            .map_err(|cause| transport(source, cause))?;
        operation.finish().map_err(|cause| transport(source, cause))
    }
    pub(super) fn promote_disk_read(
        &mut self,
        source: &TransferSource<'_>,
        id: &CacheBlockId,
        index: usize,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let proof = Promotion { source, id };
        let load = &mut self.loads[index];
        let completed = load
            .read_operation
            .as_mut()
            .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?
            .take_ordinary_host_source(
                &proof,
                load.read_source
                    .as_ref()
                    .ok_or_else(|| source.error(CacheSourceError::Identity.into()))?,
            )?;
        load.promotion.run_read(&proof, completed, stream)
    }
}

pub(super) fn write_controls(
    writer: &PreparedDiskWriteDestination,
    reporting: usize,
) -> Option<OrdinaryCallControls> {
    writer.ordinary_call_controls::<WriteSource<'_, '_>>(reporting)
}
pub(super) fn read_controls(
    read: &PreparedDiskReadDestination,
    reporting: usize,
) -> Option<OrdinaryCallControls> {
    read.ordinary_call_controls::<Promotion<'_, '_>>(reporting)
}

impl HostProgram {
    pub(super) fn disk_call_controls(
        &self,
        ordinals: Range<usize>,
    ) -> Option<OrdinaryCallControls> {
        if !self.requires_disk {
            return Some(OrdinaryCallControls::default());
        }
        let mut controls = OrdinaryCallControls::default();
        let mut stores = 0usize;
        let mut loads = 0usize;
        let mut writes = 0usize;
        let mut reads = 0usize;
        for store in &self.stores {
            if store.first_ordinal >= ordinals.end {
                continue;
            }
            stores = stores.checked_add(1)?;
            if let Some(writer) = store.disk_controls {
                writes = writes.checked_add(1)?;
                controls = controls.append(writer)?;
                controls = controls.append(
                    store
                        .write_retirement
                        .as_ref()?
                        .source_controls::<WriteSource<'_, '_>>()?,
                )?;
                controls = controls.metadata(
                    PreparedOrdinaryCacheHostDemotion::written_host_release_control_bytes()?
                        .checked_add(OrdinaryWrittenCacheHostSource::validation_control_bytes())?
                        .checked_add(
                            WorkspaceContext::metadata_source_bytes::<DiskWriteOperationFailure>()?
                                .checked_mul(3)?,
                        )?,
                )?;
            }
            if let Some(backed) = &store.initial_disk {
                controls =
                    controls.append(backed.ordinary_source_controls::<Eviction<'_, '_>>()?)?;
            }
        }
        for load in &self.loads {
            if load.ordinal >= ordinals.end {
                continue;
            }
            loads = loads.checked_add(1)?;
            if let Some(reader) = load.disk_controls {
                reads = reads.checked_add(1)?;
                controls = controls.append(reader)?;
            }
            // A load may return its reusable Host source to a writer once.
            controls = controls.metadata(
                OrdinaryWrittenCacheHostSource::validation_control_bytes().checked_add(
                    size_of::<(
                        &mut PreparedOrdinaryCacheHostPromotion,
                        &OrdinaryWrittenCacheHostSource,
                        Result<(), CacheSourceError>,
                    )>(),
                )?,
            )?;
        }
        // Each possible initial store or loaded Device source can be demoted
        // once. Demotion checks room before and after its actual Host move; a
        // selected file read checks room once before constructing its buffers.
        // Each writer contributes one additional successful policy iteration.
        let room_calls = stores
            .checked_add(loads)?
            .checked_mul(2)?
            .checked_add(reads)?;
        let iterations = room_calls.checked_add(writes)?;
        let bytes = outer_controls()?
            .checked_mul(iterations.checked_add(reads)?.checked_add(1)?)?
            .checked_add(
                CacheResidencyManager::prepared_host_policy_control_bytes::<TransferSource<'_>>()?
                    .checked_mul(iterations)?,
            )?;
        controls.metadata(bytes)
    }
}

fn outer_controls() -> Option<usize> {
    let frames = [
        size_of::<WriteSource<'_, '_>>(),
        size_of::<(&WriteSource<'_, '_>, &CacheBlockSourceLoan<'_>)>(),
        size_of::<Result<usize, Exception>>(),
        size_of::<(
            &mut HostProgram,
            &TransferSource<'_>,
            usize,
            Option<&CacheBlockId>,
            u64,
            bool,
        )>(),
        size_of::<(&mut HostProgram, &TransferSource<'_>, usize, usize)>(),
        size_of::<(
            &mut HostProgram,
            &TransferSource<'_>,
            &CacheBlockId,
            usize,
            &Stream,
        )>(),
        size_of::<Result<(), Exception>>(),
        size_of::<(usize, CacheBlockId, u64)>(),
        size_of::<Option<CacheBlockId>>(),
        size_of::<std::slice::Iter<'_, super::super::super::host_program::disk::Worker>>(),
        size_of::<std::slice::Iter<'_, Store>>(),
        size_of::<std::slice::IterMut<'_, Load>>(),
        size_of::<Option<&mut PreparedDiskWriteHostRetirement>>(),
        size_of::<OrdinaryWrittenCacheHostSource>(),
        size_of::<[&ImmutableHostTransferBuffer; 2]>(),
        size_of::<Result<OrdinaryWrittenCacheHostSource, DiskWriteOperationFailure>>(),
        size_of::<Result<PreparedDiskWrite, CacheSourceFailure>>(),
        size_of::<Result<DiskReadBinding, CacheSourceFailure>>(),
        size_of::<Result<PreparedDiskReadSource, CacheSourceFailure>>(),
        size_of::<Result<DiskReadOperation, CacheSourceFailure>>(),
        size_of::<Result<DiskWriteOperation, CacheSourceFailure>>(),
        InstalledManagerCatalog::source_control_bytes::<
            Result<PreparedDiskWrite, CacheSourceFailure>,
        >(size_of::<(
            PreparedDiskWriteDestination,
            &WriteSource<'_, '_>,
        )>())?,
        InstalledManagerCatalog::source_control_bytes::<Result<DiskReadBinding, CacheSourceFailure>>(
            size_of::<&mut PreparedDiskReadDestination>(),
        )?,
        WorkspaceContext::metadata_source_bytes::<DiskReadFinishFailure>()?,
        WorkspaceContext::metadata_source_bytes::<DiskReadOperationFailure>()?.checked_mul(2)?,
        WorkspaceContext::metadata_source_bytes::<DiskWriteOperationFailure>()?.checked_mul(3)?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
