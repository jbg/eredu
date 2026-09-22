//! Exact per-manager disk worker and durable writes for the shared Host itinerary.
use super::super::scan_claim::OriginalPagedScanSource;
use super::*;
use crate::backend::runtime::cache::residency::{
    CacheResidencyManager, DiskReadBinding, DiskReadOperation, DiskWriteOperation,
    InstalledDiskWorker, PreparedDiskReadDestination, PreparedDiskReadSource, PreparedDiskWorker,
};
use safemlx::error::Exception;

pub(in super::super) struct Worker {
    prepared: Option<PreparedDiskWorker>,
    installed: Option<InstalledDiskWorker>,
    source: usize,
}
impl Worker {
    pub(in super::super) fn source_index(&self) -> usize {
        self.source
    }

    pub(in super::super) fn installed(&self) -> Option<&InstalledDiskWorker> {
        self.installed.as_ref()
    }
}
pub(in super::super) fn prepare_workers<'a, S, L>(
    sources: &'a [ProjectedPagedSource],
    stores: S,
    loads: L,
    context: &WorkspaceContext,
) -> Result<Vec<Worker>, CacheSourceFailure>
where
    S: Iterator<Item = &'a ProjectedPagedSource> + Clone,
    L: Iterator<Item = &'a ProjectedPagedSource> + Clone,
{
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let frames = [
        ProjectedPagedSource::retained_file_control_bytes(),
        size_of::<std::slice::Iter<'_, crate::backend::runtime::cache::kv::PagedCacheBlockGeometry>>(
        ),
        size_of::<Worker>(),
        size_of::<Vec<Worker>>(),
        size_of::<(&[ProjectedPagedSource], S, L, &WorkspaceContext)>(),
        size_of::<Result<Vec<Worker>, CacheSourceFailure>>(),
        size_of::<std::slice::Iter<'_, ProjectedPagedSource>>(),
        size_of::<S>(),
        size_of::<L>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<(&CacheResidencyManager, bool, bool, bool)>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let first = |index: usize| {
        let manager = sources[index].manager();
        let writes = matches!(
            manager.options().live_disk_policy(),
            eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }
        ) && stores
            .clone()
            .any(|source| source.manager().same_catalog(manager));
        let reads = loads.clone().any(|source| {
            source.manager().same_catalog(manager)
                && source
                    .geometry()
                    .blocks
                    .iter()
                    .any(|block| source.retained_file(&block.id).is_some())
        });
        (writes || reads)
            && !sources[..index]
                .iter()
                .any(|prior| prior.manager().same_catalog(manager))
    };
    let mut managers = (0..sources.len()).filter(|index| first(*index));
    context
        .charge_metadata(
            std::mem::size_of_val(&first)
                .checked_add(std::mem::size_of_val(&managers))
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        &mut [Worker],
                        &WorkspaceContext,
                        std::slice::IterMut<'_, Worker>,
                    )>())
                })
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let count = managers
        .try_fold(0usize, |count, _| count.checked_add(1))
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    let mut workers = context
        .metadata_vec(count)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    for (index, source) in sources
        .iter()
        .enumerate()
        .filter(|(index, _)| first(*index))
    {
        let writes_enabled = matches!(
            source.manager().options().live_disk_policy(),
            eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }
        );
        let mut writes_iter = stores
            .clone()
            .filter(|store| writes_enabled && store.manager().same_catalog(source.manager()));
        let mut reads_iter = loads
            .clone()
            .filter(|load| load.manager().same_catalog(source.manager()));
        context
            .charge_metadata(
                std::mem::size_of_val(&writes_iter)
                    .checked_add(std::mem::size_of_val(&reads_iter))
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let writes = writes_iter
            .try_fold(0usize, |count, _| count.checked_add(1))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let reads = reads_iter
            .try_fold(0usize, |count, _| count.checked_add(1))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let maximum = writes
            .checked_add(reads)
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let prepared = source
            .manager()
            .with_source_loan(source.selection(), context, |loan| {
                loan.prepare_disk_worker(maximum, context)
            })?;
        workers.push(Worker {
            prepared: Some(prepared),
            installed: None,
            source: index,
        });
    }
    Ok(workers)
}
pub(in super::super) fn install_workers(
    workers: &mut [Worker],
    context: &WorkspaceContext,
) -> Result<(), CacheSourceFailure> {
    for worker in workers {
        let prepared = worker
            .prepared
            .take()
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Identity, context))?;
        // Failed installation retains the actual replacement and old table owners.
        worker.installed = Some(prepared.install().map_err(|cause| {
            CacheSourceFailure::metadata(context.metadata_source(cause), context)
        })?);
    }
    Ok(())
}
impl PreparedPagedHostProgram {
    pub(super) fn read_disk_source(
        &mut self,
        bank: &PagedSourceBank,
        proof: &OriginalPagedScanSource<'_>,
        ordinal: usize,
        load: usize,
    ) -> Result<(), Exception> {
        let store = self.loads[load].store;
        let id = self.stores[store].id.clone();
        let host_capacity = self.loads[load]
            .disk_read
            .as_ref()
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?
            .host_capacity();
        self.make_disk_room(bank, proof, ordinal, Some(&id), host_capacity, false)?;
        let worker = self
            .disk_workers
            .iter()
            .position(|worker| {
                bank.sources[worker.source]
                    .manager()
                    .same_catalog(proof.manager())
            })
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
        let destination = self.loads[load]
            .disk_read
            .take()
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
        let binding = proof
            .manager()
            .with_original_host_source(proof, &id, |loan| {
                destination
                    .bind_source(loan)
                    .map_err(|cause| consumer::failure(proof, cause))
            })?;
        // The native destination and its source pin are outside the manager loan
        // before file opening or any fallible prefix teardown can occur.
        let read = destination
            .bind(binding)
            .map_err(|cause| consumer::failure(proof, cause))?
            .open()
            .map_err(|cause| {
                proof.error(crate::backend::error::Error::Neural(
                    proof.context().metadata_source(cause),
                ))
            })?;
        let operation = self.disk_workers[worker]
            .installed
            .as_ref()
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?
            .prepare_read(read, proof.context())
            .map_err(|cause| consumer::failure(proof, cause))?;
        self.loads[load].read_operation = Some(operation);
        let operation = self.loads[load]
            .read_operation
            .as_mut()
            .expect("retained before I/O");
        operation.submit().map_err(|cause| {
            proof.error(crate::backend::error::Error::Neural(
                proof.context().metadata_source(cause),
            ))
        })?;
        operation.finish().map_err(|cause| {
            proof.error(crate::backend::error::Error::Neural(
                proof.context().metadata_source(cause),
            ))
        })
    }
    /// Complete the actual asynchronous write before depending on its released
    /// canonical Host tier. Its old Host backing and file stay in the finite slot.
    pub(super) fn make_disk_room(
        &mut self,
        bank: &PagedSourceBank,
        proof: &OriginalPagedScanSource<'_>,
        ordinal: usize,
        required: Option<&CacheBlockId>,
        additional_host: u64,
        proactive: bool,
    ) -> Result<(), Exception> {
        loop {
            let Some(id) = proof.manager().original_disk_write_victim(
                proof,
                required,
                additional_host,
                proactive,
            )?
            else {
                return Ok(());
            };
            let index = self
                .stores
                .iter()
                .position(|store| {
                    store.id == id
                        && store.first_ordinal <= ordinal
                        && bank.sources[store.source]
                            .manager()
                            .same_catalog(proof.manager())
                })
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
            if self.stores[index].write.is_some() {
                return Err(proof.error(CacheSourceError::Identity));
            }
            let worker = self
                .disk_workers
                .iter()
                .position(|worker| {
                    bank.sources[worker.source]
                        .manager()
                        .same_catalog(proof.manager())
                })
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
            let selected = &bank.sources[self.stores[index].source];
            let source = consumer::for_source(proof, selected);
            let write = self.prepare_disk_write(bank, &source, ordinal, &id)?;
            let operation = self.disk_workers[worker]
                .installed
                .as_ref()
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?
                .prepare_write(write, proof.context())
                .map_err(|cause| consumer::failure(proof, cause))?;
            self.stores[index].write = Some(operation);
            let operation = self.stores[index]
                .write
                .as_mut()
                .expect("retained before submission");
            operation.submit().map_err(|cause| {
                proof.error(crate::backend::error::Error::Neural(
                    proof.context().metadata_source(cause),
                ))
            })?;
            operation.finish().map_err(|cause| {
                proof.error(crate::backend::error::Error::Neural(
                    proof.context().metadata_source(cause),
                ))
            })?;
            if proactive {
                return Ok(());
            }
        }
    }
}
pub(super) fn consumer_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<(
            &mut PreparedPagedHostProgram,
            &PagedSourceBank,
            &OriginalPagedScanSource<'_>,
            usize,
            Option<&CacheBlockId>,
            u64,
            bool,
        )>(),
        size_of::<(usize, usize, CacheBlockId, u64)>(),
        size_of::<Option<CacheBlockId>>(),
        size_of::<DiskWriteOperation>(),
        size_of::<DiskReadOperation>(),
        size_of::<PreparedDiskReadDestination>(),
        size_of::<DiskReadBinding>(),
        size_of::<Result<DiskReadOperation, CacheSourceFailure>>(),
        size_of::<Result<PreparedDiskReadSource, CacheSourceFailure>>(),
        size_of::<(
            &mut PreparedPagedHostProgram,
            &PagedSourceBank,
            &OriginalPagedScanSource<'_>,
            usize,
            usize,
        )>(),
        CacheResidencyManager::source_loan_control_bytes::<DiskReadBinding>(size_of::<(
            &PreparedDiskReadDestination,
            &OriginalPagedScanSource<'_>,
        )>())?,
        size_of::<Result<DiskWriteOperation, CacheSourceFailure>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<std::slice::Iter<'_, Worker>>(),
        size_of::<std::slice::Iter<'_, StoreSlot>>(),
        CacheResidencyManager::original_host_policy_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
