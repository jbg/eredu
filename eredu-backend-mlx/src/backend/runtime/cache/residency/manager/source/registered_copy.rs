//! Canonical immutable page copies through the existing registered-copy worker.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::{
        IsolatedArrayCopy, PreparedHostArrayCopy, PreparedOriginalCopy, PreparedSavedHostCopy,
        RegisteredArrayCopyCustody, SavedHostCopyPlan,
    },
    error::Error as NativeError,
    nn::workspace::MlxMetalWorkspaceMechanisms,
    runtime::cache::{
        original_copy::{PreparedPredictionCacheCopy, construct},
        state::CompletedResidentSource,
    },
};
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{PrefillRootsRuntime, PreparedArrayClone};

// All native aliases retire before their pins/context; this whole pending value
// lives outside the lexical manager callback, including partial construction.
struct Row {
    arrays: [Option<Array>; 2],
    host: Option<[Arc<ImmutableHostTransferBuffer>; 2]>,
    disk: Option<DiskLocation>,
    id: CacheBlockId,
    metadata: Option<CacheBlockMetadata>,
    protected_prefix: bool,
    imported: bool,
    pin: Option<PinnedCacheBlock>,
}
// Execution-only wrappers retire on their creating thread. Existing recovery
// roots retain every submitted native prefix and its Scope independently.
struct HostWork {
    loads: [Option<PreparedHostArrayCopy>; 2],
    stores: [Option<PreparedSavedHostCopy>; 2],
}
#[derive(Debug, thiserror::Error)]
enum PublicationFailure {
    #[error("copied cache destination identity changed")]
    Identity,
    #[error(transparent)]
    Source(CacheSourceError),
    #[error(transparent)]
    Residency(CacheResidencyError),
}
struct Pending {
    host_work: Vec<HostWork>,
    rows: Vec<Row>,
    clones: Vec<Option<[PreparedArrayClone; 2]>>,
    destination: Option<PreparedIndependentCacheManager>,
    context: WorkspaceContext,
    copy_budget: Option<safemlx::OriginalBufferBudget>,
    copy_custody: Option<eredu_runtime::working_memory::WorkspaceCopyRetention>,
}
impl Pending {
    fn prepare(&mut self, mut source: CacheBlockSourceLoan<'_>) -> Result<(), CacheSourceFailure> {
        let context = &self.context;
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(controls().ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        // The destination remains unreachable as executable state. Its source
        // generation and reservation precede every array alias and native copy.
        self.destination = Some(source.prepare_independent_manager(context)?);
        self.destination
            .as_mut()
            .expect("prepared destination")
            .reserve_copy_occupancy(&source, context)?;
        let count = source.all_blocks().count();
        if source.catalog_population().0 != count {
            return Err(fail(CacheSourceError::Identity));
        }
        self.host_work = context
            .metadata_vec(count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        self.rows = context
            .metadata_vec(count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        self.clones = context
            .metadata_vec(count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for block in source.all_blocks() {
            if self.rows.len() == count {
                return Err(fail(CacheSourceError::Identity));
            }
            let dtypes = source_types(block).map_err(fail)?;
            let metadata = CacheBlockMetadata::prepare_floating(
                block.id().representation,
                block.shapes(),
                dtypes,
                context,
            )?;
            let controls = row_controls().ok_or_else(|| fail(CacheSourceError::Overflow))?;
            context
                .charge_metadata(controls)
                .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
            let clones = if block.device().is_some() {
                Some([
                    PreparedArrayClone::try_prepare_for_inspection()
                        .map_err(|cause| fail(cause.into()))?,
                    PreparedArrayClone::try_prepare_for_inspection()
                        .map_err(|cause| fail(cause.into()))?,
                ])
            } else {
                None
            };
            self.clones.push(clones);
            self.host_work.push(HostWork {
                loads: [None, None],
                stores: [None, None],
            });
            self.rows.push(Row {
                arrays: [None, None],
                host: None,
                disk: block.record.disk().cloned(),
                id: block.id().clone(),
                metadata: Some(metadata),
                protected_prefix: source
                    .lifecycle
                    .is_protected_prefix(block.id())
                    .map_err(|cause| fail(cause.into()))?,
                imported: block.imported(),
                pin: None,
            });
        }
        for (row_index, row) in self.rows.iter_mut().enumerate() {
            let selection = CacheBlockSelection::new(
                row.id.global_layer,
                row.id.representation,
                row.id.start,
                row.id.end,
                0,
            );
            row.pin = Some(
                source
                    .selected(selection)
                    .pin_prepared_block(&row.id, context.metadata_funding())
                    .map_err(fail)?,
            );
            let record = source
                .records
                .get(&row.id)
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            if let Some(host) = record.physical.host_resource() {
                row.host = Some(match host {
                    HostCacheBlock::KeyValue { keys, values } => {
                        [Arc::clone(keys), Arc::clone(values)]
                    }
                    HostCacheBlock::CompressedLatentRotary { latent, rotary_key } => {
                        [Arc::clone(latent), Arc::clone(rotary_key)]
                    }
                });
                continue;
            }
            let Some(device) = record.physical.device_resource() else {
                if row.disk.is_some() && record.physical.phase() == CacheStoragePhase::DiskReady {
                    continue;
                }
                return Err(fail(CacheSourceError::PromotionRequired));
            };
            let arrays = device.arrays();
            for (index, array) in arrays.into_iter().enumerate() {
                // No MLX graph operation or allocation hook under the manager
                // guard. Slot preparation and exact source pin already exist.
                row.arrays[index] = Some(
                    self.clones[row_index]
                        .as_mut()
                        .ok_or_else(|| fail(CacheSourceError::Identity))?[index]
                        .fill_for_inspection(array)
                        .map_err(|cause| fail(cause.into()))?,
                );
            }
        }
        Ok(())
    }
}

impl Pending {
    // Both independent registered copies and the enclosing saved-state copy
    // supply their existing admitted numerical worker. This body only performs
    // exact canonical publication; it creates no completion or allocation grant.
    fn copy_arrays(
        &mut self,
        copy: &mut dyn FnMut(&Array) -> Result<Array, NativeError>,
    ) -> Result<(), NativeError> {
        self.copy_rows(copy, &mut |_, _, _| {
            Err(NativeError::PrefillControl(
                WorkingMemoryError::UnknownBound,
            ))
        })
    }
    fn copy_rows(
        &mut self,
        copy: &mut dyn FnMut(&Array) -> Result<Array, NativeError>,
        host_copy: &mut dyn FnMut(
            &mut Row,
            &mut HostWork,
            &WorkspaceContext,
        ) -> Result<HostCacheBlock, NativeError>,
    ) -> Result<(), NativeError> {
        if self.rows.len() != self.host_work.len() {
            return Err(NativeError::PrefillControl(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        for (row, work) in self.rows.iter_mut().zip(&mut self.host_work) {
            let record = if row.host.is_some() {
                let host = host_copy(row, work, &self.context)?;
                let metadata = row.metadata.take().expect("one copied publication");
                metadata
                    .validate_host(&host)
                    .map_err(|cause| NativeError::Neural(self.context.metadata_source(cause)))?;
                let mut id = row.id.clone();
                id.session_id = self
                    .destination
                    .as_ref()
                    .expect("prepared destination")
                    .destination()
                    .session_id();
                metadata.into_storage_record(
                    MlxCacheBlockStorage::host(id, host, row.disk.clone()),
                    row.imported,
                )
            } else if row.arrays.iter().all(Option::is_none) {
                let backing = row.disk.clone().ok_or_else(|| {
                    NativeError::Neural(self.context.metadata_source(CacheSourceError::Identity))
                })?;
                let mut id = row.id.clone();
                id.session_id = self
                    .destination
                    .as_ref()
                    .expect("prepared destination")
                    .destination()
                    .session_id();
                row.metadata
                    .take()
                    .expect("one copied publication")
                    .into_storage_record(MlxCacheBlockStorage::disk(id, backing), row.imported)
            } else {
                let first = copy(row.arrays[0].as_ref().expect("bound source"))?;
                let second = copy(row.arrays[1].as_ref().expect("bound source"))?;
                let arrays = match row.id.representation {
                    CacheRepresentation::KeyValue => CacheBlockArrays::KeyValue {
                        keys: first,
                        values: second,
                    },
                    CacheRepresentation::CompressedLatentRotary => {
                        CacheBlockArrays::CompressedLatentRotary {
                            latent: first,
                            rotary_key: second,
                        }
                    }
                };
                let metadata = row.metadata.take().expect("one copied publication");
                metadata
                    .validate_arrays(&arrays)
                    .map_err(|cause| NativeError::Neural(self.context.metadata_source(cause)))?;
                let destination = self.destination.as_mut().expect("prepared destination");
                let mut id = row.id.clone();
                id.session_id = destination.destination().session_id();
                metadata.into_storage_record(
                    MlxCacheBlockStorage::device(id, arrays, row.disk.clone()),
                    row.imported,
                )
            };
            let destination = self.destination.as_mut().expect("prepared destination");
            let mut id = row.id.clone();
            id.session_id = destination.destination().session_id();
            let result = {
                // On every rejection, the record survives this guard;
                // native teardown is deferred until after unlock.
                let manager = destination.destination();
                match manager.inner.state.try_lock() {
                    Ok(mut state) => {
                        if state.generation != destination.installed().initial_generation()
                            || !manager.borrowed_storage_complete(&state)
                        {
                            Err((PublicationFailure::Identity, record))
                        } else {
                            match publication::insert_record(
                                &mut state,
                                id,
                                record,
                                row.protected_prefix,
                                true,
                            ) {
                                Ok(()) => {
                                    state.telemetry.report.block_seals += 1;
                                    Ok(())
                                }
                                Err((cause, record)) => {
                                    Err((PublicationFailure::Residency(cause), record))
                                }
                            }
                        }
                    }
                    Err(cause) => {
                        let cause = match cause {
                            TryLockError::WouldBlock => CacheSourceError::Busy,
                            TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                        };
                        Err((PublicationFailure::Source(cause), record))
                    }
                }
            };
            if let Err((cause, record)) = result {
                drop(record);
                return Err(NativeError::Neural(self.context.metadata_source(cause)));
            }
            destination
                .publish_copy_occupancy(&self.context)
                .map_err(|cause| NativeError::Neural(self.context.metadata_source(cause)))?;
        }
        Ok(())
    }
}

/// Paid, source-pinned page inputs and an independent canonical destination.
/// No numerical work has begun. It must remain owned by the enclosing snapshot
/// recovery until all page/tail copies and final publication have settled.
pub(crate) struct PreparedPagedArrayCopy {
    pending: Pending,
}
impl PreparedPagedArrayCopy {
    pub(crate) fn retain_failure(self, cause: NativeError) -> NativeError {
        let Pending {
            host_work,
            rows,
            clones,
            destination,
            context,
            copy_budget,
            copy_custody,
        } = self.pending;
        // Native wrappers have embedded deferred teardown. Submitted outputs
        // and Scope remain in the enclosing recovery; neutral copy custody is
        // retained below after source pins and the partial destination.
        drop(host_work);
        drop(copy_budget);
        drop(clones);
        NativeError::StorageSource(eredu_core::BackendFailure::from_error(PagedCopyFailure {
            cause,
            _rows: rows,
            _destination: destination,
            _copy_custody: copy_custody,
            _funding: context.metadata_funding(),
        }))
    }
    pub(crate) fn failure_control_bytes() -> Option<usize> {
        eredu_core::BackendFailure::source_retention_peak_bytes::<PagedCopyFailure>()
            .and_then(|n| n.checked_add(size_of::<PagedCopyFailure>()))
    }
    pub(crate) fn destination(&mut self) -> &mut PreparedIndependentCacheManager {
        self.pending
            .destination
            .as_mut()
            .expect("one prepared paged copy")
    }
    pub(crate) fn copy_retained(
        &mut self,
        stream: &safemlx::Stream,
        roots: &std::cell::RefCell<Vec<Array>>,
    ) -> Result<(), NativeError> {
        self.copy_retained_with(stream, roots, &mut |_| Ok(()))
    }
    pub(crate) fn copy_retained_with(
        &mut self,
        stream: &safemlx::Stream,
        roots: &std::cell::RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), NativeError>,
    ) -> Result<(), NativeError> {
        self.copy_retained_with_host(stream, roots, observe, &mut |_| Ok(()))
    }
    pub(crate) fn finish(
        &mut self,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<(), NativeError> {
        let destination = self
            .pending
            .destination
            .as_mut()
            .expect("one prepared paged copy");
        destination
            .finish_copy(host, &self.pending.context)
            .map_err(|cause| NativeError::Neural(self.pending.context.metadata_source(cause)))?;
        Ok(())
    }
}

#[derive(thiserror::Error)]
#[error("{cause}")]
struct PagedCopyFailure {
    #[source]
    cause: NativeError,
    _rows: Vec<Row>,
    _destination: Option<PreparedIndependentCacheManager>,
    _copy_custody: Option<eredu_runtime::working_memory::WorkspaceCopyRetention>,
    _funding: Option<HostMetadataFunding>,
}
impl std::fmt::Debug for PagedCopyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedCopyFailure")
            .field("cause", &self.cause)
            .field("source_rows", &self._rows.len())
            .finish_non_exhaustive()
    }
}

impl CacheResidencyManager {
    pub(crate) fn prepare_paged_array_copy(
        &self,
        layout: &PagedArrayCopyLayout,
        context: &WorkspaceContext,
    ) -> Result<PreparedPagedArrayCopy, NativeError> {
        if context.metadata_funding().is_none() {
            return Err(NativeError::PrefillControl(
                WorkingMemoryError::UnknownBound,
            ));
        }
        let mut pending = Pending {
            host_work: Vec::new(),
            rows: Vec::new(),
            clones: Vec::new(),
            destination: None,
            context: context.clone(),
            copy_budget: None,
            copy_custody: None,
        };
        self.with_source_loan(
            CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
            context,
            |source| {
                if source
                    .paged_copy_layout()
                    .map_err(|cause| CacheSourceFailure::source(cause, context))?
                    != *layout
                {
                    return Err(CacheSourceFailure::source(
                        CacheSourceError::Identity,
                        context,
                    ));
                }
                pending.prepare(source)
            },
        )
        .map_err(|cause| NativeError::Neural(context.metadata_source(cause)))?;
        Ok(PreparedPagedArrayCopy { pending })
    }

    /// Copies actual sealed device pages into an independent prepared manager.
    /// Every array uses the same registered/completed source binder and original
    /// numerical copy admission as resident cache leaves. Tail occupancy remains
    /// reserved for the enclosing exact layer-state copy/assembly; this value is
    /// deliberately not a runnable paged state or an execution/source grant.
    pub(crate) fn copy_completed_paged_manager(
        &self,
        completed: Option<&CompletedResidentSource>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        capacity: u64,
    ) -> Result<PreparedPredictionCacheCopy<PreparedIndependentCacheManager>, NativeError> {
        self.copy_completed_paged_with(
            completed,
            environment,
            initialized,
            mechanisms,
            context,
            capacity,
            0,
            |destination, _| Ok(destination),
        )
    }

    pub(in crate::backend::runtime::cache) fn copy_completed_paged_with<C, F>(
        &self,
        completed: Option<&CompletedResidentSource>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        capacity: u64,
        tail_operands: usize,
        finish: F,
    ) -> Result<PreparedPredictionCacheCopy<C>, NativeError>
    where
        F: FnOnce(
            PreparedIndependentCacheManager,
            &mut crate::backend::runtime::cache::original_copy::Worker<'_, '_>,
        ) -> Result<C, NativeError>,
    {
        let funding = context
            .metadata_funding()
            .ok_or_else(|| NativeError::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let mut pending = Pending {
            host_work: Vec::new(),
            rows: Vec::new(),
            clones: Vec::new(),
            destination: None,
            context: context.clone(),
            copy_budget: None,
            copy_custody: None,
        };
        // Selection controls only the descriptive view. The closed worker
        // enumerates the actual whole catalog and obtains each exact row pin.
        self.with_source_loan(
            CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
            context,
            |source| pending.prepare(source),
        )
        .map_err(|cause| NativeError::Neural(context.metadata_source(cause)))?;
        construct(
            completed,
            environment,
            initialized,
            mechanisms,
            &funding,
            capacity,
            controls(),
            move |worker| {
                let count = pending
                    .rows
                    .iter()
                    .filter(|row| row.arrays[0].is_some() || row.host.is_some())
                    .count()
                    .checked_mul(2)
                    .and_then(|count| count.checked_add(tail_operands))
                    .ok_or_else(|| NativeError::PrefillControl(WorkingMemoryError::Overflow))?;
                worker.slots(count)?;
                pending.copy_arrays(&mut |array| worker.copy(IsolatedArrayCopy::new(array)))?;
                finish(
                    pending
                        .destination
                        .take()
                        .expect("completed page copy destination"),
                    worker,
                )
            },
        )
    }
}
fn source_types(block: CacheBlockSource<'_>) -> Result<[Dtype; 2], CacheSourceError> {
    let backing = block.retained_disk_types()?;
    let types = if let Some(arrays) = block.device() {
        [arrays[0].dtype(), arrays[1].dtype()]
    } else if let Some(host) = block.host() {
        let a = host[0]
            .try_fixed_descriptor::<4>()
            .map_err(CacheSourceError::HostDescriptor)?;
        let b = host[1]
            .try_fixed_descriptor::<4>()
            .map_err(CacheSourceError::HostDescriptor)?;
        if [a.shape(), b.shape()] != block.shapes() {
            return Err(CacheSourceError::Geometry);
        }
        [a.dtype(), b.dtype()]
    } else {
        backing.ok_or(CacheSourceError::PromotionRequired)?
    };
    if backing.is_some_and(|stored| stored != types)
        || CacheBlockMetadata::floating_bytes(block.shapes(), types)? != block.logical_bytes()
    {
        return Err(CacheSourceError::Geometry);
    }
    Ok(types)
}
fn controls() -> Option<usize> {
    let frames = [
        size_of::<Option<DiskLocation>>(),
        size_of::<MlxCacheBlockStorage>(),
        size_of::<Option<[PreparedArrayClone; 2]>>(),
        CacheBlockSource::retained_disk_control_bytes(),
        size_of::<Pending>(),
        size_of::<HostWork>(),
        size_of::<Vec<HostWork>>(),
        size_of::<Option<eredu_runtime::working_memory::WorkspaceCopyRetention>>(),
        size_of::<std::iter::Zip<std::slice::IterMut<'_, Row>, std::slice::IterMut<'_, HostWork>>>(), size_of::<Row>(), size_of::<Vec<Row>>(),
        size_of::<Option<safemlx::OriginalBufferBudget>>(),
        size_of::<&mut dyn FnMut(&mut Row, &mut HostWork, &WorkspaceContext) -> Result<HostCacheBlock, NativeError>>(),
        size_of::<Vec<Option<[PreparedArrayClone; 2]>>>(),
        size_of::<PreparedPagedArrayCopy>(),
        size_of::<(&mut Pending, &mut dyn FnMut(&Array) -> Result<Array, NativeError>)>(),
        size_of::<(&safemlx::Stream, &std::cell::RefCell<Vec<Array>>)>(),
        size_of::<&mut dyn FnMut(&Array) -> Result<(), NativeError>>(),
        size_of::<Result<PreparedPagedArrayCopy, NativeError>>(),
        size_of::<Result<CacheResidencyManager, NativeError>>(),
        size_of::<Option<PreparedIndependentCacheManager>>(),
        size_of::<(&mut Pending, CacheBlockSourceLoan<'_>)>(),
        size_of::<[Option<Array>; 2]>(), size_of::<[PreparedArrayClone; 2]>(),
        size_of::<[&Array; 2]>(), size_of::<CacheBlockMetadata>(), size_of::<CacheBlockRecord>(),
        size_of::<CacheBlockArrays>(), size_of::<CacheBlockId>(), size_of::<CacheBlockSelection>(),
        size_of::<Option<CacheBlockMetadata>>(), size_of::<Result<(), CacheSourceFailure>>(),
        size_of::<Result<(), NativeError>>(), size_of::<Result<(), (PublicationFailure, CacheBlockRecord)>>(),
        size_of::<PublicationFailure>(),
        size_of::<Result<(), (CacheResidencyError, CacheBlockRecord)>>(),
        size_of::<(Array, RegisteredArrayCopyCustody)>(),
        size_of::<Option<usize>>(), size_of::<usize>(),
        size_of::<MutexGuard<'_, CacheManagerState>>(),
        size_of::<Result<MutexGuard<'_, CacheManagerState>, TryLockError<MutexGuard<'_, CacheManagerState>>>>(),
        size_of::<std::slice::IterMut<'_, Row>>(),
        size_of::<std::array::IntoIter<&Array, 2>>(),
        size_of::<std::iter::Enumerate<std::array::IntoIter<&Array, 2>>>(),
        size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(),
        CacheBlockMetadata::fixed_controls()?,
        eredu_runtime::cache::CacheRecordTable::<CacheBlockId, CacheBlockRecord>::mutation_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

fn row_controls() -> Option<usize> {
    PreparedArrayClone::control_bytes()
        .and_then(|n| n.checked_add(CacheBlockSource::retained_disk_control_bytes()))
        .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes()))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(PinnedCacheBlock::fixed_controls()?))
        .and_then(|n| {
            n.checked_add(safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?)
        })
        .and_then(|n| n.checked_add(Array::descriptor_control_bytes()?.checked_mul(2)?))
        .and_then(|n| n.checked_add(CacheBlockLifecycle::prepared_mutation_control_bytes()?))
}
#[path = "registered_copy/layout.rs"]
mod layout;
pub(crate) use layout::PagedArrayCopyLayout;

#[path = "registered_copy/host.rs"]
mod host;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
