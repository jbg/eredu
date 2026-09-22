//! Actual ordinary Host itinerary, independently paid from the retained trace.
use super::*;
mod disk;
mod runtime;
use crate::backend::runtime::cache::residency::{
    DiskReadOperation, DiskWriteOperation, OrdinaryDiskReadSource, OrdinaryWrittenCacheHostSource,
    PinnedCacheBlock, PinnedCacheBlockLease, PreparedCacheTransferSource,
    PreparedDiskReadDestination, PreparedDiskWriteDestination, PreparedDiskWriteHostRetirement,
    PreparedHostEviction, PreparedHostPromotion, PreparedHostReturn, PreparedInitialDiskReturn,
    PreparedOrdinaryCacheHostDemotion, PreparedOrdinaryCacheHostPromotion,
};
use eredu_core::DomainMemoryRequirements;
use eredu_runtime::working_memory::{MemoryLedger, WorkspaceReportMetadata};
use safemlx::{Array, ImmutableHostTransferBuffer, Stream};

pub(super) struct PreparedHostProgram {
    pub(super) program: HostProgram,
    pub(super) requirements: DomainMemoryRequirements,
}
pub(super) struct HostProgram {
    stores: Vec<Store>,
    loads: Vec<Load>,
    requires_disk: bool,
    disk_workers: Vec<super::super::host_program::disk::Worker>,
    _funding: Option<HostMetadataFunding>,
}
struct Store {
    disk: Option<PreparedDiskWriteDestination>,
    disk_controls: Option<OrdinaryCallControls>,
    write_retirement: Option<PreparedDiskWriteHostRetirement>,
    write: Option<DiskWriteOperation>,
    initial_disk: Option<PreparedInitialDiskReturn>,
    mover: Option<PreparedOrdinaryCacheHostDemotion>,
    initial_host: Option<[std::sync::Arc<ImmutableHostTransferBuffer>; 2]>,
    current_load: Option<usize>,
    pair: super::super::scan_program::ScanSourcePair,
    pin: Option<PinnedCacheBlock>,
    id: CacheBlockId,
    source: usize,
    entry: usize,
    first_ordinal: usize,
}
struct Load {
    disk_read: Option<PreparedDiskReadDestination>,
    disk_controls: Option<OrdinaryCallControls>,
    read_source: Option<OrdinaryDiskReadSource>,
    read_operation: Option<DiskReadOperation>,
    promotion: PreparedOrdinaryCacheHostPromotion,
    read: Option<Read>,
    store: usize,
    ordinal: usize,
    pass: usize,
    consumed: bool,
}
#[derive(Clone, Copy)]
enum Read {
    Initial(usize),
    Promoted(usize),
}
impl HostProgram {
    pub(super) fn prepare(
        sources: &[ProjectedPagedSource],
        rows: &[Option<Row>],
        plan: &InferenceSpanWorkspacePlan,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
    ) -> Result<Option<PreparedHostProgram>, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let mut counts = (0usize, 0usize);
        for (source_index, source) in sources.iter().enumerate() {
            let Some(trace) = source.host_trace() else {
                continue;
            };
            trace
                .with_entries(|entries| -> Result<(), CacheSourceFailure> {
                    for entry in entries {
                        if super::super::host_program::select_append(
                            source_index,
                            entry.first_invocation(),
                            rows,
                            plan,
                            context,
                        )?
                        .is_some()
                        {
                            counts.0 = counts
                                .0
                                .checked_add(1)
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                        }
                    }
                    Ok(())
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
            trace
                .with_loads(|loads| -> Result<(), CacheSourceFailure> {
                    for load in loads {
                        if super::super::host_program::select_append(
                            source_index,
                            (load.query_start, load.context_end),
                            rows,
                            plan,
                            context,
                        )?
                        .is_some()
                        {
                            counts.1 = counts
                                .1
                                .checked_add(1)
                                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
                        }
                    }
                    Ok(())
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
        }
        if counts == (0, 0) {
            return Ok(None);
        }
        let fields = [
            size_of::<Self>(),
            size_of::<PreparedHostProgram>(),
            size_of::<Store>(),
            size_of::<Load>(),
            size_of::<(
                &[ProjectedPagedSource],
                &[Option<Row>],
                &InferenceSpanWorkspacePlan,
                &MemoryLedger,
                &WorkspaceContext,
            )>(),
            size_of::<Result<Option<PreparedHostProgram>, CacheSourceFailure>>(),
            size_of::<super::super::host_program::PagedHostStoreDeclaration<'_>>(),
            size_of::<safemlx::PreparedInputRuntime>(),
            size_of::<(usize, usize, Option<&Row>, u64)>(),
            safemlx::InitializedInputAllocator::borrow_control_bytes()
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        ];
        context
            .charge_metadata(
                fields
                    .into_iter()
                    .try_fold(size_of_val(&fields), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let requires_disk = sources.iter().any(|source| {
            matches!(
                source.manager().options().live_disk_policy(),
                eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }
            ) || source
                .geometry()
                .blocks
                .iter()
                .any(|block| source.retained_file(&block.id).is_some())
        });
        // The shared finite table covers one demotion and two writer tokens per
        // store, or its independent backed-page return. A load can consume one
        // promotion, two read-staging tokens and one later Device return token.
        let reservations = if requires_disk {
            counts
                .0
                .checked_mul(3)
                .and_then(|n| n.checked_add(counts.1.checked_mul(4)?))
        } else {
            counts.0.checked_add(counts.1)
        }
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let allocator = crate::backend::managed_memory::input_allocator::admitted_initializer(pool)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let runtime = allocator.try_borrow_runtime().map_err(|cause| {
            CacheSourceFailure::metadata(context.metadata_source(cause), context)
        })?;
        let report = WorkspaceReportMetadata::new(context);
        let mut requirements = report
            .placed_requirements(pool.topology(), 0, pool.host_placement())
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let mut stores: Vec<Store> = context
            .metadata_vec(counts.0)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let mut loads: Vec<Load> = context
            .metadata_vec(counts.1)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for (source_index, source) in sources.iter().enumerate() {
            let Some(trace) = source.host_trace() else {
                continue;
            };
            trace.with_entries(|entries| -> Result<(), CacheSourceFailure> {
                for (entry_index, entry) in entries.iter().enumerate() {
                    let Some(row) = super::super::host_program::select_append(source_index,
                        entry.first_invocation(), rows, plan, context)? else { continue };
                    let (declaration, first_ordinal) = super::super::host_program::declaration(
                        source, source_index, entry, rows, row, context)?;
                    let mover = if entry.requires_store() && source.retained_file(declaration.id()).is_none() {
                        let mover = source.manager().with_source_loan(source.selection(), context, |loan| {
                            loan.prepare_declared_ordinary_host_demotion(&declaration, &runtime,
                                reservations, context)
                        })?;
                        for (capacity, placement) in mover.capacities() {
                            let placement = crate::backend::managed_memory::placement_fact_handle(placement, pool).map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                            let part = report.placed_requirements(pool.topology(), u64::try_from(capacity)
                                .map_err(|_| fail(CacheSourceError::Overflow))?, &placement).map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                            requirements = report.combine_domain_requirements(&requirements, &part, true).map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                            let part = report.placed_requirements(pool.topology(),
                                crate::backend::managed_memory::ordinary_root_metadata_bytes().map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?, pool.host_placement()).map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                            requirements = report.combine_domain_requirements(&requirements, &part, true).map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                        }
                        Some(mover)
                    } else { None };
                    let initial_host = if !entry.requires_store() && declaration.is_retained() {
                        source.manager().with_source_loan(source.selection(), context, |loan| {
                            declaration.validate_loan(&loan, context).map_err(fail)?;
                            let block = loan.blocks().find(|block| block.id() == declaration.id())
                                .ok_or_else(|| fail(CacheSourceError::Identity))?;
                            Ok(block.host().map(|buffers| buffers.map(std::sync::Arc::clone)))
                        })?
                    } else { None };
                    let disk = if source.retained_file(declaration.id()).is_none()
                        && matches!(source.manager().options().live_disk_policy(),
                            eredu_runtime::LiveCacheDiskPolicy::Enabled { .. }) {
                        Some(source.manager().with_source_loan(source.selection(), context, |loan| {
                            loan.prepare_declared_ordinary_disk_write(&declaration, &runtime,
                                reservations, context)
                        })?)
                    } else { None };
                    let initial_disk = if declaration.is_retained()
                        && source.retained_file(declaration.id()).is_some()
                        && source.geometry().blocks.iter().any(|block|
                            &block.id == declaration.id()
                                && block.phase == eredu_runtime::CacheStoragePhase::Device) {
                        Some(source.manager().with_source_loan(source.selection(), context, |loan|
                            loan.prepare_initial_ordinary_disk_return(&declaration, reservations, context))?)
                    } else { None };
                    let disk_controls = disk.as_ref().map(|writer|
                        disk::write_controls(writer, row.reporting)
                            .ok_or_else(|| fail(CacheSourceError::Overflow))).transpose()?;
                    let write_retirement = disk.as_ref().map(|_|
                        PreparedDiskWriteHostRetirement::prepare(context, row.reporting)).transpose()?;
                    stores.push(Store { disk, disk_controls, write_retirement, write: None, initial_disk,
                        mover, initial_host, current_load: None, pair: super::super::scan_program::ScanSourcePair::prepare(context)?,
                        pin: None, id: declaration.id().clone(), source: source_index, entry: entry_index,
                        first_ordinal });
                }
                Ok(())
            }).map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
            trace
                .with_loads(|entries| -> Result<(), CacheSourceFailure> {
                    for load in entries {
                        let Some(row) = super::super::host_program::select_append(
                            source_index,
                            (load.query_start, load.context_end),
                            rows,
                            plan,
                            context,
                        )?
                        else {
                            continue;
                        };
                        let store = stores
                            .iter()
                            .position(|store| {
                                store.source == source_index && store.entry == load.entry
                            })
                            .ok_or_else(|| {
                                fail(CacheSourceError::HostIdentity("ordinary load store"))
                            })?;
                        if load.pass > 1 || row.ordinal < stores[store].first_ordinal {
                            return Err(fail(CacheSourceError::HostIdentity(
                                "ordinary load order",
                            )));
                        }
                        let (promotion, disk_read) = trace
                            .with_entries(|entries| {
                                let entry = entries
                                    .get(load.entry)
                                    .ok_or_else(|| fail(CacheSourceError::Identity))?;
                                let first = super::super::host_program::select_append(
                                    source_index,
                                    entry.first_invocation(),
                                    rows,
                                    plan,
                                    context,
                                )?
                                .ok_or_else(|| fail(CacheSourceError::Identity))?;
                                let (declaration, _) = super::super::host_program::declaration(
                                    source,
                                    source_index,
                                    entry,
                                    rows,
                                    first,
                                    context,
                                )?;
                                source.manager().with_source_loan(
                                    source.selection(),
                                    context,
                                    |loan| {
                                        let disk_layout = source.retained_file(declaration.id())
                                            .and_then(|file| file.layout())
                                            .or_else(|| stores[store].disk.as_ref()
                                                .map(|writer| writer.read_layout()));
                                        let promotion = loan.prepare_declared_ordinary_host_promotion(
                                            &declaration, &runtime, reservations, disk_layout.is_some(), context)?;
                                        let disk_read = disk_layout.map(|layout| {
                                            loan.prepare_declared_ordinary_disk_read(&declaration,
                                                layout, &runtime, reservations, context)
                                        }).transpose()?;
                                        Ok((promotion, disk_read))
                                    },
                                )
                            })
                            .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
                        let (disk_read, read_source) = match disk_read {
                            Some((read, source)) => {
                                for (capacity, placement) in read.ordinary_capacities()
                                    .ok_or_else(|| fail(CacheSourceError::Identity))? {
                                    let placement = crate::backend::managed_memory::placement_fact_handle(placement, pool)
                                        .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                                    let part = report.placed_requirements(pool.topology(), u64::try_from(capacity)
                                        .map_err(|_| fail(CacheSourceError::Overflow))?, &placement)
                                        .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                                    requirements = report.combine_domain_requirements(&requirements, &part, true)
                                        .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                                    let part = report.placed_requirements(pool.topology(),
                                        crate::backend::managed_memory::ordinary_root_metadata_bytes()
                                            .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?, pool.host_placement())
                                        .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                                    requirements = report.combine_domain_requirements(&requirements, &part, true)
                                        .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?;
                                }
                                (Some(read), Some(source))
                            }
                            None => (None, None),
                        };
                        let disk_controls = disk_read.as_ref().map(|read|
                            disk::read_controls(read, row.reporting)
                                .ok_or_else(|| fail(CacheSourceError::Overflow))).transpose()?;
                        loads.push(Load {
                            disk_read, disk_controls, read_source, read_operation: None,
                            promotion,
                            read: None,
                            store,
                            ordinal: row.ordinal,
                            pass: load.pass,
                            consumed: false,
                        });
                    }
                    Ok(())
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))??;
        }
        if stores.len() != counts.0 || loads.len() != counts.1 {
            return Err(fail(CacheSourceError::Identity));
        }
        let disk_workers = super::super::host_program::disk::prepare_workers(
            sources,
            stores.iter().map(|store| &sources[store.source]),
            loads.iter().map(|load| &sources[stores[load.store].source]),
            context,
        )?;
        Ok(Some(PreparedHostProgram {
            program: Self {
                disk_workers,
                stores,
                loads,
                requires_disk,
                _funding: context.metadata_funding(),
            },
            requirements,
        }))
    }
}

/// Only the actual checked-out append/scan row can mint this lexical source.
/// Its Work, installed catalog and retained projection are compared before use.
struct TransferSource<'a> {
    work: &'a OrdinaryPagedWork,
    source: &'a ProjectedPagedSource,
    installed: &'a InstalledManagerCatalog,
    host: &'a HostPreparationAuthority,
}
impl PreparedCacheTransferSource for TransferSource<'_> {
    type Cause = OrdinaryPagedCause;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        let owner = crate::backend::nn::shared::current_ordinary_execution_owner()?
            .ok_or_else(|| self.error(CacheSourceError::Identity.into()))?;
        if owner.paged().is_none_or(|actual| {
            !Rc::ptr_eq(&actual.program.inner, &self.work.program.inner)
                || actual.ordinals != self.work.ordinals
        }) || !self.source.manager().same_catalog(manager)
            || !self.installed.manager().same_catalog(manager)
            || generation != self.installed.initial_generation()
            || generation != self.source.geometry().generation
        {
            return Err(self.error(CacheSourceError::Identity.into()));
        }
        Ok(())
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        failure(cause, self.host)
    }
}
struct Eviction<'a, 'source> {
    source: &'a TransferSource<'source>,
    id: &'a CacheBlockId,
    arrays: [&'a Array; 2],
    owned_pins: usize,
}
impl PreparedHostEviction for Eviction<'_, '_> {
    type Cause = OrdinaryPagedCause;
    fn id(&self) -> &CacheBlockId {
        self.id
    }
    fn arrays(&self) -> [&Array; 2] {
        self.arrays
    }
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        self.source.validate_manager(manager, generation)
    }
    fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception> {
        if lifecycle
            .is_device_leased(self.id)
            .map_err(|cause| self.error(CacheSourceError::Lifecycle(cause).into()))?
            || lifecycle
                .source_pin_count(self.id)
                .map_err(|cause| self.error(CacheSourceError::Lifecycle(cause).into()))?
                != self.owned_pins
        {
            return Err(self.error(CacheSourceError::Busy.into()));
        }
        Ok(())
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        self.source.error(cause)
    }
}

/// Descriptive source loan minted while one authenticated scan Row is checked
/// out. It borrows the same admitted Work and cannot escape its callback.
pub(crate) struct OrdinaryPagedHostScan<'a> {
    pub(super) work: &'a OrdinaryPagedWork,
    pub(super) source: &'a ProjectedPagedSource,
    pub(super) installed: &'a InstalledManagerCatalog,
    pub(super) ordinal: usize,
    pub(super) host: &'a HostPreparationAuthority,
}
struct Checkout<'a> {
    work: &'a OrdinaryPagedWork,
    value: Option<HostProgram>,
}
impl Drop for Checkout<'_> {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        match self.work.program.inner.host.try_borrow_mut() {
            Ok(mut slot) if slot.is_none() => *slot = Some(value),
            // Conflicting source ownership cannot prove that a native prefix
            // is retired. Keep the complete destination and source custody.
            _ => std::mem::forget((value, self.work.clone())),
        }
    }
}
impl OrdinaryPagedHostScan<'_> {
    pub(crate) fn has_itinerary(&self) -> Result<bool, Exception> {
        let program = self
            .work
            .program
            .inner
            .host
            .try_borrow()
            .map_err(|_| failure(CacheSourceError::Busy, self.host))?;
        Ok(program.as_ref().is_some_and(|program| {
            program.loads.iter().any(|load| {
                let store = &program.stores[load.store];
                load.ordinal == self.ordinal
                    && self.work.program.inner.sources.sources()[store.source]
                        .manager()
                        .same_catalog(self.source.manager())
                    && store.id.global_layer == self.source.geometry().global_layer
            })
        }))
    }
    pub(crate) fn source_control_bytes<R>(callback_bytes: usize) -> Option<usize> {
        let fields = [
            size_of::<Self>(),
            size_of::<Checkout<'_>>(),
            size_of::<TransferSource<'_>>(),
            size_of::<(&Self, &CacheBlockId, usize)>(),
            size_of::<Result<R, Exception>>(),
            size_of::<std::cell::RefMut<'_, Option<HostProgram>>>(),
            size_of::<Option<HostProgram>>(),
            size_of::<Option<&mut Load>>(),
            size_of::<[&Array; 2]>(),
            size_of::<PinnedCacheBlockLease>(),
            size_of::<Result<(usize, PinnedCacheBlockLease), Exception>>(),
            size_of::<(&Stream, &CacheBlockId, usize)>(),
            callback_bytes,
            InstalledManagerCatalog::source_control_bytes::<()>(size_of::<(
                &CacheBlockId,
                [&ImmutableHostTransferBuffer; 2],
            )>())?,
        ];
        fields
            .into_iter()
            .try_fold(size_of_val(&fields), usize::checked_add)
    }
}
