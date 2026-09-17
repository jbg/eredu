//! Finite header/file/output slots before their actual Host source is ready.
use super::*;
use crate::backend::nn::workspace::{OriginalPagedDiskWriteSource, PagedHostStoreDeclaration};
use eredu_runtime::cache::PreparedCachePoolReservation;
use safemlx::{PreparedHostTransferPlan, PreparedInputRuntime};

/// This owns only actual prepared destinations and immutable declarations.
/// A task exists only after exact canonical Host buffers and source pins bind.
pub(crate) struct PreparedDiskWriteDestination {
    task: PreparedCacheIoTaskSlot<DiskTask, DiskResult>,
    publication: PreparedLiveCachePublication,
    layout: CacheShardLayout,
    location: DiskLocation,
    output: PreparedDiskWriteOutput,
    disk: PreparedCachePoolReservation,
    transfer: PreparedCachePoolReservation,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    bytes: [usize; 2],
    capacities: [usize; 2],
    manager: CacheResidencyManager,
    worker: Arc<DiskWorker>,
    id: CacheBlockId,
    generation: u64,
    context: WorkspaceContext,
    funding: WorkspaceMetadataFunding,
}
impl CacheBlockSourceLoan<'_> {
    /// Uses the exact selected retained/future page declaration. Header layout,
    /// file names, output shell and pool-table slots precede actual publication.
    pub(crate) fn prepare_declared_disk_write(
        &self,
        source: &PagedHostStoreDeclaration<'_>,
        runtime: &PreparedInputRuntime,
        additional_reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedDiskWriteDestination, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        source.validate_loan(self, context).map_err(fail)?;
        context
            .charge_metadata(
                PreparedDiskWrite::fixed_control_bytes()
                    .and_then(|n| {
                        n.checked_add(size_of::<(
                            &Self,
                            &PagedHostStoreDeclaration<'_>,
                            &PreparedInputRuntime,
                            usize,
                            &WorkspaceContext,
                        )>())
                    })
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let shapes = source.shapes();
        let dtypes = source.dtypes();
        let mut bytes = [0; 2];
        let mut capacities = [0; 2];
        for index in 0..2 {
            if source.requires_store() {
                let plan = PreparedHostTransferPlan::new(runtime, &shapes[index], dtypes[index], 0)
                    .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
                context
                    .charge_metadata(
                        plan.control_bytes()
                            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
                    )
                    .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
                bytes[index] = plan.logical_bytes();
                capacities[index] = plan.backing_bytes();
            } else {
                let row = self
                    .blocks()
                    .find(|row| row.id() == source.id())
                    .ok_or_else(|| fail(CacheSourceError::Identity))?;
                let buffers = row.host().ok_or_else(|| fail(CacheSourceError::Identity))?;
                let descriptor = buffers[index]
                    .try_fixed_descriptor::<4>()
                    .map_err(|cause| fail(cause.into()))?;
                if descriptor.shape() != shapes[index] || descriptor.dtype() != dtypes[index] {
                    return Err(fail(CacheSourceError::Geometry));
                }
                bytes[index] = descriptor.nbytes();
                capacities[index] = descriptor.allocation().bytes();
            }
        }
        prepare(
            self,
            source.id(),
            shapes,
            dtypes,
            bytes,
            capacities,
            additional_reservations,
            context,
        )
    }
}
pub(super) fn prepare(
    loan: &CacheBlockSourceLoan<'_>,
    id: &CacheBlockId,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    bytes: [usize; 2],
    capacities: [usize; 2],
    additional_reservations: usize,
    context: &WorkspaceContext,
) -> Result<PreparedDiskWriteDestination, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let funding = context
        .metadata_funding()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    context
        .charge_metadata(control_bytes().ok_or_else(|| fail(CacheSourceError::Overflow))?)
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let directory = match loan.manager.inner.options.live_disk_policy() {
        LiveCacheDiskPolicy::Enabled { directory, .. } => directory,
        LiveCacheDiskPolicy::Disabled => return Err(fail(CacheSourceError::PromotionRequired)),
    };
    let worker = loan
        .manager
        .inner
        .disk_worker
        .as_ref()
        .ok_or_else(|| fail(CacheSourceError::Identity))?
        .clone();
    let mut unsigned = [[0usize; 4]; 2];
    for index in 0..2 {
        for (out, input) in unsigned[index].iter_mut().zip(shapes[index]) {
            *out = usize::try_from(input).map_err(|_| fail(CacheSourceError::Geometry))?;
        }
    }
    let metadata = CacheShardMetadata::prepare(
        id.representation,
        [&unsigned[0], &unsigned[1]],
        dtypes.map(host_scalar_to_stored),
        bytes,
        context,
    )
    .map_err(|cause| fail(CacheSourceError::Shard(cause)))?;
    let layout = metadata
        .into_prepared_layout(context)
        .map_err(|cause| fail(CacheSourceError::Shard(cause)))?;
    let publication = PreparedLiveCachePublication::prepare(directory, id, context)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    let location = DiskLocation::prepare_live(&publication, id.representation, context)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    let inner = context
        .metadata_arc(OnceLock::new())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let disk = loan
        .pool()
        .prepare_reservation_population(additional_reservations, context)
        .map_err(|cause| fail(CacheSourceError::DiskReservation(cause)))?;
    let transfer = loan
        .pool()
        .prepare_reservation_population(additional_reservations, context)
        .map_err(|cause| fail(CacheSourceError::DiskReservation(cause)))?;
    context
        .charge_metadata(
            DiskWriteOperation::control_bytes()
                .and_then(|n| n.checked_add(loan.publication_controls.checked_mul(6)?))
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let task = worker
        .inner
        .prepare_task_slot(
            CacheIoOperationKey {
                generation: loan.generation(),
                id: id.clone(),
                kind: CacheIoOperationKind::Write,
            },
            context,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    Ok(PreparedDiskWriteDestination {
        task,
        publication,
        layout,
        location,
        output: PreparedDiskWriteOutput {
            inner,
            funding: funding.clone(),
        },
        disk,
        transfer,
        shapes,
        dtypes,
        bytes,
        capacities,
        manager: loan.manager.clone(),
        worker,
        id: id.clone(),
        generation: loan.generation(),
        context: context.clone(),
        funding,
    })
}
impl PreparedDiskWriteDestination {
    pub(crate) fn read_layout(&self) -> &CacheShardLayout {
        &self.layout
    }

    pub(crate) fn bind(
        self,
        loan: &mut CacheBlockSourceLoan<'_>,
        source: &OriginalPagedDiskWriteSource<'_, '_>,
    ) -> Result<PreparedDiskWrite, CacheSourceFailure> {
        if source.id() != &self.id || !self.context.shares_trace(source.source().context()) {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                &self.context,
            ));
        }
        let pins = source.validate(loan).map_err(|cause| {
            CacheSourceFailure::metadata(self.context.metadata_source(cause), &self.context)
        })?;
        self.bind_with_pins(loan, pins)
    }
    pub(super) fn bind_with_pins(
        self,
        loan: &mut CacheBlockSourceLoan<'_>,
        source_pins: usize,
    ) -> Result<PreparedDiskWrite, CacheSourceFailure> {
        let context = &self.context;
        let fail = |cause| CacheSourceFailure::source(cause, context);
        if !loan.manager().same_catalog(&self.manager)
            || loan.generation() != self.generation
            || loan
                .manager
                .inner
                .disk_worker
                .as_ref()
                .is_none_or(|worker| !Arc::ptr_eq(worker, &self.worker))
            || loan
                .lifecycle
                .lease_count(&self.id)
                .map_err(|cause| fail(cause.into()))?
                != source_pins
            || loan
                .lifecycle
                .source_pin_count(&self.id)
                .map_err(|cause| fail(cause.into()))?
                != source_pins
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let row = loan
            .blocks()
            .find(|row| row.id() == &self.id)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if row.phase() != CacheStoragePhase::HostUnbacked || row.disk().is_some() {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let buffers = row.host().ok_or_else(|| fail(CacheSourceError::Identity))?;
        let descriptors = [
            buffers[0]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| fail(cause.into()))?,
            buffers[1]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| fail(cause.into()))?,
        ];
        let mut transfer_bytes = 0u64;
        let mut logical_bytes = 0u64;
        for index in 0..2 {
            let descriptor = &descriptors[index];
            if descriptor.shape() != self.shapes[index]
                || descriptor.shape() != row.shapes()[index]
                || descriptor.dtype() != self.dtypes[index]
                || !host_promotion::dtype_matches(descriptor.dtype(), row.dtypes()[index])
                || descriptor.nbytes() != self.bytes[index]
                || descriptor.allocation().bytes() != self.capacities[index]
            {
                return Err(fail(CacheSourceError::Geometry));
            }
            logical_bytes = logical_bytes
                .checked_add(
                    u64::try_from(descriptor.nbytes())
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
            transfer_bytes = transfer_bytes
                .checked_add(
                    u64::try_from(descriptor.allocation().bytes())
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        if logical_bytes != row.logical_bytes() {
            return Err(fail(CacheSourceError::Geometry));
        }
        let source_pins = source_pins
            .checked_add(1)
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let disk = self
            .disk
            .reserve(CachePoolUsage {
                disk_bytes: u64::try_from(self.layout.file_bytes())
                    .map_err(|_| fail(CacheSourceError::Overflow))?,
                ..CachePoolUsage::default()
            })
            .map_err(|cause| fail(CacheSourceError::DiskAdmission(cause)))?;
        let transfer = self
            .transfer
            .reserve(CachePoolUsage {
                transfer_in_flight_bytes: transfer_bytes,
                ..CachePoolUsage::default()
            })
            .map_err(|cause| fail(CacheSourceError::DiskAdmission(cause)))?;
        let transfer = DiskWriteOccupancy {
            inner: context
                .metadata_arc(Occupancy {
                    reservation: Mutex::new(transfer),
                    host_bytes: transfer_bytes,
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?,
            funding: self.funding.clone(),
        };
        let host = row
            .record
            .host_block()
            .expect("validated Host source")
            .clone();
        // Every fallible destination/source step precedes this final pin.
        let pin = loan
            .pin_prepared_block(&self.id, Some(self.funding.clone()))
            .map_err(fail)?;
        Ok(PreparedDiskWrite {
            task: Some(self.task),
            body: WriteBody {
                file: None,
                failed_source: None,
                manager: self.manager,
                publication: Some(self.publication),
                layout: self.layout,
                location: self.location,
                host,
                descriptors,
                id: self.id,
                generation: self.generation,
                source_pins,
                pin,
                transfer,
                disk: Some(disk),
                funding: self.funding,
            },
            output: self.output,
            worker: self.worker,
        })
    }
}
fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<PreparedDiskWriteDestination>(),
        size_of::<Result<PreparedDiskWriteDestination, CacheSourceFailure>>(),
        size_of::<Result<PreparedDiskWrite, CacheSourceFailure>>(),
        size_of::<[PreparedHostTransferPlan; 2]>(),
        size_of::<[[usize; 4]; 2]>(),
        size_of::<[[i32; 4]; 2]>(),
        size_of::<[usize; 2]>(),
        size_of::<[Dtype; 2]>(),
        size_of::<[HostTransferDescriptor<4>; 2]>(),
        size_of::<(
            &CacheBlockSourceLoan<'_>,
            &CacheBlockId,
            [[i32; 4]; 2],
            [Dtype; 2],
            [usize; 2],
            [usize; 2],
            usize,
            &WorkspaceContext,
        )>(),
        size_of::<(
            PreparedDiskWriteDestination,
            &mut CacheBlockSourceLoan<'_>,
            usize,
        )>(),
        size_of::<(&OriginalPagedDiskWriteSource<'_, '_>, usize, u64, u64)>(),
        HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
