//! Finite real read destinations before the selected writer publishes its file.
use super::*;
use crate::backend::nn::workspace::PagedHostStoreDeclaration;
use eredu_runtime::cache::PreparedCacheIoTaskSlot;

pub(crate) struct PreparedDiskReadDestination {
    filling: [Option<filled_host::Pending>; 2],
    bytes: Vec<u8>,
    output: PreparedDiskReadOutput,
    task: PreparedCacheIoTaskSlot<DiskTask, DiskResult>,
    operation: operation::CompletionSlots,
    layout: CacheShardLayout,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    capacities: [usize; 2],
    source_bytes: u64,
    source_custody: Option<OriginalHostSourceCustody>,
    reservation: DiskReadOccupancy,
    transfer: CachePoolReservation,
    manager: CacheResidencyManager,
    worker: Arc<DiskWorker>,
    id: CacheBlockId,
    generation: u64,
    context: WorkspaceContext,
    funding: HostMetadataFunding,
}
pub(crate) struct DiskReadBinding {
    output: PreparedDiskReadOutput,
    location: DiskLocation,
    source: LiveCacheBlockSource,
    pin: PinnedCacheBlock,
}
impl CacheBlockSourceLoan<'_> {
    pub(crate) fn prepare_declared_disk_read(
        &self,
        declaration: &PagedHostStoreDeclaration<'_>,
        layout: &CacheShardLayout,
        runtime: &PreparedInputRuntime,
        reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedDiskReadDestination, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        let destination = prepare(
            self,
            declaration.id(),
            layout,
            runtime,
            reservations,
            context,
        )?;
        if destination.shapes != declaration.shapes() || destination.dtypes != declaration.dtypes()
        {
            return Err(fail(CacheSourceError::Geometry));
        }
        Ok(destination)
    }
}
pub(super) fn prepare(
    loan: &CacheBlockSourceLoan<'_>,
    id: &CacheBlockId,
    layout: &CacheShardLayout,
    runtime: &PreparedInputRuntime,
    reservations: usize,
    context: &WorkspaceContext,
) -> Result<PreparedDiskReadDestination, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let funding = context
        .metadata_funding()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    context
        .charge_metadata(
            PreparedDiskRead::control_bytes()
                .and_then(|n| n.checked_add(control_bytes()?))
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let metadata = layout.tensor_metadata();
    let mut shapes = [[0i32; 4]; 2];
    let mut dtypes = [Dtype::Float32; 2];
    let mut capacities = [0usize; 2];
    let mut source_bytes = 0u64;
    let mut host_bytes = 0u64;
    for index in 0..2 {
        let (shape, stored, bytes) = metadata[index];
        if shape.len() != 4 {
            return Err(fail(CacheSourceError::Geometry));
        }
        for (out, value) in shapes[index].iter_mut().zip(shape) {
            *out = i32::try_from(*value).map_err(|_| fail(CacheSourceError::Geometry))?;
        }
        dtypes[index] = match stored {
            StoredDtype::F32 => Dtype::Float32,
            StoredDtype::F16 => Dtype::Float16,
            StoredDtype::BF16 => Dtype::Bfloat16,
            _ => return Err(fail(CacheSourceError::Geometry)),
        };
        let plan = PreparedHostTransferPlan::new(runtime, &shapes[index], dtypes[index], 0)
            .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
        if plan.logical_bytes() != bytes {
            return Err(fail(CacheSourceError::Geometry));
        }
        context
            .charge_metadata(
                plan.control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        capacities[index] = plan.backing_bytes();
        source_bytes = source_bytes
            .checked_add(source_component_bytes(&plan).map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?)
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        host_bytes = host_bytes
            .checked_add(
                u64::try_from(plan.backing_bytes())
                    .map_err(|_| fail(CacheSourceError::Overflow))?,
            )
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    }
    if host_bytes > loan.manager.options().host_budget_bytes() {
        return Err(fail(CacheSourceError::PromotionRequired));
    }
    let worker = loan
        .manager
        .inner
        .disk_worker
        .as_ref()
        .ok_or_else(|| fail(CacheSourceError::Identity))?
        .clone();
    let mut bytes = context
        .metadata_vec::<u8>(layout.file_bytes())
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    bytes.resize(layout.file_bytes(), 0);
    let output = PreparedDiskReadOutput {
        inner: context
            .metadata_arc(OnceLock::new())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?,
        funding: funding.clone(),
    };
    let reservation = loan
        .pool()
        .prepare_reservation_population(reservations, context)
        .map_err(|cause| fail(CacheSourceError::DiskReservation(cause)))?
        .reserve(CachePoolUsage {
            host_bytes,
            ..CachePoolUsage::default()
        })
        .map_err(|cause| fail(CacheSourceError::DiskAdmission(cause)))?;
    let transfer = loan
        .pool()
        .prepare_reservation_population(reservations, context)
        .map_err(|cause| fail(CacheSourceError::DiskReservation(cause)))?
        .reserve(CachePoolUsage {
            transfer_in_flight_bytes: host_bytes,
            ..CachePoolUsage::default()
        })
        .map_err(|cause| fail(CacheSourceError::DiskAdmission(cause)))?;
    let reservation = DiskReadOccupancy {
        inner: context
            .metadata_arc(Mutex::new(reservation))
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?,
        host_bytes,
        funding: funding.clone(),
    };
    let task = worker
        .inner
        .prepare_task_slot(
            CacheIoOperationKey {
                generation: loan.generation,
                id: id.clone(),
                kind: CacheIoOperationKind::Read,
            },
            context,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    let operation = operation::CompletionSlots::prepare(context, loan.publication_controls)?;
    Ok(PreparedDiskReadDestination {
        filling: [None, None],
        bytes,
        output,
        task,
        operation,
        layout: layout.clone(),
        shapes,
        dtypes,
        capacities,
        source_bytes,
        source_custody: None,
        reservation,
        transfer,
        manager: loan.manager.clone(),
        worker,
        id: id.clone(),
        generation: loan.generation,
        context: context.clone(),
        funding,
    })
}
impl PreparedDiskReadDestination {
    pub(crate) fn source_facts(&self) -> Result<HostSourceConstructionFacts, WorkingMemoryError> {
        HostSourceConstructionFacts::new(self.source_bytes, 2, 0)
    }
    pub(crate) fn host_capacity(&self) -> u64 {
        self.reservation.host_bytes
    }
    pub(crate) fn construct(
        &mut self,
        runtime: &PreparedInputRuntime,
        bank: &mut OriginalHostSourceBank,
        custody: &OriginalHostSourceCustody,
    ) -> Result<(), CacheSourceFailure> {
        begin_filling(
            Initialization {
                filling: &mut self.filling,
                source_custody: &mut self.source_custody,
                shapes: &self.shapes,
                dtypes: self.dtypes,
                capacities: self.capacities,
                reservation: &self.reservation,
            },
            runtime,
            bank,
            custody,
        )
        .map_err(|cause| {
            CacheSourceFailure::metadata(self.context.metadata_source(cause), &self.context)
        })
    }
    /// Immutable comparison under the one manager loan; native destination
    /// owners remain outside that loan. All fallible work precedes the final pin.
    pub(crate) fn bind_source(
        &self,
        loan: &mut CacheBlockSourceLoan<'_>,
    ) -> Result<DiskReadBinding, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, &self.context);
        if !loan.manager().same_catalog(&self.manager) || loan.generation() != self.generation {
            return Err(fail(CacheSourceError::Identity));
        }
        let row = loan
            .blocks()
            .find(|row| row.id() == &self.id)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let disk = row.disk().ok_or_else(|| fail(CacheSourceError::Identity))?;
        let file = disk
            .live_file()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        if row.phase() != CacheStoragePhase::DiskReady
            || row.host().is_some()
            || disk.persistent()
            || disk.buffered().is_some()
            || disk.names() != self.layout.names()
            || disk.path() != file.path()
            || file.file_bytes() != Some(self.layout.file_bytes())
            || file
                .writer_layout()
                .is_none_or(|layout| !layout.same_layout(&self.layout))
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let mut bytes = 0u64;
        for index in 0..2 {
            if self.shapes[index] != row.shapes()[index]
                || !host_promotion::dtype_matches(self.dtypes[index], row.dtypes()[index])
            {
                return Err(fail(CacheSourceError::Geometry));
            }
            bytes = bytes
                .checked_add(
                    u64::try_from(self.layout.tensor_metadata()[index].2)
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        if bytes != row.logical_bytes() {
            return Err(fail(CacheSourceError::Geometry));
        }
        if loan
            .telemetry
            .report
            .current_host_bytes
            .checked_add(self.reservation.host_bytes)
            .is_none_or(|n| n > loan.manager.options().host_budget_bytes())
        {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let location = disk.location.clone();
        let source = file.clone();
        let pin = loan
            .pin_prepared_block(&self.id, Some(self.funding.clone()))
            .map_err(fail)?;
        Ok(DiskReadBinding {
            output: self.output.clone(),
            location,
            source,
            pin,
        })
    }
    pub(crate) fn bind(
        self,
        binding: DiskReadBinding,
    ) -> Result<PreparedDiskReadSource, CacheSourceFailure> {
        if !Arc::ptr_eq(&self.output.inner, &binding.output.inner) {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                &self.context,
            ));
        }
        Ok(PreparedDiskReadSource {
            task: PreparedDiskRead {
                task: Some(self.task),
                operation: Some(self.operation),
                body: ReadBody {
                    filling: self.filling,
                    ready: [None, None],
                    host: None,
                    bytes: self.bytes,
                    read: None,
                    location: binding.location,
                    layout: self.layout,
                    source: binding.source,
                    shapes: self.shapes,
                    dtypes: self.dtypes,
                    capacities: self.capacities,
                    source_bytes: self.source_bytes,
                    source_custody: self.source_custody,
                    manager: self.manager,
                    id: self.id,
                    generation: self.generation,
                    pin: binding.pin,
                    reservation: self.reservation,
                    transfer: self.transfer,
                    funding: self.funding,
                },
                output: self.output,
                worker: self.worker,
            },
            context: self.context,
        })
    }
}
pub(super) struct Initialization<'a> {
    pub(super) filling: &'a mut [Option<filled_host::Pending>; 2],
    pub(super) source_custody: &'a mut Option<OriginalHostSourceCustody>,
    pub(super) shapes: &'a [[i32; 4]; 2],
    pub(super) dtypes: [Dtype; 2],
    pub(super) capacities: [usize; 2],
    pub(super) reservation: &'a DiskReadOccupancy,
}
pub(super) fn begin_filling(
    init: Initialization<'_>,
    runtime: &PreparedInputRuntime,
    bank: &mut OriginalHostSourceBank,
    custody: &OriginalHostSourceCustody,
) -> Result<(), output::FinishCause> {
    if init.source_custody.is_some() || init.filling.iter().any(Option::is_some) {
        return Err(output::FinishCause::Identity);
    }
    *init.source_custody = Some(custody.clone());
    for index in 0..2 {
        let plan =
            PreparedHostTransferPlan::new(runtime, &init.shapes[index], init.dtypes[index], 0)
                .map_err(output::FinishCause::Input)?;
        if plan.backing_bytes() != init.capacities[index] {
            return Err(output::FinishCause::Identity);
        }
        init.filling[index] = Some(
            filled_host::begin(
                bank,
                plan,
                DiskReadPermit(init.reservation.clone()),
                custody,
            )
            .map_err(output::FinishCause::Publication)?,
        );
    }
    Ok(())
}
fn source_component_bytes(plan: &PreparedHostTransferPlan<'_>) -> Result<u64, WorkingMemoryError> {
    let controls = filled_host::control_bytes::<DiskReadPermit>(plan)?;
    controls
        .checked_add(plan.backing_bytes())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
pub(crate) fn disk_read_source_facts(
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    runtime: &PreparedInputRuntime,
    context: &WorkspaceContext,
) -> Result<HostSourceConstructionFacts, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    context
        .charge_metadata(
            size_of::<(
                [[i32; 4]; 2],
                [Dtype; 2],
                &PreparedInputRuntime,
                &WorkspaceContext,
                u64,
            )>()
            .checked_add(size_of::<
                Result<HostSourceConstructionFacts, CacheSourceFailure>,
            >())
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut bytes = 0u64;
    for index in 0..2 {
        let plan = PreparedHostTransferPlan::new(runtime, &shapes[index], dtypes[index], 0)
            .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
        context
            .charge_metadata(
                plan.control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        bytes = bytes
            .checked_add(source_component_bytes(&plan).map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?)
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    }
    HostSourceConstructionFacts::new(bytes, 2, 0)
        .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))
}
fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<PreparedDiskReadDestination>(),
        size_of::<DiskReadBinding>(),
        size_of::<Initialization<'_>>(),
        size_of::<Result<PreparedDiskReadDestination, CacheSourceFailure>>(),
        size_of::<Result<DiskReadBinding, CacheSourceFailure>>(),
        size_of::<(PreparedDiskReadDestination, DiskReadBinding)>(),
        size_of::<(&PreparedDiskReadDestination, &mut CacheBlockSourceLoan<'_>)>(),
        size_of::<(
            &mut PreparedDiskReadDestination,
            &PreparedInputRuntime,
            &mut OriginalHostSourceBank,
            &OriginalHostSourceCustody,
        )>(),
        size_of::<(
            &CacheBlockSourceLoan<'_>,
            &CacheBlockId,
            &CacheShardLayout,
            &PreparedInputRuntime,
            usize,
            &WorkspaceContext,
        )>(),
        size_of::<(
            &CacheBlockSourceLoan<'_>,
            &PagedHostStoreDeclaration<'_>,
            &CacheShardLayout,
            &PreparedInputRuntime,
            usize,
            &WorkspaceContext,
        )>(),
        size_of::<(
            Initialization<'_>,
            &PreparedInputRuntime,
            &mut OriginalHostSourceBank,
            &OriginalHostSourceCustody,
        )>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
