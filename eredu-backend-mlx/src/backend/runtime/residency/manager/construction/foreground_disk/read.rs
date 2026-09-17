//! Foreground reads into final source buffers, before the existing copy worker.
use super::*;
use crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity;
use eredu_runtime::working_memory::{
    OriginalHostSourceCustody, OriginalOperationMetadataCustody, WorkingMemoryReservation,
};
use safemlx::{
    ImmutableHostTransferBuffer, InitializedInputAllocator,
    InputAllocatorCause, PreparedInputArena, PreparedInputCause, PreparedSubmissionGraphQuota,
    RetirementCapacityCause, RetirementCapacityOwner, RetirementCapacityPermit,
};
use std::mem::size_of_val;
mod publication;
use eredu_runtime::working_memory::OriginalHostSourceBank;

/// Separate source backing and host controls. Device outputs, native copy Eval,
/// final aggregate completion and publication remain in the enclosing transfer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ForegroundDiskReadLayout {
    pub(crate) host_bytes: usize,
    pub(crate) source_control_bytes: usize,
    pub(crate) source_backing_bytes: usize,
    pub(crate) output_logical_bytes: usize,
    pub(crate) outputs: usize,
    pub(crate) maximum_rank: usize,
}
/// Borrowed actual source and initialized allocator; no grant or payload read.
pub(crate) struct ForegroundDiskReadPlan<'a> {
    source: &'a ForegroundDiskDescriptors,
    domain: &'a eredu_core::SharedStorageDomain,
    initializer: &'static InitializedInputAllocator,
    unit: usize,
    layout: ForegroundDiskReadLayout,
}
/// One attempted unit read. Preparation reserves only finite metadata. Final
/// native buffers are constructed under the shared live capacity when consumed,
/// and become immutable sources only after the complete payload read succeeds.
/// Allocation and publication stay on the caller. Only the exclusive prepared
/// I/O handoff crosses a worker; no native scope, tensor or observer does.
pub(crate) struct PreparedForegroundDiskRead {
    source: ForegroundDiskDescriptors,
    unit: usize,
    ready: Vec<ImmutableHostTransferBuffer>,
    names: Vec<String>,
    named: Vec<(String, RetainedHostBuffer)>,
    initializer: &'static InitializedInputAllocator,
    outputs: usize,
    source_backing_bytes: usize,
    output_logical_bytes: u64,
    completed_capacity_bytes: u64,
    capacity: ForegroundDiskSourceCapacity,
    source_bank: OriginalHostSourceBank,
    _custody: OriginalHostSourceCustody,
}
#[derive(Debug, thiserror::Error)]
enum ReadCause {
    #[error("foreground source bank: {0}")]
    SourceBank(#[source] eredu_runtime::working_memory::HostDestinationCause),
    #[error("foreground source publication: {0}")]
    Publication(#[from] publication::SourceError),
    #[error("foreground disk read geometry or state mismatch")]
    Identity,
    #[error("foreground disk request custody: {0}")]
    Custody(#[source] WorkingMemoryError),
    #[error("foreground disk read reservation: {0}")]
    Reserve(#[from] std::collections::TryReserveError),
    #[error("foreground disk live source capacity: {0}")]
    Capacity(#[from] RetirementCapacityCause),
    #[error("foreground disk source allocator: {0}")]
    Allocator(#[from] InputAllocatorCause),
    #[error("foreground disk source constructor: {0}")]
    Native(#[from] PreparedInputCause),
    #[error("foreground disk source arena: {0}")]
    Arena(#[from] safemlx::SubmissionGraphQuotaCause),
    #[error("foreground disk final destination: {0}")]
    Destination(#[from] safemlx::error::Exception),
    #[error("foreground disk payload: {0}")]
    Read(#[from] eredu_checkpoint::store::DetachedReadFailure<ManagerCustody>),
}
/// The precise failure retains request custody after all unpublished buffers
/// retire; a read failure also keeps the independent admitted source custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct ForegroundDiskReadError {
    #[source]
    cause: ReadCause,
    _custody: OriginalHostSourceCustody,
}
impl ForegroundDiskDescriptors {
    pub(crate) fn read_plan<'a>(
        &'a self,
        id: &OffloadUnitId,
        pool: &'a WorkingMemoryPool,
    ) -> Result<ForegroundDiskReadPlan<'a>, WorkingMemoryError> {
        self._custody.validate_pool(pool)?;
        let initializer =
            crate::backend::managed_memory::input_allocator::admitted_initializer(pool)?;
        let runtime = initializer
            .try_borrow_runtime()
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let overflow = || WorkingMemoryError::Overflow;
        let unit = self
            .value
            .units
            .iter()
            .position(|row| row.definition.id() == id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let row = &self.value.units[unit];
        let count = row.own_reads.len();
        let read = self
            .value
            .source
            .read_slice(row.own_reads.clone())
            .and_then(|reads| reads.read_layout())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let mut layout = ForegroundDiskReadLayout {
            host_bytes: read.required_bytes(),
            source_control_bytes: 0,
            source_backing_bytes: 0,
            output_logical_bytes: 0,
            outputs: count,
            maximum_rank: 0,
        };
        let transfer_controls = crate::backend::runtime::residency::manager::materialization::foreground_disk_control_bytes().ok_or_else(overflow)?;
        let controls = [
            Layout::array::<publication::Pending>(count)
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<ImmutableHostTransferBuffer>(count)
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<String>(count)
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<(String, RetainedHostBuffer)>(count)
                .map_err(|_| overflow())?
                .size(),
            usize::try_from(ResidentHostOwner::storage_bytes()?).map_err(|_| overflow())?,
            size_of::<ResidentHostOwner>(),
            size_of::<Result<ResidentHostOwner, ForegroundDiskReadError>>(),
            size_of::<String>(),
            size_of::<(String, RetainedHostBuffer)>(),
            size_of::<rows::Rows<String, RetainedHostBuffer>>(),
            size_of::<ResidentHostBuffers>(),
            size_of::<
                std::iter::Zip<
                    std::vec::Drain<'_, String>,
                    std::vec::Drain<'_, ImmutableHostTransferBuffer>,
                >,
            >(),
            size_of::<std::slice::Iter<'_, WeightBinding>>(),
            Layout::array::<&mut [u8]>(count)
                .map_err(|_| overflow())?
                .size(),
            size_of::<ForegroundDiskReadPlan<'_>>(),
            size_of::<ForegroundDiskReadLayout>(),
            size_of::<PreparedForegroundDiskRead>(),
            size_of::<PreparedForegroundDiskIo>(),
            size_of::<(&mut PreparedForegroundDiskRead, &mut Vec<publication::Pending>)>(),
            size_of::<&mut PreparedForegroundDiskIo>(),
            size_of::<Result<(), publication::SourceCause>>(),
            size_of::<Result<PreparedForegroundDiskIo, ForegroundDiskReadError>>(),
            // One transaction vector moves intact through allocation/I/O/return;
            // its allocation is the exact per-output Pending array priced above.
            size_of::<Vec<publication::Pending>>(),
            size_of::<ReadForegroundDiskBatch>(),
            size_of::<Result<ReadForegroundDiskBatch, ForegroundDiskReadError>>(),
            size_of::<ForegroundDiskReadError>(),
            size_of::<Result<PreparedForegroundDiskRead, ForegroundDiskReadError>>(),
            size_of::<Result<(), ForegroundDiskReadError>>(),
            size_of::<Vec<&mut [u8]>>(),
            size_of::<OriginalHostSourceCustody>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<&WorkingMemoryReservation>(),
            size_of::<Option<&WorkingMemoryReservation>>(),
            size_of::<PreparedInputArena>(),
            size_of::<
                PreparedSubmissionGraphQuota<RetirementCapacityOwner<OriginalHostSourceCustody>>,
            >(),
            size_of::<RetirementCapacityOwner<OriginalHostSourceCustody>>(),
            size_of::<RetirementCapacityPermit>(),
            size_of::<Result<RetirementCapacityPermit, RetirementCapacityCause>>(),
            size_of::<ForegroundDiskSourceCapacity>(),
            size_of::<&ForegroundDiskSourceCapacity>(),
            // One runtime loan when this deferred attempt is consumed. Cold
            // planning's earlier loan has already retired before admission.
            InitializedInputAllocator::borrow_control_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<Result<(), ReadCause>>(),
            size_of::<std::vec::Drain<'_, publication::Pending>>(),
            size_of::<std::slice::IterMut<'_, publication::Pending>>(),
            size_of::<
                Result<OriginalHostSourceBank, eredu_runtime::working_memory::HostDestinationCause>,
            >(),
            size_of::<PreparedHostTransferPlan<'_>>(),
            transfer_controls,
        ];
        layout.host_bytes = layout
            .host_bytes
            .checked_add(size_of_val(&controls))
            .ok_or_else(overflow)?;
        for bytes in controls {
            layout.host_bytes = layout.host_bytes.checked_add(bytes).ok_or_else(overflow)?;
        }
        for binding in row
            .definition
            .bindings()
            .iter()
            .filter(|binding| !binding.is_alias())
        {
            layout.host_bytes = layout
                .host_bytes
                .checked_add(binding.name().len())
                .and_then(|bytes| {
                    bytes.checked_add(
                        usize::try_from(RetainedHostBuffer::storage_bytes().ok()?).ok()?,
                    )
                })
                .ok_or_else(overflow)?;
        }
        for index in row.own_reads.clone() {
            let native = &self.value.native_reads[index];
            let plan = PreparedHostTransferPlan::new(&runtime, &native.shape, native.dtype, 0)
                .map_err(|_| WorkingMemoryError::UnknownBound)?;
            layout.source_control_bytes = layout
                .source_control_bytes
                .checked_add(publication::control_bytes(&plan)?)
                .ok_or_else(overflow)?;
            layout.source_backing_bytes = layout
                .source_backing_bytes
                .checked_add(plan.backing_bytes())
                .ok_or_else(overflow)?;
            layout.output_logical_bytes = layout
                .output_logical_bytes
                .checked_add(plan.logical_bytes())
                .ok_or_else(overflow)?;
            layout.maximum_rank = layout.maximum_rank.max(native.shape.len());
        }
        Ok(ForegroundDiskReadPlan {
            source: self,
            domain: pool.shared_storage_domain(),
            initializer,
            unit,
            layout,
        })
    }
}
impl ForegroundDiskReadPlan<'_> {
    pub(crate) fn layout(&self) -> ForegroundDiskReadLayout {
        self.layout
    }
    /// Called by the enclosing once-only operation-bank constructor after it
    /// accepts this exact host/source layout. Custody alone is not a new grant.
    pub(crate) fn prepare(
        self,
        custody: eredu_runtime::working_memory::OriginalTextControlGuard,
        reservation: &WorkingMemoryReservation,
        capacity: &ForegroundDiskSourceCapacity,
    ) -> Result<PreparedForegroundDiskRead, ForegroundDiskReadError> {
        self.prepare_source(custody.into(), Some(reservation), capacity)
    }
    pub(crate) fn prepare_source(
        self,
        custody: OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
        capacity: &ForegroundDiskSourceCapacity,
    ) -> Result<PreparedForegroundDiskRead, ForegroundDiskReadError> {
        // No source clone, vector reservation or native allocation precedes
        // matching both the source domain and the exact accepted request.
        if !custody.metadata_custody().matches_domain(self.domain) {
            return Err(ForegroundDiskReadError {
                cause: ReadCause::Identity,
                _custody: custody,
            });
        }
        if let Err(cause) = custody
            .validate_account(reservation)
            .and_then(|()| capacity.validate_account(reservation))
        {
            return Err(ForegroundDiskReadError {
                cause: ReadCause::Custody(cause),
                _custody: custody,
            });
        }
        if !capacity.matches_source(self.source) || !custody.same_source(capacity.custody()) {
            return Err(ForegroundDiskReadError {
                cause: ReadCause::Identity,
                _custody: custody,
            });
        }
        let source_bank = capacity
            .source_bank(self.layout.source_control_bytes as u64, self.layout.outputs)
            .map_err(|cause| ForegroundDiskReadError {
                cause: ReadCause::SourceBank(cause),
                _custody: custody.clone(),
            })?;
        let mut batch = PreparedForegroundDiskRead {
            source: self.source.clone(),
            unit: self.unit,
            ready: Vec::new(),
            names: Vec::new(),
            named: Vec::new(),
            initializer: self.initializer,
            outputs: self.layout.outputs,
            source_backing_bytes: self.layout.source_backing_bytes,
            output_logical_bytes: u64::try_from(self.layout.output_logical_bytes).map_err(
                |_| ForegroundDiskReadError {
                    cause: ReadCause::Identity,
                    _custody: custody.clone(),
                },
            )?,
            completed_capacity_bytes: 0,
            capacity: capacity.clone(),
            source_bank,
            _custody: custody,
        };
        let result = (|| -> Result<(), ReadCause> {
            batch.ready.try_reserve_exact(self.layout.outputs)?;
            batch.names.try_reserve_exact(self.layout.outputs)?;
            batch.named.try_reserve_exact(self.layout.outputs)?;
            for binding in batch.source.value.units[batch.unit]
                .definition
                .bindings()
                .iter()
                .filter(|binding| !binding.is_alias())
            {
                let mut name = String::new();
                name.try_reserve_exact(binding.name().len())?;
                name.push_str(binding.name());
                batch.names.push(name);
            }
            if batch.names.len() != self.layout.outputs {
                return Err(ReadCause::Identity);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(batch),
            Err(cause) => Err(ForegroundDiskReadError {
                cause,
                _custody: batch._custody.clone(),
            }),
        }
    }
}
impl PreparedForegroundDiskRead {
    pub(in crate::backend::runtime::residency::manager) fn matches_source(
        &self,
        source: &ForegroundDiskDescriptors,
    ) -> bool {
        self.source.same_source(source)
    }

    pub(in crate::backend::runtime::residency::manager) fn unit(&self) -> &OffloadUnit {
        &self.source.value.units[self.unit].definition
    }
    pub(in crate::backend::runtime::residency::manager) fn control_custody(
        &self,
    ) -> &OriginalHostSourceCustody {
        &self._custody
    }
    pub(in crate::backend::runtime::residency::manager) fn outputs(&self) -> usize {
        self.outputs
    }

    /// Consume the same allocate/read/publish worker synchronously for ordinary
    /// foreground use. No read attempt or capacity allowance is duplicated.
    pub(crate) fn read(self) -> Result<ReadForegroundDiskBatch, ForegroundDiskReadError> {
        self.allocate()?.read_payload()?.finish()
    }
    fn allocate_into(&mut self, filling: &mut Vec<publication::Pending>) -> Result<(), ReadCause> {
            let unit = &self.source.value.units[self.unit];
            if self.outputs != unit.own_reads.len()
                || !self.ready.is_empty()
                || self.ready.capacity() < self.outputs
            {
                return Err(ReadCause::Identity);
            }
            filling.try_reserve_exact(self.outputs)?;
            // Charge the complete unit before any native quota/backing allocation.
            // Split only transfers that charge. A failed prefix returns unused
            // bytes now and submitted source bytes only at final native release.
            let mut permit = self.capacity.try_acquire(self.source_backing_bytes)?;
            let runtime = self.initializer.try_borrow_runtime()?;
            // Validate every selected output/layout and the complete byte sum
            // before the first per-buffer transaction performs payload I/O.
            let mut actual_backing = 0usize;
            for index in unit.own_reads.clone() {
                let native = &self.source.value.native_reads[index];
                let plan = PreparedHostTransferPlan::new(&runtime, &native.shape, native.dtype, 0)?;
                let metadata = self
                    .source
                    .value
                    .source
                    .read_output(index)
                    .ok_or(ReadCause::Identity)?;
                if u64::try_from(plan.logical_bytes()).ok() != Some(metadata.byte_len())
                    || self
                        .source
                        .value
                        .source
                        .read_slice(index..index + 1)
                        .and_then(|read| read.read_layout())
                        .is_none()
                {
                    return Err(ReadCause::Identity);
                }
                actual_backing = actual_backing
                    .checked_add(plan.backing_bytes())
                    .ok_or(ReadCause::Identity)?;
            }
            if actual_backing != self.source_backing_bytes {
                return Err(ReadCause::Identity);
            }
            for index in unit.own_reads.clone() {
                let native = &self.source.value.native_reads[index];
                let plan = PreparedHostTransferPlan::new(&runtime, &native.shape, native.dtype, 0)?;
                let part = permit.try_split(plan.backing_bytes())?;
                filling.push(publication::begin(
                    &mut self.source_bank,
                    plan,
                    &self.capacity,
                    part,
                    &self._custody,
                )?);
            }
            Ok(())
    }
    /// Allocate all exact final destinations on the calling thread before the
    /// prepared job becomes visible to a worker or performs any file I/O.
    pub(crate) fn allocate(mut self) -> Result<PreparedForegroundDiskIo, ForegroundDiskReadError> {
        let mut filling = Vec::new();
        let result = self.allocate_into(&mut filling);
        match result {
            Ok(()) => Ok(PreparedForegroundDiskIo { filling, read: self, initialized: false }),
            Err(cause) => Err(ForegroundDiskReadError { cause, _custody: self._custody.clone() }),
        }
    }
}
/// Exclusive final destinations, their exact detached source, and paid custody.
/// Worker code can only perform the grouped checkpoint read. Publication checks
/// that the handoff has returned to the native buffers' creating thread.
/// Filling drops before read custody; a worker drop defers native retirement.
pub(crate) struct PreparedForegroundDiskIo {
    filling: Vec<publication::Pending>,
    read: PreparedForegroundDiskRead,
    initialized: bool,
}
impl PreparedForegroundDiskIo {
    fn read_into(&mut self) -> Result<(), ReadCause> {
            if self.initialized { return Err(ReadCause::Identity); }
            let mut destinations = Vec::new();
            destinations.try_reserve_exact(self.filling.len())?;
            for pending in &mut self.filling {
                destinations.push(pending.completed_mut().bytes_mut());
            }
            let unit = &self.read.source.value.units[self.read.unit];
            self.read.source.value.source.read_slice(unit.own_reads.clone())
                .ok_or(ReadCause::Identity)?.read_many_into(&mut destinations)?;
            Ok(())
    }
    /// File I/O only: no native allocator, lock, registration or finalizer.
    pub(crate) fn read_payload(mut self) -> Result<Self, ForegroundDiskReadError> {
        let result = self.read_into();
        match result {
            Ok(()) => { self.initialized = true; Ok(self) }
            Err(cause) => Err(ForegroundDiskReadError { cause, _custody: self.read._custody.clone() }),
        }
    }
    fn publish(&mut self) -> Result<(), ReadCause> {
            if !self.initialized { return Err(ReadCause::Identity); }
            for pending in &mut self.filling {
                pending.completed_mut().freeze().map_err(|_| ReadCause::Identity)?;
            }
            for pending in self.filling.drain(..) {
                let (buffer, allocation) = publication::finish(pending)?;
                let bytes = u64::try_from(allocation.bytes()).map_err(|_| ReadCause::Identity)?;
                self.read.completed_capacity_bytes = self.read.completed_capacity_bytes
                    .checked_add(bytes).ok_or(ReadCause::Identity)?;
                self.read.ready.push(buffer);
            }
            Ok(())
    }
    /// Complete the existing source transaction on the caller, after the actual
    /// worker callback returned. Failed publication never reopens the read slot.
    pub(crate) fn finish(mut self) -> Result<ReadForegroundDiskBatch, ForegroundDiskReadError> {
        let result = self.publish();
        match result {
            Ok(()) => {
                // Deallocate the completed transaction vector before its source
                // account can be moved into and retired with the published batch.
                drop(self.filling);
                Ok(ReadForegroundDiskBatch(self.read))
            }
            Err(cause) => Err(ForegroundDiskReadError { cause, _custody: self.read._custody.clone() }),
        }
    }
}
/// All final source buffers are initialized. They retain their native source
/// quotas independently through submitted CopyFromHostTransfer primitives.
pub(crate) struct ReadForegroundDiskBatch(PreparedForegroundDiskRead);

// These are the actual move-only job and terminal transports. Enforce the
// thread boundary structurally. Only safemlx's closed zero-handle writer is
// movable while mutable; no OriginalScopeObserver enters the worker.
const _: () = {
    fn sendable<T: Send + 'static>() {}
    let _ = sendable::<PreparedForegroundDiskRead>;
    let _ = sendable::<PreparedForegroundDiskIo>;
    let _ = sendable::<Result<ReadForegroundDiskBatch, ForegroundDiskReadError>>;
};
impl ReadForegroundDiskBatch {
    /// These facts come from the same successful source publication: logical
    /// bytes were validated before read, capacity is the actual observed union
    /// of this unit's own canonical buffers. Aliases contribute no new backing.
    pub(in crate::backend::runtime::residency::manager) fn completed_bytes(&self) -> (u64, u64) {
        (self.0.output_logical_bytes, self.0.completed_capacity_bytes)
    }

    pub(in crate::backend::runtime::residency::manager) fn prepared(
        &self,
    ) -> &PreparedForegroundDiskRead {
        &self.0
    }

    /// Publish only initialized buffers into the same final named host owner
    /// consumed by host-to-device copies. All names/row slots were reserved by
    /// prepare; Arc controls are included in this exact attempted read layout.
    /// Source capacity stays in each native buffer until its final actual alias.
    pub(in crate::backend::runtime::residency::manager) fn into_host(
        mut self,
    ) -> Result<ResidentHostOwner, ForegroundDiskReadError> {
        if self.0.names.len() != self.0.outputs
            || self.0.ready.len() != self.0.outputs
            || !self.0.named.is_empty()
            || self.0.named.capacity() < self.0.outputs
        {
            return Err(ForegroundDiskReadError {
                cause: ReadCause::Identity,
                _custody: self.0._custody.clone(),
            });
        }
        for (name, buffer) in self.0.names.drain(..).zip(self.0.ready.drain(..)) {
            self.0.named.push((
                name,
                RetainedHostBuffer::request(buffer, self.0._custody.metadata_custody()),
            ));
        }
        let rows = rows::Rows::from_sorted(std::mem::take(&mut self.0.named));
        Ok(ResidentHostOwner::request(
            ResidentHostBuffers { buffers: rows },
            self.0._custody.metadata_custody(),
        ))
    }
}
