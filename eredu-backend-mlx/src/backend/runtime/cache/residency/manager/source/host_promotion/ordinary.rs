//! An ordinary reload retains its exact Host source and every submitted prefix.
use super::super::host_demotion::PreparedHostEviction;
use super::*;
use crate::backend::nn::{
    shared::current_ordinary_execution_owner,
    workspace::{OrdinaryCallControls, OrdinaryNativeControls, OrdinaryPagedCause},
};
use crate::backend::runtime::cache::residency::OrdinaryReadCacheHostSource;
use safemlx::{Event, PreparedInputRuntime};

/// A selected itinerary can request return of its own completed reload. The
/// destination binds its actual arrays internally, before canonical validation.
pub(crate) trait PreparedHostReturn {
    fn id(&self) -> &CacheBlockId;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception>;
    fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception>;
    fn error(&self, cause: OrdinaryPagedCause) -> Exception;
}
struct ReturnEviction<'a, P> {
    proof: &'a P,
    arrays: [&'a Array; 2],
}
impl<P: PreparedHostReturn> PreparedHostEviction for ReturnEviction<'_, P> {
    type Cause = OrdinaryPagedCause;
    fn id(&self) -> &CacheBlockId {
        self.proof.id()
    }
    fn arrays(&self) -> [&Array; 2] {
        self.arrays
    }
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        self.proof.validate_manager(manager, generation)
    }
    fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception> {
        self.proof.validate_pin_counts(lifecycle)
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        self.proof.error(cause)
    }
}

pub(crate) struct PreparedOrdinaryCacheHostPromotion {
    outputs: [Option<Array>; 2],
    events: [Option<Event>; 2],
    canonical: [Option<Array>; 2],
    aliases: [PreparedArrayClone; 2],
    replaced: Option<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>>,
    host: Option<HostCacheBlock>,
    read_source: Option<OrdinaryReadCacheHostSource>,
    backing: Option<DiskLocation>,
    descriptors: Option<[HostTransferDescriptor<4>; 2]>,
    pin: Option<PinnedCacheBlock>,
    reservation: Option<CachePoolReservation>,
    prepared_reservation: Option<eredu_runtime::cache::PreparedCachePoolReservation>,
    prepared_return_reservation: Option<eredu_runtime::cache::PreparedCachePoolReservation>,
    return_reservation: Option<CachePoolReservation>,
    host_retirement: Option<super::super::device_retirement::DeviceRetirement>,
    device_bytes: u64,
    device_retirement: super::super::device_retirement::DeviceRetirement,
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    capacity: u64,
    attempted: bool,
    completed: bool,
    published: bool,
    demoted: bool,
    context: WorkspaceContext,
    _funding: HostMetadataFunding,
}

impl CacheBlockSourceLoan<'_> {
    /// Describes the exact traced load before its Host source exists. Neither
    /// preparation nor matching geometry grants permission to submit a copy.
    pub(crate) fn prepare_declared_ordinary_host_promotion(
        &self,
        declaration: &crate::backend::nn::workspace::PagedHostStoreDeclaration<'_>,
        runtime: &PreparedInputRuntime,
        additional_reservations: usize,
        may_read_disk: bool,
        context: &WorkspaceContext,
    ) -> Result<PreparedOrdinaryCacheHostPromotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        let shapes = declaration.shapes();
        let dtypes = declaration.dtypes();
        let funding = context
            .metadata_funding()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        context
            .charge_metadata(
                PreparedOrdinaryCacheHostPromotion::preparation_control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let mut capacity = 0u64;
        for index in 0..2 {
            let width = CacheBlockMetadata::floating_dtype_bytes(dtypes[index])
                .ok_or_else(|| fail(CacheSourceError::Geometry))?;
            let bytes = shapes[index]
                .into_iter()
                .try_fold(width, |size, dimension| {
                    let dimension = u64::try_from(dimension)
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or_else(|| fail(CacheSourceError::Geometry))?;
                    size.checked_mul(dimension)
                        .ok_or_else(|| fail(CacheSourceError::Overflow))
                })?;
            let retained = self
                .blocks()
                .find(|row| row.id() == declaration.id())
                .and_then(|row| row.host());
            let bytes = if let Some(buffers) = retained {
                let descriptor = buffers[index]
                    .try_fixed_descriptor::<4>()
                    .map_err(|cause| fail(CacheSourceError::HostDescriptor(cause)))?;
                if descriptor.shape() != shapes[index]
                    || descriptor.dtype() != dtypes[index]
                    || descriptor.policy() != HostTransferPolicy::Transfer
                {
                    return Err(fail(CacheSourceError::Geometry));
                }
                descriptor.allocation().bytes()
            } else {
                HostTransferBuffer::ordinary_capacity(
                    runtime,
                    usize::try_from(bytes).map_err(|_| fail(CacheSourceError::Overflow))?,
                )
                .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?
                .0
            };
            capacity = capacity
                .checked_add(u64::try_from(bytes).map_err(|_| fail(CacheSourceError::Overflow))?)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        let bytes =
            CacheBlockMetadata::floating_bytes([&shapes[0], &shapes[1]], dtypes).map_err(fail)?;
        let prepared_reservation = self
            .pool()
            .prepare_reservation_population(additional_reservations, context)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let device_retirement =
            super::super::device_retirement::DeviceRetirement::prepare(context)?;
        let (prepared_return_reservation, host_retirement) = if may_read_disk {
            let reservation = self
                .pool()
                .prepare_reservation_population(additional_reservations, context)
                .map_err(|cause| {
                    CacheSourceFailure::metadata(context.metadata_source(cause), context)
                })?;
            let retirement =
                super::super::device_retirement::DeviceRetirement::prepare_host(context)?;
            (Some(reservation), Some(retirement))
        } else {
            (None, None)
        };
        let aliases = [
            PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| fail(cause.into()))?,
            PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| fail(cause.into()))?,
        ];
        Ok(PreparedOrdinaryCacheHostPromotion {
            outputs: [None, None],
            events: [None, None],
            canonical: [None, None],
            aliases,
            replaced: None,
            host: None,
            read_source: None,
            backing: None,
            descriptors: None,
            pin: None,
            reservation: None,
            prepared_reservation: Some(prepared_reservation),
            prepared_return_reservation,
            return_reservation: None,
            host_retirement,
            device_bytes: bytes,
            device_retirement,
            id: declaration.id().clone(),
            manager: self.manager.clone(),
            generation: self.generation,
            shapes,
            dtypes,
            capacity,
            attempted: false,
            completed: false,
            published: false,
            demoted: false,
            context: context.clone(),
            _funding: funding,
        })
    }
}

impl PreparedOrdinaryCacheHostPromotion {
    pub(crate) fn values(&self) -> Option<[&Array; 2]> {
        if !self.published || self.demoted {
            return None;
        }
        Some([self.outputs[0].as_ref()?, self.outputs[1].as_ref()?])
    }
    pub(crate) fn acquire(&self) -> Result<PinnedCacheBlockLease, CacheSourceError> {
        if !self.published || self.demoted {
            return Err(CacheSourceError::Identity);
        }
        self.pin
            .as_ref()
            .ok_or(CacheSourceError::Identity)?
            .acquire()
    }
    pub(crate) fn owns_pin(&self) -> bool {
        self.pin.is_some()
    }
    pub(crate) fn read_source_pin_count(&self) -> usize {
        usize::from(self.read_source.is_some())
    }
    pub(crate) fn was_demoted(&self) -> bool {
        self.demoted
    }
    pub(crate) fn host_capacity(&self) -> Option<u64> {
        Some(self.capacity)
    }
    pub(crate) fn reclaim_replaced_device(&mut self) {
        self.device_retirement.reclaim();
        if let Some(retirement) = &mut self.host_retirement {
            retirement.reclaim();
        }
    }

    pub(crate) fn has_backing(&self) -> bool {
        self.backing.is_some()
    }

    /// A positively completed canonical read supplies its own immutable Host
    /// buffers. The witness remains owned through every submitted copy prefix.
    pub(crate) fn run_read<P: PreparedHostPromotion<Cause = OrdinaryPagedCause>>(
        &mut self,
        proof: &P,
        source: OrdinaryReadCacheHostSource,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.read_source.is_some()
            || self.host_retirement.is_none()
            || self.attempted
            || self.published
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        let host = source.host().clone();
        self.read_source = Some(source);
        let result = self.run(proof, host.buffers(), stream);
        drop(host);
        self.reclaim_replaced_device();
        result
    }

    /// The source proof comes from the checked-out actual invocation. Failure
    /// retains Host custody, submitted arrays and events in this destination.
    pub(crate) fn run<P: PreparedHostPromotion<Cause = OrdinaryPagedCause>>(
        &mut self,
        proof: &P,
        buffers: [&ImmutableHostTransferBuffer; 2],
        stream: &Stream,
    ) -> Result<(), Exception> {
        let owner = current_ordinary_execution_owner()?
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
        if owner.paged().is_none()
            || self.attempted
            || self.published
            || (self.read_source.is_some() && self.host_retirement.is_none())
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        proof.validate_manager(&self.manager, self.generation)?;
        proof.validate_id(&self.id)?;
        proof.validate_context(&self.context)?;
        if let Some(source) = &self.read_source {
            source.validate(proof, &self.id, buffers)?;
        }
        let descriptors = [
            buffers[0]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| proof.error(CacheSourceError::HostDescriptor(cause).into()))?,
            buffers[1]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| proof.error(CacheSourceError::HostDescriptor(cause).into()))?,
        ];
        let mut capacity = 0u64;
        for (index, descriptor) in descriptors.iter().enumerate() {
            if descriptor.shape() != self.shapes[index]
                || descriptor.dtype() != self.dtypes[index]
                || descriptor.policy() != HostTransferPolicy::Transfer
            {
                return Err(proof.error(CacheSourceError::Geometry.into()));
            }
            capacity = capacity
                .checked_add(
                    u64::try_from(descriptor.allocation().bytes())
                        .map_err(|_| proof.error(CacheSourceError::Overflow.into()))?,
                )
                .ok_or_else(|| proof.error(CacheSourceError::Overflow.into()))?;
        }
        if capacity > self.capacity {
            return Err(proof.error(CacheSourceError::Geometry.into()));
        }
        let selection = CacheBlockSelection::new(
            self.id.global_layer,
            self.id.representation,
            self.id.start,
            self.id.end,
            0,
        );
        self.manager.with_source_loan_inner(
            selection,
            Some(proof.publication_controls()),
            |cause| proof.error(cause.into()),
            |mut loan| {
                proof.validate_manager(loan.manager(), loan.generation)?;
                let row = loan
                    .blocks()
                    .find(|row| row.id() == &self.id)
                    .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
                require_device_capacity(&loan, row.logical_bytes())
                    .map_err(|cause| proof.error(cause.into()))?;
                let host = row
                    .record
                    .host_block()
                    .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
                let actual = host.buffers();
                if !std::ptr::eq(actual[0], buffers[0]) || !std::ptr::eq(actual[1], buffers[1]) {
                    return Err(proof.error(CacheSourceError::Identity.into()));
                }
                commit::validate_record(row.record, &self.id, host, &descriptors)
                    .map_err(|cause| proof.error(cause.into()))?;
                let backing = row.record.disk().cloned();
                if backing.is_some() && self.host_retirement.is_none() {
                    return Err(proof.error(CacheSourceError::Identity.into()));
                }
                if let Some(read) = &self.read_source {
                    let file = read
                        .backing()
                        .file_source()
                        .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
                    if backing
                        .as_ref()
                        .and_then(DiskLocation::file_source)
                        .is_none_or(|actual| !actual.same_source(&file))
                    {
                        return Err(proof.error(CacheSourceError::Identity.into()));
                    }
                }
                // Occupancy is reserved only when this exact source is loaded;
                // preparation has already paid the finite reservation table.
                self.reservation = Some(
                    self.prepared_reservation
                        .take()
                        .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                        .reserve(CachePoolUsage {
                            device_bytes: self.device_bytes,
                            transfer_in_flight_bytes: capacity,
                            ..CachePoolUsage::default()
                        })
                        .map_err(|cause| {
                            proof.error(CacheSourceError::ReservationAdmission(cause).into())
                        })?,
                );
                self.capacity = capacity;
                self.backing = backing;
                self.host = Some(host.clone());
                self.descriptors = Some(descriptors);
                self.pin = Some(
                    loan.pin_prepared_block(&self.id, Some(self._funding.clone()))
                        .map_err(|cause| proof.error(cause.into()))?,
                );
                Ok(())
            },
        )?;
        if self.backing.is_some() {
            self.return_reservation = Some(
                self.prepared_return_reservation
                    .take()
                    .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                    .reserve(CachePoolUsage::default())
                    .map_err(|cause| {
                        proof.error(CacheSourceError::ReservationAdmission(cause).into())
                    })?,
            );
            self.host_retirement
                .as_mut()
                .expect("prepared read staging retirement")
                .attach_host(
                    self.host
                        .as_ref()
                        .expect("authenticated read buffers")
                        .buffers(),
                )
                .map_err(|cause| proof.error(cause.into()))?;
        }
        let started = Instant::now();
        self.attempted = true;
        for (index, buffer) in self
            .host
            .as_ref()
            .expect("authenticated Host source")
            .buffers()
            .into_iter()
            .enumerate()
        {
            let (array, event) = buffer
                .copy_to_array(stream)
                .map_err(|cause| proof.error(cause.into()))?
                .into_parts();
            self.outputs[index] = Some(array);
            self.events[index] = Some(event);
        }
        for event in self.events.iter().flatten() {
            event
                .synchronize()
                .map_err(|cause| proof.error(cause.into()))?;
        }
        for index in 0..2 {
            let array = self.outputs[index].as_ref().expect("submitted copy");
            // The transfer Event has completed; observe this Array's own event
            // before the strict descriptor query, without evaluating or waiting.
            if array
                .try_observe_availability()
                .map_err(|cause| proof.error(cause.into()))?
                != Some(true)
            {
                return Err(proof.error(CacheSourceError::PendingStorage.into()));
            }
            if array
                .try_descriptor()
                .map_err(|cause| proof.error(CacheSourceError::Descriptor(cause).into()))?
                .facts()
                .allocation()
                .is_none()
                || array.shape() != self.shapes[index]
                || array.dtype() != self.dtypes[index]
            {
                return Err(proof.error(CacheSourceError::PendingStorage.into()));
            }
            self.canonical[index] = Some(
                self.aliases[index]
                    .fill_for_inspection(array)
                    .map_err(|cause| proof.error(OrdinaryPagedCause::Source(cause.into())))?,
            );
        }
        self.completed = true;
        commit::publish(
            &self.manager,
            proof,
            &self.id,
            self.generation,
            self.host.as_ref().expect("retained source"),
            self.descriptors.as_ref().expect("retained descriptors"),
            &mut self.canonical,
            self.reservation.as_mut().expect("unpublished reservation"),
            self.read_source.is_some(),
            started,
        )?;
        self.published = true;
        if self.backing.is_some() {
            self.host_retirement
                .as_mut()
                .expect("prepared read staging retirement")
                .publish(&mut self.reservation);
            self.reservation = self.return_reservation.take();
            // Both Events positively completed and Device publication retains
            // the exact durable backing. Read staging has no remaining use.
            self.events = [None, None];
            self.host = None;
            self.read_source = None;
        }
        Ok(())
    }

    pub(crate) fn return_to_host<P: PreparedHostReturn>(
        &mut self,
        proof: &P,
    ) -> Result<(), Exception> {
        if !self.published
            || !self.completed
            || self.demoted
            || self.replaced.is_some()
            || self.host.is_none()
            || proof.id() != &self.id
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        proof.validate_manager(&self.manager, self.generation)?;
        let arrays = [
            self.outputs[0].as_ref().expect("completed first"),
            self.outputs[1].as_ref().expect("completed second"),
        ];
        let eviction = ReturnEviction { proof, arrays };
        self.device_retirement
            .attach(arrays)
            .map_err(|cause| proof.error(cause.into()))?;
        self.replaced = Some(super::super::host_demotion::commit_host(
            &self.manager,
            &eviction,
            self.host.as_ref().expect("retained Host source").clone(),
            self.capacity,
            self.reservation
                .as_mut()
                .expect("completed promotion reservation"),
            false,
        )?);
        self.demoted = true;
        self.device_retirement.publish(&mut self.reservation);
        self.events = [None, None];
        self.outputs = [None, None];
        self.replaced = None;
        Ok(())
    }

    /// A committed writer authenticates the exact old reusable Host source.
    /// Once returned to Host, this destination needs no further native access.
    pub(crate) fn release_written_host_source(
        &mut self,
        written: &crate::backend::runtime::cache::residency::OrdinaryWrittenCacheHostSource,
    ) -> Result<(), CacheSourceError> {
        let Some(host) = &self.host else {
            return Ok(());
        };
        if !self.demoted {
            return Err(CacheSourceError::Identity);
        }
        written.validate_buffers(&self.id, host.buffers(), &self.context)?;
        self.events = [None, None];
        self.host = None;
        Ok(())
    }

    /// Returns a completed Device allocation to the actual retained file using
    /// the same canonical transition as every other backed cache owner.
    pub(crate) fn return_to_disk<P: PreparedHostReturn>(
        &mut self,
        proof: &P,
    ) -> Result<(), Exception> {
        if !self.published
            || !self.completed
            || self.demoted
            || self.replaced.is_some()
            || proof.id() != &self.id
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        proof.validate_manager(&self.manager, self.generation)?;
        let backing = self
            .backing
            .as_ref()
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
        let arrays = [
            self.outputs[0].as_ref().expect("completed first"),
            self.outputs[1].as_ref().expect("completed second"),
        ];
        let eviction = ReturnEviction { proof, arrays };
        self.device_retirement
            .attach(arrays)
            .map_err(|cause| proof.error(cause.into()))?;
        let replaced = super::return_disk::return_device_prepared(
            &self.manager,
            &self.id,
            backing,
            self.reservation
                .as_mut()
                .expect("completed promotion reservation"),
            &eviction,
        )?;
        self.demoted = true;
        self.device_retirement.publish(&mut self.reservation);
        self.events = [None, None];
        self.outputs = [None, None];
        drop(replaced);
        self.host = None;
        self.read_source = None;
        Ok(())
    }

    fn preparation_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<(
                &CacheBlockSourceLoan<'_>,
                &crate::backend::nn::workspace::PagedHostStoreDeclaration<'_>,
                &PreparedInputRuntime,
                usize,
                bool,
                &WorkspaceContext,
            )>(),
            size_of::<([[i32; 4]; 2], [Dtype; 2], u64, usize)>(),
            PreparedArrayClone::control_bytes()?.checked_mul(2)?,
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            size_of::<Result<PreparedArrayClone, safemlx::PreparedArrayCloneCause>>()
                .checked_mul(2)?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn caller_controls<P: PreparedHostPromotion, E: PreparedHostReturn>(
        &self,
    ) -> Option<OrdinaryCallControls> {
        Self::controls::<P, E>(true, self.host_retirement.is_some())
    }
    /// The typed transfer trace separately prices both safe copies and waits.
    pub(crate) fn source_controls<P: PreparedHostPromotion, E: PreparedHostReturn>(
        &self,
    ) -> Option<OrdinaryCallControls> {
        Self::controls::<P, E>(false, self.host_retirement.is_some())
    }
    fn controls<P: PreparedHostPromotion, E: PreparedHostReturn>(
        include_calls: bool,
        disk_read: bool,
    ) -> Option<OrdinaryCallControls> {
        let frames = [
            commit::control_bytes::<P>()?,
            super::return_disk::prepared_control_bytes::<ReturnEviction<'_, E>>()?,
            if disk_read {
                OrdinaryReadCacheHostSource::validation_control_bytes::<P>()?.checked_add(
                    size_of::<(&mut Self, &P, OrdinaryReadCacheHostSource, &Stream)>(),
                )?
            } else {
                0
            },
            size_of::<HostCacheBlock>(),
            size_of::<(CacheFileSource, Option<CacheFileSource>)>(),
            super::super::host_demotion::prepared_host_commit_control_bytes::<ReturnEviction<'_, E>>(
            )?,
            size_of::<ReturnEviction<'_, E>>(),
            eredu_runtime::cache::PreparedCachePoolReservation::admission_control_bytes()?
                .checked_mul(if disk_read { 2 } else { 1 })?,
            size_of::<(&mut Self, &P, [&ImmutableHostTransferBuffer; 2], &Stream)>(),
            size_of::<(&mut Self, &E)>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
            size_of::<(
                [HostTransferDescriptor<4>; 2],
                Instant,
                CacheBlockSelection,
                u64,
            )>(),
            if include_calls {
                ImmutableHostTransferBuffer::ordinary_copy_wrapper_control_bytes()?
                    .checked_mul(2)?
                    .checked_add(Event::ordinary_synchronize_control_bytes()?.checked_mul(2)?)?
            } else {
                0
            },
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            PreparedArrayClone::control_bytes()?.checked_mul(2)?,
            Array::ordinary_availability_control_bytes()?.checked_mul(2)?,
            Array::descriptor_control_bytes()?.checked_mul(2)?,
            CacheResidencyManager::source_loan_control_bytes::<()>(size_of::<(
                &mut Self,
                &P,
                [&ImmutableHostTransferBuffer; 2],
                [HostTransferDescriptor<4>; 2],
            )>())?,
            PinnedCacheBlock::fixed_controls()?,
        ];
        Some(OrdinaryCallControls {
            metadata_bytes: u64::try_from(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)?,
            )
            .ok()?,
            observed: OrdinaryNativeControls::default(),
        })
    }
}
