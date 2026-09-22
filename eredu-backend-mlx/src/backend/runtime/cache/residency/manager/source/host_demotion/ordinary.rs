//! Ordinary Host destinations bound to the actual declared canonical source.
use super::*;
use crate::backend::nn::{
    shared::current_ordinary_execution_owner,
    workspace::{OrdinaryCallControls, OrdinaryNativeControls, OrdinaryPagedCause},
};
use safemlx::{AllocationPlacement, Event, HostTransferDescriptor, PreparedAllocationOwner};

type HostAttachment = PreparedAllocationOwner<HostMetadataFunding>;

/// The enclosing ordinary Work retains this attempted prefix until its native
/// scope settles. Canonical Host aliases retain their own metadata payer through
/// a prepared attachment to the actual allocation, without retaining a manager.
pub(crate) struct PreparedOrdinaryCacheHostDemotion {
    pending: [Option<(HostTransferBuffer, Event)>; 2],
    host: [Option<Arc<ImmutableHostTransferBuffer>>; 2],
    attachments: [Option<HostAttachment>; 2],
    replaced: Option<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>>,
    source: Option<[AllocationInfo; 2]>,
    shapes: [[i32; 4]; 2],
    dtypes: [Dtype; 2],
    capacities: [usize; 2],
    placements: [AllocationPlacement; 2],
    observed: OrdinaryNativeControls,
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    prepared_reservation: Option<eredu_runtime::cache::PreparedCachePoolReservation>,
    reservation: Option<CachePoolReservation>,
    device_retirement: super::super::device_retirement::DeviceRetirement,
    attempted: bool,
    published: bool,
    _funding: HostMetadataFunding,
}

impl CacheBlockSourceLoan<'_> {
    pub(crate) fn prepare_declared_ordinary_host_demotion(
        &self,
        declaration: &crate::backend::nn::workspace::PagedHostStoreDeclaration<'_>,
        runtime: &PreparedInputRuntime,
        additional_reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedOrdinaryCacheHostDemotion, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        if !declaration.requires_store() {
            return Err(fail(CacheSourceError::Identity));
        }
        let id = declaration.id();
        let shapes = declaration.shapes();
        let dtypes = declaration.dtypes();
        let funding = context
            .metadata_funding()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        context
            .charge_metadata(
                PreparedOrdinaryCacheHostDemotion::preparation_control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let source = if declaration.is_retained() {
            let row = self
                .blocks()
                .find(|row| row.id() == id)
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            if row.phase() != CacheStoragePhase::Device || row.disk().is_some() {
                return Err(fail(CacheSourceError::PromotionRequired));
            }
            let arrays = row
                .device()
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            let mut allocation = [None, None];
            for (index, array) in arrays.into_iter().enumerate() {
                if array.shape() != shapes[index] || array.dtype() != dtypes[index] {
                    return Err(fail(CacheSourceError::Geometry));
                }
                allocation[index] = Some(
                    array
                        .try_allocation_info()
                        .map_err(|cause| fail(CacheSourceError::ArrayMetadata(cause)))?
                        .ok_or_else(|| fail(CacheSourceError::Identity))?,
                );
            }
            Some(allocation.map(Option::unwrap))
        } else {
            None
        };
        let mut capacities = [0; 2];
        let mut placements = [AllocationPlacement::Unknown; 2];
        let mut observed = OrdinaryNativeControls::default();
        let mut capacity = 0u64;
        for index in 0..2 {
            let width = CacheBlockMetadata::floating_dtype_bytes(dtypes[index])
                .ok_or_else(|| fail(CacheSourceError::Geometry))?;
            let bytes = shapes[index]
                .into_iter()
                .try_fold(width, |size, dimension| {
                    let dimension = u64::try_from(dimension)
                        .ok()
                        .filter(|value| *value != 0)
                        .ok_or_else(|| fail(CacheSourceError::Geometry))?;
                    size.checked_mul(dimension)
                        .ok_or_else(|| fail(CacheSourceError::Overflow))
                })?;
            let (size, placement) = HostTransferBuffer::ordinary_capacity(
                runtime,
                usize::try_from(bytes).map_err(|_| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| fail(CacheSourceError::HostInput(cause)))?;
            let controls = HostTransferBuffer::ordinary_observed_control_bytes(runtime, 4)
                .ok_or_else(|| fail(CacheSourceError::Geometry))?;
            capacities[index] = size;
            placements[index] = placement;
            capacity = capacity
                .checked_add(u64::try_from(size).map_err(|_| fail(CacheSourceError::Overflow))?)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
            observed = observed
                .append(OrdinaryNativeControls {
                    observed_host_bytes: u64::try_from(controls)
                        .map_err(|_| fail(CacheSourceError::Overflow))?,
                    // HostTransferControlAllocator performs one shared-owner allocation.
                    // The separately described physical Host payload is another birth.
                    control_allocations: 1,
                    platform_events: 0,
                })
                .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        }
        if capacity > self.manager.options().host_budget_bytes() {
            return Err(fail(CacheSourceError::PromotionRequired));
        }
        let prepared_reservation = self
            .pool()
            .prepare_reservation_population(additional_reservations, context)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let device_retirement =
            super::super::device_retirement::DeviceRetirement::prepare(context)?;
        let mut attachments = [None, None];
        for destination in &mut attachments {
            *destination = Some(HostAttachment::try_new(funding.clone()).map_err(|error| {
                let (cause, retained) = error.into_parts();
                let failure = fail(CacheSourceError::OrdinaryHostOwner(cause));
                drop(retained);
                failure
            })?);
        }
        Ok(PreparedOrdinaryCacheHostDemotion {
            pending: [None, None],
            host: [None, None],
            attachments,
            replaced: None,
            source,
            shapes,
            dtypes,
            capacities,
            placements,
            observed,
            id: id.clone(),
            manager: self.manager.clone(),
            generation: self.generation,
            prepared_reservation: Some(prepared_reservation),
            reservation: None,
            device_retirement,
            attempted: false,
            published: false,
            _funding: funding,
        })
    }
}
impl PreparedOrdinaryCacheHostDemotion {
    pub(crate) fn capacities(&self) -> [(usize, AllocationPlacement); 2] {
        [
            (self.capacities[0], self.placements[0]),
            (self.capacities[1], self.placements[1]),
        ]
    }
    pub(crate) fn constructor_controls(&self) -> OrdinaryNativeControls {
        self.observed
    }
    pub(crate) fn reclaim_replaced_device(&mut self) {
        self.device_retirement.reclaim();
    }
    pub(crate) fn host_capacity(&self) -> Option<u64> {
        self.capacities.iter().try_fold(0u64, |sum, value| {
            sum.checked_add(u64::try_from(*value).ok()?)
        })
    }
    pub(crate) fn buffers(&self) -> Option<[&ImmutableHostTransferBuffer; 2]> {
        if !self.published {
            return None;
        }
        Some([self.host[0].as_deref()?, self.host[1].as_deref()?])
    }
    pub(crate) fn run<P: PreparedHostEviction<Cause = OrdinaryPagedCause>>(
        &mut self,
        proof: &P,
        stream: &Stream,
    ) -> Result<(), Exception> {
        // This lookup requires the exact physical observer of the active native
        // scope. A retained source or equal geometry alone cannot submit a copy.
        let owner = current_ordinary_execution_owner()?
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
        if owner.paged().is_none() || self.attempted || self.published || proof.id() != &self.id {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        proof.validate_manager(&self.manager, self.generation)?;
        for (index, array) in proof.arrays().into_iter().enumerate() {
            if array
                .try_observe_availability()
                .map_err(|cause| proof.error(cause.into()))?
                != Some(true)
                || array.shape() != self.shapes[index]
                || array.dtype() != self.dtypes[index]
            {
                return Err(proof.error(CacheSourceError::PendingStorage.into()));
            }
            let actual = array
                .try_allocation_info()
                .map_err(|cause| proof.error(CacheSourceError::ArrayMetadata(cause).into()))?
                .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
            if self.source.is_some_and(|source| source[index] != actual) {
                return Err(proof.error(CacheSourceError::Identity.into()));
            }
        }
        let capacity = self
            .host_capacity()
            .ok_or_else(|| proof.error(CacheSourceError::Overflow.into()))?;
        super::commit::preflight(&self.manager, proof, capacity)?;
        self.reservation = Some(
            self.prepared_reservation
                .take()
                .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                .reserve(CachePoolUsage {
                    host_bytes: capacity,
                    transfer_in_flight_bytes: capacity,
                    ..CachePoolUsage::default()
                })
                .map_err(|cause| {
                    proof.error(CacheSourceError::ReservationAdmission(cause).into())
                })?,
        );
        self.attempted = true;
        for (index, array) in proof.arrays().into_iter().enumerate() {
            self.pending[index] = Some(
                HostTransferBuffer::copy_from_array(array, HostTransferPolicy::Transfer, stream)
                    .map_err(|cause| proof.error(cause.into()))?
                    .into_parts(),
            );
        }
        for index in 0..2 {
            self.pending[index]
                .as_ref()
                .expect("submitted Host destination")
                .1
                .synchronize()
                .map_err(|cause| proof.error(cause.into()))?;
            let (buffer, event) = self.pending[index]
                .take()
                .expect("completed Host destination");
            let buffer = buffer.freeze();
            // Keep the completed payload even when fixed metadata or attachment
            // validation fails; failure never reconstructs a new source.
            self.host[index] = Some(Arc::new(buffer));
            drop(event);
            let buffer = self.host[index].as_deref().unwrap();
            let descriptor = buffer
                .try_fixed_descriptor::<4>()
                .map_err(|cause| proof.error(CacheSourceError::HostDescriptor(cause).into()))?;
            if descriptor.shape() != self.shapes[index]
                || descriptor.dtype() != self.dtypes[index]
                || descriptor.policy() != HostTransferPolicy::Transfer
                || descriptor.allocation().bytes() > self.capacities[index]
                || descriptor.allocation().placement() != self.placements[index]
            {
                return Err(proof.error(CacheSourceError::Geometry.into()));
            }
            let attachment = self.attachments[index]
                .take()
                .expect("one prepared Host owner");
            if let Err(error) = buffer.try_attach_prepared_allocation_owner(attachment) {
                let (cause, attachment) = error.into_parts();
                self.attachments[index] = Some(attachment);
                return Err(proof.error(CacheSourceError::OrdinaryHostOwner(cause).into()));
            }
        }
        self.device_retirement
            .attach(proof.arrays())
            .map_err(|cause| proof.error(cause.into()))?;
        let host = match self.id.representation {
            CacheRepresentation::KeyValue => HostCacheBlock::KeyValue {
                keys: Arc::clone(self.host[0].as_ref().unwrap()),
                values: Arc::clone(self.host[1].as_ref().unwrap()),
            },
            CacheRepresentation::CompressedLatentRotary => HostCacheBlock::CompressedLatentRotary {
                latent: Arc::clone(self.host[0].as_ref().unwrap()),
                rotary_key: Arc::clone(self.host[1].as_ref().unwrap()),
            },
        };
        let capacity = host.buffers().into_iter().try_fold(0u64, |sum, buffer| {
            let actual = buffer
                .try_allocation_info()
                .map_err(|cause| proof.error(CacheSourceError::HostDescriptor(cause).into()))?;
            sum.checked_add(
                u64::try_from(actual.bytes())
                    .map_err(|_| proof.error(CacheSourceError::Overflow.into()))?,
            )
            .ok_or_else(|| proof.error(CacheSourceError::Overflow.into()))
        })?;
        self.replaced = Some(commit_host(
            &self.manager,
            proof,
            host,
            capacity,
            self.reservation.as_mut().expect("unpublished occupancy"),
            true,
        )?);
        self.published = true;
        self.device_retirement.publish(&mut self.reservation);
        self.replaced = None;
        Ok(())
    }
    /// Only the completed writer's closed source can retire this reusable Host
    /// alias. Pending copies and failed writes cannot obtain that witness.
    pub(crate) fn release_written_host_source(
        &mut self,
        written: &crate::backend::runtime::cache::residency::OrdinaryWrittenCacheHostSource,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceError> {
        if !self.published || self.pending.iter().any(Option::is_some) {
            return Err(CacheSourceError::Identity);
        }
        let buffers = self.buffers().ok_or(CacheSourceError::Identity)?;
        written.validate_buffers(&self.id, buffers, context)?;
        self.host = [None, None];
        Ok(())
    }
    pub(crate) fn written_host_release_control_bytes() -> Option<usize> {
        size_of::<(&mut Self,
            &crate::backend::runtime::cache::residency::OrdinaryWrittenCacheHostSource,
            &WorkspaceContext, [&ImmutableHostTransferBuffer; 2], Result<(), CacheSourceError>)>()
            .checked_add(crate::backend::runtime::cache::residency::OrdinaryWrittenCacheHostSource::validation_control_bytes())
    }
    fn preparation_control_bytes() -> Option<usize> {
        let layout = HostAttachment::layout();
        let fields = [
            size_of::<Self>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<(
                &CacheBlockSourceLoan<'_>,
                &crate::backend::nn::workspace::PagedHostStoreDeclaration<'_>,
                &PreparedInputRuntime,
                usize,
                &WorkspaceContext,
            )>(),
            size_of::<(
                Option<[AllocationInfo; 2]>,
                [[i32; 4]; 2],
                [Dtype; 2],
                [usize; 2],
                [AllocationPlacement; 2],
            )>(),
            size_of::<OrdinaryNativeControls>(),
            size_of::<Option<OrdinaryNativeControls>>(),
            size_of::<Option<HostAttachment>>().checked_mul(2)?,
            layout.allocation_bytes()?.checked_mul(2)?,
            layout.preparation_control_bytes().checked_mul(2)?,
            layout.preparation_failure_bytes().checked_mul(2)?,
            WorkspaceContext::metadata_arc_bytes::<ImmutableHostTransferBuffer>()?
                .checked_mul(2)?,
        ];
        fields
            .into_iter()
            .try_fold(std::mem::size_of_val(&fields), usize::checked_add)
    }
    pub(crate) fn caller_controls<P: PreparedHostEviction>() -> Option<OrdinaryCallControls> {
        Self::controls::<P>(true)
    }
    /// Safe copy and Event transports are already priced by the typed trace.
    pub(crate) fn source_controls<P: PreparedHostEviction>() -> Option<OrdinaryCallControls> {
        Self::controls::<P>(false)
    }
    fn controls<P: PreparedHostEviction>(include_calls: bool) -> Option<OrdinaryCallControls> {
        let layout = HostAttachment::layout();
        let fields = [
            size_of::<(&mut Self, &P, &Stream)>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
            size_of::<HostCacheBlock>(),
            size_of::<(usize, u64, Option<AllocationInfo>)>(),
            size_of::<[(usize, AllocationPlacement); 2]>(),
            if include_calls {
                HostTransferBuffer::ordinary_detach_wrapper_control_bytes()?
                    .checked_mul(2)?
                    .checked_add(Event::ordinary_synchronize_control_bytes()?.checked_mul(2)?)?
            } else {
                0
            },
            eredu_runtime::cache::PreparedCachePoolReservation::admission_control_bytes()?,
            Array::ordinary_availability_control_bytes()?.checked_mul(2)?,
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            layout.attachment_failure_bytes().checked_mul(2)?,
            layout.original_attachment_control_bytes().checked_mul(2)?,
            // Pre-copy validation and final publication use the same source worker.
            prepared_host_commit_control_bytes::<P>()?.checked_mul(2)?,
        ];
        Some(OrdinaryCallControls {
            metadata_bytes: u64::try_from(
                fields
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&fields), usize::checked_add)?,
            )
            .ok()?,
            observed: OrdinaryNativeControls::default(),
        })
    }
}
