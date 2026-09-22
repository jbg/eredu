//! Ordinary destinations share the selected file worker and retain independent owners.
use super::super::filling::{OrdinaryAttachment, OrdinaryPreparation, OrdinaryReadOwner};
use super::*;
use crate::backend::nn::{
    shared::current_ordinary_execution_owner,
    workspace::{OrdinaryCallControls, OrdinaryNativeControls, OrdinaryPagedCause},
};
use crate::backend::runtime::cache::residency::PreparedHostPromotion;
use eredu_runtime::cache::PreparedCachePoolReservation;
use safemlx::{AllocationPlacement, PreparedHostTransferWriter, error::Exception};

impl CacheBlockSourceLoan<'_> {
    pub(crate) fn prepare_declared_ordinary_disk_read(
        &self,
        declaration: &PagedHostStoreDeclaration<'_>,
        layout: &CacheShardLayout,
        runtime: &PreparedInputRuntime,
        reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<(PreparedDiskReadDestination, OrdinaryDiskReadSource), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        let destination = prepare_for(
            self,
            declaration.id(),
            layout,
            runtime,
            reservations,
            context,
            true,
        )?;
        if destination.shapes != declaration.shapes() || destination.dtypes != declaration.dtypes()
        {
            return Err(fail(CacheSourceError::Geometry));
        }
        let identity = destination
            .ordinary
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .identity
            .clone();
        let source = OrdinaryDiskReadSource::from_prepared_identity(identity, context);
        Ok((destination, source))
    }
}

pub(super) fn prepare_owners(
    host: PreparedCachePoolReservation,
    transfer: PreparedCachePoolReservation,
    placements: [AllocationPlacement; 2],
    occupancy: &DiskReadOccupancy,
    context: &WorkspaceContext,
) -> Result<OrdinaryPreparation, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let funding = context
        .metadata_funding()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let layout = OrdinaryAttachment::layout();
    let fields = [
        size_of::<OrdinaryPreparation>(),
        size_of::<OrdinaryReadOwner>(),
        size_of::<Result<OrdinaryPreparation, CacheSourceFailure>>(),
        size_of::<(
            PreparedCachePoolReservation,
            PreparedCachePoolReservation,
            [AllocationPlacement; 2],
            &DiskReadOccupancy,
            &WorkspaceContext,
        )>(),
        size_of::<Result<(PreparedDiskReadDestination, OrdinaryDiskReadSource), CacheSourceFailure>>(
        ),
        layout
            .allocation_bytes()
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        layout
            .preparation_control_bytes()
            .checked_mul(2)
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        layout
            .preparation_failure_bytes()
            .checked_mul(2)
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
    let mut attachments = [None, None];
    for attachment in &mut attachments {
        *attachment = Some(
            OrdinaryAttachment::try_new(OrdinaryReadOwner {
                occupancy: occupancy.clone(),
                funding: funding.clone(),
            })
            .map_err(|error| {
                let (cause, owner) = error.into_parts();
                let failure = fail(CacheSourceError::OrdinaryHostOwner(cause));
                drop(owner);
                failure
            })?,
        );
    }
    let identity = context
        .metadata_arc(())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    Ok(OrdinaryPreparation {
        attachments,
        host: Some(host),
        transfer: Some(transfer),
        placements,
        identity,
    })
}

impl PreparedDiskReadDestination {
    /// This same immutable descriptor supplies the physical payload populations.
    pub(crate) fn ordinary_capacities(&self) -> Option<[(usize, AllocationPlacement); 2]> {
        let source = self.ordinary.as_ref()?;
        Some([
            (self.capacities[0], source.placements[0]),
            (self.capacities[1], source.placements[1]),
        ])
    }
    pub(crate) fn ordinary_constructor_controls(&self) -> Option<OrdinaryNativeControls> {
        self.observed
    }
    /// Called after the canonical source was bound. The actual Work supplies
    /// both the physical observer and the source proof; the cold descriptor alone
    /// cannot create either permission. Failed prefixes remain in this destination.
    pub(crate) fn construct_ordinary<P: PreparedHostPromotion<Cause = OrdinaryPagedCause>>(
        &mut self,
        proof: &P,
        binding: &DiskReadBinding,
    ) -> Result<(), Exception> {
        let owner = current_ordinary_execution_owner()?
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
        proof.validate_context(&self.context)?;
        proof.validate_manager(&self.manager, self.generation)?;
        proof.validate_id(&self.id)?;
        if !Arc::ptr_eq(&self.output.inner, &binding.output.inner)
            || owner.paged().is_none()
            || self.source_custody.is_some()
            || self.filling.iter().any(Option::is_some)
            || self.transfer.is_some()
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        let source = self
            .ordinary
            .as_mut()
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
        // The logical reservations are prepared cold but activated only for this
        // selected load. Sequential reads do not reserve simultaneous payloads.
        let host = source
            .host
            .take()
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
            .reserve(CachePoolUsage {
                host_bytes: self.reservation.host_bytes,
                ..CachePoolUsage::default()
            })
            .map_err(|cause| proof.error(CacheSourceError::ReservationAdmission(cause).into()))?;
        {
            let mut occupancy = self.reservation.inner.try_lock().map_err(|cause| {
                proof.error(
                    match cause {
                        TryLockError::WouldBlock => CacheSourceError::Busy,
                        TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                    }
                    .into(),
                )
            })?;
            if occupancy.is_some() {
                return Err(proof.error(CacheSourceError::Identity.into()));
            }
            *occupancy = Some(host);
        }
        self.transfer = Some(
            source
                .transfer
                .take()
                .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                .reserve(CachePoolUsage {
                    transfer_in_flight_bytes: self.reservation.host_bytes,
                    ..CachePoolUsage::default()
                })
                .map_err(|cause| {
                    proof.error(CacheSourceError::ReservationAdmission(cause).into())
                })?,
        );
        for index in 0..2 {
            self.filling[index] = Some(Filling::Ordinary(
                PreparedHostTransferWriter::ordinary_with_prepared_owner(
                    &self.shapes[index],
                    self.dtypes[index],
                    &mut source.attachments[index],
                )
                .map_err(|cause| proof.error(CacheSourceError::OrdinaryHostWriter(cause).into()))?,
            ));
        }
        Ok(())
    }
    pub(crate) fn ordinary_construction_call_controls<P: PreparedHostPromotion>(
        &self,
    ) -> Option<OrdinaryCallControls> {
        self.ordinary.as_ref()?;
        let fields = [
            size_of::<(&mut Self, &P, &DiskReadBinding)>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
            size_of::<MutexGuard<'_, Option<CachePoolReservation>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, Option<CachePoolReservation>>,
                    TryLockError<MutexGuard<'_, Option<CachePoolReservation>>>,
                >,
            >(),
            PreparedCachePoolReservation::admission_control_bytes()?.checked_mul(2)?,
            PreparedHostTransferWriter::ordinary_with_prepared_owner_control_bytes::<
                OrdinaryReadOwner,
            >(4)?
            .checked_mul(2)?,
        ];
        Some(OrdinaryCallControls {
            metadata_bytes: u64::try_from(
                fields
                    .into_iter()
                    .try_fold(size_of_val(&fields), usize::checked_add)?,
            )
            .ok()?,
            observed: self.observed?,
        })
    }
    /// Conservative transports of this selected shared read worker. Cold
    /// destination allocations remain in their retained context; native payload
    /// and constructor observer facts are composed separately by the caller.
    pub(crate) fn ordinary_call_controls<P: PreparedHostPromotion>(
        &self,
        publication_controls: usize,
    ) -> Option<OrdinaryCallControls> {
        let controls = self.ordinary_construction_call_controls::<P>()?;
        let frames = [
            PreparedDiskRead::control_bytes()?,
            super::control_bytes()?,
            super::super::operation::DiskReadOperation::control_bytes()?,
            self.source_read_controls,
            OrdinaryReadCacheHostSource::handoff_control_bytes::<P>(publication_controls)?,
            publication_controls.checked_mul(6)?,
            WorkspaceContext::metadata_source_bytes::<DiskReadFinishFailure>()?,
            WorkspaceContext::metadata_source_bytes::<DiskReadOperationFailure>()?
                .checked_mul(2)?,
            size_of::<Result<PreparedDiskRead, DiskReadFinishFailure>>(),
            size_of::<Result<DiskReadOperation, CacheSourceFailure>>(),
            size_of::<Result<(), DiskReadOperationFailure>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?;
        Some(OrdinaryCallControls {
            metadata_bytes: controls
                .metadata_bytes
                .checked_add(u64::try_from(bytes).ok()?)?,
            observed: controls.observed,
        })
    }
}
