//! Initial hot storage returns to its actual file through the shared worker.
use super::super::host_demotion::PreparedHostEviction;
use super::*;
use crate::backend::nn::workspace::{
    OrdinaryCallControls, OrdinaryNativeControls, OrdinaryPagedCause, OriginalPagedHostReturn,
    PagedHostStoreDeclaration,
};

#[derive(Clone, Copy)]
enum InitialDiskMode {
    Original,
    Ordinary,
}

pub(crate) struct PreparedInitialDiskReturn {
    replaced: Option<CacheBlockArrays>,
    backing: DiskLocation,
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    reservation: Option<CachePoolReservation>,
    prepared_reservation: Option<eredu_runtime::cache::PreparedCachePoolReservation>,
    retirement: Option<super::super::device_retirement::DeviceRetirement>,
    attempted: bool,
    publication_controls: usize,
    context: WorkspaceContext,
    funding: Option<HostMetadataFunding>,
}
impl CacheBlockSourceLoan<'_> {
    pub(crate) fn prepare_initial_disk_return(
        &self,
        declaration: &PagedHostStoreDeclaration<'_>,
        reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedInitialDiskReturn, CacheSourceFailure> {
        self.prepare_initial_disk_return_with(
            declaration,
            reservations,
            context,
            InitialDiskMode::Original,
        )
    }
    pub(crate) fn prepare_initial_ordinary_disk_return(
        &self,
        declaration: &PagedHostStoreDeclaration<'_>,
        reservations: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedInitialDiskReturn, CacheSourceFailure> {
        self.prepare_initial_disk_return_with(
            declaration,
            reservations,
            context,
            InitialDiskMode::Ordinary,
        )
    }
    fn prepare_initial_disk_return_with(
        &self,
        declaration: &PagedHostStoreDeclaration<'_>,
        reservations: usize,
        context: &WorkspaceContext,
        mode: InitialDiskMode,
    ) -> Result<PreparedInitialDiskReturn, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        context
            .charge_metadata(
                match mode {
                    InitialDiskMode::Original => return_disk::control_bytes()
                        .and_then(|n| n.checked_add(self.publication_controls.checked_mul(3)?)),
                    InitialDiskMode::Ordinary => Some(0),
                }
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        PreparedInitialDiskReturn,
                        &Self,
                        &PagedHostStoreDeclaration<'_>,
                        usize,
                        InitialDiskMode,
                        &WorkspaceContext,
                        Result<PreparedInitialDiskReturn, CacheSourceFailure>,
                    )>())
                })
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if matches!(mode, InitialDiskMode::Original) {
            context
                .charge_metadata(size_of::<(
                    &mut PreparedInitialDiskReturn,
                    &OriginalPagedHostReturn<'_, '_>,
                    [&Array; 2],
                    Result<(), Exception>,
                )>())
                .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        }
        let row = self
            .blocks()
            .find(|row| row.id() == declaration.id())
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if !declaration.is_retained()
            || row.phase() != CacheStoragePhase::Device
            || row.disk().and_then(|disk| disk.file_source()).is_none()
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let backing = row
            .record
            .disk()
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .clone();
        let prepared = self
            .pool()
            .prepare_reservation_population(reservations, context)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let (reservation, prepared_reservation, retirement) = match mode {
            InitialDiskMode::Original => (
                Some(
                    prepared
                        .reserve(CachePoolUsage::default())
                        .map_err(|cause| {
                            CacheSourceFailure::metadata(context.metadata_source(cause), context)
                        })?,
                ),
                None,
                None,
            ),
            InitialDiskMode::Ordinary => (
                None,
                Some(prepared),
                Some(super::super::device_retirement::DeviceRetirement::prepare(
                    context,
                )?),
            ),
        };
        Ok(PreparedInitialDiskReturn {
            replaced: None,
            backing,
            id: declaration.id().clone(),
            manager: self.manager().clone(),
            generation: self.generation(),
            reservation,
            prepared_reservation,
            retirement,
            attempted: false,
            publication_controls: self.publication_controls,
            context: context.clone(),
            funding: context.metadata_funding(),
        })
    }
}
impl PreparedInitialDiskReturn {
    pub(crate) fn run(
        &mut self,
        proof: &OriginalPagedHostReturn<'_, '_>,
        arrays: [&Array; 2],
    ) -> Result<(), Exception> {
        let source = proof.source();
        source.validate_manager(&self.manager, self.generation)?;
        if self.retirement.is_some()
            || self.replaced.is_some()
            || !self.context.shares_trace(source.context())
        {
            return Err(source.error(CacheSourceError::Identity));
        }
        self.replaced = Some(return_disk::return_device(
            &self.manager,
            &self.id,
            &self.backing,
            arrays,
            self.reservation.as_mut().expect("prepared Original return"),
            proof,
        )?);
        Ok(())
    }

    /// Return this exact retained Device row to its authenticated durable file.
    /// A failed attachment keeps the reservation with this one-use destination.
    pub(crate) fn run_ordinary<P: PreparedHostEviction<Cause = OrdinaryPagedCause>>(
        &mut self,
        proof: &P,
        context: &WorkspaceContext,
    ) -> Result<(), Exception> {
        proof.validate_manager(&self.manager, self.generation)?;
        if self.attempted
            || self.retirement.is_none()
            || proof.id() != &self.id
            || !self.context.shares_trace(context)
            || self
                .funding
                .as_ref()
                .zip(context.metadata_funding().as_ref())
                .is_none_or(|(expected, actual)| !expected.same_account(actual))
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        for array in proof.arrays() {
            if array
                .try_observe_availability()
                .map_err(|cause| proof.error(cause.into()))?
                != Some(true)
            {
                return Err(proof.error(CacheSourceError::PendingStorage.into()));
            }
        }
        self.reservation = Some(
            self.prepared_reservation
                .take()
                .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                .reserve(CachePoolUsage::default())
                .map_err(|cause| {
                    proof.error(CacheSourceError::ReservationAdmission(cause).into())
                })?,
        );
        self.attempted = true;
        self.retirement
            .as_mut()
            .expect("prepared ordinary return")
            .attach(proof.arrays())
            .map_err(|cause| proof.error(cause.into()))?;
        self.replaced = Some(return_disk::return_device_prepared(
            &self.manager,
            &self.id,
            &self.backing,
            self.reservation.as_mut().expect("admitted ordinary return"),
            proof,
        )?);
        self.retirement
            .as_mut()
            .expect("prepared ordinary return")
            .publish(&mut self.reservation);
        self.replaced = None;
        Ok(())
    }

    pub(crate) fn reclaim_replaced_device(&mut self) {
        if let Some(retirement) = &mut self.retirement {
            retirement.reclaim();
        }
    }

    pub(crate) fn ordinary_source_controls<P: PreparedHostEviction>(
        &self,
    ) -> Option<OrdinaryCallControls> {
        self.retirement.as_ref()?;
        let frames = [
            return_disk::prepared_control_bytes::<P>()?,
            self.publication_controls.checked_mul(3)?,
            eredu_runtime::cache::PreparedCachePoolReservation::admission_control_bytes()?,
            Array::ordinary_availability_control_bytes()?.checked_mul(2)?,
            size_of::<(&mut Self, &P, &WorkspaceContext)>(),
            size_of::<Result<(), Exception>>(),
            size_of::<std::array::IntoIter<&Array, 2>>(),
            size_of::<(Option<&HostMetadataFunding>, Option<HostMetadataFunding>)>(),
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

use eredu_nn::workspace::WorkspaceMetadataAllocation;
