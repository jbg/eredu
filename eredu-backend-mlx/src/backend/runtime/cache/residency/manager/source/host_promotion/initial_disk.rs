//! Initial hot storage returns to its actual file through the shared worker.
use super::*;
use crate::backend::nn::workspace::{OriginalPagedHostReturn, PagedHostStoreDeclaration};

pub(crate) struct PreparedInitialDiskReturn {
    replaced: Option<CacheBlockArrays>,
    backing: DiskLocation,
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    reservation: CachePoolReservation,
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
        let fail = |cause| CacheSourceFailure::source(cause, context);
        declaration.validate_loan(self, context).map_err(fail)?;
        context
            .charge_metadata(
                return_disk::control_bytes()
                    .and_then(|n| n.checked_add(self.publication_controls.checked_mul(3)?))
                    .and_then(|n| {
                        n.checked_add(size_of::<(
                            PreparedInitialDiskReturn,
                            &Self,
                            &PagedHostStoreDeclaration<'_>,
                            usize,
                            &WorkspaceContext,
                            Result<PreparedInitialDiskReturn, CacheSourceFailure>,
                        )>())
                    })
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        context
            .charge_metadata(size_of::<(
                &mut PreparedInitialDiskReturn,
                &OriginalPagedHostReturn<'_, '_>,
                [&Array; 2],
                Result<(), Exception>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let row = self
            .blocks()
            .find(|row| row.id() == declaration.id())
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if !declaration.is_retained()
            || row.phase() != CacheStoragePhase::Device
            || row.disk().and_then(|disk| disk.live_file()).is_none()
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let backing = row
            .record
            .disk()
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .clone();
        let reservation = self
            .pool()
            .prepare_reservation_population(reservations, context)
            .map_err(|cause| CacheSourceFailure::metadata(context.metadata_source(cause), context))?
            .reserve(CachePoolUsage::default())
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        Ok(PreparedInitialDiskReturn {
            replaced: None,
            backing,
            id: declaration.id().clone(),
            manager: self.manager().clone(),
            generation: self.generation(),
            reservation,
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
        if self.replaced.is_some() || !self.context.shares_trace(source.context()) {
            return Err(source.error(CacheSourceError::Identity));
        }
        self.replaced = Some(return_disk::return_device(
            &self.manager,
            &self.id,
            &self.backing,
            arrays,
            &mut self.reservation,
            proof,
        )?);
        Ok(())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
