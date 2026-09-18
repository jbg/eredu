//! Completed-demand construction through the existing exact window worker.
use super::*;
use crate::backend::runtime::residency::manager::SupplementaryResidencySource;
use eredu_runtime::{residency::ResidencyClosureSlot, working_memory::{OriginalHostSourceBank,
    OriginalHostSourceCustody, OriginalHostSourceReceipt, WorkingMemoryReservation}};

pub(in crate::backend::runtime::execution::generic::original_operations) struct PreparedSelectedResidency {
    window: PreparedResidencyAttempt,
    closure: Vec<ResidencyClosureSlot>,
    _catalog: NameCatalogOwner,
    _source: SupplementaryResidencySource,
    // Both reached constructor receipts outlive their allocations, including
    // cancellation before any native submission. Neither can refund its bank.
    _scratch_receipt: OriginalHostSourceReceipt,
    _window_receipt: OriginalHostSourceReceipt,
    _custody: OriginalHostSourceCustody,
}
impl PreparedSelectedResidency {
    pub(in crate::backend::runtime::execution::generic::original_operations) fn scratch_bytes(source:&SupplementaryResidencySource)->Option<u64> {
        let units=source.source().controller_units;
        let parts = [ResidencyClosureSlot::layout(units)?.size(), size_of::<Self>(),
            size_of::<Vec<ResidencyClosureSlot>>(), size_of::<OriginalHostSourceCustody>(),
            size_of::<OriginalHostSourceReceipt>(), size_of::<OriginalHostSourceReceipt>(),
            size_of::<OriginalOperationMetadataCustody>(), size_of::<WindowPopulation>(),
            size_of::<Result<WindowPopulation,crate::backend::runtime::residency::manager::OperationSourceFailure>>(),
            size_of::<Result<Self,Error>>(), size_of::<Result<(),Error>>(),
            size_of::<super::super::super::OriginalSelectedResidencyAttempt>(),
            size_of::<Result<super::super::super::OriginalSelectedResidencyAttempt,Error>>(),
            size_of::<(&ResidencyManager,&SupplementaryResidencySource,&[OffloadUnitId],&mut OriginalHostSourceBank)>(),
            size_of::<Result<OriginalHostSourceReceipt,eredu_runtime::working_memory::OriginalHostSourceError>>(),
            super::super::super::OriginalSelectedResidencyAccess::control_bytes()?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<SelectedPreparationFailure>()?];
        let fixed=u64::try_from(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)?).ok()?;
        fixed.checked_add(crate::backend::runtime::residency::manager::OriginalResidencySource::constructor_storage_bytes(units,1)?)?
            .checked_add(u64::try_from(source.projection_control_bytes()?).ok()?)
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn window_bytes(source:WindowPopulation)->Option<u64> {
        // Same transfer, controller, closure and catalog factories as static
        // windows; source counts describe the selected canonical closure.
        PreparedResidencyAttempt::control_bytes(source)?.checked_add(named_error_control_bytes()?)?
            .checked_add(u64::try_from(source.named.catalog_requested_bytes).ok()?)?
            .checked_add(u64::try_from(size_of::<NameCatalogOwner>()
                .checked_add(size_of::<Result<NameCatalogOwner,crate::backend::runtime::residency::manager::NamePreparationError>>())?
                .checked_add(size_of::<Result<PreparedResidencyAttempt,Error>>())?).ok()?)
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn prepare(
        manager:&ResidencyManager, source:&SupplementaryResidencySource, roots:&[OffloadUnitId],
        bank:&mut OriginalHostSourceBank, custody:OriginalHostSourceCustody,
        reservation:Option<&WorkingMemoryReservation>,
    )->Result<Self,Error> {
        Self::prepare_with_disk(manager,source,roots,bank,custody,reservation,None)
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn prepare_with_disk(
        manager:&ResidencyManager,source:&SupplementaryResidencySource,roots:&[OffloadUnitId],
        bank:&mut OriginalHostSourceBank,custody:OriginalHostSourceCustody,
        reservation:Option<&WorkingMemoryReservation>,
        disk:Option<(&eredu_runtime::working_memory::WorkingMemoryPool,
            &ForegroundDiskSourceCapacity,&eredu_nn::workspace::HostMetadataFunding)>,
    )->Result<Self,Error> {
        if !bank.belongs_to_source(&custody) { return Err(identity()); }
        if let Some((pool,capacity,_))=disk {
            if !custody.same_source(capacity.custody())
                || !custody.metadata_custody().matches_domain(pool.shared_storage_domain())
                || !source.foreground().is_some_and(|source|capacity.matches_source(source.source())) {
                return Err(identity());
            }
            capacity.validate_account(reservation).map_err(memory)?;
        }
        let metadata=custody.metadata_custody();
        // These are actual retained Host sources. Foreground-disk construction
        // must supply its distinct source plan/bank and is never silently used.
        let units=source.source().controller_units;
        let scratch_receipt=bank.try_debit(Self::scratch_bytes(source).ok_or_else(overflow)?)
            .map_err(|cause|failure(SelectedCause::Debit(cause),&custody))?;
        manager.validate_supplementary_source(source).map_err(|cause|failure(SelectedCause::Source(cause),&custody))?;
        if disk.is_none(){manager.validate_supplementary_host(source).map_err(|_|identity())?;}
        let mut closure=Vec::new();
        closure.try_reserve_exact(units).map_err(|cause|reserve_error(cause,&metadata))?;
        closure.resize(units,ResidencyClosureSlot::default());
        let population=manager.selected_operation_population(source,roots,&mut closure)
            .map_err(|cause|failure(SelectedCause::Source(cause),&custody))?;
        // The exact selected read descriptor is built with counted metadata;
        // its native output/source capacity was supplied by the invocation.
        let disk_plan=match disk {
            Some((pool,_,funding))=>Some(ForegroundDiskWindowPlan::new_with_metadata(
                manager,pool,population,roots,&mut closure,Some(funding))?),
            None=>None,
        };
        let window_bytes=Self::window_bytes(population).and_then(|bytes|bytes.checked_add(
            disk_plan.as_ref().map(ForegroundDiskWindowPlan::attempt_control_bytes).unwrap_or(Some(0))?))
            .ok_or_else(overflow)?;
        let window_receipt=bank.try_debit(window_bytes)
            .map_err(|cause|failure(SelectedCause::Debit(cause),&custody))?;
        let catalog=manager.prepare_name_catalog(roots,&mut closure,metadata.clone())
            .map_err(|cause|named_error(cause.into_source(),&metadata))?;
        let window=PreparedResidencyAttempt::new(population,MemoryTier::Device,None,&metadata,None,
            manager,&catalog,roots,&mut closure,Default::default(),&[],None,
            disk.zip(disk_plan.as_ref()).map(|((pool,capacity,_),plan)|(plan,pool,capacity)),reservation)?;
        Ok(Self{window,closure,_catalog:catalog,_source:source.clone(),_scratch_receipt:scratch_receipt,
            _window_receipt:window_receipt,_custody:custody})
    }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn residency<'a>(
        &'a mut self,reservation:Option<&'a WorkingMemoryReservation>,
    )->OriginalResidencySlots<'a> { self.window.residency(&mut self.closure,reservation) }
}
#[derive(Debug,thiserror::Error)]
enum SelectedCause {
    #[error("selected residency source: {0}")]
    Source(#[source] crate::backend::runtime::residency::manager::OperationSourceFailure),
    #[error("selected residency constructor: {0}")]
    Debit(#[source] eredu_runtime::working_memory::OriginalHostSourceError),
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct SelectedPreparationFailure {
    #[source] cause:SelectedCause,
    _custody:OriginalHostSourceCustody,
}
fn failure(cause:SelectedCause,custody:&OriginalHostSourceCustody)->Error {
    Error::with_original_control_source(eredu_core::BackendFailure::from_error(
        SelectedPreparationFailure{cause,_custody:custody.clone()}),false)
}
