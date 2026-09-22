//! Explicit loans of finite selected-operation storage; no installed/TLS lookup.
use super::{
    PreparedClosureIds, PreparedHostMaterialization, PreparedResidentTransfer,
    PreparedTransferObservation,
};
use crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots;
use crate::backend::submission_recovery::observed::bank::PreparedOperationBank;

mod foreground_disk;
pub(crate) use foreground_disk::{
    ForegroundDiskPopulation, ForegroundDiskSlotError, ForegroundDiskSourceCapacity,
    ForegroundDiskSourceSeries, ForegroundDiskSubsetCeiling, ForegroundDiskWindowPlan,
    PreparedForegroundDiskSlots,
};

/// Stack-only loan of the actual read service and its once-only final host rows.
pub(crate) struct OriginalHostPublicationSlots<'a> {
    pub(crate) publication: &'a mut Option<super::PreparedHostPublication>,
    pub(crate) reads:
        &'a crate::backend::runtime::residency::dense_stream::BackgroundHostReadService,
}
impl OriginalHostPublicationSlots<'_> {
    fn reborrow(&mut self) -> OriginalHostPublicationSlots<'_> {
        OriginalHostPublicationSlots {
            publication: self.publication,
            reads: self.reads,
        }
    }
}

/// Borrowed backing and metadata of the authenticated enclosing native role.
/// The descriptor still supplies its exact source program and the caller's
/// validation slots. This loan creates no reservation or observer authority.
#[derive(Clone, Copy)]
pub(crate) struct OriginalMaterializedLoan<'a> {
    pub(crate) budget: &'a safemlx::OriginalBufferBudget,
    pub(crate) funding: &'a eredu_nn::workspace::HostMetadataFunding,
}

/// The closed request-bound projection authenticates the current registered
/// role before issuing this loan. Holding these references grants no native
/// producer permission; every operation still validates its exact current role.
pub(crate) struct OriginalResidencySlots<'a> {
    pub(crate) controller: &'a mut Option<super::PreparedControllerAttempt>,
    pub(crate) closure_ids: &'a mut Option<PreparedClosureIds>,
    pub(crate) closure: &'a mut [eredu_runtime::residency::ResidencyClosureSlot],
    pub(crate) transfers: &'a mut PreparedOperationBank<PreparedResidentTransfer>,
    pub(crate) observations: &'a mut PreparedOperationBank<PreparedTransferObservation>,
    pub(crate) host_materializations: &'a mut PreparedOperationBank<PreparedHostMaterialization>,
    // An exact request loan, independent of native permission. Present only
    // after the operation registry authenticates the current accepted role.
    pub(crate) reservation: Option<&'a eredu_runtime::working_memory::WorkingMemoryReservation>,
    pub(crate) foreground_disk: &'a mut PreparedForegroundDiskSlots,
    pub(crate) materialization: OriginalMaterializationSlots<'a>,
    pub(crate) materialized_recipe: Option<OriginalMaterializedLoan<'a>>,
    pub(crate) background_host: Option<OriginalHostPublicationSlots<'a>>,
}
impl OriginalResidencySlots<'_> {
    pub(crate) fn reborrow(&mut self) -> OriginalResidencySlots<'_> {
        OriginalResidencySlots {
            controller: self.controller,
            closure_ids: self.closure_ids,
            closure: self.closure,
            transfers: self.transfers,
            observations: self.observations,
            host_materializations: self.host_materializations,
            reservation: self.reservation,
            foreground_disk: self.foreground_disk,
            materialization: self.materialization.reborrow(),
            materialized_recipe: self.materialized_recipe,
            background_host: self
                .background_host
                .as_mut()
                .map(OriginalHostPublicationSlots::reborrow),
        }
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        size_of::<Self>()
            .checked_add(size_of::<&mut Self>())?
            .checked_add(size_of::<Option<Self>>())?
            .checked_add(OriginalMaterializationSlots::control_bytes()?)
    }
}
