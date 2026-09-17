//! Selected dense source permit for the shared filled-Host destination worker.
use super::*;
use crate::backend::runtime::residency::storage::filled_host;
use eredu_runtime::working_memory::{HostSourcePeakSelection, OriginalHostSourceReceipt};
pub(super) use filled_host::{Pending, SourceCause, SourceError};

struct Peak<'a> {
    capacity: &'a ForegroundDiskSourceCapacity,
    permit: RetirementCapacityPermit,
}
impl filled_host::HostFillPermit for Peak<'_> {
    type Owner = RetirementCapacityOwner<OriginalHostSourceReceipt>;
    fn validate(
        &self,
        selection: HostSourcePeakSelection,
        owner: usize,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        self.capacity
            .validate_permit(selection, owner, bytes, &self.permit)
    }
    fn create_quota(
        self,
        bytes: usize,
        receipt: OriginalHostSourceReceipt,
    ) -> Result<PreparedSubmissionGraphQuota<Self::Owner>, SourceCause> {
        PreparedSubmissionGraphQuota::try_new_with_retirement(bytes, receipt, self.permit)
            .map_err(|error| SourceCause::Arena(error.cause()))
    }
}
pub(super) fn control_bytes(
    plan: &PreparedHostTransferPlan<'_>,
) -> Result<usize, WorkingMemoryError> {
    filled_host::control_bytes::<Peak<'_>>(plan)
}
pub(super) fn begin<'a>(
    bank: &mut OriginalHostSourceBank,
    plan: PreparedHostTransferPlan<'a>,
    capacity: &'a ForegroundDiskSourceCapacity,
    permit: RetirementCapacityPermit,
    controls: &'a OriginalHostSourceCustody,
) -> Result<Pending, SourceError> {
    filled_host::begin(bank, plan, Peak { capacity, permit }, controls)
}
pub(super) fn finish(
    pending: Pending,
) -> Result<(ImmutableHostTransferBuffer, safemlx::AllocationInfo), SourceError> {
    filled_host::finish(pending)
}
