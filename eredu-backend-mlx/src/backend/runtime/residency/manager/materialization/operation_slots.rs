//! Complete typed slots for explicitly supplied original-operation banks.
//! The closed selected caller supplies counts and original custody before work.
//! No lookup, native observer, stream, lease, or payload is constructed here.
use super::*;
use crate::backend::submission_recovery::observed::{
    bank::{BankLayout, PreparedOperationBank},
    operation::OperationRecovery,
    PreparedObservedRecovery,
};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use safemlx::{error::Exception, OriginalScopeObserver};

/// One final node; its payload remains absent until the selected caller activates it.
pub(crate) struct PreparedHostMaterialization {
    ready: PreparedObservedRecovery<HostMaterialization, OriginalOperationMetadataCustody>,
}
impl PreparedHostMaterialization {
    /// Called only during the closed caller's prepaid bank construction.
    /// Uses the existing Box abort-on-OOM contract; custody remains in the node.
    pub(crate) fn new(custody: OriginalOperationMetadataCustody) -> Self {
        Self {
            ready: PreparedObservedRecovery::new(custody),
        }
    }

    /// F/E must be the actual factory and its owning failure type. This prices
    /// named controls, not dynamic payloads or a proof of the supplied count.
    pub(crate) fn bank_layout<F, E>(count: usize) -> Option<BankLayout> {
        let node = PreparedObservedRecovery::<HostMaterialization, OriginalOperationMetadataCustody>
            ::control_bytes::<Exception>()?;
        let node = node.checked_add(u64::try_from(size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>()).ok()?)?;
        let dispatch =
            OperationRecovery::<HostMaterialization, OriginalOperationMetadataCustody>::control_bytes()?;
        let native = u64::try_from(OriginalScopeObserver::control_bytes()?).ok()?;
        let slot = node.checked_add(dispatch)?.checked_add(native)?;
        PreparedOperationBank::<Self>::layout::<F, E>(count, slot)
    }

    /// The caller authenticated the current innermost role before any producer.
    /// Moving the final prepared node never allocates or creates a child Scope.
    pub(super) fn activate(
        self,
        value: HostMaterialization,
        observer: OriginalScopeObserver,
    ) -> OperationRecovery<HostMaterialization, OriginalOperationMetadataCustody> {
        OperationRecovery::original(self.ready, value, observer)
    }
}
