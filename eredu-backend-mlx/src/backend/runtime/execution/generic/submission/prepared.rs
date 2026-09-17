//! Two final unit nodes prepared before any selected unit is constructed.
use super::*;
use crate::backend::submission_recovery::observed::{ObservedRecovery, PreparedObservedRecovery};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use std::mem::size_of;

type Ready<U> = PreparedObservedRecovery<UnitRetention<U>, OriginalOperationMetadataCustody>;

/// Concrete unit slot. Its provider must price the complete finite population
/// against the actual selected policy and request before constructing any slot.
/// This storage alone supplies neither a quote nor permission for native work.
pub(crate) struct PreparedUnit<U: 'static> {
    ready: Ready<U>,
    resources: OrdinaryRetirement<UnitResourcesSlot<U>>,
}

impl<U: 'static> PreparedUnit<U> {
    /// Both independently retiring allocations exist before the first unit.
    /// The existing guard is custody only; no native scope or callback is entered.
    pub(crate) fn new(controls: OriginalOperationMetadataCustody) -> Self {
        let resources = OrdinaryRetirement::new(UnitResourcesSlot {
            value: None,
            cleanup: None,
            _custody: Some(controls.clone()),
        });
        Self {
            ready: Ready::new(controls),
            resources,
        }
    }

    /// Fill both prepaid nodes by move after authenticating the exact current
    /// original native scope. No Box/Vec/Rc is allocated here.
    pub(super) fn activate(
        mut self,
        unit: MlxModule<U>,
        transfer: MlxUnitTransfer,
        observer: safemlx::OriginalScopeObserver,
    ) -> ObservedRecovery<UnitRetention<U>, OriginalOperationMetadataCustody> {
        self.resources.value = Some(UnitResources {
            unit,
            _transfer: transfer,
        });
        self.ready.activate(
            UnitRetention {
                resources: Some(self.resources),
                event: None,
                failed: Cell::new(false),
            },
            observer,
        )
    }

    /// Actual two allocated node layouts and named move/return controls. Dynamic
    /// module/transfer children, pending/request buffers and native arena use are
    /// separate selected-policy facts; this does not claim Complete coverage.
    pub(crate) fn control_bytes() -> Option<u64> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<UnitResourcesSlot<U>>(),
            size_of::<UnitResources<U>>(),
            size_of::<MlxModule<U>>(),
            size_of::<MlxUnitTransfer>(),
            size_of::<UnitRetention<U>>(),
            size_of::<Option<OrdinaryRetirement<UnitResourcesSlot<U>>>>(),
            size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<Option<OriginalOperationMetadataCustody>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        Ready::<U>::control_bytes::<safemlx::error::Exception>()?
            .checked_add(OrdinaryRetirement::<UnitResourcesSlot<U>>::control_bytes()?)?
            .checked_add(UnitRecovery::<U>::control_bytes()?)?
            .checked_add(ResidentTransfer::original_retirement_control_bytes()?)?
            .checked_add(
                crate::backend::submission_recovery::observed::CompletedObservedRetention::<
                    UnitRetention<U>,
                    OriginalOperationMetadataCustody,
                >::release_with_control_bytes::<fn(&mut UnitRetention<U>)>()?,
            )?
            .checked_add(u64::try_from(fixed).ok()?)
    }
}
