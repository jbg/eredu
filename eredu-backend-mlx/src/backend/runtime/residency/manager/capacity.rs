//! Immutable declaration ceilings; neither a current inventory nor a grant.
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorageInspectionError;
use std::sync::TryLockError;

/// Bound to the exact manager instance whose immutable declarations were read.
/// Pending transfer owners, source/catalog metadata and manager bookkeeping are
/// outside these map/source counts. A separate complete current visit is required.
pub(crate) struct PreparedWeightOwnerSlotBounds<'a> {
    source: &'a ResidencyManager,
    array_slots: usize,
    host_slots: usize,
    source_slots: usize,
}
impl<'a> PreparedWeightOwnerSlotBounds<'a> {
    pub(crate) fn source(&self) -> &'a ResidencyManager {
        self.source
    }
    pub(crate) fn matches(&self, source: &ResidencyManager) -> bool {
        self.source.inner.ptr_eq(&source.inner)
    }
    pub(crate) fn array_slots(&self) -> usize {
        self.array_slots
    }
    pub(crate) fn host_slots(&self) -> usize {
        self.host_slots
    }
    pub(crate) fn source_slots(&self) -> usize {
        self.source_slots
    }
}

impl ResidencyManager {
    /// Counts every declared binding, including currently unloaded units and
    /// aliases. Both device and host maps may retain one owner per binding.
    /// Uses only the state try-lock and source-owned immutable count diagnostics;
    /// no source visitation, clone, native operation or housekeeping occurs.
    pub(crate) fn prepare_owner_slot_bounds(
        &self,
    ) -> Result<Option<PreparedWeightOwnerSlotBounds<'_>>, RetainedStorageInspectionError> {
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(RetainedStorageInspectionError::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(ResidencyError::StatePoisoned.into()),
        };
        let bindings = state.control.units().try_fold(0usize, |n, unit| {
            n.checked_add(unit.bindings().len())
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "weight owner binding slots",
                })
        })?;
        let mut source_slots = 0usize;
        let mut complete = true;
        for source in self.inner.sources.ordinary_sources() {
            match source
                .source_storage_slot_bound()
                .map_err(ResidencyError::from)?
            {
                Some(n) => {
                    source_slots =
                        source_slots
                            .checked_add(n)
                            .ok_or(ResidencyError::ArithmeticOverflow {
                                context: "weight source owner slots",
                            })?
                }
                None => complete = false,
            }
        }
        Ok(complete.then_some(PreparedWeightOwnerSlotBounds {
            source: self,
            array_slots: bindings,
            host_slots: if matches!(&self.inner.sources, ResidencySources::Original(_)) {
                // Immutable source rows remain alive alongside ledger host rows.
                bindings
                    .checked_mul(2)
                    .ok_or(ResidencyError::ArithmeticOverflow {
                        context: "immutable host source owner slots",
                    })?
            } else {
                bindings
            },
            source_slots,
        }))
    }
}
