//! Explicit neutral destinations for ordered acquisition and atomic admission.
use super::*;
use std::{alloc::Layout, sync::Arc};

mod capacity;
pub use capacity::{ResidencyBlockerRef, ResidencyCapacityRef};
mod batch;
pub use batch::{ResidencyBatchError, ResidencyBatchFailure};
mod storage;
use storage::{Candidate, Failure};
mod reservation;
pub use reservation::ResidencyProtection;
pub use storage::{
    PreparedResidencyAdmissionFailure, ResidencyAdmissionPreparationError,
    ResidencyAdmissionStorage, ResidencyBlockerRow, ResidencyEvictedRow, ResidencyReservationRow,
};
#[cfg(test)]
mod tests;

/// Retained identity and immutable declarations of one actual residency ledger.
///
/// Equal independently constructed ledgers do not share this identity. This
/// owns no mutable ledger, backend storage, native completion or admission lease.
#[derive(Clone, Debug)]
pub struct ResidencyPlanSource {
    plan: Arc<OffloadPlan>,
}
impl ResidencyPlanSource {
    /// Number of actual planned units, in stable identifier order.
    pub fn len(&self) -> usize {
        self.plan.units().len()
    }
    /// Whether this actual plan has no units.
    pub fn is_empty(&self) -> bool {
        self.plan.units().is_empty()
    }
    /// Exact retained identifier at a planned ordinal.
    pub fn id(&self, ordinal: usize) -> Option<&OffloadUnitId> {
        self.plan.units().get(ordinal).map(OffloadUnitSpec::id)
    }
    /// Resolves an identifier without copying it or allocating lookup storage.
    pub fn ordinal(&self, id: &OffloadUnitId) -> Option<usize> {
        self.plan
            .units()
            .binary_search_by(|unit| unit.id().cmp(id))
            .ok()
    }
    /// Largest actual planned identifier, excluding allocator rounding.
    pub fn maximum_id_bytes(&self) -> usize {
        self.plan
            .units()
            .iter()
            .map(|unit| unit.id().as_str().len())
            .max()
            .unwrap_or(0)
    }
    /// Whether two retained source aliases name the same actual ledger plan.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.plan, &other.plan)
    }
    /// Fixed source-alias representation; this does not price its shared payload.
    pub const fn control_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
    /// Requested shared-header layout for the supported Rust 1.98 ArcInner.
    ///
    /// Existing plan vector/ID payload and allocator rounding remain separate.
    /// The caller must bind this representation to its actual toolchain before
    /// treating it as a host-control bound.
    pub fn shared_plan_layout() -> Option<Layout> {
        Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .align_to(2)
            .ok()?
            .extend(Layout::new::<OffloadPlan>())
            .ok()
            .map(|(layout, _)| layout.pad_to_align())
    }
}

impl ResidencyLedger {
    /// Retains the immutable source already owned by this exact ledger.
    /// This only increments an existing shared reference; it creates no map.
    pub fn plan_source(&self) -> ResidencyPlanSource {
        ResidencyPlanSource {
            plan: Arc::clone(&self.plan),
        }
    }
    /// Compares a still-retained source with this ledger's actual source owner.
    pub fn matches_plan_source(&self, source: &ResidencyPlanSource) -> bool {
        Arc::ptr_eq(&self.plan, &source.plan)
    }

    /// Validates the same ordered batch using caller-provided index scratch.
    ///
    /// Target refusal comes first. For valid scratch, each position's duplicate
    /// check precedes that position's unknown-unit check. The input is unchanged.
    /// A failure borrows its actual ID and does not allocate an error string.
    pub fn validate_batch_in<'a>(
        &self,
        ids: &'a [OffloadUnitId],
        tier: MemoryTier,
        order: &mut [usize],
    ) -> Result<(), ResidencyBatchFailure<'a>> {
        batch::validate(self, ids.len(), |index| &ids[index], tier, order)
    }
}
