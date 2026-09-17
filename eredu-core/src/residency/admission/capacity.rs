use super::*;

/// Borrowed typed capacity failure, shared by ordinary and prepared owners.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyCapacityRef<'a> {
    /// Exact requested ID from the failure's still-retained owner.
    pub requested: &'a OffloadUnitId,
    /// Target tier.
    pub tier: MemoryTier,
    /// Complete batch physical request.
    pub required_bytes: u64,
    /// Actual configured budget.
    pub budget_bytes: u64,
    /// Actual charged bytes at refusal.
    pub resident_bytes: u64,
    blockers: Blockers<'a>,
}
#[derive(Clone, Copy, Debug)]
enum Blockers<'a> {
    Ordinary(&'a [ResidencyBlocker]),
    Prepared(&'a ResidencyPlanSource, &'a [ResidencyBlockerRow]),
}
/// One borrowed actual blocker; no ID copy or fallback lookup is performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyBlockerRef<'a> {
    /// Actual retained logical ID.
    pub id: &'a OffloadUnitId,
    /// Lifetime policy forbids eviction.
    pub pinned: bool,
    /// Current live lease count.
    pub in_use: u64,
    /// Current named window protects the copy.
    pub active_window: bool,
    /// The complete batch protects the copy.
    pub request_protected: bool,
}
impl<'a> ResidencyCapacityRef<'a> {
    /// Borrows blockers in the existing stable identifier order.
    pub fn blockers(self) -> impl ExactSizeIterator<Item = ResidencyBlockerRef<'a>> {
        let count = match self.blockers {
            Blockers::Ordinary(rows) => rows.len(),
            Blockers::Prepared(_, rows) => rows.len(),
        };
        (0..count).map(move |index| match self.blockers {
            Blockers::Ordinary(rows) => {
                let row = &rows[index];
                ResidencyBlockerRef {
                    id: &row.id,
                    pinned: row.pinned,
                    in_use: row.in_use,
                    active_window: row.active_window,
                    request_protected: row.request_protected,
                }
            }
            Blockers::Prepared(source, rows) => {
                let row = rows[index];
                ResidencyBlockerRef {
                    id: source.id(row.plan).expect("actual blocker source"),
                    pinned: row.pinned,
                    in_use: row.in_use,
                    active_window: row.active_window,
                    request_protected: row.request_protected,
                }
            }
        })
    }
}
impl ResidencyLedgerError {
    /// Exact capacity classification; unrelated failures are never retriable.
    pub fn capacity(&self) -> Option<ResidencyCapacityRef<'_>> {
        let Self::BudgetExhausted {
            requested,
            tier,
            required_bytes,
            budget_bytes,
            resident_bytes,
            blocking_units,
        } = self
        else {
            return None;
        };
        Some(ResidencyCapacityRef {
            requested,
            tier: *tier,
            required_bytes: *required_bytes,
            budget_bytes: *budget_bytes,
            resident_bytes: *resident_bytes,
            blockers: Blockers::Ordinary(blocking_units),
        })
    }
}
impl PreparedResidencyAdmissionFailure {
    /// Borrows the exact same typed capacity cause without allocating IDs.
    pub fn capacity(&self) -> Option<ResidencyCapacityRef<'_>> {
        let Failure::Budget {
            plan,
            tier,
            required,
            budget,
            resident,
        } = self.cause
        else {
            return None;
        };
        Some(ResidencyCapacityRef {
            requested: self
                .storage
                .source
                .id(plan)
                .expect("actual requested source"),
            tier,
            required_bytes: required,
            budget_bytes: budget,
            resident_bytes: resident,
            blockers: Blockers::Prepared(&self.storage.source, &self.storage.blockers),
        })
    }
}
