//! Explicit eviction with a borrowed identifier and fixed failure values.
use super::*;
use std::mem::{size_of, size_of_val};

/// Allocation-free explicit-eviction refusal. The caller retains the actual
/// identifier/tier loan or source owner; equality of byte counts grants nothing.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum ResidencyEvictionError {
    /// Disk has no materialized copy.
    #[error("evict requires a host or device tier")]
    InvalidTargetTier,
    /// The supplied identifier is not in this ledger.
    #[error("unknown residency unit")]
    UnknownUnit,
    /// The actual unit has a pinned lifetime policy.
    #[error("cannot evict pinned residency unit")]
    Pinned,
    /// Live ownership leases still protect the copy.
    #[error("cannot evict residency unit with {pin_count} live leases")]
    InUse {
        /// Actual retained lease count.
        pin_count: u64,
    },
    /// The original caller must settle its actual transfer before removal.
    #[error("resident copy has an unresolved transfer")]
    PendingTransfer,
    /// The checked subtraction would violate the ledger accounting.
    #[error("resident copy removal accounting is inconsistent")]
    Accounting,
}
impl ResidencyEvictionError {
    pub(super) fn into_owned(self, id: &OffloadUnitId, tier: MemoryTier) -> ResidencyLedgerError {
        match self {
            Self::InvalidTargetTier => {
                ResidencyLedgerError::InvalidTargetTier { operation: "evict" }
            }
            Self::UnknownUnit => ResidencyLedgerError::UnknownUnit { id: id.clone() },
            Self::Pinned => ResidencyLedgerError::PinnedEviction {
                id: id.clone(),
                tier,
            },
            Self::InUse { pin_count } => ResidencyLedgerError::InUseEviction {
                id: id.clone(),
                tier,
                pin_count,
            },
            Self::PendingTransfer => inconsistent(id, tier, "evict pending transfer"),
            Self::Accounting => inconsistent(id, tier, "copy removal accounting"),
        }
    }
}
impl ResidencyLedger {
    /// Fixed controls of borrowed explicit eviction. No identifier, error text,
    /// or output collection is allocated by this worker.
    pub fn borrowed_eviction_control_bytes() -> Option<usize> {
        let controls = [
            size_of::<(&mut Self, &OffloadUnitId, MemoryTier)>(),
            size_of::<Result<Option<u64>, ResidencyEvictionError>>(),
            size_of::<Option<&LedgerUnit>>(),
            size_of::<Option<&LedgerCopy>>(),
            size_of::<Option<ResidentCopyStatus>>(),
            size_of::<LedgerCopy>(),
            size_of::<u64>(),
            size_of::<bool>(),
            size_of::<std::slice::Iter<'_, LedgerUnit>>(),
            size_of::<Option<&mut LedgerUnit>>(),
            size_of::<Option<&mut Option<LedgerCopy>>>(),
            size_of::<(MemoryTier, u64, bool)>(),
            size_of::<(MemoryTier, u64, usize)>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    /// Same explicit policy/pin/accounting checks and committed removal as
    /// `evict`, returning the removed byte count while borrowing the actual ID.
    /// Additionally refuses an unresolved transfer. Actual native completion
    /// and external source authority remain with the caller.
    pub fn evict_settled_borrowed(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<Option<u64>, ResidencyEvictionError> {
        let bytes = self.check_eviction(id, tier)?;
        if let Some(bytes) = bytes {
            let copy = self
                .units
                .get(id)
                .and_then(|unit| unit.copy(tier))
                .expect("checked eviction source");
            if copy.in_flight.is_some() {
                return Err(ResidencyEvictionError::PendingTransfer);
            }
            self.remove_copy_committed(id, tier, bytes, true);
        }
        Ok(bytes)
    }
    pub(super) fn check_eviction(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<Option<u64>, ResidencyEvictionError> {
        if tier == MemoryTier::Disk {
            return Err(ResidencyEvictionError::InvalidTargetTier);
        }
        let unit = self
            .units
            .get(id)
            .ok_or(ResidencyEvictionError::UnknownUnit)?;
        let Some(copy) = unit.copy(tier).and_then(|copy| copy.status()) else {
            return Ok(None);
        };
        if unit.spec.policy() == ResidencyPolicy::Pinned {
            return Err(ResidencyEvictionError::Pinned);
        }
        if copy.pins != 0 {
            return Err(ResidencyEvictionError::InUse {
                pin_count: copy.pins,
            });
        }
        self.tier_bytes(tier)
            .checked_sub(copy.bytes)
            .ok_or(ResidencyEvictionError::Accounting)?;
        Ok(Some(copy.bytes))
    }
}
#[cfg(test)]
mod tests;
