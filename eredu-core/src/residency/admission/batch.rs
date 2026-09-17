use super::*;

/// Exact ordered ledger failure, borrowing the supplied unknown identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyBatchError<'a> {
    /// Disk is not a materialized target.
    #[error("{operation} requires a host or device tier")]
    InvalidTargetTier {
        /// Stable operation description.
        operation: &'static str,
    },
    /// At the same position, duplicate refusal precedes unknown-unit lookup.
    /// An unknown unit at an earlier position still wins.
    #[error("batched residency acquisition contains a duplicate unit")]
    DuplicateBatchUnit,
    /// The first failing position named this actual unknown unit.
    #[error("unknown residency unit: {id}")]
    UnknownUnit {
        /// Caller-owned identifier; no owned copy was created.
        id: &'a OffloadUnitId,
    },
}
impl ResidencyBatchError<'_> {
    /// Ordinary adapter to the existing public owned ledger error.
    /// Only an unknown identifier needs an owned ID copy.
    pub fn into_owned(self) -> ResidencyLedgerError {
        match self {
            Self::InvalidTargetTier { operation } => {
                ResidencyLedgerError::InvalidTargetTier { operation }
            }
            Self::DuplicateBatchUnit => ResidencyLedgerError::DuplicateBatchUnit,
            Self::UnknownUnit { id } => ResidencyLedgerError::UnknownUnit { id: id.clone() },
        }
    }
}

/// A borrowed ledger refusal or a caller's insufficient neutral destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyBatchFailure<'a> {
    /// Actual ledger validation failed in caller order.
    #[error("{0}")]
    Ledger(ResidencyBatchError<'a>),
    /// The supplied index scratch cannot hold the complete batch.
    #[error("residency batch index destination needs {required} slots, has {available}")]
    Destination {
        /// Actual required batch length.
        required: usize,
        /// Actual available scratch length.
        available: usize,
    },
}
impl<'a> From<ResidencyBatchError<'a>> for ResidencyBatchFailure<'a> {
    fn from(cause: ResidencyBatchError<'a>) -> Self {
        Self::Ledger(cause)
    }
}

pub(super) fn validate<'a>(
    ledger: &ResidencyLedger,
    len: usize,
    id: impl Fn(usize) -> &'a OffloadUnitId,
    tier: MemoryTier,
    order: &mut [usize],
) -> Result<(), ResidencyBatchFailure<'a>> {
    if tier == MemoryTier::Disk {
        return Err(ResidencyBatchError::InvalidTargetTier {
            operation: "residency batch",
        }
        .into());
    }
    if order.len() < len {
        return Err(ResidencyBatchFailure::Destination {
            required: len,
            available: order.len(),
        });
    }
    let order = &mut order[..len];
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }
    // Unstable slice sort needs no heap. The original position is an explicit
    // tie-break, so equal IDs have deterministic ascending occurrence order.
    order.sort_unstable_by(|&left, &right| id(left).cmp(id(right)).then(left.cmp(&right)));
    let first_duplicate = order
        .windows(2)
        .filter(|pair| id(pair[0]) == id(pair[1]))
        .map(|pair| pair[1])
        .min();
    for index in 0..len {
        if first_duplicate == Some(index) {
            return Err(ResidencyBatchError::DuplicateBatchUnit.into());
        }
        let actual = id(index);
        if !ledger.units.contains_key(actual) {
            return Err(ResidencyBatchError::UnknownUnit { id: actual }.into());
        }
    }
    Ok(())
}
