//! Exact coexistent resource admission for independently leased bank acquisitions.
use crate::ParameterBankKey;
use std::collections::BTreeMap;

/// Invalid bank demand or its compact resource footprint.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankDemandError {
    /// Ordinary entry acquisition requires a nonempty demand.
    #[error("addressable entry demand must not be empty")]
    EmptyDemand,
    /// One selected entry has no uses.
    #[error("addressable entry {identity:?} has zero demand")]
    ZeroDemand {
        /// Invalid physical storage identity.
        identity: ParameterBankKey,
    },
    /// Demand must already coalesce each independently identified member.
    #[error("addressable entry demand repeats {identity:?}")]
    DuplicateDemand {
        /// Repeated physical storage identity.
        identity: ParameterBankKey,
    },
    /// The selected rank does not own this member.
    #[error("no owned bank entry {identity:?}")]
    MissingOwnedEntry {
        /// Missing physical storage identity.
        identity: ParameterBankKey,
    },
    /// An owned member has no current executable extent.
    #[error("owned bank entry {identity:?} has no effective byte extent")]
    MissingExtent {
        /// Physical storage whose extent is absent.
        identity: ParameterBankKey,
    },
    /// A compact acquisition cannot combine unrelated bank namespaces.
    #[error("compact parameter-bank demand mixes bank identities")]
    MixedBanks,
    /// Selection or compact payload accounting is not representable.
    #[error("bank demand byte or use-count accounting overflowed")]
    ByteOverflow,
    /// Independent selected payloads cannot coexist within the compact allowance.
    #[error("compact entry bank for {distinct_entries} entries requires {required_bytes} bytes, exceeding the {limit_bytes}-byte scratch limit")]
    ScratchLimitExceeded {
        /// Total required executable member bytes.
        required_bytes: u64,
        /// Admitted compact allowance.
        limit_bytes: u64,
        /// Independently charged member count.
        distinct_entries: usize,
    },
}

/// Admitted independent resources in canonical global identity order.
/// Default represents an empty logical selection with no resources or uses;
/// ordinary explicit entry acquisition still rejects an empty request.
#[derive(Debug, Default)]
pub struct AddressableBankDemand {
    entries: Vec<(ParameterBankKey, u64)>,
    selection_count: u64,
    scratch_bytes: u64,
}
impl AddressableBankDemand {
    /// Validates already coalesced entry demand before native acquisition.
    /// Each physical member is charged once using its current executable extent,
    /// independently of use count or shared checkpoint source recipes.
    pub fn new(
        entries: &[(ParameterBankKey, u64)],
        catalog: &BTreeMap<ParameterBankKey, u64>,
        effective_bytes: &BTreeMap<ParameterBankKey, u64>,
        scratch_limit: u64,
    ) -> Result<Self, AddressableBankDemandError> {
        if entries.is_empty() {
            return Err(AddressableBankDemandError::EmptyDemand);
        }
        let mut demand = BTreeMap::new();
        let mut selection_count = 0u64;
        for &(identity, count) in entries {
            if count == 0 {
                return Err(AddressableBankDemandError::ZeroDemand { identity });
            }
            if !catalog.contains_key(&identity) {
                return Err(AddressableBankDemandError::MissingOwnedEntry { identity });
            }
            if demand.insert(identity, count).is_some() {
                return Err(AddressableBankDemandError::DuplicateDemand { identity });
            }
            selection_count = selection_count
                .checked_add(count)
                .ok_or(AddressableBankDemandError::ByteOverflow)?;
        }
        let bank = demand.first_key_value().unwrap().0.bank();
        if demand.keys().any(|key| key.bank() != bank) {
            return Err(AddressableBankDemandError::MixedBanks);
        }
        let scratch_bytes = demand.keys().try_fold(0u64, |total, identity| {
            let bytes =
                effective_bytes
                    .get(identity)
                    .ok_or(AddressableBankDemandError::MissingExtent {
                        identity: *identity,
                    })?;
            total
                .checked_add(*bytes)
                .ok_or(AddressableBankDemandError::ByteOverflow)
        })?;
        if scratch_bytes > scratch_limit {
            return Err(AddressableBankDemandError::ScratchLimitExceeded {
                required_bytes: scratch_bytes,
                limit_bytes: scratch_limit,
                distinct_entries: demand.len(),
            });
        }
        Ok(Self {
            entries: demand.into_iter().collect(),
            selection_count,
            scratch_bytes,
        })
    }
    /// Moves exact physical demands, total logical uses and compact byte allowance
    /// to the existing backend transfer and completion mechanism.
    pub fn into_parts(self) -> (Vec<(ParameterBankKey, u64)>, u64, u64) {
        (self.entries, self.selection_count, self.scratch_bytes)
    }
}
