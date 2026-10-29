//! Exact row-resource addressing projected from the retained cold descriptor.
use super::RowLookupDescriptor;
use crate::{AddressableBankDemandError, ParameterBankKey};
use eredu_core::residency::OffloadUnitId;
use std::collections::BTreeSet;

/// Admitted row demand and exact source-backed residency IDs in request order.
/// No vocabulary-sized catalog or source payload is constructed or acquired.
pub struct RowBankDemand<'a> {
    entries: &'a [(ParameterBankKey, u64)],
    requests: Vec<(OffloadUnitId, u64)>,
}
impl RowBankDemand<'_> {
    /// Original logical row identities and use counts, preserving request order.
    pub fn entries(&self) -> &[(ParameterBankKey, u64)] {
        self.entries
    }
    /// Exact residency IDs and use counts for ordinary acquisition mechanisms.
    pub fn requests(&self) -> &[(OffloadUnitId, u64)] {
        &self.requests
    }
}
impl RowLookupDescriptor {
    /// Returns the encoded extent only for a row owned by this exact descriptor.
    pub fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        (key.bank() == self.spec().bank
            && key.unit() == self.spec().unit
            && (key.member() as u64) < self.spec().rows)
            .then_some(self.range().bytes())
    }
    /// Admits exact unique row demands and projects logical rows through the
    /// retained namespace offset. Rows preserve caller order rather than the
    /// canonical key ordering used by grouped banks. Workspace/decoder limits
    /// and native completion are admitted by their existing owners separately.
    pub fn plan_demand<'a>(
        &self,
        entries: &'a [(ParameterBankKey, u64)],
    ) -> Result<RowBankDemand<'a>, AddressableBankDemandError> {
        if entries.is_empty() {
            return Err(AddressableBankDemandError::EmptyDemand);
        }
        let mut seen = BTreeSet::new();
        let mut requests = Vec::with_capacity(entries.len());
        for &(key, demand) in entries {
            if self.member_bytes(key).is_none() {
                return Err(AddressableBankDemandError::MissingOwnedEntry { identity: key });
            }
            if demand == 0 {
                return Err(AddressableBankDemandError::ZeroDemand { identity: key });
            }
            if !seen.insert(key) {
                return Err(AddressableBankDemandError::DuplicateDemand { identity: key });
            }
            // Descriptor construction proves start + rows == end and bounds
            // each logical row, so this addition and projection are exact.
            let id = self
                .range()
                .member_id(self.range().start() + key.member() as u64)
                .expect("validated row descriptor");
            requests.push((id, demand));
        }
        Ok(RowBankDemand { entries, requests })
    }
}
