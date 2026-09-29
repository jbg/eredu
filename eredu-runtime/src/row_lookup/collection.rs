//! Compact preparation and total binding of a set of row-addressable parameters.
use super::*;
use std::collections::BTreeMap;

/// Cold storage and invocation bounds, independent of vocabulary cardinality.
/// Invocation bounds describe one serial lookup; retained scalars are cumulative.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowLookupRequirements {
    /// Encoded source bytes across all tables; these need not be resident.
    pub source_bytes: u64,
    /// Floating scalar companions retained beside the row cache.
    pub scalar_bytes: u64,
    /// Largest encoded acquisition allowed by the prepared lookup limits.
    pub acquisition_bytes: u64,
    /// Maximum host planning allowance for one lookup.
    pub host_bytes: u64,
    /// Maximum completed/concatenated/reordered output allowance for one lookup.
    pub output_bytes: u64,
}

/// Complete compact row binding authority for a prepared execution graph.
/// Construction rejects duplicate identities and overlapping bank ownership
/// before a backend is called. All retained sources remain payload-lazy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PreparedRowLookups {
    entries: BTreeMap<ParameterId, PreparedRowLookup>,
    descriptors: RowLookupDescriptors,
}
impl PreparedRowLookups {
    /// Validates every owner against the graph's flat execution-unit count.
    pub fn new(
        entries: impl IntoIterator<Item = PreparedRowLookup>,
        execution_units: usize,
    ) -> Result<Self, RowLookupError> {
        let entries: Vec<_> = entries.into_iter().collect();
        let descriptors = RowLookupDescriptors::new(
            entries.iter().map(|entry| entry.descriptor().clone()),
            execution_units,
        )?;
        Ok(Self {
            entries: entries
                .into_iter()
                .map(|entry| (entry.spec().parameter.clone(), entry))
                .collect(),
            descriptors,
        })
    }
    /// Metadata-only authority for cold selection before binding readable sources.
    pub fn descriptors(&self) -> &RowLookupDescriptors {
        &self.descriptors
    }
    /// Exact table contracts; one entry per parameter, never per vocabulary row.
    pub fn entries(&self) -> &BTreeMap<ParameterId, PreparedRowLookup> {
        &self.entries
    }
    /// Compact ranges to place in the same residency pool as other parameter banks.
    pub fn ranges(&self) -> impl Iterator<Item = &crate::RowResidencyRange> {
        self.entries.values().map(PreparedRowLookup::range)
    }
    /// Cold source, retained-companion and per-invocation workspace bounds.
    pub const fn requirements(&self) -> RowLookupRequirements {
        self.descriptors.requirements()
    }
    /// Rejects a pool that cannot hold the largest admitted row acquisition.
    /// Other live owners and transfer buffers remain charged by the shared ledger.
    pub fn validate_pool_budget(
        &self,
        config: eredu_core::residency::OffloadConfig,
    ) -> Result<(), RowLookupError> {
        self.descriptors.validate_pool_budget(config)
    }
    /// Checks that row namespaces do not alias another selected bank's identity.
    pub fn validate_other_banks(
        &self,
        banks: impl IntoIterator<Item = usize>,
    ) -> Result<(), RowLookupError> {
        self.descriptors.validate_other_banks(banks)
    }
    /// Binds one actual tensor-group member to the complete selected row set.
    /// Only `owner` supplies native banks; peers retain declarations and receive
    /// compact results through the already bound collective context. Each owner
    /// bank keeps the ordinary acquisition, completion and source-lease contract.
    pub fn bind_tensor_parallel<B, K, G>(
        &self,
        mut banks: BTreeMap<ParameterId, K>,
        parallel: G,
        status_parallel: G,
        rank: usize,
        owner: usize,
    ) -> Result<
        RowLookupProviders<TensorParallelRowLookup<B, BoundedRowLookup<K>, G>>,
        RowLookupError,
    >
    where
        B: eredu_nn::DistributedNeuralBackend,
        K: RowLookupBank<B>,
        G: Clone + std::borrow::Borrow<B::ParallelContext>,
    {
        let size = B::parallel_size(parallel.borrow());
        if size == 0 || rank >= size || owner >= size || rank != B::parallel_rank(parallel.borrow())
        {
            return Err(RowLookupError::Geometry);
        }
        if rank == owner {
            if let Some(id) = self.entries.keys().find(|id| !banks.contains_key(*id)) {
                return Err(RowLookupError::Missing(id.clone()));
            }
            if let Some(id) = banks.keys().find(|id| !self.entries.contains_key(*id)) {
                return Err(RowLookupError::Specification(id.clone()));
            }
        } else if let Some(id) = banks.keys().next() {
            return Err(RowLookupError::Specification(id.clone()));
        }
        RowLookupProviders::new(
            self.entries
                .iter()
                .map(|(id, entry)| {
                    let provider = banks.remove(id).map(|bank| entry.bind(bank)).transpose()?;
                    TensorParallelRowLookup::new(
                        entry.spec().clone(),
                        entry.limits().requests,
                        provider,
                        parallel.clone(),
                        status_parallel.clone(),
                        rank,
                        owner,
                    )
                    .map(|provider| (id.clone(), provider))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    /// Binds exactly the prepared set after native construction has consumed the
    /// same contracts. Missing or extra banks reject before any lookup can run.
    pub fn bind<K>(
        &self,
        mut banks: BTreeMap<ParameterId, K>,
    ) -> Result<RowLookupProviders<BoundedRowLookup<K>>, RowLookupError> {
        if let Some(id) = self.entries.keys().find(|id| !banks.contains_key(*id)) {
            return Err(RowLookupError::Missing(id.clone()));
        }
        if let Some(id) = banks.keys().find(|id| !self.entries.contains_key(*id)) {
            return Err(RowLookupError::Specification(id.clone()));
        }
        RowLookupProviders::new(
            self.entries
                .iter()
                .map(|(id, entry)| {
                    entry
                        .bind(banks.remove(id).expect("validated exact native row set"))
                        .map(|provider| (id.clone(), provider))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}
