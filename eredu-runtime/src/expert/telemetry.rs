//! Parameter-bank telemetry and pass statistics.

use std::collections::BTreeMap;

use eredu_checkpoint::{quantization::WeightMaterializationReport, WeightQuantization};
use eredu_core::residency::MemoryTier;

use super::BankPassStatistics;
use crate::{ParameterBankKey, ResidencyReport};

/// Point-in-time entry residency and execution report.
pub struct ParameterBankResidencyReport {
    pub(super) pool_id: u64,
    /// Every packed encoding used by exact load-time transformed bindings.
    pub(super) weight_quantizations: Vec<WeightQuantization>,
    /// Exact architecture-selected ownership for every bank entry.
    pub(super) placements: Vec<(ParameterBankKey, crate::AddressableBankMemberPlacement)>,
    /// Owned logical entry count.
    pub(super) owned_entries: usize,
    /// Owned logical entry bytes, including cold checkpoint-only entries.
    pub(super) owned_bytes: u64,
    /// Current host-resident entry count.
    pub(super) host_resident_entries: usize,
    /// Current device-resident entry count.
    pub(super) device_resident_entries: usize,
    /// Current physical capacity of host-resident entry allocations.
    pub(super) host_resident_bytes: u64,
    /// Current device-resident entry bytes.
    pub(super) device_resident_bytes: u64,
    /// Peak physical capacity of host-resident entry allocations.
    pub(super) peak_host_resident_bytes: u64,
    /// Peak device-resident entry bytes.
    pub(super) peak_device_resident_bytes: u64,
    /// Prompt-processing statistics.
    pub(super) bulk: BankPassStatistics,
    /// Autoregressive incremental statistics.
    pub(super) incremental: BankPassStatistics,
    /// Underlying logical transfer and checkpoint diagnostics.
    pub(super) residency: ResidencyReport,
    /// Bounded load-time entry materialisation telemetry, when the catalog
    /// was transformed from floating checkpoint weights.
    pub(super) materialization: Option<WeightMaterializationReport>,
}

impl ParameterBankResidencyReport {
    /// Returns all packed load-time encodings in deterministic first-binding order.
    pub fn weight_quantizations(&self) -> &[WeightQuantization] {
        &self.weight_quantizations
    }
    /// Returns exact selected entry placement in deterministic key order.
    pub fn placements(&self) -> &[(ParameterBankKey, crate::AddressableBankMemberPlacement)] {
        &self.placements
    }
    /// Returns the number of owned entries.
    pub const fn owned_entries(&self) -> usize {
        self.owned_entries
    }
    /// Returns total owned bytes, including cold entries.
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }
    /// Returns host-resident entry count.
    pub const fn host_resident_entries(&self) -> usize {
        self.host_resident_entries
    }
    /// Returns device-resident entry count.
    pub const fn device_resident_entries(&self) -> usize {
        self.device_resident_entries
    }
    /// Returns current host-resident capacity.
    pub const fn host_resident_bytes(&self) -> u64 {
        self.host_resident_bytes
    }
    /// Returns current device-resident bytes.
    pub const fn device_resident_bytes(&self) -> u64 {
        self.device_resident_bytes
    }
    /// Returns peak host-resident capacity.
    pub const fn peak_host_resident_bytes(&self) -> u64 {
        self.peak_host_resident_bytes
    }
    /// Returns peak device-resident bytes.
    pub const fn peak_device_resident_bytes(&self) -> u64 {
        self.peak_device_resident_bytes
    }
    /// Returns bulk-access statistics.
    pub const fn bulk(&self) -> &BankPassStatistics {
        &self.bulk
    }
    /// Returns incremental-access statistics.
    pub const fn incremental(&self) -> &BankPassStatistics {
        &self.incremental
    }
    /// Returns underlying residency diagnostics.
    pub const fn residency(&self) -> &ResidencyReport {
        &self.residency
    }
    /// Returns bounded load-time materialization telemetry, when present.
    pub const fn materialization(&self) -> Option<&WeightMaterializationReport> {
        self.materialization.as_ref()
    }
}

/// Row companion/workspace facts beside the one shared physical cache report.
pub struct RowLookupPoolReport {
    pub(super) pool_id: u64,
    pub(super) residency: ResidencyReport,
    pub(super) requirements: crate::SelectedRowLookupRequirements,
}
impl RowLookupPoolReport {
    /// Records the retained row requirements and physical residency of this pool.
    pub const fn new(
        pool_id: u64,
        residency: ResidencyReport,
        requirements: crate::SelectedRowLookupRequirements,
    ) -> Self {
        Self {
            pool_id,
            residency,
            requirements,
        }
    }

    /// Whole-pool residency, including any grouped owners sharing this pool.
    pub fn residency(&self) -> &ResidencyReport {
        &self.residency
    }
    /// Retained scalars and selected serial invocation bounds, separate from cache occupancy.
    pub const fn requirements(&self) -> crate::SelectedRowLookupRequirements {
        self.requirements
    }
}

/// Reports independently identified banks and their total occupancy.
/// Shared pool peaks are counted once; logical bank counters remain separate.
pub struct ParameterBanksResidencyReport {
    rows: Option<RowLookupPoolReport>,
    banks: std::collections::BTreeMap<crate::RoutedBankId, ParameterBankResidencyReport>,
    bulk: BankPassStatistics,
    incremental: BankPassStatistics,
}
impl ParameterBanksResidencyReport {
    /// Retains all bank identities, counters, and exact placements.
    pub fn new(
        banks: std::collections::BTreeMap<crate::RoutedBankId, ParameterBankResidencyReport>,
    ) -> Self {
        let bulk = banks.values().map(|bank| *bank.bulk()).sum();
        let incremental = banks.values().map(|bank| *bank.incremental()).sum();
        Self {
            banks,
            bulk,
            incremental,
            rows: None,
        }
    }
    /// Attaches rows without double-counting their shared grouped-weight pool.
    pub fn with_rows(mut self, rows: RowLookupPoolReport) -> Self {
        self.rows = Some(rows);
        self
    }
    /// Independent row source, scalar and workspace facts when selected.
    pub fn rows(&self) -> Option<&RowLookupPoolReport> {
        self.rows.as_ref()
    }
    fn pools(&self) -> BTreeMap<u64, &ResidencyReport> {
        let mut pools: BTreeMap<_, _> = self
            .banks
            .values()
            .map(|b| (b.pool_id, b.residency()))
            .collect();
        if let Some(rows) = &self.rows {
            pools.insert(rows.pool_id, &rows.residency);
        }
        pools
    }
    /// Reports indexed by the selected architecture bank.
    pub fn banks(
        &self,
    ) -> &std::collections::BTreeMap<crate::RoutedBankId, ParameterBankResidencyReport> {
        &self.banks
    }
    /// Total prompt-processing counters across banks.
    pub const fn bulk(&self) -> &BankPassStatistics {
        &self.bulk
    }
    /// Total cached-decoding counters across banks.
    pub const fn incremental(&self) -> &BankPassStatistics {
        &self.incremental
    }
    /// Sum of `owned_entries` across independently retained banks.
    pub fn owned_entries(&self) -> usize {
        self.banks
            .values()
            .fold(0, |sum, bank| sum.saturating_add(bank.owned_entries()))
    }
    /// Sum of `owned_bytes` across independently retained banks.
    pub fn owned_bytes(&self) -> u64 {
        self.banks
            .values()
            .fold(0, |sum, bank| sum.saturating_add(bank.owned_bytes()))
    }
    /// Current shared-pool host entries, counting each physical pool once.
    pub fn host_resident_entries(&self) -> usize {
        self.pools().values().fold(0usize, |n, p| {
            n.saturating_add(p.offload().resident_units().get(MemoryTier::Host))
        })
    }
    /// Current shared-pool device entries, counting each physical pool once.
    pub fn device_resident_entries(&self) -> usize {
        self.pools().values().fold(0usize, |n, p| {
            n.saturating_add(p.offload().resident_units().get(MemoryTier::Device))
        })
    }
    /// Current shared-pool host bytes, including rows once.
    pub fn host_resident_bytes(&self) -> u64 {
        self.pools().values().fold(0u64, |n, p| {
            n.saturating_add(p.offload().resident_bytes().get(MemoryTier::Host))
        })
    }
    /// Current shared-pool device bytes, including rows once.
    pub fn device_resident_bytes(&self) -> u64 {
        self.pools().values().fold(0u64, |n, p| {
            n.saturating_add(p.offload().resident_bytes().get(MemoryTier::Device))
        })
    }
    /// Peak shared-pool host occupancy, counting each physical pool once.
    pub fn peak_host_resident_bytes(&self) -> u64 {
        self.pools().values().fold(0u64, |n, p| {
            n.saturating_add(p.offload().peak_resident_bytes().get(MemoryTier::Host))
        })
    }
    /// Peak shared-pool device occupancy, counting each physical pool once.
    pub fn peak_device_resident_bytes(&self) -> u64 {
        self.pools().values().fold(0u64, |n, p| {
            n.saturating_add(p.offload().peak_resident_bytes().get(MemoryTier::Device))
        })
    }
}
