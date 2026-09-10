//! Parameter-bank telemetry and pass statistics.

use super::*;

/// Tier-local cache request counters.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct BankTierStatistics {
    /// Logical entry acquisition requests after duplicate coalescing.
    pub(super) requests: u64,
    /// Requests served by an already resident copy.
    pub(super) hits: u64,
    /// Requests that materialized or promoted a copy.
    pub(super) misses: u64,
    /// Copies evicted while satisfying cache requests.
    pub(super) evictions: u64,
    /// Bytes evicted while satisfying cache requests.
    pub(super) eviction_bytes: u64,
}

impl BankTierStatistics {
    /// Returns logical acquisition requests after coalescing.
    pub const fn requests(&self) -> u64 {
        self.requests
    }
    /// Returns requests served by a resident copy.
    pub const fn hits(&self) -> u64 {
        self.hits
    }
    /// Returns requests requiring materialization or promotion.
    pub const fn misses(&self) -> u64 {
        self.misses
    }
    /// Returns evicted copies.
    pub const fn evictions(&self) -> u64 {
        self.evictions
    }
    /// Returns bytes removed by eviction.
    pub const fn eviction_bytes(&self) -> u64 {
        self.eviction_bytes
    }
}

/// Cumulative statistics for one public execution-path class.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct BankPassStatistics {
    /// Selection rows requested by the selector, including duplicates.
    pub(super) requested_selections: u64,
    /// Distinct logical entries requested after coalescing.
    pub(super) distinct_entries: u64,
    /// Duplicate requests eliminated before materialization.
    pub(super) coalesced_duplicates: u64,
    /// Temporary compact banks built.
    pub(super) compact_banks: u64,
    /// Cumulative compact-bank bytes.
    pub(super) compact_bank_bytes: u64,
    /// Peak temporary compact-bank bytes.
    pub(super) peak_compact_bank_bytes: u64,
    /// Cumulative compact-bank construction time.
    pub(super) compact_bank_time: Duration,
    /// Time preparing and reserving entry materialization or promotion.
    ///
    /// Deferred device completion is charged to the dependent entry output,
    /// not this counter.
    pub(super) materialization_wait: Duration,
    /// Host-tier cache activity.
    pub(super) host: BankTierStatistics,
    /// Device-tier cache activity.
    pub(super) device: BankTierStatistics,
}

impl BankPassStatistics {
    /// Returns requested selection rows, including duplicates.
    pub const fn requested_selections(&self) -> u64 {
        self.requested_selections
    }
    /// Returns distinct entries after coalescing.
    pub const fn distinct_entries(&self) -> u64 {
        self.distinct_entries
    }
    /// Returns duplicates eliminated before materialization.
    pub const fn coalesced_duplicates(&self) -> u64 {
        self.coalesced_duplicates
    }
    /// Returns temporary compact banks built.
    pub const fn compact_banks(&self) -> u64 {
        self.compact_banks
    }
    /// Returns cumulative compact-bank bytes.
    pub const fn compact_bank_bytes(&self) -> u64 {
        self.compact_bank_bytes
    }
    /// Returns the peak compact-bank byte size.
    pub const fn peak_compact_bank_bytes(&self) -> u64 {
        self.peak_compact_bank_bytes
    }
    /// Returns cumulative compact-bank construction time.
    pub const fn compact_bank_time(&self) -> Duration {
        self.compact_bank_time
    }
    /// Returns time spent preparing entry materialization or promotion.
    pub const fn materialization_wait(&self) -> Duration {
        self.materialization_wait
    }
    /// Returns host-tier activity.
    pub const fn host(&self) -> &BankTierStatistics {
        &self.host
    }
    /// Returns device-tier activity.
    pub const fn device(&self) -> &BankTierStatistics {
        &self.device
    }
}

/// Point-in-time entry residency and execution report.
pub struct ParameterBankResidencyReport {
    pub(super) pool_id: u64,
    /// Every packed encoding used by exact load-time transformed bindings.
    pub(super) weight_quantizations: Vec<WeightQuantization>,
    /// Exact architecture-selected ownership for every bank entry.
    pub(super) placements: Vec<(
        ParameterBankKey,
        eredu_runtime::AddressableBankMemberPlacement,
    )>,
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
    pub fn placements(
        &self,
    ) -> &[(
        ParameterBankKey,
        eredu_runtime::AddressableBankMemberPlacement,
    )] {
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

#[derive(Default)]
pub(super) struct ParameterBankStatistics {
    pub(super) peak_host_bytes: u64,
    pub(super) peak_device_bytes: u64,
    pub(super) bulk: BankPassStatistics,
    pub(super) incremental: BankPassStatistics,
}

impl ParameterBankStatistics {
    pub(super) fn pass_mut(&mut self, pass: BankAccessClass) -> &mut BankPassStatistics {
        match pass {
            BankAccessClass::Bulk => &mut self.bulk,
            BankAccessClass::Incremental => &mut self.incremental,
        }
    }
}

/// Reports independently identified banks and their total occupancy.
/// Shared pool peaks are counted once; logical bank counters remain separate.
pub struct ParameterBanksResidencyReport {
    banks: std::collections::BTreeMap<eredu_runtime::RoutedBankId, ParameterBankResidencyReport>,
    bulk: BankPassStatistics,
    incremental: BankPassStatistics,
}
impl ParameterBanksResidencyReport {
    /// Retains all bank identities, counters, and exact placements.
    pub fn new(
        banks: std::collections::BTreeMap<
            eredu_runtime::RoutedBankId,
            ParameterBankResidencyReport,
        >,
    ) -> Self {
        let bulk = banks.values().map(|bank| *bank.bulk()).sum();
        let incremental = banks.values().map(|bank| *bank.incremental()).sum();
        Self {
            banks,
            bulk,
            incremental,
        }
    }
    /// Reports indexed by the selected architecture bank.
    pub fn banks(
        &self,
    ) -> &std::collections::BTreeMap<eredu_runtime::RoutedBankId, ParameterBankResidencyReport>
    {
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
    /// Sum of `host_resident_entries` across independently retained banks.
    pub fn host_resident_entries(&self) -> usize {
        self.banks.values().fold(0, |sum, bank| {
            sum.saturating_add(bank.host_resident_entries())
        })
    }
    /// Sum of `device_resident_entries` across independently retained banks.
    pub fn device_resident_entries(&self) -> usize {
        self.banks.values().fold(0, |sum, bank| {
            sum.saturating_add(bank.device_resident_entries())
        })
    }
    /// Sum of `host_resident_bytes` across independently retained banks.
    pub fn host_resident_bytes(&self) -> u64 {
        self.banks.values().fold(0, |sum, bank| {
            sum.saturating_add(bank.host_resident_bytes())
        })
    }
    /// Sum of `device_resident_bytes` across independently retained banks.
    pub fn device_resident_bytes(&self) -> u64 {
        self.banks.values().fold(0, |sum, bank| {
            sum.saturating_add(bank.device_resident_bytes())
        })
    }
    /// Peak shared-pool occupancy, counting each physical pool once.
    pub fn peak_host_resident_bytes(&self) -> u64 {
        self.banks
            .values()
            .map(|bank| {
                (
                    bank.pool_id,
                    bank.residency()
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Host),
                )
            })
            .collect::<BTreeMap<_, _>>()
            .values()
            .fold(0, |sum: u64, bytes| sum.saturating_add(*bytes))
    }
    /// Peak shared-pool occupancy, counting each physical pool once.
    pub fn peak_device_resident_bytes(&self) -> u64 {
        self.banks
            .values()
            .map(|bank| {
                (
                    bank.pool_id,
                    bank.residency()
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device),
                )
            })
            .collect::<BTreeMap<_, _>>()
            .values()
            .fold(0, |sum: u64, bytes| sum.saturating_add(*bytes))
    }
}
impl std::iter::Sum for BankTierStatistics {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut total, value| {
            total.requests = total.requests.saturating_add(value.requests);
            total.hits = total.hits.saturating_add(value.hits);
            total.misses = total.misses.saturating_add(value.misses);
            total.evictions = total.evictions.saturating_add(value.evictions);
            total.eviction_bytes = total.eviction_bytes.saturating_add(value.eviction_bytes);
            total
        })
    }
}
impl std::iter::Sum for BankPassStatistics {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut total, value| {
            total.requested_selections = total
                .requested_selections
                .saturating_add(value.requested_selections);
            total.distinct_entries = total
                .distinct_entries
                .saturating_add(value.distinct_entries);
            total.coalesced_duplicates = total
                .coalesced_duplicates
                .saturating_add(value.coalesced_duplicates);
            total.compact_banks = total.compact_banks.saturating_add(value.compact_banks);
            total.compact_bank_bytes = total
                .compact_bank_bytes
                .saturating_add(value.compact_bank_bytes);
            total.compact_bank_time = total
                .compact_bank_time
                .saturating_add(value.compact_bank_time);
            total.materialization_wait = total
                .materialization_wait
                .saturating_add(value.materialization_wait);
            total.host = [total.host, value.host].into_iter().sum();
            total.device = [total.device, value.device].into_iter().sum();
            total.peak_compact_bank_bytes = total
                .peak_compact_bank_bytes
                .max(value.peak_compact_bank_bytes);
            total
        })
    }
}
