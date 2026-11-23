//! Host staging, exact device acquisition, and accounting for grouped banks.
use std::{
    collections::BTreeMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use eredu_core::residency::{MemoryTier, OffloadUnitId, ResidencyTransferOrdering};
use eredu_nn::{GroupedNeuralBackend, Tensor};

use crate::residency::ResidencyAccess;
use crate::{
    AddressableBankDemand, AddressableBankDemandError, AddressableGroupedBank, ParameterBankAccess,
    ParameterBankAcquisition, ParameterBankKey,
};

/// State locked by the backend while the portable driver acquires one group.
pub struct BankAcquisitionResources<'a, R> {
    /// Physical cache used by the shared pool.
    pub residency: &'a R,
    /// Exact owned member geometry.
    pub catalog: &'a BTreeMap<ParameterBankKey, u64>,
    /// Effective logical member sizes after parameter replacement.
    pub effective_member_bytes: &'a BTreeMap<ParameterBankKey, u64>,
    /// Lookup from physical residency units to logical bank ordinals.
    pub unit_banks: &'a BTreeMap<OffloadUnitId, usize>,
    /// Selected host cache capacity, if bounded.
    pub host_budget: Option<u64>,
    /// Selected compact-bank working-set limit.
    pub scratch_limit: u64,
    /// Counters retained with the shared pool.
    pub statistics: &'a Mutex<BTreeMap<usize, ParameterBankStatistics>>,
    /// The only logical bank permitted by a scoped view.
    pub scope: Option<usize>,
}

/// Stable access to a shared cache during grouped acquisition.
pub trait GroupedBankAcquisitionGuard {
    /// Exact installed residency implementation.
    type Residency: ResidencyAccess;

    /// Borrows the state protected by this guard.
    fn resources(&self) -> BankAcquisitionResources<'_, Self::Residency>;

    /// Identifies a saturated host-only cache for the portable staging fallback.
    fn host_capacity_exhausted(error: &<Self::Residency as ResidencyAccess>::Error) -> bool;
}

/// Failure admitting or completing admission for a grouped bank.
#[derive(Debug, thiserror::Error)]
pub enum BankAcquisitionError<E: std::error::Error + 'static> {
    /// The demand disagrees with selected ownership or working-set limits.
    #[error(transparent)]
    Demand(#[from] AddressableBankDemandError),
    /// The acquisition reached outside the selected view.
    #[error("acquisition exceeds selected bank scope")]
    Scope,
    /// Native residency could not provide the selected copy.
    #[error(transparent)]
    Residency(E),
    /// Statistics are unavailable after a panic.
    #[error("sparse entry cache statistics are unavailable after a panic")]
    StatisticsPoisoned,
}

/// Acquired compact order, accounting and completion authority.
pub struct BankTransfer<T> {
    /// Physical member keys in compact order.
    pub identities: Vec<ParameterBankKey>,
    /// Coalesced use counts in compact order.
    pub demand: Vec<u64>,
    /// Admitted compact-bank bytes.
    pub scratch_bytes: u64,
    /// Execution path used for accounting.
    pub access: ParameterBankAccess,
    /// Native transfer that retains the acquired leases.
    pub transfer: T,
}

impl<T> BankTransfer<T> {
    pub(crate) fn begin_loan<L, E>(
        self,
        open: impl FnOnce(T) -> Result<L, E>,
    ) -> Result<BankTransfer<L>, E> {
        Ok(BankTransfer {
            identities: self.identities,
            demand: self.demand,
            scratch_bytes: self.scratch_bytes,
            access: self.access,
            transfer: open(self.transfer)?,
        })
    }
}

/// Admits a grouped request and orders the selected device copy before execution.
pub fn acquire_grouped_bank<B, G>(
    bank: &G,
    request: ParameterBankAcquisition<'_>,
    queue: &<B::Tensor as Tensor>::Context,
) -> Result<G::Acquisition, G::Error>
where
    B: GroupedNeuralBackend,
    G: AddressableGroupedBank<B>,
{
    let guard = bank.acquisition_guard()?;
    let resources = guard.resources();
    if resources
        .scope
        .is_some_and(|scope| request.entries().iter().any(|(key, _)| key.bank() != scope))
    {
        return Err(eredu_core::BackendFailure::from_error(
            BankAcquisitionError::<<G::Residency as ResidencyAccess>::Error>::Scope,
        )
        .into());
    }
    let plan = AddressableBankDemand::new(
        request.entries(),
        resources.catalog,
        resources.effective_member_bytes,
        resources.scratch_limit,
    )
    .map_err(BankAcquisitionError::<<G::Residency as ResidencyAccess>::Error>::Demand)
    .map_err(eredu_core::BackendFailure::from_error)?;
    let transfer = acquire_addressable_demand(&guard, resources, plan, request.access(), queue)
        .map_err(eredu_core::BackendFailure::from_error)?;
    drop(guard);
    bank.begin_consumption(transfer, queue)
}

/// Acquires an admitted exact demand and updates telemetry alongside it.
pub fn acquire_addressable_demand<Q, G>(
    _: &G,
    resources: BankAcquisitionResources<'_, G::Residency>,
    plan: AddressableBankDemand,
    pass: ParameterBankAccess,
    queue: &Q,
) -> Result<
    BankTransfer<<G::Residency as ResidencyAccess>::Transfer>,
    BankAcquisitionError<<G::Residency as ResidencyAccess>::Error>,
>
where
    Q: ?Sized,
    G: GroupedBankAcquisitionGuard,
    <G::Residency as ResidencyAccess>::Transfer:
        ResidencyTransferOrdering<Q, Error = <G::Residency as ResidencyAccess>::Error>,
{
    let (entries, selection_count, scratch_bytes) = plan.into_parts();
    let bank = entries.first().map(|(key, _)| key.bank());
    let before = resident_snapshot(resources.residency)?;
    let started = Instant::now();
    let mut host_hits = 0u64;
    let mut host_misses = 0u64;
    let mut device_hits = 0u64;
    let mut device_misses = 0u64;
    let mut requests = Vec::with_capacity(entries.len());
    let mut host_requests = Vec::new();
    for &(identity, selection_demand) in &entries {
        let unit = identity.unit_id();
        let host_hit = resources
            .residency
            .is_resident(&unit, MemoryTier::Host)
            .map_err(BankAcquisitionError::Residency)?;
        let device_hit = resources
            .residency
            .is_resident(&unit, MemoryTier::Device)
            .map_err(BankAcquisitionError::Residency)?;
        if host_hit {
            host_hits = host_hits.saturating_add(1);
        } else {
            host_misses = host_misses.saturating_add(1);
        }
        if device_hit {
            device_hits = device_hits.saturating_add(1);
        } else {
            device_misses = device_misses.saturating_add(1);
        }
        if host_hit || (!device_hit && resources.host_budget != Some(0)) {
            host_requests.push((unit.clone(), selection_demand));
        }
        requests.push((unit, selection_demand));
    }
    if !host_requests.is_empty() {
        match resources
            .residency
            .acquire_many_with_demand(&host_requests, MemoryTier::Host)
        {
            Ok(host) => drop(host),
            Err(error) if G::host_capacity_exhausted(&error) => {}
            Err(error) => return Err(BankAcquisitionError::Residency(error)),
        }
    }
    let transfer = resources
        .residency
        .acquire_many_with_transfer(&requests, MemoryTier::Device)
        .map_err(BankAcquisitionError::Residency)?;
    transfer
        .order_after(queue)
        .map_err(BankAcquisitionError::Residency)?;
    let wait = started.elapsed();
    let after = resident_snapshot(resources.residency)?;
    let (host_evictions, host_eviction_bytes) = before.evicted(&after, MemoryTier::Host);
    let (device_evictions, device_eviction_bytes) = before.evicted(&after, MemoryTier::Device);

    let mut statistics = resources
        .statistics
        .lock()
        .map_err(|_| BankAcquisitionError::StatisticsPoisoned)?;
    if let Some(bank) = bank {
        let stats = statistics.entry(bank).or_default().pass_mut(pass);
        let distinct = entries.len() as u64;
        stats.requested_selections = stats.requested_selections.saturating_add(selection_count);
        stats.distinct_entries = stats.distinct_entries.saturating_add(distinct);
        stats.coalesced_duplicates = stats
            .coalesced_duplicates
            .saturating_add(selection_count.saturating_sub(distinct));
        stats.materialization_wait = stats.materialization_wait.saturating_add(wait);
        stats.host.requests = stats.host.requests.saturating_add(distinct);
        stats.host.hits = stats.host.hits.saturating_add(host_hits);
        stats.host.misses = stats.host.misses.saturating_add(host_misses);
        stats.host.evictions = stats.host.evictions.saturating_add(host_evictions);
        stats.host.eviction_bytes = stats
            .host
            .eviction_bytes
            .saturating_add(host_eviction_bytes);
        stats.device.requests = stats.device.requests.saturating_add(distinct);
        stats.device.hits = stats.device.hits.saturating_add(device_hits);
        stats.device.misses = stats.device.misses.saturating_add(device_misses);
        stats.device.evictions = stats.device.evictions.saturating_add(device_evictions);
        stats.device.eviction_bytes = stats
            .device
            .eviction_bytes
            .saturating_add(device_eviction_bytes);
    }
    let mut occupancy = BTreeMap::<usize, (u64, u64)>::new();
    for (id, bytes) in &after.host {
        if let Some(bank) = resources.unit_banks.get(id) {
            occupancy.entry(*bank).or_default().0 += bytes;
        }
    }
    for (id, bytes) in &after.device {
        if let Some(bank) = resources.unit_banks.get(id) {
            occupancy.entry(*bank).or_default().1 += bytes;
        }
    }
    for (bank, (host, device)) in occupancy {
        let stats = statistics.entry(bank).or_default();
        stats.peak_host_bytes = stats.peak_host_bytes.max(host);
        stats.peak_device_bytes = stats.peak_device_bytes.max(device);
    }
    drop(statistics);

    let (identities, demand) = entries.into_iter().unzip();
    Ok(BankTransfer {
        identities,
        demand,
        scratch_bytes,
        access: pass,
        transfer,
    })
}

fn resident_snapshot<R: ResidencyAccess>(
    residency: &R,
) -> Result<ResidentSnapshot, BankAcquisitionError<R::Error>> {
    let report = residency
        .residency_snapshot()
        .map_err(BankAcquisitionError::Residency)?;
    Ok(ResidentSnapshot {
        host: report
            .units
            .iter()
            .filter(|unit| unit.host_resident())
            .map(|unit| (unit.id().clone(), unit.host_allocated_bytes()))
            .collect(),
        device: report
            .units
            .iter()
            .filter(|unit| unit.device_resident())
            .map(|unit| (unit.id().clone(), unit.device_allocated_bytes()))
            .collect(),
    })
}

struct ResidentSnapshot {
    host: BTreeMap<OffloadUnitId, u64>,
    device: BTreeMap<OffloadUnitId, u64>,
}

impl ResidentSnapshot {
    fn evicted(&self, after: &Self, tier: MemoryTier) -> (u64, u64) {
        let (before, after) = match tier {
            MemoryTier::Host => (&self.host, &after.host),
            MemoryTier::Device => (&self.device, &after.device),
            MemoryTier::Disk => return (0, 0),
        };
        before
            .iter()
            .filter(|(id, _)| !after.contains_key(*id))
            .fold((0u64, 0u64), |(count, bytes), (_, size)| {
                (count.saturating_add(1), bytes.saturating_add(*size))
            })
    }
}

/// Tier-local cache request counters.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct BankTierStatistics {
    /// Logical entry acquisition requests after duplicate coalescing.
    requests: u64,
    /// Requests served by an already resident copy.
    hits: u64,
    /// Requests that materialized or promoted a copy.
    misses: u64,
    /// Copies evicted while satisfying cache requests.
    evictions: u64,
    /// Bytes evicted while satisfying cache requests.
    eviction_bytes: u64,
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
    requested_selections: u64,
    /// Distinct logical entries requested after coalescing.
    distinct_entries: u64,
    /// Duplicate requests eliminated before materialization.
    coalesced_duplicates: u64,
    /// Temporary compact banks built.
    compact_banks: u64,
    /// Cumulative compact-bank bytes.
    compact_bank_bytes: u64,
    /// Peak temporary compact-bank bytes.
    peak_compact_bank_bytes: u64,
    /// Cumulative compact-bank construction time.
    compact_bank_time: Duration,
    /// Time preparing and reserving entry materialization or promotion.
    ///
    /// Deferred device completion is charged to the dependent entry output,
    /// not this counter.
    materialization_wait: Duration,
    /// Host-tier cache activity.
    host: BankTierStatistics,
    /// Device-tier cache activity.
    device: BankTierStatistics,
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

/// Counters and occupancy peaks for one logical bank in a shared cache.
#[derive(Default)]
pub struct ParameterBankStatistics {
    /// Peak host occupancy.
    pub peak_host_bytes: u64,
    /// Peak device occupancy.
    pub peak_device_bytes: u64,
    /// Prefill counters.
    pub bulk: BankPassStatistics,
    /// Decode counters.
    pub incremental: BankPassStatistics,
}

impl ParameterBankStatistics {
    /// Records completed native compaction using the admitted logical byte count.
    pub fn record_compaction(
        &mut self,
        access: ParameterBankAccess,
        bytes: u64,
        elapsed: Duration,
    ) {
        let stats = self.pass_mut(access);
        stats.compact_banks = stats.compact_banks.saturating_add(1);
        stats.compact_bank_bytes = stats.compact_bank_bytes.saturating_add(bytes);
        stats.peak_compact_bank_bytes = stats.peak_compact_bank_bytes.max(bytes);
        stats.compact_bank_time = stats.compact_bank_time.saturating_add(elapsed);
    }

    pub(super) fn pass_mut(&mut self, pass: ParameterBankAccess) -> &mut BankPassStatistics {
        match pass {
            ParameterBankAccess::Bulk => &mut self.bulk,
            ParameterBankAccess::Incremental => &mut self.incremental,
        }
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
