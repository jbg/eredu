//! Backend-neutral policy, ownership, capacity, accounting, and telemetry for weight residency.
//!
//! Placement determines which tensors a rank owns. The types in this module
//! describe a separate residency decision: the tier in which an owned logical
//! unit is intended to reside and its lifetime policy. This module validates
//! explicit plans and records observations without knowing how a backend
//! materializes, transfers, or synchronizes resources.

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    time::Duration,
};

/// Current serialized residency-plan schema.
pub const OFFLOAD_PLAN_SCHEMA_VERSION: u32 = 1;

/// A storage or execution-memory tier used by an offload plan.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// Memory directly used for device execution.
    Device,
    /// Host-accessible memory.
    Host,
    /// Disk-backed storage.
    Disk,
}

impl MemoryTier {
    const fn index(self) -> usize {
        match self {
            Self::Device => 0,
            Self::Host => 1,
            Self::Disk => 2,
        }
    }
}

/// The intended lifetime behavior of an offload unit within a tier.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyPolicy {
    /// Keep the unit resident for the lifetime of the residency manager.
    Pinned,
    /// Keep the unit resident only within a bounded execution window.
    Windowed,
    /// Allow the residency manager to retain or evict the unit as cache policy permits.
    Cacheable,
}

/// Deterministic eviction ordering for cacheable residency units.
#[derive(
    Debug, Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CacheEvictionPolicy {
    /// Evict the least recently used cacheable copy first.
    #[default]
    LeastRecentlyUsed,
    /// Evict the least frequently used copy, using recency and unit id as ties.
    LeastFrequentlyUsed,
}

/// A stable logical identifier for one independently managed offload unit.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct OffloadUnitId(String);

impl OffloadUnitId {
    /// Creates an identifier from a non-empty string.
    pub fn new(id: impl Into<String>) -> Result<Self, OffloadError> {
        let id = id.into();
        if id.trim().is_empty() {
            Err(OffloadError::EmptyUnitId)
        } else {
            Ok(Self(id))
        }
    }

    /// Returns the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for OffloadUnitId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for OffloadUnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'de> Deserialize<'de> for OffloadUnitId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

/// Global limits and lookahead used when validating an explicit offload plan.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct OffloadConfig {
    device_budget_bytes: Option<u64>,
    host_budget_bytes: Option<u64>,
    prefetch_depth: usize,
    eviction_policy: CacheEvictionPolicy,
}

impl OffloadConfig {
    /// Creates a configuration with optional finite device and host budgets.
    ///
    /// A zero-byte budget is meaningful and forbids assigning non-empty units
    /// to that tier. `prefetch_depth` must be nonzero.
    pub fn new(
        device_budget_bytes: Option<u64>,
        host_budget_bytes: Option<u64>,
        prefetch_depth: usize,
    ) -> Result<Self, OffloadError> {
        if prefetch_depth == 0 {
            return Err(OffloadError::ZeroPrefetchDepth);
        }
        Ok(Self {
            device_budget_bytes,
            host_budget_bytes,
            prefetch_depth,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
        })
    }

    /// Returns the finite device-tier budget, if configured.
    pub const fn device_budget_bytes(self) -> Option<u64> {
        self.device_budget_bytes
    }

    /// Returns the finite physical host-allocation budget, if configured.
    pub const fn host_budget_bytes(self) -> Option<u64> {
        self.host_budget_bytes
    }

    /// Returns the number of logical units the executor may prefetch ahead.
    pub const fn prefetch_depth(self) -> usize {
        self.prefetch_depth
    }

    /// Selects deterministic cache eviction without changing tier budgets.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Returns the configured cache eviction ordering.
    pub const fn eviction_policy(self) -> CacheEvictionPolicy {
        self.eviction_policy
    }
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            device_budget_bytes: None,
            host_budget_bytes: None,
            prefetch_depth: 1,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
        }
    }
}

#[derive(Deserialize)]
struct SerializedOffloadConfig {
    device_budget_bytes: Option<u64>,
    host_budget_bytes: Option<u64>,
    prefetch_depth: usize,
    eviction_policy: CacheEvictionPolicy,
}

impl<'de> Deserialize<'de> for OffloadConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = SerializedOffloadConfig::deserialize(deserializer)?;
        Self::new(
            value.device_budget_bytes,
            value.host_budget_bytes,
            value.prefetch_depth,
        )
        .map(|config| config.with_eviction_policy(value.eviction_policy))
        .map_err(D::Error::custom)
    }
}

/// One explicit logical unit assignment in an offload plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct OffloadUnitSpec {
    id: OffloadUnitId,
    bytes: u64,
    policy: ResidencyPolicy,
    tier: MemoryTier,
}

impl OffloadUnitSpec {
    /// Creates and validates one explicit assignment.
    pub fn new(
        id: OffloadUnitId,
        bytes: u64,
        policy: ResidencyPolicy,
        tier: MemoryTier,
    ) -> Result<Self, OffloadError> {
        if bytes == 0 {
            return Err(OffloadError::ZeroSizedUnit { id });
        }
        if policy == ResidencyPolicy::Pinned && tier == MemoryTier::Disk {
            return Err(OffloadError::ContradictoryAssignment {
                id,
                policy,
                tier,
                reason: "pinned units must be assigned to a resident memory tier",
            });
        }
        Ok(Self {
            id,
            bytes,
            policy,
            tier,
        })
    }

    /// Returns the logical unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the planned unit size in bytes.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the planned residency policy.
    pub const fn policy(&self) -> ResidencyPolicy {
        self.policy
    }

    /// Returns the explicitly assigned tier.
    pub const fn tier(&self) -> MemoryTier {
        self.tier
    }
}

#[derive(Deserialize)]
struct SerializedOffloadUnitSpec {
    id: OffloadUnitId,
    bytes: u64,
    policy: ResidencyPolicy,
    tier: MemoryTier,
}

impl<'de> Deserialize<'de> for OffloadUnitSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = SerializedOffloadUnitSpec::deserialize(deserializer)?;
        Self::new(value.id, value.bytes, value.policy, value.tier).map_err(D::Error::custom)
    }
}

/// Byte totals indexed by memory tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct TierByteTotals {
    device: u64,
    host: u64,
    disk: u64,
}

/// Current or peak logical resident-unit counts by tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct TierUnitTotals {
    device: usize,
    host: usize,
    disk: usize,
}

impl TierUnitTotals {
    /// Creates explicit device, host, and disk unit totals.
    pub const fn new(device: usize, host: usize, disk: usize) -> Self {
        Self { device, host, disk }
    }

    /// Returns the unit total for `tier`.
    pub const fn get(self, tier: MemoryTier) -> usize {
        match tier {
            MemoryTier::Device => self.device,
            MemoryTier::Host => self.host,
            MemoryTier::Disk => self.disk,
        }
    }

    fn set(&mut self, tier: MemoryTier, units: usize) {
        match tier {
            MemoryTier::Device => self.device = units,
            MemoryTier::Host => self.host = units,
            MemoryTier::Disk => self.disk = units,
        }
    }
}

impl TierByteTotals {
    /// Creates explicit device, host, and disk byte totals.
    pub const fn new(device: u64, host: u64, disk: u64) -> Self {
        Self { device, host, disk }
    }

    /// Returns the byte total for `tier`.
    pub const fn get(self, tier: MemoryTier) -> u64 {
        match tier {
            MemoryTier::Device => self.device,
            MemoryTier::Host => self.host,
            MemoryTier::Disk => self.disk,
        }
    }

    fn set(&mut self, tier: MemoryTier, bytes: u64) {
        match tier {
            MemoryTier::Device => self.device = bytes,
            MemoryTier::Host => self.host = bytes,
            MemoryTier::Disk => self.disk = bytes,
        }
    }

    fn checked_add(&mut self, tier: MemoryTier, bytes: u64) -> Result<(), OffloadError> {
        let total = self
            .get(tier)
            .checked_add(bytes)
            .ok_or(OffloadError::ByteTotalOverflow { tier })?;
        self.set(tier, total);
        Ok(())
    }
}

/// A deterministic, validated explicit offload plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct OffloadPlan {
    schema_version: u32,
    config: OffloadConfig,
    units: Vec<OffloadUnitSpec>,
    #[serde(skip)]
    planned_bytes: TierByteTotals,
}

impl OffloadPlan {
    /// Validates explicit assignments and sorts them by logical identifier.
    ///
    /// This constructor does not materialize tensors or choose assignments.
    pub fn new(
        config: OffloadConfig,
        units: impl IntoIterator<Item = OffloadUnitSpec>,
    ) -> Result<Self, OffloadError> {
        let mut units = units.into_iter().collect::<Vec<_>>();
        units.sort_by(|left, right| left.id.cmp(&right.id));

        if let Some(pair) = units.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(OffloadError::DuplicateUnitId {
                id: pair[0].id.clone(),
            });
        }

        let mut planned_bytes = TierByteTotals::default();
        for unit in &units {
            planned_bytes.checked_add(unit.tier, unit.bytes)?;
        }

        validate_budget(
            MemoryTier::Device,
            planned_bytes.device,
            config.device_budget_bytes,
        )?;
        validate_budget(
            MemoryTier::Host,
            planned_bytes.host,
            config.host_budget_bytes,
        )?;

        Ok(Self {
            schema_version: OFFLOAD_PLAN_SCHEMA_VERSION,
            config,
            units,
            planned_bytes,
        })
    }

    /// Returns the stable serialized schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Revalidates every invariant represented by this plan.
    pub fn validate(&self) -> Result<(), OffloadError> {
        Self::new(self.config, self.units.clone()).map(|_| ())
    }

    /// Returns the configuration used to validate this plan.
    pub const fn config(&self) -> OffloadConfig {
        self.config
    }

    /// Returns assignments in stable logical-identifier order.
    pub fn units(&self) -> &[OffloadUnitSpec] {
        &self.units
    }

    /// Looks up a unit by its logical identifier.
    pub fn unit(&self, id: &OffloadUnitId) -> Option<&OffloadUnitSpec> {
        self.units
            .binary_search_by(|unit| unit.id.cmp(id))
            .ok()
            .map(|index| &self.units[index])
    }

    /// Returns checked planned byte totals for every tier.
    pub const fn planned_bytes(&self) -> TierByteTotals {
        self.planned_bytes
    }
}

#[derive(Deserialize)]
struct SerializedOffloadPlan {
    schema_version: u32,
    config: OffloadConfig,
    units: Vec<OffloadUnitSpec>,
}

impl<'de> Deserialize<'de> for OffloadPlan {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = SerializedOffloadPlan::deserialize(deserializer)?;
        if value.schema_version != OFFLOAD_PLAN_SCHEMA_VERSION {
            return Err(D::Error::custom(OffloadError::UnsupportedSchemaVersion(
                value.schema_version,
            )));
        }
        Self::new(value.config, value.units).map_err(D::Error::custom)
    }
}

fn validate_budget(
    tier: MemoryTier,
    planned_bytes: u64,
    budget_bytes: Option<u64>,
) -> Result<(), OffloadError> {
    if let Some(budget_bytes) = budget_bytes {
        if planned_bytes > budget_bytes {
            return Err(OffloadError::BudgetExceeded {
                tier,
                planned_bytes,
                budget_bytes,
            });
        }
    }
    Ok(())
}

/// Structured validation failures for offload contracts.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum OffloadError {
    /// A serialized plan used an unsupported schema version.
    #[error("unsupported offload plan schema version {0}")]
    UnsupportedSchemaVersion(u32),
    /// A logical identifier was empty or whitespace-only.
    #[error("offload unit identifiers must not be empty")]
    EmptyUnitId,
    /// A unit had no bytes to manage.
    #[error("offload unit {id} must contain at least one byte")]
    ZeroSizedUnit {
        /// The invalid unit identifier.
        id: OffloadUnitId,
    },
    /// More than one unit used the same stable identifier.
    #[error("duplicate offload unit identifier: {id}")]
    DuplicateUnitId {
        /// The duplicated identifier.
        id: OffloadUnitId,
    },
    /// Summing unit sizes overflowed the stable byte counter.
    #[error("planned byte total overflowed for the {tier:?} tier")]
    ByteTotalOverflow {
        /// The tier whose total overflowed.
        tier: MemoryTier,
    },
    /// Explicit assignments exceeded a configured finite budget.
    #[error(
        "planned {planned_bytes} bytes for the {tier:?} tier exceed its {budget_bytes}-byte budget"
    )]
    BudgetExceeded {
        /// The over-budget tier.
        tier: MemoryTier,
        /// The checked planned total.
        planned_bytes: u64,
        /// The configured finite budget.
        budget_bytes: u64,
    },
    /// A policy and tier assignment had incompatible meanings.
    #[error("offload unit {id} has contradictory {policy:?}/{tier:?} assignment: {reason}")]
    ContradictoryAssignment {
        /// The invalid unit identifier.
        id: OffloadUnitId,
        /// The requested policy.
        policy: ResidencyPolicy,
        /// The requested tier.
        tier: MemoryTier,
        /// A stable explanation of the contradiction.
        reason: &'static str,
    },
    /// Prefetching was configured with a meaningless zero-unit lookahead.
    #[error("offload prefetch depth must be nonzero")]
    ZeroPrefetchDepth,
}

/// A strongly typed transfer direction between two distinct tiers.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferDirection {
    /// Device memory to host memory.
    DeviceToHost,
    /// Device memory to disk.
    DeviceToDisk,
    /// Host memory to device memory.
    HostToDevice,
    /// Host memory to disk.
    HostToDisk,
    /// Disk to device memory.
    DiskToDevice,
    /// Disk to host memory.
    DiskToHost,
}

impl TransferDirection {
    /// All directions in stable reporting order.
    pub const ALL: [Self; 6] = [
        Self::DeviceToHost,
        Self::DeviceToDisk,
        Self::HostToDevice,
        Self::HostToDisk,
        Self::DiskToDevice,
        Self::DiskToHost,
    ];

    /// Returns the source tier.
    pub const fn source(self) -> MemoryTier {
        match self {
            Self::DeviceToHost | Self::DeviceToDisk => MemoryTier::Device,
            Self::HostToDevice | Self::HostToDisk => MemoryTier::Host,
            Self::DiskToDevice | Self::DiskToHost => MemoryTier::Disk,
        }
    }

    /// Returns the destination tier.
    pub const fn destination(self) -> MemoryTier {
        match self {
            Self::DeviceToHost | Self::DiskToHost => MemoryTier::Host,
            Self::DeviceToDisk | Self::HostToDisk => MemoryTier::Disk,
            Self::HostToDevice | Self::DiskToDevice => MemoryTier::Device,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::DeviceToHost => 0,
            Self::DeviceToDisk => 1,
            Self::HostToDevice => 2,
            Self::HostToDisk => 3,
            Self::DiskToDevice => 4,
            Self::DiskToHost => 5,
        }
    }
}

/// Accumulated transfer observations for one direction.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransferMetrics {
    count: u64,
    bytes: u64,
    duration: Duration,
}

impl TransferMetrics {
    /// Returns the number of recorded transfers.
    pub const fn count(self) -> u64 {
        self.count
    }

    /// Returns the number of recorded bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Returns the accumulated transfer duration.
    pub const fn duration(self) -> Duration {
        self.duration
    }

    fn record(&mut self, bytes: u64, duration: Duration) {
        self.count = self.count.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        self.duration = self.duration.saturating_add(duration);
    }
}

/// The result of one completed prefetch request.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefetchOutcome {
    /// The requested unit was already available at the required tier.
    Hit,
    /// The request required a transfer or load.
    Miss,
}

/// Accumulated prefetch observations.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct PrefetchMetrics {
    requests: u64,
    hits: u64,
    misses: u64,
    stalls: u64,
    stall_duration: Duration,
}

impl PrefetchMetrics {
    /// Returns the number of completed prefetch requests.
    pub const fn requests(self) -> u64 {
        self.requests
    }

    /// Returns the number of prefetch hits.
    pub const fn hits(self) -> u64 {
        self.hits
    }

    /// Returns the number of prefetch misses.
    pub const fn misses(self) -> u64 {
        self.misses
    }

    /// Returns the number of demand waits attributed to prefetching.
    pub const fn stalls(self) -> u64 {
        self.stalls
    }

    /// Returns the accumulated prefetch stall duration.
    pub const fn stall_duration(self) -> Duration {
        self.stall_duration
    }
}

/// Accumulated eviction observations.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvictionMetrics {
    count: u64,
    bytes: u64,
}

impl EvictionMetrics {
    /// Returns the number of recorded evictions.
    pub const fn count(self) -> u64 {
        self.count
    }

    /// Returns the number of recorded evicted bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// A point-in-time sample of backend-managed allocator memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct AllocatorMemoryMetrics {
    active_bytes: u64,
    cached_bytes: u64,
    peak_bytes: u64,
}

impl AllocatorMemoryMetrics {
    /// Creates an explicit backend allocator sample.
    pub const fn new(active_bytes: u64, cached_bytes: u64, peak_bytes: u64) -> Self {
        Self {
            active_bytes,
            cached_bytes,
            peak_bytes,
        }
    }

    /// Returns active backend-managed bytes.
    pub const fn active_bytes(self) -> u64 {
        self.active_bytes
    }

    /// Returns bytes retained by the backend allocator cache.
    pub const fn cached_bytes(self) -> u64 {
        self.cached_bytes
    }

    /// Returns peak active backend-managed bytes.
    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

/// Optional process-level memory and page-fault observations.
///
/// Individual values are absent when they cannot be obtained safely on the
/// current platform. The built-in sampler currently reads Linux `/proc`; it
/// makes no availability guarantee on Apple or Windows targets.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessMetrics {
    rss_bytes: Option<u64>,
    minor_page_faults: Option<u64>,
    major_page_faults: Option<u64>,
}

impl ProcessMetrics {
    /// Creates an explicit process sample.
    pub const fn new(
        rss_bytes: Option<u64>,
        minor_page_faults: Option<u64>,
        major_page_faults: Option<u64>,
    ) -> Self {
        Self {
            rss_bytes,
            minor_page_faults,
            major_page_faults,
        }
    }

    /// Returns resident-set bytes when available.
    pub const fn rss_bytes(self) -> Option<u64> {
        self.rss_bytes
    }

    /// Returns minor page faults when available.
    pub const fn minor_page_faults(self) -> Option<u64> {
        self.minor_page_faults
    }

    /// Returns major page faults when available.
    pub const fn major_page_faults(self) -> Option<u64> {
        self.major_page_faults
    }
}

/// Samples optional process metrics without adding a platform runtime dependency.
pub fn sample_process_metrics() -> ProcessMetrics {
    platform_process_metrics()
}

#[cfg(target_os = "linux")]
fn platform_process_metrics() -> ProcessMetrics {
    let rss_bytes = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let value = line.strip_prefix("VmRSS:")?.trim();
                let kibibytes = value.strip_suffix("kB")?.trim().parse::<u64>().ok()?;
                kibibytes.checked_mul(1024)
            })
        });

    let faults = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            // The parenthesized command name may contain spaces. Fields after
            // its final ')' begin at process-stat field 3 (state).
            let fields = stat.get(stat.rfind(')')? + 1..)?.split_whitespace();
            let fields = fields.collect::<Vec<_>>();
            Some((fields.get(7)?.parse().ok()?, fields.get(9)?.parse().ok()?))
        });

    ProcessMetrics::new(
        rss_bytes,
        faults.map(|value| value.0),
        faults.map(|value| value.1),
    )
}

#[cfg(not(target_os = "linux"))]
fn platform_process_metrics() -> ProcessMetrics {
    ProcessMetrics::default()
}

/// Mutable, single-threaded offload telemetry collector.
///
/// Updates use saturating arithmetic for monotonic counters and durations.
/// Resident bytes are set explicitly, and setting a new value updates the
/// corresponding peak. Wrap this value in a mutex if multiple threads need to
/// record into one collector.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OffloadTelemetry {
    planned_bytes: TierByteTotals,
    resident_bytes: TierByteTotals,
    peak_resident_bytes: TierByteTotals,
    resident_units: TierUnitTotals,
    peak_resident_units: TierUnitTotals,
    transfers: [TransferMetrics; 6],
    prefetch: PrefetchMetrics,
    tier_prefetch: [PrefetchMetrics; 3],
    evictions: EvictionMetrics,
    tier_evictions: [EvictionMetrics; 3],
    allocator_memory: Option<AllocatorMemoryMetrics>,
    process: ProcessMetrics,
    process_sampled: bool,
}

impl OffloadTelemetry {
    /// Creates a collector initialized with a validated plan's byte totals.
    pub fn from_plan(plan: &OffloadPlan) -> Self {
        Self {
            planned_bytes: plan.planned_bytes,
            ..Self::default()
        }
    }

    /// Replaces the planned byte totals recorded by this collector.
    pub fn set_planned_bytes(&mut self, planned_bytes: TierByteTotals) {
        self.planned_bytes = planned_bytes;
    }

    /// Sets current resident bytes and updates the peak for `tier`.
    pub fn set_resident_bytes(&mut self, tier: MemoryTier, bytes: u64) {
        self.resident_bytes.set(tier, bytes);
        if bytes > self.peak_resident_bytes.get(tier) {
            self.peak_resident_bytes.set(tier, bytes);
        }
    }

    /// Sets current resident units and updates the peak for `tier`.
    pub fn set_resident_units(&mut self, tier: MemoryTier, units: usize) {
        self.resident_units.set(tier, units);
        if units > self.peak_resident_units.get(tier) {
            self.peak_resident_units.set(tier, units);
        }
    }

    /// Records one completed transfer using saturating counter updates.
    pub fn record_transfer(
        &mut self,
        direction: TransferDirection,
        bytes: u64,
        duration: Duration,
    ) {
        self.transfers[direction.index()].record(bytes, duration);
    }

    /// Records one completed prefetch request and its outcome.
    pub fn record_prefetch(&mut self, outcome: PrefetchOutcome) {
        self.prefetch.requests = self.prefetch.requests.saturating_add(1);
        match outcome {
            PrefetchOutcome::Hit => {
                self.prefetch.hits = self.prefetch.hits.saturating_add(1);
            }
            PrefetchOutcome::Miss => {
                self.prefetch.misses = self.prefetch.misses.saturating_add(1);
            }
        }
    }

    /// Records a cache request both globally and for its target tier.
    pub fn record_tier_prefetch(&mut self, tier: MemoryTier, outcome: PrefetchOutcome) {
        self.record_prefetch(outcome);
        let metrics = &mut self.tier_prefetch[tier.index()];
        metrics.requests = metrics.requests.saturating_add(1);
        match outcome {
            PrefetchOutcome::Hit => metrics.hits = metrics.hits.saturating_add(1),
            PrefetchOutcome::Miss => metrics.misses = metrics.misses.saturating_add(1),
        }
    }

    /// Records a demand stall while waiting for a prefetched unit.
    pub fn record_prefetch_stall(&mut self, duration: Duration) {
        self.prefetch.stalls = self.prefetch.stalls.saturating_add(1);
        self.prefetch.stall_duration = self.prefetch.stall_duration.saturating_add(duration);
    }

    /// Records one eviction using saturating counter updates.
    pub fn record_eviction(&mut self, bytes: u64) {
        self.evictions.count = self.evictions.count.saturating_add(1);
        self.evictions.bytes = self.evictions.bytes.saturating_add(bytes);
    }

    /// Records an eviction both globally and for its source tier.
    pub fn record_tier_eviction(&mut self, tier: MemoryTier, bytes: u64) {
        self.record_eviction(bytes);
        let metrics = &mut self.tier_evictions[tier.index()];
        metrics.count = metrics.count.saturating_add(1);
        metrics.bytes = metrics.bytes.saturating_add(bytes);
    }

    /// Records an allocator sample supplied by the selected backend.
    pub fn record_allocator_memory(&mut self, metrics: AllocatorMemoryMetrics) {
        self.allocator_memory = Some(metrics);
    }

    /// Records an externally obtained process sample.
    pub fn record_process_metrics(&mut self, metrics: ProcessMetrics) {
        self.process = metrics;
        self.process_sampled = true;
    }

    /// Updates process observations using the built-in optional sampler.
    pub fn sample_process_metrics(&mut self) {
        self.process = sample_process_metrics();
        self.process_sampled = true;
    }

    /// Returns an immutable point-in-time report.
    pub fn snapshot(&self) -> OffloadReport {
        OffloadReport {
            planned_bytes: self.planned_bytes,
            resident_bytes: self.resident_bytes,
            peak_resident_bytes: self.peak_resident_bytes,
            resident_units: self.resident_units,
            peak_resident_units: self.peak_resident_units,
            transfers: self.transfers,
            prefetch: self.prefetch,
            tier_prefetch: self.tier_prefetch,
            evictions: self.evictions,
            tier_evictions: self.tier_evictions,
            allocator_memory: self.allocator_memory,
            process: self.process,
            process_sampled: self.process_sampled,
        }
    }

    /// Clears all configuration and observations, including resident peaks.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Immutable point-in-time offload telemetry report.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct OffloadReport {
    planned_bytes: TierByteTotals,
    resident_bytes: TierByteTotals,
    peak_resident_bytes: TierByteTotals,
    resident_units: TierUnitTotals,
    peak_resident_units: TierUnitTotals,
    transfers: [TransferMetrics; 6],
    prefetch: PrefetchMetrics,
    tier_prefetch: [PrefetchMetrics; 3],
    evictions: EvictionMetrics,
    tier_evictions: [EvictionMetrics; 3],
    allocator_memory: Option<AllocatorMemoryMetrics>,
    process: ProcessMetrics,
    process_sampled: bool,
}

impl OffloadReport {
    /// Returns planned bytes per tier.
    pub const fn planned_bytes(&self) -> TierByteTotals {
        self.planned_bytes
    }

    /// Returns current resident bytes per tier.
    pub const fn resident_bytes(&self) -> TierByteTotals {
        self.resident_bytes
    }

    /// Returns peak resident bytes per tier.
    pub const fn peak_resident_bytes(&self) -> TierByteTotals {
        self.peak_resident_bytes
    }

    /// Returns current resident-unit counts per tier.
    pub const fn resident_units(&self) -> TierUnitTotals {
        self.resident_units
    }

    /// Returns peak resident-unit counts per tier.
    pub const fn peak_resident_units(&self) -> TierUnitTotals {
        self.peak_resident_units
    }

    /// Returns accumulated metrics for one transfer direction.
    pub const fn transfer(&self, direction: TransferDirection) -> TransferMetrics {
        self.transfers[direction.index()]
    }

    /// Returns accumulated prefetch metrics.
    pub const fn prefetch(&self) -> PrefetchMetrics {
        self.prefetch
    }

    /// Returns cache request metrics for one target tier.
    pub const fn tier_prefetch(&self, tier: MemoryTier) -> PrefetchMetrics {
        self.tier_prefetch[tier.index()]
    }

    /// Returns accumulated eviction metrics.
    pub const fn evictions(&self) -> EvictionMetrics {
        self.evictions
    }

    /// Returns eviction metrics for one source tier.
    pub const fn tier_evictions(&self, tier: MemoryTier) -> EvictionMetrics {
        self.tier_evictions[tier.index()]
    }

    /// Returns the latest backend allocator sample, if one was recorded.
    pub const fn allocator_memory(&self) -> Option<AllocatorMemoryMetrics> {
        self.allocator_memory
    }

    /// Returns the latest optional process sample.
    pub const fn process_metrics(&self) -> ProcessMetrics {
        self.process
    }

    /// Returns whether process sampling was requested, including unsupported platforms.
    pub const fn process_sampled(&self) -> bool {
        self.process_sampled
    }
}

/// Logical state of one materialized tier copy.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidentCopyStatus {
    bytes: u64,
    pins: u64,
    in_flight: Option<u64>,
}

impl ResidentCopyStatus {
    /// Charged physical bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Active ownership leases preventing eviction.
    pub const fn pins(self) -> u64 {
        self.pins
    }

    /// Exact transfer generation awaiting disposition.
    pub const fn in_flight(self) -> Option<u64> {
        self.in_flight
    }
}

/// Point-in-time logical residency for one planned unit.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnitResidencyReport {
    id: OffloadUnitId,
    planned_tier: MemoryTier,
    policy: ResidencyPolicy,
    expected_bytes: u64,
    host_allocated_bytes: u64,
    device_allocated_bytes: u64,
    host_resident: bool,
    device_resident: bool,
    host_pins: u64,
    device_pins: u64,
    active_window: bool,
}

impl UnitResidencyReport {
    /// Stable unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }
    /// Initial tier selected by the plan.
    pub const fn planned_tier(&self) -> MemoryTier {
        self.planned_tier
    }
    /// Operational lifetime policy.
    pub const fn policy(&self) -> ResidencyPolicy {
        self.policy
    }
    /// Planned logical bytes.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }
    /// Charged host allocation capacity.
    pub const fn host_allocated_bytes(&self) -> u64 {
        self.host_allocated_bytes
    }
    /// Charged execution-memory bytes.
    pub const fn device_allocated_bytes(&self) -> u64 {
        self.device_allocated_bytes
    }
    /// Whether a host copy is logically resident.
    pub const fn host_resident(&self) -> bool {
        self.host_resident
    }
    /// Whether an execution-memory copy is logically resident.
    pub const fn device_resident(&self) -> bool {
        self.device_resident
    }
    /// Active host-copy leases.
    pub const fn host_pins(&self) -> u64 {
        self.host_pins
    }
    /// Active execution-copy leases.
    pub const fn device_pins(&self) -> u64 {
        self.device_pins
    }
    /// Whether any named execution window protects the unit.
    pub const fn active_window(&self) -> bool {
        self.active_window
    }
}

/// A copy the backend must release after a ledger transition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EvictedResidencyCopy {
    /// Logical unit whose backend storage must be released.
    pub id: OffloadUnitId,
    /// Tier of the released backend storage.
    pub tier: MemoryTier,
    /// Bytes removed from ledger accounting.
    pub bytes: u64,
}

/// One unit preventing an automatic capacity reservation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidencyBlocker {
    /// Stable logical identifier.
    pub id: OffloadUnitId,
    /// Whether lifetime policy forbids eviction.
    pub pinned: bool,
    /// Active ownership leases.
    pub in_use: u64,
    /// Whether an execution window protects the unit.
    pub active_window: bool,
    /// Whether the current atomic request protects the unit.
    pub request_protected: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum CopyLifecycle {
    Reserved,
    Resident,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct LedgerCopy {
    lifecycle: CopyLifecycle,
    bytes: u64,
    pins: u64,
    last_used: u64,
    frequency: u64,
    in_flight: Option<u64>,
}

impl LedgerCopy {
    fn status(self) -> Option<ResidentCopyStatus> {
        (self.lifecycle == CopyLifecycle::Resident).then_some(ResidentCopyStatus {
            bytes: self.bytes,
            pins: self.pins,
            in_flight: self.in_flight,
        })
    }
}

#[derive(Debug, Clone)]
struct LedgerUnit {
    spec: OffloadUnitSpec,
    host: Option<LedgerCopy>,
    device: Option<LedgerCopy>,
}

impl LedgerUnit {
    fn copy(&self, tier: MemoryTier) -> Option<&LedgerCopy> {
        match tier {
            MemoryTier::Host => self.host.as_ref(),
            MemoryTier::Device => self.device.as_ref(),
            MemoryTier::Disk => None,
        }
    }

    fn copy_mut(&mut self, tier: MemoryTier) -> Option<&mut LedgerCopy> {
        match tier {
            MemoryTier::Host => self.host.as_mut(),
            MemoryTier::Device => self.device.as_mut(),
            MemoryTier::Disk => None,
        }
    }

    fn slot_mut(&mut self, tier: MemoryTier) -> Option<&mut Option<LedgerCopy>> {
        match tier {
            MemoryTier::Host => Some(&mut self.host),
            MemoryTier::Device => Some(&mut self.device),
            MemoryTier::Disk => None,
        }
    }
}

/// Backend-neutral ownership and capacity state for one residency plan.
///
/// Backends mirror each resident ledger copy with concrete storage. Ledger
/// transitions return every copy whose storage must be released, so tensor or
/// buffer destruction never needs to be hidden behind an untyped callback.
#[derive(Debug)]
pub struct ResidencyLedger {
    plan: OffloadPlan,
    units: BTreeMap<OffloadUnitId, LedgerUnit>,
    group_windows: BTreeMap<(String, MemoryTier), BTreeSet<OffloadUnitId>>,
    active_windows: BTreeMap<MemoryTier, BTreeSet<OffloadUnitId>>,
    telemetry: OffloadTelemetry,
    resident_bytes: TierByteTotals,
    tick: u64,
    next_transfer_generation: u64,
    initialized: bool,
}

impl ResidencyLedger {
    /// Creates empty ownership state for every unit in a validated plan.
    pub fn new(plan: OffloadPlan) -> Self {
        let units = plan
            .units()
            .iter()
            .cloned()
            .map(|spec| {
                (
                    spec.id().clone(),
                    LedgerUnit {
                        spec,
                        host: None,
                        device: None,
                    },
                )
            })
            .collect();
        let telemetry = OffloadTelemetry::from_plan(&plan);
        Self {
            plan,
            units,
            group_windows: BTreeMap::new(),
            active_windows: BTreeMap::new(),
            telemetry,
            resident_bytes: TierByteTotals::default(),
            tick: 0,
            next_transfer_generation: 1,
            initialized: false,
        }
    }

    /// Validated plan governing this ledger.
    pub const fn plan(&self) -> &OffloadPlan {
        &self.plan
    }

    /// Whether initial planned materialization has completed.
    pub const fn initialized(&self) -> bool {
        self.initialized
    }

    /// Marks initial planned materialization complete.
    pub fn mark_initialized(&mut self) {
        self.initialized = true;
    }

    /// Fails unless initial planned materialization completed.
    pub fn require_initialized(&self) -> Result<(), ResidencyLedgerError> {
        self.initialized
            .then_some(())
            .ok_or(ResidencyLedgerError::NotInitialized)
    }

    /// Returns whether the plan contains a unit.
    pub fn contains(&self, id: &OffloadUnitId) -> bool {
        self.units.contains_key(id)
    }

    /// Returns one planned unit specification.
    pub fn spec(&self, id: &OffloadUnitId) -> Result<&OffloadUnitSpec, ResidencyLedgerError> {
        self.units
            .get(id)
            .map(|unit| &unit.spec)
            .ok_or_else(|| ResidencyLedgerError::UnknownUnit { id: id.clone() })
    }

    /// Returns a resident copy's logical state.
    pub fn copy_status(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<Option<ResidentCopyStatus>, ResidencyLedgerError> {
        validate_ledger_tier(tier, "copy status")?;
        Ok(self
            .units
            .get(id)
            .ok_or_else(|| ResidencyLedgerError::UnknownUnit { id: id.clone() })?
            .copy(tier)
            .and_then(|copy| copy.status()))
    }

    /// Returns whether a materialized copy exists, including in-flight copies.
    pub fn is_resident(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<bool, ResidencyLedgerError> {
        Ok(self.copy_status(id, tier)?.is_some())
    }

    /// Validates one ordered, duplicate-free batch of known units.
    pub fn validate_batch(
        &self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        validate_ledger_tier(tier, "residency batch")?;
        let mut seen = BTreeSet::new();
        for id in ids {
            if !seen.insert(id) {
                return Err(ResidencyLedgerError::DuplicateBatchUnit);
            }
            self.spec(id)?;
        }
        Ok(())
    }

    /// Reserves capacity for one missing copy and returns backend storage evictions.
    pub fn reserve_copy(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        required_bytes: u64,
        protected: &BTreeSet<OffloadUnitId>,
    ) -> Result<Vec<EvictedResidencyCopy>, ResidencyLedgerError> {
        self.reserve_copies(&[(id.clone(), required_bytes)], tier, protected)
    }

    /// Atomically reserves capacity for a batch of missing copies.
    ///
    /// Admission is planned before any ledger copy is removed or inserted. A
    /// failed request therefore leaves ownership and accounting unchanged.
    pub fn reserve_copies(
        &mut self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        protected: &BTreeSet<OffloadUnitId>,
    ) -> Result<Vec<EvictedResidencyCopy>, ResidencyLedgerError> {
        validate_ledger_tier(tier, "capacity reservation")?;
        let ids = requests
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        self.validate_batch(&ids, tier)?;
        let mut total_required = 0u64;
        for (id, required_bytes) in requests {
            if *required_bytes == 0 {
                return Err(ResidencyLedgerError::ZeroReservation { id: id.clone() });
            }
            let unit = self
                .units
                .get(id)
                .ok_or_else(|| ResidencyLedgerError::UnknownUnit { id: id.clone() })?;
            if unit.copy(tier).is_some() {
                return Err(ResidencyLedgerError::CopyAlreadyExists {
                    id: id.clone(),
                    tier,
                });
            }
            total_required = total_required.checked_add(*required_bytes).ok_or(
                ResidencyLedgerError::ArithmeticOverflow {
                    context: "batch capacity reservation",
                },
            )?;
        }

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut victims = Vec::new();
        if let Some(budget) = self.budget(tier) {
            let needed = self.tier_bytes(tier).checked_add(total_required).ok_or(
                ResidencyLedgerError::ArithmeticOverflow {
                    context: "budget reservation",
                },
            )?;
            let required_release = needed.saturating_sub(budget);
            let mut releasable = 0u64;
            if required_release != 0 {
                for victim in self.eviction_candidates(tier, protected) {
                    let bytes = self
                        .units
                        .get(&victim)
                        .and_then(|unit| unit.copy(tier))
                        .ok_or_else(|| inconsistent(&victim, tier, "capacity eviction planning"))?
                        .bytes;
                    releasable = releasable.checked_add(bytes).ok_or(
                        ResidencyLedgerError::ArithmeticOverflow {
                            context: "eviction capacity planning",
                        },
                    )?;
                    victims.push(victim);
                    if releasable >= required_release {
                        break;
                    }
                }
                if releasable < required_release {
                    return Err(ResidencyLedgerError::BudgetExhausted {
                        requested: requests[0].0.clone(),
                        tier,
                        required_bytes: total_required,
                        budget_bytes: budget,
                        resident_bytes: self.tier_bytes(tier),
                        blocking_units: self.blockers(tier, protected),
                    });
                }
            }
        }

        let mut evicted = Vec::with_capacity(victims.len());
        for victim in victims {
            evicted.push(self.remove_copy(&victim, tier, true)?);
        }

        let charged = self.tier_bytes(tier).checked_add(total_required).ok_or(
            ResidencyLedgerError::ArithmeticOverflow {
                context: "resident byte reservation",
            },
        )?;
        self.set_tier_bytes(tier, charged);
        for (id, required_bytes) in requests {
            let tick = self.next_tick();
            *self
                .units
                .get_mut(id)
                .and_then(|unit| unit.slot_mut(tier))
                .ok_or_else(|| inconsistent(id, tier, "reservation insertion"))? =
                Some(LedgerCopy {
                    lifecycle: CopyLifecycle::Reserved,
                    bytes: *required_bytes,
                    pins: 0,
                    last_used: tick,
                    frequency: 0,
                    in_flight: None,
                });
        }
        self.update_resident_telemetry(tier);
        Ok(evicted)
    }

    /// Publishes backend storage into an existing reservation.
    pub fn publish_reserved(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        actual_bytes: u64,
        in_flight: Option<u64>,
    ) -> Result<(), ResidencyLedgerError> {
        validate_ledger_tier(tier, "copy publication")?;
        let reserved = *self
            .units
            .get(id)
            .and_then(|unit| unit.copy(tier))
            .ok_or_else(|| inconsistent(id, tier, "publication lookup"))?;
        if reserved.lifecycle != CopyLifecycle::Reserved {
            return Err(inconsistent(id, tier, "publication lifecycle"));
        }
        if actual_bytes == 0 {
            return Err(ResidencyLedgerError::ZeroPublication {
                id: id.clone(),
                tier,
            });
        }
        if actual_bytes > reserved.bytes {
            return Err(ResidencyLedgerError::PublicationExceedsReservation {
                id: id.clone(),
                tier,
                reserved_bytes: reserved.bytes,
                actual_bytes,
            });
        }
        let adjusted = self
            .tier_bytes(tier)
            .checked_sub(reserved.bytes)
            .and_then(|bytes| bytes.checked_add(actual_bytes))
            .ok_or_else(|| inconsistent(id, tier, "publication accounting"))?;
        self.set_tier_bytes(tier, adjusted);
        let tick = self.next_tick();
        let copy = self
            .units
            .get_mut(id)
            .and_then(|unit| unit.copy_mut(tier))
            .ok_or_else(|| inconsistent(id, tier, "publication mutation"))?;
        *copy = LedgerCopy {
            lifecycle: CopyLifecycle::Resident,
            bytes: actual_bytes,
            pins: 0,
            last_used: tick,
            frequency: 0,
            in_flight,
        };
        self.update_resident_telemetry(tier);
        Ok(())
    }

    /// Rolls back an unpublished reservation without recording an eviction.
    pub fn rollback_reserved(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        validate_ledger_tier(tier, "reservation rollback")?;
        let copy = self.units.get(id).and_then(|unit| unit.copy(tier)).copied();
        let Some(copy) = copy else {
            return Ok(());
        };
        if copy.lifecycle != CopyLifecycle::Reserved {
            return Ok(());
        }
        self.remove_copy(id, tier, false)?;
        Ok(())
    }

    /// Allocates a stable generation for one exact transfer submission.
    pub fn next_transfer_generation(&mut self) -> Result<u64, ResidencyLedgerError> {
        let generation = self.next_transfer_generation;
        self.next_transfer_generation = self.next_transfer_generation.checked_add(1).ok_or(
            ResidencyLedgerError::ArithmeticOverflow {
                context: "resident transfer generation",
            },
        )?;
        Ok(generation)
    }

    /// Resolves exact transfer completion and returns failed backend copies to release.
    pub fn resolve_transfer(
        &mut self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
        generation: u64,
        succeeded: bool,
    ) -> Result<Vec<EvictedResidencyCopy>, ResidencyLedgerError> {
        validate_ledger_tier(tier, "transfer resolution")?;
        self.validate_batch(ids, tier)?;
        let mut removed = Vec::new();
        for id in ids {
            let matches = self
                .units
                .get(id)
                .and_then(|unit| unit.copy(tier))
                .is_some_and(|copy| {
                    copy.lifecycle == CopyLifecycle::Resident && copy.in_flight == Some(generation)
                });
            if !matches {
                continue;
            }
            if succeeded {
                self.units
                    .get_mut(id)
                    .and_then(|unit| unit.copy_mut(tier))
                    .ok_or_else(|| inconsistent(id, tier, "transfer success resolution"))?
                    .in_flight = None;
            } else {
                removed.push(self.remove_copy(id, tier, false)?);
            }
        }
        Ok(removed)
    }

    /// Pins a resident copy and records weighted demand.
    pub fn pin(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        demand: u64,
    ) -> Result<ResidentCopyStatus, ResidencyLedgerError> {
        validate_ledger_tier(tier, "pin")?;
        let tick = self.next_tick();
        let copy = self
            .units
            .get_mut(id)
            .ok_or_else(|| ResidencyLedgerError::UnknownUnit { id: id.clone() })?
            .copy_mut(tier)
            .filter(|copy| copy.lifecycle == CopyLifecycle::Resident)
            .ok_or_else(|| inconsistent(id, tier, "pin"))?;
        copy.pins = copy
            .pins
            .checked_add(1)
            .ok_or(ResidencyLedgerError::ArithmeticOverflow {
                context: "resident lease count",
            })?;
        copy.last_used = tick;
        copy.frequency = copy.frequency.saturating_add(demand);
        Ok(copy.status().expect("resident copy has status"))
    }

    /// Releases one pin. Unknown or already released copies are ignored for drop safety.
    pub fn unpin(&mut self, id: &OffloadUnitId, tier: MemoryTier) {
        if let Some(copy) = self.units.get_mut(id).and_then(|unit| unit.copy_mut(tier)) {
            copy.pins = copy.pins.saturating_sub(1);
        }
    }

    /// Updates recency for an already resident copy.
    pub fn touch(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        validate_ledger_tier(tier, "touch")?;
        let tick = self.next_tick();
        self.units
            .get_mut(id)
            .ok_or_else(|| ResidencyLedgerError::UnknownUnit { id: id.clone() })?
            .copy_mut(tier)
            .filter(|copy| copy.lifecycle == CopyLifecycle::Resident)
            .ok_or_else(|| inconsistent(id, tier, "touch"))?
            .last_used = tick;
        Ok(())
    }

    /// Replaces one named protected window.
    pub fn set_group_window(
        &mut self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), ResidencyLedgerError> {
        validate_ledger_tier(tier, "protected window")?;
        if group.trim().is_empty() {
            return Err(ResidencyLedgerError::InvalidGroupId);
        }
        self.validate_batch(active, tier)?;
        let key = (group.to_string(), tier);
        if active.is_empty() {
            self.group_windows.remove(&key);
        } else {
            self.group_windows
                .insert(key, active.iter().cloned().collect());
        }
        let union = self
            .group_windows
            .iter()
            .filter(|((_, candidate_tier), _)| *candidate_tier == tier)
            .flat_map(|(_, window)| window.iter().cloned())
            .collect();
        self.active_windows.insert(tier, union);
        Ok(())
    }

    /// Explicitly evicts an unpinned copy and returns backend storage to release.
    pub fn evict(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<Option<EvictedResidencyCopy>, ResidencyLedgerError> {
        validate_ledger_tier(tier, "evict")?;
        let unit = self
            .units
            .get(id)
            .ok_or_else(|| ResidencyLedgerError::UnknownUnit { id: id.clone() })?;
        let Some(copy) = unit.copy(tier).and_then(|copy| copy.status()) else {
            return Ok(None);
        };
        if unit.spec.policy() == ResidencyPolicy::Pinned {
            return Err(ResidencyLedgerError::PinnedEviction {
                id: id.clone(),
                tier,
            });
        }
        if copy.pins != 0 {
            return Err(ResidencyLedgerError::InUseEviction {
                id: id.clone(),
                tier,
                pin_count: copy.pins,
            });
        }
        self.remove_copy(id, tier, true).map(Some)
    }

    /// Records one prefetch result in neutral telemetry.
    pub fn record_prefetch(&mut self, tier: MemoryTier, outcome: PrefetchOutcome) {
        self.telemetry.record_tier_prefetch(tier, outcome);
    }

    /// Records backend transfer observations.
    pub fn record_transfer(
        &mut self,
        direction: TransferDirection,
        bytes: u64,
        duration: Duration,
    ) {
        self.telemetry.record_transfer(direction, bytes, duration);
    }

    /// Records time spent waiting for demand materialization.
    pub fn record_prefetch_stall(&mut self, duration: Duration) {
        self.telemetry.record_prefetch_stall(duration);
    }

    /// Records a backend allocator observation.
    pub fn record_allocator_memory(&mut self, metrics: AllocatorMemoryMetrics) {
        self.telemetry.record_allocator_memory(metrics);
    }

    /// Samples optional process memory using the portable core sampler.
    pub fn sample_process_metrics(&mut self) {
        self.telemetry.sample_process_metrics();
    }

    /// Returns immutable accounting telemetry.
    pub fn telemetry(&self) -> OffloadReport {
        self.telemetry.snapshot()
    }

    /// Returns logical unit reports in stable identifier order.
    pub fn unit_reports(&self) -> Vec<UnitResidencyReport> {
        let active = self.active_window();
        self.units
            .values()
            .map(|unit| {
                let host = unit.host.and_then(LedgerCopy::status);
                let device = unit.device.and_then(LedgerCopy::status);
                UnitResidencyReport {
                    id: unit.spec.id().clone(),
                    planned_tier: unit.spec.tier(),
                    policy: unit.spec.policy(),
                    expected_bytes: unit.spec.bytes(),
                    host_allocated_bytes: host.map_or(0, ResidentCopyStatus::bytes),
                    device_allocated_bytes: device.map_or(0, ResidentCopyStatus::bytes),
                    host_resident: host.is_some(),
                    device_resident: device.is_some(),
                    host_pins: host.map_or(0, ResidentCopyStatus::pins),
                    device_pins: device.map_or(0, ResidentCopyStatus::pins),
                    active_window: active.contains(unit.spec.id()),
                }
            })
            .collect()
    }

    /// Returns the union of every named protected window.
    pub fn active_window(&self) -> BTreeSet<OffloadUnitId> {
        self.active_windows
            .values()
            .flat_map(|window| window.iter().cloned())
            .collect()
    }

    fn budget(&self, tier: MemoryTier) -> Option<u64> {
        match tier {
            MemoryTier::Host => self.plan.config().host_budget_bytes(),
            MemoryTier::Device => self.plan.config().device_budget_bytes(),
            MemoryTier::Disk => None,
        }
    }

    fn tier_bytes(&self, tier: MemoryTier) -> u64 {
        self.resident_bytes.get(tier)
    }

    fn set_tier_bytes(&mut self, tier: MemoryTier, bytes: u64) {
        self.resident_bytes.set(tier, bytes);
    }

    fn eviction_candidates(
        &self,
        tier: MemoryTier,
        protected: &BTreeSet<OffloadUnitId>,
    ) -> Vec<OffloadUnitId> {
        let mut candidates = self
            .units
            .values()
            .filter_map(|unit| {
                let copy = unit.copy(tier)?;
                if copy.lifecycle != CopyLifecycle::Resident
                    || unit.spec.policy() == ResidencyPolicy::Pinned
                    || copy.pins != 0
                    || protected.contains(unit.spec.id())
                    || self
                        .active_windows
                        .get(&tier)
                        .is_some_and(|window| window.contains(unit.spec.id()))
                {
                    return None;
                }
                let priority = match unit.spec.policy() {
                    ResidencyPolicy::Windowed => 0u8,
                    ResidencyPolicy::Cacheable => 1u8,
                    ResidencyPolicy::Pinned => return None,
                };
                let frequency = match self.plan.config().eviction_policy() {
                    CacheEvictionPolicy::LeastRecentlyUsed => 0,
                    CacheEvictionPolicy::LeastFrequentlyUsed => copy.frequency,
                };
                Some((priority, frequency, copy.last_used, unit.spec.id().clone()))
            })
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.into_iter().map(|(_, _, _, id)| id).collect()
    }

    fn blockers(
        &self,
        tier: MemoryTier,
        protected: &BTreeSet<OffloadUnitId>,
    ) -> Vec<ResidencyBlocker> {
        self.units
            .values()
            .filter_map(|unit| {
                let copy = unit.copy(tier)?;
                if copy.lifecycle != CopyLifecycle::Resident {
                    return None;
                }
                let pinned = unit.spec.policy() == ResidencyPolicy::Pinned;
                let active_window = self
                    .active_windows
                    .get(&tier)
                    .is_some_and(|window| window.contains(unit.spec.id()));
                let request_protected = protected.contains(unit.spec.id());
                (pinned || copy.pins != 0 || active_window || request_protected).then(|| {
                    ResidencyBlocker {
                        id: unit.spec.id().clone(),
                        pinned,
                        in_use: copy.pins,
                        active_window,
                        request_protected,
                    }
                })
            })
            .collect()
    }

    fn remove_copy(
        &mut self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        record_eviction: bool,
    ) -> Result<EvictedResidencyCopy, ResidencyLedgerError> {
        let copy = self
            .units
            .get_mut(id)
            .and_then(|unit| unit.slot_mut(tier))
            .and_then(Option::take)
            .ok_or_else(|| inconsistent(id, tier, "copy removal"))?;
        let bytes = self
            .tier_bytes(tier)
            .checked_sub(copy.bytes)
            .ok_or_else(|| inconsistent(id, tier, "copy removal accounting"))?;
        self.set_tier_bytes(tier, bytes);
        self.update_resident_telemetry(tier);
        if record_eviction {
            self.telemetry.record_tier_eviction(tier, copy.bytes);
        }
        Ok(EvictedResidencyCopy {
            id: id.clone(),
            tier,
            bytes: copy.bytes,
        })
    }

    fn update_resident_telemetry(&mut self, tier: MemoryTier) {
        let units = self
            .units
            .values()
            .filter(|unit| unit.copy(tier).and_then(|copy| copy.status()).is_some())
            .count();
        self.telemetry
            .set_resident_bytes(tier, self.tier_bytes(tier));
        self.telemetry.set_resident_units(tier, units);
    }

    fn next_tick(&mut self) -> u64 {
        if self.tick == u64::MAX {
            for unit in self.units.values_mut() {
                if let Some(copy) = unit.host.as_mut() {
                    copy.last_used /= 2;
                }
                if let Some(copy) = unit.device.as_mut() {
                    copy.last_used /= 2;
                }
            }
            self.tick /= 2;
        }
        self.tick += 1;
        self.tick
    }
}

fn validate_ledger_tier(
    tier: MemoryTier,
    operation: &'static str,
) -> Result<(), ResidencyLedgerError> {
    if tier == MemoryTier::Disk {
        Err(ResidencyLedgerError::InvalidTargetTier { operation })
    } else {
        Ok(())
    }
}

fn inconsistent(
    id: &OffloadUnitId,
    tier: MemoryTier,
    operation: &'static str,
) -> ResidencyLedgerError {
    ResidencyLedgerError::StateInconsistent {
        id: id.clone(),
        tier,
        operation,
    }
}

/// Backend-neutral residency ownership failure.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidencyLedgerError {
    /// Initial planned materialization has not completed.
    #[error("residency manager has not been initialized")]
    NotInitialized,
    /// A named protected window had no stable identity.
    #[error("resident execution group id must not be empty")]
    InvalidGroupId,
    /// One batch named the same logical unit more than once.
    #[error("batched residency acquisition contains a duplicate unit")]
    DuplicateBatchUnit,
    /// A request referenced an unknown logical unit.
    #[error("unknown residency unit: {id}")]
    UnknownUnit {
        /// Missing unit.
        id: OffloadUnitId,
    },
    /// Disk is not a materialized target tier.
    #[error("{operation} requires a host or device tier")]
    InvalidTargetTier {
        /// Invalid operation.
        operation: &'static str,
    },
    /// A zero-byte physical reservation was requested.
    #[error("residency unit {id} cannot reserve zero bytes")]
    ZeroReservation {
        /// Invalid unit.
        id: OffloadUnitId,
    },
    /// A backend attempted to reserve an existing copy.
    #[error("residency unit {id} already has a {tier:?} copy")]
    CopyAlreadyExists {
        /// Existing unit.
        id: OffloadUnitId,
        /// Existing tier.
        tier: MemoryTier,
    },
    /// Backend storage exceeded the capacity reserved for its publication.
    #[error("residency unit {id} published {actual_bytes} bytes to {tier:?} after reserving only {reserved_bytes}")]
    PublicationExceedsReservation {
        /// Published unit.
        id: OffloadUnitId,
        /// Target tier.
        tier: MemoryTier,
        /// Capacity charged before materialization.
        reserved_bytes: u64,
        /// Actual materialized capacity.
        actual_bytes: u64,
    },
    /// Backend storage publication had no physical capacity.
    #[error("residency unit {id} cannot publish a zero-byte copy to {tier:?}")]
    ZeroPublication {
        /// Published unit.
        id: OffloadUnitId,
        /// Target tier.
        tier: MemoryTier,
    },
    /// Checked lifecycle arithmetic overflowed.
    #[error("residency arithmetic overflow during {context}")]
    ArithmeticOverflow {
        /// Stable operation description.
        context: &'static str,
    },
    /// No safe victim could satisfy a finite tier budget.
    #[error("cannot reserve {required_bytes} bytes for {requested} in {tier:?}: {resident_bytes}/{budget_bytes} bytes resident")]
    BudgetExhausted {
        /// Requested unit.
        requested: OffloadUnitId,
        /// Requested tier.
        tier: MemoryTier,
        /// Required physical bytes.
        required_bytes: u64,
        /// Finite tier budget.
        budget_bytes: u64,
        /// Currently charged bytes.
        resident_bytes: u64,
        /// Units preventing eviction.
        blocking_units: Vec<ResidencyBlocker>,
    },
    /// Explicit eviction contradicted pinned lifetime policy.
    #[error("cannot evict pinned residency unit {id} from {tier:?}")]
    PinnedEviction {
        /// Pinned unit.
        id: OffloadUnitId,
        /// Resident tier.
        tier: MemoryTier,
    },
    /// Explicit eviction targeted leased storage.
    #[error("cannot evict residency unit {id} from {tier:?} while {pin_count} leases are active")]
    InUseEviction {
        /// Leased unit.
        id: OffloadUnitId,
        /// Resident tier.
        tier: MemoryTier,
        /// Active leases.
        pin_count: u64,
    },
    /// Backend storage and ledger transitions were applied out of order.
    #[error("residency ledger state is inconsistent for {id} in {tier:?} during {operation}")]
    StateInconsistent {
        /// Unit whose ownership invariant was violated.
        id: OffloadUnitId,
        /// Tier whose copy state was inconsistent.
        tier: MemoryTier,
        /// Stable transition description.
        operation: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(id: &str, bytes: u64, policy: ResidencyPolicy, tier: MemoryTier) -> OffloadUnitSpec {
        OffloadUnitSpec::new(OffloadUnitId::new(id).unwrap(), bytes, policy, tier).unwrap()
    }

    #[test]
    fn explicit_plan_is_validated_sorted_and_inspectable() {
        let config = OffloadConfig::new(Some(80), Some(40), 2).unwrap();
        let plan = OffloadPlan::new(
            config,
            [
                unit("layer.2", 20, ResidencyPolicy::Windowed, MemoryTier::Disk),
                unit("layer.0", 40, ResidencyPolicy::Pinned, MemoryTier::Device),
                unit("layer.1", 30, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
        )
        .unwrap();

        assert_eq!(
            plan.units()
                .iter()
                .map(|unit| unit.id().as_str())
                .collect::<Vec<_>>(),
            ["layer.0", "layer.1", "layer.2"]
        );
        assert_eq!(plan.config(), config);
        assert_eq!(plan.planned_bytes(), TierByteTotals::new(40, 30, 20));
        assert_eq!(
            plan.unit(&OffloadUnitId::new("layer.1").unwrap())
                .unwrap()
                .bytes(),
            30
        );
    }

    #[test]
    fn plan_and_report_serialization_preserve_validated_state() {
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(80), Some(40), 2)
                .unwrap()
                .with_eviction_policy(CacheEvictionPolicy::LeastFrequentlyUsed),
            [
                unit("layer.1", 20, ResidencyPolicy::Windowed, MemoryTier::Disk),
                unit("layer.0", 40, ResidencyPolicy::Pinned, MemoryTier::Device),
            ],
        )
        .unwrap();
        let encoded = serde_json::to_string(&plan).unwrap();
        let decoded: OffloadPlan = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.schema_version(), OFFLOAD_PLAN_SCHEMA_VERSION);

        let report = OffloadTelemetry::from_plan(&plan).snapshot();
        let encoded = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<OffloadReport>(&encoded).unwrap(),
            report
        );
    }

    #[test]
    fn deserialization_rejects_invalid_plan_instead_of_bypassing_constructors() {
        let invalid = serde_json::json!({
            "schema_version": OFFLOAD_PLAN_SCHEMA_VERSION,
            "config": {
                "device_budget_bytes": 1,
                "host_budget_bytes": null,
                "prefetch_depth": 1,
                "eviction_policy": "least_recently_used"
            },
            "units": [{
                "id": "layer.0",
                "bytes": 2,
                "policy": "pinned",
                "tier": "device"
            }]
        });
        assert!(serde_json::from_value::<OffloadPlan>(invalid).is_err());
    }

    #[test]
    fn duplicate_identifiers_are_rejected_deterministically() {
        let duplicate = OffloadPlan::new(
            OffloadConfig::default(),
            [
                unit("b", 1, ResidencyPolicy::Cacheable, MemoryTier::Host),
                unit("a", 1, ResidencyPolicy::Pinned, MemoryTier::Device),
                unit("a", 2, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
        )
        .unwrap_err();
        assert_eq!(
            duplicate,
            OffloadError::DuplicateUnitId {
                id: OffloadUnitId::new("a").unwrap()
            }
        );
    }

    #[test]
    fn finite_tier_budgets_are_enforced() {
        let error = OffloadPlan::new(
            OffloadConfig::new(Some(9), None, 1).unwrap(),
            [unit(
                "weights",
                10,
                ResidencyPolicy::Pinned,
                MemoryTier::Device,
            )],
        )
        .unwrap_err();
        assert_eq!(
            error,
            OffloadError::BudgetExceeded {
                tier: MemoryTier::Device,
                planned_bytes: 10,
                budget_bytes: 9,
            }
        );
    }

    #[test]
    fn byte_total_overflow_is_reported() {
        let error = OffloadPlan::new(
            OffloadConfig::default(),
            [
                unit("a", u64::MAX, ResidencyPolicy::Cacheable, MemoryTier::Host),
                unit("b", 1, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            OffloadError::ByteTotalOverflow {
                tier: MemoryTier::Host
            }
        );
    }

    #[test]
    fn meaningless_and_contradictory_inputs_are_rejected() {
        assert_eq!(
            OffloadConfig::new(None, None, 0),
            Err(OffloadError::ZeroPrefetchDepth)
        );
        let id = OffloadUnitId::new("empty").unwrap();
        assert_eq!(
            OffloadUnitSpec::new(id.clone(), 0, ResidencyPolicy::Cacheable, MemoryTier::Host),
            Err(OffloadError::ZeroSizedUnit { id })
        );
        assert!(matches!(
            OffloadUnitSpec::new(
                OffloadUnitId::new("pinned-disk").unwrap(),
                1,
                ResidencyPolicy::Pinned,
                MemoryTier::Disk
            ),
            Err(OffloadError::ContradictoryAssignment { .. })
        ));
    }

    #[test]
    fn telemetry_accounts_for_residency_activity_and_runtime_samples() {
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [unit("a", 10, ResidencyPolicy::Pinned, MemoryTier::Device)],
        )
        .unwrap();
        let mut telemetry = OffloadTelemetry::from_plan(&plan);
        telemetry.set_resident_bytes(MemoryTier::Device, 8);
        telemetry.set_resident_bytes(MemoryTier::Device, 12);
        telemetry.set_resident_bytes(MemoryTier::Device, 6);
        telemetry.set_resident_units(MemoryTier::Device, 1);
        telemetry.set_resident_units(MemoryTier::Device, 2);
        telemetry.set_resident_units(MemoryTier::Device, 1);
        telemetry.record_transfer(TransferDirection::HostToDevice, 5, Duration::from_millis(2));
        telemetry.record_transfer(TransferDirection::HostToDevice, 7, Duration::from_millis(3));
        telemetry.record_tier_prefetch(MemoryTier::Device, PrefetchOutcome::Hit);
        telemetry.record_tier_prefetch(MemoryTier::Host, PrefetchOutcome::Miss);
        telemetry.record_prefetch_stall(Duration::from_millis(4));
        telemetry.record_tier_eviction(MemoryTier::Device, 3);
        telemetry.record_tier_eviction(MemoryTier::Host, 4);
        telemetry.record_allocator_memory(AllocatorMemoryMetrics::new(11, 12, 13));
        telemetry.record_process_metrics(ProcessMetrics::new(Some(14), Some(15), Some(16)));

        let report = telemetry.snapshot();
        assert_eq!(report.planned_bytes().get(MemoryTier::Device), 10);
        assert_eq!(report.resident_bytes().get(MemoryTier::Device), 6);
        assert_eq!(report.peak_resident_bytes().get(MemoryTier::Device), 12);
        assert_eq!(report.resident_units().get(MemoryTier::Device), 1);
        assert_eq!(report.peak_resident_units().get(MemoryTier::Device), 2);
        assert_eq!(report.transfer(TransferDirection::HostToDevice).count(), 2);
        assert_eq!(report.transfer(TransferDirection::HostToDevice).bytes(), 12);
        assert_eq!(
            report.transfer(TransferDirection::HostToDevice).duration(),
            Duration::from_millis(5)
        );
        assert_eq!(report.prefetch().requests(), 2);
        assert_eq!(report.prefetch().hits(), 1);
        assert_eq!(report.prefetch().misses(), 1);
        assert_eq!(report.prefetch().stalls(), 1);
        assert_eq!(report.prefetch().stall_duration(), Duration::from_millis(4));
        assert_eq!(report.evictions().count(), 2);
        assert_eq!(report.evictions().bytes(), 7);
        assert_eq!(report.tier_prefetch(MemoryTier::Device).hits(), 1);
        assert_eq!(report.tier_prefetch(MemoryTier::Host).misses(), 1);
        assert_eq!(report.tier_evictions(MemoryTier::Device).bytes(), 3);
        assert_eq!(report.tier_evictions(MemoryTier::Host).bytes(), 4);
        assert_eq!(report.allocator_memory().unwrap().peak_bytes(), 13);
        assert_eq!(report.process_metrics().rss_bytes(), Some(14));
        assert!(report.process_sampled());
    }

    #[test]
    fn snapshot_is_immutable_and_reset_clears_everything() {
        let mut telemetry = OffloadTelemetry::default();
        telemetry.set_planned_bytes(TierByteTotals::new(1, 2, 3));
        telemetry.set_resident_bytes(MemoryTier::Host, 4);
        telemetry.set_resident_units(MemoryTier::Host, 1);
        let snapshot = telemetry.snapshot();

        telemetry.set_resident_bytes(MemoryTier::Host, 9);
        telemetry.record_eviction(5);
        assert_eq!(snapshot.resident_bytes().get(MemoryTier::Host), 4);
        assert_eq!(snapshot.resident_units().get(MemoryTier::Host), 1);
        assert_eq!(snapshot.evictions(), EvictionMetrics::default());
        assert!(!snapshot.process_sampled());

        telemetry.reset();
        assert_eq!(telemetry.snapshot(), OffloadTelemetry::default().snapshot());
    }

    #[test]
    fn telemetry_counters_saturate() {
        let mut telemetry = OffloadTelemetry::default();
        telemetry.transfers[TransferDirection::DiskToHost.index()] = TransferMetrics {
            count: u64::MAX,
            bytes: u64::MAX,
            duration: Duration::MAX,
        };
        telemetry.evictions = EvictionMetrics {
            count: u64::MAX,
            bytes: u64::MAX,
        };
        telemetry.record_transfer(TransferDirection::DiskToHost, 1, Duration::from_nanos(1));
        telemetry.record_eviction(1);
        let report = telemetry.snapshot();
        assert_eq!(
            report.transfer(TransferDirection::DiskToHost).count(),
            u64::MAX
        );
        assert_eq!(
            report.transfer(TransferDirection::DiskToHost).bytes(),
            u64::MAX
        );
        assert_eq!(
            report.transfer(TransferDirection::DiskToHost).duration(),
            Duration::MAX
        );
        assert_eq!(report.evictions().count(), u64::MAX);
        assert_eq!(report.evictions().bytes(), u64::MAX);
    }

    #[test]
    fn optional_process_sampler_never_requires_platform_support() {
        let metrics = sample_process_metrics();
        if let Some(rss_bytes) = metrics.rss_bytes() {
            assert!(rss_bytes > 0);
        }
        let mut telemetry = OffloadTelemetry::default();
        telemetry.sample_process_metrics();
        let report = telemetry.snapshot();
        let _ = report.process_metrics();
        assert!(report.process_sampled());
    }

    fn disk_ledger(
        device_budget: Option<u64>,
        units: impl IntoIterator<Item = (&'static str, u64, ResidencyPolicy)>,
    ) -> ResidencyLedger {
        let plan = OffloadPlan::new(
            OffloadConfig::new(device_budget, None, 1).unwrap(),
            units
                .into_iter()
                .map(|(id, bytes, policy)| unit(id, bytes, policy, MemoryTier::Disk)),
        )
        .unwrap();
        ResidencyLedger::new(plan)
    }

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    fn publish_device(
        ledger: &mut ResidencyLedger,
        unit: &str,
        bytes: u64,
        generation: Option<u64>,
    ) {
        ledger
            .reserve_copy(&id(unit), MemoryTier::Device, bytes, &BTreeSet::new())
            .unwrap();
        ledger
            .publish_reserved(&id(unit), MemoryTier::Device, bytes, generation)
            .unwrap();
    }

    #[test]
    fn ledger_owns_publication_pins_and_exact_completion() {
        let mut ledger = disk_ledger(Some(8), [("a", 8, ResidencyPolicy::Cacheable)]);
        assert_eq!(
            ledger.require_initialized(),
            Err(ResidencyLedgerError::NotInitialized)
        );
        ledger.mark_initialized();

        let generation = ledger.next_transfer_generation().unwrap();
        ledger
            .reserve_copy(&id("a"), MemoryTier::Device, 8, &BTreeSet::new())
            .unwrap();
        assert_eq!(
            ledger.copy_status(&id("a"), MemoryTier::Device).unwrap(),
            None
        );
        ledger
            .publish_reserved(&id("a"), MemoryTier::Device, 8, Some(generation))
            .unwrap();
        let pinned = ledger.pin(&id("a"), MemoryTier::Device, 3).unwrap();
        assert_eq!(pinned.pins(), 1);
        assert_eq!(pinned.in_flight(), Some(generation));
        assert!(matches!(
            ledger.evict(&id("a"), MemoryTier::Device),
            Err(ResidencyLedgerError::InUseEviction { pin_count: 1, .. })
        ));

        assert!(ledger
            .resolve_transfer(&[id("a")], MemoryTier::Device, generation + 1, true)
            .unwrap()
            .is_empty());
        assert_eq!(
            ledger
                .copy_status(&id("a"), MemoryTier::Device)
                .unwrap()
                .unwrap()
                .in_flight(),
            Some(generation)
        );
        ledger
            .resolve_transfer(&[id("a")], MemoryTier::Device, generation, true)
            .unwrap();
        ledger.unpin(&id("a"), MemoryTier::Device);
        assert!(ledger
            .evict(&id("a"), MemoryTier::Device)
            .unwrap()
            .is_some());
        assert_eq!(
            ledger.telemetry().resident_bytes().get(MemoryTier::Device),
            0
        );
        assert_eq!(ledger.telemetry().evictions().count(), 1);
    }

    #[test]
    fn failed_exact_completion_discards_copy_and_accounting() {
        let mut ledger = disk_ledger(Some(8), [("a", 8, ResidencyPolicy::Cacheable)]);
        let generation = ledger.next_transfer_generation().unwrap();
        publish_device(&mut ledger, "a", 8, Some(generation));

        let removed = ledger
            .resolve_transfer(&[id("a")], MemoryTier::Device, generation, false)
            .unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].id, id("a"));
        assert!(!ledger.is_resident(&id("a"), MemoryTier::Device).unwrap());
        assert_eq!(
            ledger.telemetry().resident_bytes().get(MemoryTier::Device),
            0
        );
        assert_eq!(ledger.telemetry().evictions().count(), 0);
    }

    #[test]
    fn failed_batch_admission_is_atomic() {
        let mut ledger = disk_ledger(
            Some(16),
            [
                ("a", 8, ResidencyPolicy::Cacheable),
                ("b", 8, ResidencyPolicy::Cacheable),
                ("c", 8, ResidencyPolicy::Cacheable),
                ("d", 8, ResidencyPolicy::Cacheable),
            ],
        );
        publish_device(&mut ledger, "a", 8, None);
        publish_device(&mut ledger, "b", 8, None);
        ledger.pin(&id("b"), MemoryTier::Device, 1).unwrap();

        let error = ledger
            .reserve_copies(
                &[(id("c"), 8), (id("d"), 8)],
                MemoryTier::Device,
                &BTreeSet::new(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ResidencyLedgerError::BudgetExhausted {
                required_bytes: 16,
                resident_bytes: 16,
                ..
            }
        ));
        assert!(ledger.is_resident(&id("a"), MemoryTier::Device).unwrap());
        assert!(ledger.is_resident(&id("b"), MemoryTier::Device).unwrap());
        assert!(!ledger.is_resident(&id("c"), MemoryTier::Device).unwrap());
        assert!(!ledger.is_resident(&id("d"), MemoryTier::Device).unwrap());
        assert_eq!(ledger.telemetry().evictions().count(), 0);
    }

    #[test]
    fn batch_reservation_evicts_only_after_complete_admission() {
        let mut ledger = disk_ledger(
            Some(16),
            [
                ("a", 8, ResidencyPolicy::Cacheable),
                ("b", 8, ResidencyPolicy::Cacheable),
                ("c", 8, ResidencyPolicy::Cacheable),
                ("d", 8, ResidencyPolicy::Cacheable),
            ],
        );
        publish_device(&mut ledger, "a", 8, None);
        publish_device(&mut ledger, "b", 8, None);

        let evicted = ledger
            .reserve_copies(
                &[(id("c"), 8), (id("d"), 8)],
                MemoryTier::Device,
                &BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(
            evicted
                .iter()
                .map(|copy| copy.id.clone())
                .collect::<Vec<_>>(),
            [id("a"), id("b")]
        );
        assert!(!ledger.is_resident(&id("a"), MemoryTier::Device).unwrap());
        assert!(!ledger.is_resident(&id("b"), MemoryTier::Device).unwrap());
        assert_eq!(
            ledger.copy_status(&id("c"), MemoryTier::Device).unwrap(),
            None
        );
        assert_eq!(
            ledger.copy_status(&id("d"), MemoryTier::Device).unwrap(),
            None
        );
        ledger
            .rollback_reserved(&id("c"), MemoryTier::Device)
            .unwrap();
        ledger
            .rollback_reserved(&id("d"), MemoryTier::Device)
            .unwrap();
        assert_eq!(
            ledger.telemetry().resident_bytes().get(MemoryTier::Device),
            0
        );
    }

    #[test]
    fn equal_eviction_priority_uses_stable_unit_identity() {
        let mut ledger = disk_ledger(
            Some(16),
            [
                ("a", 8, ResidencyPolicy::Cacheable),
                ("b", 8, ResidencyPolicy::Cacheable),
                ("c", 8, ResidencyPolicy::Cacheable),
            ],
        );
        publish_device(&mut ledger, "a", 8, None);
        publish_device(&mut ledger, "b", 8, None);
        for unit in [id("a"), id("b")] {
            ledger
                .units
                .get_mut(&unit)
                .unwrap()
                .device
                .as_mut()
                .unwrap()
                .last_used = 10;
        }

        let evicted = ledger
            .reserve_copy(&id("c"), MemoryTier::Device, 8, &BTreeSet::new())
            .unwrap();
        assert_eq!(evicted[0].id, id("a"));
    }

    #[test]
    fn request_protection_is_reported_as_a_capacity_blocker() {
        let mut ledger = disk_ledger(
            Some(8),
            [
                ("active", 8, ResidencyPolicy::Cacheable),
                ("next", 8, ResidencyPolicy::Cacheable),
            ],
        );
        publish_device(&mut ledger, "active", 8, None);
        let protected = BTreeSet::from([id("active")]);
        let error = ledger
            .reserve_copy(&id("next"), MemoryTier::Device, 8, &protected)
            .unwrap_err();
        let ResidencyLedgerError::BudgetExhausted { blocking_units, .. } = error else {
            panic!("unexpected error: {error}");
        };
        assert_eq!(blocking_units.len(), 1);
        assert!(blocking_units[0].request_protected);
    }

    #[test]
    fn windows_are_group_scoped_and_protect_capacity() {
        let mut ledger = disk_ledger(
            Some(16),
            [
                ("text", 8, ResidencyPolicy::Windowed),
                ("vision", 8, ResidencyPolicy::Windowed),
                ("next", 8, ResidencyPolicy::Cacheable),
            ],
        );
        publish_device(&mut ledger, "text", 8, None);
        publish_device(&mut ledger, "vision", 8, None);
        ledger
            .set_group_window("text", &[id("text")], MemoryTier::Device)
            .unwrap();
        ledger
            .set_group_window("vision", &[id("vision")], MemoryTier::Device)
            .unwrap();
        assert_eq!(
            ledger.active_window(),
            BTreeSet::from([id("text"), id("vision")])
        );
        assert!(matches!(
            ledger.reserve_copy(&id("next"), MemoryTier::Device, 8, &BTreeSet::new()),
            Err(ResidencyLedgerError::BudgetExhausted { .. })
        ));

        ledger
            .set_group_window("text", &[], MemoryTier::Device)
            .unwrap();
        assert_eq!(ledger.active_window(), BTreeSet::from([id("vision")]));
        let evicted = ledger
            .reserve_copy(&id("next"), MemoryTier::Device, 8, &BTreeSet::new())
            .unwrap();
        assert_eq!(evicted[0].id, id("text"));
    }

    #[test]
    fn unit_reports_round_trip_without_backend_state() {
        let mut ledger = disk_ledger(Some(8), [("a", 8, ResidencyPolicy::Cacheable)]);
        publish_device(&mut ledger, "a", 8, None);
        ledger.pin(&id("a"), MemoryTier::Device, 2).unwrap();
        let report = ledger.unit_reports().remove(0);
        let encoded = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<UnitResidencyReport>(&encoded).unwrap(),
            report
        );
        assert_eq!(report.device_allocated_bytes(), 8);
        assert_eq!(report.device_pins(), 1);
    }

    #[test]
    fn publication_cannot_exceed_reserved_capacity() {
        let mut ledger = disk_ledger(Some(16), [("a", 8, ResidencyPolicy::Cacheable)]);
        ledger
            .reserve_copy(&id("a"), MemoryTier::Device, 8, &BTreeSet::new())
            .unwrap();
        assert!(matches!(
            ledger.publish_reserved(&id("a"), MemoryTier::Device, 0, None),
            Err(ResidencyLedgerError::ZeroPublication { .. })
        ));
        assert!(matches!(
            ledger.publish_reserved(&id("a"), MemoryTier::Device, 9, None),
            Err(ResidencyLedgerError::PublicationExceedsReservation {
                reserved_bytes: 8,
                actual_bytes: 9,
                ..
            })
        ));
        ledger
            .rollback_reserved(&id("a"), MemoryTier::Device)
            .unwrap();
        assert_eq!(
            ledger.telemetry().resident_bytes().get(MemoryTier::Device),
            0
        );
    }
}
