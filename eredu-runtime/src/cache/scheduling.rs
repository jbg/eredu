//! Cache pressure policy over existing page and transfer facts.

use super::{
    CacheBlockLifecycle, CachePoolReport, CacheStoragePhase, LiveCacheDiskPolicy, PagedCacheOptions,
};
use eredu_core::{
    cache::{
        mechanisms::{CachePrefetchFailure, CachePrefetchMechanisms},
        CacheBlockId, CacheTier,
    },
    residency::CacheEvictionPolicy,
};

/// Physical observations for one page in the catalog.
#[derive(Clone, Copy)]
pub struct CachePage<'a> {
    /// Exact logical identity of the page.
    pub id: &'a CacheBlockId,
    /// Physical phase established by the current transfer owner.
    pub phase: CacheStoragePhase,
    /// Whether a completed backing can be used without another transfer.
    pub backed: bool,
}

/// One instruction selected while the cache catalog is locked.
pub enum CacheRebalanceAction {
    /// No work is needed.
    Idle,
    /// Settle the current device-to-host transfer before opening another.
    SettleDemotion,
    /// Release a device copy backed by an existing file.
    ReleaseDevice(CacheBlockId),
    /// Transfer the exact device page to host storage.
    DemoteDevice(CacheBlockId),
    /// Release one host page. Urgent releases are settled before retry.
    ReleaseHost {
        /// Exact victim selected by residency policy.
        id: CacheBlockId,
        /// Whether space is required before the next step.
        urgent: bool,
    },
    /// Settle pending spill I/O before rechecking host pressure.
    SettleSpill,
    /// No permitted movement can make room for this allocation.
    Reject(CachePressure),
}

/// Exact budget and attribution for a rejected rebalancing step.
pub struct CachePressure {
    /// Tier whose budget cannot admit more storage.
    pub tier: CacheTier,
    /// Projected bytes required in that tier.
    pub required: u64,
    /// Admitted byte limit.
    pub budget: u64,
    /// Layer to which the failure is attributed, if any.
    pub layer: Option<usize>,
    /// Whether the shared pool was the limiting budget.
    pub shared_pool: bool,
    /// Whether host pressure requires live spill to be enabled.
    pub needs_live_disk: bool,
}

/// Admits spill space against both published files and work that still owns a reservation.
pub fn admit_cache_spill(
    options: &PagedCacheOptions,
    host_bytes: u64,
    published_bytes: u64,
    pending_bytes: u64,
    block_bytes: u64,
    layer: usize,
) -> Result<&std::path::Path, CachePressure> {
    let LiveCacheDiskPolicy::Enabled {
        directory,
        budget_bytes,
        ..
    } = options.live_disk_policy()
    else {
        return Err(CachePressure {
            tier: CacheTier::Host,
            required: host_bytes,
            budget: options.host_budget_bytes(),
            layer: Some(layer),
            shared_pool: false,
            needs_live_disk: true,
        });
    };
    let required = published_bytes
        .saturating_add(pending_bytes)
        .saturating_add(block_bytes);
    if required > *budget_bytes {
        return Err(CachePressure {
            tier: CacheTier::Disk,
            required,
            budget: *budget_bytes,
            layer: Some(layer),
            shared_pool: false,
            needs_live_disk: false,
        });
    }
    Ok(directory)
}

/// Selects one legal victim using logical leases and observed physical phases.
pub fn select_cache_victim<'a>(
    lifecycle: &CacheBlockLifecycle,
    pages: impl Iterator<Item = CachePage<'a>>,
    tier: CacheTier,
    required: Option<&CacheBlockId>,
    recent_per_layer: usize,
    policy: CacheEvictionPolicy,
) -> Option<CacheBlockId> {
    lifecycle
        .eviction_candidate(
            pages
                .filter(|page| {
                    matches!(
                        (page.phase, tier),
                        (CacheStoragePhase::Device, CacheTier::Device)
                            | (
                                CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked,
                                CacheTier::Host
                            )
                            | (CacheStoragePhase::DiskReady, CacheTier::Disk)
                    )
                })
                .map(|page| page.id.clone()),
            required,
            recent_per_layer,
            policy,
        )
        .expect("physical cache pages have logical lifecycle state")
}

/// The locked observations used for one pressure decision.
pub struct CacheRebalance<'a> {
    /// Lease ownership, access order and protected pages.
    pub lifecycle: &'a CacheBlockLifecycle,
    /// Admitted local tier limits and eviction policy.
    pub options: &'a PagedCacheOptions,
    /// Page that must remain available to the caller.
    pub required: Option<&'a CacheBlockId>,
    /// Whether recent protection may yield to the required page.
    pub allow_recent_eviction: bool,
    /// Observed local device occupancy.
    pub device_bytes: u64,
    /// Observed local host occupancy, including reserved transfers.
    pub host_bytes: u64,
    /// Shared pool occupancy and limits.
    pub pool: CachePoolReport,
}

impl CacheRebalance<'_> {
    /// Chooses the next transfer or wait. Queries host capacity only for a selected page.
    pub fn next<'a, E>(
        &self,
        pages: impl Iterator<Item = CachePage<'a>> + Clone,
        host_capacity: impl FnOnce(&CacheBlockId) -> Result<u64, E>,
        spill_pending: impl Fn() -> bool,
    ) -> Result<CacheRebalanceAction, E> {
        use CacheRebalanceAction as Action;

        let victim = |tier, recent| {
            select_cache_victim(
                self.lifecycle,
                pages.clone(),
                tier,
                self.required,
                recent,
                self.options.eviction_policy(),
            )
        };
        let evict_host = || {
            victim(CacheTier::Host, 0)
                .map(|id| Action::ReleaseHost { id, urgent: true })
                .or_else(|| spill_pending().then_some(Action::SettleSpill))
        };
        let local_device_over = self.device_bytes > self.options.device_budget_bytes();
        let pool_device_over = self.pool.current_device_bytes > self.pool.limits.device_bytes();
        let attributed_layer = self.required.map(|id| id.global_layer);
        if local_device_over || pool_device_over {
            if pages
                .clone()
                .any(|page| page.phase == CacheStoragePhase::DemotingToHost)
            {
                return Ok(Action::SettleDemotion);
            }
            let Some(id) =
                victim(CacheTier::Device, self.options.recent_device_blocks()).or_else(|| {
                    self.allow_recent_eviction
                        .then(|| victim(CacheTier::Device, 0))
                        .flatten()
                })
            else {
                let shared_pool = !local_device_over;
                return Ok(Action::Reject(CachePressure {
                    tier: CacheTier::Device,
                    required: if shared_pool {
                        self.pool.current_device_bytes
                    } else {
                        self.device_bytes
                    },
                    budget: if shared_pool {
                        self.pool.limits.device_bytes()
                    } else {
                        self.options.device_budget_bytes()
                    },
                    layer: attributed_layer,
                    shared_pool,
                    needs_live_disk: false,
                }));
            };
            if pages.clone().any(|page| page.id == &id && page.backed) {
                return Ok(Action::ReleaseDevice(id));
            }
            let capacity = host_capacity(&id)?;
            let budget = self.options.host_budget_bytes();
            let required = self.host_bytes.saturating_add(capacity);
            if required <= budget {
                return Ok(Action::DemoteDevice(id));
            }
            if capacity <= budget {
                if let Some(action) = evict_host() {
                    return Ok(action);
                }
            }
            return Ok(Action::Reject(CachePressure {
                tier: CacheTier::Host,
                required: if capacity > budget {
                    capacity
                } else {
                    required
                },
                budget,
                layer: Some(id.global_layer),
                shared_pool: false,
                needs_live_disk: capacity <= budget
                    && matches!(
                        self.options.live_disk_policy(),
                        LiveCacheDiskPolicy::Disabled
                    ),
            }));
        }

        let local_host_over = self.host_bytes > self.options.host_budget_bytes();
        let pool_host_over = self.pool.current_host_bytes > self.pool.limits.host_bytes();
        if local_host_over || pool_host_over {
            let shared_pool = !local_host_over;
            return Ok(evict_host().unwrap_or(Action::Reject(CachePressure {
                tier: CacheTier::Host,
                required: if shared_pool {
                    self.pool.current_host_bytes
                } else {
                    self.host_bytes
                },
                budget: if shared_pool {
                    self.pool.limits.host_bytes()
                } else {
                    self.options.host_budget_bytes()
                },
                layer: attributed_layer,
                shared_pool,
                needs_live_disk: false,
            })));
        }

        // Start one write before the next host allocation requires a wait.
        let host_full = self.host_bytes >= self.options.host_budget_bytes()
            || self.pool.current_host_bytes >= self.pool.limits.host_bytes();
        if host_full
            && self.host_bytes != 0
            && matches!(
                self.options.live_disk_policy(),
                LiveCacheDiskPolicy::Enabled { .. }
            )
        {
            if let Some(id) = victim(CacheTier::Host, 0) {
                return Ok(Action::ReleaseHost { id, urgent: false });
            }
        }
        Ok(Action::Idle)
    }
}

/// Ordered current-plus-next prefetch. Pending leases retain the prefetched pages.
pub struct CachePagePrefetch<L> {
    ids: std::collections::VecDeque<CacheBlockId>,
    device_budget_bytes: u64,
    pending: std::collections::VecDeque<L>,
}

impl<L> CachePagePrefetch<L> {
    const WINDOW: usize = 2;

    /// Retains the exact page order requested by the consumer.
    pub fn new(ids: Vec<CacheBlockId>, device_budget_bytes: u64) -> Self {
        Self {
            ids: ids.into(),
            device_budget_bytes,
            pending: std::collections::VecDeque::with_capacity(Self::WINDOW),
        }
    }

    /// Number of prepared leases waiting for their consumer.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Gets the next ordered lease, falling back to demand reads under device pressure.
    pub fn next<M: CachePrefetchMechanisms<Lease = L>>(
        &mut self,
        mechanisms: &M,
        context: &M::Context,
    ) -> Result<Option<L>, M::Error> {
        while self.pending.len() < Self::WINDOW {
            let Some(id) = self.ids.front() else { break };
            if !self.pending.is_empty() {
                let pending_bytes = self
                    .pending
                    .iter()
                    .map(M::leased_bytes)
                    .fold(0u64, u64::saturating_add);
                if pending_bytes.saturating_add(mechanisms.page_bytes(id)?)
                    > self.device_budget_bytes
                {
                    break;
                }
            }
            match mechanisms.prepare_page(id, context) {
                Ok(lease) => {
                    self.ids.pop_front();
                    self.pending.push_back(lease);
                }
                Err(CachePrefetchFailure::DevicePressure(_)) if !self.pending.is_empty() => break,
                Err(
                    CachePrefetchFailure::DevicePressure(error)
                    | CachePrefetchFailure::Transfer(error),
                ) => return Err(error),
            }
        }
        Ok(self.pending.pop_front())
    }
}
