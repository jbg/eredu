//! Backend-neutral ownership and admission algorithms for live model caches.
//!
//! A cache backend owns concrete tensors, transfer objects, and persistent
//! files. This module owns both each session's block/lease/tail lifecycle and
//! the process-wide accounting boundary shared by independently managed cache
//! sessions. Reservations are RAII tokens: an admitted resource remains charged
//! until its exact backend transition either publishes the bytes into a
//! registered manager or drops the reservation.

mod executor;
mod lifecycle;
mod persistence;
mod policy;
mod storage;
mod telemetry;
mod worker;

pub use lifecycle::{CacheBlockLifecycle, CacheLifecycleError, MutableCacheTail};
pub use persistence::{
    finalize_prompt_cache_shard, hash_prompt_cache_shard_payload, inspect_prompt_cache,
    resolve_prompt_cache_root, safe_prompt_cache_shard_path, validate_prompt_cache_manifest,
    LiveCacheBlockPublication, LiveCachePublicationError, PromptCachePersistenceError,
    PromptCachePublication, MAX_PROMPT_CACHE_SHARD_HEADER_BYTES, PROMPT_CACHE_CURRENT_FILE,
    PROMPT_CACHE_GENERATIONS_DIRECTORY,
};
pub use policy::{
    CacheResidencyConfigurationError, CacheResidencyPolicy, LiveCacheDiskPolicy, PagedCacheOptions,
};
pub use storage::{
    CacheBlockStorage, CacheHostDemotionOperation, CacheHostPromotion, CacheIoOperation,
    CacheIoOperationKey, CacheIoOperationKind, CacheStorageError, CacheStoragePhase,
};
pub use telemetry::{
    CacheLayerResidencyReport, CacheLayerResidencyStats, CacheResidencyReport,
    CacheResidencyTelemetry, CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
};
pub use worker::{
    CacheIoSubmission, CacheIoSubmissionOutcome, CacheIoTicket, CacheIoWorker, CacheIoWorkerError,
};

use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

static NEXT_CACHE_POOL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CACHE_POOL_RESERVATION_ID: AtomicU64 = AtomicU64::new(1);

/// Process-wide finite limits shared by independently owned live caches.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct CachePoolLimits {
    device_bytes: u64,
    host_bytes: u64,
    transfer_in_flight_bytes: u64,
    disk_bytes: u64,
}

impl CachePoolLimits {
    /// Creates finite aggregate cache limits.
    ///
    /// Device and transfer capacity must be nonzero. Zero host or disk
    /// capacity explicitly disables that tier.
    pub fn new(
        device_bytes: u64,
        host_bytes: u64,
        transfer_in_flight_bytes: u64,
        disk_bytes: u64,
    ) -> Result<Self, CachePoolError> {
        if device_bytes == 0 {
            return Err(CachePoolError::InvalidLimits(
                "cache pool device budget must be nonzero",
            ));
        }
        if transfer_in_flight_bytes == 0 {
            return Err(CachePoolError::InvalidLimits(
                "cache pool transfer-in-flight budget must be nonzero",
            ));
        }
        Ok(Self {
            device_bytes,
            host_bytes,
            transfer_in_flight_bytes,
            disk_bytes,
        })
    }

    /// Aggregate execution-device cache capacity.
    pub const fn device_bytes(self) -> u64 {
        self.device_bytes
    }

    /// Aggregate physical host-allocation capacity.
    pub const fn host_bytes(self) -> u64 {
        self.host_bytes
    }

    /// Aggregate bytes retained until exact transfer completion.
    pub const fn transfer_in_flight_bytes(self) -> u64 {
        self.transfer_in_flight_bytes
    }

    /// Aggregate live-cache disk capacity.
    pub const fn disk_bytes(self) -> u64 {
        self.disk_bytes
    }
}

/// Occupancy on each independently admitted cache resource axis.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CachePoolUsage {
    /// Concrete execution-device allocations.
    pub device_bytes: u64,
    /// Concrete physical host allocations.
    pub host_bytes: u64,
    /// Resources retained by submitted transfers.
    pub transfer_in_flight_bytes: u64,
    /// Live-cache files plus pending file reservations.
    pub disk_bytes: u64,
}

impl CachePoolUsage {
    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            device_bytes: self.device_bytes.checked_add(other.device_bytes)?,
            host_bytes: self.host_bytes.checked_add(other.host_bytes)?,
            transfer_in_flight_bytes: self
                .transfer_in_flight_bytes
                .checked_add(other.transfer_in_flight_bytes)?,
            disk_bytes: self.disk_bytes.checked_add(other.disk_bytes)?,
        })
    }

    fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            device_bytes: self.device_bytes.checked_sub(other.device_bytes)?,
            host_bytes: self.host_bytes.checked_sub(other.host_bytes)?,
            transfer_in_flight_bytes: self
                .transfer_in_flight_bytes
                .checked_sub(other.transfer_in_flight_bytes)?,
            disk_bytes: self.disk_bytes.checked_sub(other.disk_bytes)?,
        })
    }
}

/// Resource axis named by an aggregate admission failure.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePoolResource {
    /// Execution-device cache allocations.
    Device,
    /// Physical host cache allocations.
    Host,
    /// Resources retained until exact transfer completion.
    TransferInFlight,
    /// Live-cache disk files and pending writes.
    Disk,
}

/// Aggregate process-pool occupancy and high-water marks.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct CachePoolReport {
    /// Stable process-local pool identity.
    pub pool_id: u64,
    /// Number of registered live cache managers.
    pub managers: usize,
    /// Current aggregate device bytes.
    pub current_device_bytes: u64,
    /// Peak aggregate device bytes successfully admitted.
    pub peak_device_bytes: u64,
    /// Current aggregate physical host capacity.
    pub current_host_bytes: u64,
    /// Peak aggregate physical host capacity successfully admitted.
    pub peak_host_bytes: u64,
    /// Current bytes retained by in-flight transfers.
    pub current_transfer_in_flight_bytes: u64,
    /// Peak bytes retained by in-flight transfers.
    pub peak_transfer_in_flight_bytes: u64,
    /// Current disk bytes, including pending write reservations.
    pub current_disk_bytes: u64,
    /// Peak disk bytes successfully admitted.
    pub peak_disk_bytes: u64,
    /// Aggregate finite limits.
    pub limits: CachePoolLimits,
}

#[derive(Debug)]
struct CachePoolState {
    managers: BTreeMap<u64, CachePoolUsage>,
    reservations: BTreeMap<u64, CachePoolUsage>,
    current: CachePoolUsage,
    peak: CachePoolUsage,
}

/// Shareable aggregate ownership boundary for scheduler and standalone caches.
#[derive(Clone)]
pub struct CacheResidencyPool {
    id: u64,
    limits: CachePoolLimits,
    state: Arc<Mutex<CachePoolState>>,
}

impl std::fmt::Debug for CacheResidencyPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CacheResidencyPool")
            .field("id", &self.id)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl PartialEq for CacheResidencyPool {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

impl Eq for CacheResidencyPool {}

impl CacheResidencyPool {
    /// Creates an empty process pool under aggregate finite limits.
    pub fn new(limits: CachePoolLimits) -> Self {
        Self {
            id: NEXT_CACHE_POOL_ID.fetch_add(1, Ordering::Relaxed),
            limits,
            state: Arc::new(Mutex::new(CachePoolState {
                managers: BTreeMap::new(),
                reservations: BTreeMap::new(),
                current: CachePoolUsage::default(),
                peak: CachePoolUsage::default(),
            })),
        }
    }

    /// Returns this pool's stable process-local identity.
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns aggregate finite limits.
    pub const fn limits(&self) -> CachePoolLimits {
        self.limits
    }

    /// Registers a manager and returns its exact RAII membership token.
    pub fn register_manager(&self, manager: u64) -> Result<CachePoolMembership, CachePoolError> {
        let mut state = self.state.lock().map_err(|_| CachePoolError::Poisoned)?;
        if state.managers.contains_key(&manager) {
            return Err(CachePoolError::DuplicateManager { manager });
        }
        state.managers.insert(manager, CachePoolUsage::default());
        Ok(CachePoolMembership {
            manager,
            pool: self.clone(),
        })
    }

    /// Replaces the published occupancy owned by one registered manager.
    ///
    /// Backends use reservations to admit growth before allocating it, publish
    /// the resulting concrete occupancy here, and then release the reservation.
    /// Publication records physical truth even when an unavoidable allocation
    /// boundary temporarily exceeds a limit; subsequent admissions fail and
    /// the backend must synchronously rebalance or roll back that allocation.
    pub fn update_manager(
        &self,
        manager: u64,
        usage: CachePoolUsage,
    ) -> Result<CachePoolUsage, CachePoolError> {
        let mut state = self.state.lock().map_err(|_| CachePoolError::Poisoned)?;
        let old = *state
            .managers
            .get(&manager)
            .ok_or(CachePoolError::UnknownManager { manager })?;
        let current = state
            .current
            .checked_sub(old)
            .and_then(|current| current.checked_add(usage))
            .ok_or(CachePoolError::AccountingOverflow {
                operation: "manager occupancy publication",
            })?;
        state.managers.insert(manager, usage);
        state.current = current;
        update_peaks(&mut state, self.limits);
        Ok(state.current)
    }

    /// Atomically admits additional occupancy until its RAII token is dropped.
    pub fn reserve(&self, usage: CachePoolUsage) -> Result<CachePoolReservation, CachePoolError> {
        let mut state = self.state.lock().map_err(|_| CachePoolError::Poisoned)?;
        validate_additional(state.current, usage, self.limits)?;
        let required =
            state
                .current
                .checked_add(usage)
                .ok_or(CachePoolError::AccountingOverflow {
                    operation: "temporary admission",
                })?;
        let reservation = NEXT_CACHE_POOL_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        state.reservations.insert(reservation, usage);
        state.current = required;
        update_peaks(&mut state, self.limits);
        Ok(CachePoolReservation {
            reservation,
            pool: self.clone(),
        })
    }

    /// Atomically admits resources retained by one submitted transfer.
    pub fn reserve_transfer(&self, bytes: u64) -> Result<CachePoolReservation, CachePoolError> {
        self.reserve(CachePoolUsage {
            transfer_in_flight_bytes: bytes,
            ..CachePoolUsage::default()
        })
    }

    /// Returns aggregate current occupancy and high-water marks.
    pub fn report(&self) -> Result<CachePoolReport, CachePoolError> {
        let state = self.state.lock().map_err(|_| CachePoolError::Poisoned)?;
        Ok(CachePoolReport {
            pool_id: self.id,
            managers: state.managers.len(),
            current_device_bytes: state.current.device_bytes,
            peak_device_bytes: state.peak.device_bytes,
            current_host_bytes: state.current.host_bytes,
            peak_host_bytes: state.peak.host_bytes,
            current_transfer_in_flight_bytes: state.current.transfer_in_flight_bytes,
            peak_transfer_in_flight_bytes: state.peak.transfer_in_flight_bytes,
            current_disk_bytes: state.current.disk_bytes,
            peak_disk_bytes: state.peak.disk_bytes,
            limits: self.limits,
        })
    }

    fn remove_manager(&self, manager: u64) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(previous) = state.managers.get(&manager).copied() {
                if let Some(current) = state.current.checked_sub(previous) {
                    state.managers.remove(&manager);
                    state.current = current;
                }
            }
        }
    }
}

/// Exact ownership of a temporary aggregate cache admission.
#[derive(Debug)]
pub struct CachePoolReservation {
    reservation: u64,
    pool: CacheResidencyPool,
}

impl Drop for CachePoolReservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.pool.state.lock() {
            if let Some(usage) = state.reservations.get(&self.reservation).copied() {
                if let Some(current) = state.current.checked_sub(usage) {
                    state.reservations.remove(&self.reservation);
                    state.current = current;
                }
            }
        }
    }
}

/// Exact ownership of one manager's aggregate pool contribution.
#[derive(Debug)]
pub struct CachePoolMembership {
    manager: u64,
    pool: CacheResidencyPool,
}

impl CachePoolMembership {
    /// Pool to which this manager is registered.
    pub const fn pool(&self) -> &CacheResidencyPool {
        &self.pool
    }
}

impl Drop for CachePoolMembership {
    fn drop(&mut self) {
        self.pool.remove_manager(self.manager);
    }
}

fn validate_additional(
    current: CachePoolUsage,
    usage: CachePoolUsage,
    limits: CachePoolLimits,
) -> Result<(), CachePoolError> {
    let required = current
        .checked_add(usage)
        .ok_or(CachePoolError::AccountingOverflow {
            operation: "admission validation",
        })?;
    for (resource, additional, required, budget) in [
        (
            CachePoolResource::Device,
            usage.device_bytes,
            required.device_bytes,
            limits.device_bytes,
        ),
        (
            CachePoolResource::Host,
            usage.host_bytes,
            required.host_bytes,
            limits.host_bytes,
        ),
        (
            CachePoolResource::TransferInFlight,
            usage.transfer_in_flight_bytes,
            required.transfer_in_flight_bytes,
            limits.transfer_in_flight_bytes,
        ),
        (
            CachePoolResource::Disk,
            usage.disk_bytes,
            required.disk_bytes,
            limits.disk_bytes,
        ),
    ] {
        if additional != 0 && required > budget {
            return Err(CachePoolError::BudgetExceeded {
                resource,
                required,
                budget,
            });
        }
    }
    Ok(())
}

fn update_peaks(state: &mut CachePoolState, limits: CachePoolLimits) {
    if state.current.device_bytes <= limits.device_bytes {
        state.peak.device_bytes = state.peak.device_bytes.max(state.current.device_bytes);
    }
    if state.current.host_bytes <= limits.host_bytes {
        state.peak.host_bytes = state.peak.host_bytes.max(state.current.host_bytes);
    }
    if state.current.transfer_in_flight_bytes <= limits.transfer_in_flight_bytes {
        state.peak.transfer_in_flight_bytes = state
            .peak
            .transfer_in_flight_bytes
            .max(state.current.transfer_in_flight_bytes);
    }
    if state.current.disk_bytes <= limits.disk_bytes {
        state.peak.disk_bytes = state.peak.disk_bytes.max(state.current.disk_bytes);
    }
}

/// Backend-neutral aggregate cache ownership failure.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CachePoolError {
    /// Aggregate limits had no device or transfer capacity.
    #[error("invalid cache pool limits: {0}")]
    InvalidLimits(&'static str),
    /// A manager identity was registered more than once.
    #[error("cache pool manager {manager} is already registered")]
    DuplicateManager {
        /// Duplicate manager identity.
        manager: u64,
    },
    /// Occupancy was published for an unregistered manager.
    #[error("cache pool manager {manager} is not registered")]
    UnknownManager {
        /// Missing manager identity.
        manager: u64,
    },
    /// Aggregate cache accounting exceeded one finite resource limit.
    #[error(
        "cache pool {resource:?} budget exceeded: required {required} bytes, budget {budget} bytes"
    )]
    BudgetExceeded {
        /// Exhausted resource axis.
        resource: CachePoolResource,
        /// Aggregate bytes required.
        required: u64,
        /// Finite pool budget.
        budget: u64,
    },
    /// Checked ownership accounting overflowed.
    #[error("cache pool accounting overflow during {operation}")]
    AccountingOverflow {
        /// Stable accounting transition.
        operation: &'static str,
    },
    /// Shared ownership state was poisoned by a panic.
    #[error("cache residency pool state is poisoned")]
    Poisoned,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Barrier};

    fn pool() -> CacheResidencyPool {
        CacheResidencyPool::new(CachePoolLimits::new(16, 12, 10, 8).unwrap())
    }

    #[test]
    fn manager_membership_owns_published_occupancy() {
        let pool = pool();
        let membership = pool.register_manager(7).unwrap();
        pool.update_manager(
            7,
            CachePoolUsage {
                device_bytes: 8,
                host_bytes: 4,
                ..CachePoolUsage::default()
            },
        )
        .unwrap();
        let report = pool.report().unwrap();
        assert_eq!(report.managers, 1);
        assert_eq!(report.current_device_bytes, 8);
        assert_eq!(report.current_host_bytes, 4);

        drop(membership);
        let report = pool.report().unwrap();
        assert_eq!(report.managers, 0);
        assert_eq!(report.current_device_bytes, 0);
        assert_eq!(report.current_host_bytes, 0);
    }

    #[test]
    fn reservation_is_atomic_and_released_by_exact_owner() {
        let pool = pool();
        let first = pool.reserve_transfer(6).unwrap();
        assert_eq!(pool.report().unwrap().current_transfer_in_flight_bytes, 6);
        assert_eq!(
            pool.reserve_transfer(5).unwrap_err(),
            CachePoolError::BudgetExceeded {
                resource: CachePoolResource::TransferInFlight,
                required: 11,
                budget: 10,
            }
        );
        assert_eq!(pool.report().unwrap().current_transfer_in_flight_bytes, 6);
        drop(first);
        assert_eq!(pool.report().unwrap().current_transfer_in_flight_bytes, 0);
    }

    #[test]
    fn independent_resource_axes_fail_closed() {
        let pool = pool();
        for (usage, resource, required, budget) in [
            (
                CachePoolUsage {
                    device_bytes: 17,
                    ..CachePoolUsage::default()
                },
                CachePoolResource::Device,
                17,
                16,
            ),
            (
                CachePoolUsage {
                    host_bytes: 13,
                    ..CachePoolUsage::default()
                },
                CachePoolResource::Host,
                13,
                12,
            ),
            (
                CachePoolUsage {
                    disk_bytes: 9,
                    ..CachePoolUsage::default()
                },
                CachePoolResource::Disk,
                9,
                8,
            ),
        ] {
            assert_eq!(
                pool.reserve(usage).unwrap_err(),
                CachePoolError::BudgetExceeded {
                    resource,
                    required,
                    budget,
                }
            );
        }
        assert_eq!(pool.report().unwrap().current_device_bytes, 0);
        assert_eq!(pool.report().unwrap().current_host_bytes, 0);
        assert_eq!(pool.report().unwrap().current_disk_bytes, 0);
    }

    #[test]
    fn reports_and_limits_round_trip_without_backend_types() {
        let pool = pool();
        let _membership = pool.register_manager(1).unwrap();
        pool.update_manager(
            1,
            CachePoolUsage {
                device_bytes: 8,
                disk_bytes: 4,
                ..CachePoolUsage::default()
            },
        )
        .unwrap();
        let report = pool.report().unwrap();
        let encoded = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<CachePoolReport>(&encoded).unwrap(),
            report
        );
    }

    #[test]
    fn concurrent_admission_has_one_atomic_winner() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(64, 10, 64, 0).unwrap());
        let start = Arc::new(Barrier::new(3));
        let finish = Arc::new(Barrier::new(3));
        let (sender, receiver) = mpsc::channel();
        let handles = (0..2)
            .map(|_| {
                let pool = pool.clone();
                let start = Arc::clone(&start);
                let finish = Arc::clone(&finish);
                let sender = sender.clone();
                std::thread::spawn(move || {
                    start.wait();
                    let admission = pool.reserve(CachePoolUsage {
                        host_bytes: 6,
                        ..CachePoolUsage::default()
                    });
                    sender.send(admission.is_ok()).unwrap();
                    finish.wait();
                    drop(admission);
                })
            })
            .collect::<Vec<_>>();
        drop(sender);
        start.wait();
        let admitted = [receiver.recv().unwrap(), receiver.recv().unwrap()];
        assert_eq!(admitted.into_iter().filter(|value| *value).count(), 1);
        assert_eq!(pool.report().unwrap().current_host_bytes, 6);
        finish.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(pool.report().unwrap().current_host_bytes, 0);
    }

    #[test]
    fn manager_identity_and_accounting_fail_closed() {
        let pool = pool();
        let _membership = pool.register_manager(9).unwrap();
        assert_eq!(
            pool.register_manager(9).unwrap_err(),
            CachePoolError::DuplicateManager { manager: 9 }
        );
        assert_eq!(
            pool.update_manager(10, CachePoolUsage::default())
                .unwrap_err(),
            CachePoolError::UnknownManager { manager: 10 }
        );
        let overflow = CacheResidencyPool::new(CachePoolLimits::new(u64::MAX, 1, 1, 0).unwrap());
        let _reservation = overflow
            .reserve(CachePoolUsage {
                device_bytes: u64::MAX,
                ..CachePoolUsage::default()
            })
            .unwrap();
        assert!(matches!(
            overflow.reserve(CachePoolUsage {
                device_bytes: 1,
                ..CachePoolUsage::default()
            }),
            Err(CachePoolError::AccountingOverflow { .. })
        ));
    }
}
pub use executor::{
    CacheIoAdmission, CacheIoCompletionDisposition, CacheIoExecutionState,
    CacheIoExecutionStateError, CacheIoPreparation, CacheIoStartDisposition,
};
