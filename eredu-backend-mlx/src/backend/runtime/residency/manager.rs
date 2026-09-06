//! MLX materialization and transfer execution for immutable weight units.
//!
//! A [`crate::backend::runtime::residency::manager::ResidencyManager`] moves caller-defined groups of
//! checkpoint selections from an [`eredu_checkpoint::store::CheckpointSource`] into
//! immutable typed host-transfer buffers or execution-stream arrays. The
//! manager accounts for logical host and device copies independently, even on
//! unified-memory systems.
//! Missing units can be reserved and submitted as one batch. Caller-owned
//! [`crate::backend::runtime::residency::manager::ResidentTransfer`] values retain
//! source leases until MLX reports exact completion of the submitted
//! transfer.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Instant,
};

use safemlx::{
    host_transfer_capacity_upper_bound, transforms::async_eval_with_event, Array, DeviceType,
    Event, HostTransferBuffer, HostTransferPolicy, ImmutableHostTransferBuffer, Stream,
};

use crate::{
    backend::nn::shared::MlxNeuralBackend,
    backend::residency::sample_allocator_memory,
    backend::runtime::checkpoint::recipe::{MlxWeightRecipeExt, WeightRecipeError},
    backend::runtime::checkpoint::store::{
        CheckpointMaterializationError, MlxParameterMaterializationContext,
        PendingWeightMaterialization, WeightMaterialization,
    },
};
use eredu_core::residency::{
    EvictedResidencyCopy, MemoryTier, OffloadPlan, OffloadReport, OffloadUnitId, PrefetchOutcome,
    ResidencyLedgerError, TransferDirection, UnitResidencyReport,
};

use eredu_runtime::residency::{
    OffloadUnit, ResidencyController, ResidencyControllerError, ResidencyLease,
    ResidencyLeaseOwner, ResidencyLeaseStorage, ResidencyWindowError, ResidencyWindowManager,
    WeightBinding,
};
use eredu_runtime::ResidencyReport;

/// A resident unit that prevents eviction of one tier until it is dropped.
pub type ResidentUnitLease = ResidencyLease<ResidentLeaseStorage, ManagerInner>;

/// Host or device storage retained by a weight-residency lease.
pub enum ResidentLeaseStorage {
    /// Immutable host-transfer buffers.
    Host(Arc<ResidentHostBuffers>),
    /// Materialized device arrays.
    Device(Arc<ResidentArrays>),
}

impl ResidencyLeaseStorage for ResidentLeaseStorage {
    type DeviceValue = Array;
    type HostValue = ImmutableHostTransferBuffer;
    type Error = ResidencyError;
    type BindingNames<'a> = Box<dyn Iterator<Item = &'a str> + 'a>;

    fn device_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::DeviceValue, Self::Error> {
        match self {
            ResidentLeaseStorage::Device(arrays) => {
                arrays
                    .arrays
                    .get(name)
                    .ok_or_else(|| ResidencyError::UnknownBinding {
                        id: id.clone(),
                        name: name.to_string(),
                    })
            }
            ResidentLeaseStorage::Host(_) => Err(ResidencyError::HostBindingIsNotArray {
                id: id.clone(),
                name: name.to_string(),
            }),
        }
    }

    fn host_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::HostValue, Self::Error> {
        match self {
            ResidentLeaseStorage::Host(buffers) => {
                buffers.buffers.get(name).map(Arc::as_ref).ok_or_else(|| {
                    ResidencyError::UnknownBinding {
                        id: id.clone(),
                        name: name.to_string(),
                    }
                })
            }
            ResidentLeaseStorage::Device(_) => Err(ResidencyError::DeviceBindingIsNotHostBuffer {
                id: id.clone(),
                name: name.to_string(),
            }),
        }
    }

    fn binding_names(&self) -> Self::BindingNames<'_> {
        match self {
            ResidentLeaseStorage::Host(buffers) => {
                Box::new(buffers.buffers.keys().map(String::as_str))
            }
            ResidentLeaseStorage::Device(arrays) => {
                Box::new(arrays.arrays.keys().map(String::as_str))
            }
        }
    }
}

/// Structured failures from residency validation and state transitions.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// A complete owner binding set is unsupported by the MLX parameter backend.
    #[error("MLX residency binding preflight failed: {0}")]
    BindingPreflight(String),
    /// A backend-neutral binding or offload-unit declaration was invalid.
    #[error(transparent)]
    Declaration(#[from] eredu_runtime::residency::ResidencyDeclarationError),
    /// Backend-neutral ownership or capacity transition failed.
    #[error(transparent)]
    Ledger(#[from] ResidencyLedgerError),
    /// Backend-neutral plan and declaration validation failed.
    #[error(transparent)]
    Controller(#[from] ResidencyControllerError),
    /// Backend-neutral binding selection rewrite failed.
    #[error(transparent)]
    BindingSelection(#[from] eredu_runtime::residency::WeightBindingSelectionError),
    /// Backend-neutral ordered-window validation or accounting failed.
    #[error(transparent)]
    Window(#[from] ResidencyWindowError),
    /// Binding sizes did not sum to the plan's unit size.
    #[error(
        "residency unit {id} defines {actual_bytes} bytes but its plan reserves {planned_bytes}"
    )]
    UnitByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Bytes reserved by the plan.
        planned_bytes: u64,
        /// Sum of binding sizes.
        actual_bytes: u64,
    },
    /// A binding's selected checkpoint size contradicted its definition.
    #[error("binding {binding:?} in unit {id} selects {actual_bytes} bytes but declares {expected_bytes}")]
    BindingByteMismatch {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Binding name.
        binding: String,
        /// Declared size.
        expected_bytes: u64,
        /// Store-validated size.
        actual_bytes: u64,
    },
    /// A derived-weight recipe was invalid or could not be materialized.
    #[error("derived-weight recipe for binding {binding:?} failed: {source}")]
    Recipe {
        /// Local binding name.
        binding: String,
        /// Recipe failure.
        #[source]
        source: WeightRecipeError,
    },
    /// The configured source stream was not a CPU stream.
    #[error("the residency source stream must target the CPU")]
    InvalidSourceStream,
    /// A binding lookup failed on a valid resident unit.
    #[error("residency unit {id} has no binding named {name:?}")]
    UnknownBinding {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Unknown local name.
        name: String,
    },
    /// A caller requested an executable array from typed host storage.
    #[error(
        "host-resident binding {name:?} in unit {id} is a typed transfer buffer, not an MLX array"
    )]
    HostBindingIsNotArray {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Requested binding name.
        name: String,
    },
    /// A caller requested typed host storage from a device-resident copy.
    #[error(
        "device-resident binding {name:?} in unit {id} is an MLX array, not a host-transfer buffer"
    )]
    DeviceBindingIsNotHostBuffer {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Requested binding name.
        name: String,
    },
    /// A backend allocated beyond its advertised pre-allocation capacity bound.
    #[error(
        "host-transfer allocation for residency unit {id} used {actual_bytes} bytes, exceeding reserved upper bound {reserved_bytes}"
    )]
    HostCapacityBoundExceeded {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Capacity reserved before materialization.
        reserved_bytes: u64,
        /// Exact allocated capacity.
        actual_bytes: u64,
    },
    /// Checked byte or recency arithmetic overflowed.
    #[error("residency arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        context: &'static str,
    },
    /// Backend-neutral checkpoint inspection or lease acquisition failed.
    #[error(transparent)]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),
    /// MLX checkpoint materialization failed.
    #[error(transparent)]
    CheckpointMaterialization(#[from] CheckpointMaterializationError),
    /// An MLX copy or evaluation failed.
    #[error("MLX {operation} failed for residency unit {id}: {source}")]
    Mlx {
        /// Unit identifier.
        id: OffloadUnitId,
        /// Failed operation.
        operation: &'static str,
        /// MLX exception.
        #[source]
        source: safemlx::error::Exception,
    },
    /// Serialized manager state was poisoned by a prior panic.
    #[error("residency manager state is poisoned")]
    StatePoisoned,
}

/// Serialized, shareable manager for immutable checkpoint weight residency.
#[derive(Clone)]
pub struct ResidencyManager {
    inner: Arc<ManagerInner>,
}

fn preflight_residency_owner_bindings(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    control: &ResidencyController,
) -> Result<(), ResidencyError> {
    for unit in control.units() {
        eredu_runtime::preflight_bindings::<MlxNeuralBackend>(store, unit.bindings())
            .map_err(|error| ResidencyError::BindingPreflight(error.to_string()))?;
    }
    Ok(())
}

impl ResidencyManager {
    /// Returns the MLX stream index used for device residency transfers.
    pub fn device_stream_index(&self) -> Result<i32, ResidencyError> {
        self.lock()?
            .device_stream
            .get_index()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "device residency stream index",
                source,
            })
    }

    /// Validates plan/unit identity, binding sizes, selections, and streams.
    ///
    /// Construction does not create MLX arrays. Call [`Self::initialize`] to
    /// materialize units assigned to host or device by the plan.
    pub fn new<S>(
        store: Arc<S>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError>
    where
        S: eredu_checkpoint::store::CheckpointSource + 'static,
    {
        let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = store;
        Self::new_shared(store, plan, units, source_stream, device_stream)
    }

    /// Creates a manager from an already type-erased checkpoint store.
    pub fn new_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        plan: OffloadPlan,
        units: impl IntoIterator<Item = OffloadUnit>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, ResidencyError> {
        let units = units.into_iter().collect::<Vec<_>>();
        let control = ResidencyController::new(store.as_ref(), plan, units)?;
        preflight_residency_owner_bindings(store.as_ref(), &control)?;

        let source_device = source_stream
            .get_device()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source stream inspection",
                source,
            })?;
        if source_device
            .get_type()
            .map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "source device inspection",
                source,
            })?
            != DeviceType::Cpu
        {
            return Err(ResidencyError::InvalidSourceStream);
        }

        let storage = control
            .units()
            .map(|unit| (unit.id().clone(), UnitStorage::default()))
            .collect();
        let failed_transfer = Arc::new(std::sync::atomic::AtomicBool::new(false));
        Ok(Self {
            inner: Arc::new(ManagerInner {
                store,
                failed_transfer: Arc::clone(&failed_transfer),
                state: Mutex::new(ManagerState {
                    failed_transfer,
                    control,
                    storage,
                    alias_owner_pins: BTreeSet::new(),
                    materialization: MlxParameterMaterializationContext::new(
                        &source_stream,
                        &device_stream,
                    ),
                    source_stream,
                    device_stream,
                }),
                changed: Condvar::new(),
            }),
        })
    }

    /// Materializes all planned host and device units in identifier order.
    ///
    /// Disk units remain array-free. A failure never publishes a partial unit;
    /// units completed earlier remain resident and fully accounted, allowing a
    /// caller to inspect the report and retry initialization.
    pub fn initialize(&self) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if state.control.ledger_mut().initialized() {
            return Ok(());
        }
        let assignments = state
            .control
            .ledger_mut()
            .plan()
            .units()
            .iter()
            .map(|unit| (unit.id().clone(), unit.tier()))
            .collect::<Vec<_>>();
        for (id, tier) in assignments {
            if tier != MemoryTier::Disk {
                ensure_resident(&mut state, self.inner.store.as_ref(), &id, tier, true)?;
            }
        }
        state.control.ledger_mut().mark_initialized();
        Ok(())
    }

    /// Synchronously prepares one host or device copy and records hit/miss telemetry.
    ///
    /// This provides caller-directed lookahead but does not overlap transfer
    /// with computation.
    pub fn prefetch(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<PrefetchOutcome, ResidencyError> {
        validate_target(tier, "prefetch")?;
        let mut state = self.lock()?;
        loop {
            state.control.ledger_mut().require_initialized()?;
            let copy = state.control.ledger_mut().copy_status(id, tier)?;
            if !copy.is_some_and(|copy| copy.in_flight().is_some()) {
                break;
            }
            state = self.wait_for_transfer(state)?;
        }
        prefetch_locked(&mut state, self.inner.store.as_ref(), id, tier)
    }

    /// Ensures residency and returns an RAII lease protecting the requested copy.
    pub fn acquire(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_with_demand(id, tier, 1)
    }

    /// Ensures residency and records weighted demand for eviction policy.
    ///
    /// `demand` may be larger than one when duplicate entry requests
    /// share a single acquisition. Frequency counters saturate on overflow.
    pub fn acquire_with_demand(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
        demand: u64,
    ) -> Result<ResidentUnitLease, ResidencyError> {
        self.acquire_many_with_demand(&[(id.clone(), demand)], tier)?
            .pop()
            .ok_or(ResidencyError::StatePoisoned)
    }

    /// Acquires a deterministic entry set with one batched residency transition.
    ///
    /// Missing copies reserve capacity before any materialization starts. All
    /// requested units are protected from eviction, all lazy outputs are
    /// evaluated together, and leases are published only after the batch is
    /// complete.
    pub fn acquire_many_with_demand(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<ResidentUnitLease>, ResidencyError> {
        self.acquire_many_with_mode(requests, tier, false)
            .map(|(leases, _)| leases)
    }

    /// Submits one residency batch and returns its owning completion lease.
    ///
    /// Missing copies are submitted to MLX for asynchronous evaluation, but
    /// this method does not block the host for their completion. Call
    /// [`ResidentTransfer::order_after`] before evaluating work on another
    /// compatible stream. The transfer guard owns every source dependency
    /// until it is synchronized or dropped.
    pub fn acquire_many_with_transfer(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<ResidentTransfer, ResidencyError> {
        let (leases, submitted) = self.acquire_many_with_mode(requests, tier, true)?;
        let transfer = match submitted {
            None => ResidentTransfer::immediate(leases, tier),
            Some(submitted) => ResidentTransfer::submitted(leases, submitted),
        };
        Ok(transfer)
    }

    fn acquire_many_with_mode(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
        return_transfer: bool,
    ) -> Result<(Vec<ResidentUnitLease>, Option<SubmittedResidentTransfer>), ResidencyError> {
        validate_target(tier, "acquire")?;
        let mut state = self.lock()?;
        let ids = requests
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        state.control.ledger_mut().validate_batch(&ids, tier)?;
        loop {
            state.control.ledger_mut().require_initialized()?;
            for (id, _) in requests {
                state.control.ledger_mut().spec(id)?;
            }
            let waiting = requests.iter().any(|(id, _)| {
                state
                    .control
                    .ledger_mut()
                    .copy_status(id, tier)
                    .ok()
                    .flatten()
                    .is_some_and(|copy| copy.in_flight().is_some())
            });
            if !waiting {
                break;
            }
            state = self.wait_for_transfer(state)?;
        }
        let missing = requests
            .iter()
            .filter(|(id, _)| {
                !state
                    .control
                    .ledger_mut()
                    .is_resident(id, tier)
                    .unwrap_or(false)
            })
            .count();
        let started = Instant::now();
        let residency = ensure_many_resident(
            &mut state,
            self.inner.store.as_ref(),
            &ids,
            tier,
            return_transfer,
            false,
        );
        if missing > 0 {
            state
                .control
                .ledger_mut()
                .record_prefetch_stall(started.elapsed());
        }
        let (_, submitted) = residency?;
        if let Some(submitted) = &submitted {
            submitted.attach_owner(Arc::downgrade(&self.inner));
        }
        let mut leases = crate::backend::ordinary_retirement::OrdinaryRetirement::new(Vec::new());
        let acquired = (|| -> Result<(), ResidencyError> {
            for (id, demand) in requests {
                let unit = state.storage.get(id).ok_or(ResidencyError::StatePoisoned)?;
                let storage = match tier {
                    MemoryTier::Host => ResidentLeaseStorage::Host(Arc::clone(
                        unit.host.as_ref().ok_or(ResidencyError::StatePoisoned)?,
                    )),
                    MemoryTier::Device => ResidentLeaseStorage::Device(Arc::clone(
                        unit.device.as_ref().ok_or(ResidencyError::StatePoisoned)?,
                    )),
                    MemoryTier::Disk => unreachable!("validated above"),
                };
                state.control.ledger_mut().pin(id, tier, *demand)?;
                leases.push(ResidentUnitLease::new(
                    id.clone(),
                    tier,
                    storage,
                    Arc::downgrade(&self.inner),
                ));
            }
            Ok(())
        })();
        if let Err(error) = acquired {
            if let Some(submitted) = &submitted {
                submitted.retain_partial_leases(leases.into_inner());
            }
            return Err(error);
        }
        Ok((leases.into_inner(), submitted))
    }

    /// Returns whether a logical copy currently resides in a memory tier.
    pub fn is_resident(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<bool, ResidencyError> {
        validate_target(tier, "is_resident")?;
        let state = self.lock()?;
        Ok(state.control.ledger().is_resident(id, tier)?)
    }

    /// Replaces the protected window and synchronously prepares bounded lookahead.
    ///
    /// `active` units are protected from automatic eviction. At most the first
    /// configured number of distinct `upcoming` units are prefetched, in caller
    /// order. Repeated and overlapping windows are deterministic.
    pub fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        self.prepare_group_window("default", active, upcoming, tier)
    }

    /// Replaces one named group's protected window and prepares bounded lookahead.
    ///
    /// Protection owned by other groups remains active. This permits independent
    /// text, vision, audio, temporal, and depth stack scheduling.
    pub fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, ResidencyError> {
        validate_target(tier, "prepare_group_window")?;
        let mut state = self.lock()?;
        loop {
            state.control.ledger_mut().require_initialized()?;
            for id in active.iter().chain(upcoming) {
                state.control.ledger_mut().spec(id)?;
            }
            let waiting = active.iter().chain(upcoming).any(|id| {
                state
                    .control
                    .ledger_mut()
                    .copy_status(id, tier)
                    .ok()
                    .flatten()
                    .is_some_and(|copy| copy.in_flight().is_some())
            });
            if !waiting {
                break;
            }
            state = self.wait_for_transfer(state)?;
        }
        let selected = state
            .control
            .commit_group_window(group, active, upcoming, tier)?;
        selected
            .into_iter()
            .map(|id| {
                prefetch_locked(&mut state, self.inner.store.as_ref(), &id, tier)
                    .map(|outcome| (id, outcome))
            })
            .collect()
    }

    /// Replaces one named protected window without materializing its units.
    ///
    /// This is used by schedulers that submit materialization through a
    /// separate bounded service. Protection owned by other named windows is
    /// preserved.
    pub fn protect_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        validate_target(tier, "protect_group_window")?;
        state.control.protect_group_window(group, active, tier)?;
        Ok(())
    }

    /// Explicitly evicts one host or device copy.
    ///
    /// Evicting an absent copy is an idempotent success returning `false`.
    pub fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, ResidencyError> {
        validate_target(tier, "evict")?;
        let mut state = self.lock()?;
        let Some(evicted) = state.control.ledger_mut().evict(id, tier)? else {
            return Ok(false);
        };
        if !state
            .storage
            .get_mut(id)
            .is_some_and(|unit| unit.remove_storage(tier))
        {
            return Err(ResidencyError::StatePoisoned);
        }
        debug_assert_eq!(evicted.id, *id);
        Ok(true)
    }

    /// Samples optional MLX allocator and process metrics on explicit request.
    pub fn sample_memory(
        &self,
        include_mlx: bool,
        include_process: bool,
    ) -> Result<(), ResidencyError> {
        let mut state = self.lock()?;
        if include_mlx {
            let metrics = sample_allocator_memory().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "allocator memory sampling",
                source,
            })?;
            state.control.ledger_mut().record_allocator_memory(metrics);
        }
        if include_process {
            state.control.ledger_mut().sample_process_metrics();
        }
        Ok(())
    }

    /// Returns an immutable point-in-time residency and storage report.
    pub fn report(&self) -> Result<ResidencyReport, ResidencyError> {
        let (initialized, offload, units, active_window) = self.telemetry_snapshot()?;
        Ok(ResidencyReport::new(
            initialized,
            offload,
            units,
            active_window,
            self.inner.store.source_diagnostics()?,
        ))
    }

    /// Returns initialized state, aggregate telemetry, unit reports, and active window.
    pub fn telemetry_snapshot(
        &self,
    ) -> Result<
        (
            bool,
            OffloadReport,
            Vec<UnitResidencyReport>,
            Vec<OffloadUnitId>,
        ),
        ResidencyError,
    > {
        let state = self.lock()?;
        let active = state.control.ledger().active_window();
        let units = state.control.ledger().unit_reports();
        Ok((
            state.control.ledger().initialized(),
            state.control.ledger().telemetry(),
            units,
            active.into_iter().collect(),
        ))
    }

    fn lock(&self) -> Result<MutexGuard<'_, ManagerState>, ResidencyError> {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        if self
            .inner
            .failed_transfer
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(ResidencyError::Mlx {
                id: internal_id(),
                operation: "resident transfer admission",
                source: safemlx::error::Exception::custom(
                    "residency manager is poisoned by a failed or unobservable native transfer",
                ),
            });
        }
        self.inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)
    }

    fn wait_for_transfer<'a>(
        &'a self,
        state: MutexGuard<'a, ManagerState>,
    ) -> Result<MutexGuard<'a, ManagerState>, ResidencyError> {
        let (state, _) = self
            .inner
            .changed
            .wait_timeout(state, std::time::Duration::from_millis(25))
            .map_err(|_| ResidencyError::StatePoisoned)?;
        drop(state);
        // Recovery may stage manager-lock owners; never reclaim under this mutex.
        self.lock()
    }
}

impl ResidencyWindowManager for ResidencyManager {
    type Error = ResidencyError;

    fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
        ResidencyManager::prepare_window(self, active, upcoming, tier)
    }

    fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
        ResidencyManager::prepare_group_window(self, group, active, upcoming, tier)
    }

    fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, Self::Error> {
        ResidencyManager::evict(self, id, tier)
    }

    fn unit_reports(&self) -> Result<Vec<UnitResidencyReport>, Self::Error> {
        self.telemetry_snapshot().map(|(_, _, units, _)| units)
    }
}

mod transfer;
use transfer::*;
pub use transfer::{
    ManagerInner, ResidentArrays, ResidentHostBuffers, ResidentTransfer, ResidentTransferResources,
};

mod materialization;
pub use materialization::host_capacity_upper_bound_for_bindings;
use materialization::*;

#[cfg(test)]
#[path = "manager/tests.rs"]
mod tests;
