//! MLX materialization and transfer execution for immutable weight units.
//!
//! A [`crate::backend::runtime::residency::manager::ResidencyManager`] moves caller-defined groups of
//! checkpoint selections from an [`eredu_checkpoint::store::CheckpointSource`] into
//! immutable typed host-transfer buffers or execution-stream arrays. The
//! manager accounts for logical host and device copies independently, even on
//! unified-memory systems.
//! Missing units can be reserved and submitted as one batch. Caller-owned
//! [`crate::backend::runtime::residency::manager::ResidentTransfer`] values retain
//! source mappings until MLX reports exact completion of the submitted
//! transfer.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Instant,
};

use safemlx::{
    host_transfer_capacity_upper_bound,
    transforms::{async_eval_with_event, eval},
    Array, DeviceType, Event, HostTransferBuffer, HostTransferPolicy, ImmutableHostTransferBuffer,
    Stream,
};

use crate::{
    backend::residency::sample_allocator_memory,
    backend::runtime::checkpoint::recipe::{MlxWeightRecipeExt, WeightRecipeError},
    backend::runtime::checkpoint::store::{
        MlxParameterMaterializationContext, PendingWeightMaterialization, WeightStoreError,
    },
};
use eredu_core::residency::{
    EvictedResidencyCopy, MemoryTier, OffloadPlan, OffloadReport, OffloadUnitId, PrefetchOutcome,
    ResidencyLedgerError, TransferDirection, UnitResidencyReport,
};

use eredu_runtime::residency::{
    OffloadUnit, ResidencyController, ResidencyControllerError, ResidencyLease,
    ResidencyLeaseOwner, ResidencyLeaseStorage, ResidencyTransfer, ResidencyTransferOwner,
    ResidencyWindowError, ResidencyWindowManager, WeightBinding,
};
use eredu_runtime::ResidencyReport;

/// A resident unit that prevents eviction of one tier until it is dropped.
pub type ResidentUnitLease = ResidencyLease<ResidentLeaseStorage, ManagerInner>;

pub enum ResidentLeaseStorage {
    Host(Arc<ResidentHostBuffers>),
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

/// Caller-owned completion and source-lifetime guard for one residency batch.
pub type ResidentTransfer =
    ResidencyTransfer<ResidentUnitLease, Event, ResidentTransferResources, ManagerInner>;

/// Structured failures from residency validation and state transitions.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
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
    /// Persistent store validation or materialization failed.
    #[error(transparent)]
    WeightStore(#[from] WeightStoreError),
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

        let units = units.into_iter().collect::<Vec<_>>();
        let control = ResidencyController::new(store.as_ref(), plan, units)?;
        for unit in control.units() {
            for binding in unit.bindings() {
                if binding.is_alias() {
                    continue;
                }
                binding
                    .source_recipe()
                    .preflight_bounded(store.as_ref())
                    .map_err(|source| ResidencyError::Recipe {
                        binding: binding.name().to_owned(),
                        source: source.into(),
                    })?;
            }
        }
        let storage = control
            .units()
            .map(|unit| (unit.id().clone(), UnitStorage::default()))
            .collect();
        Ok(Self {
            inner: Arc::new(ManagerInner {
                store,
                state: Mutex::new(ManagerState {
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
            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| ResidencyError::StatePoisoned)?;
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

    /// Ensures residency and records route-weighted demand for eviction policy.
    ///
    /// `demand` may be larger than one when duplicate routed-expert requests
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

    /// Acquires a deterministic expert set with one batched residency transition.
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
            Some(submitted) => ResidentTransfer::submitted(
                leases,
                submitted.event,
                submitted.retained,
                Arc::downgrade(&self.inner),
                submitted.ids,
                tier,
                submitted.generation,
            ),
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
            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| ResidencyError::StatePoisoned)?;
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
        let leases = requests
            .iter()
            .map(|(id, demand)| {
                state.control.ledger_mut().pin(id, tier, *demand)?;
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
                Ok(ResidentUnitLease::new(
                    id.clone(),
                    tier,
                    storage,
                    Arc::downgrade(&self.inner),
                ))
            })
            .collect::<Result<Vec<_>, ResidencyError>>()?;
        Ok((leases, submitted))
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
            state = self
                .inner
                .changed
                .wait(state)
                .map_err(|_| ResidencyError::StatePoisoned)?;
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
            self.inner
                .store
                .source_diagnostics()
                .map_err(crate::backend::runtime::checkpoint::store::neutral_store_error)?,
        ))
    }

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
        self.inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)
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

pub struct ManagerInner {
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    state: Mutex<ManagerState>,
    changed: Condvar,
}

impl ResidencyLeaseOwner for ManagerInner {
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier) {
        if let Ok(mut state) = self.state.lock() {
            state.control.ledger_mut().unpin(id, tier);
        }
    }
}

impl ResidencyTransferOwner<Event, ResidentTransferResources> for ManagerInner {
    type Executor = Stream;
    type Error = ResidencyError;

    fn order_after(
        completion: &Event,
        executor: &Stream,
        id: &OffloadUnitId,
    ) -> Result<(), Self::Error> {
        completion
            .wait_on(executor)
            .map_err(|source| ResidencyError::Mlx {
                id: id.clone(),
                operation: "resident transfer stream wait",
                source,
            })
    }

    fn is_complete(completion: &Event, id: &OffloadUnitId) -> Result<bool, Self::Error> {
        completion
            .is_complete()
            .map_err(|source| ResidencyError::Mlx {
                id: id.clone(),
                operation: "resident transfer query",
                source,
            })
    }

    fn wait(completion: &Event, id: &OffloadUnitId) -> Result<(), Self::Error> {
        completion
            .synchronize()
            .map_err(|source| ResidencyError::Mlx {
                id: id.clone(),
                operation: "resident transfer completion",
                source,
            })
    }

    fn finish_resources(resources: ResidentTransferResources, succeeded: bool) {
        if succeeded {
            for source in resources.sources {
                source.complete();
            }
            drop(resources.retained_arrays);
            drop(resources.retained_host);
            drop(resources.retained_events);
        } else {
            // Preserve PendingWeightMaterialization's conservative failure
            // cleanup: it synchronizes the involved streams and retains a
            // mapping permanently if backend state is unknowable.
            drop(resources);
        }
    }

    fn resolve_transfer(
        &self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
        generation: u64,
        succeeded: bool,
    ) -> Result<(), Self::Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        let removed = state
            .control
            .resolve_transfer(ids, tier, generation, succeeded)?;
        release_backend_copies(&mut state, &removed)?;
        self.changed.notify_all();
        Ok(())
    }
}

struct ManagerState {
    control: ResidencyController,
    storage: BTreeMap<OffloadUnitId, UnitStorage>,
    alias_owner_pins: BTreeSet<(OffloadUnitId, MemoryTier)>,
    materialization: MlxParameterMaterializationContext,
    source_stream: Stream,
    device_stream: Stream,
}

// SAFETY: every access to the MLX stream handles and resident arrays in this
// state is serialized by `ManagerInner::state`. No stream reference escapes
// the lock, and MLX operations use safemlx's runtime guard internally.
unsafe impl Send for ManagerState {}

#[derive(Default)]
struct UnitStorage {
    host: Option<Arc<ResidentHostBuffers>>,
    device: Option<Arc<ResidentArrays>>,
}

impl UnitStorage {
    fn remove_storage(&mut self, tier: MemoryTier) -> bool {
        match tier {
            MemoryTier::Host => self.host.take().is_some(),
            MemoryTier::Device => self.device.take().is_some(),
            MemoryTier::Disk => false,
        }
    }
}

fn release_backend_copies(
    state: &mut ManagerState,
    copies: &[EvictedResidencyCopy],
) -> Result<(), ResidencyError> {
    for copy in copies {
        let removed = state
            .storage
            .get_mut(&copy.id)
            .is_some_and(|unit| unit.remove_storage(copy.tier));
        if !removed {
            return Err(ResidencyError::StatePoisoned);
        }
    }
    Ok(())
}

pub struct ResidentArrays {
    arrays: BTreeMap<String, Array>,
}

pub struct ResidentHostBuffers {
    buffers: BTreeMap<String, Arc<ImmutableHostTransferBuffer>>,
}

pub struct ResidentTransferResources {
    sources: Vec<PendingWeightMaterialization>,
    retained_arrays: Vec<Array>,
    retained_host: Vec<Arc<ResidentHostBuffers>>,
    retained_events: Vec<Event>,
}

struct SubmittedResidentTransfer {
    event: Event,
    retained: ResidentTransferResources,
    ids: Vec<OffloadUnitId>,
    generation: u64,
}

fn validate_target(tier: MemoryTier, operation: &'static str) -> Result<(), ResidencyError> {
    if tier == MemoryTier::Disk {
        Err(ResidencyLedgerError::InvalidTargetTier { operation }.into())
    } else {
        Ok(())
    }
}

fn internal_id() -> OffloadUnitId {
    OffloadUnitId::new("residency-manager").expect("static identifier is valid")
}

fn prefetch_locked(
    state: &mut ManagerState,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<PrefetchOutcome, ResidencyError> {
    let outcome = state.control.begin_prefetch(id, tier)?;
    ensure_resident(state, store, id, tier, false)?;
    Ok(outcome)
}

fn ensure_resident(
    state: &mut ManagerState,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    id: &OffloadUnitId,
    tier: MemoryTier,
    initializing: bool,
) -> Result<bool, ResidencyError> {
    ensure_many_resident(
        state,
        store,
        std::slice::from_ref(id),
        tier,
        false,
        initializing,
    )
    .map(|(created, _)| created[0])
}

fn ensure_many_resident(
    state: &mut ManagerState,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    return_transfer: bool,
    initializing: bool,
) -> Result<(Vec<bool>, Option<SubmittedResidentTransfer>), ResidencyError> {
    validate_target(tier, "residency transition")?;
    if ids.is_empty() {
        return Ok((Vec::new(), None));
    }
    let mut owner_ids = BTreeSet::new();
    for id in ids {
        let unit = state
            .control
            .unit(id)
            .ok_or(ResidencyError::StatePoisoned)?;
        for binding in unit.bindings() {
            if let Some((owner, _)) = state.control.binding_owner(id, binding) {
                // Materialize cross-unit owners before preparing the requested
                // batch, even when that owner also appears later in `ids`.
                // Batch preparation is atomic, so it cannot consume an owner
                // that has only been prepared (rather than published) by the
                // same batch.
                if owner != id {
                    owner_ids.insert(owner.clone());
                }
            }
        }
    }
    for owner in owner_ids {
        let owner_tier = tier;
        ensure_resident(state, store, &owner, owner_tier, initializing)?;
        if state.alias_owner_pins.insert((owner.clone(), owner_tier)) {
            state.control.ledger_mut().pin(&owner, owner_tier, 1)?;
        }
    }
    let acquisition = if initializing {
        state.control.plan_initialization_acquisition(ids, tier)?
    } else {
        state.control.plan_acquisition(ids, tier)?
    };
    let created = acquisition.missing().to_vec();
    if acquisition.is_hit() {
        state.control.touch_acquisition_hits(&acquisition, tier)?;
        return Ok((created, None));
    }

    let started = Instant::now();
    let result = (|| {
        let mut reservations = Vec::new();
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                continue;
            }
            let planned = state.control.ledger().spec(id)?.bytes();
            let bindings = state
                .control
                .unit(id)
                .ok_or(ResidencyError::StatePoisoned)?
                .bindings();
            let required = resident_capacity_requirement(bindings, planned, tier)?;
            reservations.push((id.clone(), required));
        }
        let evicted = state
            .control
            .reserve_acquisition(&acquisition, &reservations, tier)?;
        release_backend_copies(state, &evicted)?;

        if tier == MemoryTier::Host {
            let mut prepared = Vec::new();
            for (id, is_missing) in ids.iter().zip(&created) {
                if !is_missing {
                    continue;
                }
                let bindings = state
                    .control
                    .unit(id)
                    .ok_or(ResidencyError::StatePoisoned)?
                    .bindings()
                    .to_vec();
                let shared = shared_host_buffers_for_unit(state, id)?;
                let buffers = materialize_host_buffers(
                    id,
                    store,
                    &bindings,
                    &state.source_stream,
                    &state.materialization,
                    &shared,
                )?;
                let logical = host_buffers_nbytes(&buffers, &bindings)?;
                let planned = state.control.ledger_mut().spec(id)?.bytes();
                if logical != planned {
                    return Err(ResidencyError::UnitByteMismatch {
                        id: id.clone(),
                        planned_bytes: planned,
                        actual_bytes: logical,
                    });
                }
                let capacity = host_buffers_capacity(&buffers, &bindings)?;
                let reserved_capacity = resident_capacity_requirement(&bindings, planned, tier)?;
                if capacity > reserved_capacity {
                    return Err(ResidencyError::HostCapacityBoundExceeded {
                        id: id.clone(),
                        reserved_bytes: reserved_capacity,
                        actual_bytes: capacity,
                    });
                }
                prepared.push((id.clone(), buffers, logical, capacity, reserved_capacity));
            }
            for (id, buffers, logical, capacity, _) in prepared {
                state.control.publish_acquisition_copy(
                    &id,
                    tier,
                    capacity,
                    logical,
                    None,
                    TransferDirection::DiskToHost,
                    started.elapsed(),
                )?;
                state
                    .storage
                    .get_mut(&id)
                    .ok_or(ResidencyError::StatePoisoned)?
                    .host = Some(Arc::new(buffers));
            }
            state.control.touch_acquisition_hits(&acquisition, tier)?;
            return Ok((created.clone(), None));
        }

        let mut prepared = Vec::new();
        for (id, is_missing) in ids.iter().zip(&created) {
            if !is_missing {
                continue;
            }
            let bindings = state
                .control
                .unit(id)
                .ok_or(ResidencyError::StatePoisoned)?
                .bindings()
                .to_vec();
            let shared = shared_arrays_for_unit(state, store, id)?;
            let item = loop {
                let item = match tier {
                    MemoryTier::Device => {
                        if let Some(host) = state.storage[id].host.as_ref().map(Arc::clone) {
                            prepare_copy_to_device(id, host, &state.device_stream)
                        } else {
                            prepare_from_disk(
                                store,
                                &bindings,
                                &state.source_stream,
                                &state.device_stream,
                                &state.materialization,
                                TransferDirection::DiskToDevice,
                                &shared,
                            )
                        }
                    }
                    MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
                };
                match item {
                    Ok(item) => break item,
                    Err(error)
                        if is_mapping_capacity_error(&error)
                            && prepared.iter().any(
                                |(_, item): &(OffloadUnitId, PreparedResidentArrays)| {
                                    !item.pending_sources.is_empty()
                                },
                            ) =>
                    {
                        // Earlier units in this batch can pin the only mapped
                        // shard while a later cross-shard expert is prepared.
                        // Their output arrays are complete evaluation roots, so
                        // detach those mappings and retry the current unit.
                        eval(prepared.iter().flat_map(|(_, item)| item.arrays.values())).map_err(
                            |source| ResidencyError::Mlx {
                                id: internal_id(),
                                operation: "mapping-capacity batch evaluation",
                                source,
                            },
                        )?;
                        for (_, item) in &mut prepared {
                            for source in item.pending_sources.drain(..) {
                                source.complete();
                            }
                            item.retained_arrays.clear();
                            item.retained_host = None;
                            item.retained_events.clear();
                        }
                    }
                    Err(error) => return Err(error),
                }
            };
            prepared.push((id.clone(), item));
        }

        for (id, item) in &prepared {
            let bindings = state
                .control
                .unit(id)
                .ok_or(ResidencyError::StatePoisoned)?
                .bindings();
            let actual = arrays_nbytes(&item.arrays, bindings)?;
            let required = state.control.ledger_mut().spec(id)?.bytes();
            if actual != required {
                return Err(ResidencyError::UnitByteMismatch {
                    id: id.clone(),
                    planned_bytes: required,
                    actual_bytes: actual,
                });
            }
        }

        let event =
            async_eval_with_event(prepared.iter().flat_map(|(_, item)| item.arrays.values()))
                .map_err(|source| ResidencyError::Mlx {
                    id: internal_id(),
                    operation: "batched residency submission",
                    source,
                })?;
        let generation = if return_transfer {
            state.control.ledger_mut().next_transfer_generation()?
        } else {
            let completion = event.synchronize();
            for (_, item) in &mut prepared {
                for source in item.pending_sources.drain(..) {
                    source.complete();
                }
                item.retained_arrays.clear();
                item.retained_host = None;
                item.retained_events.clear();
            }
            completion.map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "batched residency completion",
                source,
            })?;
            0
        };

        let submitted_ids = prepared
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let mut retained = ResidentTransferResources {
            sources: Vec::new(),
            retained_arrays: Vec::new(),
            retained_host: Vec::new(),
            retained_events: Vec::new(),
        };

        for (id, mut item) in prepared {
            if return_transfer {
                retained.sources.append(&mut item.pending_sources);
                retained.retained_arrays.append(&mut item.retained_arrays);
                if let Some(host) = item.retained_host.take() {
                    retained.retained_host.push(host);
                }
                retained.retained_events.append(&mut item.retained_events);
            }
            let bindings = state
                .control
                .unit(&id)
                .ok_or(ResidencyError::StatePoisoned)?
                .bindings();
            let actual = arrays_nbytes(&item.arrays, bindings)?;
            state.control.publish_acquisition_copy(
                &id,
                tier,
                actual,
                actual,
                return_transfer.then_some(generation),
                item.direction,
                started.elapsed(),
            )?;
            let unit = state
                .storage
                .get_mut(&id)
                .ok_or(ResidencyError::StatePoisoned)?;
            match tier {
                MemoryTier::Device => {
                    unit.device = Some(Arc::new(ResidentArrays {
                        arrays: item.arrays,
                    }))
                }
                MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
            }
        }
        state.control.touch_acquisition_hits(&acquisition, tier)?;
        let submitted = return_transfer.then_some(SubmittedResidentTransfer {
            event,
            retained,
            ids: submitted_ids,
            generation,
        });
        Ok((created.clone(), submitted))
    })();

    if result.is_err() {
        state.control.rollback_acquisition(&acquisition, tier)?;
    }
    result
}

struct PreparedResidentArrays {
    arrays: BTreeMap<String, Array>,
    pending_sources: Vec<PendingWeightMaterialization>,
    retained_arrays: Vec<Array>,
    retained_host: Option<Arc<ResidentHostBuffers>>,
    retained_events: Vec<Event>,
    direction: TransferDirection,
}

fn shared_arrays_for_unit(
    state: &ManagerState,
    _store: &dyn eredu_checkpoint::store::CheckpointSource,
    id: &OffloadUnitId,
) -> Result<BTreeMap<String, Array>, ResidencyError> {
    let shared = state
        .control
        .unit(id)
        .ok_or(ResidencyError::StatePoisoned)?
        .bindings()
        .iter()
        .filter_map(|binding| {
            state
                .control
                .binding_owner(id, binding)
                .map(|(owner_unit, owner)| {
                    (
                        binding.name().to_owned(),
                        (owner_unit.clone(), owner.name().to_owned()),
                        owner.name().to_owned(),
                    )
                })
        })
        .collect::<Vec<_>>();
    let mut arrays = BTreeMap::new();
    for (target, (owner_unit, owner_name), _) in shared {
        if owner_unit == *id {
            continue;
        }
        let array = state
            .storage
            .get(&owner_unit)
            .and_then(|storage| storage.device.as_ref())
            .and_then(|owner| owner.arrays.get(&owner_name))
            .ok_or(ResidencyError::StatePoisoned)?;
        arrays.insert(target, array.clone());
    }
    Ok(arrays)
}

fn shared_host_buffers_for_unit(
    state: &ManagerState,
    id: &OffloadUnitId,
) -> Result<BTreeMap<String, Arc<ImmutableHostTransferBuffer>>, ResidencyError> {
    let unit = state
        .control
        .unit(id)
        .ok_or(ResidencyError::StatePoisoned)?;
    let mut shared = BTreeMap::new();
    for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
        let (owner_unit, owner) = state
            .control
            .binding_owner(id, binding)
            .ok_or(ResidencyError::StatePoisoned)?;
        if owner_unit == id {
            continue;
        }
        let buffer = state
            .storage
            .get(owner_unit)
            .and_then(|storage| storage.host.as_ref())
            .and_then(|buffers| buffers.buffers.get(owner.name()))
            .ok_or(ResidencyError::StatePoisoned)?;
        shared.insert(binding.name().to_owned(), Arc::clone(buffer));
    }
    Ok(shared)
}

fn materialize_host_buffers(
    id: &OffloadUnitId,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    context: &MlxParameterMaterializationContext,
    shared: &BTreeMap<String, Arc<ImmutableHostTransferBuffer>>,
) -> Result<ResidentHostBuffers, ResidencyError> {
    let mut buffers = shared.clone();
    for binding in bindings {
        if buffers.contains_key(binding.name()) {
            continue;
        }
        let (array, sources) = match binding.recipe() {
            Some(recipe) => {
                let pending = recipe
                    .prepare_materialization(store, context)
                    .map_err(|source| ResidencyError::Recipe {
                        binding: binding.name().to_owned(),
                        source,
                    })?;
                pending.into_parts()
            }
            None => {
                let lease = store
                    .acquire_lease(eredu_checkpoint::store::TensorReadRequest {
                        key: binding.checkpoint_key().to_owned(),
                        selection: binding.selection().clone(),
                        policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                    })
                    .map_err(crate::backend::runtime::checkpoint::store::neutral_store_error)?;
                let lease = context.weight_lease(lease)?;
                let pending = lease.prepare_materialization(source_stream, source_stream)?;
                (pending.output().clone(), vec![pending])
            }
        };
        let buffer = HostTransferBuffer::copy_from_array(
            &array,
            HostTransferPolicy::Transfer,
            source_stream,
        )
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "weight array-to-host-buffer submission",
            source,
        })?
        .synchronize()
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "weight array-to-host-buffer completion",
            source,
        })?;
        for source in sources {
            source.complete();
        }
        let actual = u64::try_from(buffer.nbytes().map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "host-buffer byte inspection",
            source,
        })?)
        .map_err(|_| ResidencyError::ArithmeticOverflow {
            context: "host-buffer byte conversion",
        })?;
        if actual != binding.expected_bytes() {
            return Err(ResidencyError::BindingByteMismatch {
                id: id.clone(),
                binding: binding.name().to_owned(),
                expected_bytes: binding.expected_bytes(),
                actual_bytes: actual,
            });
        }
        buffers.insert(binding.name().to_owned(), Arc::new(buffer.freeze()));
    }
    Ok(ResidentHostBuffers { buffers })
}

fn prepare_from_disk(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    context: &MlxParameterMaterializationContext,
    direction: TransferDirection,
    shared: &BTreeMap<String, Array>,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = shared.clone();
    let mut pending_sources = Vec::new();
    let mut retained_arrays = Vec::new();
    for binding in bindings {
        if arrays.contains_key(binding.name()) {
            continue;
        }
        let mut retried_after_capacity = false;
        loop {
            let prepared = (|| match binding.recipe() {
                Some(recipe) => {
                    let pending =
                        recipe
                            .prepare_materialization(store, context)
                            .map_err(|source| ResidencyError::Recipe {
                                binding: binding.name().to_owned(),
                                source,
                            })?;
                    let (host, sources) = pending.into_parts();
                    if execution_stream == source_stream {
                        Ok((host, sources, None))
                    } else {
                        let output = host.copy(execution_stream).map_err(|source| {
                            ResidencyError::Recipe {
                                binding: binding.name().to_owned(),
                                source: WeightRecipeError::Mlx(source),
                            }
                        })?;
                        Ok((output, sources, Some(host)))
                    }
                }
                None => {
                    let lease = store
                        .acquire_lease(eredu_checkpoint::store::TensorReadRequest {
                            key: binding.checkpoint_key().to_owned(),
                            selection: binding.selection().clone(),
                            policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                        })
                        .map_err(crate::backend::runtime::checkpoint::store::neutral_store_error)?;
                    let lease = context.weight_lease(lease)?;
                    let pending = lease.prepare_materialization(source_stream, execution_stream)?;
                    let output = pending.output().clone();
                    Ok((output, vec![pending], None))
                }
            })();
            match prepared {
                Ok((output, sources, retained)) => {
                    pending_sources.extend(sources);
                    retained_arrays.extend(retained);
                    arrays.insert(binding.name().to_owned(), output);
                    break;
                }
                Err(error)
                    if !retried_after_capacity
                        && !pending_sources.is_empty()
                        && is_mapping_capacity_error(&error) =>
                {
                    eval(arrays.values()).map_err(|source| ResidencyError::Mlx {
                        id: internal_id(),
                        operation: "mapping-capacity residency evaluation",
                        source,
                    })?;
                    for source in pending_sources.drain(..) {
                        source.complete();
                    }
                    retained_arrays.clear();
                    retried_after_capacity = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources,
        retained_arrays,
        retained_host: None,
        retained_events: Vec::new(),
        direction,
    })
}

fn is_mapping_capacity_error(error: &ResidencyError) -> bool {
    matches!(
        error,
        ResidencyError::WeightStore(WeightStoreError::CapacityExhausted { .. })
            | ResidencyError::Recipe {
                source: WeightRecipeError::WeightStore(WeightStoreError::CapacityExhausted { .. }),
                ..
            }
    )
}

fn prepare_copy_to_device(
    id: &OffloadUnitId,
    host: Arc<ResidentHostBuffers>,
    device_stream: &Stream,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = BTreeMap::new();
    let mut retained_events = Vec::new();
    for (name, buffer) in &host.buffers {
        let submitted =
            buffer
                .copy_to_array(device_stream)
                .map_err(|source| ResidencyError::Mlx {
                    id: id.clone(),
                    operation: "host-buffer-to-device copy",
                    source,
                })?;
        let (array, completion) = submitted.into_parts();
        arrays.insert(name.clone(), array);
        retained_events.push(completion);
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources: Vec::new(),
        retained_arrays: Vec::new(),
        retained_host: Some(host),
        retained_events,
        direction: TransferDirection::HostToDevice,
    })
}

fn arrays_nbytes(
    arrays: &BTreeMap<String, Array>,
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let array = arrays
                .get(binding.name())
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes =
                u64::try_from(array.nbytes()).map_err(|_| ResidencyError::ArithmeticOverflow {
                    context: "array byte conversion",
                })?;
            total
                .checked_add(bytes)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "resident array byte total",
                })
        })
}

fn resident_capacity_requirement(
    bindings: &[WeightBinding],
    planned_bytes: u64,
    tier: MemoryTier,
) -> Result<u64, ResidencyError> {
    if tier != MemoryTier::Host {
        return Ok(planned_bytes);
    }
    host_capacity_upper_bound_for_bindings(bindings)
}

/// Returns the complete charged host-transfer capacity for one atomic unit.
pub fn host_capacity_upper_bound_for_bindings(
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let logical = usize::try_from(binding.expected_bytes()).map_err(|_| {
                ResidencyError::ArithmeticOverflow {
                    context: "host capacity-bound input conversion",
                }
            })?;
            let capacity =
                host_transfer_capacity_upper_bound(logical, HostTransferPolicy::Transfer).map_err(
                    |source| ResidencyError::Mlx {
                        id: internal_id(),
                        operation: "host-transfer capacity-bound query",
                        source,
                    },
                )?;
            let capacity =
                u64::try_from(capacity).map_err(|_| ResidencyError::ArithmeticOverflow {
                    context: "host capacity-bound output conversion",
                })?;
            total
                .checked_add(capacity)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "host unit capacity bound",
                })
        })
}

fn host_buffers_nbytes(
    buffers: &ResidentHostBuffers,
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let buffer = buffers
                .buffers
                .get(binding.name())
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes = u64::try_from(buffer.nbytes().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "host-buffer byte inspection",
                source,
            })?)
            .map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "host-buffer byte conversion",
            })?;
            total
                .checked_add(bytes)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "resident host-buffer byte total",
                })
        })
}

fn host_buffers_capacity(
    buffers: &ResidentHostBuffers,
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let buffer = buffers
                .buffers
                .get(binding.name())
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes = u64::try_from(buffer.capacity().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "host-buffer capacity inspection",
                source,
            })?)
            .map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "host-buffer capacity conversion",
            })?;
            total
                .checked_add(bytes)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "resident host-buffer capacity total",
                })
        })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{mpsc, Arc, Barrier},
        time::Duration,
    };

    use eredu_checkpoint::store::{CheckpointSource, SafetensorsWeightStore, TensorSelection};
    use eredu_runtime::{DeviceLayerWindow, ResidentLayerGroup};
    use safemlx::{
        host_transfer_capacity_upper_bound, Device, DeviceType, HostTransferPolicy,
        HostTransferStorageKind,
    };
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::*;
    use eredu_core::residency::{
        OffloadConfig, OffloadUnitSpec, ResidencyLedgerError, ResidencyPolicy,
    };

    fn cpu_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    fn write_fixture(path: &std::path::Path) {
        let a = [1i32, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let b = [3i32, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let c = [5i32, 6]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let matrix = [10i32, 11, 12, 13, 14, 15]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                ("a", TensorView::new(Dtype::I32, vec![2], &a).unwrap()),
                ("b", TensorView::new(Dtype::I32, vec![2], &b).unwrap()),
                ("c", TensorView::new(Dtype::I32, vec![2], &c).unwrap()),
                (
                    "matrix",
                    TensorView::new(Dtype::I32, vec![3, 2], &matrix).unwrap(),
                ),
            ],
            None,
            path,
        )
        .unwrap();
    }

    fn fixture_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(&dir.path().join("model.safetensors"));
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    fn cross_shard_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        for (file, key, values) in [
            ("model-00001-of-00002.safetensors", "left", [1i32, 2]),
            ("model-00002-of-00002.safetensors", "right", [3i32, 4]),
        ] {
            let bytes = values
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            serialize_to_file(
                [(key, TensorView::new(Dtype::I32, vec![2], &bytes).unwrap())],
                None,
                &dir.path().join(file),
            )
            .unwrap();
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "left": "model-00001-of-00002.safetensors",
                    "right": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let store =
            Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap());
        (dir, store)
    }

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    fn binding(name: &str, key: &str, selection: TensorSelection, bytes: u64) -> WeightBinding {
        WeightBinding::new(name, key, selection, bytes).unwrap()
    }

    fn unit(name: &str, bindings: impl IntoIterator<Item = WeightBinding>) -> OffloadUnit {
        OffloadUnit::new(id(name), bindings).unwrap()
    }

    fn spec(name: &str, bytes: u64, policy: ResidencyPolicy, tier: MemoryTier) -> OffloadUnitSpec {
        OffloadUnitSpec::new(id(name), bytes, policy, tier).unwrap()
    }

    fn manager(
        store: Arc<SafetensorsWeightStore>,
        config: OffloadConfig,
        specs: impl IntoIterator<Item = OffloadUnitSpec>,
        units: impl IntoIterator<Item = OffloadUnit>,
    ) -> ResidencyManager {
        // Most fixtures below use one independently allocated eight-byte
        // binding as their accounting unit. Preserve those unit-count tests
        // while exercising the production contract, which charges the full
        // backing extent of every host-transfer allocation.
        let host_budget = config.host_budget_bytes().map(fixture_physical_host_budget);
        let config = OffloadConfig::new(
            config.device_budget_bytes(),
            host_budget,
            config.prefetch_depth(),
        )
        .unwrap()
        .with_eviction_policy(config.eviction_policy());
        ResidencyManager::new(
            store,
            OffloadPlan::new(config, specs).unwrap(),
            units,
            cpu_stream(),
            cpu_stream(),
        )
        .unwrap()
    }

    fn fixture_physical_host_budget(logical_bytes: u64) -> u64 {
        if logical_bytes == 0 {
            return 0;
        }
        let minimum_capacity =
            host_transfer_capacity_upper_bound(1, HostTransferPolicy::Transfer).unwrap() as u64;
        if logical_bytes >= minimum_capacity {
            return logical_bytes;
        }
        let complete_bindings = logical_bytes / 8;
        let remainder = logical_bytes % 8;
        let binding_capacity = fixture_binding_capacity();
        complete_bindings
            .checked_mul(binding_capacity)
            .and_then(|bytes| {
                (remainder != 0)
                    .then(|| {
                        host_transfer_capacity_upper_bound(
                            remainder as usize,
                            HostTransferPolicy::Transfer,
                        )
                        .unwrap() as u64
                    })
                    .map_or(Some(bytes), |tail| bytes.checked_add(tail))
            })
            .unwrap()
    }

    fn fixture_binding_capacity() -> u64 {
        host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64
    }

    fn fixture_host_capacity(bindings: u64) -> u64 {
        bindings.checked_mul(fixture_binding_capacity()).unwrap()
    }

    fn single(name: &str, key: &str) -> OffloadUnit {
        unit(name, [binding("weight", key, TensorSelection::Full, 8)])
    }

    fn host_i32(lease: &ResidentUnitLease, name: &str) -> Vec<i32> {
        lease
            .host_value(name)
            .unwrap()
            .as_bytes()
            .unwrap()
            .chunks_exact(size_of::<i32>())
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn named_execution_groups_keep_independent_windows_and_clear_in_isolation() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, None, 1).unwrap(),
            [
                spec("text.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("text.1", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("vision.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            ],
            [
                single("text.0", "a"),
                single("text.1", "b"),
                single("vision.0", "c"),
            ],
        );
        manager.initialize().unwrap();
        let text = ResidentLayerGroup::new("text", [id("text.0"), id("text.1")], 1).unwrap();
        let vision = ResidentLayerGroup::new("vision", [id("vision.0")], 1).unwrap();

        text.prepare(&manager, 0).unwrap();
        vision.prepare(&manager, 0).unwrap();
        let report = manager.report().unwrap();
        assert_eq!(report.active_window(), &[id("text.0"), id("vision.0")]);
        assert!(state(&report, "text.0").device_resident());
        assert!(state(&report, "vision.0").device_resident());

        text.clear(&manager).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "text.0").device_resident());
        assert!(state(&report, "vision.0").device_resident());
        assert_eq!(report.active_window(), &[id("vision.0")]);
        let vision_report = vision.report(&manager).unwrap();
        assert_eq!(vision_report.device_units(), 1);
        assert_eq!(vision_report.device_bytes(), 8);
    }

    fn state<'a>(report: &'a ResidencyReport, name: &str) -> &'a UnitResidencyReport {
        report
            .units()
            .iter()
            .find(|unit| unit.id() == &id(name))
            .unwrap()
    }

    #[test]
    fn failed_batch_reservation_rolls_back_and_cache_remains_usable() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b")],
        );
        manager.initialize().unwrap();
        assert!(matches!(
            manager.acquire_many_with_demand(&[(id("a"), 1), (id("b"), 1)], MemoryTier::Device),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::BudgetExhausted { .. }
            ))
        ));
        let report = manager.report().unwrap();
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 0);
        assert!(!state(&report, "a").device_resident());
        assert!(!state(&report, "b").device_resident());

        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(lease.device_value("weight").unwrap().shape(), &[2]);
    }

    #[test]
    fn batched_units_detach_prior_shards_at_mapping_capacity() {
        let (_dir, store) = cross_shard_store();
        let manager = manager(
            Arc::clone(&store),
            OffloadConfig::new(None, None, 1).unwrap(),
            [
                spec("left", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("right", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("left", "left"), single("right", "right")],
        );
        manager.initialize().unwrap();

        let leases = manager
            .acquire_many_with_demand(&[(id("left"), 1), (id("right"), 1)], MemoryTier::Host)
            .unwrap();

        assert_eq!(host_i32(&leases[0], "weight"), [1, 2]);
        assert_eq!(host_i32(&leases[1], "weight"), [3, 4]);
        let diagnostics = store.source_diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert!(diagnostics.evictions >= 1);
    }

    #[test]
    #[ignore = "requires local MLX host-transfer device support"]
    fn cross_unit_alias_reacquisition_reuses_one_pinned_owner_read() {
        let (_dir, store) = fixture_store();
        let owner = unit(
            "owner",
            [binding("weight", "a", TensorSelection::Full, 8)
                .with_logical_target("shared.weight")
                .unwrap()],
        );
        let alias = unit(
            "alias",
            [
                WeightBinding::alias("shared", "shared.weight", 8).unwrap(),
                binding("local", "b", TensorSelection::Full, 8),
            ],
        );
        let manager = manager(
            Arc::clone(&store),
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("owner", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [owner, alias],
        );
        manager.initialize().unwrap();

        let first = manager.acquire(&id("alias"), MemoryTier::Host).unwrap();
        assert_eq!(host_i32(&first, "shared"), [1, 2]);
        assert_eq!(host_i32(&first, "local"), [3, 4]);
        drop(first);
        let reads = store.source_diagnostics().unwrap().physical_reads;
        assert_eq!(reads, 2, "one owner plus one local tensor must be read");
        let report = manager.report().unwrap();
        assert_eq!(
            report
                .units()
                .iter()
                .map(|unit| unit.expected_bytes())
                .sum::<u64>(),
            16
        );

        assert!(manager.evict(&id("alias"), MemoryTier::Host).unwrap());
        let second = manager.acquire(&id("alias"), MemoryTier::Host).unwrap();
        assert_eq!(host_i32(&second, "shared"), [1, 2]);
        assert_eq!(
            store.source_diagnostics().unwrap().physical_reads,
            reads + 1,
            "only the evicted alias-local tensor may be reread; the pinned shared owner must not"
        );
    }

    #[test]
    fn caller_owned_transfer_publishes_only_after_exact_completion() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 2)], MemoryTier::Device)
            .unwrap();
        let lease = &transfer.leases()[0];
        {
            let state = manager.lock().unwrap();
            let copy = state
                .control
                .ledger()
                .copy_status(&id("a"), MemoryTier::Device)
                .unwrap()
                .unwrap();
            assert_eq!(copy.pins(), 1);
            assert!(copy.in_flight().is_some());
        }

        let consumer = cpu_stream();
        transfer.order_after(&consumer).unwrap();
        let dependent = lease
            .device_value("weight")
            .unwrap()
            .add(Array::from_int(1), &consumer)
            .unwrap();
        eval([&dependent]).unwrap();
        transfer.synchronize().unwrap();
        {
            let state = manager.lock().unwrap();
            let copy = state
                .control
                .ledger()
                .copy_status(&id("a"), MemoryTier::Device)
                .unwrap()
                .unwrap();
            assert!(copy.in_flight().is_none());
        }
        let ready = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        assert!(ready.is_complete().unwrap());
        drop(ready);
        drop(transfer);
        assert_eq!(state(&manager.report().unwrap(), "a").device_pins(), 0);
    }

    #[test]
    fn empty_caller_owned_transfer_is_immediately_complete() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut transfer = manager
            .acquire_many_with_transfer(&[], MemoryTier::Device)
            .unwrap();
        assert!(transfer.is_empty());
        assert!(transfer.is_complete().unwrap());
        transfer.order_after(&cpu_stream()).unwrap();
        transfer.synchronize().unwrap();
        assert!(!manager.is_resident(&id("a"), MemoryTier::Device).unwrap());
    }

    #[test]
    fn dropping_transfer_retains_queued_consumer_and_publishes_copy() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        let consumer = cpu_stream();
        transfer.order_after(&consumer).unwrap();
        let dependent = transfer.leases()[0]
            .device_value("weight")
            .unwrap()
            .add(Array::from_int(1), &consumer)
            .unwrap();
        drop(transfer);
        assert!(manager
            .lock()
            .unwrap()
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Device)
            .unwrap()
            .unwrap()
            .in_flight()
            .is_none());
        assert_eq!(dependent.evaluated().unwrap().as_slice::<i32>(), &[2, 3]);
    }

    #[test]
    fn synchronous_acquisition_waits_for_caller_owned_transfer() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();

        let mut transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        let waiting_manager = manager.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let result = waiting_manager
                .acquire(&id("a"), MemoryTier::Device)
                .map(|lease| {
                    lease
                        .device_value("weight")
                        .unwrap()
                        .evaluated()
                        .unwrap()
                        .as_slice::<i32>()
                        .to_vec()
                });
            sender.send(result).unwrap();
        });
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        transfer.synchronize().unwrap();
        assert_eq!(
            receiver
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap(),
            vec![1, 2]
        );
        waiter.join().unwrap();
    }

    #[test]
    fn validates_unit_identity_bindings_sizes_and_targets() {
        let (_dir, store) = fixture_store();
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        )
        .unwrap();
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::Controller(
                ResidencyControllerError::MissingUnitDefinition { .. }
            ))
        ));
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [single("a", "a"), single("a", "a")],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::Controller(
                ResidencyControllerError::DuplicateUnitDefinition { .. }
            ))
        ));
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [single("a", "a"), single("b", "b")],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::Controller(
                ResidencyControllerError::UnexpectedUnitDefinition { .. }
            ))
        ));
        assert!(matches!(
            OffloadUnit::new(id("empty"), []),
            Err(eredu_runtime::ResidencyDeclarationError::EmptyUnit { .. })
        ));
        let duplicate = binding("same", "a", TensorSelection::Full, 8);
        assert!(matches!(
            OffloadUnit::new(id("duplicate"), [duplicate.clone(), duplicate]),
            Err(eredu_runtime::ResidencyDeclarationError::DuplicateBindingName { .. })
        ));
        let wrong = unit("a", [binding("weight", "a", TensorSelection::Full, 4)]);
        assert!(matches!(
            ResidencyManager::new(
                Arc::clone(&store),
                plan.clone(),
                [wrong],
                cpu_stream(),
                cpu_stream()
            ),
            Err(ResidencyError::Controller(
                ResidencyControllerError::BindingByteMismatch { .. }
            ))
        ));

        let valid =
            ResidencyManager::new(store, plan, [single("a", "a")], cpu_stream(), cpu_stream())
                .unwrap();
        assert!(matches!(
            valid.prefetch(&id("a"), MemoryTier::Disk),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::InvalidTargetTier { .. }
            ))
        ));
    }

    #[test]
    fn detects_unit_total_overflow_before_checkpoint_access() {
        let (_dir, store) = fixture_store();
        let overflowing = unit(
            "a",
            [
                binding("a", "a", TensorSelection::Full, 8),
                binding("z", "missing", TensorSelection::Full, u64::MAX),
            ],
        );
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        )
        .unwrap();
        assert!(matches!(
            ResidencyManager::new(store, plan, [overflowing], cpu_stream(), cpu_stream()),
            Err(ResidencyError::Controller(
                ResidencyControllerError::ArithmeticOverflow { .. }
            ))
        ));
    }

    #[test]
    fn initialization_honors_planned_tiers_and_pinning() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                spec("disk", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("host", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
                spec("device", 8, ResidencyPolicy::Cacheable, MemoryTier::Device),
            ],
            [
                single("disk", "a"),
                single("host", "b"),
                single("device", "c"),
            ],
        );
        assert!(matches!(
            manager.acquire(&id("disk"), MemoryTier::Host),
            Err(ResidencyError::Ledger(ResidencyLedgerError::NotInitialized))
        ));
        manager.initialize().unwrap();
        let report = manager.report().unwrap();
        assert!(report.initialized());
        assert!(!state(&report, "disk").host_resident());
        assert!(state(&report, "host").host_resident());
        assert!(state(&report, "device").device_resident());
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
        assert!(matches!(
            manager.evict(&id("host"), MemoryTier::Host),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::PinnedEviction { .. }
            ))
        ));
    }

    #[test]
    fn host_residency_charges_physical_allocation_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let values = (0..5000u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![5000], &values).unwrap(),
            )],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let logical = u64::try_from(values.len()).unwrap();
        let capacity =
            safemlx::host_transfer_capacity_upper_bound(values.len(), HostTransferPolicy::Transfer)
                .map(|capacity| u64::try_from(capacity).unwrap())
                .unwrap();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(capacity), 1).unwrap(),
            [spec(
                "host",
                logical,
                ResidencyPolicy::Pinned,
                MemoryTier::Host,
            )],
            [unit(
                "host",
                [binding("weight", "weight", TensorSelection::Full, logical)],
            )],
        );
        manager.initialize().unwrap();
        let report = manager.report().unwrap();
        let unit = state(&report, "host");
        assert_eq!(unit.expected_bytes(), logical);
        assert_eq!(unit.host_allocated_bytes(), capacity);
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            capacity
        );
    }

    #[test]
    fn partial_initialization_failure_remains_consistent_and_inspectable() {
        let dir = tempfile::tempdir().unwrap();
        let good = [7u8, 8];
        let bad = [9u8, 10];
        serialize_to_file(
            [
                ("good", TensorView::new(Dtype::U8, vec![2], &good).unwrap()),
                (
                    "unsupported",
                    TensorView::new(Dtype::F8_E5M2, vec![2], &bad).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(fixture_host_capacity(3)), 1).unwrap(),
            [
                spec("a-good", 2, ResidencyPolicy::Cacheable, MemoryTier::Host),
                spec("z-bad", 4, ResidencyPolicy::Cacheable, MemoryTier::Host),
            ],
            [
                unit(
                    "a-good",
                    [binding("weight", "good", TensorSelection::Full, 2)],
                ),
                unit(
                    "z-bad",
                    [
                        binding("a-good-copy", "good", TensorSelection::Full, 2),
                        binding("z-unsupported", "unsupported", TensorSelection::Full, 2),
                    ],
                ),
            ],
        );
        assert!(matches!(
            manager.initialize(),
            Err(ResidencyError::WeightStore(
                WeightStoreError::UnsupportedStoredDtype { .. }
            ))
        ));
        let report = manager.report().unwrap();
        assert!(!report.initialized());
        assert!(state(&report, "a-good").host_resident());
        assert!(!state(&report, "z-bad").host_resident());
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
    }

    #[test]
    fn materializes_promotes_and_publishes_multi_tensor_units_atomically() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(16), Some(16), 1).unwrap(),
            [spec(
                "quantized",
                16,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )],
            [unit(
                "quantized",
                [
                    binding("scales", "b", TensorSelection::Full, 8),
                    binding("weight", "a", TensorSelection::Full, 8),
                ],
            )],
        );
        manager.initialize().unwrap();
        assert_eq!(
            manager
                .prefetch(&id("quantized"), MemoryTier::Host)
                .unwrap(),
            PrefetchOutcome::Miss
        );
        let host = manager.acquire(&id("quantized"), MemoryTier::Host).unwrap();
        assert_eq!(
            host.binding_names().collect::<Vec<_>>(),
            ["scales", "weight"]
        );
        assert_eq!(host_i32(&host, "weight"), [1, 2]);
        assert!(matches!(
            host.device_value("weight"),
            Err(ResidencyError::HostBindingIsNotArray { .. })
        ));
        assert!(matches!(
            host.host_value("unknown"),
            Err(ResidencyError::UnknownBinding { .. })
        ));
        assert!(matches!(
            host.host_value("weight").unwrap().storage_kind().unwrap(),
            HostTransferStorageKind::Cpu
                | HostTransferStorageKind::MetalShared
                | HostTransferStorageKind::CudaPinned
        ));
        drop(host);
        assert_eq!(
            manager
                .prefetch(&id("quantized"), MemoryTier::Device)
                .unwrap(),
            PrefetchOutcome::Miss
        );
        let device = manager
            .acquire(&id("quantized"), MemoryTier::Device)
            .unwrap();
        assert_eq!(device.device_value("scales").unwrap().shape(), &[2]);
        assert!(matches!(
            device.host_value("scales"),
            Err(ResidencyError::DeviceBindingIsNotHostBuffer { .. })
        ));
        let report = manager.report().unwrap();
        assert!(state(&report, "quantized").host_resident());
        assert!(state(&report, "quantized").device_resident());
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .bytes(),
            16
        );
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .bytes(),
            16
        );
    }

    #[test]
    fn direct_disk_to_device_does_not_create_a_host_copy() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "a").device_resident());
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToDevice)
                .bytes(),
            8
        );
    }

    #[test]
    fn budgets_use_deterministic_policy_then_lru_eviction() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("cache-a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("window-b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("cache-c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                single("cache-a", "a"),
                single("window-b", "b"),
                single("cache-c", "c"),
            ],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(state(&report, "cache-a").host_resident());
        assert!(!state(&report, "window-b").host_resident());
        assert!(state(&report, "cache-c").host_resident());
        assert_eq!(report.offload().evictions().count(), 1);
        assert_eq!(
            report.offload().evictions().bytes(),
            fixture_binding_capacity()
        );

        manager.evict(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.evict(&id("cache-c"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "cache-a").host_resident());
        assert!(state(&report, "cache-c").host_resident());
        assert!(state(&report, "window-b").host_resident());
        assert!(
            report.offload().resident_bytes().get(MemoryTier::Host) <= fixture_host_capacity(2)
        );
    }

    #[test]
    fn oldest_copy_is_evicted_deterministically() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("c"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "b").host_resident());
        assert!(state(&report, "c").host_resident());
    }

    #[test]
    fn host_and_device_budgets_are_independent() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
        manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
        let report = manager.report().unwrap();
        assert!(!state(&report, "a").host_resident());
        assert!(state(&report, "a").device_resident());
        assert!(state(&report, "b").host_resident());
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
    }

    #[test]
    fn leases_block_eviction_and_drop_or_unwind_releases_pins() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
        assert!(matches!(
            manager.evict(&id("a"), MemoryTier::Host),
            Err(ResidencyError::Ledger(
                ResidencyLedgerError::InUseEviction { pin_count: 1, .. }
            ))
        ));
        drop(lease);
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
        assert!(!manager.evict(&id("a"), MemoryTier::Host).unwrap());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let manager = manager.clone();
            move || {
                let _lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                panic!("exercise lease unwinding");
            }
        }));
        assert!(result.is_err());
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    }

    #[test]
    fn concurrent_acquisition_materializes_once_and_counts_pins() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let handles = (0..2)
            .map(|_| {
                let manager = manager.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                    barrier.wait();
                    barrier.wait();
                    drop(lease);
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let report = manager.report().unwrap();
        assert_eq!(state(&report, "a").host_pins(), 2);
        assert_eq!(
            report
                .offload()
                .transfer(TransferDirection::DiskToHost)
                .count(),
            1
        );
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_binding_capacity()
        );
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(state(&manager.report().unwrap(), "a").host_pins(), 0);
    }

    #[test]
    fn windows_bound_lookahead_protect_active_units_and_record_hits() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 2).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
                spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        let first = manager
            .prepare_window(&[id("a")], &[id("a"), id("b"), id("c")], MemoryTier::Host)
            .unwrap();
        assert_eq!(first.len(), 2);
        assert!(first
            .iter()
            .all(|(_, value)| *value == PrefetchOutcome::Miss));
        let report_before = manager.report().unwrap();
        assert!(state(&report_before, "a").host_resident());
        assert!(state(&report_before, "b").host_resident());
        assert!(!state(&report_before, "c").host_resident());

        let second = manager
            .prepare_window(&[id("b")], &[id("b"), id("c")], MemoryTier::Host)
            .unwrap();
        assert_eq!(second[0].1, PrefetchOutcome::Hit);
        assert_eq!(second[1].1, PrefetchOutcome::Miss);
        let report_after = manager.report().unwrap();
        assert!(!state(&report_after, "a").host_resident());
        assert!(state(&report_after, "b").host_resident());
        assert!(state(&report_after, "c").host_resident());
        assert_eq!(report_after.active_window(), &[id("b")]);
        assert_eq!(report_after.offload().prefetch().requests(), 4);
        assert_eq!(report_after.offload().prefetch().hits(), 1);
        assert_eq!(report_after.offload().prefetch().misses(), 3);
        assert_eq!(report_before.offload().prefetch().requests(), 2);
    }

    #[test]
    fn exhaustion_reports_pinned_in_use_and_active_blockers() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("pinned", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
                spec("leased", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("wanted", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                single("pinned", "a"),
                single("leased", "b"),
                single("wanted", "c"),
            ],
        );
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("leased"), MemoryTier::Host).unwrap();
        let error = manager
            .prefetch(&id("wanted"), MemoryTier::Host)
            .unwrap_err();
        match error {
            ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                required_bytes,
                budget_bytes,
                blocking_units,
                ..
            }) => {
                assert_eq!(required_bytes, fixture_binding_capacity());
                assert_eq!(budget_bytes, fixture_host_capacity(2));
                assert_eq!(blocking_units.len(), 2);
                assert!(blocking_units.iter().any(|unit| unit.pinned));
                assert!(blocking_units.iter().any(|unit| unit.in_use == 1));
            }
            other => panic!("unexpected error: {other}"),
        }
        drop(lease);
    }

    #[test]
    fn demand_stalls_and_rank_local_selections_are_reported() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(16), 1).unwrap(),
            [
                spec("range", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("indices", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
            [
                unit(
                    "range",
                    [binding(
                        "weight",
                        "matrix",
                        TensorSelection::Range {
                            axis: 0,
                            start: 1,
                            end: 2,
                        },
                        8,
                    )],
                ),
                unit(
                    "indices",
                    [binding(
                        "weight",
                        "matrix",
                        TensorSelection::Indices {
                            axis: 0,
                            indices: vec![2],
                        },
                        8,
                    )],
                ),
            ],
        );
        manager.initialize().unwrap();
        let range = manager.acquire(&id("range"), MemoryTier::Host).unwrap();
        assert_eq!(host_i32(&range, "weight"), [12, 13]);
        drop(range);
        let indices = manager.acquire(&id("indices"), MemoryTier::Host).unwrap();
        assert_eq!(host_i32(&indices, "weight"), [14, 15]);
        let report = manager.report().unwrap();
        assert_eq!(report.offload().prefetch().stalls(), 2);
        assert_eq!(
            report.offload().resident_bytes().get(MemoryTier::Host),
            fixture_host_capacity(2)
        );
        assert_eq!(
            report.offload().peak_resident_bytes().get(MemoryTier::Host),
            fixture_host_capacity(2)
        );
        assert!(report.weight_store().mapping_hits > 0);
    }

    #[test]
    fn ordered_device_window_trims_stale_units_with_unlimited_budget() {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(None, Some(24), 2).unwrap(),
            [
                spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
                spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            ],
            [single("a", "a"), single("b", "b"), single("c", "c")],
        );
        manager.initialize().unwrap();
        let window = DeviceLayerWindow::new([id("a"), id("b"), id("c")], 2).unwrap();

        window.prepare(&manager, 0).unwrap();
        let first = manager.report().unwrap();
        assert!(state(&first, "a").device_resident());
        assert!(state(&first, "b").device_resident());
        assert!(!state(&first, "c").device_resident());

        let lease = manager.acquire(&id("b"), MemoryTier::Device).unwrap();
        window.prepare(&manager, 1).unwrap();
        let second = manager.report().unwrap();
        assert!(!state(&second, "a").device_resident());
        assert!(state(&second, "b").device_resident());
        assert!(state(&second, "c").device_resident());
        assert_eq!(state(&second, "b").device_pins(), 1);
        drop(lease);

        window.clear(&manager).unwrap();
        assert!(manager
            .report()
            .unwrap()
            .units()
            .iter()
            .all(|unit| !unit.device_resident()));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn promotes_host_buffers_to_a_real_metal_stream() {
        let (_dir, store) = fixture_store();
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(fixture_binding_capacity()), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
        )
        .unwrap();
        let manager = ResidencyManager::new(
            store,
            plan,
            [single("a", "a")],
            cpu_stream(),
            Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
        )
        .unwrap();
        manager.initialize().unwrap();
        let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
        assert_eq!(
            lease
                .device_value("weight")
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            manager
                .report()
                .unwrap()
                .offload()
                .transfer(TransferDirection::HostToDevice)
                .count(),
            1
        );
    }
}
