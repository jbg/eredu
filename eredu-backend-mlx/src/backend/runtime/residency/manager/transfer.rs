//! Residency transfer ownership, storage lifecycle, and acquisition.

use super::*;
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::{Recovery, Retention, Status},
};
use std::{
    cell::{Cell, OnceCell, RefCell},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        Weak,
    },
};

/// Shared lease and transfer owner for a residency manager.
pub struct ManagerInner {
    pub(super) store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    pub(super) state: Mutex<ManagerState>,
    pub(super) changed: Condvar,
    pub(super) failed_transfer: Arc<AtomicBool>,
}

impl ResidencyLeaseOwner for ManagerInner {
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier) {
        if let Ok(mut state) = self.state.lock() {
            state.control.ledger_mut().unpin(id, tier);
        }
    }
}

impl ManagerInner {
    fn resolve_transfer(
        &self,
        ids: &[OffloadUnitId],
        tier: MemoryTier,
        generation: u64,
        succeeded: bool,
    ) -> Result<(), ResidencyError> {
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

pub(super) struct ManagerState {
    pub(super) failed_transfer: Arc<AtomicBool>,
    pub(super) control: ResidencyController,
    pub(super) storage: BTreeMap<OffloadUnitId, UnitStorage>,
    pub(super) alias_owner_pins: BTreeSet<(OffloadUnitId, MemoryTier)>,
    pub(super) materialization: MlxParameterMaterializationContext,
    pub(super) source_stream: Stream,
    pub(super) device_stream: Stream,
}

#[derive(Default)]
pub(super) struct UnitStorage {
    pub(super) host: Option<Arc<ResidentHostBuffers>>,
    pub(super) device: Option<Arc<ResidentArrays>>,
}

impl UnitStorage {
    pub(super) fn remove_storage(&mut self, tier: MemoryTier) -> bool {
        match tier {
            MemoryTier::Host => self.host.take().is_some(),
            MemoryTier::Device => self.device.take().is_some(),
            MemoryTier::Disk => false,
        }
    }
}

pub(super) fn release_backend_copies(
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

/// Named device arrays retained by one resident unit.
pub struct ResidentArrays {
    pub(super) arrays: BTreeMap<String, Array>,
}

/// Named immutable host buffers retained by one resident unit.
pub struct ResidentHostBuffers {
    pub(super) buffers: BTreeMap<String, Arc<ImmutableHostTransferBuffer>>,
}

/// Source and destination resources retained through an asynchronous transfer.
pub struct ResidentTransferResources {
    pub(super) sources: Vec<PendingWeightMaterialization>,
    pub(super) retained_arrays: Vec<Array>,
    pub(super) retained_host: Vec<Arc<ResidentHostBuffers>>,
    pub(super) retained_events: Vec<Event>,
    event: Option<Event>,
    application: Rc<OrdinaryRetirement<TransferApplication>>,
}

pub(super) struct SubmittedResidentTransfer {
    retained: Recovery<Rc<ResidentTransferResources>>,
}

#[derive(Default)]
struct TransferStatus {
    settled: Cell<bool>,
    failed: Cell<bool>,
    children: Cell<usize>,
}

struct TransferApplication {
    leases: OnceCell<Vec<ResidentUnitLease>>,
    owner: RefCell<Weak<ManagerInner>>,
    ids: Vec<OffloadUnitId>,
    tier: MemoryTier,
    generation: Cell<u64>,
    status: TransferStatus,
    failed_transfer: Arc<AtomicBool>,
}

impl TransferApplication {
    fn new(ids: Vec<OffloadUnitId>, tier: MemoryTier, failed_transfer: Arc<AtomicBool>) -> Self {
        Self {
            leases: OnceCell::new(),
            owner: RefCell::new(Weak::new()),
            ids,
            tier,
            generation: Cell::new(0),
            status: TransferStatus::default(),
            failed_transfer,
        }
    }

    fn mark_failed(&self) {
        self.status.failed.set(true);
        self.failed_transfer.store(true, Ordering::Release);
    }

    fn resolve(&self) -> Result<(), ResidencyError> {
        let generation = self.generation.get();
        if generation == 0 {
            return Ok(());
        }
        let Some(owner) = self.owner.borrow().upgrade() else {
            return Ok(());
        };
        owner.resolve_transfer(
            &self.ids,
            self.tier,
            generation,
            self.status.settled.get()
                && !self.status.failed.get()
                && self.status.children.get() == 0,
        )?;
        self.generation.set(0);
        Ok(())
    }
}

impl Drop for TransferApplication {
    fn drop(&mut self) {
        // Only ordinary, unlocked reclamation reaches this manager-lock owner.
        let _ = self.resolve();
    }
}

impl Retention for ResidentTransferResources {
    fn observe(&self, status: Status) {
        self.application.status.settled.set(status.settled);
        if status.failed || status.blocked {
            self.application.mark_failed();
        }
    }
}

struct TransferObservation {
    owner: Rc<ResidentTransferResources>,
    _stream: Stream,
}

impl Retention for TransferObservation {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.owner.application.mark_failed();
        }
    }
}

impl Drop for TransferObservation {
    fn drop(&mut self) {
        let children = &self.owner.application.status.children;
        children.set(children.get() - 1);
    }
}

impl SubmittedResidentTransfer {
    pub(super) fn attach_owner(&self, owner: Weak<ManagerInner>) {
        *self.retained.retention().application.owner.borrow_mut() = owner;
    }

    pub(super) fn retain_partial_leases(&self, leases: Vec<ResidentUnitLease>) {
        let _ = self.retained.retention().application.leases.set(leases);
    }
}

struct TransferUnwind<'a>(&'a TransferApplication);
impl Drop for TransferUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.mark_failed();
        }
    }
}

/// Native transfer owner with nonblocking polling and teardown.
///
/// Pending native work keeps source and application leases independently of this
/// handle. Manager publication and unpinning run only at ordinary unlocked entry.
pub struct ResidentTransfer {
    retained: Option<Recovery<Rc<ResidentTransferResources>>>,
    application: Rc<OrdinaryRetirement<TransferApplication>>,
}

fn transfer_error(operation: &'static str, source: safemlx::error::Exception) -> ResidencyError {
    ResidencyError::Mlx {
        id: internal_id(),
        operation,
        source,
    }
}

impl ResidentTransfer {
    #[cfg(test)]
    pub(super) fn mark_failed_for_test(&self) {
        self.application.mark_failed();
    }

    /// Creates a transfer for copies already resident in the requested tier.
    pub fn immediate(leases: Vec<ResidentUnitLease>, tier: MemoryTier) -> Self {
        let app = TransferApplication::new(Vec::new(), tier, Arc::new(AtomicBool::new(false)));
        let _ = app.leases.set(leases);
        app.status.settled.set(true);
        Self {
            retained: None,
            application: Rc::new(OrdinaryRetirement::new(app)),
        }
    }

    pub(super) fn submitted(
        leases: Vec<ResidentUnitLease>,
        submitted: SubmittedResidentTransfer,
    ) -> Self {
        let application = Rc::clone(&submitted.retained.retention().application);
        let _ = application.leases.set(leases);
        Self {
            retained: Some(submitted.retained),
            application,
        }
    }

    /// The exact resident unit leases carried by this transfer.
    pub fn leases(&self) -> &[ResidentUnitLease] {
        self.application.leases.get().map_or(&[], Vec::as_slice)
    }

    /// Whether this transfer carries no resident unit leases.
    pub fn is_empty(&self) -> bool {
        self.leases().is_empty()
    }

    fn check_native_status(&self) -> Result<bool, ResidencyError> {
        crate::backend::submission_recovery::reap();
        if let Some(retained) = &self.retained {
            let status = retained.progress();
            if status.failed || status.blocked {
                self.application.mark_failed();
            }
        }
        let status = &self.application.status;
        if status.failed.get() {
            Err(transfer_error(
                "resident transfer completion",
                safemlx::error::Exception::custom(
                    "native transfer failed; unresolved resources remain retained",
                ),
            ))
        } else {
            Ok(status.settled.get() && status.children.get() == 0)
        }
    }

    /// Orders an independently retained consumer-stream dependency.
    pub fn order_after(&self, stream: &Stream) -> Result<(), ResidencyError> {
        self.check_native_status()?;
        let Some(retained) = &self.retained else {
            return Ok(());
        };
        let owner = retained.retention();
        let _unwind = TransferUnwind(&owner.application);
        let count = &owner.application.status.children;
        let next = count.get().checked_add(1).ok_or_else(|| {
            transfer_error(
                "prepare transfer consumer",
                safemlx::error::Exception::custom("transfer observation count exhausted"),
            )
        })?;
        count.set(next);
        let mut observation = Recovery::begin(TransferObservation {
            owner: Rc::clone(owner),
            _stream: stream.clone(),
        })
        .map_err(|error| transfer_error("prepare transfer consumer", error))?;
        let result = owner
            .event
            .as_ref()
            .expect("submitted transfer")
            .wait_on(stream);
        if result.is_err() {
            self.application.mark_failed();
        }
        observation.seal();
        let status = observation.progress();
        if status.failed || status.blocked {
            return Err(transfer_error(
                "transfer consumer",
                safemlx::error::Exception::custom(
                    "native transfer consumer failed; unresolved resources remain retained",
                ),
            ));
        }
        result.map_err(|error| transfer_error("resident transfer stream wait", error))
    }

    /// Nonblocking exact completion query; runtime contention returns pending.
    pub fn is_complete(&self) -> Result<bool, ResidencyError> {
        // Already-resident transfers have no native work or event to observe.
        if self.retained.is_none() {
            return self.check_native_status();
        }
        safemlx::try_with_submission_retirement(|| {
            if !self.check_native_status()? {
                return Ok(false);
            }
            match &self.retained {
                Some(retained) => retained
                    .retention()
                    .event
                    .as_ref()
                    .expect("submitted transfer")
                    .is_complete()
                    .map_err(|error| {
                        self.application.mark_failed();
                        transfer_error("resident transfer query", error)
                    }),
                None => Ok(true),
            }
        })
        .unwrap_or(Ok(false))
    }

    /// Explicitly waits for completion and publishes the exact transfer generation.
    pub fn synchronize(&mut self) -> Result<(), ResidencyError> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        if let Some(retained) = self.retained.take() {
            let status = retained.finish();
            if status.failed || status.blocked {
                self.application.mark_failed();
                return Err(transfer_error(
                    "resident transfer retirement",
                    safemlx::error::Exception::custom("native transfer failed"),
                ));
            }
        }
        self.application.resolve()?;
        Ok(())
    }
}

pub(super) fn validate_target(
    tier: MemoryTier,
    operation: &'static str,
) -> Result<(), ResidencyError> {
    if tier == MemoryTier::Disk {
        Err(ResidencyLedgerError::InvalidTargetTier { operation }.into())
    } else {
        Ok(())
    }
}

pub(super) fn internal_id() -> OffloadUnitId {
    OffloadUnitId::new("residency-manager").expect("static identifier is valid")
}

pub(super) fn prefetch_locked(
    state: &mut ManagerState,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<PrefetchOutcome, ResidencyError> {
    let outcome = state.control.begin_prefetch(id, tier)?;
    ensure_resident(state, store, id, tier, false)?;
    Ok(outcome)
}

pub(super) fn ensure_resident(
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

pub(super) fn ensure_many_resident(
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

        let application = Rc::new(OrdinaryRetirement::new(TransferApplication::new(
            ids.iter()
                .zip(&created)
                .filter(|(_, missing)| **missing)
                .map(|(id, _)| id.clone())
                .collect(),
            tier,
            Arc::clone(&state.failed_transfer),
        )));
        let mut retained = Recovery::begin(Rc::new(ResidentTransferResources {
            sources: Vec::new(),
            retained_arrays: Vec::new(),
            retained_host: Vec::new(),
            retained_events: Vec::new(),
            event: None,
            application,
        }))
        .map_err(|error| transfer_error("prepare resident transfer recovery", error))?;
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
                            prepare_copy_to_device(
                                id,
                                host,
                                &state.device_stream,
                                Rc::get_mut(retained.retention_mut())
                                    .expect("unpublished transfer"),
                            )
                        } else {
                            prepare_from_disk(
                                store,
                                &bindings,
                                &state.source_stream,
                                &state.device_stream,
                                &state.materialization,
                                TransferDirection::DiskToDevice,
                                &shared,
                                Rc::get_mut(retained.retention_mut())
                                    .expect("unpublished transfer"),
                            )
                        }
                    }
                    MemoryTier::Host | MemoryTier::Disk => unreachable!("validated above"),
                };
                match item {
                    Ok(item) => break item,
                    Err(error)
                        if is_shard_cache_capacity_error(&error)
                            && !retained.retention().sources.is_empty() =>
                    {
                        // Earlier units in this batch can pin the only cached
                        // shard while a later cross-shard entry is prepared.
                        // Their output arrays are complete evaluation roots, so
                        // detach those leases and retry the current unit.
                        let resources =
                            Rc::get_mut(retained.retention_mut()).expect("unpublished transfer");
                        let prior = WeightMaterialization::prepare_retained(
                            Vec::new(),
                            std::mem::take(&mut resources.sources),
                        )?
                        .submit_outputs(resources.retained_arrays.clone())?;
                        prior.wait()?;
                        prior.finish()?;
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

        let submitted =
            async_eval_with_event(prepared.iter().flat_map(|(_, item)| item.arrays.values()));
        retained.seal();
        if submitted.is_err() {
            retained.retention().application.mark_failed();
        }
        let event =
            submitted.map_err(|error| transfer_error("batched residency submission", error))?;
        Rc::get_mut(retained.retention_mut())
            .expect("unpublished transfer")
            .event = Some(event);
        let progress = retained.progress();
        if progress.failed || progress.blocked {
            return Err(transfer_error(
                "batched residency submission",
                safemlx::error::Exception::custom(
                    "native transfer failed; unresolved resources remain retained",
                ),
            ));
        }
        let generation = if return_transfer {
            state.control.ledger_mut().next_transfer_generation()?
        } else {
            loop {
                let status = retained.progress();
                if status.failed || status.blocked {
                    return Err(transfer_error(
                        "batched residency completion",
                        safemlx::error::Exception::custom(
                            "native transfer failed; unresolved resources remain retained",
                        ),
                    ));
                }
                if status.settled {
                    break;
                }
                std::thread::yield_now();
            }
            0
        };
        retained.retention().application.generation.set(generation);

        for (id, item) in prepared {
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
        let submitted = return_transfer.then_some(SubmittedResidentTransfer { retained });
        Ok((created.clone(), submitted))
    })();

    if result.is_err() {
        state.control.rollback_acquisition(&acquisition, tier)?;
    }
    result
}
