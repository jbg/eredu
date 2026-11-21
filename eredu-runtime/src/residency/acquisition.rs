//! Portable reservation and publication policy for immutable parameter storage.

use super::*;
use eredu_core::residency::{
    RealizedResidency, ResidencyCopyRequest, ResidencyLease, ResidencyLeaseOwner, ResidencyStorage,
    ResidencyTransferTicket, ResidentBindingAlias,
};
use eredu_core::{residency::parameter_binding::preflight_bindings, BackendFailure};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

/// The portable residency controller paired with its native storage under
/// one serialized access boundary.
pub struct ResidencyState<M> {
    controller: ResidencyController,
    native: M,
}

impl<M: ResidencyStorage> ResidencyState<M> {
    /// Adopts native storage and installs slots for the admitted unit catalog.
    fn new(controller: ResidencyController, mut native: M) -> Self {
        for unit in controller.units() {
            native.prepare_storage(unit.id());
        }
        Self { controller, native }
    }

    /// Observes the current portable controller.
    pub fn controller(&self) -> &ResidencyController {
        &self.controller
    }

    /// Mutates the portable controller while holding the storage guard.
    pub fn controller_mut(&mut self) -> &mut ResidencyController {
        &mut self.controller
    }

    /// Observes the storage owned by the driver.
    pub fn native(&self) -> &M {
        &self.native
    }

    /// Borrows native storage under the same serialized guard.
    pub fn native_mut(&mut self) -> &mut M {
        &mut self.native
    }
}

/// Consistent logical residency telemetry captured without progressing native work.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencySnapshot {
    /// Whether initial copies have been published.
    pub initialized: bool,
    /// Aggregate counters, pool usage, and optional conversion retention.
    pub offload: OffloadReport,
    /// Unit reports in the ledger's stable identifier order.
    pub units: Vec<UnitResidencyReport>,
    /// Protected units in the active windows.
    pub active_window: Vec<OffloadUnitId>,
}

/// The shared residency lock cannot be used after a panic interrupted a transaction.
#[derive(Debug, thiserror::Error)]
#[error("residency state mutex is poisoned")]
pub struct ResidencyStatePoisoned;

/// The portable owner of one admitted catalog, its serialized ledger and native storage.
pub struct ResidencyOwner<M> {
    sources: ResidencySources,
    state: Mutex<ResidencyState<M>>,
    changed: Condvar,
}

impl<M: ResidencyStorage> ResidencyOwner<M> {
    /// Installs the controller with empty native slots for the exact admitted catalog.
    pub(super) fn new(
        sources: ResidencySources,
        controller: ResidencyController,
        native: M,
    ) -> Arc<Self> {
        Arc::new(Self {
            sources,
            state: Mutex::new(ResidencyState::new(controller, native)),
            changed: Condvar::new(),
        })
    }

    /// Retains the exact checkpoint source catalog.
    pub const fn sources(&self) -> &ResidencySources {
        &self.sources
    }

    /// Observes storage without polling native work or changing residency.
    pub fn observe(
        &self,
    ) -> Result<impl std::ops::Deref<Target = ResidencyState<M>> + '_, ResidencyStatePoisoned> {
        self.lock()
    }

    /// Observes the exact logical state and telemetry under one lock.
    pub fn snapshot(&self) -> Result<ResidencySnapshot, ResidencyStatePoisoned> {
        let state = self.lock()?;
        let control = state.controller();
        let ledger = control.ledger();
        let mut offload = ledger.telemetry();
        if let Some(retention) = control.conversion_retention() {
            offload = offload.with_parameter_conversion_retention(eredu_core::Observed::exact(
                vec![retention.report()],
                "residency conversion admission ledger",
            ));
        }
        Ok(ResidencySnapshot {
            initialized: ledger.initialized(),
            offload,
            units: ledger.unit_reports(),
            active_window: ledger.active_window().into_iter().collect(),
        })
    }

    /// Assembles logical residency accounting and checkpoint source diagnostics.
    pub fn report(&self) -> Result<ResidencyReport, M::Error>
    where
        M::Error: From<eredu_checkpoint::store::StoreError> + From<BackendFailure>,
    {
        let snapshot = self.snapshot().map_err(BackendFailure::from_error)?;
        let unit_sources = self
            .sources
            .unit_sources()
            .map(|(id, source)| {
                source
                    .source_diagnostics()
                    .map(|diagnostics| (id.clone(), diagnostics))
            })
            .collect::<Result<_, _>>()?;
        Ok(ResidencyReport::new(
            snapshot.initialized,
            snapshot.offload,
            snapshot.units,
            snapshot.active_window,
            self.sources.primary().source_diagnostics()?,
        )
        .with_unit_sources(unit_sources))
    }

    fn lock(&self) -> Result<MutexGuard<'_, ResidencyState<M>>, ResidencyStatePoisoned> {
        self.state.lock().map_err(|_| ResidencyStatePoisoned)
    }
}

impl<M: Send> ResidencyLeaseOwner for ResidencyOwner<M> {
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier) {
        if let Ok(mut state) = self.state.lock() {
            state.controller.ledger_mut().unpin(id, tier);
        }
    }
}

/// One pending publication owned by the exact native transfer, including its consumers.
///
/// Completion owners call settlement only after they have established a terminal
/// disposition and released any obligation to keep the original transfer live.
struct ResidencyTransferSettlement<M> {
    owner: Weak<ResidencyOwner<M>>,
    ids: Vec<OffloadUnitId>,
    tier: MemoryTier,
    generation: u64,
}

impl<M> ResidencyTransferTicket for ResidencyTransferSettlement<M>
where
    M: ResidencyStorage,
    M::Error: From<BackendFailure> + From<ResidencyLedgerError>,
{
    type Error = M::Error;

    /// Publishes completion or drops the copies invalidated by a terminal failure.
    ///
    /// Successive calls after successful settlement have no effect.
    fn settle(&mut self, succeeded: bool) -> Result<(), M::Error> {
        if self.generation == 0 {
            return Ok(());
        }
        if let Some(owner) = self.owner.upgrade() {
            let mut state = owner.lock().map_err(BackendFailure::from_error)?;
            settle_residency_transfer(
                &mut *state,
                &self.ids,
                self.tier,
                self.generation,
                succeeded,
            )?;
            owner.changed.notify_all();
        }
        self.generation = 0;
        Ok(())
    }
}

/// Resident storage and the portable pin that prevents eviction of its exact copy.
pub type ResidentLease<M> = ResidencyLease<<M as ResidencyStorage>::Copy>;

/// Native progress and completion retention for the shared residency driver.
pub trait ResidencyAccess {
    /// Error shared by storage access and acquisition.
    type Error: std::error::Error
        + Send
        + Sync
        + From<ResidencyLedgerError>
        + From<BackendFailure>
        + 'static;
    /// Transfer returned to the caller.
    type Transfer;
    /// Exact native storage and transfer primitives.
    type Storage: ResidencyStorage<Error = Self::Error>;

    /// Shared portable residency owner for this catalog.
    fn residency_owner(&self) -> &Arc<ResidencyOwner<Self::Storage>>;

    /// Progresses native deferred work and checks that further submission is safe.
    /// This runs with no residency lock held.
    fn progress_native_residency(&self) -> Result<(), Self::Error>;

    /// Retains leases and completion together for work that may outlive the caller.
    fn transfer(
        leases: Vec<ResidentLease<Self::Storage>>,
        submitted: Option<<Self::Storage as ResidencyStorage>::Transfer>,
    ) -> Self::Transfer;

    /// Enters serialized access after progressing native recovery.
    fn lock_storage(&self) -> Result<MutexGuard<'_, ResidencyState<Self::Storage>>, Self::Error> {
        self.progress_native_residency()?;
        Ok(self
            .residency_owner()
            .lock()
            .map_err(BackendFailure::from_error)?)
    }

    /// Observes a consistent logical snapshot without progressing native work.
    fn residency_snapshot(&self) -> Result<ResidencySnapshot, Self::Error> {
        Ok(self
            .residency_owner()
            .snapshot()
            .map_err(BackendFailure::from_error)?)
    }

    /// Exact permanently pinned units among the supplied IDs.
    /// Compact ranges are resolved only for the supplied IDs.
    fn pinned_unit_ids(
        &self,
        ids: &[OffloadUnitId],
    ) -> Result<BTreeSet<OffloadUnitId>, Self::Error> {
        let state = self
            .residency_owner()
            .observe()
            .map_err(BackendFailure::from_error)?;
        let plan = state.controller().ledger().plan();
        Ok(ids
            .iter()
            .filter(|id| {
                plan.unit(id).is_some_and(|unit| {
                    unit.policy() == eredu_core::residency::ResidencyPolicy::Pinned
                })
            })
            .cloned()
            .collect())
    }

    /// Tests whether the selected tier currently owns a logical copy.
    fn is_resident(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, Self::Error> {
        validate_target(tier, "is_resident")?;
        Ok(self
            .lock_storage()?
            .controller()
            .ledger()
            .is_resident(id, tier)?)
    }

    /// Retained pool limits before any parameter acquisition.
    fn offload_config(&self) -> Result<eredu_core::residency::OffloadConfig, Self::Error> {
        Ok(self.lock_storage()?.controller().ledger().plan().config())
    }

    /// Observes ledger unit reports without progressing native work.
    fn residency_unit_reports(&self) -> Result<Vec<UnitResidencyReport>, Self::Error> {
        Ok(self
            .residency_owner()
            .observe()
            .map_err(BackendFailure::from_error)?
            .controller()
            .ledger()
            .unit_reports())
    }

    /// Materializes the planned host and device copies and opens demand acquisition.
    fn initialize(&self) -> Result<(), Self::Error> {
        let mut state = self.lock_storage()?;
        let ledger = state.controller().ledger();
        if ledger.initialized() {
            return Ok(());
        }
        let assignments: Vec<_> = ledger
            .plan()
            .units()
            .iter()
            .filter(|unit| unit.tier() != MemoryTier::Disk)
            .map(|unit| (unit.id().clone(), unit.tier()))
            .collect();
        for (id, tier) in assignments {
            acquire_residency(
                &mut *state,
                self.residency_owner().sources(),
                std::slice::from_ref(&id),
                tier,
                false,
                true,
            )?;
        }
        state.controller_mut().ledger_mut().mark_initialized();
        Ok(())
    }

    /// Synchronously prepares one copy ahead of demand and records its outcome.
    fn prefetch(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<PrefetchOutcome, Self::Error> {
        validate_target(tier, "prefetch")?;
        let state = self.lock_storage()?;
        let (mut state, _) = settled_storage(self, state, std::iter::once(id), tier)?;
        prefetch_locked(&mut *state, self.residency_owner().sources(), id, tier)
    }

    /// Changes a named protected window without loading lookahead.
    fn protect_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<(), Self::Error> {
        let mut state = self.lock_storage()?;
        validate_target(tier, "protect_group_window")?;
        state
            .controller_mut()
            .protect_group_window(group, active, tier)?;
        Ok(())
    }

    /// Acquires one copy with one unit of demand.
    fn acquire(
        &self,
        id: &OffloadUnitId,
        tier: MemoryTier,
    ) -> Result<ResidentLease<Self::Storage>, Self::Error> {
        let (lease, _) = acquire_leases(self, &[(id.clone(), 1)], tier, false)?;
        Ok(lease.into_iter().next().expect("one requested lease"))
    }

    /// Acquires a complete batch before handing its leases to the caller.
    fn acquire_many_with_demand(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Vec<ResidentLease<Self::Storage>>, Self::Error> {
        acquire_leases(self, requests, tier, false).map(|(leases, _)| leases)
    }

    /// Submits a batch and returns leases owned through the exact transfer.
    fn acquire_many_with_transfer(
        &self,
        requests: &[(OffloadUnitId, u64)],
        tier: MemoryTier,
    ) -> Result<Self::Transfer, Self::Error> {
        let (leases, submitted) = acquire_leases(self, requests, tier, true)?;
        Ok(Self::transfer(leases, submitted))
    }
}

type Lease<A> = ResidentLease<<A as ResidencyAccess>::Storage>;
type Pending<A> = <<A as ResidencyAccess>::Storage as ResidencyStorage>::Transfer;

fn acquire_leases<A: ResidencyAccess + ?Sized>(
    access: &A,
    requests: &[(OffloadUnitId, u64)],
    tier: MemoryTier,
    submit: bool,
) -> Result<(Vec<Lease<A>>, Option<Pending<A>>), A::Error> {
    validate_target(tier, "acquire")?;
    let ids: Vec<_> = requests.iter().map(|(id, _)| id.clone()).collect();
    let state = access.lock_storage()?;
    state.controller().ledger().validate_batch(&ids, tier)?;
    let (mut state, missing) = settled_storage(access, state, ids.iter(), tier)?;
    let started = std::time::Instant::now();
    let result = acquire_residency(
        &mut *state,
        access.residency_owner().sources(),
        &ids,
        tier,
        submit,
        false,
    );
    if missing {
        state
            .controller_mut()
            .ledger_mut()
            .record_prefetch_stall(started.elapsed());
    }
    let submitted = result?.map(|pending| {
        A::Storage::retain_settlement(
            &pending.transfer,
            Box::new(ResidencyTransferSettlement {
                owner: Arc::downgrade(access.residency_owner()),
                ids: pending.ids,
                tier,
                generation: pending.generation,
            }),
        );
        pending.transfer
    });

    // Gather all handles first. A failed lookup or pin admission leaves no
    // partially constructed leases that could release pins under the lock.
    let storage = ids
        .iter()
        .map(|id| state.native().resident_storage(id, tier))
        .collect::<Result<Vec<_>, _>>()?;
    let owner = Arc::downgrade(access.residency_owner());
    let owner: Weak<dyn ResidencyLeaseOwner> = owner;
    state
        .controller_mut()
        .ledger_mut()
        .pin_batch(requests, tier)?;
    let leases = ids
        .into_iter()
        .zip(storage)
        .map(|(id, copy)| ResidencyLease::new(id, tier, copy, owner.clone()))
        .collect();
    Ok((leases, submitted))
}

fn settled_storage<'a, 'r, A: ResidencyAccess + ?Sized>(
    access: &'a A,
    mut state: MutexGuard<'a, ResidencyState<A::Storage>>,
    ids: impl Iterator<Item = &'r OffloadUnitId> + Clone,
    tier: MemoryTier,
) -> Result<(MutexGuard<'a, ResidencyState<A::Storage>>, bool), A::Error> {
    loop {
        let ledger = state.controller().ledger();
        ledger.require_initialized()?;
        let mut waiting = false;
        let mut missing = false;
        for id in ids.clone() {
            match ledger.copy_status(id, tier)? {
                Some(copy) => waiting |= copy.in_flight().is_some(),
                None => missing = true,
            }
        }
        if !waiting {
            return Ok((state, missing));
        }
        let (guard, _) = access
            .residency_owner()
            .changed
            .wait_timeout(state, std::time::Duration::from_millis(25))
            .map_err(|_| BackendFailure::from_error(ResidencyStatePoisoned))?;
        drop(guard);
        state = access.lock_storage()?;
    }
}

fn validate_target(tier: MemoryTier, operation: &'static str) -> Result<(), ResidencyLedgerError> {
    if tier == MemoryTier::Disk {
        return Err(ResidencyLedgerError::InvalidTargetTier { operation });
    }
    Ok(())
}

fn prefetch_locked<M: ResidencyStorage>(
    state: &mut ResidencyState<M>,
    sources: &ResidencySources,
    id: &OffloadUnitId,
    tier: MemoryTier,
) -> Result<PrefetchOutcome, M::Error>
where
    M::Error: From<BackendFailure> + From<ResidencyLedgerError>,
{
    let outcome = state.controller_mut().begin_prefetch(id, tier)?;
    acquire_residency(state, sources, std::slice::from_ref(id), tier, false, false)?;
    Ok(outcome)
}

impl<A> ResidencyWindowManager for A
where
    A: ResidencyAccess,
{
    type Error = A::Error;

    fn prepare_window(
        &self,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
        self.prepare_group_window("default", active, upcoming, tier)
    }

    fn prepare_group_window(
        &self,
        group: &str,
        active: &[OffloadUnitId],
        upcoming: &[OffloadUnitId],
        tier: MemoryTier,
    ) -> Result<Vec<(OffloadUnitId, PrefetchOutcome)>, Self::Error> {
        validate_target(tier, "prepare_group_window")?;
        let state = self.lock_storage()?;
        let (mut state, _) = settled_storage(self, state, active.iter().chain(upcoming), tier)?;
        let selected = state
            .controller_mut()
            .commit_group_window(group, active, upcoming, tier)?;
        selected
            .into_iter()
            .map(|id| {
                prefetch_locked(&mut *state, self.residency_owner().sources(), &id, tier)
                    .map(|outcome| (id, outcome))
            })
            .collect()
    }

    fn evict(&self, id: &OffloadUnitId, tier: MemoryTier) -> Result<bool, Self::Error> {
        validate_target(tier, "evict")?;
        let mut state = self.lock_storage()?;
        let Some(copy) = state.controller_mut().ledger_mut().evict(id, tier)? else {
            return Ok(false);
        };
        release_copies(&mut *state, std::slice::from_ref(&copy))?;
        Ok(true)
    }

    fn unit_reports(&self) -> Result<Vec<UnitResidencyReport>, Self::Error> {
        self.residency_unit_reports()
    }
}

/// Settles one exact transfer and drops native storage invalidated by its result.
fn settle_residency_transfer<M: ResidencyStorage>(
    mechanisms: &mut ResidencyState<M>,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    generation: u64,
    succeeded: bool,
) -> Result<(), M::Error>
where
    M::Error: From<ResidencyLedgerError>,
{
    let removed = mechanisms
        .controller_mut()
        .resolve_transfer(ids, tier, generation, succeeded)?;
    release_copies(mechanisms, &removed)
}

fn release_copies<M: ResidencyStorage>(
    mechanisms: &mut ResidencyState<M>,
    copies: &[EvictedResidencyCopy],
) -> Result<(), M::Error>
where
    M::Error: From<ResidencyLedgerError>,
{
    for copy in copies {
        mechanisms.native_mut().release_copy(&copy.id, copy.tier)?;
        if mechanisms
            .controller_mut()
            .retire_unit_if_unused(&copy.id)?
        {
            mechanisms.native_mut().retire_storage(&copy.id);
        }
    }
    Ok(())
}

struct PendingResidencyTransfer<T> {
    transfer: T,
    ids: Vec<OffloadUnitId>,
    generation: u64,
}

/// Acquires an exact batch under the portable residency ledger.
fn acquire_residency<M: ResidencyStorage>(
    mechanisms: &mut ResidencyState<M>,
    sources: &ResidencySources,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
    submit: bool,
    initializing: bool,
) -> Result<Option<PendingResidencyTransfer<M::Transfer>>, M::Error>
where
    M::Error: From<BackendFailure> + From<ResidencyLedgerError>,
{
    let control = mechanisms.controller_mut();
    control.ledger().validate_batch(ids, tier)?;
    if !initializing {
        control.ledger().require_initialized()?;
    }
    let new: Vec<_> = ids
        .iter()
        .filter(|id| control.unit(id).is_none())
        .cloned()
        .collect();
    control
        .prepare_units(ids)
        .map_err(BackendFailure::from_error)?;
    let result = acquire_prepared(mechanisms, sources, ids, &new, tier, submit, initializing);
    // Native transfers with outstanding work remain charged by the ledger.
    for id in ids {
        if mechanisms.controller_mut().retire_unit_if_unused(id)? {
            mechanisms.native_mut().retire_storage(id);
        }
    }
    result
}

fn acquire_prepared<M: ResidencyStorage>(
    mechanisms: &mut ResidencyState<M>,
    sources: &ResidencySources,
    ids: &[OffloadUnitId],
    new: &[OffloadUnitId],
    tier: MemoryTier,
    submit: bool,
    initializing: bool,
) -> Result<Option<PendingResidencyTransfer<M::Transfer>>, M::Error>
where
    M::Error: From<BackendFailure> + From<ResidencyLedgerError>,
{
    for id in new {
        let unit = mechanisms.controller().unit(id).expect("prepared unit");
        preflight_bindings::<M::Backend>(sources.source(id), unit.bindings()).map_err(|error| {
            BackendFailure::from_error(PreparedResidencyError::BindingPreflight(error.to_string()))
        })?;
        mechanisms.native_mut().prepare_storage(id);
    }

    // A batch cannot use an alias owner until its storage has been published.
    let mut owners = BTreeSet::new();
    let control = mechanisms.controller();
    for id in ids {
        let unit = control.unit(id).expect("prepared unit");
        for binding in unit.bindings() {
            if let Some((owner, _)) = control.binding_owner(id, binding) {
                if owner != id {
                    owners.insert(owner.clone());
                }
            }
        }
    }
    for owner in owners {
        acquire_residency(
            mechanisms,
            sources,
            std::slice::from_ref(&owner),
            tier,
            false,
            initializing,
        )?;
        let control = mechanisms.controller_mut();
        if !control.alias_owner_pins.contains(&(owner.clone(), tier)) {
            control.ledger_mut().pin(&owner, tier, 1)?;
            control.alias_owner_pins.insert((owner, tier));
        }
    }

    let control = mechanisms.controller_mut();
    let acquisition = if initializing {
        control.plan_initialization_acquisition(ids, tier)?
    } else {
        control.plan_acquisition(ids, tier)?
    };
    if acquisition.is_hit() {
        control.touch_acquisition_hits(&acquisition, tier)?;
        return Ok(None);
    }

    let started = std::time::Instant::now();
    let result = (|| {
        let requests = acquisition
            .missing_ids()
            .map(|id| {
                let control = mechanisms.controller();
                let logical_bytes = control.ledger().spec(id)?.bytes();
                let unit = control.unit(id).expect("prepared unit");
                let bindings = unit.bindings();
                let reserved_capacity =
                    mechanisms
                        .native()
                        .capacity(bindings, logical_bytes, tier)?;
                let aliases = bindings
                    .iter()
                    .filter_map(|binding| {
                        let (owner, value) = control.binding_owner(id, binding)?;
                        Some(ResidentBindingAlias {
                            target: binding.name().to_owned(),
                            owner: owner.clone(),
                            binding: value.name().to_owned(),
                        })
                    })
                    .collect();
                Ok(ResidencyCopyRequest {
                    id,
                    source: sources.source(id),
                    bindings: bindings.to_vec(),
                    aliases,
                    logical_bytes,
                    reserved_capacity,
                })
            })
            .collect::<Result<Vec<_>, M::Error>>()?;
        let reservations: Vec<_> = requests
            .iter()
            .map(|request| (request.id.clone(), request.reserved_capacity))
            .collect();
        let evicted =
            mechanisms
                .controller_mut()
                .reserve_acquisition(&acquisition, &reservations, tier)?;
        release_copies(mechanisms, &evicted)?;
        let RealizedResidency { copies, transfer } =
            super::materialization::realize(mechanisms.native_mut(), &requests, tier, submit)?;
        let generation = if transfer.is_some() {
            Some(
                mechanisms
                    .controller_mut()
                    .ledger_mut()
                    .next_transfer_generation()?,
            )
        } else {
            None
        };
        for copy in copies {
            mechanisms.controller_mut().publish_acquisition_copy(
                &copy.id,
                tier,
                copy.capacity,
                copy.bytes,
                generation,
                copy.direction,
                started.elapsed(),
            )?;
            mechanisms.native_mut().install_copy(copy.id, copy.storage);
        }
        mechanisms
            .controller_mut()
            .touch_acquisition_hits(&acquisition, tier)?;
        Ok(transfer
            .zip(generation)
            .map(|(transfer, generation)| PendingResidencyTransfer {
                transfer,
                ids: acquisition.missing_ids().cloned().collect(),
                generation,
            }))
    })();
    if result.is_err() {
        mechanisms
            .controller_mut()
            .rollback_acquisition(&acquisition, tier)?;
    }
    result
}
