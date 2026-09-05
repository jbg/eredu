//! Residency transfer ownership, storage lifecycle, and acquisition.

use super::*;

/// Shared lease and transfer owner for a residency manager.
pub struct ManagerInner {
    pub(super) store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    pub(super) state: Mutex<ManagerState>,
    pub(super) changed: Condvar,
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
            // source lease permanently if backend state is unknowable.
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

pub(super) struct ManagerState {
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
}

pub(super) struct SubmittedResidentTransfer {
    pub(super) event: Event,
    pub(super) retained: ResidentTransferResources,
    pub(super) ids: Vec<OffloadUnitId>,
    pub(super) generation: u64,
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
                        if is_shard_cache_capacity_error(&error)
                            && prepared.iter().any(
                                |(_, item): &(OffloadUnitId, PreparedResidentArrays)| {
                                    !item.pending_sources.is_empty()
                                },
                            ) =>
                    {
                        // Earlier units in this batch can pin the only cached
                        // shard while a later cross-shard entry is prepared.
                        // Their output arrays are complete evaluation roots, so
                        // detach those leases and retry the current unit.
                        eval(prepared.iter().flat_map(|(_, item)| item.arrays.values())).map_err(
                            |source| ResidencyError::Mlx {
                                id: internal_id(),
                                operation: "shard-cache-capacity batch evaluation",
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
