//! Tier transitions, eviction, asynchronous retirement, and rebalancing.

use super::*;

pub(super) enum HostDemotionProgress<M: LiveCacheBackend> {
    Retry,
    Freed,
    Pending(DiskTicket<M>),
}

pub(super) enum PendingCacheOperation<M: LiveCacheBackend> {
    Disk(DiskTicket<M>),
    HostDemotion(HostDemotionTicket<M>),
}

impl<M: LiveCacheBackend> CacheResidencyManager<M> {
    pub(super) fn retire_tickets(
        &self,
        tickets: &[PendingCacheOperation<M>],
    ) -> Result<(), CacheResidencyError> {
        let mut first_error = None;
        for ticket in tickets {
            match ticket {
                PendingCacheOperation::Disk(ticket) => {
                    let result = ticket.wait_for_task_resources();
                    if result.is_ok() {
                        if let Some(worker) = &self.inner.disk_worker {
                            worker.retire(ticket);
                        }
                    }
                    let mut state = self.lock()?;
                    state.retiring_disk_reads.remove(&ticket.key);
                    update_report_totals(&mut state);
                    drop(state);
                    if let Err(error) = result {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
                PendingCacheOperation::HostDemotion(ticket) => {
                    let result = self.finish_device_demotion(ticket);
                    let mut state = self.lock()?;
                    state.retiring_host_demotions.remove(&ticket.operation_id);
                    update_report_totals(&mut state);
                    drop(state);
                    if let Err(error) = result {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) fn begin_host_demotion(
        &self,
        id: &CacheBlockId,
    ) -> Result<HostDemotionProgress<M>, CacheResidencyError> {
        let mut state = self.lock()?;
        let Some(record) = state.blocks.get(id) else {
            return Ok(HostDemotionProgress::Retry);
        };
        if record.tier() != CacheTier::Host
            || state.lifecycle.is_leased(id)?
            || record.pending_disk().is_some()
        {
            return Ok(HostDemotionProgress::Retry);
        }

        // Persistent prompt-cache blocks and completed live-cache writes can be
        // released immediately; they do not require live writeback to be enabled.
        if record.disk().is_some() {
            let record = state.blocks.get_mut(id).expect("host block exists");
            record.physical.release_host_to_disk()?;
            state.telemetry.report.disk_demotions += 1;
            state.layer_activity_mut(id.global_layer).disk_demotions += 1;
            update_report_totals(&mut state);
            return Ok(HostDemotionProgress::Freed);
        }

        let record = state.blocks.get(id).expect("host block exists");
        let published_bytes = state
            .blocks
            .values()
            .filter(|record| record.disk().is_some_and(|location| !location.persistent))
            .map(|record| record.bytes)
            .fold(0u64, u64::saturating_add);
        let pending_bytes = state
            .host_write_reservations
            .iter()
            .filter(|(key, _)| {
                !state.blocks.get(&key.id).is_some_and(|record| {
                    record.disk().is_some_and(|location| !location.persistent)
                })
            })
            .map(|(_, reservation)| reservation.logical_bytes)
            .fold(0u64, u64::saturating_add);
        let directory = crate::cache::scheduling::admit_cache_spill(
            self.options(),
            state.telemetry.report.current_host_bytes,
            published_bytes,
            pending_bytes,
            record.bytes,
            id.global_layer,
        )
        .map_err(|pressure| reject_cache_pressure(&mut state, &pressure))?;
        let worker = self.inner.disk_worker.as_ref().ok_or_else(|| {
            CacheResidencyError::Runtime("live cache disk worker is unavailable".into())
        })?;
        let record = state.blocks.get(id).expect("host block exists");
        let block = record
            .host_block()
            .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
            .clone();
        let host_capacity = block.capacity()?;
        let pool_admission = state.pool.reserve(CachePoolUsage {
            transfer_in_flight_bytes: host_capacity,
            disk_bytes: record.bytes,
            ..CachePoolUsage::default()
        })?;
        let submission = worker.prepare_write(
            state.generation,
            directory,
            id,
            &block,
            Arc::downgrade(&self.inner.state),
        )?;
        let ticket = submission.ticket.clone();
        let record_bytes = record.bytes;
        if let Some(reservation_id) = submission.write_reservation_id {
            state.host_write_reservations.insert(
                ticket.key.clone(),
                HostWriteReservation {
                    reservation_id,
                    logical_bytes: record_bytes,
                    host_capacity,
                    ticket: ticket.clone(),
                },
            );
        }
        let record = state.blocks.get_mut(id).expect("host block exists");
        record.physical.begin_write(CacheIoEntry {
            ticket: ticket.clone(),
            reserved_host_bytes: None,
        })?;
        update_report_totals(&mut state);
        drop(pool_admission);
        drop(state);

        let enqueue_started = Instant::now();
        let outcome = match submission.enqueue() {
            Ok(outcome) => outcome,
            Err(error) => {
                let mut state = self.lock()?;
                if let Some(record) = state.blocks.get_mut(&ticket.key.id) {
                    record.physical.fail_io_if_matches(&ticket.key);
                }
                state.telemetry.report.failures += 1;
                state
                    .layer_activity_mut(ticket.key.id.global_layer)
                    .failures += 1;
                update_report_totals(&mut state);
                drop(state);
                worker.retire(&ticket);
                return Err(error);
            }
        };
        let enqueue_wait = enqueue_started.elapsed();
        let mut state = self.lock()?;
        if outcome.joined {
            state.telemetry.report.in_flight_waits += 1;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .in_flight_waits += 1;
        }
        state.telemetry.report.queue_peak_occupancy = state
            .telemetry
            .report
            .queue_peak_occupancy
            .max(outcome.peak_occupancy);
        state.telemetry.report.queue_backpressure += u64::from(outcome.backpressure);
        if outcome.backpressure {
            state.telemetry.report.transfer_wait += enqueue_wait;
            state
                .layer_activity_mut(ticket.key.id.global_layer)
                .transfer_wait += enqueue_wait;
        }
        update_report_totals(&mut state);
        Ok(HostDemotionProgress::Pending(ticket))
    }

    pub(super) fn wait_for_host_release(
        &self,
        ticket: &DiskTicket<M>,
    ) -> Result<(), CacheResidencyError> {
        let started = Instant::now();
        let result = ticket.wait();
        ticket.wait_for_task_resources()?;
        let elapsed = started.elapsed();
        let mut state = self.lock()?;
        state.telemetry.report.in_flight_waits += 1;
        state.telemetry.report.transfer_wait += elapsed;
        let activity = state.layer_activity_mut(ticket.key.id.global_layer);
        activity.in_flight_waits += 1;
        activity.transfer_wait += elapsed;
        if result.is_err() {
            // The write commit records its error for asynchronous callers. This
            // caller observed it directly, so do not surface the same failure twice.
            state.background_disk_error = None;
        }
        update_report_totals(&mut state);
        drop(state);
        match result {
            Ok(DiskResult::Write) => Ok(()),
            Ok(_) => Err(CacheResidencyError::Runtime(
                "cache disk worker returned an unexpected write result".into(),
            )),
            Err(error) => Err(error),
        }
    }

    pub(super) fn begin_device_demotion(
        &self,
        id: &CacheBlockId,
    ) -> Result<HostDemotionTicket<M>, CacheResidencyError> {
        let mut state = self.lock()?;
        let record = state
            .blocks
            .get(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
        if state.lifecycle.is_leased(id)? {
            return Err(CacheLifecycleError::BlockLeased(id.clone()).into());
        }
        if record.physical.phase() != CacheStoragePhase::Device
            || record.physical.backing().is_some()
        {
            return Err(CacheResidencyError::MissingResidentArrays(id.clone()));
        }
        let arrays = record
            .physical
            .device_resource()
            .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?
            .clone();
        let reserved_host_bytes = host_cache_capacity_upper_bound::<M>(&arrays)?;
        let required_host_bytes = state
            .telemetry
            .report
            .current_host_bytes
            .saturating_add(reserved_host_bytes);
        if required_host_bytes > self.options().host_budget_bytes() {
            return Err(CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Host,
                required: required_host_bytes,
                budget: self.options().host_budget_bytes(),
            });
        }
        let pool_admission = state.pool.reserve(CachePoolUsage {
            host_bytes: reserved_host_bytes,
            transfer_in_flight_bytes: reserved_host_bytes,
            ..CachePoolUsage::default()
        })?;
        let device = state.transfer_device.clone().unwrap_or_else(M::host_device);
        let ticket = self.inner.host_demotion_worker.submit(
            id,
            arrays.clone(),
            device,
            reserved_host_bytes,
        )?;
        let record = state.blocks.get_mut(id).expect("demotion block exists");
        record.physical.begin_host_demotion(ticket.clone())?;
        update_report_totals(&mut state);
        drop(pool_admission);
        Ok(ticket)
    }

    pub(super) fn finish_device_demotion(
        &self,
        ticket: &HostDemotionTicket<M>,
    ) -> Result<(), CacheResidencyError> {
        let started = Instant::now();
        let result = ticket.wait();
        let elapsed = started.elapsed();
        let mut state = self.lock()?;
        let matches = state
            .blocks
            .get(&ticket.id)
            .and_then(CacheBlockRecord::host_demotion_ticket)
            .is_some_and(|pending| pending.operation_id == ticket.operation_id);
        if !matches {
            return Ok(());
        }
        state.telemetry.report.in_flight_waits += 1;
        state.telemetry.report.transfer_wait += elapsed;
        let activity = state.layer_activity_mut(ticket.id.global_layer);
        activity.in_flight_waits += 1;
        activity.transfer_wait += elapsed;
        match result {
            Ok(block) => {
                let capacity = block.capacity()?;
                if capacity > ticket.reserved_host_bytes {
                    let record = state
                        .blocks
                        .get_mut(&ticket.id)
                        .expect("demotion block exists");
                    record.physical.fail_host_demotion(ticket.operation_id)?;
                    state.telemetry.report.failures += 1;
                    state.layer_activity_mut(ticket.id.global_layer).failures += 1;
                    update_report_totals(&mut state);
                    return Err(CacheResidencyError::Runtime(format!(
                        "host cache allocation used {capacity} bytes, exceeding reserved upper bound {}",
                        ticket.reserved_host_bytes
                    )));
                }
                let bytes = state
                    .blocks
                    .get(&ticket.id)
                    .expect("demotion block exists")
                    .bytes;
                let record = state
                    .blocks
                    .get_mut(&ticket.id)
                    .expect("demotion block exists");
                record
                    .physical
                    .finish_host_demotion(ticket.operation_id, block)?;
                state.telemetry.report.host_demotions += 1;
                state.telemetry.report.transfer_bytes += bytes;
                let activity = state.layer_activity_mut(ticket.id.global_layer);
                activity.host_demotions += 1;
                activity.transfer_bytes += bytes;
                update_report_totals(&mut state);
                Ok(())
            }
            Err(error) => {
                let record = state
                    .blocks
                    .get_mut(&ticket.id)
                    .expect("demotion block exists");
                record.physical.fail_host_demotion(ticket.operation_id)?;
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(ticket.id.global_layer).failures += 1;
                update_report_totals(&mut state);
                Err(error)
            }
        }
    }

    pub(super) fn rebalance(
        &self,
        required: Option<&CacheBlockId>,
        allow_recent_eviction: bool,
    ) -> Result<(), CacheResidencyError> {
        loop {
            let mut state = self.lock()?;
            if let Some(error) = state.background_disk_error.take() {
                return Err(error.into());
            }
            update_report_totals(&mut state);
            use crate::cache::scheduling::{CacheRebalance, CacheRebalanceAction};
            let action = CacheRebalance {
                lifecycle: &state.lifecycle,
                options: self.options(),
                required,
                allow_recent_eviction,
                device_bytes: state.telemetry.report.current_device_bytes,
                host_bytes: state.telemetry.report.current_host_bytes,
                pool: state.pool.report()?,
            }
            .next(
                cache_pages(&state),
                |id| {
                    state.blocks[id]
                        .physical
                        .device_resource()
                        .map(host_cache_capacity_upper_bound::<M>)
                        .transpose()?
                        .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))
                },
                || state.pending_spill().is_some(),
            )?;
            match action {
                CacheRebalanceAction::Idle => return Ok(()),
                CacheRebalanceAction::SettleDemotion => {
                    let ticket = state
                        .blocks
                        .values()
                        .find_map(CacheBlockRecord::host_demotion_ticket)
                        .expect("scheduled demotion owns its completion")
                        .clone();
                    drop(state);
                    self.finish_device_demotion(&ticket)?;
                }
                CacheRebalanceAction::ReleaseDevice(id) => {
                    let record = state.blocks.get_mut(&id).expect("scheduled page exists");
                    record.physical.release_device_to_disk()?;
                    state.telemetry.report.disk_demotions += 1;
                    state.layer_activity_mut(id.global_layer).disk_demotions += 1;
                }
                CacheRebalanceAction::DemoteDevice(id) => {
                    drop(state);
                    self.begin_device_demotion(&id)?;
                }
                CacheRebalanceAction::ReleaseHost { id, urgent } => {
                    drop(state);
                    let release = self.begin_host_demotion(&id)?;
                    if !urgent {
                        return Ok(());
                    }
                    if let HostDemotionProgress::Pending(ticket) = release {
                        self.wait_for_host_release(&ticket)?;
                    }
                }
                CacheRebalanceAction::SettleSpill => {
                    let ticket = state
                        .pending_spill()
                        .expect("scheduled spill owns its completion")
                        .clone();
                    drop(state);
                    self.wait_for_host_release(&ticket)?;
                }
                CacheRebalanceAction::Reject(pressure) => {
                    return Err(reject_cache_pressure(&mut state, &pressure));
                }
            }
        }
    }
}

fn cache_pages<M: LiveCacheBackend>(
    state: &CacheManagerState<M>,
) -> impl Iterator<Item = crate::cache::scheduling::CachePage<'_>> + Clone {
    state
        .blocks
        .values()
        .map(|record| crate::cache::scheduling::CachePage {
            id: record.physical.id(),
            phase: record.physical.phase(),
            backed: record.disk().is_some(),
        })
}

pub(super) fn eviction_candidate<M: LiveCacheBackend>(
    state: &CacheManagerState<M>,
    tier: CacheTier,
    required: Option<&CacheBlockId>,
    recent_per_layer: usize,
    policy: CacheEvictionPolicy,
) -> Option<CacheBlockId> {
    crate::cache::scheduling::select_cache_victim(
        &state.lifecycle,
        cache_pages(state),
        tier,
        required,
        recent_per_layer,
        policy,
    )
}

fn reject_cache_pressure<M: LiveCacheBackend>(
    state: &mut CacheManagerState<M>,
    pressure: &crate::cache::scheduling::CachePressure,
) -> CacheResidencyError {
    state.telemetry.report.failures += 1;
    if let Some(layer) = pressure.layer {
        state.layer_activity_mut(layer).failures += 1;
    } else {
        state.telemetry.unassigned_activity_mut().failures += 1;
    }
    if pressure.shared_pool {
        let resource = match pressure.tier {
            CacheTier::Device => CachePoolResource::Device,
            CacheTier::Host => CachePoolResource::Host,
            CacheTier::Disk => CachePoolResource::Disk,
        };
        CachePoolError::BudgetExceeded {
            resource,
            required: pressure.required,
            budget: pressure.budget,
        }
        .into()
    } else if pressure.needs_live_disk {
        CacheResidencyError::LiveDiskRequired {
            required: pressure.required,
            budget: pressure.budget,
        }
    } else {
        CacheResidencyError::BudgetExceeded {
            tier: pressure.tier,
            required: pressure.required,
            budget: pressure.budget,
        }
    }
}
