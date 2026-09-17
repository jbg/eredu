//! Tier transitions, eviction, asynchronous retirement, and rebalancing.

use super::*;

pub(super) enum HostDemotionProgress {
    Retry,
    Freed,
    Pending(DiskTicket),
}

pub(super) enum PendingCacheOperation {
    Disk(DiskTicket),
    HostDemotion(HostDemotionTicket),
}

impl CacheResidencyManager {
    pub(super) fn retire_tickets(
        &self,
        tickets: &[PendingCacheOperation],
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
    ) -> Result<HostDemotionProgress, CacheResidencyError> {
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

        let (directory, budget_bytes) = match self.options().live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => {
                state.telemetry.report.failures += 1;
                state.layer_activity_mut(id.global_layer).failures += 1;
                return Err(CacheResidencyError::LiveDiskRequired {
                    required: state.telemetry.report.current_host_bytes,
                    budget: self.options().host_budget_bytes(),
                });
            }
            LiveCacheDiskPolicy::Enabled {
                directory,
                budget_bytes,
                ..
            } => (directory.clone(), *budget_bytes),
        };
        let worker = self.inner.disk_worker.as_ref().ok_or_else(|| {
            CacheResidencyError::Runtime("live cache disk worker is unavailable".into())
        })?;
        let record = state.blocks.get(id).expect("host block exists");
        let live_disk_bytes = live_disk_bytes(&state).unwrap_or(u64::MAX);
        let projected = live_disk_bytes.saturating_add(record.bytes);
        if projected > budget_bytes {
            state.telemetry.report.failures += 1;
            state.layer_activity_mut(id.global_layer).failures += 1;
            return Err(CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Disk,
                required: projected,
                budget: budget_bytes,
            });
        }
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
            &directory,
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
                    global_layer: id.global_layer,
                    logical_bytes: record_bytes,
                    host_capacity,
                    ticket: ticket.clone(),
                    prepared: None,
                },
            );
        }
        let record = state.blocks.get_mut(id).expect("host block exists");
        record.physical.begin_write(MlxCacheIoOperation {
            ticket: ticket.clone(),
            reserved_host_bytes: None,
            prepared_read: None,
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
        ticket: &DiskTicket,
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
            Ok(DiskResult::Write(_)) => Ok(()),
            Ok(_) => Err(CacheResidencyError::Runtime(
                "cache disk worker returned an unexpected write result".into(),
            )),
            Err(error) => Err(error),
        }
    }

    pub(super) fn begin_device_demotion(
        &self,
        id: &CacheBlockId,
    ) -> Result<HostDemotionTicket, CacheResidencyError> {
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
        let reserved_host_bytes = host_cache_capacity_upper_bound(&arrays)?;
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
        let device = state.transfer_device.unwrap_or(CacheTransferDevice::CPU);
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
        ticket: &HostDemotionTicket,
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
                return Err(CacheResidencyError::Runtime(format!(
                    "background cache disk write failed: {error}"
                )));
            }
            update_report_totals(&mut state);
            let budget = budget_snapshot(&state, self.options())?;
            let pool_report = budget.pool;
            let pool_device_over = budget.pool_device_over;
            let local_device_over = budget.local_device_over;
            if local_device_over || pool_device_over {
                let pending_demotion = state
                    .blocks
                    .values()
                    .find_map(CacheBlockRecord::host_demotion_ticket)
                    .cloned();
                if let Some(ticket) = pending_demotion {
                    drop(state);
                    self.finish_device_demotion(&ticket)?;
                    continue;
                }
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Device,
                    required,
                    self.options().recent_device_blocks(),
                    self.options().eviction_policy(),
                )
                .or_else(|| {
                    // Recent protection remains strict for mutation capacity,
                    // but may yield to an existing block demanded by attention.
                    if allow_recent_eviction {
                        eviction_candidate(
                            &state,
                            CacheTier::Device,
                            required,
                            0,
                            self.options().eviction_policy(),
                        )
                    } else {
                        None
                    }
                });
                let Some(id) = candidate else {
                    state.telemetry.report.failures += 1;
                    if let Some(required) = required {
                        state.layer_activity_mut(required.global_layer).failures += 1;
                    } else {
                        state.telemetry.unassigned_activity_mut().failures += 1;
                    }
                    return Err(if pool_device_over && !local_device_over {
                        CachePoolError::BudgetExceeded {
                            resource: CachePoolResource::Device,
                            required: pool_report.current_device_bytes,
                            budget: pool_report.limits.device_bytes(),
                        }
                        .into()
                    } else {
                        CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Device,
                            required: state.telemetry.report.current_device_bytes,
                            budget: self.options().device_budget_bytes(),
                        }
                    });
                };
                if state
                    .blocks
                    .get(&id)
                    .and_then(CacheBlockRecord::disk)
                    .is_some()
                {
                    let record = state.blocks.get_mut(&id).expect("candidate exists");
                    record.physical.release_device_to_disk()?;
                    state.telemetry.report.disk_demotions += 1;
                    state.layer_activity_mut(id.global_layer).disk_demotions += 1;
                    continue;
                }
                let candidate_host_bytes = state
                    .blocks
                    .get(&id)
                    .expect("candidate exists")
                    .physical
                    .device_resource()
                    .map(host_cache_capacity_upper_bound)
                    .transpose()?
                    .ok_or_else(|| CacheResidencyError::MissingResidentArrays(id.clone()))?;
                let required_host_bytes = state
                    .telemetry
                    .report
                    .current_host_bytes
                    .saturating_add(candidate_host_bytes);
                if required_host_bytes > self.options().host_budget_bytes() {
                    if candidate_host_bytes > self.options().host_budget_bytes() {
                        state.telemetry.report.failures += 1;
                        state.layer_activity_mut(id.global_layer).failures += 1;
                        return Err(CacheResidencyError::BudgetExceeded {
                            tier: CacheTier::Host,
                            required: candidate_host_bytes,
                            budget: self.options().host_budget_bytes(),
                        });
                    }
                    let host_candidate = eviction_candidate(
                        &state,
                        CacheTier::Host,
                        required,
                        0,
                        self.options().eviction_policy(),
                    );
                    let pending = state
                        .host_write_reservations
                        .values()
                        .next()
                        .map(|reservation| reservation.ticket.clone())
                        .or_else(|| {
                            state.blocks.values().find_map(|record| {
                                record
                                    .pending_disk()
                                    .filter(|pending| {
                                        pending.ticket.key.kind == CacheIoOperationKind::Write
                                    })
                                    .map(|pending| pending.ticket.clone())
                            })
                        });
                    drop(state);
                    if let Some(id) = host_candidate {
                        match self.begin_host_demotion(&id)? {
                            HostDemotionProgress::Retry | HostDemotionProgress::Freed => continue,
                            HostDemotionProgress::Pending(ticket) => {
                                self.wait_for_host_release(&ticket)?;
                                continue;
                            }
                        }
                    }
                    if let Some(ticket) = pending {
                        self.wait_for_host_release(&ticket)?;
                        continue;
                    }
                    let mut state = self.lock()?;
                    state.telemetry.report.failures += 1;
                    state.layer_activity_mut(id.global_layer).failures += 1;
                    return Err(match self.options().live_disk_policy() {
                        LiveCacheDiskPolicy::Disabled => CacheResidencyError::LiveDiskRequired {
                            required: required_host_bytes,
                            budget: self.options().host_budget_bytes(),
                        },
                        LiveCacheDiskPolicy::Enabled { .. } => {
                            CacheResidencyError::BudgetExceeded {
                                tier: CacheTier::Host,
                                required: required_host_bytes,
                                budget: self.options().host_budget_bytes(),
                            }
                        }
                    });
                }
                drop(state);
                self.begin_device_demotion(&id)?;
                continue;
            }

            let budget = budget_snapshot(&state, self.options())?;
            let pool_report = budget.pool;
            let pool_host_over = budget.pool_host_over;
            let local_host_over = budget.local_host_over;
            if local_host_over || pool_host_over {
                let candidate = eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options().eviction_policy(),
                );
                let pending = state
                    .host_write_reservations
                    .values()
                    .next()
                    .map(|reservation| reservation.ticket.clone())
                    .or_else(|| {
                        state.blocks.values().find_map(|record| {
                            record
                                .pending_disk()
                                .filter(|pending| {
                                    pending.ticket.key.kind == CacheIoOperationKind::Write
                                })
                                .map(|pending| pending.ticket.clone())
                        })
                    });
                let required_host_bytes = state.telemetry.report.current_host_bytes;
                drop(state);
                if let Some(id) = candidate {
                    match self.begin_host_demotion(&id)? {
                        HostDemotionProgress::Retry | HostDemotionProgress::Freed => continue,
                        HostDemotionProgress::Pending(ticket) => {
                            self.wait_for_host_release(&ticket)?;
                            continue;
                        }
                    }
                }
                if let Some(ticket) = pending {
                    self.wait_for_host_release(&ticket)?;
                    continue;
                }
                let mut state = self.lock()?;
                state.telemetry.report.failures += 1;
                if let Some(required) = required {
                    state.layer_activity_mut(required.global_layer).failures += 1;
                } else {
                    state.telemetry.unassigned_activity_mut().failures += 1;
                }
                return Err(if pool_host_over && !local_host_over {
                    CachePoolError::BudgetExceeded {
                        resource: CachePoolResource::Host,
                        required: pool_report.current_host_bytes,
                        budget: pool_report.limits.host_bytes(),
                    }
                    .into()
                } else {
                    CacheResidencyError::BudgetExceeded {
                        tier: CacheTier::Host,
                        required: required_host_bytes,
                        budget: self.options().host_budget_bytes(),
                    }
                });
            }

            // Start one background write as soon as the finite host tier fills.
            // It remains charged to host memory until the worker commits and
            // releases its buffers; a later demotion waits only if it needs space.
            let proactive = matches!(
                self.options().live_disk_policy(),
                LiveCacheDiskPolicy::Enabled { .. }
            ) && state.telemetry.report.current_host_bytes != 0
                && (state.telemetry.report.current_host_bytes
                    >= self.options().host_budget_bytes()
                    || pool_report.current_host_bytes >= pool_report.limits.host_bytes());
            let candidate = if proactive {
                eviction_candidate(
                    &state,
                    CacheTier::Host,
                    required,
                    0,
                    self.options().eviction_policy(),
                )
            } else {
                None
            };
            drop(state);
            if let Some(id) = candidate {
                let _ = self.begin_host_demotion(&id)?;
            }
            return Ok(());
        }
    }
}

pub(super) fn eviction_candidate(
    state: &CacheManagerState,
    tier: CacheTier,
    required: Option<&CacheBlockId>,
    recent_per_layer: usize,
    policy: CacheEvictionPolicy,
) -> Option<CacheBlockId> {
    let candidates = state
        .blocks
        .iter()
        .filter(|(_, record)| {
            physical_state_is_tier_candidate(&record.physical, tier)
                && record.pending_disk().is_none()
        })
        .map(|(_, record)| record.physical.id());
    state
        .lifecycle
        .eviction_candidate_borrowed(candidates, required, recent_per_layer, policy)
        .expect("canonical storage blocks have lifecycle state")
        .cloned()
}

fn physical_state_is_tier_candidate(physical: &MlxCacheBlockStorage, tier: CacheTier) -> bool {
    matches!(
        (physical.phase(), tier),
        (CacheStoragePhase::Device, CacheTier::Device)
            | (
                CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked,
                CacheTier::Host
            )
            | (CacheStoragePhase::DiskReady, CacheTier::Disk)
    )
}

/// Same actual local/pool comparisons for ordinary transitions and the admitted
/// device-resident publication. This read grants no tier-transition permission.
pub(super) struct BudgetSnapshot {
    pub(super) pool: eredu_runtime::CachePoolReport,
    pub(super) pool_device_over: bool,
    pub(super) local_device_over: bool,
    pub(super) pool_host_over: bool,
    pub(super) local_host_over: bool,
}
pub(super) fn budget_snapshot(
    state: &CacheManagerState,
    options: &PagedCacheOptions,
) -> Result<BudgetSnapshot, CachePoolError> {
    let pool = state.pool.report()?;
    Ok(BudgetSnapshot {
        pool_device_over: pool.current_device_bytes > pool.limits.device_bytes(),
        local_device_over: state.telemetry.report.current_device_bytes
            > options.device_budget_bytes(),
        pool_host_over: pool.current_host_bytes > pool.limits.host_bytes(),
        local_host_over: state.telemetry.report.current_host_bytes > options.host_budget_bytes(),
        pool,
    })
}

/// Logical local-policy usage shared by ordinary and prepared Host writeback.
/// Actual file allocation/reservation remains a separate physical pool charge.
pub(super) fn live_disk_bytes(state: &CacheManagerState) -> Option<u64> {
    let mut bytes = 0u64;
    for (_, record) in state.blocks.iter() {
        if record.disk().is_some_and(|location| !location.persistent) {
            bytes = bytes.checked_add(record.bytes)?;
        }
    }
    for (key, reservation) in state.host_write_reservations.iter() {
        if !state
            .blocks
            .get(&key.id)
            .is_some_and(|record| record.disk().is_some_and(|location| !location.persistent))
        {
            bytes = bytes.checked_add(reservation.logical_bytes)?;
        }
    }
    Some(bytes)
}
pub(super) fn live_disk_query_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<(&CacheManagerState, u64, Option<u64>)>(),
        size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(),
        size_of::<
            eredu_runtime::cache::CacheRecordTableIter<
                '_,
                CacheIoOperationKey,
                HostWriteReservation,
            >,
        >(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
