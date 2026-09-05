//! Ordered block acquisition, promotion, prefetch, and leases.

use super::*;

/// Fixed current-plus-next cache-block promotion window.
pub struct CacheBlockPrefetch {
    pub(super) manager: CacheResidencyManager,
    pub(super) ids: VecDeque<CacheBlockId>,
    pub(super) pending: VecDeque<CacheBlockLease>,
    pub(super) transfer_stream: Stream,
    pub(super) execution_stream: Stream,
}

impl CacheBlockPrefetch {
    pub(super) fn new(
        manager: CacheResidencyManager,
        ids: Vec<CacheBlockId>,
        execution_stream: &Stream,
    ) -> Result<Self, CacheResidencyError> {
        let device = execution_stream
            .get_device()
            .map_err(|source| transfer_error("inspect cache execution stream", source))?;
        Ok(Self {
            manager,
            ids: ids.into(),
            pending: VecDeque::with_capacity(PAGED_CACHE_PREFETCH_BLOCKS),
            transfer_stream: Stream::new_with_device(&device),
            execution_stream: execution_stream.clone(),
        })
    }

    /// Returns the next ordered block. At most the current and following block
    /// hold leases; a one-block device budget falls back to demand promotion.
    pub fn next_block(&mut self) -> Result<Option<CacheBlockLease>, CacheResidencyError> {
        while self.pending.len() < PAGED_CACHE_PREFETCH_BLOCKS {
            let Some(id) = self.ids.front() else {
                break;
            };
            if !self.pending.is_empty() && !self.window_has_capacity_for(id)? {
                break;
            }
            match self
                .manager
                .prepare_block_transfer(id, &self.transfer_stream)
            {
                Ok(lease) => {
                    self.ids.pop_front();
                    self.pending.push_back(lease);
                }
                Err(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Device,
                    ..
                }) if !self.pending.is_empty() => break,
                Err(error) => return Err(error),
            }
        }
        let Some(lease) = self.pending.pop_front() else {
            return Ok(None);
        };
        lease.wait_on(&self.execution_stream)?;
        Ok(Some(lease))
    }

    fn window_has_capacity_for(&self, id: &CacheBlockId) -> Result<bool, CacheResidencyError> {
        let pending_bytes = self
            .pending
            .iter()
            .fold(0u64, |total, lease| total.saturating_add(lease.bytes()));
        let state = self.manager.lock()?;
        let next_bytes = state
            .blocks
            .get(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?
            .bytes;
        Ok(
            pending_bytes.saturating_add(next_bytes)
                <= self.manager.options().device_budget_bytes(),
        )
    }

    #[cfg(test)]
    pub(super) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    pub(super) fn stream_indices(&self) -> Result<(i32, i32), CacheResidencyError> {
        Ok((
            self.execution_stream
                .get_index()
                .map_err(|source| transfer_error("inspect cache execution stream", source))?,
            self.transfer_stream
                .get_index()
                .map_err(|source| transfer_error("inspect cache transfer stream", source))?,
        ))
    }
}

/// One device-cataloged block and its single-shot promotion completion.
///
/// The lease inserts its event dependency before exposing arrays to a consumer.
/// Dropping it after that wait is safe: MLX retains the event for queued work,
/// while lazy consumer graphs retain the arrays they reference. Asynchronous
/// promotion failures poison the consumer stream and surface when that work is
/// evaluated. The lease is intentionally neither `Send` nor `Sync` because it
/// owns `safemlx`'s thread-affine [`Event`].
pub struct CacheBlockLease {
    pub(super) id: CacheBlockId,
    pub(super) arrays: CacheBlockArrays,
    pub(super) manager: CacheResidencyManager,
    pub(super) completions: Vec<Event>,
    pub(super) _transfer_reservation: Option<CachePoolReservation>,
    pub(super) released: bool,
}

impl CacheBlockLease {
    /// Returns the leased block identity.
    pub fn id(&self) -> &CacheBlockId {
        &self.id
    }

    /// Returns the leased device arrays.
    pub fn arrays(&self) -> &CacheBlockArrays {
        &self.arrays
    }

    /// Returns the logical device byte count of the leased arrays.
    pub fn bytes(&self) -> u64 {
        self.arrays.bytes()
    }

    pub(super) fn wait_on(&self, stream: &Stream) -> Result<(), CacheResidencyError> {
        for completion in &self.completions {
            completion
                .wait_on(stream)
                .map_err(|source| transfer_error("wait for cache block promotion", source))?;
        }
        Ok(())
    }
}

impl Drop for CacheBlockLease {
    fn drop(&mut self) {
        if !self.released {
            self.manager.release_lease(&self.id);
            self.released = true;
        }
    }
}

impl CacheResidencyManager {
    /// Promotes and leases one cache block for use on `stream`.
    pub fn lease_block(
        &self,
        id: &CacheBlockId,
        stream: &Stream,
    ) -> Result<CacheBlockLease, CacheResidencyError> {
        let lease = self.prepare_block_transfer(id, stream)?;
        lease.wait_on(stream)?;
        Ok(lease)
    }

    /// Creates a fixed two-block promotion window on a dedicated stream for
    /// the execution stream's device.
    pub fn prefetch_blocks(
        &self,
        ids: Vec<CacheBlockId>,
        execution_stream: &Stream,
    ) -> Result<CacheBlockPrefetch, CacheResidencyError> {
        CacheBlockPrefetch::new(self.clone(), ids, execution_stream)
    }

    pub(super) fn prepare_block_transfer(
        &self,
        id: &CacheBlockId,
        transfer_stream: &Stream,
    ) -> Result<CacheBlockLease, CacheResidencyError> {
        self.bind_transfer_device(transfer_stream)?;
        let started = Instant::now();
        let mut loaded_from_disk = false;
        loop {
            let mut state = self.lock()?;
            let generation = state.generation;
            let physical = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?
                .physical
                .clone();

            match physical.phase() {
                CacheStoragePhase::DemotingToHost => {
                    let ticket = physical
                        .host_demotion()
                        .expect("demoting phase owns its exact completion")
                        .clone();
                    drop(state);
                    self.finish_device_demotion(&ticket)?;
                }
                CacheStoragePhase::DiskReady | CacheStoragePhase::DiskReading => {
                    let location = physical
                        .backing()
                        .expect("disk phase owns its backing")
                        .clone();
                    let worker = self.inner.disk_worker.as_ref().ok_or_else(|| {
                        CacheResidencyError::Runtime("cache disk worker is unavailable".into())
                    })?;
                    let (ticket, submission, joined, _transfer_reservation, reserved_host_bytes) =
                        match physical.phase() {
                            CacheStoragePhase::DiskReading => {
                                let pending = physical
                                    .io()
                                    .expect("disk-reading phase owns its exact completion");
                                (
                                    pending.ticket.clone(),
                                    None,
                                    true,
                                    None,
                                    pending
                                        .reserved_host_bytes
                                        .expect("disk-read completion owns its host reservation"),
                                )
                            }
                            CacheStoragePhase::DiskReady => {
                                let record = state
                                    .blocks
                                    .get(id)
                                    .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                                let bytes = record.bytes;
                                let reserved_host_bytes = host_cache_layout_capacity_upper_bound(
                                    &record.shapes,
                                    &record.dtypes,
                                )?;
                                let required_host_bytes = state
                                    .telemetry
                                    .report
                                    .current_host_bytes
                                    .saturating_add(reserved_host_bytes);
                                if required_host_bytes > self.options().host_budget_bytes() {
                                    let candidate = eviction_candidate(
                                        &state,
                                        CacheTier::Host,
                                        Some(id),
                                        0,
                                        self.options().eviction_policy(),
                                    );
                                    drop(state);
                                    if let Some(candidate) = candidate {
                                        match self.begin_host_demotion(&candidate)? {
                                            HostDemotionProgress::Retry
                                            | HostDemotionProgress::Freed => continue,
                                            HostDemotionProgress::Pending(ticket) => {
                                                self.wait_for_host_release(&ticket)?;
                                                continue;
                                            }
                                        }
                                    }
                                    return Err(CacheResidencyError::BudgetExceeded {
                                        tier: CacheTier::Host,
                                        required: required_host_bytes,
                                        budget: self.options().host_budget_bytes(),
                                    });
                                }
                                let host_admission = state.pool.reserve(CachePoolUsage {
                                    host_bytes: reserved_host_bytes,
                                    ..CachePoolUsage::default()
                                })?;
                                let transfer_reservation =
                                    Some(state.pool.reserve_transfer(bytes)?);
                                let submission = worker.prepare_read(
                                    generation,
                                    id,
                                    &location,
                                    id.representation,
                                )?;
                                let ticket = submission.ticket.clone();
                                let record = state
                                    .blocks
                                    .get_mut(id)
                                    .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                                record.physical.begin_read(MlxCacheIoOperation {
                                    ticket: ticket.clone(),
                                    reserved_host_bytes: Some(reserved_host_bytes),
                                })?;
                                update_report_totals(&mut state);
                                drop(host_admission);
                                (
                                    ticket,
                                    Some(submission),
                                    false,
                                    transfer_reservation,
                                    reserved_host_bytes,
                                )
                            }
                            _ => unreachable!("disk phase was matched above"),
                        };
                    drop(state);

                    let outcome = match submission {
                        Some(submission) => Some(submission.enqueue()?),
                        None => None,
                    };
                    let result = ticket.wait();
                    let mut state = self.lock()?;
                    if joined || outcome.as_ref().is_some_and(|outcome| outcome.joined) {
                        state.telemetry.report.in_flight_waits += 1;
                        state.layer_activity_mut(id.global_layer).in_flight_waits += 1;
                    }
                    if let Some(outcome) = &outcome {
                        state.telemetry.report.queue_peak_occupancy = state
                            .telemetry
                            .report
                            .queue_peak_occupancy
                            .max(outcome.peak_occupancy);
                        state.telemetry.report.queue_backpressure +=
                            u64::from(outcome.backpressure);
                    }
                    let stale = state.generation != ticket.key.generation;
                    match result {
                        Ok(DiskResult::Read(block)) if !stale => {
                            let shapes = block.shapes()?;
                            let dtypes = block.dtypes()?;
                            let bytes = block.bytes()?;
                            let capacity = block.capacity()?;
                            let record = state
                                .blocks
                                .get_mut(id)
                                .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                            if shapes != record.shapes
                                || dtypes != record.dtypes
                                || bytes != record.bytes
                            {
                                record.physical.fail_io_if_matches(&ticket.key);
                                drop(state);
                                worker.retire(&ticket);
                                return Err(CacheResidencyError::MalformedShard {
                                    path: location.path,
                                    reason: "array shape or dtype does not match the manifest"
                                        .into(),
                                });
                            }
                            if capacity > reserved_host_bytes {
                                record.physical.fail_io_if_matches(&ticket.key);
                                drop(state);
                                worker.retire(&ticket);
                                return Err(CacheResidencyError::Runtime(format!(
                                    "host cache allocation capacity {capacity} exceeded its pre-admitted bound {reserved_host_bytes}"
                                )));
                            }
                            if record.physical.io_matches(&ticket.key) {
                                record.physical.finish_read(&ticket.key, block)?;
                            }
                            loaded_from_disk = true;
                            update_report_totals(&mut state);
                        }
                        Ok(DiskResult::Read(_))
                        | Err(CacheResidencyError::DiskOperationCancelled { .. })
                            if stale =>
                        {
                            drop(state);
                            worker.retire(&ticket);
                            return Err(CacheResidencyError::DiskOperationCancelled {
                                generation: ticket.key.generation,
                            });
                        }
                        Ok(_) => {
                            drop(state);
                            worker.retire(&ticket);
                            return Err(CacheResidencyError::Runtime(
                                "cache disk worker returned an unexpected operation result".into(),
                            ));
                        }
                        Err(error) => {
                            if let Some(record) = state.blocks.get_mut(id) {
                                record.physical.fail_io_if_matches(&ticket.key);
                            }
                            state.telemetry.report.failures += 1;
                            state.layer_activity_mut(id.global_layer).failures += 1;
                            drop(state);
                            worker.retire(&ticket);
                            return Err(error);
                        }
                    }
                    drop(state);
                    worker.retire(&ticket);
                }
                CacheStoragePhase::HostUnbacked
                | CacheStoragePhase::HostBacked
                | CacheStoragePhase::HostWriting => {
                    if physical.phase() == CacheStoragePhase::HostWriting {
                        let pending = physical
                            .io()
                            .expect("host-writing phase owns its exact completion");
                        drop(state);
                        self.wait_for_host_release(&pending.ticket)?;
                        continue;
                    }
                    let block = physical
                        .host_resource()
                        .expect("stable host phase owns host resources")
                        .clone();
                    let transfer_reservation = state.pool.reserve_transfer(block.capacity()?)?;
                    let (device_arrays, completions) = block.copy_to_device(transfer_stream)?;
                    if state.generation != generation {
                        return Err(CacheResidencyError::DiskOperationCancelled { generation });
                    }
                    state.lifecycle.acquire(id)?;
                    let (bytes, promotion) = {
                        let record = state
                            .blocks
                            .get_mut(id)
                            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
                        let promotion = record.physical.promote_host(device_arrays.clone())?;
                        (record.bytes, promotion)
                    };
                    state.telemetry.report.demand_misses += 1;
                    if loaded_from_disk {
                        state.telemetry.report.disk_promotions += 1;
                    } else {
                        state.telemetry.report.host_promotions += 1;
                    }
                    state.telemetry.report.transfer_bytes += bytes;
                    let transfer_wait = started.elapsed();
                    state.telemetry.report.transfer_wait += transfer_wait;
                    let activity = state.layer_activity_mut(id.global_layer);
                    activity.demand_misses += 1;
                    if loaded_from_disk {
                        activity.disk_promotions += 1;
                    } else {
                        activity.host_promotions += 1;
                    }
                    activity.transfer_bytes += bytes;
                    activity.transfer_wait += transfer_wait;
                    drop(state);
                    if let Err(error) = self.rebalance(Some(id), true) {
                        let mut state = self.lock()?;
                        if state.generation == generation {
                            state.lifecycle.release(id)?;
                            if let Some(record) = state.blocks.get_mut(id) {
                                record.physical.restore_host(promotion)?;
                            }
                            update_report_totals(&mut state);
                        }
                        return Err(error);
                    }
                    return Ok(CacheBlockLease {
                        id: id.clone(),
                        arrays: device_arrays,
                        manager: self.clone(),
                        completions,
                        _transfer_reservation: Some(transfer_reservation),
                        released: false,
                    });
                }
                CacheStoragePhase::Device => {
                    let arrays = physical
                        .device_resource()
                        .expect("device phase owns device arrays")
                        .clone();
                    state.lifecycle.acquire(id)?;
                    drop(state);
                    let completion = match async_eval_with_event(arrays.arrays()) {
                        Ok(completion) => completion,
                        Err(source) => {
                            if let Ok(mut state) = self.lock() {
                                let _ = state.lifecycle.release(id);
                                update_report_totals(&mut state);
                            }
                            return Err(transfer_error(
                                "submit resident cache block completion",
                                source,
                            ));
                        }
                    };
                    let mut state = self.lock()?;
                    if state.generation != generation {
                        state.lifecycle.release(id)?;
                        return Err(CacheResidencyError::DiskOperationCancelled { generation });
                    }
                    state.telemetry.report.demand_hits += 1;
                    let transfer_wait = started.elapsed();
                    state.telemetry.report.transfer_wait += transfer_wait;
                    let activity = state.layer_activity_mut(id.global_layer);
                    activity.demand_hits += 1;
                    activity.transfer_wait += transfer_wait;
                    drop(state);
                    if let Err(error) = self.rebalance(Some(id), true) {
                        if let Ok(mut state) = self.lock() {
                            let _ = state.lifecycle.release(id);
                            update_report_totals(&mut state);
                        }
                        return Err(error);
                    }
                    return Ok(CacheBlockLease {
                        id: id.clone(),
                        arrays,
                        manager: self.clone(),
                        completions: vec![completion],
                        _transfer_reservation: None,
                        released: false,
                    });
                }
            }
        }
    }

    fn release_lease(&self, id: &CacheBlockId) {
        let mut background_failed = false;
        if let Ok(mut state) = self.inner.state.lock() {
            let released = state.lifecycle.release(id);
            debug_assert!(
                released.is_ok(),
                "cache lease release lost ownership: {released:?}"
            );
            background_failed = released.is_err() || state.background_disk_error.is_some();
        }
        if !background_failed {
            let _ = self.rebalance(None, false);
        }
    }
}
