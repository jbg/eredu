//! Session lifecycle, block catalog mutation, and manager reset operations.

use super::*;

impl CacheResidencyManager {
    /// Creates an empty manager with globally shared finite limits.
    pub fn new(options: PagedCacheOptions) -> Result<Self, CacheResidencyError> {
        if let LiveCacheDiskPolicy::Enabled { directory, .. } = options.live_disk_policy() {
            fs::create_dir_all(directory).map_err(|source| CacheResidencyError::Io {
                action: "create live cache directory",
                path: directory.clone(),
                source,
            })?;
        }
        let queue_capacity = match options.live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => 0,
            LiveCacheDiskPolicy::Enabled { queue_capacity, .. } => *queue_capacity,
        };
        let effective_queue_capacity = queue_capacity.max(1);
        let telemetry = CacheResidencyTelemetry::new(effective_queue_capacity);
        let disk_worker = Some(Arc::new(DiskWorker::new(effective_queue_capacity)?));
        let host_demotion_worker = Arc::new(HostDemotionWorker::new()?);
        let recent_device_blocks = options.recent_device_blocks();
        let device_budget_bytes = options.device_budget_bytes();
        let host_budget_bytes = options.host_budget_bytes();
        let disk_budget_bytes = match options.live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => None,
            LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => Some(*budget_bytes),
        };
        let session_id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let pool = match options.pool().cloned() {
            Some(pool) => pool,
            None => options.create_pool()?,
        };
        let pool_membership = Arc::new(pool.register_manager(session_id)?);
        Ok(Self {
            session_id,
            inner: Arc::new(CacheResidencyManagerInner {
                options,
                state: Arc::new(Mutex::new(CacheManagerState {
                    pool,
                    pool_manager_id: session_id,
                    generation: 0,
                    background_disk_error: None,
                    lifecycle: CacheBlockLifecycle::new(),
                    blocks: BTreeMap::new(),
                    host_write_reservations: HashMap::new(),
                    retiring_host_demotions: HashMap::new(),
                    retiring_disk_reads: HashMap::new(),
                    transfer_device: None,
                    telemetry,
                    recent_device_blocks,
                    device_budget_bytes,
                    host_budget_bytes,
                    disk_budget_bytes,
                })),
                host_demotion_worker,
                disk_worker,
                pool_membership,
            }),
        })
    }

    /// Returns the live cache identity included in every block id.
    pub const fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Forks one quiescent cache session into an independent block namespace.
    ///
    /// Immutable block arrays may be shared, but lifecycle ownership, mutable-tail accounting,
    /// budgets, and future block identities belong exclusively to the returned manager.
    pub(crate) fn fork_session(&self, stream: &Stream) -> Result<Self, CacheResidencyError> {
        let (blocks, tails) = {
            let state = self.lock()?;
            let blocks = state
                .blocks
                .keys()
                .map(|id| {
                    state
                        .lifecycle
                        .is_protected_prefix(id)
                        .map(|protected| (id.clone(), protected))
                        .map_err(CacheResidencyError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tails = state.lifecycle.tails().collect::<Vec<_>>();
            (blocks, tails)
        };
        let fork = Self::new(self.options().clone())?;
        for (id, protected) in blocks {
            let lease = self.lease_block(&id, stream)?;
            let arrays = lease.arrays().clone();
            fork.seal_block(
                id.global_layer,
                id.start,
                id.end,
                id.rank,
                arrays,
                protected,
            )?;
        }
        for (layer, tail) in tails {
            fork.set_tail_state(layer, tail.bytes, tail.end)?;
        }
        Ok(fork)
    }

    /// Returns validated paged-cache options.
    pub fn options(&self) -> &PagedCacheOptions {
        &self.inner.options
    }

    /// Returns the aggregate process pool accounting for this manager.
    pub fn pool(&self) -> &CacheResidencyPool {
        self.inner.pool_membership.pool()
    }

    pub(super) fn lock(&self) -> Result<MutexGuard<'_, CacheManagerState>, CacheResidencyError> {
        self.inner
            .state
            .lock()
            .map_err(|_| CacheResidencyError::ManagerPoisoned)
    }

    /// Binds the manager to the transfer device selected by `stream`.
    pub fn bind_transfer_device(&self, stream: &Stream) -> Result<(), CacheResidencyError> {
        let device = CacheTransferDevice::from_stream(stream)?;
        let mut state = self.lock()?;
        match state.transfer_device {
            Some(bound) if bound != device => Err(CacheResidencyError::Runtime(format!(
                "paged cache is bound to {:?} device {} but received {:?} device {}",
                bound.device_type, bound.index, device.device_type, device.index
            ))),
            Some(_) => Ok(()),
            None => {
                state.transfer_device = Some(device);
                Ok(())
            }
        }
    }

    /// Updates the mutable tail's logical byte count and end position.
    pub fn set_tail_state(
        &self,
        layer: usize,
        bytes: u64,
        end: i64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        let previous = state
            .lifecycle
            .set_tail(layer, MutableCacheTail { bytes, end });
        let allocated = previous.is_none_or(|tail| tail.bytes == 0) && bytes > 0;
        if allocated {
            state.telemetry.report.tail_allocations += 1;
        }
        drop(state);
        if let Err(error) = self.rebalance(None, false) {
            let mut state = self.lock()?;
            state.lifecycle.restore_tail(layer, previous);
            if allocated {
                state.telemetry.report.tail_allocations =
                    state.telemetry.report.tail_allocations.saturating_sub(1);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        Ok(())
    }

    /// Publishes a completed mutable tail range as an immutable cache block.
    pub fn seal_block(
        &self,
        global_layer: usize,
        start: i64,
        end: i64,
        rank: Option<CacheRankIdentity>,
        arrays: CacheBlockArrays,
        protected_prefix: bool,
    ) -> Result<CacheBlockId, CacheResidencyError> {
        if start < 0 || end <= start {
            return Err(CacheResidencyError::InvalidTokenRange { start, end });
        }
        let representation = arrays.representation();
        validate_block_arrays(&arrays, end - start)?;
        eval(arrays.arrays()).map_err(|source| CacheResidencyError::Runtime(source.to_string()))?;
        let id = CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start,
            end,
            rank,
        };
        let mut state = self.lock()?;
        if state.blocks.contains_key(&id) {
            return Err(CacheLifecycleError::DuplicateBlock(id).into());
        }
        let bytes = arrays.bytes();
        let record = CacheBlockRecord {
            shapes: arrays.shapes(),
            dtypes: arrays.dtypes(),
            physical: MlxCacheBlockStorage::device(id.clone(), arrays, None),
            bytes,
            imported: false,
        };
        state
            .lifecycle
            .insert(id.clone(), protected_prefix)
            .map_err(CacheResidencyError::from)?;
        state.blocks.insert(id.clone(), record);
        drop(state);
        if let Err(error) = self.rebalance(Some(&id), false) {
            let mut state = self.lock()?;
            if let Some(record) = state.blocks.remove(&id) {
                state.lifecycle.remove(&id)?;
                cancel_record_operation(&record, &mut state.telemetry.report);
                remove_ephemeral_file(&record);
            }
            update_report_totals(&mut state);
            return Err(error);
        }
        let mut state = self.lock()?;
        state.telemetry.report.block_seals += 1;
        Ok(id)
    }

    /// Returns ordered block identities for a layer and visible token range.
    pub fn layer_block_ids(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        visible_end: i64,
        prefix_tokens: i64,
    ) -> Result<Vec<CacheBlockId>, CacheResidencyError> {
        let state = self.lock()?;
        Ok(state
            .blocks
            .keys()
            .filter(|id| {
                id.global_layer == layer
                    && id.representation == representation
                    && id.start < visible_end
                    && (id.end > visible_start || id.start < prefix_tokens)
            })
            .cloned()
            .collect())
    }

    /// Returns the greatest logical end position stored for a layer.
    pub fn layer_end(
        &self,
        layer: usize,
        representation: CacheRepresentation,
    ) -> Result<i64, CacheResidencyError> {
        let state = self.lock()?;
        Ok(state
            .blocks
            .keys()
            .filter(|id| id.global_layer == layer && id.representation == representation)
            .map(|id| id.end)
            .max()
            .unwrap_or(0))
    }

    /// Removes one unleased cache block from all tiers.
    pub fn remove_block(&self, id: &CacheBlockId) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if !state.blocks.contains_key(id) {
            return Err(CacheLifecycleError::MissingBlock(id.clone()).into());
        }
        state.lifecycle.remove(id)?;
        let tickets = advance_generation_locked(&mut state);
        let record = state
            .blocks
            .remove(id)
            .expect("validated cache block still present");
        remove_ephemeral_file(&record);
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    /// Atomically truncates a layer while preserving its protected prefix.
    pub fn truncate_layer_transaction(
        &self,
        global_layer: usize,
        representation: CacheRepresentation,
        end: i64,
        replacement: Option<(CacheBlockLease, CacheBlockArrays)>,
        protected_prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if end < 0 {
            return Err(CacheResidencyError::InvalidTokenRange { start: 0, end });
        }
        if let Some((lease, arrays)) = &replacement {
            let old_id = lease.id();
            if old_id.global_layer != global_layer
                || old_id.representation != representation
                || old_id.start >= end
                || old_id.end <= end
                || arrays.representation() != representation
            {
                return Err(CacheResidencyError::ArrayMismatch(
                    "trailing cache replacement does not match the truncated layer".into(),
                ));
            }
            validate_block_arrays(arrays, end - old_id.start)?;
            eval(arrays.arrays())
                .map_err(|source| CacheResidencyError::Runtime(source.to_string()))?;
        }

        let mut state = self.lock()?;
        if let Some(error) = state.background_disk_error.take() {
            return Err(CacheResidencyError::Runtime(format!(
                "background cache disk write failed: {error}"
            )));
        }
        let affected = state
            .blocks
            .keys()
            .filter(|id| {
                id.global_layer == global_layer
                    && id.representation == representation
                    && id.end > end
            })
            .cloned()
            .collect::<Vec<_>>();
        let crossing = affected.iter().find(|id| id.start < end);
        match (crossing, replacement.as_ref()) {
            (Some(crossing), Some((lease, _))) if crossing == lease.id() => {}
            (None, None) => {}
            _ => {
                return Err(CacheResidencyError::ArrayMismatch(
                    "trailing cache replacement does not match the block crossing the truncation boundary"
                        .into(),
                ))
            }
        }
        for id in &affected {
            let record = state
                .blocks
                .get(id)
                .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
            let owned_replacement_lease = replacement
                .as_ref()
                .is_some_and(|(lease, _)| lease.id() == id);
            if owned_replacement_lease && record.tier() != CacheTier::Device {
                return Err(CacheResidencyError::Runtime(
                    "truncated cache replacement lease is not device resident".into(),
                ));
            }
        }

        let replacement_id = replacement.as_ref().map(|(lease, _)| CacheBlockId {
            session_id: self.session_id,
            global_layer,
            representation,
            start: lease.id().start,
            end,
            rank: lease.id().rank,
        });
        if let Some(id) = &replacement_id {
            if state.blocks.contains_key(id) && !affected.contains(id) {
                return Err(CacheLifecycleError::DuplicateBlock(id.clone()).into());
            }
        }

        let removals = affected
            .iter()
            .map(|id| {
                let owned_replacement_lease = replacement
                    .as_ref()
                    .is_some_and(|(lease, _)| lease.id() == id);
                (id.clone(), usize::from(owned_replacement_lease))
            })
            .collect::<Vec<_>>();
        state.lifecycle.replace(
            &removals,
            replacement_id
                .clone()
                .map(|id| (id, end <= protected_prefix_tokens)),
            global_layer,
            MutableCacheTail { bytes: 0, end },
        )?;

        let tickets = advance_generation_locked(&mut state);
        let mut removed = Vec::with_capacity(affected.len());
        for id in &affected {
            if let Some(record) = state.blocks.remove(id) {
                removed.push(record);
            }
        }
        if let Some((mut lease, arrays)) = replacement {
            let old_id = lease.id().clone();
            lease.released = true;
            let id = replacement_id.expect("validated replacement id is available");
            debug_assert_eq!(id.start, old_id.start);
            let record = CacheBlockRecord {
                shapes: arrays.shapes(),
                dtypes: arrays.dtypes(),
                bytes: arrays.bytes(),
                physical: MlxCacheBlockStorage::device(id.clone(), arrays, None),
                imported: false,
            };
            let previous = state.blocks.insert(id, record);
            debug_assert!(previous.is_none());
            state.telemetry.report.block_seals += 1;
        }
        update_report_totals(&mut state);
        drop(state);

        for record in &removed {
            remove_ephemeral_file(record);
        }
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    /// Discards layer blocks entirely before the visible window.
    pub fn discard_before(
        &self,
        layer: usize,
        representation: CacheRepresentation,
        visible_start: i64,
        prefix_tokens: i64,
    ) -> Result<(), CacheResidencyError> {
        if self.options().retains_discarded_for_persistence() {
            return Ok(());
        }
        let mut state = self.lock()?;
        let ids = state
            .blocks
            .iter()
            .filter(|(id, record)| {
                id.global_layer == layer
                    && id.representation == representation
                    && id.end <= visible_start
                    && id.end > prefix_tokens
                    && state.lifecycle.lease_count(id).ok() == Some(0)
                    && !record.imported
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let tickets = if ids.is_empty() {
            Vec::new()
        } else {
            advance_generation_locked(&mut state)
        };
        for id in ids {
            if let Some(record) = state.blocks.remove(&id) {
                state.lifecycle.remove(&id)?;
                remove_ephemeral_file(&record);
                state.telemetry.report.discarded_sliding_blocks += 1;
            }
        }
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }

    /// Clears every live block and advances the manager generation.
    pub fn clear(&self) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        state.lifecycle.clear()?;
        let tickets = advance_generation_locked(&mut state);
        for record in state.blocks.values() {
            remove_ephemeral_file(record);
        }
        state.blocks.clear();
        update_report_totals(&mut state);
        drop(state);
        self.retire_tickets(&tickets)?;
        Ok(())
    }
}

fn validate_block_arrays(
    arrays: &CacheBlockArrays,
    token_count: i64,
) -> Result<(), CacheResidencyError> {
    let [first, second] = arrays.arrays();
    if first.dtype() != second.dtype() {
        return Err(CacheResidencyError::ArrayMismatch(
            "both arrays in a cache block must share a dtype".into(),
        ));
    }
    let sequence_axis = match arrays {
        CacheBlockArrays::KeyValue { .. } => {
            if first.ndim() < 2 || second.ndim() < 2 {
                return Err(CacheResidencyError::ArrayMismatch(
                    "key/value blocks must have a sequence axis".into(),
                ));
            }
            first.ndim() - 2
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => {
            if first.ndim() != 3 || second.ndim() != 3 {
                return Err(CacheResidencyError::ArrayMismatch(
                    "compressed latent blocks must be rank-3".into(),
                ));
            }
            1
        }
    };
    if first.dim(sequence_axis as i32) as i64 != token_count
        || second.dim(sequence_axis as i32) as i64 != token_count
    {
        return Err(CacheResidencyError::ArrayMismatch(
            "cache block range does not match its sequence dimensions".into(),
        ));
    }
    match arrays {
        CacheBlockArrays::KeyValue { .. } => {
            if first.ndim() != second.ndim()
                || first.shape()[..first.ndim() - 2] != second.shape()[..second.ndim() - 2]
            {
                return Err(CacheResidencyError::ArrayMismatch(
                    "key and value blocks must share leading dimensions".into(),
                ));
            }
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => {
            if first.dim(0) != second.dim(0) || first.dim(1) != second.dim(1) {
                return Err(CacheResidencyError::ArrayMismatch(
                    "compressed latent and rotary blocks must share batch and sequence dimensions"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

fn cancel_record_operation(record: &CacheBlockRecord, report: &mut CacheResidencyReport) {
    if let Some(pending) = record.pending_disk() {
        if pending.ticket.cancel() {
            report.cancellations += 1;
        }
    }
}

fn advance_generation_locked(state: &mut CacheManagerState) -> Vec<PendingCacheOperation> {
    state.generation = state.generation.wrapping_add(1);
    state.background_disk_error = None;
    let mut tickets = Vec::new();
    let mut demotion_reservations = Vec::new();
    let mut read_reservations = Vec::new();
    for record in state.blocks.values_mut() {
        if let Some(ticket) = record.host_demotion_ticket().cloned() {
            demotion_reservations.push((
                ticket.operation_id,
                RetiringHostDemotion {
                    id: record.physical.id().clone(),
                    device_bytes: record.bytes,
                    host_bytes: ticket.reserved_host_bytes,
                },
            ));
            tickets.push(PendingCacheOperation::HostDemotion(ticket));
        }
        if let Some(pending) = record.pending_disk().cloned() {
            if pending.ticket.cancel() {
                state.telemetry.report.cancellations += 1;
            }
            tickets.push(PendingCacheOperation::Disk(pending.ticket.clone()));
            if pending.ticket.key.kind != CacheIoOperationKind::Write {
                if let Some(reserved_host_bytes) = pending.reserved_host_bytes {
                    read_reservations.push((
                        pending.ticket.key.clone(),
                        (record.physical.id().global_layer, reserved_host_bytes),
                    ));
                }
                record.physical.fail_io_if_matches(&pending.ticket.key);
            }
        }
    }
    state.retiring_host_demotions.extend(demotion_reservations);
    state.retiring_disk_reads.extend(read_reservations);
    tickets
}
