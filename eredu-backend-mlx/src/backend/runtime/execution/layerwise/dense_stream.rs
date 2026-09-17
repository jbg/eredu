//! Dense streamed-weight scheduling and transfer guards.

use super::*;
mod prepared;
pub(crate) use prepared::{
    DenseControllerHandle, OriginalDenseControllerFacts, PreparedDenseController,
    PreparedDenseControllerError,
};

pub(crate) fn dense_window_names(group: &str) -> [String; 2] {
    [
        format!("dense:{group}:host"),
        format!("dense:{group}:device"),
    ]
}

fn is_temporary_residency_contention(error: &Error) -> bool {
    match error {
        Error::Residency(cause) | Error::LayerwiseModel(LayerwiseModelError::Residency(cause)) => {
            cause.capacity().is_some()
        }
        _ => false,
    }
}

/// Opens a shared SafeTensors source with a bounded shard-buffer cache.
pub fn open_safetensors_weight_store(
    model_dir: &Path,
    max_cached_shards: usize,
) -> Result<RetainedCheckpointSource, Error> {
    Ok(
        Arc::new(SafetensorsWeightStore::open_with_max_cached_shards(
            model_dir,
            max_cached_shards,
        )?)
        .into(),
    )
}

/// Coordinates bounded host prefetch, device transfers, and dense-stream telemetry.
pub struct DenseStreamController {
    options: DenseDiskStreamLoadOptions,
    background: Option<BackgroundLayerPrefetch>,
    telemetry: DenseStreamTelemetry,
    prepared: Option<prepared::PreparedOrigin>,
}

impl DenseStreamController {
    /// Whether all transfers execute under the calling operation's recovery.
    pub(crate) fn is_foreground(&self) -> bool {
        self.options.host_budget_bytes() == 0 && self.background.is_none()
    }

    /// Actual retained source-only worker policy. Ordinary controllers retain
    /// their existing worker and cannot be installed into an original bank.
    pub(crate) fn original_background_options(&self) -> Option<DenseDiskStreamLoadOptions> {
        (self.prepared.is_some() && self.background.is_none() && self.options.host_budget_bytes() > 0)
            .then_some(self.options)
    }

    /// Creates a controller for a validated dense disk-stream plan.
    pub fn new(
        manager: &ResidencyManager,
        options: DenseDiskStreamLoadOptions,
        planned_layer_count: usize,
        planned_layer_bytes: u64,
        maximum_host_layer_bytes: u64,
        pinned_static_device_bytes: u64,
        groups: impl IntoIterator<Item = (String, Vec<OffloadUnitId>)>,
    ) -> Result<Self, Error> {
        let background = (options.host_budget_bytes() > 0)
            .then(|| {
                BackgroundLayerPrefetch::new(manager.clone(), options.background_queue_capacity())
            })
            .transpose()?;
        let transfer_stream_index = manager.device_stream_index()?;
        Ok(Self {
            options,
            background,
            prepared: None,
            telemetry: DenseStreamTelemetry::new(
                planned_layer_count,
                planned_layer_bytes,
                maximum_host_layer_bytes,
                pinned_static_device_bytes,
                transfer_stream_index,
                groups,
            ),
        })
    }

    /// Opens a bounded device-transfer window for selected group-local units.
    pub fn transfer_window(
        self: &Arc<Self>,
        manager: &ResidencyManager,
        group: impl Into<String>,
        units: &[OffloadUnitId],
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<DenseTransferWindow, Error> {
        Self::transfer_window_owned(
            DenseControllerHandle::Ordinary(Arc::clone(self)),
            manager,
            group,
            units,
            indices,
            prefill,
        )
    }
    fn transfer_window_owned(
        controller: DenseControllerHandle,
        manager: &ResidencyManager,
        group: impl Into<String>,
        units: &[OffloadUnitId],
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<DenseTransferWindow, Error> {
        let indices = indices.into_iter().collect::<Vec<_>>();
        if let Some(&index) = indices.iter().find(|&&index| index >= units.len()) {
            return Err(LayerwiseModelError::InvalidDenseTransferWindow {
                index,
                unit_count: units.len(),
            }
            .into());
        }
        // The same source-prepared manager can serve an ordinary call. Its
        // original controller cannot own a manager-capturing worker (a cycle),
        // so this ordinary window owns that existing worker through cleanup.
        // Original operations bypass this scheduler via their admitted bank.
        let background = if let Some(options) = controller.original_background_options() {
            if manager.admitted_disk_route_active() { return Err(Error::PrefillScopeUnavailable); }
            Some(BackgroundLayerPrefetch::new(manager.clone(), options.background_queue_capacity())?)
        } else { None };
        let mut window = DenseTransferWindow {
            controller,
            background,
            manager: manager.clone(),
            group: group.into(),
            units: units.to_vec(),
            schedule: DenseTransferSchedule::new(indices, DENSE_TRANSFER_WINDOW)?,
            prefill,
        };
        if let Err(error) = window.refill() {
            if !window.schedule.has_ready() || !is_temporary_residency_contention(&error) {
                return Err(error);
            }
        }
        Ok(window)
    }

    pub(crate) fn observe_group(
        &self,
        manager: &ResidencyManager,
        group: &str,
        prefill: bool,
    ) -> Result<(), Error> {
        if let Some(origin) = &self.prepared {
            manager.with_original_dense_ledger(&origin.manager, |ledger| {
                self.telemetry.observe_group_ledger(group, prefill, ledger)
            })??;
        } else {
            let (_, _, units, _) = manager.telemetry_snapshot()?;
            self.telemetry.observe_group(group, prefill, &units)?;
        }
        Ok(())
    }

    pub(crate) fn record_group_execution(&self, group: &str) -> Result<(), Error> {
        self.telemetry.record_group_execution(group)?;
        Ok(())
    }

    /// Cancels prefetch and removes host/device protection for one group.
    pub fn clear_group(&self, manager: &ResidencyManager, group: &str) -> Result<(), Error> {
        let [host, device] = dense_window_names(group);
        manager.protect_group_window(&host, &[], MemoryTier::Host)?;
        manager.protect_group_window(&device, &[], MemoryTier::Device)?;
        if let Some(background) = &self.background {
            background.cancel()?;
        }
        Ok(())
    }

    /// Starts transactional telemetry for one model forward pass.
    pub fn forward_guard(
        self: &Arc<Self>,
        prefill: bool,
        manager: &ResidencyManager,
    ) -> Result<DenseStreamForwardGuard, Error> {
        Self::forward_guard_owned(
            DenseControllerHandle::Ordinary(Arc::clone(self)),
            prefill,
            manager,
        )
    }
    fn forward_guard_owned(
        controller: DenseControllerHandle,
        prefill: bool,
        manager: &ResidencyManager,
    ) -> Result<DenseStreamForwardGuard, Error> {
        if let Some(origin) = &controller.prepared {
            manager.with_original_dense_ledger(&origin.manager, |ledger| {
                controller
                    .telemetry
                    .begin_forward(prefill, &ledger.telemetry())
            })??;
        } else {
            let (_, offload, _, _) = manager.telemetry_snapshot()?;
            controller.telemetry.begin_forward(prefill, &offload)?;
        }
        Ok(DenseStreamForwardGuard {
            controller,
            manager: manager.clone(),
            armed: true,
        })
    }

    fn commit_forward(&self, manager: &ResidencyManager) -> Result<(), Error> {
        if let Some(origin) = &self.prepared {
            manager.with_original_dense_ledger(&origin.manager, |ledger| {
                self.telemetry.commit_forward(&ledger.telemetry())
            })??;
        } else {
            if self.options.samples_backend_memory() || self.options.samples_process_memory() {
                manager.sample_memory(
                    self.options.samples_backend_memory(),
                    self.options.samples_process_memory(),
                )?;
            }
            let (_, offload, _, _) = manager.telemetry_snapshot()?;
            self.telemetry.commit_forward(&offload)?;
        }
        Ok(())
    }

    fn abort_forward(&self) {
        self.telemetry.abort_forward();
    }

    /// Starts cleanup and execution accounting for one group.
    pub fn group_guard(
        self: &Arc<Self>,
        manager: &ResidencyManager,
        group: &str,
    ) -> DenseStreamGroupGuard {
        Self::group_guard_owned(
            DenseControllerHandle::Ordinary(Arc::clone(self)),
            manager,
            group,
        )
    }
    fn group_guard_owned(
        controller: DenseControllerHandle,
        manager: &ResidencyManager,
        group: &str,
    ) -> DenseStreamGroupGuard {
        DenseStreamGroupGuard {
            controller,
            manager: manager.clone(),
            group: group.to_string(),
            armed: true,
        }
    }

    pub(crate) fn record_background(&self, report: eredu_core::residency::BackgroundPrefetchReport) -> Result<(), Error> {
        Ok(self.telemetry.record_background(report)?)
    }

    /// Returns current dense-stream and residency telemetry.
    pub fn report(&self, manager: &ResidencyManager) -> Result<DenseDiskStreamReport, Error> {
        let residency = manager.report()?;
        let background = self
            .background
            .as_ref()
            .map(BackgroundLayerPrefetch::report)
            .transpose()?
            .unwrap_or_default();
        Ok(self.telemetry.report(residency, background)?)
    }
}
/// thread supported by MLX events. A window submits at most two device copies.
/// Callers consume one entry, evaluate and synchronize its compute work, drop
/// that entry, and then call [`Self::refill`] to submit the following layer.
pub struct DenseTransferWindow {
    controller: DenseControllerHandle,
    background: Option<BackgroundLayerPrefetch>,
    manager: ResidencyManager,
    group: String,
    units: Vec<OffloadUnitId>,
    schedule: DenseTransferSchedule<DensePreparedTransfer>,
    prefill: bool,
}

impl DenseTransferWindow {
    fn background(&self) -> Option<&BackgroundLayerPrefetch> {
        self.background.as_ref().or(self.controller.background.as_ref())
    }
    /// Takes the next transfer after ordering `consumer` behind its event.
    pub fn next(&mut self, consumer: &Stream) -> Result<DensePreparedTransfer, Error> {
        let (index, transfer) = self.schedule.pop_ready().ok_or({
            LayerwiseModelError::InvalidDenseTransferWindow {
                index: self.units.len(),
                unit_count: self.units.len(),
            }
        })?;
        debug_assert_eq!(transfer.index(), index);
        transfer.transfer.order_after(consumer)?;
        Ok(transfer)
    }

    /// Reprotects the current/next units and submits one replacement transfer.
    ///
    /// The completed [`DensePreparedTransfer`] must be dropped before this is
    /// called so the fixed two-layer device budget can admit the replacement.
    pub fn refill(&mut self) -> Result<(), Error> {
        // A managed direct route cannot start a worker or select a host source.
        // The policy has already completed and evicted its preceding window.
        if self.manager.admitted_disk_route_active() && !self.controller.is_foreground() {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )));
        }
        let device_indices = self.schedule.desired_indices(DENSE_TRANSFER_WINDOW);
        let host_indices = if self.background().is_some() {
            self.schedule
                .desired_indices(self.controller.options.host_lookahead())
        } else {
            Vec::new()
        };
        let device_units = device_indices
            .iter()
            .map(|&index| self.units[index].clone())
            .collect::<Vec<_>>();
        let host_units = host_indices
            .iter()
            .map(|&index| self.units[index].clone())
            .collect::<Vec<_>>();
        let [host_window, device_window] = dense_window_names(&self.group);
        self.manager
            .protect_group_window(&host_window, &host_units, MemoryTier::Host)?;
        self.manager
            .protect_group_window(&device_window, &device_units, MemoryTier::Device)?;
        if let Some(background) = self.background() {
            for id in &host_units {
                background.submit(id)?;
            }
        }
        while self.schedule.can_admit() {
            let Some(index) = self.schedule.next_pending() else {
                break;
            };
            let id = &self.units[index];
            let _host = if host_indices.contains(&index) {
                self.background()
                    .map(|background| background.acquire(id))
                    .transpose()?
            } else {
                None
            };
            let transfer = self
                .manager
                .acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)?;
            self.schedule
                .admit(index, DensePreparedTransfer { index, transfer })?;
        }
        self.controller
            .observe_group(&self.manager, &self.group, self.prefill)?;
        Ok(())
    }
}

impl Drop for DenseTransferWindow {
    fn drop(&mut self) {
        let _ = self.controller.clear_group(&self.manager, &self.group);
        if let Some(background) = self.background.take() {
            let _ = background.cancel();
            if let Ok(report) = background.report() {
                let _ = self.controller.record_background(report);
            }
        }
    }
}

/// One populated-unit dependency taken from a [`DenseTransferWindow`].
pub struct DensePreparedTransfer {
    index: usize,
    transfer: ResidentTransfer,
}

impl DensePreparedTransfer {
    pub(crate) fn synchronize(&mut self) -> Result<(), Error> {
        self.transfer.synchronize().map_err(Into::into)
    }

    pub(crate) fn retire_completed_original(self) {
        self.transfer.retire_completed_original();
    }

    /// Returns the index in the group's authoritative unit list.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns the single resident lease protected by this transfer.
    pub fn lease(&self) -> &ResidentUnitLease {
        self.transfer
            .leases()
            .first()
            .expect("dense transfer always acquires one unit")
    }
}

/// Transactional guard for dense-stream forward telemetry.
pub struct DenseStreamForwardGuard {
    controller: DenseControllerHandle,
    manager: ResidencyManager,
    armed: bool,
}

impl DenseStreamForwardGuard {
    /// Commits the forward telemetry and disarms rollback-on-drop.
    pub fn complete(mut self) -> Result<(), Error> {
        let result = self.controller.commit_forward(&self.manager);
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for DenseStreamForwardGuard {
    fn drop(&mut self) {
        if self.armed {
            self.controller.abort_forward();
        }
    }
}

/// Cleanup guard for one dense-stream execution group.
pub struct DenseStreamGroupGuard {
    controller: DenseControllerHandle,
    manager: ResidencyManager,
    group: String,
    armed: bool,
}

impl DenseStreamGroupGuard {
    /// Clears group protections and records successful execution.
    pub fn complete(mut self) -> Result<(), Error> {
        let result = self
            .controller
            .clear_group(&self.manager, &self.group)
            .and_then(|()| self.controller.record_group_execution(&self.group));
        self.armed = false;
        result
    }
}

impl Drop for DenseStreamGroupGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.controller.clear_group(&self.manager, &self.group);
        }
    }
}
