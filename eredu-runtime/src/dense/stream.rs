//! Portable host lookahead, device transfer scheduling, and dense-stream telemetry.

use std::sync::Arc;

pub use eredu_core::residency::observations::{
    DenseResidencyMechanisms, NativeResidencyObservations,
};
use eredu_core::residency::{
    AllocatorMemoryMetrics, BackgroundPrefetchReport, MemoryTier, OffloadUnitId,
    PrefetchDemandResolution,
};

use crate::{
    detached_resources::{DetachedResourceInventory, DetachedResourceLimits},
    residency::{ResidencyAccess, ResidentLease},
    BackgroundPrefetchWorker, BackgroundPrefetchWorkerError, DenseDiskStreamLoadOptions,
    DenseDiskStreamReport, DenseStreamTelemetry, DenseStreamTelemetryError, DenseTransferSchedule,
    DenseTransferScheduleError, DenseTransferStep, ParameterUnitLayout, ResidencyReport,
    DENSE_TRANSFER_WINDOW,
};

fn residency_capacity_exhausted(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if matches!(
            error.downcast_ref::<eredu_core::residency::ResidencyLedgerError>(),
            Some(eredu_core::residency::ResidencyLedgerError::BudgetExhausted { .. })
        ) {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

/// Native primitives used by the portable dense-stream driver.
pub trait DenseStreamMechanisms: ResidencyAccess + Clone + Send + Sync + 'static {
    /// Identity of the native transfer stream used for telemetry.
    fn transfer_stream_index(&self) -> Result<i32, Self::Error>;

    /// Settles exact transfer ownership before another acquisition of the same copy.
    fn settle_transfer(transfer: &mut Self::Transfer) -> Result<(), Self::Error>;

    /// Samples the native allocator's current counters.
    fn allocator_memory(&self) -> Result<AllocatorMemoryMetrics, Self::Error>;

    /// Logical residency report augmented with backend physical observations.
    fn report_residency(&self) -> Result<ResidencyReport, Self::Error>;
}

impl<N: DenseResidencyMechanisms> crate::residency::ResidencyDriver<N> {
    /// Returns the transfer stream identity used by the native queue.
    pub fn device_stream_index(&self) -> Result<i32, N::Error> {
        N::transfer_stream_index(self.lock_storage()?.native())
    }

    /// Combines the portable point-in-time report with native observations.
    pub fn report(&self) -> Result<ResidencyReport, N::Error> {
        let report = self.residency_owner().report()?;
        let state = self.observation_lock()?;
        let facts = self
            .native_resources()
            .observe_residency(state.native(), state.controller().observation_bindings())?;
        drop(state);
        let limits = DetachedResourceLimits::new(64, 256)?;
        let observation = self.native_resources().observe_detached(limits)?;
        let detached = DetachedResourceInventory::from_observation(limits, observation)?;
        Ok(report
            .with_host_storage_resources(facts.host_storage)?
            .with_detached_resources(detached)
            .with_device_parameter_conversions(facts.conversions)?
            .with_projection_storage(facts.projection_storage)
            .with_f32_rms_normalization_gains(facts.f32_normalization_gains))
    }
}

impl<N> DenseStreamMechanisms for crate::residency::ResidencyDriver<N>
where
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,

    Self: Send + Sync,
{
    fn transfer_stream_index(&self) -> Result<i32, Self::Error> {
        self.device_stream_index()
    }

    fn settle_transfer(transfer: &mut Self::Transfer) -> Result<(), Self::Error> {
        N::settle_transfer(transfer)
    }

    fn allocator_memory(&self) -> Result<AllocatorMemoryMetrics, Self::Error> {
        self.native_resources().allocator_memory()
    }

    fn report_residency(&self) -> Result<ResidencyReport, Self::Error> {
        self.report()
    }
}

/// Records the memory samples selected by portable residency policy.
pub(crate) fn sample_residency_memory<M: DenseStreamMechanisms>(
    manager: &M,
    backend: bool,
    process: bool,
) -> Result<(), M::Error> {
    if !backend && !process {
        return Ok(());
    }

    let mut state = manager.lock_storage()?;
    let ledger = state.controller_mut().ledger_mut();
    if backend {
        ledger.record_allocator_memory(manager.allocator_memory()?);
    }
    if process {
        ledger.sample_process_metrics();
    }
    Ok(())
}

/// Failure during portable dense-stream scheduling or native transfer.
#[derive(Debug, thiserror::Error)]
pub enum DenseStreamError<E: std::error::Error + 'static> {
    /// A selected storage index is outside the retained execution units.
    #[error("dense transfer window index {index} is outside {unit_count} planned units")]
    InvalidTransferWindow {
        /// Requested storage index.
        index: usize,
        /// Number of admitted units.
        unit_count: usize,
    },
    /// There is no device transfer ready for the next invocation.
    #[error("dense transfer window has no ready transfer")]
    NoReadyTransfer,
    /// Worker-side materialization failed before demand.
    #[error("background materialization of {id} failed: {message}")]
    PrefetchFailed {
        /// Exact unit requested by demand.
        id: OffloadUnitId,
        /// Worker diagnostic retained before the consumer arrived.
        message: String,
    },
    /// A residency or native transfer primitive failed.
    #[error("dense residency operation failed: {0}")]
    Residency(#[source] E),
    /// A bounded background worker operation failed.
    #[error(transparent)]
    PrefetchWorker(#[from] BackgroundPrefetchWorkerError),
    /// A transfer schedule rejected an invalid operation.
    #[error(transparent)]
    Schedule(#[from] DenseTransferScheduleError),
    /// Dense-stream telemetry rejected a transition.
    #[error(transparent)]
    Telemetry(#[from] DenseStreamTelemetryError),
}

/// One bounded disk-to-host worker with explicit cancellation and nonjoining Drop.
pub struct BackgroundLayerPrefetch<M> {
    manager: M,
    worker: BackgroundPrefetchWorker,
}

impl<M> BackgroundLayerPrefetch<M>
where
    M: ResidencyAccess + Clone + Send + Sync + 'static,
{
    /// Starts a bounded worker over the same portable residency owner as demand.
    pub fn new(manager: M, capacity: usize) -> Result<Self, DenseStreamError<M::Error>> {
        let prefetch = manager.clone();
        let worker =
            BackgroundPrefetchWorker::new(capacity, "eredu-dense-layer-prefetch", move |id| {
                prefetch
                    .prefetch(id, MemoryTier::Host)
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            })?;
        Ok(Self {
            manager,
            worker: worker.with_nonblocking_drop(),
        })
    }

    /// Queues one unit unless it is already host-resident.
    pub fn submit(&self, id: &OffloadUnitId) -> Result<(), DenseStreamError<M::Error>> {
        let resident = self
            .manager
            .is_resident(id, MemoryTier::Host)
            .map_err(DenseStreamError::Residency)?;
        Ok(self.worker.submit(id, resident)?)
    }

    /// Waits for queued prefetch and acquires a host-resident lease.
    pub fn acquire(
        &self,
        id: &OffloadUnitId,
    ) -> Result<ResidentLease<M::Storage>, DenseStreamError<M::Error>> {
        let resolution = self.worker.wait(id)?;
        if let PrefetchDemandResolution::Failed(message) = resolution {
            return Err(DenseStreamError::PrefetchFailed {
                id: id.clone(),
                message,
            });
        }
        self.manager
            .acquire(id, MemoryTier::Host)
            .map_err(DenseStreamError::Residency)
    }

    /// Cancels queued work and fences in-flight work.
    pub fn cancel(&self) -> Result<(), DenseStreamError<M::Error>> {
        Ok(self.worker.cancel()?)
    }

    /// Returns current background-prefetch telemetry.
    pub fn report(&self) -> Result<BackgroundPrefetchReport, DenseStreamError<M::Error>> {
        Ok(self.worker.report()?)
    }

    /// Waits for the queued host work to reach a terminal result.
    pub fn wait_idle(&self) -> Result<(), DenseStreamError<M::Error>> {
        Ok(self.worker.wait_idle()?)
    }
}

/// Coordinates bounded host prefetch, device transfers, and dense-stream telemetry.
pub struct DenseStreamController<M: DenseStreamMechanisms> {
    manager: M,
    options: DenseDiskStreamLoadOptions,
    background: Option<BackgroundLayerPrefetch<M>>,
    telemetry: DenseStreamTelemetry,
}

impl<M: DenseStreamMechanisms> DenseStreamController<M> {
    /// Prepares dense lookahead and telemetry for the selected parameter layout.
    pub(crate) fn for_parameter_units(
        manager: &M,
        units: &ParameterUnitLayout,
    ) -> Result<Option<Arc<Self>>, DenseStreamError<M::Error>> {
        let Some(options) = units.residency.dense() else {
            return Ok(None);
        };
        let layout = units.invocations.storage_layout();
        let groups = (0..layout.group_count()).filter_map(|group| {
            let range = layout.group_range(group)?;
            if range.is_empty() {
                return None;
            }
            let id = layout
                .group_id(group)
                .expect("validated layout names every group");
            Some((id.as_str().to_owned(), units.unit_ids[range].to_vec()))
        });
        let metadata = &units.metadata;
        let background = (options.host_budget_bytes() > 0)
            .then(|| {
                BackgroundLayerPrefetch::new(manager.clone(), options.background_queue_capacity())
            })
            .transpose()?;
        let transfer_stream_index = manager
            .transfer_stream_index()
            .map_err(DenseStreamError::Residency)?;
        Ok(Some(Arc::new(Self {
            manager: manager.clone(),
            options,
            background,
            telemetry: DenseStreamTelemetry::new(
                metadata.layer_count(),
                metadata.layer_parameter_bytes(),
                metadata.maximum_host_layer_bytes(),
                metadata.static_device_bytes(),
                transfer_stream_index,
                groups,
            ),
        })))
    }

    /// Opens a bounded device-transfer window for selected group-local units.
    pub(crate) fn transfer_window(
        self: &Arc<Self>,
        group: impl Into<String>,
        units: &[OffloadUnitId],
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<DenseTransferWindow<M>, DenseStreamError<M::Error>> {
        let indices = indices.into_iter().collect::<Vec<_>>();
        if let Some(&index) = indices.iter().find(|&&index| index >= units.len()) {
            return Err(DenseStreamError::InvalidTransferWindow {
                index,
                unit_count: units.len(),
            });
        }
        Ok(DenseTransferWindow {
            controller: Arc::clone(self),
            group: group.into(),
            units: units.to_vec(),
            schedule: DenseTransferSchedule::new(indices, DENSE_TRANSFER_WINDOW)?,
            prefill,
        })
    }

    fn observe_group(&self, group: &str, prefill: bool) -> Result<(), DenseStreamError<M::Error>> {
        let units = self
            .manager
            .residency_snapshot()
            .map_err(DenseStreamError::Residency)?
            .units;
        self.telemetry.observe_group(group, prefill, &units)?;
        Ok(())
    }

    fn record_group_execution(&self, group: &str) -> Result<(), DenseStreamError<M::Error>> {
        self.telemetry.record_group_execution(group)?;
        Ok(())
    }

    /// Cancels prefetch and removes host/device protection for one group.
    fn clear_group(&self, group: &str) -> Result<(), DenseStreamError<M::Error>> {
        let host = self.manager.protect_group_window(
            &format!("dense:{group}:host"),
            &[],
            MemoryTier::Host,
        );
        let device = self.manager.protect_group_window(
            &format!("dense:{group}:device"),
            &[],
            MemoryTier::Device,
        );
        let prefetch = self
            .background
            .as_ref()
            .map(BackgroundLayerPrefetch::cancel);
        host.map_err(DenseStreamError::Residency)?;
        device.map_err(DenseStreamError::Residency)?;
        prefetch.transpose()?;
        Ok(())
    }

    /// Starts transactional telemetry for one model forward pass.
    pub(crate) fn forward_guard(
        self: &Arc<Self>,
        prefill: bool,
    ) -> Result<DenseStreamForwardGuard<M>, DenseStreamError<M::Error>> {
        let offload = self
            .manager
            .residency_snapshot()
            .map_err(DenseStreamError::Residency)?
            .offload;
        self.telemetry.begin_forward(prefill, &offload)?;
        Ok(DenseStreamForwardGuard {
            controller: Arc::clone(self),
            armed: true,
        })
    }

    fn commit_forward(&self) -> Result<(), DenseStreamError<M::Error>> {
        sample_residency_memory(
            &self.manager,
            self.options.samples_backend_memory(),
            self.options.samples_process_memory(),
        )
        .map_err(DenseStreamError::Residency)?;
        let offload = self
            .manager
            .residency_snapshot()
            .map_err(DenseStreamError::Residency)?
            .offload;
        self.telemetry.commit_forward(&offload)?;
        Ok(())
    }

    fn abort_forward(&self) {
        self.telemetry.abort_forward();
    }

    /// Starts cleanup and execution accounting for one group.
    pub(crate) fn group_guard(self: &Arc<Self>, group: &str) -> DenseStreamGroupGuard<M> {
        DenseStreamGroupGuard {
            controller: Arc::clone(self),
            group: group.to_string(),
            armed: true,
        }
    }

    /// Returns current dense-stream and residency telemetry.
    pub fn report(&self) -> Result<DenseDiskStreamReport, DenseStreamError<M::Error>> {
        let residency = self
            .manager
            .report_residency()
            .map_err(DenseStreamError::Residency)?;
        let background = self
            .background
            .as_ref()
            .map(BackgroundLayerPrefetch::report)
            .transpose()?
            .unwrap_or_default();
        Ok(self.telemetry.report(residency, background)?)
    }
}

/// Bounded window of independently retained, ordered device transfers.
/// A window submits at most two device copies.
/// Callers consume one entry, evaluate and synchronize its compute work, drop
/// that entry, and then call [`Self::refill`] to submit the following layer.
pub(crate) struct DenseTransferWindow<M: DenseStreamMechanisms> {
    controller: Arc<DenseStreamController<M>>,
    group: String,
    units: Vec<OffloadUnitId>,
    schedule: DenseTransferSchedule<M::Transfer>,
    prefill: bool,
}

impl<M: DenseStreamMechanisms> DenseTransferWindow<M> {
    /// Takes the scheduled invocation and its exact transfer.
    pub(crate) fn next(
        &mut self,
    ) -> Result<(DenseTransferStep, M::Transfer), DenseStreamError<M::Error>> {
        self.schedule
            .pop_ready()
            .ok_or(DenseStreamError::NoReadyTransfer)
    }

    /// A replacement can wait when consuming a ready unit can release capacity.
    pub(crate) fn can_advance_after(&self, error: &DenseStreamError<M::Error>) -> bool {
        self.schedule.has_ready()
            && matches!(error, DenseStreamError::Residency(error) if residency_capacity_exhausted(error))
    }

    /// Reprotects the current/next units and submits one replacement transfer.
    ///
    /// Retiring the prior consumer releases the transfer's lease and makes its
    /// device capacity available to the replacement.
    pub(crate) fn refill(&mut self) -> Result<(), DenseStreamError<M::Error>> {
        let device_indices = self.schedule.storage_window(DENSE_TRANSFER_WINDOW);
        let host_indices = if self.controller.background.is_some() {
            self.schedule
                .storage_window(self.controller.options.host_lookahead())
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
        self.controller
            .manager
            .protect_group_window(
                &format!("dense:{}:host", self.group),
                &host_units,
                MemoryTier::Host,
            )
            .map_err(DenseStreamError::Residency)?;
        self.controller
            .manager
            .protect_group_window(
                &format!("dense:{}:device", self.group),
                &device_units,
                MemoryTier::Device,
            )
            .map_err(DenseStreamError::Residency)?;
        if let Some(background) = &self.controller.background {
            for id in &host_units {
                background.submit(id)?;
            }
        }
        while self.schedule.can_admit() {
            let Some(step) = self.schedule.next_pending() else {
                break;
            };
            let index = step.storage_index();
            let id = &self.units[index];
            // Resolve this window's existing source transfer before the manager
            // waits for the same copy. Each occurrence retains its own lease.
            if let Some(previous) = self.schedule.ready_storage_mut(index) {
                M::settle_transfer(previous).map_err(DenseStreamError::Residency)?;
            }
            let _host = if host_indices.contains(&index) {
                self.controller
                    .background
                    .as_ref()
                    .map(|background| background.acquire(id))
                    .transpose()?
            } else {
                None
            };
            let transfer = self
                .controller
                .manager
                .acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)
                .map_err(DenseStreamError::Residency)?;
            self.schedule.admit(step, transfer)?;
        }
        self.controller.observe_group(&self.group, self.prefill)?;
        Ok(())
    }
}

/// Transactional guard for dense-stream forward telemetry.
pub(crate) struct DenseStreamForwardGuard<M: DenseStreamMechanisms> {
    controller: Arc<DenseStreamController<M>>,
    armed: bool,
}

impl<M: DenseStreamMechanisms> DenseStreamForwardGuard<M> {
    /// Commits the forward telemetry and disarms rollback-on-drop.
    pub fn complete(mut self) -> Result<(), DenseStreamError<M::Error>> {
        let result = self.controller.commit_forward();
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl<M: DenseStreamMechanisms> Drop for DenseStreamForwardGuard<M> {
    fn drop(&mut self) {
        if self.armed {
            self.controller.abort_forward();
        }
    }
}

/// Cleanup guard for one dense-stream execution group.
pub(crate) struct DenseStreamGroupGuard<M: DenseStreamMechanisms> {
    controller: Arc<DenseStreamController<M>>,
    group: String,
    armed: bool,
}

impl<M: DenseStreamMechanisms> DenseStreamGroupGuard<M> {
    /// Clears group protections and records successful execution.
    pub fn complete(mut self) -> Result<(), DenseStreamError<M::Error>> {
        let result = self
            .controller
            .clear_group(&self.group)
            .and_then(|()| self.controller.record_group_execution(&self.group));
        self.armed = false;
        result
    }
}

impl<M: DenseStreamMechanisms> Drop for DenseStreamGroupGuard<M> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.controller.clear_group(&self.group);
        }
    }
}
