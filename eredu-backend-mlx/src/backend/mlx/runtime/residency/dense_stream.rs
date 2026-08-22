//! Experimental bounded streaming of dense execution units from safetensors.
//!
//! The worker in this module performs disk-to-host materialization. Device
//! promotion uses a dedicated MLX stream and a caller-owned two-layer transfer
//! window. The current layer's completion orders its compute stream while the
//! next layer may continue transferring independently.

use std::sync::Arc;

use crate::backend::mlx::runtime::residency::manager::{ResidencyManager, ResidentUnitLease};
use eredu_core::residency::{
    BackgroundPrefetchReport, MemoryTier, OffloadUnitId, PrefetchDemandResolution,
};
use eredu_runtime::BackgroundPrefetchWorker;

type HostPrefetchOperation =
    Arc<dyn Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static>;

/// One bounded, deterministically joined disk-to-host worker.
pub struct BackgroundLayerPrefetch {
    manager: ResidencyManager,
    worker: BackgroundPrefetchWorker,
}

impl BackgroundLayerPrefetch {
    pub fn new(manager: ResidencyManager, capacity: usize) -> Result<Self, DenseStreamError> {
        let worker_manager = manager.clone();
        let operation = Arc::new(move |id: &OffloadUnitId| {
            worker_manager
                .prefetch(id, MemoryTier::Host)
                .map(|_| ())
                .map_err(|error| error.to_string())
        });
        Self::new_with_operation(manager, capacity, operation)
    }

    fn new_with_operation(
        manager: ResidencyManager,
        capacity: usize,
        operation: HostPrefetchOperation,
    ) -> Result<Self, DenseStreamError> {
        let worker =
            BackgroundPrefetchWorker::new(capacity, "safemlx-dense-layer-prefetch", move |id| {
                operation(id)
            })?;
        Ok(Self { manager, worker })
    }

    pub fn submit(&self, id: &OffloadUnitId) -> Result<(), DenseStreamError> {
        let resident = self.manager.is_resident(id, MemoryTier::Host)?;
        Ok(self.worker.submit(id, resident)?)
    }

    pub fn acquire(&self, id: &OffloadUnitId) -> Result<ResidentUnitLease, DenseStreamError> {
        let resolution = self.worker.wait(id)?;
        if let PrefetchDemandResolution::Failed(message) = resolution {
            return Err(DenseStreamError::PrefetchFailed {
                id: id.clone(),
                message,
            });
        }
        Ok(self.manager.acquire(id, MemoryTier::Host)?)
    }

    pub fn cancel(&self) -> Result<(), DenseStreamError> {
        Ok(self.worker.cancel()?)
    }

    pub fn report(&self) -> Result<BackgroundPrefetchReport, DenseStreamError> {
        Ok(self.worker.report()?)
    }

    #[cfg(test)]
    fn wait_idle(&self) -> Result<(), DenseStreamError> {
        Ok(self.worker.wait_idle()?)
    }
}

/// Structured validation and worker failures for dense disk streaming.
#[derive(Debug, thiserror::Error)]
pub enum DenseStreamError {
    /// Shared dense-stream telemetry state was poisoned.
    #[error("dense streaming state is poisoned")]
    StatePoisoned,
    /// Forward telemetry lifecycle calls were inconsistent.
    #[error("invalid dense streaming forward telemetry state: {0}")]
    InvalidForwardTelemetry(&'static str),
    /// A worker-side materialization failed and was observed by demand.
    #[error("background materialization of {id} failed: {message}")]
    PrefetchFailed {
        /// Failed unit.
        id: OffloadUnitId,
        /// Original residency error.
        message: String,
    },
    /// A residency transition failed.
    #[error(transparent)]
    Residency(#[from] crate::backend::mlx::runtime::residency::manager::ResidencyError),
    /// Backend-neutral background-prefetch execution failed.
    #[error(transparent)]
    PrefetchWorker(#[from] eredu_runtime::BackgroundPrefetchWorkerError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mlx::runtime::checkpoint::store::{
        SafetensorsWeightStore, TensorSelection,
    };
    use eredu_core::residency::{OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy};
    use eredu_runtime::{OffloadUnit, WeightBinding};
    use safemlx::{
        host_transfer_capacity_upper_bound, Device, DeviceType, HostTransferPolicy, Stream,
    };
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::{
        sync::{mpsc, Condvar, Mutex},
        thread,
    };

    #[derive(Default)]
    struct OperationGate {
        state: Mutex<(usize, bool)>,
        changed: Condvar,
    }

    impl OperationGate {
        fn enter_and_wait(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            self.changed.notify_all();
            while !state.1 {
                state = self.changed.wait(state).unwrap();
            }
        }

        fn wait_for_starts(&self, expected: usize) {
            let mut state = self.state.lock().unwrap();
            while state.0 < expected {
                state = self.changed.wait(state).unwrap();
            }
        }

        fn release(&self) {
            let mut state = self.state.lock().unwrap();
            state.1 = true;
            self.changed.notify_all();
        }
    }

    fn test_manager(
        tensors: Vec<(String, Dtype, Vec<u8>)>,
        host_budget: u64,
    ) -> (tempfile::TempDir, ResidencyManager, Vec<OffloadUnitId>) {
        let directory = tempfile::tempdir().unwrap();
        serialize_to_file(
            tensors.iter().map(|(name, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, vec![2], bytes).unwrap(),
                )
            }),
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let mut specs = Vec::new();
        let mut definitions = Vec::new();
        let mut ids = Vec::new();
        for (index, (key, _, bytes)) in tensors.iter().enumerate() {
            let id = OffloadUnitId::new(format!("layer.{index}")).unwrap();
            let expected_bytes = bytes.len() as u64;
            let binding =
                WeightBinding::new("weight", key, TensorSelection::Full, expected_bytes).unwrap();
            specs.push(
                OffloadUnitSpec::new(
                    id.clone(),
                    expected_bytes,
                    ResidencyPolicy::Cacheable,
                    MemoryTier::Disk,
                )
                .unwrap(),
            );
            definitions.push(OffloadUnit::new(id.clone(), [binding]).unwrap());
            ids.push(id);
        }
        let binding_capacity =
            host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64;
        let physical_host_budget = (host_budget / 8).checked_mul(binding_capacity).unwrap();
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(u64::MAX), Some(physical_host_budget), 1).unwrap(),
            specs,
        )
        .unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let manager =
            ResidencyManager::new(store, plan, definitions, stream.clone(), stream).unwrap();
        manager.initialize().unwrap();
        (directory, manager, ids)
    }

    fn i32_manager(
        count: usize,
        host_budget: u64,
    ) -> (tempfile::TempDir, ResidencyManager, Vec<OffloadUnitId>) {
        test_manager(
            (0..count)
                .map(|index| {
                    (
                        format!("weight.{index}"),
                        Dtype::I32,
                        [index as i32, index as i32 + 1]
                            .into_iter()
                            .flat_map(i32::to_le_bytes)
                            .collect(),
                    )
                })
                .collect(),
            host_budget,
        )
    }

    fn blocking_operation(
        manager: ResidencyManager,
        gate: Arc<OperationGate>,
    ) -> HostPrefetchOperation {
        Arc::new(move |id| {
            gate.enter_and_wait();
            manager
                .prefetch(id, MemoryTier::Host)
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
    }

    #[test]
    fn duplicate_background_requests_coalesce_and_join_demand() {
        let directory = tempfile::tempdir().unwrap();
        let bytes = [1i32, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::I32, vec![2], &bytes).unwrap(),
            )],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let id = OffloadUnitId::new("layer.0").unwrap();
        let binding = WeightBinding::new("weight", "weight", TensorSelection::Full, 8).unwrap();
        let host_capacity =
            host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64;
        let plan = OffloadPlan::new(
            OffloadConfig::new(Some(8), Some(host_capacity), 1).unwrap(),
            [
                OffloadUnitSpec::new(id.clone(), 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                    .unwrap(),
            ],
        )
        .unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let manager = ResidencyManager::new(
            store,
            plan,
            [OffloadUnit::new(id.clone(), [binding]).unwrap()],
            stream.clone(),
            stream,
        )
        .unwrap();
        manager.initialize().unwrap();

        let prefetch = BackgroundLayerPrefetch::new(manager, 1).unwrap();
        prefetch.submit(&id).unwrap();
        prefetch.submit(&id).unwrap();
        let lease = prefetch.acquire(&id).unwrap();
        assert_eq!(lease.host_value("weight").unwrap().shape().unwrap(), [2]);
        let report = prefetch.report().unwrap();
        assert_eq!(report.submitted(), 1);
        assert!(report.coalesced() >= 1);
        assert_eq!(report.completed(), 1);
    }

    #[test]
    fn bounded_queue_backpressures_without_dropping_required_work() {
        let (_directory, manager, ids) = i32_manager(3, 24);
        let gate = Arc::new(OperationGate::default());
        let prefetch = Arc::new(
            BackgroundLayerPrefetch::new_with_operation(
                manager.clone(),
                1,
                blocking_operation(manager, Arc::clone(&gate)),
            )
            .unwrap(),
        );
        prefetch.submit(&ids[0]).unwrap();
        gate.wait_for_starts(1);
        prefetch.submit(&ids[1]).unwrap();

        let (started_tx, started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let submitting = Arc::clone(&prefetch);
        let required = ids[2].clone();
        let submitter = thread::spawn(move || {
            started_tx.send(()).unwrap();
            let outcome = submitting.submit(&required);
            result_tx.send(outcome).unwrap();
        });
        started_rx.recv().unwrap();
        gate.release();
        result_rx.recv().unwrap().unwrap();
        submitter.join().unwrap();
        prefetch.wait_idle().unwrap();

        let lease = prefetch.acquire(&ids[2]).unwrap();
        assert_eq!(lease.host_value("weight").unwrap().shape().unwrap(), [2]);
        let report = prefetch.report().unwrap();
        assert_eq!(report.queue_capacity(), 1);
        assert_eq!(report.peak_queue_occupancy(), 1);
        assert_eq!(report.submitted(), 3);
        assert_eq!(report.started(), 3);
        assert_eq!(report.completed(), 3);
        assert_eq!(report.failed(), 0);
        assert!(report.backpressure_count() >= 1);
    }

    #[test]
    fn worker_errors_and_panics_reach_demand_and_release_reservations() {
        let unsupported = vec![("unsupported".to_string(), Dtype::F8_E5M2, vec![0x3c, 0x40])];
        let (_directory, manager, ids) = test_manager(unsupported, 2);
        let prefetch = BackgroundLayerPrefetch::new(manager.clone(), 1).unwrap();
        prefetch.submit(&ids[0]).unwrap();
        let error = match prefetch.acquire(&ids[0]) {
            Ok(_) => panic!("unsupported stored dtype unexpectedly prefetched"),
            Err(error) => error,
        };
        assert!(matches!(error, DenseStreamError::PrefetchFailed { .. }));
        let report = manager.report().unwrap();
        assert_eq!(report.offload().resident_bytes().get(MemoryTier::Host), 0);
        assert!(!report.units()[0].host_resident());
        assert_eq!(prefetch.report().unwrap().failed(), 1);

        let (_directory, manager, ids) = i32_manager(1, 8);
        let operation: HostPrefetchOperation = Arc::new(|_| -> Result<(), String> {
            panic!("controlled worker panic");
        });
        let prefetch =
            BackgroundLayerPrefetch::new_with_operation(manager.clone(), 1, operation).unwrap();
        prefetch.submit(&ids[0]).unwrap();
        let error = match prefetch.acquire(&ids[0]) {
            Ok(_) => panic!("panicking operation unexpectedly prefetched"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("controlled worker panic"));
        assert_eq!(prefetch.report().unwrap().failed(), 1);
        assert!(!manager.is_resident(&ids[0], MemoryTier::Host).unwrap());
    }

    #[test]
    fn completed_prefetch_evicted_before_demand_is_counted() {
        let (_directory, manager, ids) = i32_manager(2, 8);
        let prefetch = BackgroundLayerPrefetch::new(manager.clone(), 1).unwrap();
        prefetch.submit(&ids[0]).unwrap();
        prefetch.wait_idle().unwrap();
        prefetch.submit(&ids[1]).unwrap();
        prefetch.wait_idle().unwrap();
        assert!(!manager.is_resident(&ids[0], MemoryTier::Host).unwrap());

        prefetch.submit(&ids[0]).unwrap();
        prefetch.wait_idle().unwrap();
        assert_eq!(prefetch.report().unwrap().evicted_before_use(), 1);
    }
}
