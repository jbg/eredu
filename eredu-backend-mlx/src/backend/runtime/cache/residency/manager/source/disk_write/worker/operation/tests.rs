use super::*;
use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger};
#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("disk worker preparation runs no tensor equations")
    }
}
impl eredu_nn::workspace::WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("no tensor quote")
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("no tensor quote")
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("no host equation quote")
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("no host equation quote")
    }
}
fn idle(manager: &CacheResidencyManager) {
    let until = Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let state = manager.inner.state.lock().unwrap();
        if manager.borrowed_storage_complete(&state) {
            return;
        }
        drop(state);
        assert!(Instant::now() < until, "actual cache worker did not retire");
        std::thread::yield_now();
    }
}
fn source(directory: &Path) -> (CacheResidencyManager, CacheBlockId) {
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_live_disk(directory, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let id = manager
        .seal_block(
            7,
            0,
            2,
            None,
            CacheBlockArrays::KeyValue {
                keys: Array::from_slice(&[1.25f32, -2.5], &[1, 1, 2, 1]),
                values: Array::from_slice(&[3.5f32, 4.75], &[1, 1, 2, 1]),
            },
            false,
        )
        .unwrap();
    let ticket = manager.begin_device_demotion(&id).unwrap();
    manager.finish_device_demotion(&ticket).unwrap();
    drop(ticket);
    idle(&manager);
    (manager, id)
}
#[test]
#[ignore = "requires native cache Host sources"]
fn prepared_disk_write_preserves_real_host_file_occupancy_and_failed_completion_custody() {
    for collide in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let (manager, id) = source(directory.path());
        let cache_pool = manager.pool().clone();
        let host_bytes = manager.report().unwrap().current_host_bytes;
        assert!(host_bytes > 0);
        let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
        let funding = pool
            .prepare_workspace_metadata(
                &InferenceExecutionIdentity::default(),
                crate::memory_fixture::resolved_limits(1 << 24),
            )
            .unwrap();
        let context =
            WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
        let selection = CacheBlockSelection::new(7, CacheRepresentation::KeyValue, 0, 2, 0);
        let (catalog, worker) = manager
            .with_source_loan(selection, &context, |loan| {
                Ok((
                    loan.prepare_catalog(1, 1, &context)?,
                    loan.prepare_disk_worker(1, &context)?,
                ))
            })
            .unwrap();
        let catalog = catalog.install().unwrap();
        let worker = worker.install().unwrap();
        let write = manager
            .with_source_loan(selection, &context, |mut loan| {
                loan.prepare_disk_write(&id, &context)
            })
            .unwrap();
        let destination = write.body.location.path.clone();
        let file_bytes = write.body.layout.file_bytes() as u64;
        if collide {
            fs::write(&destination, b"keep existing file").unwrap();
        }
        let mut operation = worker.prepare_write(write, &context).unwrap();
        operation.submit().unwrap();
        assert!(manager.clear().is_err(), "actual source pin excludes reset");
        if collide {
            let failure = operation.finish().unwrap_err();
            assert!(std::error::Error::source(&failure).is_some());
            let output = operation.output.clone();
            drop(operation); // exact canonical rollback, while failed task survives
            assert_eq!(manager.report().unwrap().current_host_bytes, host_bytes);
            assert_eq!(fs::read(&destination).unwrap(), b"keep existing file");
            assert_eq!(cache_pool.report().unwrap().current_disk_bytes, file_bytes);
            assert!(
                manager.clear().is_err(),
                "escaped error retains the real source pin"
            );
            drop((failure, output));
            idle(&manager);
            assert_eq!(cache_pool.report().unwrap().current_disk_bytes, 0);
        } else {
            operation.finish().unwrap();
            let report = manager.report().unwrap();
            assert_eq!(report.current_host_bytes, 0);
            assert_eq!(report.disk_demotions, 1);
            assert_eq!(cache_pool.report().unwrap().current_host_bytes, host_bytes);
            let file = operation.output.result().unwrap().unwrap().clone();
            assert_eq!(
                file.writer_layout().unwrap().file_bytes() as u64,
                file_bytes
            );
            assert_eq!(cache_pool.report().unwrap().current_disk_bytes, file_bytes);
            let bytes = fs::read(file.path()).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let first = tensors.tensor("keys").unwrap();
            assert_eq!(
                first.data(),
                &[1.25f32, -2.5].map(f32::to_le_bytes).concat()
            );
            drop(operation);
            idle(&manager);
            assert_eq!(cache_pool.report().unwrap().current_host_bytes, 0);
            manager.clear().unwrap();
            assert!(file.path().exists());
            assert_eq!(cache_pool.report().unwrap().current_disk_bytes, file_bytes);
            drop(file);
            assert!(!destination.exists());
        }
        manager.clear().unwrap();
        drop((catalog, worker, context, funding, manager));
        assert_eq!(cache_pool.report().unwrap().current_disk_bytes, 0);
        // The empty prepared reservation table is a live metadata owner too.
        assert!(pool.fixture_host_charge().unwrap() > 0);
        drop(cache_pool);
        // The ordinary disk worker deliberately has nonblocking Drop. Its
        // receiver keeps the paid empty queue/registry until the real thread
        // exits; final account retirement must follow that actual owner.
        let until = Instant::now() + std::time::Duration::from_secs(5);
        while pool.fixture_host_charge().unwrap() != 0 {
            assert!(
                Instant::now() < until,
                "disk worker metadata did not retire after all callers dropped"
            );
            std::thread::yield_now();
        }
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;

#[test]
#[ignore = "requires actual native Host sources and durable file completion"]
fn ordinary_disk_write_retires_staging_independently_of_file_and_source_pin() {
    struct Proof<'a> {
        manager: &'a CacheResidencyManager,
        id: &'a CacheBlockId,
        context: &'a WorkspaceContext,
        generation: u64,
    }
    impl PreparedCacheDiskWriteSource for Proof<'_> {
        fn id(&self) -> &CacheBlockId {
            self.id
        }
        fn context(&self) -> &WorkspaceContext {
            self.context
        }
        fn validate(&self, loan: &CacheBlockSourceLoan<'_>) -> Result<usize, Exception> {
            if !loan.manager().same_catalog(self.manager) || loan.generation() != self.generation {
                return Err(Exception::from_source(CacheSourceError::Identity));
            }
            let (pins, demand) = loan
                .source_ownership(self.id)
                .map_err(Exception::from_source)?;
            if demand {
                return Err(Exception::from_source(CacheSourceError::Busy));
            }
            Ok(pins)
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let (manager, id) = source(directory.path());
    let cache_pool = manager.pool().clone();
    let host_bytes = manager.report().unwrap().current_host_bytes;
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(1 << 24),
        )
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let selection = CacheBlockSelection::new(7, CacheRepresentation::KeyValue, 0, 2, 0);
    let (catalog, worker, mut retirement, generation) = manager
        .with_source_loan(selection, &context, |loan| {
            Ok((
                loan.prepare_catalog(1, 1, &context)?,
                loan.prepare_disk_worker(1, &context)?,
                PreparedDiskWriteHostRetirement::prepare(&context, loan.publication_controls)?,
                loan.generation(),
            ))
        })
        .unwrap();
    let catalog = catalog.install().unwrap();
    let worker = worker.install().unwrap();
    let write = manager
        .with_source_loan(selection, &context, |mut loan| {
            loan.prepare_disk_write(&id, &context)
        })
        .unwrap();
    let file_bytes = u64::try_from(write.body.layout.file_bytes()).unwrap();
    let escaped = write.body.host.as_ref().unwrap().clone();
    let mut operation = worker.prepare_write(write, &context).unwrap();
    let proof = Proof {
        manager: &manager,
        id: &id,
        context: &context,
        generation,
    };
    assert!(retirement.source_controls::<Proof<'_>>().is_some());
    let before = cache_pool.report().unwrap();
    assert!(
        operation
            .retire_ordinary_host_source(&proof, &mut retirement)
            .is_err()
    );
    assert_eq!(cache_pool.report().unwrap(), before);
    assert!(!retirement.consumed);
    operation.submit().unwrap();
    operation.finish().unwrap();
    let output = operation.output.clone();
    let witness = operation
        .retire_ordinary_host_source(&proof, &mut retirement)
        .unwrap();
    witness
        .validate_buffers(&id, escaped.buffers(), &context)
        .unwrap();
    let foreign =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    assert!(
        witness
            .validate_buffers(&id, escaped.buffers(), &foreign)
            .is_err()
    );
    drop(foreign);
    let file = witness.file().clone();
    let bytes = fs::read(file.path()).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(
        tensors.tensor("keys").unwrap().data(),
        &[1.25f32, -2.5].map(f32::to_le_bytes).concat()
    );
    assert_eq!(operation.completed_source_pin_count(), 1);
    drop(witness);
    retirement.reclaim();
    assert_eq!(cache_pool.report().unwrap().current_host_bytes, host_bytes);
    drop(escaped);
    retirement.reclaim();
    let usage = cache_pool.report().unwrap();
    assert_eq!(usage.current_host_bytes, 0);
    assert_eq!(usage.current_transfer_in_flight_bytes, 0);
    assert_eq!(usage.current_disk_bytes, file_bytes);
    assert!(
        manager.clear().is_err(),
        "completed output still owns the source pin"
    );
    assert!(
        operation
            .retire_ordinary_host_source(&proof, &mut retirement)
            .is_err()
    );
    drop(proof);
    drop((operation, output));
    idle(&manager);
    manager.clear().unwrap();
    assert!(file.path().exists());
    drop(file);
    assert_eq!(cache_pool.report().unwrap().current_disk_bytes, 0);
    assert!(fs::read_dir(directory.path()).unwrap().next().is_none());
    drop((
        catalog, worker, retirement, context, funding, manager, cache_pool,
    ));
    let until = Instant::now() + std::time::Duration::from_secs(5);
    while pool.fixture_host_charge().unwrap() != 0 {
        assert!(
            Instant::now() < until,
            "completed writer metadata remains owned"
        );
        std::thread::yield_now();
    }
}
