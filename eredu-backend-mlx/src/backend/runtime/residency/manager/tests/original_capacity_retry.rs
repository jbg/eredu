// Invoked by the genuine composition-admission fixture. This helper supplies
// actual source/cache behavior; it cannot manufacture original scope authority.
pub(crate) fn exercise_original_capacity_retry(
    custody: &eredu_runtime::working_memory::OriginalTextControlGuard,
    observer: &safemlx::OriginalScopeObserver,
    source_stream: Stream,
    device_stream: Stream,
) {
    use crate::backend::runtime::checkpoint::store::{
        MaterializationPayloadShape, OriginalMaterializationSlots,
        PreparedMaterializationObservation, PreparedPendingWeight, PreparedWeightMaterialization,
    };
    use crate::backend::submission_recovery::observed::bank::PreparedOperationBank;
    use eredu_checkpoint::store::{
        CheckpointLease, EncodedReadBatch, EncodedTensorLease, ReadPolicy, StoreError,
        TensorMetadata, TensorReadRequest, WeightStoreDiagnostics,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // A real one-shard store supplies the cause. This adapter deliberately
    // chooses the ordinary encoded-lease path, and repeats the captured actual
    // right-reader refusal during its direct-plan query. The independent left
    // lease remains held through both probes. Planning reads no payload bytes.
    struct Source {
        inner: Arc<SafetensorsWeightStore>,
        armed: AtomicBool,
        preparing: AtomicBool,
        manager: std::sync::Mutex<Option<ManagerWeak>>,
        right_attempts: AtomicUsize,
        left_acquisitions: AtomicUsize,
        right_refusal: StoreError,
    }
    fn request(key: &str) -> TensorReadRequest {
        TensorReadRequest {
            key: key.to_owned(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        }
    }
    impl CheckpointSource for Source {
        fn prepared_acquisition_source(
            &self,
        ) -> eredu_checkpoint::store::PreparedAcquisitionSource<'_> {
            if self.preparing.load(Ordering::SeqCst) {
                let owner = self
                    .manager
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap();
                assert!(
                    owner.state.try_lock().is_ok(),
                    "cold source callbacks must run after the manager loan ends"
                );
            }
            self.inner.prepared_acquisition_source()
        }

        fn source_keys(&self) -> Vec<String> {
            self.inner.source_keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.inner.source_metadata(key)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            if self.armed.load(Ordering::SeqCst) && request.key == "left" {
                self.left_acquisitions.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.inner.source_diagnostics()
        }
        fn prepare_encoded_read(
            &self,
            keys: &[String],
        ) -> Result<Option<EncodedReadBatch>, StoreError> {
            if self.armed.load(Ordering::SeqCst) && keys == ["right"] {
                self.right_attempts.fetch_add(1, Ordering::SeqCst);
                // Inject the actual store cause captured while the same
                // independent left lease was held, rather than performing a
                // payload acquisition in the metadata-only planning method.
                return Err(self.right_refusal.clone());
            }
            Ok(None)
        }
    }
    fn bank<S>(count: usize, mut slot: impl FnMut() -> S) -> PreparedOperationBank<S> {
        PreparedOperationBank::try_new(count, |_| Ok::<_, std::convert::Infallible>(slot()))
            .unwrap()
    }
    fn values(lease: &CheckpointLease) -> Vec<i32> {
        lease
            .encoded_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| i32::from_ne_bytes(*bytes))
            .collect()
    }

    let (_directory, inner) = cross_shard_store();
    let external = inner.acquire_lease(request("left")).unwrap();
    assert_eq!(values(&external), [1, 2]);
    let left_path = external.backing_path().unwrap().to_path_buf();
    let right_refusal = inner.acquire_lease(request("right")).unwrap_err();
    assert!(
        matches!(&right_refusal, StoreError::CapacityExhausted { maximum: 1, leased } if leased == &[left_path.clone()])
    );
    let source = Arc::new(Source {
        inner: Arc::clone(&inner),
        armed: AtomicBool::new(false),
        preparing: AtomicBool::new(false),
        manager: std::sync::Mutex::new(None),
        right_attempts: AtomicUsize::new(0),
        left_acquisitions: AtomicUsize::new(0),
        right_refusal,
    });
    let manager = ResidencyManager::new(
        Arc::clone(&source),
        OffloadPlan::new(
            OffloadConfig::new(Some(16), Some(0), 1).unwrap(),
            [
                spec("0.left", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("1.right", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
        )
        .unwrap(),
        [single("0.left", "left"), single("1.right", "right")],
        source_stream,
        device_stream,
    )
    .unwrap();
    manager.initialize().unwrap();

    // These exact test slots expose the real checkout cardinality. This is not
    // a claim that the test's additional manager is admitted by the model Q.
    let roots = [id("0.left"), id("1.right")];
    let mut named_scratch = [eredu_runtime::residency::ResidencyClosureSlot::default(); 2];
    let catalog = manager
        .prepare_name_catalog(&roots, &mut named_scratch, custody.metadata_custody().into())
        .unwrap_or_else(|_| panic!("prepare exact test names"));
    let mut transfers = bank(2, || {
        PreparedResidentTransfer::try_new_for_window(
            custody.metadata_custody().into(),
            TransferPayloadShape {
                unit_ids: 2,
                prepared_units: 2,
                unit_id_bytes: roots.iter().map(|id| id.as_str().len()).sum(),
                leases: roots.len(),
                lease_id_bytes: roots.iter().map(|id| id.as_str().len()).sum(),
                pending_sources: 1,
                retained_arrays: 2,
                ..Default::default()
            },
            &manager,
            &catalog,
            &roots,
            &mut named_scratch,
        )
        .unwrap_or_else(|_| panic!("prepare test transfer payload"))
    });
    let mut observations = bank(0, || PreparedTransferObservation::new(custody.metadata_custody().into()));
    let mut host = bank(0, || PreparedHostMaterialization::new(custody.metadata_custody().into()));
    let mut pending = bank(1, || PreparedPendingWeight::new(custody.clone()));
    let mut weights = bank(1, || {
        PreparedWeightMaterialization::try_new(
            custody.clone(),
            MaterializationPayloadShape {
                inputs: 0,
                outputs: 2,
                pending_sources: 1,
            },
        )
        .unwrap_or_else(|_| panic!("prepare test detach payload"))
    });
    let mut consumers = bank(0, || {
        PreparedMaterializationObservation::new(custody.clone())
    });
    let mut closure = [eredu_runtime::residency::ResidencyClosureSlot::default(); 2];
    let mut closure_ids = Some(
        manager
            .prepare_closure_ids(
                &roots,
                &mut closure,
                roots.len(),
                roots.iter().map(|id| id.as_str().len()).sum(),
                custody.metadata_custody().into(),
            )
            .unwrap_or_else(|_| panic!("prepare actual closure IDs")),
    );
    let mut controller =
        Some(manager.prepare_controller_for_test(&roots, &mut closure, custody.clone()));
    *source.manager.lock().unwrap() = Some(manager.inner.downgrade());
    source.preparing.store(true, Ordering::SeqCst);
    let mut acquisitions = manager
        .prepare_source_acquisitions(&roots, &mut closure, custody.clone())
        .unwrap();
    source.preparing.store(false, Ordering::SeqCst);
    let acquisition_count = acquisitions.remaining();
    source.armed.store(true, Ordering::SeqCst);
    let error = manager
        .acquire_many_with_original_transfer(
            &[(id("0.left"), 1), (id("1.right"), 1)],
            MemoryTier::Device,
            &mut OriginalResidencySlots {
            background_host: None,
                controller: &mut controller,
                closure_ids: &mut closure_ids,
                closure: &mut closure,
                transfers: &mut transfers,
                observations: &mut observations,
                host_materializations: &mut host,
                reservation: None,
                foreground_disk: &mut crate::backend::runtime::residency::manager::PreparedForegroundDiskSlots::unavailable(),
                materialization: OriginalMaterializationSlots {
                    acquisitions: &mut acquisitions,
                    pending_weights: &mut pending,
                    weight_materializations: &mut weights,
                    observations: &mut consumers,
                },
            },
            observer,
        )
        .err()
        .expect("the independently retained shard still excludes right");
    assert!(
        closure_ids.is_none(),
        "source failure follows closure destination take"
    );
    match &error {
        ResidencyError::Recipe {
            source:
                WeightRecipeError::Neutral(eredu_checkpoint::recipe::RecipeError::Store(
                    StoreError::CapacityExhausted { maximum, leased },
                )),
            ..
        } => {
            assert_eq!(*maximum, 1);
            assert_eq!(leased, &[left_path]);
        }
        other => panic!("actual capacity cause was replaced: {other}; debug: {other:?}"),
    }
    assert_eq!(source.right_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        source.left_acquisitions.load(Ordering::SeqCst),
        0,
        "explicit prepared path does not call ordinary acquisition"
    );
    assert_eq!(
        acquisitions.remaining(),
        acquisition_count - 1,
        "one actual left destination consumed; failed right planning skips its slots"
    );
    assert_eq!(transfers.remaining(), 0);
    assert_eq!(pending.remaining(), 0);
    assert_eq!(weights.remaining(), 0, "one real detach submission");
    assert!(!manager.inner.failed_transfer.load(Ordering::Acquire));
    {
        let state = manager.lock_original(observer).unwrap();
        assert!(state.storage.values().all(|unit| unit.device.is_none()));
        assert_eq!(
            state
                .control
                .ledger()
                .telemetry()
                .resident_bytes()
                .get(MemoryTier::Device),
            0
        );
    }
    assert_eq!(
        values(&external),
        [1, 2],
        "retry must not consume the independent owner"
    );
    assert!(matches!(
        inner.acquire_lease(request("right")),
        Err(StoreError::CapacityExhausted { .. })
    ));
    drop(error);
    drop(external);
    // Success now proves the pending source from the first unit was actually
    // released by the explicit detach. No global reaper runs between the failed
    // acquisition and this probe; fixture initialization happened beforehand.
    let right = inner.acquire_lease(request("right")).unwrap();
    assert_eq!(values(&right), [3, 4]);
    assert!(inner.source_diagnostics().unwrap().evictions >= 1);
    drop(right);
    let (outcome, status) = observer.progress().unwrap();
    assert_eq!(outcome, safemlx::ScopedSubmissionProgress::Observed);
    assert!(status.is_settled() && !status.failed() && !status.blocked());
    assert_eq!(
        observer.retire_completed_records().unwrap(),
        safemlx::SubmissionRetirement::CompleteSnapshot
    );
}
