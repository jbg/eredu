use super::*;
use crate::backend::submission_recovery::Probe;

struct TestExecution(Rc<Cell<usize>>);

impl Drop for TestExecution {
    fn drop(&mut self) {
        assert!(safemlx::can_reclaim_submission_resources());
        self.0.set(self.0.get() + 1);
    }
}

impl ErasedRealtimeExecutionContract for TestExecution {
    fn selected(&self) -> &SelectedRealtimeRealization {
        panic!("recovery fixture does not execute architecture policy")
    }
    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Err(Error::ArchitectureModel("unused fixture report".into()))
    }
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        Ok(None)
    }
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        Ok(Vec::new())
    }
    fn execute_decisions(
        &mut self,
        _: &mut MlxKeyValueState,
        _: &[crate::MlxTensor],
        _: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        _: &Stream,
    ) -> Result<(crate::MlxTensor, moshi::ForwardContext<crate::MlxTensor>), Error> {
        Err(Error::ArchitectureModel(
            "injected execution failure".into(),
        ))
    }
}

fn model(drops: Rc<Cell<usize>>) -> MlxRealtimeExecution {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let poisoned = Rc::new(Cell::new(false));
    MlxRealtimeExecution {
        artifact_identity: eredu_core::artifact::DeferredArtifactIdentity::ready(
            eredu_core::artifact::fingerprint_artifact(
                "recovery fixture",
                [eredu_core::artifact::ArtifactMemberIdentity::new(
                    "fixture", 0, [0; 32],
                )],
            )
            .unwrap(),
        ),
        metadata: LayerwiseModelMetadata::new(
            "recovery fixture",
            None,
            0,
            0,
            eredu_runtime::ExecutionResidency::FullyResident,
            0,
            0,
            0,
            0,
        ),
        payload: Rc::new(OrdinaryRetirement::new(RealtimeExecutionPayload {
            execution: Box::new(TestExecution(drops)),
            resources: Arc::new(SelectedRealtimeResources {
                inner: OrdinaryRetirement::new(RealtimeResourcePayload {
                    _store: Arc::new(eredu_checkpoint::store::MemoryWeightStore::default()),
                    world: None,
                    stream: stream.clone(),
                    _weights_stream: stream,
                    poisoned: Rc::clone(&poisoned),
                }),
            }),
        })),
        poisoned,
    }
}

struct FakeProbe(Rc<Cell<Status>>);
impl Probe for FakeProbe {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        self.0.get()
    }
}

#[test]
fn realtime_host_error_poison_prevents_reentry_after_resources_settle() {
    let mut model = model(Rc::new(Cell::new(0)));
    let error =
        model.with_submission(|_| Err::<(), _>(Error::ArchitectureModel("host failure".into())));
    assert!(error.is_err());
    crate::backend::submission_recovery::wait_for_retirement(|| {
        Rc::strong_count(&model.payload) == 1
    });
    assert_eq!(Rc::strong_count(&model.payload), 1);
    let mut entered = false;
    assert!(model
        .with_submission(|_| {
            entered = true;
            Ok(())
        })
        .is_err());
    assert!(
        !entered,
        "poison is checked before another execution operation"
    );
}

#[test]
fn realtime_unwind_poison_is_caught_without_process_termination() {
    let mut model = model(Rc::new(Cell::new(0)));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model
            .with_submission::<()>(|_| panic!("injected host unwind"))
            .unwrap();
    }));
    assert!(result.is_err());
    assert!(model.ensure_healthy().is_err());
}

#[test]
fn unresolved_realtime_work_owns_entire_execution_after_public_model_drop() {
    let drops = Rc::new(Cell::new(0));
    let model = model(Rc::clone(&drops));
    let native = Rc::new(Cell::new(Status {
        settled: false,
        failed: false,
        blocked: false,
    }));
    let recovery = Recovery::with_probe(
        RealtimeSubmissionRetention {
            payload: RefCell::new(Some(Rc::clone(&model.payload))),
            poisoned: Rc::clone(&model.poisoned),
        },
        FakeProbe(Rc::clone(&native)),
    );
    let poison = Rc::clone(&model.poisoned);
    drop(model);
    drop(recovery);
    submission_recovery::reap();
    assert_eq!(drops.get(), 0);
    native.set(Status {
        settled: true,
        failed: true,
        blocked: true,
    });
    submission_recovery::reap();
    assert_eq!(
        drops.get(),
        0,
        "terminal native retirement only stages semantic owners"
    );
    ordinary_retirement::reclaim();
    crate::backend::submission_recovery::wait_for_retirement(|| drops.get() == 1);
    assert_eq!(drops.get(), 1);
    assert!(
        poison.get(),
        "terminal recovery never unpoisons the execution"
    );
}

#[test]
fn completion_resources_share_the_executions_poison_authority() {
    let model = model(Rc::new(Cell::new(0)));
    model.completion_resources().poison();
    assert!(model.ensure_healthy().is_err());
}

#[test]
fn last_completion_resource_owner_stages_store_drop_outside_native_retirement() {
    use eredu_checkpoint::store::{
        CheckpointLease, StoreError, TensorMetadata, TensorReadRequest, WeightStoreDiagnostics,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Store {
        inner: eredu_checkpoint::store::MemoryWeightStore,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Store {
        fn drop(&mut self) {
            assert!(safemlx::can_reclaim_submission_resources());
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl CheckpointSource for Store {
        fn source_keys(&self) -> Vec<String> {
            self.inner.source_keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.inner.source_metadata(key)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.inner.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.inner.source_diagnostics()
        }
    }

    ordinary_retirement::reclaim();
    let mut model = model(Rc::new(Cell::new(0)));
    let drops = Arc::new(AtomicUsize::new(0));
    let payload = Rc::get_mut(&mut model.payload).unwrap();
    Arc::get_mut(&mut payload.resources).unwrap().inner._store = Arc::new(Store {
        inner: Default::default(),
        drops: Arc::clone(&drops),
    });
    let resources = model.completion_resources();
    drop(model);
    ordinary_retirement::reclaim();
    assert_eq!(Arc::strong_count(&resources), 1);
    let mut resources = Some(resources);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::try_with_submission_retirement(|| {
            drop(resources.take());
            ordinary_retirement::reclaim();
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        })
        .is_some()
    });
    ordinary_retirement::reclaim();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
