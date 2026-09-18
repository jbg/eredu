//! Real original C publication, sealed finite S, and canonical SessionPrefill.
//! Payloads are neutral host fixtures; no native allocation/completeness claim.
use super::*;
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PublicationKey {
    Capture(eredu_core::SharedStorageIdentity),
    Payload(Key),
}
impl CapturePlanStorageKey for PublicationKey {
    fn capture_plan_identity(&self) -> Option<&eredu_core::SharedStorageIdentity> {
        match self {
            Self::Capture(k) => Some(k),
            Self::Payload(_) => None,
        }
    }
}
fn pkey(key: &Key) -> PublicationKey {
    PublicationKey::Payload(key.clone())
}
fn publication_quote(
    pool: &WorkingMemoryPool,
    source: &SharedCapturePlan,
    root: &Key,
    g: InferenceGeometry,
    slots: usize,
) -> IncrementalInferenceQuote {
    let context = WorkspaceContext::new(Facts);
    let backing = WorkspaceExistingStorage::new(Some(64), &context);
    let tensor = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap(),
        &backing,
        &context,
    )
    .unwrap();
    let storage =
        RegisteredWorkspaceStorage::bind(pool, &context, [(pkey(root), backing)]).unwrap();
    let report = quote_inference_workspace(g, |_| {
        context.begin_state_span([&tensor])?;
        context.report(&[tensor.clone()])
    })
    .unwrap();
    let b = |n| WorkspaceBound::bounded(n, "actual neutral publication fixture envelope");
    let outside = ExecutionWorkspaceEstimate {
        geometry: g,
        activations: b(384),
        attention: b(0),
        vocabulary: b(0),
        state_update: b(0),
        materialization: b(0),
        retained: b(CaptureRunHostPlan::prepare(source)
            .unwrap()
            .initialization_peak_bytes()),
    };
    let q = ResidualInferenceQuote::compose(
        &report,
        mock_inference_admission(g).state,
        outside,
        &storage,
    )
    .unwrap()
    .into_incremental();
    let publication = PreparedCapturePlanPublication::prepare(
        pool,
        q.span_workspace().plan(),
        source,
        PublicationKey::Capture(source.storage_identity().clone()),
        None,
    )
    .unwrap();
    let rows = PreparedPrefillStoragePublicationPlan::<PublicationKey>::prepare(
        q.span_workspace().plan(),
        |_| Some(slots),
    )
    .unwrap();
    let s = rows.control_peak_bytes();
    let controls = PreparedTextControlWorkspace::prepare(
        source,
        g,
        q.span_workspace().plan(),
        TextHostControlFacts::new(Some(11), Some(17), Some(23)),
    )
    .unwrap()
    .with_capture_plan_publication(publication)
    .unwrap()
    .with_prefill_storage_publications(rows)
    .unwrap();
    assert_eq!(controls.storage_publication_control_bytes(), s);
    q.with_span_workspace_and_text_controls(controls).unwrap()
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublicationMode {
    Success,
    Empty,
    Capacity,
    DuplicateCapacity,
    Budget,
    Foreign,
    Busy,
    Panic,
    LateHealth,
    LateZero,
}
struct PublicationBackend {
    scope: Option<WorkingMemoryFundingScope>,
    sibling: Option<WorkingMemoryFundingScope>,
    segment: Option<CaptureSourceSegment>,
    slots: OriginalPrefillStoragePublicationSlots<PublicationKey>,
    attempt: Option<BoundedPublicationAttempt<PublicationKey>>,
    keys: Vec<Key>,
    outputs: Vec<BoundedPublishedAllocation<PublicationKey>>,
    ordinary: Vec<WorkingMemoryStorage<PublicationKey>>,
    origin: Option<WorkingMemoryFundingScope>,
    mode: PublicationMode,
    failure: Option<BoundedPublicationError>,
    retired: usize,
    contention: Option<Contention>,
}
impl ScheduledCaptureBackend for PublicationBackend {
    type Tensor = FakeTensor;
    type Error = Error;
    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, FundedCaptureError<Error>> {
        let scope = self.scope.as_mut().unwrap();
        let (segment, registration) = bootstrap
            .begin_segment(scope, context)
            .map_err(CaptureRunHostError::from)?;
        self.segment = Some(segment);
        let segment = self.segment.as_ref().unwrap();
        if self.mode == PublicationMode::Foreign {
            assert!(matches!(
                self.slots
                    .begin(context, self.sibling.as_ref().unwrap(), segment),
                Err(BoundedPublicationError::Storage(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert_eq!(
                self.slots.spent_rows(),
                context.chunk().input.start as usize
            );
        }
        self.attempt = Some(self.slots.begin(context, scope, segment).unwrap());
        let attempt = self.attempt.as_mut().unwrap();
        assert!(
            attempt.published_input_index(0).is_none(),
            "pending inputs have no successful canonical mapping"
        );
        if self.mode != PublicationMode::Empty {
            // Mixed legacy alias + new/fixed alias + genuine zero-byte key.
            attempt.push_owned(pkey(&self.keys[0]), 64).unwrap();
            attempt
                .push_owned(
                    pkey(&self.keys[1]),
                    if self.mode == PublicationMode::Budget {
                        100_000
                    } else {
                        32
                    },
                )
                .unwrap();
            attempt.push_owned(pkey(&self.keys[2]), 0).unwrap();
            attempt
                .push_owned(
                    pkey(&self.keys[0]),
                    if self.mode == PublicationMode::DuplicateCapacity {
                        63
                    } else {
                        64
                    },
                )
                .unwrap();
            if self.mode == PublicationMode::Capacity {
                attempt.push_owned(pkey(&self.keys[3]), 17).unwrap();
            } else {
                attempt.push_owned(pkey(&self.keys[3]), 16).unwrap();
            }
            let rejected = attempt.push_owned(pkey(&self.keys[0]), 64).unwrap_err();
            assert_eq!(rejected, pkey(&self.keys[0]));
        }
        if matches!(
            self.mode,
            PublicationMode::LateHealth | PublicationMode::LateZero
        ) {
            drop(self.origin.take());
        }
        if self.mode == PublicationMode::Panic {
            self.keys[0]
                .probe
                .panic_after
                .store(1, AtomicOrdering::SeqCst);
        }
        let used = scope.pool().used_bytes().unwrap();
        let peak = scope.pool().peak_bytes().unwrap();
        let worker = self.contention.as_ref().map(|c| {
            c.key.0.armed.store(true, AtomicOrdering::SeqCst);
            let key = c.key.clone();
            let pool = scope.pool().clone();
            let worker =
                std::thread::spawn(move || pool.pin_registered_storage([(key, 1)]).unwrap());
            c.entered
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            worker
        });
        let result = attempt.publish(scope, segment);
        if let Some(worker) = worker {
            self.contention.as_ref().unwrap().release.send(()).unwrap();
            drop(worker.join().unwrap());
        }
        assert_eq!(
            scope.pool().used_bytes().unwrap(),
            used,
            "adoption converts original charge only"
        );
        assert_eq!(scope.pool().peak_bytes().unwrap(), peak);
        match result {
            Ok(()) => {
                assert_eq!(
                    attempt.published_count(),
                    if self.mode == PublicationMode::Empty {
                        0
                    } else {
                        4
                    }
                );
                assert!(matches!(
                    attempt.publish(scope, segment),
                    Err(BoundedPublicationError::Storage(
                        WorkingMemoryError::PreparationAlreadyStarted
                    ))
                ));
                if self.mode != PublicationMode::Empty {
                    assert_eq!(
                        (0..4)
                            .map(|i| attempt.published_input_index(i))
                            .collect::<Vec<_>>(),
                        [Some(0), Some(1), Some(2), Some(4)],
                        "duplicate input 3 maps to the first input's allocation"
                    );
                    assert!(attempt.published_input_index(4).is_none());
                    for i in 0..4 {
                        let output = attempt.take_allocation(i).unwrap();
                        output.validate_source(scope.pool()).unwrap();
                        self.outputs.push(output.clone());
                        self.outputs.push(output);
                        assert!(attempt.take_allocation(i).is_none());
                        assert_eq!(attempt.published_input_index(i), Some([0, 1, 2, 4][i]));
                    }
                    // Both ordinary grouped and per-key aliases use the same fixed
                    // canonical entries; these may outlive every bounded wrapper.
                    self.ordinary.push(
                        scope
                            .pool()
                            .pin_registered_storage([
                                (pkey(&self.keys[1]), 32),
                                (pkey(&self.keys[2]), 0),
                            ])
                            .unwrap(),
                    );
                    self.ordinary.extend(
                        scope
                            .pool()
                            .register_storage_individually([(pkey(&self.keys[1]), 32)])
                            .unwrap()
                            .into_values(),
                    );
                }
                self.attempt.take();
                Ok(Some(registration))
            }
            Err(error) => {
                assert_eq!(attempt.published_count(), 0);
                assert!(attempt.published_input_index(0).is_none());
                assert!(attempt.take_allocation(0).is_none());
                assert_eq!(attempt.retained_input_count(), 5);
                assert!(
                    scope
                        .pool()
                        .pin_registered_storage([(
                            pkey(&self.keys[1]),
                            if self.mode == PublicationMode::Budget {
                                100_000
                            } else {
                                32
                            }
                        )])
                        .is_err(),
                    "first new key did not leak from rejected mixed batch"
                );
                if self.mode == PublicationMode::LateZero {
                    // This real zero-byte entry preceded the batch. Ordinary
                    // pinning retains its existing registration; the bounded
                    // publication's separate origin-health check rejected it.
                    let prior = scope
                        .pool()
                        .pin_registered_storage([(pkey(&self.keys[2]), 0)])
                        .unwrap();
                    assert_eq!(prior.bytes(), 0);
                    drop(prior);
                } else {
                    assert!(scope
                        .pool()
                        .pin_registered_storage([(pkey(&self.keys[2]), 0)])
                        .is_err());
                }
                self.failure = Some(error);
                Err(FundedCaptureError::Backend(Error::backend_retained_source(
                    Original(Arc::new(())),
                )))
            }
        }
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Error>> {
        let parcel = self
            .segment
            .as_ref()
            .unwrap()
            .take_settled_sources(self.scope.as_mut().unwrap(), &ticket)
            .map_err(CaptureRunHostError::from)?;
        drop(parcel);
        self.segment.take();
        self.retired += 1;
        Ok(())
    }
    fn validate_source(
        &self,
        _: &FakeTensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<checkpoint::TensorDtype, Error> {
        Ok(checkpoint::TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &FakeTensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        unreachable!()
    }
    fn transform(
        &mut self,
        _: &FakeTensor,
        _: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        unreachable!()
    }
    fn validate_prefill_source(
        &self,
        _: &FakeTensor,
        _: &CapturePrefillFragment<'_, '_>,
    ) -> Result<checkpoint::TensorDtype, FundedCaptureError<Error>> {
        Ok(checkpoint::TensorDtype::F32)
    }
    fn estimate_prefill(
        &self,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 12,
            host_bytes: 12,
            encoded_bytes: 4096,
        })
    }
}
fn exercise_publication(mode: PublicationMode, rows: u64) {
    INFERENCE_SCOPE_TRACE.with(|s| {
        *s.borrow_mut() = InferenceScopeTrace {
            required: true,
            ..Default::default()
        }
    });
    let (mut session, _) = session();
    let g = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: rows,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let source = source(g);
    let c = source.capacity_bytes().unwrap();
    let ckey = PublicationKey::Capture(source.storage_identity().clone());
    let runtime = eredu_runtime::ResidentRuntime::<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >::new(
        OrdinaryTextFixture {
            static_modules: FakeOperator,
            trace: vec![],
            counters: Default::default(),
            inconsistent_transport: false,
            inconsistent_identity: false,
        },
        &(),
    )
    .unwrap();
    let paths = runtime.prepare_observation_paths().unwrap();
    let selected = paths.source().prepare_capture_selection(&source).unwrap();
    let bound = selected.bind_geometry(g).unwrap();
    let pool = WorkingMemoryPool::new(8_000_000, 0).unwrap();
    let keys = vec![
        key(1, 64),
        key(
            2,
            if mode == PublicationMode::Budget {
                100_000
            } else {
                32
            },
        ),
        key(3, 0),
        key(4, 16),
    ];
    let weak = Arc::downgrade(&keys[1].payload);
    let root = pool.register_storage([(pkey(&keys[0]), 64)]).unwrap();
    let (origin_r, origin_run) = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(g),
        )
        .unwrap()
        .into_funding()
        .unwrap();
    let origin = origin_run.scope().unwrap();
    let external = origin
        .adopt_storage_individually(if mode == PublicationMode::LateZero {
            vec![(pkey(&keys[3]), 16), (pkey(&keys[2]), 0)]
        } else {
            vec![(pkey(&keys[3]), 16)]
        })
        .unwrap();
    let contention = (mode == PublicationMode::Busy).then(contention);
    let hold = contention
        .as_ref()
        .map(|c| pool.register_storage([(c.key.clone(), 1)]).unwrap());
    let q = publication_quote(
        &pool,
        &source,
        &keys[0],
        g,
        if mode == PublicationMode::Empty { 0 } else { 5 },
    );
    let exact = pool.used_bytes().unwrap() + q.incremental_bytes();
    let caps = ModelCapabilities {
        effective_model_type: "ordinary-text-fixture".into(),
        native_max_context: Observed::exact(128, "fixture"),
        effective_max_context: Observed::exact(128, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let shape = AdmissionRequest {
        input: InputTokenCount::text(rows),
        max_output_tokens: 1,
        batch_size: 1,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    assert!(plan_prefill_incremental_with_capacity(
        session.inference_execution_identity(),
        &pool,
        &caps,
        shape,
        g,
        exact - 1,
        |_| Ok(q.clone())
    )
    .is_err());
    let (r, accepted) = plan_prefill_incremental_with_capacity(
        session.inference_execution_identity(),
        &pool,
        &caps,
        shape,
        g,
        exact,
        |_| Ok(q.clone()),
    )
    .unwrap();
    let (r, run) = r.into_funding().unwrap();
    let native = run.scope().unwrap();
    let pending = accepted
        .begin_capture_plan_publication::<PublicationKey>(&run, &r, &source)
        .unwrap();
    let (mut owner, witness) = pending.publish_and_finish(&native).unwrap();
    let protected = owner.protected_host_bytes();
    let slots = owner
        .take_prefill_storage_publications::<PublicationKey>()
        .unwrap();
    assert!(owner
        .take_prefill_storage_publications::<PublicationKey>()
        .is_err());
    let mut funded = run
        .prepare_capture_run(&r, CaptureRunHostPlan::prepare(&source).unwrap())
        .unwrap()
        .into_capture_session()
        .unwrap();
    let mut backend = PublicationBackend {
        scope: Some(native),
        sibling: Some(run.scope().unwrap()),
        segment: None,
        slots,
        attempt: None,
        keys,
        outputs: vec![],
        ordinary: vec![],
        origin: Some(origin),
        mode,
        failure: None,
        retired: 0,
        contention,
    };
    let request = InferenceRequest::from(&r);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        funded.with_prefill_observer(
            &mut backend,
            bound,
            &|e| Error::backend_retained_source(e),
            |observer| {
                let mut driver = eredu_runtime::prefill::PrefillDriver::new(
                    session.inference_execution_identity(),
                    request.clone(),
                    g,
                    GenerationCancellationToken::new(),
                )
                .unwrap();
                let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
                let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
                    &mut session,
                    Input {
                        geometry: g,
                        prepared: Rc::new(Cell::new(0)),
                        failure: Failure::None,
                        identity: Arc::new(()),
                    },
                    request.clone(),
                    &(),
                    &mut borrowed,
                )
                .unwrap();
                let result = driver.run(&mut executor, |_, _| {});
                drop(executor);
                borrowed.finish_prefill(matches!(
                    result,
                    Ok(eredu_runtime::prefill::PrefillOutcome::Complete)
                ));
                result
            },
        )
    }));
    if mode == PublicationMode::Panic {
        assert!(result.is_err());
        assert_eq!(backend.attempt.as_ref().unwrap().retained_input_count(), 5);
        assert!(pool.used_bytes().is_ok());
    } else if matches!(
        mode,
        PublicationMode::Capacity
            | PublicationMode::DuplicateCapacity
            | PublicationMode::Budget
            | PublicationMode::Busy
            | PublicationMode::LateHealth
            | PublicationMode::LateZero
    ) {
        assert!(result.unwrap().unwrap().is_err());
        assert_eq!(backend.retired, 0);
        assert_eq!(backend.slots.spent_rows(), 1);
        match (mode, backend.failure.as_ref().unwrap()) {
            (PublicationMode::Busy, BoundedPublicationError::Busy) => {}
            (
                PublicationMode::Budget,
                BoundedPublicationError::Storage(WorkingMemoryError::BudgetExceeded { .. }),
            ) => {}
            (
                PublicationMode::LateHealth | PublicationMode::LateZero,
                BoundedPublicationError::Storage(WorkingMemoryError::ExecutionFenced),
            ) => {}
            (
                PublicationMode::Capacity | PublicationMode::DuplicateCapacity,
                BoundedPublicationError::Storage(WorkingMemoryError::StorageCapacityMismatch {
                    ..
                }),
            ) => {}
            (_, other) => panic!("unexpected original cause: {other}"),
        }
    } else {
        assert!(matches!(
            result.unwrap().unwrap(),
            Ok(eredu_runtime::prefill::PrefillOutcome::Complete)
        ));
        assert_eq!(backend.retired, rows as usize);
    }
    INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
    let outputs = std::mem::take(&mut backend.outputs);
    let ordinary = std::mem::take(&mut backend.ordinary);
    let failed = backend.attempt.take();
    // The scalar fixture submits no native work. Successful publication is the
    // provider's actual complete managed-payload inventory for these callbacks.
    backend.scope.take().unwrap().certify().unwrap();
    backend.sibling.take().unwrap().certify().unwrap();
    if let Some(origin) = backend.origin.take() {
        origin.certify().unwrap();
    }
    drop((
        funded, backend, owner, witness, q, request, r, run, root, external, origin_r, origin_run,
        session, selected, paths, runtime, source, hold,
    ));
    if matches!(
        mode,
        PublicationMode::LateHealth | PublicationMode::LateZero
    ) {
        assert!(pool.used_bytes().unwrap() > 0);
        drop((failed, outputs, ordinary));
        return;
    }
    if !outputs.is_empty() {
        // Escaped legacy output controls keep the original source witness free:
        // C can retire even while the new fixed entry's raw S custody survives.
        assert_eq!(pool.used_bytes().unwrap(), protected + 112);
        for output in &outputs {
            output.validate_source(&pool).unwrap();
        }
        assert_eq!(&*weak.upgrade().unwrap(), &[2u8; 32]);
        drop(outputs);
        assert_eq!(pool.used_bytes().unwrap(), protected + 32);
        assert!(pool.pin_registered_storage([(ckey, c)]).is_err());
        drop(ordinary);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(weak.upgrade().is_none());
    } else {
        drop((outputs, ordinary));
        if failed.is_some() {
            assert!(pool.used_bytes().unwrap() >= protected);
        }
        drop(failed);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn canonical_three_batch_rows_share_legacy_and_fixed_origins_until_last_alias() {
    exercise_publication(PublicationMode::Success, 3);
}
#[test]
fn canonical_empty_and_foreign_scope_publication_preserve_original_custody() {
    exercise_publication(PublicationMode::Empty, 1);
    exercise_publication(PublicationMode::Foreign, 1);
}
#[test]
fn mixed_batch_capacity_and_duplicate_rejection_is_atomic() {
    exercise_publication(PublicationMode::Capacity, 1);
    exercise_publication(PublicationMode::DuplicateCapacity, 1);
}
#[test]
fn new_batch_cannot_spend_original_host_hold_or_exceed_residual() {
    exercise_publication(PublicationMode::Budget, 1);
}
#[test]
fn busy_issued_batch_retains_inputs_and_cannot_retry() {
    exercise_publication(PublicationMode::Busy, 1);
}
#[test]
fn provider_comparison_panic_keeps_original_inputs_before_usage() {
    exercise_publication(PublicationMode::Panic, 1);
}
#[test]
fn late_zero_and_nonzero_origin_health_blocks_whole_publication() {
    exercise_publication(PublicationMode::LateHealth, 1);
    exercise_publication(PublicationMode::LateZero, 1);
}
