//! Real SessionPrefill contexts/tickets, neutral scalar equations and actual
//! original P+Q+S/H funding. No native allocator or capture transform claim.
use super::*;
#[path = "bounded_pins/opening_groups.rs"]
mod opening_groups;
#[path = "bounded_pins/retirement.rs"]
mod retirement;
use eredu_nn::workspace::*;
use opening_groups::OpeningMode;
use std::{
    cmp::Ordering,
    sync::atomic::{AtomicBool, Ordering as AtomicOrdering},
};

#[derive(Debug)]
struct Probe {
    panic_after: std::sync::atomic::AtomicUsize,
    checker:
        std::sync::Weak<std::sync::OnceLock<(WorkingMemoryPool, BoundedRegisteredStorage<Key>)>>,
    drop_checks: std::sync::atomic::AtomicUsize,
    drop_saw_busy: AtomicBool,
}
#[derive(Clone, Debug)]
struct Key {
    id: u32,
    payload: Arc<[u8]>,
    probe: Arc<Probe>,
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        let prior = self.probe.panic_after.fetch_update(
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
            |n| n.checked_sub(1),
        );
        assert!(prior != Ok(1), "provider comparison sentinel");
        self.id.cmp(&other.id)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        if let Some(checker) = self.probe.checker.upgrade() {
            if let Some((pool, group)) = checker.get() {
                self.probe.drop_checks.fetch_add(1, AtomicOrdering::SeqCst);
                if matches!(group.validate_source(pool), Err(BoundedPinError::Busy)) {
                    self.probe.drop_saw_busy.store(true, AtomicOrdering::SeqCst);
                }
            }
        }
    }
}
fn key(id: u32, n: usize) -> Key {
    Key {
        id,
        payload: vec![id as u8; n].into(),
        probe: Arc::new(Probe {
            panic_after: std::sync::atomic::AtomicUsize::new(0),
            checker: std::sync::Weak::new(),
            drop_checks: std::sync::atomic::AtomicUsize::new(0),
            drop_saw_busy: AtomicBool::new(false),
        }),
    }
}
// A distinct existing registry namespace supplies deterministic Usage contention
// through provider comparison. The held lock performs no native work.
#[derive(Debug)]
struct HoldGate {
    armed: AtomicBool,
    entered: std::sync::mpsc::Sender<()>,
    release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}
#[derive(Clone, Debug)]
struct HoldKey(Arc<HoldGate>);
impl PartialEq for HoldKey {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for HoldKey {}
impl PartialOrd for HoldKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HoldKey {
    fn cmp(&self, _: &Self) -> Ordering {
        if self.0.armed.swap(false, AtomicOrdering::SeqCst) {
            self.0.entered.send(()).unwrap();
            self.0
                .release
                .lock()
                .unwrap()
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
        }
        Ordering::Equal
    }
}
struct Contention {
    key: HoldKey,
    entered: std::sync::mpsc::Receiver<()>,
    release: std::sync::mpsc::Sender<()>,
}
fn contention() -> Contention {
    let (etx, erx) = std::sync::mpsc::channel();
    let (rtx, rrx) = std::sync::mpsc::channel();
    Contention {
        key: HoldKey(Arc::new(HoldGate {
            armed: AtomicBool::new(false),
            entered: etx,
            release: std::sync::Mutex::new(rrx),
        })),
        entered: erx,
        release: rtx,
    }
}
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|x| x.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "scalar fixture exact output backing".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "no disjoint scalar workspace".into(),
        }))
    }
}
fn source(g: InferenceGeometry) -> SharedCapturePlan {
    let template = capture_source();
    let mut point = template.admission().points()[0].clone();
    point.path = "decoder.unit.0.output".into();
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let mut plan = template.admission().plan().clone();
    plan.selections[0].path = point.path.clone();
    // Exercise the actual hook/causal binding, but skip its transform through the
    // existing logical quota. These tests cover pin custody, not fragment fills.
    plan.limits.per_step.captures = 0;
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![point],
                completeness: DescriptionCompleteness::Complete,
            },
            &support,
            &CaptureCapabilities {
                transformations: vec![CaptureTransformKind::FullTensor],
                ..Default::default()
            },
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: g.input_positions,
                max_predictions: 1,
            },
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    )
}
fn quote(
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
        RegisteredWorkspaceStorage::bind(pool, &context, [(root.clone(), backing)]).unwrap();
    let report = quote_inference_workspace(g, |_| {
        context.begin_state_span([&tensor])?;
        context.report(&[tensor.clone()])
    })
    .unwrap();
    let h = CaptureRunHostPlan::prepare(source)
        .unwrap()
        .initialization_peak_bytes();
    let b = |n| WorkspaceBound::bounded(n, "actual scalar retention fixture enclosing owner");
    let outside = ExecutionWorkspaceEstimate {
        geometry: g,
        activations: b(384),
        attention: b(0),
        vocabulary: b(0),
        state_update: b(0),
        materialization: b(0),
        retained: b(h),
    };
    let q = ResidualInferenceQuote::compose(
        &report,
        mock_inference_admission(g).state,
        outside,
        &storage,
    )
    .unwrap()
    .into_incremental();
    let pins =
        PreparedPrefillStoragePinPlan::<Key>::prepare(q.span_workspace().plan(), |_| Some(slots))
            .unwrap();
    let controls = PreparedTextControlWorkspace::prepare(
        source,
        g,
        q.span_workspace().plan(),
        TextHostControlFacts::new(Some(11), Some(17), Some(23)),
    )
    .unwrap()
    .with_prefill_storage_pins(pins)
    .unwrap();
    q.with_span_workspace_and_text_controls(controls).unwrap()
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Success,
    Empty,
    Missing,
    Capacity,
    DuplicateCapacity,
    ForeignScope,
    OtherSource,
    LateHealth,
    Panic,
    RegistryPanic,
    Busy,
}
struct Backend {
    scope: Option<WorkingMemoryFundingScope>,
    sibling: Option<WorkingMemoryFundingScope>,
    segment: Option<CaptureSourceSegment>,
    slots: OriginalPrefillStoragePinSlots<Key>,
    keys: Vec<Key>,
    groups: Vec<BoundedRegisteredStorage<Key>>,
    failure: Option<FailedBoundedPinAttempt<Key>>,
    origin: Option<WorkingMemoryFundingScope>,
    mode: Mode,
    retired: usize,
    contention: Option<Contention>,
    drop_checker: Arc<std::sync::OnceLock<(WorkingMemoryPool, BoundedRegisteredStorage<Key>)>>,
    opening_mode: Option<OpeningMode>,
    opening_failure_identity: Arc<()>,
    opening_pending: Option<BoundedRegisteredStorage<Key>>,
    opening_parcels: Vec<SettledCaptureSourceParcel>,
}
impl ScheduledCaptureBackend for Backend {
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
        if self.mode == Mode::ForeignScope {
            assert!(matches!(
                self.slots
                    .begin(context, self.sibling.as_ref().unwrap(), segment),
                Err(BoundedPinError::Storage(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert_eq!(self.slots.spent_rows(), 0);
        }
        let mut attempt = match self.slots.begin(context, scope, segment) {
            Ok(value) => value,
            Err(error) => return Err(FundedCaptureError::Backend(Error::backend_retained_source(error))),
        };
        let empty_probe = self.mode == Mode::RegistryPanic && context.chunk().input.start == 0;
        if self.mode != Mode::Empty && !empty_probe {
            attempt.push_owned(self.keys[0].clone(), 64).unwrap();
            match self.mode {
                Mode::Missing => attempt.push_owned(key(99, 8), 8).unwrap(),
                Mode::Capacity => attempt.push_owned(self.keys[1].clone(), 33).unwrap(),
                Mode::DuplicateCapacity => attempt.push_owned(self.keys[0].clone(), 63).unwrap(),
                _ => {
                    attempt.push_owned(self.keys[1].clone(), 32).unwrap();
                    attempt.push_owned(self.keys[2].clone(), 0).unwrap();
                    attempt.push_owned(self.keys[0].clone(), 64).unwrap();
                    let extra = key(101, 3);
                    let pointer = Arc::as_ptr(&extra.payload);
                    let rejected = attempt.push_owned(extra, 3).unwrap_err();
                    assert_eq!(Arc::as_ptr(&rejected.payload), pointer);
                }
            }
        }
        if self.mode == Mode::LateHealth {
            drop(self.origin.take());
        }
        if self.mode == Mode::Panic {
            self.keys[0]
                .probe
                .panic_after
                .store(1, AtomicOrdering::SeqCst);
        }
        if self.mode == Mode::RegistryPanic && !empty_probe {
            // Three comparisons on the first key occur in dedup. The fourth
            // is registry ordinal search while Usage is held.
            self.keys[0]
                .probe
                .panic_after
                .store(4, AtomicOrdering::SeqCst);
        }
        let worker = self
            .contention
            .as_ref()
            .filter(|_| self.opening_mode.is_none())
            .map(|c| {
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
        let result = attempt.pin_registered(scope, segment);
        if let Some(worker) = worker {
            self.contention.as_ref().unwrap().release.send(()).unwrap();
            drop(worker.join().unwrap());
        }
        match result {
            Ok(group) => {
                group.validate_source(scope.pool()).unwrap();
                assert_eq!(
                    group.bytes(),
                    if self.mode == Mode::Empty || empty_probe {
                        0
                    } else {
                        96
                    }
                );
                if empty_probe {
                    self.drop_checker
                        .set((scope.pool().clone(), group.clone()))
                        .unwrap();
                }
                self.groups.push(group.clone());
                self.groups.push(group);
                if self.opening_mode.is_some() {
                    self.install_opening(context)?;
                }
                Ok(Some(registration))
            }
            Err(error) => {
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
        if self.opening_mode == Some(OpeningMode::Abandon) {
            return Err(FundedCaptureError::Backend(Error::backend_retained_source(
                Original(self.opening_failure_identity.clone()),
            )));
        }
        let segment = self.segment.as_ref().unwrap();
        let parcel = segment
            .take_settled_sources(self.scope.as_mut().unwrap(), &ticket)
            .map_err(CaptureRunHostError::from)?;
        self.retired += 1;
        if self.opening_mode.is_some() {
            self.opening_parcels.push(parcel);
        } else {
            drop(parcel);
        }
        self.segment.take();
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
fn exercise_pins(mode: Mode, rows: u64) {
    exercise_pins_inner(mode, rows, None);
}
fn exercise_pins_inner(mode: Mode, rows: u64, opening_mode: Option<OpeningMode>) {
    exercise_pins_retirement(mode, rows, opening_mode, false);
}
fn exercise_pins_retirement(
    mode: Mode,
    rows: u64,
    opening_mode: Option<OpeningMode>,
    concurrent_retirement: bool,
) {
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
    // Real immutable declarations of the same neutral architecture, prepared
    // under the fixture's original loading responsibility. No fabricated bound.
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
    let pool = WorkingMemoryPool::new(4_000_000, 0).unwrap();
    let mut keys = vec![key(1, 64), key(2, 32), key(3, 0)];
    let drop_checker = Arc::new(std::sync::OnceLock::new());
    if mode == Mode::RegistryPanic {
        Arc::get_mut(&mut keys[0].probe).unwrap().checker = Arc::downgrade(&drop_checker);
    }
    let contention =
        (mode == Mode::Busy || opening_mode == Some(OpeningMode::Busy)).then(contention);
    let hold_registration = contention
        .as_ref()
        .map(|c| pool.register_storage([(c.key.clone(), 1)]).unwrap());
    let weak_payload = Arc::downgrade(&keys[0].payload);
    let root = pool
        .register_storage([(keys[0].clone(), 64), (keys[2].clone(), 0)])
        .unwrap();
    let (origin_r, origin_run) = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(g),
        )
        .unwrap()
        .into_funding()
        .unwrap();
    let origin = origin_run.scope().unwrap();
    let second = origin
        .adopt_storage_individually([(keys[1].clone(), 32)])
        .unwrap();
    let quote_source = if mode == Mode::OtherSource {
        SharedCapturePlan::new(source.admission().clone())
    } else {
        source.clone()
    };
    let q = quote(
        &pool,
        &quote_source,
        &keys[0],
        g,
        if mode == Mode::Empty { 0 } else { 4 },
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
    let request_shape = AdmissionRequest {
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
        request_shape,
        g,
        exact - 1,
        |_| Ok(q.clone())
    )
    .is_err());
    let (r, accepted) = plan_prefill_incremental_with_capacity(
        session.inference_execution_identity(),
        &pool,
        &caps,
        request_shape,
        g,
        exact,
        |_| Ok(q.clone()),
    )
    .unwrap();
    let (r, run) = r.into_funding().unwrap();
    let (mut owner, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
    let pqs = owner.protected_host_bytes();
    let slots = owner.take_prefill_storage_pins::<Key>().unwrap();
    let mut funded = run
        .prepare_capture_run(&r, CaptureRunHostPlan::prepare(&source).unwrap())
        .unwrap()
        .into_capture_session()
        .unwrap();
    let mut backend = Backend {
        scope: Some(run.scope().unwrap()),
        sibling: Some(run.scope().unwrap()),
        segment: None,
        slots,
        keys,
        groups: vec![],
        failure: None,
        origin: Some(origin),
        mode,
        retired: 0,
        contention,
        drop_checker,
        opening_mode,
        opening_failure_identity: Arc::new(()),
        opening_pending: None,
        opening_parcels: vec![],
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
    if opening_mode == Some(OpeningMode::Panic) {
        assert!(result.is_err());
        assert!(
            backend.opening_pending.is_some(),
            "caller owner survives provider panic"
        );
        assert_eq!(backend.opening_pending.as_ref().unwrap().bytes(), 96);
        assert_eq!(backend.slots.spent_rows(), 1);
        assert_eq!(backend.retired, 0);
        assert!(matches!(
            pool.used_bytes(),
            Err(WorkingMemoryError::Poisoned)
        ));
        INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
        return;
    }
    if mode == Mode::RegistryPanic {
        assert!(result.is_err());
        assert_eq!(backend.slots.spent_rows(), 2);
        assert!(matches!(
            pool.used_bytes(),
            Err(WorkingMemoryError::Poisoned)
        ));
        assert!(
            backend.keys[0]
                .probe
                .drop_checks
                .load(AtomicOrdering::SeqCst)
                > 0
        );
        assert!(
            !backend.keys[0]
                .probe
                .drop_saw_busy
                .load(AtomicOrdering::SeqCst),
            "no staged key destructor runs under Usage"
        );
        // Keys hold only Weak to the preallocated checker; it cannot form a
        // key -> pool/registry -> key cycle, including on assertion failure.
        // Positive cleanup cannot be invented after poisoned Usage. Ordinary
        // Drop retains/quarantines the same original scopes and all origins.
        INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
        return;
    }
    let opening_failed = matches!(
        opening_mode,
        Some(
            OpeningMode::OriginBeforeInstall
                | OpeningMode::OriginAfterInstall
                | OpeningMode::Abandon
        )
    );
    if opening_failed {
        let error = result
            .unwrap()
            .unwrap()
            .err()
            .expect("original opening failure");
        let eredu_runtime::prefill::PrefillError::Submission(error) = error else {
            panic!("expected typed submission error");
        };
        // This fixture's policy/mechanism errors are &str, so the outer
        // generic session error has no std::error::Error implementation.
        // Inspect the actual typed variants and preserve the backend source.
        let mut inner = &error;
        while let eredu_runtime::ReplicatedTextSessionError::BeforeStateMutation(error) = inner {
            inner = error;
        }
        if opening_mode == Some(OpeningMode::OriginAfterInstall) {
            let eredu_runtime::ReplicatedTextSessionError::Architecture(error) = inner else {
                panic!("expected typed capture-account rejection: {inner}");
            };
            // The observer's first-frame account check returns this transparent
            // host variant. Its Error::source deliberately forwards the inner
            // source, so inspect the preserved enum rather than skip its payload.
            assert!(
                matches!(
                    cause::<FundedCaptureError<Error>>(error),
                    Some(FundedCaptureError::Host(CaptureRunHostError::Memory(
                        WorkingMemoryError::ExecutionFenced
                    )))
                ),
                "original source-health cause: {error:?}"
            );
        } else {
            let eredu_runtime::ReplicatedTextSessionError::Architecture(error) = inner else {
                panic!("expected original architecture observer error: {inner}");
            };
            let original = cause::<Original>(error).expect("original typed backend cause");
            assert!(Arc::ptr_eq(&original.0, &backend.opening_failure_identity));
        }
        assert_eq!(backend.retired, 0);
        assert_eq!(backend.slots.spent_rows(), 1);
        assert_eq!(
            backend.opening_pending.is_some(),
            opening_mode == Some(OpeningMode::OriginBeforeInstall)
        );
        if opening_mode == Some(OpeningMode::OriginAfterInstall) {
            assert!(matches!(
                backend
                    .segment
                    .as_ref()
                    .unwrap()
                    .validate_native_scope(backend.scope.as_ref().unwrap()),
                Err(WorkingMemoryError::ExecutionFenced)
            ));
        }
    } else if mode == Mode::Panic {
        assert!(result.is_err());
        assert_eq!(backend.slots.spent_rows(), 1);
        // The comparison is during bounded dedup before Usage, so the pool
        // remains usable and no registered owner was incremented.
        assert!(pool.used_bytes().is_ok());
    } else if matches!(
        mode,
        Mode::Missing
            | Mode::Capacity
            | Mode::DuplicateCapacity
            | Mode::LateHealth
            | Mode::OtherSource
            | Mode::Busy
    ) {
        assert!(result.unwrap().unwrap().is_err());
        assert_eq!(backend.retired, 0);
        assert_eq!(
            backend.slots.spent_rows(),
            usize::from(mode != Mode::OtherSource)
        );
        if mode != Mode::OtherSource {
            let error = backend.failure.as_ref().unwrap();
            assert!(matches!(
                error.cause(),
                BoundedPinError::Busy
                    | BoundedPinError::Storage(
                        WorkingMemoryError::IdentityMismatch | WorkingMemoryError::ExecutionFenced
                    )
            ));
            assert!(error.retained_input_count() >= 2);
        }
    } else {
        assert!(matches!(
            result.unwrap().unwrap(),
            Ok(eredu_runtime::prefill::PrefillOutcome::Complete)
        ));
        assert_eq!(backend.retired, rows as usize);
        assert_eq!(backend.slots.spent_rows(), rows as usize);
        assert_eq!(backend.groups.len(), rows as usize * 2);
    }
    INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
    let aliases = std::mem::take(&mut backend.groups);
    let failed = backend.failure.take();
    let pending_opening = backend.opening_pending.take();
    let parcels = std::mem::take(&mut backend.opening_parcels);
    let quarantine = opening_failed || opening_mode == Some(OpeningMode::Abandon);
    if quarantine {
        drop(backend.scope.take());
    } else {
        backend.scope.take().unwrap().certify().unwrap();
    }
    backend.sibling.take().unwrap().certify().unwrap();
    if let Some(origin) = backend.origin.take() {
        origin.certify().unwrap();
    }
    drop((
        funded,
        backend,
        owner,
        q,
        request,
        r,
        run,
        root,
        second,
        origin_r,
        origin_run,
        session,
        selected,
        paths,
        runtime,
        source,
        quote_source,
        hold_registration,
    ));
    if mode == Mode::LateHealth || quarantine {
        assert!(
            pool.used_bytes().unwrap() > 0,
            "original source account remains quarantined"
        );
    } else if !aliases.is_empty() {
        assert!(pool.used_bytes().unwrap() >= pqs + if mode == Mode::Empty { 0 } else { 96 });
        if mode != Mode::Empty {
            let payload = weak_payload
                .upgrade()
                .expect("actual source remains owned by escaped keys");
            assert_eq!(&*payload, &[1u8; 64]);
        }
        for alias in &aliases {
            alias.validate_source(&pool).unwrap();
        }
        assert_eq!(aliases[0].bytes(), if mode == Mode::Empty { 0 } else { 96 });
    } else if failed.is_some() {
        assert!(
            pool.used_bytes().unwrap() >= pqs,
            "terminal failure owns the actual controls"
        );
    }
    if concurrent_retirement {
        assert!(!quarantine);
        assert!(failed.is_none());
        assert!(pending_opening.is_none());
        assert!(matches!(opening_mode, None | Some(OpeningMode::Success)));
        retirement::retire_final_owners(
            aliases,
            parcels,
            &pool,
            pqs + if mode == Mode::Empty { 0 } else { 96 },
        );
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(weak_payload.upgrade().is_none());
        return;
    }
    drop((failed, aliases, pending_opening));
    if opening_mode.is_some() && !quarantine {
        let payload = weak_payload
            .upgrade()
            .expect("parcel alone retains original physical source");
        assert_eq!(&*payload, &[1u8; 64]);
        let held = pool.used_bytes().unwrap();
        assert!(held >= pqs + 96);
        // A real independent payload alias retires before the final pin parcel.
        drop(payload);
        assert_eq!(pool.used_bytes().unwrap(), held);
    }
    drop(parcels);
    if matches!(
        opening_mode,
        Some(OpeningMode::Abandon | OpeningMode::OriginAfterInstall)
    ) {
        assert!(
            weak_payload.upgrade().is_some(),
            "quarantine owns installed group after all caller aliases drop"
        );
        assert!(pool.used_bytes().unwrap() >= pqs + 96);
    }
    if mode != Mode::LateHealth && !quarantine {
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert!(weak_payload.upgrade().is_none());
    }
}
#[test]
fn canonical_three_rows_keep_aliases_and_original_controls_after_run_close() {
    exercise_pins(Mode::Success, 3);
}
#[test]
fn canonical_empty_group_still_retains_original_control_custody() {
    exercise_pins(Mode::Empty, 1);
}
#[test]
fn failed_second_key_and_duplicate_capacity_preserve_atomic_group() {
    for mode in [Mode::Missing, Mode::Capacity, Mode::DuplicateCapacity] {
        exercise_pins(mode, 1);
    }
}
#[test]
fn canonical_sibling_scope_and_equal_independent_source_are_rejected() {
    for mode in [Mode::ForeignScope, Mode::OtherSource] {
        exercise_pins(mode, 1);
    }
}
#[test]
fn canonical_late_funding_origin_failure_keeps_failed_attempt_and_quarantine() {
    exercise_pins(Mode::LateHealth, 1);
}
#[test]
fn comparison_panic_consumes_row_without_publishing_or_poisoning_usage() {
    exercise_pins(Mode::Panic, 1);
}

#[test]
fn busy_commit_consumes_its_row_and_preserves_terminal_inputs() {
    exercise_pins(Mode::Busy, 1);
}

#[test]
fn registry_comparison_unwind_drops_keys_after_unlock_and_keeps_quarantine() {
    exercise_pins(Mode::RegistryPanic, 2);
}

#[path = "bounded_pins/publications.rs"]
mod publications;
