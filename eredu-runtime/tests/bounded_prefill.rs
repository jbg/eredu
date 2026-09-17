use eredu_core::{
    cache::LayerCachePolicy, Admission, AdmissionRequest, AdmissionResult, AttentionPolicy,
    CacheStateStrategy, Completion, EstimationCompleteness, ExecutionWorkspaceEstimate,
    GenerationCancellationToken, InferenceGeometry, InputModalities, InputTokenCount,
    LayerSchedule, ModelCapabilities, Observed, OutputDemand, RuntimeStateEstimate,
    SessionAuthority, StateMemoryLayout, Submission, WorkspaceBound,
};
use eredu_runtime::{prefill::*, working_memory::*};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
};

fn geometry(chunk: u64, output: OutputDemand) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 7,
        max_output_tokens: 3,
        prefill_chunk_positions: chunk,
        output,
    }
}

fn capabilities() -> ModelCapabilities {
    ModelCapabilities {
        effective_model_type: "neutral recurrent fixture".into(),
        native_max_context: Observed::exact(64, "fixture"),
        effective_max_context: Observed::exact(64, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::PersistentStateOnly,
    }
}

fn request(g: InferenceGeometry) -> AdmissionRequest {
    AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    }
}

fn quote(g: InferenceGeometry) -> Result<RuntimeStateEstimate, eredu_core::CapabilityError> {
    // Nonzero persistent storage and explicit component bounds for the fixture.
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 2).unwrap()],
        )
        .unwrap(),
        vec![0],
        2,
        1,
        EstimationCompleteness::Complete,
    )?;
    let state = eredu_core::estimate_runtime_state(
        &layout,
        request(g).input,
        g.max_output_tokens,
        g.batch_size,
        std::num::NonZeroU8::new(4).unwrap(),
    )?;
    let bound = |n| WorkspaceBound::bounded(n, "neutral fixture storage bound");
    state.with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry: g,
        activations: bound(g.prefill_chunk_positions * 32),
        attention: bound(0),
        vocabulary: bound(g.output.positions(g.prefill_chunk_positions) * 16),
        state_update: bound(80),
        materialization: bound(0),
        retained: bound(32),
    })
}

fn admission(g: InferenceGeometry) -> Admission {
    match eredu_core::apply_admission_policy(&capabilities(), request(g), quote(g).unwrap(), None)
        .unwrap()
    {
        AdmissionResult::Admitted(admitted) => admitted,
        other => panic!("{other:?}"),
    }
}

struct NativeCompletion {
    status: Arc<AtomicU8>,
    _reservation: InferenceRequest,
    quarantine: Rc<RefCell<Vec<InferenceRequest>>>,
}
impl Completion for NativeCompletion {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        match self.status.load(Ordering::SeqCst) {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(std::io::Error::other("observation failed")),
        }
    }
    fn wait(&self) -> Result<(), Self::Error> {
        if self.status.load(Ordering::SeqCst) == 0 {
            self.status.store(1, Ordering::SeqCst);
        }
        self.is_complete().map(|_| ())
    }
    fn resources_releasable(&self) -> bool {
        self.status.load(Ordering::SeqCst) == 1
    }
}
impl Drop for NativeCompletion {
    fn drop(&mut self) {
        if !self.resources_releasable() {
            self.quarantine.borrow_mut().push(self._reservation.clone());
        }
    }
}

struct RecurrentExecutor {
    state: f64,
    projected: Vec<usize>,
    submitted: Vec<PrefillChunk>,
    status: Arc<AtomicU8>,
    quarantine: Rc<RefCell<Vec<InferenceRequest>>>,
    fail_submission: bool,
}
impl RecurrentExecutor {
    fn new() -> Self {
        Self {
            state: 0.25,
            projected: vec![],
            submitted: vec![],
            status: Arc::new(AtomicU8::new(0)),
            quarantine: Default::default(),
            fail_submission: false,
        }
    }
    fn decode(&mut self, token: f64) -> f64 {
        self.state = 0.75 * self.state + token;
        self.state * 1.5 - 0.125
    }
}
impl PrefillExecutor for RecurrentExecutor {
    type Output = Vec<f64>;
    type Completion = NativeCompletion;
    type Error = std::io::Error;
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        reservation: InferenceRequest,
    ) -> Result<Submission<Option<Self::Output>, Self::Completion>, Self::Error> {
        if self.fail_submission {
            return Err(std::io::Error::other("preparation failed"));
        }
        self.submitted.push(chunk.clone());
        let mut hidden = vec![];
        for position in chunk.input.clone() {
            self.state = 0.75 * self.state + (position as f64 + 1.0) / 3.0;
            hidden.push(self.state);
        }
        // The output equation is invoked only after position selection.
        let output = match chunk.output {
            OutputDemand::StateOnly => None,
            OutputDemand::LastPosition => {
                self.projected.push(1);
                Some(vec![hidden.last().unwrap() * 1.5 - 0.125])
            }
            OutputDemand::Sequence => {
                self.projected.push(hidden.len());
                Some(hidden.into_iter().map(|x| x * 1.5 - 0.125).collect())
            }
        };
        Ok(Submission {
            output,
            completion: NativeCompletion {
                status: self.status.clone(),
                _reservation: reservation,
                quarantine: self.quarantine.clone(),
            },
        })
    }
}

fn run(chunk: u64, output: OutputDemand, controlled: bool) -> (RecurrentExecutor, Vec<f64>) {
    let g = geometry(chunk, output);
    let pool = WorkingMemoryPool::new(10000, 100).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let reserve = pool.reserve(&execution, &admission(g)).unwrap();
    let mut driver =
        PrefillDriver::new(&execution, reserve, g, GenerationCancellationToken::new()).unwrap();
    let mut executor = RecurrentExecutor::new();
    let mut scores = vec![];
    if controlled {
        loop {
            match driver.step(&mut executor).unwrap() {
                PrefillProgress::Pending => executor.status.store(1, Ordering::SeqCst),
                PrefillProgress::Chunk { output, .. } => {
                    scores.extend(output.into_iter().flatten())
                }
                PrefillProgress::Complete => break,
                PrefillProgress::Cancelled => panic!("unexpected cancellation"),
            }
        }
    } else {
        assert_eq!(
            driver
                .run(&mut executor, |_, output| scores
                    .extend(output.into_iter().flatten()))
                .unwrap(),
            PrefillOutcome::Complete
        );
    }
    drop(driver);
    assert_eq!(pool.used_bytes().unwrap(), 100);
    (executor, scores)
}

#[test]
fn nonzero_chunked_state_matches_full_prompt_and_cached_decodes_in_both_drivers() {
    for demand in [
        OutputDemand::StateOnly,
        OutputDemand::LastPosition,
        OutputDemand::Sequence,
    ] {
        for controlled in [false, true] {
            for chunk in [1, 2, 3, 6, 7] {
                let (mut ordinary, expected) = run(7, demand, false);
                let (mut chunked, actual) = run(chunk, demand, controlled);
                assert_eq!(actual, expected);
                assert_eq!(chunked.state, ordinary.state);
                for token in [0.1, 1.7, -0.8] {
                    assert_eq!(chunked.decode(token), ordinary.decode(token));
                }
                match demand {
                    OutputDemand::StateOnly => assert!(chunked.projected.is_empty()),
                    OutputDemand::LastPosition => {
                        assert_eq!(chunked.projected, [1]);
                        assert!(chunked
                            .submitted
                            .iter()
                            .rev()
                            .skip(1)
                            .all(|c| c.output == OutputDemand::StateOnly));
                    }
                    OutputDemand::Sequence => {
                        assert_eq!(chunked.projected.iter().sum::<usize>(), 7)
                    }
                }
            }
        }
    }
}

#[test]
fn cancellation_polls_the_same_submission_and_holds_capacity_until_completion() {
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 20).unwrap();
    let id = InferenceExecutionIdentity::default();
    let capacity = 20 + admission(g).incremental_required_bytes;
    let cancellation = GenerationCancellationToken::new();
    let mut driver = PrefillDriver::new(
        &id,
        pool.reserve_with_capacity(&id, &admission(g), capacity)
            .unwrap(),
        g,
        cancellation.clone(),
    )
    .unwrap();
    let mut executor = RecurrentExecutor::new();
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Pending
    ));
    cancellation.cancel();
    for _ in 0..3 {
        assert!(matches!(
            driver.step(&mut executor).unwrap(),
            PrefillProgress::Pending
        ));
    }
    assert_eq!(executor.submitted.len(), 1);
    assert!(pool.used_bytes().unwrap() > 20);
    assert_eq!(pool.effective_capacity().unwrap(), capacity);
    assert!(matches!(
        pool.reserve(&id, &admission(g)),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    executor.status.store(1, Ordering::SeqCst);
    assert!(matches!(
        driver.step(&mut executor).unwrap(),
        PrefillProgress::Cancelled
    ));
    drop(driver);
    assert_eq!(pool.used_bytes().unwrap(), 20);
    assert_eq!(pool.effective_capacity().unwrap(), 1000);
}

#[test]
fn cancellation_before_first_step_submits_nothing() {
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let id = InferenceExecutionIdentity::default();
    let cancel = GenerationCancellationToken::new();
    cancel.cancel();
    let mut driver =
        PrefillDriver::new(&id, pool.reserve(&id, &admission(g)).unwrap(), g, cancel).unwrap();
    let mut executor = RecurrentExecutor::new();
    assert_eq!(
        driver
            .run(&mut executor, |_, _| panic!("cancelled"))
            .unwrap(),
        PrefillOutcome::Cancelled
    );
    assert!(executor.submitted.is_empty());
}

#[test]
fn failed_completion_never_refunds_or_replays_unresolved_work() {
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let id = InferenceExecutionIdentity::default();
    let capacity = admission(g).incremental_required_bytes;
    let mut driver = PrefillDriver::new(
        &id,
        pool.reserve_with_capacity(&id, &admission(g), capacity)
            .unwrap(),
        g,
        GenerationCancellationToken::new(),
    )
    .unwrap();
    let mut executor = RecurrentExecutor::new();
    executor.status.store(2, Ordering::SeqCst);
    assert!(matches!(
        driver.step(&mut executor),
        Err(PrefillError::Completion(_))
    ));
    assert!(!driver.release_settled_failure());
    assert!(matches!(
        driver.step(&mut executor),
        Err(PrefillError::Failed)
    ));
    assert_eq!(executor.submitted.len(), 1);
    drop(driver);
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(pool.effective_capacity().unwrap(), capacity);
    assert!(matches!(
        pool.register_storage([(1_u32, 1)]),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    // The native recovery owner independently proves safe release.
    executor.status.store(1, Ordering::SeqCst);
    executor.quarantine.borrow_mut().clear();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.effective_capacity().unwrap(), 1000);
}

#[test]
fn retained_source_capacity_selects_chunks_and_competes_with_concurrent_requests() {
    use eredu_checkpoint::store::{
        CheckpointSource, MemoryWeightStore, RestrictedCheckpointSource, SourceStorage,
    };
    let mut payload = Vec::with_capacity(4096);
    payload.extend([7u8, 9]);
    let capacity = payload.capacity() as u64;
    let source = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "weight".into(),
            safetensors::Dtype::U8,
            vec![2],
            payload,
        )])
        .unwrap(),
    );
    let view = RestrictedCheckpointSource::including(
        source.clone(),
        "target",
        std::collections::BTreeSet::from(["weight".into()]),
    )
    .unwrap();
    let storage = SourceStorage::collect([source.as_ref() as &dyn CheckpointSource, &view, &view])
        .unwrap()
        .unwrap();
    assert_eq!(storage.bytes().unwrap(), capacity);
    assert!(matches!(
        WorkingMemoryPool::new(capacity - 1, storage.bytes().unwrap()),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    let baseline = run(7, OutputDemand::LastPosition, false).1;
    for controlled in [false, true] {
        let g = geometry(7, OutputDemand::LastPosition);
        let id = InferenceExecutionIdentity::default();
        let available = admission(geometry(3, g.output)).incremental_required_bytes;
        let pool =
            WorkingMemoryPool::new(capacity + 2 * available, storage.bytes().unwrap()).unwrap();
        let (admitted, reservation) = plan_prefill_with_capacity(
            &id,
            &pool,
            &capabilities(),
            request(g),
            g,
            capacity + available,
            quote,
        )
        .unwrap();
        assert_eq!(reservation.geometry().prefill_chunk_positions, 3);
        assert!(matches!(
            pool.reserve(&id, &admitted),
            Err(WorkingMemoryError::BudgetExceeded { .. })
        ));
        let geometry = reservation.geometry();
        let mut driver = PrefillDriver::new(
            &id,
            reservation,
            geometry,
            GenerationCancellationToken::new(),
        )
        .unwrap();
        let mut executor = RecurrentExecutor::new();
        let mut scores = Vec::new();
        if controlled {
            loop {
                match driver.step(&mut executor).unwrap() {
                    PrefillProgress::Pending => executor.status.store(1, Ordering::SeqCst),
                    PrefillProgress::Chunk { output, .. } => {
                        scores.extend(output.into_iter().flatten())
                    }
                    PrefillProgress::Complete => break,
                    PrefillProgress::Cancelled => panic!("unexpected cancellation"),
                }
            }
        } else {
            assert_eq!(
                driver
                    .run(&mut executor, |_, output| scores
                        .extend(output.into_iter().flatten()))
                    .unwrap(),
                PrefillOutcome::Complete
            );
        }
        assert_eq!(scores, baseline);
        assert_eq!(pool.used_bytes().unwrap(), capacity + available);
        assert_eq!(pool.effective_capacity().unwrap(), capacity + available);
        drop(driver);
        assert_eq!(pool.used_bytes().unwrap(), capacity);
        assert_eq!(pool.effective_capacity().unwrap(), capacity + 2 * available);
        assert!(pool.reserve(&id, &admitted).is_ok());
    }
}

#[test]
fn planner_reduces_chunk_under_shared_capacity_and_keeps_decode_allowance() {
    let g = geometry(7, OutputDemand::LastPosition);
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(400, 80).unwrap();
    let (admitted, reservation) =
        plan_prefill(&id, &pool, &capabilities(), request(g), g, quote).unwrap();
    assert_eq!(reservation.geometry().prefill_chunk_positions, 3);
    assert_eq!(admitted.requested_positions, 10);
    assert_eq!(pool.used_bytes().unwrap(), 384);
    assert!(matches!(
        pool.reserve(&id, &admitted),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    let clone = reservation.clone();
    drop(reservation);
    assert_eq!(pool.used_bytes().unwrap(), 384);
    drop(clone);
    assert_eq!(pool.used_bytes().unwrap(), 80);
    assert_eq!(pool.peak_bytes().unwrap(), 384);
}

#[test]
fn reservation_is_bound_to_execution_and_geometry_and_retained_by_submission_authority() {
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let id = InferenceExecutionIdentity::default();
    let reservation = pool.reserve(&id, &admission(g)).unwrap();
    assert_eq!(
        reservation.validate(&InferenceExecutionIdentity::default(), g),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        reservation.validate(&id, geometry(2, OutputDemand::LastPosition)),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    let mut authority = SessionAuthority::new();
    let mut lease = authority.begin_submission().unwrap();
    lease.retain_resource(reservation);
    assert!(lease.resolve());
    assert!(pool.used_bytes().unwrap() > 0);
    drop(lease);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn selected_state_backing_gaps_and_geometry_reject_even_with_stale_coverage() {
    let execution = InferenceExecutionIdentity::default();
    let g = geometry(3, OutputDemand::LastPosition);
    for unknown in [false, true] {
        let mut admitted = admission(g);
        admitted.state.selected_state_backing = Some(eredu_core::SelectedStateBacking {
            geometry: if unknown {
                g
            } else {
                InferenceGeometry {
                    prefill_chunk_positions: 1,
                    ..g
                }
            },
            bound: if unknown {
                WorkspaceBound::Unknown {
                    reason: "missing selected native backing".into(),
                }
            } else {
                WorkspaceBound::bounded(0, "fixture mismatched schedule")
            },
        });
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let result = pool.reserve(&execution, &admitted);
        assert!(
            matches!(result, Err(ref error) if *error == if unknown { WorkingMemoryError::UnknownBound } else { WorkingMemoryError::IdentityMismatch })
        );
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.peak_bytes().unwrap(), 0);
        let result =
            eredu_core::apply_admission_policy(&capabilities(), request(g), admitted.state, None);
        if unknown {
            assert!(matches!(
                result,
                Ok(AdmissionResult::Rejected(
                    eredu_core::AdmissionRejection::EstimationUnsupported { .. }
                ))
            ));
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn strict_admission_rejects_unknown_workspace_even_with_a_large_safety_reserve() {
    let g = geometry(3, OutputDemand::LastPosition);
    let mut state = quote(g).unwrap();
    state.execution_workspace.as_mut().unwrap().attention = WorkspaceBound::Unknown {
        reason: "native scratch not bounded".into(),
    };
    let mut req = request(g);
    req.safety_reserve_bytes = 1_000_000;
    assert!(matches!(
        eredu_core::apply_admission_policy(&capabilities(), req, state, None).unwrap(),
        AdmissionResult::Rejected(eredu_core::AdmissionRejection::EstimationUnsupported { .. })
    ));
}

#[test]
fn chunk_planning_checks_nonmonotone_native_bounds_and_rejects_before_submission() {
    let g = geometry(7, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let id = InferenceExecutionIdentity::default();
    let (_, reservation) = plan_prefill(&id, &pool, &capabilities(), request(g), g, |geometry| {
        let mut estimate = quote(geometry)?;
        if geometry.prefill_chunk_positions != 5 {
            estimate.execution_workspace.as_mut().unwrap().attention = WorkspaceBound::Unknown {
                reason: "only the five-position fixture kernel is bounded".into(),
            };
        }
        Ok(estimate)
    })
    .unwrap();
    assert_eq!(reservation.geometry().prefill_chunk_positions, 5);
    let too_small = WorkingMemoryPool::new(1, 0).unwrap();
    assert!(plan_prefill(&id, &too_small, &capabilities(), request(g), g, quote).is_err());
    assert_eq!(too_small.used_bytes().unwrap(), 0);
}

#[test]
fn submission_failure_fences_driver_without_advancing_state() {
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let id = InferenceExecutionIdentity::default();
    let mut driver = PrefillDriver::new(
        &id,
        pool.reserve(&id, &admission(g)).unwrap(),
        g,
        GenerationCancellationToken::new(),
    )
    .unwrap();
    let mut executor = RecurrentExecutor::new();
    executor.fail_submission = true;
    assert!(matches!(
        driver.step(&mut executor),
        Err(PrefillError::Submission(_))
    ));
    assert!(matches!(
        driver.step(&mut executor),
        Err(PrefillError::Failed)
    ));
    assert_eq!(executor.state, 0.25);
    assert!(executor.submitted.is_empty());
    drop(driver);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn cloned_retention_cannot_start_another_request_or_refund_started_authority() {
    let g = geometry(3, OutputDemand::LastPosition);
    let pool = WorkingMemoryPool::new(1000, 0).unwrap();
    let id = InferenceExecutionIdentity::default();
    let reserve = pool.reserve(&id, &admission(g)).unwrap();
    let driver: PrefillDriver<Vec<f64>, NativeCompletion> =
        PrefillDriver::new(&id, reserve.clone(), g, GenerationCancellationToken::new()).unwrap();
    drop(driver);
    assert!(matches!(
        PrefillDriver::<Vec<f64>, NativeCompletion>::new(
            &id,
            reserve,
            g,
            GenerationCancellationToken::new()
        ),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn concurrent_admissions_cannot_overbook_shared_physical_capacity() {
    let g = geometry(3, OutputDemand::LastPosition);
    let admitted = admission(g);
    let pool = WorkingMemoryPool::new(2 * admitted.incremental_required_bytes + 20, 20).unwrap();
    let workers = (0..8)
        .map(|_| {
            let pool = pool.clone();
            let admitted = admitted.clone();
            std::thread::spawn(move || {
                pool.reserve(&InferenceExecutionIdentity::default(), &admitted)
            })
        })
        .collect::<Vec<_>>();
    let held = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(held.iter().filter(|result| result.is_ok()).count(), 2);
    assert_eq!(
        pool.used_bytes().unwrap(),
        2 * admitted.incremental_required_bytes + 20
    );
    drop(held);
    assert_eq!(pool.used_bytes().unwrap(), 20);
}

#[test]
fn one_rank_cancellation_stops_every_rank_at_the_same_completed_boundary() {
    use std::sync::{atomic::AtomicBool, Barrier};
    struct CollectiveExecutor {
        inner: RecurrentExecutor,
        cancellation: GenerationCancellationToken,
        cancel_after_submit: bool,
        barrier: Arc<Barrier>,
        any_cancelled: Arc<AtomicBool>,
    }
    impl PrefillExecutor for CollectiveExecutor {
        type Output = Vec<f64>;
        type Completion = NativeCompletion;
        type Error = std::io::Error;
        fn agree_cancellation(
            &mut self,
            local: bool,
            _: InferenceRequest,
        ) -> Result<bool, Self::Error> {
            self.any_cancelled.fetch_or(local, Ordering::SeqCst);
            self.barrier.wait();
            let cancelled = self.any_cancelled.load(Ordering::SeqCst);
            self.barrier.wait();
            Ok(cancelled)
        }
        fn submit_chunk(
            &mut self,
            chunk: &PrefillChunk,
            reservation: InferenceRequest,
        ) -> Result<Submission<Option<Self::Output>, Self::Completion>, Self::Error> {
            let submission = self.inner.submit_chunk(chunk, reservation)?;
            if self.cancel_after_submit {
                self.cancellation.cancel();
            }
            Ok(submission)
        }
    }
    for cancel_before_first in [false, true] {
        let barrier = Arc::new(Barrier::new(2));
        let any_cancelled = Arc::new(AtomicBool::new(false));
        let workers = (0..2)
            .map(|rank| {
                let barrier = barrier.clone();
                let any_cancelled = any_cancelled.clone();
                std::thread::spawn(move || {
                    let g = geometry(2, OutputDemand::LastPosition);
                    let execution = InferenceExecutionIdentity::default();
                    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
                    let reservation = pool.reserve(&execution, &admission(g)).unwrap();
                    let cancellation = GenerationCancellationToken::new();
                    if rank == 0 && cancel_before_first {
                        cancellation.cancel();
                    }
                    let mut executor = CollectiveExecutor {
                        inner: RecurrentExecutor::new(),
                        cancellation: cancellation.clone(),
                        cancel_after_submit: rank == 0,
                        barrier,
                        any_cancelled,
                    };
                    executor.inner.status.store(1, Ordering::SeqCst);
                    let mut driver =
                        PrefillDriver::new(&execution, reservation, g, cancellation.clone())
                            .unwrap();
                    assert_eq!(
                        driver
                            .run(&mut executor, |_, _| panic!(
                                "cancelled output must not publish"
                            ))
                            .unwrap(),
                        PrefillOutcome::Cancelled
                    );
                    assert!(
                        cancellation.is_cancelled(),
                        "peer cancellation must reach local commitment"
                    );
                    // Terminal calls must not start another collective after a peer
                    // has returned its outcome.
                    assert!(matches!(
                        driver.step(&mut executor).unwrap(),
                        PrefillProgress::Cancelled
                    ));
                    assert_eq!(
                        executor.inner.submitted.len(),
                        usize::from(!cancel_before_first)
                    );
                    let state = executor.inner.state;
                    drop(driver);
                    drop(executor);
                    assert_eq!(pool.used_bytes().unwrap(), 0);
                    state
                })
            })
            .collect::<Vec<_>>();
        let states = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(states[0], states[1]);
    }
}

#[test]
fn explicit_unbudgeted_requests_share_scheduling_without_claiming_memory_coverage() {
    let g = geometry(3, OutputDemand::LastPosition);
    let id = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let mut results = Vec::new();
    for budgeted in [false, true] {
        let request: InferenceRequest = if budgeted {
            pool.reserve(&id, &admission(g)).unwrap().into()
        } else {
            InferenceRequest::without_memory_budget(&id, g).unwrap()
        };
        assert_eq!(request.memory_reservation().is_some(), budgeted);
        let same_geometry = InferenceRequest::without_memory_budget(&id, g).unwrap();
        assert_eq!(
            request.validate_same_request(&same_geometry),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(request.validate_same_request(&request.clone()), Ok(()));
        let mut executor = RecurrentExecutor::new();
        let mut driver =
            PrefillDriver::new(&id, &request, g, GenerationCancellationToken::new()).unwrap();
        assert!(matches!(
            PrefillDriver::<Vec<f64>, NativeCompletion>::new(
                &id,
                &request,
                g,
                GenerationCancellationToken::new()
            ),
            Err(WorkingMemoryError::AlreadyStarted)
        ));
        let mut scores = Vec::new();
        assert_eq!(
            driver
                .run(&mut executor, |_, output| scores
                    .extend(output.into_iter().flatten()))
                .unwrap(),
            PrefillOutcome::Complete
        );
        results.push((executor.state, scores, executor.submitted));
        drop(driver);
        drop(request);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn reservation_domain_validation_uses_identity_without_changing_charges() {
    let g = geometry(2, OutputDemand::LastPosition);
    let id = InferenceExecutionIdentity::default();
    let first_pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let second_pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let first = first_pool.reserve(&id, &admission(g)).unwrap();
    let second = second_pool.reserve(&id, &admission(g)).unwrap();
    let charged = first.bytes();
    assert!(charged > 0);
    assert_eq!(second.bytes(), charged);
    assert_eq!(first_pool.used_bytes().unwrap(), charged);
    assert_eq!(second_pool.used_bytes().unwrap(), charged);
    for _ in 0..2 {
        first.validate(&id, g).unwrap();
        second.validate(&id, g).unwrap();
        first.validate_domain(&first_pool.clone()).unwrap();
        second.validate_domain(&second_pool).unwrap();
        assert_eq!(
            first.validate_domain(&second_pool),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!(
            second.validate_domain(&first_pool),
            Err(WorkingMemoryError::IdentityMismatch)
        );
    }
    for pool in [&first_pool, &second_pool] {
        assert_eq!(pool.used_bytes().unwrap(), charged);
        assert_eq!(pool.peak_bytes().unwrap(), charged);
    }
    drop(first);
    assert_eq!(first_pool.used_bytes().unwrap(), 0);
    assert_eq!(second_pool.used_bytes().unwrap(), charged);
    drop(second);
    assert_eq!(second_pool.used_bytes().unwrap(), 0);
}

#[test]
fn equal_geometry_cannot_replace_the_exact_reserved_charge_owner() {
    let g = geometry(2, OutputDemand::LastPosition);
    let id = InferenceExecutionIdentity::default();
    let first_pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let second_pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let reservation = first_pool.reserve(&id, &admission(g)).unwrap();
    let request = InferenceRequest::from(&reservation);
    assert_eq!(
        request.validate_same_request(&InferenceRequest::from(&reservation)),
        Ok(())
    );
    let other = InferenceRequest::from(second_pool.reserve(&id, &admission(g)).unwrap());
    assert_eq!(
        request.validate_same_request(&other),
        Err(WorkingMemoryError::IdentityMismatch)
    );
}

#[test]
fn state_retention_survives_request_drop_and_never_refunds_newer_charges_on_restore() {
    use eredu_nn::workspace::WorkspaceBackend;
    type State = eredu_runtime::DeviceState<WorkspaceBackend, Vec<u32>>;
    let g = geometry(3, OutputDemand::LastPosition);
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(g);
    let charge = admitted.incremental_required_bytes;
    let pool = WorkingMemoryPool::new(charge * 2, 0).unwrap();
    let first = pool.reserve(&execution, &admitted).unwrap();
    let first_alias: InferenceRequest = (&first).into();
    let first_request: InferenceRequest = first.into();
    let second: InferenceRequest = pool.reserve(&execution, &admitted).unwrap().into();
    let layout = eredu_runtime::StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 2).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = State::create(layout, |_, _| Ok::<_, ()>(vec![7, 11, 13])).unwrap();
    state.retain_inference(&first_request);
    state.retain_inference(&first_alias);
    assert_eq!(state.inference_retention().requests().len(), 1);
    let saved = state.clone();
    state.as_mut()[0].push(17);
    state.retain_inference(&second);
    assert_eq!(state.inference_retention().requests().len(), 2);
    drop(first_alias);
    drop(first_request);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), charge * 2);
    assert!(matches!(
        pool.reserve(&execution, &admitted),
        Err(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    state.clone_from(&saved);
    assert_eq!(state.as_ref()[0], [7, 11, 13]);
    assert_eq!(state.inference_retention().requests().len(), 2);
    assert_eq!(pool.used_bytes().unwrap(), charge * 2);
    drop(state);
    assert_eq!(pool.used_bytes().unwrap(), charge);
    drop(saved);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), charge * 2);
}

#[test]
fn exchanged_state_and_descendants_keep_their_exact_charges_without_charging_handle_clones() {
    use eredu_nn::workspace::WorkspaceBackend;
    type State = eredu_runtime::DeviceState<WorkspaceBackend, Vec<u32>>;
    let g = geometry(2, OutputDemand::LastPosition);
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(g);
    let charge = admitted.incremental_required_bytes;
    let pool = WorkingMemoryPool::new(charge, 0).unwrap();
    let request: InferenceRequest = pool.reserve(&execution, &admitted).unwrap().into();
    let unbudgeted = InferenceRequest::without_memory_budget(&execution, g).unwrap();
    let mut installed = State::stateless();
    installed.retain_inference(&request);
    installed.retain_inference(&unbudgeted);
    drop(request);
    let mut branch = installed.clone();
    let saved = branch.clone();
    assert_eq!(pool.used_bytes().unwrap(), charge);
    assert_eq!(installed.inference_retention().requests().len(), 1);
    installed = State::stateless();
    std::mem::swap(&mut installed, &mut branch);
    assert_eq!(installed.inference_retention().requests().len(), 1);
    assert_eq!(branch.inference_retention().requests().len(), 0);
    drop(branch);
    drop(installed);
    assert_eq!(pool.used_bytes().unwrap(), charge);
    drop(saved);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let mut state = State::stateless();
    state.retain_inference(&unbudgeted);
    assert_eq!(state.inference_retention().requests().len(), 0);
}

#[test]
fn active_admission_checks_exact_span_geometry_phase_readout_and_native_frontier() {
    for output in [
        OutputDemand::StateOnly,
        OutputDemand::LastPosition,
        OutputDemand::Sequence,
    ] {
        for chunk in [1, 3, 7] {
            let g = InferenceGeometry {
                batch_size: 2,
                cached_positions: 2,
                ..geometry(chunk, output)
            };
            let execution = InferenceExecutionIdentity::default();
            let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
            let request: InferenceRequest = pool.reserve(&execution, &admission(g)).unwrap().into();
            let mut retention = InferenceRetention::new();
            retention.admit(&request);
            let active = retention.admission().unwrap();
            let demand = output.for_chunk(chunk == 7);
            for frontier in [Some(2), None] {
                assert_eq!(
                    active
                        .validate_span(&execution, Some([2, chunk]), true, demand, frontier)
                        .unwrap(),
                    2 + chunk
                );
            }
            assert!(matches!(
                active.validate_span(&execution, Some([2, chunk]), false, demand, Some(2)),
                Err(WorkingMemoryError::InvocationPhaseMismatch),
            ));
            assert!(matches!(
                active.validate_span(&execution, Some([1, chunk]), true, demand, Some(2)),
                Err(WorkingMemoryError::SpanShapeMismatch {
                    expected: [2, _],
                    actual: [1, _]
                }),
            ));
            assert!(matches!(
                active.validate_span(&execution, Some([2, chunk + 1]), true, demand, Some(2)),
                Err(WorkingMemoryError::SpanShapeMismatch { .. }),
            ));
            if chunk > 1 {
                assert!(matches!(
                    active.validate_span(&execution, Some([2, chunk - 1]), true, demand, Some(2)),
                    Err(WorkingMemoryError::SpanShapeMismatch { .. }),
                ));
            }
            assert!(matches!(
                active.validate_span(&execution, Some([2, chunk]), true, demand, Some(3)),
                Err(WorkingMemoryError::StateFrontierMismatch {
                    expected: 2,
                    actual: 3
                }),
            ));
            assert!(matches!(
                active.validate_span(
                    &InferenceExecutionIdentity::default(),
                    Some([2, chunk]),
                    true,
                    demand,
                    Some(2)
                ),
                Err(WorkingMemoryError::IdentityMismatch),
            ));
            assert!(matches!(
                active.validate_span(&execution, None, true, demand, Some(2)),
                Err(WorkingMemoryError::UnknownBound),
            ));
            let different = if demand == OutputDemand::StateOnly {
                OutputDemand::LastPosition
            } else {
                OutputDemand::StateOnly
            };
            assert!(matches!(
                active.validate_span(&execution, Some([2, chunk]), true, different, Some(2)),
                Err(WorkingMemoryError::OutputDemandMismatch { .. }),
            ));
        }
    }
}

#[test]
fn single_row_readouts_are_equivalent_but_state_only_is_not() {
    for output in [OutputDemand::LastPosition, OutputDemand::Sequence] {
        let g = InferenceGeometry {
            input_positions: 1,
            max_output_tokens: 0,
            ..geometry(1, output)
        };
        let execution = InferenceExecutionIdentity::default();
        let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
        let request: InferenceRequest = pool.reserve(&execution, &admission(g)).unwrap().into();
        let mut retention = InferenceRetention::new();
        retention.admit(&request);
        for demanded in [OutputDemand::LastPosition, OutputDemand::Sequence] {
            assert_eq!(
                retention
                    .admission()
                    .unwrap()
                    .validate_span(&execution, Some([1, 1]), true, demanded, Some(0))
                    .unwrap(),
                1
            );
        }
        assert!(matches!(
            retention.admission().unwrap().validate_span(
                &execution,
                Some([1, 1]),
                true,
                OutputDemand::StateOnly,
                Some(0)
            ),
            Err(WorkingMemoryError::OutputDemandMismatch { .. })
        ));
    }
}

#[test]
fn restoring_logical_admission_keeps_newer_charges_and_empty_snapshots_cannot_erase_it() {
    let execution = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let old_g = geometry(3, OutputDemand::LastPosition);
    let new_g = InferenceGeometry {
        cached_positions: 2,
        ..old_g
    };
    let old: InferenceRequest = pool.reserve(&execution, &admission(old_g)).unwrap().into();
    let new: InferenceRequest = pool.reserve(&execution, &admission(new_g)).unwrap().into();
    let charged = pool.used_bytes().unwrap();
    let mut retention = InferenceRetention::new();
    retention.admit(&old);
    let snapshot = retention.clone();
    retention.admit(&new);
    assert_eq!(retention.admission().unwrap().position(), 2);
    retention.restore_admission(&snapshot);
    assert_eq!(retention.admission().unwrap().position(), 0);
    assert!(retention
        .admission()
        .unwrap()
        .request()
        .validate_same_request(&old)
        .is_ok());
    retention.restore_admission(&InferenceRetention::new());
    retention.admit(&InferenceRequest::without_memory_budget(&execution, old_g).unwrap());
    assert!(retention
        .admission()
        .unwrap()
        .request()
        .validate_same_request(&old)
        .is_ok());
    assert_eq!(retention.requests().len(), 2);
    drop(old);
    drop(new);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    drop(snapshot);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    drop(retention);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[path = "bounded_prefill/text_preparation.rs"]
mod text_preparation;

#[path = "bounded_prefill/text_run.rs"]
mod text_run;

#[path = "bounded_prefill/storage.rs"]
mod storage;

#[path = "bounded_prefill/capacity.rs"]
mod capacity;

#[path = "bounded_prefill/unquoted.rs"]
mod unquoted;

#[path = "bounded_prefill/final_output.rs"]
mod final_output;

#[path = "bounded_prefill/controls.rs"]
mod controls;
