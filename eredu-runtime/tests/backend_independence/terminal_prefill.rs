//! Genuine shared source/session transactions. The transport is a bounded neutral
//! rendezvous; it is not evidence of native multi-device cancellation/completion.
use super::*;
use eredu_core::{GenerationCancellationToken, InferenceGeometry, OutputDemand};
use eredu_runtime::replicated_session::{PrefillSourceOutcome, PreparedPrefillSource};
use eredu_runtime::working_memory::*;
use eredu_runtime::{ActivationObserver, PartitionExecutionError, ReplicatedTextSessionError};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Round {
    votes: [Option<(DistributedExecutionPhase, bool)>; 2],
}
#[derive(Default)]
pub(super) struct Coordinator {
    rounds: Mutex<Vec<Round>>,
    changed: Condvar,
    inject: Option<DistributedExecutionPhase>,
}
pub(super) struct Agreement {
    rank: usize,
    next: usize,
    coordinator: Arc<Coordinator>,
}
fn deadline(phase: DistributedExecutionPhase) -> PartitionExecutionError {
    PartitionExecutionError::CommunicationDeadlineExceeded {
        operation: CommunicationOperation::FailureAgreement,
        phase,
        route: None,
        cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    }
}
impl Agreement {
    pub(super) fn agree(
        &mut self,
        phase: DistributedExecutionPhase,
        ready: bool,
    ) -> Result<bool, PartitionExecutionError> {
        check_inference_scope();
        let ordinal = self.next;
        self.next += 1;
        let mut rounds = self.coordinator.rounds.lock().unwrap();
        if rounds.len() <= ordinal {
            rounds.resize_with(ordinal + 1, Round::default);
        }
        assert!(rounds[ordinal].votes[self.rank]
            .replace((phase, ready))
            .is_none());
        self.coordinator.changed.notify_all();
        let end = Instant::now() + Duration::from_secs(2);
        loop {
            if let [Some(a), Some(b)] = rounds[ordinal].votes {
                assert_eq!(a.0, b.0, "ranks entered different protocols");
                return if self.coordinator.inject == Some(phase) {
                    Err(deadline(phase))
                } else {
                    Ok(a.1 && b.1)
                };
            }
            let Some(left) = end.checked_duration_since(Instant::now()) else {
                return Err(deadline(phase));
            };
            let (next, timeout) = self.coordinator.changed.wait_timeout(rounds, left).unwrap();
            rounds = next;
            if timeout.timed_out() {
                return Err(deadline(phase));
            }
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("original prepared input cause")]
struct InputCause(Arc<()>);
thread_local! { static INPUT_CAUSE: RefCell<Option<Arc<()>>> = const { RefCell::new(None) }; }
fn input_cause(error: &Error) -> &InputCause {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(cause) = source.downcast_ref::<InputCause>() {
            return cause;
        }
        source = source.source().expect("retained original input cause");
    }
}

struct Source {
    geometry: InferenceGeometry,
    calls: Rc<Cell<usize>>,
    fail: bool,
}
impl
    PreparedPrefillSource<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    > for Source
{
    type Chunk = FakeTensor;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(
        &self,
        _: &eredu_runtime::prefill::PrefillChunk,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        check_inference_scope();
        self.calls.set(self.calls.get() + 1);
        if self.fail {
            Err(Error::backend_source(InputCause(INPUT_CAUSE.with(
                |slot| slot.borrow_mut().take().expect("test cause"),
            ))))
        } else {
            Ok(FakeTensor(vec![1, 7]))
        }
    }
    fn input<'a>(&'a self, chunk: &'a FakeTensor) -> &'a FakeTensor {
        chunk
    }
}
#[derive(Default)]
struct Observer {
    begun: usize,
    finished: Vec<bool>,
    chunks: Vec<bool>,
    panic: Option<Arc<()>>,
    opening_panic: Option<Arc<()>>,
}
impl ActivationObserver<FakeTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        // This fixture observes lifecycle and final-position output only.
        false
    }
    fn requires_prefill_opening_state(&self) -> bool {
        self.opening_panic.is_some()
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        _: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
        opening: &eredu_runtime::inspection::PrefillOpeningState<'_, FakeTensor>,
    ) -> Result<Option<eredu_runtime::inspection::PreparedPrefillChunkRetention>, Error> {
        check_inference_scope();
        opening.visit(&mut |_| {}).map_err(Error::backend_source)?;
        std::panic::panic_any(self.opening_panic.take().expect("opening panic fixture"))
    }
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        _: DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.begun += 1;
        Ok(())
    }
    fn finish_transaction(&mut self, _: DistributedCommitEpoch, commit: bool) {
        self.chunks.push(commit);
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.finished.push(committed);
    }
    fn observe(&mut self, _: &str, _: &FakeTensor) -> Result<(), Error> {
        check_inference_scope();
        if let Some(payload) = self.panic.take() {
            std::panic::panic_any(payload);
        }
        Ok(())
    }
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    }
}
fn run(
    session: &mut ReferencePartitionedSession,
    request: &InferenceRequest,
    calls: &Rc<Cell<usize>>,
    factory: &Rc<Cell<usize>>,
    observer: &mut Observer,
    fail: bool,
) -> Result<
    PrefillSourceOutcome<FakeTensor>,
    ReplicatedTextSessionError<Error, &'static str, &'static str>,
> {
    session.try_prefill_source_cancellable(
        Some(request),
        Some([1, 1]),
        None,
        |geometry| {
            check_inference_scope();
            factory.set(factory.get() + 1);
            Ok(Some(Source {
                geometry,
                calls: calls.clone(),
                fail,
            }))
        },
        &GenerationCancellationToken::new(),
        &(),
        observer,
    )
}
fn reset() {
    INFERENCE_SCOPE_TRACE.with(|t| *t.borrow_mut() = InferenceScopeTrace::default());
    ENTRY_COORDINATION.with(|t| *t.borrow_mut() = EntryCoordination::default());
}

#[test]
fn every_failed_guard_entry_stops_before_the_next_vote_or_native_callback() {
    // source, initial cancellation, chunk, nested model, completion agreement,
    // post-chunk cancellation, final indexing: exactly seven healthy attempts.
    for failed in 1..=7 {
        reset();
        let phases = Rc::new(RefCell::new(Vec::new()));
        let (mut session, counters, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
                failed_phase: DistributedExecutionPhase::Commit,
                calls: phases.clone(),
                commits: Default::default(),
            }),
        );
        let pool = WorkingMemoryPool::new(384, 0).unwrap();
        let request: InferenceRequest = pool
            .reserve(
                session.inference_execution_identity(),
                &mock_inference_admission(geometry()),
            )
            .unwrap()
            .into();
        INFERENCE_SCOPE_TRACE.with(|t| {
            *t.borrow_mut() = InferenceScopeTrace {
                required: true,
                fail_begin_at: Some(failed),
                retain_abandoned: true,
                ..Default::default()
            }
        });
        let calls = Rc::new(Cell::new(0));
        let factory = Rc::new(Cell::new(0));
        let mut observer = Observer::default();
        let error = run(
            &mut session,
            &request,
            &calls,
            &factory,
            &mut observer,
            false,
        )
        .err()
        .expect("failed entry");
        assert!(matches!(
            error,
            ReplicatedTextSessionError::Mechanism("injected reservation scope failure")
        ));
        assert_eq!(factory.get(), usize::from(failed > 1));
        assert_eq!(calls.get(), usize::from(failed > 3));
        assert_eq!(counters.snapshot().forward_calls, usize::from(failed > 4));
        assert_eq!(
            observer.finished,
            if failed == 1 { vec![] } else { vec![false] }
        );
        if failed == 4 {
            assert!(!phases
                .borrow()
                .iter()
                .any(|(phase, _)| *phase == DistributedExecutionPhase::InputPreparation));
        }
        let prior = phases.borrow().len();
        assert!(run(
            &mut session,
            &request,
            &calls,
            &factory,
            &mut observer,
            false
        )
        .is_err());
        assert_eq!(phases.borrow().len(), prior, "fenced retry must not vote");
        INFERENCE_SCOPE_TRACE.with(|t| {
            let t = t.borrow();
            assert_eq!(t.active, 0);
            assert_eq!(t.begin_attempts, failed);
            assert_eq!(t.opened, t.finished + t.abandoned);
        });
        drop((session, request));
        let quarantined = INFERENCE_SCOPE_TRACE.with(|t| !t.borrow().quarantine.is_empty());
        assert_eq!(
            pool.used_bytes().unwrap(),
            if quarantined { 384 } else { 0 }
        );
        reset(); // Separate scalar completion evidence, never deadline-as-completion.
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn earlier_prepared_input_error_survives_nested_begin_failure_and_abandons_outer_scope() {
    reset();
    let (mut session, _, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: Default::default(),
            commits: Default::default(),
        }),
    );
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry()),
        )
        .unwrap()
        .into();
    INFERENCE_SCOPE_TRACE.with(|t| {
        *t.borrow_mut() = InferenceScopeTrace {
            required: true,
            fail_begin_at: Some(4),
            retain_abandoned: true,
            ..Default::default()
        }
    });
    let identity = Arc::new(());
    INPUT_CAUSE.with(|slot| *slot.borrow_mut() = Some(identity.clone()));
    let error = run(
        &mut session,
        &request,
        &Rc::new(Cell::new(0)),
        &Rc::new(Cell::new(0)),
        &mut Observer::default(),
        true,
    )
    .err()
    .unwrap();
    let ReplicatedTextSessionError::Architecture(ref original) = error else {
        panic!("original architecture error");
    };
    assert!(Arc::ptr_eq(&input_cause(original).0, &identity));
    assert!(error.to_string().contains("original prepared input cause"));
    assert!(!error.to_string().contains("reservation scope failure"));
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!(t.finished, 2);
        assert_eq!(t.abandoned, 1);
    });
    assert!(
        session.report().unwrap().distributed_commit().is_none(),
        "failed entry does not fabricate Aborted"
    );
    drop((session, request));
    assert_eq!(pool.used_bytes().unwrap(), 384);
    reset();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

fn deadline_phase(
    mut error: &ReplicatedTextSessionError<Error, &'static str, &'static str>,
) -> Option<DistributedExecutionPhase> {
    loop {
        match error {
            ReplicatedTextSessionError::Partition(
                PartitionExecutionError::CommunicationDeadlineExceeded { phase, .. },
            ) => return Some(*phase),
            ReplicatedTextSessionError::BeforeStateMutation(inner) => error = inner,
            _ => return None,
        }
    }
}

fn pair(
    fail_begin: Option<usize>,
    inject: Option<DistributedExecutionPhase>,
) -> Vec<(usize, usize)> {
    let coordinator = Arc::new(Coordinator {
        inject,
        ..Default::default()
    });
    let threads: Vec<_> = (0..2)
        .map(|rank| {
            let coordinator = coordinator.clone();
            std::thread::spawn(move || {
                reset();
                TERMINAL_SUBMISSION_MARKS.with(|m| m.set(0));
                let (mut session, counters, _) = partitioned_agreement_session(
                    rank,
                    LocalPartitionFailure::None,
                    TestPhaseAgreement::Terminal(Agreement {
                        rank,
                        next: 0,
                        coordinator,
                    }),
                );
                let pool = WorkingMemoryPool::new(384, 0).unwrap();
                let request: InferenceRequest = pool
                    .reserve(
                        session.inference_execution_identity(),
                        &mock_inference_admission(geometry()),
                    )
                    .unwrap()
                    .into();
                INFERENCE_SCOPE_TRACE.with(|t| {
                    *t.borrow_mut() = InferenceScopeTrace {
                        required: true,
                        retain_abandoned: true,
                        fail_begin_at: if rank == 0 { fail_begin } else { None },
                        ..Default::default()
                    }
                });
                let mut observer = Observer::default();
                let result = run(
                    &mut session,
                    &request,
                    &Rc::new(Cell::new(0)),
                    &Rc::new(Cell::new(0)),
                    &mut observer,
                    false,
                );
                let failing = fail_begin.is_some() || inject.is_some();
                assert_eq!(result.is_err(), failing);
                if failing {
                    let error = result.err().unwrap();
                    if rank == 0 && fail_begin.is_some() {
                        assert!(matches!(
                            error,
                            ReplicatedTextSessionError::Mechanism(
                                "injected reservation scope failure"
                            )
                        ));
                    } else {
                        let expected = inject.unwrap_or(match fail_begin {
                            Some(1) => DistributedExecutionPhase::PrefillSourcePreparation,
                            Some(2) => DistributedExecutionPhase::PrefillReservation,
                            Some(4) => DistributedExecutionPhase::InputPreparation,
                            _ => DistributedExecutionPhase::Execution,
                        });
                        assert_eq!(deadline_phase(&error), Some(expected));
                    }
                    assert!(TERMINAL_SUBMISSION_MARKS.with(Cell::get) > 0);
                    assert!(!observer.finished.contains(&true));
                    let before = counters.snapshot().forward_calls;
                    assert!(session.decode(&FakeTensor(vec![2, 9]), &()).is_err());
                    assert_eq!(counters.snapshot().forward_calls, before);
                    if inject == Some(DistributedExecutionPhase::Execution) {
                        assert_eq!(
                            session.report().unwrap().state_report(),
                            &[1],
                            "no native rollback after terminal agreement"
                        );
                        assert!(session.report().unwrap().distributed_commit().is_none());
                        assert_eq!(observer.chunks, vec![false]);
                    }
                } else {
                    assert_eq!(observer.finished, vec![true]);
                    assert_eq!(observer.chunks, vec![true]);
                }
                let (finished, abandoned) = INFERENCE_SCOPE_TRACE.with(|t| {
                    let t = t.borrow();
                    assert_eq!(t.active, 0);
                    (t.finished, t.abandoned)
                });
                if inject == Some(DistributedExecutionPhase::Execution) {
                    assert_eq!(finished, 2);
                    assert_eq!(abandoned, 2);
                }
                drop((session, request));
                assert_eq!(
                    pool.used_bytes().unwrap(),
                    if abandoned > 0 { 384 } else { 0 }
                );
                reset();
                assert_eq!(pool.used_bytes().unwrap(), 0);
                (finished, abandoned)
            })
        })
        .collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
}
#[test]
fn two_rank_missing_source_cancel_and_nested_guard_votes_end_by_bounded_deadline() {
    for begin in [1, 2, 4] {
        pair(Some(begin), None);
    }
}
#[test]
fn two_rank_pending_collective_failure_abandons_nested_and_outer_retention_without_rollback() {
    pair(None, Some(DistributedExecutionPhase::Execution));
}
#[test]
fn healthy_two_rank_prefill_keeps_all_seven_guard_completions() {
    assert_eq!(pair(None, None), vec![(7, 0), (7, 0)]);
}

#[test]
fn observer_unwind_keeps_exact_payload_and_abandons_both_scopes_before_fenced_retry() {
    reset();
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: Default::default(),
            commits: Default::default(),
        }),
    );
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry()),
        )
        .unwrap()
        .into();
    INFERENCE_SCOPE_TRACE.with(|t| {
        *t.borrow_mut() = InferenceScopeTrace {
            required: true,
            retain_abandoned: true,
            ..Default::default()
        }
    });
    let identity = Arc::new(());
    let mut observer = Observer {
        panic: Some(identity.clone()),
        ..Default::default()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run(
            &mut session,
            &request,
            &Rc::new(Cell::new(0)),
            &Rc::new(Cell::new(0)),
            &mut observer,
            false,
        );
    }));
    let payload = result
        .unwrap_err()
        .downcast::<Arc<()>>()
        .expect("original panic type");
    assert!(Arc::ptr_eq(&payload, &identity));
    assert_eq!(observer.chunks, vec![false]);
    assert_eq!(observer.finished, vec![false]);
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!((t.active, t.finished, t.abandoned), (0, 2, 2));
    });
    let before = counters.snapshot().forward_calls;
    assert!(session.decode(&FakeTensor(vec![2, 9]), &()).is_err());
    assert_eq!(counters.snapshot().forward_calls, before);
    assert!(session.report().unwrap().distributed_commit().is_none());
    drop((session, request));
    assert_eq!(pool.used_bytes().unwrap(), 384);
    reset();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn agreed_source_rejection_preserves_before_state_mutation_and_ordinary_retry() {
    reset();
    TERMINAL_SUBMISSION_MARKS.with(|m| m.set(0));
    let phases = Rc::new(RefCell::new(Vec::new()));
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: phases.clone(),
            commits: Default::default(),
        }),
    );
    INFERENCE_SCOPE_TRACE.with(|t| {
        *t.borrow_mut() = InferenceScopeTrace {
            required: true,
            ..Default::default()
        }
    });
    let cancellation = GenerationCancellationToken::new();
    let mut observer = Observer::default();
    let error = session
        .try_prefill_unbudgeted_source_cancellable::<Source, _>(
            Some([1, 1]),
            None,
            |_| {
                check_inference_scope();
                Err(Error::backend("original source factory failure"))
            },
            &cancellation,
            &(),
            &mut observer,
        )
        .err()
        .unwrap();
    assert!(matches!(
        &error,
        ReplicatedTextSessionError::BeforeStateMutation(_)
    ));
    assert!(error
        .to_string()
        .contains("original source factory failure"));
    assert_eq!(
        phases.borrow().as_slice(),
        &[(DistributedExecutionPhase::PrefillSourcePreparation, false)]
    );
    assert_eq!(counters.snapshot().forward_calls, 0);
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!((t.finished, t.abandoned, t.active), (0, 1, 0));
    });
    assert_eq!(TERMINAL_SUBMISSION_MARKS.with(Cell::get), 0);
    assert!(observer.finished.is_empty());
    let output = session
        .try_prefill_unbudgeted_source_cancellable(
            Some([1, 1]),
            None,
            |geometry| {
                Ok(Some(Source {
                    geometry,
                    calls: Rc::new(Cell::new(0)),
                    fail: false,
                }))
            },
            &cancellation,
            &(),
            &mut observer,
        )
        .unwrap();
    assert!(matches!(output, PrefillSourceOutcome::Complete(_)));
    assert_eq!(counters.snapshot().forward_calls, 1);
    assert_eq!(observer.finished, vec![true]);
    reset();
}

#[test]
fn source_factory_unwind_abandons_before_vote_and_fences_ordinary_gateway() {
    reset();
    let phases = Rc::new(RefCell::new(Vec::new()));
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: phases.clone(),
            commits: Default::default(),
        }),
    );
    INFERENCE_SCOPE_TRACE.with(|t| {
        *t.borrow_mut() = InferenceScopeTrace {
            required: true,
            retain_abandoned: true,
            ..Default::default()
        }
    });
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry()),
        )
        .unwrap()
        .into();
    let identity = Arc::new(());
    let thrown = identity.clone();
    let mut observer = Observer::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = session.try_prefill_source_cancellable::<Source, _>(
            Some(&request),
            Some([1, 1]),
            None,
            |_| {
                check_inference_scope();
                std::panic::panic_any(thrown)
            },
            &GenerationCancellationToken::new(),
            &(),
            &mut observer,
        );
    }));
    let payload = result
        .unwrap_err()
        .downcast::<Arc<()>>()
        .expect("original panic type");
    assert!(Arc::ptr_eq(&payload, &identity));
    assert!(phases.borrow().is_empty());
    assert_eq!(counters.snapshot().forward_calls, 0);
    assert!(observer.finished.is_empty());
    assert!(session.decode(&FakeTensor(vec![2, 9]), &()).is_err());
    assert!(phases.borrow().is_empty());
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!((t.active, t.finished, t.abandoned), (0, 0, 1));
    });
    drop((session, request));
    assert_eq!(pool.used_bytes().unwrap(), 384);
    reset();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn direct_session_opening_view_unwind_ends_state_loan_before_terminal_cleanup() {
    reset();
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: Default::default(),
            commits: Default::default(),
        }),
    );
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let execution = session.inference_execution_identity().clone();
    let request: InferenceRequest = pool
        .reserve(&execution, &mock_inference_admission(geometry()))
        .unwrap()
        .into();
    INFERENCE_SCOPE_TRACE.with(|t| {
        *t.borrow_mut() = InferenceScopeTrace {
            required: true,
            retain_abandoned: true,
            ..Default::default()
        }
    });
    let identity = Arc::new(());
    let calls = Rc::new(Cell::new(0));
    let mut observer = Observer {
        opening_panic: Some(identity.clone()),
        ..Default::default()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut driver = eredu_runtime::prefill::PrefillDriver::new(
            &execution,
            &request,
            geometry(),
            GenerationCancellationToken::new(),
        )
        .unwrap();
        let mut executor = eredu_runtime::replicated_session::SessionPrefill::new(
            &mut session,
            Source {
                geometry: geometry(),
                calls: calls.clone(),
                fail: false,
            },
            &request,
            &(),
            &mut observer,
        )
        .unwrap();
        let _ = driver.run(&mut executor, |_, _| {});
    }));
    let payload = result
        .unwrap_err()
        .downcast::<Arc<()>>()
        .expect("original panic type");
    assert!(Arc::ptr_eq(&payload, &identity));
    assert_eq!(calls.get(), 0);
    assert_eq!(counters.snapshot().forward_calls, 0);
    assert!(session.decode(&FakeTensor(vec![2, 9]), &()).is_err());
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!((t.active, t.finished, t.abandoned), (0, 1, 1));
    });
    drop((session, request));
    assert_eq!(pool.used_bytes().unwrap(), 384);
    reset();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn state_only_selected_source_finishes_without_final_scores_on_both_ranks() {
    state_only_pair(false, false);
}
#[test]
fn one_rank_cancels_state_only_after_completed_span_before_next_source_call() {
    state_only_pair(true, false);
}
#[test]
fn state_only_completion_agreement_failure_fences_before_next_source_call() {
    state_only_pair(false, true);
}
fn state_only_pair(cancel_after_first: bool, fail_completion: bool) {
    let coordinator = Arc::new(Coordinator {
        inject: fail_completion.then_some(DistributedExecutionPhase::PrefillReservationCompletion),
        ..Default::default()
    });
    let threads: Vec<_> = (0..2)
        .map(|rank| {
            let coordinator = coordinator.clone();
            std::thread::spawn(move || {
                reset();
                let (mut session, counters, _) = partitioned_agreement_session(
                    rank,
                    LocalPartitionFailure::None,
                    TestPhaseAgreement::Terminal(Agreement {
                        rank,
                        next: 0,
                        coordinator,
                    }),
                );
                let cancel = GenerationCancellationToken::new();
                struct StateSource {
                    inner: Source,
                    cancel: GenerationCancellationToken,
                    inject: bool,
                }
                impl
                    PreparedPrefillSource<
                        OrdinaryTextFixture,
                        FakeBackend,
                        DeviceState<FakeBackend, FakeLayerState>,
                    > for StateSource
                {
                    type Chunk = FakeTensor;
                    fn geometry(&self) -> InferenceGeometry {
                        self.inner.geometry
                    }
                    fn prepare_chunk(
                        &self,
                        chunk: &eredu_runtime::prefill::PrefillChunk,
                        context: &(),
                    ) -> Result<FakeTensor, Error> {
                        let value = self.inner.prepare_chunk(chunk, context)?;
                        if self.inject {
                            self.cancel.cancel();
                        }
                        Ok(value)
                    }
                    fn input<'a>(&'a self, chunk: &'a FakeTensor) -> &'a FakeTensor {
                        chunk
                    }
                }
                let calls = Rc::new(Cell::new(0));
                let mut observer = Observer::default();
                let outcome = session.try_prefill_unbudgeted_state_source_cancellable(
                    Some([1, 3]),
                    std::num::NonZeroU64::new(1),
                    |geometry| {
                        Ok(Some(StateSource {
                            inner: Source {
                                geometry,
                                calls: calls.clone(),
                                fail: false,
                            },
                            cancel: cancel.clone(),
                            inject: cancel_after_first && rank == 0,
                        }))
                    },
                    &cancel,
                    &(),
                    &mut observer,
                );
                if fail_completion {
                    let error = outcome
                        .err()
                        .expect("same scope completion agreement failed");
                    assert_eq!(
                        deadline_phase(&error),
                        Some(DistributedExecutionPhase::PrefillReservationCompletion)
                    );
                    assert_eq!(calls.get(), 1);
                    assert_eq!(observer.finished, vec![false]);
                    assert!(session
                        .try_prefill_unbudgeted_state_source_cancellable::<Source, _>(
                            Some([1, 1]),
                            None,
                            |_| panic!("fenced retry entered source factory"),
                            &cancel,
                            &(),
                            &mut Observer::default()
                        )
                        .is_err());
                    reset();
                    return;
                }
                let progress = outcome.unwrap();
                assert_eq!(
                    progress.completed_positions,
                    if cancel_after_first { 1 } else { 3 }
                );
                let outcome = progress.outcome;
                assert_eq!(calls.get(), if cancel_after_first { 1 } else { 3 });
                assert_eq!(
                    counters.snapshot().readout_demand,
                    Some(OutputDemand::StateOnly)
                );
                assert_eq!(observer.finished, vec![!cancel_after_first]);
                if cancel_after_first {
                    assert!(matches!(outcome, PrefillSourceOutcome::Cancelled));
                    assert!(cancel.is_cancelled());
                } else {
                    assert!(matches!(outcome, PrefillSourceOutcome::Complete(())));
                }
                reset();
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}

// Opt-in test caller coordination. The real low-level begin remains unchanged:
// one attempt, including on failure. No clock, retry or native lock is simulated.
#[derive(Default)]
struct EntryCoordination {
    enabled: bool,
    active: bool,
    roles: Vec<Option<eredu_runtime::prefill::PrefillControlRole>>,
    cancel_at: Option<(
        eredu_runtime::prefill::PrefillControlRole,
        GenerationCancellationToken,
    )>,
}
thread_local! {
    static ENTRY_COORDINATION: RefCell<EntryCoordination> = RefCell::default();
}
struct CoordinationExit;
impl Drop for CoordinationExit {
    fn drop(&mut self) {
        ENTRY_COORDINATION.with(|t| t.borrow_mut().active = false);
    }
}
pub(super) fn coordinate_entry<T>(
    role: Option<eredu_runtime::prefill::PrefillControlRole>,
    begin: impl FnOnce() -> T,
) -> T {
    let enabled = ENTRY_COORDINATION.with(|t| t.borrow().enabled);
    if !enabled {
        return begin();
    }
    let cancel = ENTRY_COORDINATION.with(|t| {
        let mut t = t.borrow_mut();
        assert!(!t.active, "caller coordination reentered");
        t.active = true;
        t.roles.push(role);
        if t.cancel_at
            .as_ref()
            .is_some_and(|(at, _)| Some(*at) == role)
        {
            t.cancel_at.take().map(|(_, token)| token)
        } else {
            None
        }
    });
    let _exit = CoordinationExit;
    if let Some(token) = cancel {
        assert!(
            !token.is_cancelled(),
            "cancellation must arise during coordination"
        );
        token.cancel();
    }
    begin()
}
pub(super) fn assert_coordination_released() {
    ENTRY_COORDINATION.with(|t| {
        assert!(
            !t.borrow().active,
            "callback/vote ran inside caller coordination"
        )
    });
}
fn coordinated_roles() -> Vec<Option<eredu_runtime::prefill::PrefillControlRole>> {
    ENTRY_COORDINATION.with(|t| t.borrow().roles.clone())
}
fn span_role(
    phase: eredu_runtime::prefill::PrefillSpanControlPhase,
    start: u64,
    output: OutputDemand,
) -> eredu_runtime::prefill::PrefillControlRole {
    eredu_runtime::prefill::PrefillControlRole::Span {
        phase,
        input_start: start,
        input_end: start + 1,
        position: start,
        output,
    }
}
fn enable_coordination() {
    ENTRY_COORDINATION.with(|t| t.borrow_mut().enabled = true);
    INFERENCE_SCOPE_TRACE.with(|t| t.borrow_mut().required = true);
}

#[test]
fn coordination_routes_every_role_and_direct_input_with_one_low_level_attempt() {
    use eredu_runtime::prefill::{PrefillControlRole as Role, PrefillSpanControlPhase as Phase};
    for fail_direct in [false, true] {
        reset();
        let phases = Rc::new(RefCell::new(Vec::new()));
        let commits = Rc::new(Cell::new(0));
        let (mut session, counters, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
                failed_phase: DistributedExecutionPhase::Commit,
                calls: phases.clone(),
                commits: commits.clone(),
            }),
        );
        let pool = WorkingMemoryPool::new(384, 0).unwrap();
        let request: InferenceRequest = pool
            .reserve(
                session.inference_execution_identity(),
                &mock_inference_admission(InferenceGeometry {
                    max_output_tokens: 2,
                    ..geometry()
                }),
            )
            .unwrap()
            .into();
        enable_coordination();
        let calls = Rc::new(Cell::new(0));
        let factory = Rc::new(Cell::new(0));
        let mut observer = Observer::default();
        let output = run(
            &mut session,
            &request,
            &calls,
            &factory,
            &mut observer,
            false,
        )
        .unwrap();
        let PrefillSourceOutcome::Complete(output) = output else {
            panic!("complete scores")
        };
        assert_eq!(output, FakeTensor(vec![5]));
        assert_eq!(
            (
                factory.get(),
                calls.get(),
                counters.snapshot().forward_calls
            ),
            (1, 1, 1)
        );
        assert_eq!(session.report().unwrap().state_report(), &[1]);
        assert_eq!(commits.get(), 1);
        assert_eq!(observer.chunks, vec![true]);
        assert_eq!(observer.finished, vec![true]);
        assert_eq!(
            coordinated_roles(),
            vec![
                Some(Role::SourcePreparation),
                Some(span_role(Phase::PreBoundary, 0, OutputDemand::LastPosition)),
                Some(span_role(Phase::SpanOuter, 0, OutputDemand::LastPosition)),
                Some(span_role(
                    Phase::InputTransaction,
                    0,
                    OutputDemand::LastPosition
                )),
                Some(span_role(
                    Phase::SettlementAgreement,
                    0,
                    OutputDemand::LastPosition
                )),
                Some(span_role(
                    Phase::PostBoundary,
                    0,
                    OutputDemand::LastPosition
                )),
                Some(Role::FinalIndex),
            ]
        );
        INFERENCE_SCOPE_TRACE.with(|t| {
            let mut t = t.borrow_mut();
            assert_eq!(
                (t.begin_attempts, t.opened, t.finished, t.active),
                (7, 7, 7, 0)
            );
            t.fail_begin = fail_direct;
        });
        ENTRY_COORDINATION.with(|t| t.borrow_mut().roles.clear());
        phases.borrow_mut().clear();
        let result = session.decode_input_result_with_observer(
            Ok(&FakeTensor(vec![2, 9])),
            &(),
            &mut observer,
        );
        assert_eq!(coordinated_roles(), vec![None]);
        if fail_direct {
            assert!(matches!(
                result,
                Err(ReplicatedTextSessionError::Mechanism(
                    "injected reservation scope failure"
                ))
            ));
            assert!(
                phases.borrow().is_empty(),
                "failed begin cannot issue even a false vote"
            );
            assert_eq!(session.report().unwrap().state_report(), &[1]);
            assert_eq!(counters.snapshot().forward_calls, 1);
            assert_eq!(commits.get(), 1);
            assert!(session.decode(&FakeTensor(vec![2, 9]), &()).is_err());
            assert_eq!(
                coordinated_roles(),
                vec![None],
                "fenced retry must not coordinate again"
            );
        } else {
            assert_eq!(result.unwrap(), FakeTensor(vec![5]));
            assert_eq!(session.report().unwrap().state_report(), &[2]);
            assert_eq!(counters.snapshot().forward_calls, 2);
            assert_eq!(commits.get(), 2);
            assert!(phases
                .borrow()
                .contains(&(DistributedExecutionPhase::InputPreparation, true)));
        }
        INFERENCE_SCOPE_TRACE.with(|t| {
            let t = t.borrow();
            assert_eq!(t.begin_attempts, 8);
            assert_eq!(t.active, 0);
            assert_eq!(t.abandoned, 0);
            assert_eq!(t.opened, t.finished);
        });
        drop(session);
        drop(request);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        reset();
    }
}

#[test]
fn cancellation_during_coordination_is_voted_before_the_next_source_or_native_chunk() {
    use eredu_runtime::prefill::PrefillSpanControlPhase as Phase;
    for after_first in [false, true] {
        reset();
        let phases = Rc::new(RefCell::new(Vec::new()));
        let commits = Rc::new(Cell::new(0));
        let (mut session, counters, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
                failed_phase: DistributedExecutionPhase::Commit,
                calls: phases.clone(),
                commits: commits.clone(),
            }),
        );
        let g = InferenceGeometry {
            input_positions: 3,
            ..geometry()
        };
        let pool = WorkingMemoryPool::new(384, 0).unwrap();
        let request: InferenceRequest = pool
            .reserve(
                session.inference_execution_identity(),
                &mock_inference_admission(g),
            )
            .unwrap()
            .into();
        let cancellation = GenerationCancellationToken::new();
        let selected = span_role(
            if after_first {
                Phase::PostBoundary
            } else {
                Phase::PreBoundary
            },
            0,
            OutputDemand::StateOnly,
        );
        enable_coordination();
        ENTRY_COORDINATION
            .with(|t| t.borrow_mut().cancel_at = Some((selected, cancellation.clone())));
        let calls = Rc::new(Cell::new(0));
        let factory = Rc::new(Cell::new(0));
        let mut observer = Observer::default();
        let outcome = session
            .try_prefill_source_cancellable(
                Some(&request),
                Some([1, 3]),
                None,
                |geometry| {
                    check_inference_scope();
                    factory.set(factory.get() + 1);
                    Ok(Some(Source {
                        geometry,
                        calls: calls.clone(),
                        fail: false,
                    }))
                },
                &cancellation,
                &(),
                &mut observer,
            )
            .unwrap();
        assert!(matches!(outcome, PrefillSourceOutcome::Cancelled));
        assert!(cancellation.is_cancelled());
        let completed = usize::from(after_first);
        assert_eq!(factory.get(), 1);
        assert_eq!(calls.get(), completed);
        assert_eq!(counters.snapshot().forward_calls, completed);
        assert_eq!(commits.get(), completed);
        assert_eq!(
            session.report().unwrap().state_report(),
            &[i32::from(after_first)]
        );
        assert_eq!(
            observer.chunks,
            if after_first { vec![true] } else { vec![] }
        );
        assert_eq!(observer.finished, vec![false]);
        let votes = phases.borrow();
        let cancellation_votes: Vec<_> = votes
            .iter()
            .filter(|(phase, _)| *phase == DistributedExecutionPhase::PrefillCancellation)
            .copied()
            .collect();
        assert_eq!(
            cancellation_votes,
            if after_first {
                vec![
                    (DistributedExecutionPhase::PrefillCancellation, true),
                    (DistributedExecutionPhase::PrefillCancellation, false),
                ]
            } else {
                vec![(DistributedExecutionPhase::PrefillCancellation, false)]
            }
        );
        assert_eq!(
            votes.last(),
            Some(&(DistributedExecutionPhase::PrefillCancellation, false))
        );
        drop(votes);
        assert_eq!(coordinated_roles().last(), Some(&Some(selected)));
        INFERENCE_SCOPE_TRACE.with(|t| {
            let t = t.borrow();
            assert_eq!(t.begin_attempts, coordinated_roles().len());
            assert_eq!((t.active, t.abandoned), (0, 0));
            assert_eq!(t.opened, t.finished);
        });
        drop(session);
        drop(request);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        reset();
    }
}

#[test]
fn second_span_begin_failure_keeps_committed_prefix_and_stops_without_false_vote_or_retry() {
    use eredu_runtime::prefill::PrefillSpanControlPhase as Phase;
    reset();
    let phases = Rc::new(RefCell::new(Vec::new()));
    let commits = Rc::new(Cell::new(0));
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: phases.clone(),
            commits: commits.clone(),
        }),
    );
    let g = InferenceGeometry {
        input_positions: 3,
        ..geometry()
    };
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(g),
        )
        .unwrap()
        .into();
    enable_coordination();
    // Source + five first-span roles + second PreBoundary succeed. The existing
    // low-level entry then refuses the second SpanOuter before any chunk work.
    INFERENCE_SCOPE_TRACE.with(|t| t.borrow_mut().fail_begin_at = Some(8));
    let calls = Rc::new(Cell::new(0));
    let mut observer = Observer::default();
    let error = session
        .try_prefill_source_cancellable(
            Some(&request),
            Some([1, 3]),
            None,
            |geometry| {
                Ok(Some(Source {
                    geometry,
                    calls: calls.clone(),
                    fail: false,
                }))
            },
            &GenerationCancellationToken::new(),
            &(),
            &mut observer,
        )
        .err()
        .expect("second entry refusal");
    assert!(matches!(
        error,
        ReplicatedTextSessionError::Mechanism("injected reservation scope failure")
    ));
    assert_eq!(
        (
            calls.get(),
            counters.snapshot().forward_calls,
            commits.get()
        ),
        (1, 1, 1)
    );
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    assert_eq!(observer.chunks, vec![true]);
    assert_eq!(observer.finished, vec![false]);
    assert_eq!(
        coordinated_roles().last(),
        Some(&Some(span_role(
            Phase::SpanOuter,
            1,
            OutputDemand::StateOnly
        )))
    );
    let prior_votes = phases.borrow().clone();
    assert!(
        prior_votes.iter().all(|(phase, ready)| {
            !matches!(
                phase,
                DistributedExecutionPhase::InputPreparation
                    | DistributedExecutionPhase::PrefillReservation
                    | DistributedExecutionPhase::PrefillReservationCompletion
            ) || *ready
        }),
        "no failed-entry vote is authorized"
    );
    assert_eq!(
        prior_votes.last(),
        Some(&(DistributedExecutionPhase::PrefillCancellation, true))
    );
    assert_eq!(
        prior_votes
            .iter()
            .filter(|(phase, _)| *phase == DistributedExecutionPhase::InputPreparation)
            .count(),
        1
    );
    assert!(session.decode(&FakeTensor(vec![2, 9]), &()).is_err());
    assert_eq!(*phases.borrow(), prior_votes);
    assert_eq!(coordinated_roles().len(), 8);
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    INFERENCE_SCOPE_TRACE.with(|t| {
        let t = t.borrow();
        assert_eq!(
            (
                t.begin_attempts,
                t.opened,
                t.finished,
                t.active,
                t.abandoned
            ),
            (8, 7, 7, 0, 0)
        );
    });
    assert_eq!(pool.used_bytes().unwrap(), 384);
    drop(session);
    drop(request);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    reset();
}
