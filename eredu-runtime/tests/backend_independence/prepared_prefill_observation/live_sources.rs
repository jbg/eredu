//! Actual SessionPrefill boundary with current state and execution values.
//! Mock guards establish ordering only, not native accounting or settlement.
use super::*;
use eredu_runtime::inspection::{
    PrefillChunkRetentionContext, PrefillOpeningExecution, PrefillOpeningState,
    PreparedPrefillChunkRetention,
};
type State = DeviceState<FakeBackend, FakeLayerState>;
#[path = "live_sources/architecture.rs"]
mod architecture;
use architecture::{Chunked, Helper};
type LiveSession = ReplicatedTextSession<Chunked, FakeBackend, ReferenceTextMechanisms>;

#[derive(Debug, PartialEq, Eq)]
enum Log {
    Begin(u64),
    Mechanism(u64, i32, i32, bool),
    Opening(u64),
    Input(u64),
    Commit,
    Finish(bool),
}
struct Record {
    enabled: bool,
    fail_at: Option<u64>,
    events: Vec<Log>,
    request: Option<InferenceRequest>,
    epochs: Vec<DistributedCommitEpoch>,
    state_pointer: usize,
    helper_pointer: usize,
}
thread_local! {
    static LIVE: RefCell<Option<Rc<RefCell<Record>>>> = const { RefCell::new(None) };
}
pub(crate) fn enabled() -> bool {
    LIVE.with(|slot| slot.borrow().as_ref().is_some_and(|r| r.borrow().enabled))
}
pub(crate) fn prepare(
    state: &State,
    context: &PrefillChunkRetentionContext<'_>,
    execution: &PrefillOpeningExecution<'_, FakeTensor>,
) -> Result<(), &'static str> {
    guarded();
    LIVE.with(|slot| {
        let record = slot.borrow().as_ref().expect("opted in").clone();
        let mut r = record.borrow_mut();
        let start = context.chunk().input.start;
        assert_eq!(r.events.last(), Some(&Log::Begin(start)));
        let expected = r.request.as_ref().unwrap();
        expected.validate_same_request(context.request()).unwrap();
        assert_eq!(context.geometry(), expected.geometry());
        assert_eq!(
            context.chunk().position,
            context.geometry().cached_positions + start
        );
        assert!(!r.epochs.contains(&context.epoch()));
        r.epochs.push(context.epoch());
        if r.fail_at == Some(start) {
            return Err("original live-source mechanism failure");
        }
        let value = state.as_ref()[0].1.as_ref().unwrap();
        assert_eq!(value.0.as_ptr() as usize, r.state_pointer);
        let mut helpers = Vec::new();
        let complete = execution.visit(&mut |v| {
            assert_eq!(v.0.as_ptr() as usize, r.helper_pointer);
            helpers.push(v.0[0]);
        });
        assert_eq!(helpers.len(), 1, "one actual static helper, no cloned list");
        r.events
            .push(Log::Mechanism(start, value.0[0], helpers[0], complete));
        if complete {
            Ok(())
        } else {
            Err("incomplete current execution inventory")
        }
    })
}
fn guarded() {
    INFERENCE_SCOPE_TRACE.with(|s| {
        let s = s.borrow();
        assert_eq!(s.active, 1);
        assert_eq!(s.opened, s.finished + 1);
    });
}
struct Fixture(Rc<RefCell<Record>>);
impl Fixture {
    fn new(enabled: bool, fail_at: Option<u64>) -> Self {
        let r = Rc::new(RefCell::new(Record {
            enabled,
            fail_at,
            events: vec![],
            request: None,
            epochs: vec![],
            state_pointer: 0,
            helper_pointer: 0,
        }));
        LIVE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(r.clone());
        });
        INFERENCE_SCOPE_TRACE.with(|s| {
            *s.borrow_mut() = InferenceScopeTrace {
                required: true,
                ..Default::default()
            }
        });
        Self(r)
    }
    fn settled(&self) {
        INFERENCE_SCOPE_TRACE.with(|s| {
            let s = s.borrow();
            assert_eq!(s.active, 0);
            assert_eq!(s.opened, s.finished + s.abandoned);
            assert!(s.quarantine.is_empty());
        });
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        LIVE.with(|slot| *slot.borrow_mut() = None);
        INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
    }
}
struct Observer(Rc<RefCell<Record>>);
impl ActivationObserver<FakeTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn requires_prefill_opening_state(&self) -> bool {
        true
    }
    fn transactional(&self) -> bool {
        true
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        guarded();
        self.0
            .borrow_mut()
            .events
            .push(Log::Begin(chunk.input.start));
        Ok(())
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        c: &PrefillChunkRetentionContext<'_>,
        state: &PrefillOpeningState<'_, FakeTensor>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        guarded();
        let mut r = self.0.borrow_mut();
        let start = c.chunk().input.start;
        if r.enabled {
            assert!(matches!(r.events.last(),Some(Log::Mechanism(i,_,_,true)) if *i==start));
        } else {
            assert_eq!(r.events.last(), Some(&Log::Begin(start)));
        }
        let mut visits = 0;
        state
            .visit(&mut |value| {
                assert_eq!(value.0.as_ptr() as usize, r.state_pointer);
                visits += 1;
            })
            .map_err(Error::backend_source)?;
        assert_eq!(visits, 1);
        r.events.push(Log::Opening(start));
        Ok(None)
    }
    fn observe(&mut self, _: &str, _: &FakeTensor) -> Result<(), Error> {
        Ok(())
    }
    fn finish_transaction(&mut self, _: DistributedCommitEpoch, committed: bool) {
        if committed {
            self.0.borrow_mut().events.push(Log::Commit);
        }
    }
    fn finish_prefill(&mut self, committed: bool) {
        INFERENCE_SCOPE_TRACE.with(|s| assert_eq!(s.borrow().active, 0));
        self.0.borrow_mut().events.push(Log::Finish(committed));
    }
}
struct Input {
    geometry: InferenceGeometry,
    record: Rc<RefCell<Record>>,
}
impl<A> PreparedPrefillSource<A, FakeBackend, State> for Input
where
    A: LayeredArchitecture<FakeBackend, State, Error = Error> + 'static,
    for<'a> A: LayeredArchitecture<FakeBackend, State, Input<'a> = &'a FakeTensor>,
{
    type Chunk = FakeTensor;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(&self, chunk: &PrefillChunk, _: &()) -> Result<FakeTensor, Error> {
        guarded();
        let mut r = self.record.borrow_mut();
        assert_eq!(r.events.last(), Some(&Log::Opening(chunk.input.start)));
        r.events.push(Log::Input(chunk.input.start));
        Ok(FakeTensor(vec![
            7;
            usize::try_from(
                chunk.input.end - chunk.input.start
            )
            .unwrap()
        ]))
    }
    fn input<'a>(&'a self, value: &'a FakeTensor) -> &'a FakeTensor {
        value
    }
}
fn setup(
    f: &Fixture,
    complete: bool,
) -> (
    LiveSession,
    ReplicatedSessionCounters,
    Rc<Cell<usize>>,
    WorkingMemoryPool,
    InferenceRequest,
) {
    let counters = ReplicatedSessionCounters::default();
    let visits = Rc::new(Cell::new(0));
    let architecture = Chunked {
        inner: OrdinaryTextFixture {
            static_modules: FakeOperator,
            trace: vec![],
            counters: counters.clone(),
            inconsistent_transport: false,
            inconsistent_identity: false,
        },
        helper: Helper {
            value: FakeTensor(vec![41, 53]),
            visits: visits.clone(),
            complete,
        },
    };
    f.0.borrow_mut().helper_pointer = architecture.helper.value.0.as_ptr() as usize;
    let selected =
        selected_reference_text(&architecture.inner, LayerWeightResidency::FullyResident);
    let identity = selected.requirements().architecture_identity().to_owned();
    let contract = prepare_replicated_text_contract::<_, FakeBackend, State>(
        &architecture,
        None,
        selected,
        &identity,
        &(),
    )
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Default::default(),
        completions: Default::default(),
        counters: counters.clone(),
        fail_completion: Default::default(),
        fail_checkpoint: false,
        fail_construction_report: false,
        prepared_partition: None,
        prompt_cache: None,
    };
    let mut session = construct_replicated_text_session::<_, FakeBackend, _>(
        architecture,
        None,
        contract,
        mechanisms,
        &(),
    )
    .unwrap();
    let layout = session
        .inspect_runtime_state(|s| Ok(s.layout().clone()))
        .unwrap();
    let mut state = State::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState(4, Some(FakeTensor(vec![17, 29]))))
    })
    .unwrap();
    f.0.borrow_mut().state_pointer = state.as_ref()[0].1.as_ref().unwrap().0.as_ptr() as usize;
    session
        .exchange_prediction_target_state(&mut state, &())
        .unwrap();
    drop(state);
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 5,
        max_output_tokens: 0,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry),
        )
        .unwrap()
        .into();
    f.0.borrow_mut().request = Some(request.clone());
    visits.set(0);
    (session, counters, visits, pool, request)
}
fn run(
    session: &mut LiveSession,
    request: &InferenceRequest,
    f: &Fixture,
) -> Result<PrefillSourceOutcome<FakeTensor>, SessionError> {
    session.try_prefill_source_cancellable(
        Some(request),
        Some([1, 5]),
        std::num::NonZeroU64::new(2),
        |geometry| {
            Ok(Some(Input {
                geometry,
                record: f.0.clone(),
            }))
        },
        &GenerationCancellationToken::new(),
        &(),
        &mut Observer(f.0.clone()),
    )
}
fn mechanism_error(error: SessionError) -> &'static str {
    match error {
        ReplicatedTextSessionError::BeforeStateMutation(e) => mechanism_error(*e),
        ReplicatedTextSessionError::Mechanism(e) => e,
        other => panic!("lost mechanism error: {other}"),
    }
}
#[test]
fn live_state_and_current_execution_are_borrowed_before_each_uneven_chunk() {
    let f = Fixture::new(true, None);
    let (mut session, counters, visits, pool, request) = setup(&f, true);
    assert!(matches!(
        run(&mut session, &request, &f).unwrap(),
        PrefillSourceOutcome::Complete(_)
    ));
    assert_eq!(
        f.0.borrow().events,
        [
            Log::Begin(0),
            Log::Mechanism(0, 17, 41, true),
            Log::Opening(0),
            Log::Input(0),
            Log::Commit,
            Log::Begin(2),
            Log::Mechanism(2, 19, 43, true),
            Log::Opening(2),
            Log::Input(2),
            Log::Commit,
            Log::Begin(4),
            Log::Mechanism(4, 21, 45, true),
            Log::Opening(4),
            Log::Input(4),
            Log::Commit,
            Log::Finish(true),
        ]
    );
    assert_eq!(visits.get(), 3);
    assert_eq!(counters.snapshot().forward_calls, 3);
    assert_eq!(session.report().unwrap().state_report(), &[9]);
    assert_eq!(f.0.borrow().epochs.len(), 3);
    f.settled();
    drop((session, request, f));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn opt_out_performs_no_execution_visit_or_mechanism_callback() {
    let f = Fixture::new(false, None);
    let (mut session, counters, visits, pool, request) = setup(&f, true);
    assert!(matches!(
        run(&mut session, &request, &f).unwrap(),
        PrefillSourceOutcome::Complete(_)
    ));
    assert_eq!(visits.get(), 0);
    assert!(!f
        .0
        .borrow()
        .events
        .iter()
        .any(|e| matches!(e, Log::Mechanism(..))));
    assert_eq!(
        f.0.borrow()
            .events
            .iter()
            .filter(|e| matches!(e, Log::Input(_)))
            .count(),
        3
    );
    assert_eq!(counters.snapshot().forward_calls, 3);
    f.settled();
    drop((session, request, f));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn typed_mechanism_failure_preserves_prior_chunk_and_stops_later_preparation() {
    let f = Fixture::new(true, Some(2));
    let (mut session, counters, visits, pool, request) = setup(&f, true);
    assert_eq!(
        mechanism_error(run(&mut session, &request, &f).err().expect("failure")),
        "original live-source mechanism failure"
    );
    assert_eq!(
        f.0.borrow().events,
        [
            Log::Begin(0),
            Log::Mechanism(0, 17, 41, true),
            Log::Opening(0),
            Log::Input(0),
            Log::Commit,
            Log::Begin(2),
            Log::Finish(false)
        ]
    );
    assert_eq!(visits.get(), 1);
    assert_eq!(counters.snapshot().forward_calls, 1);
    assert_eq!(session.report().unwrap().state_report(), &[6]);
    f.settled();
    drop((session, request, f));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn incomplete_execution_preserves_partial_visits_and_rejects_before_source() {
    let f = Fixture::new(true, None);
    let (mut session, counters, visits, pool, request) = setup(&f, false);
    assert_eq!(
        mechanism_error(run(&mut session, &request, &f).err().expect("incomplete")),
        "incomplete current execution inventory"
    );
    assert_eq!(
        f.0.borrow().events,
        [
            Log::Begin(0),
            Log::Mechanism(0, 17, 41, false),
            Log::Finish(false)
        ]
    );
    assert_eq!(visits.get(), 1);
    assert_eq!(counters.snapshot().forward_calls, 0);
    assert_eq!(session.report().unwrap().state_report(), &[4]);
    f.settled();
    drop((session, request, f));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn mechanism_failure_enters_existing_partition_input_agreement() {
    let f = Fixture::new(true, Some(0));
    let calls = Rc::new(RefCell::new(vec![]));
    let (mut session, counters, _) = partitioned_agreement_session(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase: DistributedExecutionPhase::Commit,
            calls: calls.clone(),
            commits: Default::default(),
        }),
    );
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request: InferenceRequest = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry),
        )
        .unwrap()
        .into();
    f.0.borrow_mut().request = Some(request.clone());
    let result = session.try_prefill_source_cancellable(
        Some(&request),
        Some([1, 3]),
        std::num::NonZeroU64::new(1),
        |geometry| {
            Ok(Some(Input {
                geometry,
                record: f.0.clone(),
            }))
        },
        &GenerationCancellationToken::new(),
        &(),
        &mut Observer(f.0.clone()),
    );
    assert_eq!(
        mechanism_error(result.err().expect("failure")),
        "original live-source mechanism failure"
    );
    assert!(calls
        .borrow()
        .contains(&(DistributedExecutionPhase::InputPreparation, false)));
    assert!(!calls
        .borrow()
        .iter()
        .any(|(phase, _)| matches!(phase, DistributedExecutionPhase::Execution)));
    assert_eq!(counters.snapshot().forward_calls, 0);
    assert_eq!(f.0.borrow().events, [Log::Begin(0), Log::Finish(false)]);
    f.settled();
    drop((session, request, f));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
