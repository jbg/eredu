//! Actual replicated source scheduling and its prediction-level terminal hook.
//! Scalar fixture charges/guards are neutral evidence, not native byte facts.
use super::*;
use eredu_core::{GenerationCancellationToken, InferenceGeometry, OutputDemand};
use eredu_runtime::{
    prefill::PrefillChunk,
    replicated_session::{PrefillSourceOutcome, PreparedPrefillSource},
    working_memory::{InferenceRequest, MemoryLedger},
    ActivationObserver, ReplicatedTextSessionError,
};

type Session = ReplicatedTextSession<OrdinaryTextFixture, FakeBackend, ReferenceTextMechanisms>;
type SessionError = ReplicatedTextSessionError<Error, &'static str, &'static str>;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Begin,
    Prepare,
    Index,
    Settlement,
}
#[derive(Debug, PartialEq, Eq)]
enum Event {
    Begin(u64),
    Prepare(u64),
    Chunk(bool),
    Index(usize),
    Finish(bool, usize),
}
struct Trace {
    events: Vec<Event>,
    fault: Fault,
    identity: Arc<()>,
}
thread_local! {
    static INDEX_TRACE: RefCell<Option<Rc<RefCell<Trace>>>> = const { RefCell::new(None) };
}
struct Fixture(Rc<RefCell<Trace>>);
impl Fixture {
    fn new(fault: Fault) -> Self {
        let trace = Rc::new(RefCell::new(Trace {
            events: Vec::new(),
            fault,
            identity: Arc::new(()),
        }));
        INDEX_TRACE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(trace.clone());
        });
        INFERENCE_SCOPE_TRACE.with(|scope| {
            *scope.borrow_mut() = InferenceScopeTrace {
                required: true,
                ..Default::default()
            }
        });
        Self(trace)
    }
    fn observer(
        &self,
        cancellation: &GenerationCancellationToken,
        cancel_after: Option<u64>,
        prepared: bool,
    ) -> Observer {
        Observer {
            trace: self.0.clone(),
            cancellation: cancellation.clone(),
            cancel_after,
            current: 0,
            prepared,
        }
    }
    fn finish(&self, success: bool, indexed: bool) {
        let trace = self.0.borrow();
        assert_eq!(
            trace
                .events
                .iter()
                .filter(|e| matches!(e, Event::Finish(..)))
                .count(),
            1
        );
        let Event::Finish(actual, finished) = trace.events.last().unwrap() else {
            panic!("terminal callback must be last")
        };
        assert_eq!(*actual, success);
        let index = trace.events.iter().find_map(|event| match event {
            Event::Index(before) => Some(*before),
            _ => None,
        });
        assert_eq!(index.is_some(), indexed);
        if let Some(before) = index {
            assert_eq!(
                *finished,
                before + usize::from(trace.fault != Fault::Index),
                "successful indexing settles; failed indexing abandons before terminal callback"
            );
        }
        INFERENCE_SCOPE_TRACE.with(|scope| {
            let scope = scope.borrow();
            assert_eq!(scope.active, 0);
            assert_eq!(scope.opened, scope.finished + scope.abandoned);
        });
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        INDEX_TRACE.with(|slot| *slot.borrow_mut() = None);
        // Existing fixture's independent completion/teardown evidence releases
        // quarantined mock requests only after each test inspects their custody.
        INFERENCE_SCOPE_TRACE.with(|scope| *scope.borrow_mut() = InferenceScopeTrace::default());
    }
}
// Called only from the existing test mechanism. In all unrelated tests it is
// inactive. No production fault hook or generic runtime callback is introduced.
pub(super) fn before_index() -> Result<(), &'static str> {
    INDEX_TRACE.with(|slot| {
        let Some(trace) = slot.borrow().as_ref().cloned() else {
            return Ok(());
        };
        let finished = INFERENCE_SCOPE_TRACE.with(|scope| {
            let scope = scope.borrow();
            assert_eq!(scope.active, 1);
            scope.finished
        });
        let mut trace = trace.borrow_mut();
        trace.events.push(Event::Index(finished));
        match trace.fault {
            Fault::Index => Err("injected final score indexing failure"),
            Fault::Settlement => {
                INFERENCE_SCOPE_TRACE.with(|scope| scope.borrow_mut().fail_finish = true);
                Ok(())
            }
            _ => Ok(()),
        }
    })
}
#[derive(Debug, thiserror::Error)]
#[error("prepared prefill callback failed")]
struct CallbackFailure {
    point: Fault,
    identity: Arc<()>,
}
fn callback_error(trace: &Trace) -> Error {
    Error::backend_retained_source(CallbackFailure {
        point: trace.fault,
        identity: trace.identity.clone(),
    })
}
struct Source {
    geometry: InferenceGeometry,
    trace: Rc<RefCell<Trace>>,
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
    fn prepare_chunk(&self, chunk: &PrefillChunk, _: &()) -> Result<FakeTensor, Error> {
        check_inference_scope();
        let mut trace = self.trace.borrow_mut();
        assert_eq!(trace.events.last(), Some(&Event::Begin(chunk.input.start)));
        trace.events.push(Event::Prepare(chunk.input.start));
        if trace.fault == Fault::Prepare && chunk.input.start == 1 {
            return Err(callback_error(&trace));
        }
        Ok(FakeTensor(vec![3, 7]))
    }
    fn input<'a>(&'a self, chunk: &'a FakeTensor) -> &'a FakeTensor {
        chunk
    }
}
struct Observer {
    trace: Rc<RefCell<Trace>>,
    cancellation: GenerationCancellationToken,
    cancel_after: Option<u64>,
    current: u64,
    prepared: bool,
}
impl ActivationObserver<FakeTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn requires_prepared_traversal(&self) -> bool {
        self.prepared
    }
    fn transactional(&self) -> bool {
        true
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        check_inference_scope();
        assert_eq!(chunk.input.end, chunk.input.start + 1);
        self.current = chunk.input.start;
        let mut trace = self.trace.borrow_mut();
        trace.events.push(Event::Begin(self.current));
        if trace.fault == Fault::Begin && self.current == 1 {
            return Err(callback_error(&trace));
        }
        Ok(())
    }
    fn observe(&mut self, _: &str, value: &FakeTensor) -> Result<(), Error> {
        assert!(!value.0.is_empty());
        Ok(())
    }
    fn finish_transaction(&mut self, _: DistributedCommitEpoch, committed: bool) {
        self.trace.borrow_mut().events.push(Event::Chunk(committed));
        if committed && self.cancel_after == Some(self.current) {
            self.cancellation.cancel();
        }
    }
    fn finish_prefill(&mut self, committed: bool) {
        let finished = INFERENCE_SCOPE_TRACE.with(|scope| {
            let scope = scope.borrow();
            assert_eq!(scope.active, 0);
            scope.finished
        });
        self.trace
            .borrow_mut()
            .events
            .push(Event::Finish(committed, finished));
    }
}
fn session() -> (Session, ReplicatedSessionCounters) {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let identity = selected.requirements().architecture_identity().to_owned();
    let contract = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(&architecture, None, selected, &identity, &())
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
    (
        construct_replicated_text_session::<_, FakeBackend, _>(
            architecture,
            None,
            contract,
            mechanisms,
            &(),
        )
        .unwrap(),
        counters,
    )
}
fn request(
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
) -> (MemoryLedger, InferenceRequest) {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let pool = crate::memory::host_ledger(mock_reservation_bytes(), 0).unwrap();
    let request = pool
        .reserve(execution, &mock_inference_admission(geometry))
        .unwrap()
        .into();
    (pool, request)
}
fn run(
    session: &mut Session,
    request: &InferenceRequest,
    fixture: &Fixture,
    cancellation: &GenerationCancellationToken,
    observer: &mut Observer,
) -> Result<PrefillSourceOutcome<FakeTensor>, SessionError> {
    session.try_prefill_source_cancellable(
        Some(request),
        Some([1, 3]),
        std::num::NonZeroU64::new(1),
        |geometry| {
            Ok(Some(Source {
                geometry,
                trace: fixture.0.clone(),
            }))
        },
        cancellation,
        &(),
        observer,
    )
}

#[test]
fn span_callback_precedes_preparation_and_one_terminal_success_follows_final_index_settlement() {
    let fixture = Fixture::new(Fault::None);
    let (mut session, counters) = session();
    let (pool, request) = request(session.inference_execution_identity());
    let cancellation = GenerationCancellationToken::new();
    let mut observer = fixture.observer(&cancellation, None, true);
    let PrefillSourceOutcome::Complete(output) = run(
        &mut session,
        &request,
        &fixture,
        &cancellation,
        &mut observer,
    )
    .unwrap() else {
        panic!("complete prefill")
    };
    assert_eq!(output, FakeTensor(vec![5]));
    fixture.finish(true, true);
    assert_eq!(counters.snapshot().forward_calls, 3);
    assert_eq!(session.report().unwrap().state_report(), &[3]);
    let events = &fixture.0.borrow().events;
    for start in 0..3 {
        assert!(events
            .windows(2)
            .any(|e| e == [Event::Begin(start), Event::Prepare(start)]));
    }
    assert_eq!(
        events.iter().filter(|e| **e == Event::Chunk(true)).count(),
        3
    );
    drop((observer, request, session));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn cancellation_after_intermediate_or_final_committed_chunk_aborts_outer_prefill_without_index() {
    for last in [0, 2] {
        let fixture = Fixture::new(Fault::None);
        let (mut session, counters) = session();
        let (pool, request) = request(session.inference_execution_identity());
        let cancellation = GenerationCancellationToken::new();
        let mut observer = fixture.observer(&cancellation, Some(last), true);
        assert!(matches!(
            run(
                &mut session,
                &request,
                &fixture,
                &cancellation,
                &mut observer
            )
            .unwrap(),
            PrefillSourceOutcome::Cancelled
        ));
        fixture.finish(false, false);
        assert_eq!(counters.snapshot().forward_calls as u64, last + 1);
        // Cancellation does not roll back already committed individual chunks.
        assert_eq!(session.report().unwrap().state_report(), &[last as i32 + 1]);
        assert_eq!(
            fixture
                .0
                .borrow()
                .events
                .iter()
                .filter(|e| matches!(e, Event::Prepare(_)))
                .count() as u64,
            last + 1
        );
        drop((observer, request, session));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn final_index_and_final_reservation_failure_abort_once_after_all_chunk_commits() {
    for fault in [Fault::Index, Fault::Settlement] {
        let fixture = Fixture::new(fault);
        let (mut session, counters) = session();
        let (pool, request) = request(session.inference_execution_identity());
        let cancellation = GenerationCancellationToken::new();
        let mut observer = fixture.observer(&cancellation, None, true);
        let error = run(
            &mut session,
            &request,
            &fixture,
            &cancellation,
            &mut observer,
        )
        .err()
        .expect("terminal failure");
        assert!(
            matches!(error, ReplicatedTextSessionError::Mechanism(message) if message == if fault == Fault::Index { "injected final score indexing failure" } else { "injected reservation settlement failure" })
        );
        fixture.finish(false, true);
        assert_eq!(counters.snapshot().forward_calls, 3);
        assert_eq!(session.report().unwrap().state_report(), &[3]);
        INFERENCE_SCOPE_TRACE.with(|scope| {
            assert_eq!(
                scope.borrow().quarantine.len(),
                usize::from(fault == Fault::Settlement)
            )
        });
        drop((observer, request, session));
        assert_eq!(
            pool.live_charge_bytes().unwrap(),
            if fault == Fault::Settlement {
                mock_reservation_bytes()
            } else {
                0
            }
        );
        drop(fixture); // independent mock teardown, not terminal callback certification
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

fn original(error: &SessionError) -> &CallbackFailure {
    let architecture = match error {
        ReplicatedTextSessionError::BeforeStateMutation(error) => return original(error),
        ReplicatedTextSessionError::Architecture(error) => error,
        _ => panic!("original architecture callback error: {error}"),
    };
    let mut cause: &(dyn std::error::Error + 'static) = architecture;
    loop {
        if let Some(error) = cause.downcast_ref::<CallbackFailure>() {
            return error;
        }
        cause = cause.source().expect("original typed callback cause");
    }
}
#[test]
fn callback_failure_uses_existing_input_agreement_and_prevents_later_preparation() {
    for fault in [Fault::Begin, Fault::Prepare] {
        let fixture = Fixture::new(fault);
        let calls = Rc::new(RefCell::new(Vec::new()));
        let (mut session, counters, _) = partitioned_agreement_session(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
                failed_phase: DistributedExecutionPhase::Commit,
                calls: calls.clone(),
                commits: Default::default(),
            }),
        );
        let (pool, request) = request(session.inference_execution_identity());
        let cancellation = GenerationCancellationToken::new();
        let mut observer = fixture.observer(&cancellation, None, false);
        let error = session
            .try_prefill_source_cancellable(
                Some(&request),
                Some([1, 3]),
                std::num::NonZeroU64::new(1),
                |geometry| {
                    Ok(Some(Source {
                        geometry,
                        trace: fixture.0.clone(),
                    }))
                },
                &cancellation,
                &(),
                &mut observer,
            )
            .err()
            .expect("callback failure");
        let cause = original(&error);
        assert_eq!(cause.point, fault);
        assert!(Arc::ptr_eq(&cause.identity, &fixture.0.borrow().identity));
        assert!(calls
            .borrow()
            .contains(&(DistributedExecutionPhase::InputPreparation, false)));
        fixture.finish(false, false);
        assert_eq!(counters.snapshot().forward_calls, 1);
        assert_eq!(session.report().unwrap().state_report(), &[1]);
        let trace = fixture.0.borrow();
        assert!(!trace.events.contains(&Event::Begin(2)));
        assert_eq!(
            trace
                .events
                .iter()
                .filter(|e| matches!(e, Event::Prepare(_)))
                .count(),
            if fault == Fault::Begin { 1 } else { 2 }
        );
        drop(trace);
        drop((observer, request, session));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[path = "prepared_prefill_observation/retention.rs"]
pub(super) mod retention;

#[path = "prepared_prefill_observation/opening.rs"]
mod opening;

#[path = "prepared_prefill_observation/live_sources.rs"]
pub(super) mod live_sources;

#[path = "prepared_prefill_observation/sequence_gateway.rs"]
mod sequence_gateway;
