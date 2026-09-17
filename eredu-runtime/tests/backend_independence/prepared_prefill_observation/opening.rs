//! Real SessionPrefill source access; scalar guards are not native byte facts.
use super::*;
use eredu_runtime::inspection::{
    BorrowedActivationObserver, ObserverErrorBridge, PrefillChunkRetentionContext,
    PrefillOpeningState, PreparedPrefillChunkRetention,
};
use std::ops::Range;
type State = DeviceState<FakeBackend, FakeLayerState>;
#[derive(Debug, PartialEq, Eq)]
enum Log {
    Begin(Range<u64>),
    Open(u64, Vec<i32>),
    Legacy(u64),
    Prepare(u64),
    Commit,
    Finish(bool),
}
#[derive(Default)]
struct Record {
    events: Vec<Log>,
    prepared: usize,
    visited: usize,
    epochs: Vec<DistributedCommitEpoch>,
}
struct ScopeTrace;
impl ScopeTrace {
    fn new() -> Self {
        INFERENCE_SCOPE_TRACE.with(|s| {
            *s.borrow_mut() = InferenceScopeTrace {
                required: true,
                ..Default::default()
            }
        });
        Self
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
impl Drop for ScopeTrace {
    fn drop(&mut self) {
        INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
    }
}
fn guarded() {
    INFERENCE_SCOPE_TRACE.with(|s| {
        let s = s.borrow();
        assert_eq!(s.active, 1);
        assert_eq!(s.opened, s.finished + 1);
    });
}
fn seed(session: &mut Session) -> *const i32 {
    let layout = session
        .inspect_runtime_state(|s| Ok(s.layout().clone()))
        .unwrap();
    let mut state = State::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState(4, Some(FakeTensor(vec![17, 29]))))
    })
    .unwrap();
    let pointer = state.as_ref()[0].1.as_ref().unwrap().0.as_ptr();
    session
        .exchange_prediction_target_state(&mut state, &())
        .unwrap();
    drop(state);
    pointer
}
fn original_request(session: &Session) -> (WorkingMemoryPool, InferenceRequest) {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 4,
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let pool = WorkingMemoryPool::new(384, 0).unwrap();
    let request = pool
        .reserve(
            session.inference_execution_identity(),
            &mock_inference_admission(geometry),
        )
        .unwrap()
        .into();
    (pool, request)
}
#[derive(Debug, thiserror::Error)]
#[error("original opening source failure")]
struct Original(Arc<()>);
fn cause<'a>(mut e: &'a (dyn std::error::Error + 'static)) -> &'a Original {
    loop {
        if let Some(e) = e.downcast_ref::<Original>() {
            return e;
        }
        e = e.source().expect("original typed cause");
    }
}
fn architecture_error(error: SessionError) -> Error {
    match error {
        ReplicatedTextSessionError::BeforeStateMutation(error) => architecture_error(*error),
        ReplicatedTextSessionError::Architecture(error) => error,
        other => panic!("expected the original architecture error: {other}"),
    }
}
struct Input {
    geometry: InferenceGeometry,
    record: Rc<RefCell<Record>>,
    failure: Option<Arc<()>>,
}
impl PreparedPrefillSource<OrdinaryTextFixture, FakeBackend, State> for Input {
    type Chunk = FakeTensor;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(&self, chunk: &PrefillChunk, _: &()) -> Result<FakeTensor, Error> {
        guarded();
        let mut r = self.record.borrow_mut();
        assert!(
            matches!(r.events.last(),Some(Log::Open(start,_)|Log::Legacy(start)) if *start==chunk.input.start)
        );
        r.prepared += 1;
        r.events.push(Log::Prepare(chunk.input.start));
        if let Some(id) = &self.failure {
            return Err(Error::backend_source(Original(id.clone())));
        }
        // This original scalar architecture advances exactly one position.
        assert_eq!(chunk.input.end - chunk.input.start, 1);
        Ok(FakeTensor(vec![3, 7]))
    }
    fn input<'a>(&'a self, value: &'a FakeTensor) -> &'a FakeTensor {
        value
    }
}
struct Opening {
    record: Rc<RefCell<Record>>,
    pointer: *const i32,
    fail_at: Option<u64>,
    identity: Arc<()>,
    epoch: Option<DistributedCommitEpoch>,
}
impl Opening {
    fn new(record: Rc<RefCell<Record>>, pointer: *const i32) -> Self {
        Self {
            record,
            pointer,
            fail_at: None,
            identity: Arc::new(()),
            epoch: None,
        }
    }
}
impl ActivationObserver<FakeTensor, Error> for Opening {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn transactional(&self) -> bool {
        true
    }
    fn requires_prefill_opening_state(&self) -> bool {
        true
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        guarded();
        assert_eq!(chunk.position, 4 + chunk.input.start);
        self.record
            .borrow_mut()
            .events
            .push(Log::Begin(chunk.input.clone()));
        Ok(())
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        _: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        panic!("opening callback must not fall back")
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        c: &PrefillChunkRetentionContext<'_>,
        source: &PrefillOpeningState<'_, FakeTensor>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        guarded();
        assert_eq!(c.geometry().cached_positions, 4);
        assert_eq!(c.geometry(), c.request().geometry());
        assert_eq!(
            self.record.borrow().events.last(),
            Some(&Log::Begin(c.chunk().input.clone()))
        );
        let mut values = Vec::new();
        source
            .visit(&mut |v| {
                assert_eq!(v.0.as_ptr(), self.pointer);
                values.extend_from_slice(&v.0);
            })
            .map_err(Error::backend_source)?;
        let mut adapted = Vec::new();
        source
            .with_tensor_adapter(
                |v| &v.0,
                |s| {
                    s.visit(&mut |v| {
                        assert_eq!(v.as_ptr(), self.pointer);
                        adapted.extend_from_slice(v);
                    })
                },
            )
            .map_err(Error::backend_source)?;
        assert_eq!(
            adapted, values,
            "lexical bridge preserves the borrowed source"
        );
        let mut r = self.record.borrow_mut();
        r.visited += 1;
        if let Some(previous) = r.epochs.last() {
            assert_ne!(*previous, c.epoch());
        }
        r.epochs.push(c.epoch());
        r.events.push(Log::Open(c.chunk().input.start, values));
        self.epoch = Some(c.epoch());
        if self.fail_at == Some(c.chunk().input.start) {
            return Err(Error::backend_source(Original(self.identity.clone())));
        }
        Ok(None) // This view does not manufacture a bank, stamp or ticket.
    }
    fn prepare_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        assert_eq!(self.epoch, Some(epoch));
        Ok(())
    }
    fn observe(&mut self, _: &str, _: &FakeTensor) -> Result<(), Error> {
        Ok(())
    }
    fn finish_transaction(&mut self, _: DistributedCommitEpoch, committed: bool) {
        if committed {
            self.record.borrow_mut().events.push(Log::Commit);
        }
    }
    fn finish_prefill(&mut self, committed: bool) {
        INFERENCE_SCOPE_TRACE.with(|s| assert_eq!(s.borrow().active, 0));
        self.record.borrow_mut().events.push(Log::Finish(committed));
    }
}
fn run<O: ActivationObserver<FakeTensor, Error> + ?Sized>(
    session: &mut Session,
    request: &InferenceRequest,
    record: &Rc<RefCell<Record>>,
    cancel: &GenerationCancellationToken,
    observer: &mut O,
    failure: Option<Arc<()>>,
) -> Result<PrefillSourceOutcome<FakeTensor>, SessionError> {
    session.try_prefill_source_cancellable(
        Some(request),
        Some([1, 3]),
        std::num::NonZeroU64::new(1),
        |geometry| {
            Ok(Some(Input {
                geometry,
                record: record.clone(),
                failure,
            }))
        },
        cancel,
        &(),
        observer,
    )
}
#[test]
fn opening_precedes_source_under_guard_and_borrows_updated_cached_state() {
    let scopes = ScopeTrace::new();
    let (mut session, counters) = session();
    let pointer = seed(&mut session);
    let (pool, request) = original_request(&session);
    let record = Rc::new(RefCell::new(Record::default()));
    let mut observer = Opening::new(record.clone(), pointer);
    let mut borrowed = BorrowedActivationObserver(&mut observer);
    assert!(matches!(
        run(
            &mut session,
            &request,
            &record,
            &GenerationCancellationToken::new(),
            &mut borrowed,
            None
        )
        .unwrap(),
        PrefillSourceOutcome::Complete(_)
    ));
    assert_eq!(
        record.borrow().events,
        [
            Log::Begin(0..1),
            Log::Open(0, vec![17, 29]),
            Log::Prepare(0),
            Log::Commit,
            Log::Begin(1..2),
            Log::Open(1, vec![18, 29]),
            Log::Prepare(1),
            Log::Commit,
            Log::Begin(2..3),
            Log::Open(2, vec![19, 29]),
            Log::Prepare(2),
            Log::Commit,
            Log::Finish(true)
        ]
    );
    assert_eq!(counters.snapshot().forward_calls, 3);
    assert_eq!(session.report().unwrap().state_report(), &[7]);
    session
        .inspect_runtime_state(|s| {
            assert_eq!(s.as_ref()[0].1.as_ref().unwrap().0, [20, 29]);
            assert_eq!(s.as_ref()[0].1.as_ref().unwrap().0.as_ptr(), pointer);
            Ok(())
        })
        .unwrap();
    scopes.settled();
    drop((observer, session, request));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn opening_callback_failure_prevents_next_preparation_and_bridge_keeps_cause() {
    let scopes = ScopeTrace::new();
    let (mut session, counters) = session();
    let pointer = seed(&mut session);
    let (pool, request) = original_request(&session);
    let record = Rc::new(RefCell::new(Record::default()));
    let mut observer = Opening::new(record.clone(), pointer);
    observer.fail_at = Some(1);
    let identity = observer.identity.clone();
    let mut bridge = ObserverErrorBridge::new(
        &mut observer,
        |e: Error| e,
        |_: &Error| Error::backend("opening propagation signal"),
    );
    let result = run(
        &mut session,
        &request,
        &record,
        &GenerationCancellationToken::new(),
        &mut bridge,
        None,
    );
    let error = bridge
        .resolve(result.map_err(architecture_error))
        .err()
        .expect("opening callback failure");
    assert!(Arc::ptr_eq(&cause(&error).0, &identity));
    assert_eq!(record.borrow().visited, 2);
    assert_eq!(record.borrow().prepared, 1);
    assert_eq!(record.borrow().events.last(), Some(&Log::Finish(false)));
    assert_eq!(counters.snapshot().forward_calls, 1);
    assert_eq!(session.report().unwrap().state_report(), &[5]);
    scopes.settled();
    drop((observer, request, session));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn input_preparation_failure_after_opening_preserves_original_cause() {
    let scopes = ScopeTrace::new();
    let (mut session, counters) = session();
    let pointer = seed(&mut session);
    let (pool, request) = original_request(&session);
    let record = Rc::new(RefCell::new(Record::default()));
    let mut observer = Opening::new(record.clone(), pointer);
    let identity = Arc::new(());
    let error = run(
        &mut session,
        &request,
        &record,
        &GenerationCancellationToken::new(),
        &mut observer,
        Some(identity.clone()),
    )
    .err()
    .expect("input failure");
    let error = architecture_error(error);
    assert!(Arc::ptr_eq(&cause(&error).0, &identity));
    assert_eq!(record.borrow().visited, 1);
    assert_eq!(record.borrow().prepared, 1);
    assert_eq!(record.borrow().events.last(), Some(&Log::Finish(false)));
    assert_eq!(counters.snapshot().forward_calls, 0);
    assert_eq!(session.report().unwrap().state_report(), &[4]);
    scopes.settled();
    drop((observer, request, session));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn initial_cancellation_visits_no_opening_state_or_input() {
    let scopes = ScopeTrace::new();
    let (mut session, counters) = session();
    let pointer = seed(&mut session);
    let (pool, request) = original_request(&session);
    let record = Rc::new(RefCell::new(Record::default()));
    let mut observer = Opening::new(record.clone(), pointer);
    let cancel = GenerationCancellationToken::new();
    cancel.cancel();
    assert!(matches!(
        run(
            &mut session,
            &request,
            &record,
            &cancel,
            &mut observer,
            None
        )
        .unwrap(),
        PrefillSourceOutcome::Cancelled
    ));
    assert_eq!(record.borrow().events, [Log::Finish(false)]);
    assert_eq!(record.borrow().visited, 0);
    assert_eq!(record.borrow().prepared, 0);
    assert_eq!(counters.snapshot().forward_calls, 0);
    scopes.settled();
    drop((observer, request, session));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
struct Legacy {
    inner: Opening,
    opted_in: bool,
}
impl ActivationObserver<FakeTensor, Error> for Legacy {
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_prefill_opening_state(&self) -> bool {
        self.opted_in
    }
    fn begin_prefill_chunk(&mut self, c: &PrefillChunk) -> Result<(), Error> {
        self.inner.begin_prefill_chunk(c)
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        c: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        guarded();
        assert_eq!(
            self.inner.record.borrow().events.last(),
            Some(&Log::Begin(c.chunk().input.clone()))
        );
        self.inner
            .record
            .borrow_mut()
            .events
            .push(Log::Legacy(c.chunk().input.start));
        Ok(None)
    }
    fn observe(&mut self, _: &str, _: &FakeTensor) -> Result<(), Error> {
        Ok(())
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.inner.finish_prefill(committed);
    }
}
#[test]
fn old_retention_callback_remains_exact_with_default_or_opted_in_fallback() {
    for opted_in in [false, true] {
        let scopes = ScopeTrace::new();
        let (mut session, counters) = session();
        let pointer = seed(&mut session);
        let (pool, request) = original_request(&session);
        let record = Rc::new(RefCell::new(Record::default()));
        let mut observer = Legacy {
            inner: Opening::new(record.clone(), pointer),
            opted_in,
        };
        assert!(matches!(
            run(
                &mut session,
                &request,
                &record,
                &GenerationCancellationToken::new(),
                &mut observer,
                None
            )
            .unwrap(),
            PrefillSourceOutcome::Complete(_)
        ));
        assert_eq!(
            record.borrow().events,
            [
                Log::Begin(0..1),
                Log::Legacy(0),
                Log::Prepare(0),
                Log::Begin(1..2),
                Log::Legacy(1),
                Log::Prepare(1),
                Log::Begin(2..3),
                Log::Legacy(2),
                Log::Prepare(2),
                Log::Finish(true)
            ]
        );
        assert_eq!(record.borrow().visited, 0);
        assert_eq!(record.borrow().prepared, 3);
        assert_eq!(counters.snapshot().forward_calls, 3);
        scopes.settled();
        drop((observer, request, session));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[path = "opening/uneven.rs"]
mod uneven;

#[test]
fn whole_state_inventory_uses_actual_local_slots_independently_of_global_unit_addresses() {
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap();
    let layout =
        StateLayout::new(LayerSchedule::new(2, vec![policy.clone(), policy]).unwrap()).unwrap();
    let partition = eredu_runtime::PartitionState::new(layout, 5).unwrap();
    let state = State::create(partition.layout().clone(), |ordinal, _| {
        Ok::<_, Infallible>(FakeLayerState(
            4,
            Some(FakeTensor(vec![11 + ordinal as i32])),
        ))
    })
    .unwrap();
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [7]).unwrap();
    // Existing per-unit retention receives a local policy ordinal alongside a
    // global semantic address. Whole-state opening access needs neither guess.
    let second = RuntimeState::retained_values(&state, 1, units.address(6).unwrap())
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(second.0, [12]);
    let mut visited = Vec::new();
    RuntimeState::visit_all_retained_values(&state, &mut |value| {
        visited.push((value.0.as_ptr(), value.0[0]))
    })
    .unwrap();
    assert_eq!(
        visited,
        [
            (state.as_ref()[0].1.as_ref().unwrap().0.as_ptr(), 11),
            (state.as_ref()[1].1.as_ref().unwrap().0.as_ptr(), 12),
        ]
    );
    assert_eq!(partition.global_layer_offset(), 5);
    let absent = State::stateless();
    assert!(absent.optional_layout().is_none());
    let mut calls = 0;
    RuntimeState::visit_all_retained_values(&absent, &mut |_| calls += 1).unwrap();
    assert_eq!(calls, 0);
}
