//! Shared original admission -> authenticated Sequence -> actual bank fragments.
//! This integer fixture exercises portable behavior, not native allocation bounds.
use super::*;
use eredu_core::{capture::*, observation::TensorObservationData, *};
use eredu_nn::workspace::*;
use eredu_runtime::{
    capture::{FundedCaptureError, FundedCaptureSession, ScheduledCaptureBackend},
    inspection::{
        BorrowedActivationObserver, ObserverErrorBridge, PrefillChunkRetentionContext,
        PreparedPrefillChunkRetention, SettledPrefillChunkRetention,
    },
    layered::{
        BoundCaptureSelection, PrefillObservationDeclaration, PrefillReadoutStage,
        PreparedCaptureSelection,
    },
    working_memory::*,
};
#[path = "sequence_gateway/fixture.rs"]
mod fixture;
use fixture::{accept, candidate_quote, controls, quote, source, state, Rows};
type State = DeviceState<FakeBackend, FakeLayerState>;
type Session = ReplicatedTextSession<Rows, FakeBackend, ReferenceTextMechanisms>;
const PATHS: [&str; 5] = [
    "decoder.unit.0.input",
    "decoder.unit.0.input.effective",
    "decoder.unit.0.output",
    "decoder.unit.0.output.effective",
    "model.logits",
];
fn geometry(chunk: u64, output: OutputDemand) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: 5,
        max_output_tokens: 4,
        prefill_chunk_positions: chunk,
        output,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fail {
    None,
    Factory,
    Unavailable,
    Source,
    Observer,
    Retirement,
    Panic,
}
#[derive(Default)]
struct Events {
    factory: usize,
    prepared: Vec<std::ops::Range<u64>>,
    retired: Vec<std::ops::Range<u64>>,
    finishes: Vec<bool>,
}
#[derive(Debug, thiserror::Error)]
#[error("sequence gateway sentinel")]
struct Sentinel(Arc<()>);
struct Input {
    geometry: InferenceGeometry,
    events: Rc<RefCell<Events>>,
    fail: Fail,
    identity: Arc<()>,
}
impl PreparedPrefillSource<Rows, FakeBackend, State> for Input {
    type Chunk = FakeTensor;
    fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }
    fn prepare_chunk(&self, chunk: &PrefillChunk, _: &()) -> Result<FakeTensor, Error> {
        self.events.borrow_mut().prepared.push(chunk.input.clone());
        if chunk.input.start > 0 && self.fail == Fail::Source {
            return Err(Error::backend_retained_source(Sentinel(self.identity.clone())));
        }
        if chunk.input.start > 0 && self.fail == Fail::Panic {
            std::panic::panic_any(self.identity.clone());
        }
        Ok(FakeTensor(
            [2, 5, 7, 11, 13][chunk.input.start as usize..chunk.input.end as usize].to_vec(),
        ))
    }
    fn input<'a>(&'a self, value: &'a FakeTensor) -> &'a FakeTensor {
        value
    }
}
struct Backend {
    scope: Option<WorkingMemoryFundingScope>,
    segment: Option<CaptureSourceSegment>,
    events: Rc<RefCell<Events>>,
    fail: Fail,
    identity: Arc<()>,
    cancellation: GenerationCancellationToken,
    cancel_at: Option<u64>,
}
fn usage(n: usize) -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        retained_bytes: n as u64 * 4,
        host_bytes: n as u64 * 4,
        encoded_bytes: 4096,
    }
}
impl ScheduledCaptureBackend for Backend {
    type Tensor = FakeTensor;
    type Error = Error;
    fn prepare_prefill_chunk_retention(
        &mut self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, FundedCaptureError<Error>> {
        assert!(self.segment.is_none());
        let (segment, ticket) = bootstrap
            .begin_segment(self.scope.as_mut().unwrap(), context)
            .map_err(CaptureRunHostError::from)?;
        self.segment = Some(segment);
        Ok(Some(ticket))
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.events
            .borrow_mut()
            .retired
            .push(ticket.chunk().input.clone());
        if self.fail == Fail::Retirement {
            return Err(FundedCaptureError::Backend(Error::backend_retained_source(
                Sentinel(self.identity.clone()),
            )));
        }
        if self.cancel_at == Some(ticket.chunk().input.end) {
            self.cancellation.cancel();
        }
        let parcel = self
            .segment
            .take()
            .unwrap()
            .take_settled_sources(self.scope.as_mut().unwrap(), &ticket)
            .map_err(CaptureRunHostError::from)?;
        // CPU integer reads/writes finished synchronously, with no native graph.
        drop(parcel);
        Ok(())
    }
    fn validate_source(
        &self,
        value: &FakeTensor,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<checkpoint::TensorDtype, Error> {
        assert_eq!(g.source_shape(), &[value.0.len()]);
        Ok(checkpoint::TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &FakeTensor,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(g.elements()))
    }
    fn transform(
        &mut self,
        value: &FakeTensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        let mut writer = claim.prepare().map_err(Error::backend_retained_source)?;
        for &n in &value.0 {
            writer.push_f32(n as f32).map_err(Error::backend_retained_source)?;
        }
        Ok(writer.finish().unwrap())
    }
    fn validate_prefill_source(
        &self,
        value: &FakeTensor,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<checkpoint::TensorDtype, FundedCaptureError<Error>> {
        assert_eq!(fragment.source_shape(), &[value.0.len()]);
        Ok(checkpoint::TensorDtype::F32)
    }
    fn estimate_prefill(
        &self,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(g.elements()))
    }
    fn transform_prefill_fragment(
        &mut self,
        value: &FakeTensor,
        claim: CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        if self.fail == Fail::Observer {
            return Err(FundedCaptureError::Backend(Error::backend_retained_source(
                Sentinel(self.identity.clone()),
            )));
        }
        let fragment = claim.fragment();
        let mut writer = claim.prepare()?;
        for item in fragment.mappings() {
            writer.push_f32(value.0[item.source_index()] as f32)?;
        }
        writer.finish()?;
        Ok(())
    }
}
struct Original {
    pool: WorkingMemoryPool,
    root: WorkingMemoryStorage<u32>,
    source: SharedCapturePlan,
    selection: PreparedCaptureSelection,
    owner: OwnedTextSpanWorkspace,
    request: InferenceRequest,
    run: WorkingMemoryFundingRun,
}
impl Original {
    fn new(session: &Session, g: InferenceGeometry, body: bool, seal: bool) -> Self {
        Self::with_explicit_source(session, g, body, seal, false)
    }
    fn with_explicit_source(
        session: &Session,
        g: InferenceGeometry,
        body: bool,
        seal: bool,
        explicit_source: bool,
    ) -> Self {
        let pool = WorkingMemoryPool::new(4_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let source = source(g, body);
        let selection = session
            .shared_observation_paths()
            .unwrap()
            .prepare_capture_selection(&source)
            .unwrap();
        // The residual decoder pin moves to the run. Only this explicit join
        // also belongs to escaped full span custody; apply the same policy to
        // every cold candidate rather than rebinding after acceptance.
        let joined = explicit_source.then_some(&root);
        let q = candidate_quote(&pool, &source, &selection, g, seal, joined);
        let exact = 64 + q.incremental_bytes();
        let mut rejected = Vec::new();
        let rejection = accept(session, &pool, g, exact - 1, |candidate| {
            let quoted = if candidate == g {
                q.clone()
            } else {
                candidate_quote(&pool, &source, &selection, candidate, seal, joined)
            };
            assert_eq!(quoted.geometry(), candidate);
            // For this scalar fixture, smaller chunks keep the same tensor/H
            // demand and may enlarge the actual retained record capacity.
            // Inspect each real quote; do not invent a retry budget failure.
            assert!(quoted.incremental_bytes() >= q.incremental_bytes());
            rejected.push((candidate, quoted.incremental_bytes()));
            quoted
        })
        .err()
        .expect("one-byte-short original admission must reject every real candidate");
        assert_eq!(
            rejected
                .iter()
                .map(|(g, _)| g.prefill_chunk_positions)
                .collect::<Vec<_>>(),
            (1..=g.prefill_chunk_positions).rev().collect::<Vec<_>>()
        );
        let last_required = rejected.last().unwrap().1;
        assert!(
            matches!(
                &rejection,
                PrefillPlanningError::Reservation(WorkingMemoryError::BudgetExceeded {
                    required_bytes, available_bytes,
                }) if *required_bytes == last_required && *available_bytes == q.incremental_bytes() - 1
            ),
            "{rejection:?}"
        );
        assert_eq!(pool.used_bytes().unwrap(), 64);
        let mut admitted_candidates = Vec::new();
        let (r, accepted) = accept(session, &pool, g, exact, |candidate| {
            admitted_candidates.push(candidate);
            assert_eq!(candidate, g, "the exact original candidate must fit");
            q.clone()
        })
        .unwrap();
        assert_eq!(admitted_candidates, [g]);
        assert_eq!(r.geometry(), g);
        assert_eq!(r.bytes(), q.incremental_bytes());
        assert_eq!(r.admission().incremental_required_bytes, q.incremental_bytes());
        assert!(accepted
            .span_workspace()
            .plan()
            .same_plan(q.span_workspace().plan()));
        assert_eq!(pool.used_bytes().unwrap(), exact);
        let (r, run) = r.into_funding().unwrap();
        let (owner, _) = accepted.into_funded_text_span_workspace(&run, &r).unwrap();
        Self {
            pool,
            root,
            source,
            selection,
            owner,
            request: InferenceRequest::from(&r),
            run,
        }
    }
    fn bound(&self) -> BoundCaptureSelection<'_> {
        self.selection
            .bind_geometry(self.request.geometry())
            .unwrap()
    }
    fn bank(&self) -> FundedCaptureSession {
        self.run
            .prepare_capture_run(
                self.request.memory_reservation().unwrap(),
                CaptureRunHostPlan::prepare(&self.source).unwrap(),
            )
            .unwrap()
            .into_capture_session()
            .unwrap()
    }
    fn backend(
        &self,
        events: Rc<RefCell<Events>>,
        fail: Fail,
        cancellation: &GenerationCancellationToken,
        cancel_at: Option<u64>,
    ) -> Backend {
        Backend {
            scope: Some(self.run.scope().unwrap()),
            segment: None,
            events,
            fail,
            identity: Arc::new(()),
            cancellation: cancellation.clone(),
            cancel_at,
        }
    }
}
// This fixture-only adapter observes notifications while forwarding every
// callback used by this exact ordinary tensor/transaction route. It lends the
// actual accepted view; it cannot create one or synthesize a capture event.
struct Notifications<'a> {
    inner: &'a mut dyn ActivationObserver<FakeTensor, Error>,
    events: &'a Rc<RefCell<Events>>,
}
impl ActivationObserver<FakeTensor, Error> for Notifications<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        self.inner.requires_prepared_traversal()
    }
    fn requires_sequence_readout(&self) -> bool {
        self.inner.requires_sequence_readout()
    }
    fn admitted_prefill_capture(&self) -> Option<&AdmittedPrefillCapture<'_>> {
        self.inner.admitted_prefill_capture()
    }
    fn transactional(&self) -> bool {
        self.inner.transactional()
    }
    fn begin_prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<(), Error> {
        self.inner.begin_prefill_chunk(chunk)
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        cx: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, Error> {
        self.inner.prepare_prefill_chunk_retention(cx)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), Error> {
        self.inner.retire_prefill_chunk_retention(ticket)
    }
    fn prepare_transaction(
        &mut self,
        e: DistributedCommitEpoch,
        pass: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        self.inner.prepare_transaction(e, pass)
    }
    fn coordinate_transaction(&mut self, e: DistributedCommitEpoch) -> Result<(), Error> {
        self.inner.coordinate_transaction(e)
    }
    fn complete_transaction(&mut self, e: DistributedCommitEpoch) -> Result<(), Error> {
        self.inner.complete_transaction(e)
    }
    fn finish_transaction(&mut self, e: DistributedCommitEpoch, committed: bool) {
        self.inner.finish_transaction(e, committed);
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.events.borrow_mut().finishes.push(committed);
        self.inner.finish_prefill(committed);
    }
    fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
        self.inner.observe(path, value)
    }
    fn intervene(&mut self, path: &str, value: &FakeTensor) -> Result<Option<FakeTensor>, Error> {
        self.inner.intervene(path, value)
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.inner.finish()
    }
}
fn invoke(
    session: &mut Session,
    original: &Original,
    events: &Rc<RefCell<Events>>,
    fail: Fail,
    identity: &Arc<()>,
    cancellation: &GenerationCancellationToken,
    observer: &mut dyn ActivationObserver<FakeTensor, Error>,
) -> Result<PrefillSourceOutcome<FakeTensor>, SessionError> {
    let mut observer = Notifications {
        inner: observer,
        events,
    };
    session.try_prefill_source_cancellable(
        Some(&original.request),
        Some([1, 5]),
        std::num::NonZeroU64::new(original.request.geometry().prefill_chunk_positions),
        |geometry| {
            events.borrow_mut().factory += 1;
            match fail {
                Fail::Factory => Err(Error::backend_retained_source(Sentinel(identity.clone()))),
                Fail::Unavailable => Ok(None),
                _ => Ok(Some(Input {
                    geometry,
                    events: events.clone(),
                    fail,
                    identity: identity.clone(),
                })),
            }
        },
        cancellation,
        &(),
        &mut observer,
    )
}
fn frame_values(frame: &SharedCapturedStep) -> Vec<Vec<f32>> {
    frame
        .records()
        .iter()
        .map(|record| {
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            let TensorObservationData::F32(values) =
                record.payload.as_ref().unwrap().as_tensor().unwrap().data()
            else {
                panic!("F32")
            };
            values.to_vec()
        })
        .collect()
}
fn run(
    chunk: u64,
    body: bool,
    explicit_source: bool,
) -> (Vec<i32>, Vec<Vec<Vec<f32>>>, (i32, Vec<i32>)) {
    let mut session = fixture::session();
    let original = Original::with_explicit_source(
        &session,
        geometry(
            chunk,
            if body {
                OutputDemand::LastPosition
            } else {
                OutputDemand::Sequence
            },
        ),
        body,
        true,
        explicit_source,
    );
    let events = Rc::new(RefCell::new(Events::default()));
    let cancel = GenerationCancellationToken::new();
    let mut backend = original.backend(events.clone(), Fail::None, &cancel, None);
    let mut bank = original.bank();
    let accepted = original.owner.prefill_capture(original.bound()).unwrap();
    let result = bank
        .with_admitted_prefill_observer(
            &mut backend,
            &accepted,
            &Error::backend_retained_source,
            |observer| {
                let mut borrowed = BorrowedActivationObserver(observer);
                let mut bridge = ObserverErrorBridge::new(
                    &mut borrowed,
                    |e: Error| e,
                    |_: &Error| Error::backend_retained_source(Sentinel(Arc::new(()))),
                );
                assert!(std::ptr::eq(
                    bridge.admitted_prefill_capture().unwrap(),
                    &accepted
                ));
                let result = invoke(
                    &mut session,
                    &original,
                    &events,
                    Fail::None,
                    &Arc::new(()),
                    &cancel,
                    &mut bridge,
                );
                result
            },
        )
        .unwrap()
        .unwrap();
    let PrefillSourceOutcome::Complete(output) = result else {
        panic!("complete")
    };
    assert_eq!(output.0, vec![121]);
    assert_eq!(state(&session), (7, vec![55, 29]));
    assert_eq!(events.borrow().factory, 1);
    assert_eq!(events.borrow().finishes, [true]);
    assert_eq!(events.borrow().prepared, events.borrow().retired);
    let first = bank.take_shared_step().unwrap().unwrap();
    assert_eq!(first.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(first.prediction_index(), 0);
    assert_eq!(first.phase(), CapturePhase::Prefill);
    let mut frames = vec![first];
    let mut outputs = output.0;
    for prediction in 1..4 {
        let value = FakeTensor(vec![prediction as i32 + 1]);
        let result = bank
            .with_observer(&mut backend, prediction, &Error::backend_retained_source, |o| {
                session.decode_input_with_observer(&value, &(), o)
            })
            .unwrap()
            .unwrap();
        outputs.extend(result.0);
        let decoded = (2..=prediction as i32 + 1).sum::<i32>();
        assert_eq!(
            state(&session),
            (7 + prediction as i32, vec![55 + decoded, 29])
        );
        frames.push(bank.take_shared_step().unwrap().unwrap());
    }
    let values = frames.iter().map(frame_values).collect();
    let final_state = state(&session);
    let pool = original.pool.clone();
    let h = CaptureRunHostPlan::prepare(&original.source)
        .unwrap()
        .initialization_peak_bytes();
    let protected = original.owner.protected_host_bytes();
    let plan = original.owner.workspace().plan().clone();
    drop(accepted);
    backend.scope.take().unwrap().certify().unwrap();
    drop((backend, bank, session));
    drop(original);
    let source_bytes = if explicit_source { 64 } else { 0 };
    assert_eq!(pool.used_bytes().unwrap(), h + protected + source_bytes);
    let source_pin = pool.pin_registered_storage([(1u32, 64)]);
    if explicit_source {
        drop(source_pin.expect("full original source witness still pins key 1"));
    } else {
        assert!(matches!(
            source_pin,
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    drop(frames);
    assert_eq!(pool.used_bytes().unwrap(), protected + source_bytes);
    drop(plan);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 64)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    (outputs, values, final_state)
}
#[test]
fn admitted_sequence_gateway_preserves_every_real_row_state_decode_and_original_custody() {
    let reference = run(5, false, false);
    for chunk in [1, 2, 3] {
        assert_eq!(run(chunk, false, false), reference);
    }
    // The same numerical/capture assertions also run with a genuine explicit
    // source join, whose extra lifetime differs from a residual decoder pin.
    assert_eq!(run(2, false, true), reference);
    let body = run(2, true, false);
    assert_eq!(body.0, reference.0);
    assert_eq!(body.2, reference.2);
    for (body, full) in body.1.iter().zip(&reference.1) {
        assert_eq!(body, &full[..4]);
    }
}

#[test]
fn admitted_sequence_seal_rejects_missing_duplicate_equal_content_foreign_sources_and_plans() {
    let session = fixture::session();
    let g = geometry(2, OutputDemand::Sequence);
    let original = Original::new(&session, g, false, true);
    let q = quote(&original.pool, &original.source, g);
    let c = controls(&q, &original.source);
    let bound = original.bound();
    let sealed = c.clone().with_prefill_capture_selection(bound).unwrap();
    assert!(!sealed.same_binding(&c));
    assert!(matches!(
        sealed.clone().with_prefill_capture_selection(bound),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let foreign_source = SharedCapturePlan::new(original.source.admission().clone());
    let other = session
        .shared_observation_paths()
        .unwrap()
        .prepare_capture_selection(&foreign_source)
        .unwrap();
    assert!(matches!(
        c.with_prefill_capture_selection(other.bind_geometry(g).unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        original
            .owner
            .prefill_capture(other.bind_geometry(g).unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    // A second real loaded runtime has equal strings/declarations but a distinct
    // physical path source. Current semantic validity cannot replace the old seal.
    let second = self::fixture::session();
    let paths = second.shared_observation_paths().unwrap();
    assert_eq!(
        paths.unit_paths(0, 0),
        session.shared_observation_paths().unwrap().unit_paths(0, 0)
    );
    let selected = paths.prepare_capture_selection(&original.source).unwrap();
    assert!(matches!(
        original
            .owner
            .prefill_capture(selected.bind_geometry(g).unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let changed = InferenceGeometry {
        prefill_chunk_positions: 1,
        ..g
    };
    assert!(matches!(
        original
            .owner
            .prefill_capture(original.selection.bind_geometry(changed).unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let unknown = PreparedTextControlWorkspace::prepare(
        &original.source,
        g,
        q.span_workspace().plan(),
        TextHostControlFacts::new(None, Some(0), Some(0)),
    )
    .unwrap()
    .with_prefill_capture_selection(bound)
    .unwrap();
    assert!(matches!(
        q.clone().with_span_workspace_and_text_controls(unknown),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::UnknownBound
        ))
    ));
    let missing = Original::new(&session, g, false, false);
    assert!(matches!(
        missing.owner.prefill_capture(missing.bound()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let another = quote(&original.pool, &original.source, g);
    assert!(matches!(
        another.with_span_workspace_and_text_controls(sealed),
        Err(ResidualQuoteError::Storage(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let held = original.pool.used_bytes().unwrap();
    drop(original.run.scope().unwrap());
    assert!(matches!(
        original.owner.prefill_capture(bound),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(original.pool.used_bytes().unwrap(), held);
}

#[test]
fn sequence_gateway_rejects_bound_only_or_foreign_reservation_before_factory_and_claim() {
    let mut session = fixture::session();
    let g = geometry(2, OutputDemand::Sequence);
    let original = Original::new(&session, g, false, true);
    let other = Original::new(&session, g, false, true);
    let events = Rc::new(RefCell::new(Events::default()));
    let cancel = GenerationCancellationToken::new();
    let mut backend = original.backend(events.clone(), Fail::None, &cancel, None);
    let mut bank = original.bank();
    let before = state(&session);
    let result = bank
        .with_prefill_observer(
            &mut backend,
            original.bound(),
            &Error::backend_retained_source,
            |observer| {
                assert!(observer.requires_sequence_readout());
                assert!(observer.admitted_prefill_capture().is_none());
                invoke(
                    &mut session,
                    &original,
                    &events,
                    Fail::None,
                    &Arc::new(()),
                    &cancel,
                    observer,
                )
            },
        )
        .unwrap();
    assert!(
        matches!(result,Err(ReplicatedTextSessionError::BeforeStateMutation(error)) if matches!(*error,ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::OutputDemandMismatch {..})))
    );
    let admitted = original.owner.prefill_capture(original.bound()).unwrap();
    let result = bank
        .with_admitted_prefill_observer(
            &mut backend,
            &admitted,
            &Error::backend_retained_source,
            |observer| {
                invoke(
                    &mut session,
                    &other,
                    &events,
                    Fail::None,
                    &Arc::new(()),
                    &cancel,
                    observer,
                )
            },
        )
        .unwrap();
    assert!(
        matches!(result,Err(ReplicatedTextSessionError::BeforeStateMutation(error)) if matches!(*error,ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch)))
    );
    assert_eq!(state(&session), before);
    assert_eq!(events.borrow().factory, 0);
    assert!(events.borrow().finishes.is_empty());
    assert_eq!(bank.spent_steps(), 0);
    // Both prior checks precede the original one-use driver claim.
    let result = bank
        .with_admitted_prefill_observer(
            &mut backend,
            &admitted,
            &Error::backend_retained_source,
            |observer| {
                invoke(
                    &mut session,
                    &original,
                    &events,
                    Fail::None,
                    &Arc::new(()),
                    &cancel,
                    observer,
                )
            },
        )
        .unwrap()
        .unwrap();
    assert!(matches!(result, PrefillSourceOutcome::Complete(_)));
    assert_eq!(events.borrow().factory, 1);
    bank.take_shared_step().unwrap().unwrap();
    backend.scope.take().unwrap().certify().unwrap();
}

#[test]
fn sequence_gateway_checks_current_token_and_state_only_before_source_work() {
    for state_only in [false, true] {
        let mut session = fixture::session();
        let g = geometry(
            2,
            if state_only {
                OutputDemand::StateOnly
            } else {
                OutputDemand::Sequence
            },
        );
        let original = Original::new(&session, g, state_only, true);
        if !state_only {
            // Parameter replacement publication invalidates the token while preserving
            // the immutable original path source and quote.
            session.publish_parameter_replacements(&Default::default(), false).unwrap();
        }
        let before = state(&session);
        let events = Rc::new(RefCell::new(Events::default()));
        let cancel = GenerationCancellationToken::new();
        let mut backend = original.backend(events.clone(), Fail::None, &cancel, None);
        let mut bank = original.bank();
        let admitted = original.owner.prefill_capture(original.bound()).unwrap();
        let result = bank
            .with_admitted_prefill_observer(
                &mut backend,
                &admitted,
                &Error::backend_retained_source,
                |observer| {
                    invoke(
                        &mut session,
                        &original,
                        &events,
                        Fail::None,
                        &Arc::new(()),
                        &cancel,
                        observer,
                    )
                },
            )
            .unwrap();
        match result {
            Err(ReplicatedTextSessionError::BeforeStateMutation(error)) if state_only => {
                assert!(matches!(
                    *error,
                    ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::OutputDemandMismatch { .. }
                    )
                ))
            }
            Err(ReplicatedTextSessionError::BeforeStateMutation(error)) => assert!(matches!(
                *error,
                ReplicatedTextSessionError::PreparedObservation(_)
            )),
            _ => panic!("must reject before source"),
        }
        assert_eq!(events.borrow().factory, 0);
        assert_eq!(bank.spent_steps(), 0);
        assert_eq!(state(&session), before);
        backend.scope.take().unwrap().certify().unwrap();
    }
}

fn sentinel(error: &SessionError) -> &Sentinel {
    match error {
        ReplicatedTextSessionError::BeforeStateMutation(inner) => sentinel(inner),
        ReplicatedTextSessionError::Architecture(inner) => {
            let mut error: &dyn std::error::Error = inner;
            loop {
                if let Some(cause) = error.downcast_ref::<Sentinel>() {
                    return cause;
                }
                error = error.source().expect("original source");
            }
        }
        _ => panic!("unexpected failure: {error:?}"),
    }
}
#[test]
fn sequence_gateway_preserves_factory_observer_source_retirement_and_index_failures() {
    for fail in [
        Fail::Factory,
        Fail::Unavailable,
        Fail::Source,
        Fail::Observer,
        Fail::Retirement,
    ] {
        let mut session = fixture::session();
        let original = Original::new(&session, geometry(2, OutputDemand::Sequence), false, true);
        let events = Rc::new(RefCell::new(Events::default()));
        let cancel = GenerationCancellationToken::new();
        let mut backend = original.backend(events.clone(), fail, &cancel, None);
        let identity = backend.identity.clone();
        let mut bank = original.bank();
        let admitted = original.owner.prefill_capture(original.bound()).unwrap();
        let result = bank
            .with_admitted_prefill_observer(
                &mut backend,
                &admitted,
                &Error::backend_retained_source,
                |observer| {
                    invoke(
                        &mut session,
                        &original,
                        &events,
                        fail,
                        &identity,
                        &cancel,
                        observer,
                    )
                },
            )
            .unwrap();
        let error = result.err().expect("exact selected failure must execute");
        if fail == Fail::Unavailable {
            assert!(
                matches!(error,ReplicatedTextSessionError::BeforeStateMutation(error) if matches!(*error,ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::PreparedSourceUnavailable)))
            );
        } else {
            assert!(Arc::ptr_eq(&sentinel(&error).0, &identity));
        }
        assert_eq!(events.borrow().factory, 1);
        if matches!(fail, Fail::Factory | Fail::Unavailable) {
            assert!(events.borrow().finishes.is_empty());
            assert_eq!(bank.spent_steps(), 0);
            assert_eq!(state(&session), (2, vec![17, 29]));
            assert!(!bank.has_pending_step());
        } else {
            let frame = bank.take_shared_step().unwrap().unwrap();
            assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
            assert_eq!(frame.prediction_index(), 0);
            assert_eq!(frame.phase(), CapturePhase::Prefill);
            assert!(
                frame.records().iter().all(|r| r.payload.is_none()),
                "incomplete fragments never become delivered tensors"
            );
            assert_eq!(events.borrow().finishes, [false]);
            assert_eq!(bank.spent_steps(), 1);
            assert_eq!(
                session.report().unwrap().state_report(),
                &[if matches!(fail, Fail::Source | Fail::Retirement) {
                    4
                } else {
                    2
                }]
            );
        }
        // Failed native-equivalent scopes remain conservative; resetting the
        // fixture's independent quarantine does not claim native completion.
        drop((bank, backend, admitted));
        drop((original, session));
        INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
    }
    let mut session = fixture::session();
    let original = Original::new(&session, geometry(2, OutputDemand::Sequence), false, true);
    let fixture = Fixture::new(Fault::Index);
    let events = Rc::new(RefCell::new(Events::default()));
    let cancel = GenerationCancellationToken::new();
    let mut backend = original.backend(events.clone(), Fail::None, &cancel, None);
    let mut bank = original.bank();
    let admitted = original.owner.prefill_capture(original.bound()).unwrap();
    let error = bank
        .with_admitted_prefill_observer(&mut backend, &admitted, &Error::backend_retained_source, |o| {
            invoke(
                &mut session,
                &original,
                &events,
                Fail::None,
                &Arc::new(()),
                &cancel,
                o,
            )
        })
        .unwrap()
        .err()
        .unwrap();
    assert!(matches!(
        error,
        ReplicatedTextSessionError::Mechanism("injected final score indexing failure")
    ));
    assert_eq!(events.borrow().finishes, [false]);
    assert!(fixture
        .0
        .borrow()
        .events
        .iter()
        .any(|e| matches!(e, Event::Index(_))));
    assert_eq!(session.report().unwrap().state_report(), &[7]);
    assert_eq!(
        bank.take_shared_step().unwrap().unwrap().outcome(),
        CaptureStepOutcome::Aborted
    );
}

#[test]
fn sequence_gateway_cancellation_and_panic_keep_real_committed_frontier_and_abort_frame() {
    for end in [0, 2, 5] {
        let mut session = fixture::session();
        let original = Original::new(&session, geometry(2, OutputDemand::Sequence), false, true);
        let events = Rc::new(RefCell::new(Events::default()));
        let cancel = GenerationCancellationToken::new();
        if end == 0 {
            cancel.cancel();
        }
        let mut backend = original.backend(events.clone(), Fail::None, &cancel, Some(end));
        let mut bank = original.bank();
        let admitted = original.owner.prefill_capture(original.bound()).unwrap();
        let result = bank
            .with_admitted_prefill_observer(&mut backend, &admitted, &Error::backend_retained_source, |o| {
                invoke(
                    &mut session,
                    &original,
                    &events,
                    Fail::None,
                    &Arc::new(()),
                    &cancel,
                    o,
                )
            })
            .unwrap()
            .unwrap();
        assert!(matches!(result, PrefillSourceOutcome::Cancelled));
        assert_eq!(events.borrow().factory, 1);
        assert_eq!(events.borrow().finishes, [false]);
        assert_eq!(state(&session).0, 2 + end as i32);
        let frame = bank.take_shared_step().unwrap();
        if end == 0 {
            assert!(frame.is_none());
            assert_eq!(bank.spent_steps(), 0);
        } else {
            assert_eq!(frame.unwrap().outcome(), CaptureStepOutcome::Aborted);
        }
        backend.scope.take().unwrap().certify().unwrap();
    }
    let mut session = fixture::session();
    let original = Original::new(&session, geometry(2, OutputDemand::Sequence), false, true);
    let events = Rc::new(RefCell::new(Events::default()));
    let cancel = GenerationCancellationToken::new();
    let mut backend = original.backend(events.clone(), Fail::None, &cancel, None);
    let identity = backend.identity.clone();
    let mut bank = original.bank();
    let admitted = original.owner.prefill_capture(original.bound()).unwrap();
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bank.with_admitted_prefill_observer(&mut backend, &admitted, &Error::backend_retained_source, |o| {
            invoke(
                &mut session,
                &original,
                &events,
                Fail::Panic,
                &identity,
                &cancel,
                o,
            )
        })
    }))
    .err()
    .expect("original panic");
    assert!(Arc::ptr_eq(
        payload.downcast_ref::<Arc<()>>().unwrap(),
        &identity
    ));
    assert_eq!(session.report().unwrap().state_report(), &[4]);
    assert!(session.inspect_runtime_state(|_| Ok(())).is_err());
    assert_eq!(
        bank.take_shared_step().unwrap().unwrap().outcome(),
        CaptureStepOutcome::Aborted
    );
    assert_eq!(events.borrow().prepared, vec![0..2, 2..4]);
    assert_eq!(events.borrow().finishes, [false]);
    INFERENCE_SCOPE_TRACE.with(|s| *s.borrow_mut() = InferenceScopeTrace::default());
}

#[path = "sequence_gateway/score_layout.rs"]
mod score_layout;
