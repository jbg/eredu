use super::*;
use eredu_core::{
    BackendFailure, BackendFailureKind, GenerationTokenIdStorage, RetainedGenerationStorage,
};
use std::{
    collections::VecDeque,
    convert::Infallible,
    error::Error as _,
    sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        Arc, Mutex,
    },
};

#[derive(Debug, Default)]
struct Probe {
    calls: Mutex<Vec<&'static str>>,
    prepared: AtomicUsize,
    retired: AtomicUsize,
    errors_retired: AtomicUsize,
    slot_address: AtomicUsize,
}
impl Probe {
    fn record(&self, call: &'static str) {
        self.calls.lock().unwrap().push(call);
    }
}
#[derive(Debug)]
struct Custody(Arc<Probe>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.retired.fetch_add(1, SeqCst);
    }
}
#[derive(Debug)]
struct Payload {
    slots: Vec<u32>,
    eos: Vec<u32>,
    committed: usize,
    custody: Custody,
}
impl GenerationTokenIdStorage for Payload {
    fn token_ids(&self) -> &[u32] {
        &self.slots[..self.committed]
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug, Clone, Copy)]
enum Behavior {
    Ready,
    Fail,
    Panic,
}
#[derive(Debug)]
struct Provider {
    maximum: usize,
    consumer: Option<eredu_core::GenerationSequenceConsumerLayout>,
    behavior: Behavior,
    payload: Arc<Payload>,
}
#[derive(Debug)]
struct LocalFailure(Arc<Probe>);
impl fmt::Display for LocalFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("actual retained preparation sentinel")
    }
}
impl std::error::Error for LocalFailure {}
impl Drop for LocalFailure {
    fn drop(&mut self) {
        assert_eq!(self.0.retired.load(SeqCst), 0);
        self.0.errors_retired.fetch_add(1, SeqCst);
    }
}
fn unbox_provider(provider: Box<Provider>) -> Provider {
    *provider
}
impl RetainedGenerationStorage for Provider {
    fn consumer_layout(&self) -> Option<&eredu_core::GenerationSequenceConsumerLayout> {
        self.consumer.as_ref()
    }
    fn max_tokens(&self) -> usize {
        self.maximum
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.payload.eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        let payload = Arc::get_mut(&mut self.payload).expect("sole mutable token owner");
        payload.custody.0.record("prepare");
        payload.custody.0.prepared.fetch_add(1, SeqCst);
        let extent = if matches!(self.behavior, Behavior::Ready) {
            self.maximum
        } else {
            self.maximum.min(1)
        };
        payload.slots.resize(extent, 0);
        payload
            .custody
            .0
            .slot_address
            .store(payload.slots.as_ptr() as usize, SeqCst);
        match self.behavior {
            Behavior::Ready => Ok(()),
            Behavior::Fail => Err(BackendFailure::new(
                BackendFailureKind::ResourceExhausted,
                LocalFailure(payload.custody.0.clone()),
            )),
            Behavior::Panic => std::panic::panic_any("exact cursor preparation panic"),
        }
    }
    fn token_slots(&self) -> &[u32] {
        &self.payload.slots
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut Arc::get_mut(&mut self.payload)
            .expect("sole mutable token owner")
            .slots
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let Self { mut payload, .. } = unbox_provider(self);
        Arc::get_mut(&mut payload)
            .expect("sole owner before freeze")
            .committed = committed;
        GenerationTokenIds::from_owner(payload)
    }
    fn retire(self: Box<Self>) {
        drop(unbox_provider(self));
    }
}
fn retained(
    maximum: usize,
    eos: &[u32],
    behavior: Behavior,
) -> (
    CommittedGenerationCursor<RetainedGenerationSequenceStorage>,
    Arc<Probe>,
) {
    retained_for(maximum, eos, behavior, None)
}
fn retained_for(
    maximum: usize,
    eos: &[u32],
    behavior: Behavior,
    consumer: Option<eredu_core::GenerationSequenceConsumerLayout>,
) -> (
    CommittedGenerationCursor<RetainedGenerationSequenceStorage>,
    Arc<Probe>,
) {
    let probe = Arc::new(Probe::default());
    let mut eos = eos.to_vec();
    eos.sort_unstable();
    eos.dedup();
    let sequence = RetainedGenerationSequence::from_retained_storage(Box::new(Provider {
        maximum,
        consumer,
        behavior,
        payload: Arc::new(Payload {
            slots: Vec::new(),
            eos,
            committed: 0,
            custody: Custody(probe.clone()),
        }),
    }))
    .unwrap();
    (
        CommittedGenerationCursor::from_retained_sequence(sequence),
        probe,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceFailure {
    Readiness,
}
impl fmt::Display for SourceFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("source readiness sentinel")
    }
}
impl std::error::Error for SourceFailure {}
#[derive(Clone, Copy)]
enum Disposition {
    Healthy,
    PeerCancel,
    Fail,
}
struct Source {
    tokens: VecDeque<u32>,
    probe: Arc<Probe>,
    disposition: Disposition,
    readiness: usize,
    next: usize,
    grammar_after: Option<usize>,
    delivery_failure: bool,
    admitted: bool,
}
impl Source {
    fn new(probe: Arc<Probe>, disposition: Disposition) -> Self {
        Self {
            tokens: b"abcd".iter().copied().map(u32::from).collect(),
            probe,
            disposition,
            readiness: 0,
            next: 0,
            grammar_after: None,
            delivery_failure: false,
            admitted: false,
        }
    }
}
impl CommittedTokenSource for Source {
    type Error = SourceFailure;
    fn next_token(&mut self, _: &GenerationCancellationToken) -> Result<Option<u32>, Self::Error> {
        assert!(
            self.admitted,
            "no token request before successful first readiness"
        );
        self.probe.record("next");
        self.next += 1;
        Ok(self.tokens.pop_front())
    }
    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(self.grammar_after.is_some_and(|n| self.next >= n))
    }
    fn finish_step<T, E>(
        &mut self,
        local: Result<T, E>,
        cancelled: bool,
        map_source: impl FnOnce(Self::Error) -> E,
    ) -> Result<Option<T>, E> {
        self.probe.record(if local.is_err() {
            "readiness-error"
        } else {
            "readiness"
        });
        self.readiness += 1;
        let value = local?; // Original local cause precedes peer disposition.
        if self.readiness == 1 {
            match self.disposition {
                Disposition::Fail => return Err(map_source(SourceFailure::Readiness)),
                Disposition::PeerCancel => return Ok(None),
                Disposition::Healthy => {}
            }
        }
        if cancelled {
            Ok(None)
        } else {
            self.admitted = true;
            Ok(Some(value))
        }
    }
    fn delivery_failure(&self) -> Option<eredu_core::capture::CaptureError> {
        self.delivery_failure
            .then_some(eredu_core::capture::CaptureError::Overflow)
    }
}
#[derive(Clone)]
struct Decoder;
impl TokenDecoderBackend for Decoder {
    type Error = Infallible;
    fn decode_token(&mut self, token: u32, _: bool) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![u8::try_from(token).unwrap()])
    }
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        Some(std::mem::size_of::<Self>() as u64)
    }
}
fn pipeline(stops: &[&str]) -> CommittedTokenPipeline<Decoder> {
    CommittedTokenPipeline::new(
        RawTokenDecoder::new(Decoder, []),
        ToolRuntimeParser::text(stops.iter().copied()),
    )
}

#[test]
fn retained_cursor_preparation_failure_reaches_first_readiness_with_exact_owner() {
    let (mut cursor, probe) = retained(3, &[], Behavior::Fail);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    let mut pipeline = pipeline(&[]);
    let mut events = Vec::new();
    let error = cursor
        .step(
            &mut source,
            &mut pipeline,
            &GenerationCancellationToken::new(),
            &mut |e| events.push(e),
        )
        .unwrap_err();
    let CommittedGenerationError::Preparation(error) = error else {
        panic!("lost actual local preparation error")
    };
    let actual = error
        .cause()
        .source()
        .unwrap()
        .downcast_ref::<LocalFailure>()
        .unwrap();
    assert!(Arc::ptr_eq(&actual.0, &probe));
    assert_eq!(error.cause().kind(), BackendFailureKind::ResourceExhausted);
    assert_eq!(error.sequence().max_tokens(), 3);
    assert!(error.sequence().tokens().is_empty());
    assert_eq!(
        probe.calls.lock().unwrap().as_slice(),
        ["prepare", "readiness-error"]
    );
    assert_eq!((source.next, source.readiness), (0, 1));
    assert!(events.is_empty());
    assert!(matches!(
        cursor.step(
            &mut source,
            &mut pipeline,
            &GenerationCancellationToken::new(),
            &mut |_| {}
        ),
        Err(CommittedGenerationError::Lifecycle(
            eredu_core::GenerationError::FailedGeneration
        ))
    ));
    drop((cursor, source, pipeline));
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(error);
    assert_eq!(probe.errors_retired.load(SeqCst), 1);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

#[test]
fn retained_cursor_source_readiness_failure_precedes_any_token_and_keeps_storage() {
    let (mut cursor, probe) = retained(3, &[], Behavior::Ready);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    let error = cursor
        .step(
            &mut source,
            &mut pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| panic!("no event before readiness"),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CommittedGenerationError::Source(SourceFailure::Readiness)
    ));
    assert_eq!(
        probe.calls.lock().unwrap().as_slice(),
        ["prepare", "readiness"]
    );
    assert_eq!((source.next, source.readiness), (0, 1));
    assert_eq!(probe.retired.load(SeqCst), 0);
    let cursor = cursor
        .into_retained_tokens()
        .expect_err("failed cursor must not publish a token result");
    drop(cursor);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

#[test]
fn retained_cursor_initial_peer_cancellation_and_zero_output_preserve_disposition() {
    for (maximum, initial_cancel, peer_cancel, expected_prepares) in [
        (3, true, false, 0),
        (0, false, false, 0),
        (0, true, false, 0),
        (3, false, true, 1),
        (0, false, true, 0),
    ] {
        let (mut cursor, probe) = retained(maximum, &[], Behavior::Ready);
        let cancellation = GenerationCancellationToken::new();
        if initial_cancel {
            cancellation.cancel();
        }
        let mut source = Source::new(
            probe.clone(),
            if peer_cancel {
                Disposition::PeerCancel
            } else {
                Disposition::Healthy
            },
        );
        let mut events = Vec::new();
        cursor
            .step(&mut source, &mut pipeline(&[]), &cancellation, &mut |e| {
                events.push(e)
            })
            .unwrap();
        let cancelled = initial_cancel || peer_cancel;
        let reason = if cancelled {
            FinishReason::Cancelled
        } else {
            FinishReason::MaxTokens
        };
        assert_eq!(source.next, 0);
        assert_eq!(source.readiness, if cancelled { 3 } else { 2 });
        assert_eq!(probe.prepared.load(SeqCst), expected_prepares);
        assert_eq!(events, vec![SemanticEvent::Finished { reason }]);
        let (tokens, actual) = cursor.into_retained_tokens().unwrap();
        assert_eq!(actual, reason);
        assert!(tokens.is_empty());
        assert_eq!(probe.retired.load(SeqCst), 0);
        drop(tokens);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}

fn run<S: CursorStorage>(
    cursor: &mut CommittedGenerationCursor<S>,
    source: &mut Source,
    stops: &[&str],
    one_at_a_time: bool,
) -> Vec<SemanticEvent> {
    let mut pipeline = pipeline(stops);
    let cancellation = GenerationCancellationToken::new();
    let mut events = Vec::new();
    if one_at_a_time {
        for _ in 0..4 {
            if cursor.finish_reason().is_some() {
                break;
            }
            let before = cursor.token_ids().len();
            cursor
                .step(source, &mut pipeline, &cancellation, &mut |e| {
                    events.push(e)
                })
                .unwrap();
            assert_eq!(cursor.token_ids().len(), before + 1);
            assert_eq!(source.next, before + 1);
            // Merely retaining/reading a controlled cursor cannot advance it.
            assert_eq!(cursor.token_ids().len(), before + 1);
        }
    } else {
        while cursor.finish_reason().is_none() {
            cursor
                .step(source, &mut pipeline, &cancellation, &mut |e| {
                    events.push(e)
                })
                .unwrap();
        }
    }
    assert!(cursor.finish_reason().is_some());
    events
}

#[test]
fn retained_cursor_loop_and_controlled_steps_match_legacy_without_token_copy() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<CommittedGenerationCursor>();
    send_sync::<CommittedGenerationCursor<RetainedGenerationSequenceStorage>>();
    send_sync::<
        CommittedGenerationError<SourceFailure, Infallible, RetainedSequencePreparationError>,
    >();
    for (maximum, eos, grammar, stops, reason) in [
        (4, vec![], None, vec![], FinishReason::MaxTokens),
        (4, vec![b'b' as u32], None, vec![], FinishReason::Eos),
        (
            4,
            vec![b'b' as u32],
            Some(2),
            vec![],
            FinishReason::GrammarComplete,
        ),
        (
            3,
            vec![b'c' as u32],
            Some(3),
            vec!["bc"],
            FinishReason::StopSequence,
        ),
    ] {
        for one_at_a_time in [false, true] {
            let (mut cursor, probe) = retained(maximum, &eos, Behavior::Ready);
            let mut source = Source::new(probe.clone(), Disposition::Healthy);
            source.grammar_after = grammar;
            let events = run(&mut cursor, &mut source, &stops, one_at_a_time);
            let mut legacy =
                CommittedGenerationCursor::new(&eos, NonZeroUsize::new(maximum).unwrap());
            let mut legacy_source = Source::new(Arc::new(Probe::default()), Disposition::Healthy);
            legacy_source.grammar_after = grammar;
            assert_eq!(
                events,
                run(&mut legacy, &mut legacy_source, &stops, one_at_a_time)
            );
            assert_eq!(cursor.token_ids(), legacy.token_ids());
            assert_eq!(cursor.finish_reason(), Some(reason));
            assert_eq!(source.readiness, legacy_source.readiness);
            assert_eq!(source.readiness, source.next * 2);
            assert_eq!(probe.prepared.load(SeqCst), 1);
            let before = cursor.token_ids().as_ptr();
            assert_eq!(before as usize, probe.slot_address.load(SeqCst));
            let (tokens, actual) = cursor.into_retained_tokens().unwrap();
            assert_eq!(actual, reason);
            assert_eq!(tokens.as_ptr(), before);
            let alias = tokens.clone();
            let mut iter = tokens.into_iter();
            assert_eq!(iter.next(), Some(b'a' as u32));
            drop(source);
            assert_eq!(probe.retired.load(SeqCst), 0);
            drop(alias);
            assert_eq!(probe.retired.load(SeqCst), 0);
            drop(iter);
            assert_eq!(probe.retired.load(SeqCst), 1);
            let copy = legacy.clone();
            assert_eq!(copy.token_ids(), legacy.token_ids());
            assert_eq!(
                copy.snapshot_storage_bytes(),
                legacy.snapshot_storage_bytes()
            );
            assert!(copy.snapshot_storage_bytes().is_some());
        }
    }
}

#[test]
fn retained_cursor_callback_cancellation_keeps_legacy_committed_prefix_and_stop_priority() {
    for stop in [false, true] {
        let cancellation = GenerationCancellationToken::new();
        let (mut cursor, probe) = retained(1, &[b'a' as u32], Behavior::Ready);
        let mut source = Source::new(probe, Disposition::Healthy);
        let stops = if stop { vec!["a"] } else { vec![] };
        let mut events = Vec::new();
        cursor
            .step(
                &mut source,
                &mut pipeline(&stops),
                &cancellation,
                &mut |e| {
                    cancellation.cancel();
                    events.push(e);
                },
            )
            .unwrap();
        let reason = if stop {
            FinishReason::StopSequence
        } else {
            FinishReason::Cancelled
        };
        assert_eq!(cursor.finish_reason(), Some(reason));
        assert_eq!(cursor.token_ids(), [b'a' as u32]);
        let legacy_cancel = GenerationCancellationToken::new();
        let mut legacy =
            CommittedGenerationCursor::new(&[b'a' as u32], NonZeroUsize::new(1).unwrap());
        let mut legacy_source = Source::new(Arc::new(Probe::default()), Disposition::Healthy);
        let mut legacy_events = Vec::new();
        legacy
            .step(
                &mut legacy_source,
                &mut pipeline(&stops),
                &legacy_cancel,
                &mut |e| {
                    legacy_cancel.cancel();
                    legacy_events.push(e);
                },
            )
            .unwrap();
        assert_eq!(events, legacy_events);
        assert_eq!(cursor.finish_reason(), legacy.finish_reason());
        assert_eq!(source.readiness, legacy_source.readiness);
    }
}

#[test]
fn retained_cursor_existing_delivery_failure_precedes_dormant_materialization() {
    let (mut cursor, probe) = retained(3, &[], Behavior::Fail);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    source.delivery_failure = true;
    let error = cursor
        .step(
            &mut source,
            &mut pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CommittedGenerationError::Delivery(eredu_core::capture::CaptureError::Overflow)
    ));
    assert_eq!(probe.prepared.load(SeqCst), 0);
    assert_eq!(source.next, 0);
    assert_eq!(probe.calls.lock().unwrap().as_slice(), ["readiness-error"]);
    drop(cursor);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

#[test]
fn retained_cursor_preparation_unwind_retires_prefix_and_prevents_retry() {
    let (mut cursor, probe) = retained(3, &[], Behavior::Panic);
    let mut source = Source::new(probe.clone(), Disposition::Healthy);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = cursor.step(
            &mut source,
            &mut pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        );
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&'static str>(),
        Some(&"exact cursor preparation panic")
    );
    assert_eq!(probe.retired.load(SeqCst), 1);
    assert_eq!((source.readiness, source.next), (0, 0));
    assert!(matches!(
        cursor.step(
            &mut source,
            &mut pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {}
        ),
        Err(CommittedGenerationError::Lifecycle(
            eredu_core::GenerationError::FailedGeneration
        ))
    ));
    assert_eq!(probe.prepared.load(SeqCst), 1);
}

#[test]
fn funded_semantic_state_uses_the_committed_cursor_without_speculative_execution() {
    use eredu_runtime::working_memory::{
        InferenceExecutionIdentity, PreparedSemanticSource, WorkingMemoryPool,
    };
    use eredu_text::{stop_storage::StopCompilePlan, tokenizer_storage::TokenizerPlan};
    type Consumer = RetainedConsumerCursor<SourceFailure, eredu_core::SpeculativeOutputError, true>;
    let json = br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}}"#;

    for cancel_after_first in [false, true] {
        let pool = WorkingMemoryPool::new(64 * 1024 * 1024, 0).unwrap();
        let tokenizer = pool.compile_tokenizer(TokenizerPlan::prepare_json(json).unwrap()).unwrap();
        let stops = pool.compile_stop_source(StopCompilePlan::prepare_refs(&[]).unwrap()).unwrap();
        let preparation = PreparedSemanticSource::new(
            &tokenizer, &InferenceExecutionIdentity::default(), 64 * 1024 * 1024,
        ).unwrap();
        let mut semantic = preparation.prepare(&stops, 2, NonZeroUsize::new(1).unwrap(), true).unwrap();
        let (mut cursor, probe) = retained_for(2, &[], Behavior::Ready, Consumer::layout());
        let mut cursor = Consumer::from_sequence(cursor.sequence.take().unwrap()).unwrap();
        let mut source = Source::new(probe.clone(), Disposition::Healthy);
        source.tokens = [0, 1].into();
        let cancellation = GenerationCancellationToken::new();
        let mut events = Vec::new();
        while cursor.finish_reason().is_none() {
            cursor = cursor.advance_semantic(&mut source, &mut semantic, &cancellation, &mut |event| {
                events.push(event);
                if cancel_after_first && events.len() == 1 { cancellation.cancel(); }
            }).unwrap();
        }
        let output = cursor.into_output(Default::default()).unwrap();
        assert_eq!(output.token_ids.as_ref(), if cancel_after_first { &[0][..] } else { &[0, 1][..] });
        let reason = if cancel_after_first { FinishReason::Cancelled } else { FinishReason::MaxTokens };
        assert_eq!(output.finish_reason, reason);
        let mut expected = vec![SemanticEvent::TextDelta("a".into())];
        if !cancel_after_first { expected.push(SemanticEvent::TextDelta("b".into())); }
        expected.push(SemanticEvent::Finished { reason });
        assert_eq!(events, expected);
        semantic.publish_events(&mut |_| panic!("committed event delivered twice"));
        drop((semantic, preparation, tokenizer, stops));
        assert!(pool.used_bytes().unwrap() > 0, "escaped text retains its actual payer");
        drop(events);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(source.next, if cancel_after_first { 1 } else { 2 });
        drop(output);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}

mod consumer;

mod decoder;
