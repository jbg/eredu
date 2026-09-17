//! Actual compiled decoder/stop storage under the same private cursor. The
//! runtime tests separately exercise original admission; these probes witness
//! event loans, native-free readiness and exact owner retirement behavior.
use super::*;
use eredu_core::{GenerationDecoderOutput, GenerationPlainText, GenerationPlainTextEvent};
use eredu_text::stop_storage::{OwnedStopStorage, PreparedStopSource};
type PlainConsumer = RetainedConsumerCursor<SourceFailure, GenerationDecoderError, false, true>;
#[derive(Debug)]
struct PlainProvider {
    decoder: DecoderOwner,
    stops: Option<OwnedStopStorage>,
    inner: Provider,
    behavior: Behavior,
}
fn take(owner: Box<PlainProvider>) -> PlainProvider {
    *owner
}
impl RetainedGenerationStorage for PlainProvider {
    fn consumer_layout(&self) -> Option<&eredu_core::GenerationSequenceConsumerLayout> {
        self.inner.consumer_layout()
    }
    fn decoder_output(&self) -> GenerationDecoderOutput {
        GenerationDecoderOutput::PlainText
    }
    fn max_tokens(&self) -> usize {
        self.inner.max_tokens()
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.inner.eos_token_ids()
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        self.inner.prepare_tokens()?;
        self.decoder
            .storage
            .as_mut()
            .unwrap()
            .prepare_destinations()
            .unwrap();
        self.stops.as_mut().unwrap().prepare_destination().unwrap();
        self.decoder.probe.record("plain-prepared");
        match self.behavior {
            Behavior::Ready => Ok(()),
            Behavior::Fail => Err(BackendFailure::new(
                BackendFailureKind::ResourceExhausted,
                LocalFailure(self.decoder.probe.clone()),
            )),
            Behavior::Panic => panic!("after decoder and stop destination preparation"),
        }
    }
    fn project_plain_text(
        &mut self,
        id: u32,
    ) -> Result<GenerationPlainText<'_>, GenerationDecoderError> {
        self.decoder.probe.record("plain-step");
        let text = self
            .decoder
            .storage
            .as_mut()
            .unwrap()
            .step(id)
            .map_err(|_| GenerationDecoderError::InvalidStorage)?
            .unwrap_or("");
        let out = self
            .stops
            .as_mut()
            .unwrap()
            .step(text)
            .map_err(|_| GenerationDecoderError::InvalidStorage)?;
        Ok(GenerationPlainText {
            visible: out.visible,
            stop_matched: out.matched.is_some(),
        })
    }
    fn finish_plain_text(&mut self) -> Result<&str, GenerationDecoderError> {
        self.decoder.probe.record("plain-finish");
        self.decoder
            .storage
            .as_mut()
            .unwrap()
            .finish()
            .map_err(|_| GenerationDecoderError::IncompleteByteSequence)?;
        self.stops
            .as_mut()
            .unwrap()
            .finish()
            .map_err(|_| GenerationDecoderError::InvalidStorage)
    }
    fn token_slots(&self) -> &[u32] {
        self.inner.token_slots()
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        self.inner.token_slots_mut()
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let Self {
            stops,
            decoder,
            inner,
            ..
        } = take(self);
        drop(stops);
        drop(decoder);
        let Provider { mut payload, .. } = inner;
        Arc::get_mut(&mut payload).unwrap().committed = committed;
        GenerationTokenIds::from_owner(payload)
    }
    fn retire(self: Box<Self>) {
        drop(take(self));
    }
}
fn plain(maximum: usize, stops: &[&str], behavior: Behavior) -> (PlainConsumer, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let source = PreparedDecodeSource::prepare(&tokenizer().snapshot()).unwrap();
    let extent =
        eredu_text::decoder_storage::DecodeStreamLayout::for_source(&source, maximum, false)
            .unwrap()
            .text_capacity();
    let sequence = RetainedGenerationSequence::from_retained_storage(Box::new(PlainProvider {
        decoder: DecoderOwner {
            storage: Some(OwnedDecodeStorage::new(source, maximum, false).unwrap()),
            probe: probe.clone(),
        },
        stops: Some(
            OwnedStopStorage::new(
                PreparedStopSource::prepare(stops.iter().copied()).unwrap(),
                extent,
            )
            .unwrap(),
        ),
        inner: Provider {
            maximum,
            consumer: PlainConsumer::layout(),
            behavior: Behavior::Ready,
            payload: Arc::new(Payload {
                slots: vec![],
                eos: vec![],
                committed: 0,
                custody: Custody(probe.clone()),
            }),
        },
        behavior,
    }))
    .unwrap();
    (PlainConsumer::from_sequence(sequence).unwrap(), probe)
}
fn own(event: GenerationPlainTextEvent<'_>) -> SemanticEvent {
    // Test observer-owned copies only; production borrowed adapter has no such conversion.
    match event {
        GenerationPlainTextEvent::TextDelta(text) => SemanticEvent::TextDelta(text.to_owned().into()),
        GenerationPlainTextEvent::Finished { reason } => SemanticEvent::Finished { reason },
    }
}
#[test]
fn borrowed_plain_events_match_actual_hf_legacy_steps_and_pointer_preserving_result() {
    for stops in [vec![], vec!["ab"], vec!["a b"], vec!["b "]] {
        let ids = [0, 2, 1, 2, 0];
        let (mut cursor, probe) = plain(ids.len(), &stops, Behavior::Ready);
        let mut source = Source::new(probe.clone(), Disposition::Healthy);
        source.tokens = ids.into_iter().collect();
        let mut reference =
            CommittedGenerationCursor::new(&[], NonZeroUsize::new(ids.len()).unwrap());
        let mut reference_source = Source::new(Arc::new(Probe::default()), Disposition::Healthy);
        reference_source.tokens = ids.into_iter().collect();
        let mut pipeline = legacy_pipeline(tokenizer().snapshot(), &stops);
        let (mut actual, mut expected) = (vec![], vec![]);
        let cancellation = GenerationCancellationToken::new();
        loop {
            cursor = cursor
                .advance_plain(&mut source, &cancellation, &mut |event| {
                    actual.push(own(event))
                })
                .unwrap();
            reference
                .step(
                    &mut reference_source,
                    &mut pipeline,
                    &cancellation,
                    &mut |event| expected.push(event),
                )
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(cursor.token_ids(), reference.token_ids());
            assert_eq!(cursor.finish_reason(), reference.finish_reason());
            assert_eq!(source.readiness, reference_source.readiness);
            if cursor.finish_reason().is_some() {
                break;
            }
        }
        let address = cursor.token_ids().as_ptr();
        let (tokens, reason) = cursor.into_tokens().unwrap();
        assert_eq!(tokens.as_ptr(), address);
        assert_eq!(Some(reason), reference.finish_reason());
        assert_eq!(probe.retired.load(SeqCst), 0);
        assert!(probe.calls.lock().unwrap().contains(&"decoder-retired"));
        drop(tokens);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}
#[test]
fn borrowed_plain_stop_wins_callback_cancel_and_nonterminal_cancel_does_not_flush() {
    for matched in [false, true] {
        // The real token "<stop>" yields visible "<" and a stop in one call.
        // Cancelling during that TextDelta must still deliver normal Finished.
        let stops = if matched { vec!["stop>"] } else { vec!["bb"] };
        let first = if matched { 5 } else { 0 };
        let (cursor, probe) = plain(4, &stops, Behavior::Ready);
        let mut source = Source::new(probe.clone(), Disposition::Healthy);
        source.tokens = [first, 1, 0, 0].into_iter().collect();
        let cancel = GenerationCancellationToken::new();
        let mut events = vec![];
        let cursor = cursor
            .advance_plain(&mut source, &cancel, &mut |event| {
                events.push(own(event));
                cancel.cancel();
            })
            .unwrap();
        let reason = if matched {
            FinishReason::StopSequence
        } else {
            FinishReason::Cancelled
        };
        assert_eq!(cursor.finish_reason(), Some(reason));
        assert_eq!(source.next, 1);
        assert_eq!(
            events,
            [
                SemanticEvent::TextDelta(if matched { "<" } else { "a" }.into()),
                SemanticEvent::Finished { reason }
            ]
        );
        let (tokens, _) = cursor.into_tokens().unwrap();
        assert_eq!(&*tokens, [first]);
        drop(tokens);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
    // A real withheld suffix is never exposed by initial/peer cancellation of
    // the next step. Reuse the same provider and cursor instead of a fresh parser.
    let (mut cursor, probe) = plain(4, &["ab"], Behavior::Ready);
    let mut source = Source::new(probe, Disposition::Healthy);
    source.tokens = [0, 1, 0, 0].into_iter().collect();
    let cancel = GenerationCancellationToken::new();
    let mut events = vec![];
    cursor = cursor
        .advance_plain(&mut source, &cancel, &mut |e| events.push(own(e)))
        .unwrap();
    assert!(events.is_empty());
    cancel.cancel();
    cursor = cursor
        .advance_plain(&mut source, &cancel, &mut |e| events.push(own(e)))
        .unwrap();
    assert_eq!(
        events,
        [SemanticEvent::Finished {
            reason: FinishReason::Cancelled
        }]
    );
    assert_eq!(source.next, 1);
    drop(cursor);
}
#[test]
fn plain_readiness_errors_zero_and_peer_cancel_preserve_owning_disposition() {
    for (maximum, initial, disposition) in [
        (0, false, Disposition::Healthy),
        (3, true, Disposition::Healthy),
        (3, false, Disposition::PeerCancel),
    ] {
        let (cursor, probe) = plain(maximum, &["ab"], Behavior::Ready);
        let mut source = Source::new(probe.clone(), disposition);
        let cancel = GenerationCancellationToken::new();
        if initial {
            cancel.cancel();
        }
        let mut events = vec![];
        let cursor = cursor
            .advance_plain(&mut source, &cancel, &mut |e| events.push(own(e)))
            .unwrap();
        assert_eq!(source.next, 0);
        assert_eq!(
            probe.prepared.load(SeqCst),
            usize::from(maximum > 0 && !initial)
        );
        assert_eq!(events.len(), 1);
        drop(cursor);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
    let (cursor, probe) = plain(3, &["ab"], Behavior::Fail);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    let error = cursor
        .advance_plain(
            &mut source,
            &GenerationCancellationToken::new(),
            &mut |_| panic!("no delivery before readiness"),
        )
        .unwrap_err();
    assert_eq!((source.readiness, source.next), (1, 0));
    assert!(error.cause_owns_sequence_for_test());
    let CommittedGenerationError::Preparation(cause) = error.cause() else {
        panic!("original local failure wins")
    };
    assert!(cause.cause().source().unwrap().is::<LocalFailure>());
    assert!(probe.calls.lock().unwrap().contains(&"plain-prepared"));
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(error);
    assert_eq!(probe.errors_retired.load(SeqCst), 1);
    assert_eq!(probe.retired.load(SeqCst), 1);
}
#[test]
fn borrowed_plain_callback_and_partial_preparation_unwind_retire_exact_provider() {
    for prepare in [false, true] {
        let (cursor, probe) = plain(
            3,
            &[],
            if prepare {
                Behavior::Panic
            } else {
                Behavior::Ready
            },
        );
        let mut source = Source::new(probe.clone(), Disposition::Healthy);
        source.tokens = [0, 1, 0].into_iter().collect();
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cursor.advance_plain(
                &mut source,
                &GenerationCancellationToken::new(),
                &mut |_| panic!("borrowed plain callback"),
            );
        }))
        .unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&if prepare {
                "after decoder and stop destination preparation"
            } else {
                "borrowed plain callback"
            })
        );
        assert_eq!(probe.retired.load(SeqCst), 1);
        assert_eq!(source.next, usize::from(!prepare));
    }
}
