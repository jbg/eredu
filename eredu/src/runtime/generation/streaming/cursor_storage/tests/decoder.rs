//! Concrete kernel/provider/cursor composition. Original-account admission is
//! exercised separately by the runtime genuine-core fixture; this owner probe
//! certifies callback/drop ordering, not a funding grant or native completion.
use super::*;
use eredu_core::GenerationDecoderError;
use eredu_text::{
    decoder_storage::{OwnedDecodeStorage, PreparedDecodeSource},
    tokenizer::{Tokenizer, TokenizerSnapshot},
};
type Consumer = RetainedConsumerCursor<SourceFailure, GenerationDecoderError>;

fn tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"▁"},"content":" "}]},"model":{"type":"WordLevel","vocab":{"a":0,"b":1,"▁":2,"<0x61>":3,"<0xFF>":4,"<stop>":5,"[UNK]":6},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap()
}
struct HfDecoder(crate::api::TextDecoder);
impl TokenDecoderBackend for HfDecoder {
    type Error = crate::api::TextDecoderError;
    fn decode_token(&mut self, id: u32, _: bool) -> Result<Vec<u8>, Self::Error> {
        Ok(self.0.step(id)?.unwrap_or_default().into_bytes())
    }
    fn finish(&mut self) -> Result<Vec<u8>, Self::Error> {
        let decoded = self
            .0
            .tokenizer
            .decode(&self.0.ids, self.0.skip_special_tokens)
            .map_err(crate::api::TextDecoderError::Tokenizer)?;
        if decoded.len() > self.0.prefix.len() {
            return Err(crate::api::TextDecoderError::IncompleteByteSequence);
        }
        Ok(vec![])
    }
}
#[derive(Debug)]
struct DecoderOwner {
    storage: Option<OwnedDecodeStorage>,
    probe: Arc<Probe>,
}
impl Drop for DecoderOwner {
    fn drop(&mut self) {
        drop(self.storage.take());
        assert_eq!(self.probe.retired.load(SeqCst), 0);
        self.probe.record("decoder-retired");
    }
}
#[derive(Debug)]
struct DecoderProvider {
    decoder: DecoderOwner,
    inner: Provider,
    failure: Behavior,
}
fn unbox(owner: Box<DecoderProvider>) -> DecoderProvider {
    *owner
}
impl RetainedGenerationStorage for DecoderProvider {
    fn consumer_layout(&self) -> Option<&eredu_core::GenerationSequenceConsumerLayout> {
        self.inner.consumer_layout()
    }
    fn matches_decoder_input(&self, _: Option<&dyn eredu_core::GenerationDecoderInput>) -> bool {
        false
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
            .map_err(|e| BackendFailure::new(BackendFailureKind::ResourceExhausted, e))?;
        self.decoder.probe.record("decoder-prepared");
        match self.failure {
            Behavior::Ready => Ok(()),
            Behavior::Fail => Err(BackendFailure::new(
                BackendFailureKind::ResourceExhausted,
                Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err(),
            )),
            Behavior::Panic => panic!("after actual decoder destinations"),
        }
    }
    fn decode_token(&mut self, id: u32) -> Result<Option<&str>, GenerationDecoderError> {
        self.decoder.probe.record("decoder-step");
        self.decoder
            .storage
            .as_mut()
            .unwrap()
            .step(id)
            .map_err(|error| match error {
                eredu_text::decoder_storage::OwnedDecodeStorageError::Storage(
                    eredu_text::decoder_storage::DecodeStorageError::InvalidPrefix {
                        token_id,
                        expected_bytes,
                        actual_bytes,
                    },
                ) => GenerationDecoderError::InvalidPrefix {
                    token_id,
                    expected_bytes,
                    actual_bytes,
                },
                _ => GenerationDecoderError::InvalidStorage,
            })
    }
    fn finish_decoder(&mut self) -> Result<(), GenerationDecoderError> {
        self.decoder.probe.record("decoder-finish");
        self.decoder
            .storage
            .as_mut()
            .unwrap()
            .finish()
            .map_err(|_| GenerationDecoderError::IncompleteByteSequence)
    }
    fn token_slots(&self) -> &[u32] {
        self.inner.token_slots()
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        self.inner.token_slots_mut()
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let Self { decoder, inner, .. } = unbox(self);
        drop(decoder);
        let Provider { mut payload, .. } = inner;
        Arc::get_mut(&mut payload).unwrap().committed = committed;
        GenerationTokenIds::from_owner(payload)
    }
    fn retire(self: Box<Self>) {
        drop(unbox(self));
    }
}
fn owned(
    snapshot: &TokenizerSnapshot,
    maximum: usize,
    failure: Behavior,
) -> (Consumer, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let sequence = RetainedGenerationSequence::from_retained_storage(Box::new(DecoderProvider {
        decoder: DecoderOwner {
            storage: Some(
                OwnedDecodeStorage::new(
                    PreparedDecodeSource::prepare(snapshot).unwrap(),
                    maximum,
                    false,
                )
                .unwrap(),
            ),
            probe: probe.clone(),
        },
        inner: Provider {
            maximum,
            consumer: Consumer::layout(),
            behavior: Behavior::Ready,
            payload: Arc::new(Payload {
                slots: vec![],
                eos: vec![],
                committed: 0,
                custody: Custody(probe.clone()),
            }),
        },
        failure,
    }))
    .unwrap();
    (Consumer::from_sequence(sequence).unwrap(), probe)
}
fn original_pipeline(stops: &[&str]) -> CommittedTokenPipeline<OriginalProviderDecoder> {
    CommittedTokenPipeline::new(
        RawTokenDecoder::with_structural_tokens(OriginalProviderDecoder, [(5, "<stop>".into())]),
        ToolRuntimeParser::text(stops.iter().copied()),
    )
}
fn legacy_pipeline(
    snapshot: TokenizerSnapshot,
    stops: &[&str],
) -> CommittedTokenPipeline<HfDecoder> {
    let decoder = crate::api::TextDecoder {
        tokenizer: snapshot,
        skip_special_tokens: false,
        ids: vec![],
        prefix: String::new(),
        prefix_index: 0,
    };
    CommittedTokenPipeline::new(
        RawTokenDecoder::with_structural_tokens(HfDecoder(decoder), [(5, "<stop>".into())]),
        ToolRuntimeParser::text(stops.iter().copied()),
    )
}
#[test]
fn source_contained_decoder_and_actual_hf_adapter_use_same_commit_delivery_and_stop_body() {
    for stop in [false, true] {
        let snapshot = tokenizer().snapshot();
        let stops = if stop { vec!["ab"] } else { vec![] };
        let ids = [0, 1, 2, 5, 0];
        let (mut cursor, probe) = owned(&snapshot, ids.len(), Behavior::Ready);
        let mut actual_source = Source::new(probe.clone(), Disposition::Healthy);
        actual_source.tokens = ids.into_iter().collect();
        let legacy_probe = Arc::new(Probe::default());
        let mut reference_source = Source::new(legacy_probe, Disposition::Healthy);
        reference_source.tokens = ids.into_iter().collect();
        let mut reference =
            CommittedGenerationCursor::new(&[], NonZeroUsize::new(ids.len()).unwrap());
        let (mut actual_pipeline, mut reference_pipeline) =
            (original_pipeline(&stops), legacy_pipeline(snapshot, &stops));
        let (mut actual_events, mut reference_events) = (vec![], vec![]);
        let cancellation = GenerationCancellationToken::new();
        loop {
            cursor = cursor
                .advance(
                    &mut actual_source,
                    &mut actual_pipeline,
                    &cancellation,
                    &mut |e| actual_events.push(e),
                )
                .unwrap();
            reference
                .step(
                    &mut reference_source,
                    &mut reference_pipeline,
                    &cancellation,
                    &mut |e| reference_events.push(e),
                )
                .unwrap();
            assert_eq!(cursor.token_ids(), reference.token_ids());
            assert_eq!(cursor.finish_reason(), reference.finish_reason());
            assert_eq!(actual_events, reference_events);
            assert_eq!(actual_source.readiness, reference_source.readiness);
            if cursor.finish_reason().is_some() {
                break;
            }
        }
        let address = cursor.token_ids().as_ptr();
        let (tokens, _) = cursor.into_tokens().unwrap();
        assert_eq!(tokens.as_ptr(), address);
        assert!(probe.calls.lock().unwrap().contains(&"decoder-retired"));
        assert_eq!(probe.retired.load(SeqCst), 0);
        drop(tokens);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}
#[test]
fn original_decoder_zero_initial_peer_and_callback_cancel_share_existing_readiness_counts() {
    for (maximum, initial, peer, callback) in [
        (0, false, false, false),
        (3, true, false, false),
        (3, false, true, false),
        (3, false, false, true),
    ] {
        let (mut cursor, probe) = owned(&tokenizer().snapshot(), maximum, Behavior::Ready);
        let cancellation = GenerationCancellationToken::new();
        if initial {
            cancellation.cancel();
        }
        let mut source = Source::new(
            probe.clone(),
            if peer {
                Disposition::PeerCancel
            } else {
                Disposition::Healthy
            },
        );
        source.tokens = [0, 1, 0].into_iter().collect();
        cursor = cursor
            .advance(
                &mut source,
                &mut original_pipeline(&[]),
                &cancellation,
                &mut |event| {
                    if callback && matches!(event, SemanticEvent::TextDelta(_)) {
                        cancellation.cancel();
                    }
                },
            )
            .unwrap();
        let cancelled = initial || peer || callback;
        assert_eq!(
            cursor.finish_reason(),
            Some(if cancelled {
                FinishReason::Cancelled
            } else {
                FinishReason::MaxTokens
            })
        );
        assert_eq!(source.next, usize::from(callback));
        assert_eq!(source.readiness, if cancelled { 3 } else { 2 });
        let calls = probe.calls.lock().unwrap();
        assert_eq!(calls.contains(&"decoder-prepared"), peer || callback);
        assert_eq!(calls.contains(&"decoder-finish"), !cancelled);
        drop(calls);
        let (tokens, _) = cursor.into_tokens().unwrap();
        assert_eq!(tokens.len(), usize::from(callback));
        drop(tokens);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}
#[test]
fn failure_after_actual_decoder_destinations_stays_inside_first_readiness_owner() {
    let (cursor, probe) = owned(&tokenizer().snapshot(), 3, Behavior::Fail);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    let error = cursor
        .advance(
            &mut source,
            &mut original_pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap_err();
    assert_eq!((source.next, source.readiness), (0, 1));
    assert_eq!(
        probe.calls.lock().unwrap().as_slice(),
        ["prepare", "decoder-prepared", "readiness-error"]
    );
    let CommittedGenerationError::Preparation(preparation) = error.cause() else {
        panic!("lost actual provider")
    };
    assert!(preparation
        .cause()
        .source()
        .unwrap()
        .is::<std::collections::TryReserveError>());
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(error);
    assert_eq!(probe.calls.lock().unwrap().last(), Some(&"decoder-retired"));
    assert_eq!(probe.retired.load(SeqCst), 1);

    let (cursor, probe) = owned(&tokenizer().snapshot(), 3, Behavior::Panic);
    let mut source = Source::new(probe.clone(), Disposition::Healthy);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = cursor.advance(
            &mut source,
            &mut original_pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        );
    }));
    assert!(panic.is_err());
    // Actual destinations existed when preparation unwound. The consumed
    // provider retires before its probe; no readiness callback was entered.
    assert_eq!((source.next, source.readiness), (0, 0));
    assert_eq!(
        probe.calls.lock().unwrap().as_slice(),
        ["prepare", "decoder-prepared", "decoder-retired"]
    );
    assert_eq!(probe.retired.load(SeqCst), 1);
}
#[test]
fn real_invalid_prefix_is_typed_and_failed_cursor_retains_uncommitted_decoder_state() {
    let (mut cursor, probe) = owned(&tokenizer().snapshot(), 4, Behavior::Ready);
    let mut source = Source::new(probe.clone(), Disposition::Healthy);
    source.tokens = [3, 4, 1, 0].into_iter().collect();
    let mut pipeline = original_pipeline(&[]);
    let cancellation = GenerationCancellationToken::new();
    for _ in 0..2 {
        cursor = cursor
            .advance(&mut source, &mut pipeline, &cancellation, &mut |_| {})
            .unwrap();
    }
    assert_eq!(cursor.token_ids(), [3, 4]);
    let error = cursor
        .advance(&mut source, &mut pipeline, &cancellation, &mut |_| {})
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        CommittedGenerationError::Pipeline(CommittedTokenPipelineError::OriginalDecoder(
            GenerationDecoderError::InvalidPrefix { token_id: 1, .. }
        ))
    ));
    assert_eq!((source.next, source.readiness), (3, 6));
    assert_eq!(probe.retired.load(SeqCst), 0);
    let erased = error.into_backend_failure(BackendFailureKind::InvalidInput);
    assert!(erased
        .source()
        .unwrap()
        .source()
        .unwrap()
        .is::<GenerationDecoderError>());
    drop(erased);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

mod plain_text;
