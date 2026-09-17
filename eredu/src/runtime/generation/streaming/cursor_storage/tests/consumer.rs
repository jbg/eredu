use super::*;
mod ordinary_output;
type Consumer = RetainedConsumerCursor<SourceFailure, Infallible>;
fn consumer(maximum: usize, behavior: Behavior) -> (Consumer, Arc<Probe>) {
    let layout = Consumer::layout().unwrap();
    let (mut cursor, probe) = retained_for(maximum, &[], behavior, Some(layout));
    (
        Consumer::from_sequence(cursor.sequence.take().unwrap()).unwrap(),
        probe,
    )
}

#[test]
fn original_consumer_rejects_unassociated_and_foreign_same_size_before_readiness() {
    let expected = Consumer::layout().unwrap();
    #[repr(transparent)]
    struct Foreign(Consumer);
    let foreign = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
        Foreign,
        CommittedGenerationError<SourceFailure, Infallible, RetainedSequencePreparationError>,
        RetainedCursorFailure<SourceFailure, Infallible>,
    >()
    .unwrap();
    assert_eq!(
        expected.retention_peak_bytes(),
        foreign.retention_peak_bytes()
    );
    assert_ne!(expected, foreign);
    for layout in [None, Some(foreign)] {
        let (mut cursor, probe) = retained_for(3, &[], Behavior::Ready, layout);
        let error = Consumer::from_sequence(cursor.sequence.take().unwrap()).unwrap_err();
        assert_eq!(error.consumer_layout(), layout.as_ref());
        assert!(error.tokens().is_empty());
        drop(cursor);
        assert_eq!(probe.retired.load(SeqCst), 0);
        assert_eq!(probe.prepared.load(SeqCst), 0);
        assert!(probe.calls.lock().unwrap().is_empty());
        drop(error);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}

#[test]
fn original_consumer_consumes_source_failure_and_preserves_concrete_error_through_erasure() {
    let (cursor, probe) = consumer(3, Behavior::Ready);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    let error = cursor
        .advance(
            &mut source,
            &mut pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        CommittedGenerationError::Source(SourceFailure::Readiness)
    ));
    assert_eq!((source.next, source.readiness), (0, 1));
    drop(source);
    assert_eq!(probe.retired.load(SeqCst), 0);
    let erased = error.into_backend_failure(BackendFailureKind::InvalidSession);
    let concrete = erased
        .source()
        .unwrap()
        .downcast_ref::<RetainedCursorFailure<SourceFailure, Infallible>>()
        .unwrap();
    assert_eq!(
        concrete.source().unwrap().downcast_ref::<SourceFailure>(),
        Some(&SourceFailure::Readiness)
    );
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(erased);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

struct FailDecoder(Arc<Probe>);
impl TokenDecoderBackend for FailDecoder {
    type Error = LocalFailure;
    fn decode_token(&mut self, _: u32, _: bool) -> Result<Vec<u8>, Self::Error> {
        Err(LocalFailure(self.0.clone()))
    }
}
#[test]
fn original_consumer_pipeline_cause_retires_before_its_consumed_cursor() {
    type Failing = RetainedConsumerCursor<SourceFailure, LocalFailure>;
    let (mut raw, probe) = retained_for(3, &[], Behavior::Ready, Failing::layout());
    let cursor = Failing::from_sequence(raw.sequence.take().unwrap()).unwrap();
    let mut source = Source::new(probe.clone(), Disposition::Healthy);
    let mut pipeline = CommittedTokenPipeline::new(
        RawTokenDecoder::new(FailDecoder(probe.clone()), []),
        ToolRuntimeParser::text([] as [&str; 0]),
    );
    let error = cursor
        .advance(
            &mut source,
            &mut pipeline,
            &GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        CommittedGenerationError::Pipeline(CommittedTokenPipelineError::Decoder(_))
    ));
    assert!(Arc::ptr_eq(
        &error
            .source()
            .unwrap()
            .downcast_ref::<LocalFailure>()
            .unwrap()
            .0,
        &probe
    ));
    assert_eq!((source.next, source.readiness), (1, 2));
    drop((raw, source, pipeline));
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(error.into_backend_failure(BackendFailureKind::InvalidInput));
    assert_eq!(probe.errors_retired.load(SeqCst), 1);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

#[test]
fn original_consumer_preparation_cause_keeps_original_owner_inside_existing_first_readiness() {
    let (cursor, probe) = consumer(3, Behavior::Fail);
    let mut source = Source::new(probe.clone(), Disposition::Fail);
    let error = cursor
        .advance(
            &mut source,
            &mut pipeline(&[]),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap_err();
    assert!(error.cause_owns_sequence_for_test());
    let CommittedGenerationError::Preparation(preparation) = error.cause() else {
        panic!("lost owning preparation cause")
    };
    assert!(Arc::ptr_eq(
        &preparation
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<LocalFailure>()
            .unwrap()
            .0,
        &probe
    ));
    assert_eq!((source.next, source.readiness), (0, 1));
    assert_eq!(
        probe.calls.lock().unwrap().as_slice(),
        ["prepare", "readiness-error"]
    );
    drop(source);
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(error);
    assert_eq!(probe.errors_retired.load(SeqCst), 1);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

#[test]
fn original_consumer_ordinary_and_controlled_advance_freeze_without_copy_and_keep_cancel_counts() {
    for controlled in [false, true] {
        for (maximum, initial_cancel, peer_cancel, prepares) in [
            (3, false, false, 1),
            (3, true, false, 0),
            (0, false, false, 0),
            (3, false, true, 1),
            (0, false, true, 0),
        ] {
            let (mut cursor, probe) = consumer(maximum, Behavior::Ready);
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
            let mut pipeline = pipeline(&[]);
            let mut events = Vec::new();
            loop {
                let before = source.next;
                cursor = cursor
                    .advance(&mut source, &mut pipeline, &cancellation, &mut |e| {
                        events.push(e)
                    })
                    .unwrap();
                if controlled {
                    assert!(source.next <= before + 1);
                    assert_eq!(cursor.token_ids().len(), source.next);
                }
                if cursor.finish_reason().is_some() {
                    break;
                }
            }
            let cancelled = initial_cancel || peer_cancel;
            let expected = if cancelled {
                FinishReason::Cancelled
            } else {
                FinishReason::MaxTokens
            };
            assert_eq!(cursor.finish_reason(), Some(expected));
            assert_eq!(source.next, if cancelled { 0 } else { maximum });
            assert_eq!(
                source.readiness,
                if cancelled { 3 } else { 2 * maximum.max(1) }
            );
            assert_eq!(probe.prepared.load(SeqCst), prepares);
            let pointer = cursor.token_ids().as_ptr();
            let (tokens, reason) = cursor.into_tokens().unwrap();
            assert_eq!(reason, expected);
            assert_eq!(tokens.as_ptr(), pointer);
            assert!(
                matches!(events.last(),Some(SemanticEvent::Finished {reason}) if *reason == expected)
            );
            let alias = tokens.clone();
            let mut iter = tokens.into_iter();
            assert_eq!(
                iter.next(),
                (!cancelled && maximum > 0).then_some(b'a' as u32)
            );
            drop((source, pipeline, alias));
            assert_eq!(probe.retired.load(SeqCst), 0);
            drop(iter);
            assert_eq!(probe.retired.load(SeqCst), 1);
        }
    }
}

#[test]
fn original_consumer_delivery_lifecycle_and_missing_terminal_failures_keep_cursor_owner() {
    for mode in 0..3 {
        let (mut cursor, probe) = consumer(1, Behavior::Ready);
        let mut source = Source::new(probe.clone(), Disposition::Healthy);
        let mut pipeline = pipeline(&[]);
        let cancellation = GenerationCancellationToken::new();
        match mode {
            0 => source.delivery_failure = true,
            1 => source.tokens.clear(),
            _ => {
                cursor = cursor
                    .advance(&mut source, &mut pipeline, &cancellation, &mut |_| {})
                    .unwrap();
            }
        }
        let error = cursor
            .advance(&mut source, &mut pipeline, &cancellation, &mut |_| {})
            .unwrap_err();
        assert!(!error.cause_owns_sequence_for_test());
        match (mode, error.cause()) {
            (
                0,
                CommittedGenerationError::Delivery(eredu_core::capture::CaptureError::Overflow),
            ) => {
                assert_eq!(source.next, 0);
                assert_eq!(probe.prepared.load(SeqCst), 0);
            }
            (1, CommittedGenerationError::MissingTerminalToken) => assert_eq!(source.next, 1),
            (
                2,
                CommittedGenerationError::Lifecycle(eredu_core::GenerationError::AlreadyFinished),
            ) => assert_eq!(source.next, 1),
            _ => panic!("wrong original step cause"),
        }
        drop((source, pipeline));
        assert_eq!(probe.retired.load(SeqCst), 0);
        drop(error);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
}
