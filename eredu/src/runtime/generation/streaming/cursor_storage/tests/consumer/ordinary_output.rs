use super::*;
use eredu_core::{GenerationOutput, GenerationSequenceConsumerLayout, GenerationTiming};
type Ordinary = RetainedConsumerCursor<SourceFailure, Infallible, true>;

fn ordinary(maximum: usize, behavior: Behavior) -> (Ordinary, Arc<Probe>) {
    let (mut raw, probe) = retained_for(maximum, &[], behavior, Ordinary::layout());
    let result = Ordinary::from_sequence(raw.sequence.take().unwrap()).unwrap();
    (result, probe)
}

#[test]
fn ordinary_output_requires_original_mode_and_preserves_nonterminal_rejection_owner() {
    type Error =
        CommittedGenerationError<SourceFailure, Infallible, RetainedSequencePreparationError>;
    type Failure = RetainedCursorFailure<SourceFailure, Infallible, true>;
    let tuple =
        GenerationSequenceConsumerLayout::for_driver_types::<Ordinary, Error, Failure>().unwrap();
    #[repr(transparent)]
    struct Foreign(Ordinary);
    let foreign = GenerationSequenceConsumerLayout::for_driver_with_ordinary_output::<
        Foreign,
        Error,
        Failure,
    >()
    .unwrap();
    assert_eq!(
        foreign.retention_peak_bytes(),
        Ordinary::layout().unwrap().retention_peak_bytes()
    );
    assert_ne!(foreign, Ordinary::layout().unwrap());
    for association in [None, Some(tuple), Some(foreign)] {
        let (mut raw, probe) = retained_for(2, &[], Behavior::Ready, association);
        let rejected = Ordinary::from_sequence(raw.sequence.take().unwrap()).unwrap_err();
        assert_eq!(rejected.consumer_layout(), association.as_ref());
        assert_eq!(probe.prepared.load(SeqCst), 0);
        drop(raw);
        assert_eq!(probe.retired.load(SeqCst), 0);
        drop(rejected);
        assert_eq!(probe.retired.load(SeqCst), 1);
    }
    let (cursor, probe) = ordinary(2, Behavior::Ready);
    let cursor = cursor.into_output(GenerationTiming::default()).unwrap_err();
    assert!(cursor.token_ids().is_empty());
    assert!(cursor.finish_reason().is_none());
    assert_eq!(probe.prepared.load(SeqCst), 0);
    assert_eq!(probe.retired.load(SeqCst), 0);
    drop(cursor);
    assert_eq!(probe.retired.load(SeqCst), 1);
}

#[test]
fn ordinary_output_loop_and_controlled_steps_preserve_legacy_terminal_values_and_owner() {
    for controlled in [false, true] {
        for (maximum, initial_cancel, peer_cancel, prepares) in [
            (3, false, false, 1),
            (3, true, false, 0),
            (0, false, false, 0),
            (3, false, true, 1),
            (0, false, true, 0),
        ] {
            let (mut cursor, probe) = ordinary(maximum, Behavior::Ready);
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
                }
                if cursor.finish_reason().is_some() {
                    break;
                }
            }
            let cancelled = initial_cancel || peer_cancel;
            let reason = if cancelled {
                FinishReason::Cancelled
            } else {
                FinishReason::MaxTokens
            };
            assert_eq!(source.next, if cancelled { 0 } else { maximum });
            assert_eq!(
                source.readiness,
                if cancelled { 3 } else { 2 * maximum.max(1) }
            );
            assert_eq!(probe.prepared.load(SeqCst), prepares);
            let pointer = cursor.token_ids().as_ptr();
            let timing = GenerationTiming::new(Some(std::time::Duration::from_millis(31)));
            let output = cursor.into_output(timing).unwrap();
            assert_eq!(output.token_ids().as_ptr(), pointer);
            assert_eq!(output.finish_reason(), reason);
            assert_eq!(*output.timing(), timing);
            assert_eq!(output.stats(), &());
            assert!(
                matches!(events.last(), Some(SemanticEvent::Finished { reason: actual }) if *actual == reason)
            );
            assert_eq!(
                probe.prepared.load(SeqCst),
                prepares,
                "freeze must not prepare again"
            );
            if maximum > 0 {
                let mut legacy =
                    CommittedGenerationCursor::new(&[], NonZeroUsize::new(maximum).unwrap());
                let mut legacy_source = Source::new(
                    Arc::new(Probe::default()),
                    if peer_cancel {
                        Disposition::PeerCancel
                    } else {
                        Disposition::Healthy
                    },
                );
                let legacy_cancel = GenerationCancellationToken::new();
                if initial_cancel {
                    legacy_cancel.cancel();
                }
                let mut legacy_pipeline = super::super::pipeline(&[]);
                let mut legacy_events = Vec::new();
                while legacy.finish_reason().is_none() {
                    legacy
                        .step(
                            &mut legacy_source,
                            &mut legacy_pipeline,
                            &legacy_cancel,
                            &mut |e| legacy_events.push(e),
                        )
                        .unwrap();
                }
                let legacy_output = GenerationOutput::new(
                    legacy.token_ids().to_vec(),
                    legacy.finish_reason().unwrap(),
                    timing,
                    (),
                );
                assert_eq!(output.token_ids(), legacy_output.token_ids());
                assert_eq!(output.finish_reason(), legacy_output.finish_reason());
                assert_eq!(events, legacy_events);
                assert_eq!(source.readiness, legacy_source.readiness);
            }
            let mut iter = output.token_ids.into_iter();
            assert_eq!(
                iter.next(),
                (!cancelled && maximum > 0).then_some(b'a' as u32)
            );
            drop((source, pipeline));
            assert_eq!(probe.retired.load(SeqCst), 0);
            drop(iter);
            assert_eq!(probe.retired.load(SeqCst), 1);
        }
    }
}

#[test]
fn ordinary_output_specialization_keeps_readiness_failure_and_preparation_owner_order() {
    for behavior in [Behavior::Ready, Behavior::Fail] {
        let (cursor, probe) = ordinary(2, behavior);
        let mut source = Source::new(probe.clone(), Disposition::Fail);
        let error = cursor
            .advance(
                &mut source,
                &mut pipeline(&[]),
                &GenerationCancellationToken::new(),
                &mut |_| panic!("no events before readiness"),
            )
            .unwrap_err();
        assert_eq!((source.next, source.readiness), (0, 1));
        match behavior {
            Behavior::Ready => assert!(matches!(
                error.cause(),
                CommittedGenerationError::Source(SourceFailure::Readiness)
            )),
            Behavior::Fail => {
                assert!(error.cause_owns_sequence_for_test());
                assert!(matches!(
                    error.cause(),
                    CommittedGenerationError::Preparation(_)
                ));
            }
            Behavior::Panic => unreachable!(),
        }
        drop(source);
        assert_eq!(probe.retired.load(SeqCst), 0);
        let erased = error.into_backend_failure(BackendFailureKind::InvalidSession);
        assert!(erased
            .source()
            .unwrap()
            .is::<RetainedCursorFailure<SourceFailure, Infallible, true>>());
        drop(erased);
        assert_eq!(probe.retired.load(SeqCst), 1);
        assert_eq!(
            probe.errors_retired.load(SeqCst),
            usize::from(matches!(behavior, Behavior::Fail))
        );
    }
}
