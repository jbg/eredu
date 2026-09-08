use super::*;
use eredu::api::GenerationOutput;
use eredu_core::GenerationCancellationToken;
use std::time::{Duration, Instant};

// Both public output aliases support the same consumer without mode-specific
// forwarding methods or a second terminal-output representation.
fn terminal<S>(output: GenerationOutput<S>) -> (Vec<u32>, FinishReason, Option<Duration>) {
    let ttft = output.timing().time_to_first_token();
    (output.token_ids, output.finish_reason, ttft)
}

fn chat(model: &mut LoadedModel<MockBackend>) -> PreparedChat {
    model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap()
}

fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn prepared_ttft_counts_eos_and_stop_tokens_without_visible_output() {
    for speculative in [false, true] {
        for eos in [false, true] {
            // The speculative mock predicts 7; ordinary prefill predicts prompt length.
            // With first=5, the Unicode fixture assigns EOS to token 7.
            let mut model = unicode_model(eos.then_some(5));
            let chat = chat(&mut model);
            let stops = if eos {
                vec![]
            } else {
                vec!["ordinary_6".into()]
            };
            let mut events = Vec::new();
            let started = Instant::now();
            let mut first_delivery = None;
            let on_event = |event| {
                first_delivery.get_or_insert_with(|| started.elapsed());
                events.push(event);
            };
            let input = PreparedChatInput::prepared_backend_input(&chat, vec![0; 7]);
            let (ids, reason, ttft) = if speculative {
                let output = model
                    .generate_prepared_chat_speculative(PreparedChatSpeculativeGenerationRequest {
                        input,
                        drafting: SpeculativeDraft::Embedded,
                        settings: settings(),
                        options: Default::default(),
                        caller_stop_sequences: &stops,
                        cancellation: Default::default(),
                        on_event,
                    })
                    .unwrap();
                terminal(output)
            } else {
                let output = model
                    .generate_prepared_chat(PreparedChatGenerationRequest {
                        input,
                        settings: settings(),
                        caller_stop_sequences: &stops,
                        cancellation: Default::default(),
                        on_event,
                    })
                    .unwrap();
                terminal(output)
            };
            assert_eq!(ids, [7]);
            assert!(matches!(
                reason,
                FinishReason::Eos | FinishReason::StopSequence
            ));
            assert_eq!(events, [SemanticEvent::Finished { reason }]);
            assert!(
                ttft.expect("invisible committed token still has TTFT") <= first_delivery.unwrap()
            );
        }
    }
}

#[test]
fn prepared_ttft_distinguishes_cancellation_before_and_after_commitment() {
    for speculative in [false, true] {
        for pre_cancel in [false, true] {
            let mut model = unicode_model(None);
            let chat = chat(&mut model);
            let cancellation = GenerationCancellationToken::new();
            if pre_cancel {
                cancellation.cancel();
            }
            let cancel_on_event = cancellation.clone();
            let on_event = move |_| cancel_on_event.cancel();
            let input = PreparedChatInput::prepared_backend_input(&chat, vec![0; 7]);
            let (ids, reason, ttft) = if speculative {
                let output = model
                    .generate_prepared_chat_speculative(PreparedChatSpeculativeGenerationRequest {
                        input,
                        drafting: SpeculativeDraft::Embedded,
                        settings: settings(),
                        options: Default::default(),
                        caller_stop_sequences: &[],
                        cancellation,
                        on_event,
                    })
                    .unwrap();
                terminal(output)
            } else {
                let output = model
                    .generate_prepared_chat(PreparedChatGenerationRequest {
                        input,
                        settings: settings(),
                        caller_stop_sequences: &[],
                        cancellation,
                        on_event,
                    })
                    .unwrap();
                terminal(output)
            };
            assert_eq!(reason, FinishReason::Cancelled);
            assert_eq!(ids.len(), usize::from(!pre_cancel));
            assert_eq!(ttft.is_some(), !pre_cancel);
        }
    }
}

#[test]
fn speculative_batch_ttft_uses_a_shared_origin_and_excludes_cancelled_lanes() {
    let mut model = unicode_model(Some(5));
    let chat = chat(&mut model);
    let cancelled = GenerationCancellationToken::new();
    cancelled.cancel();
    let started = Instant::now();
    let output = model
        .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: [
                GenerationCancellationToken::new(),
                cancelled,
                GenerationCancellationToken::new(),
            ]
            .into_iter()
            .map(|cancellation| PreparedChatSpeculativeBatchLane {
                input: PreparedChatInput::rendered_prompt(&chat),
                settings: settings(),
                max_draft_tokens: NonZeroUsize::new(2).unwrap(),
                caller_stop_sequences: &[],
                cancellation,
                on_event: Box::new(|_| {}),
            })
            .collect(),
            scheduler: Default::default(),
        })
        .unwrap();
    let first = output.requests()[0].timing().time_to_first_token().unwrap();
    let last = output.requests()[2].timing().time_to_first_token().unwrap();
    assert!(first <= last);
    assert!(last <= started.elapsed());
    assert_eq!(output.requests()[1].timing().time_to_first_token(), None);
    assert!(output.requests()[1].token_ids().is_empty());
    for index in [0, 2] {
        let lane = &output.requests()[index];
        assert_eq!(lane.token_ids(), [7]);
        assert!(lane.stats().submission_to_first_token().unwrap() <= lane.stats().elapsed());
        assert!(
            lane.stats().submission_to_first_token().unwrap()
                <= lane.timing().time_to_first_token().unwrap()
        );
    }
}

#[test]
fn speculative_ttft_counts_a_buffered_unicode_token_cancelled_before_text() {
    let mut model = unicode_model(Some(7));
    let chat = chat(&mut model);
    let cancellation = GenerationCancellationToken::new();
    let pre_cancelled = GenerationCancellationToken::new();
    pre_cancelled.cancel();
    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: vec![
                PreparedChatSpeculativeBatchLane {
                    input: PreparedChatInput::rendered_prompt(&chat),
                    settings: settings(),
                    max_draft_tokens: NonZeroUsize::new(2).unwrap(),
                    caller_stop_sequences: &[],
                    cancellation: cancellation.clone(),
                    on_event: Box::new(|event| events.push(event)),
                },
                PreparedChatSpeculativeBatchLane {
                    input: PreparedChatInput::rendered_prompt(&chat),
                    settings: settings(),
                    max_draft_tokens: NonZeroUsize::new(2).unwrap(),
                    caller_stop_sequences: &[],
                    cancellation: pre_cancelled,
                    // Lane 0 has prefetched a byte fragment, but published no text.
                    on_event: Box::new(|_| cancellation.cancel()),
                },
            ],
            scheduler: Default::default(),
        })
        .unwrap();
    assert_eq!(output.requests()[0].token_ids(), [7]);
    assert_eq!(
        output.requests()[0].finish_reason(),
        FinishReason::Cancelled
    );
    assert!(output.requests()[0]
        .timing()
        .time_to_first_token()
        .is_some());
    assert_eq!(output.requests()[1].timing().time_to_first_token(), None);
    assert_eq!(
        events,
        [SemanticEvent::Finished {
            reason: FinishReason::Cancelled
        }]
    );
}

#[test]
fn observed_ttft_counts_a_buffered_unicode_token_before_record_delivery() {
    use eredu::api::ObservedGenerationEvent;
    use std::ops::ControlFlow;

    let mut probe = unicode_model(None);
    let prepared = chat(&mut probe);
    let first = probe
        .encode(prepared.rendered_prompt(), false)
        .unwrap()
        .len() as u32;
    let mut model = unicode_model(Some(first));
    let chat = chat(&mut model);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings(),
            eredu_core::capture::CapturePlan::none(),
            eredu::api::TraceLimits {
                per_record_bytes: 65536,
                total_bytes: 1024 * 1024,
            },
        )
        .unwrap();
    let started = Instant::now();
    let mut token_delivered = None;
    let output = model
        .generate_observed_chat(prepared, &[], Default::default(), |record| {
            match record.event {
                ObservedGenerationEvent::Token { token_id, .. } => {
                    assert_eq!(token_id, first);
                    token_delivered = Some(started.elapsed());
                    ControlFlow::Break(())
                }
                ObservedGenerationEvent::Semantic {
                    event: SemanticEvent::TextDelta(_),
                    ..
                } => {
                    panic!("a byte fragment must not produce visible text")
                }
                _ => ControlFlow::Continue(()),
            }
        })
        .unwrap();
    assert_eq!(output.token_ids, [first]);
    assert_eq!(output.finish_reason, FinishReason::Cancelled);
    assert!(output.timing().time_to_first_token().unwrap() <= token_delivered.unwrap());
}
