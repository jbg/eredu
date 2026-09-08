use super::*;
use eredu::api::{ControlledGenerationRecord, ObservedGenerationEvent, TraceLimits};
use eredu_core::{capture::CapturePlan, execution_control::*};
use std::ops::ControlFlow;

#[path = "control/snapshots.rs"]
mod snapshots;

#[path = "control/text.rs"]
mod text;

fn setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
    u32,
) {
    let request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let mut probe = unicode_model(None);
    let chat = probe.prepare_chat(request()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let mut model = unicode_model(Some(first));
    let chat = model.prepare_chat(request()).unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(8),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    (model, chat, settings, first)
}
fn limits() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    }
}

impl eredu_runtime::execution_control::TextSamplingControlBackend for MockBackend {
    fn sampling_control_facts(state: &observed_mock::State) -> eredu::api::SamplingStateFacts {
        eredu::api::SamplingStateFacts {
            temperature: state.sampling.temperature,
            requires_positive_temperature: false,
            has_rng: state.sampling.seed.is_some(),
        }
    }
    fn install_sampling_override(
        _: &mut ModelRuntime<Self>,
        state: &mut observed_mock::State,
        request: eredu_runtime::execution_control::ValidatedSamplingOverride,
    ) -> Result<(), MockError> {
        if let Some(seed) = request.reseed() {
            state.sampling.seed = Some(seed);
        }
        state.sampling.temperature = request.temperature();
        Ok(())
    }
}

#[test]
fn prospective_sampling_changes_preserve_position_and_emit_bounded_provenance() {
    use eredu::api::SamplingOverride;
    let (mut model, chat, settings, first) = setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut records = vec![];
    let mut session = model
        .start_controlled_chat(prepared, &[], Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(!session.sampling_state().unwrap().has_rng);
    assert_eq!(
        session.capabilities().sampling_overrides,
        ControlSupport::Supported
    );
    let request = SamplingOverride {
        temperature: Some(0.7),
        reseed: None,
    };
    assert!(session
        .override_sampling(request, |_| panic!("invalid change emitted"))
        .is_err());
    assert_eq!(session.next_prediction(), 0);
    let request = SamplingOverride {
        reseed: Some(99),
        ..request
    };
    session
        .override_sampling(request, |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.next_prediction(), 0);
    assert_eq!(session.sampling_state().unwrap().temperature, 0.7);
    let record = records.last().unwrap();
    assert!(
        matches!(record.generation.event, ObservedGenerationEvent::SamplingChanged {
        next_prediction: 0, request: r, ..
    } if r == request)
    );
    session.force_next_token(first).unwrap();
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    session
        .override_sampling(
            SamplingOverride {
                temperature: Some(0.0),
                reseed: None,
            },
            |r| {
                records.push(r);
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    assert_eq!(session.next_prediction(), 1);
    assert_eq!(session.token_ids(), [first]);
    assert!(session.sampling_state().unwrap().has_rng);
    let bytes: usize = records
        .iter()
        .map(|r| serde_json::to_vec(r).unwrap().len())
        .sum();
    assert_eq!(session.emitted_bytes(), bytes as u64);
    session
        .override_sampling(SamplingOverride::default(), |_| ControlFlow::Break(()))
        .unwrap();
    assert!(session
        .override_sampling(request, |_| panic!("closed consumer called"))
        .is_err());
    assert_eq!(
        session.step(|_| panic!("closed consumer called")).unwrap(),
        GenerationStatus::Cancelled
    );
}

#[test]
fn decoder_forks_preserve_pending_unicode_without_replaying_tokens() {
    let (model, _, _, first) = setup();
    let mut original = model.text_decoder(true);
    assert!(original.step(first).unwrap().is_none());
    let saved = original.clone();
    let mut left = saved.clone();
    let mut right = saved.clone();
    assert_eq!(original.step(first + 1).unwrap().as_deref(), Some("é"));
    assert_eq!(left.step(first + 1).unwrap().as_deref(), Some("é"));
    // Advancing/dropping one fork cannot consume another fork's retained bytes.
    left.step(first + 2).unwrap();
    drop(left);
    assert_eq!(right.step(first + 1).unwrap().as_deref(), Some("é"));
}

#[test]
fn forced_canonical_tokens_use_ordinary_commitment_decoding_and_termination() {
    let (mut model, chat, settings, first) = setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, observed_mock::plan(), limits())
        .unwrap();
    let mut records = vec![];
    let mut session = model
        .start_controlled_chat(prepared, &[], Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(session.force_next_token(u32::MAX).is_err());
    assert_eq!(session.next_prediction(), 0);
    assert_eq!(session.pending_forced_token(), None);
    session.force_next_token(first).unwrap();
    assert!(session.force_next_token(first + 1).is_err());
    assert_eq!(session.pending_forced_token(), Some(first));
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.token_ids(), [first]);
    assert_eq!(session.pending_forced_token(), None);
    // The grammar requires the second UTF-8 byte before EOS can be consumed.
    assert!(session.force_next_token(first + 2).is_err());
    assert_eq!(session.pending_forced_token(), None);
    session.force_next_token(first + 1).unwrap();
    assert!(session.clear_forced_token().unwrap());
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    session.force_next_token(first + 2).unwrap();
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.status(), GenerationStatus::Completed);
    assert!(session.force_next_token(first).is_err());
    let committed: Vec<_> = records
        .iter()
        .filter_map(|r| match r.generation.event {
            ObservedGenerationEvent::Token {
                token_id, forced, ..
            } => Some((token_id, forced)),
            _ => None,
        })
        .collect();
    assert_eq!(
        committed,
        [(first, true), (first + 1, false), (first + 2, true)]
    );
    let text: String = semantic(&records)
        .iter()
        .filter_map(|e| match e {
            SemanticEvent::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "é");
}

#[test]
fn forced_alternative_changes_canonical_history_and_semantic_text_once() {
    let (mut model, chat, settings, first) = setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let alternative = first + 3;
    let expected_text = model.decode(&[alternative], true).unwrap();
    let mut records = vec![];
    let mut session = model
        .start_controlled_chat(prepared, &[], Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    session.force_next_token(alternative).unwrap();
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.token_ids(), [alternative]);
    session.force_next_token(first + 2).unwrap();
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.token_ids(), [alternative, first + 2]);
    // ChatML's semantic terminator takes the same precedence as ordinary EOS.
    assert_eq!(session.finish_reason(), Some(FinishReason::StopSequence));
    let text: String = semantic(&records)
        .iter()
        .filter_map(|e| match e {
            SemanticEvent::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, expected_text);
    assert_ne!(text, "é");
}
fn semantic(records: &[ControlledGenerationRecord]) -> Vec<SemanticEvent> {
    records
        .iter()
        .filter_map(|r| match &r.generation.event {
            ObservedGenerationEvent::Semantic { event, .. } => Some(event.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn controlled_facade_pauses_unicode_and_resumes_the_ordinary_pipeline_in_all_modes() {
    for mode in 0..4 {
        let (mut model, chat, settings, first) = setup();
        let mut ordinary = vec![];
        let baseline = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                settings,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |event| ordinary.push(event),
            })
            .unwrap();
        let capture = if mode & 1 == 0 {
            CapturePlan::none()
        } else {
            observed_mock::plan()
        };
        // Intervention-only still needs nonzero explicit evidence budgets.
        let mut capture = capture;
        capture.limits = observed_mock::plan().limits;
        let prepared = if mode & 2 == 0 {
            model
                .prepare_observed_chat(&chat, settings, capture, limits())
                .unwrap()
        } else {
            model
                .prepare_intervened_chat(
                    &chat,
                    settings,
                    capture,
                    observed_mock::intervention_plan(1.0),
                    limits(),
                )
                .unwrap()
        };
        let mut records = vec![];
        let control = GenerationControlHandle::default();
        let mut session = model
            .start_controlled_chat(prepared, &[], control.clone(), |record| {
                records.push(record);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Prepared);
        assert_eq!(session.next_prediction(), 0);
        assert_eq!(session.capabilities().step, ControlSupport::Supported);
        let initial_storage = session
            .semantic_snapshot_bytes()
            .expect("built-in semantic estimate");
        session
            .step(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), [first]);
        assert!(session.semantic_snapshot_bytes().unwrap() > initial_storage);
        assert_eq!(session.next_prediction(), 1);
        assert!(semantic(&records)
            .iter()
            .all(|e| !matches!(e, SemanticEvent::TextDelta(_))));
        session
            .pause(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), [first]);
        assert_eq!(session.status(), GenerationStatus::Paused);
        // A consumer can request pause from a separate thread without moving the
        // native owner. The request stops the run after this token's delivery.
        session
            .resume(|r| {
                if matches!(
                    r.generation.event,
                    ObservedGenerationEvent::Token {
                        prediction_index: 1,
                        ..
                    }
                ) {
                    let control = control.clone();
                    std::thread::spawn(move || control.request_pause())
                        .join()
                        .unwrap();
                }
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), [first, first + 1]);
        assert_eq!(session.next_prediction(), 2);
        assert_eq!(session.status(), GenerationStatus::Paused);
        session
            .resume(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), baseline.token_ids);
        assert_eq!(session.finish_reason(), Some(baseline.finish_reason));
        assert_eq!(session.status(), GenerationStatus::Completed);
        assert_eq!(semantic(&records), ordinary);
        assert!(session
            .step(|_| panic!("terminal session emitted"))
            .is_err());
        assert!(session
            .pause(|_| panic!("terminal session emitted"))
            .is_err());
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.sequence, index as u64);
            assert_eq!(record.epoch, 0);
        }
        let bytes: usize = records
            .iter()
            .map(|r| serde_json::to_vec(r).unwrap().len())
            .sum();
        assert_eq!(session.emitted_bytes(), bytes as u64);
    }
}

#[test]
fn controlled_facade_stop_and_cancellation_do_not_flush_partial_unicode_or_advance_again() {
    for cancel_at in [None, Some(0), Some(1)] {
        let (mut model, chat, settings, first) = setup();
        let prepared = model
            .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
            .unwrap();
        let control = GenerationControlHandle::default();
        let mut records = vec![];
        let mut session = model
            .start_controlled_chat(prepared, &["é".into()], control.clone(), |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        if cancel_at == Some(0) {
            control.cancel();
        }
        session
            .step(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        if cancel_at == Some(1) {
            session
                .cancel(|r| {
                    records.push(r);
                    ControlFlow::Continue(())
                })
                .unwrap();
        }
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.next_prediction(), session.token_ids().len() as u64);
        match cancel_at {
            None => {
                assert_eq!(session.token_ids(), [first, first + 1]);
                assert_eq!(session.finish_reason(), Some(FinishReason::StopSequence));
            }
            Some(n) => {
                assert_eq!(session.token_ids().len(), n);
                assert_eq!(session.finish_reason(), Some(FinishReason::Cancelled));
            }
        }
        assert!(semantic(&records)
            .iter()
            .all(|e| !matches!(e, SemanticEvent::TextDelta(_))));
    }
}

#[test]
fn controlled_facade_closes_broken_consumers_and_fences_panics_and_transport_failures() {
    for failure in ["break", "panic", "limit"] {
        let (mut model, chat, settings, _) = setup();
        let mut limits = limits();
        if failure == "limit" {
            limits.total_bytes = 1500;
        }
        let prepared = model
            .prepare_observed_chat(&chat, settings, observed_mock::plan(), limits)
            .unwrap();
        let mut session = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        let mut calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            session.step(|record| {
                calls += 1;
                if matches!(
                    record.generation.event,
                    ObservedGenerationEvent::Token { .. }
                ) {
                    if failure == "panic" {
                        panic!("consumer disconnected");
                    }
                    if failure == "break" {
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            })
        }));
        match failure {
            "break" => {
                assert_eq!(result.unwrap().unwrap(), GenerationStatus::Cancelled);
                assert_eq!(calls, 1);
            }
            "panic" => {
                assert!(result.is_err());
                assert_eq!(session.status(), GenerationStatus::Failed);
            }
            "limit" => {
                assert!(result.unwrap().is_err());
                assert_eq!(session.status(), GenerationStatus::Failed);
            }
            _ => unreachable!(),
        }
        let position = session.next_prediction();
        assert!(session
            .step(|_| panic!("failed consumer called again"))
            .is_err());
        assert_eq!(session.next_prediction(), position);
    }
}

#[test]
fn controlled_choices_use_sparse_tokenizer_ids_instead_of_entry_count() {
    let base = unicode_model(None);
    let mut value: serde_json::Value =
        serde_json::from_str(&base.tokenizer().to_string(false).unwrap()).unwrap();
    // Leave IDs 2 and 3 unmapped, preserving the high mapped IDs and EOS.
    let vocabulary = value["model"]["vocab"].as_object_mut().unwrap();
    vocabulary.remove("ordinary_1");
    vocabulary.remove("ordinary_2");
    // Keep the special token's explicit ID when the tokenizer is reconstructed.
    vocabulary.insert("<|im_end|>".into(), serde_json::json!(65));
    let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(tokenizer.get_vocab_size(true) < 65);
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut model = LoadedModel::from_runtime(
        ModelRuntime::prepare(MockBackend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "sparse-control".into(),
            chat_template: Some(QWEN_TEMPLATE.into()),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    );
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let prepared = model
        .prepare_observed_chat(
            &chat,
            PreparedChatGenerationSettings::default(),
            CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut session = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(session.force_next_token(2).is_err());
    assert!(session.force_next_token(eos + 1).is_err());
    assert_eq!(session.next_prediction(), 0);
    session.force_next_token(64).unwrap();
    session.step(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), [64]);
    session.force_next_token(eos).unwrap();
    session.step(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), [64, eos]);
}
