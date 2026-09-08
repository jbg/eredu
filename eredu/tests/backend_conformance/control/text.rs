use super::*;
use eredu::api::{GenerationBranchOptions, PreparedObservedGeneration};
use eredu::runtime::chat::{CapabilitySupport, SemanticSupport, ToolChoice};

// Deliberately unlike any recognized protocol, including updated LFM templates.
const TEMPLATE: &str = "{% for message in messages %}{{ message.role }} says: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}reply begins: {% endif %}";

fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        add_generation_prompt: true,
        ..Default::default()
    }
}

fn text_setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
    u32,
) {
    let mut probe = unicode_model_with_template(None, 64, TEMPLATE);
    let chat = probe.prepare_chat(request()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let mut model = unicode_model_with_template(Some(first), 64, TEMPLATE);
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

fn prepare(
    model: &LoadedModel<MockBackend>,
    chat: &PreparedChat,
    settings: PreparedChatGenerationSettings,
    mode: u8,
) -> PreparedObservedGeneration {
    let mut capture = if mode & 1 == 0 {
        CapturePlan::none()
    } else {
        observed_mock::plan()
    };
    capture.limits = observed_mock::plan().limits;
    if mode & 2 == 0 {
        model
            .prepare_observed_chat(chat, settings, capture, limits())
            .unwrap()
    } else {
        model
            .prepare_intervened_chat(
                chat,
                settings,
                capture,
                observed_mock::intervention_plan(1.0),
                limits(),
            )
            .unwrap()
    }
}

fn collect(
    records: &mut Vec<ControlledGenerationRecord>,
) -> impl FnMut(ControlledGenerationRecord) -> ControlFlow<()> + '_ {
    |record| {
        records.push(record);
        ControlFlow::Continue(())
    }
}

#[test]
fn unrecognized_chat_stays_strict_while_text_retains_exact_prompt_admissions_and_controls() {
    for mode in 0..4 {
        let (mut model, chat, settings, first) = text_setup();
        assert!(matches!(
            chat.semantic_support(),
            SemanticSupport::Unsupported { .. }
        ));
        assert_eq!(
            chat.text_generation_support(),
            &CapabilitySupport::Supported
        );
        assert_eq!(chat.rendered_prompt(), "user says: hello\nreply begins: ");
        assert_eq!(chat.generation_prompt(), "reply begins: ");
        let prepared = prepare(&model, &chat, settings, mode);
        let prompt = prepared.prompt_token_ids().to_vec();
        assert_eq!(prompt, model.encode(chat.rendered_prompt(), false).unwrap());
        let error = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                panic!("rejected chat emitted")
            })
            .err()
            .unwrap();
        assert!(error.to_string().contains("executable semantic plan"));
        let prepared = prepare(&model, &chat, settings, mode);
        let capture_id = prepared.capture_plan().identity().to_owned();
        let intervention_id = prepared
            .intervention_plan()
            .map(|p| p.identity().to_owned());
        let resolved = prepared.generation_config();
        let control = GenerationControlHandle::default();
        let mut records = vec![];
        let mut run = model
            .start_controlled_text(prepared, &[], control.clone(), collect(&mut records))
            .unwrap();
        assert_eq!(run.status(), GenerationStatus::Prepared);
        assert!(run.token_ids().is_empty());
        assert_eq!(run.next_prediction(), 0);
        assert_eq!(records.len(), 1);
        assert!(
            matches!(&records[0].generation.event, ObservedGenerationEvent::Started { prompt_token_ids, generation, seed } if prompt_token_ids == &prompt && generation == &resolved && *seed == 17)
        );
        assert_eq!(run.capabilities().step, ControlSupport::Supported);
        run.step(collect(&mut records)).unwrap();
        // This backend's prefill predicts prompt length, proving it received these IDs.
        assert_eq!(run.token_ids(), [first]);
        assert!(semantic(&records).is_empty());
        run.pause(collect(&mut records)).unwrap();
        run.run(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), [first]);
        run.resume(|record| {
            if matches!(
                record.generation.event,
                ObservedGenerationEvent::Token {
                    prediction_index: 1,
                    ..
                }
            ) {
                control.request_pause();
            }
            records.push(record);
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(run.status(), GenerationStatus::Paused);
        assert_eq!(run.token_ids(), [first, first + 1]);
        run.resume(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), [first, first + 1, first + 2]);
        assert_eq!(run.finish_reason(), Some(FinishReason::Eos));
        assert_eq!(
            semantic(&records),
            [
                SemanticEvent::TextDelta("é".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::Eos
                }
            ]
        );
        assert!(run.step(|_| panic!("completed run emitted")).is_err());
        for (sequence, record) in records.iter().enumerate() {
            assert_eq!(record.sequence, sequence as u64);
            assert_eq!(record.generation.capture_plan_id, capture_id);
            assert_eq!(record.generation.intervention_plan_id, intervention_id);
            if let ObservedGenerationEvent::Token {
                captures,
                committed,
                ..
            } = &record.generation.event
            {
                assert!(*committed);
                assert_eq!(captures.is_some(), mode != 0);
                if let Some(step) = captures {
                    assert_eq!(step.records.len(), usize::from(mode & 1 != 0));
                    assert_eq!(step.interventions.len(), usize::from(mode & 2 != 0));
                    assert!(step.step_usage.captures > 0);
                    assert!(step.cumulative_usage.captures >= step.step_usage.captures);
                    for applied in &step.interventions {
                        assert_eq!(Some(applied.plan_id.as_str()), intervention_id.as_deref());
                        assert_eq!(
                            applied.outcome,
                            eredu_core::intervention::InterventionOutcome::Applied
                        );
                        assert_eq!(applied.evidence.len(), 2);
                    }
                }
            }
        }
        assert_eq!(
            run.emitted_bytes(),
            records
                .iter()
                .map(|r| serde_json::to_vec(r).unwrap().len() as u64)
                .sum::<u64>()
        );
    }
}

#[test]
fn text_stops_cancellation_and_callback_closure_preserve_committed_prefix() {
    for cancel_at in [None, Some(0), Some(1)] {
        let (mut model, chat, settings, first) = text_setup();
        let prepared = prepare(&model, &chat, settings, 1);
        let control = GenerationControlHandle::default();
        let mut records = vec![];
        let mut run = model
            .start_controlled_text(
                prepared,
                &["é".into()],
                control.clone(),
                collect(&mut records),
            )
            .unwrap();
        if cancel_at == Some(0) {
            control.cancel();
        }
        run.step(collect(&mut records)).unwrap();
        if cancel_at == Some(1) {
            run.cancel(collect(&mut records)).unwrap();
        }
        run.run(collect(&mut records)).unwrap();
        let reason = if let Some(count) = cancel_at {
            assert_eq!(run.token_ids().len(), count);
            FinishReason::Cancelled
        } else {
            assert_eq!(run.token_ids(), [first, first + 1]);
            FinishReason::StopSequence
        };
        assert_eq!(run.finish_reason(), Some(reason));
        assert_eq!(semantic(&records), [SemanticEvent::Finished { reason }]);
    }
    let (mut model, chat, settings, _) = text_setup();
    let prepared = prepare(&model, &chat, settings, 0);
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    let mut calls = 0;
    run.step(|_| {
        calls += 1;
        ControlFlow::Break(())
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(run.status(), GenerationStatus::Cancelled);
    assert!(run.step(|_| panic!("closed callback invoked")).is_err());
}

#[test]
fn text_snapshots_and_siblings_preserve_unicode_stop_lookbehind_and_bounded_storage() {
    for mode in 0..4 {
        for stop in ["never", "é!"] {
            let (mut model, chat, settings, first) = text_setup();
            let prepared = prepare(&model, &chat, settings, mode);
            let mut records = vec![];
            let mut run = model
                .start_controlled_text(
                    prepared,
                    &[stop.into()],
                    Default::default(),
                    collect(&mut records),
                )
                .unwrap();
            let bounds = SnapshotLimits {
                max_snapshots: 1,
                max_branches: 2,
                retained_bytes: 64_000_000,
                cumulative_copy_bytes: 256_000_000,
            };
            run.enable_snapshots(bounds).unwrap();
            run.step(collect(&mut records)).unwrap();
            // Save either partial UTF-8 or decoded text held back by a caller stop.
            if stop == "é!" {
                run.step(collect(&mut records)).unwrap();
            }
            assert!(semantic(&records).is_empty());
            let saved = run.snapshot(collect(&mut records)).unwrap();
            assert_eq!(run.capabilities().fork, ControlSupport::Supported);
            assert!(run.snapshot(collect(&mut records)).is_err());
            let options = || GenerationBranchOptions {
                trace_limits: TraceLimits {
                    per_record_bytes: 16384,
                    total_bytes: 65536,
                },
                capture_limits: Some(observed_mock::plan().limits),
                sampling: None,
                intervention: None,
            };
            let mut left = run.fork(&saved, options(), collect(&mut records)).unwrap();
            let mut right = run.fork(&saved, options(), collect(&mut records)).unwrap();
            assert!(run.fork(&saved, options(), collect(&mut records)).is_err());
            let before = records.len();
            run.run(collect(&mut records)).unwrap();
            let continuation = semantic(&records[before..]);
            assert_eq!(
                continuation,
                [
                    SemanticEvent::TextDelta("é".into()),
                    SemanticEvent::Finished {
                        reason: FinishReason::Eos
                    }
                ]
            );
            let parent_id = run.output_checkpoint().run_id;
            let usage = run.snapshot_usage().unwrap();
            run.restore(&saved, collect(&mut records)).unwrap();
            assert!(
                run.snapshot_usage().unwrap().cumulative_copy_bytes > usage.cumulative_copy_bytes
            );
            let before = records.len();
            run.run(collect(&mut records)).unwrap();
            assert_eq!(semantic(&records[before..]), continuation);
            run.exchange(&mut left, collect(&mut records)).unwrap();
            run.cancel(collect(&mut records)).unwrap();
            run.exchange(&mut right, collect(&mut records)).unwrap();
            let before = records.len();
            run.run(collect(&mut records)).unwrap();
            assert_eq!(semantic(&records[before..]), continuation);
            assert_eq!(run.token_ids(), [first, first + 1, first + 2]);
            run.exchange(&mut left, collect(&mut records)).unwrap();
            assert_eq!(run.output_checkpoint().run_id, parent_id);
            assert_eq!(run.status(), GenerationStatus::Completed);
            drop((left, right, saved));
            assert_eq!(run.snapshot_usage().unwrap().retained_bytes, 0);
        }
    }
}

#[test]
fn text_admission_never_discards_requested_tools_or_explicit_thinking() {
    for template in [TEMPLATE, QWEN_TEMPLATE] {
        let mut model = unicode_model_with_template(None, 512, template);
        for choice in [ToolChoice::None, ToolChoice::Auto, ToolChoice::Required] {
            let chat = model.prepare_chat(ChatTemplateRequest {
                tools: vec![serde_json::json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}})],
                tool_choice: choice,
                ..request()
            }).unwrap();
            assert!(matches!(
                chat.text_generation_support(),
                CapabilitySupport::Unsupported { .. }
            ));
            let prepared = prepare(&model, &chat, Default::default(), 0);
            let error = model
                .start_controlled_text(prepared, &[], Default::default(), |_| {
                    panic!("rejected request emitted")
                })
                .err()
                .unwrap();
            assert!(error.to_string().contains("tool declarations"));
        }
    }
    let (mut model, _, settings, _) = text_setup();
    assert!(model
        .prepare_chat(ChatTemplateRequest {
            enable_thinking: Some(true),
            ..request()
        })
        .is_err());
    let required = model
        .prepare_chat(ChatTemplateRequest {
            tool_choice: ToolChoice::Required,
            ..request()
        })
        .unwrap();
    let prepared = prepare(&model, &required, settings, 0);
    assert!(model
        .start_controlled_text(prepared, &[], Default::default(), |_| panic!(
            "required tools admitted"
        ))
        .is_err());
    let raw = model
        .prepare_chat(ChatTemplateRequest {
            enable_thinking: Some(true),
            allow_unparsed_reasoning: true,
            ..request()
        })
        .unwrap();
    let prepared = prepare(&model, &raw, settings, 0);
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    run.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(run.finish_reason(), Some(FinishReason::Eos));
}

#[test]
fn text_sampling_and_forcing_exclude_sparse_and_padded_ids() {
    let base = unicode_model_with_template(None, 64, TEMPLATE);
    let mut value: serde_json::Value =
        serde_json::from_str(&base.tokenizer().to_string(false).unwrap()).unwrap();
    let vocab = value["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove("ordinary_1"); // Hole at ID 2.
    vocab.insert("<|im_end|>".into(), serde_json::json!(65));
    let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap();
    let mut model = LoadedModel::from_runtime(
        ModelRuntime::prepare(MockBackend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "sparse-text".into(),
            chat_template: Some(TEMPLATE.into()),
            eos_token_ids: vec![65],
            checkpoint_generation_config: None,
        },
    );
    let chat = model.prepare_chat(request()).unwrap();
    let prepared = prepare(
        &model,
        &chat,
        PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            seed: 17,
            ..Default::default()
        },
        0,
    );
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    for id in [2, 66, u32::MAX] {
        assert!(run.force_next_token(id).is_err());
    }
    run.force_next_token(64).unwrap();
    run.step(|_| ControlFlow::Continue(())).unwrap();
    run.force_next_token(1).unwrap();
    run.step(|_| ControlFlow::Continue(())).unwrap();
    run.step(|_| ControlFlow::Continue(())).unwrap(); // Backend proposes unmapped 2.
    assert_eq!(run.token_ids(), [64, 1, 0]);
    run.force_next_token(65).unwrap();
    run.step(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(run.finish_reason(), Some(FinishReason::Eos));
}

#[test]
fn text_skips_non_eos_special_tokens_and_delivers_protocol_like_text_literally() {
    let base = unicode_model_with_template(None, 64, TEMPLATE);
    let mut value: serde_json::Value =
        serde_json::from_str(&base.tokenizer().to_string(false).unwrap()).unwrap();
    let vocab = value["model"]["vocab"].as_object_mut().unwrap();
    for (id, token) in [(1, "<think>literal"), (2, "<tool_call>"), (3, "<hidden>")] {
        vocab.remove(&format!("ordinary_{}", id - 1));
        vocab.insert(token.into(), serde_json::json!(id));
    }
    let mut tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("<hidden>", true).normalized(false)])
        .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut model = LoadedModel::from_runtime(
        ModelRuntime::prepare(MockBackend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "literal-text".into(),
            chat_template: Some(TEMPLATE.into()),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    );
    let chat = model.prepare_chat(request()).unwrap();
    let prepared = prepare(&model, &chat, Default::default(), 0);
    let mut records = vec![];
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), collect(&mut records))
        .unwrap();
    for id in [1, 3, 2, eos] {
        run.force_next_token(id).unwrap();
        run.step(collect(&mut records)).unwrap();
    }
    assert_eq!(run.token_ids(), [1, 3, 2, eos]);
    assert_eq!(
        semantic(&records),
        [
            SemanticEvent::TextDelta("<think>literal".into()),
            SemanticEvent::TextDelta("<tool_call>".into()),
            SemanticEvent::Finished {
                reason: FinishReason::Eos
            }
        ]
    );
}
