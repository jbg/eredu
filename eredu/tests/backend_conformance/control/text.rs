use super::*;
use eredu::api::GenerationBranchOptions;
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
    original_sources::Fixture<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
    u32,
) {
    let mut probe = unicode_model_with_template(None, 64, TEMPLATE);
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = probe
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        probe
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let mut model = unicode_model_with_template(Some(first), 64, TEMPLATE);
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
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

struct Instrumentation {
    capture: CapturePlan,
    intervention: Option<eredu_core::intervention::InterventionPlan>,
}
fn declarations(mode: u8) -> Instrumentation {
    let mut capture = if mode & 1 == 0 {
        CapturePlan::none()
    } else {
        observed_mock::plan()
    };
    capture.limits = observed_mock::plan().limits;
    Instrumentation {
        capture,
        intervention: (mode & 2 != 0).then(|| observed_mock::intervention_plan(1.0)),
    }
}
impl Instrumentation {
    fn request<'a>(
        &'a self,
        chat: &'a PreparedChat,
        settings: PreparedChatGenerationSettings,
    ) -> eredu::api::PreparedChatRequest<'a, Prompt> {
        let mut request =
            eredu::api::PreparedChatRequest::new(chat, original_sources::settings(settings));
        request.capture = Some(&self.capture);
        request.intervention = self.intervention.as_ref();
        request.output_mode = eredu::api::PreparedChatOutputMode::Text;
        request
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
fn observed_literal_text_matches_controlled_text_for_unrecognized_templates_and_plans() {
    for mode in 0..4 {
        let (mut model, chat, settings, _) = text_setup();
        let instrumentation = declarations(mode);
        let mut prepared = instrumentation.request(&chat, settings);
        let mut ordinary = vec![];
        let output = {
            let mut run = model
                .start_controlled_chat(
                    prepared,
                    limits(),
                    Default::default(),
                    collect(&mut ordinary),
                )
                .unwrap()
                .unwrap();
            run.run(collect(&mut ordinary)).unwrap();
            (run.token_ids().to_vec(), run.finish_reason().unwrap())
        };
        model.reset().unwrap();
        let instrumentation = declarations(mode);
        let mut prepared = instrumentation.request(&chat, settings);
        let mut controlled = vec![];
        let mut run = model
            .start_controlled_chat(
                prepared,
                limits(),
                Default::default(),
                collect(&mut controlled),
            )
            .unwrap()
            .unwrap();
        run.run(collect(&mut controlled)).unwrap();
        assert_eq!(run.token_ids(), output.0.as_slice());
        assert_eq!(run.finish_reason(), Some(output.1));
        let events: Vec<_> = ordinary
            .iter()
            .filter_map(|r| match r.event.progress() {
                Some(ObservedGenerationEvent::Semantic { event, .. }) => Some(event.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(events, semantic(&controlled));
        let evidence = |event: &eredu::api::ObservedGenerationEvent| match event {
            ObservedGenerationEvent::Token {
                forced, captures, ..
            } => {
                assert!(!forced);
                Some(
                    captures
                        .as_ref()
                        .map(|step| (step.records.clone(), step.interventions.clone())),
                )
            }
            _ => None,
        };
        assert_eq!(
            ordinary
                .iter()
                .filter_map(|r| r.event.progress().and_then(evidence))
                .collect::<Vec<_>>(),
            controlled
                .iter()
                .filter_map(|r| r.event.progress().and_then(evidence))
                .collect::<Vec<_>>()
        );
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
        let instrumentation = declarations(mode);
        let mut prepared = instrumentation.request(&chat, settings);
        let prompt = model.encode(chat.rendered_prompt(), false).unwrap();
        prepared.output_mode = eredu::api::PreparedChatOutputMode::Semantic;
        assert_eq!(prompt, model.encode(chat.rendered_prompt(), false).unwrap());
        let error = model
            .start_controlled_chat(prepared, limits(), Default::default(), |_| {
                panic!("rejected chat emitted")
            })
            .err()
            .unwrap();
        assert!(matches!(
            error.session_failure().unwrap().semantic_output_rejection(),
            Some(SemanticSupport::Unsupported { .. })
        ));
        let instrumentation = declarations(mode);
        let mut prepared = instrumentation.request(&chat, settings);
        let resolved = model.resolve_generation_config(settings.overrides).unwrap();
        let control = GenerationControlHandle::default();
        let mut records = vec![];
        let mut run = model
            .start_controlled_chat(prepared, limits(), control.clone(), collect(&mut records))
            .unwrap()
            .unwrap();
        let (capture_id, intervention_id) = match &records[0].instrumentation {
            eredu::api::PreparedInstrumentationRecord::Captured { plan_id } => {
                (plan_id.clone(), None)
            }
            eredu::api::PreparedInstrumentationRecord::Intervened {
                capture_plan_id,
                intervention_plan_id,
            } => (capture_plan_id.clone(), Some(intervention_plan_id.clone())),
            _ => panic!("requested source was not retained"),
        };
        assert!(!capture_id.is_empty());
        assert_eq!(intervention_id.is_some(), mode & 2 != 0);
        assert_eq!(run.status(), GenerationStatus::Prepared);
        assert!(run.token_ids().is_empty());
        assert_eq!(run.next_prediction(), 0);
        assert_eq!(records.len(), 1);
        assert!(
            matches!(&records[0].event, eredu::api::ControlledGenerationEvent::Started { prompt_attribution, generation, seed } if prompt_attribution.attribution().complete_token_ids() == Some(prompt.as_slice()) && generation == &resolved && *seed == 17)
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
                &record.event,
                eredu::api::ControlledGenerationEvent::Progress {
                    event: ObservedGenerationEvent::Token {
                        prediction_index: 1,
                        ..
                    }
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
            match &record.instrumentation {
                eredu::api::PreparedInstrumentationRecord::Captured { plan_id } => {
                    assert_eq!(plan_id, &capture_id);
                    assert!(intervention_id.is_none());
                }
                eredu::api::PreparedInstrumentationRecord::Intervened {
                    capture_plan_id,
                    intervention_plan_id,
                } => {
                    assert_eq!(capture_plan_id, &capture_id);
                    assert_eq!(Some(intervention_plan_id), intervention_id.as_ref());
                }
                eredu::api::PreparedInstrumentationRecord::Unobserved => {
                    panic!("capture owner lost")
                }
            }
            if let eredu::api::ControlledGenerationEvent::Progress {
                event:
                    ObservedGenerationEvent::Token {
                        captures,
                        committed,
                        ..
                    },
            } = &record.event
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
        let instrumentation = declarations(1);
        let mut prepared = instrumentation.request(&chat, settings);
        let stops = ["é".into()];
        prepared.stop_sequences = &stops;
        let control = GenerationControlHandle::default();
        let mut records = vec![];
        let mut run = model
            .start_controlled_chat(prepared, limits(), control.clone(), collect(&mut records))
            .unwrap()
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
    let instrumentation = declarations(0);
    let mut prepared = instrumentation.request(&chat, settings);
    let mut run = model
        .start_controlled_chat(prepared, limits(), Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
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
            let pool = model.original_pool().clone();
            let instrumentation = declarations(mode);
            let mut prepared = instrumentation.request(&chat, settings);
            let stops = [stop.into()];
            prepared.stop_sequences = &stops;
            let mut records = vec![];
            let mut run = model
                .start_controlled_chat(
                    prepared,
                    limits(),
                    Default::default(),
                    collect(&mut records),
                )
                .unwrap()
                .unwrap();
            let bounds = SnapshotLimits {
                max_snapshots: 1,
                max_branches: 2,
                retained_bytes: 64_000_000,
                cumulative_copy_bytes: 256_000_000,
            };
            run.enable_snapshots(
                bounds,
                original_sources::CAPACITY,
                eredu_runtime::working_memory::WorkspaceCopyLimits::new(original_sources::CAPACITY),
            )
            .unwrap();
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
            let parent_id = run.output_checkpoint().unwrap().run_id.clone();
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
            assert_eq!(run.output_checkpoint().unwrap().run_id, parent_id);
            assert_eq!(run.status(), GenerationStatus::Completed);
            drop((left, right, saved));
            // The restored parent and escaping semantic aliases still retain
            // independently copied destinations after saved handles retire.
            assert!(run.snapshot_usage().unwrap().retained_bytes > 0);
            drop(records);
            assert!(run.snapshot_usage().unwrap().retained_bytes > 0);
            drop(run);
            drop((continuation, chat, model));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn text_admission_never_discards_requested_tools_or_explicit_thinking() {
    for template in [TEMPLATE, QWEN_TEMPLATE] {
        let mut model = unicode_model_with_template(None, 512, template);
        for choice in [ToolChoice::None, ToolChoice::Auto, ToolChoice::Required] {
            let chat = {
                let request = ChatTemplateRequest {
                    tools: vec![
                        serde_json::json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}),
                    ],
                    tool_choice: choice,
                    ..request()
                };
                let cancellation = eredu_core::GenerationCancellationToken::new();
                let source = model
                    .chat_source(!request.tools.is_empty(), &cancellation)
                    .unwrap()
                    .unwrap();
                model
                    .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                    .unwrap()
                    .unwrap()
            };
            assert!(matches!(
                chat.text_generation_support(),
                CapabilitySupport::Unsupported { .. }
            ));
            let instrumentation = declarations(0);
            let mut prepared = instrumentation.request(&chat, Default::default());
            let error = model
                .start_controlled_chat(prepared, limits(), Default::default(), |_| {
                    panic!("rejected request emitted")
                })
                .err()
                .unwrap();
            assert!(error.to_string().contains("tool declarations"));
        }
    }
    let (mut model, _, settings, _) = text_setup();
    let requested = ChatTemplateRequest {
        enable_thinking: Some(true),
        ..request()
    };
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let source = model.chat_source(false, &cancellation).unwrap().unwrap();
    assert!(model
        .prepare_chat(
            &source,
            &requested,
            original_sources::CAPACITY,
            &cancellation
        )
        .is_err());
    let required = {
        let request = ChatTemplateRequest {
            tool_choice: ToolChoice::Required,
            ..request()
        };
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let instrumentation = declarations(0);
    let mut prepared = instrumentation.request(&required, settings);
    assert!(model
        .start_controlled_chat(prepared, limits(), Default::default(), |_| panic!(
            "required tools admitted"
        ))
        .is_err());
    let raw = {
        let request = ChatTemplateRequest {
            enable_thinking: Some(true),
            allow_unparsed_reasoning: true,
            ..request()
        };
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let instrumentation = declarations(0);
    let mut prepared = instrumentation.request(&raw, settings);
    let mut run = model
        .start_controlled_chat(prepared, limits(), Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    run.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(run.finish_reason(), Some(FinishReason::Eos));
}

#[test]
fn text_sampling_and_forcing_exclude_sparse_and_padded_ids() {
    let tokenizer = unicode_tokenizer(None, 64);
    let mut value: serde_json::Value =
        serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
    let vocab = value["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove("ordinary_1"); // Hole at ID 2.
    vocab.insert("<|im_end|>".into(), serde_json::json!(65));
    let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap();
    let mut model = original_sources::Fixture::from_runtime(
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
    )
    .unwrap();
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let instrumentation = declarations(0);
    let mut prepared = instrumentation.request(
        &chat,
        PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            seed: 17,
            ..Default::default()
        },
    );
    let mut run = model
        .start_controlled_chat(prepared, limits(), Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
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
    let tokenizer = unicode_tokenizer(None, 64);
    let mut value: serde_json::Value =
        serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
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
    let mut model = original_sources::Fixture::from_runtime(
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
    )
    .unwrap();
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let instrumentation = declarations(0);
    let mut prepared = instrumentation.request(&chat, Default::default());
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(
            prepared,
            limits(),
            Default::default(),
            collect(&mut records),
        )
        .unwrap()
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
