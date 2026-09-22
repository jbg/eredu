//! Exact chat/media association, semantic state copies and speculative generation.
use super::*;
use eredu_core::{
    InputExtent, InputMetadataKey, InputModality, InputPayloadKind, TokenInputRejection,
};
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView};

#[test]
#[ignore = "requires an accessible Metal device"]
fn authenticated_image_tools_refuse_foreign_chat_and_preserve_manual_and_speculative_parity() {
    check(false, CAPACITY);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn authenticated_image_tool_snapshots_preserve_pending_branches_and_partial_events() {
    check(true, CAPACITY);
}

// Both cases use the same actual image, source authentication, uneven chunks,
// deterministic decoder and event oracle. Their independent lifetimes let a
// branch-admission failure report separately from speculative execution.
fn check(snapshots: bool, capacity: u64) {
    for choice in [ToolChoice::Required, ToolChoice::Auto] {
        let (root, tokenizer, script) = fixture(true);
        let execution = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "metal:0").unwrap());
        let speculative_execution =
            execution
                .clone()
                .with_drafting(eredu_core::DraftingPlan::External {
                    model: root.path().display().to_string(),
                    placement: eredu_core::DraftPlacementPlan::Target,
                    max_draft_tokens: 2,
                    lookahead: false,
                    adaptive_lookahead: false,
                });
        let loaded = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            root.path(),
            &speculative_execution,
        )
        .unwrap_or_else(fail);
        let speculative_options = loaded
            .speculative_generation_options()
            .unwrap_or_else(fail)
            .unwrap();
        let (mut model, mut drafting) = loaded.into_parts();
        let (mut foreign_model, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            root.path(),
            &execution,
        )
        .unwrap_or_else(fail)
        .into_parts();
        let cancel = GenerationCancellationToken::new();
        let tokens = model
            .compile_managed_plain_text_source(
                std::fs::File::open(root.path().join("tokenizer.json")).unwrap(),
            )
            .unwrap_or_else(fail);
        let source = model
            .compile_managed_chat_source(
                &tokens,
                std::fs::File::open(root.path().join("tokenizer_config.json")).unwrap(),
                true,
                &cancel,
            )
            .unwrap_or_else(fail)
            .unwrap();
        let policy = image_policy(choice);
        let chat = model
            .prepare_chat(&source, &policy, &native_limits(capacity), &cancel)
            .unwrap_or_else(fail)
            .unwrap();
        let encoded = tokenizer.encode(chat.rendered_prompt(), false).unwrap();
        let image_id = tokenizer.token_to_id("<|image_pad|>").unwrap();
        let marker = encoded
            .get_ids()
            .iter()
            .position(|&id| id == image_id)
            .unwrap();
        assert_eq!(
            encoded
                .get_ids()
                .iter()
                .filter(|&&id| id == image_id)
                .count(),
            1
        );
        let leading = &encoded.get_ids()[..marker];
        let trailing = &encoded.get_ids()[marker + 1..];
        let image = ImagePrompt::new(leading, trailing);
        let parts = image.parts();
        let cancelled = GenerationCancellationToken::new();
        cancelled.cancel();
        assert!(model
            .prepare_chat_input(&chat, &parts, &cancelled)
            .unwrap_or_else(fail)
            .is_none());
        let decoder_positions = encoded.len() as u64 + 3; // one marker expands to four image rows
        let chunk = if decoder_positions % 17 == 0 { 19 } else { 17 };
        assert!(decoder_positions > chunk && decoder_positions % chunk != 0);
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(8),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                memory_limits: eredu_core::MemoryLimitDeclarations::new([(
                    "host".into(),
                    eredu_core::MemoryLimit::Finite(capacity),
                )]),
                prefill_chunk_positions: NonZeroU64::new(chunk),
                ..Default::default()
            },
            seed: 37,
            ..Default::default()
        };
        let input = model
            .prepare_chat_input(&chat, &parts, &cancel)
            .unwrap_or_else(fail)
            .unwrap();
        let binding = input.chat_binding().unwrap();
        assert_eq!(binding.coordinates().len(), 3);
        assert_eq!(
            binding.coordinates()[1].rendered_range(),
            [marker, marker + 1]
        );
        assert_eq!(
            binding.coordinates()[1].decoder_range(),
            Some([marker as u64, marker as u64 + 4])
        );
        assert_eq!(
            binding
                .coordinates()
                .last()
                .unwrap()
                .decoder_range()
                .unwrap()[1],
            decoder_positions
        );
        let mut foreign_execution = PreparedChatRequest::new(&chat, settings.clone());
        foreign_execution.input = eredu::api::PreparedChatPrompt::Media(input);
        let error = foreign_model
            .start_prepared_chat(foreign_execution, &cancel)
            .err()
            .expect("identical artifacts cannot substitute execution identity");
        assert_eq!(
            error.input_rejection(),
            Some(TokenInputRejection::IdentityMismatch)
        );
        drop((error, foreign_model));
        let input = model
            .prepare_chat_input(&chat, &parts, &cancel)
            .unwrap_or_else(fail)
            .unwrap();
        let foreign_chat = model
            .prepare_chat(&source, &policy, &native_limits(capacity), &cancel)
            .unwrap_or_else(fail)
            .unwrap();
        let mut foreign = PreparedChatRequest::new(&foreign_chat, settings.clone());
        foreign.input = eredu::api::PreparedChatPrompt::Media(input);
        let error = model
            .start_prepared_chat(foreign, &cancel)
            .err()
            .expect("equal text cannot substitute render identity");
        assert_eq!(
            error.input_rejection(),
            Some(TokenInputRejection::IdentityMismatch)
        );
        drop((error, foreign_chat));
        let mut baseline: Option<(Vec<u32>, eredu_core::FinishReason, serde_json::Value)> = None;
        for manual in [false, true] {
            if manual {
                model
                    .prepare_reset_ordinary()
                    .unwrap_or_else(fail)
                    .reset_admitted(eredu_core::SessionResetLimits::new(native_limits(capacity)))
                    .unwrap_or_else(fail);
                model.synchronize().unwrap_or_else(fail);
            }
            let input = model
                .prepare_chat_input(&chat, &parts, &cancel)
                .unwrap_or_else(|error| {
                    panic!("prepare image for {choice:?}, manual={manual}: {error:?}")
                })
                .unwrap();
            let mut request = PreparedChatRequest::new(&chat, settings.clone());
            request.input = eredu::api::PreparedChatPrompt::Media(input);
            let mut session = model
                .start_prepared_chat(request, &cancel)
                .unwrap_or_else(fail)
                .unwrap();
            if snapshots {
                let report = session
                    .preparation_report()
                    .expect("original media admission");
                eprintln!(
                    "MEDIA_ADMISSION choice={choice:?} manual={manual} required={:?} limits={:?} geometry={:?}",
                    report.admission.incremental_required_bytes,
                    report.admission.memory_limits,
                    report.geometry,
                );
            }
            let retained_attribution = session
                .prompt_attribution()
                .expect("original media attribution")
                .clone();
            let attribution = retained_attribution.attribution();
            assert_eq!(attribution.segments.len(), 3);
            assert_eq!(attribution.decoder_positions, decoder_positions);
            assert_eq!(attribution.opening_position, 0);
            assert_eq!(
                attribution.segments[1].plan.decoder_range,
                [marker as u64, marker as u64 + 4]
            );
            assert!(matches!(
                attribution.segments[1].tokens,
                eredu_core::PromptTokenAttribution::NotTokenized
            ));
            assert!(attribution.complete_token_ids().is_none());
            assert_eq!(
                attribution.canonical_token_ids.len(),
                leading.len() + trailing.len()
            );
            assert_eq!(&attribution.canonical_token_ids[..leading.len()], leading);
            assert_eq!(&attribution.canonical_token_ids[leading.len()..], trailing);
            let mut events = Vec::new();
            let output = if manual && snapshots {
                let expected = baseline.as_ref().expect("ordinary media baseline");
                let budget = SnapshotBudget::new(SnapshotLimits {
                    max_snapshots: 2,
                    max_branches: 1,
                    retained_bytes: capacity,
                    cumulative_copy_bytes: capacity * 8,
                });
                // Copy genuine pending media before encoder/decoder work. A
                // fork receives its own admitted state and exact source binding.
                assert!(session.token_ids().is_empty());
                assert_eq!(session.next_prediction(), 0);
                let prepared = session
                    .snapshot(
                        &budget,
                        native_limits(capacity),
                        WorkspaceCopyLimits::new(native_limits(capacity)),
                    )
                    .unwrap_or_else(|error| panic!("{choice:?} pending media snapshot: {error:?}"));
                assert!(prepared.token_ids().is_empty());
                assert_eq!(prepared.next_prediction(), 0);
                let copied = budget.usage().cumulative_copy_bytes;
                assert!(copied > 0);
                eprintln!(
                    "MEDIA_PENDING_SNAPSHOT choice={choice:?} retained={} cumulative_copy={copied}",
                    prepared.retained_bytes(),
                );
                let mut branch = session
                    .fork_snapshot(
                        &prepared,
                        PreparedChatResumeSettings::default(),
                        native_limits(capacity),
                        &cancel,
                    )
                    .unwrap_or_else(|error| panic!("{choice:?} pending media fork: {error:?}"))
                    .expect("live pending media branch");
                assert!(session.token_ids().is_empty());
                assert_eq!(session.next_prediction(), 0);
                assert!(budget.usage().cumulative_copy_bytes > copied);
                session.exchange(&mut branch).unwrap_or_else(|error| {
                    panic!("{choice:?} pending media branch installation: {error:?}")
                });
                assert_eq!(
                    session.prompt_attribution().unwrap().attribution(),
                    attribution
                );
                let mut branch_events = Vec::new();
                while session.finish_reason().is_none() {
                    session = session
                        .advance(&cancel, &mut |event| branch_events.push(event))
                        .unwrap_or_else(|error| {
                            panic!("{choice:?} pending media branch advance: {error:?}")
                        });
                }
                assert_eq!(session.token_ids(), expected.0);
                assert_eq!(session.finish_reason(), Some(expected.1));
                assert_eq!(serde_json::to_value(&branch_events).unwrap(), expected.2);
                assert_eq!(
                    session.prompt_attribution().unwrap().attribution(),
                    attribution
                );
                session.exchange(&mut branch).unwrap_or_else(|error| {
                    panic!("{choice:?} return to pending media parent: {error:?}")
                });
                assert!(session.token_ids().is_empty());
                assert_eq!(session.next_prediction(), 0);
                assert!(session.finish_reason().is_none());
                assert_eq!(
                    session.prompt_attribution().unwrap().attribution(),
                    attribution
                );
                assert_eq!(branch.token_ids(), expected.0);
                let spent = budget.usage().cumulative_copy_bytes;
                drop((branch, branch_events));
                assert_eq!(budget.usage().cumulative_copy_bytes, spent);

                // Three script tokens leave the JSON argument object and tool
                // event unfinished; restoration must preserve both parser states.
                while session.token_ids().len() < 3 {
                    session = session
                        .advance(&cancel, &mut |event| events.push(event))
                        .unwrap_or_else(|error| {
                            panic!("{choice:?} media partial tool advance: {error:?}")
                        });
                }
                assert_eq!(session.token_ids(), &script[..3]);
                assert_eq!(session.next_prediction(), 3);
                assert!(session.finish_reason().is_none());
                assert!(!events.iter().any(|event| matches!(
                    event,
                    SemanticEvent::ToolCallEnd | SemanticEvent::Finished { .. }
                )));
                let prefix = events.clone();
                let partial = session
                    .snapshot(
                        &budget,
                        native_limits(capacity),
                        WorkspaceCopyLimits::new(native_limits(capacity)),
                    )
                    .unwrap_or_else(|error| {
                        panic!("{choice:?} partial media tool snapshot: {error:?}")
                    });
                assert_eq!(partial.token_ids(), &script[..3]);
                assert_eq!(partial.next_prediction(), 3);
                assert_eq!(partial.remaining_tokens(), Some(5));
                while session.finish_reason().is_none() {
                    session = session
                        .advance(&cancel, &mut |event| events.push(event))
                        .unwrap_or_else(|error| {
                            panic!("{choice:?} media tool completion: {error:?}")
                        });
                }
                assert_eq!(session.token_ids(), expected.0);
                assert_eq!(session.finish_reason(), Some(expected.1));
                assert_eq!(serde_json::to_value(&events).unwrap(), expected.2);
                let suffix = serde_json::to_value(&events[prefix.len()..]).unwrap();
                let copied = budget.usage().cumulative_copy_bytes;
                assert!(session
                    .restore_snapshot(
                        &partial,
                        PreparedChatResumeSettings::default(),
                        native_limits(capacity),
                        &cancel
                    )
                    .unwrap_or_else(|error| panic!(
                        "{choice:?} partial media tool restore: {error:?}"
                    )));
                assert!(budget.usage().cumulative_copy_bytes > copied);
                assert_eq!(session.token_ids(), &script[..3]);
                assert_eq!(session.next_prediction(), 3);
                assert!(session.finish_reason().is_none());
                assert_eq!(
                    session.prompt_attribution().unwrap().attribution(),
                    attribution
                );
                let prefix_len = prefix.len();
                events = prefix;
                while session.finish_reason().is_none() {
                    session = session
                        .advance(&cancel, &mut |event| events.push(event))
                        .unwrap_or_else(|error| {
                            panic!("{choice:?} restored media tool completion: {error:?}")
                        });
                }
                assert_eq!(serde_json::to_value(&events[prefix_len..]).unwrap(), suffix);
                assert_eq!(session.token_ids(), expected.0);
                assert_eq!(session.finish_reason(), Some(expected.1));
                assert_eq!(
                    session.prompt_attribution().unwrap().attribution(),
                    attribution
                );
                let spent = budget.usage().cumulative_copy_bytes;
                drop((prepared, partial));
                assert_eq!(budget.usage().cumulative_copy_bytes, spent);
                session
                    .into_output()
                    .unwrap_or_else(|_| panic!("terminal session"))
            } else if manual {
                while session.finish_reason().is_none() {
                    session = session
                        .advance(&cancel, &mut |event| events.push(event))
                        .unwrap_or_else(fail);
                }
                session
                    .into_output()
                    .unwrap_or_else(|_| panic!("terminal manual media session"))
            } else {
                session
                    .run(&cancel, &mut |event| events.push(event))
                    .unwrap_or_else(fail)
            };
            assert!(output.token_ids.starts_with(&script[..5]));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
                    .count(),
                1
            );
            let result = (
                output.token_ids.to_vec(),
                output.finish_reason,
                serde_json::to_value(&events).unwrap(),
            );
            if let Some(expected) = &baseline {
                assert_eq!(&result, expected);
            } else {
                baseline = Some(result);
            }
        }
        if snapshots {
            continue;
        }
        // Recompile media semantics for each independently owned decoder state.
        // Both roles consume the same authenticated host/native source; the
        // image crosses an uneven bounded prefill rather than becoming IDs.
        model
            .prepare_reset_ordinary()
            .unwrap_or_else(fail)
            .reset_admitted(eredu_core::SessionResetLimits::new(native_limits(capacity)))
            .unwrap_or_else(fail);
        model.synchronize().unwrap_or_else(fail);
        let input = model
            .prepare_chat_input(&chat, &parts, &cancel)
            .unwrap_or_else(fail)
            .unwrap();
        let mut events = Vec::new();
        let output = model
            .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                chat: &chat,
                input: PreparedChatPrompt::Media(input),
                drafting: drafting.as_speculative_draft().unwrap(),
                settings,
                output_mode: PreparedChatOutputMode::Semantic,
                skip_special_tokens: true,
                options: speculative_options,
                caller_stop_sequences: &[],
                cancellation: cancel.clone(),
                on_event: |event| events.push(event),
            })
            .unwrap_or_else(fail);
        assert!(output.stats().rounds() > 0);
        let expected = baseline.as_ref().unwrap();
        assert_eq!(output.token_ids(), expected.0);
        assert_eq!(output.finish_reason(), expected.1);
        assert_eq!(serde_json::to_value(&events).unwrap(), expected.2);
    }
}

// Shared source geometry for single-device and distributed public callers.
struct ImagePrompt<'a> {
    leading: &'a [u32],
    trailing: &'a [u32],
    shapes: [[usize; 2]; 2],
    pixels: [f32; 192],
    metadata: [(InputMetadataKey, HostTensorView<'static>); 1],
}
impl<'a> ImagePrompt<'a> {
    fn new(leading: &'a [u32], trailing: &'a [u32]) -> Self {
        Self {
            leading,
            trailing,
            shapes: [[1, leading.len()], [1, trailing.len()]],
            pixels: std::array::from_fn(|i| (i as f32 - 93.0) / 193.0),
            metadata: [(
                InputMetadataKey::PatchGrid,
                HostTensorView {
                    shape: &[1, 3],
                    values: HostTensorValues::I32(&[1, 4, 4]),
                },
            )],
        }
    }
    fn parts(&self) -> [HostInputPart<'_>; 3] {
        [
            HostInputPart {
                modality: InputModality::Text,
                kind: InputPayloadKind::TokenIds,
                payload: HostTensorView {
                    shape: &self.shapes[0],
                    values: HostTensorValues::U32(self.leading),
                },
                metadata: &[],
                extents: &[],
            },
            HostInputPart {
                modality: InputModality::Image,
                kind: InputPayloadKind::Tensor,
                payload: HostTensorView {
                    shape: &[16, 12],
                    values: HostTensorValues::F32(&self.pixels),
                },
                metadata: &self.metadata,
                extents: &[InputExtent::PatchGrid {
                    time: 1,
                    height: 4,
                    width: 4,
                }],
            },
            HostInputPart {
                modality: InputModality::Text,
                kind: InputPayloadKind::TokenIds,
                payload: HostTensorView {
                    shape: &self.shapes[1],
                    values: HostTensorValues::U32(self.trailing),
                },
                metadata: &[],
                extents: &[],
            },
        ]
    }
}

fn image_policy(choice: ToolChoice) -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![
            serde_json::json!({"role":"user","content":"Read value 17. <|vision_start|><|image_pad|><|vision_end|>"}),
        ],
        tools: vec![
            serde_json::json!({"type":"function","function":{"name":"reading",
                "parameters":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object",
                "properties":{"value":{"enum":[17]}},"required":["value"],"additionalProperties":false}}}),
        ],
        tool_choice: choice,
        add_generation_prompt: true,
        ..Default::default()
    }
}

#[path = "media/distributed.rs"]
mod distributed;
