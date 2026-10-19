use super::*;
use eredu::api::{
    GenerationControlOptions, GenerationOutput, GenerationStep, OutputMode, PlannedModel,
};
use eredu_core::decoding::{DecodingMode, DecodingModeError, SpeculationOptionsRequest};

fn load(
    source: usize,
) -> (
    TestDirectory,
    TestDirectory,
    PlannedModel<MockBackend, RealizedDrafting<MockDrafter>>,
) {
    let target = TestDirectory::new();
    let assistant = TestDirectory::new();
    write_loadable_text_artifact(target.path());
    write_loadable_assistant_artifact(assistant.path());
    let tokenizer = unicode_model(None).tokenizer().to_string(false).unwrap();
    for path in [target.path(), assistant.path()] {
        std::fs::write(path.join("tokenizer.json"), &tokenizer).unwrap();
    }
    let drafting = match source {
        0 => DraftingPlan::Disabled,
        1 => DraftingPlan::Embedded {
            max_draft_tokens: 2,
            lookahead: false,
            adaptive_lookahead: false,
        },
        2 => DraftingPlan::External {
            residency: eredu_core::ResidencyPlan::FullyResident,
            model: assistant.path().display().to_string(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 2,
            lookahead: false,
            adaptive_lookahead: false,
        },
        _ => unreachable!(),
    };
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mock", "gpu:0").unwrap())
        .with_drafting(drafting);
    let session = LoadedModel::load_execution_plan_with_text_options(
        &MockBackend,
        target.path(),
        &plan,
        eredu::api::TextModelOptions {
            chat_template: Some(QWEN_TEMPLATE.into()),
        },
    )
    .unwrap();
    (target, assistant, session)
}

fn run(
    source: usize,
    mode: OutputMode,
    controlled: bool,
    cancel: Option<bool>,
) -> (GenerationOutput, Vec<SemanticEvent>) {
    let (_target, _assistant, mut session) = load(source);
    let chat = session
        .model_mut()
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let cancellation = eredu_core::GenerationCancellationToken::new();
    if cancel == Some(true) {
        cancellation.cancel();
    }
    let cancel_on_event = cancellation.clone();
    let mut events = Vec::new();
    let control = eredu_core::execution_control::GenerationControlHandle::default();
    let request = GenerationRequest {
        input: PreparedChatInput::token_ids(&chat, vec![3, 4]),
        output_mode: mode,
        settings: PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(6),
                ..Default::default()
            },
            ..Default::default()
        },
        caller_stop_sequences: &[],
        cancellation,
        control_handle: Some(&control),
        on_event: |event| {
            if cancel == Some(false) && matches!(event, SemanticEvent::TextDelta(_)) {
                cancel_on_event.cancel();
            }
            events.push(event);
        },
    };
    let forecast = session
        .forecast_generation(&request, &Default::default())
        .unwrap();
    if source == 0 {
        assert!(forecast.speculative.is_none());
    } else {
        let plan = forecast.speculative.unwrap();
        assert_eq!(plan.max_draft_tokens, 2);
        assert_eq!(plan.scheduler.lookahead_blocks, 0);
        assert!(!plan.scheduler.adaptive_lookahead);
    }
    let output = if controlled {
        session
            .with_controlled(request, GenerationControlOptions::default(), |control| {
                while let Some(step) = control.step()? {
                    if let GenerationStep::Speculative(step) = step {
                        if let Some(draft) = step.drafted {
                            assert!(draft.token_ids.len() <= 2);
                        }
                    }
                }
                Ok(())
            })
            .unwrap_or_else(|error| {
                panic!("source={source}, mode={mode:?}, cancel={cancel:?}: {error}")
            })
    } else {
        session.generate(request).unwrap()
    };
    (output, events)
}

#[test]
fn planned_generation_shares_defaults_events_and_results_with_explicit_advancement() {
    for source in 0..3 {
        for mode in [OutputMode::Semantic, OutputMode::Text] {
            for cancel in [None, Some(false), Some(true)] {
                let (ordinary, events) = run(source, mode, false, cancel);
                let (controlled, controlled_events) = run(source, mode, true, cancel);
                assert_eq!(ordinary.token_ids, controlled.token_ids);
                assert_eq!(ordinary.finish_reason, controlled.finish_reason);
                assert_eq!(events, controlled_events);
                assert_eq!(
                    ordinary.timing().time_to_first_token().is_some(),
                    cancel != Some(true)
                );
                assert_eq!(
                    controlled.timing().time_to_first_token().is_some(),
                    cancel != Some(true)
                );
                assert_eq!(ordinary.stats().drafting.is_some(), source != 0);
                assert_eq!(controlled.stats().drafting.is_some(), source != 0);
                if let (Some(a), Some(b)) =
                    (&ordinary.stats().drafting, &controlled.stats().drafting)
                {
                    assert_eq!(a.draft_tokens(), b.draft_tokens());
                    assert_eq!(a.accepted_tokens(), b.accepted_tokens());
                    assert_eq!(a.rounds(), b.rounds());
                    if cancel.is_none() {
                        assert!(a.draft_tokens() > 0);
                    }
                }
                if cancel.is_some() {
                    assert_eq!(ordinary.finish_reason, FinishReason::Cancelled);
                } else {
                    assert_eq!(ordinary.token_ids.len(), 6);
                }
            }
        }
    }
}

#[test]
fn planned_generation_rejects_larger_proposals_and_lookahead_before_execution() {
    for source in 0..3 {
        for larger_proposals in [false, true] {
            let (_target, _assistant, mut session) = load(source);
            let chat = session
                .model_mut()
                .prepare_chat(ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
                    ..Default::default()
                })
                .unwrap();
            let mut options = SpeculationOptionsRequest::default();
            options.max_draft_tokens =
                NonZeroUsize::new(if larger_proposals { 3 } else { 2 }).unwrap();
            options.scheduler = options.scheduler.with_lookahead(!larger_proposals);
            let request = GenerationRequest {
                input: PreparedChatInput::token_ids(&chat, vec![3, 4]),
                output_mode: OutputMode::Text,
                settings: PreparedChatGenerationSettings {
                    decoding_mode: Some(DecodingMode::SelfSpeculation(options.try_into().unwrap())),
                    ..Default::default()
                },
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                control_handle: None,
                on_event: |_| panic!("invalid policy must fail before publication"),
            };
            assert!(matches!(
                session.forecast_generation(&request, &Default::default()),
                Err(eredu::api::GenerationForecastError::Generation(
                    eredu_core::GenerationError::DecodingMode(DecodingModeError::Unsupported(_))
                ))
            ));
            assert!(matches!(
                session.generate(request),
                Err(PreparedChatError::Generation(
                    eredu_core::GenerationError::DecodingMode(DecodingModeError::Unsupported(_))
                ))
            ));
        }
    }
}

#[test]
fn source_policy_survives_initial_ar_mode_restore_and_branch_exchange() {
    use eredu::api::{GenerationControlOptions, SpeculativeControlError};
    for source in [1, 2] {
        let (_target, _assistant, mut session) = load(source);
        let selected = session.drafting_options().unwrap().unwrap();
        let permitted = selected.with_proposal_capacity(NonZeroUsize::new(1).unwrap());
        let too_large = selected.with_proposal_capacity(NonZeroUsize::new(3).unwrap());
        let chat = session
            .model_mut()
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
                ..Default::default()
            })
            .unwrap();
        let request = GenerationRequest {
            input: PreparedChatInput::token_ids(&chat, vec![3, 4]),
            output_mode: OutputMode::Text,
            settings: PreparedChatGenerationSettings {
                decoding_mode: Some(DecodingMode::Autoregressive),
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(8),
                    ..Default::default()
                },
                ..Default::default()
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            control_handle: None,
            on_event: |_| {},
        };
        let forecast = session
            .forecast_generation(&request, &Default::default())
            .unwrap();
        assert_eq!(forecast.speculative.unwrap().max_draft_tokens, 0);
        session
            .with_controlled(
                request,
                GenerationControlOptions {
                    speculative_snapshots: Some(eredu_core::execution_control::SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 1,
                        retained_bytes: 16 << 20,
                        cumulative_copy_bytes: 64 << 20,
                    }),
                    ..Default::default()
                },
                |control| {
                    let control = control.speculative().unwrap();
                    control.step()?;
                    assert!(matches!(
                        control.request_decoding_mode(DecodingMode::SelfSpeculation(too_large)),
                        Err(SpeculativeControlError::DecodingMode(
                            DecodingModeError::Unsupported(_)
                        ))
                    ));
                    control.request_decoding_mode(DecodingMode::SelfSpeculation(permitted))?;
                    while !control.can_snapshot() || control.token_ids().len() == 1 {
                        control.step()?;
                    }
                    let snapshot = control.snapshot()?;
                    control.request_decoding_mode(DecodingMode::Autoregressive)?;
                    control.step()?;
                    control.restore(&snapshot)?;
                    let child = control.fork(&snapshot)?;
                    control.exchange(&child)?;
                    assert!(matches!(
                        control.request_decoding_mode(DecodingMode::SelfSpeculation(too_large)),
                        Err(SpeculativeControlError::DecodingMode(
                            DecodingModeError::Unsupported(_)
                        ))
                    ));
                    while control.step()?.is_some() {}
                    Ok(())
                },
            )
            .unwrap();
    }
}
