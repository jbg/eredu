//! A neutral block producer exercises facade delivery independently of native equations.
use super::*;
use eredu_core::{
    refinement::GenerationPhase, PendingTextInput, TextOperationOutput, TextProgress,
};

#[allow(clippy::type_complexity)]
pub(super) fn submit(
    runtime: &mut ModelRuntime<MockBackend>,
    input: PendingTextInput<Vec<u32>, MockToken>,
    decision: Option<&eredu_core::TokenSamplingDecision<'_>>,
    state: &mut observed_mock::State,
) -> Result<
    Submission<TextOperationOutput<Vec<u32>, MockToken>, MockSessionCompletion>,
    eredu_core::TextOperationError<MockError>,
> {
    let physical = state.sampling.physical;
    let slot = physical % 5;
    let lease = runtime
        .session_mut()
        .authority
        .begin_submission()
        .map_err(MockError::from)
        .map_err(eredu_core::TextOperationError::Backend)?;
    let output = if slot < 3 {
        assert!(
            decision.is_none(),
            "tentative work queried commitment constraints"
        );
        if slot == 0 {
            if let PendingTextInput::Prefill(prompt) = input {
                state.sampling.iterative_first = prompt.len() as u32;
                runtime.session_mut().cache_positions += prompt.len() as u64;
            } else {
                runtime.session_mut().cache_positions += 2;
            }
        }
        TextOperationOutput::Progress {
            pending: PendingTextInput::Continue,
            progress: TextProgress {
                phase: if slot == 0 {
                    GenerationPhase::Context
                } else {
                    GenerationPhase::Refinement
                },
                block: physical / 5,
                iteration: slot as u32,
                tentative_token_ids: (state.sampling.draft_events && slot != 0)
                    .then(|| vec![5 + slot as u32, 9]),
            },
        }
    } else {
        let decision = decision.ok_or(eredu_core::TextOperationPolicyError::MissingDecision)?;
        let token = apply_filter(
            state.sampling.iterative_first + state.sampling.prediction as u32,
            decision.filter(),
        );
        state.sampling.prediction += 1;
        TextOperationOutput::Token(MockToken(token))
    };
    state.sampling.physical += 1;
    Ok(Submission {
        output,
        completion: MockSessionCompletion(lease),
    })
}

fn setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
) {
    let mut probe = unicode_model(None);
    let request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user","content":"hello"})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = probe.prepare_chat(request()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let mut runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    runtime.session_mut().iterative = true;
    let mut model = unicode_model_with_runtime(Some(first), 64, QWEN_TEMPLATE, runtime);
    let chat = model.prepare_chat(request()).unwrap();
    (
        model,
        chat,
        PreparedChatGenerationSettings {
            draft_events: true,
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            ..Default::default()
        },
    )
}

#[test]
fn iterative_drafts_preserve_controlled_and_ordinary_unicode_and_eos_commitment() {
    use eredu::api::{ObservedGenerationEvent, TraceLimits};
    use eredu_core::{capture::CapturePlan, refinement::DraftEvent};
    use std::ops::ControlFlow;
    let (mut model, chat, settings) = setup();
    let limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    };
    let prepare = |model: &LoadedModel<MockBackend>| {
        model
            .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits)
            .unwrap()
    };
    let mut ordinary = Vec::new();
    let output = model
        .generate_observed(
            prepare(&model),
            &[],
            Default::default(),
            |record| {
                ordinary.push(record);
                ControlFlow::Continue(())
            },
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    assert_eq!(output.token_ids.len(), 3);
    assert_eq!(output.finish_reason, FinishReason::Eos);
    model.reset().unwrap();
    let mut controlled = Vec::new();
    let mut run = model
        .start_controlled(
            prepare(&model),
            &[],
            Default::default(),
            |record| {
                controlled.push(record);
                ControlFlow::Continue(())
            },
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    run.run(|record| {
        controlled.push(record);
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(run.token_ids(), output.token_ids);
    let check = |events: Vec<&ObservedGenerationEvent>| {
        let mut active = false;
        let mut revisions = 0;
        let mut tokens = 0;
        for event in events {
            match event {
                ObservedGenerationEvent::Draft {
                    event: DraftEvent::Updated { .. },
                    ..
                } => {
                    active = true;
                    revisions += 1;
                }
                ObservedGenerationEvent::Draft {
                    event: DraftEvent::Cleared { .. },
                    ..
                } => {
                    assert!(active);
                    active = false;
                }
                ObservedGenerationEvent::Token { .. } => {
                    assert!(!active);
                    tokens += 1;
                }
                _ => {}
            }
        }
        assert!(!active);
        assert_eq!(revisions, 4);
        assert_eq!(tokens, 3);
    };
    check(ordinary.iter().map(|record| &record.event).collect());
    check(
        controlled
            .iter()
            .map(|record| &record.generation.event)
            .collect(),
    );
}

#[test]
fn iterative_facade_rejects_explicit_causal_sampling_overrides() {
    let (model, chat, mut settings) = setup();
    settings.overrides.temperature = Some(0.8);
    let result = model.prepare_observed_chat(
        &chat,
        settings,
        eredu_core::capture::CapturePlan::none(),
        eredu::api::TraceLimits {
            per_record_bytes: 65536,
            total_bytes: 1024 * 1024,
        },
    );
    assert!(matches!(
        result,
        Err(PreparedChatError::Generation(
            eredu_core::generation::GenerationError::RequiresCausalConditionalScores
        ))
    ));
}

#[test]
fn iterative_active_draft_snapshot_restore_fork_and_cancel_preserve_commitment() {
    use eredu::api::{GenerationBranchOptions, ObservedGenerationEvent, TraceLimits};
    use eredu_core::{
        capture::CapturePlan, execution_control::SnapshotLimits, refinement::DraftEvent,
    };
    use std::ops::ControlFlow;
    let (mut model, chat, settings) = setup();
    let limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    };
    let request = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits)
        .unwrap();
    let mut records = Vec::new();
    let collect = |record| {
        records.push(record);
        ControlFlow::Continue(())
    };
    let mut run = model
        .start_controlled(
            request,
            &[],
            Default::default(),
            collect,
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 1,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 256_000_000,
    })
    .unwrap();
    let mut collect = |record| {
        records.push(record);
        ControlFlow::Continue(())
    };
    run.step(&mut collect).unwrap(); // Authoritative context.
    run.step(&mut collect).unwrap(); // Tentative revision, no canonical output.
    assert!(run.token_ids().is_empty());
    let saved = run.snapshot(&mut collect).unwrap();
    assert_eq!(saved.metadata().position.committed_tokens, 0);
    assert_eq!(saved.metadata().position.execution_step, 2);
    let mut branch = run
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: limits,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            &mut collect,
        )
        .unwrap();
    run.run(&mut collect).unwrap();
    let expected = run.token_ids().to_vec();
    run.restore(&saved, &mut collect).unwrap();
    run.run(&mut collect).unwrap();
    assert_eq!(run.token_ids(), expected);
    run.exchange(&mut branch, &mut collect).unwrap();
    run.run(&mut collect).unwrap();
    assert_eq!(run.token_ids(), expected);
    run.exchange(&mut branch, &mut collect).unwrap();
    run.restore(&saved, &mut collect).unwrap();
    run.step(&mut collect).unwrap(); // Next tentative revision after restoration.
    let position = run.position();
    run.cancel(&mut collect).unwrap();
    assert!(run.token_ids().is_empty());
    assert_eq!(run.position().committed_tokens, position.committed_tokens);
    let mut active = None;
    let mut identities = std::collections::BTreeSet::new();
    for record in records {
        match record.generation.event {
            ObservedGenerationEvent::Draft {
                event: DraftEvent::Updated { draft },
                ..
            } => {
                active = Some((draft.branch, draft.epoch, draft.block));
                identities.insert((draft.branch, draft.epoch));
            }
            ObservedGenerationEvent::Draft {
                event:
                    DraftEvent::Cleared {
                        branch,
                        epoch,
                        block,
                    },
                ..
            } => {
                assert_eq!(active.take(), Some((branch, epoch, block)));
            }
            ObservedGenerationEvent::Token { .. } => assert!(active.is_none()),
            _ => {}
        }
    }
    assert!(active.is_none());
    assert!(
        identities.len() >= 4,
        "restoration epochs and fork identity are distinct"
    );
}

#[test]
fn iterative_target_rejects_causal_speculative_verification_before_backend_execution() {
    let (mut model, chat, settings) = setup();
    let request = (
        SpeculativeDraft::Embedded,
        GenerationRequest {
            output_mode: eredu::api::OutputMode::Semantic,
            control_handle: None,
            input: PreparedChatInput::prepared_backend_input(&chat, vec![3, 4]),

            settings: eredu::api::PreparedChatGenerationSettings {
                decoding_mode: Some(eredu_core::decoding::DecodingMode::SelfSpeculation(
                    Default::default(),
                )),
                ..settings
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        },
    );
    assert!(matches!(
        {
            let (drafting, mut request) = request;
            request.output_mode = eredu::api::OutputMode::Semantic;
            model.drafting_session(drafting, None).generate(request)
        },
        Err(PreparedChatError::Generation(
            eredu_core::generation::GenerationError::RequiresCausalConditionalScores
        ))
    ));
    let request = (
        SpeculativeDraft::Target,
        GenerationRequest {
            output_mode: eredu::api::OutputMode::Semantic,
            control_handle: None,
            input: PreparedChatInput::prepared_backend_input(&chat, vec![3, 4]),

            settings: PreparedChatGenerationSettings {
                decoding_mode: Some(eredu_core::decoding::DecodingMode::SelfSpeculation(
                    Default::default(),
                )),
                ..settings
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        },
    );
    assert!(matches!(
        {
            let (drafting, mut request) = request;
            request.output_mode = eredu::api::OutputMode::Semantic;
            model.drafting_session(drafting, None).generate(request)
        },
        Err(PreparedChatError::Generation(
            eredu_core::GenerationError::NoBackendDraftCapacity
        ))
    ));
}
