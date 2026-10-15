//! Mode control and delivery over finite completed blocks and one causal frontier.
//! Numerical draft/verification equations are tested by architecture fixtures.

use super::*;
use eredu::api::{ObservedGenerationEvent, TraceLimits};
use eredu_core::{
    decoding::*, refinement::*, PendingTextInput, TextOperationKind, TextOperationOutput,
    TextProgress,
};
use std::ops::ControlFlow;

#[derive(Default)]
pub(super) struct ModeAudit {
    records: Vec<(u64, u64)>,
    fail: bool,
    cancel: Option<eredu_core::execution_control::GenerationControlHandle>,
}

pub(super) fn operation_kind(state: &observed_mock::State) -> TextOperationKind {
    let sampling = &state.sampling;
    if !sampling.mode_context_ready {
        return TextOperationKind::Progress(GenerationPhase::Context);
    }
    if matches!(sampling.mode, DecodingMode::Autoregressive) {
        return TextOperationKind::Decision(PredictionScoreKind::CausalConditional);
    }
    if !sampling.mode_candidates.is_empty() {
        return TextOperationKind::Decision(
            if matches!(sampling.mode, DecodingMode::Diffusion { .. }) {
                PredictionScoreKind::RefinementPosition
            } else {
                PredictionScoreKind::CausalConditional
            },
        );
    }
    TextOperationKind::Progress(if sampling.mode_suffix.is_empty() {
        GenerationPhase::Refinement
    } else {
        GenerationPhase::Context
    })
}

fn append_suffix(
    runtime: &mut ModelRuntime<MockBackend>,
    state: &mut observed_mock::State,
) -> Result<(), MockError> {
    if !state.sampling.mode_suffix.is_empty() {
        // Only authoritative tokens enter the retained causal cache.
        runtime
            .prefill(state.sampling.mode_suffix.clone())?
            .completion
            .wait()?;
        state.sampling.mode_suffix.clear();
    }
    Ok(())
}

pub(super) fn settle(
    runtime: &mut ModelRuntime<MockBackend>,
    input: &mut Option<PendingTextInput<Vec<u32>, MockToken>>,
    state: &mut observed_mock::State,
    requested: DecodingMode,
) -> Result<bool, eredu_core::TextOperationError<MockError>> {
    if !runtime.session().switchable {
        return Err(eredu_core::TextOperationPolicyError::DecodingMode(
            DecodingModeError::Unsupported("fixture has no mode handoff"),
        )
        .into());
    }
    if !state.sampling.mode_candidates.is_empty() {
        return Ok(false);
    }
    if runtime.session().mode_settlements.lock().unwrap().fail {
        return Err(eredu_core::TextOperationError::Backend(MockError::Capture(
            "settlement failed".into(),
        )));
    }
    append_suffix(runtime, state).map_err(eredu_core::TextOperationError::Backend)?;
    let cancel = {
        let mut audit = runtime.session().mode_settlements.lock().unwrap();
        audit
            .records
            .push((runtime.session().cache_positions, state.sampling.prediction));
        audit.cancel.clone()
    };
    if let Some(cancel) = cancel {
        cancel.cancel();
    }
    if state.sampling.mode_context_ready {
        *input = Some(PendingTextInput::Continue);
    }
    state.sampling.mode = requested;
    Ok(true)
}

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
    let kind = operation_kind(state);
    let output = match kind {
        TextOperationKind::Progress(GenerationPhase::Context) => {
            assert!(decision.is_none());
            if !state.sampling.mode_context_ready {
                let PendingTextInput::Prefill(prompt) = input else {
                    panic!("initial context requires prompt")
                };
                state.sampling.iterative_first = prompt.len() as u32;
                runtime
                    .prefill(prompt)
                    .map_err(eredu_core::TextOperationError::Backend)?
                    .completion
                    .wait()
                    .map_err(eredu_core::TextOperationError::Backend)?;
                state.sampling.mode_context_ready = true;
            } else {
                append_suffix(runtime, state).map_err(eredu_core::TextOperationError::Backend)?;
            }
            TextOperationOutput::Progress {
                pending: PendingTextInput::Continue,
                progress: TextProgress {
                    phase: GenerationPhase::Context,
                    block: state.sampling.physical,
                    iteration: 0,
                    tentative_token_ids: None,
                },
            }
        }
        TextOperationKind::Progress(GenerationPhase::Refinement) => {
            assert!(decision.is_none());
            // A complete settled diffusion canvas or target-verified transaction.
            let count = if matches!(state.sampling.mode, DecodingMode::Diffusion { .. }) {
                2
            } else {
                3
            };
            state.sampling.mode_candidates = (0..count)
                .map(|i| state.sampling.iterative_first + state.sampling.prediction as u32 + i)
                .collect();
            TextOperationOutput::Progress {
                pending: PendingTextInput::Continue,
                progress: TextProgress {
                    phase: GenerationPhase::Refinement,
                    block: state.sampling.physical,
                    iteration: 1,
                    tentative_token_ids: state
                        .sampling
                        .draft_events
                        .then(|| state.sampling.mode_candidates.iter().copied().collect()),
                },
            }
        }
        TextOperationKind::Decision(_) => {
            if matches!(state.sampling.mode, DecodingMode::Autoregressive) {
                append_suffix(runtime, state).map_err(eredu_core::TextOperationError::Backend)?;
            }
            let candidate = state
                .sampling
                .mode_candidates
                .pop_front()
                .unwrap_or(state.sampling.iterative_first + state.sampling.prediction as u32);
            let token = apply_filter(
                candidate,
                decision
                    .expect("one evolving decision per commitment")
                    .filter(),
            );
            if token != candidate {
                state.sampling.mode_candidates.clear();
            }
            state.sampling.mode_suffix.push(token);
            state.sampling.prediction += 1;
            TextOperationOutput::Token(MockToken(token))
        }
        _ => unreachable!(),
    };
    state.sampling.physical += 1;
    let lease = runtime
        .session_mut()
        .authority
        .begin_submission()
        .map_err(MockError::from)
        .map_err(eredu_core::TextOperationError::Backend)?;
    Ok(Submission {
        output,
        completion: MockSessionCompletion(lease),
    })
}

type SettlementAudit = std::sync::Arc<std::sync::Mutex<ModeAudit>>;
fn setup(
    mode: DecodingMode,
    max_tokens: usize,
) -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
    u32,
    SettlementAudit,
) {
    let mut runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    runtime.session_mut().switchable = true;
    let audit = runtime.session().mode_settlements.clone();
    let mut model = unicode_model_with_runtime(None, 64, QWEN_TEMPLATE, runtime);
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let first = model.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    (
        model,
        chat,
        PreparedChatGenerationSettings {
            decoding_mode: Some(mode),
            draft_events: true,
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(max_tokens),
                ..Default::default()
            },
            ..Default::default()
        },
        first,
        audit,
    )
}

fn limits() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1 << 20,
    }
}
fn modes() -> [DecodingMode; 3] {
    [
        DecodingMode::Autoregressive,
        DecodingMode::Diffusion { policy: None },
        DecodingMode::SelfSpeculation(Default::default()),
    ]
}

#[test]
fn controlled_mode_forecast_observes_pending_handoff_without_advancement() {
    let (mut model, chat, settings, _, _) = setup(DecodingMode::Autoregressive, 10);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut run = model
        .start_controlled(
            prepared,
            &[],
            Default::default(),
            |_| ControlFlow::Continue(()),
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    while run.token_ids().is_empty() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    run.request_decoding_mode(DecodingMode::Diffusion { policy: None })
        .unwrap();
    let before = (
        run.token_ids().to_vec(),
        run.decoding_state(),
        run.next_prediction(),
    );
    let forecast = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    assert_eq!(forecast.decoding.as_ref().unwrap().state, before.1);
    // Preserve the selected sampler's larger input horizon as well as block append.
    assert_eq!(forecast.continuation.additional_input_tokens, 6);
    let zero = run
        .forecast_remaining_generation(0, &Default::default())
        .unwrap();
    assert_eq!(zero.decoding.as_ref().unwrap().settlement_input_tokens, 1);
    assert_eq!(zero.continuation.additional_input_tokens, 1);
    assert!(zero.estimate.domains[0]
        .phases
        .iter()
        .any(
            |phase| phase.phase == eredu::api::MemoryPhase::SpeculativeVerification
                && phase.query_positions == 1
        ));
    let phases = &forecast.estimate.domains[0].phases;
    assert_eq!(
        phases
            .iter()
            .find(|p| p.phase == eredu::api::MemoryPhase::Refinement)
            .unwrap()
            .query_positions,
        3
    );
    assert_eq!(
        phases
            .iter()
            .find(|p| p.phase == eredu::api::MemoryPhase::RefinementCommit)
            .unwrap()
            .query_positions,
        3
    );
    assert!(phases.iter().all(|p| !matches!(
        p.phase,
        eredu::api::MemoryPhase::Prefill | eredu::api::MemoryPhase::Loading
    )));
    assert_eq!(
        (
            run.token_ids().to_vec(),
            run.decoding_state(),
            run.next_prediction()
        ),
        before
    );
    while run.decoding_state().pending.is_some() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    let options = SpeculationOptionsRequest {
        max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
        ..Default::default()
    }
    .try_into()
    .unwrap();
    run.request_decoding_mode(DecodingMode::SelfSpeculation(options))
        .unwrap();
    let forecast = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    assert!(forecast.estimate.domains[0]
        .phases
        .iter()
        .any(|p| p.phase == eredu::api::MemoryPhase::Refinement));
    assert!(forecast.estimate.domains[0]
        .phases
        .iter()
        .any(|p| p.phase == eredu::api::MemoryPhase::SpeculativeVerification));
    assert!(forecast.estimate.domains[0]
        .phases
        .iter()
        .all(|p| p.parameters.lower_bytes == 4096));
}

#[test]
fn six_directed_handoffs_settle_committed_suffix_once_and_clear_tentative_output() {
    for previous in modes() {
        for next in modes() {
            if previous == next {
                continue;
            }
            let (mut model, chat, mut settings, first, audit) = setup(previous, 10);
            settings.overrides.temperature = Some(0.8);
            settings.overrides.repetition_penalty = Some(1.1);
            let prepared = model
                .prepare_observed_chat(
                    &chat,
                    settings,
                    eredu_core::capture::CapturePlan::none(),
                    limits(),
                )
                .unwrap();
            let mut run = model
                .start_controlled(
                    prepared,
                    &[],
                    Default::default(),
                    |_| ControlFlow::Continue(()),
                    eredu::api::OutputMode::Text,
                )
                .unwrap();
            while run.token_ids().is_empty() {
                run.step(|_| ControlFlow::Continue(())).unwrap();
            }
            let handle = run.control_handle();
            std::thread::scope(|scope| {
                scope
                    .spawn(|| handle.request_decoding_mode(next).unwrap())
                    .join()
                    .unwrap()
            });
            assert_eq!(run.decoding_state().pending, Some(next));
            assert_eq!(
                handle.request_decoding_mode(previous),
                Err(DecodingModeError::ModeChangePending)
            );
            let mut events = Vec::new();
            while run.decoding_state().pending.is_some() {
                run.step(|record| {
                    events.push(record.generation.event);
                    ControlFlow::Continue(())
                })
                .unwrap();
            }
            let position = run
                .decoding_state()
                .acknowledgement
                .unwrap()
                .committed_position;
            assert_eq!(
                position,
                if previous == modes()[0] {
                    1
                } else if previous == modes()[1] {
                    2
                } else {
                    3
                }
            );
            assert_eq!(
                audit.lock().unwrap().records.last(),
                Some(&(u64::from(first) + position, position))
            );
            assert!(matches!(
                events.last(),
                Some(ObservedGenerationEvent::Lifecycle { .. })
            ));
            assert!(events.iter().any(|event| matches!(event, ObservedGenerationEvent::ModeChanged { acknowledgement, .. } if acknowledgement.committed_position == position)));
            run.run(|_| ControlFlow::Continue(())).unwrap();
            assert_eq!(run.token_ids(), &(first..first + 10).collect::<Vec<_>>());
            assert_eq!(
                handle.request_decoding_mode(previous),
                Err(DecodingModeError::Terminal)
            );
        }
    }
}

#[test]
fn mode_snapshot_restores_pending_block_and_forks_independent_commands_without_refunding_copy_usage(
) {
    use eredu::api::GenerationBranchOptions;
    use eredu_core::execution_control::SnapshotLimits;
    let (mut model, chat, settings, first, audit) = setup(modes()[2], 10);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut run = model
        .start_controlled(
            prepared,
            &[],
            Default::default(),
            |_| ControlFlow::Continue(()),
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 1,
        retained_bytes: 4_000_000,
        cumulative_copy_bytes: 32_000_000,
    })
    .unwrap();
    while run.token_ids().is_empty() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    run.request_decoding_mode(modes()[1]).unwrap();
    let saved = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(saved.metadata().decoding.active, modes()[2]);
    assert_eq!(saved.metadata().decoding.pending, Some(modes()[1]));
    let mut child = run
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: limits(),
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |_| ControlFlow::Continue(()),
        )
        .unwrap();
    let child_handle = child.control_handle();
    let parent_handle = run.control_handle();
    while run.decoding_state().pending.is_some() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    assert_eq!(parent_handle.decoding_state().unwrap().active, modes()[1]);
    assert_eq!(
        child_handle.decoding_state().unwrap().pending,
        Some(modes()[1])
    );
    let spent = run.snapshot_usage().unwrap().cumulative_copy_bytes;
    run.restore(&saved, |_| ControlFlow::Continue(())).unwrap();
    assert_eq!(run.token_ids(), &[first]);
    assert_eq!(run.decoding_state().pending, Some(modes()[1]));
    assert!(run.snapshot_usage().unwrap().cumulative_copy_bytes > spent);
    while run.decoding_state().pending.is_some() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    assert_eq!(
        audit.lock().unwrap().records.last(),
        Some(&(u64::from(first) + 3, 3))
    );
    run.exchange(&mut child, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(
        run.control_handle().decoding_state().unwrap(),
        child_handle.decoding_state().unwrap()
    );
    while run.decoding_state().pending.is_some() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    child_handle.request_decoding_mode(modes()[0]).unwrap();
    assert_eq!(parent_handle.decoding_state().unwrap().pending, None);
    run.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(run.token_ids(), &(first..first + 10).collect::<Vec<_>>());
}

#[test]
fn cancellation_during_settlement_and_failed_settlement_never_acknowledge_or_commit_another_token()
{
    for failure in [false, true] {
        let (mut model, chat, settings, first, audit) = setup(modes()[0], 10);
        let prepared = model
            .prepare_observed_chat(
                &chat,
                settings,
                eredu_core::capture::CapturePlan::none(),
                limits(),
            )
            .unwrap();
        let mut run = model
            .start_controlled(
                prepared,
                &[],
                Default::default(),
                |_| ControlFlow::Continue(()),
                eredu::api::OutputMode::Text,
            )
            .unwrap();
        while run.token_ids().is_empty() {
            run.step(|_| ControlFlow::Continue(())).unwrap();
        }
        let handle = run.control_handle();
        {
            let mut audit = audit.lock().unwrap();
            audit.fail = failure;
            if !failure {
                audit.cancel = Some(handle.clone());
            }
        }
        handle.request_decoding_mode(modes()[1]).unwrap();
        let mut events = Vec::new();
        let result = run.step(|record| {
            events.push(record.generation.event);
            ControlFlow::Continue(())
        });
        assert_eq!(result.is_err(), failure);
        assert_eq!(run.token_ids(), &[first]);
        assert!(run.decoding_state().acknowledgement.is_none());
        assert!(run.decoding_state().pending.is_none());
        assert!(!events
            .iter()
            .any(|event| matches!(event, ObservedGenerationEvent::ModeChanged { .. })));
        assert_eq!(
            handle.request_decoding_mode(modes()[0]),
            Err(DecodingModeError::Terminal)
        );
        if !failure {
            assert_eq!(run.finish_reason(), Some(FinishReason::Cancelled));
        }
    }
}

#[test]
fn uninterrupted_and_controlled_advancement_follow_the_same_repeated_command_schedule() {
    let (mut model, chat, settings, first, _) = setup(modes()[0], 10);
    let control = eredu_core::execution_control::GenerationControlHandle::default();
    let mut committed = 0;
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits(),
        )
        .unwrap()
        .with_control_handle(control.clone());
    let output = model
        .generate_observed(
            prepared,
            &[],
            Default::default(),
            |record| {
                if matches!(record.event, ObservedGenerationEvent::Token { .. }) {
                    committed += 1;
                    if committed <= 7 && control.decoding_state().unwrap().pending.is_none() {
                        control
                            .request_decoding_mode(modes()[(committed as usize) % 3])
                            .unwrap();
                    }
                }
                ControlFlow::Continue(())
            },
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    let (mut model, chat, settings, _, _) = setup(modes()[0], 10);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut run = model
        .start_controlled(
            prepared,
            &[],
            Default::default(),
            |_| ControlFlow::Continue(()),
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    let handle = run.control_handle();
    let mut committed = 0;
    run.run(|record| {
        if matches!(
            record.generation.event,
            ObservedGenerationEvent::Token { .. }
        ) {
            committed += 1;
            if committed <= 7 && handle.decoding_state().unwrap().pending.is_none() {
                handle
                    .request_decoding_mode(modes()[(committed as usize) % 3])
                    .unwrap();
            }
        }
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(output.token_ids, run.token_ids());
    assert_eq!(run.token_ids(), &(first..first + 10).collect::<Vec<_>>());
}

#[test]
fn mirostat_rejects_position_score_modes_before_enqueuing_a_change() {
    let (mut model, chat, mut settings, first, audit) = setup(DecodingMode::Autoregressive, 4);
    settings.strategy = eredu_core::TextSamplingStrategy::MirostatV2 { tau: 5.0, eta: 0.1 };
    settings.overrides.temperature = Some(0.8);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut run = model
        .start_controlled(
            prepared,
            &[],
            Default::default(),
            |_| ControlFlow::Continue(()),
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    let before = run.decoding_state();
    assert!(matches!(
        run.control_handle()
            .request_decoding_mode(DecodingMode::Diffusion { policy: None }),
        Err(DecodingModeError::RequiresCausalConditionalScores)
    ));
    assert!(matches!(
        run.request_decoding_mode(DecodingMode::Diffusion { policy: None }),
        Err(eredu::api::ControlledGenerationError::DecodingMode(
            DecodingModeError::RequiresCausalConditionalScores
        ))
    ));
    assert_eq!(run.decoding_state(), before);
    while run.token_ids().is_empty() {
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    assert_eq!(run.token_ids(), &[first]);
    assert!(audit.lock().unwrap().records.is_empty());
}

#[test]
fn unsupported_mode_options_leave_execution_and_pending_command_unchanged() {
    let (mut model, chat, settings, first, _) = setup(modes()[0], 4);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits(),
        )
        .unwrap();
    let mut run = model
        .start_controlled(
            prepared,
            &[],
            Default::default(),
            |_| ControlFlow::Continue(()),
            eredu::api::OutputMode::Text,
        )
        .unwrap();
    let before = run.decoding_state();
    let adapter: SpeculationOptions = SpeculationOptionsRequest {
        draft_adapter: true,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    for mode in [
        DecodingMode::Diffusion {
            policy: Some(RefinementPolicy::MaskedUnmasking {
                config: Default::default(),
                mask_token_id: 3,
            }),
        },
        DecodingMode::SelfSpeculation(adapter),
    ] {
        assert!(matches!(
            run.request_decoding_mode(mode),
            Err(eredu::api::ControlledGenerationError::DecodingMode(
                DecodingModeError::Unsupported(_)
            ))
        ));
        assert_eq!(run.decoding_state(), before);
    }
    run.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(run.token_ids(), &(first..first + 4).collect::<Vec<_>>());
}
