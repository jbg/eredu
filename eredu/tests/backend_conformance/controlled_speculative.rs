use super::*;
use eredu::api::{
    ControlledSpeculativeOptions, SpeculativeControlError,
    SpeculativeProposalDisposition as Disposition,
};
use eredu_core::{
    execution_control::SnapshotLimits, generation::SpeculativeRequestStatus as Status,
};
use std::time::{Duration, Instant};

fn setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
) {
    let mut model = unicode_model(None);
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(6),
            ..Default::default()
        },
        ..Default::default()
    };
    (model, chat, settings)
}
fn options() -> ControlledSpeculativeOptions {
    ControlledSpeculativeOptions {
        snapshots: Some(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 0,
            retained_bytes: 16 << 20,
            cumulative_copy_bytes: 64 << 20,
        }),
        ..Default::default()
    }
}

#[test]
fn controlled_speculation_uses_identical_semantics_and_reports_accepted_and_failed_proposals() {
    let (mut model, chat, settings) = setup();
    let mut expected_events = Vec::new();
    let expected = model
        .generate_prepared_chat_speculative(PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::prepared_backend_input(
                &chat,
                vec![CONTROL_REJECTION_PROMPT_TOKEN],
            ),
            drafting: SpeculativeDraft::Embedded,
            settings,
            options: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |e| expected_events.push(e),
        })
        .unwrap();
    let mut events = Vec::new();
    let mut steps = Vec::new();
    let output = model
        .with_controlled_chat_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::prepared_backend_input(
                    &chat,
                    vec![CONTROL_REJECTION_PROMPT_TOKEN],
                ),
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |e| events.push(e),
            },
            options(),
            |session| {
                assert_eq!(session.status(), Status::Prefill);
                assert!(session.token_ids().is_empty());
                assert_eq!(session.timing().time_to_first_token(), None);
                while let Some(step) = session.step()? {
                    steps.push(step);
                }
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(output.token_ids(), expected.token_ids());
    assert_eq!(output.finish_reason(), expected.finish_reason());
    assert_eq!(events, expected_events);
    assert!(steps.iter().any(|s| s
        .drafted
        .as_ref()
        .is_some_and(|d| d.token_ids == [11, 11, 11])));
    let verified = steps.iter().find_map(|s| s.verification.as_ref()).unwrap();
    assert_eq!(
        verified.dispositions,
        [
            Disposition::Accepted,
            Disposition::Rejected,
            Disposition::Discarded
        ]
    );
    assert_eq!(verified.committed_token_ids, [11, 12]);
    assert_eq!(
        steps
            .iter()
            .flat_map(|s| &s.committed_token_ids)
            .copied()
            .collect::<Vec<_>>(),
        output.token_ids()
    );
    for (index, step) in steps.iter().enumerate() {
        assert_eq!(step.sequence, index as u64);
        let json = serde_json::to_value(step).unwrap();
        assert_eq!(json["epoch"], 0);
        assert_eq!(
            serde_json::from_value::<eredu::api::ControlledSpeculativeStep>(json).unwrap(),
            *step
        );
    }
}

#[test]
fn controlled_speculative_snapshots_replay_semantics_without_rewinding_delivery_or_budgets() {
    let (mut model, chat, settings) = setup();
    let mut events = Vec::new();
    let mut replayed = Vec::new();
    model
        .with_controlled_chat_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |e| events.push(e),
            },
            options(),
            |session| {
                assert!(matches!(
                    session.snapshot(),
                    Err(SpeculativeControlError::NotQuiescent)
                ));
                session.step()?;
                let timing = session.timing();
                assert!(session.can_snapshot());
                let snapshot = session.snapshot()?;
                let usage = session.snapshot_usage();
                let mut sequences = Vec::new();
                for epoch in 0..2 {
                    let mut tokens = Vec::new();
                    while let Some(step) = session.step()? {
                        assert_eq!(step.epoch, epoch);
                        sequences.push(step.sequence);
                        tokens.extend(step.committed_token_ids);
                        if step.status == Status::ReadyToSubmitVerification {
                            assert!(matches!(
                                session.snapshot(),
                                Err(SpeculativeControlError::NotQuiescent)
                            ));
                        }
                    }
                    replayed.push(tokens);
                    if epoch == 0 {
                        session.restore(&snapshot)?;
                        assert_eq!(session.token_ids(), [7]);
                        assert_eq!(session.epoch(), 1);
                        assert_eq!(session.timing(), timing);
                        assert_eq!(
                            session.snapshot_usage().retained_bytes,
                            usage.retained_bytes
                        );
                        assert!(
                            session.snapshot_usage().cumulative_copy_bytes
                                > usage.cumulative_copy_bytes
                        );
                    }
                }
                assert!(sequences.windows(2).all(|v| v[0] < v[1]));
                session.release_snapshot(&snapshot)?;
                assert_eq!(session.snapshot_usage().snapshots, 0);
                assert!(matches!(
                    session.restore(&snapshot),
                    Err(SpeculativeControlError::IncompatibleSnapshot)
                ));
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(replayed[0], replayed[1]);
    let terminal_indices = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e, SemanticEvent::Finished { .. }).then_some(i))
        .collect::<Vec<_>>();
    assert_eq!(terminal_indices.len(), 2);
    // The decoder snapshot includes the buffered prefix, so both suffixes flush identically.
    let first = &events[..=terminal_indices[0]];
    let second = &events[terminal_indices[0] + 1..];
    assert!(first.ends_with(second));
}

#[test]
fn speculative_ttft_excludes_pause_and_delivery_and_counts_invisible_first_commit() {
    let (mut model, chat, mut settings) = setup();
    settings.overrides.max_new_tokens = Some(1);
    let wall = Instant::now();
    let mut delivered_timing = None;
    let output = model
        .with_controlled_text_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| std::thread::sleep(Duration::from_millis(40)),
            },
            options(),
            |session| {
                std::thread::sleep(Duration::from_millis(60));
                let step = session.step()?.unwrap();
                assert_eq!(step.committed_token_ids, [7]);
                delivered_timing = Some(step.timing);
                Ok(())
            },
        )
        .unwrap();
    let ttft = output.timing().time_to_first_token().unwrap();
    assert_eq!(delivered_timing.unwrap(), *output.timing());
    assert!(wall.elapsed().saturating_sub(ttft) >= Duration::from_millis(100));
}

#[test]
fn cancellation_before_prefill_and_during_verification_never_commits_drafts() {
    for before in [true, false] {
        let (mut model, chat, settings) = setup();
        let output = model
            .with_controlled_chat_speculative(
                PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                    drafting: SpeculativeDraft::Embedded,
                    settings,
                    options: Default::default(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                },
                options(),
                |session| {
                    if !before {
                        session.step()?;
                        session.step()?;
                        session.step()?;
                        assert_eq!(session.status(), Status::TargetVerificationInFlight);
                    }
                    session.cancel()?;
                    assert_eq!(session.status(), Status::Cancelled);
                    assert!(session.step()?.is_none());
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(output.token_ids(), if before { vec![] } else { vec![7] });
        assert_eq!(output.timing().time_to_first_token().is_none(), before);
        assert_eq!(output.finish_reason(), FinishReason::Cancelled);
    }
}

#[test]
fn trace_limit_failure_is_terminal_even_if_the_controller_ignores_it() {
    let (mut model, chat, settings) = setup();
    let error = model
        .with_controlled_chat_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            ControlledSpeculativeOptions {
                trace_limits: eredu::api::TraceLimits {
                    per_record_bytes: 1,
                    total_bytes: 1,
                },
                ..options()
            },
            |session| {
                assert!(session.step().is_err());
                assert!(matches!(
                    session.step(),
                    Err(SpeculativeControlError::Failed)
                ));
                Ok(())
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("failed"));
}

#[test]
fn snapshot_handles_are_run_scoped_and_failed_admission_does_not_advance() {
    let mut foreign = None;
    for pass in 0..2 {
        let (mut model, chat, settings) = setup();
        model
            .with_controlled_text_speculative(
                PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                    drafting: SpeculativeDraft::Embedded,
                    settings,
                    options: Default::default(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                },
                ControlledSpeculativeOptions {
                    snapshots: Some(SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 0,
                        retained_bytes: 16 << 20,
                        cumulative_copy_bytes: 64 << 20,
                    }),
                    ..Default::default()
                },
                |session| {
                    session.step()?;
                    if pass == 0 {
                        foreign = Some(session.snapshot()?);
                        let before = session.snapshot_usage();
                        assert!(session.snapshot().is_err());
                        assert_eq!(session.snapshot_usage(), before);
                    } else {
                        assert!(matches!(
                            session.restore(foreign.as_ref().unwrap()),
                            Err(SpeculativeControlError::IncompatibleSnapshot)
                        ));
                    }
                    assert_eq!(session.token_ids(), [7]);
                    assert_eq!(session.epoch(), 0);
                    assert!(session.step()?.unwrap().drafted.is_some());
                    Ok(())
                },
            )
            .unwrap();
    }
}
