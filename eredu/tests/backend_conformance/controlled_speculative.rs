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
fn public_internal_activation_authority_preserves_parity_and_applies_edits() {
    use eredu::api::SpeculativeActivationPlan;
    use eredu_core::{capture::*, intervention::*};
    for factor in [1.0, 2.0] {
        let (mut model, chat, settings) = setup();
        let request = || PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::prepared_backend_input(&chat, vec![3, 4]),
            drafting: SpeculativeDraft::Embedded,
            settings,
            options: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        };
        let baseline = model.generate_prepared_chat_speculative(request()).unwrap();
        let plan = model
            .prepare_speculative_activations(SpeculativeActivationPlan {
                schema_version: eredu_core::speculative::SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                captures: observed_mock::plan(),
                interventions: observed_mock::intervention_plan(factor),
                bounds: CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 4,
                    max_context: None,
                    max_predictions: 6,
                },
            })
            .unwrap();
        let identity = plan.identity().to_owned();
        let mut records = Vec::new();
        let output = model
            .with_controlled_chat_speculative(
                request(),
                ControlledSpeculativeOptions {
                    activations: Some(plan.clone()),
                    ..options()
                },
                |session| {
                    while let Some(step) = session.step()? {
                        assert!(step.captures.is_empty());
                        records.extend(step.activations);
                    }
                    assert!(session.take_activation_evidence()?.is_none());
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(output.token_ids()[0], (7.0 * factor) as u32);
        if factor == 1.0 {
            assert_eq!(output.token_ids(), baseline.token_ids());
        }
        assert_eq!(records[0].captures.invocation.unwrap().sequence, 2);
        assert!(records
            .iter()
            .all(|r| r.completed && r.admission_identity.as_deref() == Some(identity.as_str())));
        assert!(records
            .iter()
            .flat_map(|r| &r.captures.interventions)
            .any(|edit| edit.outcome == InterventionOutcome::Applied));
        let mut continuous = Vec::new();
        let streamed = model
            .generate_observed_chat_speculative(
                request(),
                ControlledSpeculativeOptions {
                    activations: Some(plan.clone()),
                    ..Default::default()
                },
                |step| {
                    continuous.extend(step.activations);
                    std::ops::ControlFlow::Continue(())
                },
            )
            .unwrap();
        assert_eq!(streamed.token_ids(), output.token_ids());
        assert_eq!(continuous.len(), records.len());
        for (a, b) in continuous.iter().zip(&records) {
            assert_eq!(
                (a.origin, a.phase, a.captures.invocation),
                (b.origin, b.phase, b.captures.invocation)
            );
            assert_eq!(a.captures.records, b.captures.records);
            assert_eq!(a.captures.interventions, b.captures.interventions);
        }
        let fresh = model.generate_prepared_chat_speculative(request()).unwrap();
        assert_eq!(fresh.token_ids(), baseline.token_ids());
        let (mut other, _, _) = setup();
        let mut entered = false;
        let result = other.with_controlled_chat_speculative(
            request(),
            ControlledSpeculativeOptions {
                activations: Some(plan),
                ..Default::default()
            },
            |_| {
                entered = true;
                Ok(())
            },
        );
        assert!(result.is_err());
        assert!(!entered);
    }
}

#[test]
fn failed_internal_actions_deliver_bounded_original_evidence_without_recovery() {
    use eredu::api::SpeculativeActivationPlan;
    use eredu_core::{capture::*, intervention::InterventionPlan};
    for transport_bytes in [1, 1 << 20] {
        let (mut model, chat, settings) = setup();
        let mut captures = observed_mock::plan();
        captures.selections[0].id = "injected-native-failure".into();
        let plan = model
            .prepare_speculative_activations(SpeculativeActivationPlan {
                schema_version: eredu_core::speculative::SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                captures,
                interventions: InterventionPlan::none(),
                bounds: CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 4,
                    max_context: None,
                    max_predictions: 6,
                },
            })
            .unwrap();
        let result = model.with_controlled_chat_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::prepared_backend_input(&chat, vec![3, 4]),
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| panic!("failed prefill published"),
            },
            ControlledSpeculativeOptions {
                activations: Some(plan),
                trace_limits: eredu::api::TraceLimits {
                    per_record_bytes: transport_bytes,
                    total_bytes: transport_bytes,
                },
                ..Default::default()
            },
            |session| {
                let error = session.step().unwrap_err();
                let SpeculativeControlError::Backend(source) = &error else {
                    panic!("lost original backend error: {error}");
                };
                assert!(std::error::Error::source(source).is_some());
                let evidence = session.take_activation_evidence();
                if transport_bytes == 1 {
                    assert!(matches!(
                        evidence,
                        Err(SpeculativeControlError::Capture(CaptureError::Limit {
                            budget: CaptureBudget::Encoded,
                            ..
                        }))
                    ));
                } else {
                    let evidence = evidence?.unwrap();
                    assert_eq!(evidence.sequence, 0);
                    assert!(!evidence.activation.completed);
                    assert_eq!(evidence.activation.origin.committed_tokens, 0);
                }
                assert!(session.take_activation_evidence()?.is_none());
                assert!(matches!(
                    session.step(),
                    Err(SpeculativeControlError::Failed)
                ));
                Err(error)
            },
        );
        assert!(result.is_err());
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
    let activations = model
        .prepare_speculative_activations(eredu::api::SpeculativeActivationPlan {
            schema_version: eredu_core::speculative::SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            captures: observed_mock::plan(),
            interventions: eredu_core::intervention::InterventionPlan::none(),
            bounds: eredu_core::capture::CaptureInvocationBounds {
                batch: 1,
                max_sequence: 4,
                max_context: None,
                max_predictions: 6,
            },
        })
        .unwrap();
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
            ControlledSpeculativeOptions {
                activations: Some(activations),
                ..options()
            },
            |session| {
                assert!(matches!(
                    session.snapshot(),
                    Err(SpeculativeControlError::NotQuiescent)
                ));
                let first = session.step()?.unwrap();
                let last = first.activations.last().unwrap();
                let mut last_invocation = last.invocation;
                let mut last_usage = last.captures.cumulative_usage;
                let timing = session.timing();
                assert!(session.can_snapshot());
                let snapshot = session.snapshot()?;
                let usage = session.snapshot_usage();
                let mut sequences = Vec::new();
                for epoch in 0..2 {
                    let mut tokens = Vec::new();
                    while let Some(step) = session.step()? {
                        assert_eq!(step.epoch, epoch);
                        for activation in &step.activations {
                            assert!(activation.invocation > last_invocation);
                            assert!(
                                activation.captures.cumulative_usage.encoded_bytes
                                    > last_usage.encoded_bytes
                            );
                            last_invocation = activation.invocation;
                            last_usage = activation.captures.cumulative_usage;
                        }
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

#[test]
fn speculative_forks_isolate_choices_sampling_and_semantics_and_reject_foreign_handles() {
    let (mut model, chat, settings) = setup();
    let mut opts = options();
    opts.snapshots.as_mut().unwrap().max_branches = 1;
    let mut foreign = None;
    let events = std::cell::RefCell::new(Vec::new());
    for pass in 0..2 {
        model
            .with_controlled_text_speculative(
                PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                    drafting: SpeculativeDraft::Embedded,
                    settings,
                    options: Default::default(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |event| events.borrow_mut().push(event),
                },
                opts.clone(),
                |session| {
                    session.step()?;
                    if pass == 1 {
                        assert!(matches!(
                            session.exchange(foreign.as_ref().unwrap()),
                            Err(SpeculativeControlError::IncompatibleBranch)
                        ));
                        return Ok(());
                    }
                    let root = session.snapshot()?;
                    let child = session.fork(&root)?;
                    foreign = Some(child.clone());
                    assert_eq!(session.run_id(), 0);
                    assert_eq!(session.snapshot_usage().branches, 1);
                    let before = session.snapshot_usage();
                    assert!(session.fork(&root).is_err());
                    assert_eq!(session.snapshot_usage(), before);
                    while session.step()?.is_some() {}
                    let original = session.token_ids().to_vec();
                    let facts = session.sampling_state().unwrap();
                    session.exchange(&child)?;
                    assert_ne!(session.run_id(), 0);
                    assert!(matches!(
                        session.restore(&root),
                        Err(SpeculativeControlError::IncompatibleSnapshot)
                    ));
                    assert_eq!(session.token_ids(), [7]);
                    let unchanged = session.sampling_state();
                    assert!(session
                        .override_sampling(eredu::api::SamplingOverride {
                            temperature: Some(-1.0),
                            reseed: None
                        })
                        .is_err());
                    assert_eq!(session.sampling_state(), unchanged);
                    session.override_sampling(eredu::api::SamplingOverride {
                        temperature: Some(0.7),
                        reseed: Some(42),
                    })?;
                    assert!(session.force_next_token(u32::MAX).is_err());
                    session.force_next_token(12)?;
                    assert!(session.force_next_token(11).is_err());
                    let forced = session.snapshot()?;
                    let baseline_usage = session.snapshot_usage();
                    let mut replay = Vec::new();
                    let mut semantic_replays = Vec::new();
                    for pass in 0..2 {
                        let first_event = events.borrow().len();
                        let mut committed = Vec::new();
                        let mut forced_count = 0;
                        while let Some(step) = session.step()? {
                            assert_eq!(step.run_id, session.run_id());
                            forced_count += usize::from(step.forced_token == Some(12));
                            committed.extend(step.committed_token_ids);
                            if step.status == Status::ReadyToSubmitVerification {
                                assert!(matches!(
                                    session.force_next_token(11),
                                    Err(SpeculativeControlError::NotQuiescent)
                                ));
                                assert!(matches!(
                                    session.exchange(&child),
                                    Err(SpeculativeControlError::NotQuiescent)
                                ));
                            }
                        }
                        assert_eq!(forced_count, 1);
                        assert_eq!(committed[0], 12);
                        replay.push(committed);
                        semantic_replays.push(events.borrow()[first_event..].to_vec());
                        if pass == 0 {
                            session.restore(&forced)?;
                        }
                    }
                    assert_eq!(replay[0], replay[1]);
                    assert_eq!(semantic_replays[0], semantic_replays[1]);
                    assert!(
                        session.snapshot_usage().cumulative_copy_bytes
                            > baseline_usage.cumulative_copy_bytes
                    );
                    session.exchange(&child)?;
                    assert_eq!(session.run_id(), 0);
                    assert_eq!(session.token_ids(), original);
                    assert_eq!(session.sampling_state(), Some(facts));
                    assert_eq!(session.branch_info(&child)?.token_ids[1], 12);
                    session.release_snapshot(&forced)?;
                    session.release_snapshot(&root)?;
                    session.release_branch(&child)?;
                    assert_eq!(session.snapshot_usage().branches, 0);
                    assert_eq!(session.snapshot_usage().retained_bytes, 0);
                    Ok(())
                },
            )
            .unwrap();
    }
}

#[test]
fn conversion_retention_queries_preserve_speculative_proposals_and_budgets() {
    for embedded in [true, false] {
        let (mut model, chat, settings) = setup();
        let target = model.parameter_conversion_retention().unwrap();
        let mut draft = MockDrafter {
            retention: Some(conversion_retention::budget(&Default::default())),
        };
        let expected = eredu_core::residency::ExecutionConversionRetentionReport {
            target,
            external_drafter: (!embedded).then(|| eredu_core::residency::ParameterConversionRetentionObserver::parameter_conversion_retention(&draft).unwrap()),
        };
        let mut observed_pending = false;
        let mut committed = Vec::new();
        let output = model
            .with_controlled_chat_speculative(
                PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::prepared_backend_input(&chat, vec![0]),
                    drafting: if embedded {
                        SpeculativeDraft::Embedded
                    } else {
                        SpeculativeDraft::External(&mut draft)
                    },
                    settings,
                    options: Default::default(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                },
                options(),
                |session| {
                    loop {
                        let status = session.status();
                        let tokens = session.token_ids().to_vec();
                        let epoch = session.epoch();
                        let usage = session.snapshot_usage();
                        for _ in 0..3 {
                            assert_eq!(session.parameter_conversion_retention()?, expected);
                            assert_eq!(session.status(), status);
                            assert_eq!(session.token_ids(), tokens);
                            assert_eq!(session.epoch(), epoch);
                            assert_eq!(session.snapshot_usage(), usage);
                        }
                        if status == Status::ReadyToSubmitVerification {
                            observed_pending = true;
                            assert!(matches!(
                                session.snapshot(),
                                Err(SpeculativeControlError::NotQuiescent)
                            ));
                        }
                        let Some(step) = session.step()? else {
                            break;
                        };
                        committed.extend(step.committed_token_ids);
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert!(observed_pending);
        assert_eq!(committed, output.token_ids());
    }
}
