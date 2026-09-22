use super::*;
use eredu::api::{ControlledSpeculativeOptions, SpeculativeControlError};
use eredu_core::generation::SpeculativeRequestStatus as RequestStatus;

fn schedule_fault(fault: ScheduleFault) {
    PROBE.with(|slot| slot.borrow_mut().as_mut().unwrap().schedule_fault = fault);
}

#[test]
fn speculative_capacity_requires_matching_source_before_prompt_or_model_work() {
    for controlled in [false, true] {
        for (capacity, expected) in [
            (None, eredu_core::TokenInputRejection::IdentityMismatch),
            (
                Some(16 << 20),
                eredu_core::TokenInputRejection::IdentityMismatch,
            ),
        ] {
            let (mut model, chat, mut settings) = setup();
            settings.inference.memory_limits =
                capacity.map(crate::memory::limits).unwrap_or_default();
            let _guard = probe(Fault::None);
            let error = run_speculative(&mut model, &chat, settings, controlled).unwrap_err();
            let error = error
                .downcast_ref::<eredu::api::PreparedChatSpeculativeError>()
                .expect("one preparation error family for both drivers");
            assert_eq!(error.input_rejection(), Some(expected));
            let seen = snapshot();
            assert_eq!(seen.votes.last(), Some(&(Stage::Request, Status::Failed)));
            assert!(seen.native.is_empty());
            assert!(seen.actions.is_empty());
            assert_eq!(seen.forwards, 0);
        }
    }
}

#[test]
fn speculative_later_lane_foreign_capacity_rejects_whole_batch_before_prompt() {
    let (mut model, chat, settings) = setup();
    let mut bounded = settings.clone();
    bounded.inference.memory_limits = crate::memory::limits(16 << 20);
    let _guard = probe(Fault::None);
    let error = model
        .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: [settings, bounded]
                .into_iter()
                .map(|settings| PreparedChatSpeculativeBatchLane {
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                    output_mode: eredu::api::PreparedChatOutputMode::Text,
                    skip_special_tokens: true,
                    settings,
                    max_draft_tokens: NonZeroUsize::new(1).unwrap(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: Box::new(|_| {}),
                })
                .collect(),
            scheduler: Default::default(),
        })
        .err()
        .expect("bounded second lane must reject the batch");
    assert_eq!(
        error.input_rejection(),
        Some(eredu_core::TokenInputRejection::IdentityMismatch)
    );
    let seen = snapshot();
    assert_eq!(seen.votes, [(Stage::Request, Status::Failed)]);
    assert!(seen.native.is_empty());
    assert!(seen.actions.is_empty());
    assert_eq!(seen.forwards, 0);
}

#[test]
fn draft_sampling_failure_agrees_before_the_next_collective_forward() {
    for controlled in [false, true] {
        for peer in [false, true] {
            let (mut model, chat, mut settings) = setup();
            settings.seed = 0;
            settings.overrides.max_new_tokens = Some(6);
            let _guard = probe(Fault::DraftSampling { peer });
            SPECULATIVE_RESULT_FAULT.set(Some(CONTROL_REJECTION_PROMPT_TOKEN));
            let request = PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: PreparedChatSpeculativeGenerationOptions {
                    max_draft_tokens: NonZeroUsize::new(3).unwrap(),
                    ..Default::default()
                },
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            };
            let error: Box<dyn std::error::Error> = if controlled {
                model
                    .with_controlled_prepared_chat_speculative(
                        request,
                        Default::default(),
                        |session| {
                            while session.step()?.is_some() {}
                            Ok(())
                        },
                    )
                    .unwrap_err()
                    .into()
            } else {
                model
                    .generate_prepared_chat_speculative(request)
                    .unwrap_err()
                    .into()
            };
            if peer {
                assert!(
                    has_source::<eredu_core::run_preparation::TextPreparationRejected>(
                        error.as_ref()
                    ),
                    "{error}"
                );
            } else {
                assert!(has_source::<MockError>(error.as_ref()), "{error}");
                assert!(
                    error
                        .to_string()
                        .contains("original draft sampling failure"),
                    "{error}"
                );
            }
            assert_eq!(
                snapshot().actions,
                ["prefill", "begin_proposal", "proposal_logits"]
            );
            assert_eq!(
                snapshot().votes.last(),
                Some(&(Stage::Delivery, Status::Failed))
            );
        }
    }
}

#[test]
fn peer_cancellation_before_draft_and_during_verification_preserves_committed_prefix() {
    for controlled in [false, true] {
        for pending in [false, true] {
            let (mut model, chat, settings) = setup();
            let baseline =
                run_speculative(&mut model, &chat, settings.clone(), controlled).unwrap();
            let _guard = probe(Fault::None);
            schedule_fault(ScheduleFault::Cancel { lane: 0, pending });
            let actual = run_speculative(&mut model, &chat, settings, controlled).unwrap();
            assert_eq!(actual, baseline[..1]);
            let observed = snapshot();
            assert_eq!(observed.actions.contains(&"verification"), pending);
            assert_eq!(observed.actions.contains(&"commit"), pending);
            assert!(!observed.actions.contains(&"restore"));
            assert!(observed.schedules.iter().any(|states| states[0].status
                == if pending {
                    RequestStatus::TargetVerificationInFlight
                } else {
                    RequestStatus::ReadyToDraft
                }));
        }
    }
}

#[test]
fn peer_completion_delay_polls_without_repeating_native_work_or_publication() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        let baseline = run_speculative(&mut model, &chat, settings.clone(), controlled).unwrap();
        let _guard = probe(Fault::None);
        schedule_fault(ScheduleFault::DelayCompletion);
        assert_eq!(
            run_speculative(&mut model, &chat, settings, controlled).unwrap(),
            baseline
        );
        let observed = snapshot();
        assert_eq!(
            observed
                .actions
                .iter()
                .filter(|&&a| a == "verification")
                .count(),
            1
        );
        assert_eq!(
            observed.actions.iter().filter(|&&a| a == "commit").count(),
            1
        );
        assert_eq!(
            observed
                .schedules
                .iter()
                .filter(|states| states[0].verification_complete)
                .count(),
            2
        );
    }
}

#[test]
fn invalid_coordinated_identity_fails_before_any_draft() {
    for controlled in [false, true] {
        let (mut model, chat, settings) = setup();
        let _guard = probe(Fault::None);
        schedule_fault(ScheduleFault::InvalidIdentity);
        let error = run_speculative(&mut model, &chat, settings, controlled).unwrap_err();
        assert!(
            error.to_string().contains("request identity or lifecycle"),
            "{error}"
        );
        assert_eq!(snapshot().actions, ["prefill"]);
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Failed))
        );
    }
}

#[test]
fn peer_cancellation_is_per_lane_and_other_batch_lanes_finish() {
    let (mut model, chat, mut settings) = setup();
    settings.seed = 0;
    let _guard = probe(Fault::None);
    schedule_fault(ScheduleFault::Cancel {
        lane: 0,
        pending: true,
    });
    let batch = model
        .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: (0..2)
                .map(|_| PreparedChatSpeculativeBatchLane {
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                    output_mode: eredu::api::PreparedChatOutputMode::Text,
                    skip_special_tokens: true,
                    settings: settings.clone(),
                    max_draft_tokens: NonZeroUsize::new(1).unwrap(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: Box::new(|_| {}),
                })
                .collect(),
            scheduler: Default::default(),
        })
        .unwrap();
    assert_eq!(batch.requests()[0].token_ids(), [7]);
    assert_eq!(batch.requests()[1].token_ids(), [7, 11]);
}

#[test]
fn peer_publication_cancellation_matches_local_callback_cancellation() {
    for controlled in [false, true] {
        let (mut model, chat, mut settings) = setup();
        settings.seed = 0;
        let _guard = probe(Fault::None);
        let mut armed = false;
        let request = PreparedChatSpeculativeRequest {
            chat: &chat,
            input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
            output_mode: eredu::api::PreparedChatOutputMode::Text,
            skip_special_tokens: true,
            drafting: SpeculativeDraft::Embedded,
            settings,
            options: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {
                if !armed {
                    armed = true;
                    PROBE.with(|slot| {
                        slot.borrow_mut().as_mut().unwrap().fault =
                            Fault::CancelOnce(Stage::Delivery)
                    });
                }
            },
        };
        let output = if controlled {
            model
                .with_controlled_prepared_chat_speculative(request, Default::default(), |session| {
                    while session.step()?.is_some() {}
                    Ok(())
                })
                .unwrap()
        } else {
            model.generate_prepared_chat_speculative(request).unwrap()
        };
        assert!(armed);
        assert_eq!(output.token_ids(), [7]);
        assert_eq!(snapshot().actions, ["prefill"]);
    }
}

#[test]
fn controlled_record_budget_and_caller_error_agree_before_more_model_work() {
    for record_failure in [false, true] {
        let (mut model, chat, mut settings) = setup();
        settings.seed = 0;
        let _guard = probe(Fault::None);
        let mut options = ControlledSpeculativeOptions::default();
        if record_failure {
            options.trace_limits = TraceLimits {
                per_record_bytes: 1,
                total_bytes: 1,
            };
        }
        let error = model
            .with_controlled_prepared_chat_speculative(
                PreparedChatSpeculativeRequest {
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                    output_mode: eredu::api::PreparedChatOutputMode::Text,
                    skip_special_tokens: true,
                    drafting: SpeculativeDraft::Embedded,
                    settings,
                    options: Default::default(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                },
                options,
                |session| {
                    if record_failure {
                        let error = session.step().unwrap_err();
                        assert!(
                            matches!(error, SpeculativeControlError::Capture(_)),
                            "{error}"
                        );
                        assert!(matches!(
                            session.step(),
                            Err(SpeculativeControlError::Failed)
                        ));
                        Err(error)
                    } else {
                        assert!(session.step()?.is_some());
                        Err(SpeculativeControlError::Invalid(
                            "caller stopped after prefill",
                        ))
                    }
                },
            )
            .unwrap_err();
        assert!(
            if record_failure {
                error.to_string().contains("limit exceeded")
            } else {
                error.to_string().contains("caller stopped after prefill")
            },
            "{error}"
        );
        assert_eq!(snapshot().actions, ["prefill"]);
        assert_eq!(
            snapshot().votes.last(),
            Some(&(Stage::Delivery, Status::Failed))
        );
    }
}
