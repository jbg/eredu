use super::*;
use crate::runtime::voicechat::{
    VoiceChatToolWaitAction, VoiceChatToolWaitMessages, VoiceChatToolWaitPolicy,
    VoiceChatTurnLimits, VoiceChatTurnPolicy,
    tools::tests::{call, channel, limits},
};
use eredu_architectures::voicechat::{
    channel_decision::FeedbackTokens,
    observation::{OwnedSpeechChunk, TranscriptionOutput},
    output::OutputDecisions,
    schedule::{AudioSource, Progress},
};

pub(crate) fn conversation(channel: VoiceChatToolChannel) -> VoiceChatConversation {
    conversation_with_policy(channel, VoiceChatTurnPolicy::default())
}

pub(crate) fn conversation_with_policy(
    channel: VoiceChatToolChannel,
    policy: VoiceChatTurnPolicy,
) -> VoiceChatConversation {
    VoiceChatConversation::new(
        channel,
        policy,
        VoiceChatTurnLimits {
            max_audio_frames: 64,
            max_model_steps: 64,
            max_history: 40,
        },
        RnntTextDecoder::from_vocabulary(
            &["<unk>", "a", "b", "c", "d", "▁hi"].map(str::to_owned),
            64,
        )
        .unwrap(),
        VoiceChatDecodeLimits {
            max_pending_tokens: 64,
            max_pending_bytes: 1024,
        },
    )
    .unwrap()
}

pub(crate) fn capacity() -> VoiceChatPublicationCapacity {
    VoiceChatPublicationCapacity {
        speech_samples: 1764,
        transcription_tokens: 4,
        agent_text_bytes: 256,
        transcription_text_bytes: 256,
        audio_codes: 31,
        tool_events: 8,
        tool_bytes: 256,
        diagnostics: 0,
    }
}

// Inputs represent architecture-observed frames. Expected conversation behavior
// below uses fixed token IDs and explicit audio/model coordinates.
pub(crate) fn frame(
    invocation: u64,
    before: Progress,
    step: Step,
    function: Option<u32>,
) -> OwnedHostOutput<()> {
    let mut after = before;
    let fresh = matches!(
        step,
        Step::LiveAudio
            | Step::PerceptionOnly
            | Step::Tool {
                audio: AudioSource::Fresh
            }
    );
    let transcription = fresh.then(|| {
        after.audio_frames += 1;
        after.perception_frames += 1;
        after.rnnt_frames += 1;
        TranscriptionOutput {
            source: eredu_media::audio::streaming::PcmFrameSource::Input,
            frame_index: before.audio_frames,
            sample_rate: 16000,
            start_sample: before.audio_frames * 1280,
            valid_samples: 1280,
            emitted: vec![5],
            first_blank: false,
            ended_blank: true,
            hit_symbol_limit: false,
        }
    });
    if step == Step::ReplayAudio {
        after.backbone_positions += 1;
    }
    let decisions = function.map(|function| {
        after.backbone_positions += 1;
        OutputDecisions {
            raw_agent: 12,
            raw_function: function,
            feedback: FeedbackTokens::new(12, function).unwrap(),
            spoken_text: 12,
        }
    });
    OwnedHostOutput {
        invocation_index: invocation,
        step,
        before,
        after,
        decisions,
        generated_codes: None,
        speech: None,
        transcription,
        reminder_token: None,
        diagnostics: vec![],
    }
}

fn collect(owner: &mut VoiceChatConversation, payload: u32) -> Progress {
    let start = frame(0, Progress::default(), Step::LiveAudio, Some(20));
    let payload = frame(
        1,
        start.after,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(payload),
    );
    let after = payload.after;
    for frame in [start, payload] {
        let published = owner
            .admit(frame.step, capacity())
            .unwrap()
            .stage(frame)
            .unwrap()
            .commit();
        assert!(published.tools.is_none());
    }
    assert_eq!((after.audio_frames, after.backbone_positions), (1, 2));
    after
}

#[test]
fn malformed_call_rejects_the_whole_host_successor_after_valid_turn_evidence() {
    let mut owner = conversation(channel(601, limits()));
    let before = collect(&mut owner, 32); // city must be a string.
    let end = frame(
        2,
        before,
        Step::Tool {
            audio: AudioSource::Fresh,
        },
        Some(21),
    );

    for _ in 0..2 {
        assert!(matches!(
            owner
                .admit(end.step, capacity())
                .unwrap()
                .stage(end.clone()),
            Err(VoiceChatConversationError::Tool(
                VoiceChatToolError::Protocol(_)
            ))
        ));
        assert!(owner.pending().is_none());
    }

    // Both failures preserve the invocation and acoustic frontier, so a PAD
    // retry at exactly that position succeeds without discarding the payload.
    let retry = frame(2, before, end.step, Some(12));
    let published = owner
        .admit(retry.step, capacity())
        .unwrap()
        .stage(retry)
        .unwrap()
        .commit();
    assert_eq!(
        (
            published.frame.after.audio_frames,
            published.frame.after.backbone_positions
        ),
        (2, 3)
    );
    assert!(published.tools.is_none());
    let still_invalid = frame(3, published.frame.after, end.step, Some(21));
    assert!(matches!(
        owner
            .admit(end.step, capacity())
            .unwrap()
            .stage(still_invalid),
        Err(VoiceChatConversationError::Tool(
            VoiceChatToolError::Protocol(_)
        ))
    ));
}

#[test]
fn reserved_capacity_and_discard_preserve_a_complete_unpublished_call() {
    let mut owner = conversation(channel(602, limits()));
    let before = collect(&mut owner, 30);
    let mut end = frame(
        2,
        before,
        Step::Tool {
            audio: AudioSource::Fresh,
        },
        Some(21),
    );
    end.speech = Some(OwnedSpeechChunk {
        samples: vec![0.25, -0.5],
        sample_rate: 22050,
        start_sample: 0,
        delay_samples: 0,
        is_tail: false,
    });
    for (reserved, resource) in [
        (
            VoiceChatPublicationCapacity {
                speech_samples: 1,
                ..capacity()
            },
            "speech samples",
        ),
        (
            VoiceChatPublicationCapacity {
                tool_events: 0,
                ..capacity()
            },
            "tool events",
        ),
        (
            VoiceChatPublicationCapacity {
                tool_bytes: 0,
                ..capacity()
            },
            "tool bytes",
        ),
    ] {
        assert!(matches!(
            owner.admit(end.step, reserved)
                .unwrap().stage(end.clone()),
            Err(VoiceChatConversationError::Capacity(actual)) if actual == resource
        ));
        assert!(owner.pending().is_none());
    }
    let staged = owner
        .admit(end.step, capacity())
        .unwrap()
        .stage(end.clone())
        .unwrap();
    assert!(owner.pending().is_none(), "staged calls are not published");
    assert!(matches!(
        owner.interrupt(),
        Err(VoiceChatConversationError::Busy)
    ));
    drop(staged);
    assert!(owner.pending().is_none());

    let published = owner
        .admit(end.step, capacity())
        .unwrap()
        .stage(end)
        .unwrap()
        .commit();
    assert_eq!(published.frame.speech.unwrap().samples, [0.25, -0.5]);
    let batch = published.tools.unwrap();
    assert_eq!(batch.ticket.batch(), 0);
    assert_eq!(owner.pending(), Some(batch.ticket));
    assert_eq!(batch.events.len(), 3);
    assert!(
        matches!(&batch.events[0], SemanticEvent::ToolCallStart { index: 0, name, .. } if name == "weather")
    );
    assert!(
        matches!(&batch.events[1], SemanticEvent::ToolArgumentsDelta { index: 0, json_fragment }
        if serde_json::from_str::<serde_json::Value>(json_fragment).unwrap() == serde_json::json!({"city":"Paris"}))
    );
    assert!(matches!(&batch.events[2], SemanticEvent::ToolCallEnd));
}

#[test]
fn discarded_reminder_replays_its_receipt_and_only_committed_eos_waits_for_playback() {
    let mut channel = channel(603, limits());
    let ticket = call(&mut channel).ticket;
    let mut owner = conversation(channel);
    owner
        .start_wait(
            ticket,
            Duration::ZERO,
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                generic: &[],
                max_phrase_tokens: 32,
            },
        )
        .unwrap();
    let VoiceChatToolWaitAction::Reminder(first) = owner.advance_wait(Duration::ZERO).unwrap()
    else {
        panic!("initial reminder missing");
    };
    let mut output = frame(0, Progress::default(), Step::Reminder, None);
    output.reminder_token = Some(12);
    assert!(matches!(
        owner
            .admit_reminder(first, Duration::ZERO, capacity())
            .unwrap()
            .stage(output.clone()),
        Err(VoiceChatConversationError::Reminder)
    ));
    output.reminder_token = Some(first.spoken());
    let abandoned = owner
        .admit_reminder(first, Duration::ZERO, capacity())
        .unwrap()
        .stage(output)
        .unwrap();
    assert!(matches!(
        owner.advance_wait(Duration::ZERO),
        Err(VoiceChatConversationError::Busy)
    ));
    assert!(matches!(
        owner.submit_response(ticket, "ok"),
        Err(VoiceChatConversationError::Busy)
    ));
    drop(abandoned);
    assert_eq!(
        owner.advance_wait(Duration::ZERO).unwrap(),
        VoiceChatToolWaitAction::Reminder(first)
    );

    // An already available result still waits for the initial phrase's EOS and
    // explicit presentation completion. "ok" is BOS, content, 17 PADs, EOS.
    owner.submit_response(ticket, "ok").unwrap();
    let expected = [1, 51].into_iter().chain([12; 17]).chain([2]);
    let mut progress = Progress::default();
    let mut playback = None;
    for (invocation, spoken) in expected.enumerate() {
        let VoiceChatToolWaitAction::Reminder(receipt) =
            owner.advance_wait(Duration::ZERO).unwrap()
        else {
            panic!("initial phrase ended before EOS");
        };
        assert_eq!(receipt.spoken(), spoken);
        let mut output = frame(invocation as u64, progress, Step::Reminder, None);
        output.reminder_token = Some(receipt.spoken());
        output.after.tts_conditional_positions += 1;
        output.after.tts_unconditional_positions += 1;
        output.after.codec_frames += 1;
        output.after.output_samples += 1764;
        progress = output.after;
        let published = owner
            .admit_reminder(receipt, Duration::ZERO, capacity())
            .unwrap()
            .stage(output)
            .unwrap()
            .commit();
        assert_eq!(published.playback.is_some(), spoken == 2);
        playback = published.playback;
    }
    assert_eq!(
        owner.advance_wait(Duration::ZERO).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly
    );
    owner
        .finish_playback(playback.unwrap(), Duration::from_millis(1600))
        .unwrap();
    assert!(matches!(
        owner.advance_wait(Duration::from_millis(1600)).unwrap(),
        VoiceChatToolWaitAction::Response(_)
    ));
}

#[test]
fn sustained_tool_speech_interrupts_only_when_its_publication_commits() {
    fn run(discard: bool, phase: usize) -> (VoiceChatTurnIdentity, Progress) {
        let mut owner = conversation_with_policy(
            channel(680 + phase as u64, limits()),
            VoiceChatTurnPolicy {
                bou_frames: 2,
                ..VoiceChatTurnPolicy::default()
            },
        );
        // A real prefill admission opens the same next-work policy used by sessions.
        let prefill = Step::Prefill { backbone_tokens: 1 };
        let mut output = frame(0, Progress::default(), prefill, None);
        output.after.backbone_positions = 1;
        let mut progress = output.after;
        owner
            .prepare_prefill(1, Duration::ZERO, capacity())
            .unwrap()
            .stage(output)
            .unwrap()
            .commit();
        let mut invocation = 1;
        let mut ticket = None;
        for function in [20, 30, 21]
            .into_iter()
            .take(if phase == 0 { 2 } else { 3 })
        {
            let step = if function == 20 {
                Step::LiveAudio
            } else {
                Step::Tool {
                    audio: AudioSource::Retained,
                }
            };
            let output = frame(invocation, progress, step, Some(function));
            progress = output.after;
            let published = owner
                .admit(step, capacity())
                .unwrap()
                .stage(output)
                .unwrap()
                .commit();
            if let Some(batch) = published.tools {
                ticket = Some(batch.ticket);
            }
            invocation += 1;
        }
        if let Some(ticket) = ticket {
            owner
                .start_wait(
                    ticket,
                    Duration::ZERO,
                    VoiceChatToolWaitPolicy::default(),
                    VoiceChatToolWaitMessages {
                        initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                        generic: &[],
                        max_phrase_tokens: 64,
                    },
                )
                .unwrap();
            if phase == 2 {
                owner.submit_response(ticket, "ok ok").unwrap();
                // Retain one token of an already begun response injection.
                let step = Step::Tool {
                    audio: AudioSource::Retained,
                };
                let output = frame(invocation, progress, step, Some(51));
                progress = output.after;
                owner
                    .admit(step, capacity())
                    .unwrap()
                    .stage(output)
                    .unwrap()
                    .commit();
                invocation += 1;
            }
        }
        // The first utterance is separated by a blank; onset alone does not
        // invalidate a tool. The same bound applies while the agent is silent.
        for first_blank in [true, false] {
            let mut output = frame(invocation, progress, Step::PerceptionOnly, None);
            let transcription = output.transcription.as_mut().unwrap();
            transcription.first_blank = first_blank;
            if first_blank {
                transcription.emitted.clear();
            }
            progress = output.after;
            let published = owner
                .admit(output.step, capacity())
                .unwrap()
                .stage(output)
                .unwrap()
                .commit();
            assert_eq!(published.turn, None);
            assert_eq!(owner.pending(), ticket);
            invocation += 1;
        }
        let identity = owner.turn_identity();
        let response = owner.response_policy();
        let step = if phase == 0 {
            // An EOTC from the interrupted invocation must not publish its old call.
            Step::Tool {
                audio: AudioSource::Fresh,
            }
        } else {
            Step::PerceptionOnly
        };
        let output = frame(invocation, progress, step, (phase == 0).then_some(21));
        if discard {
            // Reject after turn projection has already staged the interruption.
            let insufficient = VoiceChatPublicationCapacity {
                transcription_text_bytes: 0,
                ..capacity()
            };
            assert!(
                owner
                    .admit(step, insufficient)
                    .unwrap()
                    .stage(output.clone())
                    .is_err()
            );
            let staged = owner
                .admit(step, capacity())
                .unwrap()
                .stage(output.clone())
                .unwrap();
            assert_eq!(
                staged.publication().turn,
                Some(VoiceChatTurnEvent::Interrupted)
            );
            assert_eq!(owner.turn_identity(), identity);
            assert_eq!(owner.pending(), ticket);
            assert_eq!(owner.response_policy(), response);
            drop(staged);
            assert_eq!(owner.owner.borrow().state.turn.frontier(), progress);
        }
        let published = owner
            .admit(step, capacity())
            .unwrap()
            .stage(output)
            .unwrap()
            .commit();
        assert_eq!(published.turn, Some(VoiceChatTurnEvent::Interrupted));
        assert!(published.tools.is_none());
        assert!(published.playback.is_none());
        assert_eq!(
            published.frame.after.audio_frames,
            progress.audio_frames + 1
        );
        assert_eq!(
            published.frame.after.backbone_positions,
            progress.backbone_positions + u64::from(phase == 0)
        );
        assert_eq!(owner.turn_identity().turn(), identity.turn() + 1);
        assert!(owner.pending().is_none());
        assert!(owner.response_policy().is_none());
        if let Some(ticket) = ticket {
            assert!(matches!(
                owner.submit_response(ticket, "ok"),
                Err(VoiceChatConversationError::Tool(
                    VoiceChatToolError::StaleResult
                ))
            ));
            assert!(matches!(
                owner.advance_wait(Duration::ZERO),
                Err(VoiceChatConversationError::NoWait)
            ));
        }
        // Every perception-only frame is handed to the backbone before fresh
        // queued PCM can advance. Tool-Fresh already reached the backbone.
        progress = published.frame.after;
        invocation += 1;
        for index in 0..if phase == 0 { 2 } else { 3 } {
            let (kind, replay) = owner
                .prepare_next(
                    super::super::VoiceChatAdvance::default(),
                    true,
                    false,
                    capacity(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(kind, super::super::VoiceChatWorkKind::ReplayAudio);
            let output = frame(invocation, progress, Step::ReplayAudio, None);
            if discard && index == 0 {
                let staged = replay.stage(output.clone()).unwrap();
                drop(staged);
                assert_eq!(owner.owner.borrow().state.turn.frontier(), progress);
                let (retry_kind, retry) = owner
                    .prepare_next(
                        super::super::VoiceChatAdvance::default(),
                        true,
                        false,
                        capacity(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(retry_kind, kind);
                retry.stage(output.clone()).unwrap().commit();
            } else {
                let replayed = replay.stage(output.clone()).unwrap().commit();
                assert!(replayed.turn.is_none());
                assert!(replayed.tools.is_none());
                assert!(replayed.transcription.is_none());
                assert!(replayed.agent_text.is_none());
                assert!(replayed.frame.speech.is_none());
            }
            assert_eq!(output.after.audio_frames, progress.audio_frames);
            assert_eq!(output.after.rnnt_frames, progress.rnnt_frames);
            assert_eq!(
                output.after.backbone_positions,
                progress.backbone_positions + 1
            );
            progress = output.after;
            invocation += 1;
        }
        let (kind, next) = owner
            .prepare_next(
                super::super::VoiceChatAdvance::default(),
                true,
                false,
                capacity(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(kind, super::super::VoiceChatWorkKind::Live);
        drop(next);
        (owner.turn_identity(), progress)
    }
    for phase in 0..3 {
        assert_eq!(run(false, phase), run(true, phase));
    }
}

#[test]
fn acknowledgement_is_committed_speech_independent_of_function_feedback() {
    use crate::runtime::voicechat::{VoiceChatAdvance, VoiceChatWorkKind};
    use eredu_architectures::voicechat::channel_decision::ChannelDecisionPolicy;
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_runtime::execution_control::SnapshotBudget;

    let mut owner = conversation(channel(690, limits()));
    owner
        .prepare_acknowledgement(&VoiceChatPhrasePolicy::Fixed("ok".into()), 20)
        .unwrap();
    let prefill = Step::Prefill { backbone_tokens: 1 };
    let mut output = frame(0, Progress::default(), prefill, None);
    output.after.backbone_positions = 1;
    let mut progress = output.after;
    let admission = owner
        .prepare_prefill(1, Duration::ZERO, capacity())
        .unwrap();
    assert!(matches!(
        owner.prepare_acknowledgement(&VoiceChatPhrasePolicy::Disabled, 20),
        Err(VoiceChatConversationError::Busy)
    ));
    admission.stage(output).unwrap().commit();
    assert!(matches!(
        owner.prepare_acknowledgement(&VoiceChatPhrasePolicy::Disabled, 20),
        Err(VoiceChatConversationError::AlreadyPrefilled)
    ));

    let mut invocation = 1;
    let start = frame(invocation, progress, Step::LiveAudio, Some(20));
    progress = start.after;
    invocation += 1;
    owner
        .admit(start.step, capacity())
        .unwrap()
        .stage(start)
        .unwrap()
        .commit();

    // Independent expected framing: BOS, the word "ok", seventeen tail PADs, EOS.
    let expected: Vec<u32> = [1, 51].into_iter().chain([12; 17]).chain([2]).collect();
    for (index, spoken) in expected.into_iter().enumerate() {
        if index == 3 {
            let perception = frame(invocation, progress, Step::PerceptionOnly, None);
            progress = perception.after;
            invocation += 1;
            owner
                .admit(perception.step, capacity())
                .unwrap()
                .stage(perception)
                .unwrap()
                .commit();
        }
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 0,
            retained_bytes: 1 << 20,
            cumulative_copy_bytes: 1 << 20,
        });
        let saved = if index == 2 {
            let retention = budget
                .reserve(
                    SnapshotResourceKind::Snapshot,
                    Some(owner.estimate_snapshot().unwrap()),
                )
                .unwrap();
            Some((owner.copy_snapshot(&retention).unwrap(), retention))
        } else {
            None
        };
        let (kind, admission) = owner
            .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
            .unwrap()
            .unwrap();
        assert!(matches!(kind, VoiceChatWorkKind::Tool {
            decisions: ChannelDecisionPolicy::Tool { function_feedback: None, spoken_text: Some(token) }, ..
        } if token == spoken));
        let policy = admission.tool_output_policy(true);
        assert_eq!(policy.silence_on_pad, &[false]);
        assert_eq!(policy.waveform, WaveformSilence::PreserveGenerated);
        assert!(!policy.force_silence_on_eos);
        let mut output = frame(
            invocation,
            progress,
            kind.step(),
            Some(if index == 0 { 30 } else { 12 }),
        );
        output.decisions.as_mut().unwrap().spoken_text = spoken;
        assert_eq!(output.decisions.as_ref().unwrap().feedback.agent(), 12);
        if index == 1 {
            // A rejected spoken token and a discarded staged successor leave the
            // same word available to a retried neural invocation.
            let mut wrong = output.clone();
            wrong.decisions.as_mut().unwrap().spoken_text = 12;
            assert!(matches!(
                admission.stage(wrong),
                Err(VoiceChatConversationError::Acknowledgement)
            ));
            let (retry_kind, retry) = owner
                .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
                .unwrap()
                .unwrap();
            assert_eq!(retry_kind, kind);
            drop(retry.stage(output.clone()).unwrap());
            let (retry_kind, retry) = owner
                .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
                .unwrap()
                .unwrap();
            assert_eq!(retry_kind, kind);
            retry.stage(output.clone()).unwrap().commit();
        } else {
            admission.stage(output.clone()).unwrap().commit();
        }
        if let Some((saved, _retention)) = saved {
            owner.install_snapshot(saved);
            let (replayed_kind, replayed) = owner
                .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
                .unwrap()
                .unwrap();
            assert_eq!(replayed_kind, kind);
            replayed.stage(output.clone()).unwrap().commit();
        }
        progress = output.after;
        invocation += 1;
    }
    let (kind, admission) = owner
        .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
        .unwrap()
        .unwrap();
    assert!(matches!(
        kind,
        VoiceChatWorkKind::Tool {
            decisions: ChannelDecisionPolicy::Tool {
                spoken_text: None,
                ..
            },
            ..
        }
    ));
    assert_eq!(
        admission.tool_output_policy(true).waveform,
        WaveformSilence::ApplySilence
    );
    let end = frame(invocation, progress, kind.step(), Some(21));
    progress = end.after;
    invocation += 1;
    let batch = admission.stage(end).unwrap().commit().tools.unwrap();
    owner.submit_response(batch.ticket, "ok").unwrap();
    let (kind, admission) = owner
        .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
        .unwrap()
        .unwrap();
    assert!(matches!(
        kind,
        VoiceChatWorkKind::Tool {
            decisions: ChannelDecisionPolicy::Tool {
                function_feedback: Some(51),
                spoken_text: None
            },
            ..
        }
    ));
    assert_eq!(
        admission.tool_output_policy(true).waveform,
        WaveformSilence::ApplySilence
    );
    let response = frame(invocation, progress, kind.step(), Some(51));
    progress = response.after;
    invocation += 1;
    admission.stage(response).unwrap().commit();
    owner.interrupt().unwrap();
    // A new turn begins its acknowledgement at BOS, never at the old PAD tail.
    let start = frame(invocation, progress, Step::LiveAudio, Some(20));
    owner
        .admit(start.step, capacity())
        .unwrap()
        .stage(start)
        .unwrap()
        .commit();
    let (kind, admission) = owner
        .prepare_next(VoiceChatAdvance::default(), false, false, capacity())
        .unwrap()
        .unwrap();
    assert!(matches!(
        kind,
        VoiceChatWorkKind::Tool {
            decisions: ChannelDecisionPolicy::Tool {
                spoken_text: Some(1),
                ..
            },
            ..
        }
    ));
    drop(admission);
}

#[test]
fn streaming_phrase_selections_replay_across_discard_restore_and_fork() {
    use eredu_core::{
        execution_control::{SnapshotLimits, SnapshotResourceKind},
        scheduler::RequestId,
    };
    use eredu_runtime::execution_control::SnapshotBudget;

    let mut owner = conversation(channel(691, limits()));
    owner
        .prepare_acknowledgement(&VoiceChatPhrasePolicy::Streaming { seed: 7 }, 64)
        .unwrap();
    owner
        .prepare_initial_reminder(&VoiceChatPhrasePolicy::Streaming { seed: 11 }, 64)
        .unwrap();
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let estimate = owner.estimate_snapshot().unwrap();
    let saved_retention = budget
        .reserve(SnapshotResourceKind::Snapshot, Some(estimate))
        .unwrap();
    let saved = owner.copy_snapshot(&saved_retention).unwrap();
    let child_retention = budget
        .reserve(SnapshotResourceKind::Branch, Some(estimate))
        .unwrap();
    let mut child = owner
        .copy_snapshot(&child_retention)
        .unwrap()
        .fork_into(RequestId::new(692), &child_retention)
        .unwrap();

    fn next_call(
        owner: &mut VoiceChatConversation,
        invocation: u64,
        progress: &mut Progress,
        discard: bool,
    ) -> (usize, usize) {
        let output = frame(invocation, *progress, Step::LiveAudio, Some(20));
        if discard {
            let staged = owner
                .admit(output.step, capacity())
                .unwrap()
                .stage(output.clone())
                .unwrap();
            let tentative = staged
                .transition
                .candidate
                .as_ref()
                .unwrap()
                .acknowledgement
                .selected()
                .unwrap();
            let tentative_reminder = staged
                .transition
                .candidate
                .as_ref()
                .unwrap()
                .initial_reminder
                .selected()
                .unwrap();
            assert!(
                owner
                    .owner
                    .borrow()
                    .state
                    .initial_reminder
                    .selected()
                    .is_none()
            );
            assert!(
                owner
                    .owner
                    .borrow()
                    .state
                    .acknowledgement
                    .selected()
                    .is_none()
            );
            drop(staged);
            let retry = owner
                .admit(output.step, capacity())
                .unwrap()
                .stage(output.clone())
                .unwrap();
            assert_eq!(
                retry
                    .transition
                    .candidate
                    .as_ref()
                    .unwrap()
                    .acknowledgement
                    .selected(),
                Some(tentative)
            );
            assert_eq!(
                retry
                    .transition
                    .candidate
                    .as_ref()
                    .unwrap()
                    .initial_reminder
                    .selected(),
                Some(tentative_reminder)
            );
            retry.commit();
        } else {
            owner
                .admit(output.step, capacity())
                .unwrap()
                .stage(output.clone())
                .unwrap()
                .commit();
        }
        *progress = output.after;
        let selected = owner
            .owner
            .borrow()
            .state
            .acknowledgement
            .selected()
            .unwrap();
        let reminder = owner
            .owner
            .borrow()
            .state
            .initial_reminder
            .selected()
            .unwrap();
        assert!(selected < 20);
        assert!(reminder < 10);
        assert_eq!(owner.owner.borrow().state.acknowledgement_token(), Some(1));
        owner.interrupt().unwrap();
        assert!(
            owner
                .owner
                .borrow()
                .state
                .acknowledgement
                .selected()
                .is_none()
        );
        assert!(
            owner
                .owner
                .borrow()
                .state
                .initial_reminder
                .selected()
                .is_none()
        );
        (selected, reminder)
    }

    let mut progress = Progress::default();
    let mut child_progress = Progress::default();
    let choices: Vec<_> = (0..8)
        .map(|invocation| {
            let selected = next_call(&mut owner, invocation, &mut progress, true);
            assert_eq!(
                selected,
                next_call(&mut child, invocation, &mut child_progress, false)
            );
            selected
        })
        .collect();
    assert!(
        choices.iter().any(|choice| choice.0 != choices[0].0),
        "successive calls advance acknowledgment randomness"
    );
    assert!(
        choices.iter().any(|choice| choice.1 != choices[0].1),
        "successive calls advance initial reminder randomness"
    );
    owner.install_snapshot(saved);
    progress = Progress::default();
    for (invocation, selected) in choices.into_iter().enumerate() {
        assert_eq!(
            next_call(&mut owner, invocation as u64, &mut progress, false),
            selected
        );
    }
}

#[test]
fn configured_initial_reminder_survives_eotc_and_holds_ready_result_until_playback() {
    use crate::runtime::voicechat::VoiceChatInitialReminder;

    let mut owner = conversation(channel(693, limits()));
    owner
        .prepare_initial_reminder(&VoiceChatPhrasePolicy::Streaming { seed: 7 }, 64)
        .unwrap();
    let before = collect(&mut owner, 30);
    let selected = owner
        .owner
        .borrow()
        .state
        .initial_reminder
        .selected()
        .unwrap();
    let end = frame(
        2,
        before,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(21),
    );
    let mut progress = end.after;
    let batch = owner
        .admit(end.step, capacity())
        .unwrap()
        .stage(end)
        .unwrap()
        .commit()
        .tools
        .unwrap();
    assert_eq!(
        owner.owner.borrow().state.initial_reminder.selected(),
        Some(selected)
    );
    owner
        .start_wait(
            batch.ticket,
            Duration::ZERO,
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: VoiceChatInitialReminder::Configured,
                generic: &[],
                max_phrase_tokens: 64,
            },
        )
        .unwrap();
    owner.submit_response(batch.ticket, "ok").unwrap();

    // The whitespace fixture maps every released reminder word to UNK. All
    // ten phrases are shorter than 34 ASCII characters, hence 17 tail PADs.
    let words = [4, 4, 4, 6, 6, 5, 5, 5, 4, 5][selected];
    let expected = [1]
        .into_iter()
        .chain(std::iter::repeat_n(0, words))
        .chain([12; 17])
        .chain([2]);
    let mut playback = None;
    let mut next_invocation = 3;
    for spoken in expected {
        let VoiceChatToolWaitAction::Reminder(receipt) =
            owner.advance_wait(Duration::ZERO).unwrap()
        else {
            panic!("ready response displaced initial reminder");
        };
        assert_eq!(receipt.spoken(), spoken);
        let mut output = frame(next_invocation, progress, Step::Reminder, None);
        output.reminder_token = Some(spoken);
        output.after.tts_conditional_positions += 1;
        output.after.tts_unconditional_positions += 1;
        output.after.codec_frames += 1;
        output.after.output_samples += 1764;
        progress = output.after;
        next_invocation += 1;
        playback = owner
            .admit_reminder(receipt, Duration::ZERO, capacity())
            .unwrap()
            .stage(output)
            .unwrap()
            .commit()
            .playback;
        assert_eq!(playback.is_some(), spoken == 2);
    }
    assert_eq!(
        owner.advance_wait(Duration::ZERO).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly
    );
    owner
        .finish_playback(playback.unwrap(), Duration::from_secs(3))
        .unwrap();
    assert!(matches!(
        owner.advance_wait(Duration::from_secs(3)).unwrap(),
        VoiceChatToolWaitAction::Response(_)
    ));
    // Response injection and its generated EOTR clear the selection for this batch.
    for function in [51, 22] {
        let output = frame(
            next_invocation,
            progress,
            Step::Tool {
                audio: AudioSource::Retained,
            },
            Some(function),
        );
        progress = output.after;
        next_invocation += 1;
        owner
            .admit(output.step, capacity())
            .unwrap()
            .stage(output)
            .unwrap()
            .commit();
    }
    assert!(
        owner
            .owner
            .borrow()
            .state
            .initial_reminder
            .selected()
            .is_none()
    );
    assert!(owner.pending().is_none());
}
