use super::super::tests::{capacity, conversation, frame};
use super::*;
use crate::runtime::voicechat::tools::tests::{call, channel, limits};
use eredu_architectures::voicechat::schedule::Progress;
use eredu_architectures::voicechat::{
    channel_decision::ChannelDecisionPolicy, schedule::AudioSource,
};
use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
use eredu_core::scheduler::RequestId;
use eredu_runtime::execution_control::SnapshotBudget;

fn millis(value: u64) -> Duration {
    Duration::from_millis(value)
}

fn commit_reminder(
    owner: &mut VoiceChatConversation,
    invocation: u64,
    progress: &mut Progress,
    at: u64,
    spoken: u32,
) -> Option<VoiceChatReminderPlayback> {
    let VoiceChatToolWaitAction::Reminder(receipt) = owner.advance_wait(millis(at)).unwrap() else {
        panic!("expected reminder token {spoken}");
    };
    assert_eq!(receipt.spoken(), spoken);
    let mut output = frame(invocation, *progress, Step::Reminder, None);
    output.reminder_token = Some(spoken);
    output.after.tts_conditional_positions += 1;
    output.after.tts_unconditional_positions += 1;
    output.after.codec_frames += 1;
    output.after.output_samples += 1764;
    *progress = output.after;
    owner
        .admit_reminder(receipt, millis(at), capacity())
        .unwrap()
        .stage(output)
        .unwrap()
        .commit()
        .playback
}

#[test]
fn saved_midphrase_replays_receipt_playback_gap_and_timeout() {
    let mut channel = channel(611, limits());
    let ticket = call(&mut channel).ticket;
    let mut owner = conversation(channel);
    owner
        .start_wait(
            ticket,
            Duration::ZERO,
            VoiceChatToolWaitPolicy {
                generic_gap: millis(100),
                timeout: millis(200),
            },
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                generic: &["ok"],
                max_phrase_tokens: 20,
            },
        )
        .unwrap();
    let mut progress = Progress::default();
    assert!(commit_reminder(&mut owner, 0, &mut progress, 0, 1).is_none());
    let pending = owner.advance_wait(millis(80)).unwrap();
    assert!(matches!(pending, VoiceChatToolWaitAction::Reminder(step) if step.spoken() == 51));

    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let reservation = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(owner.estimate_snapshot().unwrap()),
        )
        .unwrap();
    let saved = owner.copy_snapshot(&reservation).unwrap();
    let saved_progress = progress;

    for replay in 0..2 {
        if replay == 1 {
            let restore = budget
                .reserve(
                    SnapshotResourceKind::Restore,
                    Some(saved.estimate_snapshot_value().unwrap()),
                )
                .unwrap();
            let replacement = saved.copy_snapshot_value(&restore).unwrap();
            owner.validate_snapshot_idle().unwrap();
            owner.install_snapshot(replacement);
            progress = saved_progress;
        }
        assert_eq!(owner.pending(), Some(ticket));
        assert_eq!(owner.advance_wait(millis(80)).unwrap(), pending);
        assert!(matches!(
            owner.advance_wait(millis(79)),
            Err(VoiceChatConversationError::Wait(
                VoiceChatToolWaitError::Time
            ))
        ));

        let mut playback = None;
        for (offset, spoken) in [51].into_iter().chain([12; 17]).chain([2]).enumerate() {
            let invocation = offset as u64 + 1;
            playback = commit_reminder(
                &mut owner,
                invocation,
                &mut progress,
                invocation * 80,
                spoken,
            );
        }
        let mut playback = playback.expect("initial EOS requires recorded playback completion");
        assert_eq!(
            owner.advance_wait(millis(1599)).unwrap(),
            VoiceChatToolWaitAction::PerceptionOnly
        );
        owner.finish_playback(playback, millis(1600)).unwrap();
        assert_eq!(
            owner.advance_wait(millis(1699)).unwrap(),
            VoiceChatToolWaitAction::PerceptionOnly
        );
        for (offset, spoken) in [1, 51].into_iter().chain([12; 17]).chain([2]).enumerate() {
            playback = commit_reminder(
                &mut owner,
                offset as u64 + 20,
                &mut progress,
                1700 + offset as u64 * 80,
                spoken,
            )
            .unwrap_or(playback);
        }
        owner.finish_playback(playback, millis(3300)).unwrap();
        assert_eq!(
            owner.advance_wait(millis(3499)).unwrap(),
            VoiceChatToolWaitAction::PerceptionOnly
        );
        assert!(matches!(
            owner.advance_wait(millis(3500)).unwrap(),
            VoiceChatToolWaitAction::Response(_)
        ));
        assert_eq!(
            owner.pending(),
            Some(ticket),
            "timeout injects a response for the saved ticket"
        );
    }
}

#[test]
fn fork_rekeys_wait_receipts_and_preserves_partial_response_injection() {
    let mut channel = channel(711, limits());
    let source_ticket = call(&mut channel).ticket;
    let mut source = conversation(channel);
    source
        .start_wait(
            source_ticket,
            Duration::ZERO,
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                generic: &[],
                max_phrase_tokens: 20,
            },
        )
        .unwrap();
    let mut progress = Progress::default();
    commit_reminder(&mut source, 0, &mut progress, 0, 1);
    let VoiceChatToolWaitAction::Reminder(source_step) = source.advance_wait(millis(80)).unwrap()
    else {
        panic!("expected pending phrase content");
    };

    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 0,
        max_branches: 3,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let fork = |owner: &VoiceChatConversation, request: u64| {
        let retention = budget
            .reserve(
                SnapshotResourceKind::Branch,
                Some(owner.estimate_snapshot().unwrap()),
            )
            .unwrap();
        let copied = owner.copy_snapshot(&retention).unwrap();
        (
            copied
                .fork_into(RequestId::new(request), &retention)
                .unwrap(),
            retention,
        )
    };
    let (mut child, _child_retention) = fork(&source, 712);
    assert!(matches!(
        child.submit_response(source_ticket, "ok"),
        Err(VoiceChatConversationError::Tool(
            VoiceChatToolError::StaleResult
        ))
    ));
    assert!(matches!(
        child.admit_reminder(source_step, millis(80), capacity()),
        Err(VoiceChatConversationError::Wait(
            VoiceChatToolWaitError::Receipt
        ))
    ));
    assert_eq!(
        child.advance_wait(millis(80)).unwrap(),
        VoiceChatToolWaitAction::Reminder(source_step.with_request(RequestId::new(712)))
    );
    child
        .submit_response(
            source_ticket.with_request(RequestId::new(712)),
            "<TOOL_RESPONSE> ok </TOOL_RESPONSE>",
        )
        .unwrap();
    let mut playback = None;
    for (offset, spoken) in [51].into_iter().chain([12; 17]).chain([2]).enumerate() {
        let invocation = offset as u64 + 1;
        playback = commit_reminder(
            &mut child,
            invocation,
            &mut progress,
            invocation * 80,
            spoken,
        );
    }
    let playback = playback.expect("EOS retains playback receipt");
    assert_eq!(
        source.advance_wait(millis(80)).unwrap(),
        VoiceChatToolWaitAction::Reminder(source_step)
    );
    assert!(
        source.response_policy().is_none(),
        "child result cannot modify source wait"
    );

    let (mut after_playback, _playback_retention) = fork(&child, 713);
    assert!(matches!(
        after_playback.finish_playback(playback, millis(1600)),
        Err(VoiceChatConversationError::Wait(
            VoiceChatToolWaitError::Receipt
        ))
    ));
    after_playback
        .finish_playback(playback.with_request(RequestId::new(713)), millis(1600))
        .unwrap();
    assert_eq!(
        child.advance_wait(millis(1600)).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly,
        "source still waits for its own playback"
    );
    let policy = |token| {
        Some(ChannelDecisionPolicy::Tool {
            function_feedback: Some(token),
            spoken_text: None,
        })
    };
    assert_eq!(after_playback.response_policy(), policy(50));
    let first_response = frame(
        20,
        progress,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(50),
    );
    progress = first_response.after;
    after_playback
        .admit(first_response.step, capacity())
        .unwrap()
        .stage(first_response)
        .unwrap()
        .commit();
    assert_eq!(after_playback.response_policy(), policy(51));

    let (mut response_child, _response_retention) = fork(&after_playback, 714);
    assert_eq!(
        response_child.response_policy(),
        policy(51),
        "fork must not replay the consumed response opener"
    );
    assert_eq!(
        response_child.advance_wait(millis(1600)).unwrap(),
        VoiceChatToolWaitAction::Response(policy(51).unwrap())
    );
    let next_response = frame(
        21,
        progress,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(51),
    );
    response_child
        .admit(next_response.step, capacity())
        .unwrap()
        .stage(next_response)
        .unwrap()
        .commit();
    assert_eq!(response_child.response_policy(), policy(52));
    assert_eq!(
        after_playback.response_policy(),
        policy(51),
        "fork injection positions are independent"
    );
}

#[test]
fn rejected_external_response_growth_preserves_pending_ticket_for_retry() {
    let mut channel = channel(721, limits());
    let source_ticket = call(&mut channel).ticket;
    let source = conversation(channel);
    let maximum = 1 << 20;
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 0,
        max_branches: 1,
        retained_bytes: maximum,
        cumulative_copy_bytes: maximum,
    });
    let retention = budget
        .reserve(
            SnapshotResourceKind::Branch,
            Some(source.estimate_snapshot().unwrap()),
        )
        .unwrap();
    let mut child = source
        .copy_snapshot(&retention)
        .unwrap()
        .fork_into(RequestId::new(722), &retention)
        .unwrap();
    let ticket = source_ticket.with_request(RequestId::new(722));
    let blocker = budget
        .reserve(
            SnapshotResourceKind::BranchCopy,
            Some(SnapshotEstimate {
                retained_bytes: maximum - budget.usage().retained_bytes,
                copy_bytes: 0,
            }),
        )
        .unwrap();
    let usage = budget.usage();
    let response = "ok ".repeat(128);
    assert!(matches!(
        child.submit_response(ticket, &response),
        Err(VoiceChatConversationError::Snapshot(_))
    ));
    assert_eq!(child.pending(), Some(ticket));
    assert!(child.response_policy().is_none());
    assert_eq!(
        budget.usage(),
        usage,
        "rejected state never raises the retention frontier"
    );
    drop(blocker);
    let before = budget.usage();
    child.submit_response(ticket, &response).unwrap();
    assert_eq!(
        child.response_policy(),
        Some(ChannelDecisionPolicy::Tool {
            function_feedback: Some(51),
            spoken_text: None,
        })
    );
    assert!(budget.usage().retained_bytes > before.retained_bytes);
    assert_eq!(
        budget.usage().cumulative_copy_bytes,
        before.cumulative_copy_bytes
    );
    assert!(source.response_policy().is_none());
    assert_eq!(source.pending(), Some(source_ticket));
}
