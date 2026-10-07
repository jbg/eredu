use super::*;
use crate::runtime::voicechat::tools::tests::{call, channel, limits};

fn millis(value: u64) -> Duration {
    Duration::from_millis(value)
}

fn reminder(
    wait: &mut VoiceChatToolWait,
    channel: &mut VoiceChatToolChannel,
    at: u64,
) -> VoiceChatReminderStep {
    match wait.advance(channel, millis(at)).unwrap() {
        VoiceChatToolWaitAction::Reminder(step) => step,
        other => panic!("expected reminder, received {other:?}"),
    }
}

// Each compact phrase has BOS, one content token, seventeen tail PADs, EOS.
fn finish_phrase(
    wait: &mut VoiceChatToolWait,
    channel: &mut VoiceChatToolChannel,
    start: u64,
) -> Vec<u32> {
    let mut spoken = Vec::new();
    let mut playback = None;
    for frame in 0..20 {
        let at = start + frame * 80;
        let step = reminder(wait, channel, at);
        spoken.push(step.spoken());
        playback = wait.commit_reminder(channel, step, millis(at)).unwrap();
    }
    let playback = playback.expect("EOS declares the outstanding playback");
    assert_eq!(
        wait.advance(channel, millis(start + 1999)).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly
    );
    wait.finish_playback(channel, playback, millis(start + 2000))
        .unwrap();
    assert!(matches!(
        wait.finish_playback(channel, playback, millis(start + 2000)),
        Err(VoiceChatToolWaitError::Receipt)
    ));
    spoken
}

#[test]
fn default_wait_times_gaps_and_final_timeout_from_reminder_playback() {
    let mut channel = channel(101, limits());
    let ticket = call(&mut channel).ticket;
    let messages = VoiceChatToolWaitMessages {
        initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
        generic: &["one", "two"],
        max_phrase_tokens: 20,
    };
    let mut wait = VoiceChatToolWait::new(
        &channel,
        ticket,
        millis(0),
        VoiceChatToolWaitPolicy::default(),
        messages,
    )
    .unwrap();
    let spoken = finish_phrase(&mut wait, &mut channel, 0);
    let mut expected = vec![1, 51];
    expected.extend([12; 17]);
    expected.push(2);
    assert_eq!(spoken, expected);
    for (before, start) in [(2999, 3000), (5999, 6000)] {
        assert_eq!(
            wait.advance(&mut channel, millis(before)).unwrap(),
            VoiceChatToolWaitAction::PerceptionOnly
        );
        let spoken = finish_phrase(&mut wait, &mut channel, start);
        assert_eq!(spoken.first(), Some(&1));
        assert_eq!(spoken.last(), Some(&2));
    }
    // Final speech playback ends at 8 s. The silent timeout is a separate 15 s,
    // rather than a deadline measured from the call or the first reminder.
    for at in [15_000, 22_999] {
        assert_eq!(
            wait.advance(&mut channel, millis(at)).unwrap(),
            VoiceChatToolWaitAction::PerceptionOnly
        );
        assert!(!wait.timed_out());
    }
    assert!(matches!(
        wait.advance(&mut channel, millis(23_000)).unwrap(),
        VoiceChatToolWaitAction::Response(_)
    ));
    assert!(wait.timed_out());

    let mut another = crate::runtime::voicechat::tools::tests::channel(102, limits());
    let ticket = call(&mut another).ticket;
    for generic in [&["one", "one"][..], &["one", "two", "three"][..]] {
        assert!(matches!(
            VoiceChatToolWait::new(
                &another,
                ticket,
                millis(0),
                VoiceChatToolWaitPolicy::default(),
                VoiceChatToolWaitMessages {
                    generic,
                    ..messages
                },
            ),
            Err(VoiceChatToolWaitError::Messages)
        ));
    }
}

#[test]
fn early_result_finishes_initial_speech_and_receipts_resist_retry_and_interruption() {
    let mut channel = channel(103, limits());
    let ticket = call(&mut channel).ticket;
    let mut wait = VoiceChatToolWait::new(
        &channel,
        ticket,
        millis(0),
        VoiceChatToolWaitPolicy::default(),
        VoiceChatToolWaitMessages {
            initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
            generic: &["later"],
            max_phrase_tokens: 20,
        },
    )
    .unwrap();
    channel.submit_response(ticket, "ok").unwrap();
    let abandoned = reminder(&mut wait, &mut channel, 0);
    assert_eq!(reminder(&mut wait, &mut channel, 0), abandoned);
    wait.commit_reminder(&channel, abandoned, millis(80))
        .unwrap();
    assert!(matches!(
        wait.commit_reminder(&channel, abandoned, millis(80)),
        Err(VoiceChatToolWaitError::Receipt)
    ));
    assert!(matches!(
        wait.advance(&mut channel, millis(79)),
        Err(VoiceChatToolWaitError::Time)
    ));
    let mut playback = None;
    for frame in 1..20 {
        let at = 80 + frame * 80;
        let step = reminder(&mut wait, &mut channel, at);
        if frame == 1 {
            assert_eq!(step.spoken(), 51);
            assert!(matches!(
                wait.commit_reminder(&channel, step, millis(at - 1)),
                Err(VoiceChatToolWaitError::Time)
            ));
        }
        playback = wait.commit_reminder(&channel, step, millis(at)).unwrap();
    }
    assert_eq!(
        wait.advance(&mut channel, millis(1999)).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly
    );
    wait.finish_playback(&channel, playback.unwrap(), millis(2000))
        .unwrap();
    // The initial EOS commits even though the result preceded its BOS. The
    // unstarted generic phrase is skipped immediately, without its 1 s gap.
    assert_eq!(
        wait.advance(&mut channel, millis(2000)).unwrap(),
        VoiceChatToolWaitAction::Response(ChannelDecisionPolicy::Tool {
            function_feedback: Some(51),
            spoken_text: None,
        })
    );
    channel.interrupt().unwrap();
    assert!(matches!(
        wait.commit_reminder(&channel, abandoned, millis(2000)),
        Err(VoiceChatToolWaitError::Stale)
    ));
    assert!(matches!(
        wait.advance(&mut channel, millis(2000)),
        Err(VoiceChatToolWaitError::Stale)
    ));
    assert!(matches!(
        channel.submit_response(ticket, "ok"),
        Err(VoiceChatToolError::StaleResult)
    ));
}

#[test]
fn speech_disabled_timeout_uses_committed_response_and_rejects_late_results() {
    let mut channel = channel(104, limits());
    let ticket = call(&mut channel).ticket;
    let mut wait = VoiceChatToolWait::new(
        &channel,
        ticket,
        millis(500),
        VoiceChatToolWaitPolicy::default(),
        VoiceChatToolWaitMessages {
            initial: crate::runtime::voicechat::VoiceChatInitialReminder::Disabled,
            generic: &[],
            max_phrase_tokens: 20,
        },
    )
    .unwrap();
    assert_eq!(
        wait.advance(&mut channel, millis(15_499)).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly
    );
    let response = wait.advance(&mut channel, millis(15_500)).unwrap();
    assert!(matches!(response, VoiceChatToolWaitAction::Response(_)));
    assert_eq!(
        wait.advance(&mut channel, millis(15_500)).unwrap(),
        response
    );
    assert!(matches!(
        channel.submit_response(ticket, "ok"),
        Err(VoiceChatToolError::StaleResult)
    ));
    let mut tokens = 0;
    loop {
        let VoiceChatToolWaitAction::Response(ChannelDecisionPolicy::Tool {
            function_feedback,
            ..
        }) = wait.advance(&mut channel, millis(15_500)).unwrap()
        else {
            panic!("timeout response was discarded before EOTR");
        };
        let Some(token) = function_feedback else {
            break;
        };
        tokens += 1;
        assert!(tokens < 64);
        channel
            .observe_committed_function(channel.turn_identity(), token)
            .unwrap();
    }
    assert!(tokens > 0);
    assert_eq!(channel.pending(), Some(ticket));
    channel
        .observe_committed_function(channel.turn_identity(), 22)
        .unwrap();
    assert_eq!(
        wait.advance(&mut channel, millis(15_500)).unwrap(),
        VoiceChatToolWaitAction::Complete
    );
}

#[test]
fn admitted_generic_speech_survives_result_arrival_and_discard_invalidates_receipt() {
    let mut channel = channel(106, limits());
    let ticket = call(&mut channel).ticket;
    let mut wait = VoiceChatToolWait::new(
        &channel,
        ticket,
        millis(0),
        VoiceChatToolWaitPolicy::default(),
        VoiceChatToolWaitMessages {
            initial: crate::runtime::voicechat::VoiceChatInitialReminder::Disabled,
            generic: &["ok"],
            max_phrase_tokens: 20,
        },
    )
    .unwrap();
    let abandoned = reminder(&mut wait, &mut channel, 1000);
    wait.discard_reminder(&channel, abandoned, millis(1000))
        .unwrap();
    let admitted = reminder(&mut wait, &mut channel, 1000);
    assert_ne!(abandoned, admitted);
    assert!(matches!(
        wait.commit_reminder(&channel, abandoned, millis(1000)),
        Err(VoiceChatToolWaitError::Receipt)
    ));
    channel.submit_response(ticket, "ok").unwrap();
    assert_eq!(
        wait.advance(&mut channel, millis(1080)).unwrap(),
        VoiceChatToolWaitAction::Reminder(admitted)
    );
    // An admitted transition retains priority over the later result until its
    // native branch is explicitly discarded. It must not be silently lost.
    wait.discard_reminder(&channel, admitted, millis(1080))
        .unwrap();
    assert!(matches!(
        wait.advance(&mut channel, millis(1080)).unwrap(),
        VoiceChatToolWaitAction::Response(_)
    ));
    assert!(matches!(
        wait.commit_reminder(&channel, admitted, millis(1080)),
        Err(VoiceChatToolWaitError::Receipt)
    ));
}

#[test]
fn recorded_wait_replay_matches_with_abandoned_controlled_admissions() {
    #[derive(Debug, PartialEq, Eq)]
    enum RecordedAction {
        Spoken(u32),
        Response(ChannelDecisionPolicy),
    }

    fn replay(controlled: bool) -> Vec<(u64, RecordedAction)> {
        let mut channel = channel(105, limits());
        let ticket = call(&mut channel).ticket;
        let mut wait = VoiceChatToolWait::new(
            &channel,
            ticket,
            millis(0),
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                generic: &["later"],
                max_phrase_tokens: 20,
            },
        )
        .unwrap();
        let mut recorded = Vec::new();
        for frame in 0..=21 {
            let at = frame * 80;
            if frame == 2 {
                channel.submit_response(ticket, "ok").unwrap();
            }
            let mut action = wait.advance(&mut channel, millis(at)).unwrap();
            if controlled {
                // An unpublished candidate and repeated host polls leave both
                // speech and function feedback available to the later commit.
                assert_eq!(wait.advance(&mut channel, millis(at)).unwrap(), action);
                assert_eq!(wait.advance(&mut channel, millis(at)).unwrap(), action);
                if frame % 3 == 0 {
                    if let VoiceChatToolWaitAction::Reminder(abandoned) = action {
                        wait.discard_reminder(&channel, abandoned, millis(at))
                            .unwrap();
                        let replacement = reminder(&mut wait, &mut channel, at);
                        assert_ne!(replacement, abandoned);
                        assert_eq!(replacement.spoken(), abandoned.spoken());
                        action = VoiceChatToolWaitAction::Reminder(replacement);
                    }
                }
            }
            let recorded_action = match action {
                VoiceChatToolWaitAction::Reminder(step) => {
                    if let Some(playback) =
                        wait.commit_reminder(&channel, step, millis(at)).unwrap()
                    {
                        wait.finish_playback(&channel, playback, millis(at))
                            .unwrap();
                    }
                    RecordedAction::Spoken(step.spoken())
                }
                VoiceChatToolWaitAction::Response(
                    policy @ ChannelDecisionPolicy::Tool {
                        function_feedback, ..
                    },
                ) => {
                    channel
                        .observe_committed_function(
                            channel.turn_identity(),
                            function_feedback.unwrap_or(22),
                        )
                        .unwrap();
                    RecordedAction::Response(policy)
                }
                other => panic!("unexpected timeline action {other:?}"),
            };
            recorded.push((at, recorded_action));
        }
        assert_eq!(
            wait.advance(&mut channel, millis(1760)).unwrap(),
            VoiceChatToolWaitAction::Complete
        );
        recorded
    }
    assert_eq!(replay(false), replay(true));
}
