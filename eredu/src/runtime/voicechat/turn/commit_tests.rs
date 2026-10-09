use super::*;
use crate::runtime::voicechat::{
    tools::tests::{call, channel, limits},
    VoiceChatToolWait, VoiceChatToolWaitAction, VoiceChatToolWaitError, VoiceChatToolWaitMessages,
    VoiceChatToolWaitPolicy,
};
use eredu_architectures::voicechat::{
    channel_decision::FeedbackTokens, observation::TranscriptionOutput, output::OutputDecisions,
    schedule::AudioSource,
};
use std::time::Duration;

fn turn(channel: &VoiceChatToolChannel) -> VoiceChatTurnTaking {
    VoiceChatTurnTaking::new(
        channel,
        VoiceChatTurnPolicy {
            lookback: 2,
            eou_frames: 2,
            first_eou_frames: 2,
            bou_frames: 2,
            min_speech_frames: 2,
            first_min_speech_frames: 2,
            density_threshold: 0.0,
            first_turn_fallback: 100,
            ..VoiceChatTurnPolicy::default()
        },
        VoiceChatTurnLimits {
            max_audio_frames: 64,
            max_model_steps: 64,
            max_history: 2,
        },
    )
    .unwrap()
}

// Host receipts contain no native resources. These fixtures preserve the
// independently advancing source-audio and Nemotron coordinates.
fn output(
    owner: &VoiceChatTurnTaking,
    step: Step,
    blank: Option<bool>,
    tokens: Option<(u32, u32, u32)>,
) -> OwnedHostOutput<()> {
    let mut after = owner.frontier;
    let transcription = blank.map(|first_blank| {
        let frame_index = after.audio_frames;
        after.audio_frames += 1;
        after.perception_frames += 1;
        after.rnnt_frames += 1;
        TranscriptionOutput {
            source: eredu_media::audio::streaming::PcmFrameSource::Input,
            frame_index,
            sample_rate: 16_000,
            start_sample: frame_index * 1280,
            valid_samples: 1280,
            emitted: if first_blank { vec![] } else { vec![5] },
            first_blank,
            ended_blank: true,
            hit_symbol_limit: false,
        }
    });
    let decisions = tokens.map(|(raw_agent, raw_function, effective)| {
        after.backbone_positions += 1;
        OutputDecisions {
            raw_agent,
            raw_function,
            feedback: FeedbackTokens::new(effective, raw_function).unwrap(),
            spoken_text: effective,
        }
    });
    OwnedHostOutput {
        invocation_index: owner.invocation,
        step,
        reminder_token: None,
        before: owner.frontier,
        after,
        decisions,
        generated_codes: None,
        speech: None,
        transcription,
        diagnostics: vec![],
    }
}

#[test]
fn live_resolver_is_unpublished_and_replay_matches_committed_turn_state() {
    fn run(
        abandon: bool,
    ) -> (
        State,
        Progress,
        u64,
        VoiceChatTurnIdentity,
        Vec<VoiceChatTurnEvent>,
    ) {
        let mut channel = channel(501, limits());
        let mut owner = turn(&channel);
        let initial_identity = channel.turn_identity();
        let mut events = Vec::new();
        // Two voiced frames confirm the first user; two blanks force BOS.
        // Two further voiced frames force an interruption of the active agent.
        for (blank, raw, expected) in [
            (false, 12, 12),
            (false, 12, 12),
            (true, 12, 12),
            (true, 12, 1),
            (false, 51, 51),
            (false, 51, 2),
        ] {
            let identity = channel.turn_identity();
            let input = DecisionInput {
                raw_agent: raw,
                raw_function: 12,
                evidence: DecisionEvidence {
                    first_blank: Some(blank),
                    audio_frame_index: Some(owner.state.audio_frames),
                },
            };
            let before = owner.state.clone();
            let resolver = owner.resolver(&channel).unwrap();
            let admission = resolver.admission();
            resolver
                .preflight(TokenDomain::new(TEXT_VOCABULARY_SIZE))
                .unwrap();
            let policy = resolver.resolve(input).unwrap();
            assert_eq!(
                policy,
                ChannelDecisionPolicy::Main {
                    agent_feedback: Some(expected),
                    function_feedback: None,
                    spoken_text: None,
                }
            );
            if abandon {
                // A discarded neural candidate may have resolved policy already;
                // replay still reads identical canonical state and yields the same override.
                assert_eq!(resolver.resolve(input).unwrap(), policy);
            }
            assert_eq!(owner.state, before);
            let receipt = output(
                &owner,
                Step::LiveAudio,
                Some(blank),
                Some((raw, 12, expected)),
            );
            if abandon {
                let mut wrong = receipt.clone();
                wrong.decisions.as_mut().unwrap().feedback = FeedbackTokens::new(11, 12).unwrap();
                assert!(matches!(
                    owner.observe_committed(&mut channel, admission, &wrong),
                    Err(VoiceChatTurnError::Attribution)
                ));
                assert_eq!(owner.state, before);
                assert_eq!(channel.turn_identity(), identity);
            }
            if let Some(event) = owner
                .observe_committed(&mut channel, admission, &receipt)
                .unwrap()
            {
                events.push(event);
            }
            let committed = owner.state.clone();
            assert!(
                owner
                    .observe_committed(&mut channel, admission, &receipt)
                    .is_err(),
                "duplicates or old-turn callbacks cannot advance state"
            );
            assert_eq!(owner.state, committed);
        }
        assert_eq!(
            events,
            [
                VoiceChatTurnEvent::UserEnded,
                VoiceChatTurnEvent::Interrupted
            ]
        );
        assert_eq!(
            owner.state.history.iter().copied().collect::<Vec<_>>(),
            [51, 2]
        );
        assert_ne!(owner.identity, initial_identity);
        (
            owner.state,
            owner.frontier,
            owner.invocation,
            owner.identity,
            events,
        )
    }
    assert_eq!(run(false), run(true));
}

#[test]
fn tool_wait_audio_and_model_frontiers_advance_independently_and_interruption_invalidates_work() {
    let mut channel = channel(502, limits());
    let mut owner = turn(&channel);
    let ticket = call(&mut channel).ticket;
    let mut wait = VoiceChatToolWait::new(
        &channel,
        ticket,
        Duration::ZERO,
        VoiceChatToolWaitPolicy::default(),
        VoiceChatToolWaitMessages {
            initial: crate::runtime::voicechat::VoiceChatInitialReminder::Disabled,
            generic: &[],
            max_phrase_tokens: 64,
        },
    )
    .unwrap();
    assert_eq!(
        wait.advance(&mut channel, Duration::ZERO).unwrap(),
        VoiceChatToolWaitAction::PerceptionOnly
    );
    for blank in [false, true] {
        let receipt = output(&owner, Step::PerceptionOnly, Some(blank), None);
        let admission = owner.admission(&channel, receipt.step).unwrap();
        assert_eq!(
            owner
                .observe_committed(&mut channel, admission, &receipt)
                .unwrap(),
            None
        );
    }
    assert_eq!((owner.state.audio_frames, owner.state.model_steps), (2, 0));
    for _ in 0..3 {
        let receipt = output(
            &owner,
            Step::Tool {
                audio: AudioSource::Retained,
            },
            None,
            Some((51, 12, 12)),
        );
        let admission = owner.admission(&channel, receipt.step).unwrap();
        assert_eq!(
            owner
                .observe_committed(&mut channel, admission, &receipt)
                .unwrap(),
            None
        );
    }
    assert_eq!((owner.state.audio_frames, owner.state.model_steps), (2, 3));
    assert_eq!(
        (
            owner.frontier.audio_frames,
            owner.frontier.backbone_positions
        ),
        (2, 3)
    );
    let stale_admission = owner.admission(&channel, Step::LiveAudio).unwrap();
    let pending = output(&owner, Step::LiveAudio, Some(false), Some((51, 12, 12)));
    let retained = owner.state.clone();
    owner.interrupt(&mut channel).unwrap();
    assert_eq!(
        owner.state, retained,
        "host interruption preserves delivered acoustic/model state"
    );
    assert!(matches!(
        channel.submit_response(ticket, "<TOOL_RESPONSE> ok </TOOL_RESPONSE>"),
        Err(VoiceChatToolError::StaleResult)
    ));
    assert!(matches!(
        wait.advance(&mut channel, Duration::from_millis(80)),
        Err(VoiceChatToolWaitError::Stale)
    ));
    assert!(matches!(
        owner.observe_committed(&mut channel, stale_admission, &pending),
        Err(VoiceChatTurnError::Stale)
    ));
    assert_eq!(owner.state, retained);
    assert_eq!(channel.pending(), None);
}

#[test]
fn live_commit_retains_admitted_tool_activity_when_response_finishes_before_observation() {
    let mut channel = channel(503, limits());
    let mut owner = turn(&channel);
    let ticket = call(&mut channel).ticket;
    let identity = channel.turn_identity();
    let resolver = owner.resolver(&channel).unwrap();
    let admission = resolver.admission();
    let input = DecisionInput {
        raw_agent: 1,
        raw_function: 12,
        evidence: DecisionEvidence {
            first_blank: Some(true),
            audio_frame_index: Some(0),
        },
    };
    assert_eq!(
        resolver.resolve(input).unwrap(),
        ChannelDecisionPolicy::Main {
            agent_feedback: Some(12),
            function_feedback: None,
            spoken_text: None,
        }
    );
    let receipt = output(&owner, Step::LiveAudio, Some(true), Some((1, 12, 12)));
    // External result arrival and an already-admitted function-channel stream
    // can finish while the live receipt remains unpublished at the host layer.
    channel
        .submit_response(ticket, "<TOOL_RESPONSE> ok </TOOL_RESPONSE>")
        .unwrap();
    for token in [50, 51, 52, 22] {
        assert!(channel
            .observe_committed_function(identity, token)
            .unwrap()
            .is_none());
    }
    assert!(!channel.is_active());
    assert_eq!(channel.turn_identity(), identity);
    assert_eq!(
        owner
            .observe_committed(&mut channel, admission, &receipt)
            .unwrap(),
        None
    );
    assert_eq!(
        owner.state.history.iter().copied().collect::<Vec<_>>(),
        [12]
    );
    assert_eq!((owner.state.audio_frames, owner.state.model_steps), (1, 1));
}

#[test]
fn main_pad_tail_and_tool_idle_gates_use_prior_committed_state() {
    let mut channel = channel(504, limits());
    let mut owner = VoiceChatTurnTaking::new(
        &channel,
        VoiceChatTurnPolicy {
            enabled: false,
            pad_tail_ratio: 1.0,
            ..VoiceChatTurnPolicy::default()
        },
        VoiceChatTurnLimits {
            max_audio_frames: 16,
            max_model_steps: 16,
            max_history: 40,
        },
    )
    .unwrap();
    // BOS opens a turn. One content token permits a PAD tail through equality;
    // only the third PAD sees a prior counter strictly greater than the ratio.
    for (token, silence_pad, silence_tool_pad) in [
        (1, true, true),
        (51, false, false),
        (12, false, false),
        (12, false, false),
        (12, true, false),
        (2, true, false),
        (12, true, true),
    ] {
        let resolver = owner.resolver(&channel).unwrap();
        let admission = resolver.admission();
        let policy = resolver.output_policy(true, true);
        assert_eq!(policy.silence_on_pad, &[silence_pad]);
        assert!(policy.force_silence_on_eos);
        assert_eq!(policy.waveform, WaveformSilence::PreserveGenerated);
        assert_eq!(resolver.output_policy(true, false).silence_on_pad, &[false]);
        let tool_policy = owner.tool_output_policy(true);
        assert_eq!(tool_policy.silence_on_pad, &[silence_tool_pad]);
        assert!(!tool_policy.force_silence_on_eos);
        assert_eq!(tool_policy.waveform, WaveformSilence::ApplySilence);
        assert_eq!(owner.tool_output_policy(false).silence_on_pad, &[false]);
        let input = DecisionInput {
            raw_agent: token,
            raw_function: 12,
            evidence: DecisionEvidence {
                first_blank: Some(true),
                audio_frame_index: Some(owner.state.audio_frames),
            },
        };
        assert_eq!(
            resolver.resolve(input).unwrap(),
            ChannelDecisionPolicy::Main {
                agent_feedback: Some(token),
                function_feedback: None,
                spoken_text: None,
            }
        );
        let receipt = output(
            &owner,
            Step::LiveAudio,
            Some(true),
            Some((token, 12, token)),
        );
        owner
            .observe_committed(&mut channel, admission, &receipt)
            .unwrap();
    }
    assert!(owner.state.idle);
    assert_eq!((owner.state.content, owner.state.pads), (0, 1));
}

#[test]
fn admission_checks_only_resources_consumed_by_the_selected_step() {
    let mut channel = channel(505, limits());
    let mut owner = VoiceChatTurnTaking::new(
        &channel,
        VoiceChatTurnPolicy::default(),
        VoiceChatTurnLimits {
            max_audio_frames: 1,
            max_model_steps: 1,
            max_history: 40,
        },
    )
    .unwrap();
    let receipt = output(&owner, Step::PerceptionOnly, Some(true), None);
    let admission = owner.admission(&channel, receipt.step).unwrap();
    owner
        .observe_committed(&mut channel, admission, &receipt)
        .unwrap();
    assert!(matches!(
        owner.admission(&channel, Step::PerceptionOnly),
        Err(VoiceChatTurnError::Capacity("audio frames"))
    ));
    assert!(matches!(
        owner.admission(
            &channel,
            Step::Tool {
                audio: AudioSource::Fresh
            }
        ),
        Err(VoiceChatTurnError::Capacity("audio frames"))
    ));
    // Exhausted live-audio capacity does not consume the independent model budget.
    let tool = Step::Tool {
        audio: AudioSource::Retained,
    };
    let admission = owner.admission(&channel, tool).unwrap();
    let receipt = output(&owner, tool, None, Some((51, 12, 12)));
    owner
        .observe_committed(&mut channel, admission, &receipt)
        .unwrap();
    assert!(matches!(
        owner.admission(&channel, tool),
        Err(VoiceChatTurnError::Capacity("model steps"))
    ));
    assert_eq!((owner.state.audio_frames, owner.state.model_steps), (1, 1));
    // Speech and terminal overlap can still finish without inventing more audio
    // input or backbone positions when both of those capacities are exhausted.
    for step in [Step::Reminder, Step::Drain] {
        let admission = owner.admission(&channel, step).unwrap();
        let receipt = output(&owner, step, None, None);
        owner
            .observe_committed(&mut channel, admission, &receipt)
            .unwrap();
    }
    assert_eq!((owner.state.audio_frames, owner.state.model_steps), (1, 1));
}
