use super::*;
use crate::runtime::voicechat::{VoiceChatConversationOutput, VoiceChatPublicationReceiver};
use eredu_core::execution_control::SnapshotLimits;

fn limits() -> SnapshotLimits {
    SnapshotLimits {
        max_snapshots: 4,
        max_branches: 0,
        retained_bytes: 16 << 20,
        cumulative_copy_bytes: 64 << 20,
    }
}

fn outputs(receiver: &mut VoiceChatPublicationReceiver<()>) -> Vec<String> {
    let mut outputs = vec![];
    while let Some(output) = receiver.pop_output() {
        outputs.push(format!("{output:?}"));
    }
    outputs
}

#[test]
fn snapshot_replays_partial_pcm_and_undelivered_speech_without_rewinding_budgets() {
    let run = |replay: bool| {
        let (state, mut input, mut receiver, _) = fixture_state();
        input.submit(&[0.375; 1500], INPUT_SAMPLE_RATE).unwrap();
        input
            .finish(PcmFinishPolicy {
                final_frame: FinalPcmFrame::ZeroPad,
                trailing_silence_frames: 0,
            })
            .unwrap();
        let mut session = VoiceChatSession::start(state, config()).unwrap();
        session.enable_snapshots(limits()).unwrap();
        assert!(matches!(
            session.snapshot(),
            Err(VoiceChatSessionError::Busy)
        ));
        assert_eq!(session.snapshot_usage().unwrap().cumulative_copy_bytes, 0);
        let mut position = Position::default();
        let advance = |session: &mut VoiceChatSession<
            crate::runtime::voicechat::ingress::tests::Neural,
            (),
        >,
                       position: &mut Position,
                       tick| {
            session
                .advance(tick, |work, branch| {
                    let function = matches!(work.kind(), VoiceChatWorkKind::Live).then_some(12);
                    let mut output = position.output(work, function);
                    if function.is_some() {
                        output.speech = Some(
                            eredu_architectures::voicechat::observation::OwnedSpeechChunk {
                                samples: vec![0.125, -0.5, 0.75],
                                sample_rate: 22050,
                                start_sample: output.before.output_samples,
                                delay_samples: 0,
                                is_tail: false,
                            },
                        );
                        output.after.output_samples += 3;
                    }
                    let after = output.after;
                    let submitted = submit(work, branch, output, Rc::new(Cell::new(true)))?;
                    position.invocation += 1;
                    position.progress = after;
                    Ok(submitted)
                })
                .unwrap()
        };
        advance(&mut session, &mut position, tick(0));
        receiver.pop_output().unwrap();
        advance(&mut session, &mut position, tick(80));
        assert_eq!(input.status().queued_samples, 220);
        let snapshot = replay.then(|| session.snapshot().unwrap());
        let saved_position = Position {
            invocation: position.invocation,
            progress: position.progress,
        };
        advance(&mut session, &mut position, tick(160));
        if let Some(snapshot) = &snapshot {
            let copies = session.snapshot_usage().unwrap().cumulative_copy_bytes;
            let commits = session.report().completed_work;
            session.restore(snapshot).unwrap();
            assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > copies);
            assert_eq!(session.report().completed_work, commits);
            assert_eq!(receiver.usage().queued_publications, 1);
            assert_eq!(input.status().queued_samples, 220);
            assert_eq!(
                session
                    .mutable_state()
                    .unwrap()
                    .neural_state()
                    .model_state()
                    .frames,
                1
            );
            position = saved_position;
            advance(&mut session, &mut position, tick(160));
        }
        advance(&mut session, &mut position, tick(240));
        assert_eq!(session.status(), VoiceChatSessionStatus::Finished);
        assert_eq!(input.status().committed_samples, 1500);
        let result = outputs(&mut receiver);
        assert_eq!(result.len(), 3);
        drop(snapshot);
        assert_eq!(session.snapshot_usage().unwrap().snapshots, 0);
        assert_eq!(session.snapshot_usage().unwrap().retained_bytes, 0);
        result
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn snapshot_rejects_irreversible_frontiers_and_native_restore_is_atomic() {
    let (state, mut input, mut receiver, reject) = fixture_state();
    input.submit(&[0.25; 1280], INPUT_SAMPLE_RATE).unwrap();
    let mut session = VoiceChatSession::start(state, config()).unwrap();
    let mut position = Position::default();
    session
        .advance(tick(0), |work, branch| {
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap();
    receiver.pop_output().unwrap();
    session.enable_snapshots(limits()).unwrap();
    let saved = session.snapshot().unwrap();
    assert!(matches!(
        session.enable_snapshots(limits()),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::AlreadyConfigured
        ))
    ));
    let (foreign_state, _, _, _) = fixture_state();
    let mut foreign = VoiceChatSession::start(foreign_state, config()).unwrap();
    let mut foreign_position = Position::default();
    foreign
        .advance(tick(0), |work, branch| {
            execute(
                &mut foreign_position,
                Rc::new(Cell::new(true)),
                work,
                branch,
            )
        })
        .unwrap();
    assert!(matches!(
        foreign.restore(&saved),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::Foreign
        ))
    ));
    session
        .advance(tick(80), |work, branch| {
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap();
    let before = input.status();
    let usage = receiver.usage();
    reject.set(true);
    let error = session.restore(&saved).unwrap_err();
    let mut source: &(dyn Error + 'static) = &error;
    while !source.is::<CommitFailure>() {
        source = source.source().expect("native source retained");
    }
    assert_eq!(input.status(), before);
    assert_eq!(receiver.usage(), usage);
    assert_eq!(
        session
            .mutable_state()
            .unwrap()
            .neural_state()
            .model_state()
            .frames,
        1
    );
    assert!(matches!(
        session.poll(Duration::ZERO),
        Err(VoiceChatSessionError::Time)
    ));
    reject.set(false);
    session.restore(&saved).unwrap();
    let copies = session.snapshot_usage().unwrap().cumulative_copy_bytes;
    input.submit(&[0.5], INPUT_SAMPLE_RATE).unwrap();
    assert!(matches!(
        session.restore(&saved),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::InputAdvanced
        ))
    ));
    assert_eq!(input.status().queued_samples, 1281);
    assert_eq!(
        session.snapshot_usage().unwrap().cumulative_copy_bytes,
        copies
    );
    let saved = session.snapshot().unwrap();
    session.interrupt().unwrap();
    assert!(matches!(
        session.restore(&saved),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::HostChanged
        ))
    ));
    let saved = session.snapshot().unwrap();
    input
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    assert!(matches!(
        session.restore(&saved),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::InputAdvanced
        ))
    ));
}

#[test]
fn snapshot_retains_complete_tool_calls_but_never_replays_delivered_calls() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input.submit(&[0.25; 1280], INPUT_SAMPLE_RATE).unwrap();
    let mut session = VoiceChatSession::start(state, config()).unwrap();
    assert!(!session.has_tool_work());
    let mut position = Position::default();
    session.enable_snapshots(limits()).unwrap();
    let mut incomplete = None;
    for (at, function) in [(0, None), (80, Some(20)), (160, Some(30)), (240, Some(21))] {
        // Consume earlier outputs so the saved queue contains the complete batch.
        outputs(&mut receiver);
        let mut advance = tick(at);
        advance.tool_audio = VoiceChatToolAudioPolicy::RealAudio;
        session
            .advance(advance, |work, branch| {
                let output = position.output(work, function);
                let after = output.after;
                let submitted = submit(work, branch, output, Rc::new(Cell::new(true)))?;
                position.invocation += 1;
                position.progress = after;
                Ok(submitted)
            })
            .unwrap();
        if at == 160 {
            assert!(session.has_tool_work());
            outputs(&mut receiver);
            incomplete = Some((
                session.snapshot().unwrap(),
                Position {
                    invocation: position.invocation,
                    progress: position.progress,
                },
            ));
        }
    }
    let (incomplete, mut position) = incomplete.unwrap();
    session.restore(&incomplete).unwrap();
    assert!(session.has_tool_work());
    assert!(receiver.pop_output().is_none());
    session
        .advance(tick(240), |work, branch| {
            let output = position.output(work, Some(21));
            let after = output.after;
            let submitted = submit(work, branch, output, Rc::new(Cell::new(true)))?;
            position.invocation += 1;
            position.progress = after;
            Ok(submitted)
        })
        .unwrap();
    let saved = session.snapshot().unwrap();
    session.restore(&saved).unwrap();
    let VoiceChatConversationOutput {
        tools: Some(batch), ..
    } = receiver.try_pop_output().unwrap().unwrap()
    else {
        panic!("complete call")
    };
    assert!(batch.events.iter().any(|event| matches!(event, eredu_core::generation::SemanticEvent::ToolCallStart { name, .. } if name == "weather")));
    let copies = session.snapshot_usage().unwrap().cumulative_copy_bytes;
    assert!(matches!(
        session.restore(&saved),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::OutputDelivered
        ))
    ));
    assert_eq!(
        session.snapshot_usage().unwrap().cumulative_copy_bytes,
        copies
    );
    assert!(receiver.pop_output().is_none());
    session.submit_response(batch.ticket, "ok").unwrap();
    assert!(session.has_tool_work());
    let saved = session.snapshot().unwrap();
    session.interrupt().unwrap();
    assert!(!session.has_tool_work());
    assert!(matches!(
        session.restore(&saved),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::HostChanged
        ))
    ));
    assert!(session.submit_response(batch.ticket, "stale").is_err());
}
