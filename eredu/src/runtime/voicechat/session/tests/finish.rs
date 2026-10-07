use super::*;
use crate::runtime::voicechat::{
    VoiceChatConversationOutput, VoiceChatPublicationReceiver, ingress::tests::Neural,
};
use eredu_architectures::voicechat::schedule::Step;
use eredu_core::{execution_control::SnapshotLimits, scheduler::RequestId};
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};

type Session = VoiceChatSession<Neural, ()>;

fn finish_policy() -> PcmFinishPolicy {
    PcmFinishPolicy {
        final_frame: FinalPcmFrame::ZeroPad,
        trailing_silence_frames: 2,
    }
}

fn advance(session: &mut Session, position: &mut Position, millis: u64) {
    let progress = session
        .advance(tick(millis), |work, branch| {
            execute(position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap();
    assert_eq!((progress.submitted, progress.committed), (1, 1));
}

fn outputs(
    receiver: &mut VoiceChatPublicationReceiver<()>,
) -> Vec<VoiceChatConversationOutput<()>> {
    std::iter::from_fn(|| receiver.pop_output()).collect()
}

#[test]
fn partial_finish_continuation_and_automatic_drain_match_iterator_advancement() {
    let run = |iterator: bool| {
        let (state, mut input, mut receiver, _) = fixture_state();
        input.submit(&[0.25; 500], INPUT_SAMPLE_RATE).unwrap();
        input.submit(&[0.5; 1000], INPUT_SAMPLE_RATE).unwrap();
        input.finish(finish_policy()).unwrap();
        let mut session = Session::start(state, config()).unwrap();
        let mut position = Position::default();
        let mut events = vec![];
        let mut execute_next = |work: &VoiceChatWork<()>, branch: &mut Branch| {
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        };
        let ticks = (0..6).map(|index| tick(index * 80));
        let progress = if iterator {
            session
                .drive(ticks, &mut execute_next)
                .map(|progress| {
                    events.extend(outputs(&mut receiver));
                    progress.unwrap()
                })
                .collect::<Vec<_>>()
        } else {
            ticks
                .map(|tick| {
                    let progress = session.advance(tick, &mut execute_next).unwrap();
                    events.extend(outputs(&mut receiver));
                    progress
                })
                .collect::<Vec<_>>()
        };
        assert!(
            progress
                .iter()
                .all(|step| step.submitted == 1 && step.committed == 1)
        );
        assert_eq!(session.status(), VoiceChatSessionStatus::Finished);
        let status = input.status();
        assert_eq!(
            (status.accepted_samples, status.committed_samples),
            (1500, 1500)
        );
        assert_eq!(
            (
                status.committed_frames,
                status.committed_continuation_frames
            ),
            (4, 2)
        );
        assert_eq!(status.remaining_continuation_frames, 0);
        assert!(status.finish_committed && status.closed);
        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.frame.transcription.as_ref())
                .map(|frame| frame.valid_samples)
                .collect::<Vec<_>>(),
            [1280, 220, 0, 0]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.frame.step == Step::Drain)
                .count(),
            1
        );
        assert!(
            events
                .last()
                .unwrap()
                .frame
                .speech
                .as_ref()
                .unwrap()
                .is_tail
        );
        let final_poll = session.poll(Duration::from_millis(480)).unwrap();
        assert_eq!((final_poll.submitted, final_poll.committed), (0, 0));
        assert!(receiver.pop_output().is_none());
        (
            progress,
            events
                .iter()
                .map(|event| format!("{event:?}"))
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn finish_during_pending_last_source_frame_preserves_the_full_continuation() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input
        .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let mut session = Session::start(state, config()).unwrap();
    let mut position = Position::default();
    advance(&mut session, &mut position, 0);
    receiver.pop_output().unwrap();
    let ready = Rc::new(Cell::new(false));
    let pending = session
        .advance(tick(80), |work, branch| {
            execute(&mut position, ready.clone(), work, branch)
        })
        .unwrap();
    assert_eq!(pending.status, VoiceChatSessionStatus::Pending);
    input.finish(finish_policy()).unwrap();
    assert_eq!(input.status().remaining_continuation_frames, 2);
    assert_eq!(input.status().committed_samples, 0);
    ready.set(true);
    let committed = session.poll(Duration::from_millis(160)).unwrap();
    assert_eq!((committed.submitted, committed.committed), (0, 1));
    assert_eq!(input.status().committed_samples, INPUT_FRAME_SAMPLES as u64);
    assert_eq!(input.status().remaining_continuation_frames, 2);
    for at in [240, 320, 400] {
        advance(&mut session, &mut position, at);
    }
    assert_eq!(session.status(), VoiceChatSessionStatus::Finished);
    assert_eq!(input.status().committed_frames, 3);
    assert_eq!(input.status().committed_continuation_frames, 2);
    let events = outputs(&mut receiver);
    assert_eq!(
        events
            .iter()
            .filter_map(|event| event.frame.transcription.as_ref())
            .map(|frame| frame.valid_samples)
            .collect::<Vec<_>>(),
        [1280, 0, 0]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.frame.step == Step::Drain)
            .count(),
        1
    );
}

#[test]
fn cancelling_pending_continuation_keeps_empty_source_finish_and_suffix_uncommitted() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input.finish(finish_policy()).unwrap();
    let mut session = Session::start(state, config()).unwrap();
    let mut position = Position::default();
    advance(&mut session, &mut position, 0);
    receiver.pop_output().unwrap();
    let ready = Rc::new(Cell::new(false));
    let pending = session
        .advance(tick(80), |work, branch| {
            assert_eq!(
                work.kind(),
                VoiceChatWorkKind::Live,
                "zero source samples still require the declared continuation invocation"
            );
            execute(&mut position, ready.clone(), work, branch)
        })
        .unwrap();
    assert_eq!(pending.status, VoiceChatSessionStatus::Pending);
    assert_eq!((pending.submitted, pending.committed), (1, 0));
    assert!(!input.status().finish_declared);
    assert_eq!(input.status().remaining_continuation_frames, 2);
    assert!(receiver.pop_output().is_none());
    assert_eq!(
        session.cancel(Duration::from_millis(160)).unwrap(),
        VoiceChatSessionStatus::Cancelled { retiring: true }
    );
    assert_eq!(receiver.usage().reserved_publications, 1);
    assert_eq!(
        session.poll(Duration::from_millis(240)).unwrap().status,
        VoiceChatSessionStatus::Cancelled { retiring: true }
    );
    ready.set(true);
    assert_eq!(
        session.poll(Duration::from_millis(320)).unwrap().status,
        VoiceChatSessionStatus::Cancelled { retiring: false }
    );
    let status = input.status();
    assert_eq!(
        (
            status.accepted_samples,
            status.committed_samples,
            status.queued_samples
        ),
        (0, 0, 0)
    );
    assert_eq!(
        (
            status.committed_frames,
            status.committed_continuation_frames
        ),
        (0, 0)
    );
    assert_eq!(status.remaining_continuation_frames, 2);
    assert!(status.finished && status.closed);
    assert!(!status.finish_declared && !status.finish_committed && !status.pending);
    assert_eq!(receiver.usage().reserved_publications, 0);
    assert_eq!(receiver.usage().queued_publications, 0);
    assert!(
        receiver.pop_output().is_none(),
        "cancelled continuation cannot publish output"
    );
}

#[test]
fn continuation_backpressure_snapshot_retry_and_fork_preserve_the_remaining_suffix() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input
        .submit(&[0.25; 2 * INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    input.finish(finish_policy()).unwrap();
    let mut session = Session::start(state, config()).unwrap();
    session
        .enable_snapshots(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 32 << 20,
            cumulative_copy_bytes: 128 << 20,
        })
        .unwrap();
    let mut position = Position::default();
    for at in [0, 80, 160] {
        advance(&mut session, &mut position, at);
    }
    let saved = session.snapshot().unwrap();
    let saved_position = Position {
        invocation: position.invocation,
        progress: position.progress,
    };
    advance(&mut session, &mut position, 240);
    assert_eq!(input.status().remaining_continuation_frames, 1);
    assert!(matches!(
        session.advance(tick(320), |_, _| panic!(
            "full output queue executed suffix"
        )),
        Err(VoiceChatSessionError::Publication(
            VoiceChatPublicationError::Capacity(_)
        ))
    ));
    assert_eq!(input.status().remaining_continuation_frames, 1);
    assert_eq!(input.status().committed_continuation_frames, 1);
    let copy_bytes = session.snapshot_usage().unwrap().cumulative_copy_bytes;
    session.restore(&saved).unwrap();
    assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > copy_bytes);
    position = saved_position;
    assert_eq!(input.status().remaining_continuation_frames, 2);
    assert_eq!(input.status().committed_continuation_frames, 0);
    let mut child = session.fork(RequestId::new(1902)).unwrap();
    let mut child_position = Position {
        invocation: position.invocation,
        progress: position.progress,
    };
    assert_eq!(child.input.status().remaining_continuation_frames, 2);
    assert_eq!(outputs(&mut receiver).len(), 3);
    assert_eq!(outputs(&mut child.output).len(), 3);
    for at in [320, 400] {
        advance(&mut session, &mut position, at);
        advance(&mut child.session, &mut child_position, at);
    }
    for owner in [&mut session, &mut child.session] {
        let seen = &owner
            .mutable_state()
            .unwrap()
            .neural_state()
            .model_state()
            .seen;
        assert_eq!(seen.len(), 4);
        assert!(seen[2..].iter().flatten().all(|sample| *sample == 0.));
    }
    advance(&mut session, &mut position, 480);
    advance(&mut child.session, &mut child_position, 480);
    assert_eq!(session.status(), VoiceChatSessionStatus::Finished);
    assert_eq!(child.session.status(), VoiceChatSessionStatus::Finished);
    assert_eq!(
        input.status().committed_samples,
        (2 * INPUT_FRAME_SAMPLES) as u64
    );
    assert_eq!(
        child.input.status().committed_samples,
        input.status().committed_samples
    );
    let parent_output = outputs(&mut receiver);
    let child_output = outputs(&mut child.output);
    assert_eq!(
        parent_output
            .iter()
            .map(|event| format!("{event:?}"))
            .collect::<Vec<_>>(),
        child_output
            .iter()
            .map(|event| format!("{event:?}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        parent_output
            .iter()
            .filter(|event| event.frame.step == Step::Drain)
            .count(),
        1
    );
    assert_eq!(input.status().remaining_continuation_frames, 0);
    assert_eq!(child.input.status().remaining_continuation_frames, 0);
}
