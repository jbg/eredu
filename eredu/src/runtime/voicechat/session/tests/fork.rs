use super::*;
use crate::runtime::voicechat::{
    ingress::tests::Neural, VoiceChatFork, VoiceChatPublicationReceiver,
};
use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotLimits},
    generation::SemanticEvent,
    scheduler::RequestId,
};

type Session = VoiceChatSession<Neural, ()>;
fn limits(branches: u64) -> SnapshotLimits {
    SnapshotLimits {
        max_snapshots: 4,
        max_branches: branches,
        retained_bytes: 32 << 20,
        cumulative_copy_bytes: 128 << 20,
    }
}
fn run(session: &mut Session, position: &mut Position, at: u64) {
    session
        .advance(tick(at), |work, branch| {
            execute(position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap();
}
fn copied_position(position: &Position) -> Position {
    Position {
        invocation: position.invocation,
        progress: position.progress,
    }
}
fn output(receiver: &mut VoiceChatPublicationReceiver<()>) -> Vec<String> {
    let mut values = vec![];
    while let Some(value) = receiver.pop_output() {
        values.push(format!("{value:?}"));
    }
    values
}

#[test]
fn fork_continues_partial_audio_independently_without_repeating_prefill_or_delivered_events() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input.submit(&[0.25; 1500], INPUT_SAMPLE_RATE).unwrap();
    let mut source = Session::start(state, config()).unwrap();
    source.enable_snapshots(limits(1)).unwrap();
    assert!(matches!(
        source.fork(RequestId::new(902)),
        Err(VoiceChatSessionError::Busy)
    ));
    let mut position = Position::default();
    run(&mut source, &mut position, 0);
    receiver.pop_output().unwrap(); // Already delivered output is not inherited.
    run(&mut source, &mut position, 80);
    let parent_snapshot = source.snapshot().unwrap();
    let mut child = source.fork(RequestId::new(902)).unwrap();
    assert_eq!(child.session.report().completed_work, 0);
    assert_eq!(child.output.usage().queued_publications, 1);
    assert_eq!(child.input.status().queued_samples, 220);
    assert!(matches!(
        child.session.restore(&parent_snapshot),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::Foreign
        ))
    ));
    assert!(matches!(
        child.session.enable_snapshots(limits(10)),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::AlreadyConfigured
        ))
    ));
    assert!(matches!(
        child.session.poll(Duration::ZERO),
        Err(VoiceChatSessionError::Time)
    ));
    let mut child_position = copied_position(&position);
    let retained_before_continuation = source.snapshot_usage().unwrap().retained_bytes;
    input.submit(&[0.5; 1060], INPUT_SAMPLE_RATE).unwrap();
    input
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0,
        })
        .unwrap();
    assert_eq!(child.input.status().queued_samples, 220);
    assert!(!child.input.status().finished);
    child.input.submit(&[0.5; 1060], INPUT_SAMPLE_RATE).unwrap();
    child
        .input
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0,
        })
        .unwrap();
    run(&mut source, &mut position, 160);
    assert_eq!(
        child
            .session
            .mutable_state()
            .unwrap()
            .neural_state()
            .model_state()
            .frames,
        1
    );
    run(&mut child.session, &mut child_position, 160);
    assert!(source.snapshot_usage().unwrap().retained_bytes > retained_before_continuation);
    run(&mut source, &mut position, 240);
    run(&mut child.session, &mut child_position, 240);
    assert_eq!(source.status(), VoiceChatSessionStatus::Finished);
    assert_eq!(child.session.status(), VoiceChatSessionStatus::Finished);
    assert_eq!(child.session.report().completed_work, 2);
    let parent_output = output(&mut receiver);
    assert_eq!(parent_output.len(), 3);
    assert_eq!(output(&mut child.output), parent_output);
    assert_eq!(
        source.snapshot_usage().unwrap().branches,
        1,
        "endpoints still retain child resources"
    );
    drop(child);
    assert_eq!(source.snapshot_usage().unwrap().branches, 0);
}

#[test]
fn fork_rejects_neural_growth_before_committing_audio_or_publication() {
    use eredu_core::execution_control::{SnapshotEstimate, SnapshotResourceKind};
    let (state, _, mut receiver, _) = fixture_state();
    let mut source = Session::start(state, config()).unwrap();
    source.enable_snapshots(limits(1)).unwrap();
    let mut position = Position::default();
    run(&mut source, &mut position, 0);
    receiver.pop_output().unwrap();
    let mut child = source.fork(RequestId::new(902)).unwrap();
    child
        .input
        .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let usage = source.snapshot_usage().unwrap();
    let occupied = source
        .snapshot_budget
        .as_ref()
        .unwrap()
        .reserve(
            SnapshotResourceKind::BranchCopy,
            Some(SnapshotEstimate {
                retained_bytes: limits(1).retained_bytes - usage.retained_bytes - 4096,
                copy_bytes: 0,
            }),
        )
        .unwrap();
    let mut child_position = copied_position(&position);
    let result = child.session.advance(tick(80), |work, branch| {
        // A successful native candidate can exceed retained-child capacity; it
        // must not enter canonical state or consume the admitted source frame.
        branch
            .neural_state_mut()
            .model_state_mut()
            .seen
            .push(vec![1.; 8192]);
        execute(&mut child_position, Rc::new(Cell::new(true)), work, branch)
    });
    assert!(has_source::<ExecutionControlError>(&result.unwrap_err()));
    assert_eq!(child.input.status().committed_samples, 0);
    assert_eq!(child.input.status().queued_samples, INPUT_FRAME_SAMPLES);
    assert!(child.output.pop_output().is_none());
    assert_eq!(source.status(), VoiceChatSessionStatus::Ready);
    assert_eq!(
        source.snapshot_usage().unwrap().cumulative_copy_bytes,
        usage.cumulative_copy_bytes
    );
    drop((child, occupied));
    assert_eq!(source.snapshot_usage().unwrap().branches, 0);
}

fn function_step(session: &mut Session, position: &mut Position, at: u64, function: Option<u32>) {
    session
        .advance(tick(at), |work, branch| {
            let frame = position.output(work, function);
            let after = frame.after;
            let submitted = submit(work, branch, frame, Rc::new(Cell::new(true)))?;
            position.invocation += 1;
            position.progress = after;
            Ok(submitted)
        })
        .unwrap();
}

#[test]
fn fork_rekeys_queued_tools_and_maps_delivered_waits_without_executing_them_again() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input.submit(&[0.25; 1280], INPUT_SAMPLE_RATE).unwrap();
    let mut source = Session::start(state, config()).unwrap();
    source.enable_snapshots(limits(2)).unwrap();
    let mut position = Position::default();
    for (at, token) in [(0, None), (80, Some(20)), (160, Some(30)), (240, Some(21))] {
        output(&mut receiver);
        function_step(&mut source, &mut position, at, token);
    }
    let mut child = source.fork(RequestId::new(u64::MAX)).unwrap();
    let source_batch = receiver.pop_output().unwrap().tools.unwrap();
    let child_batch = child.output.pop_output().unwrap().tools.unwrap();
    let (parent_ticket, child_ticket) = child.pending_tool.unwrap();
    assert_eq!(source_batch.ticket, parent_ticket);
    assert_eq!(child_batch.ticket, child_ticket);
    assert_ne!(parent_ticket, child_ticket);
    for (parent, fork) in source_batch.events.iter().zip(&child_batch.events) {
        match (parent, fork) {
            (
                SemanticEvent::ToolCallStart {
                    index,
                    id: parent_id,
                    name: parent_name,
                },
                SemanticEvent::ToolCallStart { id, name, .. },
            )
            | (
                SemanticEvent::ToolCallCompleted {
                    index,
                    id: parent_id,
                    name: parent_name,
                    ..
                },
                SemanticEvent::ToolCallCompleted { id, name, .. },
            ) => {
                assert_ne!(parent_id, id);
                assert_eq!(
                    id,
                    &format!(
                        "voicechat-{}-{}-{}-{index}",
                        u64::MAX,
                        child_ticket.turn(),
                        child_ticket.batch()
                    )
                );
                assert_eq!(name, parent_name);
            }
            _ => assert_eq!(parent, fork),
        }
    }
    let completed_arguments = |batch: &crate::runtime::voicechat::VoiceChatToolBatch| {
        batch
            .events
            .iter()
            .find_map(|event| match event {
                SemanticEvent::ToolCallCompleted { arguments, .. } => Some(arguments.clone()),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(
        completed_arguments(&source_batch),
        completed_arguments(&child_batch)
    );
    let mut delivered = source.fork(RequestId::new(903)).unwrap();
    assert!(delivered.output.pop_output().is_none());
    assert_eq!(delivered.pending_tool.unwrap().0, parent_ticket);
    assert!(child.session.submit_response(parent_ticket, "ok").is_err());
    child.session.submit_response(child_ticket, "ok").unwrap();
    assert!(source.submit_response(child_ticket, "ok").is_err());
    delivered
        .session
        .submit_response(delivered.pending_tool.unwrap().1, "ok")
        .unwrap();
    source.submit_response(parent_ticket, "ok").unwrap();
}

#[test]
fn fork_budget_survives_restore_pending_cancellation_and_failed_copies_without_identity_reuse() {
    let (state, mut source_input, mut receiver, reject) = fixture_state();
    source_input
        .submit(&[0.25; 1280], INPUT_SAMPLE_RATE)
        .unwrap();
    let mut source = Session::start(state, config()).unwrap();
    source.enable_snapshots(limits(1)).unwrap();
    let mut position = Position::default();
    run(&mut source, &mut position, 0);
    receiver.pop_output().unwrap();
    let mut child = source.fork(RequestId::new(902)).unwrap();
    let saved = child.session.snapshot().unwrap();
    child.session.restore(&saved).unwrap();
    assert_eq!(source.snapshot_usage().unwrap().branches, 1);
    assert!(matches!(
        child.session.fork(RequestId::new(903)),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::Admission(ExecutionControlError::Limit("branch count"))
        ))
    ));
    let mut child_position = copied_position(&position);
    let ready = Rc::new(Cell::new(false));
    child
        .session
        .advance(tick(80), |work, branch| {
            execute(&mut child_position, ready.clone(), work, branch)
        })
        .unwrap();
    assert_eq!(
        child.session.cancel(Duration::from_millis(80)).unwrap(),
        VoiceChatSessionStatus::Cancelled { retiring: true }
    );
    let VoiceChatFork {
        mut session,
        input,
        output,
        ..
    } = child;
    drop(input);
    drop(output);
    assert_eq!(source.snapshot_usage().unwrap().branches, 1);
    ready.set(true);
    session.poll(Duration::from_millis(160)).unwrap();
    assert_eq!(source.snapshot_usage().unwrap().branches, 0);
    assert!(matches!(
        source.fork(RequestId::new(902)),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::RequestReused(_)
        ))
    ));
    let usage = source.snapshot_usage().unwrap();
    reject.set(true);
    let error = match source.fork(RequestId::new(904)) {
        Ok(_) => panic!("copy rejection"),
        Err(error) => error,
    };
    let mut failure: &(dyn Error + 'static) = &error;
    while !failure.is::<CommitFailure>() {
        failure = failure.source().expect("copy error source preserved");
    }
    reject.set(false);
    assert_eq!(source.snapshot_usage().unwrap().branches, 0);
    assert!(source.snapshot_usage().unwrap().cumulative_copy_bytes > usage.cumulative_copy_bytes);
    assert_eq!(
        source
            .mutable_state()
            .unwrap()
            .neural_state()
            .model_state()
            .frames,
        0
    );
    assert_eq!(source_input.status().queued_samples, 1280);
    assert!(matches!(
        source.fork(RequestId::new(904)),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::RequestReused(_)
        ))
    ));
    let mut retry = source.fork(RequestId::new(903)).unwrap(); // Failed branch-limit admission did not issue this ID.
    retry.session.interrupt().unwrap();
    assert_ne!(
        retry
            .session
            .mutable_state()
            .unwrap()
            .conversation()
            .turn_identity(),
        source
            .mutable_state()
            .unwrap()
            .conversation()
            .turn_identity()
    );
}

#[test]
fn fork_rejects_tool_id_growth_before_copying_when_publication_capacity_is_too_small() {
    let (state, mut input, mut receiver, _) = fixture_state();
    input.submit(&[0.25; 1280], INPUT_SAMPLE_RATE).unwrap();
    let mut bounded = config();
    // Both provisional and completed payloads fit; longer fork IDs do not.
    bounded.publication_capacity.tool_bytes = 96;
    let mut source = Session::start(state, bounded).unwrap();
    source.enable_snapshots(limits(1)).unwrap();
    let mut position = Position::default();
    for (at, token) in [(0, None), (80, Some(20)), (160, Some(30)), (240, Some(21))] {
        output(&mut receiver);
        function_step(&mut source, &mut position, at, token);
    }
    let before = source.snapshot_usage().unwrap();
    assert!(matches!(
        source.fork(RequestId::new(u64::MAX)),
        Err(VoiceChatSessionError::Snapshot(
            VoiceChatSnapshotError::Publication(VoiceChatPublicationError::Capacity("tool bytes"))
        ))
    ));
    assert_eq!(source.snapshot_usage().unwrap(), before);
    assert_eq!(receiver.usage().queued_publications, 1);
    let mut child = source.fork(RequestId::new(902)).unwrap();
    assert_eq!(
        child
            .output
            .pop_output()
            .unwrap()
            .tools
            .unwrap()
            .ticket
            .request(),
        RequestId::new(902)
    );
    assert_eq!(
        receiver
            .pop_output()
            .unwrap()
            .tools
            .unwrap()
            .ticket
            .request(),
        RequestId::new(901)
    );
}
