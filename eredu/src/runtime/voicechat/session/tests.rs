use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};
use std::{cell::Cell, error::Error, rc::Rc, time::Duration};

use eredu_architectures::voicechat::frontend::{INPUT_FRAME_SAMPLES, INPUT_SAMPLE_RATE};

use super::*;
use crate::runtime::voicechat::{
    VoiceChatInputError, VoiceChatToolAudioPolicy,
    conversation::tests::capacity,
    driver::tests::{Position, submit},
    ingress::tests::{Branch, CommitFailure, state as fixture_state},
};

fn config() -> VoiceChatSessionConfig {
    VoiceChatSessionConfig {
        clock_origin: Instant::now(),
        start_at: Duration::ZERO,
        backbone_prompt_positions: 7,
        publication_capacity: capacity(),
    }
}

fn tick(millis: u64) -> VoiceChatAdvance {
    VoiceChatAdvance {
        at: Duration::from_millis(millis),
        tool_audio: VoiceChatToolAudioPolicy::RealAudio,
    }
}

fn execute(
    position: &mut Position,
    ready: Rc<Cell<bool>>,
    work: &VoiceChatWork<()>,
    branch: &mut Branch,
) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
    let function = matches!(work.kind(), VoiceChatWorkKind::Live).then_some(12);
    let output = position.output(work, function);
    let after = output.after;
    let submitted = submit(work, branch, output, ready)?;
    position.invocation += 1;
    position.progress = after;
    Ok(submitted)
}

#[test]
fn iterator_and_controlled_advancement_publish_the_same_finished_stream() {
    let run = |iterator: bool| {
        let (state, mut input, receiver, _) = fixture_state();
        input.submit(&[0.25; 500], INPUT_SAMPLE_RATE).unwrap();
        input.submit(&[0.5; 1000], INPUT_SAMPLE_RATE).unwrap();
        input
            .finish(PcmFinishPolicy {
                final_frame: FinalPcmFrame::ZeroPad,
                trailing_silence_frames: 0,
            })
            .unwrap();
        let session = VoiceChatSession::start(state, config()).unwrap();
        let mut position = Position::default();
        let executor = move |work: &VoiceChatWork<()>, branch: &mut Branch| {
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        };
        let mut session = crate::runtime::voicechat::BoundVoiceChatSession::from_parts(
            session, executor, input, receiver,
        );
        let ticks = [tick(0), tick(80), tick(160), tick(240)];
        let progress = if iterator {
            session.drive(ticks).collect::<Result<Vec<_>, _>>().unwrap()
        } else {
            ticks
                .into_iter()
                .map(|tick| session.advance(tick))
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(progress.len(), 4);
        assert!(
            progress
                .iter()
                .all(|step| step.submitted == 1 && step.committed == 1)
        );
        assert_eq!(session.status(), VoiceChatSessionStatus::Finished);
        assert_eq!(session.input().status().committed_samples, 1500);
        assert_eq!(session.input().status().committed_frames, 2);
        assert!(session.input().status().closed);
        let mut events = vec![];
        while let Some(output) = session.output_mut().pop_output() {
            events.push((
                output.frame.step,
                output.frame.after,
                output.frame.transcription.map(|text| text.valid_samples),
                output.transcription.map(|text| text.text),
                output.frame.speech.map(|speech| speech.samples),
            ));
        }
        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.2)
                .collect::<Vec<_>>(),
            [1280, 220]
        );
        assert_eq!(
            events.last().unwrap().4.as_deref(),
            Some(&[0.25, 0.5, 0.125][..])
        );
        (progress, events)
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn bound_shared_execution_waits_before_admission_and_retries() {
    use crate::runtime::voicechat::{
        BoundVoiceChatSession, VoiceChatExecutor, ingress::tests::Neural,
    };

    struct SharedExecutor {
        busy: Rc<Cell<bool>>,
        submitted: Rc<Cell<usize>>,
        position: Position,
    }
    impl VoiceChatExecutor<Neural, ()> for SharedExecutor {
        fn is_busy(&self) -> bool {
            self.busy.get()
        }

        fn submit(
            &mut self,
            work: &VoiceChatWork<()>,
            branch: &mut Branch,
        ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
            assert!(!self.busy.get(), "shared work cannot enter a busy executor");
            self.submitted.set(self.submitted.get() + 1);
            execute(&mut self.position, Rc::new(Cell::new(true)), work, branch)
        }
    }

    for iterator in [false, true] {
        let (state, input, output, _) = fixture_state();
        let session = VoiceChatSession::start(state, config()).unwrap();
        let busy = Rc::new(Cell::new(true));
        let submitted = Rc::new(Cell::new(0));
        let executor = SharedExecutor {
            busy: busy.clone(),
            submitted: submitted.clone(),
            position: Position::default(),
        };
        let mut session = BoundVoiceChatSession::from_parts(session, executor, input, output);
        if iterator {
            let (mut drive, input, output) =
                session.drive_with_endpoints([tick(0), tick(0), tick(80)]);
            assert!(matches!(
                drive.next(),
                Some(Err(VoiceChatSessionError::Busy))
            ));
            assert_eq!(submitted.get(), 0);
            assert!(output.pop_output().is_none());
            input
                .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
                .unwrap();
            busy.set(false);
            assert_eq!(drive.next().unwrap().unwrap().committed, 1);
            assert!(output.pop_output().is_some());
            assert_eq!(drive.next().unwrap().unwrap().committed, 1);
            assert!(output.pop_output().is_some());
        } else {
            assert!(matches!(
                session.advance(tick(0)),
                Err(VoiceChatSessionError::Busy)
            ));
            assert_eq!(submitted.get(), 0);
            assert!(session.output_mut().pop_output().is_none());
            session
                .input_mut()
                .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
                .unwrap();
            busy.set(false);
            assert_eq!(session.advance(tick(0)).unwrap().committed, 1);
            assert!(session.output_mut().pop_output().is_some());
            assert_eq!(session.advance(tick(80)).unwrap().committed, 1);
            assert!(session.output_mut().pop_output().is_some());
        }
        assert_eq!(submitted.get(), 2);
        assert_eq!(
            session.input().status().committed_samples,
            INPUT_FRAME_SAMPLES as u64
        );
        assert_eq!(session.status(), VoiceChatSessionStatus::Ready);
    }
}

#[test]
fn pending_work_accepts_pcm_and_cancellation_retires_at_recorded_time() {
    let (state, mut input, mut receiver, _) = fixture_state();
    let mut session = VoiceChatSession::start(state, config()).unwrap();
    let mut position = Position::default();
    session
        .advance(tick(0), |work, branch| {
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap();
    input
        .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let ready = Rc::new(Cell::new(false));
    let submitted = session
        .advance(tick(80), |work, branch| {
            execute(&mut position, ready.clone(), work, branch)
        })
        .unwrap();
    assert_eq!(submitted.status, VoiceChatSessionStatus::Pending);
    assert_eq!(submitted.submitted, 1);
    input
        .submit(&[0.5; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let pending = session
        .advance(tick(160), |_, _| panic!("pending work submitted twice"))
        .unwrap();
    assert_eq!((pending.submitted, pending.committed), (0, 0));
    assert_eq!(input.status().committed_samples, 0);
    assert_eq!(input.status().queued_samples, 2 * INPUT_FRAME_SAMPLES);
    assert!(matches!(
        session.poll(Duration::from_millis(159)),
        Err(VoiceChatSessionError::Time)
    ));
    assert_eq!(
        session.cancel(Duration::from_millis(240)).unwrap(),
        VoiceChatSessionStatus::Cancelled { retiring: true }
    );
    assert!(input.status().closed);
    assert_eq!(receiver.usage().reserved_publications, 1);
    assert!(
        receiver.pop_output().is_some(),
        "earlier committed prefill survives cancellation"
    );
    assert!(receiver.pop_output().is_none());
    assert_eq!(
        session.poll(Duration::from_millis(400)).unwrap().status,
        VoiceChatSessionStatus::Cancelled { retiring: true }
    );
    ready.set(true);
    assert_eq!(
        session.poll(Duration::from_millis(480)).unwrap().status,
        VoiceChatSessionStatus::Cancelled { retiring: false }
    );
    assert_eq!(receiver.usage().reserved_publications, 0);
    assert_eq!(input.status().reserved_samples, 0);
    assert_eq!(input.status().queued_samples, 2 * INPUT_FRAME_SAMPLES);
    assert_eq!(input.status().committed_samples, 0);
    assert_eq!(
        session.report().last_cancellation_to_release_ns,
        Some(Duration::from_millis(240).as_nanos())
    );
    assert!(
        receiver.pop_output().is_none(),
        "cancelled speech cannot publish"
    );
}

#[test]
fn publication_backpressure_resumes_after_delivery_and_drain_closes_input() {
    let (state, mut input, mut receiver, _) = fixture_state();
    let mut session = VoiceChatSession::start(state, config()).unwrap();
    let mut position = Position::default();
    for index in 0..4 {
        if index != 0 {
            input
                .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
                .unwrap();
        }
        session
            .advance(tick(index * 80), |work, branch| {
                execute(&mut position, Rc::new(Cell::new(true)), work, branch)
            })
            .unwrap();
    }
    input
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::RequireComplete,
            trailing_silence_frames: 0,
        })
        .unwrap();
    assert!(matches!(
        session.advance(tick(320), |_, _| panic!("backpressure submitted work")),
        Err(VoiceChatSessionError::Publication(
            VoiceChatPublicationError::Capacity(_)
        ))
    ));
    assert_eq!(receiver.usage().queued_publications, 4);
    assert_eq!(
        input.status().committed_samples,
        3 * INPUT_FRAME_SAMPLES as u64
    );
    assert!(!input.status().finish_committed);
    receiver.pop_output().unwrap();
    let drained = session
        .advance(tick(320), |work, branch| {
            assert_eq!(work.kind(), VoiceChatWorkKind::Drain);
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap();
    assert_eq!(drained.status, VoiceChatSessionStatus::Finished);
    assert!(input.status().finish_committed);
    assert!(matches!(
        input.submit(&[0.25], INPUT_SAMPLE_RATE),
        Err(VoiceChatInputError::Closed)
    ));
    assert_eq!(receiver.usage().queued_publications, 4);
    let mut tail = None;
    while let Some(output) = receiver.pop_output() {
        tail = output.frame.speech;
    }
    assert!(tail.unwrap().is_tail);
}

fn has_source<T: Error + 'static>(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if error.is::<T>() {
            return true;
        }
        current = error.source();
    }
    false
}

#[test]
fn submission_and_neural_commit_failures_retain_original_sources() {
    let (state, input, mut receiver, _) = fixture_state();
    let mut session = VoiceChatSession::start(state, config()).unwrap();
    let failure = session
        .advance(tick(0), |_, _| {
            Err(VoiceChatExecutionError::Backend(
                BackendFailure::from_error(std::io::Error::other(
                    "injected native submission failure",
                )),
            ))
        })
        .unwrap_err();
    assert!(has_source::<std::io::Error>(&failure));
    assert_eq!(
        session.status(),
        VoiceChatSessionStatus::Failed { retiring: false }
    );
    assert!(input.status().closed);
    assert!(receiver.pop_output().is_none());
    assert_eq!(receiver.usage().reserved_publications, 0);

    let (state, input, mut receiver, reject) = fixture_state();
    let mut session = VoiceChatSession::start(state, config()).unwrap();
    let mut position = Position::default();
    reject.set(true);
    let failure = session
        .advance(tick(0), |work, branch| {
            execute(&mut position, Rc::new(Cell::new(true)), work, branch)
        })
        .unwrap_err();
    assert!(has_source::<CommitFailure>(&failure));
    assert_eq!(
        session.status(),
        VoiceChatSessionStatus::Failed { retiring: false }
    );
    assert!(input.status().closed);
    assert!(receiver.pop_output().is_none());
    assert_eq!(receiver.usage().reserved_publications, 0);
}

mod snapshot;

mod fork;

mod finish;

#[test]
fn iterator_and_controlled_tool_acknowledgement_publish_the_same_stream() {
    use crate::runtime::voicechat::{
        BoundVoiceChatSession, VoiceChatInitialReminder, VoiceChatPhrasePolicy,
        VoiceChatToolWaitMessages, VoiceChatToolWaitPolicy,
    };
    let run = |iterator: bool, acknowledgement: &VoiceChatPhrasePolicy| {
        let (state, mut input, receiver, _) = fixture_state();
        state
            .conversation()
            .shared_owner()
            .prepare_acknowledgement(acknowledgement, 32)
            .unwrap();
        state
            .conversation()
            .shared_owner()
            .prepare_initial_reminder(&VoiceChatPhrasePolicy::Fixed("ok".to_owned()), 32)
            .unwrap();
        input
            .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
            .unwrap();
        input
            .finish(PcmFinishPolicy {
                final_frame: FinalPcmFrame::RequireComplete,
                trailing_silence_frames: 0,
            })
            .unwrap();
        let session = VoiceChatSession::start(state, config()).unwrap();
        let mut position = Position::default();
        let mut tool_position = 0;
        let executor = move |work: &VoiceChatWork<()>, branch: &mut Branch| {
            let function = match work.kind() {
                VoiceChatWorkKind::Live => Some(20),
                VoiceChatWorkKind::Tool { .. } => {
                    let token = match tool_position {
                        0 => 30,
                        1..=31 => 12,
                        32 => 21,
                        33 => 51,
                        34 => 22,
                        _ => panic!("unexpected tool step"),
                    };
                    tool_position += 1;
                    Some(token)
                }
                _ => None,
            };
            let output = position.output(work, function);
            let after = output.after;
            let submitted = submit(work, branch, output, Rc::new(Cell::new(true)))?;
            position.invocation += 1;
            position.progress = after;
            Ok(submitted)
        };
        let mut session = BoundVoiceChatSession::from_parts(session, executor, input, receiver);
        let mut events = vec![];
        let mut spoken = vec![];
        let mut reminder = vec![];
        let mut ticket = None;
        let mut playback = None;
        for (phase_index, phase) in [0..35, 35..55, 55..58].into_iter().enumerate() {
            let mut collect =
                |output: crate::runtime::voicechat::VoiceChatConversationOutput<()>| {
                    if matches!(
                        output.frame.step,
                        eredu_architectures::voicechat::schedule::Step::Tool { .. }
                    ) {
                        let decisions = output.frame.decisions.unwrap();
                        assert_eq!(decisions.feedback.agent(), 12);
                        assert!(output.agent_text.is_none());
                        spoken.push(decisions.spoken_text);
                    }
                    if let Some(batch) = &output.tools {
                        ticket = Some(batch.ticket);
                    }
                    if let Some(token) = output.frame.reminder_token {
                        assert_eq!(phase_index, 1);
                        assert!(output.agent_text.is_none());
                        reminder.push(token);
                    }
                    if output.playback.is_some() {
                        assert_eq!(output.frame.reminder_token, Some(2));
                        playback = output.playback;
                    }
                    events.push(format!("{output:?}"));
                };
            let ticks = phase.map(|step| tick(step * 80));
            if iterator {
                let (drive, _, receiver) = session.drive_with_endpoints(ticks);
                for progress in drive {
                    assert_eq!(progress.unwrap().committed, 1);
                    while let Some(output) = receiver.pop_output() {
                        collect(output);
                    }
                }
            } else {
                for at in ticks {
                    assert_eq!(session.advance(at).unwrap().committed, 1);
                    while let Some(output) = session.output_mut().pop_output() {
                        collect(output);
                    }
                }
            }
            if phase_index == 0 {
                session
                    .session_mut()
                    .start_wait(
                        ticket.unwrap(),
                        Duration::from_millis(34 * 80),
                        VoiceChatToolWaitPolicy::default(),
                        VoiceChatToolWaitMessages {
                            initial: VoiceChatInitialReminder::Configured,
                            generic: &[],
                            max_phrase_tokens: 32,
                        },
                    )
                    .unwrap();
                // An immediately available result must still wait for all of
                // the configured reminder and its presentation receipt.
                session
                    .session_mut()
                    .submit_response(ticket.unwrap(), "ok")
                    .unwrap();
            } else if phase_index == 1 {
                assert_eq!(spoken.len(), 33);
                assert_eq!(
                    reminder,
                    [1, 51]
                        .into_iter()
                        .chain([12; 17])
                        .chain([2])
                        .collect::<Vec<_>>()
                );
                session
                    .session_mut()
                    .finish_playback(playback.take().unwrap(), Duration::from_millis(54 * 80))
                    .unwrap();
            }
        }
        assert_eq!(session.status(), VoiceChatSessionStatus::Finished);
        if matches!(acknowledgement, VoiceChatPhrasePolicy::Fixed(_)) {
            assert_eq!(
                spoken,
                [1, 51]
                    .into_iter()
                    .chain([12; 17])
                    .chain([2])
                    .chain([12; 15])
                    .collect::<Vec<_>>()
            );
        } else {
            // All released phrases are ordinary unknown words in this tiny
            // tokenizer; framing and the final response silence remain exact.
            assert_eq!(spoken[0], 1);
            let eos = spoken.iter().position(|&token| token == 2).unwrap();
            assert!(spoken[1..eos - 17].iter().all(|&token| token == 0));
            assert_eq!(&spoken[eos - 17..eos], &[12; 17]);
            assert!(spoken[eos + 1..].iter().all(|&token| token == 12));
        }
        events
    };
    for policy in [
        VoiceChatPhrasePolicy::Fixed("ok".to_owned()),
        VoiceChatPhrasePolicy::Streaming { seed: 7 },
    ] {
        assert_eq!(run(false, &policy), run(true, &policy));
    }
}
