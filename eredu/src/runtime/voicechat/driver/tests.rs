use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};
use std::{
    cell::Cell,
    convert::Infallible,
    rc::Rc,
    time::{Duration, Instant},
};

use eredu_architectures::voicechat::{
    channel_decision::{ChannelDecisionPolicy, DecisionSource},
    channels::PromptPolicy,
    execution::{Invocation, ToolAudio},
    frontend::{INPUT_FRAME_SAMPLES, INPUT_SAMPLE_RATE},
    observation::{OwnedHostOutput, OwnedSpeechChunk},
    schedule::{AudioSource, Progress},
    tts_feedback::WaveformSilence,
    tts_ingress::PromptLayout,
};
use eredu_core::BackendFailure;
use eredu_core::scheduler::{
    RequestId, Scheduler, SchedulerLimits, SemanticStateTransaction, TransitionOutput,
    WorkDescriptor,
};
use eredu_runtime::{PrepublicationRealtimeFrame, submit_realtime_output};

use super::*;
use crate::runtime::voicechat::{
    PreparedVoiceChatPrompt, VoiceChatConversationError, VoiceChatConversationOutput,
    VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatPublicationReceiver,
    VoiceChatSpeechOptions, VoiceChatSubmittedWork, VoiceChatToolWaitMessages,
    VoiceChatToolWaitPolicy,
    conversation::tests::{capacity, frame},
    ingress::tests::{Branch, State, state},
    publication::tests::{Observer, Ready},
    tools::tests::{channel, limits},
};

fn advance(at: u64) -> VoiceChatAdvance {
    VoiceChatAdvance {
        at: Duration::from_millis(at),
        tool_audio: VoiceChatToolAudioPolicy::RealAudio,
    }
}

fn next(state: &mut State, at: u64) -> VoiceChatWork<()> {
    match state.prepare_next(advance(at), capacity()).unwrap() {
        VoiceChatPreparation::Ready(work) => work,
        VoiceChatPreparation::Blocked(reason) => panic!("unexpected blocked selection: {reason:?}"),
    }
}

fn blocked(state: &mut State, at: u64, expected: VoiceChatBlocked) {
    match state.prepare_next(advance(at), capacity()).unwrap() {
        VoiceChatPreparation::Blocked(reason) => assert_eq!(reason, expected),
        VoiceChatPreparation::Ready(_) => panic!("unexpected ready selection"),
    }
}

fn descriptor(work: &VoiceChatWork<()>) -> Vec<u32> {
    let mut words = vec![];
    work.encode_descriptor(&mut words).unwrap();
    words
}

fn prompt() -> PreparedVoiceChatPrompt {
    channel(901, limits())
        .text_preparation()
        .prepare_system_prompt(
            "ok ok ok ok ok",
            PromptPolicy {
                repeat: 1,
                max_tokens: 7,
            },
        )
        .unwrap()
}

// Fixed numerical decisions pass through the public submission bridge; the
// production driver, PCM receipts and shared scheduler own all publication.
pub(crate) fn submit(
    work: &VoiceChatWork<()>,
    branch: &mut Branch,
    mut output: OwnedHostOutput<()>,
    ready: Rc<Cell<bool>>,
) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
    let before = descriptor(work);
    let prompt = prompt();
    let layout = PromptLayout::new(1, 1, &[true]).unwrap();
    let inputs = VoiceChatExecutionInputs {
        prompt: Some(&prompt),
        speaker_codes: &73_i32,
        speaker_layout: &layout,
        speaker_conditioning: Some(&79),
        silence: Some(&83),
        speech: VoiceChatSpeechOptions {
            force_silence_on_eos: true,
            force_silence_on_pad: false,
        },
    };
    let result = work.submit(branch, &inputs, |neural, invocation| {
        let speech = match (work.kind(), invocation) {
            (VoiceChatWorkKind::Prefill { backbone_tokens }, Invocation::Prefill {
                prompt: Some((tokens, policy)), codes, layout, conditioning,
            }) => {
                assert_eq!(*tokens, [51; 5]);
                assert_eq!(policy.framed_length(tokens).unwrap() as u64, backbone_tokens);
                assert_eq!(**codes, 73);
                assert_eq!(layout.frames(), 1);
                assert_eq!(conditioning.copied(), Some(79));
                None
            }
            (VoiceChatWorkKind::Live, Invocation::Live { decisions, output }) => {
                assert!(matches!(decisions, DecisionSource::Resolved(_)));
                Some((output, WaveformSilence::PreserveGenerated))
            }
            (VoiceChatWorkKind::PerceptionOnly, Invocation::PerceptionOnly)
            | (VoiceChatWorkKind::ReplayAudio, Invocation::ReplayAudio) => None,
            (VoiceChatWorkKind::Tool { audio, decisions }, Invocation::Tool {
                audio: actual_audio, decisions: actual_decisions, output,
            }) => {
                match (audio, actual_audio) {
                    (AudioSource::Fresh, ToolAudio::Fresh)
                    | (AudioSource::Retained, ToolAudio::Retained) => {}
                    (AudioSource::Silence, ToolAudio::Silence(value)) => assert_eq!(**value, 83),
                    _ => panic!("tool audio source differs from prepared work"),
                }
                assert!(matches!(actual_decisions, DecisionSource::Fixed(actual) if *actual == decisions));
                let waveform = match decisions {
                    ChannelDecisionPolicy::Tool { spoken_text: Some(_), .. } => WaveformSilence::PreserveGenerated,
                    _ => WaveformSilence::ApplySilence,
                };
                Some((output, waveform))
            }
            (VoiceChatWorkKind::Reminder(receipt), Invocation::Reminder { spoken, output }) => {
                assert_eq!(*spoken, receipt.spoken());
                Some((output, WaveformSilence::PreserveGenerated))
            }
            (VoiceChatWorkKind::Drain, Invocation::Drain) => None,
            _ => panic!("architecture invocation differs from prepared work"),
        };
        if let Some((speech, waveform)) = speech {
            assert_eq!(speech.force_silence_on_eos, matches!(work.kind(), VoiceChatWorkKind::Live));
            assert_eq!(speech.silence_on_pad, [false]);
            assert_eq!(speech.waveform, waveform);
        }
        if let Some(transcription) = &mut output.transcription {
            let model = neural.model_state_mut();
            let pcm = model.input.next_frame().unwrap();
            transcription.valid_samples = pcm.valid_samples;
            transcription.source = pcm.source;
            transcription.start_sample = pcm.start_sample;
            model.frames += 1;
            model.seen.push(pcm.samples);
        }
        let submitted = submit_realtime_output(neural, output, |_, _| {
            Ok::<_, eredu_runtime::RealtimeCompletionCreationError<Ready, Infallible>>((
                Ready(ready), 0,
            ))
        }).unwrap();
        Ok(PrepublicationRealtimeFrame::new(submitted, Observer))
    });
    assert_eq!(
        before,
        descriptor(work),
        "consensus intent survives submission"
    );
    result
}

#[derive(Default)]
pub(crate) struct Position {
    pub(crate) invocation: u64,
    pub(crate) progress: Progress,
}
impl Position {
    pub(crate) fn output(
        &self,
        work: &VoiceChatWork<()>,
        function: Option<u32>,
    ) -> OwnedHostOutput<()> {
        let mut output = frame(self.invocation, self.progress, work.kind().step(), function);
        match work.kind() {
            VoiceChatWorkKind::Prefill { backbone_tokens } => {
                output.after.backbone_positions += backbone_tokens
            }
            VoiceChatWorkKind::Reminder(receipt) => output.reminder_token = Some(receipt.spoken()),
            VoiceChatWorkKind::Tool {
                decisions:
                    ChannelDecisionPolicy::Tool {
                        spoken_text: Some(spoken),
                        ..
                    },
                ..
            } => {
                output
                    .decisions
                    .as_mut()
                    .expect("tool decision")
                    .spoken_text = spoken;
            }
            VoiceChatWorkKind::Drain => {
                output.after.codec_finished = true;
                output.after.output_samples += 3;
                output.speech = Some(OwnedSpeechChunk {
                    samples: vec![0.25, 0.5, 0.125],
                    sample_rate: 22050,
                    start_sample: self.progress.output_samples,
                    delay_samples: 0,
                    is_tail: true,
                });
            }
            _ => {}
        }
        output
    }
    fn accept(&mut self, output: &VoiceChatConversationOutput<()>) {
        self.invocation += 1;
        self.progress = output.frame.after;
    }
    fn commit(
        &mut self,
        state: &mut State,
        receiver: &mut VoiceChatPublicationReceiver<()>,
        work: VoiceChatWork<()>,
        function: Option<u32>,
    ) -> VoiceChatConversationOutput<()> {
        let output = self.output(&work, function);
        let mut branch = state.branch().unwrap();
        let gate = submit(&work, &mut branch, output, Rc::new(Cell::new(true))).unwrap();
        assert!(gate.is_complete().unwrap());
        state.commit_branch(branch).unwrap();
        let output = receiver.pop_output().unwrap();
        self.accept(&output);
        output
    }
    fn prefill(&mut self, state: &mut State, receiver: &mut VoiceChatPublicationReceiver<()>) {
        let work = state
            .prepare_prefill(7, Duration::ZERO, capacity())
            .unwrap();
        self.commit(state, receiver, work, None);
    }
}

#[test]
fn immediate_and_controlled_scheduler_advancement_publish_the_same_stream() {
    let run = |controlled: bool| {
        let (mut state, mut input, mut receiver, _) = state();
        blocked(&mut state, 0, VoiceChatBlocked::NeedsPrefill);
        input
            .submit(&[0.375; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
            .unwrap();
        input
            .finish(PcmFinishPolicy {
                final_frame: FinalPcmFrame::RequireComplete,
                trailing_silence_frames: 0,
            })
            .unwrap();
        let request = RequestId::new(901);
        let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
        scheduler.register(request, state).unwrap();
        let mut position = Position::default();
        let mut stream = vec![];
        for i in 0..3 {
            let state = scheduler.request_state_mut(request).unwrap();
            let prepare = |state: &mut State| {
                if i == 0 {
                    state
                        .prepare_prefill(7, Duration::ZERO, capacity())
                        .unwrap()
                } else {
                    next(state, i * 80)
                }
            };
            let work = prepare(state);
            if controlled {
                let words = descriptor(&work);
                if i == 1 {
                    let mut branch = state.branch().unwrap();
                    let gate = submit(
                        &work,
                        &mut branch,
                        position.output(&work, Some(12)),
                        Rc::new(Cell::new(true)),
                    )
                    .unwrap();
                    assert!(gate.is_complete().unwrap());
                    State::discard_branch(branch).unwrap();
                    drop(gate);
                    assert!(receiver.pop_output().is_none());
                }
                drop(work);
                assert_eq!(
                    input.status().committed_samples,
                    if i == 2 { 1280 } else { 0 }
                );
                let retry = prepare(state);
                assert_eq!(words, descriptor(&retry));
                scheduler.enqueue(request, retry).unwrap();
            } else {
                scheduler.enqueue(request, work).unwrap();
            }
            let ready = Rc::new(Cell::new(!controlled));
            let result = scheduler
                .run_local_turn(Instant::now(), |_, work, branch| {
                    submit(
                        work,
                        branch,
                        position.output(work, (i == 1).then_some(12)),
                        ready.clone(),
                    )
                })
                .unwrap();
            if controlled {
                assert!(result.committed.is_empty());
                assert!(receiver.pop_output().is_none());
                ready.set(true);
                let result = scheduler.poll_completions(Instant::now());
                assert!(result.failed.is_empty(), "{:#?}", result.failed);
                assert_eq!(result.committed.len(), 1);
            } else {
                assert!(result.failed.is_empty(), "{:#?}", result.failed);
                assert_eq!(result.committed.len(), 1);
            }
            let output = receiver.pop_output().unwrap();
            position.accept(&output);
            stream.push((
                output.frame.step,
                output.frame.after,
                output.transcription.map(|t| t.text),
                output.frame.speech.map(|s| s.samples),
            ));
        }
        blocked(
            scheduler.request_state_mut(request).unwrap(),
            240,
            VoiceChatBlocked::Finished,
        );
        assert_eq!(input.status().committed_samples, 1280);
        assert!(receiver.pop_output().is_none());
        stream
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn queued_policy_and_capacity_are_reserved_until_drop_or_scheduler_cancellation() {
    let (mut state, mut input, mut receiver, _) = state();
    let mut position = Position::default();
    position.prefill(&mut state, &mut receiver);
    input
        .submit(&[0.5; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let work = next(&mut state, 80);
    assert!(matches!(
        state.interrupt(),
        Err(VoiceChatPublicationError::Conversation(
            VoiceChatConversationError::Busy
        ))
    ));
    assert_eq!(input.status().reserved_samples, 1280);
    drop(work);
    assert_eq!(input.status().reserved_samples, 0);
    let permits: Vec<_> = (0..4).map(|_| state.reserve(capacity()).unwrap()).collect();
    assert!(matches!(
        state.prepare_next(advance(80), capacity()),
        Err(VoiceChatPublicationError::Capacity(_))
    ));
    assert_eq!(input.status().reserved_samples, 0);
    assert_eq!(input.status().committed_samples, 0);
    drop(permits);
    let work = next(&mut state, 80);
    let request = RequestId::new(901);
    let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register(request, state).unwrap();
    scheduler.enqueue(request, work).unwrap();
    scheduler.cancel(request).unwrap();
    assert!(
        scheduler
            .run_local_turn(Instant::now(), |_, work, branch| {
                submit(
                    work,
                    branch,
                    position.output(work, Some(12)),
                    Rc::new(Cell::new(true)),
                )
            })
            .unwrap()
            .committed
            .is_empty()
    );
    assert!(input.status().closed);
    assert_eq!(input.status().reserved_samples, 0);
    assert_eq!(input.status().queued_samples, 1280);
    assert_eq!(receiver.usage().reserved_publications, 0);
}

fn waiting_call(
    state: &mut State,
    input: &mut crate::runtime::voicechat::VoiceChatInputSender,
    receiver: &mut VoiceChatPublicationReceiver<()>,
    position: &mut Position,
) -> crate::runtime::voicechat::VoiceChatToolTicket {
    position.prefill(state, receiver);
    input
        .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let live = next(state, 80);
    assert_eq!(live.kind(), VoiceChatWorkKind::Live);
    position.commit(state, receiver, live, Some(20));
    let tool = next(state, 80);
    assert!(matches!(
        tool.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Retained,
            ..
        }
    ));
    position.commit(state, receiver, tool, Some(30));
    input
        .submit(&[-0.5; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let tool = next(state, 80);
    assert!(matches!(
        tool.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Fresh,
            ..
        }
    ));
    position
        .commit(state, receiver, tool, Some(21))
        .tools
        .unwrap()
        .ticket
}

#[test]
fn tool_wait_keeps_consuming_pcm_then_injects_response_without_advancing_audio_time() {
    let (mut state, mut input, mut receiver, _) = state();
    let mut position = Position::default();
    let ticket = waiting_call(&mut state, &mut input, &mut receiver, &mut position);
    state
        .start_wait(
            ticket,
            Duration::from_millis(80),
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Disabled,
                generic: &[],
                max_phrase_tokens: 64,
            },
        )
        .unwrap();
    input.submit(&[0.75; 17], INPUT_SAMPLE_RATE).unwrap();
    input
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    let perception = next(&mut state, 160);
    assert_eq!(perception.kind(), VoiceChatWorkKind::PerceptionOnly);
    assert!(matches!(
        state.submit_response(ticket, "ok"),
        Err(VoiceChatPublicationError::Conversation(
            VoiceChatConversationError::Busy
        ))
    ));
    let output = position.commit(&mut state, &mut receiver, perception, None);
    assert_eq!(output.frame.transcription.unwrap().valid_samples, 17);
    blocked(&mut state, 240, VoiceChatBlocked::WaitingForTime);
    state.submit_response(ticket, "ok").unwrap();
    let response = next(&mut state, 240);
    assert!(matches!(
        response.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Retained,
            decisions: ChannelDecisionPolicy::Tool {
                function_feedback: Some(51),
                ..
            }
        }
    ));
    position.commit(&mut state, &mut receiver, response, Some(51));
    let end = next(&mut state, 240);
    assert!(matches!(
        end.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Retained,
            ..
        }
    ));
    position.commit(&mut state, &mut receiver, end, Some(22));
    assert_eq!(position.progress.audio_frames, 3);
    assert_eq!(input.status().committed_samples, 2577);
    let drain = next(&mut state, 320);
    assert_eq!(drain.kind(), VoiceChatWorkKind::Drain);
    assert!(
        position
            .commit(&mut state, &mut receiver, drain, None)
            .frame
            .speech
            .unwrap()
            .is_tail
    );
    blocked(&mut state, 320, VoiceChatBlocked::Finished);
}

#[test]
fn reminder_cadence_shares_pcm_fairly_and_early_result_waits_for_phrase_playback() {
    let (mut state, mut input, mut receiver, _) = state();
    let mut position = Position::default();
    let ticket = waiting_call(&mut state, &mut input, &mut receiver, &mut position);
    state
        .start_wait(
            ticket,
            Duration::from_millis(80),
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                generic: &[],
                max_phrase_tokens: 64,
            },
        )
        .unwrap();
    input
        .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let perception = next(&mut state, 80);
    assert_eq!(perception.kind(), VoiceChatWorkKind::PerceptionOnly);
    position.commit(&mut state, &mut receiver, perception, None);
    let permits: Vec<_> = (0..4).map(|_| state.reserve(capacity()).unwrap()).collect();
    assert!(matches!(
        state.prepare_next(advance(160), capacity()),
        Err(VoiceChatPublicationError::Capacity(_))
    ));
    drop(permits);
    // Rejected preparation consumes neither the due token nor recorded time.
    let reminder = next(&mut state, 80);
    assert!(
        matches!(reminder.kind(), VoiceChatWorkKind::Reminder(receipt) if receipt.spoken() == 1)
    );
    position.commit(&mut state, &mut receiver, reminder, None);
    state.submit_response(ticket, "ok").unwrap();
    input
        .submit(&[0.5; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let perception = next(&mut state, 120);
    assert_eq!(perception.kind(), VoiceChatWorkKind::PerceptionOnly);
    position.commit(&mut state, &mut receiver, perception, None);
    blocked(&mut state, 120, VoiceChatBlocked::WaitingForTime);
    let mut spoken = vec![1];
    let mut at = 160;
    let playback = loop {
        let reminder = next(&mut state, at);
        let VoiceChatWorkKind::Reminder(receipt) = reminder.kind() else {
            panic!("early result bypassed initial phrase")
        };
        spoken.push(receipt.spoken());
        let output = position.commit(&mut state, &mut receiver, reminder, None);
        if let Some(playback) = output.playback {
            break playback;
        }
        at += 80;
    };
    let mut expected = vec![1, 51];
    expected.extend([12; 17]);
    expected.push(2);
    assert_eq!(spoken, expected);
    blocked(&mut state, at, VoiceChatBlocked::WaitingForTime);
    state
        .finish_playback(playback, Duration::from_millis(at))
        .unwrap();
    let response = next(&mut state, at);
    assert!(matches!(
        response.kind(),
        VoiceChatWorkKind::Tool {
            decisions: ChannelDecisionPolicy::Tool {
                function_feedback: Some(51),
                ..
            },
            ..
        }
    ));
    drop(response);
    assert!(matches!(
        state.prepare_next(advance(79), capacity()),
        Err(VoiceChatPublicationError::Conversation(
            VoiceChatConversationError::Time
        ))
    ));
}

#[test]
fn default_tool_silence_alternates_with_perception_without_consuming_live_frontier() {
    let (mut state, mut input, mut receiver, _) = state();
    let mut position = Position::default();
    position.prefill(&mut state, &mut receiver);
    input
        .submit(&[0.125; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let live = next(&mut state, 80);
    position.commit(&mut state, &mut receiver, live, Some(20));
    input
        .submit(&[0.5; INPUT_FRAME_SAMPLES * 2], INPUT_SAMPLE_RATE)
        .unwrap();
    let prepare = |state: &mut State| match state
        .prepare_next(
            VoiceChatAdvance {
                at: Duration::from_millis(80),
                ..VoiceChatAdvance::default()
            },
            capacity(),
        )
        .unwrap()
    {
        VoiceChatPreparation::Ready(work) => work,
        VoiceChatPreparation::Blocked(reason) => panic!("unexpected block: {reason:?}"),
    };
    let perception = prepare(&mut state);
    assert_eq!(perception.kind(), VoiceChatWorkKind::PerceptionOnly);
    position.commit(&mut state, &mut receiver, perception, None);
    let tool = prepare(&mut state);
    assert!(matches!(
        tool.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Silence,
            ..
        }
    ));
    position.commit(&mut state, &mut receiver, tool, Some(30));
    assert_eq!(input.status().queued_samples, 1280);
    let perception = prepare(&mut state);
    assert_eq!(perception.kind(), VoiceChatWorkKind::PerceptionOnly);
    position.commit(&mut state, &mut receiver, perception, None);
    let tool = prepare(&mut state);
    assert!(matches!(
        tool.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Silence,
            ..
        }
    ));
    let ticket = position
        .commit(&mut state, &mut receiver, tool, Some(21))
        .tools
        .unwrap()
        .ticket;
    state.submit_response(ticket, "ok").unwrap();
    let response = prepare(&mut state);
    assert!(matches!(
        response.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Silence,
            decisions: ChannelDecisionPolicy::Tool {
                function_feedback: Some(51),
                ..
            }
        }
    ));
    position.commit(&mut state, &mut receiver, response, Some(51));
    assert_eq!(position.progress.audio_frames, 3);
    assert_eq!(input.status().committed_samples, 3840);
}

#[test]
fn invalid_execution_inputs_preserve_reserved_work_for_corrected_submission() {
    let (mut state, mut input, mut receiver, _) = state();
    let mut position = Position::default();
    let prompt = prompt();
    let layout = PromptLayout::new(1, 1, &[true]).unwrap();
    let mut inputs = VoiceChatExecutionInputs {
        prompt: None,
        speaker_codes: &73_i32,
        speaker_layout: &layout,
        speaker_conditioning: None,
        silence: None,
        speech: VoiceChatSpeechOptions {
            force_silence_on_eos: true,
            force_silence_on_pad: false,
        },
    };
    let work = state
        .prepare_prefill(7, Duration::ZERO, capacity())
        .unwrap();
    let words = descriptor(&work);
    let mut branch = state.branch().unwrap();
    let rejected = work.submit(
        &mut branch,
        &inputs,
        |_,
         _|
         -> Result<
            PrepublicationRealtimeFrame<OwnedHostOutput<()>, Ready, Observer>,
            BackendFailure,
        > { panic!("invalid prompt count reaches native submission") },
    );
    assert!(matches!(
        rejected,
        Err(VoiceChatExecutionError::PrefillCount {
            declared: 7,
            prepared: 0
        })
    ));
    assert_eq!(descriptor(&work), words);
    let gate = submit(
        &work,
        &mut branch,
        position.output(&work, None),
        Rc::new(Cell::new(true)),
    )
    .unwrap();
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
    position.accept(&receiver.pop_output().unwrap());
    drop(gate);
    drop(work);

    input
        .submit(&[0.25; INPUT_FRAME_SAMPLES], INPUT_SAMPLE_RATE)
        .unwrap();
    let live = next(&mut state, 80);
    position.commit(&mut state, &mut receiver, live, Some(20));
    let VoiceChatPreparation::Ready(work) = state
        .prepare_next(
            VoiceChatAdvance {
                at: Duration::from_millis(80),
                ..VoiceChatAdvance::default()
            },
            capacity(),
        )
        .unwrap()
    else {
        panic!("silence tool step must be ready")
    };
    assert!(matches!(
        work.kind(),
        VoiceChatWorkKind::Tool {
            audio: AudioSource::Silence,
            ..
        }
    ));
    inputs.prompt = Some(&prompt);
    let words = descriptor(&work);
    let mut branch = state.branch().unwrap();
    let rejected = work.submit(
        &mut branch,
        &inputs,
        |_,
         _|
         -> Result<
            PrepublicationRealtimeFrame<OwnedHostOutput<()>, Ready, Observer>,
            BackendFailure,
        > { panic!("missing silence reaches native submission") },
    );
    assert!(matches!(
        rejected,
        Err(VoiceChatExecutionError::MissingSilence)
    ));
    assert_eq!(descriptor(&work), words);
    let gate = submit(
        &work,
        &mut branch,
        position.output(&work, Some(30)),
        Rc::new(Cell::new(true)),
    )
    .unwrap();
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
    let output = receiver.pop_output().unwrap();
    assert_eq!(output.frame.after.audio_frames, 1);
    assert_eq!(input.status().committed_samples, 1280);
}

#[derive(Debug, thiserror::Error)]
#[error("injected native VoiceChat failure")]
struct NativeFailure;

struct RejectObservation;
impl eredu_runtime::RealtimeFrameHostObserver<OwnedHostOutput<()>> for RejectObservation {
    type Output = OwnedHostOutput<()>;
    type Error = NativeFailure;

    fn observe(&mut self, _: &Self::Output) -> Result<Self::Output, NativeFailure> {
        Err(NativeFailure)
    }
}

#[test]
fn submission_and_completed_observation_preserve_native_error_sources_without_publication() {
    for reject_submission in [true, false] {
        let (mut state, _, mut receiver, _) = state();
        let prompt = prompt();
        let layout = PromptLayout::new(1, 1, &[true]).unwrap();
        let inputs = VoiceChatExecutionInputs {
            prompt: Some(&prompt),
            speaker_codes: &73_i32,
            speaker_layout: &layout,
            speaker_conditioning: None,
            silence: None,
            speech: VoiceChatSpeechOptions {
                force_silence_on_eos: false,
                force_silence_on_pad: false,
            },
        };
        let work = state
            .prepare_prefill(7, Duration::ZERO, capacity())
            .unwrap();
        let mut branch = state.branch().unwrap();
        let result = work.submit(&mut branch, &inputs, |neural, _| {
            if reject_submission {
                return Err(BackendFailure::from_error(NativeFailure));
            }
            let submitted =
                submit_realtime_output(neural, Position::default().output(&work, None), |_, _| {
                    Ok::<_, eredu_runtime::RealtimeCompletionCreationError<Ready, Infallible>>((
                        Ready(Rc::new(Cell::new(true))),
                        0,
                    ))
                })
                .unwrap();
            Ok(PrepublicationRealtimeFrame::new(
                submitted,
                RejectObservation,
            ))
        });
        let (error, gate) = match result {
            Err(VoiceChatExecutionError::Backend(error)) => (error, None),
            Ok(gate) => (gate.is_complete().unwrap_err(), Some(gate)),
            Err(error) => panic!("unexpected execution rejection: {error}"),
        };
        let mut source: &(dyn std::error::Error + 'static) = &error;
        while !source.is::<NativeFailure>() {
            source = source
                .source()
                .expect("original typed native failure remains reachable");
        }
        assert!(receiver.pop_output().is_none());
        assert_eq!(state.neural_state().model_state().frames, 0);
        State::discard_branch(branch).unwrap();
        drop(gate);
        drop(work);
        assert_eq!(receiver.usage().reserved_publications, 0);
    }
}
