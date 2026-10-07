use eredu_architectures::voicechat::{agent_sampling::AgentSampler, state::InterruptionFeedback};
use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
    rc::Rc,
    time::{Duration, Instant},
};

use eredu_architectures::voicechat::{
    frontend::{FrontendError, INPUT_FRAME_SAMPLES, INPUT_SAMPLE_RATE, PcmPrecision},
    input::{InputError, PcmInputState, PcmInputStatus},
    observation::OwnedHostOutput,
    schedule::{AudioSource, Progress, Step},
};
use eredu_core::scheduler::{
    RequestId, Scheduler, SchedulerLimits, SemanticStateTransaction, TransitionOutput,
    WorkDescriptor,
};
use eredu_media::audio::streaming::PcmFrameQueue;
use eredu_runtime::{
    PrepublicationRealtimeFrame, RealtimeGenerationBranch, RealtimeGenerationContract,
    RealtimeGenerationState, submit_realtime_output,
};

use super::*;
use crate::runtime::voicechat::{
    VoiceChatPublicationBranch, VoiceChatPublicationError, VoiceChatPublicationPermit,
    VoiceChatPublicationReceiver, VoiceChatPublicationState, VoiceChatToolWaitAction,
    VoiceChatToolWaitMessages, VoiceChatToolWaitPolicy,
    conversation::tests::{capacity, conversation, frame},
    publication::tests::{Observer, Ready, Schedule, queue_limits},
    tools::tests::{channel, limits},
};

// A portable neural transaction retains actual PCM frame contents; completion
// and publication still use the production realtime envelope and scheduler.
#[derive(Clone)]
pub(crate) struct PcmModel {
    pub(crate) input: PcmFrameQueue,
    pub(crate) frames: u64,
    pub(crate) seen: Vec<Vec<f32>>,
    reject: Rc<Cell<bool>>,
    fork_retention: Option<eredu_runtime::execution_control::SnapshotReservation>,
}

#[derive(Debug, thiserror::Error)]
#[error("injected PCM neural commit failure")]
pub(crate) struct CommitFailure;

impl InterruptionFeedback for PcmModel {
    fn reset_interruption_feedback(&mut self) {
        // This ingress-only model retains PCM and frames, with no token feedback.
    }
}

impl SemanticStateTransaction for PcmModel {
    type Branch = Self;
    type Error = CommitFailure;
    fn branch(&self) -> Result<Self, CommitFailure> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, branch: Self) -> Result<(), CommitFailure> {
        if branch.reject.get() {
            return Err(CommitFailure);
        }
        *self = branch;
        Ok(())
    }
}

impl PcmInputState for PcmModel {
    fn pcm_input_status(&self) -> PcmInputStatus {
        PcmInputStatus {
            pcm_precision: PcmPrecision::F32,
            queue_capacity_samples: INPUT_FRAME_SAMPLES,
            queued_samples: self.input.queued_samples(),
            frames: self.frames,
            finished: self.input.is_finished(),
            remaining_continuation_frames: self.input.remaining_continuation_frames(),
            drained: self.input.is_drained(),
        }
    }
    fn submit_pcm(&mut self, samples: &[f32], rate: u32) -> Result<(), InputError> {
        self.input
            .submit(samples, rate)
            .map_err(FrontendError::from)?;
        Ok(())
    }
    fn finish_pcm(&mut self, policy: PcmFinishPolicy) -> Result<(), InputError> {
        self.input.finish(policy).map_err(FrontendError::from)?;
        Ok(())
    }
}

pub(crate) type Neural = RealtimeGenerationState<PcmModel, Schedule, (), AgentSampler, (), Ready>;
pub(crate) type Branch = VoiceChatPublicationBranch<
    RealtimeGenerationBranch<PcmModel, Schedule, (), AgentSampler, (), Ready>,
    (),
>;
pub(crate) type State = VoiceChatPublicationState<Neural, ()>;

pub(crate) fn state() -> (
    State,
    VoiceChatInputSender,
    VoiceChatPublicationReceiver<()>,
    Rc<Cell<bool>>,
) {
    let reject = Rc::new(Cell::new(false));
    let neural = RealtimeGenerationState::new(
        PcmModel {
            input: PcmFrameQueue::new(INPUT_SAMPLE_RATE, INPUT_FRAME_SAMPLES, INPUT_FRAME_SAMPLES)
                .unwrap(),
            frames: 0,
            seen: vec![],
            reject: reject.clone(),
            fork_retention: None,
        },
        RealtimeGenerationContract::new((), 0),
        Schedule,
        (),
        Vec::<AgentSampler>::new(),
        None::<()>,
    )
    .unwrap();
    let (state, input, receiver) = VoiceChatPublicationState::new_with_input(
        neural,
        conversation(channel(901, limits())),
        VoiceChatInputConfig {
            capacity_samples: 2 * INPUT_FRAME_SAMPLES,
            pcm_precision: PcmPrecision::F32,
        },
        queue_limits(4),
    )
    .unwrap();
    (state, input, receiver, reject)
}

struct Work {
    output: OwnedHostOutput<()>,
    permits: RefCell<Option<(VoiceChatPublicationPermit<()>, Option<VoiceChatInputPermit>)>>,
}
impl Work {
    fn new(state: &State, output: OwnedHostOutput<()>, with_input: bool) -> Self {
        Self {
            output,
            permits: RefCell::new(Some((
                state.reserve(capacity()).unwrap(),
                with_input.then(|| state.reserve_input().unwrap()),
            ))),
        }
    }
}
impl WorkDescriptor for Work {
    type Error = Infallible;
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Infallible> {
        output.push(self.output.invocation_index as u32);
        Ok(())
    }
}

fn submit(
    work: &Work,
    branch: &mut Branch,
    ready: Rc<Cell<bool>>,
) -> Result<impl TransitionOutput, VoiceChatPublicationError> {
    let (output, input) = work.permits.borrow_mut().take().unwrap();
    let admission = match input {
        Some(input) => branch.admit_input(work.output.step, output, input)?,
        None => branch.admit(work.output.step, output)?,
    };
    let neural = branch.neural_state_mut();
    if matches!(
        work.output.step,
        Step::PerceptionOnly
            | Step::LiveAudio
            | Step::Tool {
                audio: AudioSource::Fresh
            }
    ) {
        let model = neural.model_state_mut();
        let pcm = model
            .input
            .next_frame()
            .expect("fresh execution requires an admitted frame");
        model.frames += 1;
        model.seen.push(pcm.samples);
    }
    let submitted = submit_realtime_output(neural, work.output.clone(), |_, _| {
        Ok::<_, eredu_runtime::RealtimeCompletionCreationError<Ready, Infallible>>((
            Ready(ready),
            0,
        ))
    })
    .unwrap();
    Ok(admission.attach(PrepublicationRealtimeFrame::new(submitted, Observer)))
}

fn commit(state: &mut State, output: OwnedHostOutput<()>, with_input: bool) {
    let work = Work::new(state, output, with_input);
    let mut branch = state.branch().unwrap();
    let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
}

#[test]
fn partial_ingress_and_pending_completion_retain_charged_pcm_until_commit() {
    let (state, mut input, mut receiver, _) = state();
    input.submit(&vec![0.25; 500], INPUT_SAMPLE_RATE).unwrap();
    assert!(matches!(
        state.reserve_input(),
        Err(VoiceChatPublicationError::Input(
            VoiceChatInputError::NeedsInput
        ))
    ));
    input.submit(&vec![0.5; 780], INPUT_SAMPLE_RATE).unwrap();
    let request = RequestId::new(901);
    let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register(request, state).unwrap();
    let first = frame(0, Progress::default(), Step::PerceptionOnly, None);
    let second = frame(1, first.after, Step::PerceptionOnly, None);
    let work = Work::new(scheduler.request_state(request).unwrap(), first, true);
    scheduler.enqueue(request, work).unwrap();
    let ready = Rc::new(Cell::new(false));
    assert!(
        scheduler
            .run_local_turn(Instant::now(), |_, work, branch| submit(
                work,
                branch,
                ready.clone()
            ))
            .unwrap()
            .committed
            .is_empty()
    );
    input.submit(&vec![-0.25; 1280], INPUT_SAMPLE_RATE).unwrap();
    assert!(
        input.submit(&[1.0], INPUT_SAMPLE_RATE).is_err(),
        "submitted head remains charged while completion is pending"
    );
    assert!(
        scheduler
            .request_state(request)
            .unwrap()
            .reserve_input()
            .is_err()
    );
    assert!(receiver.pop_output().is_none());
    assert_eq!(
        (
            input.status().queued_samples,
            input.status().reserved_samples,
            input.status().committed_samples
        ),
        (2560, 1280, 0)
    );
    ready.set(true);
    let progress = scheduler.poll_completions(Instant::now());
    assert!(progress.failed.is_empty());
    assert_eq!(progress.committed.len(), 1);
    drop(progress);
    assert_eq!(
        (
            input.status().queued_samples,
            input.status().committed_samples
        ),
        (1280, 1280)
    );
    let state = scheduler.request_state_mut(request).unwrap();
    commit(state, second, true);
    let seen = &state.neural_state().model_state().seen;
    assert_eq!(seen.len(), 2);
    assert_eq!(&seen[0][..500], vec![0.25; 500]);
    assert_eq!(&seen[0][500..], vec![0.5; 780]);
    assert_eq!(seen[1], vec![-0.25; 1280]);
    assert_eq!(receiver.pop_output().unwrap().frame.after.audio_frames, 1);
    assert_eq!(receiver.pop_output().unwrap().frame.after.audio_frames, 2);
    input.submit(&vec![0.75; 2560], INPUT_SAMPLE_RATE).unwrap();
}

#[test]
fn rejected_host_or_neural_successor_preserves_head_and_later_arrivals() {
    for rejection in ["host observation", "input attribution", "neural commit"] {
        let (mut state, mut input, mut receiver, reject) = state();
        input.submit(&vec![0.125; 1280], INPUT_SAMPLE_RATE).unwrap();
        let expected = frame(0, Progress::default(), Step::PerceptionOnly, None);
        let mut output = expected.clone();
        if rejection == "host observation" {
            output.invocation_index = 99;
        } else if rejection == "input attribution" {
            output.transcription.as_mut().unwrap().valid_samples = 17;
        }
        let work = Work::new(&state, output, true);
        let mut branch = state.branch().unwrap();
        let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
        input.submit(&vec![-0.75; 1280], INPUT_SAMPLE_RATE).unwrap();
        if rejection != "host observation" {
            assert!(gate.is_complete().unwrap());
            reject.set(rejection == "neural commit");
            assert!(state.commit_branch(branch).is_err());
            reject.set(false);
        } else {
            assert!(gate.is_complete().is_err());
            State::discard_branch(branch).unwrap();
        }
        drop(gate);
        assert!(receiver.pop_output().is_none());
        assert!(state.neural_state().model_state().seen.is_empty());
        let next = frame(1, expected.after, Step::PerceptionOnly, None);
        commit(&mut state, expected, true);
        commit(&mut state, next, true);
        assert_eq!(
            state.neural_state().model_state().seen,
            [vec![0.125; 1280], vec![-0.75; 1280]]
        );
    }
}

#[test]
fn retained_tool_steps_preserve_pcm_and_drain_cannot_skip_unread_audio() {
    let (mut state, mut input, mut receiver, _) = state();
    input.submit(&vec![0.25; 2560], INPUT_SAMPLE_RATE).unwrap();
    let start = frame(0, Progress::default(), Step::LiveAudio, Some(20));
    let payload = frame(
        1,
        start.after,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(30),
    );
    let end = frame(
        2,
        payload.after,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(21),
    );
    let after_tools = end.after;
    commit(&mut state, start, true);
    commit(&mut state, payload, false);
    commit(&mut state, end, false);
    assert_eq!(state.neural_state().model_state().seen.len(), 1);
    let mut branch = state.branch().unwrap();
    assert!(
        branch
            .admit(Step::Drain, state.reserve(capacity()).unwrap())
            .is_err()
    );
    State::discard_branch(branch).unwrap();
    let mut branch = state.branch().unwrap();
    assert!(
        branch
            .admit(Step::PerceptionOnly, state.reserve(capacity()).unwrap())
            .is_err(),
        "fresh execution cannot bypass input admission"
    );
    State::discard_branch(branch).unwrap();
    assert!(receiver.pop_output().unwrap().tools.is_none());
    assert!(receiver.pop_output().unwrap().tools.is_none());
    let ticket = receiver.pop_output().unwrap().tools.unwrap().ticket;
    state
        .start_wait(
            ticket,
            Duration::ZERO,
            VoiceChatToolWaitPolicy::default(),
            VoiceChatToolWaitMessages {
                initial: crate::runtime::voicechat::VoiceChatInitialReminder::Text("ok"),
                generic: &[],
                max_phrase_tokens: 128,
            },
        )
        .unwrap();
    let VoiceChatToolWaitAction::Reminder(receipt) = state.advance_wait(Duration::ZERO).unwrap()
    else {
        panic!("expected initial reminder");
    };
    let mut reminder = frame(3, after_tools, Step::Reminder, None);
    reminder.reminder_token = Some(receipt.spoken());
    reminder.after.tts_conditional_positions += 1;
    reminder.after.tts_unconditional_positions += 1;
    reminder.after.codec_frames += 1;
    reminder.after.output_samples += 1764;
    let perception = frame(4, reminder.after, Step::PerceptionOnly, None);
    let mut branch = state.branch().unwrap();
    let admission = branch
        .admit_reminder(receipt, Duration::ZERO, state.reserve(capacity()).unwrap())
        .unwrap();
    let submitted = submit_realtime_output(branch.neural_state_mut(), reminder, |_, _| {
        Ok::<_, eredu_runtime::RealtimeCompletionCreationError<Ready, Infallible>>((
            Ready(Rc::new(Cell::new(true))),
            0,
        ))
    })
    .unwrap();
    let gate = admission.attach(PrepublicationRealtimeFrame::new(submitted, Observer));
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
    assert_eq!(
        input.status().queued_samples,
        1280,
        "reminder speech leaves accepted user audio intact"
    );
    commit(&mut state, perception, true);
    assert_eq!(state.neural_state().model_state().seen.len(), 2);
    assert_eq!(input.status().queued_samples, 0);
}

#[test]
fn final_partial_and_finish_only_drains_preserve_real_sample_attribution() {
    for samples in [0, 17, 1280] {
        let (mut state, mut input, mut receiver, _) = state();
        input
            .submit(&vec![0.375; samples], INPUT_SAMPLE_RATE)
            .unwrap();
        if samples == 17 {
            assert!(
                input
                    .finish(eredu_media::audio::streaming::PcmFinishPolicy {
                        final_frame: eredu_media::audio::streaming::FinalPcmFrame::RequireComplete,
                        trailing_silence_frames: 0
                    })
                    .is_err()
            );
        }
        if samples != 1280 {
            input
                .finish(eredu_media::audio::streaming::PcmFinishPolicy {
                    final_frame: eredu_media::audio::streaming::FinalPcmFrame::ZeroPad,
                    trailing_silence_frames: 0,
                })
                .unwrap();
        }
        let mut after = Progress::default();
        let mut invocation = 0;
        if samples != 0 {
            let mut output = frame(0, after, Step::PerceptionOnly, None);
            output.transcription.as_mut().unwrap().valid_samples = samples;
            after = output.after;
            invocation = 1;
            if samples == 1280 {
                let work = Work::new(&state, output, true);
                let mut branch = state.branch().unwrap();
                let ready = Rc::new(Cell::new(false));
                let gate = submit(&work, &mut branch, ready.clone()).unwrap();
                assert!(!gate.is_complete().unwrap());
                input
                    .finish(eredu_media::audio::streaming::PcmFinishPolicy {
                        final_frame: eredu_media::audio::streaming::FinalPcmFrame::RequireComplete,
                        trailing_silence_frames: 0,
                    })
                    .unwrap();
                ready.set(true);
                assert!(gate.is_complete().unwrap());
                state.commit_branch(branch).unwrap();
                assert!(
                    !input.status().finish_committed,
                    "a late finish remains a distinct retained marker"
                );
            } else {
                commit(&mut state, output, true);
            }
            let published = receiver.pop_output().unwrap();
            assert_eq!(
                published.frame.transcription.unwrap().valid_samples,
                samples
            );
            let seen = &state.neural_state().model_state().seen[0];
            assert_eq!(seen.len(), 1280);
            assert_eq!(&seen[..samples], vec![0.375; samples]);
            assert!(seen[samples..].iter().all(|value| *value == 0.0));
        }
        let drain = frame(invocation, after, Step::Drain, None);
        // Empty/exact finish carries a marker, never a synthetic acoustic frame.
        let with_input = match state.reserve_input() {
            Ok(permit) => {
                assert!(permit.is_finish_only());
                drop(permit);
                true
            }
            Err(_) => false,
        };
        assert_eq!(with_input, samples != 17);
        commit(&mut state, drain, with_input);
        assert_eq!(
            state.neural_state().model_state().frames,
            u64::from(samples != 0)
        );
        assert!(state.neural_state().model_state().input.is_drained());
        assert!(state.reserve_input().is_err());
        assert!(input.status().finish_committed);
        assert_eq!(input.status().committed_samples, samples as u64);
        assert_eq!(
            receiver.pop_output().unwrap().frame.after.audio_frames,
            u64::from(samples != 0)
        );
    }
}

#[test]
fn cancellation_closes_sender_but_retains_pending_pcm_until_native_retirement() {
    let (state, mut input, mut receiver, _) = state();
    input.submit(&vec![0.5; 1280], INPUT_SAMPLE_RATE).unwrap();
    let request = RequestId::new(901);
    let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register(request, state).unwrap();
    let work = Work::new(
        scheduler.request_state(request).unwrap(),
        frame(0, Progress::default(), Step::PerceptionOnly, None),
        true,
    );
    scheduler.enqueue(request, work).unwrap();
    let ready = Rc::new(Cell::new(false));
    assert!(
        scheduler
            .run_local_turn(Instant::now(), |_, work, branch| submit(
                work,
                branch,
                ready.clone()
            ))
            .unwrap()
            .committed
            .is_empty()
    );
    input.submit(&[0.75; 17], INPUT_SAMPLE_RATE).unwrap();
    scheduler.cancel(request).unwrap();
    assert!(input.status().closed);
    assert!(matches!(
        input.submit(&[0.25], INPUT_SAMPLE_RATE),
        Err(VoiceChatInputError::Closed)
    ));
    assert_eq!(
        (
            input.status().queued_samples,
            input.status().reserved_samples,
            input.status().pending
        ),
        (1297, 1280, true)
    );
    assert!(
        scheduler
            .poll_completions(Instant::now())
            .committed
            .is_empty()
    );
    assert!(input.status().pending);
    ready.set(true);
    let retired = scheduler.poll_completions(Instant::now());
    assert!(retired.committed.is_empty());
    assert!(retired.failed.is_empty());
    assert_eq!(
        (
            input.status().queued_samples,
            input.status().committed_samples,
            input.status().pending
        ),
        (1297, 0, false)
    );
    assert!(receiver.pop_output().is_none());
}

impl eredu_runtime::execution_control::IsolatedStateCopy for PcmModel {
    type Error = eredu_core::BackendFailure;
    fn estimate_isolated_copy(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, Self::Error> {
        use eredu_runtime::execution_control::{SnapshotValue, snapshot_value_estimate};
        let bytes = self.input.snapshot_storage_bytes().unwrap();
        snapshot_value_estimate::<Self>([
            Ok(eredu_core::execution_control::SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes: bytes,
            }),
            self.seen.estimate_snapshot_value(),
        ])
    }
    fn copy_isolated(
        &self,
        retention: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, Self::Error> {
        use eredu_runtime::execution_control::SnapshotValue;
        Ok(Self {
            input: self
                .input
                .snapshot_copy()
                .map_err(eredu_core::BackendFailure::from_error)?,
            frames: self.frames,
            seen: self.seen.copy_snapshot_value(retention)?,
            reject: self.reject.clone(),
            fork_retention: None,
        })
    }
    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error> {
        if !Rc::ptr_eq(&self.reject, &saved.reject) {
            return Err(eredu_core::BackendFailure::from_error(CommitFailure));
        }
        Ok(())
    }
    fn restore_isolated(&mut self, mut saved: Self) -> Result<(), Self::Error> {
        self.validate_isolated_restore(&saved)?;
        if self.reject.get() {
            return Err(eredu_core::BackendFailure::from_error(CommitFailure));
        }
        saved.fork_retention = self.fork_retention.clone();
        *self = saved;
        Ok(())
    }
}

impl eredu_runtime::execution_control::IsolatedStateFork for PcmModel {
    fn fork_isolated(
        &self,
        retention: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, Self::Error> {
        if self.reject.get() {
            return Err(eredu_core::BackendFailure::from_error(CommitFailure));
        }
        let mut copied =
            eredu_runtime::execution_control::IsolatedStateCopy::copy_isolated(self, retention)?;
        copied.fork_retention = Some(retention.clone());
        Ok(copied)
    }
}

impl eredu_runtime::execution_control::ForkStateEstimate for PcmModel {
    fn estimate_fork_branch(branch: &Self) -> Result<u64, eredu_core::BackendFailure> {
        eredu_runtime::execution_control::IsolatedStateCopy::estimate_isolated_copy(branch)
            .map(|estimate| estimate.retained_bytes)
    }
}
