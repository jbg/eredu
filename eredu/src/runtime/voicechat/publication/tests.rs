use eredu_architectures::voicechat::{agent_sampling::AgentSampler, state::InterruptionFeedback};
use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
    rc::Rc,
    time::Instant,
};

use eredu_architectures::voicechat::{
    observation::{OwnedHostOutput, OwnedSpeechChunk},
    schedule::{AudioSource, Progress, Step},
};
use eredu_core::{
    generation::SemanticEvent,
    scheduler::{
        RequestId, Scheduler, SchedulerLimits, SemanticStateTransaction, TransitionOutput,
        WorkDescriptor,
    },
    Completion,
};
use eredu_runtime::{
    submit_realtime_output, PrepublicationRealtimeFrame, RealtimeFrameHostObserver,
    RealtimeGenerationBranch, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeScheduleState,
};

use super::*;
use crate::runtime::voicechat::{
    conversation::tests::{capacity, conversation, frame},
    tools::tests::{channel, limits},
};

#[derive(Clone)]
struct Counter {
    value: usize,
    feedback: [u32; 2],
    published: Rc<Cell<usize>>,
    discarded: Rc<Cell<usize>>,
    reject_commit: Rc<Cell<bool>>,
}

#[derive(Debug, thiserror::Error)]
#[error("injected neural commit rejection")]
struct CommitFailure;

impl InterruptionFeedback for Counter {
    fn reset_interruption_feedback(&mut self) {
        self.feedback = [2, 12];
    }
}

impl SemanticStateTransaction for Counter {
    type Branch = Self;
    type Error = CommitFailure;

    fn branch(&self) -> Result<Self, CommitFailure> {
        Ok(self.clone())
    }

    fn commit_branch(&mut self, branch: Self) -> Result<(), CommitFailure> {
        if branch.reject_commit.get() {
            return Err(CommitFailure);
        }
        branch.published.set(branch.value);
        *self = branch;
        Ok(())
    }

    fn discard_branch(branch: Self) -> Result<(), CommitFailure> {
        branch.discarded.set(branch.discarded.get() + 1);
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Schedule;

impl RealtimeScheduleState for Schedule {
    type Identity = ();
    type Error = Infallible;

    fn identity(&self) -> &() {
        &()
    }

    fn validate_successor(&self, _: &Self) -> Result<(), Infallible> {
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Ready(pub(crate) Rc<Cell<bool>>);

impl Completion for Ready {
    type Error = Infallible;

    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(self.0.get())
    }

    fn wait(&self) -> Result<(), Infallible> {
        assert!(self.0.get(), "native retirement requires exact completion");
        Ok(())
    }
}

struct Work {
    output: OwnedHostOutput<()>,
    permit: RefCell<Option<VoiceChatPublicationPermit<()>>>,
}

impl Work {
    fn new(state: &State, output: OwnedHostOutput<()>) -> Self {
        Self {
            output,
            permit: RefCell::new(Some(state.reserve(capacity()).unwrap())),
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

pub(crate) struct Observer;

impl RealtimeFrameHostObserver<OwnedHostOutput<()>> for Observer {
    type Output = OwnedHostOutput<()>;
    type Error = Infallible;

    fn observe(&mut self, output: &Self::Output) -> Result<Self::Output, Infallible> {
        Ok(output.clone())
    }
}

type Neural = RealtimeGenerationState<Counter, Schedule, (), AgentSampler, (), Ready>;
type Branch = VoiceChatPublicationBranch<
    RealtimeGenerationBranch<Counter, Schedule, (), AgentSampler, (), Ready>,
    (),
>;
type State = VoiceChatPublicationState<Neural, ()>;

fn state(
    request: u64,
    publication_limits: VoiceChatPublicationLimits,
) -> (
    State,
    VoiceChatPublicationReceiver<()>,
    Rc<Cell<usize>>,
    Rc<Cell<usize>>,
) {
    let published = Rc::new(Cell::new(0));
    let discarded = Rc::new(Cell::new(0));
    let neural = RealtimeGenerationState::new(
        Counter {
            value: 0,
            feedback: [3, 20],
            published: published.clone(),
            discarded: discarded.clone(),
            reject_commit: Rc::new(Cell::new(false)),
        },
        RealtimeGenerationContract::new((), 0),
        Schedule,
        (),
        Vec::<AgentSampler>::new(),
        None::<()>,
    )
    .unwrap();
    let (state, receiver) = VoiceChatPublicationState::new(
        neural,
        conversation(channel(request, limits())),
        publication_limits,
    )
    .unwrap();
    (state, receiver, published, discarded)
}

fn submit(
    work: &Work,
    branch: &mut Branch,
    ready: Rc<Cell<bool>>,
) -> Result<impl TransitionOutput, VoiceChatPublicationError> {
    let admission = branch.admit(work.output.step, work.permit.borrow_mut().take().unwrap())?;
    let neural = branch.neural_state_mut();
    neural.model_state_mut().value += 1;
    let submitted = submit_realtime_output(neural, work.output.clone(), |_, _| {
        Ok::<_, eredu_runtime::RealtimeCompletionCreationError<Ready, Infallible>>((
            Ready(ready),
            0,
        ))
    })
    .unwrap();
    Ok(admission.attach(PrepublicationRealtimeFrame::new(submitted, Observer)))
}

fn tool_frames(payload: u32) -> [OwnedHostOutput<()>; 3] {
    let start = frame(0, Progress::default(), Step::LiveAudio, Some(20));
    let payload = frame(
        1,
        start.after,
        Step::Tool {
            audio: AudioSource::Retained,
        },
        Some(payload),
    );
    let end = frame(
        2,
        payload.after,
        Step::Tool {
            audio: AudioSource::Fresh,
        },
        Some(21),
    );
    [start, payload, end]
}

pub(crate) fn queue_limits(max_publications: usize) -> VoiceChatPublicationLimits {
    let per_frame = capacity();
    VoiceChatPublicationLimits {
        max_publications,
        capacity: VoiceChatPublicationCapacity {
            speech_samples: per_frame.speech_samples * max_publications,
            transcription_tokens: per_frame.transcription_tokens * max_publications,
            agent_text_bytes: per_frame.agent_text_bytes * max_publications,
            transcription_text_bytes: per_frame.transcription_text_bytes * max_publications,
            audio_codes: per_frame.audio_codes * max_publications,
            tool_events: per_frame.tool_events * max_publications,
            tool_bytes: per_frame.tool_bytes * max_publications,
            diagnostics: 0,
        },
    }
}

#[test]
fn scheduler_progress_can_be_dropped_without_losing_committed_speech_or_tool_calls() {
    let request = RequestId::new(801);
    let (state, mut receiver, published, _) = state(801, queue_limits(3));
    let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register(request, state).unwrap();
    let mut frames = tool_frames(30);
    frames[2].speech = Some(OwnedSpeechChunk {
        samples: vec![0.25, -0.5],
        sample_rate: 22050,
        start_sample: 0,
        delay_samples: 0,
        is_tail: false,
    });

    for (index, output) in frames.into_iter().enumerate() {
        let work = Work::new(scheduler.request_state(request).unwrap(), output);
        scheduler.enqueue(request, work).unwrap();
        let progress = scheduler
            .run_local_turn(Instant::now(), |_, work, branch| {
                submit(work, branch, Rc::new(Cell::new(true)))
            })
            .unwrap();
        assert!(progress.failed.is_empty());
        assert_eq!(progress.committed.len(), 1);
        drop(progress);
        assert_eq!(published.get(), index + 1);
        let state = scheduler.request_state(request).unwrap();
        assert_eq!(state.usage().queued_publications, index + 1);
        assert_eq!(state.usage().reserved_publications, 0);
    }

    let state = scheduler.request_state_mut(request).unwrap();
    let ticket = state
        .conversation()
        .pending()
        .expect("tool state commits with neural state");
    assert!(receiver.pop_output().unwrap().tools.is_none());
    assert!(receiver.pop_output().unwrap().tools.is_none());
    let output = receiver.pop_output().unwrap();
    assert_eq!(output.frame.speech.unwrap().samples, [0.25, -0.5]);
    let batch = output.tools.unwrap();
    assert_eq!(batch.ticket, ticket);
    assert!(
        matches!(&batch.events[0], SemanticEvent::ToolCallStart { name, .. } if name == "weather")
    );
    assert!(
        matches!(&batch.events[1], SemanticEvent::ToolArgumentsDelta { json_fragment, .. }
        if serde_json::from_str::<serde_json::Value>(json_fragment).unwrap() == serde_json::json!({"city":"Paris"}))
    );
    assert!(matches!(
        batch.events[2],
        SemanticEvent::ToolCallCompleted { .. }
    ));
    assert!(receiver.pop_output().is_none());
    assert_eq!(state.usage().queued_publications, 0);
    state.submit_response(ticket, "ok").unwrap();
}

#[test]
fn queue_backpressure_precedes_enqueue_and_delivery_allows_same_frame_retry() {
    // Exercise slot and resource exhaustion separately: capacity rejection must
    // leave the active scheduler request and the accepted audio frontier intact.
    for (request_value, publication_limits) in [
        (802, queue_limits(1)),
        (
            803,
            VoiceChatPublicationLimits {
                max_publications: 2,
                capacity: capacity(),
            },
        ),
    ] {
        let request = RequestId::new(request_value);
        let (state, mut receiver, published, _) = state(request_value, publication_limits);
        let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
        scheduler.register(request, state).unwrap();
        let first = frame(0, Progress::default(), Step::PerceptionOnly, None);
        let next = frame(1, first.after, Step::PerceptionOnly, None);
        let work = Work::new(scheduler.request_state(request).unwrap(), first);
        scheduler.enqueue(request, work).unwrap();
        drop(
            scheduler
                .run_local_turn(Instant::now(), |_, work, branch| {
                    submit(work, branch, Rc::new(Cell::new(true)))
                })
                .unwrap(),
        );
        let state = scheduler.request_state_mut(request).unwrap();
        if request_value == 803 {
            // The completed frame retains one actual transcription token;
            // the other three reserved slots become available immediately.
            let unused_allowance = state
                .reserve(VoiceChatPublicationCapacity {
                    transcription_tokens: 3,
                    ..VoiceChatPublicationCapacity::default()
                })
                .unwrap();
            assert_eq!(receiver.usage().queued_publications, 1);
            assert_eq!(receiver.usage().reserved_publications, 1);
            drop(unused_allowance);
        }
        assert!(state.reserve(capacity()).is_err());
        assert_eq!(published.get(), 1);
        assert_eq!(receiver.pop_output().unwrap().frame.invocation_index, 0);
        let retry = Work::new(state, next);
        scheduler.enqueue(request, retry).unwrap();
        let progress = scheduler
            .run_local_turn(Instant::now(), |_, work, branch| {
                submit(work, branch, Rc::new(Cell::new(true)))
            })
            .unwrap();
        assert!(progress.failed.is_empty());
        assert_eq!(progress.committed.len(), 1);
        drop(progress);
        assert_eq!(published.get(), 2);
        let output = receiver.pop_output().unwrap();
        assert_eq!(output.frame.invocation_index, 1);
        assert_eq!(output.frame.after.audio_frames, 2);
    }
}

#[test]
fn malformed_tool_completion_rolls_back_neural_policy_and_queue_together() {
    let (mut state, mut receiver, published, discarded) = state(804, queue_limits(1));
    let [start, payload, end] = tool_frames(32); // Non-string city violates the declared schema.
    for output in [start, payload] {
        let work = Work::new(&state, output);
        let mut branch = state.branch().unwrap();
        let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
        assert!(gate.is_complete().unwrap());
        state.commit_branch(branch).unwrap();
        drop(gate);
        assert!(receiver.pop_output().unwrap().tools.is_none());
    }
    assert_eq!(published.get(), 2);

    for _ in 0..2 {
        let work = Work::new(&state, end.clone());
        let mut branch = state.branch().unwrap();
        let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
        assert!(gate.is_complete().is_err());
        State::discard_branch(branch).unwrap();
        drop(gate);
        assert_eq!(published.get(), 2);
        assert!(state.conversation().pending().is_none());
        assert!(receiver.pop_output().is_none());
        assert_eq!(state.usage().reserved_publications, 0);
    }
    assert_eq!(discarded.get(), 2);
    // A non-terminating PAD at the rejected invocation proves acoustic and host
    // frontiers survived; it does not silently discard the malformed call.
    let retry = frame(2, end.before, end.step, Some(12));
    let work = Work::new(&state, retry);
    let mut branch = state.branch().unwrap();
    let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
    let output = receiver.pop_output().unwrap();
    assert_eq!(published.get(), 3);
    assert_eq!(output.frame.after.audio_frames, 2);
    assert!(output.tools.is_none());
}

#[test]
fn neural_commit_failure_preserves_policy_credit_and_original_error() {
    let (mut state, mut receiver, published, _) = state(806, queue_limits(1));
    let mut output = frame(0, Progress::default(), Step::LiveAudio, Some(12));
    let decisions = output.decisions.as_mut().unwrap();
    decisions.raw_agent = 51; // "ok"
    decisions.spoken_text = 51;
    decisions.feedback =
        eredu_architectures::voicechat::channel_decision::FeedbackTokens::new(51, 12).unwrap();
    let work = Work::new(&state, output.clone());
    let mut branch = state.branch().unwrap();
    let reject_commit = branch
        .neural_state_mut()
        .model_state_mut()
        .reject_commit
        .clone();
    reject_commit.set(true);
    let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
    assert!(
        gate.is_complete().unwrap(),
        "host successor validates first"
    );
    let error = state.commit_branch(branch).unwrap_err();
    let mut source: &(dyn std::error::Error + 'static) = &error;
    loop {
        if source.is::<CommitFailure>() {
            break;
        }
        source = source
            .source()
            .expect("original neural error remains in the source chain");
    }
    drop(gate);
    assert_eq!(published.get(), 0);
    assert!(receiver.pop_output().is_none());
    assert_eq!(receiver.usage(), VoiceChatPublicationUsage::default());
    assert!(state.conversation().pending().is_none());

    // Reusing this exact acoustic/invocation frontier requires both the neural
    // and conversation successor to have remained uncommitted.
    reject_commit.set(false);
    let work = Work::new(&state, output);
    let mut branch = state.branch().unwrap();
    let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
    assert_eq!(published.get(), 1);
    let output = receiver.pop_output().unwrap();
    assert_eq!(output.frame.invocation_index, 0);
    assert_eq!(output.frame.after.audio_frames, 1);
    assert_eq!(output.agent_text.unwrap().text, "ok");
    assert_eq!(output.transcription.unwrap().text, "hi");
}

#[test]
fn decoded_text_capacity_rejects_the_candidate_without_consuming_decoder_state() {
    let (mut state, mut receiver, published, _) = state(809, queue_limits(2));
    let mut output = frame(0, Progress::default(), Step::LiveAudio, Some(12));
    let decisions = output.decisions.as_mut().unwrap();
    decisions.raw_agent = 51;
    decisions.spoken_text = 51;
    decisions.feedback =
        eredu_architectures::voicechat::channel_decision::FeedbackTokens::new(51, 12).unwrap();
    for reserved in [
        VoiceChatPublicationCapacity {
            agent_text_bytes: 0,
            ..capacity()
        },
        VoiceChatPublicationCapacity {
            transcription_text_bytes: 0,
            ..capacity()
        },
    ] {
        let work = Work {
            output: output.clone(),
            permit: RefCell::new(Some(state.reserve(reserved).unwrap())),
        };
        let mut branch = state.branch().unwrap();
        let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
        assert!(gate.is_complete().is_err());
        State::discard_branch(branch).unwrap();
        drop(gate);
        assert_eq!(published.get(), 0);
        assert!(receiver.pop_output().is_none());
        assert_eq!(receiver.usage(), VoiceChatPublicationUsage::default());
    }
    for expected in [("ok", "hi"), (" ok", " hi")] {
        let work = Work::new(&state, output.clone());
        let mut branch = state.branch().unwrap();
        let ready = Rc::new(Cell::new(false));
        let gate = submit(&work, &mut branch, ready.clone()).unwrap();
        assert!(!gate.is_complete().unwrap());
        assert!(receiver.pop_output().is_none());
        ready.set(true);
        assert!(gate.is_complete().unwrap());
        assert!(
            receiver.pop_output().is_none(),
            "observation alone cannot deliver text"
        );
        state.commit_branch(branch).unwrap();
        assert_eq!(receiver.usage().queued.agent_text_bytes, expected.0.len());
        assert_eq!(
            receiver.usage().queued.transcription_text_bytes,
            expected.1.len()
        );
        let delivered = receiver.pop_output().unwrap();
        assert_eq!(delivered.agent_text.unwrap().text, expected.0);
        assert_eq!(delivered.transcription.unwrap().text, expected.1);
        assert_eq!(receiver.usage(), VoiceChatPublicationUsage::default());
        let mut next = frame(
            output.invocation_index + 1,
            output.after,
            Step::LiveAudio,
            Some(12),
        );
        next.decisions = output.decisions;
        output = next;
    }
}

#[test]
fn foreign_permits_and_stale_branches_cannot_admit_into_another_frontier() {
    let (mut owner, mut receiver, published, _) = state(807, queue_limits(1));
    let (foreign, foreign_receiver, _, _) = state(808, queue_limits(1));
    let mut older = owner.branch().unwrap();
    assert!(matches!(
        older.admit(Step::PerceptionOnly, foreign.reserve(capacity()).unwrap()),
        Err(VoiceChatPublicationError::Transaction(
            "foreign publication permit"
        ))
    ));
    assert_eq!(foreign_receiver.usage().reserved_publications, 0);

    let work = Work::new(
        &owner,
        frame(0, Progress::default(), Step::PerceptionOnly, None),
    );
    let mut current = owner.branch().unwrap();
    let gate = submit(&work, &mut current, Rc::new(Cell::new(true))).unwrap();
    assert!(gate.is_complete().unwrap());
    owner.commit_branch(current).unwrap();
    receiver.pop_output().unwrap();
    assert!(matches!(
        older.admit(Step::PerceptionOnly, owner.reserve(capacity()).unwrap()),
        Err(VoiceChatPublicationError::Transaction("stale branch"))
    ));
    State::discard_branch(older).unwrap();
    assert_eq!(published.get(), 1);
    assert_eq!(receiver.usage(), VoiceChatPublicationUsage::default());
}

#[test]
fn cancelled_submission_retires_only_after_exact_native_completion() {
    let request = RequestId::new(805);
    let (state, mut receiver, published, discarded) = state(805, queue_limits(2));
    let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register(request, state).unwrap();
    let first = frame(0, Progress::default(), Step::PerceptionOnly, None);
    let second = frame(1, first.after, Step::PerceptionOnly, None);
    let work = Work::new(scheduler.request_state(request).unwrap(), first);
    scheduler.enqueue(request, work).unwrap();
    drop(
        scheduler
            .run_local_turn(Instant::now(), |_, work, branch| {
                submit(work, branch, Rc::new(Cell::new(true)))
            })
            .unwrap(),
    );
    let work = Work::new(scheduler.request_state(request).unwrap(), second);
    scheduler.enqueue(request, work).unwrap();
    let ready = Rc::new(Cell::new(false));
    let progress = scheduler
        .run_local_turn(Instant::now(), |_, work, branch| {
            submit(work, branch, ready.clone())
        })
        .unwrap();
    assert!(progress.committed.is_empty());
    let state = scheduler.request_state(request).unwrap();
    assert_eq!(state.usage().reserved_publications, 1);
    assert!(state.reserve(capacity()).is_err());
    scheduler.cancel(request).unwrap();
    assert!(scheduler.request_state(request).is_none());
    assert_eq!(receiver.usage().reserved_publications, 1);
    assert_eq!(receiver.usage().queued_publications, 1);
    // Cancellation does not revoke an earlier committed publication even after
    // the scheduler removes its canonical request state.
    let committed = receiver.pop_output().unwrap();
    assert_eq!(committed.frame.invocation_index, 0);
    assert_eq!(committed.transcription.unwrap().text, "hi");
    assert!(scheduler
        .poll_completions(Instant::now())
        .committed
        .is_empty());
    assert_eq!(discarded.get(), 0);
    assert_eq!(receiver.usage().reserved_publications, 1);
    ready.set(true);
    let retired = scheduler.poll_completions(Instant::now());
    assert!(retired.committed.is_empty());
    assert!(retired.failed.is_empty());
    assert_eq!(published.get(), 1);
    assert_eq!(discarded.get(), 1);
    assert_eq!(receiver.usage().reserved_publications, 0);
    assert!(receiver.pop_output().is_none());
    scheduler.forget_terminal(request).unwrap();
}

impl eredu_runtime::execution_control::SnapshotValue for Schedule {
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        eredu_runtime::execution_control::snapshot_value_estimate::<Self>([])
    }
    fn copy_snapshot_value(
        &self,
        _: &eredu_runtime::execution_control::SnapshotReservation,
    ) -> Result<Self, BackendFailure> {
        Ok(Self)
    }
}

#[test]
fn interrupted_publication_resets_neural_feedback_only_after_successful_commit() {
    let (mut state, mut receiver, published, _) = state(811, queue_limits(1));
    state.conversation = super::super::conversation::tests::conversation_with_policy(
        channel(811, limits()),
        super::super::VoiceChatTurnPolicy {
            bou_frames: 2,
            ..super::super::VoiceChatTurnPolicy::default()
        },
    );
    let [mut start, payload, _] = tool_frames(30);
    let transcription = start.transcription.as_mut().unwrap();
    transcription.first_blank = true;
    transcription.emitted.clear();
    let first_speech = frame(2, payload.after, Step::PerceptionOnly, None);
    let interruption = frame(3, first_speech.after, Step::PerceptionOnly, None);
    for output in [start, payload, first_speech] {
        let work = Work::new(&state, output);
        let mut branch = state.branch().unwrap();
        let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
        assert!(gate.is_complete().unwrap());
        state.commit_branch(branch).unwrap();
        assert_eq!(receiver.pop_output().unwrap().turn, None);
    }
    let identity = state.conversation().turn_identity();
    let reject = state.neural_state().model_state().reject_commit.clone();
    reject.set(true);
    let work = Work::new(&state, interruption.clone());
    let mut branch = state.branch().unwrap();
    let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
    assert!(gate.is_complete().unwrap());
    assert_eq!(state.neural_state().model_state().feedback, [3, 20]);
    assert!(state.commit_branch(branch).is_err());
    drop(gate);
    assert_eq!(state.neural_state().model_state().feedback, [3, 20]);
    assert_eq!(state.conversation().turn_identity(), identity);
    assert_eq!(published.get(), 3);
    assert!(receiver.pop_output().is_none());

    reject.set(false);
    let work = Work::new(&state, interruption.clone());
    let mut branch = state.branch().unwrap();
    let gate = submit(&work, &mut branch, Rc::new(Cell::new(true))).unwrap();
    assert!(gate.is_complete().unwrap());
    state.commit_branch(branch).unwrap();
    let output = receiver.pop_output().unwrap();
    assert_eq!(
        output.turn,
        Some(super::super::VoiceChatTurnEvent::Interrupted)
    );
    assert_eq!(state.neural_state().model_state().feedback, [2, 12]);
    assert_eq!(
        state.conversation().turn_identity().turn(),
        identity.turn() + 1
    );
    assert_eq!(
        output.frame.after.backbone_positions,
        interruption.before.backbone_positions
    );
    assert_eq!(
        output.frame.after.audio_frames,
        interruption.before.audio_frames + 1
    );
    assert_eq!(published.get(), 4);
}
