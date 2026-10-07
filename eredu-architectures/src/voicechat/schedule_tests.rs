//! Schedule conformance, using independently advanced mock component counters.
//! These receipts establish transaction attribution, not neural numerical parity.

use super::*;
use crate::voicechat::config::VoiceChatConfig;
use eredu_core::{Completion, scheduler::SemanticStateTransaction};
use eredu_runtime::{
    RealtimeFrameTransition, RealtimeGenerationBranch, RealtimeGenerationState,
    RealtimeScheduleDescriptor,
};
use std::{cell::Cell, convert::Infallible, rc::Rc};

fn limits() -> ScheduleLimits {
    ScheduleLimits {
        max_audio_frames: 16,
        max_backbone_positions: 128,
        max_tts_positions: 128,
        max_codec_frames: 16,
    }
}

fn policy_with(limits: ScheduleLimits) -> SchedulePolicy {
    let config =
        VoiceChatConfig::from_json(include_str!("fixtures/official-inference-config.json"))
            .unwrap();
    SchedulePolicy::from_config(&config, limits).unwrap()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Components {
    progress: Progress,
    actions: Vec<Step>,
}

impl Components {
    // Each simulated component reports its own committed extent. Deliberately do
    // not derive these values from StepPlan::expected or its execution flags.
    fn execute(&mut self, step: Step) -> Progress {
        match step {
            Step::Prefill { backbone_tokens } => {
                self.progress.backbone_positions += backbone_tokens;
                self.progress.tts_conditional_positions += 37;
                self.progress.tts_unconditional_positions += 37;
            }
            Step::LiveAudio
            | Step::Tool {
                audio: AudioSource::Fresh,
            } => {
                self.perception();
                self.progress.backbone_positions += 1;
                self.speech();
            }
            Step::PerceptionOnly => self.perception(),
            Step::ReplayAudio => self.progress.backbone_positions += 1,
            Step::Tool {
                audio: AudioSource::Silence | AudioSource::Retained,
            } => {
                self.progress.backbone_positions += 1;
                self.speech();
            }
            Step::Reminder => self.speech(),
            Step::Drain => {
                if self.progress.codec_frames > 0 {
                    self.progress.output_samples += 8;
                }
                self.progress.codec_finished = true;
            }
        }
        self.actions.push(step);
        self.progress
    }

    fn perception(&mut self) {
        self.progress.audio_frames += 1;
        self.progress.perception_frames += 1;
        self.progress.rnnt_frames += 1;
    }

    fn speech(&mut self) {
        self.progress.tts_conditional_positions += 1;
        self.progress.tts_unconditional_positions += 1;
        self.progress.codec_frames += 1;
        self.progress.output_samples += 1764;
    }
}

impl SemanticStateTransaction for Components {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Infallible> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, branch: Self) -> Result<(), Infallible> {
        *self = branch;
        Ok(())
    }
}

#[derive(Clone)]
struct Ready;
impl Completion for Ready {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}

type Generation = RealtimeGenerationState<Components, ScheduleState, (), (), (), Ready>;
type Branch = RealtimeGenerationBranch<Components, ScheduleState, (), (), (), Ready>;

fn generation(policy: SchedulePolicy) -> Generation {
    let contract = policy.generation_contract();
    let samplers = vec![(); contract.sampler_count()];
    Generation::new(
        Components::default(),
        contract,
        ScheduleState::new(policy),
        (),
        samplers,
        None,
    )
    .unwrap()
}

#[derive(Clone, Copy)]
enum ReceiptFault {
    Rnnt,
    Guidance,
    Pcm,
}

#[derive(Default)]
struct Transition {
    calls: Rc<Cell<usize>>,
    fault: Option<ReceiptFault>,
}

impl RealtimeFrameTransition<Components, ScheduleState, (), (), (), Ready> for Transition {
    type Input = Step;
    type Output = Progress;
    type Error = ScheduleError;
    fn execute(
        &mut self,
        step: &Step,
        branch: &mut Branch,
    ) -> Result<(Progress, Ready), ScheduleError> {
        let plan = branch.schedule_state().plan(*step)?;
        self.calls.set(self.calls.get() + 1);
        let mut observed = branch.model_state_mut().execute(*step);
        match self.fault {
            Some(ReceiptFault::Rnnt) => observed.rnnt_frames += 1,
            Some(ReceiptFault::Guidance) => observed.tts_unconditional_positions += 1,
            Some(ReceiptFault::Pcm) => observed.output_samples += 1,
            None => {}
        }
        *branch.schedule_state_mut() = plan.complete(observed)?;
        Ok((observed, Ready))
    }
}

fn explicit_step(state: &mut Generation, step: Step, transition: &mut Transition) -> Progress {
    let mut branch = state.branch().unwrap();
    let (output, completion) = transition.execute(&step, &mut branch).unwrap();
    branch.attach_submission_completion(completion).unwrap();
    state.commit_branch(branch).unwrap();
    output
}

fn expected(
    audio: u64,
    backbone: u64,
    tts: u64,
    codec: u64,
    samples: u64,
    finished: bool,
) -> Progress {
    Progress {
        audio_frames: audio,
        perception_frames: audio,
        rnnt_frames: audio,
        backbone_positions: backbone,
        tts_conditional_positions: tts,
        tts_unconditional_positions: tts,
        codec_frames: codec,
        output_samples: samples,
        codec_finished: finished,
    }
}

#[test]
fn mixed_audio_tool_reminder_trace_matches_controlled_publication() {
    let policy = policy_with(limits());
    let mut ordinary = generation(policy.clone());
    let mut controlled = generation(policy);
    let mut ordinary_transition = Transition::default();
    let mut controlled_transition = Transition::default();
    let trace = [
        (
            Step::Prefill { backbone_tokens: 5 },
            expected(0, 5, 37, 0, 0, false),
        ),
        (Step::LiveAudio, expected(1, 6, 38, 1, 1764, false)),
        (
            Step::Tool {
                audio: AudioSource::Silence,
            },
            expected(1, 7, 39, 2, 3528, false),
        ),
        (Step::Reminder, expected(1, 7, 40, 3, 5292, false)),
        (Step::PerceptionOnly, expected(2, 7, 40, 3, 5292, false)),
        (
            Step::Tool {
                audio: AudioSource::Fresh,
            },
            expected(3, 8, 41, 4, 7056, false),
        ),
        (Step::Drain, expected(3, 8, 41, 4, 7064, true)),
    ];
    for (step, expected) in trace {
        let automatic = ordinary
            .execute_frame_transition(&step, &mut ordinary_transition)
            .unwrap();
        let explicit = explicit_step(&mut controlled, step, &mut controlled_transition);
        assert_eq!(automatic, expected);
        assert_eq!(explicit, expected);
        assert_eq!(*ordinary.schedule_state().progress(), expected);
        assert_eq!(*controlled.schedule_state().progress(), expected);
        assert_eq!(ordinary.model_state(), controlled.model_state());
    }
    let calls = ordinary_transition.calls.get();
    assert!(
        ordinary
            .execute_frame_transition(&Step::LiveAudio, &mut ordinary_transition)
            .is_err()
    );
    assert_eq!(ordinary_transition.calls.get(), calls);
}

#[test]
fn mismatched_component_receipts_do_not_publish_candidate_state() {
    for fault in [
        ReceiptFault::Rnnt,
        ReceiptFault::Guidance,
        ReceiptFault::Pcm,
    ] {
        let mut state = generation(policy_with(limits()));
        state
            .execute_frame_transition(
                &Step::Prefill { backbone_tokens: 5 },
                &mut Transition::default(),
            )
            .unwrap();
        let before = state.model_state().clone();
        let mut transition = Transition {
            fault: Some(fault),
            ..Default::default()
        };
        assert!(
            state
                .execute_frame_transition(&Step::LiveAudio, &mut transition)
                .is_err()
        );
        assert_eq!(transition.calls.get(), 1);
        assert_eq!(state.model_state(), &before);
        assert_eq!(
            *state.schedule_state().progress(),
            expected(0, 5, 37, 0, 0, false)
        );
        let output = state
            .execute_frame_transition(&Step::LiveAudio, &mut Transition::default())
            .unwrap();
        assert_eq!(output, expected(1, 6, 38, 1, 1764, false));
    }
}

#[test]
fn component_capacity_rejection_precedes_all_mechanisms() {
    let base = limits();
    for limits in [
        ScheduleLimits {
            max_audio_frames: 1,
            ..base
        },
        ScheduleLimits {
            max_backbone_positions: 6,
            ..base
        },
        ScheduleLimits {
            max_tts_positions: 38,
            ..base
        },
        ScheduleLimits {
            max_codec_frames: 1,
            ..base
        },
    ] {
        let mut state = generation(policy_with(limits));
        let mut transition = Transition::default();
        state
            .execute_frame_transition(&Step::Prefill { backbone_tokens: 5 }, &mut transition)
            .unwrap();
        state
            .execute_frame_transition(&Step::LiveAudio, &mut transition)
            .unwrap();
        let before = state.model_state().clone();
        assert!(
            state
                .execute_frame_transition(&Step::LiveAudio, &mut transition)
                .is_err()
        );
        assert_eq!(transition.calls.get(), 2);
        assert_eq!(state.model_state(), &before);
        assert_eq!(
            *state.schedule_state().progress(),
            expected(1, 6, 38, 1, 1764, false)
        );
    }
}

#[test]
fn first_feedback_and_sampling_follow_backbone_not_audio_or_prompt_positions() {
    let mut state = generation(policy_with(limits()));
    let mut transition = Transition::default();
    for step in [
        Step::Prefill { backbone_tokens: 5 },
        Step::PerceptionOnly,
        Step::Reminder,
    ] {
        let plan = state.schedule_state().plan(step).unwrap();
        assert_eq!(plan.initial_feedback_tokens(), None);
        assert!(!plan.agent_sampling_before_overrides());
        state
            .execute_frame_transition(&step, &mut transition)
            .unwrap();
        if matches!(step, Step::Prefill { .. }) {
            let retained = Step::Tool {
                audio: AudioSource::Retained,
            };
            assert!(matches!(
                state.schedule_state().plan(retained),
                Err(ScheduleError::MissingRetainedAudio)
            ));
            assert!(
                state
                    .execute_frame_transition(&retained, &mut transition)
                    .is_err()
            );
            assert_eq!(transition.calls.get(), 1);
        }
    }
    let plan = state.schedule_state().plan(Step::LiveAudio).unwrap();
    assert_eq!(plan.initial_feedback_tokens(), Some([12, 12]));
    assert_eq!(plan.render_mode(), Some(RenderMode::Main));
    assert!(plan.agent_sampling_before_overrides());
    state
        .execute_frame_transition(&Step::LiveAudio, &mut transition)
        .unwrap();
    let plan = state
        .schedule_state()
        .plan(Step::Tool {
            audio: AudioSource::Silence,
        })
        .unwrap();
    assert_eq!(plan.initial_feedback_tokens(), None);
    assert_eq!(plan.render_mode(), Some(RenderMode::Tool));
    assert!(plan.agent_sampling_before_overrides());
    let plan = state.schedule_state().plan(Step::LiveAudio).unwrap();
    assert_eq!(plan.render_mode(), Some(RenderMode::Main));
    let retained = state
        .execute_frame_transition(
            &Step::Tool {
                audio: AudioSource::Retained,
            },
            &mut transition,
        )
        .unwrap();
    assert_eq!(retained, expected(2, 7, 40, 3, 5292, false));
}

#[test]
fn cold_contract_retains_exact_capacity_and_schedule_identity() {
    let policy = policy_with(limits());
    let identical = policy_with(limits());
    assert_eq!(
        policy.frame_contract().unwrap(),
        identical.frame_contract().unwrap()
    );
    assert_eq!(policy.generation_contract().identity(), &policy);
    let contract = policy.frame_contract().unwrap();
    assert_eq!(contract.policy_descriptor(), policy.schedule_descriptor());
    let smaller = policy_with(ScheduleLimits {
        max_backbone_positions: 64,
        ..limits()
    });
    assert_ne!(policy, smaller);
    assert_ne!(
        contract.consensus_descriptor(),
        smaller.frame_contract().unwrap().consensus_descriptor()
    );
}

#[test]
fn empty_drain_and_stale_architecture_predecessor_preserve_publication_boundary() {
    let mut empty = generation(policy_with(limits()));
    let mut transition = Transition::default();
    empty
        .execute_frame_transition(&Step::Prefill { backbone_tokens: 5 }, &mut transition)
        .unwrap();
    let drained = empty
        .execute_frame_transition(&Step::Drain, &mut transition)
        .unwrap();
    assert_eq!(drained, expected(0, 5, 37, 0, 0, true));
    assert!(
        empty
            .execute_frame_transition(&Step::Reminder, &mut transition)
            .is_err()
    );
    assert_eq!(transition.calls.get(), 2);

    let mut state = generation(policy_with(limits()));
    state
        .execute_frame_transition(&Step::Prefill { backbone_tokens: 5 }, &mut transition)
        .unwrap();
    let old_plan = state.schedule_state().plan(Step::LiveAudio).unwrap();
    let mut old_components = state.model_state().clone();
    let old_receipt = old_components.execute(Step::LiveAudio);
    let completed_old_schedule = old_plan.complete(old_receipt).unwrap();
    state
        .execute_frame_transition(&Step::PerceptionOnly, &mut transition)
        .unwrap();
    let current_components = state.model_state().clone();
    let current_revision = state.revision();

    // Branch from the current generic revision, then inject a schedule receipt
    // belonging to the older architecture predecessor. Generic stale-branch
    // detection alone cannot reject this candidate.
    let mut current_branch = state.branch().unwrap();
    *current_branch.model_state_mut() = old_components;
    *current_branch.schedule_state_mut() = completed_old_schedule;
    current_branch.attach_submission_completion(Ready).unwrap();
    let error = state.commit_branch(current_branch).unwrap_err();
    match error {
        eredu_runtime::RealtimeGenerationTransactionError::Schedule(source) => {
            assert!(matches!(
                source.downcast_ref::<ScheduleError>(),
                Some(ScheduleError::StalePredecessor)
            ));
        }
        error => panic!("expected architecture predecessor rejection, got {error}"),
    }
    assert_eq!(state.revision(), current_revision);
    assert_eq!(state.model_state(), &current_components);
    assert_eq!(
        *state.schedule_state().progress(),
        expected(1, 5, 37, 0, 0, false)
    );
}
