use super::*;
use eredu_architectures::voicechat::{
    channel_decision::FeedbackTokens, config::VoiceChatConfig, frontend::FrontendOptions,
    input::InputState, state::State,
};
use eredu_core::scheduler::SemanticStateTransaction;

#[test]
fn voicechat_generation_preparation_admits_startup_and_retains_cold_feedback() {
    use eredu_architectures::voicechat::{
        agent_sampling::AgentSamplingPolicy,
        host::TextControls,
        preparation::{GenerationPreparationError, PreparedGeneration},
        schedule::{ScheduleLimits, Step},
    };
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let controls = TextControls {
        bos: 1,
        eos: 2,
        pad: 12,
        listening: 11,
        user_bos: 1094,
        user_eos: 1036,
        tool_call_start: 20,
        tool_call_end: 21,
        tool_response_end: 22,
    };
    let limits = ScheduleLimits {
        max_audio_frames: 4,
        max_backbone_positions: 64,
        max_tts_positions: 64,
        max_codec_frames: 4,
    };
    let sampling = AgentSamplingPolicy::streaming_defaults(8).unwrap();
    assert!(matches!(
        PreparedGeneration::new(
            &config,
            &TextControls { pad: 0, ..controls },
            limits,
            sampling,
        ),
        Err(GenerationPreparationError::Control { control: "PAD", .. })
    ));
    assert!(matches!(
        PreparedGeneration::new(
            &config,
            &controls,
            ScheduleLimits {
                max_tts_positions: 1,
                ..limits
            },
            sampling,
        ),
        Err(GenerationPreparationError::Schedule(_))
    ));
    let prepared = PreparedGeneration::new(&config, &controls, limits, sampling).unwrap();
    let pending = prepared
        .pending_audio_resources(eredu_nn::TensorElementType::F32)
        .unwrap();
    assert_eq!(pending.projected_width, 4480);
    assert_eq!(pending.frame_bytes, 17_920);
    assert_eq!(pending.payload_bytes, 71_680);
    let input = InputState::<NumericTensor>::new(
        &config.perception.preprocessor,
        FrontendOptions::default(),
    )
    .unwrap();
    let generation = prepared
        .initialize::<_, _, _, NumericCompletion>(
            input,
            NeuralOwner {
                positions: [0, 0],
                discards: Arc::new(AtomicUsize::new(0)),
            },
            123u64,
        )
        .unwrap();
    assert_eq!(
        generation.model_state().feedback(),
        FeedbackTokens::new(12, 12).unwrap()
    );
    assert_eq!(generation.random_state(), Some(&123));
    assert!(generation.samplers()[0].history().is_empty());
    assert!(generation.schedule_state().plan(Step::LiveAudio).is_err());
    assert!(
        generation
            .schedule_state()
            .plan(Step::Prefill { backbone_tokens: 5 })
            .is_ok()
    );
}

// Deliberately not Clone: coupled state must use the neural owner's transaction,
// including its rollback hook, rather than shallow-cloning mutable cache storage.
struct NeuralOwner {
    positions: [usize; 2],
    discards: Arc<AtomicUsize>,
}
struct NeuralBranch {
    positions: [usize; 2],
    reject: bool,
    discards: Arc<AtomicUsize>,
}
impl SemanticStateTransaction for NeuralOwner {
    type Branch = NeuralBranch;
    type Error = std::io::Error;
    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(NeuralBranch {
            positions: self.positions,
            reject: false,
            discards: self.discards.clone(),
        })
    }
    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        if branch.reject {
            Self::discard_branch(branch)?;
            return Err(std::io::Error::other("joint neural commit rejected"));
        }
        self.positions = branch.positions;
        Ok(())
    }
    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        branch.discards.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn voicechat_state_adopts_component_candidates_only_after_joint_neural_commit() {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let mut input = InputState::<NumericTensor>::new(
        &config.perception.preprocessor,
        FrontendOptions::default(),
    )
    .unwrap();
    input.submit(&[0.1, 0.2], 16_000).unwrap();
    let discards = Arc::new(AtomicUsize::new(0));
    let initial = FeedbackTokens::new(12, 12).unwrap();
    let next = FeedbackTokens::new(3, 5).unwrap();
    let mut state = State::new(
        input,
        NeuralOwner {
            positions: [0, 0],
            discards: discards.clone(),
        },
        initial,
    );

    let mut rejected = state.branch().unwrap();
    assert!(rejected.begin_execution());
    assert!(!rejected.begin_execution());
    rejected.neural.positions = [1, 37];
    rejected.neural.reject = true;
    rejected.input.submit(&[0.3], 16_000).unwrap();
    rejected.feedback = next;
    rejected
        .retain_pending_audio(NumericTensor::new([1, 1, 2], vec![0.25, -0.5]), 2)
        .unwrap();
    assert!(state.commit_branch(rejected).is_err());
    assert_eq!(state.neural().positions, [0, 0]);
    assert_eq!(state.input().frontend().queued_samples(), 2);
    assert_eq!(state.feedback(), initial);
    assert_eq!(state.pending_audio_frames(), 0);
    assert_eq!(discards.load(Ordering::SeqCst), 1);

    let mut accepted = state.branch().unwrap();
    assert!(accepted.begin_execution());
    accepted.neural.positions = [1, 37];
    accepted.input.submit(&[0.3], 16_000).unwrap();
    accepted.feedback = next;
    accepted
        .retain_pending_audio(NumericTensor::new([1, 1, 2], vec![0.25, -0.5]), 2)
        .unwrap();
    accepted
        .retain_pending_audio(NumericTensor::new([1, 1, 2], vec![0.75, 0.125]), 2)
        .unwrap();
    assert!(matches!(
        accepted.retain_pending_audio(NumericTensor::new([1, 1, 2], vec![9.0, 9.0]), 2),
        Err(
            eredu_architectures::voicechat::state::PendingAudioError::Capacity {
                required: 3,
                capacity: 2,
            }
        )
    ));
    state.commit_branch(accepted).unwrap();
    assert_eq!(state.neural().positions, [1, 37]);
    assert_eq!(state.input().frontend().queued_samples(), 3);
    assert_eq!(state.feedback(), next);

    assert_eq!(state.pending_audio_frames(), 2);
    let mut roots = Vec::new();
    state.visit_component_tensors(&mut |tensor| roots.push(tensor.data.clone()));
    assert!(roots.contains(&vec![0.25, -0.5]));
    assert!(roots.contains(&vec![0.75, 0.125]));

    let mut abandoned = state.branch().unwrap();
    assert!(abandoned.begin_pending_audio_replay());
    assert!(!abandoned.begin_pending_audio_replay());
    assert_eq!(abandoned.replay_pending_audio().unwrap().data, [0.25, -0.5]);
    abandoned.neural.positions = [2, 38];
    abandoned.input.submit(&[0.4], 16_000).unwrap();
    abandoned.feedback = initial;
    State::<NumericTensor, NeuralOwner>::discard_branch(abandoned).unwrap();
    assert_eq!(discards.load(Ordering::SeqCst), 2);
    assert_eq!(state.neural().positions, [1, 37]);
    assert_eq!(state.input().frontend().queued_samples(), 3);
    assert_eq!(state.feedback(), next);
    assert_eq!(state.pending_audio_frames(), 2);

    let mut replay = state.branch().unwrap();
    assert!(replay.begin_pending_audio_replay());
    assert_eq!(replay.replay_pending_audio().unwrap().data, [0.25, -0.5]);
    state.commit_branch(replay).unwrap();
    assert_eq!(state.pending_audio_frames(), 1);
    let mut replay = state.branch().unwrap();
    assert!(!replay.begin_pending_audio_replay());
    assert_eq!(replay.replay_pending_audio().unwrap().data, [0.75, 0.125]);
    assert!(matches!(
        replay.replay_pending_audio(),
        Err(eredu_architectures::voicechat::state::PendingAudioError::Empty)
    ));
    state.commit_branch(replay).unwrap();
    assert_eq!(state.pending_audio_frames(), 0);
}

#[test]
fn voicechat_interruption_replaces_feedback_without_advancing_candidate_state() {
    use eredu_architectures::voicechat::{
        agent_sampling::{AgentSampler, AgentSamplingPolicy},
        schedule::{ScheduleLimits, SchedulePolicy, ScheduleState},
        state::InterruptionFeedback,
    };
    use eredu_runtime::RealtimeGenerationState;
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let policy = SchedulePolicy::from_config(
        &config,
        ScheduleLimits {
            max_audio_frames: 4,
            max_backbone_positions: 64,
            max_tts_positions: 64,
            max_codec_frames: 4,
        },
    )
    .unwrap();
    let mut input = InputState::<NumericTensor>::new(
        &config.perception.preprocessor,
        FrontendOptions::default(),
    )
    .unwrap();
    input.submit(&[0.1, 0.2], 16_000).unwrap();
    let before = FeedbackTokens::new(51, 20).unwrap();
    let state = State::new(
        input,
        NeuralOwner {
            positions: [9, 37],
            discards: Arc::new(AtomicUsize::new(0)),
        },
        before,
    );
    let mut sampler = AgentSampler::new(AgentSamplingPolicy::streaming_defaults(3).unwrap());
    // At capacity, interruption replaces the last slot instead of admitting
    // another decision or erasing the preceding repetition context.
    sampler.set_history(&[31, 44, 51]).unwrap();
    let schedule = ScheduleState::new(policy.clone());
    type Generation = RealtimeGenerationState<
        State<NumericTensor, NeuralOwner>,
        ScheduleState,
        (),
        AgentSampler,
        u64,
        NumericCompletion,
    >;
    let generation = Generation::new(
        state,
        policy.generation_contract(),
        schedule.clone(),
        (),
        vec![sampler],
        Some(123),
    )
    .unwrap();
    let mut candidate = generation.branch().unwrap();
    candidate.reset_interruption_feedback();
    assert_eq!(
        candidate.model_state().feedback,
        FeedbackTokens::new(2, 12).unwrap()
    );
    assert_eq!(candidate.samplers()[0].history(), &[31, 44, 2]);
    assert_eq!(candidate.model_state().neural.positions, [9, 37]);
    assert_eq!(candidate.model_state().input.frontend().queued_samples(), 2);
    assert_eq!(candidate.schedule_state(), &schedule);
    assert_eq!(candidate.random_state(), Some(&123));
    Generation::discard_branch(candidate).unwrap();
    assert_eq!(generation.model_state().feedback(), before);
    assert_eq!(generation.samplers()[0].history(), &[31, 44, 51]);
}
