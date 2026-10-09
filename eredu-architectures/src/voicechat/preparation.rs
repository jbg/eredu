//! Cold generation policy and initialization of the shared realtime state owner.
//!
//! The caller supplies admitted neural caches and input carry. Initialization
//! allocates no backend resources, executes no prompt, and creates no scheduler.

use eredu_core::{scheduler::SemanticStateTransaction, Completion};
use eredu_nn::{Tensor, TensorElementType};
use eredu_runtime::{
    RealtimeFrameContract, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeGenerationTransactionError,
};

use super::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    channel_decision::FeedbackTokens,
    channels::{TEXT_BOS_ID, TEXT_EOS_ID, TEXT_PAD_ID},
    config::VoiceChatConfig,
    host::TextControls,
    input::InputState,
    schedule::{ScheduleError, ScheduleLimits, SchedulePolicy, ScheduleState, Step},
    state::State,
};

/// Existing realtime ownership specialized to the coupled architecture state.
pub type Generation<T, N, R, C> =
    RealtimeGenerationState<State<T, N>, ScheduleState, AgentSamplingPolicy, AgentSampler, R, C>;

/// Invalid cold generation policy or shared-state initialization.
#[derive(Debug, thiserror::Error)]
pub enum GenerationPreparationError {
    /// Retained tokenizer controls disagree with the released channel equations.
    #[error("VoiceChat {control} token is {actual}, expected {expected}")]
    Control {
        /// Control marker whose ID differs.
        control: &'static str,
        /// ID used by the released equations.
        expected: u32,
        /// ID supplied by retained host conditioning.
        actual: u32,
    },
    /// Scheduling geometry or startup capacity cannot be admitted.
    #[error(transparent)]
    Schedule(#[from] ScheduleError),
    /// Shared generation construction rejected the supplied state.
    #[error(transparent)]
    State(#[from] RealtimeGenerationTransactionError),
}

/// Cold payload bound for projected audio retained during tool work.
///
/// This is mutable session tensor storage, separate from learned parameter pools.
/// It excludes backend tensor handles, allocator overhead and independently copied
/// snapshot/fork storage; live snapshot estimates account for those retained values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingAudioResources {
    /// Maximum retained frames, bounded by the session's source-audio allowance.
    pub max_frames: u64,
    /// Elements per `[1, 1, hidden]` projected frame.
    pub projected_width: usize,
    /// Explicit physical precision of the retained projected tensors.
    pub element_type: TensorElementType,
    /// Payload bytes per projected frame.
    pub frame_bytes: u64,
    /// Maximum combined tensor payload at the admitted source-frame bound.
    pub payload_bytes: u64,
}

/// Invalid physical precision or unrepresentable pending-audio payload bound.
#[derive(Debug, thiserror::Error)]
pub enum PendingAudioResourceError {
    /// Projected speech features require a supported real floating-point precision.
    #[error("VoiceChat pending audio does not support {0:?} precision")]
    Precision(TensorElementType),
    /// The selected source-frame bound cannot be represented as a byte count.
    #[error("VoiceChat pending audio payload accounting overflows")]
    ArithmeticOverflow,
}

/// Immutable family policy ready to initialize an ordinary realtime generation.
///
/// This declaration does not select a residency strategy, native realization or
/// public session capability. Sampling history and component capacities remain
/// explicit, independent limits; exhausting either produces a typed failure.
pub struct PreparedGeneration {
    schedule: SchedulePolicy,
    frame: RealtimeFrameContract,
    sampling: AgentSamplingPolicy,
    feedback: FeedbackTokens,
    projected_width: usize,
}

impl PreparedGeneration {
    /// Validates retained controls and mandatory startup without native work.
    pub fn new(
        config: &VoiceChatConfig,
        controls: &TextControls,
        limits: ScheduleLimits,
        sampling: AgentSamplingPolicy,
    ) -> Result<Self, GenerationPreparationError> {
        for (control, actual, expected) in [
            ("BOS", controls.bos, TEXT_BOS_ID as u32),
            ("EOS", controls.eos, TEXT_EOS_ID as u32),
            ("PAD", controls.pad, TEXT_PAD_ID as u32),
        ] {
            if actual != expected {
                return Err(GenerationPreparationError::Control {
                    control,
                    expected,
                    actual,
                });
            }
        }
        let schedule = SchedulePolicy::from_config(config, limits)?;
        // Fixed-Aria startup is mandatory even without a system prompt. Prompt
        // token capacity is checked separately when its explicit invocation is planned.
        ScheduleState::new(schedule.clone()).plan(Step::Prefill { backbone_tokens: 0 })?;
        let frame = schedule.frame_contract()?;
        Ok(Self {
            schedule,
            frame,
            sampling,
            projected_width: config.perception.output_dim,
            feedback: FeedbackTokens::new(controls.pad, controls.pad)
                .expect("released PAD was validated against the shared vocabulary"),
        })
    }

    /// Exact component clocks and explicit capacities.
    pub const fn schedule(&self) -> &SchedulePolicy {
        &self.schedule
    }

    /// Bounds projected-audio payload without allocating native resources.
    ///
    /// Precision is supplied explicitly by the selected realization; checkpoint
    /// encoding and host PCM precision do not imply an activation element size.
    pub fn pending_audio_resources(
        &self,
        element_type: TensorElementType,
    ) -> Result<PendingAudioResources, PendingAudioResourceError> {
        let element_bytes = match element_type {
            TensorElementType::F16 | TensorElementType::Bf16 => 2,
            TensorElementType::F32 => 4,
            TensorElementType::F64 => 8,
            _ => return Err(PendingAudioResourceError::Precision(element_type)),
        };
        let frame_bytes = u64::try_from(self.projected_width)
            .ok()
            .and_then(|width| width.checked_mul(element_bytes))
            .ok_or(PendingAudioResourceError::ArithmeticOverflow)?;
        let max_frames = self.schedule.limits().max_audio_frames;
        let payload_bytes = frame_bytes
            .checked_mul(max_frames)
            .ok_or(PendingAudioResourceError::ArithmeticOverflow)?;
        Ok(PendingAudioResources {
            max_frames,
            projected_width: self.projected_width,
            element_type,
            frame_bytes,
            payload_bytes,
        })
    }

    /// Exact sparse categorical and dense publication geometry.
    pub const fn frame_contract(&self) -> &RealtimeFrameContract {
        &self.frame
    }

    /// Shared transaction identity with the one retained agent sampler.
    pub fn generation_contract(&self) -> RealtimeGenerationContract<SchedulePolicy> {
        self.schedule.generation_contract()
    }

    /// Explicit agent-text policy and feedback-history capacity.
    pub const fn sampling(&self) -> AgentSamplingPolicy {
        self.sampling
    }

    /// Both channels begin with the retained tokenizer's validated PAD token.
    pub const fn initial_feedback(&self) -> FeedbackTokens {
        self.feedback
    }

    /// Adopts caller-owned cold caches and input into the ordinary shared owner.
    ///
    /// The caller validates neural geometry before allocation and serializes PCM
    /// ingress with outstanding branches. Random state is mandatory because TTS
    /// draws remain stochastic even when the agent policy selects greedily.
    /// Prefill remains an explicit unpublished invocation after initialization.
    pub fn initialize<T, N, R, C>(
        &self,
        input: InputState<T>,
        neural: N,
        random: R,
    ) -> Result<Generation<T, N, R, C>, GenerationPreparationError>
    where
        T: Tensor,
        N: SemanticStateTransaction,
        N::Error: Send + Sync + 'static,
        C: Completion,
    {
        Ok(RealtimeGenerationState::new(
            State::new(input, neural, self.feedback),
            self.generation_contract(),
            ScheduleState::new(self.schedule.clone()),
            self.sampling,
            vec![AgentSampler::new(self.sampling)],
            Some(random),
        )?)
    }
}
