//! Architecture-owned coupled component clocks and preflight receipts.
//!
//! A step is an invocation intent, not a PCM ingress or executable inference
//! implementation. Component execution supplies observed candidate counters;
//! the ordinary realtime transaction owns completion and publication. Current
//! spoken text conditions TTS independently of effective backbone feedback;
//! the released model has no context-hidden path.

use eredu_core::{
    execution_control::SnapshotEstimate, BackendFailure, RealtimeChannelCadence,
    RealtimeChannelDeclaration, RealtimeChannelFeedback, RealtimeChannelId, RealtimeChannelLayout,
    RealtimeChannelPayload,
};
use eredu_runtime::execution_control::{SnapshotReservation, SnapshotValue};
use eredu_runtime::{
    RealtimeFrameContract, RealtimeGenerationContract, RealtimeIdentity,
    RealtimeScheduleDescriptor, RealtimeScheduleState,
};

use super::{
    channels::{TEXT_BOS_ID, TEXT_EOS_ID, TEXT_PAD_ID},
    codec::waveform_decoder_config,
    config::VoiceChatConfig,
    tts_prompt::AriaPromptPlan,
};

/// Agent text channel, sampled before any effective-token override.
pub const AGENT_CHANNEL: RealtimeChannelId = RealtimeChannelId::new(0);
/// Greedy function channel from the same normalized backbone output.
pub const FUNCTION_CHANNEL: RealtimeChannelId = RealtimeChannelId::new(1);
/// Complete residual speech-code frame, including admitted control codes.
pub const RVQ_CHANNEL: RealtimeChannelId = RealtimeChannelId::new(2);
/// One fresh source PCM frame.
pub const INPUT_CHANNEL: RealtimeChannelId = RealtimeChannelId::new(3);
/// One ordinary codec output frame.
pub const OUTPUT_CHANNEL: RealtimeChannelId = RealtimeChannelId::new(4);
/// Final codec spectral overlap, separate from an ordinary frame.
pub const TAIL_CHANNEL: RealtimeChannelId = RealtimeChannelId::new(5);

/// Explicit retained-context bounds, checked before component work is planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleLimits {
    /// Maximum fresh live-audio frames, including perception-only work.
    pub max_audio_frames: u64,
    /// Maximum Nemotron positions, including system prompt prefill.
    pub max_backbone_positions: u64,
    /// Maximum positions in each conditional/unconditional TTS lane.
    pub max_tts_positions: u64,
    /// Maximum accepted codec frames; draining consumes no additional frame.
    pub max_codec_frames: u64,
}

/// Exact immutable released scheduling policy and explicit resource limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulePolicy {
    limits: ScheduleLimits,
    input_sample_rate: u64,
    output_sample_rate: u64,
    input_samples: u64,
    output_samples: u64,
    codec_delay: u64,
    tts_prefill: u64,
    vocabulary: usize,
    codebooks: usize,
    codebook_cardinality: usize,
    speech_controls: [usize; 3],
    guidance: bool,
}

impl SchedulePolicy {
    /// Admits the normalized release before deriving any scheduling geometry.
    /// No carrier, learned parameter or native state is allocated here.
    pub fn from_config(
        config: &VoiceChatConfig,
        limits: ScheduleLimits,
    ) -> Result<Self, ScheduleError> {
        config.validate()?;
        for (name, maximum) in [
            ("audio frames", limits.max_audio_frames),
            ("backbone positions", limits.max_backbone_positions),
            ("TTS positions", limits.max_tts_positions),
            ("codec frames", limits.max_codec_frames),
        ] {
            if maximum == 0 {
                return Err(ScheduleError::InvalidLimits(name));
            }
        }
        if limits.max_backbone_positions > i32::MAX as u64 {
            return Err(ScheduleError::InvalidLimits("backbone positions"));
        }
        if limits.max_tts_positions > i32::MAX as u64
            || limits.max_tts_positions > config.tts.backbone.max_position_embeddings as u64
        {
            return Err(ScheduleError::InvalidLimits("TTS positions"));
        }
        let prompt = AriaPromptPlan::from_config(config, i32::MAX as usize)?;
        let codec = waveform_decoder_config(&config.codec)?;
        let codec_delay = codec.spectral.delay_samples()?;
        Ok(Self {
            limits,
            input_sample_rate: config.cadence.source_sample_rate as u64,
            output_sample_rate: config.cadence.target_sample_rate as u64,
            input_samples: config.cadence.source_samples_per_frame as u64,
            output_samples: codec.samples_per_frame()? as u64,
            codec_delay: codec_delay as u64,
            tts_prefill: prompt.frames() as u64,
            // Exact shared vocabulary admitted by the released checkpoint schema.
            vocabulary: 131072,
            codebooks: config.codec.num_quantizers,
            codebook_cardinality: config.codec.codebook_size,
            speech_controls: [
                config.tokens.speech_pad,
                config.tokens.speech_bos,
                config.tokens.speech_eos,
            ],
            guidance: config.tts.guidance_enabled,
        })
    }

    /// Explicit capacities retained by both cold selection and live state.
    pub const fn limits(&self) -> ScheduleLimits {
        self.limits
    }
    /// Source sampling rate, independent of codec output time.
    pub const fn input_sample_rate(&self) -> u64 {
        self.input_sample_rate
    }
    /// Codec output sampling rate.
    pub const fn output_sample_rate(&self) -> u64 {
        self.output_sample_rate
    }
    /// Samples consumed by one fresh-audio step.
    pub const fn input_samples_per_frame(&self) -> u64 {
        self.input_samples
    }
    /// Samples emitted by one ordinary codec invocation.
    pub const fn output_samples_per_frame(&self) -> u64 {
        self.output_samples
    }
    /// Startup latency and nonempty final spectral tail in output samples.
    pub const fn output_delay_samples(&self) -> u64 {
        self.codec_delay
    }
    /// Fixed-Aria prefill positions in each enabled TTS lane.
    pub const fn tts_prefill_positions(&self) -> u64 {
        self.tts_prefill
    }

    /// Only agent sampling retains a categorical sampler; function decoding is greedy.
    pub fn generation_contract(&self) -> RealtimeGenerationContract<Self> {
        RealtimeGenerationContract::new(self.clone(), 1)
    }

    /// Exact per-invocation channels, independent of training-time delay fields.
    /// Published payloads use invocation coordinates; sample timestamps and
    /// distinct component frontiers are provided by each validated step plan.
    /// Recurrent feedback belongs to retained component state, not a fixed lag
    /// through this sparse publication history. Only universally present same-step
    /// payload dependencies are declared here; effective TTS text is separate from
    /// the sampled agent channel when facade policy supplies an override.
    pub fn frame_contract(&self) -> Result<RealtimeFrameContract, ScheduleError> {
        let categorical = |id, width, cardinality, feedback| RealtimeChannelDeclaration {
            id,
            payload: RealtimeChannelPayload::Categorical { width, cardinality },
            cadence: RealtimeChannelCadence {
                period: 1,
                phase: 0,
            },
            delay: 0,
            feedback,
        };
        let dense = |id, samples| RealtimeChannelDeclaration {
            id,
            payload: RealtimeChannelPayload::Dense {
                shape: vec![samples as usize],
            },
            cadence: RealtimeChannelCadence {
                period: 1,
                phase: 0,
            },
            delay: 0,
            feedback: vec![],
        };
        let domain = self
            .speech_controls
            .iter()
            .copied()
            .max()
            .unwrap()
            .max(self.codebook_cardinality - 1)
            .checked_add(1)
            .ok_or(ScheduleError::Arithmetic("speech domain"))?;
        let channels = RealtimeChannelLayout::new(
            "voicechat.coupled_invocations.v1".into(),
            vec![
                categorical(AGENT_CHANNEL, 1, self.vocabulary, vec![]),
                categorical(FUNCTION_CHANNEL, 1, self.vocabulary, vec![]),
                categorical(RVQ_CHANNEL, self.codebooks, domain, vec![]),
                dense(INPUT_CHANNEL, self.input_samples),
                RealtimeChannelDeclaration {
                    feedback: vec![RealtimeChannelFeedback {
                        source: RVQ_CHANNEL,
                        lag: 0,
                    }],
                    ..dense(OUTPUT_CHANNEL, self.output_samples)
                },
                dense(TAIL_CHANNEL, self.codec_delay),
            ],
        )?;
        Ok(RealtimeFrameContract::new(
            RealtimeIdentity::new("voicechat.coupled_invocations.v1")
                .expect("static nonempty policy identity"),
            self.schedule_descriptor(),
            channels,
            1,
        )?)
    }
}

impl RealtimeScheduleDescriptor for SchedulePolicy {
    fn schedule_descriptor(&self) -> Vec<u32> {
        // Domain VCSP, schema 2. All scalar values are low/high u64 limbs.
        // The version also binds the Step equations and initial-feedback policy.
        let mut words = vec![0x5643_5350, 2];
        for value in [
            self.input_sample_rate,
            self.output_sample_rate,
            self.input_samples,
            self.output_samples,
            self.codec_delay,
            self.tts_prefill,
            self.vocabulary as u64,
            self.codebooks as u64,
            self.codebook_cardinality as u64,
            self.speech_controls[0] as u64,
            self.speech_controls[1] as u64,
            self.speech_controls[2] as u64,
            TEXT_BOS_ID as u64,
            TEXT_EOS_ID as u64,
            TEXT_PAD_ID as u64,
            u64::from(self.guidance),
            self.limits.max_audio_frames,
            self.limits.max_backbone_positions,
            self.limits.max_tts_positions,
            self.limits.max_codec_frames,
        ] {
            words.extend([value as u32, (value >> 32) as u32]);
        }
        words
    }
}

/// Audio feature source for one tool-serialization backbone invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSource {
    /// Consume fresh PCM and advance perception and RNNT once.
    Fresh,
    /// Use a separately prepared silence embedding without consuming live PCM.
    Silence,
    /// Reuse retained projected features without recomputing perception or RNNT.
    Retained,
}

/// One architecture invocation intent; no application tool is executed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Once-only agent prompt and fixed-Aria TTS preparation, without codec work.
    Prefill {
        /// Actual framed agent-prompt positions; zero means no agent prompt.
        backbone_tokens: u64,
    },
    /// Consume one fresh audio frame and advance all coupled neural components.
    LiveAudio,
    /// Consume speech while retaining conversational/backbone and TTS positions.
    PerceptionOnly,
    /// Consume one previously projected tool-wait frame into Nemotron. This
    /// advances only the backbone and forced PAD history, without sampling.
    ReplayAudio,
    /// One function-serialization or response-injection substep.
    Tool {
        /// Explicit source of features supplied to the backbone.
        audio: AudioSource,
    },
    /// Supplied reminder text advances TTS and codec without advancing Nemotron.
    Reminder,
    /// Finish codec overlap without inventing a frame or advancing model caches.
    Drain,
}

/// Rendering attribution; facade policy supplies effective text and silence decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Ordinary live speech preserves generated waveform before feedback silence.
    Main,
    /// Tool speech may use the same silence substitution for waveform and feedback.
    Tool,
    /// Acknowledgment or reminder supplied by the conversational policy.
    Reminder,
}

/// Actual component counters supplied as a deterministic candidate receipt.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Consumed live-audio cadence frames.
    pub audio_frames: u64,
    /// Encoder output frames committed to the perception candidate.
    pub perception_frames: u64,
    /// Encoder frames accepted by RNNT, independent of emitted tokens.
    pub rnnt_frames: u64,
    /// Exclusive Nemotron cache position, including prompt prefill.
    pub backbone_positions: u64,
    /// Exclusive conditional TTS cache position.
    pub tts_conditional_positions: u64,
    /// Exclusive unconditional TTS cache position; zero when guidance is disabled.
    pub tts_unconditional_positions: u64,
    /// Complete residual-code frames accepted by the codec.
    pub codec_frames: u64,
    /// Emitted output samples, including startup region and final spectral tail.
    pub output_samples: u64,
    /// The codec tail has been drained and further work is forbidden.
    pub codec_finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Boundary {
    progress: Progress,
    prefilled: bool,
    backbone_decodes: u64,
    revision: u64,
}

/// Private mutable schedule with a single nonrecursive predecessor boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleState {
    policy: SchedulePolicy,
    current: Boundary,
    predecessor: Option<Boundary>,
    step: Option<Step>,
}

impl SnapshotValue for SchedulePolicy {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = std::mem::size_of::<Self>() as u64;
        Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(self.clone())
    }
}

impl SnapshotValue for ScheduleState {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = std::mem::size_of::<Self>() as u64;
        Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(self.clone())
    }
}
impl ScheduleState {
    /// Creates cold state; prefill is an explicit first transition.
    pub fn new(policy: SchedulePolicy) -> Self {
        Self {
            policy,
            current: Boundary {
                progress: Progress::default(),
                prefilled: false,
                backbone_decodes: 0,
                revision: 0,
            },
            predecessor: None,
            step: None,
        }
    }
    /// Exact immutable architecture policy.
    pub const fn policy(&self) -> &SchedulePolicy {
        &self.policy
    }
    /// Component frontiers at this committed or candidate boundary.
    pub const fn progress(&self) -> &Progress {
        &self.current.progress
    }

    /// Checks all component capacities before any native work is dispatched.
    pub fn plan(&self, step: Step) -> Result<StepPlan, ScheduleError> {
        if self.current.progress.codec_finished {
            return Err(ScheduleError::Finished);
        }
        if matches!(step, Step::Prefill { .. }) {
            if self.current.prefilled {
                return Err(ScheduleError::AlreadyPrefilled);
            }
        } else if !self.current.prefilled {
            return Err(ScheduleError::PrefillRequired);
        }
        if matches!(
            step,
            Step::Tool {
                audio: AudioSource::Retained
            }
        ) && self.current.progress.perception_frames == 0
        {
            return Err(ScheduleError::MissingRetainedAudio);
        }
        let mut next = self.current;
        next.revision = checked_add(next.revision, 1, "schedule revision")?;
        let p = &mut next.progress;
        match step {
            Step::Prefill { backbone_tokens } => {
                p.backbone_positions =
                    checked_add(p.backbone_positions, backbone_tokens, "backbone positions")?;
                p.tts_conditional_positions = self.policy.tts_prefill;
                p.tts_unconditional_positions = if self.policy.guidance {
                    self.policy.tts_prefill
                } else {
                    0
                };
                next.prefilled = true;
            }
            Step::LiveAudio
            | Step::PerceptionOnly
            | Step::Tool {
                audio: AudioSource::Fresh,
            } => {
                p.audio_frames = checked_add(p.audio_frames, 1, "audio frames")?;
                p.perception_frames = checked_add(p.perception_frames, 1, "perception frames")?;
                p.rnnt_frames = checked_add(p.rnnt_frames, 1, "RNNT frames")?;
            }
            Step::Drain => {
                if p.codec_frames > 0 {
                    p.output_samples =
                        checked_add(p.output_samples, self.policy.codec_delay, "output samples")?;
                }
                p.codec_finished = true;
            }
            Step::Tool { .. } | Step::Reminder | Step::ReplayAudio => {}
        }
        if matches!(
            step,
            Step::LiveAudio | Step::Tool { .. } | Step::ReplayAudio
        ) {
            p.backbone_positions = checked_add(p.backbone_positions, 1, "backbone positions")?;
            next.backbone_decodes =
                checked_add(next.backbone_decodes, 1, "backbone decode frames")?;
        }
        if matches!(step, Step::LiveAudio | Step::Tool { .. } | Step::Reminder) {
            p.tts_conditional_positions =
                checked_add(p.tts_conditional_positions, 1, "TTS positions")?;
            if self.policy.guidance {
                p.tts_unconditional_positions =
                    checked_add(p.tts_unconditional_positions, 1, "TTS positions")?;
            }
            p.codec_frames = checked_add(p.codec_frames, 1, "codec frames")?;
            p.output_samples = checked_add(
                p.output_samples,
                self.policy.output_samples,
                "output samples",
            )?;
        }
        let limits = self.policy.limits;
        for (component, required, capacity) in [
            ("audio frames", p.audio_frames, limits.max_audio_frames),
            (
                "backbone positions",
                p.backbone_positions,
                limits.max_backbone_positions,
            ),
            (
                "TTS positions",
                p.tts_conditional_positions,
                limits.max_tts_positions,
            ),
            (
                "TTS positions",
                p.tts_unconditional_positions,
                limits.max_tts_positions,
            ),
            ("codec frames", p.codec_frames, limits.max_codec_frames),
        ] {
            if required > capacity {
                return Err(ScheduleError::Capacity {
                    component,
                    required,
                    capacity,
                });
            }
        }
        Ok(StepPlan {
            policy: self.policy.clone(),
            before: self.current,
            after: next,
            step,
        })
    }
}

impl RealtimeScheduleState for ScheduleState {
    type Identity = SchedulePolicy;
    type Error = ScheduleError;
    fn identity(&self) -> &Self::Identity {
        &self.policy
    }
    fn validate_successor(&self, next: &Self) -> Result<(), Self::Error> {
        if self.policy != next.policy {
            return Err(ScheduleError::IdentityMismatch);
        }
        if next.predecessor != Some(self.current) {
            return Err(ScheduleError::StalePredecessor);
        }
        let step = next.step.ok_or(ScheduleError::InvalidSuccessor)?;
        let planned = self.plan(step)?;
        if planned.after != next.current {
            return Err(ScheduleError::InvalidSuccessor);
        }
        Ok(())
    }
}

/// Validated independent component work and the exact expected receipt.
#[derive(Debug, Clone)]
pub struct StepPlan {
    policy: SchedulePolicy,
    before: Boundary,
    after: Boundary,
    step: Step,
}
impl StepPlan {
    /// Exact selected policy used to declare output geometry and sample rates.
    pub const fn policy(&self) -> &SchedulePolicy {
        &self.policy
    }
    /// Zero-based invocation coordinate, independent of audio and model clocks.
    pub const fn invocation_index(&self) -> u64 {
        self.before.revision
    }
    /// Invocation intent admitted by this plan.
    pub const fn step(&self) -> Step {
        self.step
    }
    /// Exact predecessor component frontiers.
    pub const fn before(&self) -> &Progress {
        &self.before.progress
    }
    /// Required candidate component frontiers after successful work.
    pub const fn expected(&self) -> &Progress {
        &self.after.progress
    }
    /// Number of new Nemotron positions, including any prefill.
    pub fn backbone_tokens(&self) -> u64 {
        self.after.progress.backbone_positions - self.before.progress.backbone_positions
    }
    /// Whether the source PCM and perception pipeline runs.
    pub fn runs_perception(&self) -> bool {
        self.after.progress.perception_frames != self.before.progress.perception_frames
    }
    /// Whether RNNT consumes new unprojected perception output.
    pub fn runs_rnnt(&self) -> bool {
        self.after.progress.rnnt_frames != self.before.progress.rnnt_frames
    }
    /// Whether either TTS prefill or cached generation advances its causal state.
    pub fn runs_tts(&self) -> bool {
        self.after.progress.tts_conditional_positions
            != self.before.progress.tts_conditional_positions
    }
    /// Whether a complete generated code frame is decoded; drain is separate.
    pub fn runs_codec(&self) -> bool {
        self.after.progress.codec_frames != self.before.progress.codec_frames
    }
    /// Whether both feedback channels start from the released PAD IDs.
    pub fn uses_initial_feedback(&self) -> bool {
        matches!(self.step, Step::LiveAudio | Step::Tool { .. })
            && self.before.backbone_decodes == 0
    }
    /// Agent/function feedback IDs for the first backbone decode after any prefill.
    pub fn initial_feedback_tokens(&self) -> Option<[i32; 2]> {
        self.uses_initial_feedback().then_some([TEXT_PAD_ID; 2])
    }
    /// Whether agent sampling precedes forced effective-token overrides.
    /// Prefill randomness is outside this clock plan and is not implied here.
    pub fn agent_sampling_before_overrides(&self) -> bool {
        matches!(self.step, Step::LiveAudio | Step::Tool { .. })
    }
    /// Rendering attribution for generated speech; supplies no facade silence decision.
    pub const fn render_mode(&self) -> Option<RenderMode> {
        match self.step {
            Step::LiveAudio => Some(RenderMode::Main),
            Step::Tool { .. } => Some(RenderMode::Tool),
            Step::Reminder => Some(RenderMode::Reminder),
            _ => None,
        }
    }
    /// Absolute start of newly emitted PCM, including startup region and drain.
    pub fn output_start_sample(&self) -> Option<u64> {
        (self.output_samples() > 0).then_some(self.before.progress.output_samples)
    }
    /// Number of newly emitted PCM samples; drain does not invent a codec frame.
    pub fn output_samples(&self) -> u64 {
        self.after.progress.output_samples - self.before.progress.output_samples
    }
    /// Codec startup latency, independent of emitted-sample counters.
    pub const fn output_delay_samples(&self) -> u64 {
        self.policy.codec_delay
    }
    /// Verifies actual candidate component frontiers without changing the parent.
    /// The caller still owes exact native completion before publishing this state.
    pub fn complete(self, observed: Progress) -> Result<ScheduleState, ScheduleError> {
        if observed != self.after.progress {
            return Err(ScheduleError::ReceiptMismatch {
                expected: self.after.progress,
                observed,
            });
        }
        Ok(ScheduleState {
            policy: self.policy,
            current: self.after,
            predecessor: Some(self.before),
            step: Some(self.step),
        })
    }
}

fn checked_add(before: u64, count: u64, component: &'static str) -> Result<u64, ScheduleError> {
    before
        .checked_add(count)
        .ok_or(ScheduleError::Arithmetic(component))
}

/// Portable admission, capacity and exact-receipt errors before publication.
#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    /// Normalized artifact configuration differs from the pinned release.
    #[error(transparent)]
    Configuration(#[from] super::config::ConfigError),
    /// Fixed-Aria startup geometry is invalid.
    #[error(transparent)]
    Prompt(#[from] super::tts_mog::TtsError),
    /// Codec geometry could not be normalized.
    #[error(transparent)]
    Codec(#[from] eredu_codec::Error),
    /// Explicit capacities are zero or exceed neural coordinate bounds.
    #[error("invalid VoiceChat schedule limit for {0}")]
    InvalidLimits(&'static str),
    /// A candidate exceeds its explicit retained-context bound.
    #[error("VoiceChat {component} requires {required}, capacity is {capacity}")]
    Capacity {
        /// Resource that would exceed its retained bound.
        component: &'static str,
        /// Required exclusive frontier.
        required: u64,
        /// Admitted maximum frontier.
        capacity: u64,
    },
    /// Counter arithmetic cannot represent a requested successor.
    #[error("VoiceChat schedule arithmetic overflow for {0}")]
    Arithmetic(&'static str),
    /// Every active stream starts with explicit startup preparation.
    #[error("VoiceChat schedule requires prefill before component advancement")]
    PrefillRequired,
    /// Retained-feature tool work requires an earlier fresh perception invocation.
    #[error("VoiceChat retained audio requires a previous perception frame")]
    MissingRetainedAudio,
    /// Prefill is once-only and cannot reset cached state.
    #[error("VoiceChat schedule is already prefilled")]
    AlreadyPrefilled,
    /// Codec drain is terminal for this state.
    #[error("VoiceChat schedule is finished")]
    Finished,
    /// A successor carries a different immutable scheduling contract.
    #[error("VoiceChat schedule policy identity differs")]
    IdentityMismatch,
    /// A successor was derived from another state or skips an intermediate transition.
    #[error("VoiceChat schedule predecessor does not match the current boundary")]
    StalePredecessor,
    /// Private transition metadata does not describe the declared step.
    #[error("VoiceChat schedule successor does not match its planned transition")]
    InvalidSuccessor,
    /// Actual candidate neural/host counters differ from the preflight plan.
    #[error("VoiceChat component progress does not match the planned receipt")]
    ReceiptMismatch {
        /// Exact progress admitted by preflight.
        expected: Progress,
        /// Observed candidate component progress.
        observed: Progress,
    },
    /// Typed frame-channel declarations failed neutral validation.
    #[error(transparent)]
    Channels(#[from] eredu_core::RealtimeChannelError),
    /// Cold frame contract could not retain the complete policy.
    #[error(transparent)]
    Contract(#[from] eredu_runtime::RealtimeFrameContractError),
}
