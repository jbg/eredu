//! Cold request admission over exact retained artifacts, before native binding.

use std::time::{Duration, Instant};

use eredu_architectures::voicechat::{
    agent_sampling::AgentSamplingPolicy,
    artifact::RetainedVoiceChatArtifacts,
    channels::PromptPolicy,
    components::{PreparedVoiceChatConstruction, VoiceChatConstructionError},
    frontend::{FrontendError, FrontendOptions, VoiceChatFrontend},
    perception_rnnt::{PerceptionRnntError, PerceptionRnntLimits},
    preparation::{GenerationPreparationError, PreparedGeneration},
    schedule::{ScheduleLimits, ScheduleState, Step},
    transcription::{RnntTextDecoder, RnntTextError},
    tts_mog::TtsError,
    tts_prompt::AriaPromptPlan,
};
use eredu_core::scheduler::RequestId;
use serde_json::Value;

use super::{
    ingress::Ingress, publication_queue::Queue, PreparedVoiceChatPrompt, VoiceChatConversation,
    VoiceChatConversationError, VoiceChatDecodeLimits, VoiceChatInputConfig, VoiceChatInputError,
    VoiceChatPhrasePolicy, VoiceChatPublicationCapacity, VoiceChatPublicationError,
    VoiceChatPublicationLimits, VoiceChatSessionConfig, VoiceChatSessionError,
    VoiceChatSpeechOptions, VoiceChatTextError, VoiceChatTextLimits, VoiceChatTextPreparation,
    VoiceChatToolChannel, VoiceChatToolError, VoiceChatToolLimits, VoiceChatTurnLimits,
    VoiceChatTurnPolicy,
};

/// Explicit portable controls for starting a single-row streaming conversation.
/// Recorded timing is supplied by the caller; preparation reads no wall clock.
#[derive(Debug, Clone)]
pub struct VoiceChatStartOptions {
    /// Shared scheduler request identity.
    pub request: RequestId,
    /// ASCII system instructions, rendered with the tool definitions.
    pub system_prompt: String,
    /// Existing facade function-tool JSON schemas.
    pub tools: Vec<Value>,
    /// Disabled, fixed, or seeded selection from the released acknowledgement pool.
    /// Selection commits with each new call and is independent of neural sampling.
    /// The text limits bound each complete BOS/text/PAD-tail/EOS framing.
    pub acknowledgement: VoiceChatPhrasePolicy,
    /// Initial tool-wait speech selected at call start and retained until playback.
    pub initial_reminder: VoiceChatPhrasePolicy,
    /// Bounded input text and tokenization.
    pub text_limits: VoiceChatTextLimits,
    /// Architecture-owned BOS/EOS framing and repetition policy.
    pub prompt_policy: PromptPolicy,
    /// Maximum incomplete function payload and complete tool batch.
    pub tool_limits: VoiceChatToolLimits,
    /// Independent audio, backbone, TTS and codec context capacities.
    pub schedule_limits: ScheduleLimits,
    /// Released agent sampling policy or explicit validated overrides.
    pub sampling: AgentSamplingPolicy,
    /// RNNT-assisted turn policy.
    pub turn_policy: VoiceChatTurnPolicy,
    /// Acoustic and model frontiers plus turn-history bounds.
    pub turn_limits: VoiceChatTurnLimits,
    /// Agent tokenizer lookbehind and decoded text bounds.
    pub decode_limits: VoiceChatDecodeLimits,
    /// Maximum incomplete RNNT text retained by its vocabulary decoder.
    pub transcription_pending_bytes: usize,
    /// Host PCM precision, rolling history and internal queue geometry.
    pub frontend: FrontendOptions,
    /// External PCM backpressure and precision matching the frontend.
    pub input_config: VoiceChatInputConfig,
    /// Perception and RNNT execution capacities.
    pub input_limits: PerceptionRnntLimits,
    /// Maximum packed character cells for prompt and cached TTS conditioning.
    pub max_character_cells: usize,
    /// Maximum single-row fixed-Aria startup positions.
    pub max_prompt_positions: usize,
    /// Total undelivered committed publications and payload capacity.
    pub publication_limits: VoiceChatPublicationLimits,
    /// Per-invocation payload reservation within the publication limits.
    pub publication_capacity: VoiceChatPublicationCapacity,
    /// Caller-selected origin for the ordinary scheduler's timestamps.
    pub clock_origin: Instant,
    /// Explicit timestamp attributed to prompt and fixed-Aria startup.
    pub start_at: Duration,
    /// Released waveform substitution policy or explicit overrides.
    pub speech: VoiceChatSpeechOptions,
}

impl VoiceChatStartOptions {
    /// Admits recorded timing and matching bounded host queues without artifacts.
    /// Text, tokenizer, turn policy and learned geometry are admitted by preparation.
    pub fn validate_host(&self) -> Result<(), VoiceChatLoadingError> {
        self.clock_origin
            .checked_add(self.start_at)
            .ok_or(VoiceChatSessionError::ClockCapacity)?;
        if self.input_config.pcm_precision != self.frontend.pcm_precision {
            return Err(VoiceChatInputError::Configuration(
                "external PCM precision differs from the retained frontend",
            )
            .into());
        }
        self.input_limits.validate_startup()?;
        // Reuse ordinary queue construction and reservation admission. These
        // temporary host owners carry no audio and cannot submit or publish work.
        let _ = Ingress::new(self.input_config)?;
        let queue = Queue::<()>::new(self.publication_limits)?;
        let _ = Queue::reserve(&queue, self.publication_capacity)?;
        Ok(())
    }
}

/// Portable host or architecture rejection before native model construction.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatLoadingError {
    /// The retained artifacts cannot construct the complete coupled task.
    #[error(transparent)]
    Construction(#[from] VoiceChatConstructionError),
    /// Input precision or bounded ingress configuration is invalid.
    #[error(transparent)]
    Input(#[from] VoiceChatInputError),
    /// Retained preprocessing or the requested frontend is invalid.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Perception or RNNT capacity cannot admit an initial acoustic frame.
    #[error(transparent)]
    Perception(#[from] PerceptionRnntError),
    /// System text, tool rendering or tokenizer bounds are invalid.
    #[error(transparent)]
    Text(#[from] VoiceChatTextError),
    /// Tool schemas or payload limits are invalid.
    #[error(transparent)]
    Tool(#[from] VoiceChatToolError),
    /// Turn-taking or agent-text decoding policy is invalid.
    #[error(transparent)]
    Conversation(#[from] VoiceChatConversationError),
    /// RNNT vocabulary decoding cannot be initialized.
    #[error(transparent)]
    Transcription(#[from] RnntTextError),
    /// Shared generation controls or prompt capacity are invalid.
    #[error(transparent)]
    Generation(#[from] GenerationPreparationError),
    /// Fixed-Aria prompt geometry or character capacity is invalid.
    #[error(transparent)]
    Speech(#[from] TtsError),
    /// The output queue cannot admit one invocation's reservation.
    #[error(transparent)]
    Publication(#[from] VoiceChatPublicationError),
    /// Recorded startup time cannot be represented by the caller's clock origin.
    #[error(transparent)]
    Session(#[from] VoiceChatSessionError),
}

/// Admitted host request and architecture construction authority.
/// No scheduler, native tensor, cache, stream or device is created here.
pub struct PreparedVoiceChatSession<'a> {
    parts: PreparedVoiceChatSessionParts<'a>,
}

/// Consuming inputs for a backend realization of the ordinary session path.
/// Learned preparation retains the original artifact guards and source views.
pub struct PreparedVoiceChatSessionParts<'a> {
    /// Exact cold construction of the coupled learned components.
    pub construction: PreparedVoiceChatConstruction<'a>,
    /// One validated shared generation contract.
    pub generation: PreparedGeneration,
    /// Initial host conversation policy with the exact tokenizer and tool schemas.
    pub conversation: VoiceChatConversation,
    /// Rendered and tokenized prompt for the explicit prefill invocation.
    pub prompt: PreparedVoiceChatPrompt,
    /// Matching host preprocessing options.
    pub frontend: FrontendOptions,
    /// External PCM queue controls.
    pub input_config: VoiceChatInputConfig,
    /// Undelivered publication queue controls.
    pub publication_limits: VoiceChatPublicationLimits,
    /// Shared scheduler startup with derived backbone prompt positions.
    pub session_config: VoiceChatSessionConfig,
    /// Perception/RNNT execution capacities.
    pub input_limits: PerceptionRnntLimits,
    /// Maximum packed character cells per conditioning invocation.
    pub max_character_cells: usize,
    /// Maximum single-row fixed-Aria startup positions.
    pub max_prompt_positions: usize,
    /// Explicit waveform substitution flags.
    pub speech: VoiceChatSpeechOptions,
}

impl<'a> PreparedVoiceChatSession<'a> {
    /// Validates one request using retained artifacts only, before native binding.
    /// The system prompt and call validator share the same tool definitions.
    pub fn from_retained(
        artifacts: &'a RetainedVoiceChatArtifacts,
        options: &VoiceChatStartOptions,
    ) -> Result<Self, VoiceChatLoadingError> {
        options.validate_host()?;
        let _ = VoiceChatFrontend::from_retained(artifacts, options.frontend)?;
        let construction = PreparedVoiceChatConstruction::from_retained(artifacts)?;
        let text =
            VoiceChatTextPreparation::new(construction.host_conditioning(), options.text_limits)?;
        let channel = VoiceChatToolChannel::new(
            text,
            options.request,
            options.tools.clone(),
            options.tool_limits,
        )?;
        let prompt =
            channel.prepare_system_prompt(&options.system_prompt, options.prompt_policy)?;
        let backbone_prompt_positions = prompt
            .invocation_prompt()
            .map(|(tokens, policy)| policy.framed_length(tokens))
            .transpose()
            .map_err(VoiceChatTextError::from)?
            .unwrap_or(0) as u64;
        let generation =
            construction.prepare_generation(options.schedule_limits, options.sampling)?;
        ScheduleState::new(generation.schedule().clone())
            .plan(Step::Prefill {
                backbone_tokens: backbone_prompt_positions,
            })
            .map_err(GenerationPreparationError::from)?;
        let aria = AriaPromptPlan::from_config(artifacts.plan().config(), usize::MAX)?;
        aria.characters(
            &construction.host_conditioning().characters,
            1,
            options.max_prompt_positions,
            options.max_character_cells,
        )?;
        let mut conversation = VoiceChatConversation::new(
            channel,
            options.turn_policy,
            options.turn_limits,
            RnntTextDecoder::from_assets(artifacts.assets(), options.transcription_pending_bytes)?,
            options.decode_limits,
        )?;
        conversation
            .prepare_acknowledgement(&options.acknowledgement, options.text_limits.max_tokens)?;
        conversation
            .prepare_initial_reminder(&options.initial_reminder, options.text_limits.max_tokens)?;
        Ok(Self {
            parts: PreparedVoiceChatSessionParts {
                construction,
                generation,
                conversation,
                prompt,
                frontend: options.frontend,
                input_config: options.input_config,
                publication_limits: options.publication_limits,
                session_config: VoiceChatSessionConfig {
                    clock_origin: options.clock_origin,
                    start_at: options.start_at,
                    backbone_prompt_positions,
                    publication_capacity: options.publication_capacity,
                },
                input_limits: options.input_limits,
                max_character_cells: options.max_character_cells,
                max_prompt_positions: options.max_prompt_positions,
                speech: options.speech,
            },
        })
    }

    #[cfg(feature = "mlx")]
    pub(crate) fn construction(&self) -> &PreparedVoiceChatConstruction<'_> {
        &self.parts.construction
    }

    /// Cold generation limits and mutable projected-audio resource geometry.
    pub fn generation(&self) -> &PreparedGeneration {
        &self.parts.generation
    }

    /// Moves the complete admitted request into its selected backend realization.
    pub fn into_parts(self) -> PreparedVoiceChatSessionParts<'a> {
        self.parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_architectures::voicechat::{frontend::PcmPrecision, rnnt::RnntLimits};

    #[test]
    fn cold_start_admits_matching_input_and_complete_publication_reservation() {
        let capacity = VoiceChatPublicationCapacity {
            speech_samples: 1764,
            ..Default::default()
        };
        let mut options = VoiceChatStartOptions {
            request: RequestId::new(1),
            system_prompt: String::new(),
            tools: Vec::new(),
            acknowledgement: VoiceChatPhrasePolicy::Disabled,
            initial_reminder: VoiceChatPhrasePolicy::Disabled,
            text_limits: VoiceChatTextLimits {
                max_bytes: 1024,
                max_tokens: 128,
            },
            prompt_policy: PromptPolicy {
                repeat: 1,
                max_tokens: 130,
            },
            tool_limits: VoiceChatToolLimits {
                max_payload_tokens: 128,
                max_payload_bytes: 1024,
                max_calls: 2,
            },
            schedule_limits: ScheduleLimits {
                max_audio_frames: 128,
                max_backbone_positions: 256,
                max_tts_positions: 256,
                max_codec_frames: 256,
            },
            sampling: AgentSamplingPolicy::streaming_defaults(256).unwrap(),
            turn_policy: VoiceChatTurnPolicy::default(),
            turn_limits: VoiceChatTurnLimits {
                max_audio_frames: 128,
                max_model_steps: 256,
                max_history: 40,
            },
            decode_limits: VoiceChatDecodeLimits::default(),
            transcription_pending_bytes: 1024,
            frontend: FrontendOptions::default(),
            input_config: VoiceChatInputConfig::default(),
            input_limits: PerceptionRnntLimits {
                frame_capacity: 128,
                rnnt: RnntLimits {
                    max_symbols_per_frame: 4,
                    hypothesis_capacity: 512,
                },
            },
            max_character_cells: 1024,
            max_prompt_positions: 37,
            publication_limits: VoiceChatPublicationLimits {
                max_publications: 2,
                capacity,
            },
            publication_capacity: capacity,
            clock_origin: Instant::now(),
            start_at: Duration::ZERO,
            speech: VoiceChatSpeechOptions {
                force_silence_on_eos: true,
                force_silence_on_pad: true,
            },
        };
        options.validate_host().unwrap();
        options.input_config.pcm_precision = PcmPrecision::F32;
        assert!(matches!(
            options.validate_host(),
            Err(VoiceChatLoadingError::Input(
                VoiceChatInputError::Configuration(_)
            ))
        ));
        options.input_config.pcm_precision = options.frontend.pcm_precision;
        options.publication_capacity.speech_samples = 1765;
        assert!(matches!(
            options.validate_host(),
            Err(VoiceChatLoadingError::Publication(
                VoiceChatPublicationError::Capacity("speech samples")
            ))
        ));
    }
}
