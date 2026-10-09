//! Facade invocation assembly over the existing backend submission boundary.

use eredu_architectures::voicechat::{
    channel_decision::DecisionSource,
    channels::ChannelError,
    execution::{Invocation, ToolAudio},
    input::PcmInputState,
    observation::OwnedHostOutput,
    schedule::AudioSource,
    tts_ingress::PromptLayout,
};
use eredu_core::{scheduler::TransitionOutput, BackendFailure, Completion};
use eredu_runtime::{PrepublicationRealtimeFrame, RealtimeFrameHostObserver};

use super::{
    PreparedVoiceChatPrompt, VoiceChatPublicationBranch, VoiceChatPublicationError, VoiceChatWork,
    VoiceChatWorkKind,
};

/// Explicit streaming output flags, independent of the TTS model's input policy.
/// The preparation owner supplies these values; execution reads no environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatSpeechOptions {
    /// Main-loop EOS feedback substitution; generated waveform remains intact.
    pub force_silence_on_eos: bool,
    /// Permits PAD silence under the committed main-tail or tool-idle policy.
    pub force_silence_on_pad: bool,
}

/// Already prepared inputs borrowed while submitting an ordinary invocation.
///
/// Speaker codes/layout come from architecture-owned Aria prompt preparation;
/// conditioning is its corresponding character-encoder output. Silence is an
/// independently prepared projected perception embedding from architecture-owned
/// `PreparedSilenceConditioning`, computed before live input advancement.
/// The native executor validates tensor geometry and retains submitted inputs
/// through exact completion. This view opens no artifacts and selects no device.
pub struct VoiceChatExecutionInputs<'a, T> {
    /// Prepared system/tool prompt; absent skips only the backbone prompt.
    pub prompt: Option<&'a PreparedVoiceChatPrompt>,
    /// Shift-ready fixed-speaker prompt codes.
    pub speaker_codes: &'a T,
    /// Matching BOS/pre-BOS mask geometry.
    pub speaker_layout: &'a PromptLayout,
    /// Prepared character conditioning; absent uses architecture-owned preparation.
    pub speaker_conditioning: Option<&'a T>,
    /// Prepared silence embedding required by silence-backed tool steps.
    pub silence: Option<&'a T>,
    /// Main/tool output gating options; reminder policy belongs to its receipt.
    pub speech: VoiceChatSpeechOptions,
}

/// Portable failure while assembling or submitting prepared work.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatExecutionError {
    /// Work admission, PCM staging or host publication was rejected.
    #[error(transparent)]
    Publication(#[from] VoiceChatPublicationError),
    /// Prompt framing cannot satisfy its declared architecture policy.
    #[error(transparent)]
    Prompt(#[from] ChannelError),
    /// The prepared prompt differs from the work admitted before enqueue.
    #[error(
        "VoiceChat prefill work declares {declared} positions but prepared prompt has {prepared}"
    )]
    PrefillCount {
        /// Count frozen in the work descriptor.
        declared: u64,
        /// Framed count of the supplied prepared prompt.
        prepared: u64,
    },
    /// The default tool policy cannot execute without its prepared silence feature.
    #[error("VoiceChat tool execution requires a prepared silence embedding")]
    MissingSilence,
    /// Native submission error preserving its complete original source chain.
    #[error("VoiceChat execution submission failed: {0}")]
    Backend(#[source] BackendFailure),
}

impl<T> VoiceChatExecutionInputs<'_, T> {
    /// Exact backbone prefill count to pass to `prepare_prefill`.
    pub fn backbone_prompt_positions(&self) -> Result<u64, VoiceChatExecutionError> {
        Ok(self
            .prompt
            .and_then(PreparedVoiceChatPrompt::invocation_prompt)
            .map(|(tokens, policy)| policy.framed_length(tokens))
            .transpose()?
            .unwrap_or(0) as u64)
    }
}

impl<D: 'static> VoiceChatWork<D> {
    /// Admits frozen work, submits the matching architecture invocation and adds
    /// mandatory host publication to its ordinary completion gate.
    ///
    /// The executor closure owns native scopes, input retention and safe recovery
    /// if submission fails after work starts. It returns the existing prepublication
    /// gate. The caller discards a failed branch through its ordinary transaction;
    /// completion failure alone never authorizes early native-resource release.
    /// Prepared-input rejections occur before consuming this work's admission.
    pub fn submit<T, B, F, C, H>(
        &self,
        branch: &mut VoiceChatPublicationBranch<B, D>,
        inputs: &VoiceChatExecutionInputs<'_, T>,
        submit: impl FnOnce(
            &mut B,
            &Invocation<'_, T>,
        ) -> Result<PrepublicationRealtimeFrame<F, C, H>, BackendFailure>,
    ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError>
    where
        B: PcmInputState,
        F: 'static,
        C: Completion + 'static,
        C::Error: Send + Sync + 'static,
        H: RealtimeFrameHostObserver<F, Output = OwnedHostOutput<D>> + 'static,
        H::Error: Send + Sync,
    {
        match self.kind() {
            VoiceChatWorkKind::Prefill { backbone_tokens } => {
                let prepared = inputs.backbone_prompt_positions()?;
                if prepared != backbone_tokens {
                    return Err(VoiceChatExecutionError::PrefillCount {
                        declared: backbone_tokens,
                        prepared,
                    });
                }
            }
            VoiceChatWorkKind::Tool {
                audio: AudioSource::Silence,
                ..
            } if inputs.silence.is_none() => {
                return Err(VoiceChatExecutionError::MissingSilence);
            }
            _ => {}
        }
        let admission = self.admit(branch)?;
        let neural = branch.neural_state_mut();
        let gate = match self.kind() {
            VoiceChatWorkKind::Prefill { .. } => submit(
                neural,
                &Invocation::Prefill {
                    prompt: inputs
                        .prompt
                        .and_then(PreparedVoiceChatPrompt::invocation_prompt),
                    codes: inputs.speaker_codes,
                    layout: inputs.speaker_layout,
                    conditioning: inputs.speaker_conditioning,
                },
            ),
            VoiceChatWorkKind::Live => {
                let resolver = admission
                    .resolver()
                    .map_err(|error| VoiceChatPublicationError::Conversation(error.into()))?;
                submit(
                    neural,
                    &Invocation::Live {
                        decisions: DecisionSource::Resolved(&resolver),
                        output: resolver.output_policy(
                            inputs.speech.force_silence_on_eos,
                            inputs.speech.force_silence_on_pad,
                        ),
                    },
                )
            }
            VoiceChatWorkKind::PerceptionOnly => submit(neural, &Invocation::PerceptionOnly),
            VoiceChatWorkKind::ReplayAudio => submit(neural, &Invocation::ReplayAudio),
            VoiceChatWorkKind::Tool { audio, decisions } => submit(
                neural,
                &Invocation::Tool {
                    audio: match audio {
                        AudioSource::Fresh => ToolAudio::Fresh,
                        AudioSource::Retained => ToolAudio::Retained,
                        AudioSource::Silence => {
                            ToolAudio::Silence(inputs.silence.expect("validated silence input"))
                        }
                    },
                    decisions: DecisionSource::Fixed(decisions),
                    output: admission.tool_output_policy(inputs.speech.force_silence_on_pad),
                },
            ),
            VoiceChatWorkKind::Reminder(receipt) => submit(
                neural,
                &Invocation::Reminder {
                    spoken: receipt.spoken(),
                    output: receipt.output_policy(),
                },
            ),
            VoiceChatWorkKind::Drain => submit(neural, &Invocation::Drain),
        }
        .map_err(VoiceChatExecutionError::Backend)?;
        Ok(VoiceChatSubmittedWork {
            inner: Box::new(Submitted(admission.attach(gate))),
        })
    }
}

/// Owned, backend-neutral completion accepted by the ordinary shared scheduler.
/// The erased inner gate retains native resources and performs host validation
/// exactly once. It borrows neither the executor nor prepared input views.
pub struct VoiceChatSubmittedWork {
    inner: Box<dyn TransitionOutput<Error = BackendFailure>>,
}

impl TransitionOutput for VoiceChatSubmittedWork {
    type Error = BackendFailure;
    fn is_complete(&self) -> Result<bool, BackendFailure> {
        self.inner.is_complete()
    }
    fn retained_resources(&self) -> usize {
        self.inner.retained_resources()
    }
    fn backend_name(&self) -> Option<String> {
        self.inner.backend_name()
    }
    fn physically_preemptible(&self) -> bool {
        self.inner.physically_preemptible()
    }
}

// Native completion/observer errors remain implementation details. The shared
// gate still owns observation, exact completion and retained-resource lifetime.
struct Submitted<O>(O);
impl<O> TransitionOutput for Submitted<O>
where
    O: TransitionOutput,
    O::Error: Send + Sync + 'static,
{
    type Error = BackendFailure;
    fn is_complete(&self) -> Result<bool, BackendFailure> {
        self.0.is_complete().map_err(|error| {
            BackendFailure::from_error(error).with_operation("VoiceChat completion and publication")
        })
    }
    fn retained_resources(&self) -> usize {
        self.0.retained_resources()
    }
    fn backend_name(&self) -> Option<String> {
        self.0.backend_name()
    }
    fn physically_preemptible(&self) -> bool {
        self.0.physically_preemptible()
    }
}

#[cfg(feature = "mlx")]
mod mlx;
#[cfg(feature = "mlx")]
pub use mlx::submit_mlx_voicechat;
