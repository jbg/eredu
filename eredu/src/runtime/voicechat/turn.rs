//! RNNT turn policy over candidate decisions and committed host output.
//!
//! The core ordering follows the pinned Speech wrapper's
//! `_rnnt_step` and `_apply_rnnt_turn_taking` (revision
//! `097dfe9e2f55baf653b83035868bdc89849f1b47`). Optional ASR-head, redirect and
//! watchdog policies are separate from this RNNT policy.

use std::collections::VecDeque;

use eredu_architectures::voicechat::{
    channel_decision::{ChannelDecisionPolicy, DecisionEvidence, DecisionInput, DecisionResolver},
    channels::TEXT_VOCABULARY_SIZE,
    host::TextControls,
    observation::OwnedHostOutput,
    schedule::{AudioSource, Progress, Step},
    tts_feedback::{OutputPolicy, WaveformSilence},
};
use eredu_core::BackendFailure;
use eredu_runtime::TokenDomain;

use super::{VoiceChatToolChannel, VoiceChatToolError, VoiceChatTurnIdentity};

/// RNNT policy defaults from the pinned streaming configuration and wrapper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceChatTurnPolicy {
    /// Enables RNNT-driven overrides; acoustic evidence still advances when off.
    pub enabled: bool,
    /// Prior generated positions examined for BOS/EOS, excluding the current one.
    pub lookback: usize,
    /// Consecutive blank frames to end a confirmed user utterance.
    pub eou_frames: u64,
    /// Blank threshold for the user's first utterance, independent of a greeting.
    pub first_eou_frames: u64,
    /// Consecutive speech frames that interrupt an active agent turn or pending
    /// tool work. The facade applies this bound during silent tool waits too.
    pub bou_frames: u64,
    /// Minimum speech evidence after the first user utterance.
    pub min_speech_frames: u64,
    /// Minimum speech evidence for the user's first utterance.
    pub first_min_speech_frames: u64,
    /// Low nonzero density increases the speech-confirmation minimum.
    pub density_threshold: f32,
    /// Minimum evidence under the low-density rule.
    pub low_density_min_speech: u64,
    /// Silence clears unconfirmed accumulated noise while the agent is not speaking.
    pub noise_reset_frames: u64,
    /// First-turn fallback position in generated decisions, excluding prompt prefill.
    pub first_turn_fallback: u64,
    /// Main TTS PAD-tail budget per content token; zero disables the tail gate.
    pub pad_tail_ratio: f64,
}

impl Default for VoiceChatTurnPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            lookback: 40,
            eou_frames: 40,
            first_eou_frames: 40,
            bou_frames: 40,
            min_speech_frames: 3,
            first_min_speech_frames: 2,
            density_threshold: 0.15,
            low_density_min_speech: 6,
            noise_reset_frames: 10,
            first_turn_fallback: 50,
            pad_tail_ratio: 3.0,
        }
    }
}

/// Independent bounds on acoustic time, generated positions and retained history.
#[derive(Debug, Clone, Copy)]
pub struct VoiceChatTurnLimits {
    /// Fresh input frames, including perception-only work during tool waits.
    pub max_audio_frames: u64,
    /// Backbone feedback positions, including tool substeps and silent audio
    /// replay but excluding prompt prefill.
    pub max_model_steps: u64,
    /// Maximum admitted lookback allocation.
    pub max_history: usize,
}

/// Why the current committed agent token changes the conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatTurnEvent {
    /// The model generated BOS without an RNNT override.
    AgentStarted,
    /// Confirmed user speech followed by silence forces agent BOS.
    UserEnded,
    /// Confirmed first-turn speech forces BOS at the fallback position.
    FirstTurnFallback,
    /// The model generated EOS without an RNNT override.
    AgentEnded,
    /// Sustained user speech terminates the current agent/tool turn and
    /// invalidates outstanding tool work.
    Interrupted,
}

/// Host-policy errors preserve canonical state on rejected observations.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatTurnError {
    /// Configuration or explicit history bounds are invalid.
    #[error("invalid VoiceChat turn policy: {0}")]
    Configuration(&'static str),
    /// Explicit acoustic or model capacity is exhausted.
    #[error("VoiceChat turn {0} capacity exhausted")]
    Capacity(&'static str),
    /// Evidence does not describe the next fresh audio frame.
    #[error("VoiceChat turn evidence differs from the audio frontier")]
    Evidence,
    /// Output is duplicated, reordered or does not match this policy's candidate.
    #[error("VoiceChat turn output differs from its expected committed transition")]
    Attribution,
    /// A request or turn identity has been invalidated.
    #[error("stale VoiceChat turn observation")]
    Stale,
    /// Invalid raw or effective text token.
    #[error("VoiceChat turn token {0} is outside the released vocabulary")]
    Token(u32),
    /// Existing tool-channel interruption rejection.
    #[error(transparent)]
    Tool(#[from] VoiceChatToolError),
}

#[derive(Debug, Clone, PartialEq)]
struct State {
    blank_count: u64,
    nonblank_consec: u64,
    nonblank_total: u64,
    speech_confirmed: bool,
    agent_speaking: bool,
    first_turn: bool,
    user_first_turn: bool,
    rolling_density: f32,
    history: VecDeque<u32>,
    model_steps: u64,
    audio_frames: u64,
    idle: bool,
    content: u64,
    pads: u64,
}

impl Default for State {
    fn default() -> Self {
        Self {
            blank_count: 0,
            nonblank_consec: 0,
            nonblank_total: 0,
            speech_confirmed: false,
            agent_speaking: false,
            first_turn: true,
            user_first_turn: true,
            rolling_density: 0.0,
            history: VecDeque::new(),
            model_steps: 0,
            audio_frames: 0,
            idle: true,
            content: 0,
            pads: 0,
        }
    }
}

/// Canonical host turn state. Neural execution borrows a pure resolver; only
/// matching committed host output adopts the recomputed deterministic successor.
/// No callback mutates this state during sampling or before native completion.
pub struct VoiceChatTurnTaking {
    policy: VoiceChatTurnPolicy,
    limits: VoiceChatTurnLimits,
    controls: TextControls,
    state: State,
    identity: VoiceChatTurnIdentity,
    frontier: Progress,
    invocation: u64,
}

/// Captured invocation and function-channel state, retained with the unpublished
/// neural candidate. Later tool-result submission cannot change this decision's
/// attribution. Interruption invalidates the captured turn identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatTurnAdmission {
    identity: VoiceChatTurnIdentity,
    invocation: u64,
    tool_active: bool,
    step: Step,
}

/// Pure per-invocation policy borrowed by the architecture's decision seam.
pub struct VoiceChatTurnResolver<'a> {
    owner: &'a VoiceChatTurnTaking,
    admission: VoiceChatTurnAdmission,
    silence_pad: [bool; 1],
}

impl VoiceChatTurnTaking {
    pub(super) fn rekey_for_fork(&mut self, request: eredu_core::scheduler::RequestId) {
        self.identity = self.identity.with_request(request);
    }

    pub(super) fn estimate_snapshot(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        super::conversation::host_storage::<Self>(
            self.state.history.capacity(),
            std::mem::size_of::<u32>(),
        )
    }

    pub(super) fn copy_snapshot(&self) -> Result<Self, BackendFailure> {
        let mut history = VecDeque::new();
        history
            .try_reserve_exact(self.state.history.len())
            .map_err(BackendFailure::from_error)?;
        history.extend(self.state.history.iter().copied());
        Ok(Self {
            policy: self.policy,
            limits: self.limits,
            controls: self.controls,
            state: State {
                blank_count: self.state.blank_count,
                nonblank_consec: self.state.nonblank_consec,
                nonblank_total: self.state.nonblank_total,
                speech_confirmed: self.state.speech_confirmed,
                agent_speaking: self.state.agent_speaking,
                first_turn: self.state.first_turn,
                user_first_turn: self.state.user_first_turn,
                rolling_density: self.state.rolling_density,
                history,
                model_steps: self.state.model_steps,
                audio_frames: self.state.audio_frames,
                idle: self.state.idle,
                content: self.state.content,
                pads: self.state.pads,
            },
            identity: self.identity,
            frontier: self.frontier,
            invocation: self.invocation,
        })
    }

    /// Committed component clocks used by facade invocation selection.
    pub(super) fn frontier(&self) -> Progress {
        self.frontier
    }

    /// Stages the bounded acoustic history under the canonical owner's identity.
    /// This unpublished candidate is not a separately usable conversation owner.
    pub(super) fn branch_candidate(&self) -> Self {
        Self {
            policy: self.policy,
            limits: self.limits,
            controls: self.controls,
            state: self.state.clone(),
            identity: self.identity,
            frontier: self.frontier,
            invocation: self.invocation,
        }
    }

    /// Admits controls from the exact prepared tool channel without native work.
    pub fn new(
        channel: &VoiceChatToolChannel,
        policy: VoiceChatTurnPolicy,
        limits: VoiceChatTurnLimits,
    ) -> Result<Self, VoiceChatTurnError> {
        if policy.lookback == 0
            || policy.lookback > limits.max_history
            || limits.max_audio_frames == 0
            || limits.max_model_steps == 0
            || [
                policy.eou_frames,
                policy.first_eou_frames,
                policy.bou_frames,
                policy.min_speech_frames,
                policy.first_min_speech_frames,
                policy.low_density_min_speech,
                policy.noise_reset_frames,
            ]
            .contains(&0)
            || !policy.density_threshold.is_finite()
            || !(0.0..=1.0).contains(&policy.density_threshold)
            || !policy.pad_tail_ratio.is_finite()
            || policy.pad_tail_ratio < 0.0
        {
            return Err(VoiceChatTurnError::Configuration("thresholds or bounds"));
        }
        Ok(Self {
            policy,
            limits,
            controls: channel.text_preparation().controls(),
            state: State::default(),
            identity: channel.turn_identity(),
            frontier: Progress::default(),
            invocation: 0,
        })
    }

    /// Captures function-channel activity before executing the live candidate.
    pub fn resolver<'a>(
        &'a self,
        channel: &VoiceChatToolChannel,
    ) -> Result<VoiceChatTurnResolver<'a>, VoiceChatTurnError> {
        let admission = self.admission(channel, Step::LiveAudio)?;
        let tail = self.policy.pad_tail_ratio > 0.0
            && !self.state.idle
            && self.state.pads as f64 > self.policy.pad_tail_ratio * self.state.content as f64;
        Ok(VoiceChatTurnResolver {
            owner: self,
            admission,
            silence_pad: [self.state.idle || tail],
        })
    }

    /// Tool substeps gate PAD only while idle and silence the emitted waveform.
    /// Main-loop PAD-tail counters do not terminate tool-response speech.
    pub(super) fn tool_output_policy(&self, force_pad: bool) -> OutputPolicy<'_> {
        OutputPolicy {
            force_silence_on_eos: false,
            silence_on_pad: if force_pad {
                std::slice::from_ref(&self.state.idle)
            } else {
                &[false]
            },
            waveform: WaveformSilence::ApplySilence,
        }
    }

    /// Captures attribution for an ordinary, perception-only or tool invocation.
    /// Use the resolver's own admission for resolved live decisions.
    pub fn admission(
        &self,
        channel: &VoiceChatToolChannel,
        step: Step,
    ) -> Result<VoiceChatTurnAdmission, VoiceChatTurnError> {
        self.validate_identity(channel, channel.turn_identity())?;
        if matches!(
            step,
            Step::LiveAudio
                | Step::PerceptionOnly
                | Step::Tool {
                    audio: AudioSource::Fresh
                }
        ) && self.state.audio_frames >= self.limits.max_audio_frames
        {
            return Err(VoiceChatTurnError::Capacity("audio frames"));
        }
        if matches!(
            step,
            Step::LiveAudio | Step::ReplayAudio | Step::Tool { .. }
        ) && self.state.model_steps >= self.limits.max_model_steps
        {
            return Err(VoiceChatTurnError::Capacity("model steps"));
        }
        if self.invocation == u64::MAX {
            return Err(VoiceChatTurnError::Capacity("invocations"));
        }
        Ok(VoiceChatTurnAdmission {
            identity: self.identity,
            invocation: self.invocation,
            tool_active: channel.is_active(),
            step,
        })
    }

    /// Adopts turn state from the architecture host observer after shared
    /// completion and publication. Call this before
    /// collecting the same output's function token. A forced interruption advances
    /// the tool-channel turn identity; do not publish that old turn's tool tokens.
    pub fn observe_committed<D>(
        &mut self,
        channel: &mut VoiceChatToolChannel,
        admission: VoiceChatTurnAdmission,
        output: &OwnedHostOutput<D>,
    ) -> Result<Option<VoiceChatTurnEvent>, VoiceChatTurnError> {
        self.validate_identity(channel, admission.identity)?;
        if admission.invocation != self.invocation
            || admission.step != output.step
            || output.invocation_index != self.invocation
            || output.before != self.frontier
        {
            return Err(VoiceChatTurnError::Attribution);
        }
        let next_invocation = self
            .invocation
            .checked_add(1)
            .ok_or(VoiceChatTurnError::Capacity("invocations"))?;
        let evidence = match &output.transcription {
            Some(frame) => DecisionEvidence {
                first_blank: Some(frame.first_blank),
                audio_frame_index: Some(frame.frame_index),
            },
            None => DecisionEvidence::default(),
        };
        let mut next = self.state.clone();
        next.accept_audio(evidence, self.limits)?;
        let mut event = None;
        if let Some(decisions) = output.decisions {
            if !matches!(output.step, Step::LiveAudio | Step::Tool { .. }) {
                return Err(VoiceChatTurnError::Attribution);
            }
            let tool = !matches!(output.step, Step::LiveAudio)
                || admission.tool_active
                || decisions.raw_function == self.controls.tool_call_start;
            let (effective, change) = next.decide(
                decisions.raw_agent,
                tool,
                self.policy,
                self.controls,
                self.limits,
            )?;
            if decisions.feedback.agent() != effective
                || (matches!(output.step, Step::LiveAudio) && decisions.spoken_text != effective)
            {
                return Err(VoiceChatTurnError::Attribution);
            }
            event = change;
            if matches!(output.step, Step::LiveAudio) {
                next.accept_spoken(effective, self.controls)?;
            }
        } else if output.step == Step::ReplayAudio {
            let mut expected = output.before;
            expected.backbone_positions = expected
                .backbone_positions
                .checked_add(1)
                .ok_or(VoiceChatTurnError::Capacity("model positions"))?;
            if output.transcription.is_some()
                || output.speech.is_some()
                || output.generated_codes.is_some()
                || output.after != expected
            {
                return Err(VoiceChatTurnError::Attribution);
            }
            // Replay contributes a real model position and PAD feedback history,
            // but it carries no new speech evidence or conversational decision.
            next.decide(
                self.controls.pad,
                true,
                self.policy,
                self.controls,
                self.limits,
            )?;
        } else if matches!(output.step, Step::LiveAudio | Step::Tool { .. }) {
            return Err(VoiceChatTurnError::Attribution);
        }
        if next.audio_frames != output.after.audio_frames {
            return Err(VoiceChatTurnError::Attribution);
        }
        // Tool work suppresses ordinary agent decisions, but fresh RNNT evidence
        // still owns the interruption boundary. Silence/retained tool substeps
        // cannot reuse the same evidence to terminate another turn.
        if admission.tool_active
            && evidence.first_blank == Some(false)
            && next.interrupt_tool(self.policy, self.controls)?
        {
            event = Some(VoiceChatTurnEvent::Interrupted);
        }
        if event == Some(VoiceChatTurnEvent::Interrupted) {
            channel.interrupt()?;
            self.identity = channel.turn_identity();
        }
        self.state = next;
        self.frontier = output.after;
        self.invocation = next_invocation;
        Ok(event)
    }

    /// Explicit host interruption invalidates borrowed-work identities while
    /// preserving committed acoustic history and already delivered output.
    pub fn interrupt(
        &mut self,
        channel: &mut VoiceChatToolChannel,
    ) -> Result<(), VoiceChatTurnError> {
        self.validate_identity(channel, channel.turn_identity())?;
        channel.interrupt()?;
        self.identity = channel.turn_identity();
        Ok(())
    }

    fn validate_identity(
        &self,
        channel: &VoiceChatToolChannel,
        identity: VoiceChatTurnIdentity,
    ) -> Result<(), VoiceChatTurnError> {
        if identity != self.identity || channel.turn_identity() != self.identity {
            Err(VoiceChatTurnError::Stale)
        } else {
            Ok(())
        }
    }
}

impl VoiceChatTurnResolver<'_> {
    /// Retain this exact stamp with the candidate's eventual committed output.
    pub const fn admission(&self) -> VoiceChatTurnAdmission {
        self.admission
    }
    /// Main-loop output gating uses prior committed speech counters. It preserves
    /// the generated waveform while optionally substituting next-step feedback.
    /// The two force flags come from the admitted architecture's output policy.
    pub fn output_policy(&self, force_eos: bool, force_pad: bool) -> OutputPolicy<'_> {
        OutputPolicy {
            force_silence_on_eos: force_eos,
            silence_on_pad: if force_pad {
                &self.silence_pad
            } else {
                &[false]
            },
            waveform: WaveformSilence::PreserveGenerated,
        }
    }
}

impl DecisionResolver for VoiceChatTurnResolver<'_> {
    fn preflight(&self, domain: TokenDomain) -> Result<(), BackendFailure> {
        if domain.cardinality() != TEXT_VOCABULARY_SIZE {
            return Err(BackendFailure::from_error(
                VoiceChatTurnError::Configuration("released vocabulary"),
            ));
        }
        if self.owner.state.model_steps >= self.owner.limits.max_model_steps {
            return Err(BackendFailure::from_error(VoiceChatTurnError::Capacity(
                "model steps",
            )));
        }
        if self.owner.state.audio_frames >= self.owner.limits.max_audio_frames {
            return Err(BackendFailure::from_error(VoiceChatTurnError::Capacity(
                "audio frames",
            )));
        }
        Ok(())
    }

    fn resolve(&self, input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
        let run = || -> Result<_, VoiceChatTurnError> {
            if input.evidence.first_blank.is_none() {
                return Err(VoiceChatTurnError::Evidence);
            }
            let mut next = self.owner.state.clone();
            next.accept_audio(input.evidence, self.owner.limits)?;
            let tool = self.admission.tool_active
                || input.raw_function == self.owner.controls.tool_call_start;
            let (agent, _) = next.decide(
                input.raw_agent,
                tool,
                self.owner.policy,
                self.owner.controls,
                self.owner.limits,
            )?;
            Ok(ChannelDecisionPolicy::Main {
                agent_feedback: Some(agent),
                function_feedback: None,
                spoken_text: None,
            })
        };
        run().map_err(BackendFailure::from_error)
    }
}

impl State {
    fn interrupt_tool(
        &mut self,
        policy: VoiceChatTurnPolicy,
        controls: TextControls,
    ) -> Result<bool, VoiceChatTurnError> {
        if !policy.enabled || self.nonblank_consec < policy.bou_frames {
            return Ok(false);
        }
        self.nonblank_consec = 0;
        self.nonblank_total = 0;
        self.speech_confirmed = false;
        self.agent_speaking = false;
        // NVIDIA's quit-to-normal boundary substitutes EOS at the last model
        // position. Perception-only work does not invent a generated position.
        if let Some(previous) = self.history.back_mut() {
            *previous = controls.eos;
        }
        self.accept_spoken(controls.eos, controls)?;
        Ok(true)
    }

    fn accept_audio(
        &mut self,
        evidence: DecisionEvidence,
        limits: VoiceChatTurnLimits,
    ) -> Result<(), VoiceChatTurnError> {
        let (blank, frame) = match (evidence.first_blank, evidence.audio_frame_index) {
            (Some(blank), Some(frame)) => (blank, frame),
            (None, None) => return Ok(()),
            _ => return Err(VoiceChatTurnError::Evidence),
        };
        if frame != self.audio_frames {
            return Err(VoiceChatTurnError::Evidence);
        }
        if frame >= limits.max_audio_frames {
            return Err(VoiceChatTurnError::Capacity("audio frames"));
        }
        self.audio_frames += 1;
        self.blank_count = if blank { self.blank_count + 1 } else { 0 };
        self.nonblank_consec = if blank { 0 } else { self.nonblank_consec + 1 };
        if !blank {
            self.nonblank_total += 1;
        }
        if !self.agent_speaking {
            self.rolling_density =
                0.1f32 * if blank { 0.0 } else { 1.0 } + 0.9f32 * self.rolling_density;
        }
        Ok(())
    }

    fn decide(
        &mut self,
        raw: u32,
        tool: bool,
        policy: VoiceChatTurnPolicy,
        controls: TextControls,
        limits: VoiceChatTurnLimits,
    ) -> Result<(u32, Option<VoiceChatTurnEvent>), VoiceChatTurnError> {
        if raw as usize >= TEXT_VOCABULARY_SIZE {
            return Err(VoiceChatTurnError::Token(raw));
        }
        if self.model_steps >= limits.max_model_steps {
            return Err(VoiceChatTurnError::Capacity("model steps"));
        }
        let (token, event) = if tool {
            (controls.pad, None)
        } else if policy.enabled {
            self.turn(raw, policy, controls)
        } else {
            (raw, None)
        };
        self.model_steps += 1;
        if self.history.len() == policy.lookback {
            self.history.pop_front();
        }
        self.history.push_back(token);
        Ok((
            token,
            event.or({
                if token == controls.bos {
                    Some(VoiceChatTurnEvent::AgentStarted)
                } else if token == controls.eos {
                    Some(VoiceChatTurnEvent::AgentEnded)
                } else {
                    None
                }
            }),
        ))
    }

    fn turn(
        &mut self,
        raw: u32,
        policy: VoiceChatTurnPolicy,
        controls: TextControls,
    ) -> (u32, Option<VoiceChatTurnEvent>) {
        let mut minimum = if self.user_first_turn {
            policy.first_min_speech_frames
        } else {
            policy.min_speech_frames
        };
        let eou = if self.user_first_turn {
            policy.first_eou_frames
        } else {
            policy.eou_frames
        };
        if !self.first_turn
            && self.rolling_density > 0.0
            && self.rolling_density < policy.density_threshold
        {
            minimum = policy.low_density_min_speech;
        }
        let had_bos = self.history.contains(&controls.bos);
        let had_eos = self.history.contains(&controls.eos);
        if had_bos || raw == controls.bos {
            self.agent_speaking = true;
            self.first_turn = false;
            if raw == controls.bos {
                self.speech_confirmed = false;
                self.nonblank_total = 0;
            }
        }
        // The reference deliberately checks EOS after BOS; any EOS in this
        // window wins, even if BOS was more recent or is the current raw token.
        if had_eos || raw == controls.eos {
            self.agent_speaking = false;
            self.speech_confirmed = false;
            self.nonblank_total = 0;
        }
        if self.blank_count >= policy.noise_reset_frames
            && !self.speech_confirmed
            && !self.agent_speaking
        {
            self.nonblank_total = 0;
        }
        if !self.agent_speaking
            && (self.nonblank_consec >= minimum || self.nonblank_total >= minimum)
        {
            self.speech_confirmed = true;
        }
        let fallback = self.first_turn
            && self.model_steps >= policy.first_turn_fallback
            && self.speech_confirmed
            && !had_bos;
        let eou_ready = self.blank_count >= eou && self.speech_confirmed && !self.agent_speaking;
        if fallback || (eou_ready && !had_bos && raw != controls.bos) {
            self.speech_confirmed = false;
            self.nonblank_total = 0;
            self.agent_speaking = true;
            self.first_turn = false;
            self.user_first_turn = false;
            return (
                controls.bos,
                Some(if fallback {
                    VoiceChatTurnEvent::FirstTurnFallback
                } else {
                    VoiceChatTurnEvent::UserEnded
                }),
            );
        }
        if eou_ready {
            return (raw, None);
        }
        if self.nonblank_consec >= policy.bou_frames
            && self.agent_speaking
            && !had_eos
            && raw != controls.eos
        {
            self.nonblank_consec = 0;
            self.nonblank_total = 0;
            self.agent_speaking = false;
            return (controls.eos, Some(VoiceChatTurnEvent::Interrupted));
        }
        (raw, None)
    }

    fn accept_spoken(
        &mut self,
        token: u32,
        controls: TextControls,
    ) -> Result<(), VoiceChatTurnError> {
        if token == controls.bos || token == controls.eos {
            self.idle = token == controls.eos;
            self.content = 0;
            self.pads = 0;
        } else if token == controls.pad {
            self.pads = self
                .pads
                .checked_add(1)
                .ok_or(VoiceChatTurnError::Capacity("speech tail"))?;
        } else {
            self.content = self
                .content
                .checked_add(1)
                .ok_or(VoiceChatTurnError::Capacity("speech text"))?;
            self.pads = 0;
        }
        Ok(())
    }
}
