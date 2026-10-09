//! Transactional host projection at the ordinary realtime publication boundary.

use std::{cell::RefCell, rc::Rc, time::Duration};

use eredu_architectures::voicechat::transcription::{RnntTextDecoder, RnntTextError};
use eredu_architectures::voicechat::{
    observation::OwnedHostOutput,
    schedule::Step,
    tts_feedback::{OutputPolicy, WaveformSilence},
};
use eredu_core::generation::SemanticEvent;
use eredu_runtime::execution_control::SnapshotRetainedGrowth;

use super::phrase_selection::{
    PreparedPhraseSelection, VoiceChatPhrasePolicy, ACK_PHRASES, INITIAL_REMINDER_PHRASES,
};

use super::decoded_text::{
    VoiceChatAgentDecoder, VoiceChatAgentText, VoiceChatDecodeError, VoiceChatDecodeLimits,
};

mod planning;
mod snapshot;

pub(super) use snapshot::host_storage;
pub(super) use snapshot::ConversationSnapshot;

use super::{
    VoiceChatReminderPlayback, VoiceChatReminderStep, VoiceChatTextError, VoiceChatToolBatch,
    VoiceChatToolChannel, VoiceChatToolError, VoiceChatToolTicket, VoiceChatToolWait,
    VoiceChatToolWaitAction, VoiceChatToolWaitError, VoiceChatToolWaitMessages,
    VoiceChatToolWaitPolicy, VoiceChatTurnAdmission, VoiceChatTurnError, VoiceChatTurnEvent,
    VoiceChatTurnIdentity, VoiceChatTurnLimits, VoiceChatTurnPolicy, VoiceChatTurnResolver,
    VoiceChatTurnTaking,
};

/// Available publication space for one invocation. Zero admits no such payload.
///
/// The enclosing driver reserves these allowances from its output queue before
/// submission and retains that reservation until delivery or discard. Native
/// observation budgets separately account for diagnostics' internal storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoiceChatPublicationCapacity {
    /// Maximum waveform samples, including codec overlap on drain.
    pub speech_samples: usize,
    /// Maximum newly emitted RNNT token IDs.
    pub transcription_tokens: usize,
    /// Maximum decoded agent-text UTF-8 bytes.
    pub agent_text_bytes: usize,
    /// Maximum decoded RNNT transcription UTF-8 bytes.
    pub transcription_text_bytes: usize,
    /// Maximum generated residual-code IDs.
    pub audio_codes: usize,
    /// Maximum complete semantic tool events.
    pub tool_events: usize,
    /// Maximum combined UTF-8 bytes in tool-event strings.
    pub tool_bytes: usize,
    /// Maximum delegated diagnostic records.
    pub diagnostics: usize,
}

/// Host validation failure before the shared scheduler publishes neural state.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatConversationError {
    /// A fork's shared retained-state budget cannot admit the host successor.
    #[error("VoiceChat conversation retention admission failed: {0}")]
    Snapshot(#[source] eredu_core::BackendFailure),
    /// An outstanding neural invocation retains exclusive host publication authority.
    #[error("VoiceChat conversation has an outstanding transition")]
    Busy,
    /// No committed tool wait is active.
    #[error("VoiceChat conversation has no active tool wait")]
    NoWait,
    /// Recorded advancement time precedes an already committed advancement.
    #[error("VoiceChat advancement time moved backwards")]
    Time,
    /// Prompt and speaker conditioning already committed for this conversation.
    #[error("VoiceChat conversation is already prefilled")]
    AlreadyPrefilled,
    /// An output exceeds its reserved publication allowance.
    #[error("VoiceChat publication {0} capacity exhausted")]
    Capacity(&'static str),
    /// Reminder work requires its matching wait owner, receipt and recorded time.
    #[error("VoiceChat reminder publication requires matching wait admission")]
    Reminder,
    /// Acknowledgement speech differs from the admitted phrase token.
    #[error("VoiceChat acknowledgement publication differs from its admitted token")]
    Acknowledgement,
    /// Spoken phrase preparation was rejected.
    #[error(transparent)]
    Text(#[from] VoiceChatTextError),
    /// Turn evidence or invocation attribution was rejected.
    #[error(transparent)]
    Turn(#[from] VoiceChatTurnError),
    /// Function framing, schema validation or response feedback was rejected.
    #[error(transparent)]
    Tool(#[from] VoiceChatToolError),
    /// Reminder completion was rejected.
    #[error(transparent)]
    Wait(#[from] VoiceChatToolWaitError),
    /// Incremental agent decoding rejected the candidate.
    #[error(transparent)]
    Decode(#[from] VoiceChatDecodeError),
    /// RNNT vocabulary decoding rejected the candidate.
    #[error(transparent)]
    Transcription(#[from] RnntTextError),
}

struct ConversationState {
    turn: VoiceChatTurnTaking,
    channel: VoiceChatToolChannel,
    wait: Option<VoiceChatToolWait>,
    acknowledgement: PreparedPhraseSelection,
    initial_reminder: PreparedPhraseSelection,
    agent_text: VoiceChatAgentDecoder,
    transcription: RnntTextDecoder,
    prefilled: bool,
    pending_audio_frames: u64,
    replay_frames: u64,
    last_step: Option<Step>,
    last_reminder_at: Option<Duration>,
    last_plan_at: Option<Duration>,
}

impl ConversationState {
    fn acknowledgement_token(&self) -> Option<u32> {
        if !self.channel.is_active() || self.channel.pending().is_some() {
            return None;
        }
        self.acknowledgement.token()
    }

    fn branch_candidate(&self) -> Self {
        Self {
            turn: self.turn.branch_candidate(),
            channel: self.channel.branch_candidate(),
            wait: self.wait.as_ref().map(VoiceChatToolWait::branch_candidate),
            acknowledgement: self.acknowledgement.clone(),
            initial_reminder: self.initial_reminder.clone(),
            agent_text: self.agent_text.clone(),
            transcription: self.transcription.clone(),
            prefilled: self.prefilled,
            pending_audio_frames: self.pending_audio_frames,
            replay_frames: self.replay_frames,
            last_step: self.last_step,
            last_reminder_at: self.last_reminder_at,
            last_plan_at: self.last_plan_at,
        }
    }
}

struct ConversationOwner {
    state: ConversationState,
    busy: bool,
    retained_growth: Option<SnapshotRetainedGrowth>,
}

/// Canonical conversation policy used by one persistent shared realtime scheduler.
///
/// One owned admission freezes mutation until its output commits or is discarded.
/// External results receive typed backpressure during that interval; applications
/// retain and retry them. This owner contains no neural state, clock, scheduler or
/// external tool executor. It is not a public session snapshot or fork handle.
pub struct VoiceChatConversation {
    owner: Rc<RefCell<ConversationOwner>>,
}

impl VoiceChatConversation {
    // The semantic transaction borrows canonical policy through an owned handle;
    // exclusive admission still prevents independently advancing aliases.
    pub(super) fn shared_owner(&self) -> Self {
        Self {
            owner: Rc::clone(&self.owner),
        }
    }

    /// Combines the existing tool channel and RNNT turn policy at their initial frontier.
    pub fn new(
        channel: VoiceChatToolChannel,
        policy: VoiceChatTurnPolicy,
        limits: VoiceChatTurnLimits,
        transcription: RnntTextDecoder,
        decode_limits: VoiceChatDecodeLimits,
    ) -> Result<Self, VoiceChatConversationError> {
        let turn = VoiceChatTurnTaking::new(&channel, policy, limits)?;
        let agent_text = VoiceChatAgentDecoder::new(channel.text_preparation(), decode_limits)?;
        Ok(Self {
            owner: Rc::new(RefCell::new(ConversationOwner {
                state: ConversationState {
                    turn,
                    channel,
                    wait: None,
                    acknowledgement: PreparedPhraseSelection::default(),
                    initial_reminder: PreparedPhraseSelection::default(),
                    agent_text,
                    transcription,
                    prefilled: false,
                    pending_audio_frames: 0,
                    replay_frames: 0,
                    last_step: None,
                    last_reminder_at: None,
                    last_plan_at: None,
                },
                busy: false,
                retained_growth: None,
            })),
        })
    }

    /// Prepares all acknowledgement phrases before prompt prefill.
    /// Per-call selection belongs to the unpublished conversation transaction.
    pub(super) fn prepare_acknowledgement(
        &mut self,
        policy: &VoiceChatPhrasePolicy,
        max_tokens: usize,
    ) -> Result<(), VoiceChatConversationError> {
        self.mutate(|state| {
            if state.prefilled {
                return Err(VoiceChatConversationError::AlreadyPrefilled);
            }
            state.acknowledgement = PreparedPhraseSelection::prepare(
                policy,
                state.channel.text_preparation(),
                max_tokens,
                &ACK_PHRASES,
            )?;
            Ok(())
        })
    }

    /// Prepares the initial tool-wait speech pool before prompt prefill.
    /// Selection shares the committed function-call boundary with acknowledgments.
    pub(super) fn prepare_initial_reminder(
        &mut self,
        policy: &VoiceChatPhrasePolicy,
        max_tokens: usize,
    ) -> Result<(), VoiceChatConversationError> {
        self.mutate(|state| {
            if state.prefilled {
                return Err(VoiceChatConversationError::AlreadyPrefilled);
            }
            state.initial_reminder = PreparedPhraseSelection::prepare(
                policy,
                state.channel.text_preparation(),
                max_tokens,
                &INITIAL_REMINDER_PHRASES,
            )?;
            Ok(())
        })
    }

    /// Identity of the canonical, committed conversation turn.
    pub fn turn_identity(&self) -> VoiceChatTurnIdentity {
        self.owner.borrow().state.channel.turn_identity()
    }

    /// Canonical external batch, including an injected response awaiting EOTR.
    pub fn pending(&self) -> Option<VoiceChatToolTicket> {
        self.owner.borrow().state.channel.pending()
    }

    /// Whether committed tool serialization, external waiting or response
    /// decoding occupies the function channel, including an incomplete call.
    /// An unpublished candidate does not change this observation.
    pub fn has_tool_work(&self) -> bool {
        self.owner.borrow().state.channel.is_active()
    }

    /// Current committed response token override, without consuming it.
    pub fn response_policy(
        &self,
    ) -> Option<eredu_architectures::voicechat::channel_decision::ChannelDecisionPolicy> {
        self.owner.borrow().state.channel.response_policy()
    }

    /// Injects an external result at an explicit boundary between invocations.
    pub fn submit_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        text: &str,
    ) -> Result<(), VoiceChatConversationError> {
        self.mutate(|state| Ok(state.channel.submit_response(ticket, text)?))
    }

    /// Invalidates conversational work after the outstanding invocation is retired.
    /// Already committed audio and acoustic/model positions are preserved.
    pub fn interrupt(&mut self) -> Result<(), VoiceChatConversationError> {
        self.mutate(|state| {
            state.turn.interrupt(&mut state.channel)?;
            state.wait = None;
            state.acknowledgement.clear();
            state.initial_reminder.clear();
            state.replay_frames = state.pending_audio_frames;
            state.agent_text.interrupt();
            Ok(())
        })
    }

    /// Prepares reminders and timeout text for a complete, committed tool batch.
    pub fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<(), VoiceChatConversationError> {
        self.mutate(|state| {
            if state.wait.is_some() {
                return Err(VoiceChatToolWaitError::NotWaiting.into());
            }
            state.wait = Some(VoiceChatToolWait::with_prepared_initial(
                &state.channel,
                ticket,
                at,
                policy,
                messages,
                state.initial_reminder.selected_phrase(),
            )?);
            Ok(())
        })
    }

    /// Selects ordinary work using recorded time; no speech token is consumed.
    pub fn advance_wait(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatToolWaitAction, VoiceChatConversationError> {
        self.mutate(|state| {
            let action = state
                .wait
                .as_mut()
                .ok_or(VoiceChatConversationError::NoWait)?
                .advance(&mut state.channel, at)?;
            if action == VoiceChatToolWaitAction::Complete {
                state.wait = None;
            }
            Ok(action)
        })
    }

    /// Records actual presentation completion, separately from native completion.
    pub fn finish_playback(
        &mut self,
        receipt: VoiceChatReminderPlayback,
        at: Duration,
    ) -> Result<(), VoiceChatConversationError> {
        self.mutate(|state| {
            Ok(state
                .wait
                .as_mut()
                .ok_or(VoiceChatConversationError::NoWait)?
                .finish_playback(&state.channel, receipt, at)?)
        })
    }

    fn mutate<O>(
        &mut self,
        apply: impl FnOnce(&mut ConversationState) -> Result<O, VoiceChatConversationError>,
    ) -> Result<O, VoiceChatConversationError> {
        let mut owner = self.owner.borrow_mut();
        if owner.busy {
            return Err(VoiceChatConversationError::Busy);
        }
        let mut candidate = owner.state.branch_candidate();
        let result = apply(&mut candidate)?;
        owner
            .admit_snapshot_retention(&candidate)
            .map_err(VoiceChatConversationError::Snapshot)?;
        owner.state = candidate;
        Ok(result)
    }

    /// Admits prefill, live, tool, perception-only or drain work with reserved output space.
    pub(super) fn admit(
        &mut self,
        step: Step,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<VoiceChatConversationTransition, VoiceChatConversationError> {
        if step == Step::Reminder {
            return Err(VoiceChatConversationError::Reminder);
        }
        self.admit_inner(step, None, capacity)
    }

    /// Admits a selected reminder receipt and explicit completion timestamp.
    /// Use that same receipt's token and output policy in the neural invocation.
    pub(super) fn admit_reminder(
        &mut self,
        receipt: VoiceChatReminderStep,
        at: Duration,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<VoiceChatConversationTransition, VoiceChatConversationError> {
        self.admit_inner(Step::Reminder, Some((receipt, at)), capacity)
    }

    fn admit_inner(
        &mut self,
        step: Step,
        reminder: Option<(VoiceChatReminderStep, Duration)>,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<VoiceChatConversationTransition, VoiceChatConversationError> {
        let mut owner = self.owner.borrow_mut();
        if owner.busy {
            return Err(VoiceChatConversationError::Busy);
        }
        let admission = owner.state.turn.admission(&owner.state.channel, step)?;
        if let Some((receipt, at)) = reminder {
            owner
                .state
                .wait
                .as_ref()
                .ok_or(VoiceChatConversationError::NoWait)?
                .branch_candidate()
                .commit_reminder(&owner.state.channel, receipt, at)?;
        }
        let candidate = owner.state.branch_candidate();
        owner.busy = true;
        Ok(VoiceChatConversationTransition {
            owner: Rc::clone(&self.owner),
            candidate: Some(candidate),
            admission,
            reminder,
            capacity,
        })
    }
}

/// Exclusive, owned host admission paired with one unpublished neural invocation.
/// Its lifetime does not borrow the facade owner or constrain the scheduler's
/// output type. Drop releases admission without adopting any tentative state.
pub(super) struct VoiceChatConversationTransition {
    owner: Rc<RefCell<ConversationOwner>>,
    candidate: Option<ConversationState>,
    admission: VoiceChatTurnAdmission,
    reminder: Option<(VoiceChatReminderStep, Duration)>,
    capacity: VoiceChatPublicationCapacity,
}

impl Drop for VoiceChatConversationTransition {
    fn drop(&mut self) {
        self.owner.borrow_mut().busy = false;
    }
}

impl VoiceChatConversationTransition {
    pub(super) fn tool_output_policy(&self, force_pad: bool) -> OutputPolicy<'_> {
        let candidate = self
            .candidate
            .as_ref()
            .expect("live conversation candidate");
        if candidate.acknowledgement_token().is_some() {
            // BOS, content, trailing PADs and EOS all belong to the phrase.
            OutputPolicy {
                force_silence_on_eos: false,
                silence_on_pad: &[false],
                waveform: WaveformSilence::PreserveGenerated,
            }
        } else {
            candidate.turn.tool_output_policy(force_pad)
        }
    }

    /// Pure live-channel resolution from the exclusively retained candidate state.
    pub fn resolver(&self) -> Result<VoiceChatTurnResolver<'_>, VoiceChatTurnError> {
        let candidate = self
            .candidate
            .as_ref()
            .expect("live conversation candidate");
        candidate.turn.resolver(&candidate.channel)
    }

    pub(super) fn stage<D>(
        mut self,
        output: OwnedHostOutput<D>,
    ) -> Result<StagedVoiceChatConversation<D>, VoiceChatConversationError> {
        if output.reminder_token != self.reminder.map(|(receipt, _)| receipt.spoken()) {
            return Err(VoiceChatConversationError::Reminder);
        }
        self.capacity.validate_frame(&output)?;
        let candidate = self
            .candidate
            .as_mut()
            .expect("live conversation candidate");
        let acknowledgement = matches!(output.step, Step::Tool { .. })
            .then(|| candidate.acknowledgement_token())
            .flatten();
        if acknowledgement.is_some()
            && output
                .decisions
                .as_ref()
                .map(|decision| decision.spoken_text)
                != acknowledgement
        {
            return Err(VoiceChatConversationError::Acknowledgement);
        }
        let turn_event =
            candidate
                .turn
                .observe_committed(&mut candidate.channel, self.admission, &output)?;
        match output.step {
            Step::PerceptionOnly => {
                candidate.pending_audio_frames =
                    candidate.pending_audio_frames.checked_add(1).ok_or(
                        VoiceChatConversationError::Capacity("retained audio frames"),
                    )?;
            }
            Step::LiveAudio => {
                candidate.pending_audio_frames = 0;
                candidate.replay_frames = 0;
            }
            Step::ReplayAudio => {
                candidate.replay_frames = candidate
                    .replay_frames
                    .checked_sub(1)
                    .ok_or(VoiceChatTurnError::Attribution)?;
                candidate.pending_audio_frames = candidate
                    .pending_audio_frames
                    .checked_sub(1)
                    .ok_or(VoiceChatTurnError::Attribution)?;
            }
            _ => {}
        }
        let was_tool_active = candidate.channel.is_active();
        // Interruption invalidates every function token from the old admitted turn.
        let tools = if turn_event == Some(VoiceChatTurnEvent::Interrupted) {
            candidate.wait = None;
            candidate.replay_frames = candidate.pending_audio_frames;
            candidate.agent_text.interrupt();
            None
        } else {
            match output.decisions {
                Some(decisions) => candidate.channel.observe_committed_function(
                    candidate.channel.turn_identity(),
                    decisions.feedback.function(),
                )?,
                None => None,
            }
        };
        if turn_event == Some(VoiceChatTurnEvent::Interrupted)
            || !candidate.channel.is_active()
            || candidate.channel.pending().is_some()
        {
            candidate.acknowledgement.clear();
        } else if !was_tool_active {
            candidate.acknowledgement.begin_call();
        } else if acknowledgement.is_some() {
            candidate.acknowledgement.advance();
        }
        if turn_event == Some(VoiceChatTurnEvent::Interrupted) || !candidate.channel.is_active() {
            candidate.initial_reminder.clear();
        } else if !was_tool_active {
            candidate.initial_reminder.begin_call();
        }
        self.capacity.validate_tools(tools.as_ref())?;
        let agent_text = candidate.agent_text.observe(&output)?;
        let transcription = output
            .transcription
            .as_ref()
            .map(|frame| {
                candidate
                    .transcription
                    .append(&frame.emitted, self.capacity.transcription_text_bytes)
            })
            .transpose()?
            .filter(|text| !text.is_empty())
            .map(|text| VoiceChatTranscriptionText { text });
        if output.step == Step::Drain {
            candidate.transcription.finish();
        }
        self.capacity
            .validate_text(agent_text.as_ref(), transcription.as_ref())?;
        let playback = match self.reminder {
            Some(_) if turn_event == Some(VoiceChatTurnEvent::Interrupted) => None,
            Some((receipt, at)) => candidate
                .wait
                .as_mut()
                .ok_or(VoiceChatConversationError::NoWait)?
                .commit_reminder(&candidate.channel, receipt, at)?,
            None => None,
        };
        candidate.last_step = Some(output.step);
        if matches!(output.step, Step::Prefill { .. }) {
            candidate.prefilled = true;
        }
        if let Some((_, at)) = self.reminder {
            candidate.last_reminder_at = Some(at);
        }
        // The ordinary publication driver stages this successor before neural
        // commitment. Retention failure therefore leaves both owners unchanged.
        self.owner
            .borrow()
            .admit_snapshot_retention(candidate)
            .map_err(VoiceChatConversationError::Snapshot)?;
        Ok(StagedVoiceChatConversation {
            transition: self,
            publication: VoiceChatConversationOutput {
                frame: output,
                turn: turn_event,
                tools,
                playback,
                agent_text,
                transcription,
            },
        })
    }
}

/// Validated successor retained by the scheduler's unpublished semantic branch.
/// Dropping it leaves canonical host state unchanged and releases admission.
#[must_use = "adopt only after the matching neural transition commits"]
pub(super) struct StagedVoiceChatConversation<D> {
    transition: VoiceChatConversationTransition,
    publication: VoiceChatConversationOutput<D>,
}

impl<D> StagedVoiceChatConversation<D> {
    pub(super) fn publication(&self) -> &VoiceChatConversationOutput<D> {
        &self.publication
    }

    /// Adopts the validated successor after successful neural publication.
    /// No tokenization, parsing, capacity check or fallible policy runs here.
    pub(super) fn commit(mut self) -> VoiceChatConversationOutput<D> {
        let candidate = self
            .transition
            .candidate
            .take()
            .expect("live conversation candidate");
        self.transition.owner.borrow_mut().state = candidate;
        self.publication
    }
}

/// One bounded publication with its timestamped architecture output and host events.
#[derive(Debug)]
pub struct VoiceChatConversationOutput<D> {
    /// Already validated speech, effective decisions and RNNT tokens.
    pub frame: OwnedHostOutput<D>,
    /// Turn change caused by this invocation.
    pub turn: Option<VoiceChatTurnEvent>,
    /// Complete schema-validated calls; external applications execute them.
    pub tools: Option<VoiceChatToolBatch>,
    /// A reminder awaiting explicit presentation completion.
    pub playback: Option<VoiceChatReminderPlayback>,
    /// Effective agent-channel text; `frame` supplies invocation and model/audio time.
    /// Function payloads and independently spoken reminders are excluded.
    pub agent_text: Option<VoiceChatAgentText>,
    /// Newly decoded RNNT text; `frame.transcription` supplies source sample time.
    /// These are token-emission timestamps, not word-level acoustic alignments.
    pub transcription: Option<VoiceChatTranscriptionText>,
}

/// Append-only transcription delta decoded with the released RNNT vocabulary policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceChatTranscriptionText {
    /// UTF-8 text; pending trailing whitespace is emitted with subsequent content.
    pub text: String,
}

impl VoiceChatPublicationCapacity {
    pub(super) fn validate_text(
        &self,
        agent: Option<&VoiceChatAgentText>,
        transcription: Option<&VoiceChatTranscriptionText>,
    ) -> Result<(), VoiceChatConversationError> {
        for (bytes, limit, name) in [
            (
                agent.map_or(0, |event| event.text.len()),
                self.agent_text_bytes,
                "agent text bytes",
            ),
            (
                transcription.map_or(0, |event| event.text.len()),
                self.transcription_text_bytes,
                "transcription text bytes",
            ),
        ] {
            if bytes > limit {
                return Err(VoiceChatConversationError::Capacity(name));
            }
        }
        Ok(())
    }
    pub(super) fn validate_frame<D>(
        &self,
        output: &OwnedHostOutput<D>,
    ) -> Result<(), VoiceChatConversationError> {
        for (size, capacity, name) in [
            (
                output.speech.as_ref().map_or(0, |v| v.samples.len()),
                self.speech_samples,
                "speech samples",
            ),
            (
                output.transcription.as_ref().map_or(0, |v| v.emitted.len()),
                self.transcription_tokens,
                "transcription tokens",
            ),
            (
                output.generated_codes.as_ref().map_or(0, Vec::len),
                self.audio_codes,
                "audio codes",
            ),
            (output.diagnostics.len(), self.diagnostics, "diagnostics"),
        ] {
            if size > capacity {
                return Err(VoiceChatConversationError::Capacity(name));
            }
        }
        Ok(())
    }

    pub(super) fn validate_tools(
        &self,
        batch: Option<&VoiceChatToolBatch>,
    ) -> Result<usize, VoiceChatConversationError> {
        let Some(batch) = batch else {
            return Ok(0);
        };
        if batch.events.len() > self.tool_events {
            return Err(VoiceChatConversationError::Capacity("tool events"));
        }
        let mut remaining = self.tool_bytes;
        for event in &batch.events {
            let parts: &[&str] = match event {
                SemanticEvent::ToolCallStart { id, name, .. } => &[id, name],
                SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => &[json_fragment],
                SemanticEvent::ReasoningDelta(text) | SemanticEvent::TextDelta(text) => &[text],
                SemanticEvent::ToolCallEnd | SemanticEvent::Finished { .. } => &[],
            };
            for part in parts {
                remaining = remaining
                    .checked_sub(part.len())
                    .ok_or(VoiceChatConversationError::Capacity("tool bytes"))?;
            }
        }
        Ok(self.tool_bytes - remaining)
    }
}

#[cfg(test)]
pub(super) mod tests;
