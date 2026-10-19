//! Recorded-time tool waiting over the existing committed function channel.
//!
//! Policy follows NVIDIA-NeMo/Speech's streaming pipeline at
//! `097dfe9e2f55baf653b83035868bdc89849f1b47`: finish the initial reminder,
//! at most two generic reminders with gaps, then the final silent timeout.
//! Neural work uses ordinary Reminder/PerceptionOnly/Tool invocations. This
//! policy never polls a clock, submits work, executes a tool or waits on a thread.

use std::{sync::Arc, time::Duration};

use eredu_architectures::voicechat::{
    channel_decision::ChannelDecisionPolicy,
    tts_feedback::{OutputPolicy, WaveformSilence},
};

use super::{
    PreparedVoiceChatPhrase, PreparedVoiceChatToolResponse, VoiceChatTextError,
    VoiceChatToolChannel, VoiceChatToolError, VoiceChatToolTicket, VoiceChatTurnIdentity,
};

const TIMEOUT_RESPONSE: &str = concat!(
    "<TOOL_RESPONSE>[{\"error\": \"timeout\", ",
    "\"message\": \"I'm sorry, that's taking longer than expected.\"}]</TOOL_RESPONSE>"
);

/// Timing overrides for tool waiting. Times are supplied by the enclosing driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatToolWaitPolicy {
    /// Silence before each optional generic reminder.
    pub generic_gap: Duration,
    /// Final silent wait after all selected reminder speech finishes playback.
    pub timeout: Duration,
}

impl Default for VoiceChatToolWaitPolicy {
    fn default() -> Self {
        Self {
            generic_gap: Duration::from_secs(1),
            timeout: Duration::from_secs(15),
        }
    }
}

/// Initial speech for a tool wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatInitialReminder<'a> {
    /// Use the startup policy's phrase, selected when this call started.
    Configured,
    /// Suppress initial speech for this wait.
    Disabled,
    /// Override the configured phrase with this ASCII text.
    Text(&'a str),
}

/// Explicit phrase choices with an optional retained conversation selection.
/// No host randomness is drawn when beginning a wait. The released configuration
/// has no generic on-hold pool by default.
#[derive(Debug, Clone, Copy)]
pub struct VoiceChatToolWaitMessages<'a> {
    /// Per-tool override, configured fallback, or explicitly disabled initial speech.
    pub initial: VoiceChatInitialReminder<'a>,
    /// Up to two distinct selected generic phrases, in playback order.
    pub generic: &'a [&'a str],
    /// Total tokens per framed phrase, including BOS, tail PAD and EOS.
    pub max_phrase_tokens: usize,
}

/// An admitted reminder token; retain it with the unpublished neural invocation.
/// A duplicate, reordered or interrupted receipt cannot advance host speech state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatReminderStep {
    ticket: VoiceChatToolTicket,
    admission: u64,
    phrase: usize,
    token: usize,
    spoken: u32,
    admitted_at: Duration,
}

impl VoiceChatReminderStep {
    pub(super) fn with_request(self, request: eredu_core::scheduler::RequestId) -> Self {
        Self {
            ticket: self.ticket.with_request(request),
            ..self
        }
    }

    pub(super) fn encode_descriptor(&self, output: &mut Vec<u32>) {
        for value in [
            self.ticket.request().value(),
            self.ticket.turn(),
            self.ticket.batch(),
            self.admission,
            self.phrase as u64,
            self.token as u64,
            self.admitted_at.as_secs(),
        ] {
            super::driver::encode_u64(output, value);
        }
        output.extend([self.spoken, self.admitted_at.subsec_nanos()]);
    }
    /// Input to the existing architecture `Invocation::Reminder`.
    pub const fn spoken(&self) -> u32 {
        self.spoken
    }

    /// Preserve generated reminder feedback/waveform, including PAD tail and EOS.
    /// Architecture-owned EOS input substitution remains a separate operation.
    pub fn output_policy(&self) -> OutputPolicy<'static> {
        OutputPolicy {
            force_silence_on_eos: false,
            silence_on_pad: &[false],
            waveform: WaveformSilence::PreserveGenerated,
        }
    }
}

/// A completely generated phrase awaiting recorded audio playback completion.
/// Native completion alone does not establish this presentation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatReminderPlayback {
    ticket: VoiceChatToolTicket,
    phrase: usize,
}

impl VoiceChatReminderPlayback {
    pub(super) fn with_request(self, request: eredu_core::scheduler::RequestId) -> Self {
        Self {
            ticket: self.ticket.with_request(request),
            ..self
        }
    }
}

/// Next host-policy action; admission never consumes speech or response tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatToolWaitAction {
    /// Speech uses the shared Reminder invocation, followed by explicit commit.
    Reminder(VoiceChatReminderStep),
    /// No speech/backbone work is due. Fresh PCM may advance perception and RNNT.
    PerceptionOnly,
    /// Use the existing Tool invocation and observe effective function feedback
    /// through `VoiceChatToolChannel` only after its transition commits.
    Response(ChannelDecisionPolicy),
    /// The model has committed EOTR for this batch.
    Complete,
}

/// Portable wait-policy rejection; no error consumes a reminder token.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatToolWaitError {
    /// Phrase selection is bounded to two distinct generic messages.
    #[error("VoiceChat tool wait requires at most two distinct generic phrases")]
    Messages,
    /// This wait must start on an externally waiting, committed call batch.
    #[error("VoiceChat tool wait does not match a pending external batch")]
    NotWaiting,
    /// Interruption or another request/batch invalidated this wait.
    #[error("stale VoiceChat tool wait")]
    Stale,
    /// Explicit timestamps must be monotonic, including completion observations.
    #[error("VoiceChat tool wait time moved backwards")]
    Time,
    /// A receipt is duplicated, reordered or otherwise differs from the candidate.
    #[error("VoiceChat reminder receipt does not match pending speech")]
    Receipt,
    /// Receipts cannot be allocated without reusing an earlier identity.
    #[error("VoiceChat reminder admission identity exhausted")]
    IdentityExhausted,
    /// Phrase or timeout text cannot be prepared with the retained text policy.
    #[error(transparent)]
    Text(#[from] VoiceChatTextError),
    /// Existing function-channel policy rejection.
    #[error(transparent)]
    Tool(#[from] VoiceChatToolError),
}

/// Tool-wait host state with explicit time and commit-driven reminder progress.
///
/// Construct once after EOTC commits and the complete batch is published. Submit
/// external results through the associated `VoiceChatToolChannel`. Initial and
/// already-started reminders finish even when a result arrives; unstarted generic
/// reminders are skipped. Generated phrases await explicit playback completion
/// before starting the next gap or timeout. A timeout submits a prevalidated
/// response through that same channel, so later external results are stale. Event ordering at equal
/// timestamps is explicit: a result submitted before `advance` wins over timeout.
///
/// The enclosing shared driver may accept PCM and perform perception-only work
/// throughout the wait. It owns native completion, publication and interruption;
/// call `commit_reminder` only after publishing the matching speech transition.
/// An admitted reminder stays pending until commit or explicit discard after
/// native retirement. Repeated advancement returns the same pending receipt.
/// No public session loading or neural snapshot support is implied by this owner.
pub struct VoiceChatToolWait {
    ticket: VoiceChatToolTicket,
    turn: VoiceChatTurnIdentity,
    policy: VoiceChatToolWaitPolicy,
    phrases: Arc<[PreparedVoiceChatPhrase]>,
    has_initial: bool,
    phrase: usize,
    token: usize,
    pending: Option<VoiceChatReminderStep>,
    next_admission: u64,
    playback: Option<VoiceChatReminderPlayback>,
    last_time: Duration,
    last_phrase_end: Duration,
    timeout_response: Arc<PreparedVoiceChatToolResponse>,
    timed_out: bool,
}

impl VoiceChatToolWait {
    pub(super) fn playback_receipt(&self) -> Option<VoiceChatReminderPlayback> {
        self.playback
    }

    pub(super) fn rekey_for_fork(&mut self, request: eredu_core::scheduler::RequestId) {
        self.ticket = self.ticket.with_request(request);
        self.turn = self.turn.with_request(request);
        self.pending = self.pending.map(|step| step.with_request(request));
        self.playback = self.playback.map(|receipt| receipt.with_request(request));
    }

    pub(super) fn estimate_snapshot(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        use eredu_runtime::execution_control::snapshot_value_estimate;
        let mut phrases = snapshot_value_estimate::<usize>(
            self.phrases
                .iter()
                .map(PreparedVoiceChatPhrase::snapshot_retention),
        )?;
        // Shared phrases may outlive a replaced wait owner. Price their complete
        // retained payload and Arc header, but no nonexistent payload copy.
        phrases.retained_bytes = phrases
            .retained_bytes
            .checked_add((2 * std::mem::size_of::<usize>()) as u64)
            .ok_or_else(|| {
                eredu_core::BackendFailure::from_error(
                    eredu_core::execution_control::ExecutionControlError::Overflow,
                )
            })?;
        phrases.copy_bytes = 0;
        snapshot_value_estimate::<Self>([Ok(phrases), self.timeout_response.snapshot_retention()])
    }

    pub(super) fn copy_snapshot(&self) -> Result<Self, eredu_core::BackendFailure> {
        Ok(self.branch_candidate())
    }

    /// Stages unpublished wait progress while retaining the canonical receipts.
    /// Only the enclosing transaction may adopt it; this is not a second wait.
    pub(super) fn branch_candidate(&self) -> Self {
        Self {
            ticket: self.ticket,
            turn: self.turn,
            policy: self.policy,
            phrases: Arc::clone(&self.phrases),
            has_initial: self.has_initial,
            phrase: self.phrase,
            token: self.token,
            pending: self.pending,
            next_admission: self.next_admission,
            playback: self.playback,
            last_time: self.last_time,
            last_phrase_end: self.last_phrase_end,
            timeout_response: Arc::clone(&self.timeout_response),
            timed_out: self.timed_out,
        }
    }

    /// Prepares every selected phrase and timeout response before native work.
    /// `Configured` has no fallback on this standalone channel constructor;
    /// conversation sessions supply their retained selection automatically.
    pub fn new(
        channel: &VoiceChatToolChannel,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<Self, VoiceChatToolWaitError> {
        Self::with_prepared_initial(channel, ticket, at, policy, messages, None)
    }

    pub(super) fn with_prepared_initial(
        channel: &VoiceChatToolChannel,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
        configured: Option<&PreparedVoiceChatPhrase>,
    ) -> Result<Self, VoiceChatToolWaitError> {
        if channel.pending() != Some(ticket) || channel.response_policy().is_some() {
            return Err(VoiceChatToolWaitError::NotWaiting);
        }
        if messages.generic.len() > 2
            || (messages.generic.len() == 2 && messages.generic[0] == messages.generic[1])
        {
            return Err(VoiceChatToolWaitError::Messages);
        }
        let text = channel.text_preparation();
        let initial = match messages.initial {
            VoiceChatInitialReminder::Configured => {
                if let Some(phrase) =
                    configured.filter(|phrase| phrase.tokens().len() > messages.max_phrase_tokens)
                {
                    return Err(VoiceChatTextError::TokenCapacity {
                        kind: super::VoiceChatTextKind::ToolResponse,
                        required: phrase.tokens().len(),
                        capacity: messages.max_phrase_tokens,
                    }
                    .into());
                }
                configured.cloned()
            }
            VoiceChatInitialReminder::Disabled => None,
            VoiceChatInitialReminder::Text(phrase) => Some(PreparedVoiceChatPhrase::prepare(
                text,
                phrase,
                messages.max_phrase_tokens,
            )?),
        };
        let has_initial = initial.is_some();
        let mut phrases: Vec<_> = initial.into_iter().collect();
        for phrase in messages.generic {
            phrases.push(PreparedVoiceChatPhrase::prepare(
                text,
                phrase,
                messages.max_phrase_tokens,
            )?);
        }
        let timeout_response = text.prepare_tool_response(TIMEOUT_RESPONSE)?;
        Ok(Self {
            ticket,
            turn: channel.turn_identity(),
            policy,
            phrases: phrases.into(),
            has_initial,
            phrase: 0,
            token: 0,
            pending: None,
            next_admission: 0,
            playback: None,
            last_time: at,
            last_phrase_end: at,
            timeout_response: Arc::new(timeout_response),
            timed_out: false,
        })
    }

    /// Whether this wait has submitted the timeout response. This is retained
    /// status, not a repeated event for each response token.
    pub const fn timed_out(&self) -> bool {
        self.timed_out
    }

    /// Advances recorded time and selects ordinary component work. It never
    /// advances a speech/function cursor. Timeout replaces the external wait once.
    pub fn advance(
        &mut self,
        channel: &mut VoiceChatToolChannel,
        at: Duration,
    ) -> Result<VoiceChatToolWaitAction, VoiceChatToolWaitError> {
        self.validate(channel, at)?;
        let action = if channel.pending().is_none() {
            VoiceChatToolWaitAction::Complete
        } else if let Some(step) = self.pending {
            VoiceChatToolWaitAction::Reminder(step)
        } else if self.playback.is_some() {
            VoiceChatToolWaitAction::PerceptionOnly
        } else if self.token > 0 || (self.has_initial && self.phrase == 0) {
            self.reminder(at)?
        } else if let Some(policy) = channel.response_policy() {
            VoiceChatToolWaitAction::Response(policy)
        } else if self.phrase < self.phrases.len() {
            if at - self.last_phrase_end >= self.policy.generic_gap {
                self.reminder(at)?
            } else {
                VoiceChatToolWaitAction::PerceptionOnly
            }
        } else if at - self.last_phrase_end >= self.policy.timeout {
            channel.submit_prepared_response(self.ticket, (*self.timeout_response).clone())?;
            self.timed_out = true;
            VoiceChatToolWaitAction::Response(
                channel
                    .response_policy()
                    .expect("prepared response is nonempty"),
            )
        } else {
            VoiceChatToolWaitAction::PerceptionOnly
        };
        self.last_time = at;
        Ok(action)
    }

    /// Adopts one already committed reminder token. Interrupted, duplicated or
    /// out-of-order receipts fail without changing speech progress or timestamps.
    pub fn commit_reminder(
        &mut self,
        channel: &VoiceChatToolChannel,
        step: VoiceChatReminderStep,
        at: Duration,
    ) -> Result<Option<VoiceChatReminderPlayback>, VoiceChatToolWaitError> {
        self.validate(channel, at)?;
        if channel.pending() != Some(self.ticket)
            || self.pending != Some(step)
            || step.ticket != self.ticket
            || step.phrase != self.phrase
            || step.token != self.token
            || step.admitted_at > at
            || self
                .phrases
                .get(self.phrase)
                .and_then(|phrase| phrase.tokens().get(self.token))
                .copied()
                != Some(step.spoken as i32)
        {
            return Err(VoiceChatToolWaitError::Receipt);
        }
        self.pending = None;
        self.token += 1;
        if self.token == self.phrases[self.phrase].tokens().len() {
            self.playback = Some(VoiceChatReminderPlayback {
                ticket: self.ticket,
                phrase: self.phrase,
            });
            self.phrase += 1;
            self.token = 0;
        }
        self.last_time = at;
        Ok(self.playback)
    }

    /// Releases an admitted reminder only after the enclosing driver has safely
    /// discarded its neural candidate. No token or playback progress is consumed.
    pub fn discard_reminder(
        &mut self,
        channel: &VoiceChatToolChannel,
        step: VoiceChatReminderStep,
        at: Duration,
    ) -> Result<(), VoiceChatToolWaitError> {
        self.validate(channel, at)?;
        if self.pending != Some(step) {
            return Err(VoiceChatToolWaitError::Receipt);
        }
        self.pending = None;
        self.last_time = at;
        Ok(())
    }

    /// Records when all audio for a completed phrase has actually played out.
    /// Offline drivers supply the corresponding recorded presentation frontier.
    /// A duplicate or interrupted callback does not restart any timer.
    pub fn finish_playback(
        &mut self,
        channel: &VoiceChatToolChannel,
        playback: VoiceChatReminderPlayback,
        at: Duration,
    ) -> Result<(), VoiceChatToolWaitError> {
        self.validate(channel, at)?;
        if self.playback != Some(playback) || channel.pending() != Some(self.ticket) {
            return Err(VoiceChatToolWaitError::Receipt);
        }
        self.playback = None;
        self.last_phrase_end = at;
        self.last_time = at;
        Ok(())
    }

    fn reminder(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatToolWaitAction, VoiceChatToolWaitError> {
        let next = self
            .next_admission
            .checked_add(1)
            .ok_or(VoiceChatToolWaitError::IdentityExhausted)?;
        let step = VoiceChatReminderStep {
            ticket: self.ticket,
            admission: self.next_admission,
            phrase: self.phrase,
            token: self.token,
            spoken: self.phrases[self.phrase].tokens()[self.token] as u32,
            admitted_at: at,
        };
        self.next_admission = next;
        self.pending = Some(step);
        Ok(VoiceChatToolWaitAction::Reminder(step))
    }

    fn validate(
        &self,
        channel: &VoiceChatToolChannel,
        at: Duration,
    ) -> Result<(), VoiceChatToolWaitError> {
        if channel.turn_identity() != self.turn
            || channel
                .pending()
                .is_some_and(|ticket| ticket != self.ticket)
        {
            return Err(VoiceChatToolWaitError::Stale);
        }
        if at < self.last_time {
            return Err(VoiceChatToolWaitError::Time);
        }
        Ok(())
    }
}
