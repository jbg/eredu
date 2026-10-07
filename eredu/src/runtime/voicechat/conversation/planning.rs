//! Recorded-time selection within the ordinary conversation transaction.

use std::{rc::Rc, time::Duration};

use eredu_architectures::voicechat::{
    channel_decision::ChannelDecisionPolicy,
    schedule::{AudioSource, Step},
};

use super::super::driver::{
    VoiceChatAdvance, VoiceChatBlocked, VoiceChatToolAudioPolicy, VoiceChatWorkKind,
};
use super::{
    ConversationState, VoiceChatConversation, VoiceChatConversationError,
    VoiceChatConversationTransition, VoiceChatPublicationCapacity, VoiceChatToolWaitAction,
};

const REMINDER_CADENCE: Duration = Duration::from_millis(80);

impl VoiceChatConversation {
    /// Reserves explicit prompt/Aria prefill and its recorded starting time.
    pub(in crate::runtime::voicechat) fn prepare_prefill(
        &mut self,
        backbone_tokens: u64,
        at: Duration,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<VoiceChatConversationTransition, VoiceChatConversationError> {
        let mut owner = self.owner.borrow_mut();
        if owner.busy {
            return Err(VoiceChatConversationError::Busy);
        }
        if owner.state.prefilled {
            return Err(VoiceChatConversationError::AlreadyPrefilled);
        }
        if owner
            .state
            .last_plan_at
            .is_some_and(|previous| at < previous)
        {
            return Err(VoiceChatConversationError::Time);
        }
        let mut candidate = owner.state.branch_candidate();
        let admission = candidate
            .turn
            .admission(&candidate.channel, Step::Prefill { backbone_tokens })?;
        candidate.last_plan_at = Some(at);
        owner.busy = true;
        Ok(VoiceChatConversationTransition {
            owner: Rc::clone(&self.owner),
            candidate: Some(candidate),
            admission,
            reminder: None,
            capacity,
        })
    }

    /// Selects a candidate without consuming canonical time or response tokens.
    /// A ready transition freezes host mutation immediately, including while its
    /// enclosing work waits in the shared scheduler. A blocked selection leaves
    /// both policy and the committed recorded-time frontier unchanged.
    pub(in crate::runtime::voicechat) fn prepare_next(
        &mut self,
        advance: VoiceChatAdvance,
        fresh_input: bool,
        input_finished: bool,
        capacity: VoiceChatPublicationCapacity,
    ) -> Result<
        Result<(VoiceChatWorkKind, VoiceChatConversationTransition), VoiceChatBlocked>,
        VoiceChatConversationError,
    > {
        let mut owner = self.owner.borrow_mut();
        if owner.busy {
            return Err(VoiceChatConversationError::Busy);
        }
        if owner.state.turn.frontier().codec_finished {
            return Ok(Err(VoiceChatBlocked::Finished));
        }
        if !owner.state.prefilled {
            return Ok(Err(VoiceChatBlocked::NeedsPrefill));
        }
        if owner.state.last_plan_at.is_some_and(|at| advance.at < at) {
            return Err(VoiceChatConversationError::Time);
        }
        let mut candidate = owner.state.branch_candidate();
        let kind = match candidate.select_next(advance, fresh_input, input_finished)? {
            Ok(kind) => kind,
            Err(blocked) => return Ok(Err(blocked)),
        };
        let admission = candidate.turn.admission(&candidate.channel, kind.step())?;
        let reminder = match kind {
            VoiceChatWorkKind::Reminder(receipt) => Some((receipt, advance.at)),
            _ => None,
        };
        candidate.last_plan_at = Some(advance.at);
        owner.busy = true;
        Ok(Ok((
            kind,
            VoiceChatConversationTransition {
                owner: Rc::clone(&self.owner),
                candidate: Some(candidate),
                admission,
                reminder,
                capacity,
            },
        )))
    }
}

impl ConversationState {
    fn select_next(
        &mut self,
        advance: VoiceChatAdvance,
        fresh_input: bool,
        input_finished: bool,
    ) -> Result<Result<VoiceChatWorkKind, VoiceChatBlocked>, VoiceChatConversationError> {
        if self.replay_frames != 0 {
            return Ok(Ok(VoiceChatWorkKind::ReplayAudio));
        }
        if let Some(wait) = &mut self.wait {
            match wait.advance(&mut self.channel, advance.at)? {
                VoiceChatToolWaitAction::Reminder(receipt) => {
                    let due = self.last_reminder_at.is_none_or(|previous| {
                        advance.at.saturating_sub(previous) >= REMINDER_CADENCE
                    });
                    // A due phrase alternates with queued perception work. Its
                    // pending receipt remains in this candidate when fresh PCM
                    // wins, without consuming a reminder token or speech tick.
                    if fresh_input && (!due || self.last_step != Some(Step::PerceptionOnly)) {
                        return Ok(Ok(VoiceChatWorkKind::PerceptionOnly));
                    }
                    return Ok(if due {
                        Ok(VoiceChatWorkKind::Reminder(receipt))
                    } else {
                        Err(VoiceChatBlocked::WaitingForTime)
                    });
                }
                VoiceChatToolWaitAction::PerceptionOnly => {
                    return Ok(if fresh_input {
                        Ok(VoiceChatWorkKind::PerceptionOnly)
                    } else {
                        Err(VoiceChatBlocked::WaitingForTime)
                    });
                }
                VoiceChatToolWaitAction::Response(decisions) => {
                    return Ok(self.select_tool(fresh_input, decisions, advance.tool_audio));
                }
                VoiceChatToolWaitAction::Complete => self.wait = None,
            }
        }
        if let Some(decisions) = self.channel.response_policy() {
            return Ok(self.select_tool(fresh_input, decisions, advance.tool_audio));
        }
        if self.channel.pending().is_some() {
            return Ok(if fresh_input {
                Ok(VoiceChatWorkKind::PerceptionOnly)
            } else {
                Err(VoiceChatBlocked::WaitingForTool)
            });
        }
        if self.channel.is_active() {
            return Ok(self.select_tool(
                fresh_input,
                ChannelDecisionPolicy::Tool {
                    function_feedback: None,
                    spoken_text: self.acknowledgement_token(),
                },
                advance.tool_audio,
            ));
        }
        Ok(if fresh_input {
            Ok(VoiceChatWorkKind::Live)
        } else if input_finished {
            Ok(VoiceChatWorkKind::Drain)
        } else {
            Err(VoiceChatBlocked::NeedsInput)
        })
    }

    fn select_tool(
        &self,
        fresh_input: bool,
        decisions: ChannelDecisionPolicy,
        policy: VoiceChatToolAudioPolicy,
    ) -> Result<VoiceChatWorkKind, VoiceChatBlocked> {
        let audio = if policy == VoiceChatToolAudioPolicy::Silence {
            // Released asynchronous function decoding uses a separately prepared
            // silence embedding. Continue ingesting speech between tool positions
            // without feeding that speech to the function-channel backbone.
            if fresh_input && self.last_step != Some(Step::PerceptionOnly) {
                return Ok(VoiceChatWorkKind::PerceptionOnly);
            }
            AudioSource::Silence
        } else if fresh_input {
            AudioSource::Fresh
        } else if self.turn.frontier().perception_frames > 0 {
            AudioSource::Retained
        } else {
            AudioSource::Silence
        };
        Ok(VoiceChatWorkKind::Tool { audio, decisions })
    }
}
