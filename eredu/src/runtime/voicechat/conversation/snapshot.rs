//! Independent host carry; the enclosing publication owns restore admission.

use super::*;
use eredu_core::{
    BackendFailure,
    execution_control::{ExecutionControlError, SnapshotEstimate},
};
use eredu_runtime::execution_control::{
    SnapshotReservation, SnapshotValue, snapshot_value_estimate,
};

/// Saved policy is not a runnable conversation owner and cannot publish events.
/// Request, turn and receipt identities remain exact. The enclosing session must
/// reject restore after external delivery or an external host action.
pub(in crate::runtime::voicechat) struct ConversationSnapshot {
    state: ConversationState,
}

impl ConversationSnapshot {
    pub(in crate::runtime::voicechat) fn pending(&self) -> Option<VoiceChatToolTicket> {
        self.state.channel.pending()
    }

    pub(in crate::runtime::voicechat) fn pending_playback(
        &self,
    ) -> Option<VoiceChatReminderPlayback> {
        self.state
            .wait
            .as_ref()
            .and_then(VoiceChatToolWait::playback_receipt)
    }

    /// Consumes an independently copied value after the enclosing driver admits
    /// the new request identity. No output is emitted and no tool is executed.
    pub(in crate::runtime::voicechat) fn fork_into(
        mut self,
        request: eredu_core::scheduler::RequestId,
        retention: &SnapshotReservation,
    ) -> Result<VoiceChatConversation, BackendFailure> {
        let retained_growth =
            SnapshotRetainedGrowth::new(retention, self.estimate_snapshot_value()?.retained_bytes);
        self.state.channel.rekey_for_fork(request);
        self.state.turn.rekey_for_fork(request);
        if let Some(wait) = &mut self.state.wait {
            wait.rekey_for_fork(request);
        }
        Ok(VoiceChatConversation {
            owner: Rc::new(RefCell::new(ConversationOwner {
                state: self.state,
                busy: false,
                retained_growth: Some(retained_growth),
            })),
        })
    }
}

impl SnapshotValue for ConversationState {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.turn.estimate_snapshot(),
            self.channel.estimate_snapshot(),
            self.wait.as_ref().map_or(
                Ok(SnapshotEstimate {
                    retained_bytes: 0,
                    copy_bytes: 0,
                }),
                |wait| wait.estimate_snapshot(),
            ),
            self.acknowledgement.snapshot_retention(),
            self.initial_reminder.snapshot_retention(),
            self.agent_text.estimate_snapshot_value(),
            self.transcription.estimate_snapshot_value(),
        ])
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            turn: self.turn.copy_snapshot()?,
            channel: self.channel.copy_snapshot()?,
            wait: self
                .wait
                .as_ref()
                .map(VoiceChatToolWait::copy_snapshot)
                .transpose()?,
            acknowledgement: self.acknowledgement.clone(),
            initial_reminder: self.initial_reminder.clone(),
            agent_text: self.agent_text.copy_snapshot_value(retention)?,
            transcription: self.transcription.copy_snapshot_value(retention)?,
            prefilled: self.prefilled,
            pending_audio_frames: self.pending_audio_frames,
            replay_frames: self.replay_frames,
            last_step: self.last_step,
            last_reminder_at: self.last_reminder_at,
            last_plan_at: self.last_plan_at,
        })
    }
}

impl SnapshotValue for ConversationSnapshot {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        conversation_estimate(&self.state)
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            state: self.state.copy_snapshot_value(retention)?,
        })
    }
}

impl VoiceChatConversation {
    /// Reserves a restored successor before the caller installs any neural state.
    pub(in crate::runtime::voicechat) fn validate_snapshot_retention(
        &self,
        saved: &ConversationSnapshot,
    ) -> Result<(), BackendFailure> {
        self.validate_snapshot_idle()
            .map_err(BackendFailure::from_error)?;
        self.owner.borrow().admit_snapshot_retention(&saved.state)
    }

    pub(in crate::runtime::voicechat) fn validate_snapshot_idle(
        &self,
    ) -> Result<(), VoiceChatConversationError> {
        let owner = self
            .owner
            .try_borrow()
            .map_err(|_| VoiceChatConversationError::Busy)?;
        if owner.busy {
            return Err(VoiceChatConversationError::Busy);
        }
        Ok(())
    }

    pub(in crate::runtime::voicechat) fn estimate_snapshot(
        &self,
    ) -> Result<SnapshotEstimate, BackendFailure> {
        self.validate_snapshot_idle()
            .map_err(BackendFailure::from_error)?;
        conversation_estimate(&self.owner.borrow().state)
    }

    pub(in crate::runtime::voicechat) fn copy_snapshot(
        &self,
        retention: &SnapshotReservation,
    ) -> Result<ConversationSnapshot, BackendFailure> {
        self.validate_snapshot_idle()
            .map_err(BackendFailure::from_error)?;
        Ok(ConversationSnapshot {
            state: self.owner.borrow().state.copy_snapshot_value(retention)?,
        })
    }

    /// Called only after idle/origin/external-frontier validation and successful
    /// neural installation, with exclusive publication ownership retained.
    pub(in crate::runtime::voicechat) fn install_snapshot(&mut self, saved: ConversationSnapshot) {
        self.owner.borrow_mut().state = saved.state;
    }
}

impl ConversationOwner {
    pub(super) fn admit_snapshot_retention(
        &self,
        state: &ConversationState,
    ) -> Result<(), BackendFailure> {
        if let Some(growth) = &self.retained_growth {
            let estimate = conversation_estimate(state)?;
            growth
                .admit(estimate.retained_bytes)
                .map_err(BackendFailure::from_error)?;
        }
        Ok(())
    }
}

fn conversation_estimate(state: &ConversationState) -> Result<SnapshotEstimate, BackendFailure> {
    snapshot_value_estimate::<ConversationSnapshot>([
        state.estimate_snapshot_value(),
        // Runnable copies retain a tracker and its independent Rc<Cell<u64>>.
        // Saved DTOs do not copy live admission ownership; pricing this metadata
        // for both copy forms conservatively admits later child construction.
        host_storage::<SnapshotRetainedGrowth>(64, 1),
    ])
}

/// Conservative logical host allocation size, including inline metadata.
pub(in crate::runtime::voicechat) fn host_storage<T>(
    capacity: usize,
    element_bytes: usize,
) -> Result<SnapshotEstimate, BackendFailure> {
    let bytes = capacity
        .checked_mul(element_bytes)
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<T>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
    Ok(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}

#[cfg(test)]
mod tests;
