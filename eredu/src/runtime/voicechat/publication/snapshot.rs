//! Full saved publication state without a second live ingress/delivery owner.

#[cfg(test)]
mod tests;

use eredu_core::execution_control::{ExecutionControlError, SnapshotResourceKind};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, IsolatedStateCopy, SnapshotBudget, SnapshotReservation, SnapshotValue,
};

use super::super::conversation::ConversationSnapshot;
use super::*;

/// Portable snapshot rejection; rejected restores leave the live session unchanged.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatSnapshotError {
    /// Fork requests cannot reuse any ID admitted in the same session lineage.
    #[error("VoiceChat fork request {0:?} was already admitted in this lineage")]
    RequestReused(eredu_core::scheduler::RequestId),
    /// Rekeyed output does not fit the admitted queue or invocation limits.
    #[error(transparent)]
    Publication(#[from] VoiceChatPublicationError),
    /// The session has not configured its persistent snapshot limits.
    #[error("VoiceChat snapshot limits are not configured")]
    NotConfigured,
    /// Limits cannot be replaced or reset after configuration.
    #[error("VoiceChat snapshot limits are already configured")]
    AlreadyConfigured,
    /// Outstanding work retains an input, publication, or conversation admission.
    #[error("VoiceChat snapshot requires idle publication state")]
    Busy,
    /// A snapshot belongs to another session.
    #[error("VoiceChat snapshot belongs to another session")]
    Foreign,
    /// Restoring would discard audio or a finish declaration accepted since capture.
    #[error("VoiceChat input admission advanced since snapshot capture")]
    InputAdvanced,
    /// Restoring could replay speech, text, or a tool call already delivered externally.
    #[error("VoiceChat output was delivered since snapshot capture")]
    OutputDelivered,
    /// An external result, interruption, or playback action occurred since capture.
    #[error("VoiceChat host policy changed since snapshot capture")]
    HostChanged,
    /// The monotonic publication frontier cannot advance.
    #[error("VoiceChat publication revision exhausted")]
    RevisionExhausted,
    /// A complete known copy exceeds the shared logical budget.
    #[error(transparent)]
    Admission(#[from] ExecutionControlError),
    /// Copy, idle validation, or native restore failure with its concrete source.
    #[error("VoiceChat snapshot component failed: {0}")]
    Component(#[source] BackendFailure),
}

fn component(error: impl std::error::Error + Send + Sync + 'static) -> VoiceChatSnapshotError {
    VoiceChatSnapshotError::Component(BackendFailure::from_error(error))
}

pub(in crate::runtime::voicechat) struct PublicationSnapshot<S, D> {
    neural: S,
    conversation: ConversationSnapshot,
    queue: Queue<D>,
    input: Ingress,
    identity: Rc<()>,
    budget: SnapshotBudget,
    _reservation: SnapshotReservation,
}

impl<S: IsolatedStateCopy + SemanticStateTransaction, D: SnapshotValue>
    VoiceChatPublicationState<S, D>
{
    pub(super) fn snapshot_idle(&self) -> Result<(), VoiceChatSnapshotError> {
        if self.queue.borrow().usage().reserved_publications != 0
            || self.input.as_ref().is_none_or(|input| {
                let status = input.borrow().status();
                status.pending || status.closed
            })
        {
            return Err(VoiceChatSnapshotError::Busy);
        }
        // Conversation admission is independent of the queue reservations.
        self.conversation
            .validate_snapshot_idle()
            .map_err(|_| VoiceChatSnapshotError::Busy)?;
        Ok(())
    }

    pub(in crate::runtime::voicechat) fn capture_snapshot(
        &self,
        budget: &SnapshotBudget,
        metadata_bytes: usize,
    ) -> Result<PublicationSnapshot<S, D>, VoiceChatSnapshotError> {
        self.snapshot_idle()?;
        let input_guard = self.input.as_ref().expect("validated ingress").borrow();
        let queue = self.queue.borrow();
        let mut estimate = snapshot_value_estimate::<PublicationSnapshot<S, D>>([
            self.neural
                .estimate_isolated_copy()
                .map_err(BackendFailure::from_error),
            self.conversation.estimate_snapshot(),
            queue.estimate_snapshot_value(),
            input_guard.estimate_snapshot_value(),
        ])
        .map_err(component)?;
        estimate.retained_bytes = estimate
            .retained_bytes
            .checked_add(metadata_bytes as u64)
            .ok_or(ExecutionControlError::Overflow)?;
        let reservation = budget.reserve(SnapshotResourceKind::Snapshot, Some(estimate))?;
        let conversation = self
            .conversation
            .copy_snapshot(&reservation)
            .map_err(component)?;
        let queue_copy = queue.copy_snapshot_value(&reservation).map_err(component)?;
        let input = input_guard
            .copy_snapshot_value(&reservation)
            .map_err(component)?;
        let neural = self.neural.copy_isolated(&reservation).map_err(component)?;
        Ok(PublicationSnapshot {
            neural,
            conversation,
            queue: queue_copy,
            input,
            identity: self.identity.clone(),
            budget: budget.clone(),
            _reservation: reservation,
        })
    }

    pub(in crate::runtime::voicechat) fn validate_snapshot_origin(
        &self,
        saved: &PublicationSnapshot<S, D>,
    ) -> Result<(), VoiceChatSnapshotError> {
        if Rc::ptr_eq(&self.identity, &saved.identity) {
            Ok(())
        } else {
            Err(VoiceChatSnapshotError::Foreign)
        }
    }

    pub(in crate::runtime::voicechat) fn restore_snapshot(
        &mut self,
        saved: &PublicationSnapshot<S, D>,
    ) -> Result<(), VoiceChatSnapshotError> {
        self.snapshot_idle()?;
        self.validate_snapshot_origin(saved)?;
        let queue_guard = self.queue.borrow();
        if !Rc::ptr_eq(
            &queue_guard.delivery_identity(),
            &saved.queue.delivery_identity(),
        ) {
            return Err(VoiceChatSnapshotError::OutputDelivered);
        }
        let input = self.input.as_ref().expect("validated ingress");
        let input_guard = input.borrow();
        if !Rc::ptr_eq(
            &input_guard.external_identity(),
            &saved.input.external_identity(),
        ) {
            return Err(VoiceChatSnapshotError::InputAdvanced);
        }
        let epoch = self
            .epoch
            .get()
            .checked_add(1)
            .ok_or(VoiceChatSnapshotError::RevisionExhausted)?;
        self.conversation
            .validate_snapshot_idle()
            .map_err(component)?;
        self.neural
            .validate_isolated_restore(&saved.neural)
            .map_err(component)?;
        let estimate = snapshot_value_estimate::<PublicationSnapshot<S, D>>([
            saved
                .neural
                .estimate_isolated_copy()
                .map_err(BackendFailure::from_error),
            saved.conversation.estimate_snapshot_value(),
            saved.queue.estimate_snapshot_value(),
            saved.input.estimate_snapshot_value(),
        ])
        .map_err(component)?;
        let reservation = saved
            .budget
            .reserve(SnapshotResourceKind::Restore, Some(estimate))?;
        let conversation = saved
            .conversation
            .copy_snapshot_value(&reservation)
            .map_err(component)?;
        let mut queue = saved
            .queue
            .copy_snapshot_value(&reservation)
            .map_err(component)?;
        let mut candidate_input = saved
            .input
            .copy_snapshot_value(&reservation)
            .map_err(component)?;
        self.conversation
            .validate_snapshot_retention(&conversation)
            .map_err(component)?;
        queue
            .validate_snapshot_retention(&queue_guard)
            .map_err(component)?;
        if let Some((growth, _)) = &self.fork_neural_growth {
            let bytes = saved
                .neural
                .estimate_isolated_copy()
                .map_err(component)?
                .retained_bytes;
            growth.admit(bytes)?;
        }
        queue.retain_fork(self.fork_retention.clone());
        candidate_input.retain_fork(self.fork_retention.clone());
        let neural = saved
            .neural
            .copy_isolated(&reservation)
            .map_err(component)?;
        // Native installation is the only fallible mutation. All host storage is
        // prepared first; existing external sender/receiver identities survive.
        self.neural.restore_isolated(neural).map_err(component)?;
        drop(queue_guard);
        drop(input_guard);
        self.conversation.install_snapshot(conversation);
        let old_queue = std::mem::replace(&mut *self.queue.borrow_mut(), queue);
        let old_input = std::mem::replace(&mut *input.borrow_mut(), candidate_input);
        self.epoch.set(epoch);
        // User diagnostic destructors see fully installed host state. Any new
        // admissions they perform cannot be overwritten by this restore.
        drop(old_input);
        drop(old_queue);
        Ok(())
    }
}
