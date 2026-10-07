//! Independent publication owners with new host identities and retained copy credit.

use super::snapshot::VoiceChatSnapshotError;
use super::*;
use eredu_core::execution_control::SnapshotResourceKind;
use eredu_core::scheduler::RequestId;
use eredu_runtime::execution_control::{
    ForkStateEstimate, IsolatedStateFork, SnapshotBudget, SnapshotRetainedGrowth, SnapshotValue,
    snapshot_value_estimate,
};

pub(in crate::runtime::voicechat) struct ForkedPublication<S: SemanticStateTransaction, D> {
    pub state: VoiceChatPublicationState<S, D>,
    pub input: VoiceChatInputSender,
    pub output: VoiceChatPublicationReceiver<D>,
    pub tool: Option<(VoiceChatToolTicket, VoiceChatToolTicket)>,
    pub playback: Option<(VoiceChatReminderPlayback, VoiceChatReminderPlayback)>,
}

fn component(error: impl std::error::Error + Send + Sync + 'static) -> VoiceChatSnapshotError {
    VoiceChatSnapshotError::Component(BackendFailure::from_error(error))
}

impl<S: IsolatedStateFork + ForkStateEstimate, D: SnapshotValue> VoiceChatPublicationState<S, D> {
    pub(in crate::runtime::voicechat) fn fork_state(
        &self,
        request: RequestId,
        budget: &SnapshotBudget,
        capacity: VoiceChatPublicationCapacity,
        metadata: usize,
        claim_identity: impl FnOnce() -> Result<(), VoiceChatSnapshotError>,
    ) -> Result<ForkedPublication<S, D>, VoiceChatSnapshotError> {
        self.snapshot_idle()?;
        // Keep both source endpoints fenced through every user copy callback.
        let input = self.input.as_ref().expect("validated ingress").borrow();
        let queue = self.queue.borrow();
        queue.validate_fork(request, capacity)?;
        let neural_estimate = self.neural.estimate_isolated_copy().map_err(component)?;
        let mut estimate = snapshot_value_estimate::<Self>([
            Ok(neural_estimate),
            self.conversation.estimate_snapshot(),
            queue.fork_estimate(request),
            input.estimate_snapshot_value(),
        ])
        .map_err(component)?;
        estimate.retained_bytes = estimate
            .retained_bytes
            .checked_add(metadata as u64)
            // High-water counters retained outside rewindable component values.
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or(eredu_core::execution_control::ExecutionControlError::Overflow)?;
        let retention = budget.reserve(SnapshotResourceKind::Branch, Some(estimate))?;
        claim_identity()?;
        let conversation = self
            .conversation
            .copy_snapshot(&retention)
            .map_err(component)?;
        let tool = conversation
            .pending()
            .map(|ticket| (ticket, ticket.with_request(request)));
        let playback = conversation
            .pending_playback()
            .map(|receipt| (receipt, receipt.with_request(request)));
        let copied_queue = queue
            .copy_snapshot_value(&retention)
            .map_err(component)?
            .fork_owner(request, capacity, &retention)?;
        let copied_input = input
            .copy_snapshot_value(&retention)
            .map_err(component)?
            .fork_owner(&retention);
        let conversation = conversation
            .fork_into(request, &retention)
            .map_err(component)?;
        let neural = self.neural.fork_isolated(&retention).map_err(component)?;
        let sender = VoiceChatInputSender::from_owner(copied_input.clone());
        let receiver = VoiceChatPublicationReceiver::new(copied_queue.clone());
        let state = VoiceChatPublicationState {
            neural,
            reset_interruption_feedback: self.reset_interruption_feedback,
            conversation,
            queue: copied_queue,
            input: Some(copied_input),
            identity: Rc::new(()),
            epoch: Rc::new(Cell::new(0)),
            fork_neural_growth: Some((
                SnapshotRetainedGrowth::new(&retention, neural_estimate.retained_bytes),
                S::estimate_fork_branch,
            )),
            fork_retention: Some(retention),
        };
        Ok(ForkedPublication {
            state,
            input: sender,
            output: receiver,
            tool,
            playback,
        })
    }
}
