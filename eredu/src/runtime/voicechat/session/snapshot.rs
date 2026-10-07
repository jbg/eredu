//! Explicit replay of an idle session before external effects advance.

use super::super::publication::snapshot::PublicationSnapshot;
use super::*;
use eredu_core::execution_control::{SnapshotLimits, SnapshotUsage};
use eredu_runtime::execution_control::{IsolatedStateCopy, SnapshotBudget, SnapshotValue};

/// Reusable complete idle-session snapshot, charged to its original copy budget.
///
/// Contains neural and host policy state, accepted PCM, undelivered publications,
/// and the recorded-time cursor. Capture and restore perform no model transition.
/// Delivered outputs, newly accepted PCM/finish, tool results, playback completion,
/// and explicit interruptions are irreversible boundaries: restore rejects crossing
/// them. Scheduler counters, state revisions, and cumulative budgets never rewind.
pub struct VoiceChatSessionSnapshot<S, D> {
    state: PublicationSnapshot<S, D>,
    host_actions: Rc<()>,
    last_time: Duration,
}

impl<S, D> VoiceChatSession<S, D>
where
    S: SemanticStateTransaction + IsolatedStateCopy,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
    D: SnapshotValue,
{
    /// Configures the shared snapshot/copy limits exactly once for this session.
    /// May be called before prefill; no state is copied or native work submitted.
    pub fn enable_snapshots(
        &mut self,
        limits: SnapshotLimits,
    ) -> Result<(), VoiceChatSessionError> {
        if let Some(status) = self.scheduler.request_status(self.request) {
            if status != RequestStatus::Active {
                return Err(VoiceChatSessionError::Terminal(status));
            }
        }
        if self.snapshot_budget.is_some() {
            return Err(VoiceChatSnapshotError::AlreadyConfigured.into());
        }
        self.snapshot_budget = Some(SnapshotBudget::new(limits));
        Ok(())
    }

    /// Reports retained snapshot storage and cumulative copy work, including
    /// failed admitted copies. Restore never replaces this accounting owner.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.snapshot_budget.as_ref().map(SnapshotBudget::usage)
    }

    /// Captures complete idle state before any externally irreversible action.
    /// Queued prefill or pending work must retire before capture.
    pub fn snapshot(&mut self) -> Result<VoiceChatSessionSnapshot<S, D>, VoiceChatSessionError> {
        let budget = self
            .snapshot_budget
            .clone()
            .ok_or(VoiceChatSnapshotError::NotConfigured)?;
        let state = self.mutable_state()?.capture_snapshot(
            &budget,
            std::mem::size_of::<VoiceChatSessionSnapshot<S, D>>(),
        )?;
        Ok(VoiceChatSessionSnapshot {
            state,
            host_actions: self.host_actions.clone(),
            last_time: self.last_time,
        })
    }

    /// Restores an idle session without undoing any externally delivered effects.
    /// Failed validation or copying changes neither live state nor recorded time.
    /// On success replay may resubmit ticks from the saved recorded-time cursor.
    pub fn restore(
        &mut self,
        saved: &VoiceChatSessionSnapshot<S, D>,
    ) -> Result<(), VoiceChatSessionError> {
        // Check lifecycle before the external-action frontier for consistent busy
        // and terminal behavior, without admitting a copy on either rejection.
        self.mutable_state()?
            .validate_snapshot_origin(&saved.state)?;
        if !Rc::ptr_eq(&self.host_actions, &saved.host_actions) {
            return Err(VoiceChatSnapshotError::HostChanged.into());
        }
        self.mutable_state()?.restore_snapshot(&saved.state)?;
        self.last_time = saved.last_time;
        Ok(())
    }
}
