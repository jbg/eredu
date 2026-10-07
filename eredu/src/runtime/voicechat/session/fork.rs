//! Live child sessions over the ordinary scheduler and persistent lineage budget.

use super::super::{VoiceChatInputSender, VoiceChatPublicationReceiver};
use super::*;
use eredu_core::execution_control::{SnapshotEstimate, SnapshotResourceKind};
use eredu_runtime::execution_control::{
    ForkStateEstimate, IsolatedStateFork, SnapshotBudget, SnapshotReservation, SnapshotValue,
};

/// An independent continuation and its input/output endpoints.
///
/// Only undelivered committed publications are copied. Already delivered speech
/// and tool events are never reemitted. The mappings let applications explicitly
/// apply an already pending tool result or playback completion to the child;
/// source tickets/receipts alone are rejected by the child's policy.
pub struct VoiceChatFork<S, D>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
{
    /// Child using the same advancement, snapshot, cancellation and drain API.
    pub session: VoiceChatSession<S, D>,
    /// Independent PCM admission, initially containing the source's unread audio.
    pub input: VoiceChatInputSender,
    /// Independent committed output; inherited tool IDs carry the child request.
    pub output: VoiceChatPublicationReceiver<D>,
    /// Source request whose current idle frontier was copied.
    pub source_request: RequestId,
    /// Fresh request identity for all child tool and reminder work.
    pub request: RequestId,
    /// Current pending tool ticket as `(source, child)`, including result injection.
    /// This mapping does not authorize executing the same external tool twice.
    pub pending_tool: Option<(VoiceChatToolTicket, VoiceChatToolTicket)>,
    /// Current playback receipt as `(source, child)`, without replaying presented audio.
    pub pending_playback: Option<(VoiceChatReminderPlayback, VoiceChatReminderPlayback)>,
}

pub(super) struct ForkLineage {
    // IDs survive child cancellation/drop and are never rewound by snapshots.
    // Each additional entry retains its own bounded metadata reservation.
    ids: Vec<(RequestId, Option<SnapshotReservation>)>,
}
impl ForkLineage {
    pub(super) fn new(root: RequestId) -> Self {
        Self {
            ids: vec![(root, None)],
        }
    }
    fn validate(&self, request: RequestId) -> Result<(), VoiceChatSnapshotError> {
        if self.ids.iter().any(|(id, _)| *id == request) {
            Err(VoiceChatSnapshotError::RequestReused(request))
        } else {
            Ok(())
        }
    }
    fn claim(
        &mut self,
        request: RequestId,
        budget: &SnapshotBudget,
    ) -> Result<(), VoiceChatSnapshotError> {
        self.validate(request)?;
        let bytes = (std::mem::size_of::<(RequestId, Option<SnapshotReservation>)>() + 64) as u64;
        let copy_bytes = if self.ids.len() == self.ids.capacity() {
            self.ids
                .len()
                .checked_add(1)
                .and_then(|count| {
                    count.checked_mul(
                        std::mem::size_of::<(RequestId, Option<SnapshotReservation>)>(),
                    )
                })
                .and_then(|bytes| bytes.checked_add(64))
                .ok_or(eredu_core::execution_control::ExecutionControlError::Overflow)?
                as u64
        } else {
            bytes
        };
        let retention = budget.reserve(
            SnapshotResourceKind::BranchCopy,
            Some(SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes,
            }),
        )?;
        self.ids.try_reserve_exact(1).map_err(|error| {
            VoiceChatSnapshotError::Component(BackendFailure::from_error(error))
        })?;
        self.ids.push((request, Some(retention)));
        Ok(())
    }
}

impl<S, D> VoiceChatSession<S, D>
where
    S: SemanticStateTransaction + IsolatedStateFork + ForkStateEstimate,
    <S as SemanticStateTransaction>::Error: Send + Sync + 'static,
    D: SnapshotValue,
{
    /// Forks the current active, idle boundary without replaying prompt prefill.
    ///
    /// Requires previously configured snapshot limits. Child and source retain
    /// one shared budget, including further forks and failed copy attempts.
    /// Request IDs must also be unique among independently constructed sessions;
    /// this lineage rejects every previously admitted fork ID, including failed
    /// copy attempts. Dropping a child never makes its tickets reusable.
    ///
    /// Input/output and model state are independently owned. Retention charges
    /// remain while the child, its endpoints, or unresolved native work own the
    /// copied resources. Delivered output belongs to the receiving application.
    /// Later host and neural growth must fit that shared retention budget before
    /// commitment. Its high-water charges survive restore and output delivery.
    pub fn fork(
        &mut self,
        request: RequestId,
    ) -> Result<VoiceChatFork<S, D>, VoiceChatSessionError> {
        self.mutable_state()?;
        let budget = self
            .snapshot_budget
            .clone()
            .ok_or(VoiceChatSnapshotError::NotConfigured)?;
        let lineage = self.lineage.clone();
        let mut identities = lineage
            .try_borrow_mut()
            .map_err(|_| VoiceChatSessionError::Busy)?;
        identities.validate(request)?;
        let config = self.config;
        let last_time = self.last_time;
        let source_request = self.request;
        let forked = self.mutable_state()?.fork_state(
            request,
            &budget,
            config.publication_capacity,
            std::mem::size_of::<Self>() + std::mem::size_of::<VoiceChatFork<S, D>>(),
            || identities.claim(request, &budget),
        )?;
        drop(identities);
        let session = Self::register_idle(forked.state, config, last_time, Some(budget), lineage)?;
        Ok(VoiceChatFork {
            session,
            input: forked.input,
            output: forked.output,
            source_request,
            request,
            pending_tool: forked.tool,
            pending_playback: forked.playback,
        })
    }
}
