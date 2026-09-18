//! Original dense edit sources and prepaid member votes on the existing transport.
use super::*;
use crate::intervention::{InterventionPrefillWindow, PreparedPartitionInterventionProjection};
use crate::working_memory::{CaptureInterventionClaim, OriginalInterventionSource};
use eredu_core::{BoundedCompletion, BoundedCompletionWait};
use eredu_nn::workspace::{
    WorkspaceContext, HostMetadataFunding, HostMetadataFundingError,
};
use std::mem::{size_of, size_of_val};
mod allowance;
mod source;
mod outcome;
pub(crate) use outcome::PartitionInterventionOutcome;
pub use allowance::PartitionInterventionLocalAllowance;
pub use source::{
    PartitionInterventionInvocationSource, PartitionInterventionMemberSource,
    PreparedPartitionInterventionSource,
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("partition intervention source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Metadata(#[from] eredu_nn::Error),
    #[error(transparent)]
    Memory(#[from] crate::working_memory::WorkingMemoryError),
    #[error(transparent)]
    Exchange(#[from] PartitionCaptureExchangeError),
    #[error(transparent)]
    Window(#[from] crate::intervention::InterventionPrefillSourceError),
}
/// Refusals retain the immutable original operation and all metadata custody.
/// Consumed logical credits remain charged to their parent on every error.
#[derive(Debug, thiserror::Error)]
#[error("original partition intervention: {cause}")]
pub struct PartitionInterventionSourceError {
    #[source]
    cause: Cause,
    _source: OriginalInterventionSource,
    _metadata: HostMetadataFunding,
}
#[derive(Debug)]
struct Identity;
#[derive(Debug)]
struct Member {
    rank: usize,
    geometry: [u8; 32],
    execution: [u8; 32],
    shape: Vec<u64>,
    usage: CaptureUsage,
    projection: [CaptureUsage; 2],
    source_usage: CaptureUsage,
}
impl Member {
    fn total(&self) -> Result<CaptureUsage, CaptureError> {
        self.usage
            .checked_add(self.projection[0])?
            .checked_add(self.projection[1])?
            .checked_add(self.source_usage)
    }
    fn global(&self, world: u64) -> Result<CaptureUsage, CaptureError> {
        let mut usage = self
            .usage
            .checked_add(self.projection[0])?
            .checked_add(self.projection[1])?;
        // Same ordinary policy: replicated Host projection work is present on
        // every rank; actual numerical dependencies are retained per member.
        usage.host_bytes = usage
            .host_bytes
            .checked_mul(world)
            .ok_or(CaptureError::Overflow)?;
        usage.checked_add(self.source_usage)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Unprepared,
    Absent,
    Ready,
    Lent,
    Accepted,
    Rejected,
}
#[derive(Debug)]
struct Invocation {
    window: Option<InterventionPrefillWindow>,
    ranks: Vec<usize>,
    members: Vec<Option<Member>>,
    local: Option<usize>,
    quota: Option<CaptureQuota>,
    vote: Option<CaptureQuota>,
    vote_usage: CaptureUsage,
    state: State,
}
type Agree<T> =
    fn(&T, &[usize], BoundedCompletionWait, bool) -> Result<bool, PartitionCaptureExchangeError>;
/// One move-only original operation. This reuses the ordinary member agreement
/// and world receipt workers; it never owns native execution or tensor values.
pub(crate) struct PreparedPartitionIntervention<'t, T: PartitionCaptureTransport> {
    transport: &'t T,
    source: PreparedPartitionInterventionSource,
    epoch: DistributedCommitEpoch,
    rank: usize,
    wait: BoundedCompletionWait,
    agree: Agree<T>,
    global: CaptureUsage,
    local: CaptureUsage,
    receipt: CaptureQuota,
    receipt_usage: CaptureUsage,
    remaining: CaptureQuota,
    delivered: bool,
}
impl<T: PartitionCaptureTransport> PreparedPartitionIntervention<'_, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    pub(crate) fn operation(&self) -> usize {
        self.source.operation
    }
    pub(crate) fn epoch(&self)->DistributedCommitEpoch {self.epoch}
    pub(crate) fn member(&self,window:Option<InterventionPrefillWindow>)->Result<bool,PartitionInterventionSourceError> {
        let index=self.window_index(window)?;Ok(self.source.invocations[index].local.is_some())
    }
    pub(crate) fn local_window_complete(&self,window:Option<InterventionPrefillWindow>)->Result<bool,PartitionInterventionSourceError> {
        let index=self.window_index(window)?;Ok(matches!(self.source.invocations[index].state,State::Absent|State::Accepted))
    }
    pub(crate) fn descriptor(&self) -> &[u8; 32] {
        &self.source.descriptor
    }
    pub(crate) fn global_reserved(&self) -> CaptureUsage {
        self.global
    }
    pub(crate) fn local_reserved(&self) -> CaptureUsage {
        self.local
    }
    pub(crate) fn original(&self) -> &OriginalInterventionSource {
        &self.source.source
    }
    pub(crate) fn coordinate(&self) -> (CapturePhase, u64) {
        (self.source.phase, self.source.prediction)
    }
    pub(crate) fn window_index(
        &self,
        window: Option<InterventionPrefillWindow>,
    ) -> Result<usize, PartitionInterventionSourceError> {
        self.source
            .invocations
            .iter()
            .position(|row| row.window == window)
            .ok_or_else(|| {
                self.source.error(Cause::Source(
                    "hook is outside the original invocation schedule",
                ))
            })
    }
    /// The ordinary preflight vote occurs before a local quota can escape.
    /// A failed local source still joins both votes and remains a failed source.
    pub(crate) fn begin(
        &mut self,
        window: Option<InterventionPrefillWindow>,
        valid: bool,
    ) -> Result<Option<PartitionInterventionLocalAllowance>, PartitionInterventionSourceError> {
        let transport = self.transport;
        self.validate_transport()?;
        let index = self.window_index(window)?;
        let result = (|| -> Result<_, Cause> {
            if self.source.invocations[..index]
                .iter()
                .any(|row| !matches!(row.state, State::Absent | State::Accepted))
            {
                return Err(Cause::Source("prior original invocation has not completed"));
            }
            let row = &mut self.source.invocations[index];
            if row.state == State::Absent {
                return Ok(None);
            }
            if row.state != State::Ready {
                return Err(Cause::Source(
                    "local hook is repeated, missing or already lent",
                ));
            }
            row.state = State::Rejected;
            let votes = row
                .vote
                .as_mut()
                .ok_or(Cause::Source("local hook has no vote allowance"))?;
            votes.reserve_quota(row.vote_usage)?;
            let first = (self.agree)(transport, &row.ranks, self.wait, valid);
            // A completed refusal still follows the same second member vote.
            // An unsafe transport failure has already failed the shared world.
            match first {
                Ok(true) if valid => (),
                Ok(_) => {
                    votes.reserve_quota(row.vote_usage)?;
                    let final_vote = (self.agree)(transport, &row.ranks, self.wait, false);
                    final_vote?;
                    return Err(Cause::Source("member preflight was rejected"));
                }
                Err(cause) => return Err(cause.into()),
            }
            let local = row
                .local
                .ok_or(Cause::Source("accepted invocation has no local member"))?;
            let member = row.members[local]
                .take()
                .ok_or(Cause::Source("local descriptor is spent"))?;
            let quota = row
                .quota
                .take()
                .ok_or(Cause::Source("local allowance is spent"))?;
            row.state = State::Lent;
            Ok(Some(PartitionInterventionLocalAllowance {
                member,
                quota,
                window,
                index,
                epoch: self.epoch,
                operation: self.source.operation,
                phase: self.source.phase,
                prediction: self.source.prediction,
                identity: Arc::clone(&self.source.identity),
                source: self.source.source.clone(),
                metadata: self.source.metadata.clone(),
                spent: false,
                charges: 0,
                validated: false,
            }))
        })();
        result.map_err(|cause| self.source.error(cause))
    }
    /// Return the exact detached loan before the second member vote. The caller
    /// preserves its original native error when this also reports rejection.
    pub(crate) fn finish(
        &mut self,
        loan: PartitionInterventionLocalAllowance,
        success: bool,
    ) -> Result<(), PartitionInterventionSourceError> {
        let transport = self.transport;
        self.validate_transport()?;
        let result = (|| -> Result<(), Cause> {
            if !Arc::ptr_eq(&loan.identity, &self.source.identity)
                || !loan.source.same_source(&self.source.source)
                || loan.epoch != self.epoch
                || loan.member.rank != self.rank
                || loan.operation != self.source.operation
                || loan.phase != self.source.phase
                || loan.prediction != self.source.prediction
            {
                return Err(Cause::Source(
                    "local allowance belongs to another operation owner",
                ));
            }
            let row = self
                .source
                .invocations
                .get_mut(loan.index)
                .ok_or(Cause::Source("invalid invocation ordinal"))?;
            if row.state != State::Lent || row.window != loan.window {
                return Err(Cause::Source("local allowance was not lent here"));
            }
            let local = row
                .local
                .ok_or(Cause::Source("returned member is absent"))?;
            let valid = success && loan.spent && loan.quota.used() == loan.member.total()?;
            row.members[local] = Some(loan.member);
            row.quota = Some(loan.quota);
            row.state = State::Rejected;
            row.vote
                .as_mut()
                .ok_or(Cause::Source("postflight vote has no allowance"))?
                .reserve_quota(row.vote_usage)?;
            let agreed = (self.agree)(transport, &row.ranks, self.wait, valid)?;
            if !valid || !agreed {
                return Err(Cause::Source("member execution was rejected"));
            }
            row.state = State::Accepted;
            Ok(())
        })();
        result.map_err(|cause| self.source.error(cause))
    }
    fn validate_transport(&self) -> Result<(), PartitionInterventionSourceError> {
        if self.transport.capture_rank() != self.rank
            || self.transport.participant_count() != self.source.world
        {
            return Err(self
                .source
                .error(Cause::Source("local rank or world changed")));
        }
        Ok(())
    }
}
