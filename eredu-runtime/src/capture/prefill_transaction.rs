//! Shared scalar transaction rules. Logical frame/funding owners remain with
//! their existing collectors; no allocation, scheduler or completion is added.
use eredu_core::DistributedCommitEpoch;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Announced,
    Prepared,
    Complete,
    Between,
    Finished,
}
pub(super) fn can_prepare(
    phase: Phase,
    failed: bool,
    current: Option<DistributedCommitEpoch>,
    expected: Option<DistributedCommitEpoch>,
    epoch: DistributedCommitEpoch,
) -> bool {
    phase == Phase::Announced
        && !failed
        && current.is_none_or(|old| old < epoch)
        && expected.is_none_or(|value| value == epoch)
}
pub(super) fn can_complete(
    phase: Phase,
    failed: bool,
    current: Option<DistributedCommitEpoch>,
    epoch: DistributedCommitEpoch,
) -> bool {
    phase == Phase::Prepared && !failed && current == Some(epoch)
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Finish {
    Ignored,
    Invalid,
    Committed,
    Aborted,
}
pub(super) fn finish(
    phase: &mut Phase,
    failed: &mut bool,
    current: Option<DistributedCommitEpoch>,
    epoch: DistributedCommitEpoch,
    committed: bool,
) -> Finish {
    if *phase == Phase::Finished {
        return Finish::Ignored;
    }
    if current != Some(epoch) || !matches!(*phase, Phase::Prepared | Phase::Complete) {
        *failed = true;
        return Finish::Invalid;
    }
    if committed && *phase == Phase::Complete && !*failed {
        *phase = Phase::Between;
        Finish::Committed
    } else {
        *phase = Phase::Finished;
        Finish::Aborted
    }
}
