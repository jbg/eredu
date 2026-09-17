use super::*;
type Error = ReplicatedTextSessionError<&'static str, &'static str, &'static str>;
fn checked(
    next: DistributedCommitEpoch,
    active: Option<DistributedCommitEpoch>,
    outcome: Option<DistributedCommitOutcome>,
) -> Result<DistributedCommitEpoch, Error> {
    checked_upcoming_epoch(next, active, outcome)
}
#[test]
fn upcoming_epoch_check_is_readonly_and_rejects_the_last_counter_before_preparation() {
    let next = DistributedCommitEpoch::new(u64::MAX - 1).unwrap();
    for _ in 0..3 {
        assert_eq!(checked(next, None, None).unwrap(), next);
    }
    let last = DistributedCommitEpoch::new(u64::MAX).unwrap();
    assert!(
        matches!(checked(last,None,None),Err(Error::Contract(message)) if message=="distributed commit epoch overflow")
    );
    assert_eq!(last.value(), u64::MAX);
}
#[test]
fn active_and_indeterminate_epochs_keep_original_rejection_before_guarded_work() {
    let next = DistributedCommitEpoch::new(8).unwrap();
    let previous = DistributedCommitEpoch::new(7).unwrap();
    assert!(
        matches!(checked(next,Some(previous),None),Err(Error::Contract(message)) if message=="distributed transaction already has an active commit epoch")
    );
    let phase = eredu_core::DistributedCommitPhase::DecisionCompletion;
    assert!(
        matches!(checked(next,None,Some(DistributedCommitOutcome::Indeterminate{epoch:previous,phase})),Err(Error::CommitIndeterminate{epoch,phase:actual}) if epoch==previous && actual==phase)
    );
    assert_eq!(
        checked(
            next,
            None,
            Some(DistributedCommitOutcome::Committed(previous))
        )
        .unwrap(),
        next
    );
    assert_eq!(
        checked(
            next,
            None,
            Some(DistributedCommitOutcome::Aborted(previous))
        )
        .unwrap(),
        next
    );
}
