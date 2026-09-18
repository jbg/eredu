//! Different prompt geometry and replay spending stay on their selected cursor.
use super::*;
use crate::speculative::autoregressive::occurrence_owners::OccurrenceOwners;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeRequestId};
use std::cell::RefCell;

#[test]
fn independent_batch_cursors_keep_geometry_spending_and_continuation_per_request() {
    let selected = selected();
    let config = SpeculativeConfig { max_tokens: 3, max_draft_tokens: 1, ..Default::default() };
    let plan = |input| AutoregressiveSchedulePlan::new(&selected,
        NonZeroUsize::new(1).unwrap(), NonZeroU64::new(input).unwrap(), NonZeroU64::new(16).unwrap(),
        &config, SpeculativeSchedulerOptions::default()).unwrap();
    let first = plan(2);
    let second = plan(5);
    let identities = [first.identity(), second.identity()];
    let proposal_attempts = first.domains().iter().find(|domain| domain.pass() == AutoregressivePass::Proposal).unwrap().attempts();
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let funding = pool.prepare_workspace_metadata(&execution, 1 << 20).unwrap();
    let bytes = SpeculativeBuffer::<RefCell<AutoregressiveOccurrenceCursor<'_>>>::retained_control_bytes(2).unwrap()
        + HostPreparationAuthority::retention_bytes::<eredu_nn::workspace::HostMetadataFunding>().unwrap();
    funding.reserve_metadata(bytes).unwrap();
    let mut cursors = SpeculativeBuffer::try_new_retained(2, HostPreparationAuthority::retain(funding.clone())).unwrap();
    cursors.try_push(RefCell::new(first.into_cursor())).unwrap();
    cursors.try_push(RefCell::new(second.into_cursor())).unwrap();
    let owner = OccurrenceOwners::Table { cursors, _host: HostPreparationAuthority::unmanaged() };
    let cursor = |index| owner.select(Some(SpeculativeRequestId::new(index))).unwrap().unwrap();
    assert!(owner.select(None).is_err());
    assert!(owner.select(Some(SpeculativeRequestId::new(2))).is_err());
    for index in 0..2 { cursor(index).borrow_mut().begin_cache().unwrap(); }
    assert!(cursor(0).borrow_mut().begin_cache().is_err());
    let call = |width| AutoregressiveInvocation::prefill(AutoregressivePass::TargetPrefill, width).unwrap();
    assert!(cursor(1).borrow_mut().claim(0, call(2)).is_err());
    assert_eq!(cursor(1).borrow().attempted(), 0);
    let first = cursor(0).borrow_mut().claim(0, call(2)).unwrap();
    let second = cursor(1).borrow_mut().claim(0, call(5)).unwrap();
    assert_eq!(first.schedule().identity(), identities[0]);
    assert_eq!(second.schedule().identity(), identities[1]);
    drop((first, second)); // Abandoned work does not restore prefill capacity.
    assert!(cursor(0).borrow_mut().claim(0, call(2)).is_err());
    assert_eq!(cursor(1).borrow().attempted(), 1);
    let proposal = AutoregressiveInvocation::decode(AutoregressivePass::Proposal, 1).unwrap();
    for _ in 0..proposal_attempts { cursor(0).borrow_mut().claim(2, proposal).unwrap(); }
    assert!(cursor(0).borrow_mut().claim(2, proposal).is_err());
    let continuation = cursor(0).borrow().continuation(1, eredu_core::SpeculativeRequestStatus::ReadyToDraft).unwrap();
    cursor(0).borrow_mut().install_continuation(continuation);
    let replay = cursor(0).borrow_mut().claim(2, proposal).unwrap();
    assert_eq!(replay.ordinal(), 1 + proposal_attempts);
    assert_eq!(cursor(1).borrow().attempted(), 1);
    drop((replay, funding));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(owner);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
