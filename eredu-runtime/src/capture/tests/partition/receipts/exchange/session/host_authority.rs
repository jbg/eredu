use super::*;
use eredu_core::HostPreparationAuthority;

struct RetiredPlan {
    plan: std::sync::Weak<AdmittedCapturePlan>,
    drops: Arc<AtomicUsize>,
}
impl Drop for RetiredPlan {
    fn drop(&mut self) {
        assert!(
            self.plan.upgrade().is_none(),
            "partition plan must retire before custody"
        );
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn existing_partition_work_keeps_late_attached_custody_after_session_retirement() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let mut session = configured(plan, 1);
    let transport = transport(world(1), 0, Fault::None);
    session
        .prepare_step_transaction(DistributedCommitEpoch::FIRST, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let work = prepare(&mut session, &transport).unwrap();
    assert!(work.global_reserved().host_bytes > 0);
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(RetiredPlan {
        plan: Arc::downgrade(session.plan.legacy_arc().expect("legacy session source")),
        drops: drops.clone(),
    });
    // Work predates attachment and has its own shared plan reference.
    session.retain_host_preparation(&authority).unwrap();
    drop((session, authority));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(work.global_reserved().host_bytes > 0);
    drop(work);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
}
