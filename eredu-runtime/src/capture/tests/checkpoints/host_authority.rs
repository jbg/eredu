use super::*;
use eredu_core::HostPreparationAuthority;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Weak,
};

struct Drops(Arc<AtomicUsize>);
impl Drop for Drops {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn authority(count: &Arc<AtomicUsize>) -> HostPreparationAuthority {
    HostPreparationAuthority::retain(Drops(count.clone()))
}

#[test]
fn checkpoint_aliases_created_before_attachment_keep_source_and_destination_custody() {
    let (session, discovery) = setup();
    let source_count = Arc::new(AtomicUsize::new(0));
    let destination_count = Arc::new(AtomicUsize::new(0));
    let source = authority(&source_count);
    let destination = authority(&destination_count);
    session.retain_host_preparation(&source).unwrap();
    let saved = session.checkpoint(&discovery).unwrap();
    let earlier_alias = saved.clone();
    saved.retain_host_preparation(&destination).unwrap();
    drop((session, saved, source, destination));
    assert_eq!(source_count.load(Ordering::SeqCst), 0);
    assert_eq!(destination_count.load(Ordering::SeqCst), 0);
    assert_eq!(earlier_alias.artifact_identity(), "source");
    assert_eq!(earlier_alias.next_prediction(), 0);
    assert!(earlier_alias.logical_storage_bytes().unwrap() > 0);
    drop(earlier_alias);
    assert_eq!(source_count.load(Ordering::SeqCst), 1);
    assert_eq!(destination_count.load(Ordering::SeqCst), 1);
}

#[test]
fn child_keeps_inherited_custody_with_distinct_run_identity_and_nonrefunding_restore() {
    let (mut session, discovery) = setup();
    let source_count = Arc::new(AtomicUsize::new(0));
    let destination_count = Arc::new(AtomicUsize::new(0));
    let source = authority(&source_count);
    session.retain_host_preparation(&source).unwrap();
    run(&mut session, 0);
    let saved = session.checkpoint(&discovery).unwrap();
    let mut child = saved
        .fork(
            CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 10,
                limits: session.plan().plan().limits.clone(),
                intervention: None,
            },
            |_, _, _| Ok(CaptureUsage::default()),
        )
        .unwrap();
    let destination = authority(&destination_count);
    child.retain_host_preparation(&destination).unwrap();
    assert!(child.restore(&saved).is_err());
    let own = child.checkpoint(&discovery).unwrap();
    run(&mut child, 1);
    let charged = child.cumulative_usage();
    child.restore(&own).unwrap();
    assert_eq!(child.cumulative_usage(), charged);
    drop((session, saved, own, source, destination));
    assert_eq!(source_count.load(Ordering::SeqCst), 0);
    assert_eq!(destination_count.load(Ordering::SeqCst), 0);
    assert_eq!(child.checkpoint(&discovery).unwrap().next_prediction(), 1);
    drop(child);
    assert_eq!(source_count.load(Ordering::SeqCst), 1);
    assert_eq!(destination_count.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_child_estimation_keeps_original_checkpoint_and_custody_without_publishing_child() {
    let (session, discovery) = setup();
    let drops = Arc::new(AtomicUsize::new(0));
    let source = authority(&drops);
    session.retain_host_preparation(&source).unwrap();
    let saved = session.checkpoint(&discovery).unwrap();
    let original = saved.inherited_usage();
    let result = saved.fork(
        CaptureForkRequest {
            discovery: &discovery,
            max_predictions: 10,
            limits: session.plan().plan().limits.clone(),
            intervention: None,
        },
        |_, _, _| {
            Err(CaptureError::Unsupported(
                "exact injected host-copy preflight failure".into(),
            ))
        },
    );
    assert!(
        matches!(result, Err(CaptureError::Unsupported(reason)) if reason == "exact injected host-copy preflight failure")
    );
    assert_eq!(saved.inherited_usage(), original);
    assert_eq!(saved.next_prediction(), 0);
    drop((session, source));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(saved);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct ReentrantRetirement {
    plan: Weak<AdmittedCapturePlan>,
    target: Arc<crate::capture::CaptureHostOwner>,
    drops: Arc<AtomicUsize>,
}
impl Drop for ReentrantRetirement {
    fn drop(&mut self) {
        assert!(
            self.plan.upgrade().is_none(),
            "plan payload must retire before its authority"
        );
        self.target
            .retain(&HostPreparationAuthority::unmanaged())
            .unwrap();
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn final_authority_destructor_runs_after_plan_payload_and_can_reenter_another_owner() {
    let (session, _) = setup();
    let (target, _) = setup();
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(ReentrantRetirement {
        plan: Arc::downgrade(session.plan.legacy_arc().expect("legacy session source")),
        target: target.owner.clone(),
        drops: drops.clone(),
    });
    session.retain_host_preparation(&authority).unwrap();
    // A second attachment replaces custody internally, without prematurely
    // invoking the first erased token or invoking it under the owner's mutex.
    session
        .retain_host_preparation(&HostPreparationAuthority::unmanaged())
        .unwrap();
    drop(authority);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(session);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    target
        .retain_host_preparation(&HostPreparationAuthority::unmanaged())
        .unwrap();
}
