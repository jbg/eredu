use super::*;
use crate::working_memory::{WorkingMemoryError, WorkingMemoryPool};
use eredu_core::{capture::SharedCapturePlan, HostPreparationAuthority};

fn shared() -> SharedCapturePlan {
    SharedCapturePlan::new(plan_for(CaptureTransform::Slice, false))
}
fn attach(source: &SharedCapturePlan, pool: &WorkingMemoryPool) {
    let key = source.storage_identity().clone();
    let bytes = source.capacity_bytes().unwrap();
    assert!(source
        .try_attach(pool.shared_storage_domain(), || {
            let registration = pool.register_storage([(key, bytes)])?;
            Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(registration))
        })
        .unwrap());
}
fn configured_shared(source: SharedCapturePlan) -> CaptureSession {
    let mut session = CaptureSession::new(source);
    session
        .configure_partition_capture(
            PartitionCaptureIdentity::new(
                "artifact-exact".into(),
                "retained-execution".into(),
                "coordinated-run".into(),
                Some("effective-overlay".into()),
                1,
            )
            .unwrap(),
        )
        .unwrap();
    session
}
fn limits() -> PartitionCaptureReceiptLimits {
    PartitionCaptureReceiptLimits {
        max_producers: 8,
        max_fragments: 32,
        max_record_bytes: 16384,
    }
}

#[test]
fn sessions_keep_actual_registered_plan_buffers_and_late_attachment_across_aliases() {
    let source = shared();
    let pointer = source.admission().plan().selections.as_ptr();
    let points = source.admission().points().as_ptr();
    let bytes = source.capacity_bytes().unwrap();
    assert!(bytes > 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let first = CaptureSession::new(source.clone());
    let second = CaptureSession::new(source.clone());
    assert_eq!(first.plan().plan().selections.as_ptr(), pointer);
    assert_eq!(first.plan().points().as_ptr(), points);
    assert!(first.shared_plan_source().same_storage(&source));
    attach(&source, &pool); // Both sessions predate source publication.
    let alias = first.shared_plan_source().clone();
    drop((source, first, second));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(alias.admission().plan().selections.as_ptr(), pointer);
    assert_eq!(alias.admission().points().as_ptr(), points);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn equal_sources_are_independent_and_two_domains_follow_actual_alias_lifetime() {
    let a = shared();
    let b = SharedCapturePlan::new(a.admission().clone());
    assert_eq!(a.admission().identity(), b.admission().identity());
    assert!(!a.same_storage(&b));
    let a_bytes = a.capacity_bytes().unwrap();
    let b_bytes = b.capacity_bytes().unwrap();
    let first = WorkingMemoryPool::new(a_bytes + b_bytes, 0).unwrap();
    let second = WorkingMemoryPool::new(a_bytes, 0).unwrap();
    attach(&a, &first);
    attach(&b, &first);
    attach(&a, &second);
    let session = CaptureSession::new(a.clone());
    let other = CaptureSession::new(b.clone());
    drop((a, b));
    assert_eq!(first.used_bytes().unwrap(), a_bytes + b_bytes);
    drop(other);
    assert_eq!(first.used_bytes().unwrap(), a_bytes);
    assert_eq!(second.used_bytes().unwrap(), a_bytes);
    drop(session);
    assert_eq!(first.used_bytes().unwrap(), 0);
    assert_eq!(second.used_bytes().unwrap(), 0);
}

#[test]
fn dense_sum_and_routed_receipts_retain_the_original_shared_source() {
    for sum in [false, true] {
        let source = shared();
        let bytes = source.capacity_bytes().unwrap();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let mut ledger = CaptureLedger::new(source.admission());
        let receipt = if sum {
            PartitionCaptureReceiptPlan::new_sum(
                source.clone(),
                context(source.admission()),
                producers(source.admission(), 1),
                1,
                limits(),
                &mut ledger,
            )
            .unwrap()
        } else {
            PartitionCaptureReceiptPlan::new(
                source.clone(),
                context(source.admission()),
                producers(source.admission(), 1),
                1,
                limits(),
                &mut ledger,
            )
            .unwrap()
        };
        assert!(receipt.shared_plan_source().same_storage(&source));
        assert_eq!(
            receipt
                .shared_plan_source()
                .admission()
                .points()
                .as_ptr(),
            source.admission().points().as_ptr()
        );
        attach(&source, &pool);
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        drop(receipt);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    let source = SharedCapturePlan::new(super::super::super::routed::plan(false));
    let bytes = source.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let mut ledger = CaptureLedger::new(source.admission());
    let producers = super::super::super::routed::producers(source.admission(), false);
    let world = producers.len();
    let receipt = PartitionCaptureReceiptPlan::new_routed(
        source.clone(),
        context(source.admission()),
        producers,
        world,
        limits(),
        &mut ledger,
    )
    .unwrap();
    assert!(receipt.shared_plan_source().same_storage(&source));
    attach(&source, &pool);
    let delivery = receipt.into_delivery();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(delivery);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

struct RetiredAfterSource {
    pool: WorkingMemoryPool,
    drops: Arc<AtomicUsize>,
}
impl Drop for RetiredAfterSource {
    fn drop(&mut self) {
        assert_eq!(
            self.pool.used_bytes().unwrap(),
            0,
            "all plan source aliases retire before final preparation custody"
        );
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn live_partition_work_outlives_session_and_retires_source_before_preparation_authority() {
    let source = shared();
    let bytes = source.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let mut session = configured_shared(source.clone());
    let transport = transport(world(1), 0, Fault::None);
    session
        .prepare_step_transaction(DistributedCommitEpoch::FIRST, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let work = prepare(&mut session, &transport).unwrap();
    assert!(work.shared_plan_source().same_storage(&source));
    attach(&source, &pool); // Work and its nested receipt predate attachment.
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(RetiredAfterSource {
        pool: pool.clone(),
        drops: drops.clone(),
    });
    session.retain_host_preparation(&authority).unwrap();
    drop((source, session, authority));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(work.global_reserved().host_bytes > 0);
    drop(work);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn independent_and_shared_sources_produce_equal_partition_results_and_usage() {
    let source = shared();
    let mut results = Vec::new();
    for use_shared in [false, true] {
        let mut session = if use_shared {
            configured_shared(source.clone())
        } else {
            configured(source.admission().clone(), 1)
        };
        let transport = transport(world(1), 0, Fault::None);
        let epoch = DistributedCommitEpoch::FIRST;
        session
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        let mut work = prepare(&mut session, &transport).unwrap();
        assert_eq!(work.shared_plan_source().same_storage(&source), use_shared);
        let coordination = session.prepare_partition_coordination(&transport).unwrap();
        session.coordinate_partition_capture(coordination).unwrap();
        let mut backend = Backend::default();
        session
            .observe_partition(&mut work, &mut backend, &global())
            .unwrap();
        assert!(backend.transforms > 0);
        session.complete_partition_capture(work).unwrap();
        session.complete_transaction(epoch).unwrap();
        session.finish_transaction(epoch, true);
        let step = session.take_step().unwrap();
        assert!(step.records[0].payload.is_some());
        results.push(step);
    }
    assert_eq!(results[0].records, results[1].records);
    assert_eq!(results[0].partitions, results[1].partitions);
    assert_eq!(results[0].step_usage, results[1].step_usage);
    assert_eq!(results[0].cumulative_usage, results[1].cumulative_usage);
    assert_eq!(results[0].outcome, results[1].outcome);
    let mut ordinary = CaptureSession::new(source);
    ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
    ordinary
        .observe(&mut Backend::default(), "block.output", &global())
        .unwrap();
    assert_eq!(
        results[0].records[0].payload,
        ordinary.take_step().unwrap().records[0].payload
    );
}

#[test]
fn checkpoints_copy_destination_payload_but_keep_original_source_and_custody_separate() {
    let source = shared();
    let bytes = source.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let session = CaptureSession::new(source.clone());
    let catalog = discovery(source.admission());
    let saved = session.checkpoint(&catalog).unwrap();
    let copy = saved.clone();
    assert!(saved.shared_plan_source().same_storage(&source));
    assert!(copy.shared_plan_source().same_storage(&source));
    let original = source.admission().plan().selections.as_ptr();
    let first = saved.copied_plan_for_test().plan().selections.as_ptr();
    let second = copy.copied_plan_for_test().plan().selections.as_ptr();
    assert_ne!(original, first);
    assert_ne!(first, second);
    assert_eq!(
        saved.logical_storage_bytes(),
        session.checkpoint_storage_bytes(&catalog)
    );
    assert!(
        saved.logical_storage_bytes().unwrap() > std::mem::size_of::<CaptureCheckpoint>() as u64
    );
    attach(&source, &pool);
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(RetiredAfterSource {
        pool: pool.clone(),
        drops: drops.clone(),
    });
    session.retain_host_preparation(&authority).unwrap();
    drop((source, session, saved, authority));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(
        copy.copied_plan_for_test().plan().selections.as_ptr(),
        second
    );
    drop(copy);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn child_readmission_is_independent_and_failed_preflight_does_not_publish_an_alias() {
    let source = shared();
    let bytes = source.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    attach(&source, &pool);
    let mut parent = CaptureSession::new(source.clone());
    let catalog = discovery(source.admission());
    let saved = parent.checkpoint(&catalog).unwrap();
    let fork_request = || CaptureForkRequest {
        discovery: &catalog,
        max_predictions: source.admission().request().max_predictions,
        limits: source.admission().plan().limits.clone(),
        intervention: None,
    };
    let before = pool.used_bytes().unwrap();
    assert!(saved
        .fork(fork_request(), |_, _, _| Err(CaptureError::Unsupported(
            "injected preflight".into()
        )))
        .is_err());
    assert_eq!(pool.used_bytes().unwrap(), before);
    let mut child = saved
        .fork(fork_request(), |_, _, _| Ok(CaptureUsage::default()))
        .unwrap();
    assert!(!child.shared_plan_source().same_storage(&source));
    assert_ne!(
        child.plan().plan().selections.as_ptr(),
        source.admission().plan().selections.as_ptr()
    );
    assert!(child.restore(&saved).is_err());
    parent.restore(&saved).unwrap();
    drop((parent, saved, source));
    assert_eq!(
        pool.used_bytes().unwrap(),
        0,
        "the child owns a distinct admission rather than the original source charge"
    );
    child.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(child.take_step().is_some());
}

#[test]
fn rejected_shared_receipt_and_checkpoint_keep_original_source_usable() {
    let source = shared();
    let bytes = source.capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    attach(&source, &pool);
    let mut ledger = CaptureLedger::new(source.admission());
    let mut wrong = context(source.admission());
    wrong.capture_plan_identity = "foreign-admission".into();
    assert!(PartitionCaptureReceiptPlan::new(
        source.clone(),
        wrong,
        producers(source.admission(), 1),
        1,
        limits(),
        &mut ledger,
    )
    .is_err());
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let session = CaptureSession::new(source.clone());
    let mut wrong = discovery(source.admission());
    wrong.catalog.points[0].meaning.push_str(" changed");
    assert!(session.checkpoint(&wrong).is_err());
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert!(session.checkpoint(&discovery(source.admission())).is_ok());
    assert!(session.shared_plan_source().same_storage(&source));
    drop((session, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
