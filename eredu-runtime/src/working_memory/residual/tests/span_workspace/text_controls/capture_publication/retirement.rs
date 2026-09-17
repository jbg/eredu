use super::*;
use crate::working_memory::storage::capture_publication::{
    CaptureSourceOwner, PublishedCaptureStorage,
};
use eredu_core::{SharedStorageOwner, SharedStorageRetirement};

#[test]
fn actual_published_c_mixed_typed_witness_and_source_owners_retire_concurrently() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let c = source.capacity_bytes().unwrap();
    let (r, run, q) = accept(&pool, publication_quote(&pool, &source, None));
    let native = run.scope().unwrap();
    let (owner, witness) = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap()
        .publish_and_finish(&native)
        .unwrap();
    let protected = owner.protected_host_bytes();
    let typed = source
        .try_attach_owned_nonblocking::<PublishedCaptureStorage<Key>, WorkingMemoryError>(
            pool.shared_storage_domain(),
            || panic!("actual published owner exists"),
        )
        .unwrap();
    let erased = CaptureSourceOwner::new(typed.clone());
    witness.validate(&pool).unwrap();
    native.certify().unwrap();
    drop((owner, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected + c);
    std::thread::scope(|scope| {
        let pool_ref = &pool;
        let a = scope.spawn(move || {
            witness.validate(pool_ref).unwrap();
            drop(witness);
        });
        let b = scope.spawn(move || {
            assert_eq!(source.capacity_bytes(), Some(c));
            drop(source);
        });
        let d = scope.spawn(move || drop(typed));
        a.join().unwrap();
        b.join().unwrap();
        d.join().unwrap();
    });
    assert_eq!(
        pool.used_bytes().unwrap(),
        protected + c,
        "escaped erased owner remains"
    );
    {
        let usage = pool.0.usage.lock().unwrap();
        erased.validate(&pool, &usage).unwrap();
    }
    drop(erased);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn actual_c_final_typed_and_erased_exit_race_has_one_original_retirement() {
    for _ in 0..4 {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let source = capture_source();
        let c = source.capacity_bytes().unwrap();
        let (r, run, q) = accept(&pool, publication_quote(&pool, &source, None));
        let native = run.scope().unwrap();
        let (owner, witness) = q
            .begin_capture_plan_publication::<Key>(&run, &r, &source)
            .unwrap()
            .publish_and_finish(&native)
            .unwrap();
        let protected = owner.protected_host_bytes();
        let typed = source
            .try_attach_owned_nonblocking::<PublishedCaptureStorage<Key>, WorkingMemoryError>(
                pool.shared_storage_domain(),
                || panic!("existing C"),
            )
            .unwrap();
        native.certify().unwrap();
        drop((owner, r, run, source, root));
        assert_eq!(pool.used_bytes().unwrap(), protected + c);
        let barrier = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            let a = scope.spawn(|| {
                barrier.wait();
                drop(typed);
            });
            let b = scope.spawn(|| {
                barrier.wait();
                drop(witness);
            });
            barrier.wait();
            a.join().unwrap();
            b.join().unwrap();
        });
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

struct ForeignOwned;
impl SharedStorageRetirement for ForeignOwned {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[test]
fn actual_original_publication_rejects_wrong_owned_type_without_partial_debit() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let c = source.capacity_bytes().unwrap();
    source
        .try_attach_owned_nonblocking(pool.shared_storage_domain(), || {
            Ok::<_, WorkingMemoryError>(SharedStorageOwner::new(ForeignOwned))
        })
        .unwrap();
    let (r, run, q) = accept(&pool, publication_quote(&pool, &source, None));
    let native = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let before = account(&pool, &r);
    let failure = pending.publish_and_finish(&native).unwrap_err();
    assert!(matches!(
        failure.cause(),
        CapturePlanPublicationCause::AttachmentMismatch
    ));
    assert_eq!(account(&pool, &r), before);
    assert!(pool.pin_registered_storage([(key(&source), c)]).is_err());
    native.certify().unwrap();
    drop((source, r, run, root));
    assert_eq!(
        pool.used_bytes().unwrap(),
        before.1 + 64,
        "failed pending retains its original registered root"
    );
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn actual_owned_c_rejects_raw_lookup_without_opening_an_ordinary_arc_exit() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let c = source.capacity_bytes().unwrap();
    let (r, run, q) = accept(&pool, publication_quote(&pool, &source, None));
    let native = run.scope().unwrap();
    let (owner, witness) = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap()
        .publish_and_finish(&native)
        .unwrap();
    let protected = owner.protected_host_bytes();
    assert!(matches!(
        source.try_attach_typed_nonblocking::<PublishedCaptureStorage<Key>, WorkingMemoryError>(
            pool.shared_storage_domain(),
            || panic!("owned type cannot become raw")
        ),
        Err(eredu_core::SharedStorageAttachmentError::AttachmentMismatch)
    ));
    assert!(!source
        .try_attach::<WorkingMemoryError>(pool.shared_storage_domain(), || panic!(
            "legacy provider remains lazy"
        ))
        .unwrap());
    native.certify().unwrap();
    drop((owner, witness, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected + c);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn actual_post_c_source_failure_preserves_published_prefix_and_earlier_source_alias() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let earlier = source.clone();
    let c = source.capacity_bytes().unwrap();
    let (old_r, old_run, old_q) = accept(&pool, quote(&pool, &source));
    let original_source_envelope = old_r.bytes();
    let abandoned = old_run.scope().unwrap();
    let mut retained = abandoned.adopt_storage_individually([(51u32, 7)]).unwrap();
    let pin = retained.remove(&51).unwrap();
    let q = publication_quote(&pool, &source, None)
        .with_registered_sources(pin.clone())
        .unwrap();
    let (r, run, q) = accept(&pool, q);
    let native = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let before = account(&pool, &r);
    let reset = pending.quarantine_source_after_publication(abandoned);
    let failed = pending.publish_and_finish(&native).unwrap_err();
    drop(reset);
    assert!(matches!(
        failed.cause(),
        CapturePlanPublicationCause::Storage(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(account(&pool, &r), (before.0 - c, before.1, before.2));
    let registered_c = pool.pin_registered_storage([(key(&source), c)]).unwrap();
    drop(registered_c);
    native.certify().unwrap();
    drop((r, run, old_r, old_run, old_q, root, pin, retained, source));
    assert_eq!(
        pool.used_bytes().unwrap(),
        original_source_envelope + 64 + before.1 + c
    );
    drop(failed);
    assert_eq!(
        pool.used_bytes().unwrap(),
        original_source_envelope + 64 + before.1 + c
    );
    assert_eq!(earlier.capacity_bytes(), Some(c));
    drop(earlier);
    assert_eq!(pool.used_bytes().unwrap(), original_source_envelope + 64);
}

#[test]
fn actual_c_key_clone_unwind_retires_constructed_prefix_outside_both_locks() {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Weak,
    };
    struct Probe {
        pool: Weak<crate::working_memory::Pool>,
        source: Weak<SharedCapturePlan>,
        armed: AtomicBool,
        clones: AtomicUsize,
        drops: AtomicUsize,
        locked: AtomicBool,
    }
    struct ClonedKey {
        id: SharedStorageIdentity,
        probe: Arc<Probe>,
    }
    impl Clone for ClonedKey {
        fn clone(&self) -> Self {
            if self.probe.armed.load(Ordering::SeqCst)
                && self.probe.clones.fetch_add(1, Ordering::SeqCst) == 1
            {
                std::panic::panic_any(57_u64);
            }
            Self {
                id: self.id.clone(),
                probe: self.probe.clone(),
            }
        }
    }
    impl std::fmt::Debug for ClonedKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.id.fmt(f)
        }
    }
    impl PartialEq for ClonedKey {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }
    impl Eq for ClonedKey {}
    impl PartialOrd for ClonedKey {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for ClonedKey {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.id.cmp(&other.id)
        }
    }
    impl CapturePlanStorageKey for ClonedKey {
        fn capture_plan_identity(&self) -> Option<&SharedStorageIdentity> {
            Some(&self.id)
        }
    }
    impl Drop for ClonedKey {
        fn drop(&mut self) {
            if !self.probe.armed.load(Ordering::SeqCst) {
                return;
            }
            self.probe.drops.fetch_add(1, Ordering::SeqCst);
            if let Some(pool) = self.probe.pool.upgrade() {
                if matches!(
                    pool.usage.try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ) {
                    self.probe.locked.store(true, Ordering::SeqCst);
                }
            }
            if let Some(source) = self.probe.source.upgrade() {
                if matches!(
                    source.try_attach_nonblocking::<WorkingMemoryError>(
                        &SharedStorageDomain::default(),
                        || Ok(Box::new(()))
                    ),
                    Err(eredu_core::SharedStorageAttachmentError::Busy)
                ) {
                    self.probe.locked.store(true, Ordering::SeqCst);
                }
            }
        }
    }
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = Arc::new(capture_source());
    let probe = Arc::new(Probe {
        pool: Arc::downgrade(&pool.0),
        source: Arc::downgrade(&source),
        armed: AtomicBool::new(false),
        clones: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        locked: AtomicBool::new(false),
    });
    let q = replacement_quote(&pool, geometry(), 0).into_incremental();
    let publication = PreparedCapturePlanPublication::prepare(
        &pool,
        q.span_workspace().plan(),
        &source,
        ClonedKey {
            id: source.storage_identity().clone(),
            probe: probe.clone(),
        },
        None,
    )
    .unwrap();
    let controls = prepared(&source, &q)
        .with_capture_plan_publication(publication)
        .unwrap();
    let q = q.with_span_workspace_and_text_controls(controls).unwrap();
    let protected = q.span_workspace().protected_peak_bytes().unwrap();
    let alias = q.span_workspace().plan().clone();
    let (r, run, q) = accept(&pool, q);
    let native = run.scope().unwrap();
    probe.armed.store(true, Ordering::SeqCst);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        q.begin_capture_plan_publication::<ClonedKey>(&run, &r, &source)
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<u64>(), Some(&57));
    assert_eq!(probe.clones.load(Ordering::SeqCst), 2);
    assert!(
        probe.drops.load(Ordering::SeqCst) > 0,
        "first staged key is destroyed"
    );
    assert!(!probe.locked.load(Ordering::SeqCst));
    assert_eq!(account(&pool, &r).1, protected);
    native.certify().unwrap();
    drop((r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), protected);
    drop(alias);
    assert!(!probe.locked.load(Ordering::SeqCst));
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
