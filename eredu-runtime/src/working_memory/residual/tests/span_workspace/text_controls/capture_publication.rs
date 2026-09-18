//! Real shared source and original accepted quotes; no synthetic execution grant.
use super::*;
use crate::working_memory::{
    CapturePlanPublicationCause, CapturePlanStorageKey, PreparedCapturePlanPublication,
};
use eredu_core::{SharedStorageDomain, SharedStorageIdentity};
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key(SharedStorageIdentity);
impl CapturePlanStorageKey for Key {
    fn capture_plan_identity(&self) -> Option<&SharedStorageIdentity> {
        Some(&self.0)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OtherKey(Key);
impl CapturePlanStorageKey for OtherKey {
    fn capture_plan_identity(&self) -> Option<&SharedStorageIdentity> {
        self.0.capture_plan_identity()
    }
}
fn key(source: &SharedCapturePlan) -> Key {
    Key(source.storage_identity().clone())
}
fn publication_quote(
    pool: &WorkingMemoryPool,
    source: &SharedCapturePlan,
    existing: Option<&WorkingMemoryStorage<Key>>,
) -> IncrementalInferenceQuote {
    let q = replacement_quote(pool, geometry(), 0).into_incremental();
    let publication = PreparedCapturePlanPublication::prepare(
        pool,
        q.span_workspace().plan(),
        source,
        key(source),
        existing,
    )
    .unwrap();
    let controls = prepared(source, &q)
        .with_capture_plan_publication(publication)
        .unwrap();
    q.with_span_workspace_and_text_controls(controls).unwrap()
}
#[test]
fn publication_seal_prices_exact_c_outside_one_original_host_hold() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let earlier_source = source.clone();
    let c = source.capacity_bytes().unwrap();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    let before = raw.incremental_bytes();
    let publication = PreparedCapturePlanPublication::prepare(
        &pool,
        raw.span_workspace().plan(),
        &source,
        key(&source),
        None,
    )
    .unwrap();
    let controls = prepared(&source, &raw)
        .with_capture_plan_publication(publication)
        .unwrap();
    let earlier_controls = controls.clone();
    assert!(controls.same_binding(&earlier_controls));
    assert_eq!(controls.capture_publication_source_bytes(), c);
    assert!(controls.capture_publication_control_bytes() > 0);
    let q = raw.with_span_workspace_and_text_controls(controls).unwrap();
    let protected = q.span_workspace().protected_peak_bytes().unwrap();
    assert_eq!(q.incremental_bytes(), before + protected + c);
    let required = q.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &q, 64 + required - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    let (r, accepted) = sealed_plan(&pool, &q, 64 + required).unwrap();
    let (r, run) = r.into_funding().unwrap();
    assert!(accepted
        .clone()
        .into_funded_text_span_workspace(&run, &r)
        .is_err());
    let native = run.scope().unwrap();
    let pending = accepted
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let held = account(&pool, &r);
    assert_eq!(held.1, protected);
    assert!(q
        .clone()
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .is_err());
    assert_eq!(account(&pool, &r), held);
    let (owner, witness) = pending.publish_and_finish(&native).unwrap();
    assert_eq!(account(&pool, &r), (held.0 - c, protected, held.2));
    assert_eq!(pool.used_bytes().unwrap(), 64 + required);
    witness.validate(&pool).unwrap();
    assert!(owner
        .workspace()
        .text_controls()
        .unwrap()
        .same_binding(&earlier_controls));
    let witness_alias = witness.clone();
    let registered = pool.pin_registered_storage([(key(&source), c)]).unwrap();
    let next = publication_quote(&pool, &source, Some(&registered));
    assert_eq!(next.incremental_bytes() + c, required);
    assert!(matches!(
        sealed_plan(&pool, &next, 1_000_000),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.used_bytes().unwrap(), 64 + required);
    drop((next, registered));
    native.certify().unwrap();
    drop((owner, witness, r, run, q, earlier_controls, root, source));
    // The earlier source and escaped witness retain exactly the original P+Q+S
    // hold and registered C; unused equation/controller headroom is terminal.
    assert_eq!(pool.used_bytes().unwrap(), protected + c);
    drop(earlier_source);
    assert_eq!(pool.used_bytes().unwrap(), protected + c);
    witness_alias.validate(&pool).unwrap();
    drop(witness_alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn existing_and_raced_c_preserve_origins_without_retroactive_quote_credit() {
    for raced in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let source = capture_source();
        let c = source.capacity_bytes().unwrap();
        let before = if raced {
            None
        } else {
            Some(pool.register_storage([(key(&source), c)]).unwrap())
        };
        let q = publication_quote(&pool, &source, before.as_ref());
        assert_eq!(
            q.span_workspace()
                .text_controls()
                .unwrap()
                .capture_publication_source_bytes(),
            if raced { c } else { 0 }
        );
        let (r, run, q) = accept(&pool, q);
        let native = run.scope().unwrap();
        let race = if raced {
            Some(pool.register_storage([(key(&source), c)]).unwrap())
        } else {
            None
        };
        let pending = q
            .begin_capture_plan_publication::<Key>(&run, &r, &source)
            .unwrap();
        let held = account(&pool, &r);
        let used = pool.used_bytes().unwrap();
        let (owner, witness) = pending.publish_and_finish(&native).unwrap();
        assert_eq!(account(&pool, &r), held);
        assert_eq!(pool.used_bytes().unwrap(), used);
        witness.validate(&pool).unwrap();
        let retained = owner.protected_host_bytes() + c;
        native.certify().unwrap();
        drop((owner, witness, run, r, root, before, race));
        assert_eq!(pool.used_bytes().unwrap(), retained);
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn publication_rejects_wrong_identity_capacity_domain_and_independent_binding() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let other = capture_source();
    let c = source.capacity_bytes().unwrap();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    assert!(PreparedCapturePlanPublication::prepare(
        &pool,
        raw.span_workspace().plan(),
        &source,
        key(&other),
        None
    )
    .is_err());
    let wrong = pool.register_storage([(key(&source), c + 1)]).unwrap();
    assert!(PreparedCapturePlanPublication::prepare(
        &pool,
        raw.span_workspace().plan(),
        &source,
        key(&source),
        Some(&wrong)
    )
    .is_err());
    drop(wrong);
    let foreign = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let foreign_pin = foreign.register_storage([(key(&source), c)]).unwrap();
    assert!(PreparedCapturePlanPublication::prepare(
        &pool,
        raw.span_workspace().plan(),
        &source,
        key(&source),
        Some(&foreign_pin)
    )
    .is_err());
    let make = || {
        PreparedCapturePlanPublication::prepare(
            &pool,
            raw.span_workspace().plan(),
            &source,
            key(&source),
            None,
        )
        .unwrap()
    };
    let a = prepared(&source, &raw)
        .with_capture_plan_publication(make())
        .unwrap();
    let b = prepared(&source, &raw)
        .with_capture_plan_publication(make())
        .unwrap();
    assert!(!a.same_binding(&b));
    assert!(a.clone().with_capture_plan_publication(make()).is_err());
    let q = raw.with_span_workspace_and_text_controls(a).unwrap();
    let (r, run, q) = accept(&pool, q);
    let before = account(&pool, &r);
    assert!(q
        .clone()
        .begin_capture_plan_publication::<Key>(&run, &r, &other)
        .is_err());
    assert_eq!(account(&pool, &r), before);
    assert!(q
        .clone()
        .begin_capture_plan_publication::<OtherKey>(&run, &r, &source)
        .is_err());
    assert_eq!(account(&pool, &r), before);
    drop((q, r, run, b, root, foreign_pin));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
}
#[test]
fn wrong_original_scope_keeps_unready_plan_alias_under_one_hold() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = publication_quote(&pool, &source, None);
    let alias = q.span_workspace().plan().clone();
    let (r, run, q) = accept(&pool, q);
    let original = run.scope().unwrap();
    let (other_r, other_run, other_q) = accept(&pool, quote(&pool, &source));
    let wrong = other_run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let before = account(&pool, &r);
    let failed = pending.publish_and_finish(&wrong).unwrap_err();
    assert!(matches!(
        failed.cause(),
        CapturePlanPublicationCause::Storage(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(account(&pool, &r), before);
    assert!(alias
        .original_host()
        .unwrap()
        .validate_control_reservation(&r)
        .is_err());
    original.certify().unwrap();
    wrong.certify().unwrap();
    drop((failed, r, run, other_r, other_run, other_q, root));
    assert!(pool.used_bytes().unwrap() >= before.1);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn busy_usage_and_poisoned_attachment_preserve_pending_custody() {
    for poisoned in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let source = capture_source();
        let q = publication_quote(&pool, &source, None);
        let alias = q.span_workspace().plan().clone();
        let (r, run, q) = accept(&pool, q);
        let native = run.scope().unwrap();
        let pending = q
            .begin_capture_plan_publication::<Key>(&run, &r, &source)
            .unwrap();
        let before = account(&pool, &r);
        let failed = if poisoned {
            assert!(catch_unwind(AssertUnwindSafe(|| source
                .try_attach::<WorkingMemoryError>(
                    &SharedStorageDomain::default(),
                    || panic!("attachment sentinel")
                )))
            .is_err());
            pending.publish_and_finish(&native).unwrap_err()
        } else {
            let lock = pool.0.usage.lock().unwrap();
            let error = pending.publish_and_finish(&native).unwrap_err();
            drop(lock);
            error
        };
        assert!(match failed.cause() {
            CapturePlanPublicationCause::Busy => !poisoned,
            CapturePlanPublicationCause::Storage(WorkingMemoryError::Poisoned) => poisoned,
            _ => false,
        });
        assert_eq!(account(&pool, &r), before);
        assert!(pool
            .pin_registered_storage([(key(&source), source.capacity_bytes().unwrap())])
            .is_err());
        native.certify().unwrap();
        drop((failed, r, run, root, source));
        assert!(pool.used_bytes().unwrap() >= before.1);
        drop(alias);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn busy_source_attachment_does_not_wait_or_consume_another_provider() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let (r, run, q) = accept(&pool, publication_quote(&pool, &source, None));
    let native = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    std::thread::scope(|threads| {
        let (entered, wait) = std::sync::mpsc::sync_channel(0);
        let (release, finish) = std::sync::mpsc::sync_channel(0);
        let source_ref = &source;
        let worker = threads.spawn(move || {
            source_ref
                .try_attach::<WorkingMemoryError>(&SharedStorageDomain::default(), || {
                    entered.send(()).unwrap();
                    finish
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    Ok(Box::new(()))
                })
                .unwrap()
        });
        wait.recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let failed = pending.publish_and_finish(&native).unwrap_err();
        release.send(()).unwrap();
        assert!(worker.join().unwrap());
        assert!(matches!(failed.cause(), CapturePlanPublicationCause::Busy));
        drop(failed);
    });
    native.certify().unwrap();
    drop((source, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn late_original_source_quarantine_rejects_before_c_and_does_not_refund_aliases() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let (old_r, old_run, old_q) = accept(&pool, quote(&pool, &source));
    let abandoned = old_run.scope().unwrap();
    let mut retained = abandoned.adopt_storage_individually([(51u32, 7)]).unwrap();
    let pin = retained.remove(&51).unwrap();
    let q = publication_quote(&pool, &source, None)
        .with_registered_sources(pin.clone())
        .unwrap();
    let alias = q.span_workspace().plan().clone();
    let (r, run, q) = accept(&pool, q);
    let native = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let before = account(&pool, &r);
    drop(abandoned);
    let failed = pending.publish_and_finish(&native).unwrap_err();
    assert!(matches!(
        failed.cause(),
        CapturePlanPublicationCause::Storage(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(account(&pool, &r), before);
    assert!(pool
        .pin_registered_storage([(key(&source), source.capacity_bytes().unwrap())])
        .is_err());
    native.certify().unwrap();
    drop((failed, r, run, source, root));
    assert!(pool.used_bytes().unwrap() >= before.1);
    drop((alias, pin, retained, old_r, old_run, old_q));
    assert!(
        pool.used_bytes().unwrap() > 0,
        "real abandoned origin stays quarantined"
    );
}
#[test]
fn existing_c_late_quarantine_is_checked_at_original_reservation_and_publication() {
    for after_reserve in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let source = capture_source();
        let c = source.capacity_bytes().unwrap();
        let (old_r, old_run, old_q) = accept(&pool, quote(&pool, &source));
        let abandoned = old_run.scope().unwrap();
        let mut storage = abandoned
            .adopt_storage_individually([(key(&source), c)])
            .unwrap();
        let pin = storage.remove(&key(&source)).unwrap();
        let q = publication_quote(&pool, &source, Some(&pin));
        if after_reserve {
            let (r, run, q) = accept(&pool, q);
            let native = run.scope().unwrap();
            let pending = q
                .begin_capture_plan_publication::<Key>(&run, &r, &source)
                .unwrap();
            drop(abandoned);
            let failed = pending.publish_and_finish(&native).unwrap_err();
            assert!(matches!(
                failed.cause(),
                CapturePlanPublicationCause::Storage(WorkingMemoryError::ExecutionFenced)
            ));
            native.certify().unwrap();
            drop((failed, r, run));
        } else {
            drop(abandoned);
            assert!(matches!(
                sealed_plan(&pool, &q, 1_000_000),
                Err(PrefillPlanningError::Reservation(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
        }
        drop((old_r, old_run, old_q, source, root, pin, storage));
        assert!(pool.used_bytes().unwrap() > 0);
    }
}

#[test]
fn exact_typed_reuse_and_typed_race_keep_the_original_source_owner() {
    for raced in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let root = pool.register_storage([(1u32, 64)]).unwrap();
        let source = capture_source();
        let earlier = source.clone();
        let c = source.capacity_bytes().unwrap();
        let cold = if raced {
            Some(publication_quote(&pool, &source, None))
        } else {
            None
        };
        let (first_r, first_run, first_q) = accept(&pool, publication_quote(&pool, &source, None));
        let first_scope = first_run.scope().unwrap();
        let (first_owner, first_witness) = first_q
            .begin_capture_plan_publication::<Key>(&first_run, &first_r, &source)
            .unwrap()
            .publish_and_finish(&first_scope)
            .unwrap();
        let first_retained = first_owner.protected_host_bytes() + c;
        let pin = pool.pin_registered_storage([(key(&source), c)]).unwrap();
        let q = cold.unwrap_or_else(|| publication_quote(&pool, &source, Some(&pin)));
        assert_eq!(
            q.span_workspace()
                .text_controls()
                .unwrap()
                .capture_publication_source_bytes(),
            if raced { c } else { 0 }
        );
        let (r, run, q) = accept(&pool, q);
        let scope = run.scope().unwrap();
        let pending = q
            .begin_capture_plan_publication::<Key>(&run, &r, &source)
            .unwrap();
        let before = account(&pool, &r);
        let (owner, witness) = pending.publish_and_finish(&scope).unwrap();
        assert_eq!(account(&pool, &r), before);
        witness.validate(&pool).unwrap();
        first_scope.certify().unwrap();
        scope.certify().unwrap();
        drop((
            first_owner,
            first_witness,
            first_run,
            first_r,
            owner,
            witness,
            run,
            r,
            pin,
            root,
            source,
        ));
        assert_eq!(pool.used_bytes().unwrap(), first_retained);
        assert_eq!(earlier.capacity_bytes(), Some(c));
        drop(earlier);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn opaque_or_other_typed_attachment_never_substitutes_for_exact_source_custody() {
    for typed in [false, true] {
        for registered in [false, true] {
            let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
            let root = pool.register_storage([(1u32, 64)]).unwrap();
            let source = capture_source();
            let c = source.capacity_bytes().unwrap();
            let pin = registered.then(|| pool.register_storage([(key(&source), c)]).unwrap());
            if typed {
                source
                    .try_attach_typed_nonblocking(pool.shared_storage_domain(), || {
                        Ok::<_, WorkingMemoryError>(Arc::new(7u32))
                    })
                    .unwrap();
            } else {
                source
                    .try_attach(pool.shared_storage_domain(), || {
                        Ok::<_, WorkingMemoryError>(Box::new(()))
                    })
                    .unwrap();
            }
            let (r, run, q) = accept(&pool, publication_quote(&pool, &source, pin.as_ref()));
            let scope = run.scope().unwrap();
            let pending = q
                .begin_capture_plan_publication::<Key>(&run, &r, &source)
                .unwrap();
            let before = account(&pool, &r);
            let used = pool.used_bytes().unwrap();
            let failed = pending.publish_and_finish(&scope).unwrap_err();
            assert!(matches!(
                failed.cause(),
                CapturePlanPublicationCause::AttachmentMismatch
            ));
            assert_eq!(account(&pool, &r), before);
            assert_eq!(pool.used_bytes().unwrap(), used);
            assert_eq!(
                pool.pin_registered_storage([(key(&source), c)]).is_ok(),
                registered
            );
            // Old opaque API compatibility: an existing typed or opaque domain is
            // still false, without invoking the legacy provider.
            assert!(!source
                .try_attach::<WorkingMemoryError>(pool.shared_storage_domain(), || panic!(
                    "legacy provider must stay lazy"
                ))
                .unwrap());
            scope.certify().unwrap();
            drop((failed, source, r, run, root, pin));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn typed_owner_attached_to_another_physical_source_is_rejected_even_with_colliding_keys() {
    use crate::working_memory::storage::capture_publication::PublishedCaptureStorage;
    #[derive(Clone, Debug)]
    struct Colliding(SharedStorageIdentity);
    impl PartialEq for Colliding {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }
    impl Eq for Colliding {}
    impl PartialOrd for Colliding {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Colliding {
        fn cmp(&self, _: &Self) -> std::cmp::Ordering {
            std::cmp::Ordering::Equal
        }
    }
    impl CapturePlanStorageKey for Colliding {
        fn capture_plan_identity(&self) -> Option<&SharedStorageIdentity> {
            Some(&self.0)
        }
    }
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let first = capture_source();
    let other = SharedCapturePlan::new(first.admission().clone());
    let make = |source: &SharedCapturePlan| {
        let q = replacement_quote(&pool, geometry(), 0).into_incremental();
        let p = PreparedCapturePlanPublication::prepare(
            &pool,
            q.span_workspace().plan(),
            source,
            Colliding(source.storage_identity().clone()),
            None,
        )
        .unwrap();
        let c = prepared(source, &q)
            .with_capture_plan_publication(p)
            .unwrap();
        q.with_span_workspace_and_text_controls(c).unwrap()
    };
    let (a, arun, aq) = accept(&pool, make(&first));
    let ascope = arun.scope().unwrap();
    let (aowner, awitness) = aq
        .begin_capture_plan_publication::<Colliding>(&arun, &a, &first)
        .unwrap()
        .publish_and_finish(&ascope)
        .unwrap();
    let actual = first
        .try_attach_owned_nonblocking::<PublishedCaptureStorage<Colliding>, WorkingMemoryError>(
            pool.shared_storage_domain(),
            || panic!("already typed"),
        )
        .unwrap();
    other
        .try_attach_owned_nonblocking(pool.shared_storage_domain(), || {
            Ok::<_, WorkingMemoryError>(actual)
        })
        .unwrap();
    let (r, run, q) = accept(&pool, make(&other));
    let scope = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Colliding>(&run, &r, &other)
        .unwrap();
    let before = account(&pool, &r);
    let failed = pending.publish_and_finish(&scope).unwrap_err();
    assert!(matches!(
        failed.cause(),
        CapturePlanPublicationCause::Storage(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(account(&pool, &r), before);
    ascope.certify().unwrap();
    scope.certify().unwrap();
    drop((
        aowner, awitness, a, arun, r, run, failed, first, other, root,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn provider_comparison_unwind_drops_staged_keys_after_both_locks_release() {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Weak,
    };
    struct Probe {
        pool: Weak<crate::working_memory::Pool>,
        source: Weak<SharedCapturePlan>,
        panic: AtomicBool,
        checking: AtomicBool,
        locked: AtomicBool,
        drops: AtomicUsize,
    }
    #[derive(Clone)]
    struct Probed {
        id: SharedStorageIdentity,
        probe: Arc<Probe>,
    }
    impl std::fmt::Debug for Probed {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.id.fmt(f)
        }
    }
    impl PartialEq for Probed {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other).is_eq()
        }
    }
    impl Eq for Probed {}
    impl PartialOrd for Probed {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probed {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if self.probe.panic.swap(false, Ordering::SeqCst) {
                panic!("registry comparison sentinel")
            }
            self.id.cmp(&other.id)
        }
    }
    impl CapturePlanStorageKey for Probed {
        fn capture_plan_identity(&self) -> Option<&SharedStorageIdentity> {
            Some(&self.id)
        }
    }
    impl Drop for Probed {
        fn drop(&mut self) {
            if !self.probe.checking.load(Ordering::SeqCst) {
                return;
            }
            self.probe.drops.fetch_add(1, Ordering::SeqCst);
            if let Some(pool) = self.probe.pool.upgrade() {
                if matches!(
                    pool.usage.try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ) {
                    self.probe.locked.store(true, Ordering::SeqCst)
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
                    self.probe.locked.store(true, Ordering::SeqCst)
                }
            }
        }
    }
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = Arc::new(capture_source());
    let other = capture_source();
    let probe = Arc::new(Probe {
        pool: Arc::downgrade(&pool.0),
        source: Arc::downgrade(&source),
        panic: AtomicBool::new(false),
        checking: AtomicBool::new(false),
        locked: AtomicBool::new(false),
        drops: AtomicUsize::new(0),
    });
    let other_pin = pool
        .register_storage([(
            Probed {
                id: other.storage_identity().clone(),
                probe: probe.clone(),
            },
            other.capacity_bytes().unwrap(),
        )])
        .unwrap();
    let q = replacement_quote(&pool, geometry(), 0).into_incremental();
    let publication = PreparedCapturePlanPublication::prepare(
        &pool,
        q.span_workspace().plan(),
        &source,
        Probed {
            id: source.storage_identity().clone(),
            probe: probe.clone(),
        },
        None,
    )
    .unwrap();
    let c = prepared(&source, &q)
        .with_capture_plan_publication(publication)
        .unwrap();
    let q = q.with_span_workspace_and_text_controls(c).unwrap();
    let alias = q.span_workspace().plan().clone();
    let (r, run, q) = accept(&pool, q);
    let native = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Probed>(&run, &r, &source)
        .unwrap();
    probe.checking.store(true, Ordering::SeqCst);
    probe.panic.store(true, Ordering::SeqCst);
    let error = catch_unwind(AssertUnwindSafe(|| pending.publish_and_finish(&native))).unwrap_err();
    assert_eq!(
        error.downcast_ref::<&str>(),
        Some(&"registry comparison sentinel")
    );
    assert!(probe.drops.load(Ordering::SeqCst) > 0);
    assert!(!probe.locked.load(Ordering::SeqCst));
    probe.checking.store(false, Ordering::SeqCst);
    assert!(matches!(
        pool.used_bytes(),
        Err(WorkingMemoryError::Poisoned)
    ));
    // Poisoned accounting is not repaired or certified by this fixture.
    drop((native, r, run, alias, source, other, root, other_pin));
}

#[test]
fn intervening_original_scope_spend_cannot_consume_protected_controls_or_partially_publish_c() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let c = source.capacity_bytes().unwrap();
    let (r, run, q) = accept(&pool, publication_quote(&pool, &source, None));
    let scope = run.scope().unwrap();
    let pending = q
        .begin_capture_plan_publication::<Key>(&run, &r, &source)
        .unwrap();
    let state = account(&pool, &r);
    let pressure = state.0 - state.1 - c + 1;
    let payload: Arc<[u8]> = vec![19; usize::try_from(pressure).unwrap()].into();
    let registered = scope
        .adopt_storage_individually([(99u64, payload.len() as u64)])
        .unwrap();
    let before = account(&pool, &r);
    let used = pool.used_bytes().unwrap();
    let failed = pending.publish_and_finish(&scope).unwrap_err();
    assert!(
        matches!(failed.cause(),CapturePlanPublicationCause::Storage(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if *required_bytes==c && *available_bytes+1==c)
    );
    assert_eq!(account(&pool, &r), before);
    assert_eq!(pool.used_bytes().unwrap(), used);
    assert!(pool.pin_registered_storage([(key(&source), c)]).is_err());
    drop(payload);
    drop(registered);
    scope.certify().unwrap();
    drop((failed, r, run, source, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn ordinary_no_publication_promotion_and_source_witness_keep_existing_behavior() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source)
        .with_registered_sources(root.clone())
        .unwrap();
    assert_eq!(
        q.span_workspace()
            .text_controls()
            .unwrap()
            .capture_publication_control_bytes(),
        0
    );
    assert_eq!(
        q.span_workspace()
            .text_controls()
            .unwrap()
            .capture_publication_source_bytes(),
        0
    );
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    let (r, run, q) = accept(&pool, q);
    let scope = run.scope().unwrap();
    let (owner, witness) = q.into_funded_text_span_workspace(&run, &r).unwrap();
    let witness = witness.unwrap();
    assert_eq!(
        owner.protected_host_bytes(),
        p + facts().total_bytes().unwrap().unwrap()
    );
    witness.validate(&pool).unwrap();
    scope.certify().unwrap();
    drop((owner, r, run, source, root));
    assert_eq!(pool.used_bytes().unwrap(), 64);
    witness.validate(&pool).unwrap();
    drop(witness);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[path = "capture_publication/bounded_publications.rs"]
mod bounded_publications;

#[path = "capture_publication/retirement.rs"]
mod retirement;
