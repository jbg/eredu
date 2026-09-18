use super::*;
use std::{
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    error::Error,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Barrier, Weak,
    },
};

struct Charge {
    used: Arc<AtomicUsize>,
    bytes: usize,
    // Real registry keys retain identity metadata, not the filter owner.
    _identity: SharedStorageIdentity,
    _domain: SharedStorageDomain,
}
impl Charge {
    fn acquire(
        filter: &SharedTokenFilter,
        domain: &SharedStorageDomain,
        used: &Arc<AtomicUsize>,
    ) -> Box<dyn Send + Sync> {
        let bytes = filter.capacity_bytes().unwrap() as usize;
        used.fetch_add(bytes, Ordering::SeqCst);
        Box::new(Self {
            used: used.clone(),
            bytes,
            _identity: filter.identity().clone(),
            _domain: domain.clone(),
        })
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

fn filter() -> SharedTokenFilter {
    let mut mask = Vec::with_capacity(37);
    mask.extend_from_slice(&[true, false, true]);
    SharedTokenFilter::new(TokenFilter::allowed(mask).unwrap())
}

#[test]
fn immutable_aliases_preserve_spare_capacity_and_compare_by_value() {
    let mut mask = Vec::with_capacity(37);
    mask.extend_from_slice(&[true, false, true]);
    let capacity = mask.capacity();
    let pointer = mask.as_ptr();
    let filter = SharedTokenFilter::new(TokenFilter::allowed(mask).unwrap());
    let alias = filter.clone();
    let equal = SharedTokenFilter::new(TokenFilter::allowed(vec![true, false, true]).unwrap());
    assert_eq!(filter.capacity_bytes(), Some(capacity as u64));
    assert!(capacity > filter.allowed_mask().unwrap().len());
    assert_eq!(filter.allowed_mask().unwrap().as_ptr(), pointer);
    assert_eq!(alias.allowed_mask().unwrap().as_ptr(), pointer);
    assert!(filter.same_storage(&alias));
    assert!(!filter.same_storage(&equal));
    assert_eq!(filter, equal);
    assert_eq!(filter.identity(), alias.identity());
    assert_ne!(filter.identity(), equal.identity());
    assert_eq!(AsRef::<TokenFilter>::as_ref(&filter), &*alias);
    assert_eq!(
        SharedTokenFilter::new(TokenFilter::All).capacity_bytes(),
        Some(0)
    );
    assert_eq!(
        SharedTokenFilter::new(TokenFilter::Allowed(Vec::new())).capacity_bytes(),
        Some(0)
    );
}

#[test]
fn identity_and_domain_keys_outlive_payload_without_retaining_custody() {
    let filter = filter();
    let weak = Arc::downgrade(&filter.0);
    let identity = filter.identity().clone();
    let domain = SharedStorageDomain::default();
    let other_domain = SharedStorageDomain::default();
    assert!(domain.same_identity(&domain.clone()));
    assert!(!domain.same_identity(&other_domain));
    let used = Arc::new(AtomicUsize::new(0));
    filter
        .try_attach(&domain, || {
            Ok::<_, Infallible>(Charge::acquire(&filter, &domain, &used))
        })
        .unwrap();
    assert!(used.load(Ordering::SeqCst) > 0);
    drop(filter);
    assert!(weak.upgrade().is_none());
    assert_eq!(used.load(Ordering::SeqCst), 0);
    let other = SharedTokenFilter::new(TokenFilter::All);
    assert_ne!(&identity, other.identity());
    assert_eq!(
        BTreeSet::from([identity.clone(), identity.clone(), other.identity().clone()]).len(),
        2
    );
    assert_eq!(
        HashSet::from([identity.clone(), identity, other.identity().clone()]).len(),
        2
    );
    assert_eq!(
        BTreeSet::from([domain.clone(), domain.clone(), other_domain.clone()]).len(),
        2
    );
    assert_eq!(
        HashSet::from([domain.clone(), domain, other_domain]).len(),
        2
    );
}

#[test]
fn preexisting_aliases_retain_each_domain_charge_until_final_retirement() {
    let filter = filter();
    let before_attachment = filter.clone();
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    let bytes = filter.capacity_bytes().unwrap() as usize;
    assert!(filter
        .try_attach(&first, || Ok::<_, Infallible>(Charge::acquire(
            &filter, &first, &used
        )))
        .unwrap());
    assert!(!before_attachment
        .try_attach::<Infallible>(&first.clone(), || panic!(
            "duplicate domain must not acquire"
        ))
        .unwrap());
    assert!(before_attachment
        .try_attach(&second, || Ok::<_, Infallible>(Charge::acquire(
            &filter, &second, &used
        )))
        .unwrap());
    assert_eq!(used.load(Ordering::SeqCst), bytes * 2);
    drop(filter);
    assert_eq!(used.load(Ordering::SeqCst), bytes * 2);
    assert_eq!(
        before_attachment.allowed_mask(),
        Some(&[true, false, true][..])
    );
    drop(before_attachment);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[derive(Debug)]
struct Rejected {
    owner: Weak<Inner>,
    retired: Arc<AtomicBool>,
}
impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("fixture capacity rejected")
    }
}
impl Error for Rejected {}
impl Drop for Rejected {
    fn drop(&mut self) {
        let owner = self.owner.upgrade().expect("caller still owns filter");
        assert!(
            owner.custody.attachments.try_lock().is_ok(),
            "provider error dropped under custody lock"
        );
        self.retired.store(true, Ordering::SeqCst);
    }
}

#[test]
fn provider_rejection_preserves_prior_charge_and_releases_error_after_unlock() {
    let filter = filter();
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    let bytes = filter.capacity_bytes().unwrap() as usize;
    filter
        .try_attach(&first, || {
            Ok::<_, Infallible>(Charge::acquire(&filter, &first, &used))
        })
        .unwrap();
    let retired = Arc::new(AtomicBool::new(false));
    let error = filter
        .try_attach(&second, || {
            Err(Rejected {
                owner: Arc::downgrade(&filter.0),
                retired: retired.clone(),
            })
        })
        .unwrap_err();
    assert!(matches!(&error, SharedStorageAttachmentError::Provider(_)));
    assert_eq!(
        error.source().unwrap().to_string(),
        "fixture capacity rejected"
    );
    assert_eq!(used.load(Ordering::SeqCst), bytes);
    assert_eq!(filter.0.custody.attachments.lock().unwrap().len(), 1);
    drop(error);
    assert!(retired.load(Ordering::SeqCst));
    assert!(filter
        .try_attach(&second, || Ok::<_, Infallible>(Charge::acquire(
            &filter, &second, &used
        )))
        .unwrap());
    assert_eq!(used.load(Ordering::SeqCst), bytes * 2);
    drop(filter);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[test]
fn acquisition_unwind_poison_preserves_existing_custody_until_last_alias() {
    let filter = filter();
    let alias = filter.clone();
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    let bytes = filter.capacity_bytes().unwrap() as usize;
    filter
        .try_attach(&first, || {
            Ok::<_, Infallible>(Charge::acquire(&filter, &first, &used))
        })
        .unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ =
            filter.try_attach::<Infallible>(&second, || panic!("accounting acquisition panicked"));
    }))
    .is_err());
    for domain in [&first, &second] {
        assert!(matches!(
            filter
                .try_attach::<Infallible>(domain, || panic!("poison must reject before provider")),
            Err(SharedStorageAttachmentError::Poisoned)
        ));
    }
    assert_eq!(used.load(Ordering::SeqCst), bytes);
    drop(filter);
    assert_eq!(used.load(Ordering::SeqCst), bytes);
    drop(alias);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[test]
fn simultaneous_alias_attachments_acquire_one_shared_domain_charge() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SharedTokenFilter>();
    send_sync::<SharedStorageIdentity>();
    send_sync::<SharedStorageDomain>();
    let filter = filter();
    let domain = SharedStorageDomain::default();
    let barrier = Arc::new(Barrier::new(8));
    let used = Arc::new(AtomicUsize::new(0));
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let threads = (0..8)
        .map(|_| {
            let filter = filter.clone();
            let domain = domain.clone();
            let barrier = barrier.clone();
            let used = used.clone();
            let acquisitions = acquisitions.clone();
            std::thread::spawn(move || {
                barrier.wait();
                filter
                    .try_attach(&domain, || {
                        acquisitions.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, Infallible>(Charge::acquire(&filter, &domain, &used))
                    })
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    // The filter remains in this thread while all provider calls race, so final
    // retirement cannot mask duplicate acquisition with immediate refunds.
    let published = threads
        .into_iter()
        .map(|thread| usize::from(thread.join().unwrap()))
        .sum::<usize>();
    assert_eq!(published, 1);
    assert_eq!(acquisitions.load(Ordering::SeqCst), 1);
    assert_eq!(
        used.load(Ordering::SeqCst),
        filter.capacity_bytes().unwrap() as usize
    );
    drop(filter);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}
