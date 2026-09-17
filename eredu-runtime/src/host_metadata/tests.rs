use super::*;
use crate::{PreparedInputCacheIdentity, StateLayout, StateSegmentLifetime, StateSegmentSpec};
use eredu_core::{
    cache::LayerCachePolicy, checkpoint::TensorDtype, AttentionPolicy, InputModality,
    InputPartDescriptor, InputPayloadKind, InputTensorIdentity, LayerSchedule,
    PreparedInputIdentity,
};
use std::{
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    error::Error,
    fmt,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
        Barrier,
    },
};

fn layout() -> SharedHostMetadata {
    let mut name = String::with_capacity(83);
    name.push_str("decoder-β");
    let layers = LayerSchedule::new(
        2,
        vec![
            LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 4).unwrap(),
            LayerCachePolicy::NoState,
        ],
    )
    .unwrap();
    let layout = StateLayout::segmented(
        layers,
        [StateSegmentSpec::new(name, 0..2, StateSegmentLifetime::Persistent, 0).unwrap()],
    )
    .unwrap();
    SharedHostMetadata::Layout(SharedStateLayout::new(layout))
}

fn input() -> SharedHostMetadata {
    let mut shape = Vec::with_capacity(7);
    shape.extend_from_slice(&[1, 5]);
    let prepared = PreparedInputIdentity::new(vec![InputPartDescriptor::new(
        InputModality::Text,
        InputPayloadKind::TokenIds,
        InputTensorIdentity::new(TensorDtype::U32, shape).unwrap(),
        [],
    )
    .unwrap()])
    .unwrap();
    let mut fingerprint = String::with_capacity(97);
    fingerprint.push_str("content-λ-123");
    SharedHostMetadata::Input(SharedPreparedInputCacheIdentity::new(
        PreparedInputCacheIdentity::new(prepared, fingerprint).unwrap(),
    ))
}

struct Charge {
    used: Arc<AtomicU64>,
    bytes: u64,
    _identity: HostMetadataIdentity,
    _domain: SharedStorageDomain,
}

impl Charge {
    fn acquire(
        source: &SharedHostMetadata,
        domain: &SharedStorageDomain,
        used: &Arc<AtomicU64>,
    ) -> Box<dyn Send + Sync> {
        let bytes = source.capacity_bytes().expect("closed fixture capacity");
        used.fetch_add(bytes, AtomicOrdering::SeqCst);
        Box::new(Self {
            used: Arc::clone(used),
            bytes,
            _identity: source.identity().clone(),
            _domain: domain.clone(),
        })
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, AtomicOrdering::SeqCst);
    }
}

fn attach(
    source: &SharedHostMetadata,
    domain: &SharedStorageDomain,
    used: &Arc<AtomicU64>,
) -> bool {
    source
        .try_attach(domain, || {
            Ok::<_, Infallible>(Charge::acquire(source, domain, used))
        })
        .unwrap()
}

#[test]
fn earlier_aliases_retain_both_domain_attachments_until_the_final_owner() {
    for source in [layout(), input()] {
        let earlier = source.clone();
        let before_attachment = earlier.clone();
        let capacity = source.capacity_bytes().unwrap();
        assert!(capacity > 97);
        let first = SharedStorageDomain::default();
        let second = SharedStorageDomain::default();
        let used = Arc::new(AtomicU64::new(0));
        assert!(attach(&source, &first, &used));
        assert!(!earlier
            .try_attach::<Infallible>(&first, || panic!("duplicate must not acquire"))
            .unwrap());
        assert!(attach(&earlier, &second, &used));
        assert_eq!(used.load(AtomicOrdering::SeqCst), capacity * 2);
        assert_eq!(source.identity(), before_attachment.identity());
        drop((source, earlier));
        assert_eq!(used.load(AtomicOrdering::SeqCst), capacity * 2);
        match &before_attachment {
            SharedHostMetadata::ObservationPaths(_) => unreachable!("layout/input fixture"),
            SharedHostMetadata::Layout(layout) => {
                assert_eq!(layout.as_ref().len(), 2);
                assert_eq!(layout.as_ref().segments()[0].id().as_str(), "decoder-β");
            }
            SharedHostMetadata::Input(input) => {
                assert_eq!(
                    input.as_ref().semantic_content_fingerprint(),
                    "content-λ-123"
                );
            }
        }
        drop(before_attachment);
        assert_eq!(used.load(AtomicOrdering::SeqCst), 0);
    }
}

#[test]
fn identity_keys_do_not_retain_payload_or_registration_and_distinguish_equal_values() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SharedHostMetadata>();
    send_sync::<HostMetadataIdentity>();
    for make in [layout as fn() -> SharedHostMetadata, input] {
        let source = make();
        let equal = make();
        assert_ne!(source.identity(), equal.identity());
        match (&source, &equal) {
            (SharedHostMetadata::Layout(left), SharedHostMetadata::Layout(right)) => {
                assert_eq!(left, right);
            }
            (SharedHostMetadata::Input(left), SharedHostMetadata::Input(right)) => {
                assert_eq!(left, right);
            }
            _ => unreachable!(),
        }
        let key = source.identity().clone();
        let ordered = BTreeSet::from([
            key.clone(),
            source.identity().clone(),
            equal.identity().clone(),
        ]);
        let hashed = HashSet::from([
            key.clone(),
            source.identity().clone(),
            equal.identity().clone(),
        ]);
        assert_eq!((ordered.len(), hashed.len()), (2, 2));
        let used = Arc::new(AtomicU64::new(0));
        let domain = SharedStorageDomain::default();
        attach(&source, &domain, &used);
        assert!(used.load(AtomicOrdering::SeqCst) > 0);
        drop(source);
        // Provider charge keys and surviving caller keys share only the small
        // identity allocation. There is no owner-registration-payload cycle.
        assert_eq!(used.load(AtomicOrdering::SeqCst), 0);
        assert!(ordered.contains(&key));
        assert!(hashed.contains(&key));
    }
}

#[test]
fn concurrent_attachments_acquire_once_per_domain_for_each_closed_owner() {
    for source in [layout(), input()] {
        let domain = SharedStorageDomain::default();
        let barrier = Barrier::new(8);
        let used = Arc::new(AtomicU64::new(0));
        let acquired = AtomicUsize::new(0);
        let published = std::thread::scope(|threads| {
            let joins = (0..8)
                .map(|_| {
                    threads.spawn(|| {
                        barrier.wait();
                        source
                            .try_attach(&domain, || {
                                acquired.fetch_add(1, AtomicOrdering::SeqCst);
                                Ok::<_, Infallible>(Charge::acquire(&source, &domain, &used))
                            })
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            joins
                .into_iter()
                .map(|join| usize::from(join.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(published, 1);
        assert_eq!(acquired.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(
            used.load(AtomicOrdering::SeqCst),
            source.capacity_bytes().unwrap()
        );
        drop(source);
        assert_eq!(used.load(AtomicOrdering::SeqCst), 0);
    }
}

#[derive(Debug)]
struct Rejected {
    source: SharedHostMetadata,
    domain: SharedStorageDomain,
    dropped: Arc<AtomicBool>,
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("metadata registration rejected")
    }
}
impl Error for Rejected {}
impl Drop for Rejected {
    fn drop(&mut self) {
        // This reenters the same owner. An error payload must reach its caller
        // with the attachment lock released, without poisoning earlier custody.
        assert!(self
            .source
            .try_attach(&self.domain, || Ok::<_, Infallible>(
                Box::new(()) as Box<dyn Send + Sync>
            ))
            .unwrap());
        self.dropped.store(true, AtomicOrdering::SeqCst);
    }
}

#[test]
fn provider_failure_preserves_earlier_charge_and_its_error_drops_after_unlock() {
    for source in [layout(), input()] {
        let first = SharedStorageDomain::default();
        let rejected = SharedStorageDomain::default();
        let used = Arc::new(AtomicU64::new(0));
        attach(&source, &first, &used);
        let dropped = Arc::new(AtomicBool::new(false));
        let error = source
            .try_attach(&rejected, || {
                Err(Rejected {
                    source: source.clone(),
                    domain: rejected.clone(),
                    dropped: Arc::clone(&dropped),
                })
            })
            .unwrap_err();
        assert!(matches!(&error, SharedStorageAttachmentError::Provider(_)));
        assert_eq!(
            error.source().unwrap().to_string(),
            "metadata registration rejected"
        );
        assert_eq!(
            used.load(AtomicOrdering::SeqCst),
            source.capacity_bytes().unwrap()
        );
        drop(error);
        assert!(dropped.load(AtomicOrdering::SeqCst));
        assert!(!source
            .try_attach::<Infallible>(&rejected, || panic!("error destructor attached"))
            .unwrap());
        drop(source);
        assert_eq!(used.load(AtomicOrdering::SeqCst), 0);
    }
}

#[test]
fn provider_panic_keeps_old_attachments_and_poison_blocks_every_later_provider() {
    for source in [layout(), input()] {
        let alias = source.clone();
        let first = SharedStorageDomain::default();
        let second = SharedStorageDomain::default();
        let used = Arc::new(AtomicU64::new(0));
        attach(&source, &first, &used);
        let capacity = source.capacity_bytes().unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ =
                source.try_attach::<Infallible>(&second, || panic!("injected provider failure"));
        }))
        .is_err());
        for domain in [&first, &second] {
            assert!(matches!(
                source
                    .try_attach::<Infallible>(domain, || panic!("poison rejects before provider")),
                Err(SharedStorageAttachmentError::Poisoned)
            ));
        }
        drop(source);
        assert_eq!(used.load(AtomicOrdering::SeqCst), capacity);
        drop(alias);
        assert_eq!(used.load(AtomicOrdering::SeqCst), 0);
    }
}

struct AttachOnDrop {
    other: SharedHostMetadata,
    domain: SharedStorageDomain,
    completed: Arc<AtomicBool>,
}

impl Drop for AttachOnDrop {
    fn drop(&mut self) {
        assert!(self
            .other
            .try_attach(&self.domain, || {
                Ok::<_, Infallible>(Box::new(()) as Box<dyn Send + Sync>)
            })
            .unwrap());
        self.completed.store(true, AtomicOrdering::SeqCst);
    }
}

#[test]
fn final_attachment_retirement_can_reenter_another_live_metadata_owner() {
    let source = layout();
    let other = input();
    let target_domain = SharedStorageDomain::default();
    let completed = Arc::new(AtomicBool::new(false));
    source
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<_, Infallible>(Box::new(AttachOnDrop {
                other: other.clone(),
                domain: target_domain.clone(),
                completed: Arc::clone(&completed),
            }) as Box<dyn Send + Sync>)
        })
        .unwrap();
    drop(source);
    assert!(completed.load(AtomicOrdering::SeqCst));
    assert!(!other
        .try_attach::<Infallible>(&target_domain, || panic!("already attached"))
        .unwrap());
}
