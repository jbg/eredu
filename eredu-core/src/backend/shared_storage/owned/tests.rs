use super::*;
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst},
        Barrier,
    },
};
#[derive(Default)]
struct Probe {
    extracted: AtomicBool,
    dropped: AtomicUsize,
}
struct Payload {
    values: Vec<u32>,
    probe: Arc<Probe>,
}
impl SharedStorageRetirement for Payload {
    fn retire(self: Arc<Self>) {
        let value = Arc::into_inner(self);
        if let Some(value) = &value {
            value.probe.extracted.store(true, SeqCst);
        }
        drop(value);
    }
}
impl Drop for Payload {
    fn drop(&mut self) {
        assert!(self.probe.extracted.load(SeqCst));
        self.probe.dropped.fetch_add(1, SeqCst);
    }
}
fn owner() -> (SharedStorageOwner<Payload>, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    (
        SharedStorageOwner::new(Payload {
            values: vec![3, 5, 8],
            probe: probe.clone(),
        }),
        probe,
    )
}
struct Other;
impl SharedStorageRetirement for Other {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}

#[test]
fn closed_typed_erased_aliases_share_payload_and_one_concurrent_retirement() {
    let (owner, probe) = owner();
    let address = std::ptr::from_ref::<Payload>(&owner);
    let erased = owner.clone().erase();
    assert!(std::ptr::eq(
        erased.downcast_ref::<Payload>().unwrap(),
        address
    ));
    assert!(erased.downcast_ref::<Other>().is_none());
    let typed = erased.clone_typed::<Payload>().unwrap();
    assert!(owner.same_owner(&typed));
    let barrier = Barrier::new(3);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            barrier.wait();
            assert_eq!(typed.values, [3, 5, 8]);
            drop(typed);
        });
        let b = scope.spawn(|| {
            barrier.wait();
            assert_eq!(erased.downcast_ref::<Payload>().unwrap().values, [3, 5, 8]);
            drop(erased);
        });
        barrier.wait();
        drop(owner);
        a.join().unwrap();
        b.join().unwrap();
    });
    assert_eq!(probe.dropped.load(SeqCst), 1);
}

struct Unused<'a> {
    custody: &'a SharedStorageCustody,
    checked: &'a AtomicBool,
}
impl Drop for Unused<'_> {
    fn drop(&mut self) {
        assert!(
            self.custody.attachments.try_lock().is_ok(),
            "unused capture dropped under custody"
        );
        self.checked.store(true, SeqCst);
    }
}
#[test]
fn owned_attachment_reuses_same_owner_and_unused_closure_drops_after_unlock() {
    let custody = SharedStorageCustody::new();
    let domain = SharedStorageAccountingId::default();
    let (owner, probe) = owner();
    let attached = custody
        .try_attach_owned_nonblocking(&domain, || Ok::<_, Infallible>(owner.clone()))
        .unwrap();
    assert!(owner.same_owner(&attached));
    let checked = AtomicBool::new(false);
    let unused = Unused {
        custody: &custody,
        checked: &checked,
    };
    let reused = custody
        .try_attach_owned_nonblocking::<Payload, Infallible>(&domain, move || {
            drop(unused);
            panic!("existing provider must not run")
        })
        .unwrap();
    assert!(checked.load(SeqCst));
    assert!(owner.same_owner(&reused));
    drop((owner, attached, reused));
    assert_eq!(probe.dropped.load(SeqCst), 0);
    drop(custody);
    assert_eq!(probe.dropped.load(SeqCst), 1);
}

#[test]
fn owned_and_raw_attachment_populations_cannot_cross_and_mismatch_is_lazy() {
    for raw in [false, true] {
        let custody = SharedStorageCustody::new();
        let domain = SharedStorageAccountingId::default();
        if raw {
            custody
                .try_attach_typed_nonblocking(&domain, || Ok::<_, Infallible>(Arc::new(Other)))
                .unwrap();
        } else {
            custody
                .try_attach_owned_nonblocking(&domain, || {
                    Ok::<_, Infallible>(SharedStorageOwner::new(Other))
                })
                .unwrap();
        }
        assert!(matches!(
            custody.try_attach_owned_nonblocking::<Payload, Infallible>(&domain, || panic!(
                "wrong type provider"
            )),
            Err(SharedStorageAttachmentError::AttachmentMismatch)
        ));
        if raw {
            assert!(matches!(
                custody.try_attach_owned_nonblocking::<Other, Infallible>(&domain, || panic!(
                    "raw cannot close"
                )),
                Err(SharedStorageAttachmentError::AttachmentMismatch)
            ));
        } else {
            assert!(matches!(
                custody.try_attach_typed_nonblocking::<Other, Infallible>(&domain, || panic!(
                    "owned cannot reopen"
                )),
                Err(SharedStorageAttachmentError::AttachmentMismatch)
            ));
        }
        assert!(!custody
            .try_attach::<Infallible>(&domain, || panic!("opaque stays lazy"))
            .unwrap());
    }
}

#[test]
fn owned_provider_error_and_unwind_keep_staged_payload_outside_source_lock() {
    let custody = SharedStorageCustody::new();
    let domain = SharedStorageAccountingId::default();
    let checked = AtomicBool::new(false);
    let error = custody.try_attach_owned_nonblocking::<Payload, _>(&domain, || {
        Err(Unused {
            custody: &custody,
            checked: &checked,
        })
    });
    assert!(matches!(
        &error,
        Err(SharedStorageAttachmentError::Provider(_))
    ));
    assert!(!checked.load(SeqCst));
    drop(error);
    assert!(checked.load(SeqCst));
    let (owner, probe) = owner();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = custody.try_attach_owned_nonblocking::<Payload, Infallible>(&domain, || {
            assert_eq!(owner.values, [3, 5, 8]);
            std::panic::panic_any(41_u64);
        });
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<u64>(), Some(&41));
    assert!(matches!(
        custody.try_attach_owned_nonblocking::<Payload, Infallible>(&domain, || panic!(
            "poisoned provider"
        )),
        Err(SharedStorageAttachmentError::Poisoned)
    ));
    assert_eq!(probe.dropped.load(SeqCst), 0);
    drop(owner);
    assert_eq!(probe.dropped.load(SeqCst), 1);
}

#[test]
fn owned_attachment_busy_and_cold_controls_do_not_invoke_provider() {
    let custody = SharedStorageCustody::new();
    let guard = custody.attachments.lock().unwrap();
    let bytes =
        crate::capture::SharedCapturePlan::owned_attachment_control_bytes::<Payload, Infallible>()
            .unwrap();
    assert!(bytes > 0);
    assert!(matches!(
        custody.try_attach_owned_nonblocking::<Payload, Infallible>(
            &SharedStorageAccountingId::default(),
            || panic!("busy provider")
        ),
        Err(SharedStorageAttachmentError::Busy)
    ));
    drop(guard);
}
