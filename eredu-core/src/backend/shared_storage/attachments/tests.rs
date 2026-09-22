use super::*;
use std::{
    convert::Infallible,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

struct Charge {
    accounting: SharedStorageAccountingId,
    bytes: usize,
    used: Arc<AtomicUsize>,
    retired: Arc<AtomicUsize>,
}
impl SharedStorageRetirement for Charge {
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        // The attachment key has retired before its final owner can refund the
        // allocation containing the independently shared accounting identity.
        assert_eq!(Arc::strong_count(&self.accounting.0), 1);
        assert!(self.used.fetch_sub(self.bytes, SeqCst) >= self.bytes);
        self.retired.fetch_add(1, SeqCst);
    }
}
fn funded(
    layout: SharedStorageAttachmentLayout,
    accounting: &SharedStorageAccountingId,
    used: &Arc<AtomicUsize>,
    retired: &Arc<AtomicUsize>,
) -> SharedStorageOwner<Charge> {
    let bytes = layout.requested_bytes();
    assert!(bytes >= layout.node_layout().size());
    used.fetch_add(bytes, SeqCst);
    SharedStorageOwner::new(Charge {
        accounting: accounting.clone(),
        bytes,
        used: used.clone(),
        retired: retired.clone(),
    })
}
#[test]
fn paid_attachment_rejection_reuse_and_escaped_owner_preserve_custody() {
    let table = SharedStorageAttachments::new();
    let first = SharedStorageAccountingId::default();
    let second = SharedStorageAccountingId::default();
    let used = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let escaped = table
        .try_attach_owned_nonblocking(&first, |layout| {
            Ok::<_, Infallible>(funded(layout, &first, &used, &retired))
        })
        .unwrap();
    let first_bytes = used.load(SeqCst);
    let refusal = table
        .try_attach_owned_nonblocking::<Charge, _>(&second, |layout| Err(layout.requested_bytes()));
    assert!(matches!(refusal, Err(SharedStorageAttachmentError::Provider(bytes)) if bytes > 0));
    assert!(!table.has_accounting_custody(&second).unwrap());
    assert_eq!(used.load(SeqCst), first_bytes);
    let reused = table
        .try_attach_owned_nonblocking::<Charge, Infallible>(&first, |_| {
            panic!("reuse must not fund again")
        })
        .unwrap();
    assert!(reused.same_owner(&escaped));
    let other = table
        .try_attach_owned_nonblocking(&second, |layout| {
            Ok::<_, Infallible>(funded(layout, &second, &used, &retired))
        })
        .unwrap();
    assert!(used.load(SeqCst) > first_bytes);
    drop((first, second, reused, other));
    drop(table);
    assert_eq!(retired.load(SeqCst), 1);
    assert_eq!(used.load(SeqCst), first_bytes);
    drop(escaped);
    assert_eq!(used.load(SeqCst), 0);
    assert_eq!(retired.load(SeqCst), 2);
}

#[test]
fn paid_attachment_rejects_type_mismatch_without_calling_provider() {
    let table = SharedStorageAttachments::new();
    let id = SharedStorageAccountingId::default();
    assert!(table
        .try_attach::<Infallible>(&id, |_| Ok(Box::new(())))
        .unwrap());
    let result = table.try_attach_owned_nonblocking::<Charge, Infallible>(&id, |_| {
        panic!("existing opaque owner must reject")
    });
    assert!(matches!(
        result,
        Err(SharedStorageAttachmentError::AttachmentMismatch)
    ));
}
