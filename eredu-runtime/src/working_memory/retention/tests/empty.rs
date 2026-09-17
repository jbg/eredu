use super::*;
use crate::working_memory::WorkingMemoryPool;

#[test]
fn empty_retention_ignores_revision_identity_across_clone_and_restore() {
    const FRESH: InferenceRetention = InferenceRetention::new();
    let mut fresh = FRESH;
    assert!(fresh.is_empty());
    let original_revision = fresh.revision().clone();
    let copied = fresh.clone();
    assert!(fresh.is_empty() && copied.is_empty());
    fresh.restore_admission(&copied);
    assert!(fresh.is_empty());
    assert_ne!(fresh.revision(), &original_revision);
}

#[test]
fn zero_byte_request_is_retained_authority_even_without_an_admission() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = InferenceRequest::from(pool.reserve(&execution, &zero_admission()).unwrap());
    let mut retained = InferenceRetention::new();
    retained.retain(&request);
    assert!(retained.admission().is_none());
    assert!(!retained.is_empty());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let mut restored = InferenceRetention::new();
    restored.restore_admission(&retained);
    assert!(!restored.is_empty());
    restored.restore_admission(&InferenceRetention::new());
    assert!(
        !restored.is_empty(),
        "empty restoration cannot refund retained authority"
    );
    assert!(!retained.clone().is_empty());
}

#[test]
fn unquoted_lease_remains_nonempty_through_original_owner_retirement() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let lease = pool.acquire_unquoted().unwrap();
    let mut retained = InferenceRetention::new();
    retained.retain_unquoted(&lease);
    let copied = retained.clone();
    assert!(!retained.is_empty() && !copied.is_empty());
    assert_eq!(retained.requests().len(), 0);
    assert!(retained.admission().is_none());
    drop((lease, retained));
    assert!(!copied.is_empty());
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(copied);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn admission_presence_is_checked_independently_of_retained_handle_vectors() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = InferenceRequest::from(pool.reserve(&execution, &zero_admission()).unwrap());
    // Private representation case: public admit also records the request, but
    // emptiness must include the admission field in its own right.
    let mut retained = InferenceRetention::new();
    retained.admission = Some(InferenceStateAdmission {
        request,
        position: 0,
    });
    assert_eq!(retained.requests().len(), 0);
    assert!(!retained.is_empty());
    let copied = retained.clone();
    assert!(!copied.is_empty());
    let mut restored = InferenceRetention::new();
    restored.restore_admission(&copied);
    assert!(!restored.is_empty());
}
