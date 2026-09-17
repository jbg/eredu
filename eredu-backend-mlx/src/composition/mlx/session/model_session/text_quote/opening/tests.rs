use super::*;

#[test]
fn unsealed_owner_rejects_queries_and_publishes_a_changed_empty_boundary_once() {
    // Owner-level conformance: these are real neutral retention identities, not
    // an installed native copy or a resumed request.
    let before = InferenceRetention::new();
    let seal = OpeningSeal::pending(&before);
    assert_eq!(
        seal.require_sealed(),
        Err(WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(
        seal.predecessor().err(),
        Some(WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(
        seal.validate(&before, 0),
        Err(WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(
        seal.publish_installed(&before),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        seal.require_sealed(),
        Err(WorkingMemoryError::ExecutionFenced)
    );
    let actual = InferenceRetention::new();
    seal.publish_installed(&actual).unwrap();
    seal.require_sealed().unwrap();
    assert!(seal.predecessor().unwrap().is_none());
    seal.validate(&actual, 0).unwrap();
    let foreign = InferenceRetention::new();
    assert_eq!(
        seal.validate(&foreign, 0),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        seal.publish_installed(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    seal.validate(&actual, 0).unwrap();
}

#[test]
fn unquoted_retention_cannot_seal_and_rejection_keeps_owner_pending() {
    let before = InferenceRetention::new();
    let seal = OpeningSeal::pending(&before);
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(100, 0).unwrap();
    let lease = pool.acquire_unquoted().unwrap();
    let mut retained = InferenceRetention::new();
    retained.retain_unquoted(&lease);
    assert_eq!(
        seal.publish_installed(&retained),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    seal.require_pending().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop((lease, retained));
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let actual = InferenceRetention::new();
    seal.publish_installed(&actual).unwrap();
    seal.validate(&actual, 0).unwrap();
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native;
