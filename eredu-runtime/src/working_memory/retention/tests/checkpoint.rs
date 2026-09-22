use super::*;
use eredu_core::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Account {
    spent: Arc<AtomicUsize>,
    retired: Arc<AtomicBool>,
    result: Arc<Mutex<Option<(u64, usize)>>>,
    pool: crate::working_memory::MemoryLedger,
    leases: crate::working_memory::MemoryLedger,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.spent
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(bytes))
            .map(|_| ())
            .map_err(|_| HostMetadataFundingError::Overflow)
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        *self.result.lock().unwrap() = Some((
            self.pool.payload_used_bytes().unwrap(),
            self.leases.unquoted_owner_count().unwrap(),
        ));
        self.retired.store(true, Ordering::SeqCst);
    }
}
#[test]
fn checkpoint_retention_authenticates_new_request_and_existing_lease_sources() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let make = || InferenceRequest::from(pool.reserve(&execution, &zero_admission(&pool)).unwrap());
    let prior = make();
    let current = make();
    let foreign = make();
    let leases = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let lease = leases.acquire_unquoted().unwrap();
    let mut source = InferenceRetention::new();
    source.admit(&prior);
    source.retain_unquoted(&lease);
    let mut actual = source.clone();
    actual.admit(&current);
    actual
        .validate_checkpoint_source(&source, &current)
        .unwrap();
    assert!(matches!(
        actual.validate_checkpoint_source(&source, &foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut missing = InferenceRetention::new();
    missing.admit(&current);
    assert!(matches!(
        missing.validate_checkpoint_source(&source, &current),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let additional = leases.acquire_unquoted().unwrap();
    actual.retain_unquoted(&additional);
    assert!(matches!(
        actual.validate_checkpoint_source(&source, &current),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
}
#[test]
fn checkpoint_retention_clone_debits_its_source_and_retires_handles_before_payer() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let mut admission = zero_admission(&pool);
    admission.additional_headroom =
        eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 19)]);
    let request = InferenceRequest::from(pool.reserve(&execution, &admission).unwrap());
    let leases = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let lease = leases.acquire_unquoted().unwrap();
    let spent = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let result = Arc::new(Mutex::new(None));
    let funding = HostMetadataFunding::new(Account {
        spent: spent.clone(),
        retired: retired.clone(),
        result: result.clone(),
        pool: pool.clone(),
        leases: leases.clone(),
    })
    .unwrap();
    let mut source = InferenceRetention::new();
    source.admit(&request);
    source.retain_unquoted(&lease);
    let bound = source.host_clone_bytes().unwrap();
    let before = spent.load(Ordering::SeqCst);
    let saved = source.clone_with_host_source(&funding).unwrap();
    assert!(spent.load(Ordering::SeqCst) - before <= bound);
    drop((source, request, lease, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(pool.payload_used_bytes().unwrap(), 19);
    assert_eq!(leases.unquoted_owner_count().unwrap(), 1);
    drop(saved);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(*result.lock().unwrap(), Some((0, 0)));
}
