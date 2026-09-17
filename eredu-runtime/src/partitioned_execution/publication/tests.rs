use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
#[derive(Debug)]
struct Account {
    live: Arc<AtomicBool>,
    used: Arc<AtomicUsize>,
    limit: Arc<AtomicUsize>,
}
impl eredu_nn::workspace::WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes)
                    .filter(|&next| next <= self.limit.load(Ordering::SeqCst))
            })
            .map(|_| ())
            .map_err(|_| WorkspaceMetadataFundingError::Unavailable)
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.live.store(false, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct NativeFailure {
    live: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    formats: Arc<AtomicUsize>,
}
impl std::fmt::Display for NativeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.formats.fetch_add(1, Ordering::SeqCst);
        f.write_str("actual completion failure")
    }
}
impl std::error::Error for NativeFailure {}
impl Drop for NativeFailure {
    fn drop(&mut self) {
        assert!(self.live.load(Ordering::SeqCst));
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
struct FailedCompletion(NativeFailure);
impl eredu_core::Completion for FailedCompletion {
    type Error = NativeFailure;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(false)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        panic!("shared publisher must use its selected bounded wait")
    }
}
impl BoundedCompletion for FailedCompletion {
    fn wait_bounded(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        assert_eq!(
            policy.cancellation(),
            CompletionCancellationMode::QuarantineUntilComplete
        );
        Err(self.0)
    }
}
#[test]
fn prepared_publication_uses_shared_wait_and_retains_unformatted_failure_custody() {
    let live = Arc::new(AtomicBool::new(true));
    let used = Arc::new(AtomicUsize::new(0));
    let limit = Arc::new(AtomicUsize::new(usize::MAX));
    let drops = Arc::new(AtomicUsize::new(0));
    let formats = Arc::new(AtomicUsize::new(0));
    let funding = WorkspaceMetadataFunding::new(Account {
        live: live.clone(),
        used: used.clone(),
        limit: limit.clone(),
    })
    .unwrap();
    let before = used.load(Ordering::SeqCst);
    let controls =
        Controls::<NativeFailure>::prepare::<[u32; 3], FailedCompletion>(&funding).unwrap();
    assert!(used.load(Ordering::SeqCst) > before);
    let policy = crate::CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let authority = PartitionCommunicationAuthority::new(Some(policy));
    let error = authority
        .wait_with_error(
            eredu_core::Submission {
                output: [11, 23, 41],
                completion: FailedCompletion(NativeFailure {
                    live: live.clone(),
                    drops: drops.clone(),
                    formats: formats.clone(),
                }),
            },
            CommunicationOperation::Broadcast,
            DistributedExecutionPhase::OutputPublication,
            None,
            |cause| {
                controls.failure(
                    &authority,
                    cause,
                    DistributedExecutionPhase::OutputPublication,
                    true,
                )
            },
        )
        .unwrap_err();
    assert!(matches!(
        &error,
        PartitionExecutionError::PreparedCommunication {
            completion: true,
            ..
        }
    ));
    assert!(matches!(
        authority.ensure_active(),
        Err(PartitionExecutionError::CommunicationPoisoned { .. })
    ));
    assert_eq!(formats.load(Ordering::SeqCst), 0);
    let spent = used.load(Ordering::SeqCst);
    limit.store(spent, Ordering::SeqCst);
    assert!(matches!(
        Controls::<NativeFailure>::prepare::<[u32; 3], FailedCompletion>(&funding),
        Err(PartitionExecutionError::PublicationMetadata(_))
    ));
    assert_eq!(used.load(Ordering::SeqCst), spent);
    drop((controls, funding, authority));
    assert!(live.load(Ordering::SeqCst));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(!live.load(Ordering::SeqCst));
    assert_eq!(formats.load(Ordering::SeqCst), 0);
}
