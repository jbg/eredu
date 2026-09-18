use super::*;
use crate::HostMetadataAccount;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize},
};

#[derive(Debug)]
struct Account {
    calls: Arc<AtomicUsize>,
    stop: usize,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(n <= self.stop, "identity producer after refusal");
        if n == self.stop {
            Err(HostMetadataFundingError::Unavailable)
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
fn account(stop: usize) -> (HostMetadataFunding, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    (
        HostMetadataFunding::new(Account {
            calls: calls.clone(),
            stop,
            retired: retired.clone(),
        })
        .unwrap(),
        calls,
        retired,
    )
}
fn source(root: &std::path::Path) -> DeferredArtifactIdentity {
    DeferredArtifactIdentity::filesystem(
        "original-source-fixture",
        [
            ArtifactFile::new("weights", root.join("weights")),
            ArtifactFile::new("config", root.join("config")),
        ],
    )
    .unwrap()
}
#[test]
fn fresh_identity_keeps_original_hash_and_each_refusal_leaves_retryable_source() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("weights"), [3, 1, 4, 1, 5, 9]).unwrap();
    std::fs::write(root.path().join("config"), b"configuration").unwrap();
    let expected = source(root.path()).resolve().unwrap();
    let original = source(root.path());
    let (funding, calls, _) = account(usize::MAX);
    assert_eq!(original.resolve_with_metadata(&funding).unwrap(), expected);
    let count = calls.load(Ordering::SeqCst);
    assert!(count > 20);
    assert_eq!(original.resolved_identity(), Some(expected));
    for stop in 1..count {
        let original = source(root.path());
        let (funding, calls, retired) = account(stop);
        let error = original.resolve_with_metadata(&funding).unwrap_err();
        assert_eq!(
            error.funding_error(),
            Some(HostMetadataFundingError::Unavailable)
        );
        assert_eq!(calls.load(Ordering::SeqCst), stop + 1);
        assert!(
            !original.is_resolved(),
            "refusal must not poison the shared source cache"
        );
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(error);
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(original.resolve().unwrap(), expected);
    }
    // A cached digest is fixed input: it does not reopen changed or removed files.
    std::fs::remove_file(root.path().join("weights")).unwrap();
    assert_eq!(original.resolve_with_metadata(&funding).unwrap(), expected);
}

#[test]
fn identity_source_error_keeps_real_filesystem_cause_and_original_account() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("weights"), [7]).unwrap();
    std::fs::write(root.path().join("config"), [8]).unwrap();
    let original = source(root.path());
    std::fs::remove_file(root.path().join("weights")).unwrap();
    let (funding, _, retired) = account(usize::MAX);
    let error = original.resolve_with_metadata(&funding).unwrap_err();
    let mut leaf: &(dyn std::error::Error + 'static) = &error;
    while !leaf.is::<std::io::Error>() {
        leaf = leaf.source().expect("original I/O leaf");
    }
    assert_eq!(
        leaf.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::NotFound
    );
    drop(funding);
    assert!(!retired.load(Ordering::SeqCst));
    drop(error);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn iterative_identity_order_agrees_with_independent_sorted_reference() {
    use sha2::Digest;
    for count in 0..130usize {
        let input = (0..count)
            .rev()
            .map(|n| ArtifactMemberIdentity::new(format!("role-{n:03}"), n as u64, [n as u8; 32]))
            .collect::<Vec<_>>();
        let mut actual = input.clone();
        sort(&mut actual);
        let mut expected = input;
        expected.sort_by(|a, b| a.logical_role.cmp(&b.logical_role));
        assert_eq!(actual, expected);
        if count == 0 {
            continue;
        }
        // Independent digest transcript, without calling the identity worker's
        // component writer or relying on the input iterator's ordering.
        let mut hash = sha2::Sha256::new();
        for component in [
            b"eredu-checkpoint-artifact-v2".as_slice(),
            b"fixture".as_slice(),
        ] {
            hash.update((component.len() as u64).to_le_bytes());
            hash.update(component);
        }
        hash.update((count as u64).to_le_bytes());
        for row in &expected {
            hash.update((row.logical_role.len() as u64).to_le_bytes());
            hash.update(row.logical_role.as_bytes());
            hash.update(row.length.to_le_bytes());
            hash.update(row.digest);
        }
        assert_eq!(
            fingerprint_artifact("fixture", actual).unwrap().digest(),
            <[u8; 32]>::from(hash.finalize())
        );
    }
}

#[test]
fn ordinary_and_funded_resolvers_share_one_completed_digest() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("weights"), vec![11; 2 * 1024 * 1024]).unwrap();
    std::fs::write(root.path().join("config"), [8]).unwrap();
    let original = source(root.path());
    let (funding, _, _) = account(usize::MAX);
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let mut tasks = Vec::new();
    for index in 0..8 {
        let original = original.clone();
        let funding = funding.clone();
        let barrier = barrier.clone();
        tasks.push(std::thread::spawn(move || {
            barrier.wait();
            if index % 2 == 0 {
                original.resolve().unwrap()
            } else {
                original.resolve_with_metadata(&funding).unwrap()
            }
        }));
    }
    let expected = tasks.remove(0).join().unwrap();
    for task in tasks {
        assert_eq!(task.join().unwrap(), expected);
    }
    assert_eq!(original.resolved_identity(), Some(expected));
}

#[test]
fn semantic_identity_errors_preserve_diagnostics_and_refuse_each_destination() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("member");
    std::fs::write(&path, [2, 7, 1, 8]).unwrap();
    for roles in [vec![], vec![""], vec!["quoted\"東京", "quoted\"東京"]] {
        let source = || {
            DeferredArtifactIdentity::filesystem(
                "source-errors",
                roles.iter().map(|role| ArtifactFile::new(*role, &path)),
            )
            .unwrap()
        };
        let ordinary = source().resolve().unwrap_err().to_string();
        let (funding, calls, retired) = account(usize::MAX);
        let original = source();
        let error = original.resolve_with_metadata(&funding).unwrap_err();
        assert_eq!(error.to_string(), ordinary);
        assert!(matches!(
            error.cause,
            ArtifactError::InvalidArtifactIdentity(_)
        ));
        assert!(!original.is_resolved());
        let count = calls.load(Ordering::SeqCst);
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(error);
        assert!(retired.load(Ordering::SeqCst));
        for stop in 1..count {
            let original = source();
            let (funding, calls, retired) = account(stop);
            let error = original.resolve_with_metadata(&funding).unwrap_err();
            assert_eq!(
                error.funding_error(),
                Some(HostMetadataFundingError::Unavailable)
            );
            assert_eq!(calls.load(Ordering::SeqCst), stop + 1);
            assert!(!original.is_resolved());
            drop(funding);
            assert!(!retired.load(Ordering::SeqCst));
            drop(error);
            assert!(retired.load(Ordering::SeqCst));
        }
    }
}
