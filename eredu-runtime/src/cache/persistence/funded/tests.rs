use super::*;
use crate::cache::worker::test_support::{AccountState, context};
use std::sync::{Arc, atomic::Ordering};

fn new_funding() -> (
    PromptCachePersistenceFunding,
    Arc<AccountState>,
    Arc<AccountState>,
) {
    let (construction, actual) = context();
    let (dependency, estimate) = context();
    let funding = PromptCachePersistenceFunding::new(
        &construction,
        dependency.metadata_funding().unwrap(),
        DependencyMemoryPolicy::default(),
        DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT,
    )
    .unwrap();
    (funding, actual, estimate)
}
fn manifest(path: &Path) -> PromptCacheManifest {
    let mut manifest = super::super::tests::manifest(path);
    let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::End(-32)).unwrap();
    file.write_all(&[
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137,
    ])
    .unwrap();
    drop(file);
    manifest.blocks[0].payload_sha256 = hash_prompt_cache_shard_payload(path).unwrap();
    manifest
}
fn publish(
    directory: &Path,
    funding: &PromptCachePersistenceFunding,
    first: &str,
) -> PromptCacheManifest {
    let publication = PreparedPromptCachePublication::begin(directory, false, funding).unwrap();
    let path = funding
        .join_path(publication.staging_directory(), "block.safetensors")
        .unwrap();
    let mut manifest = manifest(&path);
    manifest.checkpoint_fingerprint = first.into();
    publication.commit(&manifest).unwrap();
    manifest
}

#[test]
fn funded_persistence_shares_workers_and_retains_both_manifest_accounts() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("cache");
    let (funding, actual, estimate) = new_funding();
    let controlled_before = *actual.remaining.lock().unwrap();
    let estimate_before = *estimate.remaining.lock().unwrap();
    let expected = publish(&directory, &funding, "first");
    let shared = inspect_prompt_cache_funded(&directory, &funding).unwrap();
    assert_eq!(shared.as_manifest(), &expected);
    assert_eq!(inspect_prompt_cache(&directory).unwrap(), expected);
    assert!(*actual.remaining.lock().unwrap() < controlled_before);
    assert!(*estimate.remaining.lock().unwrap() < estimate_before);
    let alias = shared.clone();
    assert!(shared.same_storage(&alias));
    drop(funding);
    drop(shared);
    assert!(!actual.retired.load(Ordering::SeqCst));
    assert!(!estimate.retired.load(Ordering::SeqCst));
    drop(alias);
    assert!(actual.retired.load(Ordering::SeqCst));
    assert!(estimate.retired.load(Ordering::SeqCst));
}

#[test]
fn funded_persistence_refuses_before_construction_and_retains_typed_errors() {
    for refuse_dependency in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("cache");
        let (funding, actual, estimate) = new_funding();
        *(if refuse_dependency {
            &estimate
        } else {
            &actual
        })
        .remaining
        .lock()
        .unwrap() = 0;
        let error =
            PreparedPromptCachePublication::begin(&destination, false, &funding).unwrap_err();
        match error.cause() {
            PromptCachePersistenceError::Dependency(_) => assert!(refuse_dependency),
            PromptCachePersistenceError::Metadata(_) => assert!(!refuse_dependency),
            other => panic!("unexpected refusal: {other:?}"),
        }
        assert!(!destination.exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        drop(funding);
        assert!(!actual.retired.load(Ordering::SeqCst));
        assert!(!estimate.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(actual.retired.load(Ordering::SeqCst));
        assert!(estimate.retired.load(Ordering::SeqCst));
    }
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("cache");
    let (funding, actual, estimate) = new_funding();
    publish(&destination, &funding, "first");
    let bounded = PromptCachePersistenceFunding::new(
        funding.context(),
        funding.dependency().clone(),
        funding.policy(),
        1,
    )
    .unwrap();
    let error = inspect_prompt_cache_funded(&destination, &bounded).unwrap_err();
    assert!(matches!(
        error.cause(),
        PromptCachePersistenceError::MetadataInputLimit { limit: 1, .. }
    ));
    let file = resolve_prompt_cache_root(&destination)
        .unwrap()
        .join("manifest.json");
    fs::write(file, b"{invalid-json").unwrap();
    let syntax = inspect_prompt_cache_funded(&destination, &funding).unwrap_err();
    assert!(matches!(
        syntax.cause(),
        PromptCachePersistenceError::ManifestJson(_)
    ));
    drop(bounded);
    drop(funding);
    drop(error);
    assert!(!actual.retired.load(Ordering::SeqCst));
    assert!(!estimate.retired.load(Ordering::SeqCst));
    drop(syntax);
    assert!(actual.retired.load(Ordering::SeqCst));
    assert!(estimate.retired.load(Ordering::SeqCst));
}

#[test]
fn funded_reversible_publication_prepares_rollback_before_visibility() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("cache");
    let (funding, actual, estimate) = new_funding();
    let first = publish(&destination, &funding, "first");
    let mut replacement =
        PreparedReversiblePromptCachePublication::begin(&destination, true, &funding).unwrap();
    let second = publish(replacement.staging_destination(), &funding, "second");
    replacement.publish().unwrap();
    assert_eq!(inspect_prompt_cache(&destination).unwrap(), second);
    // No new grant may be required to reverse already-visible publication.
    *actual.remaining.lock().unwrap() = 0;
    *estimate.remaining.lock().unwrap() = 0;
    replacement.rollback().unwrap();
    assert_eq!(inspect_prompt_cache(&destination).unwrap(), first);

    for commit in [false, true] {
        let (funding, actual, estimate) = new_funding();
        let mut replacement =
            PreparedReversiblePromptCachePublication::begin(&destination, true, &funding).unwrap();
        let staging = replacement.staging_destination().to_path_buf();
        let second = publish(&staging, &funding, "second");
        if commit {
            replacement.publish().unwrap();
        }
        *actual.remaining.lock().unwrap() = 0;
        *estimate.remaining.lock().unwrap() = 0;
        if commit {
            replacement.commit().unwrap();
            assert_eq!(inspect_prompt_cache(&destination).unwrap(), second);
        } else {
            drop(replacement);
            assert_eq!(inspect_prompt_cache(&destination).unwrap(), first);
        }
        assert!(!staging.exists());
    }
}

#[test]
fn funded_shard_reader_and_hash_use_the_supplied_open_handle() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("block.safetensors");
    let expected = manifest(&path);
    let mut file = File::open(&path).unwrap();
    let moved = root.path().join("original.safetensors");
    fs::rename(&path, &moved).unwrap();
    fs::write(&path, b"different file at the same pathname").unwrap();
    let (funding, _, _) = new_funding();
    let (metadata, header, file_len) =
        read_shard_metadata_from_funded(&mut file, &path, &funding).unwrap();
    assert_eq!(metadata.data_len(), 32);
    assert_eq!(header.len() + metadata.data_len(), file_len);
    assert_eq!(
        u64::from_le_bytes(header[..8].try_into().unwrap()) as usize,
        header.len() - 8
    );
    let digest =
        hash_prompt_cache_shard_payload_from_funded(&mut file, &path, header.len(), &funding)
            .unwrap();
    assert_eq!(digest, expected.blocks[0].payload_sha256);
}
