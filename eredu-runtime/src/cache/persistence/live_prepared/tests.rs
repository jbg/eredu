use super::*;
use crate::cache::worker::test_support::context;

fn block() -> CacheBlockId {
    CacheBlockId {
        session_id: 9,
        global_layer: 12,
        representation: CacheRepresentation::KeyValue,
        start: 3,
        end: 7,
        rank: None,
    }
}

#[test]
fn prepared_publication_preserves_names_collision_files_and_last_shared_owner_custody() {
    let directory = tempfile::tempdir().unwrap();
    let id = block();
    let ordinary = ordinary_paths(directory.path(), &id, "source", 5);
    assert_eq!(
        ordinary.destination_path().file_name().unwrap(),
        "live-source-w0000000000000005-s0000000000000009-layer-00012-kv-rank-px-tx-ex-3-7.safetensors"
    );
    assert_eq!(
        ordinary.staging_path().file_name().unwrap(),
        ".live-source-w0000000000000005-s0000000000000009-layer-00012-kv-rank-px-tx-ex-3-7.tmp.safetensors"
    );
    drop(ordinary);
    LiveCacheBlockPublication::initialize_naming_source();
    let (context, account) = context();
    let prepared = PreparedLiveCachePublication::prepare(directory.path(), &id, &context).unwrap();
    let staging = prepared.staging_path().to_path_buf();
    let destination = prepared.destination_path().to_path_buf();
    std::fs::write(&staging, b"actual staged payload").unwrap();
    let remaining = *account.remaining.lock().unwrap();
    drop(context);
    let source = prepared.commit().unwrap();
    assert!(!staging.exists());
    assert_eq!(
        std::fs::read(source.path()).unwrap(),
        b"actual staged payload"
    );
    assert!(!account.retired.load(Ordering::SeqCst));
    assert_eq!(*account.remaining.lock().unwrap(), remaining);
    let escaped = source.clone();
    assert!(escaped.same_source(&source));
    drop(source);
    assert!(destination.exists());
    drop(escaped);
    assert!(!destination.exists());
    assert!(account.retired.load(Ordering::SeqCst));

    let (context, account) = crate::cache::worker::test_support::context();
    let prepared = PreparedLiveCachePublication::prepare(directory.path(), &id, &context).unwrap();
    let staging = prepared.staging_path().to_path_buf();
    let destination = prepared.destination_path().to_path_buf();
    std::fs::write(&staging, b"new").unwrap();
    std::fs::write(&destination, b"unrelated existing file").unwrap();
    drop(context);
    let refused = prepared.commit().unwrap_err();
    assert_eq!(refused.path(), destination);
    assert!(!account.retired.load(Ordering::SeqCst));
    assert!(staging.exists());
    assert_eq!(
        std::fs::read(&destination).unwrap(),
        b"unrelated existing file"
    );
    drop(refused);
    assert!(!staging.exists());
    assert_eq!(
        std::fs::read(&destination).unwrap(),
        b"unrelated existing file"
    );
    assert!(account.retired.load(Ordering::SeqCst));
}
