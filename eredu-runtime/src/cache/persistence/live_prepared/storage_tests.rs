use super::*;
use crate::cache::worker::test_support::context;
use crate::cache::{CachePoolLimits, CachePoolUsage, CacheResidencyPool};

#[test]
fn published_file_keeps_exact_disk_reservation_across_aliases_and_failed_commit() {
    let directory = tempfile::tempdir().unwrap();
    LiveCacheBlockPublication::initialize_naming_source();
    let (context, account) = context();
    let shape = [1usize, 1, 2, 1];
    let layout = CacheShardMetadata::prepare(
        CacheRepresentation::KeyValue,
        [&shape, &shape],
        [safetensors::Dtype::U8; 2],
        [2; 2],
        &context,
    )
    .unwrap()
    .into_prepared_layout(&context)
    .unwrap();
    let bytes = u64::try_from(layout.file_bytes()).unwrap();
    let pool = CacheResidencyPool::new(CachePoolLimits::new(1, 1, 1, bytes * 2).unwrap());
    let id = CacheBlockId {
        session_id: 4,
        global_layer: 7,
        representation: CacheRepresentation::KeyValue,
        start: 0,
        end: 2,
        rank: None,
    };
    let write = |publication: &PreparedLiveCachePublication| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(publication.staging_path())
            .unwrap();
        layout.write_to(&mut file, [&[3u8, 5], &[7u8, 11]]).unwrap();
        file.sync_all().unwrap();
    };
    let destination =
        PreparedLiveCachePublication::prepare(directory.path(), &id, &context).unwrap();
    write(&destination);
    let wrong = pool
        .prepare_reservation(&context)
        .unwrap()
        .reserve(CachePoolUsage {
            disk_bytes: bytes - 1,
            ..CachePoolUsage::default()
        })
        .unwrap();
    let staging = destination.staging_path().to_path_buf();
    let failure = destination
        .commit_with_storage(layout.clone(), wrong)
        .unwrap_err();
    assert!(matches!(failure.cause, PublicationCause::Storage(_)));
    assert!(staging.exists());
    assert_eq!(pool.report().unwrap().current_disk_bytes, bytes - 1);
    drop(failure);
    assert!(!staging.exists());
    assert_eq!(pool.report().unwrap().current_disk_bytes, 0);
    let destination =
        PreparedLiveCachePublication::prepare(directory.path(), &id, &context).unwrap();
    write(&destination);
    let reservation = pool
        .prepare_reservation(&context)
        .unwrap()
        .reserve(CachePoolUsage {
            disk_bytes: bytes,
            ..CachePoolUsage::default()
        })
        .unwrap();
    assert_eq!(reservation.remaining_usage().unwrap().disk_bytes, bytes);
    let source = destination
        .commit_with_storage(layout.clone(), reservation)
        .unwrap();
    let path = source.path().to_path_buf();
    let escaped = source.clone();
    drop(source);
    drop(layout);
    drop(context);
    assert!(path.exists());
    assert_eq!(pool.report().unwrap().current_disk_bytes, bytes);
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(!path.exists());
    assert_eq!(pool.report().unwrap().current_disk_bytes, 0);
    // Empty canonical reservation storage is still installed in the real pool.
    // Its metadata account retires with that table, separately from file bytes.
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(pool);
    assert!(account.retired.load(Ordering::SeqCst));
}
