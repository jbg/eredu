use super::*;
use crate::cache::worker::test_support::context;

#[test]
fn published_version_binds_the_actual_file_and_read_failure_keeps_source_and_paid_custody() {
    let directory = tempfile::tempdir().unwrap();
    let id = CacheBlockId {
        session_id: 7,
        global_layer: 1,
        representation: CacheRepresentation::KeyValue,
        start: 2,
        end: 5,
        rank: None,
    };
    let publication = LiveCacheBlockPublication::begin(directory.path(), &id);
    std::fs::write(
        publication.staging_path(),
        b"nonzero published cache source",
    )
    .unwrap();
    let source = publication.commit_owned().unwrap();
    let path = source.path().to_path_buf();
    let foreign = directory.path().join("different-source");
    std::fs::write(&foreign, std::fs::read(&path).unwrap()).unwrap();
    let (context, account) = context();
    let refused = source
        .prepare_read_from(File::open(&foreign).unwrap(), &context)
        .unwrap_err();
    assert!(matches!(
        refused.binding_error(),
        Some(ArtifactFileReadError::Changed)
    ));
    assert!(refused.source_file().same_source(&source));
    drop(refused);
    let read = source
        .prepare_read_from(File::open(&path).unwrap(), &context)
        .unwrap();
    let mut output = vec![0; read.byte_len()];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, b"nonzero published cache source");
    let read = source
        .prepare_read_from(File::open(&path).unwrap(), &context)
        .unwrap();
    let remaining = *account.remaining.lock().unwrap();
    drop(source);
    drop(context);
    assert!(path.exists());
    let failure = read.read_into(&mut [0; 1]).unwrap_err();
    assert_eq!(failure.filled_bytes(), 0);
    assert!(matches!(
        failure.read_error().unwrap().cause(),
        ArtifactFileReadError::DestinationLength
    ));
    assert!(path.exists());
    assert!(!account.retired.load(Ordering::SeqCst));
    assert_eq!(*account.remaining.lock().unwrap(), remaining);
    drop(failure);
    assert!(!path.exists());
    assert!(foreign.exists());
    assert!(account.retired.load(Ordering::SeqCst));
}
