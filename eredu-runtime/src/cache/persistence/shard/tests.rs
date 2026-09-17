use super::*;
use crate::cache::worker::test_support::context;
use safetensors::tensor::TensorView;

#[test]
fn shared_cache_writer_matches_upstream_bytes_and_retains_paid_layout_through_file_and_aliases() {
    let directory = tempfile::tempdir().unwrap();
    let payloads = [[1u8, 0, 2, 0, 3, 0], [11u8, 13, 17, 19, 23, 29]];
    let shapes = [[1usize, 3], [2usize, 3]];
    let dtypes = [Dtype::I16, Dtype::U8];
    let expected = safetensors::tensor::serialize(
        [
            (
                "keys",
                TensorView::new(dtypes[0], shapes[0].to_vec(), &payloads[0]).unwrap(),
            ),
            (
                "values",
                TensorView::new(dtypes[1], shapes[1].to_vec(), &payloads[1]).unwrap(),
            ),
        ],
        None,
    )
    .unwrap();
    let ordinary = CacheShardMetadata::ordinary(
        CacheRepresentation::KeyValue,
        [&shapes[0], &shapes[1]],
        dtypes,
        [6, 6],
    )
    .unwrap()
    .into_ordinary_layout()
    .unwrap();
    let (source_context, source_account) = context();
    let (header_context, header_account) = context();
    let source = CacheShardMetadata::prepare(
        CacheRepresentation::KeyValue,
        [&shapes[0], &shapes[1]],
        dtypes,
        [6, 6],
        &source_context,
    )
    .unwrap();
    let declared = source.layout_control_bytes().unwrap();
    assert!(declared > expected.len());
    let layout = source.into_prepared_layout(&header_context).unwrap();
    assert_eq!(layout.header(), ordinary.header());
    assert_eq!(layout.file_bytes(), expected.len());
    let id = CacheBlockId {
        session_id: 19,
        global_layer: 1,
        representation: CacheRepresentation::KeyValue,
        start: 0,
        end: 3,
        rank: None,
    };
    LiveCacheBlockPublication::initialize_naming_source();
    let publication =
        PreparedLiveCachePublication::prepare(directory.path(), &id, &header_context).unwrap();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(publication.staging_path())
        .unwrap();
    assert!(matches!(
        layout.write_to(&mut file, [&payloads[0][..5], &payloads[1]]),
        Err(CacheShardError::Payload)
    ));
    assert_eq!(file.metadata().unwrap().len(), 0);
    layout
        .write_to(&mut file, [&payloads[0], &payloads[1]])
        .unwrap();
    file.sync_all().unwrap();
    drop(file);
    let source = publication.commit_with_layout(layout.clone()).unwrap();
    let path = source.path().to_path_buf();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes, expected);
    let values = source.writer_layout().unwrap().tensors(&bytes).unwrap();
    for i in 0..2 {
        assert_eq!(values[i].data(), &payloads[i]);
        assert_eq!(values[i].shape(), shapes[i]);
        assert_eq!(values[i].dtype(), dtypes[i]);
    }
    let mut changed = bytes.clone();
    changed[8] ^= 1;
    assert!(matches!(
        layout.tensors(&changed),
        Err(CacheShardError::Header)
    ));
    drop(source_context);
    drop(header_context);
    drop(source);
    assert!(!path.exists());
    assert!(!source_account.retired.load(Ordering::SeqCst));
    assert!(!header_account.retired.load(Ordering::SeqCst));
    drop(layout);
    assert!(source_account.retired.load(Ordering::SeqCst));
    assert!(header_account.retired.load(Ordering::SeqCst));
}
