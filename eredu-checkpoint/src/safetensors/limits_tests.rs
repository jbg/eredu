use super::*;
use crate::store::{
    CheckpointSource, SafetensorsWeightStore, SourceMetadataBorrowError, WeightStore,
};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};

fn fixture() -> (tempfile::TempDir, String, u64) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("weights.safetensors");
    serialize_to_file(
        [("weight", TensorView::new(Dtype::U8, vec![1], &[7]).unwrap())],
        None,
        &path,
    )
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let index = r#"{"weight_map":{"weight":"weights.safetensors"}}"#.to_owned();
    std::fs::write(root.path().join("model.safetensors.index.json"), &index).unwrap();
    (root, index, header_len)
}

#[test]
fn index_limit_accepts_exact_bytes_and_refuses_one_less() {
    let (root, index, header_len) = fixture();
    let mut limits = SafetensorsDiscoveryLimits {
        max_index_bytes: index.len() as u64,
        max_header_bytes: header_len,
    };
    let catalog = SafetensorsMetadataCatalog::discover_with_limits(root.path(), limits).unwrap();
    assert_eq!(catalog.tensor("weight").unwrap().logical_shape, [1]);
    assert_eq!(catalog.shards().limits(), limits);
    assert!(
        SafetensorsShards::discover_with_limits(
            root.path(),
            SafetensorsDiscoveryLimits {
                max_index_bytes: u64::MAX,
                ..limits
            }
        )
        .is_ok()
    );
    limits.max_index_bytes -= 1;
    assert!(matches!(
        SafetensorsShards::discover_with_limits(root.path(), limits),
        Err(SafetensorsShardError::IndexTooLarge { limit_bytes, .. })
            if limit_bytes == index.len() as u64 - 1
    ));
    assert!(matches!(
        SafetensorsWeightStore::open_with_limits(root.path(), 1, limits),
        Err(StoreError::SafetensorsShards(
            SafetensorsShardError::IndexTooLarge { .. }
        ))
    ));
}

#[test]
fn default_index_limit_refuses_sparse_oversized_input_before_decoding() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("model.safetensors.index.json");
    let file = std::fs::File::create(&path).unwrap();
    let limit = SafetensorsDiscoveryLimits::default().max_index_bytes;
    file.set_len(limit + 1).unwrap();
    assert!(matches!(SafetensorsShards::discover(root.path()),
        Err(SafetensorsShardError::IndexTooLarge { limit_bytes, .. }) if limit_bytes == limit));
}

#[test]
fn index_read_keeps_its_exact_extent_and_probes_growth() {
    let path = Path::new("index.json");
    let mut input = std::io::Cursor::new(b"abcdef");
    assert!(matches!(
        read_index_bytes(path, &mut input, 3),
        Err(SafetensorsShardError::IndexChanged { .. })
    ));
    assert_eq!(input.position(), 4);
    let mut exact = std::io::Cursor::new(b"abc");
    assert_eq!(read_index_bytes(path, &mut exact, 3).unwrap(), "abc");
    let mut empty = std::io::Cursor::new(b"");
    assert_eq!(read_index_bytes(path, &mut empty, 0).unwrap(), "");
    assert!(matches!(
        read_index_bytes(path, &mut std::io::Cursor::new(b"x"), 0),
        Err(SafetensorsShardError::IndexChanged { .. })
    ));
    assert!(matches!(
        read_index_bytes(path, &mut std::io::Cursor::new([0xff]), 1),
        Err(SafetensorsShardError::Io { .. })
    ));
    assert!(matches!(
        read_index_bytes(path, &mut std::io::Cursor::new(b"ab"), 3),
        Err(SafetensorsShardError::Io { .. })
    ));
    assert!(matches!(
        index_buffer_length(path, u64::MAX),
        Err(SafetensorsShardError::IndexTooLarge { .. })
    ));
}

#[test]
fn lazy_header_limits_survive_shared_admissions_and_retain_failures() {
    let (root, _, header_len) = fixture();
    let limits = SafetensorsDiscoveryLimits {
        max_header_bytes: header_len - 1,
        ..Default::default()
    };
    let shards = SafetensorsShards::discover_catalog(root.path(), limits).unwrap();
    let store = SafetensorsWeightStore::open_admitted(shards.clone(), 1).unwrap();
    let other = SafetensorsWeightStore::open_admitted(shards, 1).unwrap();
    assert!(matches!(
        store.source_metadata_borrowed("weight"),
        Err(SourceMetadataBorrowError::HeaderUnavailable)
    ));
    let message = WeightStore::metadata(&store, "weight")
        .unwrap_err()
        .to_string();
    assert!(message.contains(&format!("header exceeds {} bytes", header_len - 1)));
    let SourceMetadataBorrowError::Retained(first) =
        store.source_metadata_borrowed("weight").unwrap_err()
    else {
        panic!("header failure should be retained");
    };
    let SourceMetadataBorrowError::Retained(second) =
        other.source_metadata_borrowed("weight").unwrap_err()
    else {
        panic!("shared admission should retain the same failure");
    };
    assert!(std::ptr::eq(first, second));
    assert_eq!(
        WeightStore::metadata(&other, "weight")
            .unwrap_err()
            .to_string(),
        message
    );
    assert!(SafetensorsShards::discover_with_limits(root.path(), limits).is_err());
    assert!(
        SafetensorsWeightStore::open_with_limits(
            root.path().join("weights.safetensors"),
            1,
            limits,
        )
        .is_err()
    );
    // A new admission may use a different limit; the refused shared admission stays unchanged.
    let permissive = SafetensorsWeightStore::open_with_limits(
        root.path(),
        1,
        SafetensorsDiscoveryLimits {
            max_header_bytes: header_len,
            ..limits
        },
    )
    .unwrap();
    assert_eq!(
        WeightStore::metadata(&permissive, "weight")
            .unwrap()
            .encoded_byte_len,
        1
    );
}
