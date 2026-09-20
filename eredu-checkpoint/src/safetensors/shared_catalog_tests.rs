use super::*;
use crate::store::{
    EncodedTensorLease, ReadPolicy, SafetensorsWeightStore, TensorReadRequest, TensorSelection,
    WeightStore,
};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    for (file, name, bytes) in [
        ("left.safetensors", "left", [3, 5]),
        ("right.safetensors", "right", [7, 9]),
    ] {
        serialize_to_file(
            [(name, TensorView::new(Dtype::U8, vec![2], &bytes).unwrap())],
            None,
            &root.path().join(file),
        )
        .unwrap();
    }
    std::fs::write(
        root.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"right":"right.safetensors","left":"left.safetensors"}}"#,
    )
    .unwrap();
    root
}

#[test]
fn clones_retain_one_immutable_catalog_and_independent_discovery_is_equal() {
    let root = fixture();
    let limits = SafetensorsDiscoveryLimits::default();
    let original = SafetensorsShards::discover_catalog(root.path(), limits).unwrap();
    let copy = original.clone();
    let weak = Arc::downgrade(&original.catalog);
    assert!(std::ptr::eq(original.payload_paths(), copy.payload_paths()));
    assert!(std::ptr::eq(
        original.logical_payload_paths(),
        copy.logical_payload_paths()
    ));
    assert!(std::ptr::eq(
        original.tensor_locations().unwrap(),
        copy.tensor_locations().unwrap()
    ));
    let independent = SafetensorsShards::discover_catalog(root.path(), limits).unwrap();
    assert_eq!(original, independent);
    assert!(!Arc::ptr_eq(&original.catalog, &independent.catalog));
    drop(independent);
    drop(original);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        copy.tensor_locations()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left", "right"]
    );
    let store = SafetensorsWeightStore::open_admitted(copy, 1).unwrap();
    let lease = store
        .acquire(TensorReadRequest {
            key: "right".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(lease.encoded_bytes().unwrap(), [7, 9]);
    drop(store);
    // A payload lease retains its own file/header owners, not the catalog maps.
    assert!(weak.upgrade().is_none());
    assert_eq!(lease.encoded_bytes().unwrap(), [7, 9]);
}

#[test]
fn owned_path_exports_move_unique_storage_and_copy_shared_storage() {
    let root = fixture();
    let original = SafetensorsShards::discover(root.path()).unwrap();
    let expected = [
        root.path().join("left.safetensors").canonicalize().unwrap(),
        root.path()
            .join("right.safetensors")
            .canonicalize()
            .unwrap(),
    ];
    let mut exported = original.clone().into_payload_paths();
    assert_eq!(exported, expected);
    exported[0].push("changed");
    assert_eq!(original.payload_paths(), expected);
    let original_storage = original.payload_paths().as_ptr();
    let unique = original.into_payload_paths();
    assert_eq!(unique.as_ptr(), original_storage);
    assert_eq!(unique, expected);
}
