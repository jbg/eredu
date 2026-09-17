use super::*;
use safetensors::tensor::{serialize_to_file, TensorView};

#[test]
fn encoded_pin_keeps_actual_memory_spans_after_lease_and_source_retirement() {
    let selections = [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![2, 0, 2],
        },
        TensorSelection::Contiguous {
            offset_elements: 2,
            shape: vec![2, 2],
        },
    ];
    for selection in selections {
        for policy in [ReadPolicy::RequireBounded, ReadPolicy::AllowFullTensorRead] {
            let store = MemoryWeightStore::from_safetensors([(
                "tensor".into(),
                Dtype::U8,
                vec![2, 3],
                vec![1, 2, 3, 4, 5, 6],
            )])
            .unwrap();
            let lease = store
                .acquire_lease(TensorReadRequest {
                    key: "tensor".into(),
                    selection: selection.clone(),
                    policy,
                })
                .unwrap();
            let expected = lease.encoded_bytes().unwrap().to_vec();
            let pointer = lease.encoded_bytes().unwrap().as_ptr();
            let weak = match &lease {
                CheckpointLease::Memory(lease) => Arc::downgrade(&lease.tensor),
                _ => panic!("memory source"),
            };
            let pin = lease.pin_encoded_bytes().unwrap();
            assert_eq!(pin.encoded_bytes().unwrap().as_ptr(), pointer);
            assert!(pin.backing_path().is_none());
            let clone = pin.clone();
            drop(lease);
            drop(store);
            drop(pin);
            assert!(weak.upgrade().is_some());
            assert_eq!(clone.encoded_bytes().unwrap(), expected);
            assert_eq!(clone.encoded_bytes().unwrap().as_ptr(), pointer);
            drop(clone);
            assert!(weak.upgrade().is_none());
        }
    }
}
fn file(path: &Path, name: &str, bytes: &[u8]) {
    serialize_to_file(
        [(name, TensorView::new(Dtype::U8, vec![2, 3], bytes).unwrap())],
        None,
        path,
    )
    .unwrap();
}
#[test]
fn encoded_pin_preserves_real_safetensors_cache_hits_and_pinned_eviction_refusal() {
    let dir = tempfile::tempdir().unwrap();
    file(
        &dir.path().join("one.safetensors"),
        "one",
        &[1, 2, 3, 4, 5, 6],
    );
    file(
        &dir.path().join("two.safetensors"),
        "two",
        &[11, 12, 13, 14, 15, 16],
    );
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"one":"one.safetensors","two":"two.safetensors"}}"#,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open_with_max_cached_shards(dir.path(), 1).unwrap();
    let request = |key: &str| TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let ordinary = store.acquire(request("one")).unwrap();
    let selected = store
        .acquire(TensorReadRequest {
            key: "one".into(),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![2, 0, 2],
            },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(selected.encoded_bytes().unwrap(), [3, 1, 3, 6, 4, 6]);
    let pointer = selected.encoded_bytes().unwrap().as_ptr();
    let weak = Arc::downgrade(&selected.shard);
    let pin = selected.pin_encoded_bytes();
    let retained = pin.clone();
    let path = pin.backing_path().unwrap().to_path_buf();
    drop(selected);
    drop(ordinary);
    drop(pin);
    assert!(matches!(
        store.acquire(request("two")),
        Err(StoreError::CapacityExhausted { maximum: 1, .. })
    ));
    assert_eq!(store.diagnostics().unwrap().evictions, 0);
    assert_eq!(retained.encoded_bytes().unwrap(), [3, 1, 3, 6, 4, 6]);
    assert_eq!(retained.encoded_bytes().unwrap().as_ptr(), pointer);
    assert_eq!(retained.backing_path(), Some(path.as_path()));
    drop(retained);
    let two = store.acquire(request("two")).unwrap();
    assert_eq!(two.encoded_bytes().unwrap(), [11, 12, 13, 14, 15, 16]);
    assert!(weak.upgrade().is_none());
    assert_eq!(store.diagnostics().unwrap().evictions, 1);
    let again = store.acquire(request("two")).unwrap();
    assert!(Arc::ptr_eq(&two.bytes, &again.bytes));
}
