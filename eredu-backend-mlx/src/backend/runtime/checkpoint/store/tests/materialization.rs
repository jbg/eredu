#[test]
fn materializes_full_ranges_and_ordered_indices() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_i32(&path, "matrix", &(0..12).collect::<Vec<_>>(), vec![3, 4]);
    let store = SafetensorsWeightStore::open(&path).unwrap();
    let stream = cpu_stream();

    let full = store
        .acquire("matrix", TensorSelection::Full)
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let outer = store
        .acquire(
            "matrix",
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 3,
            },
        )
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let inner = store
        .acquire(
            "matrix",
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
        )
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let indexed = store
        .acquire(
            "matrix",
            TensorSelection::Indices {
                axis: 0,
                indices: vec![2, 0],
            },
        )
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();

    assert_eq!(
        full.evaluated().unwrap().as_slice::<i32>(),
        &(0..12).collect::<Vec<_>>()
    );
    assert_eq!(outer.shape(), [2, 4]);
    assert_eq!(
        outer.evaluated().unwrap().as_slice::<i32>(),
        &[4, 5, 6, 7, 8, 9, 10, 11]
    );
    assert_eq!(inner.shape(), [3, 2]);
    assert_eq!(
        inner.evaluated().unwrap().as_slice::<i32>(),
        &[1, 2, 5, 6, 9, 10]
    );
    assert_eq!(indexed.shape(), [2, 4]);
    assert_eq!(
        indexed.evaluated().unwrap().as_slice::<i32>(),
        &[8, 9, 10, 11, 0, 1, 2, 3]
    );
}

#[test]
fn axis_zero_range_constructs_only_the_selected_safetensors_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_i32(&path, "bank", &(0..12).collect::<Vec<_>>(), vec![3, 4]);
    let store = SafetensorsWeightStore::open(&path).unwrap();
    let stream = cpu_stream();
    let pending = store
        .acquire(
            "bank",
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
        )
        .unwrap()
        .prepare_materialization(&stream, &stream)
        .unwrap();

    assert_eq!(pending._source.shape(), [1, 4]);
    assert_eq!(pending.output().shape(), [1, 4]);
    assert_eq!(
        pending
            .finish()
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[4, 5, 6, 7]
    );
}

#[test]
fn validates_selection_and_selected_shapes() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(
        &dir.path().join("model.safetensors"),
        "matrix",
        &[0, 1, 2, 3],
        vec![2, 2],
    );
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    assert!(matches!(
        store.acquire("missing", TensorSelection::Full),
        Err(CheckpointMaterializationError::Store(
            StoreError::UnknownTensor { .. }
        ))
    ));
    for selection in [
        TensorSelection::Range {
            axis: 2,
            start: 0,
            end: 1,
        },
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 1,
        },
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 3,
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![],
        },
        TensorSelection::Indices {
            axis: 1,
            indices: vec![2],
        },
    ] {
        assert!(matches!(
            store.acquire("matrix", selection),
            Err(CheckpointMaterializationError::Store(
                StoreError::InvalidSelection { .. }
            ))
        ));
    }
    let lease = store
        .acquire(
            "matrix",
            TensorSelection::Indices {
                axis: 1,
                indices: vec![1, 0, 1],
            },
        )
        .unwrap();
    assert_eq!(lease.output_shape(), [2, 3]);
    assert_eq!(lease.selected_byte_len(), 24);
    assert_eq!(
        store
            .acquire("matrix", TensorSelection::Full)
            .unwrap()
            .selected_byte_len(),
        16
    );
}

#[test]
fn preserves_storage_encodings_and_supports_encoded_fp8() {
    let dir = tempfile::tempdir().unwrap();
    let f16_bytes = [0x00u8, 0x3c, 0x00, 0x40];
    let bf16_bytes = [0x80u8, 0x3f, 0x00, 0x40];
    let fp8_bytes = [0x38u8, 0x40];
    let f16 = TensorView::new(Dtype::F16, vec![2], &f16_bytes).unwrap();
    let bf16 = TensorView::new(Dtype::BF16, vec![2], &bf16_bytes).unwrap();
    let fp8 = TensorView::new(Dtype::F8_E4M3, vec![2], &fp8_bytes).unwrap();
    serialize_to_file(
        [("f16", f16), ("bf16", bf16), ("fp8", fp8)],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    assert_eq!(
        store.source_metadata("f16").unwrap().stored_dtype,
        StoredDtype::F16
    );
    assert_eq!(
        store.source_metadata("bf16").unwrap().stored_dtype,
        StoredDtype::BF16
    );
    assert_eq!(
        store.source_metadata("fp8").unwrap().stored_dtype,
        StoredDtype::F8E4M3
    );
    let stream = cpu_stream();
    let f16 = store
        .acquire("f16", TensorSelection::Full)
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let bf16 = store
        .acquire("bf16", TensorSelection::Full)
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    let fp8 = store
        .acquire("fp8", TensorSelection::Full)
        .unwrap()
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(f16.dtype(), MlxDtype::Float16);
    assert_eq!(bf16.dtype(), MlxDtype::Bfloat16);
    assert_eq!(fp8.dtype(), MlxDtype::Uint8);
    assert_eq!(fp8.evaluated().unwrap().as_slice::<u8>(), &fp8_bytes);
}

#[test]
fn rejects_unsupported_stored_dtype_during_materialization() {
    let dir = tempfile::tempdir().unwrap();
    let encoded = [0x3cu8, 0x40];
    let view = TensorView::new(Dtype::F8_E5M2, vec![2], &encoded).unwrap();
    serialize_to_file(
        [("unsupported", view)],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    assert_eq!(
        store.source_metadata("unsupported").unwrap().stored_dtype,
        StoredDtype::F8E5M2
    );
    let stream = cpu_stream();
    assert!(matches!(
        store
            .acquire("unsupported", TensorSelection::Full)
            .unwrap()
            .materialize(&stream, &stream),
        Err(CheckpointMaterializationError::UnsupportedStoredDtype { .. })
    ));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn materializes_from_cpu_to_metal_execution_stream() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(
        &dir.path().join("model.safetensors"),
        "weight",
        &[10, 20, 30],
        vec![3],
    );
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    let source = cpu_stream();
    let execution = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let value = store
        .acquire("weight", TensorSelection::Full)
        .unwrap()
        .materialize(&source, &execution)
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(value.evaluated().unwrap().as_slice::<i32>(), &[10, 20, 30]);
}
