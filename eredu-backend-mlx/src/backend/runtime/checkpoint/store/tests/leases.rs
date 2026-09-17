#[test]
fn maps_only_acquired_shards_and_reuses_one_mapping() {
    let dir = tempfile::tempdir().unwrap();
    write_two_i32(&dir.path().join("local.safetensors"));
    write_i32(
        &dir.path().join("other.safetensors"),
        "other",
        &[5, 6],
        vec![2],
    );
    write_index(
        dir.path(),
        &[
            ("a_tensor", "local.safetensors"),
            ("z_tensor", "local.safetensors"),
            ("other", "other.safetensors"),
        ],
    );
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    let first = store.acquire("a_tensor", TensorSelection::Full).unwrap();
    let second = store.acquire("z_tensor", TensorSelection::Full).unwrap();
    assert_eq!(first.backing_shard(), second.backing_shard());
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert_eq!(diagnostics.cache_misses, 1);
    assert_eq!(diagnostics.cache_hits, 1);
    assert_eq!(diagnostics.touched_shard_paths.len(), 1);
}

#[test]
fn enforces_capacity_until_leases_drop_then_evicts_lru() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(&dir.path().join("one.safetensors"), "one", &[1], vec![1]);
    write_i32(&dir.path().join("two.safetensors"), "two", &[2], vec![1]);
    write_index(
        dir.path(),
        &[("one", "one.safetensors"), ("two", "two.safetensors")],
    );
    let store = SafetensorsWeightStore::open_with_max_cached_shards(dir.path(), 1).unwrap();
    let one = store.acquire("one", TensorSelection::Full).unwrap();
    let error = store.acquire("two", TensorSelection::Full).unwrap_err();
    assert!(matches!(
        error,
        CheckpointMaterializationError::Store(StoreError::CapacityExhausted { maximum: 1, .. })
    ));
    assert_eq!(one.metadata().logical_shape, [1]);
    let stream = cpu_stream();
    let pinned_value = one
        .materialize(&stream, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(pinned_value.evaluated().unwrap().as_slice::<i32>(), &[1]);
    drop(one);

    let two = store.acquire("two", TensorSelection::Full).unwrap();
    assert_eq!(two.metadata().logical_shape, [1]);
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert_eq!(diagnostics.evictions, 1);
    assert_eq!(diagnostics.touched_shard_paths.len(), 2);
}

#[test]
fn returned_array_survives_lease_and_store_drop() {
    let dir = tempfile::tempdir().unwrap();
    write_i32(
        &dir.path().join("model.safetensors"),
        "weight",
        &[7, 8, 9],
        vec![3],
    );
    let stream = cpu_stream();
    let value = {
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let lease = store.acquire("weight", TensorSelection::Full).unwrap();
        lease
            .materialize(&stream, &stream)
            .unwrap()
            .synchronize()
            .unwrap()
    };
    assert_eq!(value.evaluated().unwrap().as_slice::<i32>(), &[7, 8, 9]);
}

#[test]
fn weight_lease_borrows_exact_source_metadata_and_survives_source_drop() {
    fn addresses(
        metadata: &TensorMetadata,
        selection: &TensorSelection,
        shape: &[usize],
    ) -> [usize; 6] {
        let selection_buffer = match selection {
            TensorSelection::Indices { indices, .. } => indices.as_ptr() as usize,
            TensorSelection::Contiguous { shape, .. } => shape.as_ptr() as usize,
            _ => 0,
        };
        [
            metadata.name.as_ptr() as usize,
            metadata.logical_shape.as_ptr() as usize,
            metadata.physical_shape.as_ptr() as usize,
            metadata.backing_shard.as_ref().map_or(0, |path| {
                path.as_os_str().as_encoded_bytes().as_ptr() as usize
            }),
            selection_buffer,
            shape.as_ptr() as usize,
        ]
    }

    let dir = tempfile::tempdir().unwrap();
    let key = "logical/矩阵";
    let values = (1..=12).map(|value| value as f32).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let safe_path = dir.path().join("metadata.safetensors");
    serialize_to_file(
        [(
            key,
            TensorView::new(Dtype::F32, vec![3, 4], &bytes).unwrap(),
        )],
        None,
        &safe_path,
    )
    .unwrap();
    let gguf_path = dir.path().join("metadata.gguf");
    Writer::default()
        .write(
            std::fs::File::create(&gguf_path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "physical.matrix",
                dimensions: &[4, 3],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
    let stores: Vec<Box<dyn CheckpointSource>> = vec![
        Box::new(SafetensorsWeightStore::open(&safe_path).unwrap()),
        Box::new(
            eredu_checkpoint::store::MemoryWeightStore::from_safetensors([(
                key.into(),
                Dtype::F32,
                vec![3, 4],
                bytes,
            )])
            .unwrap(),
        ),
        Box::new(
            open_gguf_checkpoint_source_for_test(GgufCheckpoint::open(gguf_path).unwrap(), |_| {
                key.into()
            })
            .unwrap(),
        ),
    ];
    let stream = cpu_stream();
    for store in stores {
        let context = MlxParameterMaterializationContext::new(&stream, &stream);
        let cases = [
            (TensorSelection::Full, vec![3, 4], values.clone()),
            (
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 3,
                },
                vec![2, 4],
                values[4..].to_vec(),
            ),
            (
                TensorSelection::Indices {
                    axis: 1,
                    indices: vec![3, 1, 3],
                },
                vec![3, 3],
                vec![4.0, 2.0, 4.0, 8.0, 6.0, 8.0, 12.0, 10.0, 12.0],
            ),
            (
                TensorSelection::Contiguous {
                    offset_elements: 4,
                    shape: vec![2, 2],
                },
                vec![2, 2],
                values[4..8].to_vec(),
            ),
        ];
        let mut retained = Vec::new();
        for (selection, shape, expected) in cases {
            let source = store
                .acquire_lease(TensorReadRequest {
                    key: key.into(),
                    selection: selection.clone(),
                    policy: WeightReadPolicy::RequireBounded,
                })
                .unwrap();
            let original = addresses(source.metadata(), source.selection(), source.output_shape());
            let metadata = source.metadata().clone();
            let source_box = match &source {
                CheckpointLease::Gguf(lease) => Some(std::ptr::from_ref(lease.as_ref()) as usize),
                _ => None,
            };
            let lease = context.weight_lease(source).unwrap();
            let adapted_box = match &lease.source {
                WeightLeaseSource::Gguf(source) => {
                    Some(std::ptr::from_ref(source.lease.as_ref()) as usize)
                }
                _ => None,
            };
            assert_eq!(adapted_box, source_box);
            assert_eq!(
                addresses(lease.metadata(), lease.selection(), lease.output_shape()),
                original
            );
            assert_eq!(lease.key(), key);
            assert_eq!(lease.metadata(), &metadata);
            assert_eq!(lease.selection(), &selection);
            assert_eq!(lease.output_shape(), shape);
            assert_eq!(lease.selected_byte_len(), expected.len() * size_of::<f32>());
            let cloned = lease.clone();
            drop(lease);
            assert_eq!(cloned.metadata(), &metadata);
            assert_eq!(cloned.selection(), &selection);
            retained.push((cloned, shape, expected));
        }
        drop(context);
        drop(store);
        for (lease, shape, expected) in retained {
            let original = addresses(lease.metadata(), lease.selection(), lease.output_shape());
            let original_box = match &lease.source {
                WeightLeaseSource::Gguf(source) => {
                    Some(std::ptr::from_ref(source.lease.as_ref()) as usize)
                }
                _ => None,
            };
            let pending = lease.prepare_materialization(&stream, &stream).unwrap();
            assert_eq!(
                addresses(
                    pending.lease().metadata(),
                    pending.lease().selection(),
                    pending.lease().output_shape()
                ),
                original
            );
            let retained_box = match &pending.lease().source {
                WeightLeaseSource::Gguf(source) => {
                    Some(std::ptr::from_ref(source.lease.as_ref()) as usize)
                }
                _ => None,
            };
            assert_eq!(retained_box, original_box);
            let output = pending.finish().unwrap();
            assert!(output
                .shape()
                .iter()
                .map(|&dimension| usize::try_from(dimension).unwrap())
                .eq(shape));
            assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), expected);
        }
    }
}
