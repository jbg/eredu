use super::*;

#[test]
fn coordinate_index_uses_exact_catalog_names_and_matches_ordinary_nonzero_selected_reads() {
    for captured in [false, true] {
        let (_dir, mut checkpoint) = sharded();
        if captured {
            checkpoint =
                Checkpoint::open_with_prepared_headers(&checkpoint.shards[0].path).unwrap();
        }
        let addresses = allocations(&checkpoint);
        let mut ordinary = checkpoint.materializer();
        assert!(ordinary.prepared_index_layout().is_none());
        let mut prepared = checkpoint.into_shared_coordinate_materializer();
        assert_eq!(allocations(&prepared.checkpoint), addresses);
        let MaterializerIndex::Coordinates(index) = &prepared.locations else {
            panic!("actual coordinate owner")
        };
        let pointer = index.as_ptr();
        let capacity = index.capacity();
        assert_eq!(
            prepared.prepared_index_layout().unwrap().size(),
            capacity * std::mem::size_of::<TensorLocation>()
        );
        for name in ["packed.weight", "dense.weight", "packed.weight"] {
            assert_eq!(
                prepared.shard_source_for_tensor(name).unwrap(),
                ordinary.shard_source_for_tensor(name).unwrap()
            );
            assert_eq!(
                prepared.raw_tensor(name).unwrap(),
                ordinary.raw_tensor(name).unwrap()
            );
            assert_eq!(
                prepared.converted_tensor(name).unwrap(),
                ordinary.converted_tensor(name).unwrap()
            );
            assert_eq!(allocations(&prepared.checkpoint), addresses);
            let MaterializerIndex::Coordinates(index) = &prepared.locations else {
                panic!()
            };
            assert_eq!(index.as_ptr(), pointer);
            assert_eq!(index.capacity(), capacity);
        }
        for selection in [
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
        ] {
            assert_eq!(
                prepared
                    .converted_tensor_selected("dense.weight", &selection)
                    .unwrap(),
                ordinary
                    .converted_tensor_selected("dense.weight", &selection)
                    .unwrap()
            );
        }
        for name in ["dense.scales", "Dense.weight", "dense.weight ", "missing"] {
            let before = prepared.open_shard_path().map(Path::to_path_buf);
            assert_eq!(
                prepared.converted_tensor(name).unwrap_err().to_string(),
                ordinary.converted_tensor(name).unwrap_err().to_string()
            );
            assert_eq!(
                prepared.metadata_source(name).err().unwrap().to_string(),
                ordinary.metadata_source(name).err().unwrap().to_string()
            );
            assert_eq!(prepared.open_shard_path(), before.as_deref());
        }
        let loan = prepared
            .shared_metadata_source("packed.weight")
            .unwrap()
            .unwrap();
        let MaterializerCheckpoint::Shared(owner) = &prepared.checkpoint else {
            panic!()
        };
        let alive = owner.test_alive();
        let expected = loan
            .source()
            .layouts(MetadataSelection::Full)
            .unwrap()
            .requested_buffer_bytes();
        drop(prepared);
        assert!(alive());
        assert_eq!(allocations(loan.test_checkpoint()), addresses);
        assert_eq!(
            loan.source()
                .layouts(MetadataSelection::Full)
                .unwrap()
                .requested_buffer_bytes(),
            expected
        );
        drop(loan);
        assert!(!alive());
    }
}

#[test]
fn coordinate_lookup_preserves_last_source_location_without_normalizing_private_duplicates() {
    let (_dir, mut checkpoint) = sharded();
    // Deliberately bypass public admission only to compare the two index workers.
    // Public duplicate files remain rejected by the shared parser below.
    checkpoint.shards[1].tensors[0].descriptor.name =
        checkpoint.shards[0].tensors[0].descriptor.name.clone();
    let ordinary = checkpoint.materializer();
    let exact = checkpoint
        .clone()
        .try_into_prepared_materializer(())
        .unwrap();
    let prepared = checkpoint.into_coordinate_materializer();
    for name in ["dense.weight", "packed.weight", "DENSE.weight", ""] {
        assert_eq!(
            exact.locations.get(name, &exact.checkpoint),
            ordinary.locations.get(name, &ordinary.checkpoint)
        );
        assert_eq!(
            prepared.locations.get(name, &prepared.checkpoint),
            ordinary.locations.get(name, &ordinary.checkpoint)
        );
    }
    assert_eq!(
        prepared.locations.get("dense.weight", &prepared.checkpoint),
        Some(TensorLocation {
            shard_index: 1,
            tensor_index: 0
        })
    );
}

#[test]
fn physical_duplicate_and_malformed_file_errors_keep_original_parse_and_lookup_precedence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.gguf");
    let bytes = [1.25f32.to_le_bytes(), (-3.5f32).to_le_bytes()].concat();
    Writer::default()
        .write(
            File::create(&path).unwrap(),
            &BTreeMap::new(),
            &[
                TensorInput {
                    name: "first",
                    dimensions: &[2],
                    ggml_type: GgmlType::F32,
                    data: &bytes,
                },
                TensorInput {
                    name: "other",
                    dimensions: &[2],
                    ggml_type: GgmlType::F32,
                    data: &bytes,
                },
            ],
        )
        .unwrap();
    let original = std::fs::read(&path).unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    let mut ordinary = checkpoint.materializer();
    let mut prepared = checkpoint.into_coordinate_materializer();
    let mut duplicate = original.clone();
    let offset = duplicate.windows(5).position(|v| v == b"other").unwrap();
    duplicate[offset..offset + 5].copy_from_slice(b"first");
    std::fs::write(&path, &duplicate).unwrap();
    let Error::Shard { source, .. } = Checkpoint::open(&path).unwrap_err() else {
        panic!()
    };
    assert!(matches!(*source,Error::DuplicateTensor(ref name) if name=="first"));
    assert_eq!(
        prepared.converted_tensor("first").unwrap_err().to_string(),
        ordinary.converted_tensor("first").unwrap_err().to_string()
    );
    assert!(prepared.open_shard_path().is_none());
    std::fs::write(&path, &original[..3]).unwrap();
    assert_eq!(
        prepared
            .converted_tensor("missing")
            .unwrap_err()
            .to_string(),
        ordinary
            .converted_tensor("missing")
            .unwrap_err()
            .to_string()
    );
    assert_eq!(
        prepared.converted_tensor("first").unwrap_err().to_string(),
        ordinary.converted_tensor("first").unwrap_err().to_string()
    );
    std::fs::write(&path, original).unwrap();
    assert_eq!(
        prepared.converted_tensor("first").unwrap(),
        ordinary.converted_tensor("first").unwrap()
    );
}
