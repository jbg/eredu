use super::*;
use crate::store::*;
use safetensors::tensor::Dtype;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

fn source() -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([
        ("kept".into(), Dtype::U8, vec![2, 3], vec![1, 2, 3, 4, 5, 6]),
        ("hidden".into(), Dtype::U8, vec![1], vec![9]),
    ])
    .unwrap()
}

#[test]
fn borrowed_catalog_views_preserve_actual_owner_and_prepared_snapshot_identity() {
    let memory: SharedCheckpointSource = Arc::new(source());
    let weak = Arc::downgrade(&memory);
    let restricted: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            memory.clone(),
            "selected",
            BTreeSet::from(["kept".into()]),
        )
        .unwrap(),
    );
    let composite = CompositeCheckpointSource::new([restricted.clone()]).unwrap();
    let native = memory.source_metadata_borrowed("kept").unwrap();
    assert!(std::ptr::eq(
        native,
        restricted.source_metadata_borrowed("kept").unwrap()
    ));
    assert!(std::ptr::eq(
        native,
        composite.source_metadata_borrowed("kept").unwrap()
    ));
    assert!(matches!(
        restricted.source_metadata_borrowed("hidden"),
        Err(SourceMetadataBorrowError::UnauthorizedTensor)
    ));
    assert!(matches!(
        composite.source_metadata_borrowed("hidden"),
        Err(SourceMetadataBorrowError::UnknownTensor)
    ));
    assert_eq!(
        composite.source_key_authority_borrowed("absent").unwrap(),
        SourceKeyAuthority::Ordinary
    );
    let prepared = PreparedCheckpointSource::new(
        restricted.clone(),
        BTreeMap::from([(
            "kept".into(),
            PreparedTensorSource {
                metadata: native.clone(),
                provenance: memory.source_provenance("kept").unwrap(),
            },
        )]),
    )
    .unwrap();
    let snapshot = prepared.source_metadata_borrowed("kept").unwrap();
    assert_eq!(snapshot, native);
    assert!(!std::ptr::eq(snapshot, native));
    assert!(std::ptr::eq(snapshot, &prepared.catalog["kept"].metadata));
    drop((memory, restricted, composite));
    assert!(weak.upgrade().is_some());
    assert_eq!(
        prepared
            .source_metadata_borrowed("kept")
            .unwrap()
            .logical_shape,
        [2, 3]
    );
    drop(prepared);
    assert!(weak.upgrade().is_none());
}

#[test]
fn absent_borrowed_companion_never_invokes_allocating_source_methods() {
    struct OrdinaryOnly;
    impl CheckpointSource for OrdinaryOnly {
        fn source_keys(&self) -> Vec<String> {
            panic!("ordinary keys")
        }
        fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            panic!("ordinary metadata")
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("ordinary lease")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("ordinary diagnostics")
        }
        fn is_authoritative_materialized_key(&self, _: &str) -> bool {
            panic!("ordinary authority")
        }
    }
    let source: &dyn CheckpointSource = &OrdinaryOnly;
    assert!(matches!(
        source.source_metadata_borrowed("key"),
        Err(SourceMetadataBorrowError::Unavailable)
    ));
    assert!(matches!(
        source.source_key_authority_borrowed("key"),
        Err(SourceMetadataBorrowError::Unavailable)
    ));
}

#[test]
fn selected_copy_layouts_match_real_nonzero_leases_for_every_selection_variant() {
    let source = source();
    let cases = [
        (TensorSelection::Full, vec![1, 2, 3, 4, 5, 6]),
        (
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
            vec![4, 5, 6],
        ),
        (
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
            vec![4, 5, 6, 1, 2, 3, 4, 5, 6],
        ),
        (
            TensorSelection::Contiguous {
                offset_elements: 1,
                shape: vec![2, 2],
            },
            vec![2, 3, 4, 5],
        ),
    ];
    for (selection, bytes) in cases {
        let metadata = source.source_metadata_borrowed("kept").unwrap();
        let before = SelectedMetadataCloneLayout::for_selection(metadata, &selection).unwrap();
        let lease = source
            .acquire(TensorReadRequest {
                key: "kept".into(),
                selection,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.encoded_bytes().unwrap(), bytes);
        assert!(lease.bounded_read_proof().physically_bounded);
        assert_eq!(
            before,
            SelectedMetadataCloneLayout::from_lease(&lease).unwrap()
        );
        assert!(std::ptr::eq(metadata, lease.metadata())); // memory leases share metadata
        assert_eq!(
            before.output_shape.size(),
            std::mem::size_of_val(lease.output_shape())
        );
        let metadata_bytes = metadata.name.len()
            + std::mem::size_of_val(metadata.logical_shape.as_slice())
            + std::mem::size_of_val(metadata.physical_shape.as_slice());
        let selection_bytes = match lease.selection() {
            TensorSelection::Full | TensorSelection::Range { .. } => 0,
            TensorSelection::Indices { indices, .. } => std::mem::size_of_val(indices.as_slice()),
            TensorSelection::Contiguous { shape, .. } => std::mem::size_of_val(shape.as_slice()),
        };
        assert_eq!(
            before.payload_bytes(),
            Some(metadata_bytes + selection_bytes + std::mem::size_of_val(lease.output_shape()))
        );
    }
}

#[test]
fn metadata_clone_layout_keeps_dtype_path_and_empty_fields_distinct() {
    let metadata = TensorMetadata {
        name: "δοκιμή".into(),
        logical_shape: Vec::new(),
        physical_shape: vec![1],
        stored_dtype: crate::StoredDtype::Other("encoded-kind".into()),
        encoded_byte_len: 1,
        backing_shard: Some(std::path::PathBuf::from("shard/é.data")),
    };
    let layout = MetadataCloneLayout::of(&metadata).unwrap();
    assert_eq!(layout.value, Layout::new::<TensorMetadata>());
    assert_eq!(layout.name.size(), metadata.name.len());
    assert_eq!(layout.logical_shape.size(), 0);
    assert_eq!(layout.physical_shape.size(), std::mem::size_of::<usize>());
    assert_eq!(layout.dtype_name.unwrap().size(), "encoded-kind".len());
    assert_eq!(
        layout.backing_path.unwrap().size(),
        metadata
            .backing_shard
            .as_ref()
            .unwrap()
            .as_os_str()
            .as_encoded_bytes()
            .len()
    );
    let scalar =
        SelectedMetadataCloneLayout::for_selection(&metadata, &TensorSelection::Full).unwrap();
    assert_eq!(scalar.output_shape.size(), 0);
    let empty = TensorMetadata {
        name: String::new(),
        stored_dtype: crate::StoredDtype::U8,
        backing_shard: None,
        ..metadata
    };
    let layout = MetadataCloneLayout::of(&empty).unwrap();
    assert_eq!(layout.name.size(), 0);
    assert!(layout.dtype_name.is_none());
    assert!(layout.backing_path.is_none());
}
