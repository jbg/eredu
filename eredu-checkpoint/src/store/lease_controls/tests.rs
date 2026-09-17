use super::*;
use crate::store::*;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

fn memory() -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([
        ("kept".into(), Dtype::U8, vec![2, 3], vec![1, 2, 3, 4, 5, 6]),
        ("hidden".into(), Dtype::U8, vec![1], vec![9]),
    ])
    .unwrap()
}
fn prepared(source: SharedCheckpointSource) -> SharedCheckpointSource {
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            let entry = PreparedTensorSource {
                metadata: source.source_metadata(&key).unwrap(),
                provenance: source.source_provenance(&key).unwrap(),
            };
            (key, entry)
        })
        .collect::<BTreeMap<_, _>>();
    Arc::new(PreparedCheckpointSource::new(source, catalog).unwrap())
}

#[test]
fn lease_source_loan_follows_actual_wrappers_and_counts_each_request_copy() {
    let source: SharedCheckpointSource = Arc::new(memory());
    let weak = Arc::downgrade(&source);
    let restriction: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            source.clone(),
            "selected",
            BTreeSet::from(["kept".into()]),
        )
        .unwrap(),
    );
    let union: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([restriction.clone()]).unwrap());
    let first = prepared(union);
    let second = prepared(first.clone());
    let leaf = source.source_lease_controls("kept").unwrap();
    let loan = second.source_lease_controls("kept").unwrap();
    assert!(loan.same_entry(&leaf));
    assert_eq!(loan.provider(), LeaseProvider::Memory);
    assert_eq!(loan.prepared_request_clones(), 2);
    assert_eq!(
        first
            .source_lease_controls("kept")
            .unwrap()
            .prepared_request_clones(),
        1
    );
    assert!(!std::ptr::eq(
        loan.metadata(),
        second.source_metadata_borrowed("kept").unwrap()
    ));
    assert!(matches!(
        restriction.source_lease_controls("hidden"),
        Err(LeaseControlBorrowError::Source(
            SourceMetadataBorrowError::UnauthorizedTensor
        ))
    ));
    assert!(matches!(
        second.source_lease_controls("hidden"),
        Err(LeaseControlBorrowError::Source(
            SourceMetadataBorrowError::UnknownTensor
        ))
    ));
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: vec![1, 0],
    };
    let layout = loan.request_clone_layout(&selection).unwrap();
    assert_eq!(
        layout.payload_bytes(),
        Some(4 + 2 * std::mem::size_of::<usize>())
    );
    let plan = loan.selection_validation(&selection).unwrap();
    let mut output = [0; 2];
    assert_eq!(plan.validate_into(&mut output, &mut []).unwrap(), [2, 3]);
    let lease = second
        .acquire_lease(TensorReadRequest {
            key: "kept".into(),
            selection,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(lease.encoded_bytes().unwrap(), [4, 5, 6, 1, 2, 3]);
    drop((source, restriction, first));
    assert!(weak.upgrade().is_some());
    drop(second);
    assert!(weak.upgrade().is_none()); // lease retains tensor, not source view
    assert_eq!(lease.encoded_bytes().unwrap(), [4, 5, 6, 1, 2, 3]);
}

#[test]
fn actual_memory_and_safetensors_clone_controls_preserve_shared_payload_owners() {
    let source = memory();
    let lease = source
        .acquire_lease(TensorReadRequest {
            key: "kept".into(),
            selection: TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0],
            },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let layout = lease.clone_control_layout().unwrap();
    assert!(layout.metadata.is_none());
    assert!(layout.boxed_lease.is_none());
    assert!(layout.gguf.is_none());
    assert_eq!(
        layout.payload_bytes(),
        Some(4 * std::mem::size_of::<usize>())
    );
    let clone = lease.clone();
    assert!(std::ptr::eq(lease.metadata(), clone.metadata()));
    assert_eq!(
        lease.encoded_bytes().unwrap().as_ptr(),
        clone.encoded_bytes().unwrap().as_ptr()
    );
    assert_ne!(lease.output_shape().as_ptr(), clone.output_shape().as_ptr());
    assert_eq!(clone.clone_control_layout(), Some(layout));

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    serialize_to_file(
        [(
            "kept",
            TensorView::new(Dtype::U8, vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let source = SafetensorsWeightStore::open(&path).unwrap();
    let loan = source.source_lease_controls("kept").unwrap();
    assert_eq!(loan.provider(), LeaseProvider::Safetensors);
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    let lease = source
        .acquire_lease(TensorReadRequest {
            key: "kept".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let before = source.diagnostics().unwrap();
    std::fs::remove_file(path).unwrap();
    let layout = lease.clone_control_layout().unwrap();
    assert_eq!(
        layout.metadata,
        Some(MetadataCloneLayout::of(loan.metadata()).unwrap())
    );
    assert!(layout.boxed_lease.is_none());
    assert!(layout.gguf.is_none());
    let clone = lease.clone();
    assert!(!std::ptr::eq(lease.metadata(), clone.metadata()));
    assert_eq!(
        lease.encoded_bytes().unwrap().as_ptr(),
        clone.encoded_bytes().unwrap().as_ptr()
    );
    let CheckpointLease::Safetensors(inner) = &lease else {
        panic!("SafeTensors lease")
    };
    let payload = Arc::downgrade(&inner.bytes);
    let shard = Arc::downgrade(&inner.shard);
    assert_eq!(
        source.diagnostics().unwrap().physical_reads,
        before.physical_reads
    );
    drop(source);
    drop(lease);
    assert!(payload.upgrade().is_some());
    assert!(shard.upgrade().is_some());
    assert_eq!(clone.encoded_bytes().unwrap(), [1, 2, 3, 4, 5, 6]);
    drop(clone);
    assert!(payload.upgrade().is_none());
    assert!(shard.upgrade().is_none());
}

#[test]
fn lease_source_default_refuses_without_owned_fallback() {
    struct OrdinaryOnly;
    impl CheckpointSource for OrdinaryOnly {
        fn source_keys(&self) -> Vec<String> {
            panic!("owned keys")
        }
        fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            panic!("owned metadata")
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("lease")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("diagnostics")
        }
    }
    assert!(matches!(
        OrdinaryOnly.source_lease_controls("key"),
        Err(LeaseControlBorrowError::Source(
            SourceMetadataBorrowError::Unavailable
        ))
    ));
}
