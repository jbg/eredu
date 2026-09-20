use super::tests::{fixture, warm};
use super::*;
use crate::{schema::*, validation::resolve_safetensors_plan};
use std::sync::atomic::{AtomicUsize, Ordering};

fn restricted(source: RetainedCheckpointSource, keys: &[&str]) -> RetainedCheckpointSource {
    Arc::new(
        RestrictedCheckpointSource::including(
            source,
            "selected view",
            keys.iter().map(|key| String::from(*key)).collect(),
        )
        .unwrap(),
    )
    .into()
}
fn snapshot(source: RetainedCheckpointSource) -> RetainedCheckpointSource {
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            let row = PreparedTensorSource {
                metadata: source.source_metadata(&key).unwrap(),
                provenance: source.source_provenance(&key).unwrap(),
            };
            (key, row)
        })
        .collect();
    Arc::new(PreparedCheckpointSource::new(source, catalog).unwrap()).into()
}
fn resolved(source: RetainedCheckpointSource) -> RetainedCheckpointSource {
    let plan = SafetensorsCheckpointPlan::new(
        "selected contract",
        vec![
            SafetensorsTensorConstraint::required(
                "a",
                vec![2],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            ),
            SafetensorsTensorConstraint::required(
                "c",
                vec![3],
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            ),
        ],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let contract = resolve_safetensors_plan(source.as_ref(), &plan).unwrap();
    Arc::new(ResolvedCheckpointSource::new(source, contract)).into()
}
fn composite(leaf: &Arc<SafetensorsWeightStore>) -> RetainedCheckpointSource {
    RetainedCheckpointSource::from_composite(
        CompositeCheckpointSource::new([
            restricted(leaf.clone().into(), &["a", "c"]),
            restricted(leaf.clone().into(), &["b"]),
        ])
        .unwrap(),
    )
}

#[test]
fn file_read_through_nested_views_owns_exact_bytes_after_views_retire() {
    let (_directory, leaf) = fixture();
    let leaf = Arc::new(leaf);
    let root = resolved(snapshot(composite(&leaf)));
    let keys = ["c".into(), "a".into(), "c".into()];
    let plan = SafetensorsEncodedReadPlan::from_source(&root, &keys)
        .unwrap()
        .unwrap();
    let custody = Arc::new(());
    let held = Arc::downgrade(&custody);
    let read = plan.construct(custody).unwrap();
    assert_eq!(leaf.diagnostics().unwrap().physical_reads, 0);
    assert!(leaf.diagnostics().unwrap().payload_shard_paths.is_empty());
    drop((root, leaf, keys));
    let mut output = [0; 8];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, [3, 7, 11, 13, 14, 3, 7, 11]);
    assert!(held.upgrade().is_some());
    drop(read);
    assert!(held.upgrade().is_none());
}

#[test]
fn file_view_authorization_precedes_header_access_and_keeps_batch_error_order() {
    let (_directory, leaf) = fixture();
    let leaf = Arc::new(leaf);
    let selected = restricted(leaf.clone().into(), &["a", "c"]);
    let allowed = ["a".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::from_source(&selected, &allowed),
        Err(SafetensorsEncodedReadPlanError::HeaderUnavailable { index: 0 })
    ));
    let denied = ["b".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::from_source(&selected, &denied),
        Err(SafetensorsEncodedReadPlanError::UnauthorizedTensor {
            index: 0,
            contract: "selected view"
        })
    ));
    for path in leaf.shards.payload_paths() {
        assert_eq!(
            leaf.shards
                .admission(path)
                .header_reads
                .load(Ordering::Relaxed),
            0
        );
    }
    warm(&leaf);
    for root in [
        selected.clone(),
        snapshot(selected),
        resolved(leaf.clone().into()),
        resolved(snapshot(composite(&leaf))),
    ] {
        for key in ["b", "missing"] {
            let keys = ["a".into(), key.into()];
            assert!(matches!(
                root.prepare_encoded_read(&keys),
                Err(StoreError::UnauthorizedTensor { .. })
            ));
            assert!(matches!(
                SafetensorsEncodedReadPlan::from_source(&root, &keys),
                Err(SafetensorsEncodedReadPlanError::UnauthorizedTensor { index: 1, .. })
            ));
        }
    }
    let root = composite(&leaf);
    for keys in [
        vec![],
        vec!["a".into(), "b".into()],
        vec!["a".into(), "b".into(), "missing".into()],
    ] {
        assert!(root.prepare_encoded_read(&keys).unwrap().is_none());
        assert!(
            SafetensorsEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .is_none()
        );
    }
    let keys = ["a".into(), "missing".into(), "b".into()];
    assert!(
        matches!(root.prepare_encoded_read(&keys), Err(StoreError::UnknownTensor { key }) if key == "missing")
    );
    assert!(matches!(
        SafetensorsEncodedReadPlan::from_source(&root, &keys),
        Err(SafetensorsEncodedReadPlanError::UnknownTensor { index: 1 })
    ));
}

struct Forwarded {
    leaf: Arc<SafetensorsWeightStore>,
    reads: Arc<AtomicUsize>,
    loans: Arc<AtomicUsize>,
}
impl CheckpointSource for Forwarded {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        self.leaf.clone().prepared_acquisition_owner()
    }
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        self.loans.fetch_add(1, Ordering::SeqCst);
        self.leaf.prepared_acquisition_source()
    }
    fn source_keys(&self) -> Vec<String> {
        self.leaf.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.leaf.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.leaf.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.leaf.source_diagnostics()
    }
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        CheckpointSource::recipe_cache(self.leaf.as_ref())
    }
    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.leaf.prepare_encoded_read(keys)
    }
}

#[test]
fn forwarded_file_owner_is_refused_before_borrowed_loan_or_ordinary_read() {
    let (_directory, leaf) = fixture();
    let reads = Arc::new(AtomicUsize::new(0));
    let loans = Arc::new(AtomicUsize::new(0));
    let source: RetainedCheckpointSource = Arc::new(Forwarded {
        leaf: Arc::new(leaf),
        reads: reads.clone(),
        loans: loans.clone(),
    })
    .into();
    let keys = ["a".into()];
    assert!(source.prepare_encoded_read(&keys).unwrap().is_some());
    let before = reads.load(Ordering::SeqCst);
    for root in [
        source.clone(),
        snapshot(source.clone()),
        restricted(source.clone(), &["a"]),
        resolved(source),
    ] {
        assert!(
            SafetensorsEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(reads.load(Ordering::SeqCst), before);
    assert_eq!(loans.load(Ordering::SeqCst), 0);
}

#[test]
fn routed_failed_header_is_the_original_source_owned_error() {
    let (_directory, leaf) = fixture();
    let leaf = Arc::new(leaf);
    let path = leaf.catalog["a"].shard.clone();
    std::fs::remove_file(&path).unwrap();
    assert!(WeightStore::metadata(leaf.as_ref(), "a").is_err());
    let root = restricted(leaf.clone().into(), &["a"]);
    let keys = ["a".into()];
    let Err(error) = SafetensorsEncodedReadPlan::from_source(&root, &keys) else {
        panic!("expected retained header error")
    };
    let original = leaf
        .shards
        .admission(&path)
        .header
        .get()
        .unwrap()
        .as_ref()
        .unwrap_err();
    let cause = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<StoreError>()
        .unwrap();
    assert!(std::ptr::eq(cause, original));
    assert_eq!(
        leaf.shards
            .admission(&path)
            .header_reads
            .load(Ordering::Relaxed),
        1
    );
}
