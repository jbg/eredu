use super::*;
use crate::{schema::*, validation::resolve_safetensors_plan};
use std::sync::atomic::{AtomicUsize, Ordering};

fn leaf() -> Arc<MemoryWeightStore> {
    Arc::new(
        MemoryWeightStore::from_safetensors([
            (
                "selected".into(),
                Dtype::U8,
                vec![2, 3],
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                "hidden".into(),
                Dtype::U8,
                vec![2, 3],
                vec![9, 8, 7, 6, 5, 4],
            ),
        ])
        .unwrap(),
    )
}
fn restricted(source: RetainedCheckpointSource, key: &str) -> RetainedCheckpointSource {
    Arc::new(
        RestrictedCheckpointSource::including(
            source,
            "selected view",
            BTreeSet::from([key.into()]),
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
        vec![SafetensorsTensorConstraint::required(
            "selected",
            vec![2, 3],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let contract = resolve_safetensors_plan(source.as_ref(), &plan).unwrap();
    Arc::new(ResolvedCheckpointSource::new(source, contract)).into()
}
fn composite(source: &Arc<MemoryWeightStore>) -> RetainedCheckpointSource {
    RetainedCheckpointSource::from_composite(
        CompositeCheckpointSource::new([
            restricted(source.clone().into(), "selected"),
            restricted(source.clone().into(), "hidden"),
        ])
        .unwrap(),
    )
}

#[test]
fn routed_plan_keeps_actual_memory_leaf_after_all_enclosing_views_retire() {
    let leaf = leaf();
    let alive = leaf.tensors["selected"].ordinary_weak();
    let root = resolved(snapshot(composite(&leaf)));
    let keys = ["selected".into(), "selected".into()];
    let plan = MemoryEncodedReadPlan::from_source(&root, &keys)
        .unwrap()
        .unwrap();
    let custody = Arc::new(());
    let held = Arc::downgrade(&custody);
    drop((root, leaf));
    assert!(alive.upgrade().is_some());
    let read = plan.construct(custody).unwrap();
    drop(keys);
    let mut output = [0; 12];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);
    assert!(held.upgrade().is_some());
    drop(read);
    assert!(alive.upgrade().is_none());
    assert!(held.upgrade().is_none());
}

#[test]
fn routed_batch_keeps_outer_authorization_and_composite_error_precedence() {
    let leaf = leaf();
    let restricted = restricted(leaf.clone().into(), "selected");
    let prepared = snapshot(restricted.clone());
    let resolved = resolved(leaf.clone().into());
    let nested = self::resolved(snapshot(composite(&leaf)));
    for root in [restricted, prepared, resolved, nested] {
        for key in ["hidden", "absent"] {
            let keys = ["selected".into(), key.into()];
            assert!(matches!(
                root.prepare_encoded_read(&keys),
                Err(StoreError::UnauthorizedTensor { .. })
            ));
            assert!(matches!(
                MemoryEncodedReadPlan::from_source(&root, &keys),
                Err(MemoryEncodedReadRouteError::UnauthorizedTensor { index: 1 })
            ));
        }
    }
    let root = composite(&leaf);
    for keys in [
        vec![],
        vec!["selected".into(), "hidden".into()],
        vec!["selected".into(), "hidden".into(), "absent".into()],
    ] {
        assert!(root.prepare_encoded_read(&keys).unwrap().is_none());
        assert!(
            MemoryEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .is_none()
        );
    }
    let keys = ["selected".into(), "absent".into(), "hidden".into()];
    assert!(
        matches!(root.prepare_encoded_read(&keys), Err(StoreError::UnknownTensor { key }) if key == "absent")
    );
    assert!(matches!(
        MemoryEncodedReadPlan::from_source(&root, &keys),
        Err(MemoryEncodedReadRouteError::Source(
            MemoryEncodedReadPlanError::UnknownTensor { index: 1 }
        ))
    ));
    let direct: RetainedCheckpointSource = leaf.into();
    let empty = MemoryEncodedReadPlan::from_source(&direct, &[])
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap();
    assert_eq!(empty.byte_len(), 0);
}

struct Forwarded {
    leaf: Arc<MemoryWeightStore>,
    reads: Arc<AtomicUsize>,
}
impl CheckpointSource for Forwarded {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        self.leaf.clone().prepared_acquisition_owner()
    }
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
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
        Some(&self.leaf.recipes)
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
fn matching_metadata_and_forwarded_owner_do_not_authorize_another_root() {
    let leaf = leaf();
    let reads = Arc::new(AtomicUsize::new(0));
    let custom: RetainedCheckpointSource = Arc::new(Forwarded {
        leaf,
        reads: reads.clone(),
    })
    .into();
    let keys = ["selected".into()];
    assert!(custom.prepare_encoded_read(&keys).unwrap().is_some());
    let before = reads.load(Ordering::SeqCst);
    for root in [
        custom.clone(),
        snapshot(custom.clone()),
        restricted(custom.clone(), "selected"),
        resolved(custom.clone()),
    ] {
        assert!(
            MemoryEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(reads.load(Ordering::SeqCst), before);
}

#[test]
fn prepared_route_checks_off_route_cache_dependencies_without_ordinary_fallback() {
    let selected = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "selected".into(),
            Dtype::U8,
            vec![2, 3],
            vec![1, 2, 3, 4, 5, 6],
        )])
        .unwrap(),
    );
    let hidden = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "hidden".into(),
            Dtype::U8,
            vec![2, 3],
            vec![9, 8, 7, 6, 5, 4],
        )])
        .unwrap(),
    );
    let reads = Arc::new(AtomicUsize::new(0));
    let hidden: RetainedCheckpointSource = Arc::new(Forwarded {
        leaf: hidden,
        reads: reads.clone(),
    })
    .into();
    let composite = RetainedCheckpointSource::from_composite(
        CompositeCheckpointSource::new([RetainedCheckpointSource::from(selected), hidden]).unwrap(),
    );
    let keys = ["selected".into()];
    assert!(
        MemoryEncodedReadPlan::from_source(&composite, &keys)
            .unwrap()
            .is_some()
    );
    let prepared = snapshot(composite);
    assert!(prepared.prepare_encoded_read(&keys).unwrap().is_some());
    assert!(
        MemoryEncodedReadPlan::from_source(&prepared, &keys)
            .unwrap()
            .is_none()
    );
    assert_eq!(reads.load(Ordering::SeqCst), 0);
}
