use super::*;
use crate::{schema::*, store::*, validation::resolve_safetensors_plan, StoredDtype};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

#[test]
fn identities_outlive_source_values_without_retaining_them_strongly() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Payload {
        drops: Arc<AtomicUsize>,
        _bytes: Box<[u8; 37]>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(Payload {
        drops: Arc::clone(&drops),
        _bytes: Box::new([7; 37]),
    });
    let mut original = SourceStorage::default();
    original.insert(Arc::clone(&source), 37).unwrap();
    let mut alias = SourceStorage::default();
    alias.insert(source, 37).unwrap();
    let (identity, capacity) = original.capacities().next().unwrap();
    let (alias_identity, alias_capacity) = alias.capacities().next().unwrap();
    assert_eq!(capacity, alias_capacity);
    assert_eq!(identity, alias_identity);
    assert_eq!(identity.0.strong_count(), 2);
    let copy = identity.clone();
    drop(original);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(identity.0.upgrade().is_none());
    assert_eq!(identity, alias_identity);
    assert_eq!(identity.cmp(&copy), std::cmp::Ordering::Equal);

    let mut deferred = BTreeMap::new();
    deferred.insert(identity.clone(), capacity);
    assert_eq!(deferred.get(&alias_identity), Some(&37));
    for _ in 0..128 {
        let mut later = SourceStorage::default();
        later.insert(Arc::new([9_u8; 37]), 37).unwrap();
        let (later_identity, _) = later.capacities().next().unwrap();
        assert_ne!(later_identity, identity);
        assert!(!deferred.contains_key(&later_identity));
    }
    drop(identity);
    assert_eq!(deferred.remove(&copy), Some(37));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn identity_aliases_preserve_order_hash_and_capacity_validation() {
    use std::collections::HashSet;

    let owner = Arc::new(vec![7_u8, 9]);
    let mut original = SourceStorage::default();
    original.insert(Arc::clone(&owner), 2).unwrap();
    let shared = original.clone();
    let identity = original.capacities().next().unwrap().0;
    let alias = shared.capacities().next().unwrap().0;
    assert_eq!(HashSet::from([identity.clone(), alias.clone()]).len(), 1);
    assert_eq!(BTreeSet::from([identity.clone(), alias]).len(), 1);

    let mut conflicting = SourceStorage::default();
    conflicting.insert(Arc::clone(&owner), 3).unwrap();
    assert!(original.merge(conflicting).is_err());
    assert_eq!(original.bytes().unwrap(), 2);
    assert_eq!(original.capacities().next().unwrap().0, identity);
    original.merge(shared).unwrap();
    assert_eq!(original.owner_count(), 1);
    assert_eq!(original.bytes().unwrap(), 2);

    drop(original);
    drop(owner);
    let copied = identity.clone();
    assert!(identity.0.upgrade().is_none());
    assert_eq!(HashSet::from([identity, copied]).len(), 1);
}

fn memory() -> (Arc<MemoryWeightStore>, u64) {
    let mut first = Vec::with_capacity(257);
    first.extend([7, 9]);
    let mut second = Vec::with_capacity(129);
    second.extend([11, 13]);
    let capacity = (first.capacity() + second.capacity()) as u64;
    (
        Arc::new(
            MemoryWeightStore::from_safetensors([
                ("selected".into(), Dtype::U8, vec![2], first),
                ("hidden".into(), Dtype::U8, vec![2], second),
            ])
            .unwrap(),
        ),
        capacity,
    )
}

#[test]
fn restricted_prepared_and_resolved_views_keep_full_physical_capacity_once() {
    let (source, capacity) = memory();
    let weak = Arc::downgrade(&source.tensors["selected"]);
    let target: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            source.clone(),
            "target",
            BTreeSet::from(["selected".into()]),
        )
        .unwrap(),
    );
    let extension: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            source.clone(),
            "extension",
            BTreeSet::from(["hidden".into()]),
        )
        .unwrap(),
    );
    let combined: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([target, extension]).unwrap());
    let catalog = combined
        .source_keys()
        .into_iter()
        .map(|key| {
            let item = PreparedTensorSource {
                metadata: combined.source_metadata(&key).unwrap(),
                provenance: combined.source_provenance(&key).unwrap(),
            };
            (key, item)
        })
        .collect();
    let prepared: SharedCheckpointSource =
        Arc::new(PreparedCheckpointSource::new(combined, catalog).unwrap());
    let contract = SafetensorsCheckpointPlan::new(
        "one selected tensor",
        vec![SafetensorsTensorConstraint::required(
            "selected",
            vec![2],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let resolved = resolve_safetensors_plan(prepared.as_ref(), &contract).unwrap();
    let resolved = ResolvedCheckpointSource::new(prepared, resolved);
    assert_eq!(resolved.source_keys(), ["selected"]);
    let storage = SourceStorage::collect([&resolved as &dyn CheckpointSource, source.as_ref()])
        .unwrap()
        .unwrap();
    assert_eq!(storage.bytes().unwrap(), capacity);
    assert_eq!(storage.owner_count(), 2);
    assert!(capacity > 4, "spare capacity must be included");
    drop(resolved);
    drop(source);
    assert!(
        weak.upgrade().is_some(),
        "bound retains its physical identity"
    );
    let shared = storage.clone();
    drop(storage);
    assert!(weak.upgrade().is_some());
    drop(shared);
    assert!(weak.upgrade().is_none());
}

struct UnknownStorage(Arc<MemoryWeightStore>);
impl CheckpointSource for UnknownStorage {
    fn source_keys(&self) -> Vec<String> {
        self.0.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.0.source_metadata(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("storage inspection read a payload")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.0.source_diagnostics()
    }
}

#[test]
fn unreported_source_storage_remains_unknown_through_authorization_views() {
    let (source, _) = memory();
    let known = source.source_storage().unwrap().unwrap();
    assert!(known.bytes().unwrap() > 0);
    let unknown: SharedCheckpointSource = Arc::new(UnknownStorage(source));
    let view = RestrictedCheckpointSource::including(
        unknown.clone(),
        "one",
        BTreeSet::from(["selected".into()]),
    )
    .unwrap();
    assert!(
        SourceStorage::collect([&view as &dyn CheckpointSource, unknown.as_ref()])
            .unwrap()
            .is_none()
    );
}

#[test]
fn file_source_quote_neither_reads_payloads_nor_claims_external_lease_storage() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("model.safetensors");
    let bytes = [7u8, 9];
    serialize_to_file(
        [(
            "selected",
            TensorView::new(Dtype::U8, vec![2], &bytes).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let source = SafetensorsWeightStore::open(&path).unwrap();
    let before = source.source_diagnostics().unwrap();
    assert_eq!(
        source.source_storage().unwrap().unwrap().bytes().unwrap(),
        0
    );
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        before.physical_reads
    );
    let lease = source
        .acquire_lease(TensorReadRequest {
            key: "selected".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let after = source.source_diagnostics().unwrap().physical_reads;
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        source.source_storage().unwrap().unwrap().bytes().unwrap(),
        0
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, after);
    assert_eq!(lease.encoded_bytes().unwrap(), bytes);
}

#[test]
fn source_storage_rejects_conflicting_alias_bounds_and_checked_sum_overflow() {
    let owner = Arc::new(vec![7u8]);
    let mut first = SourceStorage::default();
    first.insert(owner.clone(), 1).unwrap();
    let mut alias = SourceStorage::default();
    alias.insert(owner.clone(), 2).unwrap();
    assert!(first.merge(alias).is_err());
    assert_eq!(first.bytes().unwrap(), 1);
    let mut overflowing = SourceStorage::default();
    overflowing.insert(owner, u64::MAX).unwrap();
    overflowing.insert(Arc::new(vec![9u8]), 1).unwrap();
    assert!(matches!(
        overflowing.bytes(),
        Err(StoreError::Overflow { .. })
    ));
}
