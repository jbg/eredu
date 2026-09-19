use super::*;
use crate::{schema::*, store::*, validation::resolve_safetensors_plan, StoredDtype};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::sync::atomic::{AtomicUsize, Ordering};

fn memory() -> (Arc<MemoryWeightStore>, u64) {
    let mut first = Vec::with_capacity(257);
    first.extend([7_u8, 9]);
    let mut hidden = Vec::with_capacity(129);
    hidden.extend([11_u8, 13]);
    let capacity = (first.capacity() + hidden.capacity()) as u64;
    (
        Arc::new(
            MemoryWeightStore::from_safetensors([
                ("selected".into(), Dtype::U8, vec![2], first),
                ("hidden".into(), Dtype::U8, vec![2], hidden),
            ])
            .unwrap(),
        ),
        capacity,
    )
}

fn visit(source: &dyn CheckpointSource) -> (bool, Vec<SourceStorageOwner>) {
    let mut owners = Vec::new();
    let complete = source
        .visit_source_storage(&mut |owner| owners.push(owner.retain()))
        .unwrap();
    (complete, owners)
}

fn capacities(owners: &[SourceStorageOwner]) -> BTreeMap<SourceStorageIdentity, u64> {
    let mut unique = BTreeMap::new();
    for owner in owners {
        if let Some(prior) = unique.insert(owner.identity(), owner.bytes()) {
            assert_eq!(prior, owner.bytes());
        }
    }
    unique
}

#[test]
fn borrowed_source_fact_keeps_exact_arc_and_weak_identity_without_wrapper_owner() {
    struct Payload {
        bytes: Vec<u8>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let payload = Arc::new(Payload {
        bytes: vec![3, 5, 7],
        drops: drops.clone(),
    });
    let weak = Arc::downgrade(&payload);
    let reference = SourceStorageRef::new(&payload, 3);
    assert_eq!(Arc::strong_count(&payload), 1);
    let identity = reference.identity();
    assert_eq!(Arc::strong_count(&payload), 1, "identity stays weak");
    let retained = reference.retain();
    assert_eq!(
        Arc::strong_count(&payload),
        2,
        "one clone of the same allocation"
    );
    let erased = retained.as_storage_ref();
    assert_eq!(erased.identity(), identity);
    let alias = erased.retain();
    assert_eq!(Arc::strong_count(&payload), 3);
    let mut legacy = SourceStorage::default();
    legacy.insert(payload.clone(), 3).unwrap();
    assert_eq!(legacy.capacities().next().unwrap(), (identity.clone(), 3));
    drop((legacy, payload, retained));
    assert_eq!(weak.upgrade().unwrap().bytes, [3, 5, 7]);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(alias.bytes(), 3);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(weak.upgrade().is_none());
    assert!(identity.0.upgrade().is_none());
    assert_eq!(identity, identity.clone());
}

#[test]
fn memory_source_visits_actual_spare_capacities_and_retains_payload_after_source_drop() {
    let (source, total) = memory();
    let selected = source.tensors["selected"].ordinary_weak();
    let hidden = source.tensors["hidden"].ordinary_weak();
    let legacy = source.source_storage().unwrap().unwrap();
    let expected: BTreeMap<_, _> = legacy.capacities().collect();
    let (complete, owners) = visit(source.as_ref());
    assert!(complete);
    assert_eq!(owners.len(), 2);
    assert_eq!(capacities(&owners), expected);
    assert_eq!(expected.values().sum::<u64>(), total);
    assert!(total > 4);
    drop((legacy, source));
    assert_eq!(selected.upgrade().unwrap().bytes, [7, 9]);
    assert_eq!(hidden.upgrade().unwrap().bytes, [11, 13]);
    drop(owners);
    assert!(selected.upgrade().is_none());
    assert!(hidden.upgrade().is_none());
}

#[test]
fn authorization_forwarders_visit_hidden_physical_sources_and_repeat_aliases() {
    let (source, total) = memory();
    let weak = source.tensors["hidden"].ordinary_weak();
    let selected: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            source.clone(),
            "selected",
            BTreeSet::from(["selected".into()]),
        )
        .unwrap(),
    );
    let hidden: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(source.clone(), "hidden", BTreeSet::new()).unwrap(),
    );
    assert!(hidden.source_keys().is_empty());
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([selected, hidden]).unwrap());
    assert_eq!(composite.source_keys(), ["selected"]);
    let catalog = composite
        .source_keys()
        .into_iter()
        .map(|key| {
            let value = PreparedTensorSource {
                metadata: composite.source_metadata(&key).unwrap(),
                provenance: composite.source_provenance(&key).unwrap(),
            };
            (key, value)
        })
        .collect();
    let prepared: SharedCheckpointSource =
        Arc::new(PreparedCheckpointSource::new(composite, catalog).unwrap());
    let plan = SafetensorsCheckpointPlan::new(
        "selected only",
        vec![SafetensorsTensorConstraint::required(
            "selected",
            vec![2],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let resolved = resolve_safetensors_plan(prepared.as_ref(), &plan).unwrap();
    let resolved = ResolvedCheckpointSource::new(prepared, resolved);
    let legacy = resolved.source_storage().unwrap().unwrap();
    let (complete, owners) = visit(&resolved);
    assert!(complete);
    assert_eq!(
        owners.len(),
        4,
        "both physical sources visit their complete shared payloads"
    );
    assert_eq!(capacities(&owners), legacy.capacities().collect());
    assert_eq!(capacities(&owners).values().sum::<u64>(), total);
    drop((legacy, resolved, source));
    assert_eq!(weak.upgrade().unwrap().bytes, [11, 13]);
    drop(owners);
    assert!(weak.upgrade().is_none());
}

struct LegacyOnly;
impl CheckpointSource for LegacyOnly {
    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        panic!("legacy fallback")
    }
    fn source_keys(&self) -> Vec<String> {
        Vec::new()
    }
    fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
        panic!("metadata read")
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("payload read")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("diagnostics read")
    }
}

#[test]
fn unknown_default_does_not_call_legacy_and_composite_still_visits_later_sources() {
    let (known, total) = memory();
    let unknown: SharedCheckpointSource = Arc::new(LegacyOnly);
    let view: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(unknown, "unknown", BTreeSet::new()).unwrap(),
    );
    let (complete, owners) = visit(view.as_ref());
    assert!(!complete);
    assert!(owners.is_empty());
    let composite = CompositeCheckpointSource::new([view, known]).unwrap();
    let (complete, owners) = visit(&composite);
    assert!(!complete);
    assert_eq!(owners.len(), 2);
    assert_eq!(
        owners.iter().map(SourceStorageOwner::bytes).sum::<u64>(),
        total
    );
}

#[derive(Clone, Copy)]
enum Outcome {
    Complete,
    Incomplete,
    Error,
    Panic,
}
struct PrefixSource {
    payload: Arc<Vec<u8>>,
    outcome: Outcome,
}
impl CheckpointSource for PrefixSource {
    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        visitor(SourceStorageRef::new(
            &self.payload,
            self.payload.capacity() as u64,
        ));
        match self.outcome {
            Outcome::Complete => Ok(true),
            Outcome::Incomplete => Ok(false),
            Outcome::Error => Err(StoreError::Internal("original source failure".into())),
            Outcome::Panic => std::panic::panic_any(83_u32),
        }
    }
    fn source_keys(&self) -> Vec<String> {
        Vec::new()
    }
    fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
        panic!("metadata read")
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("payload read")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("diagnostics read")
    }
}

#[test]
fn source_error_and_unwind_leave_every_acquired_caller_prefix_alive() {
    for outcome in [Outcome::Incomplete, Outcome::Error, Outcome::Panic] {
        let first = Arc::new(vec![17_u8, 19]);
        let second = Arc::new(vec![23_u8, 29]);
        let weak_first = Arc::downgrade(&first);
        let weak_second = Arc::downgrade(&second);
        let first: SharedCheckpointSource = Arc::new(PrefixSource {
            payload: first,
            outcome: Outcome::Complete,
        });
        let second: SharedCheckpointSource = Arc::new(PrefixSource {
            payload: second,
            outcome,
        });
        let composite = CompositeCheckpointSource::new([first, second]).unwrap();
        let mut prefix = Vec::with_capacity(2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            composite.visit_source_storage(&mut |owner| prefix.push(owner.retain()))
        }));
        match outcome {
            Outcome::Incomplete => assert!(!result.unwrap().unwrap()),
            Outcome::Error => assert!(
                matches!(result.unwrap(), Err(StoreError::Internal(message)) if message == "original source failure")
            ),
            Outcome::Panic => assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 83),
            Outcome::Complete => unreachable!(),
        }
        assert_eq!(prefix.len(), 2);
        drop(composite);
        assert_eq!(weak_first.upgrade().unwrap().as_slice(), [17, 19]);
        assert_eq!(weak_second.upgrade().unwrap().as_slice(), [23, 29]);
        drop(prefix);
        assert!(weak_first.upgrade().is_none());
        assert!(weak_second.upgrade().is_none());
    }
}

#[test]
fn zero_and_large_source_facts_remain_per_owner_without_duplicate_summation() {
    let empty_payload = Arc::new([0_u8; 0]);
    let zero = SourceStorageRef::new(&empty_payload, 0);
    assert_eq!(zero.bytes(), 0);
    assert_eq!(zero.retain().bytes(), 0);
    // Synthetic conservative ceiling exercises only the neutral fact container;
    // it is never accepted as an execution or allocator authority.
    let payload = Arc::new(vec![31_u8]);
    let large = SourceStorageRef::new(&payload, u64::MAX);
    let first = large.retain();
    let second = large.retain();
    assert_eq!(first.identity(), second.identity());
    assert_eq!(
        capacities(&[first, second])
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [u64::MAX]
    );
    let empty = MemoryWeightStore::default();
    let (complete, owners) = visit(&empty);
    assert!(complete);
    assert!(owners.is_empty());
}

#[test]
fn safetensors_visitor_is_complete_empty_without_owning_external_payload_leases() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("weights.safetensors");
    let bytes = [37_u8, 41];
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::U8, vec![2], &bytes).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let source = SafetensorsWeightStore::open(&path).unwrap();
    let lease = source
        .acquire_lease(TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let before = source.source_diagnostics().unwrap().physical_reads;
    std::fs::remove_file(path).unwrap();
    let (complete, owners) = visit(&source);
    assert!(complete);
    assert!(owners.is_empty());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, before);
    assert_eq!(source.source_storage().unwrap().unwrap().owner_count(), 0);
    drop(source);
    assert_eq!(lease.encoded_bytes().unwrap(), bytes);
}

mod slot_bounds;
