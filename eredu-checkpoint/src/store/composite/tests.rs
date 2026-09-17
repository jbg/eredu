use super::*;
use crate::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, TensorOperation,
};
use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};
use std::sync::atomic::{AtomicUsize, Ordering};

fn source(path: &Path, name: &str, value: f32) -> Arc<GgufWeightStore> {
    let bytes = [value.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
    Writer::default()
        .write(
            File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name,
                dimensions: &[2],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
    let checkpoint = Checkpoint::open(path).unwrap();
    let plan = GgufCheckpointPlan::new(
        "composite",
        vec![GgufTensorConstraint::required(
            name,
            vec![2],
            GgufTypeConstraint::OperationClass(TensorOperation::Dense),
        )],
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .unwrap();
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    Arc::new(
        GgufWeightStore::builder()
            .add_checkpoint(checkpoint, &plan, &mapping)
            .unwrap()
            .build()
            .unwrap(),
    )
}
#[derive(Debug)]
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn closed_composite_duplicate_and_failed_prefix_keep_original_sources_and_custody() {
    let dir = tempfile::tempdir().unwrap();
    let first = source(&dir.path().join("a.gguf"), "same", 1.25);
    let second = source(&dir.path().join("b.gguf"), "same", 1.25);
    let weak = [Arc::downgrade(&first), Arc::downgrade(&second)];
    let ordinary = CompositeCheckpointSource::new([
        first.clone() as SharedCheckpointSource,
        second.clone() as SharedCheckpointSource,
    ])
    .err()
    .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let failure = GgufCompositePlan::new(first, second)
        .build_with_custody(Custody(drops.clone()))
        .err()
        .unwrap();
    assert_eq!(failure.completed_rows(), 1);
    assert_eq!(failure.duplicate_key(), Some("same"));
    assert!(
        matches!(ordinary, StoreError::Internal(ref message) if message == &failure.to_string())
    );
    assert!(weak.iter().all(|source| source.upgrade().is_some()));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(failure.sources().iter().all(|source| source
        .source_diagnostics()
        .unwrap()
        .physical_reads
        == 0));
    drop(failure);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(weak.iter().all(|source| source.upgrade().is_none()));

    let first = source(&dir.path().join("a.gguf"), "first", 1.25);
    let second = source(&dir.path().join("b.gguf"), "second", 7.25);
    FAIL_AFTER.with(|at| at.set(Some(1)));
    let failure = GgufCompositePlan::new(first, second)
        .build_with_custody(Custody(drops.clone()))
        .err()
        .unwrap();
    FAIL_AFTER.with(|at| at.set(None));
    assert_eq!(failure.completed_rows(), 1);
    assert!(failure.duplicate_key().is_none());
    assert!(std::error::Error::source(&failure)
        .unwrap()
        .is::<TryReserveError>());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(failure);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn closed_composite_ordinary_routing_and_retained_builtin_plan_keep_same_custody() {
    let dir = tempfile::tempdir().unwrap();
    let first = source(&dir.path().join("a.gguf"), "z", 1.25);
    let second = source(&dir.path().join("b.gguf"), "a", 7.25);
    let ordinary = CompositeCheckpointSource::new([
        first.clone() as SharedCheckpointSource,
        second.clone() as SharedCheckpointSource,
    ])
    .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let admitted = GgufCompositePlan::new(first, second)
        .build_with_custody(Custody(drops.clone()))
        .unwrap();
    assert_eq!(admitted.source_keys(), ["a", "z"]);
    assert_eq!(admitted.source_keys(), ordinary.source_keys());
    assert_eq!(
        admitted.source_metadata("a").unwrap(),
        ordinary.source_metadata("a").unwrap()
    );
    assert_eq!(admitted.source_diagnostics().unwrap().physical_reads, 0);
    for (key, expected) in [("z", 1.25_f32), ("a", 7.25_f32)] {
        let CheckpointLease::Gguf(lease) = admitted
            .acquire_lease(TensorReadRequest {
                key: key.into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap()
        else {
            panic!("GGUF lease");
        };
        let eredu_gguf::ConvertedTensor::Dense(tensor) =
            lease.materialize_portable().unwrap().into_converted()
        else {
            panic!("dense GGUF values");
        };
        let values = tensor
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [expected, -3.5]);
    }
    let source: SharedCheckpointSource = Arc::new(admitted);
    let route = SelectedGgufConversionPlan::query(
        source.clone(),
        TensorReadRequest {
            key: "a".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        },
    )
    .unwrap()
    .unwrap();
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(route);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(ordinary.source_keys(), ["a", "z"]);
}

#[test]
fn retained_pair_rejects_erased_children_and_preserves_wrapped_route_custody() {
    let dir = tempfile::tempdir().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let first = RetainedCheckpointSource::from_gguf_with_custody(
        Arc::try_unwrap(source(&dir.path().join("first.gguf"), "first", 1.25)).unwrap(),
        Custody(drops.clone()),
    );
    let first_id = first.identity();
    let ordinary: SharedCheckpointSource = source(&dir.path().join("second.gguf"), "second", 7.25);
    let wrong = RetainedCheckpointSource::from(ordinary.clone());
    let failed = GgufCompositePlan::from_retained(first.clone(), wrong).unwrap_err();
    assert!(failed.sources()[0].same_source(&first));
    assert!(failed.sources()[1].matches_ordinary(&ordinary));
    drop(failed);
    drop(ordinary);
    let second = RetainedCheckpointSource::from_gguf_with_custody(
        Arc::try_unwrap(source(&dir.path().join("second.gguf"), "second", 7.25)).unwrap(),
        Custody(drops.clone()),
    );
    let second_id = second.identity();
    let union = GgufCompositePlan::from_retained(first, second)
        .unwrap()
        .build()
        .unwrap();
    let union = RetainedCheckpointSource::from_composite(union);
    let selected: RetainedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(union, "selected", BTreeSet::from(["second".into()]))
            .unwrap(),
    )
    .into();
    let read = TensorReadRequest {
        key: "second".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let plan = SelectedGgufConversionPlan::query_retained(selected.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(plan.acquisition_storage().unwrap().is_some());
    let ticket = plan.prepare_acquisition().unwrap().unwrap();
    assert!(ticket.matches_retained_request(&selected, &read));
    drop(selected);
    drop(plan);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let CheckpointLease::Gguf(lease) = ticket.acquire().unwrap() else {
        panic!("GGUF")
    };
    let eredu_gguf::ConvertedTensor::Dense(tensor) =
        lease.materialize_portable().unwrap().into_converted()
    else {
        panic!("dense")
    };
    let values: Vec<_> = tensor
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(values, [7.25, -3.5]);
    drop(lease);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(first_id);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(second_id);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}
