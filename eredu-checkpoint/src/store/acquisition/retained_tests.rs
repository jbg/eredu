use super::*;
use crate::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, TensorOperation,
};
use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};
use std::sync::atomic::{AtomicUsize, Ordering};

fn source(path: &Path, name: &str, value: f32) -> crate::gguf_store::GgufWeightStore {
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
    crate::gguf_store::GgufWeightStore::builder()
        .add_checkpoint(checkpoint, &plan, &mapping)
        .unwrap()
        .build()
        .unwrap()
}
#[derive(Debug)]
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn request() -> TensorReadRequest {
    TensorReadRequest {
        key: "weight".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
#[test]
fn retained_root_g4_success_and_refusal_keep_same_owner_through_last_identity() {
    let dir = tempfile::tempdir().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let root = RetainedCheckpointSource::from_gguf_with_custody(
        source(&dir.path().join("source.gguf"), "weight", 7.25),
        Custody(drops.clone()),
    );
    let identity = root.identity();
    let alias_identity = identity.clone();
    assert_eq!(identity, root.clone().identity());
    let read = request();
    let plan = SelectedGgufConversionPlan::query_retained(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(plan.matches_retained_request(&root, &read));
    assert!(plan.acquisition_storage().unwrap().is_some());
    let prepared = plan.prepare_acquisition().unwrap().unwrap();
    assert!(prepared.matches_retained_request(&root, &read));
    let mut refused = plan.prepare_acquisition().unwrap().unwrap();
    // Real same-destination defense: the owned destination still names weight.
    refused.request.key = "missing".into();
    let bad_request = refused.request.clone();
    let failed = refused.acquire().unwrap_err();
    assert!(failed.matches_retained_request(&root, &bad_request));
    assert!(failed.pending.is_some());
    assert_eq!(
        root.as_ref().source_diagnostics().unwrap().physical_reads,
        0
    );
    drop(plan);
    drop(root);
    let lease = prepared.acquire().unwrap();
    let CheckpointLease::Gguf(lease) = lease else {
        panic!("GGUF")
    };
    let eredu_gguf::ConvertedTensor::Dense(tensor) =
        lease.materialize_portable().unwrap().into_converted()
    else {
        panic!("dense")
    };
    let values: Vec<f32> = tensor
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(values, [7.25, -3.5]);
    drop(lease);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(failed);
    // Last strong source retired, but both opaque identities retain its shell.
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(identity);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(alias_identity);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct Custom {
    source: Arc<crate::gguf_store::GgufWeightStore>,
    calls: Arc<AtomicUsize>,
}
impl CheckpointSource for Custom {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.source.clone().prepared_acquisition_owner()
    }
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        self.source.prepared_acquisition_source()
    }
    fn source_keys(&self) -> Vec<String> {
        self.source.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.source.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }
}
#[test]
fn ordinary_retained_adapter_preserves_arc_identity_and_rejects_forwarded_custom_owner() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let ordinary: SharedCheckpointSource = Arc::new(Custom {
        source: Arc::new(source(&dir.path().join("source.gguf"), "weight", 1.25)),
        calls: calls.clone(),
    });
    let weak = Arc::downgrade(&ordinary);
    let root = RetainedCheckpointSource::from(ordinary.clone());
    assert!(root.matches_ordinary(&ordinary));
    let identity = root.identity();
    let plan = SelectedGgufConversionPlan::query(ordinary.clone(), request())
        .unwrap()
        .unwrap();
    assert!(plan.matches_request(&ordinary, &request()));
    assert!(plan.matches_retained_request(&root, &request()));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(plan.acquisition_storage().unwrap().is_none());
    assert!(plan.prepare_acquisition().unwrap().is_none());
    assert!(root.constructor_control_owner::<Custody>().is_none());
    drop(ordinary);
    drop(root);
    assert!(weak.upgrade().is_some());
    drop(plan);
    assert!(weak.upgrade().is_none());
    drop(identity);
}
