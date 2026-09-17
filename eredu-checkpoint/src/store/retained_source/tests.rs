use super::*;
use crate::store::{
    CheckpointLease, MemoryWeightStore, StoreError, TensorMetadata, TensorReadRequest,
    WeightStoreDiagnostics,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Catalog {
    store: MemoryWeightStore,
    retired: Arc<AtomicUsize>,
    acquisition_calls: Arc<AtomicUsize>,
}
impl Drop for Catalog {
    fn drop(&mut self) {
        self.retired.fetch_add(1, Ordering::SeqCst);
    }
}
impl CheckpointSource for Catalog {
    fn source_keys(&self) -> Vec<String> {
        self.store.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.store.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.store.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.store.source_diagnostics()
    }
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        self.acquisition_calls.fetch_add(1, Ordering::SeqCst);
        None
    }
}
#[derive(Debug)]
struct Custody {
    source_retired: Arc<AtomicUsize>,
    retired: Arc<AtomicUsize>,
}
impl Drop for Custody {
    fn drop(&mut self) {
        assert_eq!(self.source_retired.load(Ordering::SeqCst), 1);
        self.retired.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn closed_custom_source_retires_value_before_final_identity_custody() {
    let source_retired = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicUsize::new(0));
    let acquisition_calls = Arc::new(AtomicUsize::new(0));
    let store = MemoryWeightStore::from_safetensors([(
        "weight".into(),
        safetensors::Dtype::F32,
        vec![2],
        [3.25f32.to_le_bytes(), (-7.5f32).to_le_bytes()].concat(),
    )])
    .unwrap();
    let request = RetainedCheckpointSource::source_storage_request::<Catalog, Custody>().unwrap();
    assert_eq!(request.source_body(), Layout::new::<Catalog>());
    assert!(request.control_bytes() > size_of::<RetainedCheckpointSource>());
    let root = RetainedCheckpointSource::from_source_with_custody(
        Catalog {
            store,
            retired: source_retired.clone(),
            acquisition_calls: acquisition_calls.clone(),
        },
        Custody {
            source_retired: source_retired.clone(),
            retired: retired.clone(),
        },
    );
    assert_eq!(root.source_metadata("weight").unwrap().logical_shape, vec![2]);
    assert!(root.constructor_control_owner::<Custody>().is_some());
    assert!(root.gguf().is_none());
    assert!(root.acquisition_owner().is_none());
    assert_eq!(acquisition_calls.load(Ordering::SeqCst), 0);
    let other = RetainedCheckpointSource::from(Arc::new(MemoryWeightStore::default()));
    assert!(!root.same_source(&other));
    let clone = root.clone();
    assert!(root.same_source(&clone));
    let identity = root.identity();
    let second_identity = clone.identity();
    assert_eq!(identity, second_identity);
    drop(root);
    assert_eq!(source_retired.load(Ordering::SeqCst), 0);
    drop(clone);
    assert_eq!(source_retired.load(Ordering::SeqCst), 1);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(identity);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(second_identity);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
