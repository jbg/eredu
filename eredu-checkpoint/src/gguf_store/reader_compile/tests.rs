use super::super::tests::test_plan;
use super::*;
use crate::store::WeightStore;
use eredu_gguf::{GgmlType, TensorInput, Writer};
use std::{
    fs::File,
    sync::atomic::{AtomicUsize, Ordering},
};
#[derive(Debug)]
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn append(builder: GgufWeightStoreBuilder, path: &Path, name: &str) -> GgufWeightStoreBuilder {
    let bytes = [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
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
    let plan = test_plan(&checkpoint);
    let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
    builder.add_checkpoint(checkpoint, &plan, &mapping).unwrap()
}
#[test]
fn source_metadata_and_storage_aliases_retain_constructor_through_last_weak_identity() {
    let dir = tempfile::tempdir().unwrap();
    let builder = append(
        GgufWeightStore::builder(),
        &dir.path().join("a.gguf"),
        "first",
    );
    let drops = Arc::new(AtomicUsize::new(0));
    let source = prepare(
        builder,
        Some(SourceControl::new(Custody(drops.clone()))),
        None,
    )
    .unwrap();
    let owner = source
        .inner
        .storage_ref(source.source_storage_bytes().unwrap())
        .retain();
    let identity = owner.identity();
    let metadata = source.inner.readers.lock().unwrap().materializers[0]
        .shared_metadata_source("first")
        .unwrap()
        .unwrap();
    let request = TensorReadRequest {
        key: "first".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let lease = source.acquire(request).unwrap();
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    let value = lease.materialize_portable().unwrap();
    assert_eq!(value.output_names(), ["first"]);
    let metadata_bytes = metadata
        .source()
        .layouts(eredu_gguf::MetadataSelection::Full)
        .unwrap()
        .requested_buffer_bytes();
    drop(value);
    drop(source);
    drop(lease);
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(
        metadata
            .source()
            .layouts(eredu_gguf::MetadataSelection::Full)
            .unwrap()
            .requested_buffer_bytes(),
        metadata_bytes
    );
    drop(metadata);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(identity);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
#[test]
fn source_later_reserve_failure_keeps_input_and_successful_materializer_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let builder = append(
        GgufWeightStore::builder(),
        &dir.path().join("a.gguf"),
        "first",
    );
    let builder = append(builder, &dir.path().join("b.gguf"), "second");
    let original = builder.checkpoints[1].shards()[0].tensors()[0]
        .descriptor()
        .name
        .as_ptr();
    let drops = Arc::new(AtomicUsize::new(0));
    FAIL_AFTER.with(|at| at.set(Some(1)));
    let failure = prepare(
        builder,
        Some(SourceControl::new(Custody(drops.clone()))),
        None,
    )
    .unwrap_err();
    FAIL_AFTER.with(|at| at.set(None));
    assert!(matches!(failure.cause, Cause::Reserve(_)));
    assert!(std::error::Error::source(&failure)
        .unwrap()
        .is::<std::collections::TryReserveError>());
    assert_eq!(failure.completed_materializers(), 1);
    assert_eq!(failure.retained_checkpoints(), 2);
    assert_eq!(
        failure.construction.builder.checkpoints[0].shards()[0].tensors()[0]
            .descriptor()
            .name
            .as_ptr(),
        original
    );
    assert!(
        failure
            .construction
            .buffers
            .as_ref()
            .unwrap()
            .storage_bytes()
            .unwrap()
            > 0
    );
    assert!(failure.construction.materializers[0]
        .open_shard_path()
        .is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(failure);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
