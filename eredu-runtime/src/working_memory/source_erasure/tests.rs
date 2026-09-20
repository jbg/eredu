use super::*;
use eredu_checkpoint::{
    gguf_store::{GgufCatalogPlan, GgufWeightStore},
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    },
    store::{
        CheckpointSource, GgufCompositePlan, ReadPolicy, SelectedGgufConversionPlan,
        SharedCheckpointSource, TensorReadRequest, TensorSelection,
    },
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
};
use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};
use std::{collections::BTreeMap, fs::File, sync::Arc};

struct Input {
    _dir: tempfile::TempDir,
    checkpoint: Checkpoint,
    resolution: ResolvedCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
}
impl Input {
    fn new(name: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.gguf");
        let bytes = [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
        Writer::default()
            .write(
                File::create(&path).unwrap(),
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
        let schema = GgufCheckpointPlan::new(
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
        let resolution = resolve_gguf_plan(&checkpoint, &schema).unwrap();
        let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
        Self {
            _dir: dir,
            checkpoint,
            resolution,
            mapping,
        }
    }
    fn plan(&self) -> GgufCatalogPlan<'_> {
        GgufCatalogPlan::new(self.checkpoint.clone(), &self.resolution, &self.mapping, 1)
    }
    fn ordinary(&self) -> Arc<GgufWeightStore> {
        Arc::new(self.plan().compile(()).unwrap().build().unwrap())
    }
    fn funded(&self, pool: &WorkingMemoryPool) -> GgufWeightStore {
        pool.compile_gguf_source(
            pool.compile_gguf_catalog(self.plan())
                .unwrap()
                .into_prepared(),
        )
        .unwrap()
    }
    fn bytes(&self) -> u64 {
        WorkingMemoryPool::gguf_catalog_required_bytes(&self.plan()).unwrap()
            + WorkingMemoryPool::gguf_source_required_bytes(&self.plan().compile(()).unwrap())
                .unwrap()
    }
}

fn requirement() -> Option<u64> {
    let result = WorkingMemoryPool::gguf_source_erasure_required_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RETAINED_SOURCE").is_some() {
        assert!(
            result.is_ok(),
            "mandatory retained source qualification: {result:?}"
        );
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => {
            assert!(matches!(
                WorkingMemoryPool::gguf_composite_erasure_required_bytes(),
                Err(WorkingMemoryError::UnknownBound)
            ));
            None
        }
        Err(cause) => panic!("unexpected erasure request: {cause}"),
    }
}
fn request(name: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: name.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
#[test]
fn retained_source_exact_one_short_and_last_opaque_identity_refund_once() {
    let Some(bytes) = requirement() else { return };
    let input = Input::new("weight");
    let base = input.bytes();
    let short = WorkingMemoryPool::new(base + bytes - 1, 0).unwrap();
    let original = input.funded(&short);
    let failure = short.retain_gguf_source(original).unwrap_err();
    assert!(
        matches!(failure.accounting_failure(), WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }
        if *required_bytes == bytes && *available_bytes == bytes - 1)
    );
    assert_eq!(
        failure
            .rejected_gguf()
            .unwrap()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        0
    );
    assert_eq!(short.used_bytes().unwrap(), base);
    drop(failure);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(base + bytes, 0).unwrap();
    let root = pool.retain_gguf_source(input.funded(&pool)).unwrap();
    pool.validate_retained_source_controls(&root).unwrap();
    let identity = root.identity();
    let other_identity = root.clone().identity();
    assert_eq!(identity, other_identity);
    let clone = root.clone();
    assert!(root.same_source(&clone));
    assert_eq!(pool.used_bytes().unwrap(), base + bytes);
    std::thread::scope(|scope| {
        scope.spawn(move || drop(root));
        scope.spawn(move || drop(clone));
    });
    // Nested source storage retires at last strong root; only outer shell and
    // its original control remain, independent of the inner catalog account.
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    std::thread::scope(|scope| {
        scope.spawn(move || drop(identity));
        scope.spawn(move || drop(other_identity));
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let root = pool.retain_gguf_source(input.funded(&pool)).unwrap();
    let identity = root.identity();
    let ledger = Arc::downgrade(&pool.0);
    drop(root);
    drop(pool);
    assert!(
        ledger.upgrade().is_none(),
        "opaque identity has no strong pool backedge"
    );
    drop(identity);
    assert!(ledger.upgrade().is_none());
}
#[test]
fn retained_composite_routes_same_root_and_refuses_foreign_or_ordinary_promotion() {
    let Some(_) = requirement() else { return };
    let first = Input::new("first");
    let second = Input::new("second");
    let base = first.bytes() + second.bytes();
    let union = WorkingMemoryPool::gguf_composite_required_bytes(&GgufCompositePlan::new(
        first.ordinary(),
        second.ordinary(),
    ))
    .unwrap();
    let outer = WorkingMemoryPool::gguf_composite_erasure_required_bytes().unwrap();
    let pool = WorkingMemoryPool::new(base + union + outer, 0).unwrap();
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let failed = foreign.retain_gguf_source(first.funded(&pool)).unwrap_err();
    assert!(matches!(
        failed.accounting_failure(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert!(failed.rejected_gguf().is_some());
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), first.bytes());
    drop(failed);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let ordinary = first.plan().compile(()).unwrap().build().unwrap();
    let failed = pool.retain_gguf_source(ordinary).unwrap_err();
    assert!(matches!(
        failed.accounting_failure(),
        WorkingMemoryError::UnknownBound
    ));
    drop(failed);
    // An existing original inner source inside exported Arc does not qualify
    // that Arc's outer allocation merely by moving it into the ordinary variant.
    let ordinary: SharedCheckpointSource = Arc::new(first.funded(&pool));
    let ordinary = RetainedCheckpointSource::from(ordinary);
    assert!(matches!(
        pool.validate_retained_source_controls(&ordinary),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(ordinary);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let union_source = pool
        .compile_gguf_composite(GgufCompositePlan::new(
            Arc::new(first.funded(&pool)),
            Arc::new(second.funded(&pool)),
        ))
        .unwrap();
    let root = pool.retain_gguf_composite(union_source).unwrap();
    pool.validate_retained_source_controls(&root).unwrap();
    assert!(matches!(
        foreign.validate_retained_source_controls(&root),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let identity = root.identity();
    let read = request("second");
    let route = SelectedGgufConversionPlan::query_retained(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(route.matches_retained_request(&root, &read));
    assert!(route.acquisition_storage().unwrap().is_some());
    let prepared = route.prepare_acquisition().unwrap().unwrap();
    assert!(prepared.matches_retained_request(&root, &read));
    assert_eq!(
        root.as_ref().source_diagnostics().unwrap().physical_reads,
        0
    );
    drop(root);
    drop(route);
    assert_eq!(pool.used_bytes().unwrap(), base + union + outer);
    drop(prepared);
    assert_eq!(pool.used_bytes().unwrap(), outer);
    drop(identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

mod safetensors;
