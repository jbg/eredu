use super::*;
use eredu_checkpoint::{
    gguf_store::{GgufCatalogPlan, GgufWeightStore},
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    },
    store::{
        CheckpointSource, ReadPolicy, SelectedGgufConversionPlan, SharedCheckpointSource,
        TensorReadRequest, TensorSelection,
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
    fn funded(&self, pool: &WorkingMemoryPool) -> Arc<GgufWeightStore> {
        Arc::new(
            pool.compile_gguf_source(
                pool.compile_gguf_catalog(self.plan())
                    .unwrap()
                    .into_prepared(),
            )
            .unwrap(),
        )
    }
    fn bytes(&self) -> u64 {
        WorkingMemoryPool::gguf_catalog_required_bytes(&self.plan()).unwrap()
            + WorkingMemoryPool::gguf_source_required_bytes(&self.plan().compile(()).unwrap())
                .unwrap()
    }
}
fn requirement(first: &Input, second: &Input) -> Option<u64> {
    let plan = GgufCompositePlan::new(first.ordinary(), second.ordinary());
    let result = WorkingMemoryPool::gguf_composite_required_bytes(&plan);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_COMPOSITE").is_some()
        || std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_SOURCE").is_some()
    {
        assert!(
            result.is_ok(),
            "mandatory composite storage qualification: {result:?}"
        );
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => {
            assert!(matches!(
                WorkingMemoryPool::gguf_composite_required_bytes(&plan),
                Err(WorkingMemoryError::UnknownBound)
            ));
            None
        }
        Err(cause) => panic!("unexpected composite quote failure: {cause}"),
    }
}

#[test]
fn composite_exact_and_one_short_preserve_source_pair_and_last_route_charge() {
    let first = Input::new("first");
    let second = Input::new("second");
    let Some(bytes) = requirement(&first, &second) else {
        return;
    };
    let longer = Input::new("second.with.longer.retained.catalog.key");
    let longer_bytes = requirement(&first, &longer).unwrap();
    assert_eq!(
        longer_bytes - bytes,
        ("second.with.longer.retained.catalog.key".len() - "second".len()) as u64
    );
    let base = first.bytes() + second.bytes();
    let short = WorkingMemoryPool::new(base + bytes - 1, 0).unwrap();
    let plan = GgufCompositePlan::new(first.funded(&short), second.funded(&short));
    let failure = short.compile_gguf_composite(plan).err().unwrap();
    assert!(
        matches!(failure.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded {
        required_bytes, available_bytes }) if *required_bytes == bytes && *available_bytes == bytes - 1)
    );
    assert!(failure.rejected_input().is_some() && failure.construction_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), base);
    drop(failure);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(base + bytes, 0).unwrap();
    let source = pool
        .compile_gguf_composite(GgufCompositePlan::new(
            first.funded(&pool),
            second.funded(&pool),
        ))
        .unwrap();
    pool.validate_gguf_composite_controls(&source).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), base + bytes);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let source: SharedCheckpointSource = Arc::new(source);
    let route = SelectedGgufConversionPlan::query(
        source.clone(),
        TensorReadRequest {
            key: "second".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        },
    )
    .unwrap()
    .unwrap();
    let alias = source.clone();
    drop(source);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), base + bytes);
    drop(route);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn composite_foreign_ordinary_and_duplicate_refusals_never_publish_or_refund_prefix() {
    let first = Input::new("same");
    let second = Input::new("same");
    let Some(bytes) = requirement(&first, &second) else {
        return;
    };
    let base = first.bytes() + second.bytes();
    let pool = WorkingMemoryPool::new(base + bytes, 0).unwrap();
    let other = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let rejected = pool
        .compile_gguf_composite(GgufCompositePlan::new(first.ordinary(), second.ordinary()))
        .err()
        .unwrap();
    assert!(matches!(
        rejected.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(rejected);
    let plan = GgufCompositePlan::new(first.funded(&pool), second.funded(&pool));
    let rejected = other.compile_gguf_composite(plan).err().unwrap();
    assert!(matches!(
        rejected.accounting_failure(),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.used_bytes().unwrap(), base);
    assert_eq!(other.used_bytes().unwrap(), 0);
    drop(rejected);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let failed = pool
        .compile_gguf_composite(GgufCompositePlan::new(
            first.funded(&pool),
            second.funded(&pool),
        ))
        .err()
        .unwrap();
    let prefix = failed.construction_failure().unwrap();
    assert_eq!(prefix.completed_rows(), 1);
    assert_eq!(prefix.duplicate_key(), Some("same"));
    assert!(failed.accounting_failure().is_none());
    assert_eq!(pool.used_bytes().unwrap(), base + bytes);
    assert!(std::error::Error::source(&failed)
        .unwrap()
        .is::<GgufCompositeBuildFailure>());
    drop(failed);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn retained_pair_exact_one_short_and_unfunded_outer_refusal_preserve_child_accounts() {
    use eredu_checkpoint::store::RetainedCheckpointSource;
    let first = Input::new("first");
    let second = Input::new("second");
    let Some(union_bytes) = requirement(&first, &second) else {
        return;
    };
    let leaf_bytes = WorkingMemoryPool::gguf_source_erasure_required_bytes().unwrap();
    let union_root = WorkingMemoryPool::gguf_composite_erasure_required_bytes().unwrap();
    let inner = first.bytes() + second.bytes();
    let base = inner + 2 * leaf_bytes;
    let short = WorkingMemoryPool::new(base + union_bytes - 1, 0).unwrap();
    let a = short
        .retain_gguf_source(Arc::try_unwrap(first.funded(&short)).unwrap())
        .unwrap();
    let b = short
        .retain_gguf_source(Arc::try_unwrap(second.funded(&short)).unwrap())
        .unwrap();
    let ids = [a.identity(), b.identity()];
    let failed = short
        .compile_gguf_composite(GgufCompositePlan::from_retained(a, b).unwrap())
        .err()
        .expect("retained pair must refuse before construction");
    assert!(
        matches!(failed.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
        if *required_bytes == union_bytes && *available_bytes == union_bytes - 1)
    );
    assert_eq!(short.used_bytes().unwrap(), base);
    drop(failed);
    assert_eq!(short.used_bytes().unwrap(), 2 * leaf_bytes);
    drop(ids);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(base + union_bytes + union_root, 0).unwrap();
    let a = RetainedCheckpointSource::from_gguf(Arc::try_unwrap(first.funded(&pool)).unwrap());
    let b = RetainedCheckpointSource::from_gguf(Arc::try_unwrap(second.funded(&pool)).unwrap());
    let failed = pool
        .compile_gguf_composite(GgufCompositePlan::from_retained(a, b).unwrap())
        .err()
        .expect("retained pair must refuse before construction");
    assert!(matches!(
        failed.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), inner);
    drop(failed);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let a = pool
        .retain_gguf_source(Arc::try_unwrap(first.funded(&pool)).unwrap())
        .unwrap();
    let b = pool
        .retain_gguf_source(Arc::try_unwrap(second.funded(&pool)).unwrap())
        .unwrap();
    let ids = [a.identity(), b.identity()];
    let plan = GgufCompositePlan::from_retained(a, b).unwrap();
    assert_eq!(
        WorkingMemoryPool::gguf_composite_required_bytes(&plan).unwrap(),
        union_bytes
    );
    let root = pool
        .retain_gguf_composite(pool.compile_gguf_composite(plan).unwrap())
        .unwrap();
    let root_id = root.identity();
    assert_eq!(pool.used_bytes().unwrap(), base + union_bytes + union_root);
    drop(root);
    assert_eq!(pool.used_bytes().unwrap(), 2 * leaf_bytes + union_root);
    drop(ids);
    assert_eq!(pool.used_bytes().unwrap(), union_root);
    drop(root_id);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
