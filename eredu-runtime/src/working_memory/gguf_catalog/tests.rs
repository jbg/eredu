use super::*;
use eredu_checkpoint::{
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    },
    store::{CheckpointSource, ReadPolicy, TensorReadRequest, TensorSelection, WeightStore},
    validation::resolve_gguf_plan,
};
use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};
use std::{collections::BTreeMap, fs::File};

struct Fixture {
    _directory: tempfile::TempDir,
    checkpoint: Checkpoint,
    resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("catalog.gguf");
        let bytes = [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "first",
                        dimensions: &[2],
                        ggml_type: GgmlType::F32,
                        data: &bytes,
                    },
                    TensorInput {
                        name: "second",
                        dimensions: &[2],
                        ggml_type: GgmlType::F32,
                        data: &bytes,
                    },
                ],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        let schema = GgufCheckpointPlan::new(
            "catalog fixture",
            ["first", "second"]
                .into_iter()
                .map(|name| {
                    GgufTensorConstraint::required(
                        name,
                        vec![2],
                        GgufTypeConstraint::OperationClass(TensorOperation::Dense),
                    )
                })
                .collect(),
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .unwrap();
        let resolution = resolve_gguf_plan(&checkpoint, &schema).unwrap();
        let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
        Self {
            _directory: directory,
            checkpoint,
            resolution,
            mapping,
        }
    }
    fn plan(&self) -> GgufCatalogPlan<'_> {
        // Actual cold input owners are outside this catalog-only component.
        GgufCatalogPlan::new(self.checkpoint.clone(), &self.resolution, &self.mapping, 1)
    }
    fn requirement(&self) -> Option<u64> {
        let result = WorkingMemoryPool::gguf_catalog_required_bytes(&self.plan());
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_CATALOG").is_some() {
            assert!(
                result.is_ok(),
                "mandatory pinned catalog qualification: {result:?}"
            );
        }
        match result {
            Ok(bytes) => Some(bytes),
            Err(WorkingMemoryError::UnknownBound) => {
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let error = pool.compile_gguf_catalog(self.plan()).unwrap_err();
                assert!(matches!(
                    error.accounting_failure(),
                    Some(WorkingMemoryError::UnknownBound)
                ));
                assert!(error.rejected_input().is_some());
                assert_eq!(pool.used_bytes().unwrap(), 0);
                None
            }
            Err(error) => panic!("unexpected catalog qualification failure: {error}"),
        }
    }
}
fn request() -> TensorReadRequest {
    TensorReadRequest {
        key: "first".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}

#[test]
fn original_immutable_catalog_exact_and_one_short_preserve_input_and_check_domain() {
    let fixture = Fixture::new();
    let Some(bytes) = fixture.requirement() else {
        return;
    };
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let plan = fixture.plan();
    let address = plan.checkpoint().shards()[0].tensors()[0]
        .descriptor()
        .name
        .as_ptr();
    let refused = short.compile_gguf_catalog(plan).unwrap_err();
    assert!(matches!(
        refused.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(
        refused.rejected_input().unwrap().checkpoint().shards()[0].tensors()[0]
            .descriptor()
            .name
            .as_ptr(),
        address
    );
    assert!(refused.compilation_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let original = pool.compile_gguf_catalog(fixture.plan()).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    original.validate_pool(&pool).unwrap();
    assert!(matches!(
        original.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    // These actual reader/shell constructors are separate unclosed cold owners.
    let source = original.into_prepared().build().unwrap();
    pool.validate_gguf_catalog_source(&source).unwrap();
    assert!(matches!(
        short.validate_gguf_catalog_source(&source),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(source.keys(), ["first", "second"]);
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    assert_eq!(source.metadata("first").unwrap().logical_shape, [2]);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_immutable_catalog_lease_and_concurrent_aliases_share_one_charge_past_source_weak() {
    let fixture = Fixture::new();
    let Some(bytes) = fixture.requirement() else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let source = pool
        .compile_gguf_catalog(fixture.plan())
        .unwrap()
        .into_prepared()
        .build()
        .unwrap();
    let mut identity = None;
    source
        .visit_source_storage(&mut |owner| {
            identity = Some(owner.identity());
        })
        .unwrap();
    let identity = identity.unwrap();
    let retained_identity = identity.clone();
    let lease = source.acquire(request()).unwrap();
    let a = source.clone();
    let b = source.clone();
    drop(source);
    std::thread::scope(|scope| {
        scope.spawn(move || drop(a));
        scope.spawn(move || drop(b));
    });
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let converted = lease.materialize_portable().unwrap();
    assert_eq!(converted.output_names(), ["first"]);
    match converted.converted() {
        eredu_gguf::ConvertedTensor::Dense(dense) => {
            assert_eq!(
                dense.data,
                [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat()
            );
        }
        other => panic!("expected actual dense output: {other:?}"),
    }
    drop(converted);
    drop(lease);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    // A weak StoreInner shell is separate; it must not retain/refund the catalog.
    assert_eq!(identity, retained_identity);
    drop(identity);
    drop(retained_identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_immutable_catalog_failed_prefix_and_unqualified_origins_cannot_promote() {
    let fixture = Fixture::new();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let ordinary = fixture.plan().compile(()).unwrap().build().unwrap();
    assert!(matches!(
        pool.validate_gguf_catalog_source(&ordinary),
        Err(WorkingMemoryError::UnknownBound)
    ));
    // External generic custody, including a pool clone, is never this module's
    // private source-specific origin and cannot self-certify catalog admission.
    let foreign_custody = fixture
        .plan()
        .compile(pool.clone())
        .unwrap()
        .build()
        .unwrap();
    assert!(matches!(
        pool.validate_gguf_catalog_source(&foreign_custody),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let zero_limit = GgufCatalogPlan::new(fixture.checkpoint.clone(), &fixture.resolution, &[], 0);
    let refused = pool.compile_gguf_catalog(zero_limit).unwrap_err();
    assert!(matches!(
        refused.input_failure(),
        Some(StoreError::InvalidShardCacheLimit)
    ));
    assert!(refused.accounting_failure().is_none());
    let Some(_) = fixture.requirement() else {
        return;
    };
    let plan = GgufCatalogPlan::new(
        fixture.checkpoint.clone(),
        &fixture.resolution,
        &fixture.mapping[..1],
        1,
    );
    let address = plan.checkpoint().shards()[0].tensors()[0]
        .descriptor()
        .name
        .as_ptr();
    let bytes = WorkingMemoryPool::gguf_catalog_required_bytes(&plan).unwrap();
    let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
    let error = exact.compile_gguf_catalog(plan).unwrap_err();
    assert!(error.accounting_failure().is_none());
    assert_eq!(error.completed_rows(), Some(1));
    assert_eq!(
        error.failed_input().unwrap().checkpoint().shards()[0].tensors()[0]
            .descriptor()
            .name
            .as_ptr(),
        address
    );
    assert_eq!(error.to_string(), "GGUF checkpoint operation failed for tensor \"second\": admitted GGUF tensor mapping omits a catalog output");
    assert_eq!(exact.used_bytes().unwrap(), bytes);
    drop(error);
    assert_eq!(exact.used_bytes().unwrap(), 0);
}
