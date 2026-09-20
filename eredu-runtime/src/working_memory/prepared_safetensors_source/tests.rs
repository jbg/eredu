use super::*;
use eredu_checkpoint::store::SafetensorsEncodedReadPlan;
use eredu_checkpoint::{
    StoredDtype,
    safetensors::SafetensorsDiscoveryLimits,
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    store::{
        EncodedTensorLease, ReadPolicy, SafetensorsWeightStore, StoreError, TensorReadRequest,
        TensorSelection,
    },
};
use eredu_core::artifact::GgufCompanionRequirement;
use eredu_core::{
    LoadingProtocol, ModelConfiguration, ModelConfigurationResolver, ResolvedModelConfiguration,
};
use std::{path::Path, sync::Arc};
const POLICY: DependencyMemoryPolicy = DependencyMemoryPolicy {
    fixed_bytes: 1024,
    bytes_per_input_byte: 8,
};
struct Resolver;
impl ModelConfigurationResolver for Resolver {
    type ArtifactPlan = ();
    fn resolve_safetensors(
        &self,
        json: &serde_json::Value,
    ) -> Result<ResolvedModelConfiguration<()>, ArtifactError> {
        assert_eq!(json["model_type"], "fixture");
        Ok(ResolvedModelConfiguration::new(
            ModelConfiguration::new(
                "fixture",
                "fixture",
                "fixture",
                LoadingProtocol::Model,
                Some(json.clone()),
            )?,
            (),
        ))
    }
    fn resolve_gguf(
        &self,
        _: &str,
        _: &eredu_gguf::Checkpoint,
    ) -> Result<ResolvedModelConfiguration<()>, ArtifactError> {
        unreachable!("SafeTensors fixture")
    }
    fn gguf_companion_requirements(
        &self,
        _: &str,
        _: &eredu_gguf::Checkpoint,
    ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
        unreachable!("SafeTensors fixture")
    }
}
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"model_type":"fixture"}"#,
    )
    .unwrap();
    for (name, payload) in [("alpha", [11, 19]), ("beta", [23, 29])] {
        let json =
            format!("{{\"{name}\":{{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}}}");
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(dir.path().join(format!("{name}.safetensors")), bytes).unwrap();
    }
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"alpha":"alpha.safetensors","beta":"beta.safetensors"}}"#,
    )
    .unwrap();
    dir
}

fn qualified() -> bool {
    let result = WorkingMemoryPool::safetensors_source_erasure_required_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RETAINED_SOURCE").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(_) => true,
        Err(WorkingMemoryError::UnknownBound) => false,
        other => panic!("{other:?}"),
    }
}
fn inspection(pool: &WorkingMemoryPool, path: &Path) -> eredu_core::ArtifactInspection<()> {
    pool.inspect_artifact_with_safetensors_pool(
        path,
        &Resolver,
        SafetensorsDiscoveryLimits::default(),
        POLICY,
    )
    .unwrap()
}
fn resolution(path: &Path) -> ResolvedCheckpointPlan {
    let store = SafetensorsWeightStore::open(path).unwrap();
    let plan = SafetensorsCheckpointPlan::new(
        "fixture",
        ["alpha", "beta"]
            .into_iter()
            .map(|key| {
                SafetensorsTensorConstraint::required(
                    key,
                    vec![2],
                    StoredDtypeConstraint::Exact(StoredDtype::U8),
                )
            })
            .collect(),
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .unwrap();
    eredu_checkpoint::validation::resolve_safetensors_plan(&store, &plan).unwrap()
}
fn leaf(pool: &WorkingMemoryPool, path: &Path) -> RetainedCheckpointSource {
    let store = pool
        .open_safetensors_source(path, 2, SafetensorsDiscoveryLimits::default(), POLICY)
        .unwrap();
    store.prepare_metadata("alpha").unwrap();
    store.prepare_metadata("beta").unwrap();
    pool.retain_safetensors_source(store).unwrap()
}
fn request(key: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
#[test]
fn prepared_views_keep_typed_file_route_and_resolution_after_inspection_retires() {
    if !qualified() {
        return;
    }
    let dir = fixture();
    let pool = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let inspection = inspection(&pool, dir.path());
    let contract = resolution(dir.path())
        .project_claimed_sources("alpha-only", ["alpha".into()].into())
        .unwrap();
    let ordinary = eredu_core::artifact::open_prepared_safetensors_artifact(
        inspection.tensors(),
        inspection.safetensors_shards().unwrap().clone(),
        contract.clone(),
        2,
    )
    .unwrap();
    std::fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
    std::fs::remove_file(dir.path().join("config.json")).unwrap();
    let source = pool
        .prepare_safetensors_artifact(
            inspection.tensors(),
            inspection.safetensors_shards().unwrap().clone(),
            &contract,
            2,
            POLICY,
        )
        .unwrap();
    pool.validate_retained_source_controls(&source).unwrap();
    assert_eq!(source.source_keys(), ordinary.source_keys());
    assert_eq!(
        source.source_metadata("alpha").unwrap(),
        ordinary.source_metadata("alpha").unwrap()
    );
    assert_eq!(
        source.source_provenance("alpha").unwrap(),
        ordinary.source_provenance("alpha").unwrap()
    );
    assert!(matches!(
        source.acquire_lease(request("beta")),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    let forbidden = vec!["beta".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::from_source(&source, &forbidden)
            .err()
            .unwrap()
            .kind(),
        eredu_checkpoint::store::SafetensorsEncodedReadPlanErrorKind::UnauthorizedTensor { .. }
    ));
    let view_bytes =
        WorkingMemoryPool::prepared_safetensors_view_bytes(inspection.tensors(), &contract, POLICY)
            .unwrap();
    let keys = vec!["alpha".into()];
    let initializer = SafetensorsEncodedReadPlan::from_source(&source, &keys)
        .unwrap()
        .unwrap();
    let read = pool.initialize_shared_native(initializer).unwrap();
    let identity = source.identity();
    let alias = source.clone();
    assert!(source.same_source(&alias));
    drop(source);
    drop(alias);
    drop(ordinary);
    drop(inspection);
    let mut output = [0u8; 2];
    read.output().read_into(&mut output).unwrap();
    assert_eq!(output, [11, 19]);
    drop(read);
    assert_eq!(pool.used_bytes().unwrap(), view_bytes);
    drop(identity);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn view_admission_exact_and_one_short_preserve_untouched_leaf() {
    if !qualified() {
        return;
    }
    let dir = fixture();
    let inspection = eredu_core::inspect_artifact(dir.path(), &Resolver).unwrap();
    let contract = resolution(dir.path());
    let quote =
        WorkingMemoryPool::prepared_safetensors_view_bytes(inspection.tensors(), &contract, POLICY)
            .unwrap();
    let measure = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let source = leaf(&measure, dir.path());
    let base = measure.used_bytes().unwrap();
    drop(source);
    assert_eq!(measure.used_bytes().unwrap(), 0);
    let short = WorkingMemoryPool::new(base + quote - 1, 0).unwrap();
    let error = short
        .prepare_safetensors_views(
            leaf(&short, dir.path()),
            inspection.tensors(),
            &contract,
            POLICY,
        )
        .unwrap_err();
    assert!(
        matches!(error.memory_failure(),Some(WorkingMemoryError::BudgetExceeded { required_bytes,available_bytes }) if *required_bytes==quote && *available_bytes==quote-1)
    );
    assert!(error.rejected_source().is_some());
    assert_eq!(short.used_bytes().unwrap(), base);
    drop(error);
    assert_eq!(short.used_bytes().unwrap(), 0);
    let exact = WorkingMemoryPool::new(base + quote, 0).unwrap();
    let source = exact
        .prepare_safetensors_views(
            leaf(&exact, dir.path()),
            inspection.tensors(),
            &contract,
            POLICY,
        )
        .unwrap();
    assert_eq!(exact.used_bytes().unwrap(), base + quote);
    drop(source);
    assert_eq!(exact.used_bytes().unwrap(), 0);
}
#[test]
fn failed_catalog_pinning_retains_original_source_and_metadata_account() {
    if !qualified() {
        return;
    }
    let dir = fixture();
    let pool = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let inspection = eredu_core::inspect_artifact(dir.path(), &Resolver).unwrap();
    let contract = resolution(dir.path());
    let tensors = TensorCatalog::new(inspection.tensors().descriptors().cloned().map(
        |mut tensor| {
            tensor.shape = vec![1, 2];
            tensor
        },
    ))
    .unwrap();
    let quote =
        WorkingMemoryPool::prepared_safetensors_view_bytes(&tensors, &contract, POLICY).unwrap();
    let source = leaf(&pool, dir.path());
    let base = pool.used_bytes().unwrap();
    let error = pool
        .prepare_safetensors_views(source, &tensors, &contract, POLICY)
        .unwrap_err();
    assert!(matches!(
        error.construction_failure(),
        Some(ArtifactError::CheckpointStore(
            StoreError::PreparedCatalogMismatch { .. }
        ))
    ));
    assert!(error.rejected_source().is_some());
    assert_eq!(pool.used_bytes().unwrap(), base + quote);
    drop(pool.acquire_unquoted().unwrap());
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn view_origin_rejects_foreign_and_ordinary_sources_without_strong_pool_cycle() {
    if !qualified() {
        return;
    }
    let dir = fixture();
    let pool = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let foreign = WorkingMemoryPool::new(2_000_000, 0).unwrap();
    let inspection = eredu_core::inspect_artifact(dir.path(), &Resolver).unwrap();
    let contract = resolution(dir.path());
    let error = foreign
        .prepare_safetensors_views(
            leaf(&pool, dir.path()),
            inspection.tensors(),
            &contract,
            POLICY,
        )
        .unwrap_err();
    assert!(matches!(
        error.memory_failure(),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    let ordinary = RetainedCheckpointSource::from_safetensors(
        SafetensorsWeightStore::open(dir.path()).unwrap(),
    );
    let error = pool
        .prepare_safetensors_views(ordinary, inspection.tensors(), &contract, POLICY)
        .unwrap_err();
    assert!(matches!(
        error.memory_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    drop(error);
    let source = pool
        .prepare_safetensors_views(
            leaf(&pool, dir.path()),
            inspection.tensors(),
            &contract,
            POLICY,
        )
        .unwrap();
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        source
            .acquire_lease(request("alpha"))
            .unwrap()
            .encoded_bytes()
            .unwrap(),
        [11, 19]
    );
}
