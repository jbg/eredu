use super::super::*;
use super::POLICY;
use eredu_checkpoint::store::{
    EncodedTensorLease, ReadPolicy, TensorReadRequest, TensorSelection, WeightStore,
};
use eredu_core::artifact::{ArtifactError, GgufCompanionRequirement};
use eredu_core::{
    ArtifactInspection, LoadingProtocol, ModelConfiguration, ModelConfigurationResolver,
    ResolvedModelConfiguration,
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
fn inspect(
    pool: &MemoryLedger,
    path: &Path,
) -> Result<ArtifactInspection<()>, OriginalArtifactInspectionError<()>> {
    pool.inspect_artifact_with_safetensors_pool(
        path,
        &Resolver,
        SafetensorsDiscoveryLimits::default(),
        POLICY,
    )
}
fn request(key: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
#[test]
fn early_inspection_shares_catalog_custody_and_builds_independent_caches_without_rediscovery() {
    let dir = fixture();
    if !super::qualified(dir.path()) {
        return;
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let inspection = inspect(&pool, dir.path()).unwrap();
    let alias = inspection.clone().map_architecture_plan(|_| 17u32);
    assert!(std::ptr::eq(
        inspection.tensors().get("alpha").unwrap(),
        alias.tensors().get("alpha").unwrap()
    ));
    assert!(
        inspection
            .admission_token()
            .same_admission(&alias.admission_token())
    );
    let tensors = inspection.tensors().clone();
    let discovery_bytes = pool.payload_used_bytes().unwrap();
    assert!(discovery_bytes > 0);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    let shards = inspection.safetensors_shards().unwrap().clone();
    // Deferred store creation must consume the admitted owners, not these files.
    std::fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
    std::fs::remove_file(dir.path().join("config.json")).unwrap();
    let one = pool
        .open_admitted_safetensors_source(shards.clone(), 1, POLICY)
        .unwrap();
    let two = pool
        .open_admitted_safetensors_source(shards.clone(), 2, POLICY)
        .unwrap();
    pool.validate_safetensors_source_controls(&one).unwrap();
    pool.validate_safetensors_source_controls(&two).unwrap();
    assert!(std::ptr::eq(
        one.prepare_metadata("alpha").unwrap(),
        two.prepare_metadata("alpha").unwrap()
    ));
    // Payload leases here are ordinary test prerequisites, separately from the
    // admitted source and catalog metadata being exercised.
    for source in [&one, &two] {
        let alpha = source.acquire(request("alpha")).unwrap();
        assert_eq!(alpha.encoded_bytes().unwrap(), [11, 19]);
        if std::ptr::eq(source, &one) {
            assert!(matches!(
                source.acquire(request("beta")),
                Err(StoreError::CapacityExhausted { maximum: 1, .. })
            ));
        }
        drop(alpha);
        let beta = source.acquire(request("beta")).unwrap();
        assert_eq!(beta.encoded_bytes().unwrap(), [23, 29]);
    }
    assert_eq!(one.diagnostics().unwrap().currently_cached_shards, 1);
    assert_eq!(two.diagnostics().unwrap().currently_cached_shards, 2);
    let ordinary = SafetensorsWeightStore::open_admitted(shards.clone(), 1).unwrap();
    assert!(matches!(
        pool.validate_safetensors_source_controls(&ordinary),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(ordinary);
    drop(one);
    drop(two);
    assert_eq!(pool.payload_used_bytes().unwrap(), discovery_bytes);
    drop(shards);
    drop(inspection);
    drop(alias);
    assert!(
        pool.payload_used_bytes().unwrap() > 0,
        "tensor catalog owns its original metadata contribution"
    );
    assert_eq!(tensors.get("beta").unwrap().shape, [2]);
    drop(tensors);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn inspection_and_fresh_store_refusals_preserve_original_inputs() {
    let dir = fixture();
    if !super::qualified(dir.path()) {
        return;
    }
    let initial = MemoryLedger::safetensors_source_initial_bytes(dir.path(), POLICY).unwrap();
    let short = crate::working_memory::memory_fixture::host_ledger(initial - 1, 0).unwrap();
    assert!(matches!(
        inspect(&short, dir.path()).unwrap_err().memory_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let inspection = inspect(&pool, dir.path()).unwrap();
    let base = pool.payload_used_bytes().unwrap();
    let shards = inspection.safetensors_shards().unwrap().clone();
    let foreign = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let error = foreign
        .open_admitted_safetensors_source(shards.clone(), 1, POLICY)
        .unwrap_err();
    assert!(matches!(
        error.memory_failure(),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(error.rejected_shards().is_some());
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    drop(error);
    let invalid = pool
        .open_admitted_safetensors_source(shards.clone(), 0, POLICY)
        .unwrap_err();
    assert!(matches!(
        invalid.construction_failure(),
        Some(StoreError::InvalidShardCacheLimit)
    ));
    assert!(invalid.rejected_shards().is_some());
    assert!(pool.payload_used_bytes().unwrap() > base);
    drop(invalid);
    assert_eq!(pool.payload_used_bytes().unwrap(), base);
    let ordinary = SafetensorsShards::discover(dir.path()).unwrap();
    assert!(matches!(
        pool.open_admitted_safetensors_source(ordinary, 1, POLICY)
            .unwrap_err()
            .memory_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    drop(shards);
    drop(inspection);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn malformed_inspection_keeps_accepted_account_and_tensor_catalog_does_not_keep_pool_alive() {
    let dir = fixture();
    if !super::qualified(dir.path()) {
        return;
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let inspection = inspect(&pool, dir.path()).unwrap();
    let catalog = inspection.tensors().clone();
    drop(inspection);
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    assert_eq!(catalog.len(), 2);
    drop(catalog);
    std::fs::write(dir.path().join("model.safetensors.index.json"), b"bad JSON").unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let error = inspect(&pool, dir.path()).unwrap_err();
    assert!(matches!(
        error.construction_failure(),
        Some(ArtifactError::CheckpointStore(
            StoreError::SafetensorsShards(_)
        ))
    ));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn loading_inspection_preserves_limits_and_refusals_across_admission_availability() {
    let dir = fixture();
    if !super::qualified(dir.path()) {
        return;
    }
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let ordinary_owner = pool.acquire_unquoted().unwrap();
    let ordinary = pool
        .inspect_artifact_for_loading(
            dir.path(),
            &Resolver,
            SafetensorsDiscoveryLimits::default(),
            POLICY,
        )
        .unwrap();
    assert!(
        !pool
            .safetensors_shards_have_source_admission(ordinary.safetensors_shards().unwrap())
            .unwrap()
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let limits = SafetensorsDiscoveryLimits {
        max_index_bytes: 0,
        ..Default::default()
    };
    let refused = pool
        .inspect_artifact_for_loading(dir.path(), &Resolver, limits, POLICY)
        .unwrap_err();
    assert!(refused.construction_failure().is_some());
    assert!(refused.to_string().contains("index"));
    drop(refused);
    drop(ordinary);
    drop(ordinary_owner);
    let admitted = pool
        .inspect_artifact_for_loading(
            dir.path(),
            &Resolver,
            SafetensorsDiscoveryLimits::default(),
            POLICY,
        )
        .unwrap();
    assert!(
        pool.safetensors_shards_have_source_admission(admitted.safetensors_shards().unwrap())
            .unwrap()
    );
    let foreign = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    assert!(matches!(
        foreign.safetensors_shards_have_source_admission(admitted.safetensors_shards().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(admitted);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let initial = MemoryLedger::safetensors_source_initial_bytes(dir.path(), POLICY).unwrap();
    let short = crate::working_memory::memory_fixture::host_ledger(initial - 1, 0).unwrap();
    assert!(matches!(
        short
            .inspect_artifact_for_loading(
                dir.path(),
                &Resolver,
                SafetensorsDiscoveryLimits::default(),
                POLICY
            )
            .unwrap_err()
            .memory_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    std::fs::write(dir.path().join("model.safetensors.index.json"), b"bad JSON").unwrap();
    let error = pool
        .inspect_artifact_for_loading(
            dir.path(),
            &Resolver,
            SafetensorsDiscoveryLimits::default(),
            POLICY,
        )
        .unwrap_err();
    let accepted = pool.payload_used_bytes().unwrap();
    assert!(accepted > 0);
    let error = eredu_core::AutomaticPlanningError::backend("inspect-loading-fixture", error);
    let alias = error.clone();
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), accepted);
    let mut cause: &(dyn std::error::Error + 'static) = &alias;
    while cause
        .downcast_ref::<OriginalArtifactInspectionError<()>>()
        .is_none()
    {
        cause = cause
            .source()
            .expect("original inspection cause remains in the chain");
    }
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
