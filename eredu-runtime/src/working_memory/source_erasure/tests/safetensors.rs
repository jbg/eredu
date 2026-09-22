use super::super::*;
use crate::working_memory::DependencyMemoryPolicy;
use eredu_checkpoint::store::SafetensorsEncodedReadPlan;
use eredu_checkpoint::{
    safetensors::{
        SafetensorsDiscoveryLimits, SafetensorsHeaderAdmission, SafetensorsIndexRequest,
        SafetensorsSourceAdmission,
    },
    store::{
        CheckpointSource, SafetensorsEncodedReadPlanErrorKind, SourceMetadataBorrowError,
        WeightStore,
    },
};
use std::{error::Error, path::Path, sync::Arc};
const POLICY: DependencyMemoryPolicy = DependencyMemoryPolicy {
    fixed_bytes: 1024,
    bytes_per_input_byte: 8,
};
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let json = br#"{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json);
    bytes.extend_from_slice(&[11, 19]);
    std::fs::write(dir.path().join("weights.safetensors"), bytes).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":"weights.safetensors"}}"#,
    )
    .unwrap();
    dir
}
fn required() -> Option<u64> {
    let result = MemoryLedger::safetensors_source_erasure_required_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RETAINED_SOURCE").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => None,
        other => panic!("{other:?}"),
    }
}
fn funded(pool: &MemoryLedger, path: &Path) -> SafetensorsWeightStore {
    pool.open_safetensors_source(path, 1, SafetensorsDiscoveryLimits::default(), POLICY)
        .unwrap()
}
#[test]
fn safetensors_erasure_compares_exact_bytes_and_keeps_weak_identity_custody() {
    let Some(outer) = required() else { return };
    let dir = fixture();
    let measurement = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = funded(&measurement, dir.path());
    let base = measurement.payload_used_bytes().unwrap();
    drop(source);
    assert_eq!(measurement.payload_used_bytes().unwrap(), 0);
    let short = crate::working_memory::memory_fixture::host_ledger(base + outer - 1, 0).unwrap();
    let failure = short
        .retain_safetensors_source(funded(&short, dir.path()))
        .unwrap_err();
    assert!(
        matches!(failure.accounting_failure(), WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })
        if *required_bytes == outer && (limit_bytes - existing_bytes) == outer-1)
    );
    let rejected = failure.rejected_safetensors().unwrap();
    short
        .validate_safetensors_source_controls(rejected)
        .unwrap();
    assert!(matches!(
        rejected.source_metadata_borrowed("weight"),
        Err(SourceMetadataBorrowError::HeaderUnavailable)
    ));
    assert_eq!(short.payload_used_bytes().unwrap(), base);
    drop(failure);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let pool = crate::working_memory::memory_fixture::host_ledger(base + outer, 0).unwrap();
    let root = pool
        .retain_safetensors_source(funded(&pool, dir.path()))
        .unwrap();
    pool.validate_retained_source_controls(&root).unwrap();
    assert!(matches!(
        root.source_metadata_borrowed("weight"),
        Err(SourceMetadataBorrowError::HeaderUnavailable)
    ));
    let identity = root.identity();
    let alias_identity = root.clone().identity();
    assert_eq!(identity, alias_identity);
    let alias = root.clone();
    assert!(root.same_source(&alias));
    assert_eq!(pool.payload_used_bytes().unwrap(), base + outer);
    std::thread::scope(|scope| {
        scope.spawn(move || drop(root));
        scope.spawn(move || drop(alias));
    });
    assert_eq!(pool.payload_used_bytes().unwrap(), outer);
    drop(identity);
    assert_eq!(pool.payload_used_bytes().unwrap(), outer);
    drop(alias_identity);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn safetensors_erasure_rejects_foreign_ordinary_and_caller_policy_origins() {
    let Some(_) = required() else { return };
    let dir = fixture();
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let foreign = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = funded(&pool, dir.path());
    let base = pool.payload_used_bytes().unwrap();
    let error = foreign.retain_safetensors_source(source).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert!(error.rejected_safetensors().is_some());
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    assert_eq!(pool.payload_used_bytes().unwrap(), base);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let error = pool
        .retain_safetensors_source(SafetensorsWeightStore::open(dir.path()).unwrap())
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        WorkingMemoryError::UnknownBound
    ));
    drop(error);
    #[derive(Debug)]
    struct CallerPolicy(MemoryLedger);
    impl SafetensorsSourceAdmission for CallerPolicy {
        fn reserve_index(
            &self,
            _: SafetensorsIndexRequest,
        ) -> Result<(), Arc<dyn Error + Send + Sync>> {
            Ok(())
        }
        fn reserve_store(&self, _: usize) -> Result<(), Arc<dyn Error + Send + Sync>> {
            Ok(())
        }
        fn headers(
            &self,
            count: usize,
        ) -> Result<Arc<dyn SafetensorsHeaderAdmission>, Arc<dyn Error + Send + Sync>> {
            self.0
                .safetensors_header_admission(count, POLICY)
                .map_err(|error| Arc::new(error) as _)
        }
    }
    let caller_policy = Arc::new(CallerPolicy(pool.clone()));
    let source = SafetensorsWeightStore::open_with_source_admission(
        dir.path(),
        1,
        SafetensorsDiscoveryLimits::default(),
        caller_policy,
    )
    .unwrap();
    assert!(source.source_admission_owner::<CallerPolicy>().is_some());
    let error = pool.retain_safetensors_source(source).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        WorkingMemoryError::UnknownBound
    ));
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let closed = RetainedCheckpointSource::from_source_with_custody(funded(&pool, dir.path()), ());
    assert!(matches!(
        pool.validate_retained_source_controls(&closed),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(closed);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let typed =
        RetainedCheckpointSource::from_safetensors_with_custody(funded(&pool, dir.path()), ());
    assert!(matches!(
        pool.validate_retained_source_controls(&typed),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(typed);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn funded_borrowed_headers_and_closed_source_feed_the_existing_prepared_read() {
    let Some(outer) = required() else { return };
    let dir = fixture();
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = funded(&pool, dir.path());
    let keys = ["weight".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::new(&source, &keys)
            .err()
            .unwrap()
            .kind(),
        SafetensorsEncodedReadPlanErrorKind::HeaderUnavailable { .. }
    ));
    let before = pool.payload_used_bytes().unwrap();
    let metadata = source.prepare_metadata("weight").unwrap();
    assert_eq!(metadata.logical_shape, [2]);
    assert!(std::ptr::eq(
        metadata,
        source.prepare_metadata("weight").unwrap()
    ));
    assert!(pool.payload_used_bytes().unwrap() > before);
    let after = pool.payload_used_bytes().unwrap();
    let ordinary = WeightStore::metadata(&source, "weight").unwrap();
    assert_eq!(&ordinary, metadata);
    assert_eq!(pool.payload_used_bytes().unwrap(), after);
    let root = pool.retain_safetensors_source(source).unwrap();
    pool.validate_retained_source_controls(&root).unwrap();
    let identity = root.identity();
    // The surrounding test view is ordinary storage; its authorization still
    // applies when the selected child has original closed source custody.
    let restricted = eredu_checkpoint::store::RestrictedCheckpointSource::excluding(
        root.clone(),
        "deny-weight",
        std::collections::BTreeSet::from(["weight".into()]),
    )
    .unwrap();
    let restricted = RetainedCheckpointSource::from(Arc::new(restricted));
    let error = SafetensorsEncodedReadPlan::from_source(&restricted, &keys)
        .err()
        .unwrap();
    assert_eq!(
        error.kind(),
        SafetensorsEncodedReadPlanErrorKind::UnauthorizedTensor { index: 0 }
    );
    assert_eq!(error.contract(), Some("deny-weight"));
    drop(error);
    drop(restricted);
    let read = pool
        .initialize_shared_native(
            SafetensorsEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    let read_bytes = read.original_bytes();
    assert_eq!(root.source_diagnostics().unwrap().physical_reads, 0);
    drop(root);
    // Completed file metadata owns its original files and source controls.
    assert!(pool.payload_used_bytes().unwrap() > outer + read_bytes);
    let mut output = [0; 2];
    read.output().read_into(&mut output).unwrap();
    assert_eq!(output, [11, 19]);
    drop(read);
    assert_eq!(pool.payload_used_bytes().unwrap(), outer);
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    drop(identity);
    assert!(weak.upgrade().is_none());
}
