use super::*;
use eredu_checkpoint::store::{CheckpointSource, WeightStore};
use eredu_checkpoint::store::{
    RetainedCheckpointSource, SafetensorsEncodedReadPlanErrorKind, SafetensorsWeightStore,
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

fn fixture() -> (tempfile::TempDir, SafetensorsWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    serialize_to_file(
        [
            (
                "first",
                TensorView::new(Dtype::U8, vec![2], &[13, 29]).unwrap(),
            ),
            (
                "second",
                TensorView::new(Dtype::U8, vec![3], &[3, 7, 11]).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(path).unwrap();
    (directory, store)
}
fn requirement(plan: &SafetensorsEncodedReadPlan<'_>) -> Option<u64> {
    let result = MemoryLedger::shared_native_initialization_required_bytes(plan);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => None,
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn file_metadata_admission_compares_before_construction_and_keeps_actual_source() {
    let (_directory, source) = fixture();
    let keys = ["second".into(), "first".into(), "second".into()];
    let plan = SafetensorsEncodedReadPlan::new(&source, &keys).unwrap();
    let Some(bytes) = requirement(&plan) else {
        return;
    };
    let short = crate::working_memory::memory_fixture::host_ledger(bytes - 1, 0).unwrap();
    let error = short.initialize_shared_native(plan).unwrap_err();
    assert!(error.rejected_plan().is_some());
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    assert!(
        source
            .source_diagnostics()
            .unwrap()
            .touched_shard_paths
            .is_empty()
    );
    drop(error);
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let read = pool
        .initialize_shared_native(SafetensorsEncodedReadPlan::new(&source, &keys).unwrap())
        .unwrap();
    assert_eq!(read.original_bytes(), bytes);
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    assert_eq!(source.diagnostics().unwrap().touched_shard_paths.len(), 1);
    assert!(source.diagnostics().unwrap().payload_shard_paths.is_empty());
    drop((source, keys));
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    assert!(read.output().read_layout().unwrap().required_bytes() > 0);
    // This fixture's read scratch and output are ordinary prerequisites, not
    // part of the metadata constructor's reservation.
    let mut output = [0; 8];
    read.output().read_into(&mut output).unwrap();
    assert_eq!(output, [3, 7, 11, 13, 29, 3, 7, 11]);
    drop(read);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn file_metadata_admission_preserves_competing_charges_and_unquoted_exclusion() {
    let (_directory, source) = fixture();
    let keys = ["first".into()];
    let plan = || SafetensorsEncodedReadPlan::new(&source, &keys).unwrap();
    let Some(bytes) = requirement(&plan()) else {
        return;
    };
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let unquoted = pool.acquire_unquoted().unwrap();
    let error = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop((error, unquoted));
    let first = pool.initialize_shared_native(plan()).unwrap();
    let second = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        second.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    drop((second, first));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let second = pool.initialize_shared_native(plan()).unwrap();
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn file_route_checks_visibility_and_releases_views_after_admitted_construction() {
    use eredu_checkpoint::store::RestrictedCheckpointSource;
    use std::{collections::BTreeSet, sync::Arc};
    let (_directory, source) = fixture();
    let leaf = Arc::new(source);
    let alive = Arc::downgrade(&leaf);
    let root: RetainedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "first only",
            BTreeSet::from(["first".into()]),
        )
        .unwrap(),
    )
    .into();
    let denied = ["second".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::from_source(&root, &denied).map_err(|error| error.kind()),
        Err(SafetensorsEncodedReadPlanErrorKind::UnauthorizedTensor { index: 0 })
    ));
    let keys = ["first".into(), "first".into()];
    let plan = SafetensorsEncodedReadPlan::from_source(&root, &keys)
        .unwrap()
        .unwrap();
    let Some(bytes) = requirement(&plan) else {
        return;
    };
    let short = crate::working_memory::memory_fixture::host_ledger(bytes - 1, 0).unwrap();
    let error = short.initialize_shared_native(plan).unwrap_err();
    assert!(error.rejected_plan().is_some());
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    assert!(leaf.diagnostics().unwrap().touched_shard_paths.is_empty());
    drop(error);
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let read = pool
        .initialize_shared_native(
            SafetensorsEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    drop((root, leaf, keys));
    assert!(alive.upgrade().is_none());
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    // The fixture supplies ordinary read scratch and output separately.
    let mut output = [0; 4];
    read.output().read_into(&mut output).unwrap();
    assert_eq!(output, [13, 29, 13, 29]);
    drop(read);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
