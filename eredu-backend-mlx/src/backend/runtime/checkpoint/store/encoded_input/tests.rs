use super::*;
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{
        CheckpointSource, EncodedReadFailureCause, MemoryWeightStore, SafetensorsWeightStore,
        TensorSelection,
    },
};
use safetensors::tensor::{Dtype as SafeDtype, TensorView, serialize_to_file};

fn bytes() -> Vec<u8> {
    (0..256)
        .flat_map(|index| (index as f32 - 17.5).to_le_bytes())
        .collect()
}

fn selected(source: &dyn CheckpointSource) -> EncodedRecipeRead {
    DerivedWeightRecipe::source(
        "weight",
        TensorSelection::Indices {
            axis: 0,
            indices: vec![3, 1],
        },
    )
    .prepare_encoded_read(source)
    .unwrap()
    .unwrap()
}

fn memory() -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([("weight".into(), SafeDtype::F32, vec![4, 64], bytes())])
        .unwrap()
}

fn file() -> (tempfile::TempDir, SafetensorsWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(SafeDtype::F32, vec![4, 64], &bytes()).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(directory.path()).unwrap();
    (directory, store)
}

fn assert_selected(alias: &safemlx::Array) {
    let expected: Vec<f32> = (192..256)
        .chain(64..128)
        .map(|index| index as f32 - 17.5)
        .collect();
    assert_eq!(alias.shape(), &[2, 64]);
    assert_eq!(alias.evaluated().unwrap().as_slice::<f32>(), expected);
}

#[test]
fn encoded_input_admits_before_read_and_alias_keeps_source_account() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let (_directory, file) = file();
    let memory = memory();
    for source in [&file as &dyn CheckpointSource, &memory] {
        let read = selected(source);
        let before = source.source_diagnostics().unwrap();
        let plan =
            PreparedEncodedInputPlan::new(&read, &runtime, &[2, 64], Dtype::Float32).unwrap();
        let required = plan.required_bytes().unwrap();
        assert_eq!(source.source_diagnostics().unwrap(), before);
        let short = WorkingMemoryPool::new(required - 1, 0).unwrap();
        let error = plan.prepare(&short).unwrap_err();
        assert!(
            matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded {
            required_bytes, available_bytes,
        }) if *required_bytes == required && *available_bytes == required - 1)
        );
        assert!(error.rejected_plan().is_some());
        assert!(error.constructor_failure().is_none());
        assert_eq!(source.source_diagnostics().unwrap(), before);
        drop(error);
        assert_eq!(short.used_bytes().unwrap(), 0);

        let exact = WorkingMemoryPool::new(required, 0).unwrap();
        let prepared = PreparedEncodedInputPlan::new(&read, &runtime, &[2, 64], Dtype::Float32)
            .unwrap()
            .prepare(&exact)
            .unwrap();
        assert_eq!(prepared.original_bytes(), required);
        prepared.validate_pool(&exact).unwrap();
        assert!(matches!(
            prepared.validate_pool(&short),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        let alias = prepared.output().try_prepared_source_array().unwrap();
        assert!(prepared.output().try_prepared_source_array().is_err());
        assert_selected(&alias);
        drop(prepared);
        safemlx::reclaim_allocation_owners();
        assert_eq!(exact.used_bytes().unwrap(), required);
        assert_selected(&alias);
        drop(alias);
        safemlx::reclaim_allocation_owners();
        assert_eq!(exact.used_bytes().unwrap(), 0);
    }
}

#[test]
fn encoded_input_checks_exact_dtype_and_shape_before_admission() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let source = memory();
    let read = selected(&source);
    for (shape, dtype) in [
        (&[1, 128][..], Dtype::Float32),
        (&[2, 64][..], Dtype::Int32),
        (&[-2, 64][..], Dtype::Float32),
    ] {
        assert!(matches!(
            PreparedEncodedInputPlan::new(&read, &runtime, shape, dtype),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
}

#[test]
fn encoded_input_source_failure_publishes_nothing_and_retains_error_account() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let (directory, source) = file();
    let read = selected(&source);
    let plan = PreparedEncodedInputPlan::new(&read, &runtime, &[2, 64], Dtype::Float32).unwrap();
    let required = plan.required_bytes().unwrap();
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(directory.path().join("model.safetensors"))
        .unwrap()
        .set_len(0)
        .unwrap();
    let error = plan.prepare(&pool).unwrap_err();
    assert!(error.rejected_plan().is_none());
    assert!(error.completed_output().is_none());
    assert!(matches!(
        error.constructor_failure(),
        Some(EncodedInputConstructionError::Read(EncodedReadFailure {
            cause: EncodedReadFailureCause::Changed,
            ..
        }))
    ));
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), required);
    drop(error);
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
