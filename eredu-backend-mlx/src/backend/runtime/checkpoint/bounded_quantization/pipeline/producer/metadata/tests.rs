use super::*;
use eredu_checkpoint::{
    recipe::RecipeError,
    store::{MemoryWeightStore, SafetensorsWeightStore, TensorSelection},
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

#[test]
fn output_metadata_is_admitted_before_inference_and_retires_with_its_owner() {
    let bytes = (0..256)
        .flat_map(|n| (n as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let memory = MemoryWeightStore::from_safetensors([(
        "weight".into(),
        Dtype::F32,
        vec![4, 64],
        bytes.clone(),
    )])
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::F32, vec![4, 64], &bytes).unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let file = SafetensorsWeightStore::open(directory.path()).unwrap();
    // Retained source/header admission precedes this output metadata owner.
    file.source_metadata("weight").unwrap();
    let recipe = DerivedWeightRecipe::SubtractOne {
        input: Box::new(DerivedWeightRecipe::source(
            "weight",
            TensorSelection::Indices {
                axis: 0,
                indices: vec![3, 1],
            },
        )),
    };
    for source in [&memory as &dyn CheckpointSource, &file] {
        let before = source.source_diagnostics().unwrap();
        let plan = MetadataPlan::new(&recipe, source).unwrap();
        let required = plan.required_bytes().unwrap();
        let short = WorkingMemoryPool::new(required - 1, 0).unwrap();
        let error = plan.prepare(&short).unwrap_err();
        assert!(
            matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == required && *available_bytes == required - 1)
        );
        assert!(error.rejected_plan().is_some());
        assert!(error.constructor_failure().is_none());
        assert_eq!(short.used_bytes().unwrap(), 0);
        drop(error);
        let pool = WorkingMemoryPool::new(required, 0).unwrap();
        let ready = MetadataPlan::new(&recipe, source)
            .unwrap()
            .prepare(&pool)
            .unwrap();
        assert_eq!(ready.output().shape(), &[2, 64]);
        assert_eq!(ready.output().inferred().shape(), &[2, 64]);
        assert_eq!(ready.output().inferred().byte_len(), 512);
        assert_eq!(ready.original_bytes(), required);
        assert_eq!(pool.used_bytes().unwrap(), required);
        ready.validate_pool(&pool).unwrap();
        assert!(matches!(
            ready.validate_pool(&short),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(source.source_diagnostics().unwrap(), before);
        drop(ready);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn inference_failure_retains_the_typed_error_and_original_account() {
    let (failure, pool, required) = {
        let source = MemoryWeightStore::from_safetensors([(
            "weight".into(),
            Dtype::F32,
            vec![2, 4],
            vec![0; 32],
        )])
        .unwrap();
        let recipe = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source("weight", TensorSelection::Full)),
            shape: vec![9],
        };
        let plan = MetadataPlan::new(&recipe, &source).unwrap();
        let required = plan.required_bytes().unwrap();
        let pool = WorkingMemoryPool::new(required, 0).unwrap();
        let (uncalled, failure) = plan.prepare(&pool).unwrap_err().into_parts();
        assert!(uncalled.is_none());
        (failure, pool, required)
    };
    assert!(matches!(
        &failure.constructor_failure().unwrap().cause,
        Cause::Inference(RecipeInferenceError::Recipe(
            RecipeError::ElementCountMismatch {
                input: 8,
                output: 9
            }
        ))
    ));
    assert!(failure.constructor_failure().unwrap()._metadata.is_none());
    assert_eq!(pool.used_bytes().unwrap(), required);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn native_dimension_failure_keeps_constructed_metadata_and_shape_prefix() {
    use eredu_checkpoint::{
        StoredDtype,
        store::{
            CheckpointLease, SourceMetadataLoan, StoreError, TensorMetadata, TensorReadRequest,
            WeightStoreDiagnostics,
        },
    };
    struct Source(TensorMetadata);
    impl CheckpointSource for Source {
        fn source_metadata_borrowed(&self, _: &str) -> SourceMetadataLoan<'_> {
            Ok(&self.0)
        }
        fn source_keys(&self) -> Vec<String> {
            panic!("no owned catalog")
        }
        fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            panic!("no owned metadata")
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("no payload")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("no diagnostic mutation")
        }
    }
    let large = i32::MAX as usize + 1;
    let (failure, pool, required) = {
        let source = Source(TensorMetadata {
            name: "weight".into(),
            logical_shape: vec![2, large],
            physical_shape: vec![2, large],
            stored_dtype: StoredDtype::F32,
            encoded_byte_len: 8 * large as u64,
            backing_shard: None,
        });
        let recipe = DerivedWeightRecipe::source("weight", TensorSelection::Full);
        let plan = MetadataPlan::new(&recipe, &source).unwrap();
        let required = plan.required_bytes().unwrap();
        let pool = WorkingMemoryPool::new(required, 0).unwrap();
        let (uncalled, failure) = plan.prepare(&pool).unwrap_err().into_parts();
        assert!(uncalled.is_none());
        (failure, pool, required)
    };
    fn require_static<T: 'static>(_: &T) {}
    require_static(&failure);
    let cause = failure.constructor_failure().unwrap();
    assert!(matches!(cause.cause, Cause::Dimension { axis: 1, dimension } if dimension == large));
    assert_eq!(cause._metadata.as_ref().unwrap().shape(), &[2, large]);
    assert_eq!(cause._shape, [2]);
    assert_eq!(pool.used_bytes().unwrap(), required);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
