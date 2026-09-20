use super::*;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, ReadBatchCatalogPlan, RecipeError, RecipeInferenceInput},
    store::{MemoryEncodedReadPlan, MemoryWeightStore, TensorSelection},
};
use safetensors::Dtype;

fn source() -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([(
        "雪".into(),
        Dtype::U8,
        vec![2, 3],
        vec![3, 7, 11, 13, 17, 19],
    )])
    .unwrap()
}

fn required<P: SharedNativeInitializer>(plan: &P) -> Option<u64> {
    let result = WorkingMemoryPool::shared_native_initialization_required_bytes(plan);
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
fn direct_source_inference_retains_output_after_borrowed_inputs_retire() {
    let source = source();
    let keys = ["雪".into()];
    let batch = MemoryEncodedReadPlan::new(&source, &keys)
        .unwrap()
        .construct(())
        .unwrap();
    let catalog = ReadBatchCatalogPlan::new(batch.tensors())
        .unwrap()
        .construct(())
        .unwrap();
    let selection = TensorSelection::Range {
        axis: 1,
        start: 1,
        end: 3,
    };
    let plan = RecipeInferencePlan::new(
        RecipeInferenceInput::Source {
            key: "雪",
            selection: &selection,
        },
        &catalog,
    )
    .unwrap();
    let Some(bytes) = required(&plan) else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let metadata = pool.initialize_shared_native(plan).unwrap();
    drop(catalog);
    drop((batch, keys, source, selection));
    assert_eq!(metadata.output().shape(), [2, 2]);
    assert_eq!(metadata.output().byte_len(), 4);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    metadata.validate_pool(&pool).unwrap();
    drop(metadata);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn invalid_permutation_keeps_owned_axes_and_charge_after_input_retirement() {
    let source = source();
    let keys = ["雪".into()];
    let batch = MemoryEncodedReadPlan::new(&source, &keys)
        .unwrap()
        .construct(())
        .unwrap();
    let catalog = ReadBatchCatalogPlan::new(batch.tensors())
        .unwrap()
        .construct(())
        .unwrap();
    let recipe = DerivedWeightRecipe::Transpose {
        input: Box::new(DerivedWeightRecipe::source("雪", TensorSelection::Full)),
        axes: vec![0, 0],
    };
    let plan = RecipeInferencePlan::new(RecipeInferenceInput::Derived(&recipe), &catalog).unwrap();
    let Some(bytes) = required(&plan) else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let failure = pool.initialize_shared_native(plan).unwrap_err();
    let (rejected, failure) = failure.into_parts();
    assert!(rejected.is_none());
    drop(rejected);
    drop(catalog);
    drop((batch, keys, recipe, source));
    assert!(failure.accounting_failure().is_none());
    assert!(matches!(
        failure.constructor_failure(),
        Some(RecipeInferenceError::Recipe(RecipeError::InvalidPermutation { axes, rank: 2 }))
            if axes == &[0, 0]
    ));
    assert!(
        std::error::Error::source(&failure)
            .unwrap()
            .is::<RecipeInferenceError>()
    );
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
