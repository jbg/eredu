//! Real read-plan composition; source birth, file scratch and output are separate
//! fixture prerequisites. This does not grant complete recipe compilation fit.
use super::*;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, EncodedRecipeKeysPlan, ReadBatchCatalogPlan, RecipeCatalog},
    store::{
        MemoryEncodedReadPlan, MemoryWeightStore, SafetensorsEncodedReadPlan,
        SafetensorsWeightStore, TensorSelection,
    },
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

fn memory() -> ((), MemoryWeightStore) {
    (
        (),
        MemoryWeightStore::from_safetensors([
            ("雪".into(), Dtype::U8, vec![2], vec![13, 29]),
            ("a".into(), Dtype::U8, vec![3], vec![3, 7, 11]),
        ])
        .unwrap(),
    )
}
fn file() -> (tempfile::TempDir, SafetensorsWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    serialize_to_file(
        [
            (
                "雪",
                TensorView::new(Dtype::U8, vec![2], &[13, 29]).unwrap(),
            ),
            (
                "a",
                TensorView::new(Dtype::U8, vec![3], &[3, 7, 11]).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let source = SafetensorsWeightStore::open(path).unwrap();
    (directory, source)
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

// Both concrete sources exercise the identical sequence and independent byte
// expectations without an erased read adapter or new production abstraction.
macro_rules! sequence {
    ($name:ident, $fixture:ident, $plan:ident) => {
        #[test]
        fn $name() {
            for short in [false, true] {
                let (_directory, source) = $fixture();
                let recipe = DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: ["a", "雪", "a"]
                        .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
                        .into(),
                };
                let keys_plan = || EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap();
                let Some(key_bytes) = required(&keys_plan()) else {
                    return;
                };
                // Ordinary construction establishes the fixture's actual batch
                // geometry for sizing, then retires before the admitted run.
                let keys = keys_plan().construct(()).unwrap();
                let plan = $plan::new(&source, keys.keys()).unwrap();
                let batch_bytes = required(&plan).unwrap();
                let batch = plan.construct(()).unwrap();
                let catalog_bytes =
                    required(&ReadBatchCatalogPlan::new(batch.tensors()).unwrap()).unwrap();
                let total = key_bytes + batch_bytes + catalog_bytes;
                drop((batch, keys));

                let pool = WorkingMemoryPool::new(total - u64::from(short), 0).unwrap();
                let keys = pool.initialize_shared_native(keys_plan()).unwrap();
                assert_eq!(pool.used_bytes().unwrap(), key_bytes);
                let batch = pool
                    .initialize_shared_native($plan::new(&source, keys.output().keys()).unwrap())
                    .unwrap();
                assert_eq!(pool.used_bytes().unwrap(), key_bytes + batch_bytes);
                drop((source, recipe));
                assert_eq!(keys.output().keys(), ["a", "雪", "a"]);
                assert_eq!(batch.output().byte_len(), 8);
                let result = pool.initialize_shared_native(
                    ReadBatchCatalogPlan::new(batch.output().tensors()).unwrap(),
                );
                if short {
                    let failure = result.unwrap_err();
                    assert!(matches!(
                        failure.accounting_failure(),
                        Some(WorkingMemoryError::BudgetExceeded { .. })
                    ));
                    assert!(failure.rejected_plan().is_some());
                    assert!(failure.constructor_failure().is_none());
                    assert_eq!(pool.used_bytes().unwrap(), key_bytes + batch_bytes);
                    drop(failure);
                } else {
                    let catalog = result.unwrap();
                    assert_eq!(pool.used_bytes().unwrap(), total);
                    assert!(std::ptr::eq(
                        catalog.output().tensor_metadata_borrowed("a").unwrap(),
                        &batch.output().tensors()[2]
                    ));
                    assert_eq!(
                        catalog
                            .output()
                            .tensor_metadata_borrowed("雪")
                            .unwrap()
                            .logical_shape,
                        [2]
                    );
                    assert!(
                        catalog
                            .output()
                            .tensor_metadata_borrowed("missing")
                            .is_none()
                    );
                    drop(catalog);
                }
                drop(keys);
                assert_eq!(pool.used_bytes().unwrap(), batch_bytes);
                let mut output = [0; 8];
                batch.output().read_into(&mut output).unwrap();
                assert_eq!(output, [3, 7, 11, 13, 29, 3, 7, 11]);
                drop(batch);
                assert_eq!(pool.used_bytes().unwrap(), 0);
            }
        }
    };
}
sequence!(
    memory_keys_batch_and_catalog_share_one_pool,
    memory,
    MemoryEncodedReadPlan
);
sequence!(
    file_keys_batch_and_catalog_share_one_pool,
    file,
    SafetensorsEncodedReadPlan
);
