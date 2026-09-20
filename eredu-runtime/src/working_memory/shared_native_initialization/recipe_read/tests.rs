use super::*;
use eredu_checkpoint::{
    recipe::EncodedRecipeKeysPlan,
    store::{MemoryEncodedReadPlan, MemoryWeightStore, TensorSelection},
};

#[test]
fn compiled_read_outlives_inputs_and_refuses_foreign_pool_without_leaking() {
    let probe = eredu_checkpoint::recipe::EncodedRecipeMappingPlan::source(0..1).unwrap();
    let qualification = WorkingMemoryPool::shared_native_initialization_required_bytes(&probe);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(qualification.is_ok(), "{qualification:?}");
    }
    if matches!(qualification, Err(WorkingMemoryError::UnknownBound)) {
        return;
    }
    qualification.unwrap();
    for foreign in [false, true] {
        let source = MemoryWeightStore::from_safetensors([(
            "a".into(),
            safetensors::Dtype::U8,
            vec![2, 3],
            vec![1, 2, 3, 4, 5, 6],
        )])
        .unwrap();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![DerivedWeightRecipe::source("a", TensorSelection::Full); 2],
            }),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![5, 0, 5],
            },
        };
        let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
        let other = WorkingMemoryPool::new(1 << 20, 0).unwrap();
        let keys = pool
            .initialize_shared_native(EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap())
            .unwrap();
        let batch = pool
            .initialize_shared_native(
                MemoryEncodedReadPlan::new(&source, keys.output().keys()).unwrap(),
            )
            .unwrap();
        let result = batch.compile_recipe(&recipe, if foreign { &other } else { &pool });
        drop((keys, recipe, source));
        if foreign {
            assert!(matches!(
                result,
                Err(EncodedRecipeReadPreparationError::Accounting(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            let read = result.unwrap().unwrap();
            assert_eq!(read.output().shape(), [2, 3]);
            assert!(pool.used_bytes().unwrap() > 0);
            let mut bytes = [0; 6];
            EncodedRecipeRead::read_many_borrowed_into(std::iter::once(&read), &mut [&mut bytes])
                .unwrap();
            assert_eq!(bytes, [3, 1, 3, 6, 4, 6]);
            drop(read);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
        assert_eq!(other.used_bytes().unwrap(), 0);
    }
}
