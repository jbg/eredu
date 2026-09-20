use super::*;
use crate::{
    StoredDtype,
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeInferenceInput, RecipeInferencePlan},
    store::TensorSelection,
};

fn tensor(name: &str, width: usize) -> TensorMetadata {
    TensorMetadata {
        name: name.into(),
        logical_shape: vec![width],
        physical_shape: vec![width],
        stored_dtype: StoredDtype::U8,
        encoded_byte_len: width as u64,
        backing_shard: None,
    }
}

#[test]
fn batch_catalog_preserves_last_occurrence_and_owned_lookup_independence() {
    let tensors = [
        tensor("z", 2),
        tensor("a", 3),
        tensor("z", 7),
        tensor("é", 5),
        tensor("", 11),
    ];
    let plan = ReadBatchCatalogPlan::new(&tensors).unwrap();
    assert_eq!(plan.indices, Layout::array::<usize>(5).unwrap());
    let catalog = plan.build();
    for (key, index, width) in [("", 4, 11), ("a", 1, 3), ("z", 2, 7), ("é", 3, 5)] {
        let borrowed = catalog.tensor_metadata_borrowed(key).unwrap();
        assert!(std::ptr::eq(borrowed, &tensors[index]));
        assert_eq!(borrowed.logical_shape, [width]);
        let mut owned = catalog.tensor_metadata(key).unwrap();
        assert_eq!(owned, tensors[index]);
        owned.logical_shape.push(99);
        assert_eq!(borrowed.logical_shape, [width]);
    }
    assert!(catalog.tensor_metadata_borrowed("missing").is_none());
    assert!(
        matches!(catalog.tensor_metadata("missing"), Err(StoreError::UnknownTensor { key }) if key == "missing")
    );
    let empty = ReadBatchCatalogPlan::new(&[]).unwrap().build();
    assert!(empty.tensor_metadata_borrowed("a").is_none());
}

#[test]
fn batch_catalog_lends_exact_metadata_to_finite_inference() {
    let mut matrix = tensor("matrix", 8);
    matrix.logical_shape = vec![2, 4];
    matrix.physical_shape = vec![2, 4];
    let tensors = [matrix];
    let catalog = ReadBatchCatalogPlan::new(&tensors).unwrap().build();
    let recipe = DerivedWeightRecipe::source(
        "matrix",
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
    );
    let inferred = RecipeInferencePlan::new(RecipeInferenceInput::Derived(&recipe), &catalog)
        .unwrap()
        .infer()
        .unwrap();
    assert_eq!(inferred.shape, [3, 4]);
    assert_eq!(inferred.dtype, RecipeDtype::U8);
    assert_eq!(inferred.byte_len, 12);
    drop(catalog);
    drop(tensors);
    assert_eq!(inferred.shape, [3, 4]);
}
