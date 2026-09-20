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
    let catalog = plan.construct(()).unwrap();
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
    let empty = ReadBatchCatalogPlan::new(&[])
        .unwrap()
        .construct(())
        .unwrap();
    assert!(empty.tensor_metadata_borrowed("a").is_none());
}

#[test]
fn batch_catalog_lends_exact_metadata_to_finite_inference() {
    let mut matrix = tensor("matrix", 8);
    matrix.logical_shape = vec![2, 4];
    matrix.physical_shape = vec![2, 4];
    let tensors = [matrix];
    let catalog = ReadBatchCatalogPlan::new(&tensors)
        .unwrap()
        .construct(())
        .unwrap();
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

#[test]
fn index_custody_retires_with_catalog_and_size_excludes_borrowed_metadata() {
    use std::{cell::Cell, rc::Rc};
    struct Custody(Rc<Cell<bool>>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let tensors = [tensor("same", 1), tensor("same", 7), tensor("雪", 3)];
    let other = [tensor(&"x".repeat(8192), 999); 1];
    let empty = ReadBatchCatalogPlan::new(&[])
        .unwrap()
        .required_bytes::<()>()
        .unwrap();
    assert_eq!(
        ReadBatchCatalogPlan::new(&tensors)
            .unwrap()
            .required_bytes::<()>()
            .unwrap(),
        empty + 3 * size_of::<usize>()
    );
    assert_eq!(
        ReadBatchCatalogPlan::new(&other)
            .unwrap()
            .required_bytes::<()>()
            .unwrap(),
        empty + size_of::<usize>()
    );
    let retired = Rc::new(Cell::new(false));
    let catalog = ReadBatchCatalogPlan::new(&tensors)
        .unwrap()
        .construct(Custody(retired.clone()))
        .ok()
        .unwrap();
    assert!(std::ptr::eq(
        catalog.tensor_metadata_borrowed("same").unwrap(),
        &tensors[1]
    ));
    assert!(!retired.get());
    drop(catalog);
    assert!(retired.get());
}

#[test]
fn ordinary_catalog_reserve_error_remains_a_typed_recipe_source() {
    use std::error::Error;
    // A deterministic capacity-overflow cause exercises error conversion only;
    // this does not inject an allocator failure into catalog construction.
    let cause = Vec::<usize>::new()
        .try_reserve_exact(usize::MAX)
        .unwrap_err();
    let error = RecipeError::from(ReadBatchCatalogBuildError {
        cause,
        _custody: (),
    });
    let source = error.source().unwrap();
    assert!(
        source
            .downcast_ref::<ReadBatchCatalogBuildError<()>>()
            .is_some()
    );
    assert!(
        source
            .source()
            .unwrap()
            .downcast_ref::<TryReserveError>()
            .is_some()
    );
    assert_eq!(error.to_string(), error.clone().to_string());
}
