use super::*;
use crate::recipe::{
    DerivedWeightRecipe, RecipeDtype, RecipeError, RecipeInferenceCache, RecipeInferenceInput,
    RecipeInferenceLayout, infer_recipe_bytes,
};
use crate::{StoredDtype, store::TensorSelection};
use std::cell::Cell;

struct Catalog {
    metadata: TensorMetadata,
    cache: RecipeInferenceCache,
    cache_calls: Cell<usize>,
}
impl Catalog {
    fn new() -> Self {
        Self {
            metadata: TensorMetadata {
                name: "weight".into(),
                logical_shape: vec![2, 3, 4],
                physical_shape: vec![2, 3, 4],
                stored_dtype: StoredDtype::F32,
                encoded_byte_len: 96,
                backing_shard: None,
            },
            cache: RecipeInferenceCache::default(),
            cache_calls: Cell::new(0),
        }
    }
}
impl RecipeCatalog for Catalog {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.tensor_metadata_borrowed(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
    fn tensor_metadata_borrowed(&self, key: &str) -> Option<&TensorMetadata> {
        (key == self.metadata.name).then_some(&self.metadata)
    }
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        self.cache_calls.set(self.cache_calls.get() + 1);
        Some(&self.cache)
    }
}
fn source() -> DerivedWeightRecipe {
    DerivedWeightRecipe::source("weight", TensorSelection::Full)
}

#[test]
fn selection_and_peak_inspection_leave_existing_cache_entries_unchanged() {
    let catalog = Catalog::new();
    source().infer(&catalog).unwrap();
    catalog
        .cache
        .validate::<Catalog>(&source(), || Ok(()))
        .unwrap();
    let entries = catalog.cache.entries.lock().unwrap().len();
    let validations = catalog.cache.validations.lock().unwrap().len();
    assert_eq!((entries, validations), (1, 1));
    catalog.cache_calls.set(0);
    let view = UncachedRecipeCatalog::new(&catalog as &dyn RecipeCatalog);
    let recipe = DerivedWeightRecipe::SubtractOne {
        input: Box::new(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![source(), source()],
        }),
    };
    assert_eq!(recipe.infer(&view).unwrap().shape(), &[4, 3, 4]);
    assert_eq!(recipe.peak_materialization_bytes(&view).unwrap(), 384);
    for matrix in 0..4 {
        for end in 1..=3 {
            let tile = recipe
                .select_bounded_matrix_rows(&view, matrix, 0, end)
                .unwrap();
            let metadata = tile.infer(&view).unwrap();
            assert_eq!(metadata.shape(), &[1, end, 4]);
            assert_eq!(metadata.dtype(), &RecipeDtype::F32);
            assert_eq!(metadata.byte_len(), (end * 16) as u64);
            assert!(tile.peak_materialization_bytes(&view).unwrap() >= metadata.byte_len());
        }
    }
    let members = recipe.select_bounded_members(&view).unwrap();
    assert_eq!(members.len(), 4);
    for member in members {
        assert_eq!(member.infer(&view).unwrap().shape(), &[1, 3, 4]);
    }
    assert_eq!(catalog.cache_calls.get(), 0);
    assert_eq!(catalog.cache.entries.lock().unwrap().len(), entries);
    assert_eq!(catalog.cache.validations.lock().unwrap().len(), validations);
}

#[test]
fn uncached_catalog_preserves_borrowed_metadata_and_typed_errors() {
    let catalog = Catalog::new();
    let view = UncachedRecipeCatalog::new(&catalog);
    assert!(std::ptr::eq(
        view.tensor_metadata_borrowed("weight").unwrap(),
        &catalog.metadata
    ));
    let recipe = DerivedWeightRecipe::Transpose {
        input: Box::new(source()),
        axes: vec![2, 0, 1],
    };
    let input = RecipeInferenceInput::Derived(&recipe);
    assert!(RecipeInferenceLayout::inspect(input, &view).is_some());
    assert_eq!(infer_recipe_bytes(input, &view).unwrap(), 96);
    assert_eq!(recipe.infer(&view).unwrap().shape(), &[4, 2, 3]);
    assert!(matches!(
        source().select_bounded_matrix_rows(&view, 2, 0, 1),
        Err(RecipeError::InvalidIndices {
            axis: 0,
            dimension: 2
        })
    ));
    assert!(matches!(
        view.tensor_metadata("missing"),
        Err(StoreError::UnknownTensor { key }) if key == "missing"
    ));
    let invalid = DerivedWeightRecipe::Reshape {
        input: Box::new(source()),
        shape: vec![25],
    };
    assert!(matches!(
        invalid.infer(&view),
        Err(RecipeError::ElementCountMismatch {
            input: 24,
            output: 25
        })
    ));
}

#[test]
fn uncached_catalog_does_not_consult_a_poisoned_inference_cache() {
    let catalog = Catalog::new();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = catalog.cache.entries.lock().unwrap();
        panic!("poison the source cache");
    }));
    assert!(source().infer(&catalog).is_err());
    catalog.cache_calls.set(0);
    let view = UncachedRecipeCatalog::new(&catalog);
    let tile = source().select_bounded_matrix_rows(&view, 1, 1, 3).unwrap();
    assert_eq!(tile.infer(&view).unwrap().shape(), &[1, 2, 4]);
    assert_eq!(tile.infer(&view).unwrap().byte_len(), 32);
    assert_eq!(catalog.cache_calls.get(), 0);
}
