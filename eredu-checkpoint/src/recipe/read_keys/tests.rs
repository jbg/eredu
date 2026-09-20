use super::*;
use crate::{recipe::RecipeDtype, store::*};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

fn source(key: &str) -> DerivedWeightRecipe {
    DerivedWeightRecipe::source(key, TensorSelection::Full)
}

#[test]
fn ordered_occurrences_keep_unicode_duplicates_and_custody_after_recipe_retirement() {
    struct Custody(Arc<AtomicBool>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let alive = Arc::new(AtomicBool::new(true));
    let recipe = DerivedWeightRecipe::Stack {
        axis: 1,
        inputs: vec![
            source("雪"),
            DerivedWeightRecipe::Select {
                input: Box::new(DerivedWeightRecipe::Cast {
                    input: Box::new(source("b")),
                    dtype: RecipeDtype::F16,
                }),
                selection: TensorSelection::Indices {
                    axis: 0,
                    indices: vec![2, 0],
                },
            },
            source("雪"),
            source(""),
        ],
    };
    let plan = EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap();
    assert!(!plan.contiguous());
    let keys = plan.construct(Custody(alive.clone())).unwrap();
    drop(recipe);
    assert_eq!(keys.keys(), ["雪", "b", "雪", ""]);
    assert!(alive.load(Ordering::SeqCst));
    drop(keys);
    assert!(!alive.load(Ordering::SeqCst));
}

#[test]
fn key_storage_depends_on_occurrences_and_names_not_tensor_geometry() {
    let small = DerivedWeightRecipe::Reshape {
        input: Box::new(source("weight")),
        shape: vec![1],
    };
    let large = DerivedWeightRecipe::View {
        input: Box::new(source("weight")),
        dtype: RecipeDtype::F32,
        shape: vec![1_000_000_000, 1_000_000_000],
    };
    let size = |recipe: &DerivedWeightRecipe| {
        EncodedRecipeKeysPlan::new(recipe)
            .unwrap()
            .unwrap()
            .required_bytes::<()>()
            .unwrap()
    };
    assert_eq!(size(&small), size(&large));
    assert_eq!(size(&source("雪")) - size(&source("")), "雪".len());
    let repeated = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![source("weight"), source("weight")],
    };
    assert_eq!(
        size(&repeated) - size(&small),
        size_of::<String>() + "weight".len()
    );
    let empty = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![],
    };
    assert!(EncodedRecipeKeysPlan::new(&empty)
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap()
        .keys()
        .is_empty());
    for recipe in [
        DerivedWeightRecipe::NegLog {
            input: Box::new(source("weight")),
        },
        DerivedWeightRecipe::SubtractOne {
            input: Box::new(source("weight")),
        },
    ] {
        assert!(EncodedRecipeKeysPlan::new(&recipe).unwrap().is_none());
    }
}

struct Recording {
    store: MemoryWeightStore,
    batches: Mutex<Vec<Vec<String>>>,
}
impl CheckpointSource for Recording {
    fn source_keys(&self) -> Vec<String> {
        self.store.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.store.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.store.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.store.source_diagnostics()
    }
    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        self.batches.lock().unwrap().push(keys.to_vec());
        self.store.prepare_encoded_read(keys)
    }
}

#[test]
fn shared_compilers_request_exact_occurrences_once_and_preserve_output_and_errors() {
    let store = Recording {
        store: MemoryWeightStore::from_safetensors([
            (
                "a".into(),
                safetensors::Dtype::U8,
                vec![2, 2],
                vec![1, 3, 5, 7],
            ),
            (
                "b".into(),
                safetensors::Dtype::U8,
                vec![2, 2],
                vec![11, 13, 17, 19],
            ),
        ])
        .unwrap(),
        batches: Mutex::new(Vec::new()),
    };
    for uncached in [false, true] {
        for (axis, expected) in [
            (0, [1, 3, 5, 7, 11, 13, 17, 19, 1, 3, 5, 7]),
            (1, [1, 3, 11, 13, 1, 3, 5, 7, 17, 19, 5, 7]),
        ] {
            store.batches.lock().unwrap().clear();
            let recipe = DerivedWeightRecipe::Concatenate {
                axis,
                inputs: vec![source("a"), source("b"), source("a")],
            };
            let read = if uncached {
                recipe.prepare_encoded_read_uncached(&store)
            } else {
                recipe.prepare_encoded_read(&store)
            }
            .unwrap()
            .unwrap();
            assert_eq!(*store.batches.lock().unwrap(), vec![vec!["a", "b", "a"]]);
            let mut output = [0; 12];
            read.read_into(&mut output).unwrap();
            assert_eq!(output, expected);
        }
    }
    let invalid = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![source("absent"), source("later")],
    };
    assert!(
        matches!(invalid.prepare_encoded_read(&store), Err(RecipeError::Store(StoreError::UnknownTensor { key })) if key == "absent")
    );
    store.batches.lock().unwrap().clear();
    let unsupported = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![
            source("absent"),
            DerivedWeightRecipe::NegLog {
                input: Box::new(source("a")),
            },
        ],
    };
    assert!(unsupported.prepare_encoded_read(&store).unwrap().is_none());
    assert!(store.batches.lock().unwrap().is_empty());
}
