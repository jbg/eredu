use super::*;
use eredu_checkpoint::{
    StoredDtype,
    recipe::{EncodedRecipeKeysPlan, ReadBatchCatalogPlan, RecipeDtype},
    store::{
        MemoryEncodedReadPlan, MemoryWeightStore, SafetensorsEncodedReadPlan,
        SafetensorsWeightStore, TensorMetadata, TensorSelection,
    },
};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

struct Measured<'a> {
    inner: AdmittedRecipeConstruction<'a>,
    peak: u64,
    calls: [usize; 4],
}
impl<'a> Measured<'a> {
    fn new(pool: &'a MemoryLedger) -> Self {
        Self {
            inner: AdmittedRecipeConstruction::new(pool),
            peak: 0,
            calls: [0; 4],
        }
    }
    fn record<P: SharedNativeInitializer>(&mut self, plan: &P, kind: usize) {
        let required = MemoryLedger::shared_native_initialization_required_bytes(plan).unwrap();
        self.peak = self
            .peak
            .max(self.inner.pool.payload_used_bytes().unwrap() + required);
        self.calls[kind] += 1;
    }
}
impl EncodedRecipeConstruction for Measured<'_> {
    type Error = EncodedRecipeConstructionError;
    type Metadata = InitializedSharedNative<RecipeMetadata>;
    type Mapping = InitializedSharedNative<EncodedRecipeMapping>;
    type Ranges = InitializedSharedNative<SelectionReadRanges>;
    type ChildCustody = (
        SharedNativeInitializationCustody,
        SharedNativeInitializationCustody,
    );
    fn infer<C: RecipeCatalog + ?Sized>(
        &mut self,
        recipe: &DerivedWeightRecipe,
        catalog: &C,
    ) -> Result<Self::Metadata, Self::Error> {
        self.record(
            &RecipeInferencePlan::new(RecipeInferenceInput::Derived(recipe), catalog).unwrap(),
            0,
        );
        self.inner.infer(recipe, catalog)
    }
    fn mapping(
        &mut self,
        plan: EncodedRecipeMappingPlan<'_>,
    ) -> Result<Self::Mapping, Self::Error> {
        self.record(&plan, 1);
        self.inner.mapping(plan)
    }
    fn ranges(
        &mut self,
        plan: SelectionReadDestinationPlan<'_, '_>,
    ) -> Result<Self::Ranges, Self::Error> {
        self.record(&plan, 2);
        self.inner.ranges(plan)
    }
    fn children(
        &mut self,
        plan: EncodedRecipeChildrenPlan<Self::Mapping>,
    ) -> Result<EncodedRecipeChildren<Self::Mapping, Self::ChildCustody>, Self::Error> {
        self.record(&plan, 3);
        self.inner.children(plan)
    }
}
fn recipe() -> DerivedWeightRecipe {
    use DerivedWeightRecipe as R;
    R::Transpose {
        axes: vec![1, 0, 2],
        input: Box::new(R::Reshape {
            shape: vec![1, 2, 3],
            input: Box::new(R::Select {
                selection: TensorSelection::Indices {
                    axis: 1,
                    indices: vec![8, 3, 0],
                },
                input: Box::new(R::Concatenate {
                    axis: 1,
                    inputs: ["a", "雪", "a"]
                        .map(|key| R::source(key, TensorSelection::Full))
                        .into(),
                }),
            }),
        }),
    }
}
fn memory() -> ((), MemoryWeightStore) {
    (
        (),
        MemoryWeightStore::from_safetensors([
            ("a".into(), Dtype::U8, vec![2, 3], vec![1, 2, 3, 4, 5, 6]),
            (
                "雪".into(),
                Dtype::U8,
                vec![2, 3],
                vec![11, 12, 13, 14, 15, 16],
            ),
        ])
        .unwrap(),
    )
}
fn file() -> (tempfile::TempDir, SafetensorsWeightStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.safetensors");
    serialize_to_file(
        [
            (
                "a",
                TensorView::new(Dtype::U8, vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap(),
            ),
            (
                "雪",
                TensorView::new(Dtype::U8, vec![2, 3], &[11, 12, 13, 14, 15, 16]).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(path).unwrap();
    (dir, store)
}
fn qualified() -> bool {
    let plan = EncodedRecipeMappingPlan::source(0..1).unwrap();
    let result = MemoryLedger::shared_native_initialization_required_bytes(&plan);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(_) => true,
        Err(WorkingMemoryError::UnknownBound) => false,
        Err(e) => panic!("{e}"),
    }
}
macro_rules! sequence {
    ($name:ident, $fixture:ident, $plan:ident) => {
        #[test]
        fn $name() {
            if !qualified() {
                return;
            }
            let mut measured_peak = 0;
            for attempt in 0..3 {
                let (_dir, source) = $fixture();
                let recipe = recipe();
                let limit = match attempt {
                    0 => 1 << 24,
                    1 => measured_peak,
                    _ => measured_peak - 1,
                };
                let pool = crate::working_memory::memory_fixture::host_ledger(limit, 0).unwrap();
                let keys = pool
                    .initialize_shared_native(EncodedRecipeKeysPlan::new(&recipe).unwrap().unwrap())
                    .unwrap();
                let batch = pool
                    .initialize_shared_native($plan::new(&source, keys.output().keys()).unwrap())
                    .unwrap();
                let catalog = pool
                    .initialize_shared_native(
                        ReadBatchCatalogPlan::new(batch.output().tensors()).unwrap(),
                    )
                    .unwrap();
                let prerequisites = pool.payload_used_bytes().unwrap();
                let mut construction = Measured::new(&pool);
                let result = catalog.output().compile_recipe(&recipe, &mut construction);
                if attempt == 2 {
                    let failure = result.unwrap_err();
                    let accounting = match &failure {
                        EncodedRecipeConstructionError::Inference(e) => e.accounting_failure(),
                        EncodedRecipeConstructionError::Mapping(e) => e.accounting_failure(),
                        EncodedRecipeConstructionError::Ranges(e) => e.accounting_failure(),
                        EncodedRecipeConstructionError::Children(e) => e.accounting_failure(),
                        e => panic!("{e:?}"),
                    };
                    assert!(matches!(
                        accounting,
                        Some(WorkingMemoryError::Domain(
                            eredu_core::MemoryDomainError::BudgetExceeded { .. }
                        ))
                    ));
                    assert_eq!(pool.payload_used_bytes().unwrap(), prerequisites);
                    drop(failure);
                    drop(catalog);
                    drop((batch, keys));
                } else {
                    assert!(construction.calls.iter().all(|n| *n > 0));
                    if attempt == 0 {
                        measured_peak = construction.peak;
                    } else {
                        assert_eq!(construction.peak, measured_peak);
                    }
                    let (output, mapping) = result.unwrap().unwrap();
                    assert_eq!(output.output().shape(), [2, 1, 3]);
                    assert_eq!(output.output().dtype(), &RecipeDtype::U8);
                    assert_eq!(
                        pool.payload_used_bytes().unwrap(),
                        prerequisites + output.original_bytes() + mapping.original_bytes()
                    );
                    assert_eq!(
                        mapping.output().ranges().collect::<Vec<_>>(),
                        [
                            (14..15, 0..1),
                            (6..7, 1..2),
                            (0..1, 2..3),
                            (17..18, 3..4),
                            (9..10, 4..5),
                            (3..4, 5..6),
                        ]
                    );
                    drop(catalog);
                    drop(keys);
                    let batch = batch.into_owned_read();
                    let projected = pool
                        .initialize_shared_native(batch.project(mapping.output()).unwrap())
                        .unwrap();
                    drop((recipe, source, mapping));
                    let mut bytes = [0; 6];
                    projected.output().read_into(&mut bytes).unwrap();
                    assert_eq!(bytes, [3, 11, 1, 6, 14, 4]);
                    let mut short = [99; 5];
                    assert!(projected.output().read_into(&mut short).is_err());
                    assert_eq!(short, [99; 5]);
                    drop((projected, output));
                }
                assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            }
        }
    };
}
sequence!(
    memory_recursive_compilation_and_projection_share_admission,
    memory,
    MemoryEncodedReadPlan
);
sequence!(
    file_recursive_compilation_and_projection_share_admission,
    file,
    SafetensorsEncodedReadPlan
);

#[test]
fn unsupported_recipes_release_temporaries_and_owned_validation_errors_keep_custody() {
    if !qualified() {
        return;
    }
    let tensors = [TensorMetadata {
        name: "a".into(),
        logical_shape: vec![2, 3],
        physical_shape: vec![2, 3],
        stored_dtype: StoredDtype::U8,
        encoded_byte_len: 6,
        backing_shard: None,
    }];
    let catalog = ReadBatchCatalogPlan::new(&tensors)
        .unwrap()
        .construct(())
        .unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let source = || Box::new(DerivedWeightRecipe::source("a", TensorSelection::Full));
    for recipe in [
        DerivedWeightRecipe::Cast {
            input: source(),
            dtype: RecipeDtype::F32,
        },
        DerivedWeightRecipe::Transpose {
            input: source(),
            axes: vec![1, 0],
        },
    ] {
        assert!(
            catalog
                .compile_recipe(&recipe, &mut AdmittedRecipeConstruction::new(&pool))
                .unwrap()
                .is_none()
        );
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    let invalid = DerivedWeightRecipe::Transpose {
        input: source(),
        axes: vec![1, 1],
    };
    let error = catalog
        .compile_recipe(&invalid, &mut AdmittedRecipeConstruction::new(&pool))
        .unwrap_err();
    drop(catalog);
    drop((invalid, tensors));
    let EncodedRecipeConstructionError::Inference(failure) = &error else {
        panic!("{error:?}")
    };
    assert!(matches!(
        failure.constructor_failure(),
        Some(RecipeInferenceError::Recipe(
            RecipeError::InvalidPermutation { .. }
        ))
    ));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
