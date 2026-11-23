use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe as Recipe, RecipeDtype},
    store::{
        CheckpointLease, CheckpointSource, EncodedTensorLease, ReadPolicy, SafetensorsWeightStore,
        StoreError, TensorReadRequest, TensorSelection,
    },
};
use eredu_core::checkpoint_mechanisms::recipe_execution::{self, RecipeExecution};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

struct RetainingImport {
    allow_detach: bool,
}

impl RecipeExecution for RetainingImport {
    type Value = Vec<u8>;
    type Retained = CheckpointLease;
    type Error = StoreError;

    fn import(
        &self,
        lease: CheckpointLease,
        retained: &mut Vec<CheckpointLease>,
    ) -> Result<Vec<u8>, StoreError> {
        assert!(lease.bounded_read_proof().physically_bounded);
        let bytes = lease.encoded_bytes().unwrap().to_vec();
        retained.push(lease);
        Ok(bytes)
    }

    fn apply(
        &self,
        operation: &Recipe,
        inputs: Vec<Vec<u8>>,
        _: &mut Vec<CheckpointLease>,
    ) -> Result<Vec<u8>, StoreError> {
        match operation {
            Recipe::Concatenate { .. } => Ok(inputs.into_iter().flatten().collect()),
            _ => Err(StoreError::Internal("unsupported fixture operation".into())),
        }
    }

    fn store_error(&self, error: StoreError) -> StoreError {
        error
    }

    fn can_retry_join_operand(&self, error: &StoreError) -> bool {
        self.allow_detach && matches!(error, StoreError::CapacityExhausted { .. })
    }

    fn detach_join_operand(
        &self,
        _: &Vec<u8>,
        retained: &mut Vec<CheckpointLease>,
    ) -> Result<(), StoreError> {
        retained.clear();
        Ok(())
    }
}

#[test]
fn checkpoint_recipe_preserves_bounded_selections_and_releases_join_leases() {
    let fixture = tempfile::tempdir().unwrap();
    for (key, values) in [("left", [1, 2, 3]), ("right", [11, 12, 13])] {
        serialize_to_file(
            [(key, TensorView::new(Dtype::U8, vec![3], &values).unwrap())],
            None,
            &fixture.path().join(format!("{key}.safetensors")),
        )
        .unwrap();
    }
    std::fs::write(
        fixture.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"left":"left.safetensors","right":"right.safetensors"}}"#,
    )
    .unwrap();
    let source = SafetensorsWeightStore::open_with_max_cached_shards(fixture.path(), 1).unwrap();
    let operand = |key| {
        Recipe::source(
            key,
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 3,
            },
        )
    };
    let join = Recipe::Concatenate {
        axis: 0,
        inputs: vec![operand("right"), operand("left")],
    };

    recipe_execution::preflight(
        &join,
        &source,
        &|_, metadata| {
            assert_eq!(metadata.dtype(), &RecipeDtype::U8);
            Ok::<_, eredu_checkpoint::recipe::RecipeError>(())
        },
        &std::convert::identity,
        &|_, validate| validate(),
    )
    .unwrap();
    assert!(source
        .source_diagnostics()
        .unwrap()
        .payload_shard_paths
        .is_empty());

    assert!(matches!(
        recipe_execution::execute(
            &join,
            &source,
            &RetainingImport {
                allow_detach: false
            },
            &mut Vec::new(),
        ),
        Err(StoreError::CapacityExhausted { .. })
    ));
    let value = recipe_execution::execute(
        &join,
        &source,
        &RetainingImport { allow_detach: true },
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(value, [12, 13, 2, 3]);
    source
        .acquire_lease(TensorReadRequest {
            key: "right".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
}
