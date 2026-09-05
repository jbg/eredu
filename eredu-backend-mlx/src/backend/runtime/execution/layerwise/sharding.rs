//! Execution-unit binding sharding and geometry tests.

use super::*;

/// Applies the local parallel layout to architecture-declared layer bindings.
pub fn shard_layer_bindings(
    bindings: Vec<WeightBinding>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<WeightBinding>, Error> {
    eredu_runtime::place_weight_bindings(bindings, store, layout)
        .map_err(|error| Error::Parallel(error.to_string()))
}

/// Applies only the primary physical placement to already member-selected bindings.
///
/// Independently addressable catalogs have already selected their expert member
/// axis. Their remaining primary placement is the ordinary tensor-parallel
/// projection selected by the architecture layout.
pub fn shard_addressable_member_bindings(
    bindings: Vec<WeightBinding>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<WeightBinding>, Error> {
    eredu_runtime::place_addressable_member_bindings(bindings, store, layout)
        .map_err(|error| Error::Parallel(error.to_string()))
}

#[cfg(test)]
mod shard_layer_bindings_tests {
    use super::*;
    use eredu_checkpoint::store::MemoryWeightStore;
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, ParameterRole, TensorPlacement};

    fn store() -> MemoryWeightStore {
        MemoryWeightStore::from_safetensors([(
            "model.weight".to_owned(),
            safetensors::Dtype::F32,
            vec![4, 2],
            vec![0; 4 * 2 * size_of::<f32>()],
        )])
        .unwrap()
    }

    fn layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "model.weight".into(),
            LocalTensorLayout::new(
                "projection",
                ParameterRole::ColumnProjection,
                vec![4, 2],
                vec![2, 2],
                TensorPlacement::Shard {
                    axis: 0,
                    index: 0,
                    parts: 2,
                },
                None,
                None,
                false,
            ),
        );
        layout
    }

    fn binding() -> WeightBinding {
        WeightBinding::new(
            "weight",
            "model.weight",
            TensorSelection::Full,
            (4 * 2 * size_of::<f32>()) as u64,
        )
        .unwrap()
    }

    #[test]
    fn sharding_uses_the_exact_architecture_logical_target() {
        let binding = binding().with_logical_target("model.weight").unwrap();

        let store = store();
        let sharded = shard_layer_bindings(vec![binding], &store, &layout()).unwrap();

        assert_eq!(
            sharded[0].source_recipe().infer(&store).unwrap().shape(),
            [2, 2]
        );
        assert_eq!(sharded[0].expected_bytes(), 16);
    }

    #[test]
    fn compound_expert_and_tensor_placement_selects_both_source_axes() {
        let store = MemoryWeightStore::from_safetensors([(
            "experts.weight".to_owned(),
            safetensors::Dtype::F32,
            vec![4, 8, 2],
            vec![0; 4 * 8 * 2 * size_of::<f32>()],
        )])
        .unwrap();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "experts.weight".into(),
            LocalTensorLayout::new(
                "experts",
                ParameterRole::ExpertIntermediate,
                vec![4, 8, 2],
                vec![2, 4, 2],
                TensorPlacement::Shard {
                    axis: 1,
                    index: 0,
                    parts: 2,
                },
                None,
                None,
                false,
            )
            .with_additional_placement(TensorPlacement::Range {
                axis: 0,
                start: 2,
                end: 4,
            }),
        );
        let binding = WeightBinding::new(
            "weight",
            "experts.weight",
            TensorSelection::Full,
            (4 * 8 * 2 * size_of::<f32>()) as u64,
        )
        .unwrap()
        .with_logical_target("experts.weight")
        .unwrap();

        let sharded = shard_layer_bindings(vec![binding], &store, &layout).unwrap();

        assert_eq!(
            sharded[0].source_recipe().infer(&store).unwrap().shape(),
            [2, 4, 2]
        );
        assert_eq!(sharded[0].expected_bytes(), 64);
    }

    #[test]
    fn sharding_rejects_a_missing_architecture_logical_target() {
        let error = shard_layer_bindings(vec![binding()], &store(), &layout()).unwrap_err();

        assert!(matches!(error, Error::Parallel(ref message)
            if message.contains("binding \"weight\" has no logical placement target")));
    }

    #[test]
    fn sharding_rejects_an_unmatched_architecture_logical_target() {
        let binding = binding().with_logical_target("model.weigth").unwrap();

        let error = shard_layer_bindings(vec![binding], &store(), &layout()).unwrap_err();

        assert!(matches!(error, Error::Parallel(ref message)
            if message.contains("binding \"weight\" targets unknown layout entry \"model.weigth\"")));
    }
}
