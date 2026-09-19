use super::*;
use crate::{
    resolve_replicated_text_transform_source as resolve, LocalModelLayout, LocalTensorLayout,
    TensorPlacement,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeMetadata},
    store::{
        CheckpointLease, CheckpointSource, MemoryWeightStore, StoreError, TensorMetadata,
        TensorReadRequest, TensorSelection, WeightStoreDiagnostics,
    },
};

struct MetadataOnly<'a>(&'a dyn CheckpointSource);
impl CheckpointSource for MetadataOnly<'_> {
    fn source_keys(&self) -> Vec<String> {
        self.0.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.0.source_metadata(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("source resolution must not acquire payloads")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("source resolution must not inspect runtime diagnostics")
    }
}

fn task(shape: &[usize], dtype: StoredDtype) -> ReplicatedTextMaterializationTask {
    let bytes = shape.iter().product::<usize>()
        * match dtype {
            StoredDtype::F16 | StoredDtype::BF16 => 2,
            StoredDtype::F32 | StoredDtype::U32 => 4,
            _ => panic!("unsupported fixture dtype"),
        };
    let encoding = SourceTensorEncoding::Safetensors(dtype);
    let format = LinearFormat::Affine(AffineQuantization::default());
    ReplicatedTextMaterializationTask::from_exact_source(
        "projection",
        ReplicatedTextPhysicalSource::new(
            "checkpoint.weight",
            "checkpoint.weight",
            "/checkpoint/model.safetensors",
            "checkpoint.weight",
            encoding.clone(),
            bytes as u64,
        )
        .unwrap(),
        vec![],
        shape.to_vec(),
        shape.to_vec(),
        ReplicatedTextParameterRole::LinearWeight,
        ReplicatedTextParameterOwner::StaticRole("projection".into()),
        format,
        WeightLoweringKind::Transform,
        WeightLoweringDescriptor::new(
            encoding,
            format,
            shape.to_vec(),
            shape.to_vec(),
            Some(shape.len() - 1),
        )
        .unwrap(),
    )
    .unwrap()
}

fn store(shape: &[usize], dtype: safetensors::Dtype) -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([(
        "checkpoint.weight".into(),
        dtype,
        shape.to_vec(),
        vec![0; shape.iter().product::<usize>() * dtype.size()],
    )])
    .unwrap()
}

fn layout(shape: &[usize], local: &[usize], placement: TensorPlacement) -> LocalModelLayout {
    let mut layout = LocalModelLayout::default();
    layout.insert(
        "projection".into(),
        LocalTensorLayout::new(
            "projection",
            ParameterRole::FeedForwardIntermediate,
            shape.to_vec(),
            local.to_vec(),
            placement,
            None,
            None,
            false,
        ),
    );
    layout
}

#[test]
fn floating_precisions_preserve_exact_source_metadata_and_recipe() {
    for (stored, safe, recipe_dtype) in [
        (StoredDtype::F16, safetensors::Dtype::F16, RecipeDtype::F16),
        (
            StoredDtype::BF16,
            safetensors::Dtype::BF16,
            RecipeDtype::BF16,
        ),
        (StoredDtype::F32, safetensors::Dtype::F32, RecipeDtype::F32),
    ] {
        let task = task(&[8, 64], stored);
        let source = store(&[8, 64], safe);
        let (recipe, metadata) = resolve(&MetadataOnly(&source), &task, None).unwrap();
        assert_eq!(
            recipe,
            DerivedWeightRecipe::source("checkpoint.weight", TensorSelection::Full)
        );
        assert_eq!(metadata.shape(), [8, 64]);
        assert_eq!(metadata.dtype(), &recipe_dtype);
        assert_eq!(metadata.byte_len(), (8 * 64 * safe.size()) as u64);
    }
}

#[test]
fn complete_derived_output_is_checked_before_rank_local_selection() {
    let source = store(&[8, 64], safetensors::Dtype::F32);
    let mut task = task(&[8, 64], StoredDtype::F32);
    task.lowering = WeightLoweringKind::DerivedTransform;
    task.derived_recipe = Some(DerivedWeightRecipe::source(
        "checkpoint.weight",
        TensorSelection::Full,
    ));
    task.derived_output = Some(RecipeMetadata {
        shape: vec![8, 64],
        dtype: RecipeDtype::F32,
        byte_len: 8 * 64 * 4,
    });
    let layout = layout(
        &[8, 64],
        &[4, 64],
        TensorPlacement::Range {
            axis: 0,
            start: 4,
            end: 8,
        },
    );
    let (_, local) = resolve(&source, &task, Some(&layout)).unwrap();
    assert_eq!(local.shape(), [4, 64]);
    assert_eq!(local.byte_len(), 4 * 64 * 4);
    // Local metadata cannot stand in for admission of the complete recipe.
    let complete = task.derived_output.clone().unwrap();
    let mut wrong_dtype = complete.clone();
    wrong_dtype.dtype = RecipeDtype::F16;
    let mut wrong_bytes = complete;
    wrong_bytes.byte_len -= 1;
    for invalid in [local, wrong_dtype, wrong_bytes] {
        task.derived_output = Some(invalid);
        let error = resolve(&source, &task, Some(&layout)).unwrap_err();
        assert!(error.to_string().contains("admitted derived output"));
    }
}

#[test]
fn additional_expert_selection_precedes_primary_tensor_partition() {
    let shape = [4, 8, 64];
    let values = (0..shape.iter().product::<usize>())
        .map(|index| index as f32 + 0.25)
        .collect::<Vec<_>>();
    let source = MemoryWeightStore::from_safetensors([(
        "checkpoint.weight".into(),
        safetensors::Dtype::F32,
        shape.to_vec(),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )])
    .unwrap();
    let task = task(&shape, StoredDtype::F32);
    let mut layout = layout(
        &shape,
        &[2, 8, 32],
        TensorPlacement::Range {
            axis: 2,
            start: 32,
            end: 64,
        },
    );
    let tensor = layout
        .tensor("projection")
        .unwrap()
        .clone()
        .with_additional_placement(TensorPlacement::Indices {
            axis: 0,
            indices: vec![3, 1],
        });
    layout.insert("projection".into(), tensor);
    let (recipe, metadata) = resolve(&source, &task, Some(&layout)).unwrap();
    assert_eq!(metadata.shape(), [2, 8, 32]);
    let mut actual = vec![0; metadata.byte_len() as usize];
    recipe
        .prepare_encoded_read(&source)
        .unwrap()
        .unwrap()
        .read_into(&mut actual)
        .unwrap();
    let mut expected = Vec::new();
    for expert in [3, 1] {
        for row in 0..8 {
            for column in 32..64 {
                expected.extend_from_slice(&values[(expert * 8 + row) * 64 + column].to_le_bytes());
            }
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn invalid_lowerings_sources_and_rank_placements_refuse() {
    let source = store(&[8, 64], safetensors::Dtype::F32);
    let mut selected = task(&[8, 64], StoredDtype::F32);
    selected.lowering = WeightLoweringKind::Direct;
    assert!(resolve(&source, &selected, None)
        .unwrap_err()
        .to_string()
        .contains("transform lowering"));
    selected.lowering = WeightLoweringKind::Transform;
    assert!(
        resolve(&source, &selected, Some(&LocalModelLayout::default()))
            .unwrap_err()
            .to_string()
            .contains("no source local placement")
    );
    let invalid = layout(
        &[8, 64],
        &[8, 64],
        TensorPlacement::Shard {
            axis: 0,
            index: 0,
            parts: 3,
        },
    );
    assert!(matches!(
        resolve(&source, &selected, Some(&invalid)),
        Err(crate::TransformSourceError::Placement(_))
    ));
    let mut missing = selected.clone();
    missing.sources = vec!["missing.weight".into()];
    assert!(matches!(
        resolve(&source, &missing, None),
        Err(crate::TransformSourceError::Recipe(
            eredu_checkpoint::recipe::RecipeError::Store(StoreError::UnknownTensor { .. })
        ))
    ));
    for (shape, dtype, stored) in [
        (vec![64], safetensors::Dtype::F32, StoredDtype::F32),
        (vec![8, 64], safetensors::Dtype::U32, StoredDtype::U32),
    ] {
        let source = store(&shape, dtype);
        let selected = task(&shape, stored);
        assert!(resolve(&source, &selected, None)
            .unwrap_err()
            .to_string()
            .contains("floating matrix"));
    }
}
