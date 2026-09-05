use super::*;

#[cfg(test)]
pub(super) fn gguf_test_plan(
    checkpoint: &eredu_gguf::Checkpoint,
) -> eredu_checkpoint::schema::GgufCheckpointPlan {
    use eredu_checkpoint::schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    };
    use eredu_gguf::GgmlType;

    let constraints = checkpoint
        .tensors()
        .map(|tensor| {
            let descriptor = tensor.descriptor();
            let operation = match descriptor.ggml_type {
                GgmlType::I32 => TensorOperation::I32,
                GgmlType::MxFp4 => TensorOperation::MxFp4Matrix,
                GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => TensorOperation::Dense,
                _ => TensorOperation::Matrix,
            };
            GgufTensorConstraint::required(
                descriptor.name.clone(),
                descriptor
                    .row_major_shape()
                    .into_iter()
                    .map(|dimension| usize::try_from(dimension).unwrap())
                    .collect::<Vec<_>>(),
                GgufTypeConstraint::OperationClass(operation),
            )
        })
        .collect();
    GgufCheckpointPlan::new(
        "test GGUF catalog",
        constraints,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .unwrap()
}

#[cfg(test)]
/// Opens a GGUF source under a catalog-derived strict test contract.
pub fn open_gguf_checkpoint_source_for_test<F>(
    checkpoint: GgufCheckpoint,
    translate: F,
) -> Result<NeutralGgufWeightStore, CheckpointMaterializationError>
where
    F: FnMut(&str) -> String,
{
    let plan = gguf_test_plan(checkpoint.catalog());
    let tensor_mapping = checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(|error| StoreError::Gguf {
            key: String::new(),
            message: error.to_string(),
        })?;
    eredu_checkpoint::gguf_store::open_prepared_gguf_source(
        checkpoint.catalog().clone(),
        &plan,
        &tensor_mapping,
        eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS,
    )
    .map_err(CheckpointMaterializationError::from)
}
