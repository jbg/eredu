//! The existing native segmented loop with ordinary or paid output storage.
use super::*;
use eredu_nn::workspace::WorkspaceContext;

pub(super) fn run(
    input: SegmentedAttentionInput<'_, MlxTensor>,
    context: &Stream,
    metadata: Option<&WorkspaceContext>,
) -> Result<MlxTensor, ComputeError> {
    if let Some(metadata) = metadata {
        metadata.charge_metadata(std::mem::size_of::<(
            SegmentedAttentionInput<'_, MlxTensor>,
            &Stream,
            Option<&WorkspaceContext>,
            Vec<Array>,
            [Array; 4],
            [i32; 5],
            [i32; 4],
            Result<MlxTensor, ComputeError>,
        )>())?;
    }
    input.validate_with_diagnostic(|message| match metadata {
        Some(metadata) => metadata.metadata_error(message),
        None => ComputeError::backend(message.to_string()),
    })?;
    let heads = input.queries.dim(1);
    let query_dimensions = input.queries.dim(2);
    let value_dimensions = input.values.dim(2);
    let mut outputs = match metadata {
        Some(metadata) => metadata.metadata_vec(input.segment_lengths.len())?,
        None => Vec::with_capacity(input.segment_lengths.len()),
    };
    let mut start = 0i32;
    for &length in input.segment_lengths {
        let end = start + length;
        let prepare = |value: &Array, dimensions: i32| -> Result<Array, ComputeError> {
            let value = compute(value.try_index_device((start..end, .., ..), context))?;
            let value = compute(value.transpose_axes(&[1, 0, 2], context))?;
            compute(value.reshape(&[1, heads, length, dimensions], context))
        };
        let queries = prepare(input.queries.as_array(), query_dimensions)?;
        let keys = prepare(input.keys.as_array(), query_dimensions)?;
        let values = prepare(input.values.as_array(), value_dimensions)?;
        let output = compute(safemlx::fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            input.scale,
            Option::<ScaledDotProductAttentionMask<'_>>::None,
            Option::<&Array>::None,
            context,
        ))?;
        let output = compute(output.reshape(&[heads, length, value_dimensions], context))?;
        outputs.push(compute(output.transpose_axes(&[1, 0, 2], context))?);
        start = end;
    }
    compute_tensor(concatenate_axis(&outputs, 0, context))
}
