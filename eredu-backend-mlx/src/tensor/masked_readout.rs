//! Ordinary centroid-selected vocabulary projection and its native transports.
use super::*;
pub(super) fn execute(
    input: MaskedOutputProjectionInput<'_, MlxTensor>,
    context: &Stream,
) -> Result<MlxTensor, Error> {
    input.validate()?;
    let hidden_shape = input.hidden.shape();
    let batch = hidden_shape[0];
    let sequence = hidden_shape[1];
    let hidden_size = hidden_shape[2];
    let vocabulary = input.output_weight.shape()[0];
    let centroids = input.centroid_logits.shape()[2];
    if batch == 0 || sequence == 0 {
        return tensor(safemlx::ops::zeros::<f32>(
            &[batch, sequence, vocabulary],
            context,
        ));
    }
    let per_centroid = vocabulary / centroids;
    let top_indices = backend(argpartition_axis(
        input.centroid_logits.as_array(),
        -input.top_centroids,
        -1,
        context,
    ))?;
    let top_indices =
        backend(top_indices.try_index_device((.., .., -input.top_centroids..), context))?;
    let ordering = backend(
        input
            .token_ordering
            .as_array()
            .reshape(&[centroids, per_centroid], context),
    )?;
    let selected_tokens = backend(ordering.try_index_device(&top_indices, context))?;
    let flat_tokens = backend(selected_tokens.reshape(&[-1], context))?;
    let selected_weight = backend(
        input
            .output_weight
            .as_array()
            .try_index_device(&flat_tokens, context),
    )?;
    let selected_weight = backend(selected_weight.reshape(
        &[
            batch,
            sequence,
            input.top_centroids * per_centroid,
            hidden_size,
        ],
        context,
    ))?;
    let hidden = backend(
        input
            .hidden
            .as_array()
            .try_index_device((.., .., NewAxis, ..), context),
    )?;
    let selected_weight = backend(selected_weight.transpose_axes(&[0, 1, 3, 2], context))?;
    let selected_logits = backend(matmul(hidden, selected_weight, context))?;
    let selected_logits = backend(selected_logits.squeeze_axes(&[-2], context))?;
    let minimum = backend(selected_logits.min_axis(-1, true, context))?;
    let masked_value = backend(minimum.subtract(
        Array::try_from_f32(input.mask_margin).map_err(Error::backend_retained_source)?,
        context,
    ))?;
    let output = backend(full::<f32>(
        &[batch, sequence, vocabulary],
        masked_value,
        context,
    ))?;
    let scatter_indices = backend(selected_tokens.reshape(&[batch, sequence, -1], context))?;
    tensor(put_along_axis(
        output,
        scatter_indices,
        selected_logits,
        -1,
        context,
    ))
}
pub(crate) fn control_bytes(empty: bool) -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let calls = if empty { 1 } else { 17 };
    let sizes = [
        size_of::<MaskedOutputProjectionInput<'_, MlxTensor>>() * 2,
        size_of::<(&MaskedOutputProjectionInput<'_, MlxTensor>, &Stream)>(),
        size_of::<[&[i32]; 4]>(),
        size_of::<[i32; 8]>(),
        size_of::<[i32; 4]>() * 2,
        size_of::<[i32; 3]>() * 3,
        size_of::<[i32; 2]>(),
        size_of::<[i32; 1]>() * 2,
        size_of::<Array>() * calls,
        size_of::<Result<Array, safemlx::error::Exception>>() * calls,
        size_of::<Result<Array, Error>>() * calls,
        size_of::<Result<MlxTensor, Error>>(),
        size_of::<Result<(), Error>>(),
        safemlx::OriginalScopeObserver::control_bytes()?,
    ];
    sizes
        .into_iter()
        .try_fold(size_of_val(&sizes), usize::checked_add)
}
