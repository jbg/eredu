//! Device-side grouped-selection helpers.

use safemlx::{
    error::Result,
    ops::{argsort, gather_mm, indexing::take_axis},
    Array, Dtype, Stream,
};

/// Device-side selection plan for grouped execution.
///
/// The plan is produced by sorting flattened group/group ids. `selection_indices` maps every sorted
/// selection back to its original flattened selection position. For top-k selection,
/// `token_indices` identifies the source token for each selection.
#[derive(Debug)]
pub struct GroupedSelectionPlan {
    /// Group or group id for each sorted selection.
    pub sorted_group_ids: Array,
    /// Original flattened selection index for each sorted selection.
    pub selection_indices: Array,
    /// Source row for each sorted selection.
    pub token_indices: Array,
}

/// Sort flattened group ids on-device and return indices useful for grouped kernels.
///
/// `group_ids` can be 1-D (`[selections]`) or 2-D (`[tokens, slots]`). The returned
/// `sorted_group_ids` are suitable for `grouped_matmul(..., sorted_indices = true)`, while
/// `token_indices` can be used to gather source rows and later reduce grouped outputs back to
/// tokens with [`segment_sum_by_index`].
pub fn group_by_id(
    group_ids: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<GroupedSelectionPlan> {
    let stream = stream.as_ref();
    let group_ids = group_ids.as_ref();
    let top_k = if group_ids.ndim() >= 2 {
        group_ids.dim(-1)
    } else {
        1
    };
    let flat_group_ids = group_ids
        .reshape(&[-1], stream)?
        .as_dtype(Dtype::Int32, stream)?;
    let order = argsort(&flat_group_ids, stream)?;
    let sorted_group_ids = flat_group_ids.take(&order, stream)?;
    let selection_indices = order.as_dtype(Dtype::Int32, stream)?;
    let token_indices = selection_indices.floor_divide(Array::from_int(top_k), stream)?;

    Ok(GroupedSelectionPlan {
        sorted_group_ids,
        selection_indices,
        token_indices,
    })
}

/// Matrix multiplication for rows assigned to variable-sized groups.
///
/// `inputs` has shape `[selections, in_dim]`, `weights` has shape
/// `[num_groups, in_dim, out_dim]`, and `group_ids` has shape `[selections]`. When `group_ids` are
/// already sorted, pass `sorted_indices = true` so MLX can use its sorted gather-matmul path.
pub fn grouped_matmul(
    inputs: impl AsRef<Array>,
    weights: impl AsRef<Array>,
    group_ids: impl AsRef<Array>,
    sorted_indices: bool,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let inputs = inputs.as_ref();
    let weights = weights.as_ref();
    let selections = inputs.dim(0);
    let in_dim = inputs.dim(-1);
    let out_dim = weights.dim(-1);
    let inputs = inputs.reshape(&[selections, 1, in_dim], stream)?;
    gather_mm(
        &inputs,
        weights,
        None::<&Array>,
        group_ids.as_ref(),
        sorted_indices,
        stream,
    )?
    .reshape(&[selections, out_dim], stream)
}

/// Gather source rows according to a selection plan.
pub fn gather_grouped_rows(
    rows: impl AsRef<Array>,
    plan: &GroupedSelectionPlan,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    take_axis(rows, &plan.token_indices, 0, stream)
}

/// Gather flattened per-selection values according to a selection plan.
///
/// This is useful for top-k selection weights with shape `[tokens, top_k]`.
pub fn gather_selection_values(
    values: impl AsRef<Array>,
    plan: &GroupedSelectionPlan,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    values
        .as_ref()
        .reshape(&[-1], stream.as_ref())?
        .take(&plan.selection_indices, stream)
}

/// Reduce grouped values back to source rows using summation.
///
/// `values` should have shape `[selections, ...]`, and `indices` should have shape `[selections]`.
#[cfg(test)]
pub fn segment_sum_by_index(
    values: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    num_segments: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    safemlx::ops::segment_sum(values, indices, num_segments, 0, stream)
}

/// Build a sorted top-k selection plan from `[tokens, top_k]` group ids.
pub fn topk_group_plan(
    group_indices: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<GroupedSelectionPlan> {
    group_by_id(group_indices, stream)
}

#[cfg(test)]
mod tests;
