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
    if stream.get_device()?.get_type()? == safemlx::DeviceType::Cpu
        && matches!(inputs.dtype(), Dtype::Bfloat16 | Dtype::Float16)
    {
        return grouped_matmul_cpu(inputs, weights, group_ids.as_ref(), stream);
    }
    if let Some(output) = super::matrix::bf16_row_projection(
        inputs,
        &weights.swap_axes(-1, -2, stream)?,
        Some(group_ids.as_ref()),
        stream,
    )? {
        return Ok(output);
    }
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

// CPU GatherMM accepts F32 only. Ordinary CPU matmul supports reduced precision;
// execute contiguous runs against views of each selected matrix, avoiding an
// expanded selection-sized weight tensor or a widened copy of the whole bank.
fn grouped_matmul_cpu(
    inputs: &Array,
    weights: &Array,
    ids: &Array,
    stream: &Stream,
) -> Result<Array> {
    use safemlx::ops::indexing::TryIndexOp;
    if inputs.ndim() != 2
        || weights.ndim() != 3
        || ids.ndim() != 1
        || ids.dim(0) != inputs.dim(0)
        || inputs.dim(1) != weights.dim(1)
        || inputs.dtype() != weights.dtype()
        || !matches!(
            ids.dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        )
    {
        return Err(safemlx::error::Exception::custom(
            "invalid grouped matrix multiplication geometry or dtype",
        ));
    }
    if inputs.dim(0) == 0 {
        return safemlx::ops::zeros_dtype(&[0, weights.dim(2)], inputs.dtype(), stream);
    }
    let valid = ids
        .ge(Array::from_int(0), stream)?
        .logical_and(ids.lt(Array::from_int(weights.dim(0)), stream)?, stream)?;
    if !valid.all(None, stream)?.try_item::<bool>(stream)? {
        return Err(safemlx::error::Exception::custom(
            "grouped matrix index is outside the bank",
        ));
    }
    let ids = ids.as_dtype(Dtype::Int32, stream)?.into_evaluated()?;
    let ids = ids.as_slice::<i32>();
    let mut products = Vec::new();
    let mut start = 0;
    while start < ids.len() {
        let mut end = start + 1;
        while end < ids.len() && ids[end] == ids[start] {
            end += 1;
        }
        let rows = inputs.try_index_device((start as i32..end as i32, ..), stream)?;
        // Integer advanced indexing materializes the transposed matrix in
        // column order. A unit slice retains the resident bank's strides and
        // avoids an extra expert-sized copy.
        let matrix = weights
            .try_index_device((ids[start]..ids[start] + 1, .., ..), stream)?
            .squeeze_axes(&[0], stream)?;
        products.push(rows.matmul(&matrix, stream)?);
        start = end;
    }
    safemlx::ops::concatenate_axis(&products, 0, stream)
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

#[cfg(test)]
mod reduction_tests {
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};

    #[test]
    fn cpu_bf16_grouped_projections_preserve_resident_bank_reduction_layout() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let fixture = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/bf16_grouped.safetensors"
            ),
            &stream,
        )
        .unwrap();
        for width in [128, 768, 2560] {
            let actual = super::grouped_matmul(
                &fixture[&format!("{width}.input")],
                fixture[&format!("{width}.weight")]
                    .swap_axes(-1, -2, &stream)
                    .unwrap(),
                &fixture["ids"],
                false,
                &stream,
            )
            .unwrap()
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
            let expected = fixture[&format!("{width}.output")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let mismatches = actual
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(mismatches, 0, "selected projection width={width}");
        }
    }
}
