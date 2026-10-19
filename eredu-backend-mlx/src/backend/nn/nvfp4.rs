//! Bounded NVFP4 fallback. Conversion never expands an entire expert bank.

use std::collections::BTreeMap;

use eredu_checkpoint::nvfp4::NvFp4MatrixFormat;
use eredu_nn::LinearRowLayout;
use safemlx::{
    error::Exception,
    ops::{
        self,
        indexing::{scatter_single, TryIndexOp},
    },
    Array, Dtype, Stream,
};

pub(crate) const ROW_TILE: usize = 128;

fn error(detail: impl ToString) -> Exception {
    Exception::custom(detail.to_string())
}

fn decode_tile(
    weight: &Array,
    scales: &Array,
    global: &Array,
    rows: std::ops::Range<usize>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let columns = usize::try_from(weight.dim(-1)).map_err(error)? * 8;
    let start = i32::try_from(rows.start).map_err(error)?;
    let end = i32::try_from(rows.end).map_err(error)?;
    let packed_view = weight.try_index_device((start..end, ..), stream)?;
    let packed = packed_view.evaluated()?;
    let scales_view = scales.try_index_device((start..end, ..), stream)?;
    let scales = scales_view.evaluated()?;
    let global = global.evaluated()?;
    let bytes = packed
        .try_to_vec::<u32>()
        .map_err(error)?
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let block_scales = scales.try_to_vec::<u8>().map_err(error)?;
    let global = global.try_to_vec::<f32>().map_err(error)?;
    let format = NvFp4MatrixFormat::new(rows.len(), columns, 1).map_err(error)?;
    let limit = rows
        .len()
        .checked_mul(columns)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| error("NVFP4 tile size overflow"))?;
    let decoded = format
        .decode_rows(&bytes, &block_scales, &global, 0..rows.len(), limit)
        .map_err(error)?;
    Array::from_slice(&decoded, &[end - start, columns as i32]).copy(stream)
}

fn geometry(
    weight: &Array,
    scales: &Array,
    global: &Array,
    layout: LinearRowLayout,
) -> Result<(usize, usize, usize), Exception> {
    if weight.ndim() != 2
        || scales.ndim() != 2
        || weight.dtype() != Dtype::Uint32
        || scales.dtype() != Dtype::Uint8
        || global.dtype() != Dtype::Float32
    {
        return Err(error(
            "NVFP4 requires a U32 matrix, U8 block scales, and F32 global scales",
        ));
    }
    let rows = weight.dim(0) as usize;
    let columns = weight.dim(1) as usize * 8;
    let width = layout.rows_per_partition(rows).map_err(error)?;
    NvFp4MatrixFormat::new(rows, columns, layout.partitions()).map_err(error)?;
    if scales.shape() != [rows as i32, (columns / 16) as i32]
        || global.size() != layout.partitions()
    {
        return Err(error("NVFP4 block/global scale geometry mismatch"));
    }
    Ok((rows, columns, width))
}

/// Executes one selected matrix using at most 128 converted output rows at once.
pub(crate) fn linear(
    input: &Array,
    weight: &Array,
    scales: &Array,
    global: &Array,
    layout: LinearRowLayout,
    stream: &Stream,
) -> Result<Array, Exception> {
    if weight.dtype().is_float() {
        return super::linear::dense_projection(input, weight, stream);
    }
    let (rows, columns, partition_rows) = geometry(weight, scales, global, layout)?;
    if input.ndim() == 0 || input.dim(-1) as usize != columns || !input.dtype().is_float() {
        return Err(error("NVFP4 projection input geometry or dtype mismatch"));
    }
    let globals = global.reshape(&[layout.partitions() as i32], stream)?;
    let mut tiles = Vec::new();
    let mut start = 0;
    while start < rows {
        let part = start / partition_rows;
        let end = (start + ROW_TILE).min((part + 1) * partition_rows);
        let scale = globals.try_index_device(part as i32, stream)?;
        let decoded = decode_tile(weight, scales, &scale, start..end, stream)?
            .as_dtype(input.dtype(), stream)?;
        let value = ops::matmul(input, decoded.transpose(stream)?, stream)?;
        // Complete before dropping the tile to bound both conversion and graph retention.
        safemlx::transforms::eval([&value])?;
        tiles.push(value);
        start = end;
    }
    ops::concatenate_axis(&tiles, -1, stream)
}

/// Projects only selected expert matrices, preserving the caller's route order.
pub(crate) fn grouped_linear(
    input: &Array,
    weight: &Array,
    scales: &Array,
    global: &Array,
    ids: &Array,
    layout: LinearRowLayout,
    stream: &Stream,
) -> Result<Array, Exception> {
    if weight.dtype().is_float() {
        return super::grouping::grouped_matmul(
            input,
            &weight.swap_axes(-1, -2, stream)?,
            ids,
            false,
            stream,
        );
    }
    if weight.ndim() != 3
        || scales.ndim() != 3
        || input.ndim() != 2
        || ids.ndim() != 1
        || ids.dim(0) != input.dim(0)
        || global.ndim() == 0
        || global.dim(0) != weight.dim(0)
    {
        return Err(error("NVFP4 selected-bank geometry mismatch"));
    }
    let native_ids = ids.as_dtype(Dtype::Int32, stream)?;
    let evaluated_ids = native_ids.evaluated()?;
    let mut routes = BTreeMap::<i32, Vec<i32>>::new();
    for (route, id) in evaluated_ids
        .try_to_vec::<i32>()
        .map_err(error)?
        .into_iter()
        .enumerate()
    {
        if id < 0 || id >= weight.dim(0) {
            return Err(error("NVFP4 selected expert is outside the bank"));
        }
        routes
            .entry(id)
            .or_default()
            .push(i32::try_from(route).map_err(error)?);
    }
    let mut output = ops::zeros_dtype(&[input.dim(0), weight.dim(1)], input.dtype(), stream)?;
    for (group, routes) in routes {
        let indices = Array::from_slice(&routes, &[routes.len() as i32]);
        let rows = input.take_axis(&indices, 0, stream)?;
        let value = linear(
            &rows,
            &weight.try_index_device(group, stream)?,
            &scales.try_index_device(group, stream)?,
            &global.try_index_device(group, stream)?,
            layout,
            stream,
        )?;
        output = scatter_single(
            output,
            &indices,
            value.reshape(&[routes.len() as i32, 1, weight.dim(1)], stream)?,
            0,
            stream,
        )?;
        safemlx::transforms::eval([&output])?;
    }
    Ok(output)
}

/// Decodes an explicitly reserved effective parameter for inspection or editing.
pub(crate) fn dequantize(
    weight: &Array,
    scales: &Array,
    global: &Array,
    layout: LinearRowLayout,
    stream: &Stream,
) -> Result<Array, Exception> {
    if weight.ndim() == 3 {
        let mut matrices = Vec::new();
        for group in 0..weight.dim(0) {
            matrices.push(dequantize(
                &weight.try_index_device(group, stream)?,
                &scales.try_index_device(group, stream)?,
                &global.try_index_device(group, stream)?,
                layout,
                stream,
            )?);
        }
        return ops::stack_axis(&matrices, 0, stream);
    }
    let (rows, _, width) = geometry(weight, scales, global, layout)?;
    let globals = global.reshape(&[layout.partitions() as i32], stream)?;
    let mut tiles = Vec::new();
    let mut start = 0;
    while start < rows {
        let part = start / width;
        let end = (start + ROW_TILE).min((part + 1) * width);
        let tile = decode_tile(
            weight,
            scales,
            &globals.try_index_device(part as i32, stream)?,
            start..end,
            stream,
        )?;
        safemlx::transforms::eval([&tile])?;
        tiles.push(tile);
        start = end;
    }
    ops::concatenate_axis(&tiles, 0, stream)
}
