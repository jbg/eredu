//! Packed Metal projections and lookups over native checkpoint byte buffers.

use candle_core::{DType, Result, Tensor};
use eredu_checkpoint::GgmlType;
use objc2_metal::MTLSize;

use crate::metal::{size, Launch};

pub(crate) fn project(
    input: &Tensor,
    weight: &Tensor,
    format: GgmlType,
    columns: usize,
    outputs: usize,
    matrix_tiles: bool,
) -> Result<Tensor> {
    let (format_id, block_width, _) = format_geometry(format)?;
    let activation_bytes = if block_width == 32 { 36 } else { 292 };
    let mut shape = input.dims().to_vec();
    let width = shape.last_mut().ok_or_else(|| {
        candle_core::Error::Msg("packed projection requires a feature axis".into())
    })?;
    if *width != columns || columns == 0 || !columns.is_multiple_of(block_width) {
        candle_core::bail!("packed projection width does not match its block layout");
    }
    let rows = input.elem_count() / columns;
    *width = outputs;
    if rows == 0 {
        return Tensor::zeros(shape, DType::F32, input.device());
    }
    let blocks = columns / block_width;
    let blocks_total = input.elem_count() / block_width;
    let activation_shape = (rows, blocks, activation_bytes).into();
    let quantized = Launch {
        kernel: if block_width == 32 {
            "quantize_activation32"
        } else {
            "quantize_activation256"
        },
        shape: &activation_shape,
        dtype: DType::U8,
        grid: size(blocks_total),
        group: 64,
    }
    .unary(input, [])?;
    let (_, row_bytes) = weight.dims2()?;
    Launch {
        shape: &shape.into(),
        dtype: DType::F32,
        grid: MTLSize {
            width: outputs,
            height: rows,
            depth: 1,
        },
        kernel: if block_width == 32 {
            "packed_matmul32"
        } else {
            "packed_matmul256"
        },
        group: 64,
    }
    .binary(
        &quantized,
        weight,
        [
            u32::try_from(blocks)?,
            u32::try_from(row_bytes)?,
            format_id,
            u32::try_from(if matrix_tiles { rows / 4 * 4 } else { 0 })?,
        ],
    )
}

pub(crate) fn embedding(
    weight: &Tensor,
    ids: &[u32],
    format: GgmlType,
    shape: Vec<usize>,
    hidden: usize,
    row_bytes: usize,
) -> Result<Tensor> {
    let (format_id, block_width, block_bytes) = format_geometry(format)?;
    if ids.is_empty() {
        return Tensor::zeros(shape, DType::F32, weight.device());
    }
    let ids = Tensor::from_slice(ids, ids.len(), weight.device())?;
    Launch {
        kernel: "packed_embedding",
        shape: &shape.into(),
        dtype: DType::F32,
        grid: MTLSize {
            width: hidden,
            height: ids.elem_count(),
            depth: 1,
        },
        group: 64,
    }
    .binary(
        weight,
        &ids,
        [
            u32::try_from(row_bytes)?,
            format_id,
            u32::try_from(block_width)?,
            u32::try_from(block_bytes)?,
        ],
    )
}

fn format_geometry(format: GgmlType) -> Result<(u32, usize, usize)> {
    let id = match format {
        GgmlType::Q5_0 => 0,
        GgmlType::Q8_0 => 1,
        GgmlType::Q4K => 2,
        GgmlType::Q5K => 3,
        GgmlType::Q6K => 4,
        _ => candle_core::bail!("unsupported packed Metal format"),
    };
    let (width, bytes) = format.block_and_bytes().map_err(candle_core::Error::wrap)?;
    Ok((id, usize::try_from(width)?, usize::try_from(bytes)?))
}
