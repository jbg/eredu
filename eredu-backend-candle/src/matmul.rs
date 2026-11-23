//! Matrix products with a fixed FP32 reduction for BF16 operands.

use candle_core::{CpuStorage, DType, Result, Storage, Tensor};
use eredu_nn::arithmetic::{bf16_row_block_ends, Bf16DotGeometry};
use half::bf16;
use rayon::prelude::*;

pub(crate) fn project(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    // ARM FP32 projections retain the reduction order across dense layers.
    if crate::float_cpu::applies(input) && crate::float_cpu::applies(weights) {
        return crate::float_cpu::dense_projection(input, weights);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if crate::metal_f32::applies(input) && crate::metal_f32::applies(weights) {
        return crate::metal_f32::project(input, weights);
    }
    if needs_bf16_dot(input, weights) {
        return bf16_product(input, weights, true, None);
    }
    input.matmul(&weights.t()?)
}

/// Projects narrowed operands using a wide sum and an FP32 result.
pub(crate) fn rounded_operand_projection(
    input: &Tensor,
    weight: &Tensor,
    precision: DType,
) -> Result<Tensor> {
    let (columns, width) = weight.dims2()?;
    let mut shape = input.dims().to_vec();
    let Some(last) = shape.last_mut() else {
        candle_core::bail!("linear requires a feature axis");
    };
    if width == 0 || *last != width {
        candle_core::bail!("linear input feature count differs from weight");
    }
    let inputs = input
        .to_dtype(precision)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let weights = weight
        .to_dtype(precision)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if inputs.device().is_metal() {
        return crate::metal_f32::rounded_operand_projection(&inputs, &weights);
    }
    let (left, left_layout) = inputs.storage_and_layout();
    let (right, right_layout) = weights.storage_and_layout();
    let (Storage::Cpu(CpuStorage::F32(inputs)), Storage::Cpu(CpuStorage::F32(weights))) =
        (&*left, &*right)
    else {
        candle_core::bail!("rounded projection requires CPU storage");
    };
    let inputs = &inputs[left_layout.start_offset()..];
    let weights = &weights[right_layout.start_offset()..];
    let rows = input.elem_count() / width;
    *last = columns;
    let length = rows.checked_mul(columns).ok_or_else(|| {
        candle_core::Error::Msg("rounded projection output geometry overflows".into())
    })?;
    let mut result = vec![0.; length];
    result
        .par_iter_mut()
        .with_min_len(64)
        .enumerate()
        .for_each(|(index, output)| {
            let row = index / columns;
            let column = index % columns;
            let left = &inputs[row * width..][..width];
            let right = &weights[column * width..][..width];
            *output = left
                .iter()
                .zip(right)
                .map(|(&a, &b)| f64::from(a * b))
                .sum::<f64>() as f32;
        });
    Tensor::from_vec(result, shape, input.device())
}

pub(crate) fn attention_product(lhs: &Tensor, rhs: &Tensor, transpose_rhs: bool) -> Result<Tensor> {
    if needs_bf16_dot(lhs, rhs) {
        let columns = rhs.dim(if transpose_rhs {
            candle_core::D::Minus2
        } else {
            candle_core::D::Minus1
        })?;
        return bf16_product(lhs, rhs, transpose_rhs, Some(columns));
    }
    if transpose_rhs {
        lhs.matmul(&rhs.t()?)
    } else {
        lhs.matmul(rhs)
    }
}

fn needs_bf16_dot(lhs: &Tensor, rhs: &Tensor) -> bool {
    (lhs.device().is_cpu() || supports_ordered_metal(lhs))
        && lhs.device().same_device(rhs.device())
        && lhs.dtype() == DType::BF16
        && rhs.dtype() == DType::BF16
}

pub(crate) fn supports_ordered_metal(value: &Tensor) -> bool {
    cfg!(all(feature = "metal", target_os = "macos")) && value.device().is_metal()
}

pub(crate) fn bf16_attention_page_scores(
    queries: &Tensor,
    keys: &Tensor,
    context: usize,
) -> Result<Tensor> {
    bf16_product(queries, keys, true, Some(context))
}

fn bf16_product(
    lhs: &Tensor,
    rhs: &Tensor,
    rhs_rows: bool,
    batched_columns: Option<usize>,
) -> Result<Tensor> {
    let lhs_shape = lhs.dims();
    let rhs_shape = rhs.dims();
    let [lhs_batch @ .., rows, width] = lhs_shape else {
        candle_core::bail!("BF16 matrix product requires two trailing axes");
    };
    let [rhs_batch @ .., rhs_rows_count, rhs_cols_count] = rhs_shape else {
        candle_core::bail!("BF16 matrix product requires two trailing axes");
    };
    let (rhs_width, columns) = if rhs_rows {
        (*rhs_cols_count, *rhs_rows_count)
    } else {
        (*rhs_rows_count, *rhs_cols_count)
    };
    if lhs_batch != rhs_batch || *width != rhs_width {
        candle_core::bail!("invalid BF16 matrix product geometry");
    }
    if *width == 0 || columns == 0 || *rows == 0 {
        return if rhs_rows {
            lhs.matmul(&rhs.t()?)
        } else {
            lhs.matmul(rhs)
        };
    }
    let geometry = Bf16DotGeometry {
        rows: *rows as u64,
        width: *width as u64,
        columns: batched_columns.unwrap_or(columns) as u64,
    };
    let sequential_dot = batched_columns.is_some() && geometry.uses_sequential();
    let value_page = geometry.value_reduction().page(0, *width);
    let row_count = lhs.elem_count() / width;
    let output_len = row_count.checked_mul(columns).ok_or_else(|| {
        candle_core::Error::Msg("BF16 matrix product output size overflows".into())
    })?;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if lhs.device().is_metal() {
        let mut output_shape = lhs_batch.to_vec();
        output_shape.extend([*rows, columns]);
        if output_len == 0 {
            return Tensor::zeros(output_shape, DType::BF16, lhs.device());
        }
        return crate::metal_bf16::product(
            lhs,
            rhs,
            rhs_rows,
            sequential_dot,
            value_page.interleaved,
            output_shape.into(),
        );
    }
    // Widen each operand once and borrow its contiguous CPU storage.
    let left = lhs.to_dtype(DType::F32)?.contiguous()?;
    let right = rhs.to_dtype(DType::F32)?.contiguous()?;
    let (left_storage, left_layout) = left.storage_and_layout();
    let (right_storage, right_layout) = right.storage_and_layout();
    let (Storage::Cpu(CpuStorage::F32(left)), Storage::Cpu(CpuStorage::F32(right))) =
        (&*left_storage, &*right_storage)
    else {
        candle_core::bail!("BF16 matrix product requires CPU storage");
    };
    let left = &left[left_layout.start_offset()..];
    let right = &right[right_layout.start_offset()..];
    let mut output = vec![bf16::ZERO; output_len];
    output
        .par_iter_mut()
        .with_min_len(256)
        .enumerate()
        .for_each(|(index, out)| {
            let row = index / columns;
            let column = index % columns;
            let x = &left[row * width..(row + 1) * width];
            let batch = row / rows;
            let y = &right[batch * width * columns..(batch + 1) * width * columns];
            let sum = match (rhs_rows, sequential_dot) {
                (true, false) => row_dot(x, &y[column * width..(column + 1) * width]),
                (true, true) => {
                    let y = &y[column * width..(column + 1) * width];
                    sequential(x, |k| y[k])
                }
                (false, false) => {
                    column_dot(x, value_page.interleaved, |k| y[k * columns + column])
                }
                (false, true) => sequential(x, |k| y[k * columns + column]),
            };
            *out = bf16::from_f32(sum);
        });
    let mut shape = lhs_batch.to_vec();
    shape.extend([*rows, columns]);
    Tensor::from_vec(output, shape, lhs.device())
}

fn sequential(x: &[f32], y: impl Fn(usize) -> f32) -> f32 {
    x.iter().enumerate().fold(0., |sum, (k, x)| sum + x * y(k))
}

fn row_dot(x: &[f32], y: &[f32]) -> f32 {
    let [block_end, tail_end] = bf16_row_block_ends(x.len());
    let mut lanes = [0.; 32];
    let blocks = x[..block_end]
        .chunks_exact(32)
        .zip(y[..block_end].chunks_exact(32));
    for (x, y) in blocks {
        for (lane, (&x, &y)) in lanes.iter_mut().zip(x.iter().zip(y)) {
            *lane += x * y;
        }
    }
    for step in [16, 8, 4] {
        let (lower, upper) = lanes.split_at_mut(step);
        for (lo, hi) in lower.iter_mut().zip(upper) {
            *lo += *hi;
        }
    }
    let sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let mut tail = [0.; 4];
    for (x, y) in x[block_end..tail_end]
        .chunks_exact(8)
        .zip(y[block_end..tail_end].chunks_exact(8))
    {
        for k in 0..4 {
            tail[k] += x[k] * y[k];
            tail[k] += x[k + 4] * y[k + 4];
        }
    }
    let sum = sum + ((tail[0] + tail[1]) + (tail[2] + tail[3]));
    x[tail_end..]
        .iter()
        .zip(&y[tail_end..])
        .fold(sum, |sum, (x, y)| sum + x * y)
}

fn column_dot(x: &[f32], interleaved: usize, y: impl Fn(usize) -> f32) -> f32 {
    let mut lanes = [0.; 4];
    for (i, block) in x[..interleaved].chunks_exact(4).enumerate() {
        for k in 0..4 {
            lanes[k] += block[k] * y(i * 4 + k);
        }
    }
    for (i, x) in x[interleaved..].iter().enumerate() {
        lanes[0] += x * y(interleaved + i);
    }
    reduce_bf16_value_lanes(lanes)
}

pub(crate) fn reduce_bf16_value_lanes([a, b, c, d]: [f32; 4]) -> f32 {
    ((a + b) + c) + d
}

pub(crate) fn ordered_bf16_affine(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    channel_major: bool,
) -> Result<Tensor> {
    let (columns, width) = weight.dims2()?;
    let mut shape = input.dims().to_vec();
    let Some(last) = shape.last_mut() else {
        candle_core::bail!("projection requires a feature axis");
    };
    if width == 0 || *last != width || bias.is_some_and(|bias| bias.dims() != [columns]) {
        candle_core::bail!("invalid ordered affine geometry");
    }
    let left = input
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let right = weight
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let bias = bias
        .map(|value| value.to_dtype(DType::F32)?.to_vec1::<f32>())
        .transpose()?;
    let count = (left.len() / width)
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("ordered affine output extent overflows".into()))?;
    let mut output = vec![0.0; count];
    output
        .par_iter_mut()
        .with_min_len(64)
        .enumerate()
        .for_each(|(index, value)| {
            let row = index / columns;
            let column = index % columns;
            let input = &left[row * width..][..width];
            let weight = &right[column * width..][..width];
            let sum = if channel_major {
                eredu_nn::arithmetic::bf16_channel_dot(width, |k| input[k] * weight[k])
            } else {
                row_dot(input, weight)
            };
            *value = match &bias {
                Some(bias) => sum + bias[column],
                None => sum,
            };
        });
    *last = columns;
    Tensor::from_vec(output, shape, input.device())
}
