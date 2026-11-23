//! CPU floating-point arithmetic with explicit accumulation and rounding order.

use candle_core::{CpuStorage, DType, Result, Storage, Tensor};
use rayon::prelude::*;

pub(crate) fn applies(input: &Tensor) -> bool {
    cfg!(target_arch = "aarch64") && input.dtype() == DType::F32 && input.device().is_cpu()
}

use eredu_nn::arithmetic::ordered_f32::{attention_value_lanes, dot, sum4, unpacked_dot};

pub(crate) fn silu(input: &Tensor) -> Result<Tensor> {
    map_rows(input, eredu_nn::arithmetic::ordered_f32::silu)
}

pub(crate) fn softmax(input: &Tensor) -> Result<Tensor> {
    map_rows(input, eredu_nn::arithmetic::ordered_f32::softmax)
}

pub(crate) fn binary16_projection(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, width) = input.dims2()?;
    let (outputs, weight_width) = weight.dims2()?;
    if width == 0 || width != weight_width {
        candle_core::bail!("invalid FP16 projection feature width");
    }
    let values = |tensor: &Tensor| {
        tensor
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    let x = values(input)?;
    let y = values(weight)?;
    let length = rows.checked_mul(outputs).ok_or_else(|| {
        candle_core::Error::Msg("FP16 projection output geometry overflows".into())
    })?;
    let mut projected = vec![0.; length];
    projected
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            let row = index / outputs;
            let column = index % outputs;
            let left = &x[row * width..(row + 1) * width];
            let right = &y[column * width..(column + 1) * width];
            let mut lanes = [0.; 8];
            for (left, right) in left.chunks(8).zip(right.chunks(8)) {
                for ((sum, &x), &y) in lanes.iter_mut().zip(left).zip(right) {
                    *sum = eredu_nn::arithmetic::binary16_mul_add(x, y, *sum);
                }
            }
            *output = sum4(std::array::from_fn(|i| lanes[i] + lanes[i + 4]));
        });
    Tensor::from_vec(projected, (rows, outputs), input.device())
}

/// Reduces a short FP32 row in four lanes before consuming the scalar tail.
pub(crate) fn sum_row(row: &[f32]) -> f32 {
    let (groups, tail) = row.as_chunks::<4>();
    let mut lanes = [0.; 4];
    for group in groups {
        for (sum, value) in lanes.iter_mut().zip(group) {
            *sum += value;
        }
    }
    tail.iter().fold(sum4(lanes), |sum, value| sum + value)
}

pub(crate) fn map_rows(input: &Tensor, op: impl Fn(&mut [f32]) + Sync + Send) -> Result<Tensor> {
    let width = input.dims().last().copied().unwrap_or(1);
    let mut data = input.flatten_all()?.to_vec1::<f32>()?;
    if width != 0 {
        data.par_chunks_mut(width).for_each(op);
    }
    Tensor::from_vec(data, input.shape(), input.device())
}

pub(crate) fn layer_norm(input: &Tensor, epsilon: f32) -> Result<Tensor> {
    map_rows(input, |row| {
        eredu_nn::arithmetic::normalize_blocked_f32(row, epsilon)
    })
}

pub(crate) fn rotary(
    input: &Tensor,
    adjacent: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> Result<Tensor> {
    let (_, _, sequence, width) = input.dims4()?;
    let tables =
        eredu_nn::arithmetic::recurrent_rotary_tables(sequence, width, base, scale, offset);
    let mut data = input.flatten_all()?.to_vec1::<f32>()?;
    data.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, data)| {
            for (pair, &(sin, cos)) in tables[row % sequence * (width / 2)..][..width / 2]
                .iter()
                .enumerate()
            {
                let (a, b) = if adjacent {
                    (2 * pair, 2 * pair + 1)
                } else {
                    (pair, pair + width / 2)
                };
                let (x, y) = (data[a], data[b]);
                data[a] = x.mul_add(cos, -(y * sin));
                data[b] = x.mul_add(sin, y * cos);
            }
        });
    Tensor::from_vec(data, input.shape(), input.device())
}

/// Uses the same F32 angle progression as the contiguous CPU rotary kernel.
pub(crate) fn round_robin_rotary_tables(
    positions: &[i32],
    spec: &eredu_nn::multimodal::MultiAxisRotarySpec,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let axes = spec.axes.len();
    let step = spec.base.powf(-2. / width as f32);
    if positions.is_empty() {
        candle_core::bail!("multi-axis rotary requires positions");
    }
    let mut cosine = Vec::with_capacity(positions.len() / axes * width);
    let mut sine = Vec::with_capacity(cosine.capacity());
    let mut angles = vec![0.0; axes];
    for row in positions.chunks_exact(axes) {
        for ((angle, &position), axis) in angles.iter_mut().zip(row).zip(&spec.axes) {
            *angle = position
                .saturating_add(axis.position_offset)
                .max(spec.minimum_position) as f32;
        }
        let begin = cosine.len();
        for frequency in 0..width / 2 {
            let angle = spec
                .round_robin_axis(frequency)
                .map_or(0.0, |axis| angles[axis]);
            let (sin, cos) = angle.sin_cos();
            cosine.push(cos);
            sine.push(sin);
            for angle in &mut angles {
                *angle *= step;
            }
        }
        cosine.extend_from_within(begin..);
        sine.extend_from_within(begin..);
    }
    Ok((cosine, sine))
}

pub(crate) fn split_rotary_tables(
    positions: &[i32],
    spec: &eredu_nn::multimodal::MultiAxisRotarySpec,
    width: usize,
) -> (Vec<f32>, Vec<f32>) {
    let axes = spec.axes.len();
    let mut cosine = Vec::with_capacity(positions.len() / axes * width);
    let mut sine = Vec::with_capacity(cosine.capacity());
    for row in positions.chunks_exact(axes) {
        let begin = cosine.len();
        for (axis, &position) in spec.axes.iter().zip(row) {
            let mut angle = position
                .saturating_add(axis.position_offset)
                .max(spec.minimum_position) as f32;
            let step = spec.base.powf(-2. / axis.dimensions as f32);
            for _ in 0..axis.dimensions / 2 {
                let (sin, cos) = angle.sin_cos();
                cosine.push(cos);
                sine.push(sin);
                angle *= step;
            }
        }
        cosine.extend_from_within(begin..);
        sine.extend_from_within(begin..);
    }
    (cosine, sine)
}

pub(crate) fn resize_bilinear_f32(
    input: &Tensor,
    vertical: &[(usize, Vec<f32>)],
    horizontal: &[(usize, Vec<f32>)],
    channels: usize,
) -> Result<Tensor> {
    let (_, source_width, _) = input.dims3()?;
    let input_values = input.flatten_all()?.to_vec1::<f32>()?;
    let elements = vertical
        .len()
        .checked_mul(horizontal.len())
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| candle_core::Error::Msg("resampled tensor size overflows".into()))?;
    if channels == 0 {
        candle_core::bail!("resampling requires at least one channel");
    }
    let mut result = vec![0.; elements];
    result
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(index, pixel)| {
            let (top, vertical) = &vertical[index / horizontal.len()];
            let (left, horizontal) = &horizontal[index % horizontal.len()];
            let mut total = 0.0f32;
            for (dy, wy) in vertical.iter().copied().enumerate() {
                for (dx, wx) in horizontal.iter().copied().enumerate() {
                    let weight = wy * wx;
                    if weight > 0.0 {
                        let source = ((top + dy) * source_width + left + dx) * channels;
                        for (out, &value) in pixel.iter_mut().zip(&input_values[source..]) {
                            *out = value.mul_add(weight, *out);
                        }
                        total += weight;
                    }
                }
            }
            if total > 0.0 {
                for out in pixel {
                    *out /= total;
                }
            }
        });
    Tensor::from_vec(
        result,
        (vertical.len(), horizontal.len(), channels),
        input.device(),
    )
}

pub(crate) fn attention_product(lhs: &Tensor, rhs: &Tensor, rhs_rows: bool) -> Result<Tensor> {
    let (batch, heads, rows, width) = lhs.dims4()?;
    let lhs = lhs.contiguous()?;
    let rhs = if rhs_rows {
        rhs.contiguous()?
    } else {
        rhs.t()?.contiguous()?
    };
    let (rhs_batch, rhs_heads, columns, rhs_width) = rhs.dims4()?;
    if (rhs_batch, rhs_heads, rhs_width) != (batch, heads, width) {
        candle_core::bail!("attention product has incompatible batch or feature dimensions");
    }
    let (left_storage, left_layout) = lhs.storage_and_layout();
    let (right_storage, right_layout) = rhs.storage_and_layout();
    let (Storage::Cpu(CpuStorage::F32(left)), Storage::Cpu(CpuStorage::F32(right))) =
        (&*left_storage, &*right_storage)
    else {
        candle_core::bail!("attention product requires FP32 CPU storage");
    };
    let left = &left[left_layout.start_offset()..];
    let right = &right[right_layout.start_offset()..];
    let count = [batch, heads, rows, columns]
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| candle_core::Error::Msg("attention product size overflows".into()))?;
    let mut result = vec![0.; count];
    // Query-key scores use a 16-lane dot. The batched value product uses
    // four lanes, while a small query batch uses a 16-lane dot. Missing
    // value positions contribute zero within the last reduction group.
    let dot = match (rhs_rows, attention_value_lanes(rows as u64) == 4) {
        (true, _) => |a: &[f32], b: &[f32]| dot::<16>(a, b, false),
        (false, true) => |a: &[f32], b: &[f32]| dot::<4>(a, b, true),
        (false, false) => |a: &[f32], b: &[f32]| dot::<16>(a, b, true),
    };
    result
        .par_iter_mut()
        .with_min_len(256)
        .enumerate()
        .for_each(|(i, out)| {
            let row = i / columns;
            let right_row = row / rows * columns + i % columns;
            let a = &left[row * width..][..width];
            let b = &right[right_row * width..][..width];
            *out = dot(a, b);
        });
    Tensor::from_vec(result, (batch, heads, rows, columns), lhs.device())
}

pub(crate) fn dense_projection(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, width) = input.dims2()?;
    let (columns, weight_width) = weight.dims2()?;
    if width != weight_width {
        candle_core::bail!("projection input and weight widths differ");
    }
    let input_rows = input.flatten_all()?.to_vec1::<f32>()?;
    let weight_rows = weight.flatten_all()?.to_vec1::<f32>()?;
    let length = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("projection output geometry overflows".into()))?;
    let mut output = vec![0.; length];
    output
        .par_iter_mut()
        .with_min_len(64)
        .enumerate()
        .for_each(|(index, result)| {
            let row = index / columns;
            let column = index % columns;
            let left = &input_rows[row * width..][..width];
            let right = &weight_rows[column * width..][..width];
            *result = if rows >= 4 {
                dot::<4>(left, right, false)
            } else {
                unpacked_dot(left, right)
            };
        });
    Tensor::from_vec(output, (rows, columns), input.device())
}

pub(crate) fn explicit_rotary(
    input: &Tensor,
    cosine: &Tensor,
    sine: &Tensor,
    dimensions: usize,
    adjacent: bool,
    arithmetic: eredu_nn::RotaryArithmetic,
) -> Result<Tensor> {
    let (batch, heads, tokens, width) = input.dims4()?;
    if dimensions == 0 || dimensions % 2 != 0 || dimensions > width {
        candle_core::bail!("invalid explicit rotary width");
    }
    let broadcast = |tensor: &Tensor| {
        crate::tensor::broadcast_rotary_table(tensor, batch, tokens, dimensions)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    let round: fn(f32) -> f32 = match input.dtype() {
        DType::F16 => |value| half::f16::from_f32(value).to_f32(),
        DType::BF16 => |value| half::bf16::from_f32(value).to_f32(),
        _ => |value| value,
    };
    let product = |x: f32, rx: f32, cosine: f32, sine: f32| match arithmetic {
        eredu_nn::RotaryArithmetic::InputProducts => {
            round(x * round(cosine)) + round(rx * round(sine))
        }
        eredu_nn::RotaryArithmetic::InputTableFloat32Products => {
            x * round(cosine) + rx * round(sine)
        }
        eredu_nn::RotaryArithmetic::Float32Products => x * cosine + rx * sine,
        _ => x.mul_add(cosine, rx * sine),
    };
    let cosine = broadcast(cosine)?;
    let sine = broadcast(sine)?;
    let mut values = input
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    values
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, data)| {
            let begin = (row / (heads * tokens) * tokens + row % tokens) * dimensions;
            for pair in 0..dimensions / 2 {
                let (a, b) = if adjacent {
                    (pair * 2, pair * 2 + 1)
                } else {
                    (pair, pair + dimensions / 2)
                };
                let (x, y) = (data[a], data[b]);
                data[a] = product(x, -y, cosine[begin + a], sine[begin + a]);
                data[b] = if arithmetic != eredu_nn::RotaryArithmetic::Native {
                    product(y, x, cosine[begin + b], sine[begin + b])
                } else {
                    x.mul_add(sine[begin + b], y * cosine[begin + b])
                };
            }
        });
    Tensor::from_vec(values, input.shape(), input.device())?.to_dtype(input.dtype())
}
