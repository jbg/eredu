//! Native execution of the ordered FP32 arithmetic used around packed operators.

use candle_core::{DType, Result, Tensor};
use objc2_metal::MTLSize;

use crate::metal::{size, Launch};

pub(crate) fn applies(value: &Tensor) -> bool {
    value.dtype() == DType::F32 && value.device().is_metal()
}

pub(crate) fn recurrent_rotary(
    input: &Tensor,
    adjacent: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> Result<Tensor> {
    let (_, _, sequence, width) = input.dims4()?;
    let tables =
        eredu_nn::arithmetic::recurrent_rotary_tables(sequence, width, base, scale, offset)
            .into_iter()
            .flat_map(|(sin, cos)| [sin, cos])
            .collect::<Vec<_>>();
    let tables = Tensor::from_vec(tables, (sequence, width), input.device())?;
    Launch {
        shape: input.shape(),
        dtype: DType::F32,
        grid: MTLSize {
            width: width / 2,
            height: input.elem_count() / width,
            depth: 1,
        },
        kernel: "recurrent_rotary",
        group: 64,
    }
    .binary(
        input,
        &tables,
        [u32::from(adjacent), u32::try_from(sequence)?],
    )
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
    let table = |value| {
        let value = crate::tensor::broadcast_rotary_table(value, batch, tokens, dimensions)?;
        let value = if arithmetic == eredu_nn::RotaryArithmetic::InputTableFloat32Products {
            value.to_dtype(input.dtype())?
        } else {
            value
        };
        value.to_dtype(DType::F32)
    };
    let table = Tensor::stack(&[table(cosine)?, table(sine)?], 3)?;
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    let dtype = input.dtype();
    let rounding = if arithmetic == eredu_nn::RotaryArithmetic::InputProducts {
        match dtype {
            DType::F16 => 1,
            DType::BF16 => 2,
            _ => 3,
        }
    } else if arithmetic == eredu_nn::RotaryArithmetic::Native {
        0
    } else {
        3
    };
    Launch {
        kernel: "explicit_rotary",
        shape: input.shape(),
        dtype: DType::F32,
        grid: MTLSize {
            width,
            height: input.elem_count() / width,
            depth: 1,
        },
        group: 64,
    }
    .binary(
        &input.to_dtype(DType::F32)?,
        &table,
        [
            u32::try_from(tokens)?,
            u32::try_from(heads)?,
            u32::try_from(dimensions)?,
            u32::from(adjacent),
            rounding,
        ],
    )?
    .to_dtype(dtype)
}

fn unary(input: &Tensor, name: &'static str, per_row: bool) -> Result<Tensor> {
    let width = input.dims().last().copied().unwrap_or(1);
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    Launch {
        kernel: name,
        shape: input.shape(),
        dtype: DType::F32,
        grid: size(if per_row {
            input.elem_count() / width
        } else {
            input.elem_count()
        }),
        group: 64,
    }
    .unary(input, [u32::try_from(width)?])
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn sigmoid(input: &Tensor) -> Result<Tensor> {
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    Launch {
        shape: input.shape(),
        dtype: DType::F32,
        grid: size(input.elem_count()),
        kernel: "f32_sigmoid",
        group: 64,
    }
    .unary(input, [])
}

pub(crate) fn silu(input: &Tensor) -> Result<Tensor> {
    unary(input, "f32_silu", false)
}

pub(crate) fn attention_softmax(input: &Tensor) -> Result<Tensor> {
    unary(input, "f32_attention_softmax", true)
}

pub(crate) fn rms_inverse(input: &Tensor, epsilon: f32) -> Result<Tensor> {
    let mut shape = input.dims().to_vec();
    let width = shape
        .last_mut()
        .ok_or_else(|| candle_core::Error::Msg("RMS requires a feature axis".into()))?;
    if *width == 0 {
        candle_core::bail!("RMS requires a nonempty feature axis");
    }
    let rows = input.elem_count() / *width;
    let features = u32::try_from(*width)?;
    *width = 1;
    if rows == 0 {
        return Tensor::zeros(shape, DType::F32, input.device());
    }
    Launch {
        kernel: "f32_rms_inverse",
        shape: &shape.into(),
        dtype: DType::F32,
        grid: size(rows),
        group: 64,
    }
    .unary(input, [features, epsilon.to_bits()])
}

pub(crate) fn attention_product(lhs: &Tensor, rhs: &Tensor, transpose_rhs: bool) -> Result<Tensor> {
    product(lhs, rhs, transpose_rhs, !transpose_rhs)
}

pub(crate) fn project(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    product(input, weight, true, input.dim(0)? >= 4)
}

fn product(lhs: &Tensor, rhs: &Tensor, transpose_rhs: bool, value_product: bool) -> Result<Tensor> {
    let [batch @ .., rows, width] = lhs.dims() else {
        candle_core::bail!("ordered Metal product requires matrix inputs");
    };
    let [right_batch @ .., a, b] = rhs.dims() else {
        candle_core::bail!("ordered Metal product requires matrix weights");
    };
    let (features, columns) = if transpose_rhs { (*b, *a) } else { (*a, *b) };
    if batch != right_batch || *width != features {
        candle_core::bail!("ordered Metal product dimensions differ");
    }
    let mut shape = batch.to_vec();
    shape.extend([*rows, columns]);
    if *width == 0 || *rows == 0 || columns == 0 || lhs.elem_count() == 0 {
        return Tensor::zeros(shape, DType::F32, lhs.device());
    }
    Launch {
        kernel: "f32_dot",
        shape: &shape.into(),
        dtype: DType::F32,
        grid: MTLSize {
            width: columns,
            height: lhs.elem_count() / *width,
            depth: 1,
        },
        group: 64,
    }
    .binary(
        lhs,
        rhs,
        [
            u32::try_from(*rows)?,
            u32::try_from(*width)?,
            u32::try_from(columns)?,
            u32::from(!transpose_rhs),
            if value_product && *rows >= 4 { 4 } else { 16 },
            u32::from(!transpose_rhs),
        ],
    )
}

pub(crate) fn gelu_approximate_f16(input: &Tensor) -> Result<Tensor> {
    use candle_core::metal_backend::DeviceId;
    use std::cell::RefCell;

    thread_local! {
        static TABLE: RefCell<Option<(DeviceId, Tensor)>> = const { RefCell::new(None) };
    }
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    let id = input.device().as_metal_device()?.id();
    let table = TABLE.with(|table| -> Result<Tensor> {
        let mut table = table.borrow_mut();
        if table.as_ref().is_none_or(|entry| entry.0 != id) {
            let data = eredu_nn::activations::gelu_approximate_f16_table();
            *table = Some((id, Tensor::from_slice(data, data.len(), input.device())?));
        }
        Ok(table
            .as_ref()
            .expect("initialized activation table")
            .1
            .clone())
    })?;
    Launch {
        kernel: "gelu_half_table",
        shape: input.shape(),
        dtype: DType::F32,
        grid: size(input.elem_count()),
        group: 64,
    }
    .binary(input, &table, [])
}

pub(crate) fn binary16_projection(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, width) = input.dims2()?;
    let (columns, weight_width) = weight.dims2()?;
    if width == 0 || width != weight_width {
        candle_core::bail!("invalid FP16 projection feature width");
    }
    if rows == 0 || columns == 0 {
        return Tensor::zeros((rows, columns), DType::F32, input.device());
    }
    Launch {
        kernel: "binary16_projection",
        shape: &(rows, columns).into(),
        dtype: DType::F32,
        grid: MTLSize {
            width: columns,
            height: rows,
            depth: 1,
        },
        group: 64,
    }
    .binary(
        &input.to_dtype(DType::F16)?,
        &weight.to_dtype(DType::F16)?,
        [u32::try_from(width)?, u32::try_from(columns)?],
    )
}

pub(crate) fn blocked_layer_norm(input: &Tensor, epsilon: f32) -> Result<Tensor> {
    let width = input.dim(candle_core::D::Minus1)?;
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    Launch {
        kernel: "blocked_layer_norm",
        shape: input.shape(),
        dtype: DType::F32,
        grid: size(input.elem_count() / width),
        group: 64,
    }
    .unary(input, [u32::try_from(width)?, epsilon.to_bits()])
}

pub(crate) fn softplus(input: &Tensor, beta: f32) -> Result<Tensor> {
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    Launch {
        kernel: "f32_softplus",
        shape: input.shape(),
        dtype: DType::F32,
        grid: size(input.elem_count()),
        group: 64,
    }
    .unary(input, [beta.to_bits()])
}

pub(crate) fn l2_normalize(input: &Tensor, epsilon: f32, floor: bool) -> Result<Tensor> {
    let width = input.dim(candle_core::D::Minus1)?;
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    Launch {
        kernel: "ordered_l2_norm",
        shape: input.shape(),
        dtype: DType::F32,
        grid: size(input.elem_count() / width),
        group: 64,
    }
    .unary(
        input,
        [u32::try_from(width)?, epsilon.to_bits(), u32::from(floor)],
    )
}

pub(crate) fn gated_delta_scan(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    initial: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (batch, sequence, heads, keys) = query.dims4()?;
    let values = value.dim(candle_core::D::Minus1)?;
    let state_shape = (batch, heads, keys, values);
    let shape = candle_core::Shape::from(state_shape);
    let state_len = shape.elem_count();
    let output_len = value.elem_count();
    let total = state_len
        .checked_add(output_len)
        .ok_or_else(|| candle_core::Error::Msg("Metal gated-delta output size overflows".into()))?;
    let initial = match initial {
        Some(state) => state.clone(),
        None => Tensor::zeros(state_shape, DType::F32, query.device())?,
    };
    let log_decay = if log_decay.rank() == 3 {
        log_decay.unsqueeze(3)?
    } else {
        log_decay.clone()
    };
    let wide = [query, key, value, &log_decay, &beta.unsqueeze(3)?]
        .into_iter()
        .map(|x| x.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let input = Tensor::cat(&wide, 3)?;
    let result = Launch {
        kernel: "gated_delta_scan",
        shape: &total.into(),
        dtype: DType::F32,
        grid: MTLSize {
            width: values,
            height: batch * heads,
            depth: 1,
        },
        group: 64,
    }
    .binary(
        &input,
        &initial,
        [
            u32::try_from(sequence)?,
            u32::try_from(heads)?,
            u32::try_from(keys)?,
            u32::try_from(values)?,
            u32::try_from(log_decay.dim(3)?)?,
            u32::try_from(state_len)?,
        ],
    )?;
    Ok((
        result.narrow(0, 0, state_len)?.reshape(state_shape)?,
        result
            .narrow(0, state_len, output_len)?
            .reshape(value.shape())?
            .to_dtype(query.dtype())?,
    ))
}

pub(crate) fn equal_i32(input: &Tensor, value: i32) -> Result<Tensor> {
    i32_strided(input, value, "equal_i32", DType::U8)
}

pub(crate) fn maximum_i32(input: &Tensor, value: i32) -> Result<Tensor> {
    i32_strided(input, value, "maximum_i32", DType::I32)
}

pub(crate) fn widen_i32(input: &Tensor) -> Result<Tensor> {
    i32_strided(input, 0, "widen_i32", DType::I64)
}

fn i32_strided(input: &Tensor, value: i32, kernel: &'static str, dtype: DType) -> Result<Tensor> {
    if input.elem_count() == 0 {
        return match dtype {
            DType::I32 => Tensor::from_vec(Vec::<i32>::new(), input.shape(), input.device()),
            _ => Tensor::zeros(input.shape(), dtype, input.device()),
        };
    }
    // Pass dimensions and strides together so integer views never require
    // Candle's missing strided I32 copy.
    let rank = input.rank();
    let mut args = [0; 64];
    if rank > (args.len() - 2) / 2 {
        candle_core::bail!("Metal integer view rank exceeds the kernel argument capacity");
    }
    args[0] = value as u32;
    args[1] = u32::try_from(rank)?;
    for (axis, (&dim, &stride)) in input.dims().iter().zip(input.stride()).enumerate() {
        args[2 + 2 * axis] = u32::try_from(dim)?;
        args[3 + 2 * axis] = u32::try_from(stride)?;
    }
    Launch {
        kernel,
        shape: input.shape(),
        dtype,
        grid: size(input.elem_count()),
        group: 64,
    }
    .unary_strided(input, args)
}

pub(crate) fn narrow_i64(input: &Tensor) -> Result<Tensor> {
    if input.elem_count() == 0 {
        return Tensor::from_vec(Vec::<i32>::new(), input.shape(), input.device());
    }
    Launch {
        kernel: "narrow_i64",
        shape: input.shape(),
        dtype: DType::I32,
        grid: size(input.elem_count()),
        group: 64,
    }
    .unary(input, [])
}

pub(crate) fn rounded_operand_projection(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (columns, width) = weight.dims2()?;
    let rows = input.elem_count() / width;
    let mut shape = input.dims().to_vec();
    *shape
        .last_mut()
        .expect("projection input has a feature axis") = columns;
    if rows == 0 || columns == 0 {
        return Tensor::zeros(shape, DType::F32, input.device());
    }
    Launch {
        kernel: "rounded_operand_projection",
        shape: &shape.into(),
        dtype: DType::F32,
        grid: MTLSize {
            width: columns,
            height: rows,
            depth: 1,
        },
        group: 64,
    }
    .binary(input, weight, [u32::try_from(width)?])
}

pub(crate) fn depthwise_convolution(
    input: &Tensor,
    weight: &Tensor,
    dilation: usize,
    split_products: bool,
) -> Result<Tensor> {
    let (batch, sequence, channels) = input.dims3()?;
    let kernel = weight.dim(1)?;
    let length = sequence - dilation * (kernel - 1);
    let shape = (batch, length, channels).into();
    if batch == 0 {
        return Tensor::zeros(shape, DType::F32, input.device());
    }
    Launch {
        kernel: "f32_depthwise_convolution",
        shape: &shape,
        dtype: DType::F32,
        grid: MTLSize {
            width: channels,
            height: batch * length,
            depth: 1,
        },
        group: 64,
    }
    .binary(
        input,
        weight,
        [
            u32::try_from(sequence)?,
            u32::try_from(length)?,
            u32::try_from(kernel)?,
            u32::try_from(dilation)?,
            u32::try_from(if split_products { kernel / 4 * 4 } else { 0 })?,
        ],
    )
}

pub(crate) fn top_indices(scores: &Tensor, count: usize) -> Result<Tensor> {
    let mut shape = scores.dims().to_vec();
    let width = shape.last_mut().expect("top indices have a feature axis");
    let rows = scores.elem_count() / *width;
    let geometry = [u32::try_from(*width)?, u32::try_from(count)?];
    *width = count;
    if rows == 0 {
        return Tensor::zeros(shape, DType::U32, scores.device());
    }
    Launch {
        kernel: "f32_top_indices",
        shape: &shape.into(),
        dtype: DType::U32,
        grid: size(rows),
        group: 32,
    }
    .unary(scores, geometry)
}
