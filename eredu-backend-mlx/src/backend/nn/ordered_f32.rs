//! MLX arrays and native kernels for ordered binary32 arithmetic.

use eredu_nn::arithmetic::ordered_f32;
use safemlx::{
    error::Exception,
    fast::{CustomKernelConfig, MetalKernel},
    Array, DeviceType, Dtype, Stream,
};
use std::cell::RefCell;

thread_local! {
    static PRODUCT: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static ROWS: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

fn metal(stream: &Stream) -> Result<bool, Exception> {
    Ok(super::native_quantization::uses_metal_device(
        stream.get_device()?.get_type()?,
    ))
}

pub(super) fn supported(stream: &Stream) -> Result<bool, Exception> {
    let device = stream.get_device()?.get_type()?;
    Ok(device == DeviceType::Cpu || super::native_quantization::uses_metal_device(device))
}

// Both operands have contiguous rows; every K/V head serves a contiguous group
// of query heads. No explicit head expansion is required.
pub(super) fn attention_product(
    left: &Array,
    right: &Array,
    values: bool,
    query_rows: u64,
    stream: &Stream,
) -> Result<Array, Exception> {
    let [batch, heads, rows, width] = left.shape() else {
        return Err(Exception::custom(
            "ordered product expects rank-four query rows",
        ));
    };
    let [kv_batch, kv_heads, columns, kv_width] = right.shape() else {
        return Err(Exception::custom(
            "ordered product expects rank-four state rows",
        ));
    };
    let (batch, heads, rows, width, kv_heads, columns) =
        (*batch, *heads, *rows, *width, *kv_heads, *columns);
    if *kv_batch != batch || *kv_width != width || kv_heads <= 0 || heads % kv_heads != 0 {
        return Err(Exception::custom("ordered attention dimensions differ"));
    }
    let shape = [batch, heads, rows, columns];
    let row_count = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(rows))
        .ok_or_else(|| Exception::custom("ordered attention row count overflows"))?;
    if row_count == 0 || columns == 0 || width == 0 {
        return safemlx::ops::zeros_dtype(&shape, Dtype::Float32, stream);
    }
    let lanes = if values {
        ordered_f32::attention_value_lanes(query_rows) as i32
    } else {
        16
    };
    if metal(stream)? {
        let dimensions = Array::from_slice(
            &[
                heads,
                rows,
                width,
                kv_heads,
                columns,
                lanes,
                i32::from(values),
            ],
            &[7],
        );
        let grid_width = columns
            .checked_mul(lanes)
            .ok_or_else(|| Exception::custom("ordered attention grid width overflows"))?;
        let config = CustomKernelConfig::new()
            .with_grid([grid_width, row_count, 1])
            .with_thread_group([128, 1, 1])
            .with_output_arg(shape, Dtype::Float32);
        return PRODUCT.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(product_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("ordered product kernel initialized")
                .apply_one_device([left, right, &dimensions], &config, stream)
        });
    }
    let left = left.contiguous(false, stream)?.into_evaluated()?;
    let right = right.contiguous(false, stream)?.into_evaluated()?;
    let left = left
        .try_as_slice::<f32>()
        .map_err(|e| Exception::custom(e.to_string()))?;
    let right = right
        .try_as_slice::<f32>()
        .map_err(|e| Exception::custom(e.to_string()))?;
    let (heads, rows, width, kv_heads, columns) = (
        heads as usize,
        rows as usize,
        width as usize,
        kv_heads as usize,
        columns as usize,
    );
    let output_len = (row_count as usize)
        .checked_mul(columns)
        .ok_or_else(|| Exception::custom("ordered attention output size overflows"))?;
    let mut output = vec![0.; output_len];
    use rayon::prelude::*;
    output
        .par_chunks_mut(columns)
        .with_min_len(64)
        .enumerate()
        .for_each(|(row, out)| {
            let batch = row / (heads * rows);
            let head = (row / rows % heads) / (heads / kv_heads);
            let base = (batch * kv_heads + head) * columns * width;
            let query = &left[row * width..][..width];
            for (out, key) in out.iter_mut().zip(right[base..].chunks_exact(width)) {
                *out = if lanes == 4 {
                    ordered_f32::dot::<4>(query, key, values)
                } else {
                    ordered_f32::dot::<16>(query, key, values)
                };
            }
        });
    Array::from_slice(&output, &shape).copy(stream)
}

pub(super) fn softmax(input: &Array, stream: &Stream) -> Result<Array, Exception> {
    map_rows(input, false, stream)
}

pub(super) fn silu(input: &Array, stream: &Stream) -> Result<Array, Exception> {
    map_rows(input, true, stream)
}

fn map_rows(input: &Array, silu: bool, stream: &Stream) -> Result<Array, Exception> {
    if input.size() == 0 || input.ndim() == 0 {
        return Ok(input.clone());
    }
    let width = input.dim(-1);
    if metal(stream)? {
        let work_items = if silu {
            input.size()
        } else {
            input.size() / width as usize
        };
        let work_items = i32::try_from(work_items)
            .map_err(|_| Exception::custom("ordered work item count overflows"))?;
        let dimensions = Array::from_slice(&[width], &[1]);
        let config = CustomKernelConfig::new()
            .with_template_arg_bool("SILU", silu)
            .with_grid([work_items, 1, 1])
            .with_thread_group([64, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        return ROWS.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(row_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("ordered row kernel initialized")
                .apply_one_device([input, &dimensions], &config, stream)
        });
    }
    let mut data = input
        .evaluated()?
        .try_to_vec::<f32>()
        .map_err(|e| Exception::custom(e.to_string()))?;
    let apply = if silu {
        ordered_f32::silu
    } else {
        ordered_f32::softmax
    };
    for row in data.chunks_exact_mut(width as usize) {
        apply(row);
    }
    Array::from_slice(&data, input.shape()).copy(stream)
}

fn product_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "ordered_f32_attention_product",
        ["left", "right", "shape"],
        ["out"],
        r#"
        uint index = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint heads = shape[0], rows = shape[1], width = shape[2], kv_heads = shape[3], columns = shape[4];
        uint batch = row / (heads * rows);
        uint head = (row / rows % heads) / (heads / kv_heads);
        uint nlanes = shape[5];
        uint column = index / nlanes;
        uint lane = index % nlanes;
        size_t a = size_t(row) * width;
        size_t b = (size_t(batch * kv_heads + head) * columns + column) * width;
        uint end = shape[6] ? width : (width / nlanes) * nlanes;
        float sum = 0.0f;
        for (uint i = lane; i < end; i += nlanes) sum = fma(left[a+i], right[b+i], sum);
        if (nlanes == 16) {
            sum += simd_shuffle_down(sum, 8);
            sum += simd_shuffle_down(sum, 4);
        }
        float pair = simd_shuffle_down(sum, 1);
        float third = simd_shuffle_down(sum, 2);
        float fourth = simd_shuffle_down(sum, 3);
        if (lane == 0) {
            float result = (sum + pair) + (third + fourth);
            for (uint i = end; i < width; ++i) result = fma(left[a+i], right[b+i], result);
            out[size_t(row) * columns + column] = result;
        }
        "#,
        "#pragma clang fp contract(off)\n",
        true,
        false,
    )
}

fn row_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "ordered_f32_attention_rows",
        ["input", "shape"],
        ["out"],
        r#"
        uint width = shape[0];
        uint index = thread_position_in_grid.x;
        if (SILU) {
            uint column = index % width;
            float x = input[index];
            float e = column < (width/4)*4 ? vector_exp_f32(-x) : stable_exp_f32(-x);
            out[index] = metal::precise::divide(x, 1.0f+e);
        } else {
            size_t base = size_t(index) * width;
            float peak = -INFINITY;
            for (uint i = 0; i < width; ++i) peak = max(peak, input[base+i]);
            float high = 0.0f;
            float low = 0.0f;
            for (uint i = 0; i < width; i += 4) {
                float4 values = 0.0f;
                for (uint j = 0; j < 4 && i+j < width; ++j) {
                    values[j] = vector_exp_f32(input[base+i+j] - peak);
                    out[base+i+j] = values[j];
                }
                float mass = (values.x + values.y) + (values.z + values.w);
                float sum = high + mass;
                float split = sum - high;
                float error = (high - (sum - split)) + (mass - split) + low;
                high = sum + error;
                low = error - (high - sum);
            }
            float quotient = metal::precise::divide(1.0f, high);
            float error = fma(-quotient, high, 1.0f) - quotient * low;
            float inverse = quotient + metal::precise::divide(error, high);
            for (uint i = 0; i < width; ++i) out[base+i] *= inverse;
        }
        "#,
        &[
            super::arithmetic::metal_arithmetic_header(),
            include_str!("exp_vector_f32.metal"),
        ]
        .concat(),
        true,
        false,
    )
}
