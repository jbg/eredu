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
    static SOFTMAX: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static FUSED_ATTENTION: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static DECODE_SCORES: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static DECODE_VALUES: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

const SOFTMAX_SIMD_WIDTH: i32 = 32;
const SOFTMAX_MIN_WIDTH: i32 = 128;
const SOFTMAX_MAX_WIDTH: i32 = 8192;
const FUSED_ATTENTION_THREADS: i32 = 256;

fn metal(stream: &Stream) -> Result<bool, Exception> {
    Ok(super::native_quantization::uses_metal_device(
        stream.get_device()?.get_type()?,
    ))
}

pub(super) fn supported(stream: &Stream) -> Result<bool, Exception> {
    let device = stream.get_device()?.get_type()?;
    Ok(device == DeviceType::Cpu || super::native_quantization::uses_metal_device(device))
}

/// Full-key attention with the ordered score, softmax and value reductions.
pub(super) fn fused_attention(
    query: &Array,
    keys: &Array,
    transposed_values: &Array,
    mask: Option<&Array>,
    scale: f32,
    query_rows: u64,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    let [batch, heads, rows, width] = *query.shape() else {
        return Ok(None);
    };
    let positions = keys.dim(2);
    let kv_heads = keys.dim(1);
    let output_width = transposed_values.dim(2);
    let decode = query_rows == 1;
    let min_positions = if decode { 1 } else { 128 };
    if !(min_positions..=4096).contains(&positions)
        || query.size() == 0
        || output_width == 0
        || !metal(stream)?
    {
        return Ok(None);
    }
    let shape = [batch, heads, rows, positions];
    let (mask, mask_mode) = if let Some(mask) = mask {
        let mode = match mask.dtype() {
            Dtype::Bool => 1,
            Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16 => 2,
            _ => return Ok(None),
        };
        if mask.ndim() > 4 {
            return Ok(None);
        }
        let mut axes = [1; 4];
        axes[4 - mask.ndim()..].copy_from_slice(mask.shape());
        if axes
            .iter()
            .zip(shape)
            .any(|(&axis, size)| axis != 1 && axis != size)
        {
            return Ok(None);
        }
        (mask.reshape(&axes, stream)?, mode)
    } else {
        (query.clone(), 0)
    };
    let row_count = batch
        .checked_mul(heads)
        .and_then(|count| count.checked_mul(rows))
        .ok_or_else(|| Exception::custom("ordered attention row count overflows"))?;
    let dimensions = Array::from_slice(
        &[heads, rows, width, kv_heads, positions, output_width],
        &[6],
    );
    let scale = Array::from_slice(&[scale], &[1]);
    if decode {
        let scores = DECODE_SCORES.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "ordered_f32_decode_scores",
                    ["query", "keys", "mask", "shape", "scale"],
                    ["out"],
                    include_str!("ordered_f32_decode_scores.metal"),
                    "#pragma clang fp contract(off)\n",
                    false,
                    false,
                )?);
            }
            let config = CustomKernelConfig::new()
                .with_template_arg_int("MASK_MODE", mask_mode)
                .with_grid([positions * 16, row_count, 1])
                .with_thread_group([128, 1, 1])
                .with_output_arg(shape, Dtype::Float32);
            cell.borrow()
                .as_ref()
                .expect("ordered decode score kernel initialized")
                .apply_one_device([query, keys, &mask, &dimensions, &scale], &config, stream)
        })?;
        return DECODE_VALUES.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "ordered_f32_decode_values",
                    ["scores", "values", "shape"],
                    ["out"],
                    include_str!("ordered_f32_decode_values.metal"),
                    &row_arithmetic_header(),
                    false,
                    false,
                )?);
            }
            const THREADS: i32 = 256;
            const COLUMNS: i32 = THREADS / 16;
            let config = CustomKernelConfig::new()
                .with_template_arg_int(
                    "CAPACITY",
                    (positions as u32).max(4).next_power_of_two() as i32,
                )
                .with_template_arg_int("THREADS", THREADS)
                .with_grid([THREADS, row_count, (output_width - 1) / COLUMNS + 1])
                .with_thread_group([THREADS, 1, 1])
                .with_output_arg([batch, heads, rows, output_width], Dtype::Float32);
            cell.borrow()
                .as_ref()
                .expect("ordered decode value kernel initialized")
                .apply_one_device([&scores, transposed_values, &dimensions], &config, stream)
                .map(Some)
        });
    }
    let config = CustomKernelConfig::new()
        .with_template_arg_int("CAPACITY", (positions as u32).next_power_of_two() as i32)
        .with_template_arg_int("MASK_MODE", mask_mode)
        .with_template_arg_int("THREADS", FUSED_ATTENTION_THREADS)
        .with_template_arg_int(
            "VALUE_LANES",
            ordered_f32::attention_value_lanes(query_rows) as i32,
        )
        .with_grid([FUSED_ATTENTION_THREADS, row_count, 1])
        .with_thread_group([FUSED_ATTENTION_THREADS, 1, 1])
        .with_output_arg([batch, heads, rows, output_width], Dtype::Float32);
    let query = query.contiguous(false, stream)?;
    let keys = keys.contiguous(false, stream)?;
    let values = transposed_values.contiguous(false, stream)?;
    FUSED_ATTENTION.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(MetalKernel::new(
                "ordered_f32_fused_attention",
                ["query", "keys", "values", "mask", "shape", "scale"],
                ["out"],
                include_str!("ordered_f32_fused_attention.metal"),
                &row_arithmetic_header(),
                false,
                false,
            )?);
        }
        cell.borrow()
            .as_ref()
            .expect("ordered fused attention kernel initialized")
            .apply_one_device(
                [&query, &keys, &values, &mask, &dimensions, &scale],
                &config,
                stream,
            )
            .map(Some)
    })
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
        if !silu && (SOFTMAX_MIN_WIDTH..=SOFTMAX_MAX_WIDTH).contains(&width) {
            // Reuse the same pipeline as the cached sequence grows.
            let capacity = (width as u32).div_ceil(4).next_power_of_two() as i32;
            let config = CustomKernelConfig::new()
                .with_template_arg_int("GROUP_CAPACITY", capacity)
                .with_grid([SOFTMAX_SIMD_WIDTH, work_items, 1])
                .with_thread_group([SOFTMAX_SIMD_WIDTH, 1, 1])
                .with_output_arg(input.shape(), Dtype::Float32);
            return SOFTMAX.with(|cell| {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(softmax_kernel()?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("ordered softmax kernel initialized")
                    .apply_one_device([input, &dimensions], &config, stream)
            });
        }
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
                add_softmax_mass(softmax_mass(values), high, low);
            }
            float inverse = softmax_inverse(high, low);
            for (uint i = 0; i < width; ++i) out[base+i] *= inverse;
        }
        "#,
        &row_arithmetic_header(),
        true,
        false,
    )
}

fn softmax_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "ordered_f32_simd_softmax",
        ["input", "shape"],
        ["out"],
        include_str!("ordered_f32_softmax.metal"),
        &row_arithmetic_header(),
        true,
        false,
    )
}

fn row_arithmetic_header() -> String {
    [
        super::arithmetic::metal_arithmetic_header(),
        include_str!("exp_vector_f32.metal"),
        r#"
        inline float softmax_mass(float4 values) {
            return (values.x + values.y) + (values.z + values.w);
        }
        inline void add_softmax_mass(float mass, thread float& high, thread float& low) {
            float sum = high + mass;
            float split = sum - high;
            float error = (high - (sum - split)) + (mass - split) + low;
            high = sum + error;
            low = error - (high - sum);
        }
        inline float softmax_inverse(float high, float low) {
            float quotient = metal::precise::divide(1.0f, high);
            float error = fma(-quotient, high, 1.0f) - quotient * low;
            return quotient + metal::precise::divide(error, high);
        }
        "#,
    ]
    .concat()
}
