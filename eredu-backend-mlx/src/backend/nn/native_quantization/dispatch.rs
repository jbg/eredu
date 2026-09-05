use super::*;
use super::{decode::*, kernels::*, storage::*};

/// Applies an group-major native quantized matrix bank.
pub fn native_grouped_linear(
    input: &Array,
    weight: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2
        || group_ids.ndim() != 1
        || input.dim(0) != group_ids.dim(0)
        || input.dim(1) != weight.columns
    {
        return Err(Exception::custom(format!(
            "native grouped linear shape mismatch: input {:?}, ids {:?}, weight {:?}",
            input.shape(),
            group_ids.shape(),
            weight.shape()
        )));
    }
    if native_execution_backend(stream)? == NativeExecutionBackend::Metal {
        return match weight.format() {
            NativeQuantizationFormat::GgufQ4K => {
                q4k_grouped_metal(input, weight, group_ids, stream)
            }
            NativeQuantizationFormat::GgufQ5_1 => {
                q5_1_grouped_metal(input, weight, group_ids, stream)
            }
            NativeQuantizationFormat::GgufQ8_0 => {
                q8_0_grouped_metal(input, weight, group_ids, stream)
            }
            _ => iq_grouped_metal(input, weight, group_ids, stream),
        };
    }
    native_grouped_linear_cpu(input, weight, group_ids, stream)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn is_gpu(stream: &Stream) -> Result<bool, Exception> {
    Ok(stream.get_device()?.get_type()? == DeviceType::Gpu)
}

/// Selects a native execution backend without consulting model architecture.
pub(super) fn native_execution_backend(
    stream: &Stream,
) -> Result<NativeExecutionBackend, Exception> {
    #[cfg(feature = "cuda")]
    {
        let _ = stream;
        Ok(NativeExecutionBackend::GenericFallback)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(if is_gpu(stream)? {
            NativeExecutionBackend::Metal
        } else {
            NativeExecutionBackend::GenericFallback
        })
    }
}

pub(super) fn q4k_config(
    rows: i32,
    view: &NativeQuantizedTensor,
    output_rows: i32,
    output_cols: i32,
    dtype: Dtype,
) -> CustomKernelConfig {
    let out_grid = ((output_cols + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", output_cols)
        .with_template_arg_int("OUT_GRID", out_grid)
        .with_template_arg_int("BLOCKS", view.columns / Q4_K_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("REDUCTION_TILE", REDUCTION_TILE)
        .with_template_arg_int("OUT_TILE", OUT_TILE)
        .with_grid([REDUCTION_TILE, rows * out_grid, 1])
        .with_thread_group([REDUCTION_TILE, OUT_TILE, 1])
        .with_output_arg([output_rows, output_cols], dtype)
}

pub(super) fn q4k_decode_config(view: &NativeQuantizedTensor, dtype: Dtype) -> CustomKernelConfig {
    let output_pairs = (view.rows + Q4K_DECODE_ROWS_PER_SIMD - 1) / Q4K_DECODE_ROWS_PER_SIMD;
    let pair_grid = ((output_pairs + Q4K_DECODE_SIMD_GROUPS - 1) / Q4K_DECODE_SIMD_GROUPS)
        * Q4K_DECODE_SIMD_GROUPS;
    CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / Q4_K_BLOCK_VALUES)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("ROWS_PER_SIMD", Q4K_DECODE_ROWS_PER_SIMD)
        .with_grid([REDUCTION_TILE, pair_grid, 1])
        .with_thread_group([REDUCTION_TILE, Q4K_DECODE_SIMD_GROUPS, 1])
        .with_output_arg([1, view.rows], dtype)
}

pub(super) fn qk_decode_config(
    view: &NativeQuantizedTensor,
    dtype: Dtype,
    rows_per_simd: i32,
) -> CustomKernelConfig {
    let output_tiles = (view.rows + rows_per_simd - 1) / rows_per_simd;
    let tile_grid = ((output_tiles + Q4K_DECODE_SIMD_GROUPS - 1) / Q4K_DECODE_SIMD_GROUPS)
        * Q4K_DECODE_SIMD_GROUPS;
    CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / Q5_K_BLOCK_VALUES)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("ROWS_PER_SIMD", rows_per_simd)
        .with_grid([REDUCTION_TILE, tile_grid, 1])
        .with_thread_group([REDUCTION_TILE, Q4K_DECODE_SIMD_GROUPS, 1])
        .with_output_arg([1, view.rows], dtype)
}

pub(super) fn validate_activation_dtype(input: &Array) -> Result<Dtype, Exception> {
    let dtype = input.dtype();
    if !matches!(dtype, Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16) {
        return Err(Exception::custom(format!(
            "native quantized activation must be float16, bfloat16, or float32, got {dtype:?}"
        )));
    }
    Ok(dtype)
}

pub(super) fn q4k_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q4_K linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let output = match k_quant_linear_kernel_class(outer) {
        KQuantLinearKernelClass::MatrixVector => {
            let config = q4k_decode_config(view, dtype);
            Q4K_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(q4k_linear_kernel()?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("Q4_K linear kernel initialized")
                    .apply_one_device([&flat, view.storage.bytes()], &config, stream)
            })?
        }
        KQuantLinearKernelClass::SmallBatch => {
            q4k_batch_linear_metal(&flat, view, outer, dtype, stream)?
        }
        KQuantLinearKernelClass::MatrixMatrix if k_quant_supports_tiled_matmul(view) => {
            qk_matmul_metal(
                &flat,
                view,
                outer,
                dtype,
                NativeQuantizationFormat::GgufQ4K,
                stream,
            )?
        }
        KQuantLinearKernelClass::MatrixMatrix => {
            q4k_batch_linear_metal(&flat, view, outer, dtype, stream)?
        }
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

pub(super) fn q5k_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q5_K linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let output = match k_quant_linear_kernel_class(outer) {
        KQuantLinearKernelClass::MatrixVector => {
            let config = qk_decode_config(view, dtype, 1);
            Q5K_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(q5k_linear_kernel()?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("Q5_K linear kernel initialized")
                    .apply_one_device([&flat, view.storage.bytes()], &config, stream)
            })?
        }
        KQuantLinearKernelClass::SmallBatch => {
            qk_batch_linear_metal(&flat, view, outer, dtype, true, stream)?
        }
        KQuantLinearKernelClass::MatrixMatrix if k_quant_supports_tiled_matmul(view) => {
            qk_matmul_metal(
                &flat,
                view,
                outer,
                dtype,
                NativeQuantizationFormat::GgufQ5K,
                stream,
            )?
        }
        KQuantLinearKernelClass::MatrixMatrix => {
            qk_batch_linear_metal(&flat, view, outer, dtype, true, stream)?
        }
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

pub(super) fn q6k_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q6_K linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let output = match k_quant_linear_kernel_class(outer) {
        KQuantLinearKernelClass::MatrixVector => {
            let config = qk_decode_config(view, dtype, 2);
            Q6K_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(q6k_linear_kernel()?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("Q6_K linear kernel initialized")
                    .apply_one_device([&flat, view.storage.bytes()], &config, stream)
            })?
        }
        KQuantLinearKernelClass::SmallBatch => {
            qk_batch_linear_metal(&flat, view, outer, dtype, false, stream)?
        }
        KQuantLinearKernelClass::MatrixMatrix if k_quant_supports_tiled_matmul(view) => {
            qk_matmul_metal(
                &flat,
                view,
                outer,
                dtype,
                NativeQuantizationFormat::GgufQ6K,
                stream,
            )?
        }
        KQuantLinearKernelClass::MatrixMatrix => {
            qk_batch_linear_metal(&flat, view, outer, dtype, false, stream)?
        }
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

pub(super) fn q4k_batch_linear_metal(
    flat: &Array,
    view: &NativeQuantizedTensor,
    rows: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array, Exception> {
    let out_grid = ((view.rows + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    let batch_tile = native_batch_tile(rows);
    let batch_tiles = (rows + batch_tile - 1) / batch_tile;
    let config = q4k_config(rows, view, rows, view.rows, dtype)
        .with_template_arg_int("ROWS", rows)
        .with_template_arg_int("BATCH_TILE", batch_tile)
        .with_grid([REDUCTION_TILE, out_grid, batch_tiles]);
    Q4K_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_batch_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K batch kernel initialized")
            .apply_one_device([flat, view.storage.bytes()], &config, stream)
    })
}

pub(super) fn qk_batch_linear_metal(
    flat: &Array,
    view: &NativeQuantizedTensor,
    rows: i32,
    dtype: Dtype,
    q5k: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    const QK_OUTPUT_ROWS_PER_GROUP: i32 = 8;
    const QK_SIMD_GROUPS: i32 = 2;
    let output_groups = (view.rows + QK_OUTPUT_ROWS_PER_GROUP - 1) / QK_OUTPUT_ROWS_PER_GROUP;
    let batch_tile = qk_small_batch_tile(rows);
    let batch_tiles = (rows + batch_tile - 1) / batch_tile;
    let config = q4k_config(rows, view, rows, view.rows, dtype)
        .with_template_arg_int("ROWS", rows)
        .with_template_arg_int("BATCH_TILE", batch_tile)
        .with_grid([REDUCTION_TILE, output_groups * QK_SIMD_GROUPS, batch_tiles])
        .with_thread_group([REDUCTION_TILE, QK_SIMD_GROUPS, 1]);
    if q5k {
        Q5K_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q5k_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q5_K batch kernel initialized")
                .apply_one_device([flat, view.storage.bytes()], &config, stream)
        })
    } else {
        Q6K_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q6k_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q6_K batch kernel initialized")
                .apply_one_device([flat, view.storage.bytes()], &config, stream)
        })
    }
}

pub(super) fn qk_matmul_metal(
    flat: &Array,
    view: &NativeQuantizedTensor,
    rows: i32,
    dtype: Dtype,
    format: NativeQuantizationFormat,
    stream: &Stream,
) -> Result<Array, Exception> {
    const OUTPUT_TILE: i32 = 64;
    const ACTIVATION_TILE: i32 = 32;
    const SIMD_GROUPS: i32 = 4;
    debug_assert!(k_quant_supports_tiled_matmul(view));
    let output_tiles = (view.rows + OUTPUT_TILE - 1) / OUTPUT_TILE;
    let activation_tiles = (rows + ACTIVATION_TILE - 1) / ACTIVATION_TILE;
    let config = CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_grid([REDUCTION_TILE, output_tiles * SIMD_GROUPS, activation_tiles])
        .with_thread_group([REDUCTION_TILE, SIMD_GROUPS, 1])
        .with_output_arg([rows, view.rows], dtype);
    match format {
        NativeQuantizationFormat::GgufQ4K => {
            Q4K_MATMUL_KERNEL.with(|cell| -> Result<_, Exception> {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(qk_matmul_kernel(format)?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("Q4_K matrix-matrix kernel initialized")
                    .apply_one_device([flat, view.storage.bytes()], &config, stream)
            })
        }
        NativeQuantizationFormat::GgufQ5K => {
            Q5K_MATMUL_KERNEL.with(|cell| -> Result<_, Exception> {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(qk_matmul_kernel(format)?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("Q5_K matrix-matrix kernel initialized")
                    .apply_one_device([flat, view.storage.bytes()], &config, stream)
            })
        }
        NativeQuantizationFormat::GgufQ6K => {
            Q6K_MATMUL_KERNEL.with(|cell| -> Result<_, Exception> {
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(qk_matmul_kernel(format)?);
                }
                cell.borrow()
                    .as_ref()
                    .expect("Q6_K matrix-matrix kernel initialized")
                    .apply_one_device([flat, view.storage.bytes()], &config, stream)
            })
        }
        _ => Err(Exception::custom(format!(
            "native tiled matrix-matrix does not support {format:?}"
        ))),
    }
}

pub(super) fn q4k_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let routes = input.dim(0);
    let config = q4k_config(routes, view, routes, view.rows, dtype)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q4K_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_grouped_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

pub(super) fn q5_1_config(
    logical_rows: i32,
    view: &NativeQuantizedTensor,
    output_rows: i32,
    output_cols: i32,
    dtype: Dtype,
) -> CustomKernelConfig {
    let out_grid = ((output_cols + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", output_cols)
        .with_template_arg_int("OUT_GRID", out_grid)
        .with_template_arg_int("BLOCKS", view.columns / Q5_1_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("REDUCTION_TILE", REDUCTION_TILE)
        .with_template_arg_int("OUT_TILE", OUT_TILE)
        .with_grid([REDUCTION_TILE, logical_rows * out_grid, 1])
        .with_thread_group([REDUCTION_TILE, OUT_TILE, 1])
        .with_output_arg([output_rows, output_cols], dtype)
}

pub(super) fn iq_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "IQ linear input {:?} does not match {} columns",
            input.shape(),
            view.columns
        )));
    }
    let rows = input.size() as i32 / view.columns;
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let flat = input.reshape(&[rows, view.columns], stream)?;
    let (_, block_bytes) = view.format().block_geometry();
    let batch_tile = native_batch_tile(rows);
    let mut config = CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("ROWS", rows)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / view.format().block_geometry().0)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("BATCH_TILE", batch_tile)
        .with_grid([32, rows * view.rows, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg([rows, view.rows], dtype);
    let output = if rows == 1 {
        IQ_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            let mut kernels = cell.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
                entry.insert(iq_linear_kernel(view.format(), big_endian)?);
            }
            kernels
                .get(&kernel_key)
                .expect("IQ linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (rows + batch_tile - 1) / batch_tile;
        config = config.with_grid([32, view.rows, batch_tiles]);
        IQ_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            let mut kernels = cell.borrow_mut();
            if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
                entry.insert(iq_batch_kernel(view.format(), big_endian)?);
            }
            kernels
                .get(&kernel_key)
                .expect("IQ batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

pub(super) fn iq_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let (block_values, block_bytes) = view.format().block_geometry();
    let config = CustomKernelConfig::new()
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("ROWS", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / block_values)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg([count, view.columns], Dtype::Float32);
    let output = IQ_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        let mut kernels = cell.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
            entry.insert(iq_embedding_kernel(view.format(), big_endian)?);
        }
        kernels
            .get(&kernel_key)
            .expect("IQ embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}

pub(super) fn iq_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let rows = input.dim(0);
    let big_endian = view.storage.endian == GgufEndian::Big;
    let kernel_key = (view.format(), big_endian);
    let (block_values, block_bytes) = view.format().block_geometry();
    let config = CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / block_values)
        .with_template_arg_int("BLOCK_BYTES", block_bytes)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([32, rows * view.rows, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg([rows, view.rows], dtype);
    IQ_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        let mut kernels = cell.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = kernels.entry(kernel_key) {
            entry.insert(iq_grouped_kernel(view.format(), big_endian)?);
        }
        kernels
            .get(&kernel_key)
            .expect("IQ grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

pub(super) fn q8_0_config(
    view: &NativeQuantizedTensor,
    output_rows: i32,
    output_cols: i32,
    dtype: Dtype,
    batch_tile: i32,
) -> CustomKernelConfig {
    let out_tile = q8_0_out_tile(output_cols);
    let out_grid = ((output_cols + out_tile - 1) / out_tile) * out_tile;
    CustomKernelConfig::new()
        .with_template_arg_dtype("T", dtype)
        .with_template_arg_int("ROWS", output_rows)
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("OUT_DIM", output_cols)
        .with_template_arg_int("OUT_GRID", out_grid)
        .with_template_arg_int("BLOCKS", view.columns / Q8_0_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_template_arg_int("OUT_TILE", out_tile)
        .with_template_arg_int("BATCH_TILE", batch_tile)
        .with_thread_group([REDUCTION_TILE, out_tile, 1])
        .with_output_arg([output_rows, output_cols], dtype)
}

pub(super) fn q8_0_out_tile(output_cols: i32) -> i32 {
    if output_cols >= Q8_LARGE_OUTPUT_ROWS {
        Q8_LARGE_OUT_TILE
    } else {
        OUT_TILE
    }
}

pub(super) fn q5_1_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let routes = input.dim(0);
    let config = q5_1_config(routes, view, routes, view.rows, dtype)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count);
    Q5_1_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q5_1_grouped_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q5_1 grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

pub(super) fn q5_1_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q5_1 linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let out_grid = ((view.rows + OUT_TILE - 1) / OUT_TILE) * OUT_TILE;
    let batch_tile = native_batch_tile(outer);
    let mut config = q5_1_config(outer, view, outer, view.rows, dtype)
        .with_template_arg_int("ROWS", outer)
        .with_template_arg_int("BATCH_TILE", batch_tile);
    let output = if outer == 1 {
        Q5_1_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q5_1_linear_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q5_1 linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (outer + batch_tile - 1) / batch_tile;
        config = config.with_grid([REDUCTION_TILE, out_grid, batch_tiles]);
        Q5_1_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q5_1_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q5_1 batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

pub(super) fn q5_1_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let config = q5_1_config(count, view, count, view.columns, Dtype::Float32)
        .with_template_arg_int("ROWS", view.rows)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1]);
    let output = Q5_1_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q5_1_embedding_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q5_1 embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}

pub(super) fn q8_0_linear_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dim(-1) != view.columns {
        return Err(Exception::custom(format!(
            "native Q8_0 linear expected input dimension {}, got {:?}",
            view.columns,
            input.shape()
        )));
    }
    let dtype = validate_activation_dtype(input)?;
    let outer = input.size() as i32 / view.columns;
    let flat = input.reshape(&[outer, view.columns], stream)?;
    let out_tile = q8_0_out_tile(view.rows);
    let out_grid = ((view.rows + out_tile - 1) / out_tile) * out_tile;
    let batch_tile = native_batch_tile(outer);
    let mut config = q8_0_config(view, outer, view.rows, dtype, batch_tile);
    let output = if outer == 1 {
        config = config.with_grid([REDUCTION_TILE, out_grid, 1]);
        Q8_0_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q8_0_linear_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q8_0 linear kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    } else {
        let batch_tiles = (outer + batch_tile - 1) / batch_tile;
        config = config.with_grid([REDUCTION_TILE, out_grid, batch_tiles]);
        Q8_0_BATCH_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(q8_0_batch_kernel()?);
            }
            cell.borrow()
                .as_ref()
                .expect("Q8_0 batch kernel initialized")
                .apply_one_device([&flat, view.storage.bytes()], &config, stream)
        })?
    };
    let mut shape = input.shape()[..input.ndim() - 1].to_vec();
    shape.push(view.rows);
    output.reshape(&shape, stream)
}

pub(super) fn q8_0_grouped_metal(
    input: &Array,
    view: &NativeQuantizedTensor,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = validate_activation_dtype(input)?;
    let routes = input.dim(0);
    let out_tile = q8_0_out_tile(view.rows);
    let out_grid = ((view.rows + out_tile - 1) / out_tile) * out_tile;
    let config = q8_0_config(view, routes, view.rows, dtype, NATIVE_BATCH_TILE)
        .with_template_arg_int("MATRIX_COUNT", view.matrix_count)
        .with_grid([REDUCTION_TILE, routes * out_grid, 1]);
    Q8_0_GROUPED_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q8_0_grouped_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q8_0 grouped kernel initialized")
            .apply_one_device([input, view.storage.bytes(), group_ids], &config, stream)
    })
}

pub(super) fn q4k_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let config = CustomKernelConfig::new()
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("ROWS", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / Q4_K_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg([count, view.columns], Dtype::Float32);
    let output = Q4K_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q4k_embedding_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q4_K embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}

pub(super) fn q8_0_embedding_metal(
    indices: &Array,
    view: &NativeQuantizedTensor,
    stream: &Stream,
) -> Result<Array, Exception> {
    let count = indices.size() as i32;
    let config = CustomKernelConfig::new()
        .with_template_arg_int("IN_DIM", view.columns)
        .with_template_arg_int("ROWS", view.rows)
        .with_template_arg_int("BLOCKS", view.columns / Q8_0_BLOCK_VALUES)
        .with_template_arg_int("PHYSICAL_ROWS", view.physical_rows)
        .with_template_arg_int("ROW_START", view.row_start)
        .with_grid([count * view.columns, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg([count, view.columns], Dtype::Float32);
    let output = Q8_0_EMBEDDING_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(q8_0_embedding_kernel()?);
        }
        cell.borrow()
            .as_ref()
            .expect("Q8_0 embedding kernel initialized")
            .apply_one_device([view.storage.bytes(), indices], &config, stream)
    })?;
    let mut shape = indices.shape().to_vec();
    shape.push(view.columns);
    output.reshape(&shape, stream)
}
