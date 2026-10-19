//! Mixed F32-activation / narrow-weight projections without full-weight casts.
use safemlx::{error::Exception, Array, Stream};

/// Shared selector for ordinary linear layers and tied embedding readout.
/// Single rows use native-equivalent GEMV; multiple rows use the pinned native
/// F32 GEMM with narrow weight loading. Neither path evaluates inputs to decide
/// eligibility. Unsupported geometry, weight layout, dtype or native arithmetic
/// path retains the caller's existing conversion-retention/matmul fallback.
/// Forecasts remove promotion allowances only for proven invocation coverage.
pub(crate) fn project(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        thread_local! {
            static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
        }
        if input.ndim() == 0
            || weight.ndim() != 2
            || input.dtype() != Dtype::Float32
            || !matches!(weight.dtype(), Dtype::Float16 | Dtype::Bfloat16)
            || input.dim(-1) <= 0
            || weight.dim(0) <= 0
            || weight.dim(1) != input.dim(-1)
            || !weight.is_available()?
            || weight.strides()[1] != 1
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        if input.size() != input.dim(-1) as usize {
            #[cfg(feature = "projection-profiling")]
            if super::projection_profile::cast_gemm_reference_enabled() {
                return Ok(None);
            }
            return safemlx::fast::try_mixed_storage_gemm(input, weight, stream);
        }
        let width = input.dim(-1);
        let outputs = weight.dim(0);
        // Match MLX's non-transposed GEMV dispatch, including small-K and wide-K
        // reductions; changing these partitions changes floating-point results.
        let (bm, bn, sm, sn) = if width <= 64 {
            (1, 1, 8, 4)
        } else if i64::from(width) >= 16 * i64::from(outputs) {
            (1, 8, 1, 32)
        } else {
            (if outputs >= 4096 { 8 } else { 4 }, 1, 1, 32)
        };
        let tm = if outputs < 4 { 1 } else { 4 };
        let block_rows = bm * sm * tm;
        let blocks = outputs / block_rows + i32::from(outputs % block_rows != 0);
        let Some(grid_x) = 32_i32.checked_mul(blocks) else {
            return Ok(None);
        };
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("OUTPUTS", outputs)
            .with_template_arg_int("BM", bm)
            .with_template_arg_int("BN", bn)
            .with_template_arg_int("SM", sm)
            .with_template_arg_int("SN", sn)
            .with_template_arg_int("TM", tm)
            .with_grid([grid_x, bn, bm])
            .with_thread_group([32, bn, bm])
            .with_output_arg([outputs], Dtype::Float32);
        let output = KERNEL.with(|cell| -> Result<Array, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "mixed_f32_narrow_gemv",
                    ["input", "weight"],
                    ["output"],
                    include_str!("mixed_projection.metal"),
                    "",
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("mixed projection kernel initialized")
                .apply_one_device([input, weight], &config, stream)
        })?;
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("nonempty input rank") = outputs;
        output.reshape(&shape, stream).map(Some)
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, stream);
        Ok(None)
    }
}

#[cfg(feature = "projection-profiling")]
pub(crate) fn profile_path(input: &Array) -> &'static str {
    if input.size() == input.dim(-1) as usize {
        "mixed_gemv"
    } else {
        "mixed_gemm"
    }
}

/// Current weight layout plus the native dispatch's row-dependent workspace.
/// This query creates host records only; it neither evaluates nor retains arrays.
pub(crate) fn storage_facts(
    weight: &Array,
    stream: &Stream,
) -> Result<Option<eredu_runtime::projection_memory::ProjectionStorageFacts>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use eredu_core::checkpoint::TensorDtype;
        use eredu_runtime::projection_memory::{ProjectionRowStorage, ProjectionStorageFacts};
        use safemlx::{DeviceType, Dtype};
        if weight.ndim() != 2
            || !matches!(weight.dtype(), Dtype::Float16 | Dtype::Bfloat16)
            || !weight.is_available()?
            || weight.dim(0) <= 1
            || weight.dim(0) > 65536
            || weight.dim(1) <= 0
            || weight.dim(1) > 8192
            || weight.strides() != [weight.dim(1) as usize, 1]
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        #[cfg(feature = "projection-profiling")]
        let reference = super::projection_profile::cast_gemm_reference_enabled();
        #[cfg(not(feature = "projection-profiling"))]
        let reference = false;
        // Geometry-only tables are reusable across equal-shaped resident bindings.
        // The native arithmetic configuration is process-immutable. Keep the cache
        // thread-local and bounded; no array, model, stream or parameter owner is kept.
        thread_local! {
            static TABLES: std::cell::RefCell<std::collections::BTreeMap<(i32, i32, i32), Vec<ProjectionRowStorage>>> = const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
        }
        let key = (
            stream.get_device()?.get_index()?,
            weight.dim(0),
            weight.dim(1),
        );
        let mut rows = vec![ProjectionRowStorage {
            first: 1,
            last: 1,
            mechanism: "mlx.mixed_f32_narrow_gemv".into(),
            partial_bytes_per_row: 0,
        }];
        if !reference {
            let cached = TABLES.with(|tables| tables.borrow().get(&key).cloned());
            let ranges = if let Some(cached) = cached {
                cached
            } else {
                let mut ranges: Vec<ProjectionRowStorage> = Vec::new();
                for m in 2..=2000 {
                    if let Some(partials) =
                        safemlx::fast::mixed_storage_gemm_workspace(m, key.1, key.2, stream)?
                    {
                        let per_row = partials / m as u64;
                        if let Some(last) = ranges.last_mut().filter(|r| {
                            r.last + 1 == m as u64 && r.partial_bytes_per_row == per_row
                        }) {
                            last.last = m as u64;
                        } else {
                            ranges.push(ProjectionRowStorage {
                                first: m as u64,
                                last: m as u64,
                                mechanism: "mlx.mixed_storage_f32_gemm".into(),
                                partial_bytes_per_row: per_row,
                            });
                        }
                    }
                }
                TABLES.with(|tables| {
                    let mut tables = tables.borrow_mut();
                    if tables.len() >= 256 {
                        tables.clear();
                    }
                    tables.insert(key, ranges.clone());
                });
                ranges
            };
            rows.extend(ranges);
        }
        Ok(Some(ProjectionStorageFacts {
            input: key.2 as u64,
            output: key.1 as u64,
            activation: TensorDtype::F32,
            weight: if weight.dtype() == Dtype::Float16 {
                TensorDtype::F16
            } else {
                TensorDtype::Bf16
            },
            output_scalar_bytes: 4,
            activation_copy_bytes_per_row: key.2 as u64 * 4,
            rows,
        }))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (weight, stream);
        Ok(None)
    }
}
