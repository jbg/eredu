//! Native pointwise and softmax arithmetic with explicit rounding.
use safemlx::{error::Exception, Array, Stream};
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn metal_arithmetic_header() -> &'static str {
    static HEADER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HEADER.get_or_init(|| {
        [
            eredu_nn::arithmetic::STABLE_EXP_SOURCE,
            include_str!("bf16_softmax_inverse.metal"),
        ]
        .concat()
    })
}

#[derive(Clone, Copy)]
pub(crate) enum Pointwise {
    Sigmoid = 1,
    Silu = 2,
}
pub(crate) fn f32_pointwise(
    input: &Array,
    operation: Pointwise,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        if input.dtype() != Dtype::Float32
            || input.size() == 0
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let size = i32::try_from(input.size())
            .map_err(|_| Exception::custom("pointwise size overflow"))?;
        // MLX declares zero-dimensional custom-kernel inputs as scalar references.
        // Keep pointer indexing and the same rounded equation for every shape;
        // the output configuration retains the caller's original scalar shape.
        let scalar_input = (input.ndim() == 0)
            .then(|| input.reshape(&[1], stream))
            .transpose()?;
        let kernel_input = scalar_input.as_ref().unwrap_or(input);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("SIZE", size)
            .with_template_arg_int("OP", operation as i32)
            .with_grid([size, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_pointwise",
                    ["input"],
                    ["output"],
                    r#"
                    uint i = thread_position_in_grid.x;
                    if (i >= SIZE) return;
                    float x = input[i];
                    if (OP == 1) {
                        output[i] = metal::fast::divide(1.0f, 1.0f + metal::fast::exp(-x));
                    } else {
                        output[i] = x / (1.0f + stable_exp_f32(-x));
                    }
                    "#,
                    metal_arithmetic_header(),
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("pointwise kernel initialized")
                .apply_device([kernel_input], &config, stream)
        })?;
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, operation, stream);
        Ok(None)
    }
}

/// Single-row BF16 sigmoid or ordinary SwiGLU with FP32 activation arithmetic.
pub(crate) fn bf16_pointwise(
    input: &Array,
    up: Option<&Array>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        if input.dtype() != Dtype::Bfloat16
            || input.ndim() == 0
            || input.size() == 0
            || input.size() != input.dim(-1) as usize
            || up.is_some_and(|up| up.shape() != input.shape() || up.dtype() != Dtype::Bfloat16)
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let config = CustomKernelConfig::new()
            .with_template_arg_bool("SWIGLU", up.is_some())
            .with_grid([input.dim(-1), 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg(input.shape(), Dtype::Bfloat16);
        KERNEL.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "bf16_pointwise",
                    ["input", "up"],
                    ["output"],
                    r#"
                    uint i = thread_position_in_grid.x;
                    float x = float(input[i]);
                    if (SWIGLU) {
                        // The multiplication consumes the rounded SiLU result.
                        bfloat16_t gate = bfloat16_t(x / (1.0f + stable_exp_f32(-x)));
                        output[i] = bfloat16_t(float(gate) * float(up[i]));
                    } else {
                        output[i] = bfloat16_t(
                            metal::fast::divide(1.0f, 1.0f + metal::fast::exp(-x)));
                    }
                    "#,
                    metal_arithmetic_header(),
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("BF16 pointwise kernel initialized")
                .apply_one_device([input, up.unwrap_or(input)], &config, stream)
                .map(Some)
        })
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, up, stream);
        Ok(None)
    }
}

/// Cooperative FP32 softmax. Narrow probability storage uses a stable
/// exponential, pairwise summation and precise reciprocal before rounding.
pub(crate) fn softmax_last(
    input: &Array,
    narrow_probabilities: bool,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        if input.dtype() != Dtype::Float32
            || input.size() == 0
            || input.dim(-1) < 4
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("softmax row count overflow"))?;
        // Shape data is a runtime input: growing histories reuse one shader.
        let dimensions = Array::from_slice(&[width, rows], &[2]);
        let config = CustomKernelConfig::new()
            .with_template_arg_bool("PAIRWISE", narrow_probabilities)
            .with_grid([32, rows, 1])
            .with_thread_group([32, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_ordered_parallel_softmax_last",
                    ["input", "dimensions"],
                    ["output"],
                    include_str!("softmax_f32.metal"),
                    metal_arithmetic_header(),
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("softmax kernel initialized")
                .apply_device([input, &dimensions], &config, stream)
        })?;
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, narrow_probabilities, stream);
        Ok(None)
    }
}

/// One correctly rounded binary16 fused update using exact FP32 products and
/// an error-free sum. An odd FP32 low bit retains the side of a half midpoint.
pub(crate) fn fused_binary16_lane(
    x: &Array,
    y: &Array,
    lane: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    use safemlx::{ops::r#where, Dtype};
    let zero = Array::from_f32(0.0);
    let one = Array::from_slice(&[1u32], &[]);
    let two = Array::from_slice(&[2u32], &[]);
    let zero_bits = Array::from_slice(&[0u32], &[]);

    let product = x.multiply(y, stream)?;
    let sum = product.add(lane, stream)?;
    let part = sum.subtract(&product, stream)?;
    let residual = product
        .subtract(sum.subtract(&part, stream)?, stream)?
        .add(lane.subtract(part, stream)?, stream)?;
    let bits = sum.view_dtype(Dtype::Uint32, stream)?;
    let even = bits.remainder(&two, stream)?.eq(&zero_bits, stream)?;
    let increment = sum
        .lt(&zero, stream)?
        .eq(residual.lt(&zero, stream)?, stream)?;
    let adjusted = r#where(
        increment,
        bits.add(&one, stream)?,
        bits.subtract(&one, stream)?,
        stream,
    )?
    .view_dtype(Dtype::Float32, stream)?;
    let adjust = even.logical_and(
        residual
            .gt(&zero, stream)?
            .logical_or(residual.lt(&zero, stream)?, stream)?,
        stream,
    )?;
    r#where(adjust, adjusted, sum, stream)?
        .as_dtype(Dtype::Float16, stream)?
        .as_dtype(Dtype::Float32, stream)
}

/// Fused FP32 pointwise update. The multiplier may broadcast to the input.
pub(crate) fn fused_f32(
    input: &Array,
    multiplier: &Array,
    addend: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    use safemlx::{ops::broadcast_to, DeviceType, Dtype};

    let shape = input.shape();
    if input.size() == 0 {
        return Ok(input.clone());
    }
    let multiplier = broadcast_to(multiplier, shape, stream)?;
    if stream.get_device()?.get_type()? == DeviceType::Cpu {
        let read = |array: &Array| array.contiguous(false, stream)?.into_evaluated();
        let x = read(input)?;
        let y = read(&multiplier)?;
        let z = read(addend)?;
        let values: Vec<_> = x
            .as_slice::<f32>()
            .iter()
            .zip(y.as_slice::<f32>())
            .zip(z.as_slice::<f32>())
            .map(|((&x, &y), &z)| x.mul_add(y, z))
            .collect();
        return Ok(Array::from_slice(&values, shape));
    }

    let size = i32::try_from(input.size())
        .map_err(|_| Exception::custom("fused FP32 output size exceeds i32"))?;
    let config = safemlx::fast::CustomKernelConfig::new()
        .with_template_arg_int("SIZE", size)
        .with_grid([size, 1, 1])
        .with_thread_group([256, 1, 1])
        .with_output_arg(shape, Dtype::Float32);
    let inputs = [input, &multiplier, addend];
    #[cfg(any(feature = "metal", feature = "cuda"))]
    {
        use std::cell::RefCell;
        #[cfg(feature = "cuda")]
        fn kernel() -> Result<safemlx::fast::CudaKernel, Exception> {
            safemlx::fast::CudaKernel::new(
                "fused_f32",
                ["x", "y", "z"],
                ["out"],
                "int i = blockIdx.x * blockDim.x + threadIdx.x;
                 if (i < SIZE) out[i] = __fmaf_rn(x[i], y[i], z[i]);",
                "",
                true,
                0,
            )
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        fn kernel() -> Result<safemlx::fast::MetalKernel, Exception> {
            safemlx::fast::MetalKernel::new(
                "fused_f32",
                ["x", "y", "z"],
                ["out"],
                "uint i = thread_position_in_grid.x;
                 if (i < SIZE) out[i] = metal::fma(x[i], y[i], z[i]);",
                "",
                true,
                false,
            )
        }
        #[cfg(feature = "cuda")]
        use safemlx::fast::CudaKernel as Kernel;
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        use safemlx::fast::MetalKernel as Kernel;
        thread_local! { static KERNEL: RefCell<Option<Kernel>> = const { RefCell::new(None) }; }
        return KERNEL.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(kernel()?);
            }
            let output = cell
                .borrow()
                .as_ref()
                .expect("fused FP32 kernel initialized")
                .apply_device(inputs, &config, stream)?;
            output
                .into_iter()
                .next()
                .ok_or_else(|| Exception::custom("missing fused FP32 output"))
        });
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let _ = (inputs, config);
        Err(Exception::custom(
            "fused FP32 accelerator kernel unavailable",
        ))
    }
}
