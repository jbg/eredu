//! Native FP32 pointwise and softmax arithmetic with explicit rounding.
use safemlx::{error::Exception, Array, Stream};
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
        let mut output=KERNEL.with(|cell| -> Result<Vec<Array>,Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut()=Some(MetalKernel::new("f32_pointwise",["input"],["output"],
                    "uint i=thread_position_in_grid.x; if(i>=SIZE) return; float x=input[i]; float denominator=1.0f+stable_exp_f32(-x); output[i]=(OP==1 ? 1.0f : x)/denominator;",
                    include_str!("exp_f32.metal"),true,false)?);
            }
            cell.borrow().as_ref().expect("pointwise kernel initialized").apply_device([kernel_input],&config,stream)
        })?;
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, operation, stream);
        Ok(None)
    }
}

/// Cooperative last-axis softmax preserving four ordered FP32 sum chains.
pub(crate) fn softmax_last(input: &Array, stream: &Stream) -> Result<Option<Array>, Exception> {
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
                    include_str!("exp_f32.metal"),
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
        let _ = (input, stream);
        Ok(None)
    }
}
