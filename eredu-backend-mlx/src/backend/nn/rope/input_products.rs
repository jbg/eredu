//! Native fusion of frequency-based rotary encoding with F32 product boundaries.
use safemlx::{error::Exception, Array, Stream};

pub(super) fn forward(
    input: &Array,
    frequencies: &Array,
    dimensions: i32,
    traditional: bool,
    offset: i32,
    amplitude: f32,
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
            || input.shape().len() < 2
            || dimensions <= 0
            || dimensions % 2 != 0
            || dimensions > input.dim(-1)
            || frequencies.dtype() != Dtype::Float32
            || frequencies.shape() != [dimensions / 2]
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("rotary row count overflow"))?;
        let work = dimensions / 2 + width - dimensions;
        let count = rows
            .checked_mul(work)
            .ok_or_else(|| Exception::custom("rotary grid extent overflow"))?;
        // Sequence extents remain runtime inputs instead of shader specializations.
        let position = Array::from_slice(&[offset, count, input.dim(-2)], &[3]);
        let scale = Array::from_f32(amplitude);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("DIMENSIONS", dimensions)
            .with_template_arg_bool("TRADITIONAL", traditional)
            .with_grid([count, 1, 1])
            .with_thread_group([128, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_input_products_rotary",
                    ["input", "frequencies", "position", "amplitude"],
                    ["output"],
                    include_str!("input_products.metal"),
                    "",
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("rotary kernel initialized")
                .apply_device([input, frequencies, &position, &scale], &config, stream)
        })?;
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (
            input,
            frequencies,
            dimensions,
            traditional,
            offset,
            amplitude,
            stream,
        );
        Ok(None)
    }
}
