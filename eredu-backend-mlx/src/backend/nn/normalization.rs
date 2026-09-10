//! Native realization of reproducible weighted-normalization arithmetic.
use safemlx::{error::Exception, Array, Stream};

pub(crate) fn f32_weightless_rms(
    input: &Array,
    epsilon: f32,
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
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("normalization row count overflow"))?;
        let squared = input.square(stream)?;
        let epsilon = Array::from_slice(&[epsilon], &[1]);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_grid([rows, 1, 1])
            .with_thread_group([32, 1, 1])
            .with_template_arg_int("ROWS", rows)
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() =
                    Some(MetalKernel::new(
                        "f32_weightless_rms",
                        ["input", "squared", "epsilon"],
                        ["output"],
                        concat!("uint row = thread_position_in_grid.x; if (row >= ROWS) return;",
                        "size_t base = size_t(row) * WIDTH;",
                        "float variance = cascade_sum_f32(squared + base, WIDTH) / float(WIDTH);",
                        "float inverse = 1.0f / metal::precise::sqrt(variance + epsilon[0]);",
                        "for (uint i=0; i<WIDTH; ++i) output[base+i] = input[base+i] * inverse;"),
                        include_str!("rms_cascade.metal"),
                        true,
                        false,
                    )?);
            }
            cell.borrow()
                .as_ref()
                .expect("normalization kernel initialized")
                .apply_device([input, &squared, &epsilon], &config, stream)
        })?;
        return Ok(output.pop());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, epsilon, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype};
    #[test]
    fn native_cascade_normalization_matches_independent_bf16_rounding() {
        let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let fixture = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/bf16_normalization.safetensors"
            ),
            &weights,
        )
        .unwrap();
        for width in [32, 768, 1280, 2560] {
            let input = fixture[&format!("{width}.input")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap();
            let weight = fixture[&format!("{width}.weight")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap();
            let normalized = f32_weightless_rms(&input, 1e-6, &stream).unwrap().unwrap();
            let output = normalized
                .multiply(&weight, &stream)
                .unwrap()
                .as_dtype(Dtype::Bfloat16, &stream)
                .unwrap()
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let expected = fixture[&format!("{width}.output")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let differences = output
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differences, 0, "width {width}");
        }
    }
}

pub(crate) fn f32_sum_last(input: &Array, stream: &Stream) -> Result<Option<Array>, Exception> {
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
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("row sum count overflow"))?;
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("row sum rank") = 1;
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("ROWS", rows)
            .with_grid([rows, 1, 1])
            .with_thread_group([32, 1, 1])
            .with_output_arg(shape.as_slice(), Dtype::Float32);
        let mut output=KERNEL.with(|cell| -> Result<Vec<Array>,Exception> {
            if cell.borrow().is_none() { *cell.borrow_mut()=Some(MetalKernel::new("f32_sum_last",["input"],["output"],
                "uint row=thread_position_in_grid.x; if(row>=ROWS) return; output[row]=cascade_sum_f32(input+size_t(row)*WIDTH,WIDTH);",
                include_str!("rms_cascade.metal"),true,false)?); }
            cell.borrow().as_ref().expect("row sum kernel initialized").apply_device([input],&config,stream)
        })?;
        return Ok(output.pop());
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, stream);
        Ok(None)
    }
}
