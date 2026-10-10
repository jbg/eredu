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

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype};

    /// Compatibility reference for the released ordered FP32 softmax arithmetic.
    #[test]
    fn cooperative_softmax_preserves_ordered_scores_tails_and_exceptional_rows() {
        use safemlx::fast::{CustomKernelConfig, MetalKernel};
        use safemlx::ops::indexing::TryIndexOp;
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let reference = MetalKernel::new(
            "serial_softmax_compatibility_reference",
            ["input"], ["output"],
            concat!("uint row=thread_position_in_grid.x; if(row>=ROWS) return; size_t base=size_t(row)*WIDTH;",
                "float maximum=-INFINITY; for(uint i=0;i<WIDTH;++i) { float v=input[base+i]; maximum=(isnan(v)||isnan(maximum)) ? NAN : max(maximum,v); }",
                "float sums[4]={}; for(uint i=0;i<WIDTH;++i) { float v=stable_exp_f32(input[base+i]-maximum); output[base+i]=v; sums[i%4]+=v; }",
                "float inverse=1.0f/((sums[0]+sums[2])+(sums[1]+sums[3]));",
                "for(uint i=0;i<WIDTH;++i) output[base+i]*=inverse;"),
            include_str!("exp_f32.metal"), true, false,
        ).unwrap();
        // Reuse one cooperative shader across vector/SIMD tails and changing
        // shapes, including strided origins and more than one SIMD of rows.
        for width in [4, 5, 7, 31, 32, 33, 257, 511, 512, 513, 1024, 1537, 4097] {
            let rows = 35;
            let mut data: Vec<f32> = (0..rows * (width + 3))
                .map(|i| (((i * 17) % 131) as f32 - 65.0) * 0.25)
                .collect();
            for row in 0..5 {
                for column in 0..width {
                    data[row * (width + 3) + column + 1] = match row {
                        0 => {
                            if column % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        1 => f32::NEG_INFINITY,
                        2 => {
                            if column == width - 1 {
                                f32::NAN
                            } else {
                                column as f32
                            }
                        }
                        3 => {
                            if column == 0 {
                                f32::INFINITY
                            } else {
                                -(column as f32)
                            }
                        }
                        _ => {
                            if column % 3 == 0 {
                                f32::NEG_INFINITY
                            } else {
                                -(column as f32)
                            }
                        }
                    };
                }
            }
            let input = Array::from_slice(&data, &[rows as i32, (width + 3) as i32])
                .try_index_device((.., 1..width as i32 + 1), &stream)
                .unwrap()
                .expand_dims(1, &stream)
                .unwrap();
            let config = CustomKernelConfig::new()
                .with_template_arg_int("WIDTH", width as i32)
                .with_template_arg_int("ROWS", rows as i32)
                .with_grid([rows as i32, 1, 1])
                .with_thread_group([32, 1, 1])
                .with_output_arg(input.shape(), Dtype::Float32);
            let expected = reference
                .apply_device([&input], &config, &stream)
                .unwrap()
                .pop()
                .unwrap()
                .into_evaluated()
                .unwrap();
            let actual = softmax_last(&input, &stream).unwrap().unwrap();
            assert_eq!(actual.shape(), input.shape());
            let actual = actual.into_evaluated().unwrap();
            for (index, (a, e)) in actual
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
                .enumerate()
            {
                assert!(
                    a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                    "width {width}, element {index}: {a} != {e}"
                );
            }
        }
    }

    #[test]
    fn native_pointwise_scalar_and_singleton_preserve_shape_and_rounding() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for shape in [&[][..], &[1], &[1, 1], &[2], &[]] {
            let count = shape.iter().map(|&extent| extent as usize).product();
            let input = Array::from_slice(&vec![-2.0_f32; count], shape);
            for (operation, expected) in [
                (Pointwise::Sigmoid, 0.11920292_f32),
                (Pointwise::Silu, -0.23840584_f32),
            ] {
                let output = f32_pointwise(&input, operation, &stream).unwrap().unwrap();
                assert_eq!(output.shape(), shape);
                assert!(output
                    .into_evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .iter()
                    .all(|&value| value == expected));
            }
        }
    }

    #[test]
    fn native_sigmoid_and_coefficients_match_independent_rounding() {
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let fixture = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/f32_sigmoid.safetensors"
            ),
            &weights,
        )
        .unwrap();
        let scores = f32_pointwise(&fixture["input"], Pointwise::Sigmoid, &stream)
            .unwrap()
            .unwrap();
        let selected =
            safemlx::ops::indexing::take_along_axis(&scores, &fixture["ids"], -1, &stream).unwrap();
        let denominator = crate::backend::nn::normalization::f32_sum_last(&selected, &stream)
            .unwrap()
            .unwrap();
        let coefficients = selected
            .divide(denominator, &stream)
            .unwrap()
            .multiply(Array::from_f32(2.5), &stream)
            .unwrap()
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap();
        for (name, values) in [("scores", scores), ("coefficients", coefficients)] {
            let output = values
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let expected = fixture[name]
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
            assert_eq!(differences, 0, "{name}");
        }
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
