//! Native realization of reproducible weighted-normalization arithmetic.
use safemlx::{error::Exception, Array, Stream};

/// Learned RMS normalization with an input-precision boundary before the gain.
/// Keep the same rounding order as the native low-precision operator, using
/// the shared float32 reduction to make half-way cases reproducible on Metal.
pub(crate) fn input_precision_rms(
    input: &Array,
    weight: &Array,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dtype() == weight.dtype()
        && matches!(
            input.dtype(),
            safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16
        )
        && input
            .shape()
            .last()
            .is_some_and(|width| weight.shape() == [*width])
    {
        let wide = input.as_dtype(safemlx::Dtype::Float32, stream)?;
        if let Some(normalized) = f32_weightless_rms(&wide, epsilon, stream)? {
            return normalized
                .as_dtype(input.dtype(), stream)?
                .multiply(weight, stream);
        }
    }
    safemlx::fast::rms_norm(input, weight, epsilon, stream)
}

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
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, epsilon, stream);
        Ok(None)
    }
}

/// Fuses ordered reduction, normalization and optional learned gain.
/// Independent cascade chains and gain application run in parallel per row.
/// Group and offset semantics come from the neutral normalization specification.
pub(crate) fn f32_grouped_rms(
    input: &Array,
    weight: Option<&Array>,
    groups: i32,
    epsilon: f32,
    offset: Option<f32>,
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
            || input.shape().is_empty()
            || groups <= 0
            || input.dim(-1) % groups != 0
            || weight.is_some_and(|weight| {
                weight.dtype() != Dtype::Float32 || weight.shape() != [input.dim(-1)]
            })
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let width = input.dim(-1) / groups;
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("normalization row count overflow"))?;
        let squared = input.square(stream)?;
        let extent = rows
            .checked_mul(32)
            .ok_or_else(|| Exception::custom("normalization grid extent overflow"))?;
        let scalars = Array::from_slice(&[epsilon, offset.unwrap_or(0.0)], &[2]);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("GROUPS", groups)
            .with_template_arg_bool("WEIGHTED", weight.is_some())
            .with_template_arg_bool("OFFSET", offset.is_some())
            .with_grid([extent, 1, 1])
            .with_thread_group([32, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_grouped_rms",
                    ["input", "squared", "weight", "scalars"],
                    ["output"],
                    r#"
                    #pragma clang fp contract(off)
                    uint row = threadgroup_position_in_grid.x;
                    uint lane = thread_position_in_threadgroup.x;
                    size_t base = size_t(row) * WIDTH;
                    float sum = parallel_cascade_sum_f32(squared + base, WIDTH, lane);
                    float variance = sum / float(WIDTH);
                    float inverse = 1.0f / metal::precise::sqrt(variance + scalars[0]);
                    size_t gain_base = size_t(row % GROUPS) * WIDTH;
                    for (uint i=lane; i<WIDTH; i+=32) {
                        // Keep the F32 product boundary before applying gain.
                        float normalized = input[base+i] * inverse;
                        if (WEIGHTED) {
                            float gain = OFFSET
                                ? weight[gain_base+i] + scalars[1]
                                : weight[gain_base+i];
                            output[base+i] = normalized * gain;
                        } else {
                            output[base+i] = normalized;
                        }
                    }
                    "#,
                    include_str!("rms_cascade.metal"),
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("normalization kernel initialized")
                .apply_device(
                    [input, &squared, weight.unwrap_or(input), &scalars],
                    &config,
                    stream,
                )
        })?;
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, groups, epsilon, offset, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype};
    #[test]
    fn fused_grouped_rms_handles_strided_rows_groups_and_reduction_tails() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for width in [17, 72, 1152] {
            let dimensions = width * 3;
            let values = (0..4 * dimensions)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
                .collect::<Vec<_>>();
            let weights = (0..dimensions)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.0625)
                .collect::<Vec<_>>();
            let input = Array::from_slice(&values, &[2, dimensions as i32, 2])
                .transpose_axes(&[0, 2, 1], &stream)
                .unwrap();
            let gain = Array::from_slice(&weights, &[dimensions as i32]);
            let result = f32_grouped_rms(&input, Some(&gain), 3, 0.125, Some(1.0), &stream)
                .unwrap()
                .unwrap()
                .into_evaluated()
                .unwrap();
            let actual = result.as_slice::<f32>();
            for batch in 0..2 {
                for token in 0..2 {
                    for group in 0..3 {
                        let row = (0..width)
                            .map(|column| {
                                values[(batch * dimensions + group * width + column) * 2 + token]
                            })
                            .collect::<Vec<_>>();
                        let mean =
                            row.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / width as f64;
                        for column in 0..width {
                            let expected = row[column] as f64 / (mean + 0.125).sqrt()
                                * (1.0 + weights[group * width + column] as f64);
                            let observed =
                                actual[(batch * 2 + token) * dimensions + group * width + column];
                            assert!((observed as f64 - expected).abs() < 2e-6,
                                "width {width} batch {batch} token {token} group {group} column {column}: {observed} vs {expected}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn parallel_cascade_matches_cpu_addition_order_at_carries_and_tails() {
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let gpu = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        // Cover partial vectors/chains and carries into all three upper levels.
        for width in [1, 7, 17, 255, 257, 4095, 4097, 65537] {
            let values = (0..3 * width)
                .map(|i| {
                    let signed = ((i * 37 % 103) as f32 - 51.0) / 13.0;
                    signed * 2.0_f32.powi((i % 19) as i32 - 9)
                })
                .collect::<Vec<_>>();
            let input = Array::from_slice(&values, &[3, width as i32]);
            let expected = input
                .sum_axes(&[-1], true, &cpu)
                .unwrap()
                .into_evaluated()
                .unwrap();
            use safemlx::fast::{CustomKernelConfig, MetalKernel};
            let kernel = MetalKernel::new(
                "parallel_cascade_parity",
                ["input"],
                ["output"],
                "uint row = threadgroup_position_in_grid.x; uint lane = thread_position_in_threadgroup.x; float sum = parallel_cascade_sum_f32(input + size_t(row) * WIDTH, WIDTH, lane); if (lane == 0) output[row] = sum;",
                include_str!("rms_cascade.metal"),
                true,
                false,
            ).unwrap();
            let config = CustomKernelConfig::new()
                .with_template_arg_int("WIDTH", width as i32)
                .with_grid([3 * 32, 1, 1])
                .with_thread_group([32, 1, 1])
                .with_output_arg([3, 1], Dtype::Float32);
            let actual = kernel
                .apply_device([&input], &config, &gpu)
                .unwrap()
                .pop()
                .unwrap()
                .into_evaluated()
                .unwrap();
            assert_eq!(
                actual
                    .as_slice::<f32>()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .as_slice::<f32>()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                "width {width}"
            );
        }
    }

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
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, stream);
        Ok(None)
    }
}
