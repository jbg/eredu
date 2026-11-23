//! Native realization of reproducible weighted-normalization arithmetic.
use safemlx::{error::Exception, Array, Dtype, Stream};

/// Portable FP32 reductions using exact row order.
pub(crate) fn ordered_layer_norm(
    input: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    epsilon: f32,
    arithmetic: eredu_nn::LayerNormalizationArithmetic,
    stream: &Stream,
) -> Result<Array, Exception> {
    use eredu_nn::LayerNormalizationArithmetic;
    use safemlx::{Device, DeviceType, Dtype};
    let blocked = arithmetic == LayerNormalizationArithmetic::BlockedFloat32;
    let width = input
        .shape()
        .last()
        .copied()
        .filter(|&n| n > 0)
        .ok_or_else(|| Exception::custom("layer normalization requires a positive final axis"))?;
    let floating = |value: &Array| {
        matches!(
            value.dtype(),
            Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 | Dtype::Float16
        )
    };
    if (blocked && width % 32 != 0)
        || !floating(input)
        || [weight, bias]
            .into_iter()
            .flatten()
            .any(|value| !floating(value))
    {
        return Err(Exception::custom(
            "ordered layer normalization requires floating representations",
        ));
    }
    if !epsilon.is_finite()
        || epsilon <= 0.0
        || [weight, bias]
            .into_iter()
            .flatten()
            .any(|x| x.shape() != [width])
    {
        return Err(Exception::custom(
            "invalid ordered layer normalization parameters",
        ));
    }
    if stream.get_device()?.get_type()? != DeviceType::Cpu {
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let host = input.copy(&cpu)?;
        let weight = weight.map(|value| value.copy(&cpu)).transpose()?;
        let bias = bias.map(|value| value.copy(&cpu)).transpose()?;
        return ordered_layer_norm(
            &host,
            weight.as_ref(),
            bias.as_ref(),
            epsilon,
            arithmetic,
            &cpu,
        )?
        .copy(stream);
    }
    let wide = input
        .as_dtype(Dtype::Float32, stream)?
        .contiguous(false, stream)?
        .into_evaluated()?;
    let convert = |value: &Array| {
        value
            .as_dtype(Dtype::Float32, stream)?
            .contiguous(false, stream)?
            .into_evaluated()
    };
    let weight = weight.map(convert).transpose()?;
    let bias = bias.map(convert).transpose()?;
    let weight = weight.as_ref().map(|value| value.as_slice::<f32>());
    let bias = bias.as_ref().map(|value| value.as_slice::<f32>());
    let reduced = matches!(input.dtype(), Dtype::Bfloat16 | Dtype::Float16);
    let mut output = Vec::with_capacity(input.size());
    for row in wide.as_slice::<f32>().chunks_exact(width as usize) {
        if blocked {
            let start = output.len();
            output.extend_from_slice(row);
            let values = &mut output[start..];
            eredu_nn::arithmetic::normalize_blocked_f32(values, epsilon);
            for (column, value) in values.iter_mut().enumerate() {
                if let Some(weight) = weight {
                    *value *= weight[column];
                }
                if let Some(bias) = bias {
                    *value += bias[column];
                }
            }
            continue;
        }
        let start = output.len();
        output.extend_from_slice(row);
        eredu_nn::arithmetic::normalize_cascaded_f32(
            &mut output[start..],
            weight,
            bias,
            epsilon,
            reduced,
        );
    }
    Array::from_slice(&output, input.shape()).as_dtype(input.dtype(), stream)
}

/// Channel-major RMS traversal without depending on the realized input strides.
pub(crate) fn channel_major_rms(
    input: &Array,
    weight: Option<&Array>,
    offset: Option<f32>,
    epsilon: f32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    use safemlx::{DeviceType, Dtype};
    if input.ndim() != 3 || input.shape().iter().any(|&n| n <= 0) {
        return Err(Exception::custom(
            "channel-major RMS requires [batch, time, channel]",
        ));
    }
    let time = input.dim(1) as usize;
    let width = input.dim(2) as usize;
    if time == 1 || width == 1 {
        return Ok(None);
    }
    let wide = input
        .as_dtype(Dtype::Float32, stream)?
        .contiguous(false, stream)?;
    let scale = weight
        .map(|value| {
            value
                .as_dtype(Dtype::Float32, stream)?
                .contiguous(false, stream)
        })
        .transpose()?;
    let offset = offset.unwrap_or(0.0);
    if stream.get_device()?.get_type()? == DeviceType::Cpu {
        let wide = wide.into_evaluated()?;
        let values = wide.as_slice::<f32>();
        let scale = scale.map(Array::into_evaluated).transpose()?;
        let scale = scale.as_ref().map(|value| value.as_slice::<f32>());
        let mut output = Vec::with_capacity(values.len());
        for (row, values) in values.chunks_exact(width).enumerate() {
            let variance =
                eredu_nn::arithmetic::channel_major_rms_sum(values, time, row) / width as f32;
            let inverse = 1.0 / (variance + epsilon).sqrt();
            for (column, value) in values.iter().enumerate() {
                let normalized = value * inverse;
                output.push(match scale {
                    Some(gain) => normalized * (gain[column] + offset),
                    None => normalized,
                });
            }
        }
        return Array::from_slice(&output, input.shape())
            .as_dtype(input.dtype(), stream)
            .map(Some);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if stream.get_device()?.get_type()? == DeviceType::Gpu {
        use safemlx::fast::{CustomKernelConfig, MetalKernel};
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let rows = i32::try_from(input.size() / width)
            .map_err(|_| Exception::custom("RMS row count overflow"))?;
        let square = wide.square(stream)?;
        let constants = Array::from_slice(&[epsilon, offset], &[2]);
        let has_weight = scale.is_some();
        let scale = scale.unwrap_or_else(|| Array::from_slice(&[1f32], &[1]));
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width as i32)
            .with_template_arg_int("TIME", time as i32)
            .with_template_arg_int("HAS_WEIGHT", i32::from(has_weight))
            .with_grid([
                rows.checked_mul(32)
                    .ok_or_else(|| Exception::custom("RMS grid overflow"))?,
                1,
                1,
            ])
            .with_thread_group([32, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut result = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new("channel_major_rms", ["input", "square", "weight", "constants"], ["output"],
                    concat!("uint row=threadgroup_position_in_grid.x; uint lane=thread_index_in_simdgroup;",
                        "size_t base=size_t(row)*WIDTH; threadgroup float inverse;",
                        "if(lane==0) { float sum=(row%TIME < (TIME/16)*16) ? cascade_sum_f32<1,1>(square+base,WIDTH) : cascade_sum_f32<1,4>(square+base,WIDTH);",
                        " inverse=1.0f/metal::precise::sqrt(sum/float(WIDTH)+constants[0]); }",
                        "threadgroup_barrier(mem_flags::mem_threadgroup);",
                        "for(uint col=lane;col<WIDTH;col+=32) { float value=input[base+col]*inverse;",
                        " output[base+col]=HAS_WEIGHT ? value*(weight[col]+constants[1]) : value; }"),
                    include_str!("rms_cascade.metal"), true, false)?);
            }
            cell.borrow().as_ref().expect("RMS kernel initialized").apply_device([&wide, &square, &scale, &constants], &config, stream)
        })?;
        return result
            .pop()
            .expect("one RMS output")
            .as_dtype(input.dtype(), stream)
            .map(Some);
    }
    // A backend without the native kernel retains the same bounded activation
    // operation through ordinary device copies; no checkpoint parameter bank is
    // expanded or made resident by this fallback.
    let cpu = Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
    let host_input = input.copy(&cpu)?;
    let host_weight = weight.map(|value| value.copy(&cpu)).transpose()?;
    channel_major_rms(
        &host_input,
        host_weight.as_ref(),
        Some(offset),
        epsilon,
        &cpu,
    )?
    .map(|value| value.copy(stream))
    .transpose()
}

/// Weightless RMS with FP32 reduction and one final low-precision cast.
pub(crate) fn weightless_rms(
    input: &Array,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if let Some(output) = fused_bf16_rms(input, None, epsilon, 4, stream)? {
        return Ok(output);
    }
    let dtype = input.dtype();
    let wide = if matches!(dtype, safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16) {
        input.as_dtype(safemlx::Dtype::Float32, stream)?
    } else {
        input.clone()
    };
    let normalized = match f32_weightless_rms(&wide, epsilon, 4, stream)? {
        Some(output) => output,
        None => {
            let variance = wide.square(stream)?.mean_axis(-1, true, stream)?;
            let denominator = variance
                .add(Array::from_f32(epsilon), stream)?
                .rsqrt(stream)?;
            wide.multiply(denominator, stream)?
        }
    };
    normalized.as_dtype(dtype, stream)
}

/// Learned RMS normalization with an input-precision boundary before the gain.
/// Keep the same rounding order as the native low-precision operator, using
/// the shared float32 reduction to make half-way cases reproducible on Metal.
pub(crate) fn input_precision_rms(
    input: &Array,
    weight: &Array,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dtype() == Dtype::Float32
        && weight.dtype() == Dtype::Float32
        && input.size() != 0
        && super::ordered_f32::supported(stream)?
    {
        return wide_square_rms(input, weight, epsilon, stream);
    }
    if matches!(
        input.dtype(),
        safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16
    ) && input
        .shape()
        .last()
        .is_some_and(|width| weight.shape() == [*width])
    {
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if let Some(output) = fused_bf16_rms(input, Some(weight), epsilon, 4, stream)? {
            return Ok(output);
        }
        return weightless_rms(input, epsilon, stream)?.multiply(weight, stream);
    }
    safemlx::fast::rms_norm(input, weight, epsilon, stream)
}

fn wide_square_rms(
    input: &Array,
    weight: &Array,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    use safemlx::transforms::eval;
    let width = input.dim(-1);
    if width <= 0 || weight.shape() != [width] {
        return Err(Exception::custom("invalid ordered RMS width"));
    }
    let rows = input.size() / width as usize;
    if crate::backend::nn::native_quantization::uses_metal_device(stream.get_device()?.get_type()?)
    {
        use safemlx::fast::{CustomKernelConfig, MetalKernel};
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let rows = i32::try_from(rows).map_err(|_| Exception::custom("RMS row count overflows"))?;
        let epsilon = Array::from_slice(&[epsilon], &[1]);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_grid([rows, 1, 1])
            .with_thread_group([64, 1, 1])
            .with_output_arg(input.shape(), Dtype::Float32);
        return KERNEL.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "wide_square_rms",
                    ["input", "gain", "epsilon"],
                    ["out"],
                    r#"
                    uint row = thread_position_in_grid.x;
                    size_t offset = size_t(row) * WIDTH;
                    float high = 0.0f;
                    float low = 0.0f;
                    for (uint i = 0; i < WIDTH; ++i) {
                        float x = input[offset + i];
                        float y = x * x;
                        float s = high + y;
                        float v = s - high;
                        float e = (high - (s - v)) + (y - v) + low;
                        high = s + e;
                        low = e - (high - s);
                    }
                    float mean = metal::precise::divide(high, float(WIDTH));
                    mean += metal::precise::divide(fma(-mean, float(WIDTH), high) + low, float(WIDTH));
                    float inverse = metal::precise::divide(1.0f, metal::precise::sqrt(mean + epsilon[0]));
                    for (uint i = 0; i < WIDTH; ++i) out[offset + i] = (input[offset + i] * inverse) * gain[i];
                    "#,
                    "#pragma clang fp contract(off)\n",
                    true,
                    false,
                )?);
            }
            cell.borrow().as_ref().expect("RMS kernel initialized")
                .apply_one_device([input, weight, &epsilon], &config, stream)
        });
    }
    let contiguous = input.contiguous(false, stream)?;
    eval([&contiguous, weight])?;
    let x = contiguous.evaluated()?;
    let gain = weight
        .evaluated()?
        .try_to_vec::<f32>()
        .map_err(|error| Exception::custom(error.to_string()))?;
    let x = x
        .try_as_slice::<f32>()
        .map_err(|error| Exception::custom(error.to_string()))?;
    let mut output = Vec::with_capacity(input.size());
    for row in x.chunks_exact(width as usize) {
        let sum: f64 = row.iter().map(|&x| f64::from(x * x)).sum();
        let mean = (sum / f64::from(width)) as f32;
        let inverse = (mean + epsilon).sqrt().recip();
        output.extend(row.iter().zip(&gain).map(|(&x, &w)| (x * inverse) * w));
    }
    Array::from_slice(&output, input.shape()).copy(stream)
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn fused_bf16_rms(
    input: &Array,
    gain: Option<&Array>,
    epsilon: f32,
    lanes: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    use safemlx::{
        fast::{CustomKernelConfig, MetalKernel},
        DeviceType, Dtype,
    };
    use std::cell::RefCell;

    thread_local! {
        static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    }
    let Some(&width) = input.shape().last() else {
        return Ok(None);
    };
    if input.dtype() != Dtype::Bfloat16
        || gain.is_some_and(|gain| gain.dtype() != Dtype::Bfloat16 || gain.shape() != [width])
        || input.size() == 0
        || stream.get_device()?.get_type()? != DeviceType::Gpu
    {
        return Ok(None);
    }
    let rows = i32::try_from(input.size() / width as usize)
        .map_err(|_| Exception::custom("normalization row count overflow"))?;
    // This geometry has four complete 16-step chunks in each of its 32 chains.
    let chunks = if lanes == 8 && width == 2048 { 4 } else { 1 };
    let epsilon = Array::from_slice(&[epsilon], &[1]);
    let config = CustomKernelConfig::new()
        .with_template_arg_int("WIDTH", width)
        .with_template_arg_int("LANES", lanes)
        .with_template_arg_int("WEIGHTED", i32::from(gain.is_some()))
        .with_template_arg_int("CHUNKS", chunks)
        .with_grid([32, rows, chunks])
        .with_thread_group([32, 1, chunks])
        .with_output_arg(input.shape(), Dtype::Bfloat16);
    KERNEL.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(MetalKernel::new(
                "fused_bf16_rms",
                ["input", "gain", "epsilon"],
                ["output"],
                include_str!("rms_bf16.metal"),
                concat!(
                    include_str!("rms_cascade.metal"),
                    r#"
                    struct SquaredBfloat {
                        const device bfloat16_t* values;
                        float operator[](size_t index) const thread {
                            float value = float(values[index]);
                            return value * value;
                        }
                    };
                    "#
                ),
                true,
                false,
            )?);
        }
        cell.borrow()
            .as_ref()
            .expect("fused RMS kernel initialized")
            .apply_one_device([input, gain.unwrap_or(input), &epsilon], &config, stream)
            .map(Some)
    })
}

fn f32_weightless_rms(
    input: &Array,
    epsilon: f32,
    lanes: i32,
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
        let grid = if lanes == 4 {
            [32, rows, 1]
        } else {
            [rows, 1, 1]
        };
        let squared = input.square(stream)?;
        let epsilon = Array::from_slice(&[epsilon], &[1]);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("LANES", lanes)
            .with_grid(grid)
            .with_thread_group([32, 1, 1])
            .with_template_arg_int("ROWS", rows)
            .with_output_arg(input.shape(), Dtype::Float32);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_weightless_rms",
                    ["input", "squared", "epsilon"],
                    ["output"],
                    r#"
                        if (LANES == 4) {
                            uint row = threadgroup_position_in_grid.y;
                            uint lane = thread_position_in_threadgroup.x;
                            size_t base = size_t(row) * WIDTH;
                            float sum = parallel_cascade_sum_f32(squared + base, WIDTH, lane);
                            float variance = sum / float(WIDTH);
                            float inverse = 1.0f / metal::precise::sqrt(variance + epsilon[0]);
                            for (uint i = lane; i < WIDTH; i += 32) {
                                output[base + i] = input[base + i] * inverse;
                            }
                        } else {
                            uint row = thread_position_in_grid.x;
                            if (row >= ROWS) return;
                            size_t base = size_t(row) * WIDTH;
                            float variance =
                                cascade_sum_f32<LANES, 4>(squared + base, WIDTH) / float(WIDTH);
                            float inverse = 1.0f / metal::precise::sqrt(variance + epsilon[0]);
                            for (uint i = 0; i < WIDTH; ++i) {
                                output[base + i] = input[base + i] * inverse;
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
                .apply_device([input, &squared, &epsilon], &config, stream)
        })?;
        Ok(output.pop())
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, epsilon, lanes, stream);
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

// Four vector lanes and four independent accumulation chains, with the same
// four-level carry and scalar-tail merge as rms_cascade.metal.
/// Explicit contiguous FP32 reduction; native backend normalization remains a
/// separate policy. Metal shares the existing grouped cascade with one group.
pub(crate) fn cascaded_rms(
    input: &Array,
    weight: Option<&Array>,
    offset: Option<f32>,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    use safemlx::{Device, DeviceType, Dtype};
    let width = input
        .shape()
        .last()
        .copied()
        .filter(|&n| n > 0)
        .ok_or_else(|| Exception::custom("cascaded RMS requires a positive feature width"))?;
    if weight.is_some_and(|value| value.shape() != [width]) {
        return Err(Exception::custom("cascaded RMS gain geometry differs"));
    }
    if stream.get_device()?.get_type()? != DeviceType::Cpu {
        if let Some(value) = f32_grouped_rms(input, weight, 1, epsilon, offset, stream)? {
            return Ok(value);
        }
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let host = input.copy(&cpu)?;
        let weight = weight.map(|value| value.copy(&cpu)).transpose()?;
        return cascaded_rms(&host, weight.as_ref(), offset, epsilon, &cpu)?.copy(stream);
    }
    let wide = input
        .as_dtype(Dtype::Float32, stream)?
        .contiguous(false, stream)?
        .into_evaluated()?;
    let weight = weight
        .map(|value| {
            value
                .as_dtype(Dtype::Float32, stream)?
                .contiguous(false, stream)?
                .into_evaluated()
        })
        .transpose()?;
    let weight = weight.as_ref().map(|value| value.as_slice::<f32>());
    let mut output = Vec::with_capacity(input.size());
    for row in wide.as_slice::<f32>().chunks_exact(width as usize) {
        let variance = eredu_nn::arithmetic::cascaded_sum_squares::<4>(row) / width as f32;
        let inverse = 1.0_f32 / (variance + epsilon).sqrt();
        for (column, &value) in row.iter().enumerate() {
            let normalized = value * inverse;
            output.push(match weight {
                Some(scale) => normalized * (scale[column] + offset.unwrap_or(0.0)),
                None => normalized,
            });
        }
    }
    Array::from_slice(&output, input.shape()).as_dtype(input.dtype(), stream)
}

pub(crate) fn eight_lane_rms(
    input: &Array,
    weight: Option<&Array>,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    use safemlx::{Device, DeviceType, Dtype};

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if let Some(output) = fused_bf16_rms(input, weight, epsilon, 8, stream)? {
        return Ok(output);
    }

    let wide = input.as_dtype(Dtype::Float32, stream)?;
    let normalized = if let Some(output) = f32_weightless_rms(&wide, epsilon, 8, stream)? {
        output
    } else {
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let values = wide.copy(&cpu)?.contiguous(false, &cpu)?.into_evaluated()?;
        let width = input.dim(-1) as usize;
        let mut output = Vec::with_capacity(input.size());
        for row in values.as_slice::<f32>().chunks_exact(width) {
            let variance = eredu_nn::arithmetic::cascaded_sum_squares::<8>(row) / width as f32;
            let inverse = 1.0_f32 / (variance + epsilon).sqrt();
            output.extend(row.iter().map(|value| value * inverse));
        }
        Array::from_slice(&output, input.shape())
    };
    let normalized = normalized.as_dtype(input.dtype(), stream)?;
    match weight {
        Some(gain) => normalized.multiply(gain, stream),
        None => Ok(normalized),
    }
}
