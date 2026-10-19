//! Native realization of reproducible weighted-normalization arithmetic.
use safemlx::{error::Exception, Array, Stream};

#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: [f32; 4],
    square: [f32; 4],
}
impl Moments {
    fn add(&mut self, other: Self) {
        let count = self.count + other.count;
        let coefficient = if count == 0 {
            0.0
        } else {
            other.count as f32 / count as f32
        };
        for lane in 0..4 {
            let delta = other.mean[lane] - self.mean[lane];
            self.mean[lane] += coefficient * delta;
            self.square[lane] +=
                other.square[lane] + delta * delta * coefficient * self.count as f32;
        }
        self.count = count;
    }
}

fn rowwise_moments(values: &[f32], reduced: bool) -> (f32, f32) {
    let vector = if reduced { 8 } else { 4 };
    let vectors = values.len() / vector;
    let chunks = vectors.div_ceil(16);
    let depth = usize::BITS as usize - chunks.saturating_sub(1).leading_zeros() as usize;
    let mut stack = vec![Moments::default(); depth.max(1)];
    for chunk in 0..chunks {
        let count = 16.min(vectors - chunk * 16);
        for half in 0..(vector / 4) {
            let mut local = Moments {
                count,
                ..Moments::default()
            };
            for index in 0..count {
                let coefficient = 1.0_f32 / (index + 1) as f32;
                for lane in 0..4 {
                    let value = values[(chunk * 16 + index) * vector + half * 4 + lane];
                    let delta = value - local.mean[lane];
                    local.mean[lane] += delta * coefficient;
                    local.square[lane] += delta * (value - local.mean[lane]);
                }
            }
            stack[0].add(local);
        }
        let mut mask = chunk + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let prior = stack[level - 1];
            stack[level].add(prior);
            stack[level - 1] = Moments::default();
            mask >>= 1;
        }
    }
    for level in 1..depth {
        let prior = stack[level];
        stack[0].add(prior);
    }
    let mut count = 0usize;
    let mut mean = 0.0_f32;
    let mut square = 0.0_f32;
    for &value in &values[vectors * vector..] {
        let delta = value - mean;
        count += 1;
        mean += delta / count as f32;
        square += delta * (value - mean);
    }
    let additional = vectors * vector / 4;
    for lane in 0..4 {
        let next = count + additional;
        let coefficient = if next == 0 {
            0.0
        } else {
            additional as f32 / next as f32
        };
        let delta = stack[0].mean[lane] - mean;
        mean = coefficient.mul_add(delta, mean);
        square += (delta * delta * coefficient).mul_add(count as f32, stack[0].square[lane]);
        count = next;
    }
    (mean, square / values.len() as f32)
}

/// Bounded FP32 moments and affine rounding for reference-ordered LayerNorm.
pub(crate) fn cascaded_layer_norm(
    input: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    use safemlx::{Device, DeviceType, Dtype};
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
    if !floating(input)
        || [weight, bias]
            .into_iter()
            .flatten()
            .any(|value| !floating(value))
    {
        return Err(Exception::custom(
            "cascaded layer normalization requires floating representations",
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
            "invalid cascaded layer normalization parameters",
        ));
    }
    if stream.get_device()?.get_type()? != DeviceType::Cpu {
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let host = input.copy(&cpu)?;
        let weight = weight.map(|value| value.copy(&cpu)).transpose()?;
        let bias = bias.map(|value| value.copy(&cpu)).transpose()?;
        return cascaded_layer_norm(&host, weight.as_ref(), bias.as_ref(), epsilon, &cpu)?
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
        let (mean, variance) = rowwise_moments(row, reduced);
        let inverse = 1.0_f32 / (variance + epsilon).sqrt();
        let shift = -inverse * mean;
        for (column, &value) in row.iter().enumerate() {
            let normalized = if reduced {
                value * inverse + shift
            } else {
                (value - mean) * inverse
            };
            output.push(
                normalized * weight.map_or(1.0, |w| w[column]) + bias.map_or(0.0, |b| b[column]),
            );
        }
    }
    Array::from_slice(&output, input.shape()).as_dtype(input.dtype(), stream)
}

fn channel_sum_squares(values: &[f32], parallel: usize) -> f32 {
    let steps = values.len() / parallel;
    let log = usize::BITS as usize - steps.saturating_sub(1).leading_zeros() as usize;
    let power = 4.max(log / 4);
    let chunk = 1usize << power;
    let mut sums = [[0.0_f32; 4]; 4];
    let mut index = 0;
    while index + chunk <= steps {
        let end = index + chunk;
        while index < end {
            for p in 0..parallel {
                let value = values[index * parallel + p];
                sums[0][p] += value * value;
            }
            index += 1;
        }
        for level in 1..4 {
            for p in 0..parallel {
                sums[level][p] += sums[level - 1][p];
                sums[level - 1][p] = 0.0;
            }
            if index & ((chunk - 1) << (level * power)) != 0 {
                break;
            }
        }
    }
    while index < steps {
        for p in 0..parallel {
            let value = values[index * parallel + p];
            sums[0][p] += value * value;
        }
        index += 1;
    }
    for level in 1..4 {
        for p in 0..parallel {
            sums[0][p] += sums[level][p];
        }
    }
    for value in &values[steps * parallel..] {
        sums[0][0] += value * value;
    }
    for p in 1..parallel {
        sums[0][0] += sums[0][p];
    }
    sums[0][0]
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
            let parallel = if row % time < (time / 16) * 16 { 1 } else { 4 };
            let variance = channel_sum_squares(values, parallel) / width as f32;
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
    let dtype = input.dtype();
    let wide = if matches!(dtype, safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16) {
        input.as_dtype(safemlx::Dtype::Float32, stream)?
    } else {
        input.clone()
    };
    let normalized = match f32_weightless_rms(&wide, epsilon, stream)? {
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
    if matches!(
        input.dtype(),
        safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16
    ) && input
        .shape()
        .last()
        .is_some_and(|width| weight.shape() == [*width])
    {
        return weightless_rms(input, epsilon, stream)?.multiply(weight, stream);
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
fn contiguous_sum_squares(values: &[f32]) -> f32 {
    let vectors = values.len() / 4;
    let steps = vectors / 4;
    let log = usize::BITS as usize - steps.saturating_sub(1).leading_zeros() as usize;
    let power = 4.max(log / 4);
    let chunk = 1usize << power;
    let mut sums = [[[0.0_f32; 4]; 4]; 4];
    let mut index = 0;
    while index + chunk <= steps {
        let end = index + chunk;
        while index < end {
            for p in 0..4 {
                for lane in 0..4 {
                    let value = values[(index * 4 + p) * 4 + lane];
                    sums[0][p][lane] += value * value;
                }
            }
            index += 1;
        }
        for level in 1..4 {
            for p in 0..4 {
                for lane in 0..4 {
                    sums[level][p][lane] += sums[level - 1][p][lane];
                    sums[level - 1][p][lane] = 0.0;
                }
            }
            if index & ((chunk - 1) << (level * power)) != 0 {
                break;
            }
        }
    }
    while index < steps {
        for p in 0..4 {
            for lane in 0..4 {
                let value = values[(index * 4 + p) * 4 + lane];
                sums[0][p][lane] += value * value;
            }
        }
        index += 1;
    }
    for level in 1..4 {
        for p in 0..4 {
            for lane in 0..4 {
                sums[0][p][lane] += sums[level][p][lane];
            }
        }
    }
    for vector in steps * 4..vectors {
        for lane in 0..4 {
            let value = values[vector * 4 + lane];
            sums[0][0][lane] += value * value;
        }
    }
    for p in 1..4 {
        for lane in 0..4 {
            sums[0][0][lane] += sums[0][p][lane];
        }
    }
    let mut sum = 0.0_f32;
    for &value in &values[vectors * 4..] {
        sum += value * value;
    }
    for lane in 0..4 {
        sum += sums[0][0][lane];
    }
    sum
}

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
        let variance = contiguous_sum_squares(row) / width as f32;
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
