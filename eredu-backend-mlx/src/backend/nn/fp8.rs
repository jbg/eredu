//! Architecture-neutral block-scaled FP8 projections.
//!
//! Checkpoints store E4M3 bytes together with one inverse scale per 128x128
//! weight block. Conventional ungrouped and grouped projections dynamically
//! quantize each 128-value activation block to E4M3 using the checkpoint's
//! declared block-scaled FP8 execution scheme. GPU operations consume both
//! packed representations directly, including rank-3 grouped banks, without
//! expanding a complete weight bank. CPU execution uses a deliberately slow
//! dequantized reference path for correctness tests and functional fallback.

pub(crate) mod kernel;
pub(crate) mod original;

use std::cell::RefCell;

use super::grouping::grouped_matmul;
#[cfg(feature = "cuda")]
use safemlx::fast::CudaKernel;
use safemlx::fast::CustomKernelConfig;
#[cfg(not(feature = "cuda"))]
use safemlx::fast::MetalKernel;
use safemlx::{
    Array, DeviceType, Dtype, Stream,
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{TryIndexOp, take},
        matmul,
    },
    transforms::eval,
};

#[cfg(not(feature = "cuda"))]
thread_local! {
    static SEGMENTED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SEGMENTED_TRANSPOSED_LINEAR_KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

#[cfg(feature = "cuda")]
thread_local! {
    static ACT_QUANT_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static GROUPED_LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static SEGMENTED_LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
    static SEGMENTED_TRANSPOSED_LINEAR_KERNEL: RefCell<Option<CudaKernel>> = const { RefCell::new(None) };
}

#[cfg(feature = "cuda")]
const OUT_TILE: i32 = 16;
#[cfg(feature = "cuda")]
const REDUCTION_TILE: i32 = 16;
const SCALE_BLOCK: i32 = 128;
static E8M0_SCALE_TABLE: [f32; 256] = {
    let mut values = [0.0; 256];
    let mut exponent = 0;
    while exponent < 256 {
        values[exponent] = f32::from_bits((exponent as u32) << 23);
        exponent += 1;
    }
    values
};
const TILED_ROW_THRESHOLD: i32 = 8;

fn ceil_div(lhs: i32, rhs: i32) -> i32 {
    lhs / rhs + i32::from(lhs % rhs > 0)
}

/// Decodes native unsigned E8M0 scale bytes without expanding FP8 weights.
/// Float scale tensors are returned unchanged, allowing F16, BF16,
/// or F32 inverse scales and native unsigned scales to share every execution
/// kernel.
pub fn decode_scale(scale: &Array, stream: &Stream) -> Result<Array, Exception> {
    match scale.dtype() {
        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 => Ok(scale.clone()),
        Dtype::Uint8 => {
            let table = &E8M0_SCALE_TABLE;
            take(
                Array::try_from_slice(table, &[256])?,
                scale.as_dtype(Dtype::Uint32, stream)?,
                stream,
            )
        }
        dtype => Err(original::invalid(format_args!(
            "block-FP8 scale must be Float16, Bfloat16, Float32, or native E8M0 bytes, got {dtype:?}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn linear_tiled_config(
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
) -> CustomKernelConfig {
    let out_grid = ceil_div(out_dim, OUT_TILE) * OUT_TILE;
    CustomKernelConfig::new()
        .with_template_arg_int("IN_DIM", in_dim)
        .with_template_arg_int("OUT_DIM", out_dim)
        .with_template_arg_int("OUT_TILE", OUT_TILE)
        .with_template_arg_int("REDUCTION_TILE", REDUCTION_TILE)
        .with_template_arg_int("SCALE_BLOCK", SCALE_BLOCK)
        .with_template_arg_int("SCALE_COLS", scale_cols)
        .with_grid([out_grid, rows * REDUCTION_TILE, 1])
        .with_thread_group([OUT_TILE, REDUCTION_TILE, 1])
        .with_output_arg([rows, out_dim], Dtype::Float32)
}

#[cfg(feature = "cuda")]
fn grouped_tiled_config(
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_out: i32,
    scale_cols: i32,
    row_width: i32,
) -> CustomKernelConfig {
    linear_tiled_config(routes, in_dim, out_dim, scale_cols)
        .with_template_arg_int("SCALE_OUT", scale_out)
        .with_template_arg_int("ROW_WIDTH", row_width)
        .with_template_arg_int("ROW_SCALES", ceil_div(row_width, SCALE_BLOCK))
}

fn activation_dtype(input: &Array) -> Result<Dtype, Exception> {
    let dtype = input.dtype();
    if !dtype.is_float() {
        return Err(original::invalid(format_args!(
            "block-FP8 activation input must be floating point, got {dtype:?}"
        )));
    }
    Ok(dtype)
}

fn restore_activation_dtype(
    output: Array,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array, Exception> {
    if dtype == Dtype::Float32 {
        Ok(output)
    } else {
        output.as_dtype(dtype, stream)
    }
}

fn is_cpu_stream(stream: &Stream) -> Result<bool, Exception> {
    Ok(stream.device_type()? == DeviceType::Cpu)
}

fn dequantize_grouped(weight: &Array, scale: &Array, stream: &Stream) -> Result<Array, Exception> {
    let groups = weight.dim(0);
    let out_dim = weight.dim(1);
    let in_dim = weight.dim(2);
    let scale = Array::repeat_axis::<f32>(decode_scale(scale, stream)?, 128, 1, stream)?;
    let scale = Array::repeat_axis::<f32>(scale, 128, 2, stream)?;
    weight.from_fp8(Dtype::Float32, stream)?.multiply(
        scale.try_index_device((..groups, ..out_dim, ..in_dim), stream)?,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn segmented_reference(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    group_stride: i32,
    row_offset: i32,
    output_dims: i32,
    transpose: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    let weight = dequantize(weight, scale, stream)?;
    let group_ids = group_ids.as_dtype(Dtype::Uint32, stream)?;
    eval([&group_ids])?;
    let evaluated = group_ids.evaluated()?;
    let mut outputs = Vec::with_capacity(input.dim(0) as usize);
    for (route, group) in evaluated.as_slice::<u32>().iter().copied().enumerate() {
        let start = group as i32 * group_stride + row_offset;
        let input_row = input.try_index_device(route as i32..route as i32 + 1, stream)?;
        let segment = weight.try_index_device(start..start + output_dims, stream)?;
        outputs.push(if transpose {
            matmul(&input_row, &segment, stream)?
        } else {
            matmul(&input_row, &segment.transpose(stream)?, stream)?
        });
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 0, stream)
}

struct QuantizedActivations {
    values: Array,
    scales: Array,
}

fn quantize_activations(
    input: &Array,
    rows: i32,
    in_dim: i32,
    stream: &Stream,
) -> Result<QuantizedActivations, Exception> {
    let scale_cols = ceil_div(in_dim, SCALE_BLOCK);
    if let Some(observer) = safemlx::OriginalScopeObserver::try_current()? {
        #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
        return Err(observer.capacity_error());
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if is_cpu_stream(stream)? {
            return Err(observer.capacity_error());
        }
    }
    if is_cpu_stream(stream)? {
        let input = input
            .as_dtype(Dtype::Float32, stream)?
            .reshape(&[rows, in_dim], stream)?;
        let mut values = Vec::with_capacity(scale_cols as usize);
        let mut scales = Vec::with_capacity(scale_cols as usize);
        for start in (0..in_dim).step_by(SCALE_BLOCK as usize) {
            let block =
                input.try_index_device((.., start..(start + SCALE_BLOCK).min(in_dim)), stream)?;
            let scale = safemlx::ops::maximum(
                block.abs(stream)?.max_axis(-1, true, stream)?,
                Array::try_from_f32(1.0e-4)?,
                stream,
            )?
            .divide(Array::try_from_f32(448.0)?, stream)?;
            values.push(block.divide(&scale, stream)?.to_fp8(stream)?);
            scales.push(scale);
        }
        return Ok(QuantizedActivations {
            values: concatenate_axis(&values, 1, stream)?,
            scales: concatenate_axis(&scales, 1, stream)?,
        });
    }
    #[cfg(feature = "cuda")]
    {
        let config = CustomKernelConfig::new()
            .with_template_arg_int("IN_DIM", in_dim)
            .with_template_arg_int("SCALE_COLS", scale_cols)
            .with_template_arg_int("SCALE_BLOCK", SCALE_BLOCK)
            .with_grid([rows * scale_cols * SCALE_BLOCK, 1, 1])
            .with_thread_group([SCALE_BLOCK, 1, 1])
            .with_output_arg([rows, in_dim], Dtype::Uint8)
            .with_output_arg([rows, scale_cols], Dtype::Float32);

        let mut outputs = ACT_QUANT_KERNEL.with(|cell| -> Result<_, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(activation_quantization_kernel_cuda()?);
            }
            cell.borrow()
                .as_ref()
                .expect("CUDA activation quantization kernel initialized")
                .apply_device([input], &config, stream)
        })?;

        let scales = outputs.pop().expect("activation scale output");
        let values = outputs.pop().expect("quantized activation output");
        return Ok(QuantizedActivations { values, scales });
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let [values, scales] = kernel::quantize(input, rows, in_dim, scale_cols, stream)?;
        Ok(QuantizedActivations { values, scales })
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Err(original::invalid(format_args!(
        "GPU FP8 requires a Metal or CUDA backend"
    )))
}

fn activation_reference(
    input: &Array,
    rows: i32,
    in_dim: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let quantized = quantize_activations(input, rows, in_dim, stream)?;
    dequantize_activations(&quantized, input.shape(), stream)
}

fn dequantize_activations(
    quantized: &QuantizedActivations,
    shape: &[i32],
    stream: &Stream,
) -> Result<Array, Exception> {
    let plan = eredu_nn::BlockFp8InputReconstructionPlan::new(shape)
        .map_err(original::reconstruction_error)?;
    plan.validate_operands(quantized.values.shape(), quantized.scales.shape())
        .map_err(original::reconstruction_error)?;
    eredu_nn::reconstruct_block_fp8_input(InputReconstruction {
        plan,
        quantized,
        stream,
    })
}

struct InputReconstruction<'a> {
    plan: eredu_nn::BlockFp8InputReconstructionPlan<'a>,
    quantized: &'a QuantizedActivations,
    stream: &'a Stream,
}
impl InputReconstruction<'_> {
    fn validate_sources(&self) -> Result<(), Exception> {
        self.plan
            .validate_operands(self.quantized.values.shape(), self.quantized.scales.shape())
            .map_err(original::reconstruction_error)?;
        if self.quantized.values.dtype() != Dtype::Uint8
            || self.quantized.scales.dtype() != Dtype::Float32
        {
            return Err(original::reconstruction_error(
                eredu_nn::ProjectionObservationError::Geometry,
            ));
        }
        Ok(())
    }
}
impl eredu_nn::RetainedGeneratedTensorFactory<Array, Exception> for InputReconstruction<'_> {
    fn program(&self) -> eredu_nn::GeneratedTensorProgram<'_> {
        eredu_nn::GeneratedTensorProgram::BlockFp8Input(self.plan)
    }
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(
            eredu_nn::GeneratedTensorSourceRole,
            &Array,
        ) -> Result<(), Exception>,
    ) -> Result<(), Exception> {
        self.validate_sources()?;
        retain(
            eredu_nn::GeneratedTensorSourceRole::CompactValues,
            &self.quantized.values,
        )?;
        retain(
            eredu_nn::GeneratedTensorSourceRole::BlockScales,
            &self.quantized.scales,
        )
    }
    fn generate(
        &mut self,
        retain: &mut dyn FnMut(&Array) -> Result<(), Exception>,
    ) -> Result<Array, Exception> {
        self.validate_sources()?;
        eredu_nn::reconstruct_block_fp8_input_retained(
            InputReconstruction {
                plan: self.plan,
                quantized: self.quantized,
                stream: self.stream,
            },
            retain,
        )
    }
}

impl eredu_nn::BlockFp8InputReconstructionMechanism for InputReconstruction<'_> {
    type Value = Array;
    type Error = Exception;
    fn expand_scales(&self) -> Result<Array, Exception> {
        self.quantized.scales.expand_dims(2, self.stream)
    }
    fn broadcast_scales(&self, expanded: &Array) -> Result<Array, Exception> {
        safemlx::ops::broadcast_to(
            expanded,
            &[self.plan.rows(), self.plan.scale_columns(), SCALE_BLOCK],
            self.stream,
        )
    }
    fn flatten_scales(&self, broadcasted: &Array) -> Result<Array, Exception> {
        broadcasted.reshape(&[self.plan.rows(), self.plan.padded_width()], self.stream)
    }
    fn decode_values(&self) -> Result<Array, Exception> {
        self.quantized.values.from_fp8(Dtype::Float32, self.stream)
    }
    fn trim_scales(&self, repeated: &Array) -> Result<Array, Exception> {
        repeated.try_index_device((.., ..self.plan.width()), self.stream)
    }
    fn multiply(&self, values: &Array, scales: &Array) -> Result<Array, Exception> {
        values.multiply(scales, self.stream)
    }
    fn restore_shape(&self, product: &Array) -> Result<Array, Exception> {
        product.reshape(self.plan.shape(), self.stream)
    }
}

#[cfg(feature = "cuda")]
fn activation_quantization_kernel_cuda() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_activation_quantization",
        ["input"],
        ["quantized", "activation_scale"],
        concat!(
            "uint32_t block = blockIdx.x;",
            "uint32_t lane = threadIdx.x;",
            "uint32_t row = block / SCALE_COLS;",
            "uint32_t scale_col = block % SCALE_COLS;",
            "uint32_t col = scale_col * SCALE_BLOCK + lane;",
            "bool valid = col < IN_DIM;",
            "float value = valid ? float(input[row * IN_DIM + col]) : 0.0f;",
            "__shared__ float maxima[SCALE_BLOCK];",
            "__shared__ float block_scale;",
            "maxima[lane] = fabsf(value);",
            "__syncthreads();",
            "for (uint32_t stride = SCALE_BLOCK / 2; stride > 0; stride /= 2) {",
            " if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);",
            " __syncthreads();",
            "}",
            "if (lane == 0) {",
            " block_scale = fmaxf(maxima[0], 1.0e-4f) / 448.0f;",
            " activation_scale[block] = block_scale;",
            "}",
            "__syncthreads();",
            "if (valid) quantized[row * IN_DIM + col] = float_to_fp8_e4m3(value / block_scale);"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

/// Expands one block-scaled E4M3 matrix or rank-3 matrix bank to floating point.
///
/// This is intended for algorithms that must absorb or slice a projection
/// matrix rather than apply it as a conventional linear operation. Callers
/// should keep the result transient.
pub fn dequantize(weight: &Array, scale: &Array, stream: &Stream) -> Result<Array, Exception> {
    dequantize_with_row_layout(weight, scale, eredu_nn::LinearRowLayout::Contiguous, stream)
}

/// Decodes architecture-declared independent row blocks without joining their tails.
pub fn dequantize_with_row_layout(
    weight: &Array,
    scale: &Array,
    layout: eredu_nn::LinearRowLayout,
    stream: &Stream,
) -> Result<Array, Exception> {
    if layout != eredu_nn::LinearRowLayout::Contiguous {
        if !(2..=3).contains(&weight.ndim())
            || weight.ndim() != scale.ndim()
            || weight.shape().iter().any(|n| *n <= 0)
        {
            return Err(Exception::custom(
                "invalid independently blocked FP8 geometry",
            ));
        }
        let rows = weight.dim(-2);
        let columns = weight.dim(-1);
        let groups = if weight.ndim() == 3 { weight.dim(0) } else { 1 };
        let width = layout
            .rows_per_partition(rows as usize)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let scale_rows = layout
            .scale_rows(rows as usize, SCALE_BLOCK as usize)
            .map_err(|error| Exception::custom(error.to_string()))?;
        if scale.dim(-2) as usize != scale_rows
            || scale.dim(-1) != ceil_div(columns, SCALE_BLOCK)
            || (weight.ndim() == 3 && scale.dim(0) != groups)
        {
            return Err(Exception::custom(
                "independent FP8 scale rows disagree with declared layout",
            ));
        }
        let count = (groups as usize)
            .checked_mul(layout.partitions())
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| Exception::custom("independent FP8 partition count overflows"))?;
        let weight_parts = weight.reshape(&[count, width as i32, columns], stream)?;
        let scale_parts = scale.reshape(
            &[
                count,
                (scale_rows / layout.partitions()) as i32,
                scale.dim(-1),
            ],
            stream,
        )?;
        return dequantize_grouped(&weight_parts, &scale_parts, stream)?
            .reshape(weight.shape(), stream);
    }
    if weight.ndim() == 3 && scale.ndim() == 3 {
        if weight.dim(0) <= 0
            || weight.dim(1) <= 0
            || weight.dim(2) <= 0
            || scale.dim(0) != weight.dim(0)
            || scale.dim(1) != ceil_div(weight.dim(1), SCALE_BLOCK)
            || scale.dim(2) != ceil_div(weight.dim(2), SCALE_BLOCK)
        {
            return Err(Exception::custom(
                "invalid grouped block-FP8 scale geometry",
            ));
        }
        return dequantize_grouped(weight, scale, stream);
    }
    if weight.ndim() != 2 || scale.ndim() != 2 {
        return Err(Exception::custom(
            "block-FP8 dequantization expects matching rank-2 or rank-3 weight and scale arrays",
        ));
    }
    let scale = decode_scale(scale, stream)?;
    let out_dim = weight.dim(0);
    let in_dim = weight.dim(1);
    let scale = Array::repeat_axis::<f32>(scale, 128, 0, stream)?;
    let scale = Array::repeat_axis::<f32>(scale, 128, 1, stream)?;
    weight.from_fp8(Dtype::Float32, stream)?.multiply(
        scale.try_index_device((..out_dim, ..in_dim), stream)?,
        stream,
    )
}

/// Applies a rank-2 block-scaled E4M3 weight matrix.
pub fn linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    linear_with_input_observer(input, weight, scale, stream, None)
}

pub(crate) fn linear_with_input_observer(
    input: &Array,
    weight: &Array,
    scale: &Array,
    stream: &Stream,
    observer: Option<&mut dyn super::linear::NativeProjectionInputObserver>,
) -> Result<Array, Exception> {
    if input.ndim() < 1 || weight.ndim() != 2 || scale.ndim() != 2 {
        return Err(original::invalid(format_args!(
            "block-FP8 linear expects an input with at least one dimension and rank-2 weight/scale arrays",
        )));
    }
    let original_rows = original::validate(input, weight, stream)?;
    let scale = decode_scale(scale, stream)?;
    let input_shape = input.shape();
    let output_dtype = activation_dtype(input)?;
    let in_dim = input.dim(-1);
    let out_dim = weight.dim(0);
    if in_dim <= 0
        || out_dim <= 0
        || weight.dim(1) != in_dim
        || scale.dim(0) != ceil_div(out_dim, SCALE_BLOCK)
        || scale.dim(1) != ceil_div(in_dim, SCALE_BLOCK)
    {
        return Err(original::invalid(format_args!(
            "invalid block-FP8 linear weight or scale dimensions",
        )));
    }
    let rows = original_rows.unwrap_or_else(|| (input.size() as i32) / in_dim);
    if is_cpu_stream(stream)? {
        let weight = dequantize(weight, &scale, stream)?;
        let input = activation_reference(input, rows, in_dim, stream)?;
        if let Some(observer) = observer {
            observer.observe(&input)?;
        }
        let output = matmul(&input, &weight.transpose(stream)?, stream)?;
        return restore_activation_dtype(output, output_dtype, stream);
    }
    let prototype = input;
    let input = input.reshape(&[rows, in_dim], stream)?;
    let input = quantize_activations(&input, rows, in_dim, stream)?;
    if let Some(observer) = observer {
        // Repeated scales retain complete blocks even when the feature axis is
        // narrower than 128. Include that padding before offering the factory.
        let plan = eredu_nn::BlockFp8InputReconstructionPlan::new(input_shape)
            .map_err(original::reconstruction_error)?;
        plan.validate_operands(input.values.shape(), input.scales.shape())
            .map_err(original::reconstruction_error)?;
        let source = plan
            .logical_capture_source()
            .map_err(original::reconstruction_error)?;
        observer.observe_generated_retained(
            prototype,
            &source,
            &mut InputReconstruction {
                plan,
                quantized: &input,
                stream,
            },
        )?;
    }
    let scale_cols = scale.dim(1);

    let out = linear_quantized(
        &input, weight, &scale, rows, in_dim, out_dim, scale_cols, stream,
    )?;

    finish_linear_output(out, prototype, output_dtype, out_dim, stream)
}

#[allow(clippy::too_many_arguments)]
fn linear_quantized(
    input: &QuantizedActivations,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    #[cfg(feature = "cuda")]
    {
        linear_tiled_cuda(
            &input.values,
            &input.scales,
            weight,
            scale,
            rows,
            in_dim,
            out_dim,
            scale_cols,
            stream,
        )
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let _ = (in_dim, scale_cols);
        kernel::linear(
            [&input.values, &input.scales, weight, scale],
            rows,
            out_dim,
            rows <= TILED_ROW_THRESHOLD,
            stream,
        )
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let _ = (
            input, weight, scale, rows, in_dim, out_dim, scale_cols, stream,
        );
        Err(original::invalid(format_args!(
            "GPU FP8 requires a Metal or CUDA backend"
        )))
    }
}

fn finish_linear_output(
    out: Array,
    prototype: &Array,
    output_dtype: Dtype,
    out_dim: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let out = safemlx::ops::reshape_like_prefix(&out, prototype, out_dim, stream)?;
    restore_activation_dtype(out, output_dtype, stream)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn linear_tiled_cuda(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    rows: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(linear_kernel_cuda()?);
        }
        let config = linear_tiled_config(rows, in_dim, out_dim, scale_cols);
        cell.borrow()
            .as_ref()
            .expect("CUDA FP8 linear kernel initialized")
            .apply_one_device([input, input_scale, weight, scale], &config, stream)
    })
}

#[cfg(feature = "cuda")]
fn linear_kernel_cuda() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_linear_k16",
        ["input", "input_scale", "weight", "scale"],
        ["out"],
        concat!(
            "uint32_t out_col = blockIdx.x * blockDim.x + threadIdx.x;",
            "uint32_t row = blockIdx.y;",
            "uint32_t lane_k = threadIdx.y;",
            "__shared__ float partial[REDUCTION_TILE][OUT_TILE];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint32_t k = lane_k; k < IN_DIM; k += REDUCTION_TILE) {",
            "  uint8_t raw = weight[out_col * IN_DIM + k];",
            "  float x = fp8_e4m3_to_float(input[row * IN_DIM + k]);",
            "  uint32_t scale_col = k / SCALE_BLOCK;",
            "  float xs = float(input_scale[row * SCALE_COLS + scale_col]);",
            "  float ws = float(scale[(out_col / SCALE_BLOCK) * SCALE_COLS + scale_col]);",
            "  acc += x * fp8_e4m3_to_float(raw) * xs * ws;",
            " }",
            "}",
            "partial[threadIdx.y][threadIdx.x] = acc;",
            "__syncthreads();",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            " float sum = 0.0f;",
            " for (uint32_t lane = 0; lane < REDUCTION_TILE; ++lane) sum += partial[lane][threadIdx.x];",
            " out[row * OUT_DIM + out_col] = sum;",
            "}"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

/// Applies a rank-3 block-scaled E4M3 group bank to group-major rows.
pub fn grouped_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    grouped_linear_with_row_layout(
        input,
        weight,
        scale,
        group_ids,
        eredu_nn::LinearRowLayout::Contiguous,
        stream,
    )
}

/// Applies a grouped projection with explicit independent row-block origins.
pub fn grouped_linear_with_row_layout(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    layout: eredu_nn::LinearRowLayout,
    stream: &Stream,
) -> Result<Array, Exception> {
    original::validate_grouped(input, weight, scale, group_ids, layout, stream)?;
    if input.ndim() != 2 || weight.ndim() != 3 || scale.ndim() != 3 || group_ids.ndim() != 1 {
        return Err(Exception::custom(
            "grouped block-FP8 linear expects rank-2 input, rank-3 weight/scale, and rank-1 group ids",
        ));
    }
    let output_dtype = activation_dtype(input)?;
    let routes = input.dim(0);
    let in_dim = input.dim(1);
    let groups = weight.dim(0);
    let out_dim = weight.dim(1);
    let row_width = layout
        .rows_per_partition_fixed(out_dim.max(0) as usize)
        .map_err(|error| original::invalid(format_args!("{error}")))? as i32;
    let scale_rows = layout
        .scale_rows_fixed(out_dim.max(0) as usize, SCALE_BLOCK as usize)
        .map_err(|error| original::invalid(format_args!("{error}")))? as i32;
    if routes != group_ids.dim(0)
        || routes < 0
        || groups <= 0
        || out_dim <= 0
        || in_dim <= 0
        || weight.dim(2) != in_dim
        || scale.dim(0) != groups
        || scale.dim(1) != scale_rows
        || scale.dim(2) != ceil_div(in_dim, SCALE_BLOCK)
    {
        return Err(Exception::custom(
            "invalid grouped block-FP8 linear weight, scale, or route dimensions",
        ));
    }
    #[cfg(feature = "cuda")]
    let scale_cols = scale.dim(2);
    if matches!(
        weight.dtype(),
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        // A parameter overlay promotes this complete bank to floating point,
        // including its projection input arithmetic, as for ordinary linears.
        return grouped_matmul(
            input,
            &weight.swap_axes(-1, -2, stream)?,
            group_ids,
            true,
            stream,
        );
    }
    if routes == 0 {
        return safemlx::ops::zeros_dtype(&[0, out_dim], output_dtype, stream);
    }
    let scale = decode_scale(scale, stream)?;
    if is_cpu_stream(stream)? {
        let weight = dequantize_with_row_layout(weight, &scale, layout, stream)?;
        let input = activation_reference(input, routes, in_dim, stream)?;
        let output = grouped_matmul(
            &input,
            &weight.swap_axes(-1, -2, stream)?,
            group_ids,
            true,
            stream,
        )?;
        return restore_activation_dtype(output, output_dtype, stream);
    }
    let input = quantize_activations(input, routes, in_dim, stream)?;

    #[cfg(feature = "cuda")]
    let out = grouped_linear_tiled_cuda(
        &input.values,
        &input.scales,
        weight,
        &scale,
        group_ids,
        routes,
        in_dim,
        out_dim,
        scale_cols,
        row_width,
        stream,
    )?;

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let out = kernel::grouped_linear(
        [&input.values, &input.scales, weight, &scale, group_ids],
        routes,
        out_dim,
        row_width,
        routes <= TILED_ROW_THRESHOLD,
        stream,
    )?;
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let out = {
        let _ = (input, row_width);
        return Err(original::invalid(format_args!(
            "GPU FP8 requires a Metal or CUDA backend"
        )));
    };

    restore_activation_dtype(out, output_dtype, stream)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn grouped_linear_tiled_cuda(
    input: &Array,
    input_scale: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    routes: i32,
    in_dim: i32,
    out_dim: i32,
    scale_cols: i32,
    row_width: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    GROUPED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(grouped_linear_kernel_cuda()?);
        }
        let config =
            grouped_tiled_config(routes, in_dim, out_dim, scale.dim(1), scale_cols, row_width);
        cell.borrow()
            .as_ref()
            .expect("CUDA grouped FP8 linear kernel initialized")
            .apply_one_device(
                [input, input_scale, weight, scale, group_ids],
                &config,
                stream,
            )
    })
}

#[cfg(feature = "cuda")]
fn grouped_linear_kernel_cuda() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_grouped_linear_k16",
        ["input", "input_scale", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint32_t out_col = blockIdx.x * blockDim.x + threadIdx.x;",
            "uint32_t route = blockIdx.y;",
            "uint32_t lane_k = threadIdx.y;",
            "uint32_t group = uint32_t(group_ids[route]);",
            "__shared__ float partial[REDUCTION_TILE][OUT_TILE];",
            "float acc = 0.0f;",
            "if (out_col < OUT_DIM) {",
            " for (uint32_t k = lane_k; k < IN_DIM; k += REDUCTION_TILE) {",
            "  uint32_t wi = (group * OUT_DIM + out_col) * IN_DIM + k;",
            "  uint32_t si = (group * SCALE_OUT + (out_col / ROW_WIDTH) * ROW_SCALES + (out_col % ROW_WIDTH) / SCALE_BLOCK) * SCALE_COLS + k / SCALE_BLOCK;",
            "  float xs = float(input_scale[route * SCALE_COLS + k / SCALE_BLOCK]);",
            "  acc += fp8_e4m3_to_float(input[route * IN_DIM + k]) * fp8_e4m3_to_float(weight[wi]) * xs * float(scale[si]);",
            " }",
            "}",
            "partial[threadIdx.y][threadIdx.x] = acc;",
            "__syncthreads();",
            "if (lane_k == 0 && out_col < OUT_DIM) {",
            " float sum = 0.0f;",
            " for (uint32_t lane = 0; lane < REDUCTION_TILE; ++lane) sum += partial[lane][threadIdx.x];",
            " out[route * OUT_DIM + out_col] = sum;",
            "}"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

/// Applies one row segment per group from a rank-2 block-FP8 matrix.
///
/// `weight` is laid out as `[groups * group_stride, input_dims]`. Each input
/// row selects a group through `group_ids`; the output uses
/// `weight[group * group_stride + row_offset .. + output_dims]`. This is useful
/// for fused projections whose logical per-group matrices are concatenated in
/// the checkpoint output dimension.
///
/// This absorbed-MLA operation deliberately keeps floating-point activations:
/// The absorb path dequantizes this weight and applies an
/// einsum rather than invoking its dynamically activation-quantized linear.
#[allow(clippy::too_many_arguments)]
pub fn segmented_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    group_stride: i32,
    row_offset: i32,
    output_dims: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2 || weight.ndim() != 2 || scale.ndim() != 2 || group_ids.ndim() != 1 {
        return Err(Exception::custom(
            "segmented block-FP8 linear expects rank-2 input/weight/scale and rank-1 group ids",
        ));
    }
    let scale = decode_scale(scale, stream)?;
    if input.dim(0) != group_ids.dim(0)
        || input.dim(1) != weight.dim(1)
        || group_stride <= 0
        || row_offset < 0
        || output_dims <= 0
        || row_offset + output_dims > group_stride
        || weight.dim(0) % group_stride != 0
    {
        return Err(Exception::custom(
            "invalid segmented block-FP8 linear dimensions",
        ));
    }
    if is_cpu_stream(stream)? {
        return segmented_reference(
            input,
            weight,
            &scale,
            group_ids,
            group_stride,
            row_offset,
            output_dims,
            false,
            stream,
        );
    }
    let routes = input.dim(0);
    SEGMENTED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(segmented_linear_kernel()?);
        }
        let config = CustomKernelConfig::new()
            .with_template_arg_int("IN_DIM", input.dim(1))
            .with_template_arg_int("OUT_DIM", output_dims)
            .with_template_arg_int("GROUP_STRIDE", group_stride)
            .with_template_arg_int("ROW_OFFSET", row_offset)
            .with_template_arg_int("SCALE_COLS", scale.dim(1))
            .with_template_arg_int("N_ELEMS", routes * output_dims)
            .with_grid([routes * output_dims, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([routes, output_dims], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("segmented FP8 linear kernel initialized")
            .apply_one_device([input, weight, &scale, group_ids], &config, stream)
    })
}

/// Applies the transpose of one row segment per group from a rank-2
/// block-FP8 matrix without dequantizing the complete matrix.
///
/// `input` has `segment_rows` columns and the result has `weight.dim(1)`
/// columns. Weight rows are selected with the same grouped segment layout as
/// [`segmented_linear`].
#[allow(clippy::too_many_arguments)]
pub fn segmented_transposed_linear(
    input: &Array,
    weight: &Array,
    scale: &Array,
    group_ids: &Array,
    group_stride: i32,
    row_offset: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.ndim() != 2 || weight.ndim() != 2 || scale.ndim() != 2 || group_ids.ndim() != 1 {
        return Err(Exception::custom(
            "segmented transposed block-FP8 linear expects rank-2 input/weight/scale and rank-1 group ids",
        ));
    }
    let scale = decode_scale(scale, stream)?;
    if input.dim(0) != group_ids.dim(0)
        || group_stride <= 0
        || row_offset < 0
        || input.dim(1) <= 0
        || row_offset + input.dim(1) > group_stride
        || weight.dim(0) % group_stride != 0
    {
        return Err(Exception::custom(
            "invalid segmented transposed block-FP8 linear dimensions",
        ));
    }
    if is_cpu_stream(stream)? {
        return segmented_reference(
            input,
            weight,
            &scale,
            group_ids,
            group_stride,
            row_offset,
            input.dim(1),
            true,
            stream,
        );
    }
    let routes = input.dim(0);
    let output_dims = weight.dim(1);
    SEGMENTED_TRANSPOSED_LINEAR_KERNEL.with(|cell| -> Result<_, Exception> {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(segmented_transposed_linear_kernel()?);
        }
        let config = CustomKernelConfig::new()
            .with_template_arg_int("SEGMENT_ROWS", input.dim(1))
            .with_template_arg_int("OUT_DIM", output_dims)
            .with_template_arg_int("GROUP_STRIDE", group_stride)
            .with_template_arg_int("ROW_OFFSET", row_offset)
            .with_template_arg_int("SCALE_COLS", scale.dim(1))
            .with_template_arg_int("N_ELEMS", routes * output_dims)
            .with_grid([routes * output_dims, 1, 1])
            .with_thread_group([256, 1, 1])
            .with_output_arg([routes, output_dims], Dtype::Float32);
        cell.borrow()
            .as_ref()
            .expect("segmented transposed FP8 linear kernel initialized")
            .apply_one_device([input, weight, &scale, group_ids], &config, stream)
    })
}

#[cfg(not(feature = "cuda"))]
fn segmented_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_segmented_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "if (elem >= N_ELEMS) return;",
            "uint out_col = elem % OUT_DIM;",
            "uint route = elem / OUT_DIM;",
            "uint group = uint(group_ids[route]);",
            "uint weight_row = group * GROUP_STRIDE + ROW_OFFSET + out_col;",
            "uint weight_base = weight_row * IN_DIM;",
            "uint input_base = route * IN_DIM;",
            "uint scale_base = (weight_row / 128) * SCALE_COLS;",
            "float acc = 0.0f;",
            "for (uint k = 0; k < IN_DIM; ++k) {",
            " float w = fp8_e4m3_to_float(weight[weight_base + k]);",
            " float s = float(scale[scale_base + k / 128]);",
            " acc += float(input[input_base + k]) * w * s;",
            "}",
            "out[elem] = acc;"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(not(feature = "cuda"))]
fn segmented_transposed_linear_kernel() -> Result<MetalKernel, Exception> {
    MetalKernel::new(
        "block_fp8_segmented_transposed_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "uint elem = thread_position_in_grid.x;",
            "if (elem >= N_ELEMS) return;",
            "uint out_col = elem % OUT_DIM;",
            "uint route = elem / OUT_DIM;",
            "uint group = uint(group_ids[route]);",
            "uint input_base = route * SEGMENT_ROWS;",
            "uint first_weight_row = group * GROUP_STRIDE + ROW_OFFSET;",
            "float acc = 0.0f;",
            "for (uint k = 0; k < SEGMENT_ROWS; ++k) {",
            " uint weight_row = first_weight_row + k;",
            " uint weight_idx = weight_row * OUT_DIM + out_col;",
            " uint scale_idx = (weight_row / 128) * SCALE_COLS + out_col / 128;",
            " acc += float(input[input_base + k]) * fp8_e4m3_to_float(weight[weight_idx]) * float(scale[scale_idx]);",
            "}",
            "out[elem] = acc;"
        ),
        METAL_HEADER,
        true,
        false,
    )
}

#[cfg(feature = "cuda")]
fn segmented_linear_kernel() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_segmented_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "auto elem = cooperative_groups::this_grid().thread_rank();",
            "if (elem >= N_ELEMS) return;",
            "uint32_t out_col = elem % OUT_DIM;",
            "uint32_t route = elem / OUT_DIM;",
            "uint32_t group = uint32_t(group_ids[route]);",
            "uint32_t weight_row = group * GROUP_STRIDE + ROW_OFFSET + out_col;",
            "uint32_t weight_base = weight_row * IN_DIM;",
            "uint32_t input_base = route * IN_DIM;",
            "uint32_t scale_base = (weight_row / 128) * SCALE_COLS;",
            "float acc = 0.0f;",
            "for (uint32_t k = 0; k < IN_DIM; ++k) {",
            " float w = fp8_e4m3_to_float(weight[weight_base + k]);",
            " float s = float(scale[scale_base + k / 128]);",
            " acc += float(input[input_base + k]) * w * s;",
            "}",
            "out[elem] = acc;"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

#[cfg(feature = "cuda")]
fn segmented_transposed_linear_kernel() -> Result<CudaKernel, Exception> {
    CudaKernel::new(
        "block_fp8_segmented_transposed_linear",
        ["input", "weight", "scale", "group_ids"],
        ["out"],
        concat!(
            "auto elem = cooperative_groups::this_grid().thread_rank();",
            "if (elem >= N_ELEMS) return;",
            "uint32_t out_col = elem % OUT_DIM;",
            "uint32_t route = elem / OUT_DIM;",
            "uint32_t group = uint32_t(group_ids[route]);",
            "uint32_t input_base = route * SEGMENT_ROWS;",
            "uint32_t first_weight_row = group * GROUP_STRIDE + ROW_OFFSET;",
            "float acc = 0.0f;",
            "for (uint32_t k = 0; k < SEGMENT_ROWS; ++k) {",
            " uint32_t weight_row = first_weight_row + k;",
            " uint32_t weight_idx = weight_row * OUT_DIM + out_col;",
            " uint32_t scale_idx = (weight_row / 128) * SCALE_COLS + out_col / 128;",
            " acc += float(input[input_base + k]) * fp8_e4m3_to_float(weight[weight_idx]) * float(scale[scale_idx]);",
            "}",
            "out[elem] = acc;"
        ),
        CUDA_HEADER,
        true,
        0,
    )
}

#[cfg(not(feature = "cuda"))]
const METAL_HEADER: &str = include_str!("fp8/header.metal");

#[cfg(feature = "cuda")]
const CUDA_HEADER: &str = concat!(
    "#include <cooperative_groups.h>\n",
    "#include <cuda_fp16.h>\n",
    "#include <math.h>\n",
    "#include <stdint.h>\n",
    "__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {",
    " if ((bits & 127) == 127) return NAN;",
    " uint16_t v = uint16_t(bits & 127) << 7;",
    " float converted = __half2float(__ushort_as_half(v)) * 256.0f;",
    " return (bits & 128) ? -converted : converted;",
    "}\n",
    "__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float value) {",
    " if (isnan(value)) return uint8_t(0x7f);",
    " uint32_t sign = (__float_as_uint(value) >> 24) & 0x80u;",
    " float magnitude = fminf(fabsf(value), 448.0f);",
    " if (magnitude == 0.0f) return uint8_t(sign);",
    " uint32_t code;",
    " if (magnitude < 0.015625f) {",
    "  int mantissa = int(nearbyintf(magnitude * 512.0f));",
    "  if (mantissa < 0) mantissa = 0;",
    "  code = mantissa >= 8 ? 8u : uint32_t(mantissa);",
    " } else {",
    "  int exponent = int(floorf(log2f(magnitude)));",
    "  float base = exp2f(float(exponent));",
    "  int mantissa = int(nearbyintf((magnitude / base - 1.0f) * 8.0f));",
    "  if (mantissa == 8) { exponent += 1; mantissa = 0; }",
    "  code = uint32_t((exponent + 7) * 8 + mantissa);",
    "  if (code > 0x7eu) code = 0x7eu;",
    " }",
    " return uint8_t(sign | code);",
    "}\n",
);

#[cfg(test)]
mod tests;
