//! Reduced-precision row projections with a complete FP32 reduction.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use safemlx::Dtype;
use safemlx::{error::Exception, Array, Stream};

/// Geometry-selected dot order retained across query tiles. Small products use
/// sequential accumulation; larger score products use the fixed row reduction.
/// Ordinary value GEMM remains independently selected. Batch/head axes do not
/// contribute to the small-product cutoff.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchedDotReduction {
    Native,
    Row32,
    Sequential,
}

pub(crate) fn batched_dot_reduction(
    rows: u64,
    width: u64,
    columns: u64,
    large: BatchedDotReduction,
) -> BatchedDotReduction {
    if rows > 0
        && width > 0
        && columns > 0
        && rows.saturating_mul(width).saturating_mul(columns) < 400
    {
        BatchedDotReduction::Sequential
    } else {
        large
    }
}

fn row_dot(width: usize, product: impl Fn(usize) -> f32) -> f32 {
    let mut k = 0;
    let mut lanes = [0.0_f32; 32];
    while k + 32 <= width {
        for lane in 0..32 {
            lanes[lane] += product(k + lane);
        }
        k += 32;
    }
    for width in [16, 8, 4] {
        for lane in 0..width {
            lanes[lane] += lanes[lane + width];
        }
    }
    let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let mut tail = [0.0_f32; 4];
    while k + 8 <= width {
        for lane in 0..4 {
            tail[lane] += product(k + lane);
            tail[lane] += product(k + lane + 4);
        }
        k += 8;
    }
    sum += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    while k < width {
        sum += product(k);
        k += 1;
    }
    sum
}

/// Reference-ordered CPU BF16 dots with one final cast. Native GEMM can change
/// a BF16 tie when physical strides change. Shared KV heads stay unreplicated;
/// bounded BF16 layouts are borrowed and products convert in registers.
pub(crate) fn ordered_bf16_batched_product(
    lhs: &Array,
    rhs: &Array,
    reduction: BatchedDotReduction,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    use safemlx::{DeviceType, Dtype};
    if !matches!(
        reduction,
        BatchedDotReduction::Sequential | BatchedDotReduction::Row32
    ) || lhs.dtype() != Dtype::Bfloat16
        || rhs.dtype() != Dtype::Bfloat16
        || stream.get_device()?.get_type()? != DeviceType::Cpu
    {
        return Ok(None);
    }
    if lhs.ndim() != 4
        || rhs.ndim() != 4
        || lhs.dim(0) != rhs.dim(0)
        || rhs.dim(1) <= 0
        || lhs.dim(1) % rhs.dim(1) != 0
        || lhs.dim(3) != rhs.dim(2)
    {
        return Err(Exception::custom("invalid ordered batched BF16 geometry"));
    }
    let (batch, heads, rows, width) = (
        lhs.dim(0) as usize,
        lhs.dim(1) as usize,
        lhs.dim(2) as usize,
        lhs.dim(3) as usize,
    );
    let (physical_heads, columns) = (rhs.dim(1) as usize, rhs.dim(3) as usize);
    let left = lhs.contiguous(false, stream)?;
    let right = rhs.contiguous(false, stream)?;
    let left = left.evaluated()?;
    let right = right.evaluated()?;
    let left = left.as_slice::<half::bf16>();
    let right = right.as_slice::<half::bf16>();
    let count = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(rows))
        .and_then(|n| n.checked_mul(columns))
        .ok_or_else(|| Exception::custom("ordered product size overflow"))?;
    let mut output = vec![half::bf16::ZERO; count];
    for b in 0..batch {
        for h in 0..heads {
            for row in 0..rows {
                for column in 0..columns {
                    let l = ((b * heads + h) * rows + row) * width;
                    let r = (b * physical_heads + h / (heads / physical_heads)) * width * columns
                        + column;
                    let product = |k: usize| left[l + k].to_f32() * right[r + k * columns].to_f32();
                    let sum = if reduction == BatchedDotReduction::Row32 {
                        row_dot(width, product)
                    } else {
                        let mut sum = 0.0_f32;
                        for k in 0..width {
                            sum += product(k);
                        }
                        sum
                    };
                    output[((b * heads + h) * rows + row) * columns + column] =
                        half::bf16::from_f32(sum);
                }
            }
        }
    }
    Ok(Some(Array::from_slice(
        &output,
        &[lhs.dim(0), lhs.dim(1), lhs.dim(2), rhs.dim(3)],
    )))
}

/// Supplies the native row-dot mechanism when its dtype and device apply.
/// Weights remain row-major, including the leading bank axis for selected rows.
pub(crate) fn bf16_row_projection(
    input: &Array,
    weight: &Array,
    groups: Option<&Array>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    bf16_projection(
        input,
        weight,
        groups,
        false,
        BatchedDotReduction::Native,
        None,
        stream,
    )
}

/// Reduced-precision column-major affine dot order. Parameters retain packed
/// BF16 storage; CPU borrowing converts individual products in registers only.
pub(crate) fn bf16_column_projection(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    ordered_bf16_affine(input, weight, None, true, stream)
}

/// Complete FP32 row-dot plus bias, with one BF16 output rounding.
pub(crate) fn bf16_affine_projection(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    ordered_bf16_affine(input, weight, bias, false, stream)
}

fn ordered_bf16_affine(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    columns_order: bool,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    use safemlx::{DeviceType, Dtype};
    if input.dtype() != Dtype::Bfloat16 || weight.dtype() != Dtype::Bfloat16 {
        return Ok(None);
    }
    if input.ndim() < 2
        || weight.ndim() != 2
        || input.dim(-1) != weight.dim(1)
        || bias.is_some_and(|value| value.shape() != [weight.dim(0)])
    {
        return Err(Exception::custom("invalid ordered affine geometry"));
    }
    if stream.get_device()?.get_type()? != DeviceType::Cpu {
        let contiguous_input = input.contiguous(false, stream)?;
        let contiguous_weight = weight.contiguous(false, stream)?;
        let contiguous_bias = bias
            .map(|value| value.contiguous(false, stream))
            .transpose()?;
        if let Some(output) = bf16_projection(
            &contiguous_input,
            &contiguous_weight,
            None,
            columns_order,
            BatchedDotReduction::Native,
            contiguous_bias.as_ref(),
            stream,
        )? {
            return Ok(Some(output));
        }
        let cpu = Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
        let host_input = input.copy(&cpu)?;
        let host_weight = weight.copy(&cpu)?;
        let host_bias = bias.map(|value| value.copy(&cpu)).transpose()?;
        return ordered_bf16_affine(
            &host_input,
            &host_weight,
            host_bias.as_ref(),
            columns_order,
            &cpu,
        )?
        .map(|value| value.copy(stream))
        .transpose();
    }
    let width = input.dim(-1) as usize;
    let rows = input.size() / width;
    let columns = weight.dim(0) as usize;
    let left = input.contiguous(false, stream)?.into_evaluated()?;
    let right = weight.contiguous(false, stream)?.into_evaluated()?;
    let left = left.as_slice::<half::bf16>();
    let right = right.as_slice::<half::bf16>();
    let bias = bias
        .map(|value| {
            value
                .as_dtype(Dtype::Float32, stream)?
                .contiguous(false, stream)?
                .into_evaluated()
        })
        .transpose()?;
    let bias = bias.as_ref().map(|value| value.as_slice::<f32>());
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| Exception::custom("affine output size overflow"))?;
    let mut output = vec![half::bf16::ZERO; count];
    for row in 0..rows {
        for column in 0..columns {
            let product =
                |k: usize| left[row * width + k].to_f32() * right[column * width + k].to_f32();
            let mut k = 0;
            let mut sum;
            if columns_order {
                let mut sums = [0.0_f32; 4];
                while k + 4 <= width {
                    for lane in 0..4 {
                        sums[lane] += product(k + lane);
                    }
                    k += 4;
                }
                while k < width {
                    sums[0] += product(k);
                    k += 1;
                }
                sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
            } else {
                sum = row_dot(width, product);
            }
            if let Some(bias) = bias {
                sum += bias[column];
            }
            output[row * columns + column] = half::bf16::from_f32(sum);
        }
    }
    let mut shape = input.shape().to_vec();
    *shape.last_mut().expect("validated affine rank") = weight.dim(0);
    Ok(Some(Array::from_slice(&output, &shape)))
}

fn bf16_projection(
    input: &Array,
    weight: &Array,
    groups: Option<&Array>,
    columns: bool,
    reduction: BatchedDotReduction,
    bias: Option<&Array>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType,
        };
        use std::cell::RefCell;
        thread_local! {
            static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
        }
        let width = input.dim(-1);
        if input.dtype() != Dtype::Bfloat16
            || weight.dtype() != Dtype::Bfloat16
            || (!columns && width % 32 != 0 && reduction != BatchedDotReduction::Sequential)
            || width == 0
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let rows = input.size() / width as usize;
        let outputs = weight.dim(-2);
        if rows == 0 {
            return Ok(None);
        }
        let rows =
            i32::try_from(rows).map_err(|_| Exception::custom("matrix row count overflow"))?;
        if weight.dim(-1) != width || weight.ndim() != if groups.is_some() { 3 } else { 2 } {
            return Err(Exception::custom("invalid row projection geometry"));
        }
        if let Some(ids) = groups {
            if ids.ndim() != 1
                || ids.dim(0) != rows
                || !matches!(
                    ids.dtype(),
                    Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
                )
            {
                return Err(Exception::custom("invalid row projection group indices"));
            }
            let valid = ids
                .ge(Array::from_int(0), stream)?
                .logical_and(ids.lt(Array::from_int(weight.dim(0)), stream)?, stream)?;
            if !valid.all(None, stream)?.try_item::<bool>(stream)? {
                return Err(Exception::custom(
                    "row projection group index is outside the bank",
                ));
            }
        }
        let ids = groups
            .cloned()
            .unwrap_or_else(|| Array::from_slice(&[0i32], &[1]));
        let ids = ids.as_dtype(Dtype::Int32, stream)?;
        let bias_values = bias
            .cloned()
            .unwrap_or_else(|| Array::from_slice(&[0f32], &[1]));
        let config = CustomKernelConfig::new()
            .with_template_arg_int("BIAS", i32::from(bias.is_some()))
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("COLUMNS", i32::from(columns))
            .with_template_arg_int(
                "SEQUENTIAL",
                i32::from(reduction == BatchedDotReduction::Sequential),
            )
            .with_template_arg_int("OUTPUTS", outputs)
            .with_template_arg_int("GROUPED", i32::from(groups.is_some()))
            .with_grid([32, outputs, rows])
            .with_thread_group([32, 1, 1])
            .with_output_arg([rows, outputs], Dtype::Bfloat16);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "bf16_row_projection", ["input", "weight", "groups", "bias"], ["output"],
                    concat!(
                        "uint lane = thread_position_in_grid.x;",
                        "uint col = thread_position_in_grid.y;",
                        "uint row = thread_position_in_grid.z;",
                        "uint group = GROUPED ? uint(groups[row]) : 0;",
                        "size_t base = (size_t(group) * OUTPUTS + col) * WIDTH;",
                        "float sum = 0.0f;",
                        "if (SEQUENTIAL) { if(lane==0) { for(uint k=0;k<WIDTH;++k) sum+=float(input[size_t(row)*WIDTH+k])*float(weight[base+k]); output[size_t(row)*OUTPUTS+col]=bfloat16_t(BIAS ? sum + float(bias[col]) : sum); } return; }",
                        "if (COLUMNS) {",
                        " if(lane<4) { for(uint k=lane; k+3-lane<WIDTH; k+=4) sum+=float(input[size_t(row)*WIDTH+k])*float(weight[base+k]); }",
                        " if(lane==0) { for(uint k=(WIDTH/4)*4; k<WIDTH; ++k) sum+=float(input[size_t(row)*WIDTH+k])*float(weight[base+k]); }",
                        " float s1=simd_shuffle(sum,1), s2=simd_shuffle(sum,2), s3=simd_shuffle(sum,3);",
                        " if(lane==0) { float result=((sum+s1)+s2)+s3; output[size_t(row)*OUTPUTS+col]=bfloat16_t(BIAS ? result+float(bias[col]) : result); }",
                        " return; }",
                        "for (uint k = lane; k < WIDTH; k += 32) {",
                        " sum += float(input[size_t(row) * WIDTH + k]) * float(weight[base + k]);",
                        "}",
                        "sum += simd_shuffle_down(sum, 16);",
                        "sum += simd_shuffle_down(sum, 8);",
                        "sum += simd_shuffle_down(sum, 4);",
                        "float s1 = simd_shuffle(sum, 1);",
                        "float s2 = simd_shuffle(sum, 2);",
                        "float s3 = simd_shuffle(sum, 3);",
                        "if (lane == 0) { float result=(sum+s1)+(s2+s3); output[size_t(row)*OUTPUTS+col]=bfloat16_t(BIAS ? result+float(bias[col]) : result); }"
                    ), "", true, false,
                )?);
            }
            cell.borrow().as_ref().expect("row projection kernel initialized")
                .apply_device([input, weight, &ids, &bias_values], &config, stream)
        })?;
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("matrix input rank") = outputs;
        output
            .pop()
            .expect("row projection output")
            .reshape(&shape, stream)
            .map(Some)
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, groups, columns, reduction, bias, stream);
        Ok(None)
    }
}

/// Reusable RHS layout for complete FP32 batched BF16 products.
/// The contiguous node belongs to the complete invocation, so the native kernel
/// does not insert a separate layout copy for every query tile. Preparation is
/// lazy and preserves the row-dot / strided-column reduction order.
pub(crate) struct PreparedBf16BatchedProduct {
    weights: Array,
    batch: i32,
    heads: i32,
    width: i32,
    outputs: i32,
    columns: bool,
    reduction: BatchedDotReduction,
}

impl PreparedBf16BatchedProduct {
    pub(crate) fn new(
        rhs: &Array,
        columns: bool,
        reduction: BatchedDotReduction,
        stream: &Stream,
    ) -> Result<Option<Self>, Exception> {
        if !cfg!(all(feature = "metal", not(feature = "cuda")))
            || rhs.size() == 0
            || rhs.dtype() != safemlx::Dtype::Bfloat16
            || rhs.ndim() != 4
            || (!columns && rhs.dim(2) % 32 != 0 && reduction != BatchedDotReduction::Sequential)
            || stream.get_device()?.get_type()? != safemlx::DeviceType::Gpu
        {
            return Ok(None);
        }
        let batches = rhs
            .dim(0)
            .checked_mul(rhs.dim(1))
            .ok_or_else(|| Exception::custom("matrix batch count overflow"))?;
        let width = rhs.dim(2);
        let outputs = rhs.dim(3);
        let weights = rhs
            .swap_axes(-1, -2, stream)?
            .reshape(&[batches, outputs, width], stream)?
            .contiguous(false, stream)?;
        Ok(Some(Self {
            weights,
            batch: rhs.dim(0),
            heads: rhs.dim(1),
            width,
            outputs,
            columns,
            reduction,
        }))
    }

    pub(crate) fn apply(&self, lhs: &Array, stream: &Stream) -> Result<Option<Array>, Exception> {
        if lhs.size() == 0
            || lhs.dtype() != safemlx::Dtype::Bfloat16
            || lhs.ndim() != 4
            || lhs.dim(0) != self.batch
            || lhs.dim(1) % self.heads != 0
            || lhs.dim(3) != self.width
        {
            return Ok(None);
        }
        let rows = lhs.dim(2);
        let heads = lhs.dim(1);
        let repeats = heads / self.heads;
        let query_banks = self
            .batch
            .checked_mul(heads)
            .ok_or_else(|| Exception::custom("matrix batch count overflow"))?;
        let input = lhs.reshape(&[-1, self.width], stream)?;
        let ids = (0..query_banks)
            .flat_map(|head| std::iter::repeat_n(head / repeats, rows as usize))
            .collect::<Vec<_>>();
        let groups = Array::from_slice(&ids, &[ids.len() as i32]);
        bf16_projection(
            &input,
            &self.weights,
            Some(&groups),
            self.columns,
            self.reduction,
            None,
            stream,
        )?
        .map(|output| output.reshape(&[self.batch, heads, rows, self.outputs], stream))
        .transpose()
    }
}
