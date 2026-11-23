use crate::CandleContext;
use candle_core::{DType, Tensor as NativeTensor};
use eredu_nn::{
    AttentionMask, Error, Index, LayerNormalizationArithmetic, PadMode, RotaryArithmetic, Tensor,
    TensorElementType,
};

/// A Candle tensor with a checked shape in Eredu's dimension representation.
#[derive(Clone, Debug)]
pub struct CandleTensor {
    pub(crate) value: NativeTensor,
    shape: Vec<i32>,
    pub(crate) boolean: bool,
    pub(crate) packed: Option<crate::quantized::PackedParameter>,
}

pub(crate) fn checked_gather_indices(
    indices: &NativeTensor,
    extent: i64,
) -> candle_core::Result<Vec<i64>> {
    if !matches!(
        indices.dtype(),
        DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
    ) {
        candle_core::bail!("gather requires integer indices");
    }
    // Metal accepts I64 gather indices but cannot widen I32 indices on device.
    let mut ids = indices
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    for id in &mut ids {
        if !(-extent..extent).contains(id) {
            candle_core::bail!("gather index {id} is outside axis extent {extent}");
        }
        if *id < 0 {
            *id += extent;
        }
    }
    Ok(ids)
}

fn size(value: i32) -> Result<usize, Error> {
    usize::try_from(value).map_err(Error::backend_source)
}

fn positive_size(value: i32) -> Result<usize, Error> {
    if value <= 0 {
        return Err(Error::backend("expected a positive dimension"));
    }
    size(value)
}

fn dimensions(shape: &[i32]) -> Result<Vec<usize>, Error> {
    shape.iter().copied().map(size).collect()
}

fn axis(index: i32, rank: usize) -> Result<usize, Error> {
    let rank = i32::try_from(rank).map_err(Error::backend_source)?;
    let normalized = if index < 0 { index + rank } else { index };
    if !(0..rank).contains(&normalized) {
        return Err(Error::backend(format!(
            "axis {index} is outside rank {rank}"
        )));
    }
    size(normalized)
}

fn float_dtype(element: TensorElementType) -> Result<DType, Error> {
    match element {
        TensorElementType::F16 => Ok(DType::F16),
        TensorElementType::Bf16 => Ok(DType::BF16),
        TensorElementType::F32 => Ok(DType::F32),
        TensorElementType::F64 => Ok(DType::F64),
        _ => Err(Error::backend("expected F16, BF16, F32, or F64")),
    }
}

fn ordinary_float(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
}

fn floating_operands(
    lhs: &NativeTensor,
    rhs: &NativeTensor,
) -> candle_core::Result<(NativeTensor, NativeTensor)> {
    let (left, right) = (lhs.dtype(), rhs.dtype());
    if left == right || !ordinary_float(left) || !ordinary_float(right) {
        return Ok((lhs.clone(), rhs.clone()));
    }
    let promoted = if left == DType::F64 || right == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    Ok((lhs.to_dtype(promoted)?, rhs.to_dtype(promoted)?))
}

fn product_operands(
    lhs: &NativeTensor,
    rhs: &NativeTensor,
) -> candle_core::Result<(NativeTensor, NativeTensor, DType)> {
    let (lhs, rhs) = floating_operands(lhs, rhs)?;
    let output = lhs.dtype();
    if lhs.device().is_cpu() && output == DType::BF16 && rhs.dtype() == output {
        Ok((lhs.to_dtype(DType::F32)?, rhs.to_dtype(DType::F32)?, output))
    } else {
        Ok((lhs, rhs, output))
    }
}

// Candle accelerator kernels use I64 for signed arithmetic and indexing.
fn widen_i32(input: &NativeTensor) -> candle_core::Result<NativeTensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if input.device().is_metal() {
        return crate::metal_f32::widen_i32(input);
    }
    input
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::I64)?
        .to_device(input.device())
}

fn narrow_i64(input: &NativeTensor) -> candle_core::Result<NativeTensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if input.device().is_metal() {
        return crate::metal_f32::narrow_i64(input);
    }
    input
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::I32)?
        .to_device(input.device())
}

fn device_binary(
    lhs: &NativeTensor,
    rhs: &NativeTensor,
    op: impl FnOnce(&NativeTensor, &NativeTensor) -> candle_core::Result<NativeTensor>,
) -> candle_core::Result<NativeTensor> {
    if !lhs.device().is_cpu() && lhs.dtype() == DType::I32 && rhs.dtype() == DType::I32 {
        let lhs = widen_i32(lhs)?;
        let rhs = widen_i32(rhs)?;
        return narrow_i64(&op(&lhs, &rhs)?);
    }
    let (lhs, rhs) = floating_operands(lhs, rhs)?;
    op(&lhs, &rhs)
}

pub(crate) fn rounded_bf16_binary(
    lhs: &NativeTensor,
    rhs: &NativeTensor,
    op: impl FnOnce(&NativeTensor, &NativeTensor) -> candle_core::Result<NativeTensor>,
) -> candle_core::Result<NativeTensor> {
    // Candle 0.11's ARM CPU BF16 vector add and multiply discard the low bits.
    // Widen operands exactly and round once at the operation's output.
    if lhs.device().is_cpu() && lhs.dtype() == DType::BF16 && rhs.dtype() == DType::BF16 {
        op(&lhs.to_dtype(DType::F32)?, &rhs.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)
    } else {
        device_binary(lhs, rhs, op)
    }
}

impl CandleTensor {
    pub(crate) fn independent_copy(&self) -> Result<Self, Error> {
        if self.value.elem_count() == 0 {
            return self.wrap_view(NativeTensor::zeros(
                self.value.shape(),
                self.value.dtype(),
                self.value.device(),
            ));
        }
        if !self.value.device().is_cpu()
            && self.value.dtype() == DType::I32
            && !self.value.is_contiguous()
        {
            return Self::wrap(widen_i32(&self.value).and_then(|value| narrow_i64(&value)));
        }
        self.wrap_view(self.value.force_contiguous())
    }

    pub(crate) fn wrap(result: candle_core::Result<NativeTensor>) -> Result<Self, Error> {
        let value = result.map_err(Error::backend_source)?;
        let shape = value
            .dims()
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<Result<_, _>>()
            .map_err(Error::backend_source)?;
        Ok(Self {
            value,
            shape,
            boolean: false,
            packed: None,
        })
    }

    fn wrap_view(&self, result: candle_core::Result<NativeTensor>) -> Result<Self, Error> {
        let mut view = Self::wrap(result)?;
        view.boolean = self.boolean;
        Ok(view)
    }

    fn axis(&self, index: i32) -> Result<usize, Error> {
        axis(index, self.shape.len())
    }

    fn zeros(shape: &[i32], dtype: DType, context: &CandleContext) -> Result<Self, Error> {
        if dtype == DType::I32 && context.placement().is_metal() {
            return Self::full_i32(0, shape, context);
        }
        Self::wrap(NativeTensor::zeros(
            dimensions(shape)?,
            dtype,
            &context.device,
        ))
    }
}

fn apply_to_integer_views(
    values: &[CandleTensor],
    apply: impl FnOnce(&[&NativeTensor]) -> candle_core::Result<NativeTensor>,
) -> Result<CandleTensor, Error> {
    if values
        .iter()
        .all(|tensor| tensor.value.dtype() == DType::I32 && !tensor.value.device().is_cpu())
    {
        return CandleTensor::wrap((|| {
            let wide = values
                .iter()
                .map(|tensor| widen_i32(&tensor.value))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let tensors: Vec<_> = wide.iter().collect();
            narrow_i64(&apply(&tensors)?)
        })());
    }
    let tensors: Vec<_> = values.iter().map(|tensor| &tensor.value).collect();
    CandleTensor::wrap(apply(&tensors))
}

impl Tensor for CandleTensor {
    type Context = CandleContext;

    fn shape(&self) -> &[i32] {
        &self.shape
    }

    fn element_type(&self) -> Option<TensorElementType> {
        Some(match self.value.dtype() {
            DType::U8 if self.boolean => TensorElementType::Bool,
            DType::U8 => TensorElementType::U8,
            DType::U32 => TensorElementType::U32,
            DType::I16 => TensorElementType::I16,
            DType::I32 => TensorElementType::I32,
            DType::I64 => TensorElementType::I64,
            DType::BF16 => TensorElementType::Bf16,
            DType::F16 => TensorElementType::F16,
            DType::F32 => TensorElementType::F32,
            DType::F64 => TensorElementType::F64,
            _ => return None,
        })
    }

    fn cast_float(&self, element: TensorElementType, _: &Self::Context) -> Result<Self, Error> {
        if !ordinary_float(self.value.dtype()) {
            return Err(Error::backend("cast_float requires floating input"));
        }
        Self::wrap(self.value.to_dtype(float_dtype(element)?))
    }

    fn unloaded_f32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        Self::zeros(shape, DType::F32, context)
    }

    fn unloaded_i32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        Self::zeros(shape, DType::I32, context)
    }

    fn unloaded_i64(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        Self::zeros(shape, DType::I64, context)
    }

    fn zeros_float(
        shape: &[i32],
        element: TensorElementType,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        Self::zeros(shape, float_dtype(element)?, context)
    }

    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        Self::wrap(NativeTensor::from_slice(
            values,
            dimensions(shape)?,
            &context.device,
        ))
    }

    fn from_i32_slice(
        values: &[i32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        Self::wrap(NativeTensor::from_slice(
            values,
            dimensions(shape)?,
            &context.device,
        ))
    }

    fn to_f32_vec(&self, _: &Self::Context) -> Result<Vec<f32>, Error> {
        self.value
            .to_dtype(DType::F32)
            .and_then(|value| value.flatten_all()?.to_vec1())
            .map_err(Error::backend_source)
    }

    fn to_i32_vec(&self, _: &Self::Context) -> Result<Vec<i32>, Error> {
        self.value
            .to_device(&candle_core::Device::Cpu)
            .and_then(|value| value.flatten_all()?.to_vec1())
            .map_err(Error::backend_source)
    }

    fn full_f32(value: f32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        let shape = dimensions(shape)?;
        if shape.contains(&0) {
            return Self::wrap(NativeTensor::zeros(shape, DType::F32, &context.device));
        }
        Self::wrap(NativeTensor::full(value, shape, &context.device))
    }

    fn full_i32(value: i32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        let shape = dimensions(shape)?;
        if shape.contains(&0) {
            return Self::wrap(NativeTensor::zeros(shape, DType::I32, &context.device));
        }
        if !context.placement().is_cpu() {
            return Self::wrap(
                NativeTensor::from_vec(vec![value], (), &context.device)
                    .and_then(|scalar| scalar.broadcast_as(shape)),
            );
        }
        Self::wrap(NativeTensor::full(value, shape, &context.device))
    }

    fn add(&self, rhs: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(rounded_bf16_binary(
            &self.value,
            &rhs.value,
            NativeTensor::broadcast_add,
        ))
    }

    fn subtract(&self, rhs: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(device_binary(
            &self.value,
            &rhs.value,
            NativeTensor::broadcast_sub,
        ))
    }

    fn multiply(&self, rhs: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(rounded_bf16_binary(
            &self.value,
            &rhs.value,
            NativeTensor::broadcast_mul,
        ))
    }

    fn multiply_scalar(&self, rhs: f32, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(self.value.affine(rhs.into(), 0.))
    }

    fn divide(&self, rhs: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(device_binary(
            &self.value,
            &rhs.value,
            NativeTensor::broadcast_div,
        ))
    }

    fn square(&self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(self.value.sqr())
    }

    fn abs(&self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(self.value.abs())
    }

    fn sign(&self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(self.value.sign())
    }

    fn sqrt(&self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(self.value.sqrt())
    }

    fn tanh(&self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(self.value.tanh())
    }

    fn maximum_scalar(&self, rhs: f32, _: &Self::Context) -> Result<Self, Error> {
        if !ordinary_float(self.value.dtype()) {
            return Err(Error::backend("maximum_scalar requires floating input"));
        }
        Self::wrap(self.value.maximum(rhs))
    }

    fn maximum_i32(&self, rhs: i32, _: &Self::Context) -> Result<Self, Error> {
        if self.value.dtype() != DType::I32 {
            return Err(Error::backend("maximum_i32 requires I32 input"));
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.value.device().is_metal() {
            return Self::wrap(crate::metal_f32::maximum_i32(&self.value, rhs));
        }
        if self.value.device().is_cuda() {
            return Self::wrap((|| {
                let wide = widen_i32(&self.value)?;
                let minimum = NativeTensor::from_vec(vec![i64::from(rhs)], (), wide.device())?;
                narrow_i64(&wide.broadcast_maximum(&minimum)?)
            })());
        }
        Self::wrap(self.value.maximum(rhs))
    }

    fn clip(&self, minimum: &Self, maximum: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap((|| {
            let (input, minimum) = floating_operands(&self.value, &minimum.value)?;
            let clipped = input.broadcast_maximum(&minimum)?;
            let (clipped, maximum) = floating_operands(&clipped, &maximum.value)?;
            clipped.broadcast_minimum(&maximum)
        })())
    }

    fn softmax_axis(&self, dim: i32, precise: bool, _: &Self::Context) -> Result<Self, Error> {
        let dim = self.axis(dim)?;
        let dtype = self.value.dtype();
        Self::wrap(if precise && matches!(dtype, DType::F16 | DType::BF16) {
            self.value
                .to_dtype(DType::F32)
                .and_then(|value| candle_nn::ops::softmax(&value, dim)?.to_dtype(dtype))
        } else {
            candle_nn::ops::softmax(&self.value, dim)
        })
    }

    fn capped_attention_probabilities(
        &self,
        mask: &Self,
        softcap: f32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        if self.value.device().is_metal()
            && self.value.dtype() == DType::F32
            && self.value.device().same_device(mask.value.device())
        {
            // Keep scalar tanh and normalization rounding stable at the
            // capped score boundary. Only the attention score block and mask
            // cross to the host; projections and the value product stay native.
            let host = CandleContext::cpu();
            let download = |value: &NativeTensor| {
                Self::wrap(
                    value
                        .force_contiguous()
                        .and_then(|value| value.to_device(&host.device)),
                )
            };
            let logits = download(&self.value)?;
            let mask = download(&mask.value)?;
            let probabilities = eredu_nn::arithmetic::capped_attention_probabilities(
                &logits, &mask, softcap, &host,
            )?;
            return Self::wrap(probabilities.value.to_device(self.value.device()));
        }
        eredu_nn::arithmetic::capped_attention_probabilities(self, mask, softcap, context)
    }

    fn reshape(&self, shape: &[i32], _: &Self::Context) -> Result<Self, Error> {
        if let Some(hole) = shape.iter().position(|&dim| dim == -1) {
            let mut resolved = shape.to_vec();
            resolved[hole] = 1;
            let known = dimensions(&resolved)?
                .into_iter()
                .try_fold(1usize, usize::checked_mul);
            let known = known
                .filter(|&count| count > 0)
                .ok_or_else(|| Error::backend("cannot infer a dimension from this shape"))?;
            if !self.value.elem_count().is_multiple_of(known) {
                return Err(Error::backend("reshape changes the number of elements"));
            }
            resolved[hole] =
                i32::try_from(self.value.elem_count() / known).map_err(Error::backend_source)?;
            return self.wrap_view(self.value.reshape(dimensions(&resolved)?));
        }
        self.wrap_view(self.value.reshape(dimensions(shape)?))
    }

    fn broadcast_to(&self, shape: &[i32], _: &Self::Context) -> Result<Self, Error> {
        self.wrap_view(self.value.broadcast_as(dimensions(shape)?))
    }

    fn transpose_axes(&self, axes: &[i32], _: &Self::Context) -> Result<Self, Error> {
        let axes = axes
            .iter()
            .map(|&dim| self.axis(dim))
            .collect::<Result<Vec<_>, _>>()?;
        self.wrap_view(self.value.permute(axes))
    }

    fn swap_axes(&self, left: i32, right: i32, _: &Self::Context) -> Result<Self, Error> {
        self.wrap_view(self.value.transpose(self.axis(left)?, self.axis(right)?))
    }

    fn transpose(&self, _: &Self::Context) -> Result<Self, Error> {
        self.wrap_view(self.value.t())
    }

    fn expand_dims(&self, dim: i32, _: &Self::Context) -> Result<Self, Error> {
        self.wrap_view(self.value.unsqueeze(axis(dim, self.shape.len() + 1)?))
    }

    fn squeeze_axes(&self, axes: &[i32], _: &Self::Context) -> Result<Self, Error> {
        let mut axes = axes
            .iter()
            .map(|&dim| self.axis(dim))
            .collect::<Result<Vec<_>, _>>()?;
        axes.sort_unstable();
        if axes.windows(2).any(|pair| pair[0] == pair[1])
            || axes.iter().any(|&dim| self.shape[dim] != 1)
        {
            return Err(Error::backend("squeeze requires distinct unit axes"));
        }
        self.wrap_view(
            axes.into_iter()
                .rev()
                .try_fold(self.value.clone(), |value, dim| value.squeeze(dim)),
        )
    }

    fn index(&self, indexes: &[Index], _: &Self::Context) -> Result<Self, Error> {
        if indexes.len() > self.shape.len() {
            return Err(Error::backend("too many tensor indexes"));
        }
        let mut value = self.value.clone();
        for (dim, &index) in indexes.iter().enumerate().rev() {
            let normalize = |index| {
                let length = self.shape[dim];
                size(if index < 0 { index + length } else { index })
            };
            value = match index {
                Index::Full => continue,
                Index::At(index) => value
                    .narrow(dim, normalize(index)?, 1)
                    .and_then(|value| value.squeeze(dim)),
                Index::Range(start, end) => {
                    let start = normalize(start)?;
                    let end = normalize(end)?;
                    let count = end
                        .checked_sub(start)
                        .ok_or_else(|| Error::backend("reversed tensor range"))?;
                    value.narrow(dim, start, count)
                }
            }
            .map_err(Error::backend_source)?;
        }
        self.wrap_view(Ok(value))
    }

    fn compact(&self, _: &Self::Context) -> Result<Self, Error> {
        self.independent_copy()
    }

    fn take_axis(&self, indexes: &Self, dim: i32, _: &Self::Context) -> Result<Self, Error> {
        let dim = self.axis(dim)?;
        let mut shape = self.shape[..dim].to_vec();
        shape.extend_from_slice(&indexes.shape);
        shape.extend_from_slice(&self.shape[dim + 1..]);
        let shape = dimensions(&shape)?;
        self.wrap_view((|| {
            let ids = checked_gather_indices(&indexes.value, i64::from(self.shape[dim]))?;
            let len = ids.len();
            let ids = NativeTensor::from_vec(ids, len, self.value.device())?;
            if !self.value.device().is_cpu() && self.value.dtype() == DType::I32 {
                let wide = widen_i32(&self.value)?;
                return narrow_i64(&wide.index_select(&ids, dim)?)?.reshape(shape);
            }
            self.value
                .contiguous()?
                .index_select(&ids, dim)?
                .reshape(shape)
        })())
    }

    fn concatenate(values: &[Self], dim: i32, _: &Self::Context) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| Error::backend("concatenate requires tensors"))?;
        let axis = first.axis(dim)?;
        apply_to_integer_views(values, |tensors| NativeTensor::cat(tensors, axis))
    }

    fn require_finite(&self, context: &CandleContext) -> Result<(), Error> {
        context
            .validate_tensor_device(self)
            .map_err(Error::backend_source)?;
        let finite = match self.value.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => self
                .to_f32_vec(context)?
                .iter()
                .all(|value| value.is_finite()),
            DType::F64 => self
                .value
                .flatten_all()
                .and_then(|value| value.to_vec1::<f64>())
                .map_err(Error::backend_source)?
                .iter()
                .all(|value| value.is_finite()),
            _ => {
                return Err(Error::backend(
                    "finite-value admission requires floating values",
                ))
            }
        };
        if !finite {
            return Err(Error::backend("non-finite floating values"));
        }
        Ok(())
    }

    fn rfft(&self, length: i32, axis: i32, context: &CandleContext) -> Result<(Self, Self), Error> {
        crate::spectral::forward(self, length, axis, context)
    }

    fn host_rfft(
        &self,
        length: i32,
        axis: i32,
        context: &CandleContext,
    ) -> Result<(Self, Self), Error> {
        self.rfft(length, axis, context)
    }

    fn irfft(
        real: &Self,
        imaginary: &Self,
        length: i32,
        axis: i32,
        context: &CandleContext,
    ) -> Result<Self, Error> {
        crate::spectral::inverse(real, imaginary, length, axis, context)
    }

    fn sin_cos(&self, _: &CandleContext) -> Result<(Self, Self), Error> {
        if self.value.dtype() != DType::F32 {
            return Err(Error::backend("sine/cosine requires F32 input"));
        }
        Ok((Self::wrap(self.value.sin())?, Self::wrap(self.value.cos())?))
    }

    fn stack(values: &[Self], dim: i32, _: &Self::Context) -> Result<Self, Error> {
        let first = values
            .first()
            .ok_or_else(|| Error::backend("stack requires tensors"))?;
        let axis = axis(dim, first.shape.len() + 1)?;
        apply_to_integer_views(values, |tensors| NativeTensor::stack(tensors, axis))
    }

    fn resize_bilinear_antialiased(
        &self,
        height: i32,
        width: i32,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        let [source_height, source_width, channels] = *self.shape() else {
            return Err(Error::backend(
                "resampling expects [height, width, channels]",
            ));
        };
        if height == source_height && width == source_width {
            return Ok(self.clone());
        }
        use eredu_nn::multimodal::antialiased_linear_weights;
        let y = antialiased_linear_weights(source_height, height)?;
        let x = antialiased_linear_weights(source_width, width)?;
        Self::wrap(crate::float_cpu::resize_bilinear_f32(
            &self.value,
            &y,
            &x,
            channels as usize,
        ))
    }

    fn matmul(lhs: &Self, rhs: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap((|| {
            let (lhs, rhs, dtype) = product_operands(&lhs.value, &rhs.value)?;
            if lhs.device().is_metal() {
                // Native GEMM rejects some sliced and broadcast matrix layouts.
                return lhs
                    .contiguous()?
                    .broadcast_matmul(&rhs.contiguous()?)?
                    .to_dtype(dtype);
            }
            lhs.broadcast_matmul(&rhs)?.to_dtype(dtype)
        })())
    }

    fn sum_axis(value: &Self, dim: i32, keep_dims: bool, _: &Self::Context) -> Result<Self, Error> {
        let dim = value.axis(dim)?;
        Self::wrap(if keep_dims {
            value.value.sum_keepdim(dim)
        } else {
            value.value.sum(dim)
        })
    }

    fn argmin_axis(
        value: &Self,
        dim: i32,
        keep_dims: bool,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        let dim = value.axis(dim)?;
        Self::wrap((|| {
            let indexes = if keep_dims {
                value.value.argmin_keepdim(dim)?
            } else {
                value.value.argmin(dim)?
            };
            if !indexes.device().is_cpu() {
                return narrow_i64(&indexes.to_dtype(DType::I64)?);
            }
            indexes.to_dtype(DType::I32)
        })())
    }

    fn pad(
        value: &Self,
        widths: &[(i32, i32)],
        mode: PadMode,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        if widths.len() != value.shape.len() {
            return Err(Error::backend("padding requires one width pair per axis"));
        }
        let mut padded = value.value.clone();
        for (dim, &(left, right)) in widths.iter().enumerate() {
            let (left, right) = (size(left)?, size(right)?);
            padded = match mode {
                PadMode::Constant => padded.pad_with_zeros(dim, left, right),
                PadMode::Edge => padded.pad_with_same(dim, left, right),
            }
            .map_err(Error::backend_source)?;
        }
        Self::wrap(Ok(padded))
    }

    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        let (stride, padding, dilation, groups) = (
            positive_size(stride)?,
            size(padding)?,
            positive_size(dilation)?,
            positive_size(groups)?,
        );
        Self::wrap((|| {
            let (_, sequence, channels) = input.value.dims3()?;
            let (outputs, kernel, inputs_per_group) = weight.value.dims3()?;
            if kernel == 0
                || channels == 0
                || outputs == 0
                || outputs % groups != 0
                || channels % groups != 0
                || channels / groups != inputs_per_group
                || sequence + 2 * padding <= dilation * (kernel - 1)
            {
                candle_core::bail!("invalid 1D convolution geometry");
            }
            if input.value.device().is_cpu()
                && input.value.dtype() == DType::F32
                && weight.value.dtype() == DType::F32
                && channels == groups
                && outputs == channels
                && stride == 1
                && padding == 0
            {
                return crate::recurrent::depthwise_convolution(
                    &input.value,
                    &weight.value,
                    dilation,
                );
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if input.value.device().is_metal()
                && ordinary_float(input.value.dtype())
                && ordinary_float(weight.value.dtype())
                && channels == groups
                && outputs == channels
                && stride == 1
                && padding == 0
            {
                let (input, weight, dtype) = product_operands(&input.value, &weight.value)?;
                return crate::metal_f32::depthwise_convolution(
                    &input.to_dtype(DType::F32)?,
                    &weight.to_dtype(DType::F32)?,
                    dilation,
                    dtype == DType::F32 && cfg!(target_arch = "aarch64"),
                )?
                .to_dtype(dtype);
            }
            let (input, weight, dtype) = product_operands(&input.value, &weight.value)?;
            let input = input.transpose(1, 2)?;
            let weight = weight.transpose(1, 2)?;
            input
                .conv1d(&weight, padding, stride, dilation, groups)?
                .transpose(1, 2)?
                .to_dtype(dtype)
        })())
    }

    fn conv2d(
        input: &Self,
        weight: &Self,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        let stride = [positive_size(stride.0)?, positive_size(stride.1)?];
        let padding = [size(padding.0)?, size(padding.1)?];
        let dilation = [positive_size(dilation.0)?, positive_size(dilation.1)?];
        let groups = positive_size(groups)?;
        Self::wrap((|| {
            let (_, height, width, channels) = input.value.dims4()?;
            let (outputs, kernel_h, kernel_w, inputs_per_group) = weight.value.dims4()?;
            if outputs == 0
                || outputs % groups != 0
                || channels == 0
                || channels % groups != 0
                || channels / groups != inputs_per_group
                || kernel_h == 0
                || kernel_w == 0
                || height + 2 * padding[0] <= dilation[0] * (kernel_h - 1)
                || width + 2 * padding[1] <= dilation[1] * (kernel_w - 1)
            {
                candle_core::bail!("invalid 2D convolution geometry");
            }
            let (input, weight, dtype) = product_operands(&input.value, &weight.value)?;
            let input = input.permute((0, 3, 1, 2))?;
            let weight = weight.permute((0, 3, 1, 2))?;
            if stride[0] == stride[1] && padding[0] == padding[1] && dilation[0] == dilation[1] {
                return input
                    .conv2d(&weight, padding[0], stride[0], dilation[0], groups)?
                    .permute((0, 2, 3, 1))?
                    .to_dtype(dtype);
            }
            let mut weight = weight;
            for (axis, dilation) in [2, 3].into_iter().zip(dilation) {
                if dilation > 1 {
                    let length = weight.dim(axis)?;
                    let indexes = (0..length)
                        .map(|index| u32::try_from(index * dilation))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                    let extent = (length - 1) * dilation + 1;
                    let mut shape = weight.dims().to_vec();
                    shape[axis] = extent;
                    let indexes = NativeTensor::from_vec(indexes, length, weight.device())?;
                    weight = NativeTensor::zeros(shape, weight.dtype(), weight.device())?
                        .index_add(&indexes, &weight.contiguous()?, axis)?;
                }
            }
            let input = input
                .pad_with_zeros(2, padding[0], padding[0])?
                .pad_with_zeros(3, padding[1], padding[1])?;
            let mut output = input.conv2d(&weight, 0, 1, 1, groups)?;
            for (axis, stride) in [2, 3].into_iter().zip(stride) {
                if stride > 1 {
                    let indexes = (0..u32::try_from(output.dim(axis)?)
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?)
                        .step_by(stride)
                        .collect::<Vec<_>>();
                    let count = indexes.len();
                    let indexes = NativeTensor::from_vec(indexes, count, output.device())?;
                    output = output.index_select(&indexes, axis)?;
                }
            }
            output.permute((0, 2, 3, 1))?.to_dtype(dtype)
        })())
    }

    fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        let (stride, padding, dilation, groups) = (
            positive_size(stride)?,
            size(padding)?,
            positive_size(dilation)?,
            positive_size(groups)?,
        );
        let output_padding = size(output_padding)?;
        if groups != 1 {
            return Err(Error::backend(
                "grouped transposed convolution is unsupported",
            ));
        }
        Self::wrap((|| {
            let (_, sequence, channels) = input.value.dims3()?;
            let (outputs, kernel, inputs) = weight.value.dims3()?;
            if sequence == 0
                || kernel == 0
                || channels == 0
                || outputs == 0
                || channels != inputs
                || (sequence - 1) * stride < 2 * padding
            {
                candle_core::bail!("unsupported transposed convolution geometry");
            }
            let input = input.value.transpose(1, 2)?;
            let weight = weight.value.permute((2, 0, 1))?;
            input
                .conv_transpose1d(&weight, padding, output_padding, stride, dilation, groups)?
                .transpose(1, 2)
        })())
    }

    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        Self::wrap((|| {
            let input_width = weight.value.dim(1)?;
            let mut shape = input.value.dims().to_vec();
            let last = shape
                .last_mut()
                .ok_or_else(|| candle_core::Error::Msg("linear requires a feature axis".into()))?;
            if *last != input_width || input_width == 0 {
                candle_core::bail!("linear input feature count differs from weight");
            }
            let rows = input.value.elem_count() / input_width;
            let (input, native_weight) = floating_operands(&input.value, &weight.value)?;
            let result =
                crate::matmul::project(&input.reshape((rows, input_width))?, &native_weight)?;
            *last = weight.value.dim(0)?;
            let result = match bias {
                Some(bias) => {
                    rounded_bf16_binary(&result, &bias.value, NativeTensor::broadcast_add)?
                }
                None => result,
            };
            result.reshape(shape)
        })())
    }

    fn linear_f16_operands(input: &Self, weight: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(crate::matmul::rounded_operand_projection(
            &input.value,
            &weight.value,
            DType::F16,
        ))
    }

    fn linear_f16_fused_accumulation(
        input: &Self,
        weight: &Self,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if input.value.device().is_metal() {
            return Self::wrap(crate::metal_f32::binary16_projection(
                &input.value,
                &weight.value,
            ));
        }
        Self::wrap(crate::float_cpu::binary16_projection(
            &input.value,
            &weight.value,
        ))
    }

    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        arithmetic: LayerNormalizationArithmetic,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        eredu_nn::operation_geometry::NormalizationGeometry::new(input.shape(), epsilon)?;
        Self::wrap((|| {
            let dtype = input.value.dtype();
            let features = input.value.dim(candle_core::D::Minus1)?;
            let device = input.value.device();
            if arithmetic == LayerNormalizationArithmetic::CascadedMoments {
                if !ordinary_float(dtype)
                    || [weight, bias].into_iter().flatten().any(|value| {
                        value.value.dims() != [features] || !ordinary_float(value.value.dtype())
                    })
                {
                    candle_core::bail!("invalid cascaded layer normalization geometry");
                }
                let mut values = input
                    .value
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let read = |value: &Self| value.value.to_dtype(DType::F32)?.to_vec1::<f32>();
                let weight = weight.map(read).transpose()?;
                let bias = bias.map(read).transpose()?;
                for row in values.chunks_exact_mut(features) {
                    eredu_nn::arithmetic::normalize_cascaded_f32(
                        row,
                        weight.as_deref(),
                        bias.as_deref(),
                        epsilon,
                        matches!(dtype, DType::BF16 | DType::F16),
                    );
                }
                return NativeTensor::from_vec(values, input.value.shape(), device)?
                    .to_dtype(dtype);
            }
            let blocked = arithmetic == LayerNormalizationArithmetic::BlockedFloat32;
            if blocked {
                if !features.is_multiple_of(32)
                    || !ordinary_float(dtype)
                    || [weight, bias]
                        .into_iter()
                        .flatten()
                        .any(|x| x.value.dims() != [features] || !ordinary_float(x.value.dtype()))
                {
                    candle_core::bail!("invalid blocked layer normalization geometry");
                }
                let input = input.value.to_dtype(DType::F32)?;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let mut output = if input.device().is_metal() {
                    crate::metal_f32::blocked_layer_norm(&input, epsilon)?
                } else {
                    crate::float_cpu::layer_norm(&input, epsilon)?
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let mut output = crate::float_cpu::layer_norm(&input, epsilon)?;
                if let Some(weight) = weight {
                    output = output.broadcast_mul(&weight.value.to_dtype(DType::F32)?)?;
                }
                if let Some(bias) = bias {
                    output = output.broadcast_add(&bias.value.to_dtype(DType::F32)?)?;
                }
                return output.to_dtype(dtype);
            }
            let gain = match weight {
                Some(weight) => weight.value.to_dtype(dtype)?,
                None => NativeTensor::ones(features, dtype, device)?,
            };
            let bias = match bias {
                Some(bias) => bias.value.to_dtype(dtype)?,
                None => NativeTensor::zeros(features, dtype, device)?,
            };
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            if crate::metal_f32::applies(&input.value) && features.is_multiple_of(32) {
                return crate::metal_f32::blocked_layer_norm(&input.value, epsilon)?
                    .broadcast_mul(&gain)?
                    .broadcast_add(&bias);
            }
            if crate::float_cpu::applies(&input.value)
                && cfg!(target_os = "macos")
                && features >= 32
                && features.is_multiple_of(32)
            {
                crate::float_cpu::layer_norm(&input.value, epsilon)?
                    .broadcast_mul(&gain)?
                    .broadcast_add(&bias)
            } else {
                candle_nn::ops::layer_norm_slow(&input.value, &gain, &bias, epsilon)
            }
        })())
    }

    fn gelu(input: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(input.value.gelu_erf())
    }

    fn elu(input: &Self, alpha: f32, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap(input.value.elu(alpha.into()))
    }

    fn zeros_like(&self, _: &Self::Context) -> Result<Self, Error> {
        self.wrap_view(self.value.zeros_like())
    }
    fn equal_i32(&self, value: i32, _: &Self::Context) -> Result<Self, Error> {
        let mut mask = Self::wrap((|| {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if self.value.device().is_metal() && self.value.dtype() == DType::I32 {
                return crate::metal_f32::equal_i32(&self.value, value);
            }
            if self.value.device().is_cuda() && self.value.dtype() == DType::I32 {
                let wide = widen_i32(&self.value)?;
                let rhs = NativeTensor::from_vec(vec![i64::from(value)], (), wide.device())?;
                return wide.broadcast_eq(&rhs);
            }
            let rhs = NativeTensor::from_vec(vec![value], (), self.value.device())?;
            self.value.broadcast_eq(&rhs)
        })())?;
        mask.boolean = true;
        Ok(mask)
    }

    fn logical_or(&self, rhs: &Self, _: &Self::Context) -> Result<Self, Error> {
        let mut result = Self::wrap(self.value.broadcast_maximum(&rhs.value))?;
        result.boolean = self.boolean && rhs.boolean;
        Ok(result)
    }

    fn where_condition(
        condition: &Self,
        when_true: &Self,
        when_false: &Self,
        _: &Self::Context,
    ) -> Result<Self, Error> {
        Self::wrap((|| {
            let shape = condition
                .value
                .shape()
                .broadcast_shape_binary_op(when_true.value.shape(), "where")?
                .broadcast_shape_binary_op(when_false.value.shape(), "where")?;
            let (when_true, when_false) = floating_operands(&when_true.value, &when_false.value)?;
            if !when_true.device().is_cpu()
                && when_true.dtype() == DType::I32
                && when_false.dtype() == DType::I32
            {
                let lhs = widen_i32(&when_true)?.broadcast_as(&shape)?;
                let rhs = widen_i32(&when_false)?.broadcast_as(&shape)?;
                return narrow_i64(
                    &condition
                        .value
                        .broadcast_as(&shape)?
                        .where_cond(&lhs, &rhs)?,
                );
            }
            condition.value.broadcast_as(&shape)?.where_cond(
                &when_true.broadcast_as(&shape)?,
                &when_false.broadcast_as(&shape)?,
            )
        })())
    }
    fn masked_scatter(&self, mask: &Self, source: &Self, _: &Self::Context) -> Result<Self, Error> {
        Self::wrap((|| {
            let (batch, tokens, width) = self.value.dims3()?;
            let (rows, source_width) = source.value.dims2()?;
            if mask.value.dims() != [batch, tokens] || width != source_width {
                candle_core::bail!("invalid row scatter geometry");
            }
            let mask = mask.value.flatten_all()?.to_vec1::<u8>()?;
            let indices = mask
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0)
                .map(|(index, _)| u32::try_from(index))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
            if indices.len() != rows {
                candle_core::bail!("mask and source row counts differ");
            }
            let indices = NativeTensor::from_vec(indices, (rows, 1), self.value.device())?
                .broadcast_as((rows, width))?
                .contiguous()?;
            self.value
                .reshape((batch * tokens, width))?
                .contiguous()?
                .scatter(&indices, &source.value.contiguous()?, 0)?
                .reshape((batch, tokens, width))
        })())
    }

    fn multi_axis_rotary_embeddings(
        position_ids: &Self,
        spec: &eredu_nn::multimodal::MultiAxisRotarySpec,
        context: &Self::Context,
    ) -> Result<(Self, Self), Error> {
        let mut shape = position_ids.shape().to_vec();
        if spec.axes.is_empty()
            || shape.len() < 2
            || shape.last().copied() != Some(spec.axes.len() as i32)
        {
            return Err(Error::backend("invalid multi-axis position geometry"));
        }
        let ids = position_ids.to_i32_vec(context)?;
        let width = spec.dimensions()?;
        let (cosine, sine) = if cfg!(target_arch = "aarch64")
            && matches!(
                spec.layout,
                eredu_nn::multimodal::MultiAxisRotaryLayout::RoundRobinSections
                    | eredu_nn::multimodal::MultiAxisRotaryLayout::RoundRobinSectionsWithZeroOverflow
            )
        {
            crate::float_cpu::round_robin_rotary_tables(&ids, spec, width as usize)
                .map_err(Error::backend_source)?
        } else if cfg!(target_arch = "aarch64") && spec.layout == eredu_nn::multimodal::MultiAxisRotaryLayout::SplitHalves {
            crate::float_cpu::split_rotary_tables(&ids, spec, width as usize)
        } else {
            eredu_nn::multimodal::reference_multi_axis_rotary_embeddings(
                &ids,
                ids.len() / spec.axes.len(),
                spec,
            )?
        };
        *shape.last_mut().expect("position axis") = width;
        Ok((
            Self::from_f32_slice(&cosine, &shape, context)?,
            Self::from_f32_slice(&sine, &shape, context)?,
        ))
    }

    fn rope(
        input: &Self,
        rotary_width: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        apply_rotary(
            input,
            rotary_width,
            traditional,
            base,
            scale,
            None,
            offset,
            None,
            RotaryArithmetic::Native,
            context,
        )
    }

    fn masked_output_projection(
        input: eredu_nn::multimodal::MaskedOutputProjectionInput<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        input.validate()?;
        let [batch, sequence, hidden] = [
            input.hidden.dim(0),
            input.hidden.dim(1),
            input.hidden.dim(2),
        ];
        let vocabulary = input.output_weight.dim(0);
        let centroids = input.centroid_logits.dim(2);
        let per_centroid = vocabulary / centroids;
        let selected = input.top_centroids * per_centroid;
        let top = Self::wrap((|| {
            let logits = &input.centroid_logits.value;
            let count = input.top_centroids as usize;
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if logits.device().is_metal() {
                let top = crate::metal_f32::top_indices(&logits.to_dtype(DType::F32)?, count)?;
                crate::CandleContext::synchronize_device(logits.device())?;
                return top.flatten_all();
            }
            logits
                .contiguous()?
                .arg_sort_last_dim(false)?
                .narrow(2, 0, count)?
                .flatten_all()
        })())?;
        let ordering = input
            .token_ordering
            .reshape(&[centroids, per_centroid], context)?;
        let ids = ordering
            .take_axis(&top, 0, context)?
            .reshape(&[-1], context)?;
        let weights = input
            .output_weight
            .take_axis(&ids, 0, context)?
            .reshape(&[batch, sequence, selected, hidden], context)?
            .transpose_axes(&[0, 1, 3, 2], context)?;
        let values = Self::matmul(&input.hidden.expand_dims(-2, context)?, &weights, context)?
            .squeeze_axes(&[-2], context)?;
        let shape = (batch as usize, sequence as usize, vocabulary as usize);
        let ids = ids.reshape(&[batch, sequence, selected], context)?;
        Self::wrap((|| {
            let values = values.value.to_dtype(DType::F32)?;
            let minimum = values
                .min_all()?
                .affine(1., -f64::from(input.mask_margin))?;
            let output = minimum.broadcast_as(shape)?.contiguous()?;
            let ids = if !ids.value.device().is_cpu() && ids.value.dtype() == DType::I32 {
                widen_i32(&ids.value)?
            } else {
                ids.value
            };
            let ids = ids.to_dtype(DType::U32)?;
            output.scatter(&ids, &values, 2)
        })())
    }

    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        attention(
            queries,
            keys,
            values,
            scale,
            ScoreAdjustments {
                mask,
                sinks: None,
                softcap: None,
            },
            eredu_nn::AttentionArithmetic::Fused,
            context,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_rotary(
    input: &CandleTensor,
    width: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    amplitude: Option<f32>,
    offset: i32,
    inverse_frequencies: Option<&[f32]>,
    arithmetic: RotaryArithmetic,
    _: &CandleContext,
) -> Result<CandleTensor, Error> {
    let width = positive_size(width)?;
    let shape = input.value.dims();
    if shape.len() < 2 || width % 2 != 0 || width > shape[shape.len() - 1] {
        return Err(Error::backend("invalid rotary tensor geometry"));
    }
    if !base.is_finite() || base <= 0. || !scale.is_finite() {
        return Err(Error::backend("invalid rotary base or position scale"));
    }
    CandleTensor::wrap((|| {
        let features = shape[shape.len() - 1];
        let sequence = shape[shape.len() - 2];
        if input.value.elem_count() == 0 {
            return Ok(input.value.clone());
        }
        let device = input.value.device();
        let rows = input.value.elem_count() / (sequence * features);
        let rotary = input
            .value
            .narrow(candle_core::D::Minus1, 0, width)?
            .reshape((rows, 1, sequence, width))?
            .contiguous()?;
        let input_products = arithmetic == RotaryArithmetic::InputProducts;
        let float32_products = matches!(
            arithmetic,
            RotaryArithmetic::Float32Products | RotaryArithmetic::InputTableFloat32Products
        );
        let table_amplitude = input_products || float32_products;
        let rotary = if let Some(amplitude) = amplitude.filter(|_| !table_amplitude) {
            let compute = if rotary.dtype() == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            rotary.to_dtype(compute)?.affine(f64::from(amplitude), 0.)?
        } else {
            rotary
        };
        let rotated = if arithmetic == RotaryArithmetic::RecurrentFloat32
            || (arithmetic == RotaryArithmetic::Native
                && inverse_frequencies.is_none()
                && (crate::float_cpu::applies(&rotary)
                    || (rotary.dtype() == DType::F32 && rotary.device().is_metal())))
        {
            let wide = rotary.to_dtype(DType::F32)?;
            recurrent_rotary(&wide, traditional, base, scale, offset)?.to_dtype(rotary.dtype())?
        } else {
            let frequencies = inverse_frequencies.map_or_else(
                || {
                    (0..width / 2)
                        .map(|i| base.powf(-2. * i as f32 / width as f32))
                        .collect()
                },
                <[f32]>::to_vec,
            );
            let positions: Vec<_> = (0..sequence)
                .map(|i| (offset as i64 + i as i64) as f32 * scale)
                .collect();
            let angle_device = if device.is_metal() && table_amplitude {
                &candle_core::Device::Cpu
            } else {
                device
            };
            let frequencies = NativeTensor::from_vec(frequencies, (1, width / 2), angle_device)?;
            let positions = NativeTensor::from_vec(positions, (sequence, 1), angle_device)?;
            let angles = positions.broadcast_mul(&frequencies)?;
            let table = |value: NativeTensor| {
                let value = if let Some(amplitude) = amplitude.filter(|_| table_amplitude) {
                    value.affine(f64::from(amplitude), 0.)?
                } else {
                    value
                };
                let table_type = if arithmetic == RotaryArithmetic::Float32Products {
                    DType::F32
                } else {
                    rotary.dtype()
                };
                value.to_dtype(table_type)?.to_device(device)
            };
            let cos = table(angles.cos()?)?;
            let sin = table(angles.sin()?)?;
            if input_products || float32_products {
                let rotary = if float32_products {
                    rotary.to_dtype(DType::F32)?
                } else {
                    rotary
                };
                rotary_pair_products(&rotary, &cos, &sin, traditional)?
            } else if traditional {
                candle_nn::rotary_emb::rope_i(&rotary, &cos, &sin)?
            } else {
                candle_nn::rotary_emb::rope(&rotary, &cos, &sin)?
            }
        };
        let mut rotated_shape = shape.to_vec();
        rotated_shape[shape.len() - 1] = width;
        let rotated = rotated
            .reshape(rotated_shape)?
            .to_dtype(input.value.dtype())?;
        if width == features {
            return Ok(rotated);
        }
        let tail = input
            .value
            .narrow(candle_core::D::Minus1, width, features - width)?;
        NativeTensor::cat(&[rotated, tail], candle_core::D::Minus1)
    })())
}

fn recurrent_rotary(
    input: &NativeTensor,
    adjacent: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> candle_core::Result<NativeTensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if input.device().is_metal() {
        return crate::metal_f32::recurrent_rotary(input, adjacent, base, scale, offset);
    }
    crate::float_cpu::rotary(input, adjacent, base, scale, offset)
}

pub(crate) fn broadcast_rotary_table(
    value: &NativeTensor,
    batch: usize,
    tokens: usize,
    dimensions: usize,
) -> candle_core::Result<NativeTensor> {
    let value = match value.dims() {
        [n, d] if *n == tokens && *d == dimensions => value.reshape((1, tokens, dimensions))?,
        [b, n, d] if (*b == 1 || *b == batch) && *n == tokens && *d == dimensions => value.clone(),
        _ => candle_core::bail!("invalid explicit rotary embedding geometry"),
    };
    value.broadcast_as((batch, tokens, dimensions))
}

fn rotary_pair_products(
    input: &NativeTensor,
    cosine: &NativeTensor,
    sine: &NativeTensor,
    adjacent: bool,
) -> candle_core::Result<NativeTensor> {
    let (batch, heads, tokens, width) = input.dims4()?;
    let (first, second) = if adjacent {
        let pairs = input.reshape((batch, heads, tokens, width / 2, 2))?;
        (
            pairs.narrow(4, 0, 1)?.squeeze(4)?,
            pairs.narrow(4, 1, 1)?.squeeze(4)?,
        )
    } else {
        (
            input.narrow(3, 0, width / 2)?,
            input.narrow(3, width / 2, width / 2)?,
        )
    };
    let dtype = input.dtype();
    let compute = if dtype == DType::F64 {
        dtype
    } else {
        DType::F32
    };
    let first = first.to_dtype(compute)?;
    let second = second.to_dtype(compute)?;
    let cosine = cosine.to_dtype(compute)?;
    let sine = sine.to_dtype(compute)?;
    let mul = |value: &NativeTensor, table: &NativeTensor| {
        value
            .broadcast_mul(table)?
            .to_dtype(dtype)?
            .to_dtype(compute)
    };
    let head = mul(&first, &cosine)?
        .sub(&mul(&second, &sine)?)?
        .to_dtype(dtype)?;
    let tail = mul(&first, &sine)?
        .add(&mul(&second, &cosine)?)?
        .to_dtype(dtype)?;
    if adjacent {
        NativeTensor::stack(&[&head, &tail], 4)?.reshape(input.shape())
    } else {
        NativeTensor::cat(&[&head, &tail], 3)
    }
}

pub(crate) fn expand_attention_heads(
    value: &NativeTensor,
    target: usize,
) -> candle_core::Result<NativeTensor> {
    let (batch, heads, sequence, features) = value.dims4()?;
    if heads == target {
        return Ok(value.clone());
    }
    if heads == 0 || target == 0 || !target.is_multiple_of(heads) {
        candle_core::bail!("invalid grouped-query attention head counts");
    }
    value
        .unsqueeze(2)?
        .broadcast_as((batch, heads, target / heads, sequence, features))?
        .reshape((batch, target, sequence, features))
}

pub(crate) struct ScoreAdjustments<'a> {
    pub mask: AttentionMask<'a, CandleTensor>,
    pub sinks: Option<&'a CandleTensor>,
    pub softcap: Option<f32>,
}

pub(crate) fn attention(
    queries: &CandleTensor,
    keys: &CandleTensor,
    values: &CandleTensor,
    scale: f32,
    adjustments: ScoreAdjustments<'_>,
    arithmetic: eredu_nn::AttentionArithmetic,
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    CandleTensor::wrap((|| {
        let (batch, heads, query_len, width) = queries.value.dims4()?;
        let (key_batch, key_heads, key_len, key_width) = keys.value.dims4()?;
        let (value_batch, value_heads, value_len, _) = values.value.dims4()?;
        if batch != key_batch
            || batch != value_batch
            || key_heads != value_heads
            || key_len != value_len
            || width != key_width
            || key_len == 0
            || !scale.is_finite()
        {
            candle_core::bail!("invalid scaled dot-product attention geometry or scale");
        }
        let keys = expand_attention_heads(&keys.value, heads)?;
        let values = expand_attention_heads(&values.value, heads)?;
        let dtype = queries.value.dtype();
        if !ordinary_float(dtype) || keys.dtype() != dtype || values.dtype() != dtype {
            candle_core::bail!("attention requires matching floating tensor dtypes");
        }
        use eredu_nn::AttentionArithmetic;
        let compute = if matches!(dtype, DType::F16 | DType::BF16)
            || arithmetic == AttentionArithmetic::Float32
        {
            DType::F32
        } else {
            dtype
        };
        let bf16_cpu = dtype == DType::BF16
            && queries.value.device().is_cpu()
            && arithmetic != AttentionArithmetic::Float32;
        let ordered_bf16 = bf16_cpu
            || (dtype == DType::BF16
                && arithmetic == AttentionArithmetic::InputScores
                && crate::matmul::supports_ordered_metal(&queries.value));
        let input_scores = ordered_bf16 || arithmetic == AttentionArithmetic::InputScores;
        let arm_f32 = crate::float_cpu::applies(&queries.value);
        let product: fn(&NativeTensor, &NativeTensor, bool) -> candle_core::Result<NativeTensor> =
            if arm_f32 {
                crate::float_cpu::attention_product
            } else {
                crate::matmul::attention_product
            };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let product = if crate::metal_f32::applies(&queries.value) {
            crate::metal_f32::attention_product
        } else {
            product
        };
        let score_dtype = if input_scores { dtype } else { compute };
        let product_dtype = if ordered_bf16 { dtype } else { compute };
        let scores = product(
            &queries.value.to_dtype(product_dtype)?,
            &keys.to_dtype(product_dtype)?,
            true,
        )?
        .to_dtype(score_dtype)?
        .to_dtype(compute)?
        .affine(scale.into(), 0.)?
        .to_dtype(score_dtype)?;
        let scores = if let Some(cap) = adjustments.softcap {
            softcap_scores(scores, cap, context).map_err(candle_core::Error::wrap)?
        } else {
            scores
        };
        let mut scores = match adjustments.mask {
            AttentionMask::None => scores,
            AttentionMask::Tensor(mask) if mask.value.dtype() == DType::U8 => {
                let visible = mask.value.broadcast_as(scores.shape())?;
                let hidden = NativeTensor::full(f32::NEG_INFINITY, (), scores.device())?
                    .to_dtype(score_dtype)?
                    .broadcast_as(scores.shape())?;
                visible.where_cond(&scores, &hidden)?
            }
            AttentionMask::Tensor(mask) => rounded_bf16_binary(
                &scores,
                &mask.value.to_dtype(score_dtype)?,
                NativeTensor::broadcast_add,
            )?,
            AttentionMask::Causal => {
                if query_len > key_len {
                    candle_core::bail!("causal attention has more queries than keys");
                }
                let device = scores.device();
                let queries = i32::try_from(query_len)?;
                let offset = i32::try_from(key_len - query_len)?;
                let geometry =
                    eredu_nn::operation_geometry::CausalMaskGeometry::new(queries, offset, None)
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                let visible = crate::operators::causal_visibility(geometry, None, device)?
                    .broadcast_as(scores.shape())?;
                let masked = NativeTensor::full(f32::NEG_INFINITY, (), device)?
                    .to_dtype(score_dtype)?
                    .broadcast_as(scores.shape())?;
                visible.where_cond(&scores, &masked)?
            }
        };
        if let Some(sinks) = adjustments.sinks {
            let sinks = sink_scores(sinks, (batch, heads, query_len, 1), score_dtype)?;
            scores = NativeTensor::cat(&[&scores, &sinks], 3)?;
        }
        let probabilities = if ordered_bf16 {
            crate::operators::bf16_softmax(&scores)?
        } else if arm_f32 {
            crate::float_cpu::softmax(&scores)?
        } else {
            let softmax_dtype =
                if dtype == DType::F64 && arithmetic == AttentionArithmetic::InputScores {
                    DType::F32
                } else {
                    compute
                };
            attention_softmax(&scores, dtype, softmax_dtype)?.to_dtype(score_dtype)?
        };
        let probabilities = if adjustments.sinks.is_some() {
            probabilities.narrow(3, 0, key_len)?
        } else {
            probabilities
        };
        product(
            &probabilities.to_dtype(product_dtype)?,
            &values.to_dtype(product_dtype)?,
            false,
        )?
        .to_dtype(dtype)
    })())
}

pub(crate) fn sink_scores(
    sinks: &CandleTensor,
    shape: (usize, usize, usize, usize),
    dtype: DType,
) -> candle_core::Result<NativeTensor> {
    sinks
        .value
        .to_dtype(dtype)?
        .reshape((1, shape.1, 1, 1))?
        .broadcast_as(shape)
}

pub(crate) fn softcap_scores(
    scores: NativeTensor,
    cap: f32,
    context: &CandleContext,
) -> Result<NativeTensor, Error> {
    let dtype = scores.dtype();
    let compute = if dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let scores = CandleTensor::wrap(scores.to_dtype(compute))?;
    let capped = eredu_nn::arithmetic::softcap_attention_scores(&scores, cap, context)?;
    capped.value.to_dtype(dtype).map_err(Error::backend_source)
}

fn attention_softmax(
    scores: &NativeTensor,
    input_dtype: DType,
    compute: DType,
) -> candle_core::Result<NativeTensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if input_dtype == DType::F32 && scores.device().is_metal() {
        return crate::metal_f32::attention_softmax(scores);
    }
    let _ = input_dtype;
    candle_nn::ops::softmax(&scores.to_dtype(compute)?, candle_core::D::Minus1)
}

impl eredu_core::execution_control::SnapshotValue for CandleTensor {
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        use eredu_core::{
            cache::mechanisms::CompressedCacheStorage, execution_control::SnapshotCostOverflow,
        };
        crate::CandleCacheStorage::snapshot_tensor_estimate(self)
            .ok_or_else(|| eredu_core::BackendFailure::from_error(SnapshotCostOverflow))
    }

    fn copy_snapshot_value(
        &self,
        retention: &eredu_core::execution_control::SnapshotRetention,
    ) -> Result<Self, eredu_core::BackendFailure> {
        use eredu_core::cache::mechanisms::CacheTensorStorage;
        if self.packed.is_some() {
            return Err(crate::text::invalid(
                "Candle realtime snapshots require a dense tensor",
            ));
        }
        let copy = crate::CandleCacheStorage::fork_tensor(self);
        if copy.is_err() && !self.value.device().is_cpu() {
            CandleContext::quarantine_device(
                self.value.device(),
                (self.clone(), retention.clone()),
            );
        }
        copy
    }
}
