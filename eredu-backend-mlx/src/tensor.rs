mod rotary;
pub(crate) mod narrow;
pub(crate) mod clip;
pub(crate) use rotary::PreparedRotaryProfile;
mod masked_readout;
pub(crate) use masked_readout::control_bytes as masked_readout_control_bytes;
mod workspace_input;
#[cfg(test)]
pub(crate) use rotary::{
    prepared_calls as prepared_rotary_calls, reset_prepared_calls as reset_prepared_rotary_calls,
};
#[cfg(test)]
pub(crate) use workspace_input::{reset_workspace_slot_projections, workspace_slot_projections};

use crate::nn;
use eredu_core::checkpoint::TensorDtype;
use eredu_nn::{
    multimodal::{MaskedOutputProjectionInput, MultiAxisRotaryLayout, MultiAxisRotarySpec},
    AttentionMask, Error, Index, PadMode, Tensor,
};
use ref_cast::RefCast;
use safemlx::{
    argmin_axis,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{
        addmm, argpartition_axis, concatenate_axis, conv_transpose1d, full,
        indexing::{put_along_axis, ArrayIndex, ArrayIndexOp, NewAxis, TryIndexOp},
        matmul, maximum, pad, softmax_axis, stack_axis, sum_axis, PadMode as MlxPadMode,
    },
    Array, Dtype, Stream,
};
use smallvec::SmallVec;

/// Native scalar metadata shared by capture and communication; no evaluation.
pub(crate) fn portable_dtype(dtype: Dtype) -> TensorDtype {
    match dtype {
        Dtype::Bool => TensorDtype::Bool,
        Dtype::Uint8 => TensorDtype::U8,
        Dtype::Uint16 => TensorDtype::U16,
        Dtype::Uint32 => TensorDtype::U32,
        Dtype::Uint64 => TensorDtype::U64,
        Dtype::Int8 => TensorDtype::I8,
        Dtype::Int16 => TensorDtype::I16,
        Dtype::Int32 => TensorDtype::I32,
        Dtype::Int64 => TensorDtype::I64,
        Dtype::Float16 => TensorDtype::F16,
        Dtype::Float32 => TensorDtype::F32,
        Dtype::Float64 => TensorDtype::F64,
        Dtype::Bfloat16 => TensorDtype::Bf16,
        Dtype::Complex64 => TensorDtype::Complex64,
    }
}

fn backend<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, Error> {
    result.map_err(Error::backend_retained_source)
}

/// Backend-native MLX tensor handle used at Eredu's neutral tensor boundary.
///
/// Conversions only move or borrow the underlying lazy array handle. They do
/// not evaluate, synchronize, copy, or materialize tensor values on the host.
#[repr(transparent)]
#[derive(Clone, Debug, RefCast)]
pub struct MlxTensor(Array);

impl MlxTensor {
    /// Wraps a backend-native array without evaluation or copying.
    pub const fn from_array(array: Array) -> Self {
        Self(array)
    }

    /// Borrows the backend-native array.
    pub const fn as_array(&self) -> &Array {
        &self.0
    }

    /// Mutably borrows the native array for parameter binding without
    /// materialization or copying.
    #[cfg(test)]
    pub(crate) fn as_array_mut(&mut self) -> &mut Array {
        &mut self.0
    }

    /// Unwraps the backend-native array without evaluation or copying.
    pub fn into_array(self) -> Array {
        self.0
    }
}

impl From<Array> for MlxTensor {
    fn from(array: Array) -> Self {
        Self::from_array(array)
    }
}

impl From<MlxTensor> for Array {
    fn from(tensor: MlxTensor) -> Self {
        tensor.into_array()
    }
}

impl AsRef<Array> for MlxTensor {
    fn as_ref(&self) -> &Array {
        self.as_array()
    }
}

fn tensor(result: Result<Array, safemlx::error::Exception>) -> Result<MlxTensor, Error> {
    backend(result).map(MlxTensor::from_array)
}

impl Tensor for MlxTensor {
    type Context = Stream;

    fn shape(&self) -> &[i32] {
        self.0.shape()
    }

    fn unloaded_f32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::zeros::<f32>(shape, context))
    }

    fn unloaded_i32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::zeros::<i32>(shape, context))
    }

    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(
            Array::try_from_slice(values, shape)
                .map_err(Error::backend_retained_source)?
                .copy(context),
        )
    }

    fn from_i32_slice(
        values: &[i32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(
            Array::try_from_slice(values, shape)
                .map_err(Error::backend_retained_source)?
                .copy(context),
        )
    }

    fn to_f32_vec(&self, context: &Self::Context) -> Result<Vec<f32>, Error> {
        let array = if self.0.dtype() == Dtype::Float32 {
            self.0.clone()
        } else {
            backend(self.0.as_dtype(Dtype::Float32, context))?
        };
        let array = backend(array.contiguous(false, context))?;
        if array.size() == 0 {
            return Ok(Vec::new());
        }
        backend(array.evaluated())?
            .try_to_vec::<f32>()
            .map_err(Error::backend_retained_source)
    }

    fn to_i32_vec(&self, context: &Self::Context) -> Result<Vec<i32>, Error> {
        let array = if self.0.dtype() == Dtype::Int32 {
            self.0.clone()
        } else {
            backend(self.0.as_dtype(Dtype::Int32, context))?
        };
        let evaluated = backend(array.evaluated())?;
        if array.size() == 0 {
            return Ok(Vec::new());
        }
        evaluated.try_to_vec::<i32>().map_err(Error::backend_retained_source)
    }

    fn full_f32(value: f32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::full::<f32>(
            shape,
            Array::try_from_f32(value).map_err(Error::backend_retained_source)?,
            context,
        ))
    }

    fn full_i32(value: i32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::full::<i32>(
            shape,
            Array::try_from_int(value).map_err(Error::backend_retained_source)?,
            context,
        ))
    }
    fn full_u32(value: u32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::full::<u32>(
            shape,
            Array::try_from_scalar(value).map_err(Error::backend_retained_source)?,
            context,
        ))
    }

    fn add(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::add(self.as_array(), rhs.as_array(), context))
    }

    fn subtract(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::subtract(self.as_array(), rhs.as_array(), context))
    }

    fn multiply(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::multiply(self.as_array(), rhs.as_array(), context))
    }

    fn multiply_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::multiply(
            self.as_array(),
            Array::try_from_f32(rhs).map_err(Error::backend_retained_source)?,
            context,
        ))
    }

    fn divide(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::divide(self.as_array(), rhs.as_array(), context))
    }

    fn square(&self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::square(self.as_array(), context))
    }

    fn tanh(&self, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::tanh(self.as_array(), context))
    }

    fn maximum_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::maximum(
            self.as_array(),
            Array::try_from_f32(rhs).map_err(Error::backend_retained_source)?,
            context,
        ))
    }

    fn maximum_i32(&self, rhs: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::maximum(
            self.as_array(),
            Array::try_from_int(rhs).map_err(Error::backend_retained_source)?,
            context,
        ))
    }

    fn clip(&self, minimum: &Self, maximum: &Self, context: &Self::Context) -> Result<Self, Error> {
        clip::run(self, minimum, maximum, context)
    }

    fn softmax_axis(
        &self,
        axis: i32,
        precise: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(softmax_axis(self.as_array(), axis, precise, context))
    }

    fn reshape(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::reshape(self.as_array(), shape, context))
    }

    fn broadcast_to(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::broadcast_to(self.as_array(), shape, context))
    }

    fn transpose_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::transpose_axes(self.as_array(), axes, context))
    }

    fn swap_axes(&self, left: i32, right: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::swap_axes(self.as_array(), left, right, context))
    }

    fn transpose(&self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::transpose(self.as_array(), context))
    }

    fn expand_dims(&self, axis: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::expand_dims(self.as_array(), axis, context))
    }

    fn squeeze_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::squeeze_axes(self.as_array(), axes, context))
    }

    fn index(&self, indexes: &[Index], context: &Self::Context) -> Result<Self, Error> {
        let indexes = indexes
            .iter()
            .map(|index| match index {
                Index::Full => (..).index_op(),
                Index::At(index) => index.index_op(),
                Index::Range(start, end) => (*start..*end).index_op(),
            })
            .collect::<SmallVec<[ArrayIndexOp<'_>; 5]>>();
        tensor(
            self.as_array()
                .try_index_device(indexes.as_slice(), context),
        )
    }

    fn narrow_axis(&self, axis: usize, start: i32, end: i32, context: &Self::Context) -> Result<Self, Error> {
        narrow::execute(self,axis,start,end,context)
    }

    fn take_axis(&self, indexes: &Self, axis: i32, context: &Self::Context) -> Result<Self, Error> {
        let rank = i32::try_from(self.as_array().ndim()).map_err(Error::backend_retained_source)?;
        let normalized = if axis < 0 { axis + rank } else { axis };
        if !(0..rank).contains(&normalized) {
            return Err(Error::backend("gather axis is outside tensor rank"));
        }
        let indexes = crate::backend::nn::tensor::validate_take_indices(
            indexes.as_array(),
            self.as_array().dim(normalized),
            context,
        )
        .map_err(Error::backend_retained_source)?;
        tensor(Array::take_axis(self.as_array(), &indexes, axis, context))
    }

    fn zeros_like(&self, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::zeros_like(self.as_array(), context))
    }

    fn equal_i32(&self, value: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(self.as_array().eq(
            Array::try_from_int(value).map_err(Error::backend_retained_source)?,
            context,
        ))
    }

    fn logical_or(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::logical_or(self.as_array(), rhs.as_array(), context))
    }

    fn where_condition(
        condition: &Self,
        when_true: &Self,
        when_false: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(safemlx::ops::r#where(
            condition.as_array(),
            when_true.as_array(),
            when_false.as_array(),
            context,
        ))
    }

    fn masked_scatter(
        &self,
        mask: &Self,
        source: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(safemlx::ops::indexing::masked_scatter(
            self.as_array(),
            mask.as_array(),
            source.as_array(),
            context,
        ))
    }

    fn rope_with_frequencies(
        &self,
        dimensions: i32,
        traditional: bool,
        offset: i32,
        frequencies: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(safemlx::fast::rope(
            self.as_array(),
            dimensions,
            traditional,
            None::<f32>,
            1.0,
            offset,
            frequencies.as_array(),
            context,
        ))
    }

    fn concatenate(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(concatenate_axis(values, axis, context))
    }

    fn stack(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(stack_axis(values, axis, context))
    }

    fn matmul(lhs: &Self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(matmul(lhs.as_array(), rhs.as_array(), context))
    }

    fn sum_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(sum_axis(value.as_array(), axis, keep_dims, context))
    }

    fn mean_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(safemlx::ops::mean_axis(
            value.as_array(),
            axis,
            keep_dims,
            context,
        ))
    }

    fn argmin_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(argmin_axis!(
            value.as_array(),
            axis,
            keep_dims = keep_dims,
            stream = context
        ))
        .map(MlxTensor::from_array)
    }

    fn pad(
        value: &Self,
        widths: &[(i32, i32)],
        mode: PadMode,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let mode = match mode {
            PadMode::Constant => MlxPadMode::Constant,
            PadMode::Edge => MlxPadMode::Edge,
        };
        tensor(pad(
            value.as_array(),
            widths,
            None::<Array>,
            Some(mode),
            context,
        ))
    }

    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        crate::backend::nn::convolution::original::conv1d(
            input, weight, stride, padding, dilation, groups, context,
        )
    }

    fn conv2d(
        input: &Self,
        weight: &Self,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        crate::backend::nn::convolution::original::conv2d(
            input, weight, stride, padding, dilation, groups, context,
        )
    }

    fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        if input.shape().len() == 3
            && output_padding >= 0
            && crate::backend::nn::convolution::transpose_1d_needs_output_crop(
                weight.shape(),
                stride,
                padding,
                dilation,
            )
        {
            // MLX implements negative transpose padding by slicing the input
            // before input dilation. A removed input position then removes
            // `stride` output positions, violating the transposed equation.
            // Crop the expanded result instead; the view retains that complete
            // backing buffer, which the selected workspace fact also prices.
            let full_positions = (i128::from(input.shape()[1]) - 1) * i128::from(stride)
                + i128::from(dilation) * (i128::from(weight.shape()[1]) - 1)
                + i128::from(output_padding)
                + 1;
            let full_positions = i32::try_from(full_positions)
                .ok()
                .filter(|&n| i64::from(n) > 2 * i64::from(padding))
                .ok_or_else(|| Error::backend("invalid transposed convolution output extent"))?;
            let full = tensor(conv_transpose1d(
                input.as_array(),
                weight.as_array(),
                stride,
                0,
                dilation,
                output_padding,
                groups,
                context,
            ))?;
            return full.index(
                &[
                    Index::Full,
                    Index::Range(padding, full_positions - padding),
                    Index::Full,
                ],
                context,
            );
        }
        tensor(conv_transpose1d(
            input.as_array(),
            weight.as_array(),
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            context,
        ))
    }

    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let weight = backend(weight.as_array().transpose(context))?;
        match bias {
            Some(bias)
                if input.shape().len() > 2
                    && input.shape().last() == Some(&0)
                    && weight.ndim() == 2 =>
            {
                // MLX addmm flattens a batched input with [-1, K]. For K=0
                // that inferred dimension is ambiguous, even though all row
                // dimensions are known here. Preserve its promotion and bias
                // behavior with an explicit two-dimensional empty product.
                let mut shape = input.shape().to_vec();
                let rows = shape[..shape.len() - 1]
                    .iter()
                    .try_fold(1i32, |rows, &n| rows.checked_mul(n))
                    .ok_or_else(|| Error::backend("empty linear row count overflow"))?;
                let flat = input.reshape(&[rows, 0], context)?;
                let output = backend(addmm(
                    bias.as_array(),
                    flat.as_array(),
                    &weight,
                    None,
                    None,
                    context,
                ))?;
                *shape.last_mut().expect("batched input") = weight.dim(-1);
                tensor(output.reshape(&shape, context))
            }
            Some(bias) => tensor(addmm(
                bias.as_array(),
                input.as_array(),
                &weight,
                None,
                None,
                context,
            )),
            None => tensor(matmul(input.as_array(), &weight, context)),
        }
    }

    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(safemlx::fast::layer_norm(
            input.as_array(),
            weight.map(MlxTensor::as_array),
            bias.map(MlxTensor::as_array),
            epsilon,
            context,
        ))
    }

    fn gelu(input: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(nn::gelu(input.as_array(), context))
    }

    fn elu(input: &Self, alpha: f32, context: &Self::Context) -> Result<Self, Error> {
        tensor(nn::elu(input.as_array(), Some(alpha), context))
    }

    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(safemlx::fast::rope(
            input.as_array(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            None,
            context,
        ))
    }

    fn multi_axis_rotary_embeddings(
        position_ids: &Self,
        spec: &MultiAxisRotarySpec,
        context: &Self::Context,
    ) -> Result<(Self, Self), Error> {
        let _ = spec.dimensions()?;
        rotary::execute(position_ids, spec.as_ref(), None, context)
    }

    fn multi_axis_rotary_embeddings_prepared(
        position_ids: &Self,
        prepared: eredu_nn::multimodal::PreparedMultiAxisRotary<'_>,
        context: &Self::Context,
    ) -> Result<(Self, Self), Error> {
        rotary::execute(
            position_ids,
            prepared.spec(),
            Some(prepared.frequencies()),
            context,
        )
    }

    fn masked_output_projection(
        input: MaskedOutputProjectionInput<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        masked_readout::execute(input, context)
    }

    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let mask = match mask {
            AttentionMask::None => None,
            AttentionMask::Causal => Some(ScaledDotProductAttentionMask::Causal),
            AttentionMask::Tensor(mask) => {
                Some(ScaledDotProductAttentionMask::Array(mask.as_array()))
            }
        };
        tensor(scaled_dot_product_attention(
            queries.as_array(),
            keys.as_array(),
            values.as_array(),
            scale,
            mask,
            None,
            context,
        ))
    }
}

#[cfg(test)]
mod tests;

mod host_alias;
