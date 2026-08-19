use crate::{AttentionMask, Error, Index, PadMode, Tensor};
use safemlx::{
    argmin_axis,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn,
    ops::{
        addmm, concatenate_axis, conv1d, conv_transpose1d,
        indexing::{ArrayIndex, ArrayIndexOp, TryIndexOp},
        matmul, pad, stack_axis, sum_axis, PadMode as MlxPadMode,
    },
    Array, Stream,
};
use smallvec::SmallVec;

fn backend<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, Error> {
    result.map_err(Error::backend)
}

impl Tensor for Array {
    type Context = Stream;

    fn shape(&self) -> &[i32] {
        Array::shape(self)
    }

    fn unloaded_f32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::zeros::<f32>(shape, context))
    }

    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(Array::from_slice(values, shape).copy(context))
    }

    fn add(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::add(self, rhs, context))
    }

    fn subtract(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::subtract(self, rhs, context))
    }

    fn multiply(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::multiply(self, rhs, context))
    }

    fn multiply_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::multiply(self, Array::from_f32(rhs), context))
    }

    fn divide(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::divide(self, rhs, context))
    }

    fn square(&self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::square(self, context))
    }

    fn maximum_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error> {
        backend(safemlx::ops::maximum(self, Array::from_f32(rhs), context))
    }

    fn reshape(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::reshape(self, shape, context))
    }

    fn transpose_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::transpose_axes(self, axes, context))
    }

    fn swap_axes(&self, left: i32, right: i32, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::swap_axes(self, left, right, context))
    }

    fn transpose(&self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::transpose(self, context))
    }

    fn expand_dims(&self, axis: i32, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::expand_dims(self, axis, context))
    }

    fn squeeze_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::squeeze_axes(self, axes, context))
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
        backend(self.try_index_device(indexes.as_slice(), context))
    }

    fn take_axis(&self, indexes: &Self, axis: i32, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::take_axis(self, indexes, axis, context))
    }

    fn concatenate(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error> {
        backend(concatenate_axis(values, axis, context))
    }

    fn stack(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error> {
        backend(stack_axis(values, axis, context))
    }

    fn matmul(lhs: &Self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(matmul(lhs, rhs, context))
    }

    fn sum_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(sum_axis(value, axis, keep_dims, context))
    }

    fn argmin_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(argmin_axis!(
            value,
            axis,
            keep_dims = keep_dims,
            stream = context
        ))
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
        backend(pad(value, widths, None::<Array>, Some(mode), context))
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
        backend(conv1d(
            input, weight, stride, padding, dilation, groups, context,
        ))
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
        backend(conv_transpose1d(
            input,
            weight,
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
        let weight = backend(weight.transpose(context))?;
        match bias {
            Some(bias) => backend(addmm(bias, input, &weight, None, None, context)),
            None => backend(matmul(input, &weight, context)),
        }
    }

    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(safemlx::fast::layer_norm(
            input, weight, bias, epsilon, context,
        ))
    }

    fn gelu(input: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(nn::gelu(input, context))
    }

    fn elu(input: &Self, alpha: f32, context: &Self::Context) -> Result<Self, Error> {
        backend(nn::elu(input, Some(alpha), context))
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
        backend(safemlx::fast::rope(
            input,
            dimensions,
            traditional,
            base,
            scale,
            offset,
            None,
            context,
        ))
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
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
            AttentionMask::Tensor(mask) => ScaledDotProductAttentionMask::Array(mask),
        };
        backend(scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            Some(mask),
            None,
            context,
        ))
    }
}
