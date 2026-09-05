use crate::nn;
use eredu_nn::{
    multimodal::{MaskedOutputProjectionInput, MultiAxisRotaryLayout, MultiAxisRotarySpec},
    AttentionMask, Error, Index, PadMode, Tensor,
};
use ref_cast::RefCast;
use safemlx::{
    argmin_axis,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    ops::{
        addmm, argpartition_axis, concatenate_axis, conv1d, conv2d, conv_transpose1d, full,
        indexing::{put_along_axis, ArrayIndex, ArrayIndexOp, NewAxis, TryIndexOp},
        matmul, maximum, pad, softmax_axis, stack_axis, sum_axis, PadMode as MlxPadMode,
    },
    Array, Dtype, Stream,
};
use smallvec::SmallVec;

fn backend<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, Error> {
    result.map_err(Error::backend)
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
        tensor(Array::from_slice(values, shape).copy(context))
    }

    fn from_i32_slice(
        values: &[i32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        tensor(Array::from_slice(values, shape).copy(context))
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
            .map_err(Error::backend)
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
        evaluated.try_to_vec::<i32>().map_err(Error::backend)
    }

    fn full_f32(value: f32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::full::<f32>(shape, Array::from_f32(value), context))
    }

    fn full_i32(value: i32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::full::<i32>(shape, Array::from_int(value), context))
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
            Array::from_f32(rhs),
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
            Array::from_f32(rhs),
            context,
        ))
    }

    fn maximum_i32(&self, rhs: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::maximum(
            self.as_array(),
            Array::from_int(rhs),
            context,
        ))
    }

    fn clip(&self, minimum: &Self, maximum: &Self, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::clip(
            self.as_array(),
            (minimum.as_array(), maximum.as_array()),
            context,
        ))
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

    fn take_axis(&self, indexes: &Self, axis: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(Array::take_axis(
            self.as_array(),
            indexes.as_array(),
            axis,
            context,
        ))
    }

    fn zeros_like(&self, context: &Self::Context) -> Result<Self, Error> {
        tensor(safemlx::ops::zeros_like(self.as_array(), context))
    }

    fn equal_i32(&self, value: i32, context: &Self::Context) -> Result<Self, Error> {
        tensor(self.as_array().eq(Array::from_int(value), context))
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
        tensor(conv1d(
            input.as_array(),
            weight.as_array(),
            stride,
            padding,
            dilation,
            groups,
            context,
        ))
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
        tensor(conv2d(
            input.as_array(),
            weight.as_array(),
            Some(stride),
            Some(padding),
            Some(dilation),
            Some(groups),
            context,
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
        let dimensions = spec.dimensions()?;
        let position_shape = position_ids.shape();
        let axes = spec.axes.len() as i32;
        let rows = position_shape[..position_shape.len() - 1].iter().try_fold(
            1_i32,
            |rows, dimension| {
                rows.checked_mul(*dimension)
                    .ok_or_else(|| Error::backend("multi-axis position dimensions overflowed i32"))
            },
        )?;
        let positions = backend(position_ids.as_array().reshape(&[rows, axes], context))?;
        let mut axis_angles = Vec::with_capacity(spec.axes.len());
        for (axis_index, axis) in spec.axes.iter().enumerate() {
            let inv = (0..axis.dimensions)
                .step_by(2)
                .map(|index| 1.0 / spec.base.powf(index as f32 / axis.dimensions as f32))
                .collect::<Vec<_>>();
            let inv = backend(Array::from_slice(&inv, &[1, inv.len() as i32]).copy(context))?;
            let positions = backend(positions.try_index_device((.., axis_index as i32), context))?;
            let positions = backend(positions.add(Array::from_int(axis.position_offset), context))?;
            let positions = backend(maximum(
                positions,
                Array::from_int(spec.minimum_position),
                context,
            ))?;
            let positions = backend(positions.as_dtype(Dtype::Float32, context))?;
            let positions = backend(positions.expand_dims(-1, context))?;
            axis_angles.push(backend(positions.multiply(inv, context))?);
        }
        let angles = match spec.layout {
            MultiAxisRotaryLayout::IndependentAxes => {
                let mut expanded = Vec::with_capacity(axis_angles.len());
                for angles in axis_angles {
                    expanded.push(backend(concatenate_axis(
                        &[angles.clone(), angles],
                        -1,
                        context,
                    ))?);
                }
                backend(concatenate_axis(&expanded, -1, context))?
            }
            MultiAxisRotaryLayout::SplitHalves => {
                let half = backend(concatenate_axis(&axis_angles, -1, context))?;
                backend(concatenate_axis(&[half.clone(), half], -1, context))?
            }
            MultiAxisRotaryLayout::RoundRobinSections => {
                let half = dimensions / 2;
                let axis_count = spec.axes.len();
                let mut selected = Vec::with_capacity(half as usize);
                for frequency in 0..half {
                    let candidate = frequency as usize % axis_count;
                    let section = spec.axes[candidate].dimensions / 2;
                    let axis = if candidate != 0 && frequency < section * axis_count as i32 {
                        candidate
                    } else {
                        0
                    };
                    let positions =
                        backend(positions.try_index_device((.., axis as i32), context))?;
                    let positions = backend(
                        positions.add(Array::from_int(spec.axes[axis].position_offset), context),
                    )?;
                    let positions = backend(maximum(
                        positions,
                        Array::from_int(spec.minimum_position),
                        context,
                    ))?;
                    selected.push(backend(positions.expand_dims(-1, context))?);
                }
                let selected = backend(concatenate_axis(&selected, -1, context))?;
                let inv = (0..half)
                    .map(|index| 1.0 / spec.base.powf(2.0 * index as f32 / dimensions as f32))
                    .collect::<Vec<_>>();
                let inv = backend(Array::from_slice(&inv, &[1, half]).copy(context))?;
                let selected = backend(selected.as_dtype(Dtype::Float32, context))?;
                let half = backend(selected.multiply(inv, context))?;
                backend(concatenate_axis(&[half.clone(), half], -1, context))?
            }
        };
        let mut output_shape = position_shape[..position_shape.len() - 1].to_vec();
        output_shape.push(dimensions);
        let cosine = backend(angles.cos(context))?;
        let cosine = backend(cosine.reshape(&output_shape, context))?;
        let sine = backend(angles.sin(context))?;
        let sine = backend(sine.reshape(&output_shape, context))?;
        Ok((MlxTensor::from_array(cosine), MlxTensor::from_array(sine)))
    }

    fn masked_output_projection(
        input: MaskedOutputProjectionInput<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let hidden_shape = input.hidden.shape();
        let batch = hidden_shape[0];
        let sequence = hidden_shape[1];
        let hidden_size = hidden_shape[2];
        let vocabulary = input.output_weight.shape()[0];
        let centroids = input.centroid_logits.shape()[2];
        let per_centroid = vocabulary / centroids;
        let top_indices = backend(argpartition_axis(
            input.centroid_logits.as_array(),
            -input.top_centroids,
            -1,
            context,
        ))?;
        let top_indices =
            backend(top_indices.try_index_device((.., .., -input.top_centroids..), context))?;
        let ordering = backend(
            input
                .token_ordering
                .as_array()
                .reshape(&[centroids, per_centroid], context),
        )?;
        let selected_tokens = backend(ordering.try_index_device(&top_indices, context))?;
        let flat_tokens = backend(selected_tokens.reshape(&[-1], context))?;
        let selected_weight = backend(
            input
                .output_weight
                .as_array()
                .try_index_device(&flat_tokens, context),
        )?;
        let selected_weight = backend(selected_weight.reshape(
            &[
                batch,
                sequence,
                input.top_centroids * per_centroid,
                hidden_size,
            ],
            context,
        ))?;
        let hidden = backend(
            input
                .hidden
                .as_array()
                .try_index_device((.., .., NewAxis, ..), context),
        )?;
        let selected_weight = backend(selected_weight.transpose_axes(&[0, 1, 3, 2], context))?;
        let selected_logits = backend(matmul(hidden, selected_weight, context))?;
        let selected_logits = backend(selected_logits.squeeze_axes(&[-2], context))?;
        let minimum = backend(selected_logits.min(None, context))?;
        let masked_value = backend(minimum.subtract(Array::from_f32(input.mask_margin), context))?;
        let output = backend(full::<f32>(
            &[batch, sequence, vocabulary],
            masked_value,
            context,
        ))?;
        let scatter_indices = backend(selected_tokens.reshape(&[batch, sequence, -1], context))?;
        tensor(put_along_axis(
            output,
            scatter_indices,
            selected_logits,
            -1,
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
