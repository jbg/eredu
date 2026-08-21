use crate::{
    multimodal::{MaskedOutputProjectionInput, MultiAxisRotaryLayout, MultiAxisRotarySpec},
    AttentionMask, Error, Index, PadMode, Tensor,
};
use safemlx::{
    argmin_axis,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    nn,
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

impl Tensor for Array {
    type Context = Stream;

    fn shape(&self) -> &[i32] {
        Array::shape(self)
    }

    fn unloaded_f32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::zeros::<f32>(shape, context))
    }

    fn unloaded_i32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::zeros::<i32>(shape, context))
    }

    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(Array::from_slice(values, shape).copy(context))
    }

    fn from_i32_slice(
        values: &[i32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(Array::from_slice(values, shape).copy(context))
    }

    fn full_f32(value: f32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::full::<f32>(shape, Array::from_f32(value), context))
    }

    fn full_i32(value: i32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::full::<i32>(shape, Array::from_int(value), context))
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

    fn tanh(&self, context: &Self::Context) -> Result<Self, Error> {
        backend(safemlx::ops::tanh(self, context))
    }

    fn maximum_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error> {
        backend(safemlx::ops::maximum(self, Array::from_f32(rhs), context))
    }

    fn clip(&self, minimum: &Self, maximum: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(safemlx::ops::clip(self, (minimum, maximum), context))
    }

    fn softmax_axis(
        &self,
        axis: i32,
        precise: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(softmax_axis(self, axis, precise, context))
    }

    fn reshape(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(Array::reshape(self, shape, context))
    }

    fn broadcast_to(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        backend(safemlx::ops::broadcast_to(self, shape, context))
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

    fn zeros_like(&self, context: &Self::Context) -> Result<Self, Error> {
        backend(safemlx::ops::zeros_like(self, context))
    }

    fn equal_i32(&self, value: i32, context: &Self::Context) -> Result<Self, Error> {
        backend(self.eq(Array::from_int(value), context))
    }

    fn logical_or(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        backend(Array::logical_or(self, rhs, context))
    }

    fn masked_scatter(
        &self,
        mask: &Self,
        source: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(safemlx::ops::indexing::masked_scatter(
            self, mask, source, context,
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
        backend(safemlx::fast::rope(
            self,
            dimensions,
            traditional,
            None::<f32>,
            1.0,
            offset,
            frequencies,
            context,
        ))
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

    fn mean_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(safemlx::ops::mean_axis(value, axis, keep_dims, context))
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

    fn conv2d(
        input: &Self,
        weight: &Self,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        backend(conv2d(
            input,
            weight,
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
        let positions = backend(position_ids.reshape(&[rows, axes], context))?;
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
        Ok((cosine, sine))
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
            input.centroid_logits,
            -input.top_centroids,
            -1,
            context,
        ))?;
        let top_indices =
            backend(top_indices.try_index_device((.., .., -input.top_centroids..), context))?;
        let ordering = backend(
            input
                .token_ordering
                .reshape(&[centroids, per_centroid], context),
        )?;
        let selected_tokens = backend(ordering.try_index_device(&top_indices, context))?;
        let flat_tokens = backend(selected_tokens.reshape(&[-1], context))?;
        let selected_weight = backend(input.output_weight.try_index_device(&flat_tokens, context))?;
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
        backend(put_along_axis(
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
            AttentionMask::Tensor(mask) => Some(ScaledDotProductAttentionMask::Array(mask)),
        };
        backend(scaled_dot_product_attention(
            queries, keys, values, scale, mask, None, context,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::{
        masked_output_projection, multi_axis_rotary_embeddings, project_flattened_patches,
        reference_flattened_patch_projection, reference_masked_output_projection,
        reference_multi_axis_rotary_embeddings, FlattenedPatchSpec, MaskedOutputProjectionInput,
        MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    };
    use safemlx::{Device, DeviceType, ExecutionContext};

    fn close(actual: &Array, expected: &[f32]) {
        let actual = actual.evaluated().unwrap();
        assert_eq!(actual.as_slice::<f32>().len(), expected.len());
        assert!(actual
            .as_slice::<f32>()
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() < 1e-5));
    }

    #[test]
    #[ignore = "explicit MLX patch-projection parity; run outside the sandbox"]
    fn mlx_flattened_patch_projection_matches_scalar_reference() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let input_values = [1.0, 2.0, 3.0, 4.0];
        let weight_values = [1.0, 0.5, -1.0, 2.0];
        let bias_values = [0.25, -0.5];
        let input = Array::from_slice(&input_values, &[2, 2]);
        let weight = Array::from_slice(&weight_values, &[2, 1, 1, 1, 2]);
        let bias = Array::from_slice(&bias_values, &[2]);
        let actual = project_flattened_patches(
            &input,
            &weight,
            Some(&bias),
            FlattenedPatchSpec {
                channels: 1,
                temporal: 1,
                height: 1,
                width: 2,
                output: 2,
            },
            stream,
        )
        .unwrap();
        let expected = reference_flattened_patch_projection(
            &input_values,
            2,
            &weight_values,
            2,
            Some(&bias_values),
        )
        .unwrap();
        close(&actual, &expected);
    }

    #[test]
    #[ignore = "explicit MLX multi-axis rotary parity; run outside the sandbox"]
    fn mlx_multi_axis_rotary_matches_scalar_reference() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let position_values = [-1, 2, 3, 4];
        let positions = Array::from_slice(&position_values, &[2, 2]);
        let spec = MultiAxisRotarySpec {
            axes: vec![
                RotaryAxisSpec {
                    dimensions: 4,
                    position_offset: 0,
                },
                RotaryAxisSpec {
                    dimensions: 4,
                    position_offset: 1,
                },
            ],
            base: 100.0,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::RoundRobinSections,
        };
        let (actual_cosine, actual_sine) =
            multi_axis_rotary_embeddings(&positions, &spec, stream).unwrap();
        let (expected_cosine, expected_sine) =
            reference_multi_axis_rotary_embeddings(&position_values, 2, &spec).unwrap();
        close(&actual_cosine, &expected_cosine);
        close(&actual_sine, &expected_sine);
    }

    #[test]
    #[ignore = "explicit MLX masked-output parity; run outside the sandbox"]
    fn mlx_masked_output_projection_matches_scalar_reference() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let hidden_values = [2.0, 1.0];
        let weight_values = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
        let centroid_values = [0.1, 0.9];
        let ordering_values = [2, 0, 3, 1];
        let hidden = Array::from_slice(&hidden_values, &[1, 1, 2]);
        let weight = Array::from_slice(&weight_values, &[4, 2]);
        let centroids = Array::from_slice(&centroid_values, &[1, 1, 2]);
        let ordering = Array::from_slice(&ordering_values, &[4]);
        let actual = masked_output_projection(
            MaskedOutputProjectionInput {
                hidden: &hidden,
                output_weight: &weight,
                centroid_logits: &centroids,
                token_ordering: &ordering,
                top_centroids: 1,
                mask_margin: 1.0,
            },
            stream,
        )
        .unwrap();
        let expected = reference_masked_output_projection(
            &hidden_values,
            1,
            2,
            &weight_values,
            4,
            &centroid_values,
            2,
            &[2, 0, 3, 1],
            1,
            1.0,
        )
        .unwrap();
        close(&actual, &expected);
    }
}
