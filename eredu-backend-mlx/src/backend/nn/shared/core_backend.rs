use super::*;
use super::{operators::*, parameters::*};

/// MLX failure while lowering a neutral checkpoint lease or recipe.
#[derive(Debug, thiserror::Error)]
pub enum MlxParameterError {
    /// Neutral lease conversion or MLX checkpoint submission failed.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),
    /// A neutral derived-weight recipe could not be lowered.
    #[error(transparent)]
    Recipe(#[from] crate::backend::runtime::checkpoint::recipe::WeightRecipeError),
    /// Final stream-to-stream weight copy failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

impl ParameterBackend for MlxNeuralBackend {
    type Parameter = MlxTensor;
    type MaterializedWeight = MlxTensor;
    type MaterializationContext =
        crate::backend::runtime::checkpoint::store::MlxParameterMaterializationContext;
    type Materialization = crate::backend::runtime::checkpoint::store::WeightMaterialization;
    type ParameterError = MlxParameterError;

    fn preflight_recipe(
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<(), Self::ParameterError> {
        crate::backend::runtime::checkpoint::recipe::preflight_mlx_recipe(recipe, source)?;
        Ok(())
    }

    fn materialize(
        lease: eredu_checkpoint::store::CheckpointLease,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError> {
        Ok(context
            .weight_lease(lease)?
            .materialize(context.source_stream(), context.execution_stream())?)
    }

    fn materialize_recipe(
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError> {
        use crate::backend::runtime::checkpoint::recipe::MlxWeightRecipeExt;

        let pending = recipe.prepare_materialization(source, context)?;
        let (output, sources) = pending.into_parts();
        let prepared =
            crate::backend::runtime::checkpoint::store::WeightMaterialization::prepare_retained(
                vec![output],
                sources,
            )?;
        let output = if context.source_stream() == context.execution_stream() {
            prepared.inputs()[0].clone()
        } else {
            prepared.inputs()[0].copy(context.execution_stream())?
        };
        Ok(prepared.submit_outputs(vec![output])?)
    }

    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight {
        MlxTensor::ref_cast(materialization.output())
    }

    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(MlxTensor::from_array(materialization.synchronize()?))
    }

    fn share_materialized_weight(
        weight: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(weight.clone())
    }

    fn validate_bind(
        parameter: &Self::Parameter,
        weight: &Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError> {
        if parameter.as_array().shape() != weight.as_array().shape() {
            return Err(MlxParameterError::Mlx(safemlx::error::Exception::custom(
                format!(
                    "parameter shape {:?} does not match materialized weight {:?}",
                    parameter.as_array().shape(),
                    weight.as_array().shape()
                ),
            )));
        }
        Ok(())
    }

    fn bind(parameter: &mut Self::Parameter, weight: Self::MaterializedWeight) {
        *parameter = weight;
    }
}

impl NeuralBackend for MlxNeuralBackend {
    const OPERATOR_CAPABILITIES: eredu_nn::NeuralOperatorCapabilities =
        eredu_nn::NeuralOperatorCapabilities::ALL;

    type Tensor = MlxTensor;
    type Linear = MlxLinear;
    type Embedding = MlxEmbedding;
    type Normalization = MlxRmsNorm;
    type Rotary = MlxRotary;
    type ParallelContext = Group;

    fn linear(spec: LinearSpec, context: &Stream) -> Result<MlxLinear, ComputeError> {
        let module = compute(common::linear::PhysicalLinear::unloaded(
            spec.input,
            spec.output,
            spec.bias.is_some(),
            spec.format.encoding(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias, &spec.format)?;
        Ok(MlxLinear {
            module,
            topology,
            vocabulary_range: None,
        })
    }

    fn embedding(spec: EmbeddingSpec, context: &Stream) -> Result<MlxEmbedding, ComputeError> {
        let module = compute(common::linear::unloaded_embedding(
            spec.vocabulary,
            spec.dimensions,
            spec.format.encoding().weight_quantization(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None, &spec.format)?;
        Ok(MlxEmbedding {
            module,
            topology,
            vocabulary: spec.vocabulary,
            vocabulary_range: None,
        })
    }

    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &Stream,
    ) -> Result<MlxRmsNorm, ComputeError> {
        spec.validate()?;
        let (weight, offset) = match spec.scale {
            NormalizationScale::Learned(weight) => (Some(weight), None),
            NormalizationScale::LearnedOffset { weight, offset } => (Some(weight), Some(offset)),
            NormalizationScale::Unit => (None, None),
        };
        let (module, topology) = match weight {
            Some(weight) => {
                let module = compute(nn::RmsNorm::unloaded(
                    spec.dimensions,
                    spec.epsilon,
                    Dtype::Float32,
                    context,
                ))?;
                let topology = exact_parameter_topology(&module, [("weight", weight)])?;
                (Some(module), topology)
            }
            None => (None, BTreeMap::new()),
        };
        Ok(MlxRmsNorm {
            module,
            topology,
            offset,
            dimensions: spec.dimensions,
            epsilon: spec.epsilon,
        })
    }

    fn rotary(spec: RotarySpec, context: &Stream) -> Result<MlxRotary, ComputeError> {
        compute(rope::initialize_rope(
            spec.dimensions,
            spec.base,
            spec.traditional,
            spec.algorithm,
            context,
        ))
        .map(MlxRotary)
    }

    fn silu(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(common::layers::silu(input.into_array(), context))
    }

    fn gelu_approximate(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(nn::gelu_approximate(input.into_array(), context))
    }

    fn sigmoid(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(safemlx::ops::sigmoid(input.into_array(), context))
    }

    fn softplus(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(nn::softplus(input.into_array(), context))
    }

    fn exp(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(safemlx::ops::exp(input.into_array(), context))
    }

    fn gated_group_rms_norm(
        input: &MlxTensor,
        gate: &MlxTensor,
        weight: &MlxTensor,
        groups: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        let gate = gate.as_array();
        let weight = weight.as_array();
        let dtype = input.dtype();
        let shape = input.shape().to_vec();
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            &shape,
            gate.shape(),
            groups,
            epsilon,
        )?;
        let width = geometry.width();
        if weight.shape() != [width] {
            return Err(ComputeError::backend(
                "invalid gated grouped RMS normalization geometry",
            ));
        }
        let input = compute(input.as_dtype(Dtype::Float32, context))?;
        let gate = compute(gate.as_dtype(Dtype::Float32, context))?;
        let gate =
            compute(gate.multiply(compute(safemlx::ops::sigmoid(&gate, context))?, context))?;
        let gated = compute(input.multiply(&gate, context))?;
        let grouped = compute(gated.reshape(&[-1, groups, width / groups], context))?;
        let variance = compute(safemlx::ops::mean_axis(
            compute(grouped.square(context))?,
            -1,
            true,
            context,
        ))?;
        let scale = compute(safemlx::ops::rsqrt(
            compute(variance.add(Array::from_f32(epsilon), context))?,
            context,
        ))?;
        let normalized = compute(grouped.multiply(&scale, context))?;
        let normalized = compute(normalized.reshape(&shape, context))?;
        let normalized = compute(normalized.as_dtype(dtype, context))?;
        compute_tensor(normalized.multiply(weight, context))
    }

    fn l2_normalize(
        input: &MlxTensor,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        eredu_nn::operation_geometry::NormalizationGeometry::new(input.shape(), epsilon)?;
        let squared = compute(input.square(context))?;
        let sum = compute(safemlx::ops::sum_axis(&squared, -1, true, context))?;
        let denominator = compute(sum.add(Array::from_f32(epsilon), context))?;
        compute_tensor(input.multiply(compute(denominator.rsqrt(context))?, context))
    }

    fn silu_gated_group_rms_norm(
        input: &MlxTensor,
        gate: &MlxTensor,
        weight: &MlxTensor,
        groups: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        let gate = gate.as_array();
        let weight = weight.as_array();
        let dtype = input.dtype();
        let shape = input.shape().to_vec();
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            &shape,
            gate.shape(),
            groups,
            epsilon,
        )?;
        let width = geometry.width();
        if weight.shape() != [width] {
            return Err(ComputeError::backend(
                "invalid SiLU-gated grouped RMS normalization geometry",
            ));
        }
        let input = compute(input.as_dtype(Dtype::Float32, context))?;
        let grouped = compute(input.reshape(&[-1, groups, width / groups], context))?;
        let variance = compute(safemlx::ops::mean_axis(
            compute(grouped.square(context))?,
            -1,
            true,
            context,
        ))?;
        let scale = compute(safemlx::ops::rsqrt(
            compute(variance.add(Array::from_f32(epsilon), context))?,
            context,
        ))?;
        let normalized = compute(grouped.multiply(&scale, context))?;
        let normalized = compute(normalized.reshape(&shape, context))?;
        let normalized = compute(normalized.multiply(weight, context))?;
        let gate = compute(gate.as_dtype(Dtype::Float32, context))?;
        let gate =
            compute(gate.multiply(compute(safemlx::ops::sigmoid(&gate, context))?, context))?;
        compute(normalized.multiply(&gate, context))?
            .as_dtype(dtype, context)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend)
    }

    fn segmented_attention(
        input: SegmentedAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        input.validate()?;
        let heads = input.queries.dim(1);
        let query_dimensions = input.queries.dim(2);
        let value_dimensions = input.values.dim(2);
        let mut outputs = Vec::with_capacity(input.segment_lengths.len());
        let mut start = 0i32;
        for &length in input.segment_lengths {
            let end = start + length;
            let prepare = |value: &Array, dimensions: i32| -> Result<Array, ComputeError> {
                let value = compute(value.try_index_device((start..end, .., ..), context))?;
                let value = compute(value.transpose_axes(&[1, 0, 2], context))?;
                compute(value.reshape(&[1, heads, length, dimensions], context))
            };
            let queries = prepare(input.queries.as_array(), query_dimensions)?;
            let keys = prepare(input.keys.as_array(), query_dimensions)?;
            let values = prepare(input.values.as_array(), value_dimensions)?;
            let output = compute(safemlx::fast::scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                input.scale,
                Option::<ScaledDotProductAttentionMask<'_>>::None,
                Option::<&Array>::None,
                context,
            ))?;
            let output = compute(output.reshape(&[heads, length, value_dimensions], context))?;
            outputs.push(compute(output.transpose_axes(&[1, 0, 2], context))?);
            start = end;
        }
        compute_tensor(concatenate_axis(&outputs, 0, context))
    }

    fn add_residual(
        residual: &MlxTensor,
        branch: &MlxTensor,
        fp32: bool,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let residual = residual.as_array();
        let branch = branch.as_array();
        if fp32 {
            let residual = compute(residual.as_dtype(Dtype::Float32, context))?;
            let branch = compute(branch.as_dtype(Dtype::Float32, context))?;
            compute_tensor(residual.add(branch, context))
        } else {
            compute_tensor(residual.add(branch, context))
        }
    }

    fn gated_delta_scan(
        input: GatedDeltaScanInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<GatedDeltaScanOutput<MlxTensor>, ComputeError> {
        let (state, output) = compute(crate::backend::nn::gated_delta::gated_delta_scan(
            input.query.as_array(),
            input.key.as_array(),
            input.value.as_array(),
            input.log_decay.as_array(),
            input.beta.as_array(),
            input.initial_state.map(|state| state.as_array().clone()),
            context,
        ))?;
        Ok(GatedDeltaScanOutput {
            state: MlxTensor::from_array(state),
            output: MlxTensor::from_array(output),
        })
    }

    fn selective_state_space_scan(
        input: SelectiveStateSpaceScanInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<SelectiveStateSpaceScanOutput<MlxTensor>, ComputeError> {
        let shape = input.values.shape();
        if shape.len() != 4 || input.chunk_size == 0 {
            return Err(ComputeError::backend(
                "selective state-space scan expects rank-four values and nonzero chunks",
            ));
        }
        let [batch, sequence, heads, head_dimensions] =
            <[i32; 4]>::try_from(shape).expect("validated rank-four shape");
        let state_dimensions = input.input_state.dim(3);
        let mut state = match input.initial_state {
            Some(state) => compute(state.as_array().as_dtype(Dtype::Float32, context))?,
            None => compute(safemlx::ops::zeros::<f32>(
                &[batch, heads, head_dimensions, state_dimensions],
                context,
            ))?,
        };
        let values = compute(input.values.as_array().as_dtype(Dtype::Float32, context))?;
        let input_state = compute(
            input
                .input_state
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?;
        let output_state = compute(
            input
                .output_state
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?;
        let transition = compute(safemlx::ops::exp(
            compute(
                input
                    .transition_log
                    .as_array()
                    .as_dtype(Dtype::Float32, context),
            )?,
            context,
        ))?;
        let transition = compute(transition.multiply(Array::from_f32(-1.0), context))?
            .reshape(&[1, heads, 1, 1], context)
            .map_err(ComputeError::backend)?;
        let skip = compute(input.skip.as_array().as_dtype(Dtype::Float32, context))?
            .reshape(&[1, heads, 1], context)
            .map_err(ComputeError::backend)?;
        let bias = compute(
            input
                .time_step_bias
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?
        .reshape(&[1, 1, heads], context)
        .map_err(ComputeError::backend)?;
        let mut outputs = Vec::with_capacity(sequence as usize);
        let chunk = i32::try_from(input.chunk_size).unwrap_or(i32::MAX).max(1);
        let mut chunk_start = 0;
        while chunk_start < sequence {
            let chunk_end = (chunk_start + chunk).min(sequence);
            for token in chunk_start..chunk_end {
                let value = compute(values.try_index_device((.., token, .., ..), context))?;
                let b = compute(input_state.try_index_device((.., token, .., ..), context))?;
                let c = compute(output_state.try_index_device((.., token, .., ..), context))?;
                let dt = compute(
                    input
                        .time_step
                        .as_array()
                        .try_index_device((.., token..token + 1, ..), context),
                )?;
                let dt = compute(dt.add(&bias, context))?;
                let dt = compute(nn::softplus(dt, context))?;
                let floor = Array::from_f32(input.time_step_floor);
                let dt = compute(maximum(dt, floor, context))?;
                let dt = compute(dt.as_dtype(Dtype::Float32, context))?
                    .reshape(&[batch, heads], context)
                    .map_err(ComputeError::backend)?;
                let dt_transition = compute(dt.reshape(&[batch, heads, 1, 1], context))?;
                let decay = compute(safemlx::ops::exp(
                    compute(dt_transition.multiply(&transition, context))?,
                    context,
                ))?;
                let dt_input = compute(dt.reshape(&[batch, heads, 1], context))?;
                let discretized_b = compute(dt_input.multiply(&b, context))?;
                let value_column = compute(value.try_index_device((.., .., .., NewAxis), context))?;
                let input_row =
                    compute(discretized_b.try_index_device((.., .., NewAxis, ..), context))?;
                let update = compute(value_column.multiply(&input_row, context))?;
                state = compute(compute(state.multiply(&decay, context))?.add(&update, context))?;
                let output_row = compute(c.try_index_device((.., .., NewAxis, ..), context))?;
                let projected = compute(safemlx::ops::sum_axis(
                    compute(state.multiply(&output_row, context))?,
                    -1,
                    false,
                    context,
                ))?;
                let output =
                    compute(projected.add(&compute(value.multiply(&skip, context))?, context))?;
                outputs.push(compute(
                    output.try_index_device((.., NewAxis, .., ..), context),
                )?);
            }
            chunk_start = chunk_end;
        }
        let output = compute(safemlx::ops::concatenate_axis(&outputs, 1, context))?;
        let output = compute(output.as_dtype(input.values.as_array().dtype(), context))?;
        Ok(SelectiveStateSpaceScanOutput {
            state: MlxTensor::from_array(state),
            output: MlxTensor::from_array(output),
        })
    }

    fn gated_product(
        gate: MlxTensor,
        up: MlxTensor,
        policy: GatedProductPolicy,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let mut gate = gate.into_array();
        let mut up = up.into_array();
        policy.validate()?;
        if let Some(bound) = policy.gate_upper_bound() {
            gate = compute(safemlx::ops::minimum(gate, Array::from_f32(bound), context))?;
        }
        if let Some(bound) = policy.up_absolute_bound() {
            up = compute(safemlx::ops::clip(up, (-bound, bound), context))?;
        }
        if policy.up_offset() != 0.0 {
            up = compute(up.add(Array::from_f32(policy.up_offset()), context))?;
        }
        let gate = match policy.activation() {
            eredu_nn::GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
                compute(common::layers::silu(gate, context))?
            }
            eredu_nn::GatedProductActivation::Silu => {
                let scaled =
                    compute(gate.multiply(Array::from_f32(policy.sigmoid_multiplier()), context))?;
                let probability = compute(sigmoid(scaled, context))?;
                compute(gate.multiply(probability, context))?
            }
            eredu_nn::GatedProductActivation::GeluApproximate => {
                compute(nn::gelu_approximate(gate, context))?
            }
            _ => {
                return Err(ComputeError::backend(
                    "unsupported grouped gated-product activation",
                ));
            }
        };
        compute_tensor(gate.multiply(up, context))
    }

    fn attention(
        queries: MlxTensor,
        keys: MlxTensor,
        values: MlxTensor,
        scale: f32,
        mask: Option<&MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        // MLX requires additive masks to promote to the Q/K/V output dtype.
        // Keep boolean masks boolean: casting them would change their meaning.
        let mask = mask
            .map(|mask| {
                let mask = mask.as_array();
                if mask.dtype() == Dtype::Bool {
                    Ok(mask.clone())
                } else {
                    let dtype = Dtype::from_promoting_types(
                        Dtype::from_promoting_types(
                            queries.as_array().dtype(),
                            keys.as_array().dtype(),
                        ),
                        values.as_array().dtype(),
                    );
                    compute(mask.as_dtype(dtype, context))
                }
            })
            .transpose()?;
        compute_tensor(safemlx::fast::scaled_dot_product_attention(
            queries.into_array(),
            keys.into_array(),
            values.into_array(),
            scale,
            mask.as_ref().map(ScaledDotProductAttentionMask::Array),
            None,
            context,
        ))
    }

    fn relative_attention(
        input: RelativeAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        input.validate()?;
        let query_shape = input.queries.shape();
        let key_shape = input.keys.shape();
        let batch = query_shape[0];
        let heads = query_shape[1];
        let query_len = query_shape[2];
        let dimensions = query_shape[3];
        let kv_heads = key_shape[1];
        let key_len = key_shape[2];
        let repeats = heads / kv_heads;
        let repeat_kv = |value: &Array| -> Result<Array, ComputeError> {
            if repeats == 1 {
                return Ok(value.clone());
            }
            let expanded =
                compute(value.reshape(&[batch, kv_heads, 1, key_len, dimensions], context))?;
            let expanded = compute(broadcast_to(
                &expanded,
                &[batch, kv_heads, repeats, key_len, dimensions],
                context,
            ))?;
            compute(expanded.reshape(&[batch, heads, key_len, dimensions], context))
        };
        let keys = repeat_kv(input.keys.as_array())?;
        let values = repeat_kv(input.values.as_array())?;
        let query_positions = compute(arange::<i32, i32>(
            input.query_offset,
            input.query_offset + query_len,
            1,
            context,
        ))?;
        let query_positions = compute(query_positions.try_index_device((.., NewAxis), context))?;
        let key_positions = compute(arange::<i32, i32>(
            input.key_offset,
            input.key_offset + key_len,
            1,
            context,
        ))?;
        let key_positions = compute(key_positions.try_index_device((NewAxis, ..), context))?;
        let distances = compute(query_positions.subtract(key_positions, context))?;
        let mut valid = compute(distances.ge(Array::from_int(0), context))?;
        if let Some(window) = input.window {
            valid = compute(valid.logical_and(
                &compute(distances.lt(Array::from_int(window), context))?,
                context,
            ))?;
        }
        let extent = input.profiles.dim(3);
        let gather = compute(clip(&distances, (0, extent - 1), context))?;
        let gather = compute(gather.as_dtype(Dtype::Int32, context))?;
        let gather = compute(gather.try_index_device((NewAxis, NewAxis, .., ..), context))?;
        let gather = compute(broadcast_to(
            &gather,
            &[batch, heads, query_len, key_len],
            context,
        ))?;
        let mut bias = compute(take_along_axis(
            input.profiles.as_array(),
            &gather,
            -1,
            context,
        ))?;
        let relative_valid = compute(
            compute(distances.ge(Array::from_int(0), context))?.logical_and(
                &compute(distances.lt(Array::from_int(extent), context))?,
                context,
            ),
        )?;
        bias = compute(r#where(
            &relative_valid,
            bias,
            Array::from_f32(0.0),
            context,
        ))?;
        let mut queries = input.queries.as_array().clone();
        if input.window.is_none() {
            if let Some(floor) = input.log_scaling_floor {
                let positions = compute(arange::<i32, i32>(
                    input.query_offset + 1,
                    input.query_offset + query_len + 1,
                    1,
                    context,
                ))?;
                let positions = compute(positions.as_dtype(Dtype::Float32, context))?;
                let ratio = compute(positions.divide(Array::from_f32(floor as f32), context))?;
                let ratio = compute(maximum(ratio, Array::from_f32(1.0), context))?;
                let tau = compute(ratio.log(context))?;
                let tau = compute(tau.multiply(Array::from_f32(input.log_scaling_alpha), context))?;
                let tau = compute(tau.add(Array::from_f32(1.0), context))?;
                let tau = compute(tau.reshape(&[1, 1, query_len, 1], context))?;
                queries = compute(queries.multiply(&tau, context))?;
                bias = compute(bias.multiply(&tau, context))?;
            }
        }
        let scaled = compute(queries.multiply(Array::from_f32(1.0 / dimensions as f32), context))?;
        let scores = compute(matmul(
            &scaled,
            &compute(keys.swap_axes(-1, -2, context))?,
            context,
        ))?;
        let scores = compute(scores.add(bias, context))?;
        let scores = compute(r#where(
            &valid,
            scores,
            Array::from_f32(f32::NEG_INFINITY),
            context,
        ))?;
        let probabilities = compute(softmax_axis(scores, -1, true, context))?;
        compute_tensor(matmul(probabilities, values, context))
    }

    fn indexed_attention(
        input: IndexedAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        input.validate()?;
        compute_tensor(common::attention::indexed_sparse_attention(
            input.queries.as_array(),
            input.local_keys.as_array(),
            input.local_values.as_array(),
            input.pooled_keys.as_array(),
            input.pooled_values.as_array(),
            input.selected_positions.as_array(),
            input.scale,
            input.local_mask.map(MlxTensor::as_array),
            input.pooled_mask.map(MlxTensor::as_array),
            input.sinks.map(MlxTensor::as_array),
            context,
        ))
    }

    fn pooled_attention(
        input: PooledAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let query_tokens = input.queries.dim(2);
        let local_tokens = input.local.dim(1);
        let pooled_tokens = input.pooled.dim(1);
        let local = compute(input.local.as_array().expand_dims(1, context))?;
        let pooled = compute(input.pooled.as_array().expand_dims(1, context))?;
        let keys = compute(safemlx::ops::concatenate_axis(&[local, pooled], 2, context))?;
        let mask = if input.local_mask.is_none() && input.pooled_mask.is_none() {
            None
        } else {
            let local = match input.local_mask {
                Some(mask) => mask.as_array().clone(),
                None => compute(Array::ones::<bool>(&[query_tokens, local_tokens], context))?,
            };
            let pooled = match input.pooled_mask {
                Some(mask) => mask.as_array().clone(),
                None => compute(Array::ones::<bool>(&[query_tokens, pooled_tokens], context))?,
            };
            Some(compute(safemlx::ops::concatenate_axis(
                &[local, pooled],
                -1,
                context,
            ))?)
        };
        compute_tensor(safemlx::fast::scaled_dot_product_attention(
            input.queries.as_array(),
            &keys,
            &keys,
            input.scale,
            mask.as_ref().map(ScaledDotProductAttentionMask::Array),
            input.sinks.map(MlxTensor::as_array),
            context,
        ))
    }

    fn select_pooled_positions(
        input: PooledPositionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let query_shape = input.queries.shape();
        let pooled_shape = input.pooled_keys.shape();
        let weight_shape = input.head_weights.shape();
        if query_shape.len() != 4
            || pooled_shape.len() != 3
            || weight_shape.len() != 3
            || query_shape[0] != pooled_shape[0]
            || query_shape[0] != weight_shape[0]
            || query_shape[1] != weight_shape[2]
            || query_shape[2] != weight_shape[1]
            || query_shape[3] != pooled_shape[2]
            || input.top_k <= 0
            || !input.scale.is_finite()
            || input.scale <= 0.0
            || !input.head_scale.is_finite()
            || input.head_scale <= 0.0
        {
            return Err(ComputeError::backend(format!(
                "invalid pooled-position geometry: queries={query_shape:?} pooled={pooled_shape:?} weights={weight_shape:?} top_k={}",
                input.top_k
            )));
        }
        let scores = compute(einsum(
            "bhld,bpd->bhlp",
            [
                &compute(input.queries.as_array().as_dtype(Dtype::Float32, context))?,
                &compute(
                    input
                        .pooled_keys
                        .as_array()
                        .as_dtype(Dtype::Float32, context),
                )?,
            ],
            context,
        ))?;
        let scores = compute(maximum(scores, Array::from_f32(0.0), context))?;
        let scores = compute(scores.multiply(Array::from_f32(input.scale), context))?;
        let weights = compute(
            input
                .head_weights
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?;
        let weights = compute(weights.multiply(Array::from_f32(input.head_scale), context))?;
        let weights = compute(weights.transpose_axes(&[0, 2, 1], context))?;
        let weights = compute(weights.expand_dims(-1, context))?;
        let mut scores = compute(scores.multiply(weights, context))?;
        scores = compute(scores.sum_axis(1, false, context))?;
        if let Some(mask) = input.mask {
            scores = compute(safemlx::ops::r#where(
                mask.as_array(),
                scores,
                Array::from_f32(f32::NEG_INFINITY),
                context,
            ))?;
        }
        let top_k = input.top_k.min(pooled_shape[1]);
        let indices = compute(argpartition_axis(&scores, -top_k, -1, context))?;
        let start = indices.dim(-1) - top_k;
        compute_tensor(indices.try_index_device((.., .., start..), context))
    }

    fn gather_pooled_mask(
        mask: &MlxTensor,
        selected_positions: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let mask = mask.as_array();
        let selected_positions = selected_positions.as_array();
        if mask.ndim() != 2 || selected_positions.ndim() != 3 {
            return Err(ComputeError::backend(format!(
                "pooled mask gathering expects [query, pool] and [batch, query, selected], got {:?} and {:?}",
                mask.shape(),
                selected_positions.shape()
            )));
        }
        let expanded = compute(mask.expand_dims(0, context))?;
        let expanded = compute(broadcast_to(
            &expanded,
            &[
                selected_positions.dim(0),
                selected_positions.dim(1),
                mask.dim(1),
            ],
            context,
        ))?;
        let selected = compute(take_along_axis(&expanded, selected_positions, 2, context))?;
        compute_tensor(selected.expand_dims(1, context))
    }

    fn attention_with_sinks(
        request: AttentionRequest<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        request.validate()?;
        compute_tensor(common::attention::attention_with_softcap(
            request.queries.as_array(),
            request.keys.as_array(),
            request.values.as_array(),
            request.scale,
            request.mask.map(MlxTensor::as_array),
            request.sinks.map(MlxTensor::as_array),
            request.softcap,
            context,
        ))
    }

    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, MlxTensor>,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        request.validate()?;
        let batch = request.queries.dim(0);
        let sequence = request.queries.dim(2);
        compute_tensor(
            common::attention::sliding_window_prefill_attention_with_softcap(
                request.queries.as_array().clone(),
                request.keys.as_array().clone(),
                request.values.as_array().clone(),
                request.scale,
                window,
                position_offset,
                batch,
                sequence,
                request.sinks.map(MlxTensor::as_array),
                request.softcap,
                context,
            ),
        )
    }

    fn rms_norm_without_weight(
        input: &MlxTensor,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        mlx_weightless_rms_norm(input.as_array(), epsilon, context).map(MlxTensor::from_array)
    }

    fn rms_norm_with_weight(
        input: &MlxTensor,
        weight: &MlxTensor,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        eredu_nn::operation_geometry::NormalizationGeometry::new(input.shape(), epsilon)?;
        let output = compute(safemlx::fast::rms_norm(
            input.as_array(),
            weight.as_array(),
            epsilon,
            context,
        ))?;
        compute_tensor(output.as_dtype(input.as_array().dtype(), context))
    }

    fn sliding_window_attention(
        queries: MlxTensor,
        keys: MlxTensor,
        values: MlxTensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let batch = queries.dim(0);
        let sequence = queries.dim(2);
        compute_tensor(common::attention::sliding_window_prefill_attention(
            queries.as_array().clone(),
            keys.as_array().clone(),
            values.as_array().clone(),
            scale,
            window,
            position_offset,
            batch,
            sequence,
            None,
            context,
        ))
    }

    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(crate::backend::nn::tensor::create_causal_mask(
            sequence,
            Some(offset),
            window,
            None,
            context,
        ))
    }

    fn row_parallel_linear(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(
            linear
                .module
                .forward_row_parallel(input.as_array(), parallel, context),
        )
    }

    fn parallel_size(parallel: &Group) -> usize {
        parallel.size()
    }
}

impl eredu_nn::DistributedNeuralBackend for MlxNeuralBackend {
    fn vocabulary_parallel_embedding(
        spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        context: &Stream,
    ) -> Result<MlxEmbedding, ComputeError> {
        range.validate_global_rows(spec.vocabulary)?;
        let global = i32::try_from(range.global_vocabulary).map_err(ComputeError::backend)?;
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend)?;
        let module = compute(common::linear::unloaded_embedding(
            local,
            spec.dimensions,
            spec.format.encoding().weight_quantization(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None, &spec.format)?;
        Ok(MlxEmbedding {
            module,
            topology,
            vocabulary: global,
            vocabulary_range: Some(range),
        })
    }

    fn vocabulary_parallel_linear(
        spec: LinearSpec,
        range: VocabularyParallelRange,
        context: &Stream,
    ) -> Result<MlxLinear, ComputeError> {
        range.validate_global_rows(spec.output)?;
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend)?;
        let module = compute(common::linear::PhysicalLinear::unloaded(
            spec.input,
            local,
            spec.bias.is_some(),
            spec.format.encoding(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias, &spec.format)?;
        Ok(MlxLinear {
            module,
            topology,
            vocabulary_range: Some(range),
        })
    }

    fn vocabulary_parallel_lookup(
        embedding: &mut MlxEmbedding,
        input: &MlxTensor,
        policy: EmbeddingLookupPolicy,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        policy.validate()?;
        let range = embedding
            .vocabulary_range
            .as_ref()
            .ok_or_else(|| ComputeError::backend("embedding has no vocabulary ownership"))?;
        let sentinel = match policy {
            EmbeddingLookupPolicy::Strict => None,
            EmbeddingLookupPolicy::ZeroSentinel(sentinel) => Some(sentinel),
        };
        let input = compute(validate_token_domain(
            input.as_array(),
            embedding.vocabulary,
            sentinel,
            context,
        ))?;
        let start =
            Array::from_int(i32::try_from(range.local.start).map_err(ComputeError::backend)?);
        let end = Array::from_int(i32::try_from(range.local.end).map_err(ComputeError::backend)?);
        let valid = compute(input.ge(&start, context))?
            .logical_and(&compute(input.lt(&end, context))?, context)
            .map_err(ComputeError::backend)?;
        let local = compute(input.subtract(&start, context))?;
        let safe = compute(safemlx::ops::r#where(
            &valid,
            &local,
            Array::from_int(0),
            context,
        ))?;
        let value = compute(embedding.module.forward(&safe, context))?;
        let mask = compute(valid.expand_dims(-1, context))?;
        let zero_value = compute(safemlx::ops::zeros_like(&value, context))?;
        let value = compute(safemlx::ops::r#where(&mask, &value, &zero_value, context))?;
        compute_tensor(crate::backend::runtime::distributed::all_sum(
            &value, parallel, context,
        ))
    }

    fn vocabulary_parallel_project(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let range = linear
            .vocabulary_range
            .as_ref()
            .ok_or_else(|| ComputeError::backend("projection has no vocabulary ownership"))?;
        let local = compute(linear.module.forward(input.as_array(), context))?;
        let widths = range
            .balanced_peer_widths(parallel.size(), parallel.rank())
            .map_err(ComputeError::backend)?;
        compute_tensor(
            crate::backend::distributed::all_gather_uneven_axis(
                &local, -1, &widths, parallel, context,
            )
            .map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "vocabulary projection gather failed for local range {:?}, shape {:?}, and widths {widths:?}: {error}",
                    range.local,
                    local.shape(),
                ))
            }),
        )
    }

    fn vocabulary_parallel_embedding_project(
        embedding: &mut MlxEmbedding,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let range = embedding
            .vocabulary_range
            .clone()
            .ok_or_else(|| ComputeError::backend("embedding has no vocabulary ownership"))?;
        let local = compute(embedding.module.as_linear(input.as_array(), context))?;
        let widths = range
            .balanced_peer_widths(parallel.size(), parallel.rank())
            .map_err(ComputeError::backend)?;
        compute_tensor(
            crate::backend::distributed::all_gather_uneven_axis(
                &local, -1, &widths, parallel, context,
            )
            .map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "tied vocabulary projection gather failed for local range {:?}, shape {:?}, and widths {widths:?}: {error}",
                    range.local,
                    local.shape(),
                ))
            }),
        )
    }

    fn sum_parallel(
        value: MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(crate::backend::runtime::distributed::all_sum(
            value.as_array(),
            parallel,
            context,
        ))
    }
}
