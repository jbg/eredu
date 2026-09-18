use super::*;
mod segmented;
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
    /// A prepared binding disagreed with the retained parameter geometry.
    #[error("prepared parameter shape does not match materialized weight")]
    BindingShape,
    /// Final stream-to-stream weight copy failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

impl MlxNeuralBackend {
    pub(crate) fn validate_prepared_bind(
        parameter: &MlxTensor,
        weight: &MlxTensor,
    ) -> Result<(), MlxParameterError> {
        if parameter.as_array().shape() == weight.as_array().shape() {
            Ok(())
        } else {
            Err(MlxParameterError::BindingShape)
        }
    }
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
        if Self::validate_prepared_bind(parameter, weight).is_err() {
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
            None => (None, NativeParameterTable::from_rows(Vec::new())?),
        };
        Ok(MlxRmsNorm {
            groups: spec.groups,
            module,
            topology,
            offset,
            dimensions: spec.dimensions,
            epsilon: spec.epsilon,
        })
    }

    fn rotary(spec: RotarySpec, context: &Stream) -> Result<MlxRotary, ComputeError> {
        spec.algorithm.validate()?;
        let native = compute(rope::initialize_rope(
            spec.dimensions,
            spec.base,
            spec.traditional,
            spec.algorithm,
            context,
        ))?;
        let explicit = if spec.arithmetic == eredu_nn::RotaryArithmetic::InputProducts {
            Some(compute(rope::ElementwiseRotary::new(
                spec, &native, context,
            ))?)
        } else {
            None
        };
        Ok(MlxRotary {
            native,
            explicit,
            dimensions: spec.dimensions,
        })
    }

    fn silu(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(common::layers::silu(input.into_array(), context))
    }

    fn gelu_approximate(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(nn::gelu_approximate(input.into_array(), context))
    }

    fn sigmoid(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(common::layers::sigmoid(input.into_array(), context))
    }

    fn softplus(input: MlxTensor, beta: f32, context: &Stream) -> Result<MlxTensor, ComputeError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(ComputeError::backend(
                "softplus beta must be positive and finite",
            ));
        }
        let input = input.into_array();
        let dtype = input.dtype();
        let wide = compute(input.as_dtype(Dtype::Float32, context))?;
        let scaled = compute(wide.multiply(
            Array::try_from_f32(beta).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
        let output = compute(scaled.exp(context))?;
        let output = compute(output.log1p(context))?;
        let output = compute(output.divide(
            Array::try_from_f32(beta).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
        let output = compute(safemlx::ops::r#where(
            scaled
                .gt(
                    Array::try_from_f32(20.0).map_err(ComputeError::backend_retained_source)?,
                    context,
                )
                .map_err(ComputeError::backend_retained_source)?,
            &wide,
            output,
            context,
        ))?;
        compute_tensor(output.as_dtype(dtype, context))
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
        let shape = input.shape();
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            shape,
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
            compute(variance.add(
                Array::try_from_f32(epsilon).map_err(ComputeError::backend_retained_source)?,
                context,
            ))?,
            context,
        ))?;
        let normalized = compute(grouped.multiply(&scale, context))?;
        let normalized = compute(normalized.reshape(shape, context))?;
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
        let denominator = compute(sum.add(
            Array::try_from_f32(epsilon).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
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
        let shape = input.shape();
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            shape,
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
            compute(variance.add(
                Array::try_from_f32(epsilon).map_err(ComputeError::backend_retained_source)?,
                context,
            ))?,
            context,
        ))?;
        let normalized = compute(grouped.multiply(&scale, context))?;
        let normalized = compute(normalized.reshape(shape, context))?;
        let normalized = compute(normalized.multiply(weight, context))?;
        let gate = compute(gate.as_dtype(Dtype::Float32, context))?;
        let gate =
            compute(gate.multiply(compute(safemlx::ops::sigmoid(&gate, context))?, context))?;
        compute(normalized.multiply(&gate, context))?
            .as_dtype(dtype, context)
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend_retained_source)
    }

    fn segmented_attention(
        input: SegmentedAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        segmented::run(input, context, None)
    }

    fn segmented_attention_with_metadata(
        input: SegmentedAttentionInput<'_, MlxTensor>,
        context: &Stream,
        metadata: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<MlxTensor, ComputeError> {
        segmented::run(input, context, Some(metadata))
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
        crate::backend::nn::selective_scan::execute(input, context)
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
            gate = compute(safemlx::ops::minimum(
                gate,
                Array::try_from_f32(bound).map_err(ComputeError::backend_retained_source)?,
                context,
            ))?;
        }
        if let Some(bound) = policy.up_absolute_bound() {
            up = compute(safemlx::ops::clip(up, (-bound, bound), context))?;
        }
        if policy.up_offset() != 0.0 {
            up = compute(up.add(
                Array::try_from_f32(policy.up_offset()).map_err(ComputeError::backend_retained_source)?,
                context,
            ))?;
        }
        let gate = match policy.activation() {
            eredu_nn::GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
                compute(common::layers::silu(gate, context))?
            }
            eredu_nn::GatedProductActivation::Silu => {
                let scaled = compute(
                    gate.multiply(
                        Array::try_from_f32(policy.sigmoid_multiplier())
                            .map_err(ComputeError::backend_retained_source)?,
                        context,
                    ),
                )?;
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
        let mut valid = compute(distances.ge(
            Array::try_from_int(0).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
        if let Some(window) = input.window {
            valid = compute(valid.logical_and(
                &compute(distances.lt(
                    Array::try_from_int(window).map_err(ComputeError::backend_retained_source)?,
                    context,
                ))?,
                context,
            ))?;
        }
        let relative = compute(
            crate::backend::nn::relative_attention::RelativeAttentionKernel::new(&input, context),
        )?;
        let bias = compute(relative.bias(
            i64::from(input.key_offset),
            i64::from(input.key_offset) + i64::from(key_len),
            context,
        ))?;
        let scaled = compute(relative.queries.multiply(
            Array::try_from_f32(1.0 / dimensions as f32).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
        let scores = compute(matmul(
            &scaled,
            &compute(keys.swap_axes(-1, -2, context))?,
            context,
        ))?;
        let scores = compute(scores.add(bias, context))?;
        let scores = compute(r#where(
            &valid,
            scores,
            Array::try_from_f32(f32::NEG_INFINITY).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
        let probabilities = compute(softmax_axis(scores, -1, true, context))?;
        compute_tensor(matmul(probabilities, values, context))
    }

    fn indexed_attention(
        input: IndexedAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        crate::backend::nn::attention::indexed::execute(input, context)
    }

    fn pooled_attention(
        input: PooledAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let q = input.queries.shape();
        let l = input.local.shape();
        let p = input.pooled.shape();
        if q.len() != 4
            || l.len() != 3
            || p.len() != 3
            || q.iter().any(|n| *n <= 0)
            || l[1] < 0
            || p[1] < 0
            || q[0] != l[0]
            || q[0] != p[0]
            || q[3] != l[2]
            || q[3] != p[2]
            || l[1].checked_add(p[1]).is_none_or(|n| n <= 0)
            || !input.scale.is_finite()
            || input.scale <= 0.0
        {
            return Err(ComputeError::backend("invalid pooled attention geometry"));
        }
        let local_tokens = input.local.dim(1);
        let pooled_tokens = input.pooled.dim(1);
        let mask_shapes = common::attention::pooled_mask_shapes(
            q,
            local_tokens,
            pooled_tokens,
            input.local_mask.map(Tensor::shape),
            input.pooled_mask.map(Tensor::shape),
        )?;
        let local = compute(input.local.as_array().expand_dims(1, context))?;
        let pooled = compute(input.pooled.as_array().expand_dims(1, context))?;
        let keys = compute(safemlx::ops::concatenate_axis(&[local, pooled], 2, context))?;
        let mask = mask_shapes
            .map(|(local_shape, pooled_shape)| {
                let additive = [input.local_mask, input.pooled_mask]
                    .into_iter()
                    .flatten()
                    .any(|mask| mask.as_array().dtype() != Dtype::Bool);
                let normalize = |mask: Option<&MlxTensor>, shape: &[i32]| {
                    let value = match mask {
                        Some(mask) if additive && mask.as_array().dtype() == Dtype::Bool => {
                            safemlx::ops::r#where(
                                mask.as_array(),
                                Array::try_from_f32(0.0)?,
                                Array::try_from_f32(f32::NEG_INFINITY)?,
                                context,
                            )?
                        }
                        Some(mask) => mask.as_array().clone(),
                        None if additive => Array::try_from_f32(0.0)?,
                        None => Array::try_from_bool(true)?,
                    };
                    broadcast_to(value, shape, context)
                };
                let joined = safemlx::ops::concatenate_axis(
                    &[
                        normalize(input.local_mask, &local_shape)?,
                        normalize(input.pooled_mask, &pooled_shape)?,
                    ],
                    -1,
                    context,
                )?;
                if additive {
                    joined.as_dtype(
                        Dtype::from_promoting_types(input.queries.as_array().dtype(), keys.dtype()),
                        context,
                    )
                } else {
                    Ok(joined)
                }
            })
            .transpose()
            .map_err(ComputeError::backend_retained_source)?;
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
        crate::backend::nn::attention::pooled_positions::run(input, context)
    }

    fn gather_pooled_mask(
        mask: &MlxTensor,
        selected_positions: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        crate::backend::nn::attention::gather_mask::run(mask, selected_positions, context)
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
            request.arithmetic,
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
                request.arithmetic,
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
        let output = compute(super::super::normalization::input_precision_rms(
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
        linear.forward_observed_input(input,Some(parallel),context,None)
    }

    fn row_parallel_linear_with_input_observer(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        linear.forward_observed_input(input, Some(parallel), context, observer)
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
        let global =
            i32::try_from(range.global_vocabulary).map_err(ComputeError::backend_retained_source)?;
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend_retained_source)?;
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
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend_retained_source)?;
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
        parallel_lookup::lookup(embedding,input,policy,parallel,context)
    }

    fn vocabulary_parallel_project(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        Self::vocabulary_parallel_project_with_input_observer(
            linear, input, parallel, context, None,
        )
    }

    fn vocabulary_parallel_project_with_input_observer(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let range = linear
            .vocabulary_range
            .clone()
            .ok_or_else(|| ComputeError::backend("projection has no vocabulary ownership"))?;
        let widths = parallel_gather::widths(&range,parallel)?;
        let local = linear.forward_with_input_observer(input, context, observer)?;
        parallel_gather::run(local.as_array(),&widths,parallel,context).map(MlxTensor::from_array)
    }

    fn vocabulary_parallel_embedding_project(
        embedding: &mut MlxEmbedding,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        Self::vocabulary_parallel_embedding_project_with_input_observer(
            embedding, input, parallel, context, None,
        )
    }

    fn vocabulary_parallel_embedding_project_with_input_observer(
        embedding: &mut MlxEmbedding,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let range = embedding
            .vocabulary_range
            .clone()
            .ok_or_else(|| ComputeError::backend("embedding has no vocabulary ownership"))?;
        let widths = parallel_gather::widths(&range,parallel)?;
        let local = embedding.as_linear_with_input_observer(input, context, observer)?;
        parallel_gather::run(local.as_array(),&widths,parallel,context).map(MlxTensor::from_array)
    }

    fn sum_parallel(
        value: MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        parallel.sum_model(value.as_array(),context).map(MlxTensor::from_array)
    }
}
