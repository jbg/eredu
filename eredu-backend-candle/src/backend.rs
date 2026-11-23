use std::convert::Infallible;

use candle_core::{DType, Tensor as NativeTensor};
use eredu_core::{eager_completion::EagerCompletion, SubmissionBackend};
use eredu_nn::{
    AttentionMask, EmbeddingSpec, Error, GatedProductPolicy, Index, LinearSpec, NeuralBackend,
    NeuralOperatorCapabilities, NormalizationConstructionSpec, PreparedRotarySpec,
    ProjectionInputObserver, RowParallelProjection, Tensor,
};

use crate::{
    operators, CandleContext, CandleEmbedding, CandleLinear, CandleNormalization, CandleRotary,
    CandleTensor,
};

/// Candle implementation of the neutral neural operator contract.
#[derive(Clone, Debug)]
pub struct CandleNeuralBackend;

fn l2_normalize(input: &CandleTensor, epsilon: f32, floor: bool) -> Result<CandleTensor, Error> {
    eredu_nn::operation_geometry::NormalizationGeometry::new(input.shape(), epsilon)?;
    let dtype = input.value.dtype();
    CandleTensor::wrap((|| {
        let wide = input.value.to_dtype(DType::F32)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if wide.device().is_metal() {
            return crate::metal_f32::l2_normalize(&wide, epsilon, floor)?.to_dtype(dtype);
        }
        crate::float_cpu::map_rows(&wide, |row| {
            let sum = row.iter().map(|x| f64::from(x * x)).sum::<f64>() as f32;
            let scale = if floor {
                sum.sqrt().max(epsilon)
            } else {
                (sum + epsilon).sqrt()
            }
            .recip();
            for x in row {
                *x *= scale;
            }
        })?
        .to_dtype(dtype)
    })())
}

impl CandleNeuralBackend {
    pub(crate) const ACCELERATOR_OPERATORS: NeuralOperatorCapabilities =
        NeuralOperatorCapabilities::GELU_APPROXIMATE
            .union(NeuralOperatorCapabilities::SEGMENTED_ATTENTION)
            .union(NeuralOperatorCapabilities::MULTI_AXIS_ROTARY_EMBEDDINGS)
            .union(NeuralOperatorCapabilities::ZEROS_LIKE)
            .union(NeuralOperatorCapabilities::LOGICAL_OR)
            .union(NeuralOperatorCapabilities::WHERE_CONDITION)
            .union(NeuralOperatorCapabilities::MASKED_SCATTER)
            .union(NeuralOperatorCapabilities::MASKED_OUTPUT_PROJECTION)
            .union(NeuralOperatorCapabilities::SIGMOID)
            .union(NeuralOperatorCapabilities::EXP)
            .union(NeuralOperatorCapabilities::OUTPUT_GATED_GROUP_RMS_NORM)
            .union(NeuralOperatorCapabilities::RMS_NORM_WITHOUT_WEIGHT)
            .union(NeuralOperatorCapabilities::UNLOADED_I32)
            .union(NeuralOperatorCapabilities::FROM_I32_SLICE)
            .union(NeuralOperatorCapabilities::TO_F32_VEC)
            .union(NeuralOperatorCapabilities::TO_I32_VEC)
            .union(NeuralOperatorCapabilities::FULL_F32)
            .union(NeuralOperatorCapabilities::FULL_I32)
            .union(NeuralOperatorCapabilities::TANH)
            .union(NeuralOperatorCapabilities::CLIP)
            .union(NeuralOperatorCapabilities::SOFTMAX_AXIS)
            .union(NeuralOperatorCapabilities::BROADCAST_TO)
            .union(NeuralOperatorCapabilities::ABS)
            .union(NeuralOperatorCapabilities::SIGN)
            .union(NeuralOperatorCapabilities::SQRT)
            .union(NeuralOperatorCapabilities::CAST_FLOAT)
            .union(NeuralOperatorCapabilities::CASCADED_LAYER_NORM)
            .union(NeuralOperatorCapabilities::CONV2D)
            .union(NeuralOperatorCapabilities::EQUAL_I32);

    pub(crate) const LOCAL_OPERATORS: NeuralOperatorCapabilities = Self::ACCELERATOR_OPERATORS
        .union(NeuralOperatorCapabilities::RELATIVE_ATTENTION)
        .union(NeuralOperatorCapabilities::INDEXED_ATTENTION)
        .union(NeuralOperatorCapabilities::ATTENTION_SOFTCAP)
        .union(NeuralOperatorCapabilities::ATTENTION_SINKS)
        .union(NeuralOperatorCapabilities::GELU_APPROXIMATE_F16)
        .union(NeuralOperatorCapabilities::FUSED_F16_PROJECTION)
        .union(NeuralOperatorCapabilities::BLOCKED_LAYER_NORM)
        .union(NeuralOperatorCapabilities::RECURRENT_F32_ROTARY)
        .union(NeuralOperatorCapabilities::SOFTPLUS)
        .union(NeuralOperatorCapabilities::L2_NORMALIZE)
        .union(NeuralOperatorCapabilities::GATED_DELTA_SCAN)
        .union(NeuralOperatorCapabilities::WEIGHTED_EMBEDDING);

    pub(crate) const REALTIME_OPERATORS: NeuralOperatorCapabilities =
        NeuralOperatorCapabilities::SIN_COS
            .union(NeuralOperatorCapabilities::HOST_RFFT)
            .union(NeuralOperatorCapabilities::RFFT)
            .union(NeuralOperatorCapabilities::IRFFT)
            .union(NeuralOperatorCapabilities::REQUIRE_FINITE)
            .union(NeuralOperatorCapabilities::SELECTIVE_STATE_SPACE_SCAN)
            .union(NeuralOperatorCapabilities::GATED_GROUP_RMS_NORM);
}

impl NeuralBackend for CandleNeuralBackend {
    const OPERATOR_CAPABILITIES: NeuralOperatorCapabilities =
        Self::LOCAL_OPERATORS.union(Self::REALTIME_OPERATORS);

    type Tensor = CandleTensor;
    type Linear = CandleLinear;
    type Embedding = CandleEmbedding;
    type Normalization = CandleNormalization;
    type Rotary = CandleRotary;
    type ParallelContext = Infallible;

    fn linear(spec: LinearSpec, context: &CandleContext) -> Result<CandleLinear, Error> {
        CandleLinear::new(spec, context)
    }

    fn embedding(spec: EmbeddingSpec, context: &CandleContext) -> Result<CandleEmbedding, Error> {
        CandleEmbedding::new(spec, context)
    }

    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &CandleContext,
    ) -> Result<CandleNormalization, Error> {
        CandleNormalization::new(spec, context)
    }

    fn rotary(spec: PreparedRotarySpec, _: &CandleContext) -> Result<CandleRotary, Error> {
        CandleRotary::new(spec)
    }

    fn silu(input: CandleTensor, _: &CandleContext) -> Result<CandleTensor, Error> {
        CandleTensor::wrap(operators::silu(&input.value))
    }

    fn sigmoid(input: CandleTensor, _: &CandleContext) -> Result<CandleTensor, Error> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if crate::metal_f32::applies(&input.value) {
            return CandleTensor::wrap(crate::metal_f32::sigmoid(&input.value));
        }
        if input.value.dtype() == DType::F32 && input.value.device().is_cpu() {
            return CandleTensor::wrap(crate::float_cpu::map_rows(&input.value, |row| {
                for x in row {
                    *x = 1. / (1. + (-*x).exp());
                }
            }));
        }
        CandleTensor::wrap(candle_nn::ops::sigmoid(&input.value))
    }

    fn exp(input: CandleTensor, _: &CandleContext) -> Result<CandleTensor, Error> {
        CandleTensor::wrap(input.value.exp())
    }

    fn softplus(input: CandleTensor, beta: f32, _: &CandleContext) -> Result<CandleTensor, Error> {
        if !beta.is_finite() || beta <= 0. {
            return Err(Error::backend("softplus beta must be positive and finite"));
        }
        let dtype = input.value.dtype();
        CandleTensor::wrap((|| {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if input.value.device().is_metal() {
                return crate::metal_f32::softplus(&input.value.to_dtype(DType::F32)?, beta)?
                    .to_dtype(dtype);
            }
            crate::float_cpu::map_rows(&input.value.to_dtype(DType::F32)?, |row| {
                for x in row {
                    let scaled = beta * *x;
                    *x = if scaled > 20. {
                        *x
                    } else {
                        (1. + scaled.exp()).ln() / beta
                    };
                }
            })?
            .to_dtype(dtype)
        })())
    }

    fn l2_normalize(
        input: &CandleTensor,
        epsilon: f32,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        l2_normalize(input, epsilon, false)
    }

    fn l2_normalize_with_floor(
        input: &CandleTensor,
        epsilon: f32,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        l2_normalize(input, epsilon, true)
    }

    fn gated_group_rms_norm(
        input: &CandleTensor,
        gate: &CandleTensor,
        weight: &CandleTensor,
        groups: i32,
        epsilon: f32,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            input.shape(),
            gate.shape(),
            groups,
            epsilon,
        )?;
        if weight.shape() != [geometry.width()] {
            return Err(Error::backend("gated RMS scale width differs from input"));
        }
        let dtype = input.value.dtype();
        let gate = CandleTensor::wrap(gate.value.to_dtype(DType::F32))?;
        let gate = Self::silu(gate, context)?;
        let gated = CandleTensor::wrap(
            input
                .value
                .to_dtype(DType::F32)
                .and_then(|value| value.mul(&gate.value)),
        )?;
        let normalized = operators::rms_norm(&gated, None, None, Some(groups), epsilon)?;
        CandleTensor::wrap(crate::tensor::rounded_bf16_binary(
            &normalized
                .value
                .to_dtype(dtype)
                .map_err(Error::backend_source)?,
            &weight.value,
            NativeTensor::broadcast_mul,
        ))
    }

    fn selective_state_space_scan(
        input: eredu_nn::SelectiveStateSpaceScanInput<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<eredu_nn::SelectiveStateSpaceScanOutput<CandleTensor>, Error> {
        crate::recurrent::selective_state_space_scan(input, context)
    }

    fn output_gated_group_rms_norm(
        input: &CandleTensor,
        gate: &CandleTensor,
        weight: &CandleTensor,
        groups: i32,
        epsilon: f32,
        activation: eredu_nn::OutputGateActivation,
        arithmetic: eredu_nn::OutputGatedNormArithmetic,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let geometry = eredu_nn::operation_geometry::GroupedNormalizationGeometry::new(
            input.shape(),
            gate.shape(),
            groups,
            epsilon,
        )?;
        if weight.shape() != [geometry.width()] {
            return Err(Error::backend(
                "invalid output-gated RMS normalization weights",
            ));
        }
        let dtype = input.value.dtype();
        let wide = CandleTensor::wrap(input.value.to_dtype(DType::F32))?;
        let normalized = operators::rms_norm(&wide, None, None, Some(groups), epsilon)?;
        CandleTensor::wrap((|| {
            let gain = weight.value.to_dtype(DType::F32)?;
            let scaled = match arithmetic {
                eredu_nn::OutputGatedNormArithmetic::Float32 => {
                    normalized.value.broadcast_mul(&gain)?
                }
                eredu_nn::OutputGatedNormArithmetic::RoundedNormalization => {
                    let rounded = normalized.value.to_dtype(dtype)?;
                    if weight.value.dtype() == dtype {
                        crate::tensor::rounded_bf16_binary(
                            &rounded,
                            &weight.value,
                            NativeTensor::broadcast_mul,
                        )?
                        .to_dtype(DType::F32)?
                    } else {
                        rounded.to_dtype(DType::F32)?.broadcast_mul(&gain)?
                    }
                }
            };
            let gate = gate.value.to_dtype(DType::F32)?;
            let gate = match activation {
                eredu_nn::OutputGateActivation::Silu => operators::silu(&gate)?,
                eredu_nn::OutputGateActivation::Sigmoid => candle_nn::ops::sigmoid(&gate)?,
            };
            scaled.broadcast_mul(&gate)?.to_dtype(dtype)
        })())
    }

    fn gated_delta_scan(
        input: eredu_nn::GatedDeltaScanInput<'_, CandleTensor>,
        _: &CandleContext,
    ) -> Result<eredu_nn::GatedDeltaScanOutput<CandleTensor>, Error> {
        crate::recurrent::gated_delta_scan(input)
    }

    fn gelu_approximate(input: CandleTensor, _: &CandleContext) -> Result<CandleTensor, Error> {
        CandleTensor::wrap(operators::gelu_approximate(&input.value))
    }

    fn gelu_approximate_f16(input: CandleTensor, _: &CandleContext) -> Result<CandleTensor, Error> {
        CandleTensor::wrap(operators::gelu_approximate_f16(&input.value))
    }

    fn rms_norm_without_weight(
        input: &CandleTensor,
        epsilon: f32,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        operators::rms_norm(input, None, None, None, epsilon)
    }

    fn rms_norm_with_weight(
        input: &CandleTensor,
        weight: &CandleTensor,
        epsilon: f32,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        operators::rms_norm(input, Some(weight), None, None, epsilon)
    }

    fn gated_product(
        gate: CandleTensor,
        up: CandleTensor,
        policy: GatedProductPolicy,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        operators::gated_product(&gate, &up, policy)
    }

    fn attention(
        queries: CandleTensor,
        keys: CandleTensor,
        values: CandleTensor,
        scale: f32,
        mask: Option<&CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        Self::attention_with_sinks(
            eredu_nn::AttentionRequest {
                queries,
                keys,
                values,
                scale,
                mask,
                sinks: None,
                softcap: None,
                arithmetic: eredu_nn::AttentionArithmetic::Fused,
            },
            context,
        )
    }

    fn attention_with_sinks(
        request: eredu_nn::AttentionRequest<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        request.validate()?;
        use crate::tensor::ScoreAdjustments;
        let mask = request
            .mask
            .map_or(AttentionMask::None, AttentionMask::Tensor);
        crate::tensor::attention(
            &request.queries,
            &request.keys,
            &request.values,
            request.scale,
            ScoreAdjustments {
                mask,
                sinks: request.sinks,
                softcap: request.softcap,
            },
            request.arithmetic,
            context,
        )
    }

    fn relative_attention(
        input: eredu_nn::RelativeAttentionInput<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        crate::cache::paged::resident_relative_attention(&input, context)
    }

    fn indexed_attention(
        input: eredu_nn::IndexedAttentionInput<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        crate::attention::resident_indexed_attention(&input, context)
    }

    fn sliding_window_attention(
        queries: CandleTensor,
        keys: CandleTensor,
        values: CandleTensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        Self::sliding_window_attention_with_sinks(
            eredu_nn::AttentionRequest {
                queries,
                keys,
                values,
                scale,
                mask: None,
                sinks: None,
                softcap: None,
                arithmetic: eredu_nn::AttentionArithmetic::Fused,
            },
            window,
            position_offset,
            context,
        )
    }

    fn sliding_window_attention_with_sinks(
        request: eredu_nn::AttentionRequest<'_, CandleTensor>,
        window: i32,
        position_offset: i32,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        request.validate()?;
        let eredu_nn::AttentionRequest {
            queries,
            keys,
            values,
            scale,
            mask,
            sinks,
            softcap,
            arithmetic,
        } = request;
        let [batch, heads, query_len, _] = *queries.shape() else {
            return Err(Error::backend(
                "sliding attention requires rank-four queries",
            ));
        };
        let key_len = keys.dim(2);
        let geometry = eredu_nn::operation_geometry::CausalMaskGeometry::new(
            query_len,
            position_offset,
            window.checked_sub(1),
        )?;
        if window <= 0 || key_len < query_len || geometry.keys() < key_len {
            return Err(Error::backend("invalid sliding attention geometry"));
        }
        let mask = mask
            .map(|mask| {
                CandleTensor::wrap(mask.value.broadcast_as((
                    batch as usize,
                    heads as usize,
                    query_len as usize,
                    key_len as usize,
                )))
            })
            .transpose()?;
        let mut rows = Vec::with_capacity(query_len as usize);
        for row in 0..query_len {
            let end = key_len - query_len + row + 1;
            let start = (end - window).max(0);
            let key_slice = [Index::Full, Index::Full, Index::Range(start, end)];
            let query_slice = [Index::Full, Index::Full, Index::Range(row, row + 1)];
            let mask =
                mask.as_ref()
                    .map(|mask| {
                        CandleTensor::wrap(mask.value.narrow(2, row as usize, 1).and_then(|mask| {
                            mask.narrow(3, start as usize, (end - start) as usize)
                        }))
                    })
                    .transpose()?;
            rows.push(Self::attention_with_sinks(
                eredu_nn::AttentionRequest {
                    queries: queries.index(&query_slice, context)?,
                    keys: keys.index(&key_slice, context)?,
                    values: values.index(&key_slice, context)?,
                    scale,
                    mask: mask.as_ref(),
                    sinks,
                    softcap,
                    arithmetic,
                },
                context,
            )?);
        }
        CandleTensor::concatenate(&rows, 2, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, query_len, -1], context)
    }

    fn segmented_attention(
        input: eredu_nn::SegmentedAttentionInput<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        input.validate()?;
        let mut offset = 0;
        let mut outputs = Vec::with_capacity(input.segment_lengths.len());
        for &length in input.segment_lengths {
            let segment = |tensor: &CandleTensor| {
                CandleTensor::wrap(
                    tensor
                        .value
                        .narrow(0, offset, length as usize)
                        .and_then(|t| t.transpose(0, 1))
                        .and_then(|t| t.unsqueeze(0)),
                )
            };
            let result = CandleTensor::scaled_dot_product_attention(
                &segment(input.queries)?,
                &segment(input.keys)?,
                &segment(input.values)?,
                input.scale,
                AttentionMask::None,
                context,
            )?;
            outputs.push(CandleTensor::wrap(
                result.value.squeeze(0).and_then(|t| t.transpose(0, 1)),
            )?);
            offset += length as usize;
        }
        CandleTensor::concatenate(&outputs, 0, context)
    }

    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        operators::causal_mask(sequence, offset, window, context)
    }

    fn prepare_row_parallel_linear(
        linear: &mut CandleLinear,
        input: &CandleTensor,
        context: &CandleContext,
        observer: Option<&mut dyn ProjectionInputObserver<CandleTensor>>,
    ) -> Result<RowParallelProjection<CandleTensor>, Error> {
        if let Some(observer) = observer {
            observer.observe(input)?;
        }
        Ok(RowParallelProjection {
            partial: CandleTensor::linear(input, linear.weight.as_ref(), None, context)?,
            bias: linear.bias.as_ref().map(|bias| bias.as_ref().clone()),
            output_cast: None,
        })
    }

    fn reduce_parallel_sum(
        _: CandleTensor,
        parallel: &Infallible,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        match *parallel {}
    }
}

impl SubmissionBackend for CandleNeuralBackend {
    type Executor = CandleContext;
    type OwnedExecutor = CandleContext;
    type Completion = EagerCompletion<Error>;

    fn fork_executors(executor: &CandleContext, count: usize) -> Result<Vec<CandleContext>, Error> {
        executor.check_healthy().map_err(Error::backend_source)?;
        Ok(vec![executor.clone(); count])
    }

    fn submit<'a, I>(executor: &CandleContext, values: I) -> Result<Self::Completion, Error>
    where
        I: IntoIterator<Item = &'a CandleTensor>,
    {
        executor.check_healthy().map_err(Error::backend_source)?;
        let roots: Vec<_> = values.into_iter().collect();
        for value in &roots {
            executor
                .validate_tensor_device(value)
                .map_err(Error::backend_source)?;
        }
        if let Err(error) = executor.synchronize() {
            executor.quarantine(roots.into_iter().cloned().collect::<Vec<_>>());
            return Err(Error::backend_source(error));
        }
        Ok(EagerCompletion::completed())
    }

    fn order_after(_: &Self::Completion, executor: &CandleContext) -> Result<(), Error> {
        executor.check_healthy().map_err(Error::backend_source)
    }

    fn retain_until_complete<T: Send + 'static>(
        executor: &CandleContext,
        _: &Self::Completion,
        _: T,
    ) -> Result<(), Error> {
        executor.check_healthy().map_err(Error::backend_source)
    }
}

// Local execution has no inhabitant of its parallel context.
impl eredu_nn::DistributedNeuralBackend for CandleNeuralBackend {
    fn parallel_rank(context: &Infallible) -> usize {
        match *context {}
    }

    fn vocabulary_parallel_embedding(
        _: EmbeddingSpec,
        _: eredu_nn::VocabularyParallelRange,
        _: &CandleContext,
    ) -> Result<CandleEmbedding, Error> {
        Err(Error::backend(
            "partitioned vocabulary is unavailable on this backend",
        ))
    }

    fn vocabulary_parallel_linear(
        _: LinearSpec,
        _: eredu_nn::VocabularyParallelRange,
        _: &CandleContext,
    ) -> Result<CandleLinear, Error> {
        Err(Error::backend(
            "partitioned vocabulary is unavailable on this backend",
        ))
    }

    fn vocabulary_linear_range(_: &CandleLinear) -> Option<&eredu_nn::VocabularyParallelRange> {
        None
    }
    fn vocabulary_embedding_range(
        _: &CandleEmbedding,
    ) -> Option<&eredu_nn::VocabularyParallelRange> {
        None
    }
    fn validate_vocabulary_gather(
        _: &CandleTensor,
        _: &[usize],
        context: &Infallible,
    ) -> Result<(), Error> {
        match *context {}
    }
    fn gather_vocabulary(
        _: CandleTensor,
        _: &[usize],
        context: &Infallible,
        _: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        match *context {}
    }
}

impl eredu_nn::HyperNeuralBackend for CandleNeuralBackend {
    type HyperConnection = eredu_nn::hyper::PortableHyperConnection<Self>;
    type HyperHead = eredu_nn::hyper::PortableHyperHead<Self>;

    fn hyper_connection(
        spec: eredu_nn::HyperConnectionSpec,
        context: &CandleContext,
    ) -> Result<Self::HyperConnection, eredu_nn::Error> {
        Self::HyperConnection::new(spec, context)
    }

    fn hyper_head(
        spec: eredu_nn::HyperHeadSpec,
        context: &CandleContext,
    ) -> Result<Self::HyperHead, eredu_nn::Error> {
        Self::HyperHead::new(spec, context)
    }
}
