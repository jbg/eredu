use super::*;
use crate::{
    AttentionRequest, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, GatedProductPolicy,
    LinearFormat, LinearOperator, LinearSpec, NeuralBackend, NeuralOperatorCapabilities,
    NormalizationConstructionSpec, NormalizationOperator, NormalizationScale, Parameter,
    ParameterSpec, RotaryOperator, RotaryPosition, RotarySpec, Tensor,
};

mod parallel;
mod projection_observation;
pub use parallel::WorkspaceParallelContext;
mod blockwise;
mod grouped;
mod hyper;
mod pooling;
pub use blockwise::{
    WorkspaceBlockwiseAccumulator, WorkspaceBlockwisePolicy, WorkspaceBlockwiseStage,
};
pub use grouped::{
    WorkspaceGroupSelector, WorkspaceGroupedBank, WorkspaceGroupedPhase, WorkspaceGroups,
};
pub use hyper::{WorkspaceHyperConnection, WorkspaceHyperHead};

/// Executes the ordinary neural contracts using host metadata only.
/// A capability here means that geometry can be traced; admission still requires
/// a proved allocation bound from the selected native mechanism for every call.
#[derive(Clone, Debug)]
pub struct WorkspaceBackend;

/// Metadata projection retaining exact physical checkpoint parameter topology.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceLinear {
    parameters: Vec<Parameter<WorkspaceTensor>>,
    #[parameter(skip, metadata)]
    spec: LinearSpec,
    #[parameter(skip, metadata)]
    vocabulary_range: Option<crate::VocabularyParallelRange>,
}

/// Metadata embedding. Tied readout reuses the same parameter storage identities.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceEmbedding {
    projection: WorkspaceLinear,
}

/// Metadata RMS normalization retaining its complete construction policy.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceNormalization {
    weight: Option<Parameter<WorkspaceTensor>>,
    #[parameter(skip, metadata)]
    spec: NormalizationConstructionSpec,
}

/// Metadata rotary operator. Explicit position tensors participate in the trace.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceRotary {
    #[parameter(skip, metadata)]
    spec: RotarySpec,
}

fn parameter(
    spec: ParameterSpec,
    shape: &[i32],
    dtype: WorkspaceDtype,
    context: &WorkspaceContext,
) -> Result<Parameter<WorkspaceTensor>, Error> {
    let layout = context.parameter_layout(&spec, shape, dtype)?;
    let backing = context.parameter_backing(&spec, &layout)?;
    Ok(Parameter::new(
        spec,
        match backing {
            Some(backing) => {
                WorkspaceTensor::parameter_placeholder_with_backing(layout, &backing, context)?
            }
            None => WorkspaceTensor::parameter_placeholder(layout, context)?,
        },
    ))
}

fn physical_parameters(
    spec: &LinearSpec,
    context: &WorkspaceContext,
) -> Result<Vec<Parameter<WorkspaceTensor>>, Error> {
    if context.uses_checked_metadata() {
        spec.format
            .validate_for_weight_fixed(&spec.weight)
            .map_err(|cause| match cause {
                crate::LinearWeightValidationError::Format(cause) => {
                    context.metadata_error(format_args!("{cause}"))
                }
                crate::LinearWeightValidationError::PrimaryIdentity => {
                    context.metadata_error(format_args!(
                        "linear format companion reuses primary weight identity {}",
                        spec.weight.id,
                    ))
                }
            })?;
    } else {
        spec.format.validate_for_weight(&spec.weight)?;
    }
    if spec.input <= 0 || spec.output <= 0 {
        return Err(context.metadata_error(format_args!(
            "workspace projection dimensions must be positive"
        )));
    }
    let narrow = |n: u64| {
        i32::try_from(n)
            .map_err(|_| context.metadata_error(format_args!("workspace physical extent overflow")))
    };
    let mut scale = None;
    let (shape, dtype) = match spec.format.encoding() {
        LinearFormat::Dense => ([spec.output, spec.input], WorkspaceDtype::Float32),
        LinearFormat::Affine(config) => {
            if spec.input % config.group_size != 0 {
                return Err(context.metadata_error(format_args!(
                    "workspace affine group splits the input width"
                )));
            }
            scale = Some((
                [spec.output, spec.input / config.group_size],
                WorkspaceDtype::Float32,
            ));
            (
                [
                    spec.output,
                    narrow((spec.input as u64 * config.bits as u64).div_ceil(32))?,
                ],
                WorkspaceDtype::Uint32,
            )
        }
        LinearFormat::MxFp4 => {
            if spec.input % 32 != 0 {
                return Err(context
                    .metadata_error(format_args!("workspace MXFP4 group splits the input width")));
            }
            scale = Some(([spec.output, spec.input / 32], WorkspaceDtype::Uint8));
            ([spec.output, spec.input / 8], WorkspaceDtype::Uint32)
        }
        LinearFormat::GgufIQuant { ggml_type, .. } => {
            let (block, bytes) = ggml_type
                .block_and_bytes_fixed()
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
            if spec.input as u64 % block as u64 != 0 {
                return Err(context
                    .metadata_error(format_args!("workspace GGML block splits the input width")));
            }
            (
                [
                    spec.output,
                    narrow(spec.input as u64 / block as u64 * bytes as u64)?,
                ],
                WorkspaceDtype::Uint8,
            )
        }
        LinearFormat::E4M3BlockFp8(config) => {
            let rows = spec
                .format
                .row_layout()
                .scale_rows(spec.output as usize, config.block_rows as usize)?;
            scale = Some((
                [
                    narrow(rows as u64)?,
                    narrow((spec.input as u64).div_ceil(config.block_columns as u64))?,
                ],
                match config.scale_encoding {
                    eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint => {
                        WorkspaceDtype::Float32
                    }
                    eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 => WorkspaceDtype::Uint8,
                },
            ));
            ([spec.output, spec.input], WorkspaceDtype::Uint8)
        }
    };
    let mut weight = context.clone_metadata(&spec.weight)?;
    weight.linear_row_layout = spec.format.row_layout();
    let mut values = context.metadata_vec(
        1 + usize::from(scale.is_some())
            + usize::from(scale.is_some() && spec.format.affine_bias().is_some())
            + usize::from(spec.bias.is_some()),
    )?;
    values.push(parameter(weight, &shape, dtype, context)?);
    if let Some((shape, dtype)) = scale {
        let mut identity = context.clone_metadata(
            spec.format
                .scale()
                .expect("validated companion cardinality"),
        )?;
        identity.linear_companion_of = Some(context.clone_metadata(&spec.weight.id)?);
        identity.linear_row_layout = spec.format.row_layout();
        values.push(parameter(identity, &shape, dtype, context)?);
        if let Some(identity) = spec.format.affine_bias() {
            let mut identity = context.clone_metadata(identity)?;
            identity.linear_companion_of = Some(context.clone_metadata(&spec.weight.id)?);
            identity.linear_row_layout = spec.format.row_layout();
            values.push(parameter(
                identity,
                &shape,
                WorkspaceDtype::Float32,
                context,
            )?);
        }
    }
    if let Some(bias) = &spec.bias {
        values.push(parameter(
            context.clone_metadata(bias)?,
            &[spec.output],
            WorkspaceDtype::Float32,
            context,
        )?);
    }
    Ok(values)
}

impl LinearOperator<WorkspaceTensor> for WorkspaceLinear {
    fn forward_with_input_observer(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
        observer: Option<&mut dyn crate::ProjectionInputObserver<WorkspaceTensor>>,
    ) -> Result<WorkspaceTensor, Error> {
        self.forward_observed(input, context, observer)
    }
    fn forward(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.trace(
            input,
            WorkspaceOperationKind::Projection(context.clone_metadata(&self.spec.format)?),
            context,
        )
    }
}

impl WorkspaceLinear {
    fn trace(
        &self,
        input: &WorkspaceTensor,
        kind: WorkspaceOperationKind,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if input.shape().last() != Some(&self.spec.input) {
            return Err(context.metadata_error(format_args!(
                "workspace projection input width differs from construction"
            )));
        }
        let mut shape = context.metadata_vec(input.shape().len())?;
        shape.extend_from_slice(input.shape());
        *shape.last_mut().expect("validated input rank") = self.spec.output;
        let mut inputs = context.metadata_vec(
            self.parameters
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.push(input);
        inputs.extend(self.parameters.iter().map(Parameter::as_ref));
        WorkspaceTensor::operation(kind, &inputs, &shape, WorkspaceDtype::Float32, context)
    }
}
impl EmbeddingOperator<WorkspaceTensor> for WorkspaceEmbedding {
    fn forward(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.lookup(input, EmbeddingLookupPolicy::Strict, context)
    }
    fn lookup(
        &mut self,
        input: &WorkspaceTensor,
        policy: EmbeddingLookupPolicy,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        policy.validate()?;
        if !matches!(
            input.layout.dtype,
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
        ) {
            return Err(
                context.metadata_error(format_args!("workspace embedding requires integer IDs"))
            );
        }
        let mut shape = context.metadata_vec(
            input
                .shape()
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        shape.extend_from_slice(input.shape());
        shape.push(self.projection.spec.input);
        let mut inputs = context.metadata_vec(
            self.projection
                .parameters
                .len()
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.push(input);
        inputs.extend(self.projection.parameters.iter().map(Parameter::as_ref));
        WorkspaceTensor::operation(
            WorkspaceOperationKind::Embedding(
                context.clone_metadata(&self.projection.spec.format)?,
                policy,
            ),
            &inputs,
            &shape,
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn as_linear(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.projection.forward(input, context)
    }
}
impl NormalizationOperator<WorkspaceTensor> for WorkspaceNormalization {
    fn forward(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if input.shape().last() != Some(&self.spec.dimensions) {
            return Err(context.metadata_error(format_args!(
                "workspace normalization feature width differs from construction"
            )));
        }
        let mut inputs = context.metadata_vec(1 + usize::from(self.weight.is_some()))?;
        inputs.push(input);
        inputs.extend(self.weight.iter().map(Parameter::as_ref));
        WorkspaceTensor::operation(
            WorkspaceOperationKind::ConstructedNormalization(context.clone_metadata(&self.spec)?),
            &inputs,
            input.shape(),
            input.layout.dtype,
            context,
        )
    }
}
impl RotaryOperator<WorkspaceTensor> for WorkspaceRotary {
    fn forward(
        &mut self,
        input: &WorkspaceTensor,
        position: RotaryPosition<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if input.shape().len() < 2
            || input
                .shape()
                .last()
                .is_none_or(|width| *width < self.spec.dimensions)
        {
            return Err(context.metadata_error(format_args!(
                "workspace rotary feature width differs from construction"
            )));
        }
        let mut inputs = context.metadata_vec(match &position {
            RotaryPosition::Offset(_) => 1,
            RotaryPosition::Embeddings { .. } => 3,
        })?;
        inputs.push(input);
        let offset = match position {
            RotaryPosition::Offset(offset) => Some(offset),
            RotaryPosition::Embeddings { cosine, sine } => {
                inputs.extend([cosine, sine]);
                None
            }
        };
        WorkspaceTensor::operation(
            WorkspaceOperationKind::Rotary(self.spec, offset),
            &inputs,
            input.shape(),
            input.layout.dtype,
            context,
        )
    }
}

macro_rules! elementwise {
    ($($name:ident),* $(,)?) => {$(fn $name(input: WorkspaceTensor, context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> { input.floating_unary(stringify!($name), context) })*};
}

impl NeuralBackend for WorkspaceBackend {
    type ParameterPreparation<'a> = ();
    fn construction_metadata(context: &WorkspaceContext) -> Option<&WorkspaceContext> {
        Some(context)
    }

    const OPERATOR_CAPABILITIES: NeuralOperatorCapabilities =
        NeuralOperatorCapabilities::GELU_APPROXIMATE
            .union(NeuralOperatorCapabilities::SUM_PARALLEL)
            .union(NeuralOperatorCapabilities::SIGMOID)
            .union(NeuralOperatorCapabilities::SOFTPLUS)
            .union(NeuralOperatorCapabilities::EXP)
            .union(NeuralOperatorCapabilities::L2_NORMALIZE)
            .union(NeuralOperatorCapabilities::GATED_GROUP_RMS_NORM)
            .union(NeuralOperatorCapabilities::SILU_GATED_GROUP_RMS_NORM)
            .union(NeuralOperatorCapabilities::GATED_DELTA_SCAN)
            .union(NeuralOperatorCapabilities::SELECTIVE_STATE_SPACE_SCAN)
            .union(NeuralOperatorCapabilities::ATTENTION_SOFTCAP)
            .union(NeuralOperatorCapabilities::ATTENTION_SINKS)
            .union(NeuralOperatorCapabilities::RMS_NORM_WITHOUT_WEIGHT)
            .union(NeuralOperatorCapabilities::UNLOADED_I32)
            .union(NeuralOperatorCapabilities::FROM_I32_SLICE)
            .union(NeuralOperatorCapabilities::FULL_F32)
            .union(NeuralOperatorCapabilities::FULL_I32)
            .union(NeuralOperatorCapabilities::FULL_U32)
            .union(NeuralOperatorCapabilities::TANH)
            .union(NeuralOperatorCapabilities::CLIP)
            .union(NeuralOperatorCapabilities::SOFTMAX_AXIS)
            .union(NeuralOperatorCapabilities::BROADCAST_TO)
            .union(NeuralOperatorCapabilities::ZEROS_LIKE)
            .union(NeuralOperatorCapabilities::EQUAL_I32)
            .union(NeuralOperatorCapabilities::LOGICAL_OR)
            .union(NeuralOperatorCapabilities::WHERE_CONDITION)
            .union(NeuralOperatorCapabilities::MASKED_SCATTER)
            .union(NeuralOperatorCapabilities::ROPE_WITH_FREQUENCIES)
            .union(NeuralOperatorCapabilities::CONV2D)
            .union(NeuralOperatorCapabilities::INDEXED_ATTENTION)
            .union(NeuralOperatorCapabilities::POOLED_ATTENTION)
            .union(NeuralOperatorCapabilities::POOLED_POSITION_SELECTION)
            .union(NeuralOperatorCapabilities::POOLED_MASK_GATHER)
            .union(NeuralOperatorCapabilities::GROUPED_LINEAR)
            .union(NeuralOperatorCapabilities::JOINT_GROUP_SELECTION)
            .union(NeuralOperatorCapabilities::RELATIVE_ATTENTION)
            .union(NeuralOperatorCapabilities::SEGMENTED_ATTENTION)
            .union(NeuralOperatorCapabilities::MULTI_AXIS_ROTARY_EMBEDDINGS)
            .union(NeuralOperatorCapabilities::MASKED_OUTPUT_PROJECTION);
    type Tensor = WorkspaceTensor;
    type Linear = WorkspaceLinear;
    type Embedding = WorkspaceEmbedding;
    type Normalization = WorkspaceNormalization;
    type Rotary = WorkspaceRotary;
    type ParallelContext = WorkspaceParallelContext;

    fn linear(spec: LinearSpec, context: &WorkspaceContext) -> Result<Self::Linear, Error> {
        Ok(WorkspaceLinear {
            parameters: physical_parameters(&spec, context)?,
            spec,
            vocabulary_range: None,
        })
    }
    fn embedding(
        spec: EmbeddingSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::Embedding, Error> {
        Ok(WorkspaceEmbedding {
            projection: Self::linear(
                LinearSpec {
                    input: spec.dimensions,
                    output: spec.vocabulary,
                    weight: spec.weight,
                    bias: None,
                    format: spec.format,
                },
                context,
            )?,
        })
    }
    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::Normalization, Error> {
        if context.uses_checked_metadata() {
            spec.validate_fixed()
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        } else {
            spec.validate()?;
        }
        let identity = match &spec.scale {
            NormalizationScale::Learned(weight)
            | NormalizationScale::LearnedOffset { weight, .. } => Some(weight),
            NormalizationScale::Unit => None,
        };
        let weight = identity
            .map(|identity| {
                parameter(
                    context.clone_metadata(identity)?,
                    &[spec.dimensions],
                    WorkspaceDtype::Float32,
                    context,
                )
            })
            .transpose()?;
        Ok(WorkspaceNormalization { weight, spec })
    }
    fn rotary(spec: RotarySpec, context: &WorkspaceContext) -> Result<Self::Rotary, Error> {
        if context.uses_checked_metadata() {
            spec.algorithm.validate_fixed().map_err(|_| {
                context.metadata_error(format_args!(
                    "invalid normalized rotary algorithm: {:?}",
                    spec.algorithm,
                ))
            })?;
        } else {
            spec.algorithm.validate()?;
        }
        if spec.dimensions <= 0
            || spec.dimensions % 2 != 0
            || !spec.base.is_finite()
            || spec.base <= 0.0
        {
            return Err(
                context.metadata_error(format_args!("invalid workspace rotary dimensions or base"))
            );
        }
        Ok(WorkspaceRotary { spec })
    }
    fn indexed_attention(
        input: crate::IndexedAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        pooling::indexed(input, context)
    }
    fn pooled_attention(
        input: crate::PooledAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        pooling::pooled(input, context)
    }
    fn select_pooled_positions(
        input: crate::PooledPositionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        pooling::positions(input, context)
    }
    fn gather_pooled_mask(
        mask: &WorkspaceTensor,
        positions: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        pooling::gather(mask, positions, context)
    }
    fn relative_attention(
        input: crate::RelativeAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        input.validate()?;
        WorkspaceTensor::operation(
            WorkspaceOperationKind::RelativeAttention {
                query_offset: input.query_offset,
                key_offset: input.key_offset,
                window: input.window,
                log_scaling_floor: input.log_scaling_floor,
                log_scaling_alpha: input.log_scaling_alpha,
            },
            &[input.queries, input.keys, input.values, input.profiles],
            input.queries.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn segmented_attention(
        input: crate::SegmentedAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        input.validate_with_diagnostic(|message| context.metadata_error(message))?;
        let heads = input.queries.shape()[1];
        let key_width = input.queries.shape()[2];
        let value_width = input.values.shape()[2];
        let mut outputs = context.metadata_vec(input.segment_lengths.len())?;
        let mut start = 0;
        for &length in input.segment_lengths {
            let end = start + length;
            let prepare = |value: &WorkspaceTensor, width| {
                value
                    .index(
                        &[
                            crate::Index::Range(start, end),
                            crate::Index::Full,
                            crate::Index::Full,
                        ],
                        context,
                    )?
                    .transpose_axes(&[1, 0, 2], context)?
                    .reshape(&[1, heads, length, width], context)
            };
            let queries = prepare(input.queries, key_width)?;
            let keys = prepare(input.keys, key_width)?;
            let values = prepare(input.values, value_width)?;
            let output = Self::attention(queries, keys, values, input.scale, None, context)?;
            outputs.push(
                output
                    .reshape(&[heads, length, value_width], context)?
                    .transpose_axes(&[1, 0, 2], context)?,
            );
            start = end;
        }
        WorkspaceTensor::concatenate(&outputs, 0, context)
    }
    fn segmented_attention_with_metadata(
        input: crate::SegmentedAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
        metadata: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if !context.shares_trace(metadata) {
            return Err(super::WorkspaceMetadataError::Unqualified.into());
        }
        Self::segmented_attention(input, context)
    }
    elementwise!(silu, gelu_approximate, sigmoid, exp);
    fn softplus(
        input: WorkspaceTensor,
        beta: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(context.metadata_error(format_args!("invalid workspace softplus beta")));
        }
        input.floating_unary("softplus", context)
    }
    fn gated_product(
        gate: WorkspaceTensor,
        up: WorkspaceTensor,
        policy: GatedProductPolicy,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        policy.validate()?;
        if gate.shape() != up.shape() {
            return Err(
                context.metadata_error(format_args!("workspace gated product dimensions disagree"))
            );
        }
        WorkspaceTensor::operation(
            WorkspaceOperationKind::GatedProduct(policy),
            &[&gate, &up],
            gate.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }
    fn attention(
        queries: WorkspaceTensor,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        scale: f32,
        mask: Option<&WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        Self::attention_with_sinks(
            AttentionRequest {
                queries,
                keys,
                values,
                scale,
                mask,
                sinks: None,
                softcap: None,
                arithmetic: crate::AttentionArithmetic::Fused,
            },
            context,
        )
    }
    fn attention_with_sinks(
        request: AttentionRequest<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        trace_attention(request, None, context)
    }
    fn sliding_window_attention(
        queries: WorkspaceTensor,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        Self::sliding_window_attention_with_sinks(
            AttentionRequest {
                queries,
                keys,
                values,
                scale,
                mask: None,
                sinks: None,
                softcap: None,
                arithmetic: crate::AttentionArithmetic::Fused,
            },
            window,
            position_offset,
            context,
        )
    }
    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, WorkspaceTensor>,
        window: i32,
        position_offset: i32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        if window <= 0 || position_offset < 0 {
            return Err(context.metadata_error(format_args!(
                "invalid workspace sliding attention positions"
            )));
        }
        trace_attention(request, Some((window, position_offset)), context)
    }
    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let geometry =
            crate::operation_geometry::CausalMaskGeometry::new_fixed(sequence, offset, window)
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        WorkspaceTensor::operation(
            WorkspaceOperationKind::CausalMask(geometry),
            &[],
            &[geometry.sequence(), geometry.keys()],
            WorkspaceDtype::Bool,
            context,
        )
    }
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &WorkspaceTensor,
        parallel: &WorkspaceParallelContext,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let local = linear.forward(input, context)?;
        <Self as crate::DistributedNeuralBackend>::sum_parallel(local, parallel, context)
    }
    fn parallel_size(parallel: &WorkspaceParallelContext) -> usize {
        parallel.size()
    }
    fn gated_group_rms_norm(
        input: &WorkspaceTensor,
        gate: &WorkspaceTensor,
        weight: &WorkspaceTensor,
        groups: i32,
        epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        grouped_norm(
            input,
            gate,
            weight,
            groups,
            epsilon,
            "gated_group_rms_norm",
            context,
        )
    }
    fn silu_gated_group_rms_norm(
        input: &WorkspaceTensor,
        gate: &WorkspaceTensor,
        weight: &WorkspaceTensor,
        groups: i32,
        epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        grouped_norm(
            input,
            gate,
            weight,
            groups,
            epsilon,
            "silu_gated_group_rms_norm",
            context,
        )
    }
    fn gated_delta_scan(
        input: crate::GatedDeltaScanInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<crate::GatedDeltaScanOutput<WorkspaceTensor>, Error> {
        let q = input.query.shape();
        let v = input.value.shape();
        if q.len() != 4
            || v.len() != 4
            || q[..3] != v[..3]
            || input.key.shape() != q
            || input.beta.shape() != &q[..3]
            || (input.log_decay.shape() != q && input.log_decay.shape() != &q[..3])
        {
            return Err(
                context.metadata_error(format_args!("invalid workspace gated-delta scan geometry"))
            );
        }
        let state_shape = [q[0], q[2], q[3], v[3]];
        if input
            .initial_state
            .is_some_and(|state| state.shape() != state_shape)
        {
            return Err(context.metadata_error(format_args!(
                "workspace gated-delta state dimensions disagree"
            )));
        }
        let mut inputs = context.metadata_vec(5 + usize::from(input.initial_state.is_some()))?;
        inputs.extend([
            input.query,
            input.key,
            input.value,
            input.log_decay,
            input.beta,
        ]);
        inputs.extend(input.initial_state);
        let mut layouts = context.metadata_vec(2)?;
        layouts.extend([
            context.layout(&state_shape, WorkspaceDtype::Float32)?,
            input.value.layout.clone(),
        ]);
        let mut outputs = context
            .execute(WorkspaceOperationKind::GatedDeltaScan, &inputs, layouts)?
            .into_iter();
        Ok(crate::GatedDeltaScanOutput {
            state: outputs.next().expect("declared scan state"),
            output: outputs.next().expect("declared scan output"),
        })
    }
    fn selective_state_space_scan(
        input: crate::SelectiveStateSpaceScanInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<crate::SelectiveStateSpaceScanOutput<WorkspaceTensor>, Error> {
        let geometry = crate::operation_geometry::SelectiveScanGeometry::new(
            [
                input.values.shape(),
                input.input_state.shape(),
                input.output_state.shape(),
                input.time_step.shape(),
                input.time_step_bias.shape(),
                input.transition_log.shape(),
                input.skip.shape(),
            ],
            input.initial_state.map(|value| value.shape()),
            input.chunk_size,
            input.time_step_floor,
        )
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
        let state_shape = geometry.state();
        let mut inputs = context.metadata_vec(7 + usize::from(input.initial_state.is_some()))?;
        inputs.extend([
            input.values,
            input.input_state,
            input.output_state,
            input.time_step,
            input.time_step_bias,
            input.transition_log,
            input.skip,
        ]);
        inputs.extend(input.initial_state);
        let mut layouts = context.metadata_vec(2)?;
        layouts.extend([
            context.layout(&state_shape, WorkspaceDtype::Float32)?,
            input.values.layout.clone(),
        ]);
        let mut outputs = context
            .execute(
                WorkspaceOperationKind::SelectiveStateSpaceScan(
                    input.chunk_size,
                    input.time_step_floor,
                ),
                &inputs,
                layouts,
            )?
            .into_iter();
        Ok(crate::SelectiveStateSpaceScanOutput {
            state: outputs.next().expect("declared scan state"),
            output: outputs.next().expect("declared scan output"),
        })
    }
    fn l2_normalize(
        input: &WorkspaceTensor,
        epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        normalization(input, None, epsilon, "l2", context)
    }
    fn rms_norm_without_weight(
        input: &WorkspaceTensor,
        epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        normalization(input, None, epsilon, "rms", context)
    }
    fn rms_norm_with_weight(
        input: &WorkspaceTensor,
        weight: &WorkspaceTensor,
        epsilon: f32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        normalization(input, Some(weight), epsilon, "rms", context)
    }
}

fn grouped_norm(
    input: &WorkspaceTensor,
    gate: &WorkspaceTensor,
    weight: &WorkspaceTensor,
    groups: i32,
    epsilon: f32,
    name: &'static str,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let geometry = crate::operation_geometry::GroupedNormalizationGeometry::new_fixed(
        input.shape(),
        gate.shape(),
        groups,
        epsilon,
    )
    .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    if weight.shape() != [geometry.width()] {
        return Err(context.metadata_error(format_args!(
            "workspace grouped normalization scale width differs"
        )));
    }
    WorkspaceTensor::operation(
        WorkspaceOperationKind::Normalization(name, Some(groups)),
        &[input, gate, weight],
        input.shape(),
        input.layout.dtype,
        context,
    )
}

fn normalization(
    input: &WorkspaceTensor,
    weight: Option<&WorkspaceTensor>,
    epsilon: f32,
    name: &'static str,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let geometry =
        crate::operation_geometry::NormalizationGeometry::new_fixed(input.shape(), epsilon)
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?;
    if weight.is_some_and(|weight| weight.shape() != [geometry.width()]) {
        return Err(
            context.metadata_error(format_args!("workspace normalization scale width differs"))
        );
    }
    let mut inputs = context.metadata_vec(1 + usize::from(weight.is_some()))?;
    inputs.push(input);
    inputs.extend(weight);
    WorkspaceTensor::operation(
        WorkspaceOperationKind::Normalization(name, None),
        &inputs,
        input.shape(),
        input.layout.dtype,
        context,
    )
}

fn trace_attention(
    request: AttentionRequest<'_, WorkspaceTensor>,
    window: Option<(i32, i32)>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    request.validate()?;
    // Named sliding attention constructs its own causal/window mask. The
    // default neural worker and native implementations do not consume the
    // general request mask; a previously constructed mask remains its own
    // recorded operation, rather than a fictitious sliding input edge.
    let mask = if window.is_some() { None } else { request.mask };
    let mut inputs = context
        .metadata_vec(3 + usize::from(mask.is_some()) + usize::from(request.sinks.is_some()))?;
    inputs.extend([&request.queries, &request.keys, &request.values]);
    inputs.extend(mask);
    inputs.extend(request.sinks);
    tensor::attention(
        &inputs,
        WorkspaceOperationKind::Attention {
            causal: window.is_some(),
            window,
            sinks: request.sinks.is_some(),
            softcap: request.softcap.is_some(),
            arithmetic: request.arithmetic,
        },
        context,
    )
}
