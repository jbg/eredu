use super::*;
use crate::{
    AttentionArithmetic, BlockwiseAttentionBackend, BlockwiseAttentionOptions,
    BlockwiseAttentionSpec,
};

/// Exact causal blockwise policy. Native score rounding/capping is explicit;
/// metadata outputs retain FP32 recurrence storage across both physical passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkspaceBlockwisePolicy {
    /// Absolute first query position.
    pub query_start: i64,
    /// Exclusive end of visible cache coordinates.
    pub context_end: i64,
    /// Positive finite query/key multiplier.
    pub scale: f32,
    /// Causal window including the current position.
    pub sliding_window: Option<i32>,
    /// Prefix positions visible outside the window.
    pub prefix_tokens: i64,
    /// Absolute position corresponding to the mask's first column.
    pub mask_origin: Option<i64>,
    /// Original query scalar observed by the selected source before the native
    /// recurrence converts Q/K/V to F32. Finish restores exactly this type.
    /// None preserves missing source evidence; logical Float32 is not a default.
    pub output_type: Option<WorkspaceFloatingType>,
    /// Actual score rounding and optional tanh cap.
    pub options: BlockwiseAttentionOptions,
}
/// One native mechanism boundary in an online attention recurrence.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WorkspaceBlockwiseStage {
    /// Prepare queries and broadcast the optional complete-context mask.
    Begin,
    /// Incorporate one reconstructed K/V block, retaining maximum, sum and
    /// weighted values. Gaps are permitted for prefix-plus-window histories.
    Accumulate {
        /// Inclusive absolute key position.
        start: i64,
        /// Exclusive absolute key position.
        end: i64,
        /// A previous block supplied running normalization and values.
        previous: bool,
        /// This block consumes previously completed global normalization.
        value_pass: bool,
        /// An actual additional bias operand is supplied for this key block.
        bias: bool,
    },
    /// Return the rounded value-pass result, or divide a fused accumulation by
    /// its safe normalization denominator, then restore the query scalar type.
    Finish,
}
/// Host metadata for the ordinary blockwise accumulator. It contains no native
/// state or completion authority; its allocations remain in the containing
/// trace span until that span's separately established completion boundary.
#[derive(Debug)]
pub struct WorkspaceBlockwiseAccumulator {
    policy: WorkspaceBlockwisePolicy,
    queries: WorkspaceTensor,
    mask: Option<WorkspaceTensor>,
    sinks: Option<WorkspaceTensor>,
    running: Option<Running>,
    value_pass: bool,
    value_width: Option<i32>,
    last_end: Option<i64>,
}
#[derive(Debug)]
struct Running {
    maximum: WorkspaceTensor,
    sum: WorkspaceTensor,
    value: Option<WorkspaceTensor>,
}
impl WorkspaceBlockwiseAccumulator {
    fn kind(&self, stage: WorkspaceBlockwiseStage) -> WorkspaceOperationKind {
        WorkspaceOperationKind::BlockwiseAttention {
            policy: self.policy,
            stage,
        }
    }
}
impl BlockwiseAttentionBackend for WorkspaceBackend {
    type BlockwiseAccumulator = WorkspaceBlockwiseAccumulator;
    fn begin_blockwise_attention(
        spec: BlockwiseAttentionSpec<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<Self::BlockwiseAccumulator, Error> {
        Self::begin_blockwise_attention_with_options(
            spec,
            BlockwiseAttentionOptions::default(),
            context,
        )
    }
    fn begin_blockwise_attention_with_options(
        spec: BlockwiseAttentionSpec<'_, WorkspaceTensor>,
        options: BlockwiseAttentionOptions,
        context: &WorkspaceContext,
    ) -> Result<Self::BlockwiseAccumulator, Error> {
        options
            .validate()
            .map_err(|cause| context.metadata_source(cause))?;
        context.charge_metadata(std::mem::size_of::<(
            BlockwiseAttentionSpec<'_, WorkspaceTensor>,
            BlockwiseAttentionOptions,
            Self::BlockwiseAccumulator,
            Result<Self::BlockwiseAccumulator, Error>,
        )>())?;
        let q = spec.queries.shape();
        if q.len() != 4
            || q.iter().any(|n| *n <= 0)
            || spec.query_start < 0
            || spec.context_end <= 0
            || spec
                .query_start
                .checked_add(q.get(2).copied().unwrap_or(0) as i64)
                .is_none_or(|end| end > spec.context_end)
            || !spec.scale.is_finite()
            || spec.scale <= 0.0
            || spec.prefix_tokens < 0
            || spec.sliding_window.is_some_and(|window| window <= 0)
        {
            return Err(context.metadata_error(format_args!(
                "invalid workspace blockwise query geometry or coordinates"
            )));
        }
        if spec.sinks.is_some_and(|s| s.shape() != [q[1]]) {
            return Err(context.metadata_error(format_args!(
                "workspace blockwise sinks must match query heads"
            )));
        }
        let mut outputs = context.metadata_vec(1 + usize::from(spec.mask.is_some()))?;
        outputs.push(context.layout(q, WorkspaceDtype::Float32)?);
        let mask_origin = if let Some(mask) = spec.mask {
            let shape = mask.shape();
            if shape.is_empty() || shape.len() > 4 || *shape.last().unwrap() <= 0 {
                return Err(context.metadata_error(format_args!(
                    "workspace blockwise mask must broadcast to rank-four scores"
                )));
            }
            let target = [q[0], q[1], q[2], *shape.last().unwrap()];
            if !WorkspaceBroadcastShape::new(shape, &target)
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
                .dimensions()
                .eq(target)
            {
                return Err(context.metadata_error(format_args!(
                    "workspace blockwise mask expands query dimensions"
                )));
            }
            outputs.push(context.layout(&target, mask.layout.dtype)?);
            Some(spec.context_end - *shape.last().unwrap() as i64)
        } else {
            None
        };
        let policy = WorkspaceBlockwisePolicy {
            query_start: spec.query_start,
            context_end: spec.context_end,
            scale: spec.scale,
            sliding_window: spec.sliding_window,
            prefix_tokens: spec.prefix_tokens,
            mask_origin,
            output_type: spec
                .queries
                .layout()
                .representation()
                .map(WorkspaceRepresentation::dtype),
            options,
        };
        let mut inputs = context.metadata_vec(
            1 + usize::from(spec.mask.is_some()) + usize::from(spec.sinks.is_some()),
        )?;
        inputs.push(spec.queries);
        inputs.extend(spec.mask);
        inputs.extend(spec.sinks);
        let mut values = context
            .execute(
                WorkspaceOperationKind::BlockwiseAttention {
                    policy,
                    stage: WorkspaceBlockwiseStage::Begin,
                },
                &inputs,
                outputs,
            )?
            .into_iter();
        Ok(WorkspaceBlockwiseAccumulator {
            policy,
            queries: values.next().unwrap(),
            mask: values.next(),
            sinks: spec.sinks.cloned(),
            running: None,
            value_pass: false,
            value_width: None,
            last_end: None,
        })
    }
    fn begin_blockwise_value_pass(
        accumulator: &mut Self::BlockwiseAccumulator,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self::BlockwiseAccumulator,
            &WorkspaceContext,
            Result<(), Error>,
        )>())?;
        context.validate_values([&accumulator.queries])?;
        if accumulator.policy.options.arithmetic != AttentionArithmetic::InputScores
            || accumulator.value_pass
            || accumulator.running.is_none()
        {
            return Err(context.metadata_error(format_args!(
                "rounded-probability attention requires completed normalization"
            )));
        }
        accumulator.running.as_mut().unwrap().value = None;
        accumulator.value_pass = true;
        accumulator.last_end = None;
        Ok(())
    }
    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<u64, Error> {
        Self::accumulate_blockwise_attention_with_bias(
            accumulator,
            start,
            end,
            keys,
            values,
            None,
            context,
        )
    }
    fn accumulate_blockwise_attention_with_bias(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        bias: Option<&WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<u64, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self::BlockwiseAccumulator,
            i64,
            i64,
            WorkspaceTensor,
            WorkspaceTensor,
            Option<&WorkspaceTensor>,
            &WorkspaceContext,
            Running,
            Result<u64, Error>,
        )>())?;
        let q = accumulator.queries.shape();
        let k = keys.shape();
        let v = values.shape();
        if start < 0
            || end <= start
            || end > accumulator.policy.context_end
            || accumulator
                .last_end
                .is_some_and(|previous| start < previous)
            || k.len() != 4
            || v.len() != 4
            || k[0] != q[0]
            || k[1] <= 0
            || q[1] % k[1] != 0
            || k[2] as i64 != end - start
            || k[3] != q[3]
            || k[..3] != v[..3]
            || v[3] <= 0
            || accumulator.value_width.is_some_and(|width| width != v[3])
        {
            return Err(context.metadata_error(format_args!(
                "workspace blockwise keys, values or ordered range are inconsistent"
            )));
        }
        if accumulator
            .policy
            .mask_origin
            .is_some_and(|origin| start < origin)
        {
            return Err(context.metadata_error(format_args!(
                "workspace blockwise mask does not cover the key block"
            )));
        }
        if let Some(bias) = bias {
            let target = [q[0], q[1], q[2], k[2]];
            if !WorkspaceBroadcastShape::new(bias.shape(), &target)
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
                .dimensions()
                .eq(target)
            {
                return Err(
                    context.metadata_error(format_args!("blockwise bias expands score dimensions"))
                );
            }
        }
        // This is the portable interface's reconstructed input-byte diagnostic,
        // matching logical K/V payloads. It is never the native scratch bound.
        let reconstructed = keys
            .layout
            .as_view()
            .bytes()
            .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
            .checked_add(
                values
                    .layout
                    .as_view()
                    .bytes()
                    .map_err(|cause| context.metadata_error(format_args!("{cause}")))?,
            )
            .ok_or_else(|| {
                context.metadata_error(format_args!("workspace reconstruction bytes overflow"))
            })?;
        let previous = accumulator
            .running
            .as_ref()
            .is_some_and(|running| running.value.is_some());
        let running_count = accumulator
            .running
            .as_ref()
            .map_or(0, |running| 2 + usize::from(running.value.is_some()));
        let mut inputs = context.metadata_vec(
            3 + usize::from(accumulator.mask.is_some())
                + usize::from(!accumulator.value_pass && accumulator.sinks.is_some())
                + usize::from(bias.is_some())
                + running_count,
        )?;
        inputs.extend([&accumulator.queries, &keys, &values]);
        inputs.extend(accumulator.mask.iter());
        if !accumulator.value_pass {
            inputs.extend(accumulator.sinks.iter());
        }
        inputs.extend(bias);
        if let Some(running) = &accumulator.running {
            inputs.extend([&running.maximum, &running.sum]);
            inputs.extend(running.value.iter());
        }
        let output = context.layout(&[q[0], q[1], q[2], v[3]], WorkspaceDtype::Float32)?;
        let mut layouts = context.metadata_vec(if accumulator.value_pass { 1 } else { 3 })?;
        if !accumulator.value_pass {
            let reduction = context.layout(&[q[0], q[1], q[2], 1], WorkspaceDtype::Float32)?;
            layouts.extend([reduction.clone(), reduction]);
        }
        layouts.push(output);
        let mut result = context
            .execute(
                accumulator.kind(WorkspaceBlockwiseStage::Accumulate {
                    start,
                    end,
                    previous,
                    value_pass: accumulator.value_pass,
                    bias: bias.is_some(),
                }),
                &inputs,
                layouts,
            )?
            .into_iter();
        if accumulator.value_pass {
            accumulator
                .running
                .as_mut()
                .expect("validated normalization")
                .value = Some(result.next().unwrap());
        } else {
            accumulator.running = Some(Running {
                maximum: result.next().unwrap(),
                sum: result.next().unwrap(),
                value: Some(result.next().unwrap()),
            });
        }
        accumulator.value_width = Some(v[3]);
        accumulator.last_end = Some(end);
        Ok(reconstructed)
    }
    fn finish_blockwise_attention(
        accumulator: Self::BlockwiseAccumulator,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        context.charge_metadata(std::mem::size_of::<(
            Self::BlockwiseAccumulator,
            &WorkspaceContext,
            Result<WorkspaceTensor, Error>,
        )>())?;
        let Some(running) = &accumulator.running else {
            return Err(context.metadata_error(format_args!(
                "workspace blockwise attention received no cache blocks"
            )));
        };
        let value = running.value.as_ref().ok_or_else(|| {
            context.metadata_error(format_args!(
                "workspace blockwise value scan received no cache blocks"
            ))
        })?;
        if accumulator.policy.options.arithmetic == AttentionArithmetic::InputScores {
            if !accumulator.value_pass {
                return Err(context.metadata_error(format_args!(
                    "rounded-probability attention is missing its value pass"
                )));
            }
            return WorkspaceTensor::operation(
                accumulator.kind(WorkspaceBlockwiseStage::Finish),
                &[value],
                value.shape(),
                WorkspaceDtype::Float32,
                context,
            );
        }
        WorkspaceTensor::operation(
            accumulator.kind(WorkspaceBlockwiseStage::Finish),
            &[value, &running.sum],
            value.shape(),
            WorkspaceDtype::Float32,
            context,
        )
    }
}
