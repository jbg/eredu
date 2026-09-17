//! Actual query tiles and sliding-window views over the shared attention worker.
//! Key-block recurrence retains its separate prepared nested completion obligation.
use super::*;
mod blockwise;
use crate::backend::nn::attention::{
    INPUT_SCORE_ROW_BUDGET, SLIDING_QUERY_TILE, input_score_query_step,
};
use eredu_nn::{
    AttentionArithmetic, operation_geometry::SlidingAttentionGeometry, workspace::WorkspaceDtype,
};
use safemlx::Array;

pub(super) fn selected(operation: WorkspaceOperationView<'_>) -> bool {
    match operation.kind {
        WorkspaceOperationKindView::BlockwiseAttention { .. } => true,
        WorkspaceOperationKindView::Attention {
            window: Some(_), ..
        } => true,
        WorkspaceOperationKindView::Attention {
            arithmetic: AttentionArithmetic::InputScores,
            ..
        } => operation
            .inputs
            .get(0)
            .and_then(|q| q.shape().get(2))
            .zip(operation.inputs.get(1).and_then(|k| k.shape().get(2)))
            .is_some_and(|(&q, &k)| {
                i64::from(q) * i64::from(k) > i64::from(INPUT_SCORE_ROW_BUDGET)
            }),
        _ => false,
    }
}

struct Plan {
    lowering: Lowering,
    controls: usize,
}
impl Plan {
    fn new() -> Self {
        let mut lowering = Lowering::plain(0, 0, 0);
        lowering.grouped_output_calls = 0;
        Self {
            lowering,
            controls: 0,
        }
    }
    fn append(&mut self, value: Lowering, controls: usize) -> Option<()> {
        let out = &mut self.lowering;
        out.nested_completions = out
            .nested_completions
            .checked_add(value.nested_completions)?;
        out.backend_shells = out.backend_shells.checked_add(value.backend_shells)?;
        out.primitives = out.primitives.checked_add(value.primitives)?;
        out.edges = out.edges.checked_add(value.edges)?;
        out.seeds = out.seeds.checked_add(value.seeds)?;
        out.validations = out.validations.checked_add(value.validations)?;
        out.maximum_births = out.maximum_births.checked_add(value.maximum_births)?;
        out.hidden_leaves = out.hidden_leaves.checked_add(value.hidden_leaves)?;
        out.maximum_operands = out.maximum_operands.max(value.maximum_operands);
        out.intermediate_rank = out.intermediate_rank.max(value.intermediate_rank);
        out.streams = out.streams.max(value.streams);
        out.bf16_projection_calls = out
            .bf16_projection_calls
            .checked_add(value.bf16_projection_calls)?;
        out.pointwise_calls = out.pointwise_calls.checked_add(value.pointwise_calls)?;
        out.row_rms_calls = out.row_rms_calls.checked_add(value.row_rms_calls)?;
        out.recurrent_calls = out.recurrent_calls.checked_add(value.recurrent_calls)?;
        out.router_cpu_partitions = out
            .router_cpu_partitions
            .checked_add(value.router_cpu_partitions)?;
        out.additional_sort_kernels = Some(
            out.additional_sort_kernels?
                .checked_add(value.additional_sort_kernels?)?,
        );
        if value.grouped_output_chunks != 0 {
            out.grouped_output_calls = out
                .grouped_output_calls
                .checked_add(value.grouped_output_calls)?;
            out.grouped_output_chunks = out.grouped_output_chunks.max(value.grouped_output_chunks);
        }
        out.unqualified_kernel_owner = out
            .unqualified_kernel_owner
            .or(value.unqualified_kernel_owner);
        self.controls = self.controls.checked_add(controls)?;
        Some(())
    }
    fn repeat(mut self, count: usize) -> Option<Self> {
        let v = &mut self.lowering;
        macro_rules! scale { ($($field:ident),* $(,)?) => { $(v.$field = v.$field.checked_mul(count)?;)* } }
        scale!(
            primitives,
            edges,
            seeds,
            validations,
            maximum_births,
            hidden_leaves,
            grouped_output_calls,
            bf16_projection_calls,
            pointwise_calls,
            row_rms_calls,
            recurrent_calls,
            router_cpu_partitions,
            nested_completions,
            backend_shells
        );
        v.additional_sort_kernels = Some(v.additional_sort_kernels?.checked_mul(count)?);
        self.controls = self.controls.checked_mul(count)?;
        Some(self)
    }
    fn views(&mut self, count: usize) -> Option<()> {
        self.append(Lowering::plain(count, count, 0), 0)
    }
    fn bank_and_concat(&mut self, chunks: usize) -> Option<()> {
        self.lowering.grouped_output_calls = self.lowering.grouped_output_calls.checked_add(1)?;
        self.lowering.grouped_output_chunks = self.lowering.grouped_output_chunks.max(chunks);
        self.lowering.maximum_operands = self.lowering.maximum_operands.max(chunks);
        self.append(
            Lowering::plain(chunks.checked_add(1)?, chunks.checked_mul(2)?, 0),
            0,
        )
    }
}

#[derive(Clone, Copy)]
struct Geometry<'a> {
    q: [i32; 4],
    k: [i32; 4],
    v: [i32; 4],
    mask: Option<WorkspaceDtype>,
    sink: Option<WorkspaceLayoutView<'a>>,
    softcap: bool,
    arithmetic: AttentionArithmetic,
    absolute: Option<eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry>,
    bias: bool,
    fixed_views: bool,
}
impl<'a> Geometry<'a> {
    fn child(self, queries: i32, keys: i32, mask: Option<WorkspaceDtype>) -> Option<Plan> {
        let qshape = [self.q[0], self.q[1], queries, self.q[3]];
        let kshape = [self.k[0], self.k[1], keys, self.k[3]];
        let vshape = [self.v[0], self.v[1], keys, self.v[3]];
        let mshape = [self.q[0], self.q[1], queries, keys];
        let oshape = [self.q[0], self.q[1], queries, self.v[3]];
        let q = WorkspaceLayoutView::new(&qshape, WorkspaceDtype::Float32).ok()?;
        let k = WorkspaceLayoutView::new(&kshape, WorkspaceDtype::Float32).ok()?;
        let v = WorkspaceLayoutView::new(&vshape, WorkspaceDtype::Float32).ok()?;
        let output = WorkspaceLayoutView::new(&oshape, WorkspaceDtype::Float32).ok()?;
        let mut inputs = [q, k, v, q, q];
        let mut count = 3;
        if let Some(dtype) = mask {
            inputs[count] = WorkspaceLayoutView::new(&mshape, dtype).ok()?;
            count += 1;
        }
        if let Some(sink) = self.sink {
            inputs[count] = sink;
            count += 1;
        }
        let op = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::Attention {
                causal: false,
                window: None,
                sinks: self.sink.is_some(),
                softcap: self.softcap,
                arithmetic: self.arithmetic,
            },
            inputs: WorkspaceLayoutList::Views(&inputs[..count]),
            outputs: WorkspaceLayoutList::Views(std::slice::from_ref(&output)),
        };
        if self.arithmetic == AttentionArithmetic::InputScores
            && i64::from(queries) * i64::from(keys) > i64::from(INPUT_SCORE_ROW_BUDGET)
        {
            return query_tiles(Self {
                q: qshape,
                k: kshape,
                v: vshape,
                mask,
                ..self
            });
        }
        let lowered = super::lowering(op)?;
        let controls = if attention_direct::selected(op) {
            attention_direct::control_bytes(op)?
        } else {
            0
        };
        let mut plan = Plan::new();
        plan.append(lowered, controls)?;
        Some(plan)
    }
}

fn query_tiles(g: Geometry<'_>) -> Option<Plan> {
    // This branch creates actual nested Recovery/eval frontiers per key block.
    // Its graph cannot be certified by pretending those are ordinary query tiles.
    if g.k[2] > INPUT_SCORE_ROW_BUDGET {
        return blockwise::plan(g);
    }
    let step = input_score_query_step(g.k[2]);
    let mut plan = Plan::new();
    // The optional caller mask is broadcast once before slicing each query tile.
    plan.views(usize::from(g.mask.is_some()))?;
    let mut start = 0;
    let mut chunks = 0usize;
    while start < g.q[2] {
        let count = step.min(g.q[2] - start);
        plan.views(1 + usize::from(g.mask.is_some()))?;
        let child = g.child(count, g.k[2], g.mask)?;
        plan.append(child.lowering, child.controls)?;
        start += count;
        chunks = chunks.checked_add(1)?;
    }
    plan.bank_and_concat(chunks)?;
    plan.controls = plan.controls.checked_add(frame_controls())?;
    Some(plan)
}

fn plan(operation: WorkspaceOperationView<'_>) -> Option<Plan> {
    if !selected(operation) {
        return None;
    }
    let WorkspaceOperationKindView::Attention {
        causal,
        window,
        sinks,
        softcap,
        arithmetic,
    } = operation.kind
    else {
        return None;
    };
    let base = 3 + usize::from(sinks);
    if operation.outputs.len() != 1
        || operation.inputs.len() < base
        || operation.inputs.len() > base + 1
    {
        return None;
    }
    if operation.outputs.get(0)?.dtype() != WorkspaceDtype::Float32 {
        return None;
    }
    let shape = |index| -> Option<[i32; 4]> {
        let a = operation.inputs.get(index)?;
        if a.dtype() != WorkspaceDtype::Float32 {
            return None;
        }
        a.shape().try_into().ok()
    };
    let q = shape(0)?;
    let k = shape(1)?;
    let v = shape(2)?;
    if q.iter().chain(&k).chain(&v).any(|&n| n <= 0)
        || q[0] != k[0]
        || k[..3] != v[..3]
        || q[3] != k[3]
        || q[1] % k[1] != 0
    {
        return None;
    }
    let mask = if operation.inputs.len() > base {
        let a = operation.inputs.get(3)?;
        if a.shape().len() > 4
            || a.shape()
                .iter()
                .rev()
                .zip([q[0], q[1], q[2], k[2]].iter().rev())
                .any(|(&x, &y)| x != 1 && x != y)
            || !matches!(a.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32)
        {
            return None;
        }
        Some(a.dtype())
    } else {
        None
    };
    let sink = if sinks {
        let a = operation.inputs.last()?;
        if a.shape() != [q[1]] || a.dtype() != WorkspaceDtype::Float32 {
            return None;
        }
        Some(a)
    } else {
        None
    };
    let g = Geometry {
        q,
        k,
        v,
        mask,
        sink,
        softcap,
        arithmetic,
        absolute: None,
        bias: false,
        fixed_views: true,
    };
    if let Some((window, offset)) = window {
        let origin = SlidingAttentionGeometry::new_fixed(q[2], k[2], window, offset)
            .ok()?
            .key_origin();
        if mask.is_some()
            || !causal
            || operation.outputs.get(0)?.shape() != [q[0], q[2], q[1].checked_mul(v[3])?]
        {
            return None;
        }
        let mut plan = Plan::new();
        if arithmetic == AttentionArithmetic::Fused && !softcap && offset == 0 && q[2] <= window {
            // Exactly the shared native causal SDPA branch, then transpose/join.
            let mut native = reduction_lowering(59, 64, 2, 12);
            native.intermediate_rank = 5;
            native.maximum_operands = 5;
            plan.append(native, frame_controls())?;
        } else {
            let mut start = 0;
            let mut chunks = 0usize;
            while start < q[2] {
                let end = start.checked_add(SLIDING_QUERY_TILE.min(q[2] - start))?;
                let absolute = offset.checked_add(start)?;
                let first = absolute.checked_sub(window - 1)?.max(origin);
                let keys = offset.checked_add(end)?.checked_sub(first)?;
                plan.views(3)?;
                // Real windowed causal-mask worker: two Aranges, two Reshapes,
                // compare/subtract/compare/and, with one eager distance scalar.
                plan.append(Lowering::plain(24, 26, 1), 0)?;
                let child = g.child(end - start, keys, Some(WorkspaceDtype::Bool))?;
                plan.append(child.lowering, child.controls)?;
                start = end;
                chunks = chunks.checked_add(1)?;
            }
            plan.bank_and_concat(chunks)?;
            plan.controls = plan.controls.checked_add(frame_controls())?;
        }
        plan.views(2)?;
        Some(plan)
    } else {
        if causal || operation.outputs.get(0)?.shape() != [q[0], q[1], q[2], v[3]] {
            return None;
        }
        query_tiles(g)
    }
}

fn frame_controls() -> usize {
    use std::mem::size_of;
    size_of::<Plan>()
        + size_of::<Option<Plan>>()
        + size_of::<Geometry<'static>>()
        + size_of::<[Array; 5]>()
        + size_of::<[Option<Array>; 2]>()
        + size_of::<Result<Array, safemlx::error::Exception>>()
        + size_of::<[i32; 4]>() * 3
        + size_of::<[i32; 11]>()
        + size_of::<std::ops::Range<i32>>()
        + size_of::<std::iter::StepBy<std::ops::Range<i32>>>()
        + size_of::<[&'static Array; 5]>()
        + size_of::<&'static safemlx::Stream>()
}
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    if matches!(operation.kind, WorkspaceOperationKindView::BlockwiseAttention { .. }) {
        return blockwise::stage_plan(operation).map(|plan| plan.lowering);
    }

    Some(plan(operation)?.lowering)
}
pub(super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    if matches!(operation.kind, WorkspaceOperationKindView::BlockwiseAttention { .. }) {
        return blockwise::stage_plan(operation).map(|plan| plan.controls);
    }

    Some(plan(operation)?.controls)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{AttentionRequest, NeuralBackend, Tensor};

    #[test]
    fn actual_attention_tiles_prepare_all_nested_output_banks() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        // First row: query-only tiling. Second: two sliding tiles each create
        // their own query tile bank. Third reaches genuine key-block completion.
        for (queries, keys, window, expected_banks) in [
            (64, 256, None, Some(1)),
            (512, 512, Some(256), Some(3)),
            (2, 8193, None, Some(1)),
        ] {
            let context = WorkspaceContext::new(mechanism);
            let q = WorkspaceTensor::unloaded_f32(&[1, 4, queries, 32], &context).unwrap();
            let k = WorkspaceTensor::unloaded_f32(&[1, 2, keys, 32], &context).unwrap();
            let v = WorkspaceTensor::unloaded_f32(&[1, 2, keys, 6], &context).unwrap();
            context.begin_span();
            let request = AttentionRequest {
                queries: q,
                keys: k,
                values: v,
                scale: 0.25,
                softcap: Some(2.0),
                mask: None,
                sinks: None,
                arithmetic: AttentionArithmetic::InputScores,
            };
            let output = if let Some(window) = window {
                WorkspaceBackend::sliding_window_attention_with_sinks(request, window, 0, &context)
            } else {
                WorkspaceBackend::attention_with_sinks(request, &context)
            }
            .unwrap();
            assert_eq!(output.shape()[0], 1);
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let actual = plan(report.operations[0].as_view());
            let Some(expected_banks) = expected_banks else {
                assert!(
                    actual.is_none(),
                    "ordinary nested completion is not original authority"
                );
                continue;
            };
            let actual = actual.unwrap();
            assert_eq!(actual.lowering.grouped_output_calls, expected_banks);
            assert!(actual.lowering.grouped_output_chunks >= 2);
            if keys > INPUT_SCORE_ROW_BUDGET {
                assert_eq!(
                    actual.lowering.nested_completions,
                    2 * queries as usize * (keys as usize).div_ceil(256)
                );
                assert_eq!(actual.lowering.validations, 0);
            } else {
                assert!(actual.lowering.validations > 2);
                assert_eq!(actual.lowering.nested_completions, 0);
            }
            assert!(actual.controls > 0);
            let storage = GroupedOutputStorage {
                calls: actual.lowering.grouped_output_calls,
                chunks: actual.lowering.grouped_output_chunks,
                        ..GroupedOutputStorage::default()
            };
            assert!(storage.control_bytes().unwrap() > 0);
        }
    }
    #[test]
    fn sliding_sink_attention_quotes_only_the_sources_consumed_by_each_native_branch() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for (keys, offset) in [(2, 0), (4, 2)] {
            let context = WorkspaceContext::new(mechanism);
            let q = WorkspaceTensor::unloaded_f32(&[1, 2, 2, 32], &context).unwrap();
            let k = WorkspaceTensor::unloaded_f32(&[1, 1, keys, 32], &context).unwrap();
            let v = WorkspaceTensor::unloaded_f32(&[1, 1, keys, 32], &context).unwrap();
            let sinks = WorkspaceTensor::unloaded_f32(&[2], &context).unwrap();
            context.begin_span();
            // The shared decoder has already created this generic causal mask.
            // Sliding generates its own window mask just like ordinary native
            // execution, but the earlier constructor still needs its own quote.
            let mask = WorkspaceBackend::causal_mask(2, offset, None, &context).unwrap();
            let output = WorkspaceBackend::sliding_window_attention_with_sinks(
                AttentionRequest { queries:q, keys:k, values:v, scale:0.25,
                    mask:Some(&mask), sinks:Some(&sinks), softcap:None,
                    arithmetic:AttentionArithmetic::Fused }, 3, offset, &context).unwrap();
            assert_eq!(output.shape(), [1,2,64]);
            let report = context.report(&[output]).unwrap();
            assert!(report.operations.iter().any(|op|matches!(op.kind,WorkspaceOperationKind::CausalMask(_))));
            let op=report.operations.iter().find(|op|matches!(op.kind,WorkspaceOperationKind::Attention{..})).unwrap();
            assert_eq!(op.inputs.len(),4, "Q/K/V and the real sink source");
            assert_eq!(op.inputs[3].shape(),[2]);
            let native=plan(op.as_view()).expect("actual sliding worker source");
            assert_eq!(native.lowering.grouped_output_calls,usize::from(offset!=0));
            assert!(native.controls>0);
            let recorder=ResidentRecipeRecorder::new(InferenceGeometry{
                batch_size:1,cached_positions:offset as u64,input_positions:2,max_output_tokens:1,
                prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence},mechanism);
            let complete=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(complete.first_missing_operation,None);
            assert!(complete.mutable_storage.is_some());
            assert!(complete.graph.is_some());
            assert!(complete.dispatch.is_some());
        }
    }

}
