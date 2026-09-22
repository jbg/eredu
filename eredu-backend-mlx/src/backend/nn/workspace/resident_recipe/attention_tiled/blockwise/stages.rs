//! Per-stage lowering shares the existing native block recurrence inventory.
use super::*;
use crate::backend::nn::workspace::attention::blockwise::descriptor::{self, Descriptor, Stage};
use std::mem::size_of;

pub(in crate::backend::nn::workspace::resident_recipe::attention_tiled) fn stage_plan(
    operation: WorkspaceOperationView<'_>,
) -> Option<Plan> {
    let Descriptor { policy, stage } = descriptor::decode(operation).ok()??;
    let mut plan = match stage {
        Stage::Begin { mask, sink, .. } => {
            let mut plan = Plan::new();
            plan.views(1 + usize::from(mask.is_some()))?;
            plan.lowering.backend_shells = usize::from(sink.is_some());
            plan.controls = begin_controls()?;
            plan
        }
        Stage::Accumulate {
            q,
            k,
            v,
            mask,
            sink,
            bias,
            previous,
            value_pass,
            absolute,
        } => {
            let g = Geometry {
                q,
                k,
                v,
                mask: mask.map(|m| m.dtype()),
                sink,
                softcap: policy.options.softcap.is_some(),
                arithmetic: policy.options.arithmetic,
                absolute: Some(absolute),
                bias: bias.is_some(),
                fixed_views: false,
            };
            if value_pass {
                values(g, !previous)?
            } else {
                normalize(g, !previous)?
            }
        }
        Stage::Finish { input_scores, .. } => {
            let mut plan = Plan::new();
            if !input_scores {
                append(&mut plan, 12, 15, 2)?;
                append(&mut plan, 5, 6, 0)?;
            }
            plan.views(1)?;
            plan.controls = finish_controls()?;
            plan
        }
    };
    let controls = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<Descriptor<'_>>(),
        size_of::<Stage<'_>>(),
        size_of::<Geometry<'_>>(),
        size_of::<Plan>(),
        size_of::<
            Result<Option<Descriptor<'_>>, crate::backend::nn::workspace::MlxWorkspaceFactError>,
        >(),
        size_of::<Option<Plan>>(),
        // The native mask helper retains coordinate/threshold/visibility
        // values until its exact returned page is included in block settlement.
        size_of::<[Array; 9]>(),
        size_of::<[&Array; 4]>(),
        size_of::<eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
    ];
    plan.controls = controls.into_iter().try_fold(
        plan.controls
            .checked_add(std::mem::size_of_val(&controls))?,
        usize::checked_add,
    )?;
    Some(plan)
}

// Native construction and finalization are separate call frames from the
// per-page recurrence. Count their owned/returned accumulator and error shells
// even when casts alias existing storage and create no payload allocation.
fn begin_controls() -> Option<usize> {
    use crate::backend::runtime::cache::kv::BlockwiseAttentionAccumulator;
    let frames = [
        size_of::<eredu_nn::BlockwiseAttentionSpec<'_, crate::MlxTensor>>(),
        size_of::<eredu_nn::BlockwiseAttentionOptions>(),
        size_of::<BlockwiseAttentionAccumulator>(),
        size_of::<Result<BlockwiseAttentionAccumulator, safemlx::error::Exception>>(),
        size_of::<Result<BlockwiseAttentionAccumulator, eredu_nn::Error>>(),
        size_of::<[Option<&Array>; 2]>(),
        size_of::<Option<Result<Array, safemlx::error::Exception>>>(),
        size_of::<Result<Option<Array>, safemlx::error::Exception>>(),
        size_of::<[i32; 8]>(),
        size_of::<Result<(), safemlx::error::Exception>>(),
        size_of::<Result<(), eredu_nn::Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
fn finish_controls() -> Option<usize> {
    use crate::backend::runtime::cache::kv::BlockwiseAttentionAccumulator;
    let frames = [
        size_of::<BlockwiseAttentionAccumulator>(),
        // Accumulator, denominator, comparison, safe denominator, quotient,
        // and the two actual scalar source arguments of the fused branch.
        size_of::<[Array; 7]>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<Result<Array, eredu_nn::Error>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{BlockwiseAttentionBackend, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    fn existing(shape: &[i32], context: &WorkspaceContext) -> WorkspaceTensor {
        WorkspaceTensor::existing(
            context.layout(shape, WorkspaceDtype::Float32).unwrap(),
            context,
        )
        .unwrap()
    }
    #[test]
    fn actual_page_stages_price_both_recurrences_and_exact_nested_frontiers() {
        let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for arithmetic in [AttentionArithmetic::Fused, AttentionArithmetic::InputScores] {
            let context = WorkspaceContext::new(facts);
            let q = existing(&[1, 4, 3, 16], &context);
            let mask = existing(&[3, 11], &context);
            let sinks = existing(&[4], &context);
            let spec = BlockwiseAttentionSpec {
                queries: &q,
                scale: 0.25,
                mask: Some(&mask),
                query_start: 8,
                context_end: 11,
                sliding_window: Some(4),
                prefix_tokens: 2,
                sinks: Some(&sinks),
            };
            // This primitive has no opening mutable cache state. Immutable
            // Q/K/V/mask/sink inputs are distinct from retained model state;
            // the inference report still requires this explicit empty seed.
            context
                .begin_state_span(std::iter::empty::<&WorkspaceTensor>())
                .unwrap();
            let options = BlockwiseAttentionOptions {
                arithmetic,
                softcap: Some(1.75),
            };
            let mut accumulator =
                WorkspaceBackend::begin_blockwise_attention_with_options(spec, options, &context)
                    .unwrap();
            for pass in 0..options.passes() {
                if pass == 1 {
                    WorkspaceBackend::begin_blockwise_value_pass(&mut accumulator, &context)
                        .unwrap();
                }
                for (start, end) in [(0, 2), (5, 9), (9, 11)] {
                    let k = existing(&[1, 2, end - start, 16], &context);
                    let v = existing(&[1, 2, end - start, 8], &context);
                    let bias_shape: &[i32] = if start == 0 { &[] } else { &[3, end - start] };
                    let bias = existing(bias_shape, &context);
                    WorkspaceBackend::accumulate_blockwise_attention_with_bias(
                        &mut accumulator,
                        start.into(),
                        end.into(),
                        k,
                        v,
                        Some(&bias),
                        &context,
                    )
                    .unwrap();
                }
            }
            let output =
                WorkspaceBackend::finish_blockwise_attention(accumulator, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            assert!(report.inference_transient_bytes().is_some());
            let mut completions = 0;
            for operation in &report.operations {
                if !matches!(
                    operation.kind,
                    WorkspaceOperationKind::BlockwiseAttention { .. }
                ) {
                    continue;
                }
                let plan = stage_plan(operation.as_view()).expect("actual supported stage recipe");
                let Descriptor { stage, .. } =
                    descriptor::decode(operation.as_view()).unwrap().unwrap();
                assert_eq!(
                    plan.lowering.nested_completions,
                    usize::from(matches!(stage, Stage::Accumulate { .. }))
                );
                assert!(plan.controls > 0);
                completions += plan.lowering.nested_completions;
            }
            assert_eq!(completions, 3 * options.passes());
        }
    }
}
