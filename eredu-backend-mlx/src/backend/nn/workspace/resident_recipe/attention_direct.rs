//! The shared direct explicit attention worker, including its BF16 alternatives.
//! Sliding and blockwise accumulation use their separate shared producer recipes.
use super::*;
use eredu_nn::{workspace::WorkspaceDtype, AttentionArithmetic};

struct Direct {
    softcap: bool,
    mask: bool,
    sinks: bool,
    qk_bf16: bool,
    qk_ids: bool,
    row_softmax: bool,
}

pub(super) fn selected(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(
        operation.kind,
        WorkspaceOperationKindView::Attention {
            arithmetic: AttentionArithmetic::InputScores,
            ..
        } | WorkspaceOperationKindView::Attention { softcap: true, .. }
    )
}

fn geometry(operation: WorkspaceOperationView<'_>) -> Option<Direct> {
    let WorkspaceOperationKindView::Attention {
        causal: false,
        window: None,
        sinks,
        softcap,
        arithmetic,
    } = operation.kind
    else {
        return None;
    };
    if !selected(operation) || operation.outputs.len() != 1 {
        return None;
    }
    let base = 3usize.checked_add(usize::from(sinks))?;
    if operation.inputs.len() < base || operation.inputs.len() > base + 1 {
        return None;
    }
    let q = operation.inputs.get(0)?.shape();
    let k = operation.inputs.get(1)?.shape();
    let v = operation.inputs.get(2)?.shape();
    if q.len() != 4
        || k.len() != 4
        || v.len() != 4
        || q.iter().chain(k).chain(v).any(|&n| n <= 0)
        || q[0] != k[0]
        || k[..3] != v[..3]
        || q[3] != k[3]
        || q[1] % k[1] != 0
        || operation
            .inputs
            .slice(0..3)?
            .iter()
            .any(|a| a.dtype() != WorkspaceDtype::Float32)
        || operation.outputs.get(0)?.shape() != [q[0], q[1], q[2], v[3]]
        || operation.outputs.get(0)?.dtype() != WorkspaceDtype::Float32
    {
        return None;
    }
    // Use the worker's actual branch, not an admission-specific length limit.
    if arithmetic == AttentionArithmetic::InputScores
        && i64::from(q[2]) * i64::from(k[2])
            > i64::from(super::super::super::attention::INPUT_SCORE_ROW_BUDGET)
    {
        return None;
    }
    let groups = usize::try_from(q[0].checked_mul(q[1])?).ok()?;
    safemlx::RepeatedI32InputPlan::new(groups, usize::try_from(q[2]).ok()?)?;
    let mask = operation.inputs.len() > base;
    if mask {
        let a = operation.inputs.get(3)?;
        let target = [q[0], q[1], q[2], k[2]];
        if a.shape().len() > 4
            || a.shape()
                .iter()
                .rev()
                .zip(target.iter().rev())
                .any(|(&n, &m)| n != 1 && n != m)
            || !matches!(a.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32)
        {
            return None;
        }
    }
    if sinks
        && (operation.inputs.last()?.shape() != [q[1]]
            || operation.inputs.last()?.dtype() != WorkspaceDtype::Float32)
    {
        return None;
    }
    Some(Direct {
        softcap,
        mask,
        sinks,
        // The neutral floating class includes BF16. InputScores preserves it;
        // Fused+softcap widens the first product to F32 before this helper.
        qk_bf16: arithmetic == AttentionArithmetic::InputScores && q[3] % 32 == 0,
        qk_ids: arithmetic == AttentionArithmetic::InputScores,
        row_softmax: k[2].checked_add(i32::from(sinks))? >= 4,
    })
}

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let d = geometry(operation)?;
    // Expand K/V (reshape,broadcast,reshape), their/query casts and K transpose.
    let (mut p, mut e, mut seeds) = (9usize, 9usize, 0usize);
    // Original BF16 product: three setup views and one final-owned I32 IDs
    // source; safe-index validation31/36/3; IDs cast, three-input CustomKernel,
    // kernel result reshape and final batched reshape4/6. Total38/45/4.
    // Width rejection occurs after setup/IDs, then ordinary matmul8/9.
    let qk = if d.qk_bf16 {
        (38, 45, 4)
    } else if d.qk_ids {
        (11, 12, 1)
    } else {
        (8, 9, 0)
    };
    p += qk.0;
    e += qk.1;
    seeds += qk.2;
    // Scale: widen, binary with eager scalar, restore score dtype.
    p += 7;
    e += 8;
    seeds += 1;
    if d.softcap {
        p += 13;
        e += 14;
        seeds += 2;
    }
    // Bool Select and additive-mask alternatives. Bool dominates edges/seeds.
    if d.mask {
        p += 7;
        e += 9;
        seeds += 1;
    }
    // Sink cast/reshape/broadcast, both concat casts and variadic result.
    if d.sinks {
        p += 6;
        e += 7;
    }
    // Widen scores, row custom or native final-axis softmax, fixed rank4
    // sink-column Slice and probability cast. Native softmax dominates2/2.
    p += 5;
    e += 5;
    // PV preserves the input dtype and columns=true always accepts BF16 width.
    p += 38;
    e += 45;
    seeds += 4;
    let mut value = reduction_lowering(p, e, seeds, 6 + 6 + 1);
    // Each product's ordinary matmul can allocate four compactions/two partials;
    // that also dominates the custom's three copies + validation's two buffers.
    // Final row-softmax may compact its input once.
    value.validations = 1 + usize::from(d.qk_bf16);
    value.bf16_projection_calls = value.validations;
    value.intermediate_rank = 5;
    value.unqualified_kernel_owner = bf16_projection_source_requirement().or_else(|| {
        (d.row_softmax && !crate::backend::managed_memory::row_kernels::softmax_source_qualified())
            .then_some(CustomKernelOwner::RowSoftmax)
    });
    Some(value)
}

pub(super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    use std::mem::size_of;
    let d = geometry(operation)?;
    let mut bytes = if d.row_softmax {
        crate::backend::managed_memory::row_kernels::softmax_control_bytes(4)?
    } else {
        0
    };
    if d.sinks {
        bytes = bytes.checked_add(safemlx::ops::concatenate_axis_control_bytes()?)?;
    }
    // Direct worker's fixed views/slice bounds, scalar geometry and live safe
    // transports. Native descriptors/C shells use the same Graph construction
    // bank. The shared host quote separately owns the synchronous I32 producer.
    [
        size_of::<[i32; 5]>() * 2,
        size_of::<[i32; 4]>() * 3,
        size_of::<safemlx::Array>() * 12,
        size_of::<Option<safemlx::Array>>() * 2,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<usize>() * 3,
        size_of::<i32>() * 6,
        size_of::<&safemlx::Array>() * 6,
        size_of::<&safemlx::Stream>(),
        size_of::<Direct>(),
        size_of::<Option<Direct>>(),
        safemlx::Stream::device_type_control_bytes()?.checked_mul(2)?,
    ]
    .into_iter()
    .try_fold(bytes, usize::checked_add)
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
    fn direct_attention_keeps_bf16_validation_and_actual_blockwise_boundary() {
        if !crate::tests::support::native_process::enter("qualified-recipe") {
            return;
        }
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for (width, keys) in [(32, 4), (4, 3), (32, 8193)] {
            let context = WorkspaceContext::new(mechanism);
            let q = WorkspaceTensor::unloaded_f32(&[1, 4, 2, width], &context).unwrap();
            let k = WorkspaceTensor::unloaded_f32(&[1, 2, keys, width], &context).unwrap();
            let v = WorkspaceTensor::unloaded_f32(&[1, 2, keys, 6], &context).unwrap();
            let mask = WorkspaceTensor::unloaded_f32(&[2, keys], &context).unwrap();
            let sinks = WorkspaceTensor::unloaded_f32(&[4], &context).unwrap();
            context.begin_span();
            let output = WorkspaceBackend::attention_with_sinks(
                AttentionRequest {
                    queries: q,
                    keys: k,
                    values: v,
                    scale: 0.25,
                    softcap: Some(2.0),
                    mask: Some(&mask),
                    sinks: Some(&sinks),
                    arithmetic: AttentionArithmetic::InputScores,
                },
                &context,
            )
            .unwrap();
            assert_eq!(output.shape(), &[1, 4, 2, 6]);
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 2,
                    max_output_tokens: 1,
                    prefill_chunk_positions: 2,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
            if keys > 8192 {
                assert!(lowering(report.operations[0].as_view()).is_none());
                assert_eq!(reduced.first_missing_operation, None);
                assert_eq!(
                    reduced.nested_completions,
                    2 * 2 * (keys as usize).div_ceil(256)
                );
                assert!(reduced.mutable_storage.is_some());
                continue;
            }
            assert!(control_bytes(report.operations[0].as_view()).unwrap() > 0);
            assert_eq!(reduced.first_missing_operation, None);
            assert_eq!(reduced.validation_roots, if width == 32 { 2 } else { 1 });
            assert!(reduced.mutable_storage.unwrap().mutable_bytes() > 0);
            assert!(reduced.graph.is_some());
            assert!(reduced.dispatch.is_some());
        }
    }
}
