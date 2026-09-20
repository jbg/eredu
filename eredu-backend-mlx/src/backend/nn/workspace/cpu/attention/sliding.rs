//! Existing sliding query tiles composed from shared CPU masks and attention.
use super::super::program::Program;
use super::*;
use crate::backend::nn::tensor::GroupedOutputStorage;
use eredu_nn::operation_geometry::{CausalMaskGeometry, SlidingAttentionGeometry};

fn tile_count(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    let WorkspaceOperationKindView::Attention {
        causal: true,
        window: Some((window, offset)),
        softcap,
        arithmetic,
        ..
    } = operation.kind
    else {
        return None;
    };
    let q = operation.inputs.get(0)?.shape();
    let k = operation.inputs.get(1)?.shape();
    if q.len() != 4 || k.len() != 4 {
        return None;
    }
    SlidingAttentionGeometry::new_fixed(q[2], k[2], window, offset).ok()?;
    if !softcap && arithmetic == AttentionArithmetic::Fused && offset == 0 && q[2] <= window {
        return None;
    }
    Some((q[2] as usize).div_ceil(crate::backend::nn::attention::SLIDING_QUERY_TILE as usize))
}
pub(super) fn output_storage(operation: WorkspaceOperationView<'_>) -> GroupedOutputStorage {
    tile_count(operation).map_or_else(GroupedOutputStorage::default, |chunks| {
        GroupedOutputStorage {
            calls: 1,
            chunks,
            unit_observers: 0,
            observer_shape_rank: 0,
        }
    })
}
fn sliced<'a>(
    input: WorkspaceLayoutView<'_>,
    shape: &'a [i32; 4],
) -> Option<WorkspaceLayoutView<'a>> {
    let r = input.representation()?;
    let strides = super::super::views::physical_strides(input, r)?;
    let mut physical = [0u64; 4];
    for (out, stride) in physical.iter_mut().zip(strides) {
        *out = u64::try_from(stride).ok()?;
    }
    let row = shape
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 1)
        .all(|(axis, _)| {
            shape[axis + 1..]
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                == Some(physical[axis])
        });
    let representation = WorkspaceRepresentation::new(r.dtype(), row)
        .with_last_axis_contiguous(r.last_axis_contiguous())
        .with_element_strides(&physical)?;
    Some(
        WorkspaceLayoutView::new(shape, WorkspaceDtype::Float32)
            .ok()?
            .with_representation(Some(representation)),
    )
}
pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let Some(chunks) = tile_count(operation) else {
        return Ok(None);
    };
    let WorkspaceOperationKindView::Attention {
        causal: true,
        window: Some((window, offset)),
        sinks,
        softcap,
        arithmetic,
    } = operation.kind
    else {
        return Ok(None);
    };
    let source = (|| {
        if operation.inputs.len() != 3 + usize::from(sinks) || operation.outputs.len() != 1 {
            return None;
        }
        let query = operation.inputs.get(0)?;
        let key = operation.inputs.get(1)?;
        let value = operation.inputs.get(2)?;
        let q: [i32; 4] = query.shape().try_into().ok()?;
        let k: [i32; 4] = key.shape().try_into().ok()?;
        let v: [i32; 4] = value.shape().try_into().ok()?;
        if q.iter().chain(k.iter()).chain(v.iter()).any(|&n| n <= 0)
            || q[0] != k[0]
            || k[..3] != v[..3]
            || q[3] != k[3]
            || q[1] % k[1] != 0
        {
            return None;
        }
        let output = operation.outputs.get(0)?;
        if output.dtype() != WorkspaceDtype::Float32
            || output.shape() != [q[0], q[2], q[1].checked_mul(v[3])?]
        {
            return None;
        }
        let origin = SlidingAttentionGeometry::new_fixed(q[2], k[2], window, offset)
            .ok()?
            .key_origin();
        let mut p = Program::new(mechanism);
        let mut start = 0;
        let mut dtype = None;
        while start < q[2] {
            let end = start
                .checked_add(crate::backend::nn::attention::SLIDING_QUERY_TILE.min(q[2] - start))?;
            let absolute = offset.checked_add(start)?;
            let first = absolute.checked_sub(window - 1)?.max(origin);
            let keys = offset.checked_add(end)?.checked_sub(first)?;
            let queries = end - start;
            let qs = [q[0], q[1], queries, q[3]];
            let ks = [k[0], k[1], keys, k[3]];
            let vs = [v[0], v[1], keys, v[3]];
            let child_q = sliced(query, &qs)?;
            let child_k = sliced(key, &ks)?;
            let child_v = sliced(value, &vs)?;
            for _ in 0..3 {
                p.slice(4)?;
            }
            let mask_shape = [queries, keys];
            let mask = WorkspaceLayoutView::new(&mask_shape, WorkspaceDtype::Bool).ok()?;
            let geometry = CausalMaskGeometry::new_fixed(
                queries,
                absolute.checked_sub(first)?,
                Some(window - 1),
            )
            .ok()?;
            let mask_plan = super::super::causal_mask::inspect(
                WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::CausalMask(geometry),
                    inputs: WorkspaceLayoutList::Views(&[]),
                    outputs: WorkspaceLayoutList::Views(std::slice::from_ref(&mask)),
                },
                mechanism,
            )
            .ok()??;
            p.child(mask_plan)?;
            let child_shape = [q[0], q[1], queries, v[3]];
            let child_output =
                WorkspaceLayoutView::new(&child_shape, WorkspaceDtype::Float32).ok()?;
            let mut inputs = [child_q, child_k, child_v, mask, child_q];
            if sinks {
                inputs[4] = operation.inputs.last()?;
            }
            let plan = super::inspect(
                WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Attention {
                        causal: false,
                        window: None,
                        sinks,
                        softcap,
                        arithmetic,
                    },
                    inputs: WorkspaceLayoutList::Views(&inputs[..4 + usize::from(sinks)]),
                    outputs: WorkspaceLayoutList::Views(std::slice::from_ref(&child_output)),
                },
                mechanism,
            )
            .ok()??;
            if dtype.is_some_and(|dtype| dtype != plan.dtype) {
                return None;
            }
            dtype = Some(plan.dtype);
            p.child(plan)?;
            start = end;
        }
        let dtype = dtype?;
        let native = native_dtype(dtype);
        let count = usize::try_from(output.elements().ok()?).ok()?;
        if chunks > 1 {
            let joined =
                OperationEvent::cpu_concatenate_many_layout(native, 4, chunks, count, false)?;
            p.native.concatenate(joined, chunks)?;
            p.bytes = p.bytes.checked_add(
                p.capacity(count, native)?
                    .checked_mul(joined.backing_births() as u64)?,
            )?;
        } else {
            p.shells = p.shells.checked_add(1)?;
        }
        p.copy(
            OperationEvent::cpu_transpose_alias_layout(4, false)?,
            1,
            0,
            native,
        )?;
        let shape = [q[0], q[2], q[1], v[3]];
        let strides = [
            i64::from(q[1])
                .checked_mul(i64::from(q[2]))?
                .checked_mul(i64::from(v[3]))?,
            i64::from(v[3]),
            i64::from(q[2]).checked_mul(i64::from(v[3]))?,
            1,
        ];
        p.copy(
            OperationEvent::cpu_reshape_layout(&shape, &strides, output.shape(), false)?,
            1,
            count,
            native,
        )?;
        let output_bytes = p.capacity(count, native)?;
        let storage = output_storage(operation);
        let frames = [
            storage.control_bytes()?,
            safemlx::ops::concatenate_axis_control_bytes()?,
            size_of::<Program>() * 2,
            size_of::<OperationPlan>(),
            size_of::<Option<OperationPlan>>(),
            size_of::<WorkspaceOperationView<'_>>() * 3,
            size_of::<WorkspaceLayoutView<'_>>() * 12,
            size_of::<[i32; 4]>() * 10,
            size_of::<[i32; 2]>(),
            size_of::<[i64; 4]>(),
            size_of::<[u64; 4]>(),
            size_of::<SlidingAttentionGeometry>(),
            size_of::<CausalMaskGeometry>(),
            size_of::<usize>() * 8,
            size_of::<i32>() * 14,
            size_of::<safemlx::Array>() * 10,
            size_of::<Option<safemlx::Array>>(),
            size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
            size_of::<crate::backend::nn::tensor::GroupedChunkOutputs>(),
            super::super::views::physical_stride_control_bytes()?.checked_mul(3)?,
            size_of::<(
                safemlx::Array,
                safemlx::Array,
                safemlx::Array,
                f32,
                i32,
                i32,
                i32,
                i32,
                Option<&safemlx::Array>,
                Option<f32>,
                AttentionArithmetic,
                &safemlx::Stream,
            )>(),
        ];
        p.controls(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)?,
        )?;
        Some(OperationPlan {
            dtype,
            population: p.native,
            output_bytes,
            scratch_bytes: p.bytes.checked_sub(output_bytes)?,
            rank: 5,
            parameter_shells: p.shells,
            seeds: p.seeds,
            validations: 0,
            alias_input: None,
        })
    })();
    Ok(source)
}
