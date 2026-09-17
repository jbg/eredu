//! The actual selected-gather and four-contraction indexed attention worker.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::IndexedAttention {
        local_mask,
        pooled_mask,
        sinks,
        ..
    } = operation.kind
    else {
        return None;
    };
    let g = super::super::pooling::indexed_geometry(operation).ok()?;
    if operation
        .inputs
        .slice(0..5)?
        .iter()
        .any(|v| v.dtype() != WorkspaceDtype::Float32)
        || sinks && operation.inputs.last()?.dtype() != WorkspaceDtype::Float32
    {
        return None;
    }
    let masks = usize::from(local_mask) + usize::from(pooled_mask);
    for (present, index, width) in [
        (local_mask, 6, g.shape.local),
        (pooled_mask, 6 + usize::from(local_mask), g.shape.selected),
    ] {
        if !present {
            continue;
        }
        let m = operation.inputs.get(index)?;
        let target = [g.shape.batch, g.shape.heads, g.shape.queries, width];
        if !matches!(m.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32)
            || m.shape().len() > 4
            || m.shape()
                .iter()
                .rev()
                .zip(target.iter().rev())
                .any(|(&n, &d)| n != 1 && n != d)
        {
            return None;
        }
    }
    // Each GatherAxis helper: two expands, two broadcasts, then native
    // take_along_axis's two broadcasts and two-input GatherAxis (7/8).
    // Query scale 5/6/1. Each contraction: eight views + Matmul8/9 (16/17).
    // Concat casts its two or three sources; softmax casts and emits one op;
    // two slices select actual probability ranges; final Add is 5/6.
    let mut value = Lowering::plain(
        2 * 7 + 5 + 4 * 16 + 3 + 2 + 2 + 5,
        2 * 8 + 6 + 4 * 17 + 4 + 2 + 2 + 6,
        1,
    );
    // Bool masks dominate additive cast+Add (6/7/0): Where7/9/1 preserves
    // the actual score dtype's finite minimum, including reduced precision.
    value.primitives += masks * 7;
    value.edges += masks * 9;
    value.seeds += masks;
    if sinks {
        value.primitives += 4;
        value.edges += 5;
    }
    // An empty local bank makes both local Matmul workers construct a
    // native eager zero before fill_gpu. Their existing physical facts price
    // those scalars; count their descriptors instead of nonempty compactions.
    let products = if g.shape.local == 0 {
        value.seeds += 2;
        2
    } else {
        4
    };
    // Each nonempty Matmul has at most four compactions/two partials; final
    // softmax may compact once. Concat/reshape worker extents are separately
    // supplied by the shared rank profile.
    value.maximum_births = value
        .primitives
        .checked_add(value.seeds)?
        .checked_add(products * 6 + 1)?;
    value.backend_shells = crate::backend::nn::attention::indexed::returned_handles(masks, sinks)?;
    value.intermediate_rank = 4;
    value.unqualified_kernel_owner = grouped_indexed_source_requirement();
    Some(value)
}
