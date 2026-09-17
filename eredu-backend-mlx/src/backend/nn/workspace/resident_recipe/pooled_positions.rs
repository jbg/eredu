//! Fixed pooled-score contraction and the actual one-pass GPU sort program.
use super::*;
pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let (g, masked) = super::super::pooling::positions_geometry(operation).ok()?;
    if operation
        .inputs
        .slice(0..3)?
        .iter()
        .any(|v| v.dtype() != WorkspaceDtype::Float32)
        || masked && operation.inputs.get(3)?.dtype() != WorkspaceDtype::Bool
    {
        return None;
    }
    if g.pooled == 0 {
        // Existing zeros<U32>: eager scalar, two Broadcasts, cast and Full.
        let mut value = Lowering::plain(4, 4, 1);
        value.intermediate_rank = 3;
        value.backend_shells = 1;
        return Some(value);
    }
    // Three explicit F32 casts. The exact two-operand contraction has eight
    // view calls and native matmul (8/9). Four binary calls follow: Maximum,
    // score scale, head-weight scale and weighted product (5/6 each).
    // Weight transpose/expand 2/2; Reduce+Squeeze 2/2; ArgPartition and Slice1/1.
    let mut value = Lowering::plain(
        3 + 8 + 8 + 4 * 5 + 2 + 2 + 1 + 1,
        3 + 8 + 9 + 4 * 6 + 2 + 2 + 1 + 1,
        3,
    );
    if masked {
        value.primitives += 7;
        value.edges += 9;
        value.seeds += 1;
    }
    // Native matmul's four possible compactions/two partials; general Reduce's
    // compaction/partial; multi-block sort's four ping-pong arrays and table.
    value.maximum_births = value
        .primitives
        .checked_add(value.seeds)?
        .checked_add(6 + 2 + if g.pooled > 2048 { 5 } else { 0 })?;
    // SortDispatchLayout depends on axis width. A kernel grid handles all rows;
    // do not multiply the complete sort or its pipeline attempts by row count.
    value.additional_sort_kernels = grouped_sort_kernels(usize::try_from(g.pooled).ok()?);
    value.intermediate_rank = 4;
    value.backend_shells = g.returned_handles(masked);
    value.unqualified_kernel_owner = grouped_indexed_source_requirement();
    Some(value)
}
